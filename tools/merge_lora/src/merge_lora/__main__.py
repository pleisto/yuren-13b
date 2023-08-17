"""
 Copyright 2023 Pleisto Inc

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Forked from https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/merge_llama2_with_chinese_lora_low_mem.py
 License: Apache-2.0 License, Copyright (c) 2023 ymcui
 """


import argparse
import gc
import json
import os
import re

import peft
import torch
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer
from transformers.modeling_utils import dtype_byte_size

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model",
    default=None,
    required=True,
    type=str,
    help="Base model path (basically Llama-2-hf)",
)
parser.add_argument(
    "--lora_model",
    default=None,
    required=True,
    type=str,
    help="LoRA model path",
)
parser.add_argument(
    "--output_dir",
    default="./merged_model",
    type=str,
    help="Output path for the merged model",
)
parser.add_argument(
    "--verbose",
    default=False,
    action="store_true",
    help="Show detailed debugging messages",
)


emb_to_model_size = {
    4096: "7B",
    5120: "13B",
    8192: "70B",
}
num_shards_of_models = {"7B": 1, "13B": 2, "70B": 8}
params_of_models = {
    "7B": {
        "dim": 4096,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-05,
        "vocab_size": -1,
    },
    "13B": {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-05,
        "vocab_size": -1,
    },
    "70B": {
        "dim": 8192,
        "multiple_of": 4096,
        "ffn_dim_multiplier": 1.3,
        "n_heads": 64,
        "n_kv_heads": 8,
        "n_layers": 80,
        "norm_eps": 1e-05,
        "vocab_size": -1,
    },
}


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


# Borrowed and modified from https://github.com/tloen/alpaca-lora
def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


def merge_shards(output_dir, num_shards: int):
    ckpt_filenames = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if re.match("L(\d+)-consolidated.(\d+).pth", f)
        ]
    )

    for i in range(num_shards):
        shards_filenames = sorted(
            [f for f in ckpt_filenames if re.match(f"L(\d+)-consolidated.0{i}.pth", f)]
        )
        print(f"Loading {shards_filenames} ...")
        shards_dicts = [
            torch.load(os.path.join(output_dir, fn)) for fn in shards_filenames
        ]
        shards_merged = {}
        for d in shards_dicts:
            shards_merged |= d

        print(
            "Saving the merged shard to "
            + os.path.join(output_dir, f"consolidated.0{i}.pth")
        )
        torch.save(shards_merged, os.path.join(output_dir, f"consolidated.0{i}.pth"))

        print("Cleaning up...")
        del shards_merged
        for d in shards_dicts:
            del d
        del shards_dicts
        gc.collect()  # Effectively enforce garbage collection
        for fn in shards_filenames:
            os.remove(os.path.join(output_dir, fn))


if __name__ == "__main__":
    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")

    tokenizers_and_loras = []
    print(f"Loading {lora_model_path}")
    if not os.path.exists(lora_model_path):
        print("Cannot find lora model on the disk. Downloading lora model from hub...")
        lora_model_path = snapshot_download(repo_id=lora_model_path)
    tokenizer = LlamaTokenizer.from_pretrained(lora_model_path, legacy=True)
    lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
    lora_state_dict = torch.load(
        os.path.join(lora_model_path, "adapter_model.bin"), map_location="cpu"
    )
    if "base_model.model.model.embed_tokens.weight" in lora_state_dict:
        lora_vocab_size = lora_state_dict[
            "base_model.model.model.embed_tokens.weight"
        ].shape[0]

    tokenizers_and_loras.append(
        {
            "tokenizer": tokenizer,
            "state_dict": lora_state_dict,
            "config": lora_config,
            "scaling": lora_config.lora_alpha / lora_config.r,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
        }
    )

    if not os.path.exists(base_model_path):
        print("Cannot find lora model on the disk. Downloading lora model from hub...")
        base_model_path = snapshot_download(repo_id=base_model_path)
    ckpt_filenames = sorted(
        [
            f
            for f in os.listdir(base_model_path)
            if re.match("pytorch_model-(\d+)-of-(\d+).bin", f)
        ]
    )
    if len(ckpt_filenames) == 0:
        raise FileNotFoundError(
            f"Cannot find base model checkpoints in ${base_model_path}. Please make sure the checkpoints are saved in"
            " the HF format."
        )
    embedding_size = None
    model_size = None
    total_size = 0
    for index, filename in enumerate(ckpt_filenames):
        print(f"Loading ckpt {filename}")
        state_dict = torch.load(
            os.path.join(base_model_path, filename), map_location="cpu"
        )
        if index == 0:
            embedding_size = state_dict["model.embed_tokens.weight"].shape[1]
            model_size = emb_to_model_size[embedding_size]

        print("Merging...")
        for k in state_dict:
            for tl_idx, t_and_l in enumerate(tokenizers_and_loras):
                saved_key = "base_model.model." + k
                lora_key_A = saved_key.replace(".weight", ".lora_A.weight")
                if saved_key in t_and_l["state_dict"]:
                    if args.verbose:
                        print(
                            f"copying {saved_key} from {tl_idx}-th LoRA weight to {k}"
                        )
                    state_dict[k] = (
                        t_and_l["state_dict"][saved_key].half().clone()
                    )  # do we need half()?
                if lora_key_A in t_and_l["state_dict"]:
                    lora_key_B = lora_key_A.replace("lora_A.weight", "lora_B.weight")
                    if args.verbose:
                        print(
                            f"merging {lora_key_A} and lora_B.weight form {tl_idx}-th LoRA weight to {k}"
                        )
                    state_dict[k] += (
                        transpose(
                            t_and_l["state_dict"][lora_key_B].float()
                            @ t_and_l["state_dict"][lora_key_A].float(),
                            t_and_l["fan_in_fan_out"],
                        )
                        * t_and_l["scaling"]
                    )
            weight_size = state_dict[k].numel() * dtype_byte_size(state_dict[k].dtype)
            total_size += weight_size

        print(f"Saving ckpt {filename} to {output_dir} in HF format...")
        torch.save(state_dict, os.path.join(output_dir, filename))

        del state_dict
        gc.collect()  # Effectively enforce garbage collection

    print("Saving tokenizer")
    tokenizers_and_loras[-1]["tokenizer"].save_pretrained(output_dir)

    configs = (
        "config.json",
        "generation_config.json",
        "pytorch_model.bin.index.json",
    )
    for config in configs:
        if os.path.exists(os.path.join(base_model_path, config)):
            print(f"Saving {config}")
            with open(os.path.join(base_model_path, config), "r") as f:
                obj = json.load(f)
            if config == "config.json":
                obj["vocab_size"] = len(tokenizers_and_loras[-1]["tokenizer"])
            if config == "pytorch_model.bin.index.json":
                obj["metadata"]["total_size"] = total_size
            with open(os.path.join(output_dir, config), "w") as f:
                json.dump(obj, f, indent=2)
    print("Done.")
    print(f"Check output dir: {output_dir}")
