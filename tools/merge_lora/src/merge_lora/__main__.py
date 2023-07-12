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
 """
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_lora(base_model_path, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True, help="Base model path")
    parser.add_argument("--lora-path", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--target-model-path", type=str, required=True, help="Merged model path")

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path)
