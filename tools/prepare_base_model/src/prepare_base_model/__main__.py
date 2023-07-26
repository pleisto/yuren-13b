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
from typing import Dict

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from yuren_core.constants import IM_END_TOKEN, IM_START_TOKEN, PAD_TOKEN

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-13b-hf")
arg_parser.add_argument("--tokenizer_path", type=str, default="./data/llama2-han-tokenizer/dist")
arg_parser.add_argument("--output_path", type=str, default="./dist/llama2-13b-hf-han-tokenizer")
args = arg_parser.parse_args()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding.
    Note: This is the optimized version that makes your embedding size divisible by 128.
    """

    # vocab size must be divisible by 128 to improve performance
    # see https://arxiv.org/abs/1909.08053
    VOCAB_MULTIPLE = 128

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    # check if current token embedding has a available size
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    has_available_size = model_vocab_size >= len(tokenizer) and model_vocab_size % VOCAB_MULTIPLE == 0

    # resize token embedding if not has available size
    if not has_available_size:
        # find the closest divisible by 64 with len(tokenizer)
        model.resize_token_embeddings((len(tokenizer) + VOCAB_MULTIPLE - 1) // VOCAB_MULTIPLE * VOCAB_MULTIPLE)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main():
    special_tokens_dict = {
        "additional_special_tokens": [
            PAD_TOKEN,
            IM_START_TOKEN,
            IM_END_TOKEN,
        ]
    }

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model_config = AutoConfig.from_pretrained(args.model_path)

    #  fix the max_position_embeddings to 4096, because llama2 has 4096 max tokens
    model_config.max_position_embeddings = 4096
    model_config.max_sequence_length = model_config.max_position_embeddings

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        config=model_config,
        trust_remote_code=True,
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    tokenizer.save_pretrained(args.output_path)
    print(f"Tokenizer saved to {args.output_path}")
    model.save_pretrained(args.output_path)
    print(f"Model saved to {args.output_path}")


if __name__ == "__main__":
    main()
