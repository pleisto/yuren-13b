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
import copy
import os
from functools import partial
from typing import Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from yuren_core.constants import IM_END_TOKEN, IM_START_TOKEN

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
PRINT_EXAMPLES_NUM = 2


def build_text_sft_dataset(
    model_max_length: int,
    cache_dir: str,
    tokenizer: AutoTokenizer,
    filename: str,
) -> Dataset:
    """
    Builds the training and validation datasets for text-only supervised training.

    Args:
        train_file: The path to the training dataset file.
        val_file: The path to the validation dataset file.
        tokenizer: The tokenizer to use.
        model_max_length: The maximum length of the model input.

    Returns:
        A tuple of the training and validation datasets.
    """
    if not os.path.exists(filename) or not (filename.endswith((".json", ".parquet"))):
        raise Exception(
            f"Dataset {filename} does not exist or is not in JSON or Parquet format."
        )

    data_processor = partial(
        _generate_and_tokenize_conversations, tokenizer, model_max_length
    )

    # Since pyarrow has a bug when loading huge json files, we use chunking to convert the json file to dataset
    # @see: https://issues.apache.org/jira/browse/ARROW-17137
    parquet_file = f"{filename}.parquet" if filename.endswith(".json") else filename
    if os.path.exists(parquet_file) is False and filename.endswith(".json"):
        df = pd.read_json(filename)
        chunk_size = 500_000
        table = pa.Table.from_batches(
            [
                pa.record_batch(df.iloc[i : i + chunk_size])
                for i in range(0, len(df), chunk_size)
            ]
        )
        pq.write_table(table, parquet_file)

    data = (
        load_dataset("parquet", data_files=parquet_file, cache_dir=cache_dir)["train"]
        .shuffle()
        .map(data_processor, num_proc=8)
    )

    cols_to_keep = ["input_ids", "labels", "attention_mask"]
    remove_cols = [col for col in data.column_names if col not in cols_to_keep]
    data = data.remove_columns(remove_cols)

    # data.save_to_disk(f"{filename}.saved.parquet")

    for i in range(PRINT_EXAMPLES_NUM):
        # since this function is called in torch_distributed_zero_first, no need rank_0_print
        print(f"{filename} tokenized example: {data[i]}")

    return data


def _generate_and_tokenize_conversations(
    tokenizer: AutoTokenizer, model_max_length: int, example: Dict
) -> Dict[str, list]:
    """
    Generates and tokenize the conversations for a dataset.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to be used for encoding the text.
        model_max_length (int): The maximum length of the model's input.
        example (dict): A dictionary containing the conversation data.

    Returns:
        dict: A dictionary containing tokenized input ids, attention masks, and labels.
    """
    input_ids = []
    labels = []
    conversations = example["conversations"]

    # mappings ShareGPT `from` value to OpenAI's ChatML format
    role_mapping = {"human": "user", "system": "system", "gpt": "assistant"}

    for sentence in conversations:
        role = role_mapping.get(sentence["from"].lower())

        if role is None:
            raise ValueError(f"Unknown sentence: {sentence}")

        if role == role_mapping["system"]:
            formatted_sentence = (
                IM_START_TOKEN + role + "\n" + sentence["value"] + IM_END_TOKEN
            )
        elif role == role_mapping["human"]:
            formatted_sentence = (
                f"\n{IM_START_TOKEN}"
                + role
                + "\n"
                + sentence["value"]
                + f"{IM_END_TOKEN}\n{IM_START_TOKEN}"
                + role_mapping["gpt"]
                + "\n"
            )
        else:
            formatted_sentence = sentence["value"] + IM_END_TOKEN

        encoded_sentence = tokenizer.encode(
            formatted_sentence, add_special_tokens=False
        )  # do not add bos_token_id
        label = (
            copy.deepcopy(encoded_sentence)
            if role == role_mapping["gpt"]
            else [IGNORE_TOKEN_ID] * len(encoded_sentence)
        )
        input_ids += encoded_sentence
        labels += label

        # add eos at every end of assistant sentence
        if role == role_mapping["gpt"]:
            input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]

    input_ids = input_ids[: model_max_length - 1]
    labels = labels[: model_max_length - 1]

    # replace the last token with eos_token_id if it is not eos_token_id
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids[-1] = tokenizer.eos_token_id
        labels[-1] = tokenizer.eos_token_id

    # labels can not have all values being -100. 18 and 24 are just random numbers
    if not any(x > IGNORE_TOKEN_ID for x in labels):
        labels[18:24] = input_ids[18:24]

    attention_mask = [1] * len(input_ids)
    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized_full_prompt
