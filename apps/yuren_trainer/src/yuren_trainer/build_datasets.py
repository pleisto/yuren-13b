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
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from yuren_core.constants import IM_END_TOKEN, IM_START_TOKEN
from .utils import is_huge_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
PROC_NUM = 8


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )


def build_dataset(
    is_chatml: bool,
    model_max_length: int,
    cache_dir: str,
    tokenizer: AutoTokenizer,
    filename: str,
) -> Dataset:
    """
    Builds the training and validation datasets for supervised training.

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
            f"Dataset {filename} does not exist or is a unsupported format."
        )

    if is_huge_dataset(filename) is True and filename.endswith(".json"):
        raise Exception(
            f"Dataset {filename} is too large. Pyarrow has a bug when loading huge json files.\n"
            f"Please convert the json file to parquet with the following command:\n"
            f"python -m json2parquet.main --input {filename}\n"
            f"More details: https://issues.apache.org/jira/browse/ARROW-17137"
        )
    format = "json" if filename.endswith(".json") else "parquet"
    data = load_dataset(format, data_files=filename, cache_dir=cache_dir)[
        "train"
    ].shuffle()
    example_processor = _tokenize_chatml if is_chatml else _batch_tokenize_texts
    data = data.map(
        partial(example_processor, tokenizer, model_max_length),
        num_proc=PROC_NUM,
        batched=not is_chatml,
        remove_columns=data.column_names,
    )

    for i in range(2):
        # since this function is called in torch_distributed_zero_first, no need rank_0_print
        print(f"{filename} tokenized example: {data[i]}")

    return data


def _tokenize_chatml(
    tokenizer: AutoTokenizer, max_len: int, example: Dict
) -> Dict[str, List[int]]:
    """
    Generates and tokenize the ChatML conversation.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to be used for encoding the text.
        max_len (int): The maximum length of the model's input.
        example (dict): A dictionary containing the conversation data.

    Returns:
        dict: A dictionary containing tokenized input ids, attention masks, and labels.
    """
    input_ids = []
    labels = []
    conversations = example["conversations"]

    for sentence in conversations:
        role = sentence["from"].lower()

        if role is None:
            raise ValueError(f"Unknown sentence: {sentence}")

        if role == "system":
            formatted_sentence = (
                IM_START_TOKEN + role + "\n" + sentence["value"] + IM_END_TOKEN
            )
        elif role == "user":
            formatted_sentence = (
                f"\n{IM_START_TOKEN}"
                + role
                + "\n"
                + sentence["value"]
                + f"{IM_END_TOKEN}\n{IM_START_TOKEN}"
                + "assistant"
                + "\n"
            )
        else:
            formatted_sentence = sentence["value"] + IM_END_TOKEN

        encoded_sentence = tokenizer.encode(formatted_sentence)
        label = (
            copy.deepcopy(encoded_sentence)
            if role == "assistant"
            else [IGNORE_TOKEN_ID] * len(encoded_sentence)
        )
        input_ids += encoded_sentence
        labels += label

    # add bos token for the first sentence in the conversation
    input_ids = [tokenizer.bos_token_id] + input_ids
    labels = [tokenizer.bos_token_id] + labels

    # truncate the input_ids and labels to model_max_length
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]

    # labels can not have all values being -100. 18 and 24 are just random numbers
    if all(x == IGNORE_TOKEN_ID for x in labels):
        raise ValueError(f"Labels can not have all values being: {conversations}")

    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    return tokenized_full_prompt


def _batch_tokenize_texts(
    tokenizer: AutoTokenizer,
    max_len: int,
    examples: Dict[str, List[str]],
    overlap: int = 40,
) -> Dict[str, List[List[int]]]:
    """
    Batch generates and tokenize the text.

    e.g. [bos] input_ids.. [bos] input_ids..

    Args:
        tokenizer (AutoTokenizer): The tokenizer to be used for encoding the text.
        max_len (int): The maximum length of the model's input.
        examples (dict): **batch** of examples.
        overlap (int): The number of tokens to overlap between chunks.

    Returns:
        dict: A dictionary containing tokenized input ids, attention masks, and labels.
    """
    encoded_examples = {
        "input_ids": [],
        "labels": [],
    }
    for context, completion in zip(examples["context"], examples["completion"]):
        encoded_input = tokenizer.encode(completion, add_special_tokens=False)
        labels = copy.deepcopy(encoded_input)

        # add context if it exists, and context should be ignored when calculating loss
        if len(context) > 0:
            encoded_context = tokenizer.encode(context, add_special_tokens=False)
            encoded_input = encoded_context + encoded_input
            labels = [IGNORE_TOKEN_ID] * len(encoded_context) + labels

        # add bos token to the beginning of the input_ids and labels
        encoded_examples["input_ids"].extend([tokenizer.bos_token_id] + encoded_input)
        encoded_examples["labels"].extend([tokenizer.bos_token_id] + labels)

    assert (
        len(encoded_examples["input_ids"]) >= max_len
    ), f"packed input ids length: {len(encoded_examples['input_ids'])} is smaller than max_len: {max_len}"

    assert len(encoded_examples["input_ids"]) == len(encoded_examples["labels"])

    def split_into_chunks(
        sequence: List[int], labels: List[int]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Splits a sequence into chunks of size max_len with an overlap.
        Drops the last chunk if it is smaller than max_len.
        """
        chunks = []
        label_chunks = []
        start_index = 0

        # handle the first chunk
        if len(sequence) >= max_len:
            chunks.append(sequence[:max_len])
            label_chunks.append(labels[:max_len])
            start_index += max_len

        # handle the rest of the chunks
        while start_index + max_len - overlap <= len(sequence):
            chunks.append(
                sequence[start_index - overlap : start_index + max_len - overlap]
            )
            label_chunks.append(
                [IGNORE_TOKEN_ID] * overlap
                + labels[start_index : start_index + max_len - overlap]
            )

            start_index += max_len - overlap

        return chunks, label_chunks

    input_chunks, label_chunks = split_into_chunks(
        encoded_examples["input_ids"], encoded_examples["labels"]
    )

    return {
        "input_ids": input_chunks,
        "labels": label_chunks,
    }
