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
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from yuren_core.constants import IM_END_TOKEN, IM_START_TOKEN

from .utils import is_huge_dataset, last_index_of_list

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
PROC_NUM = 8


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )


def preparing_dataset(
    is_chatml: bool,
    model_max_length: int,
    cache_dir: str,
    tokenizer: AutoTokenizer,
    filename: str,
) -> Dataset:
    """
    Preparing datasets for training.

    Args:
        train_file: The path to the training dataset file.
        val_file: The path to the validation dataset file.
        tokenizer: The tokenizer to use.
        model_max_length: The maximum length of the model input.

    Returns:
        A tuple of the training and validation datasets.
    """
    if not os.path.exists(filename) or not (filename.endswith((".json", ".parquet"))):
        raise Exception(f"Dataset {filename} does not exist or is a unsupported format.")

    if is_huge_dataset(filename) is True and filename.endswith(".json"):
        raise Exception(
            f"Dataset {filename} is too large. Pyarrow has a bug when loading huge json files.\n"
            "Please convert the json file to parquet with the following command:\n"
            f"python -m json2parquet.main --input {filename}\n"
            "More details: https://issues.apache.org/jira/browse/ARROW-17137"
        )
    format = "json" if filename.endswith(".json") else "parquet"
    data = load_dataset(format, data_files=filename, cache_dir=cache_dir)["train"].shuffle()
    example_processor = _tokenize_chatml if is_chatml else _batch_tokenize_texts
    data = data.map(
        partial(example_processor, tokenizer, model_max_length),
        num_proc=PROC_NUM,
        batched=not is_chatml,
        remove_columns=data.column_names,
    )

    if is_chatml:
        for i in range(1):
            # since this function is called in torch_distributed_zero_first, no need rank_0_print
            print(f"{filename} tokenized example: {data[i]}")

    return data


def _tokenize_chatml(tokenizer: AutoTokenizer, max_len: int, example: Dict) -> Dict[str, List[int]]:
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
            formatted_sentence = IM_START_TOKEN + role + "\n" + sentence["value"] + IM_END_TOKEN
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
        label = copy.deepcopy(encoded_sentence) if role == "assistant" else [IGNORE_TOKEN_ID] * len(encoded_sentence)
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
    seq_len: int,
    examples: Dict[str, List[str]],
    max_sliding_stride: int = -1,
) -> Dict[str, List[List[int]]]:
    """
    Batch tokenizes the texts and returns a dictionary containing tokenized input ids, attention masks, and labels.
    And returns a dictionary while regrouping the inputs by chunks of max_len with a stride of sliding_stride.

    Notes:
    1. example['context'] and overlapped part by the sliding window are set to IGNORE_TOKEN_ID in the labels as they
    should not contribute to the loss calculation.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used to tokenize the texts.
        seq_len (int): The fixed length of the sequences. The examples will be regrouped by chunks of seq_len.
        examples (Dict[str, List[str]]): The examples to be tokenized. Each example is a dictionary where the keys are
            'context' and 'completion', and the values are lists of strings.
        max_sliding_stride (int): The maximum sliding stride. Defaults to 128. The sliding stride is the
            number of tokens to be strided when regrouping the inputs. The stride is dynamically calculated to be the
            first bos_token_id if present, otherwise 0. default: -1 (auto set to seq_len).

    Returns:
        Dict[str, List[List[int]]]: Returns a dictionary where the keys are 'input_ids' and 'labels', and the values are
            lists of lists of integers. 'input_ids' are the tokenized sequences, and 'labels' are copies of 'input_ids'
            where the tokens to be ignored are replaced with IGNORE_TOKEN_ID.The sequences in the 'labels' corresponding
            to 'context' and the overlapped part by the sliding window are set to IGNORE_TOKEN_ID as they should not
            contribute to the loss calculation.
    """

    # Set Default max_sliding_stride if not provided
    max_sliding_stride = int(seq_len / 32) if max_sliding_stride == -1 else max_sliding_stride

    # example_buffer is used to temporarily store examples until it reaches max_len
    example_buffer = {
        "input_ids": [],
        "labels": [],
    }

    # final tokenized dataset
    tokenized_dataset = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    def tokenize_example(context, completion):
        """Tokenizes context and completion, adds bos token, and returns input ids and labels."""
        # use pad_token to separate context and completion
        input_ids = tokenizer.encode(
            f"{context if context else ''}{tokenizer.pad_token}{completion}",
            add_special_tokens=False,
        )
        labels = copy.deepcopy(input_ids)

        # find the index of pad_token_id which separates context and completion
        sep_index = input_ids.index(tokenizer.pad_token_id)

        # context is optional and should be ignored in loss calculation
        # just like user sentences in ChatML
        if context is not None and len(context) > 0:
            labels[:sep_index] = [IGNORE_TOKEN_ID] * sep_index

        # remove the pad_token_id in all cases, as it's just a separator
        input_ids.pop(sep_index)
        labels.pop(sep_index)

        return [tokenizer.bos_token_id] + input_ids, [tokenizer.bos_token_id] + labels

    def add_to_tokenized_dataset(input_ids, labels):
        """Adds input ids and labels to tokenized_dataset."""
        tokenized_dataset["input_ids"].append(input_ids)
        tokenized_dataset["attention_mask"].append([1] * len(input_ids))
        tokenized_dataset["labels"].append(labels)
        return

    def sliding_example_buffer(example_buffer):
        """Strides through the example_buffer and returns the remaining part."""

        # If the example_buffer is equal to max_len, clear the buffer and return empty lists
        if len(example_buffer["input_ids"]) == seq_len:
            return {
                "input_ids": [],
                "labels": [],
            }

        input_ids = example_buffer["input_ids"]
        labels = example_buffer["labels"]

        remaining_input_ids = input_ids[seq_len:]
        remaining_labels = labels[seq_len:]

        # Stride elements from the example_buffer
        stride_input_ids = input_ids[:seq_len][-max_sliding_stride:]

        # Reduce stride_input_ids to the first bos_token_id if present
        stride_bos_idx = (
            last_index_of_list(stride_input_ids, tokenizer.bos_token_id)
            if tokenizer.bos_token_id in stride_input_ids
            else 0
        )
        stride_input_ids = stride_input_ids[stride_bos_idx:]

        return {
            "input_ids": stride_input_ids + remaining_input_ids,
            "labels": [IGNORE_TOKEN_ID] * len(stride_input_ids) + remaining_labels,
        }

    # iterate through each example
    for context, completion in zip(examples["context"], examples["completion"]):
        # breakpoint()
        input_ids, labels = tokenize_example(context, completion)

        # add bos token to the beginning of the input_ids and labels
        example_buffer["input_ids"].extend(input_ids)
        example_buffer["labels"].extend(labels)

        if len(example_buffer["input_ids"]) < seq_len:
            # continue to add examples to the buffer until it reaches max_len
            continue
        else:  # len(example_buffer["input_ids"]) >= max_len
            # Appending the first max_len elements of example_buffer to the tokenized_dataset
            add_to_tokenized_dataset(
                example_buffer["input_ids"][:seq_len],
                example_buffer["labels"][:seq_len],
            )
            # Stride through the example_buffer and return the remaining part
            example_buffer = sliding_example_buffer(example_buffer)

    # Process the remaining part of the example_buffer
    while len(example_buffer["input_ids"]) >= seq_len:
        add_to_tokenized_dataset(
            example_buffer["input_ids"][:seq_len],
            example_buffer["labels"][:seq_len],
        )
        example_buffer = sliding_example_buffer(example_buffer)

    return tokenized_dataset
