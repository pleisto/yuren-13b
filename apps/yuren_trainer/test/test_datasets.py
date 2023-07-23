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

from functools import partial

from transformers import LlamaTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from yuren_trainer.build_datasets import build_dataset

model_max_length = 4096

tokenizer = LlamaTokenizer.from_pretrained("../../../data/llama2-han-tokenizer/dist")

load_pt_dataset = partial(build_dataset, False, model_max_length, ".", tokenizer)
pt_ds = load_pt_dataset("../../../data/pt.dev.json")

# check chunk is correct
for input_ids, labels in zip(pt_ds["input_ids"], pt_ds["labels"]):
    assert len(input_ids) == model_max_length, f"{len(input_ids)} != {model_max_length}"
    assert len(labels) == model_max_length, f"{len(labels)} != {model_max_length}"

overlap = 40

assert pt_ds["input_ids"][0][0] == tokenizer.bos_token_id
assert pt_ds["labels"][0][0] == tokenizer.bos_token_id

assert pt_ds["input_ids"][0][-overlap:] == pt_ds["input_ids"][1][:overlap]

assert pt_ds["labels"][1][:overlap] == [LabelSmoother.ignore_index] * overlap
assert pt_ds["input_ids"][-2][-overlap:] == pt_ds["input_ids"][-1][:overlap]
assert pt_ds["labels"][-1][:overlap] == [LabelSmoother.ignore_index] * overlap

print("PT PASS")


load_sft_dataset = partial(build_dataset, True, model_max_length, ".", tokenizer)
sft_ds = load_sft_dataset("../../../data/sft.dev.json")
print(sft_ds["input_ids"][0])
print("SFT PASS")
