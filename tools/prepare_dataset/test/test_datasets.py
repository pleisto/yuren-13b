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
from yuren_core.constants import PAD_TOKEN
from yuren_trainer.preparing_datasets import _batch_tokenize_texts, preparing_dataset

model_max_length = 4096

tokenizer = LlamaTokenizer.from_pretrained(
    "../../../data/llama2-han-tokenizer/dist",
    padding_side="right",
    pad_token=PAD_TOKEN,
)

load_pt_dataset = partial(preparing_dataset, False, model_max_length, "./cache", tokenizer)
pt_ds = load_pt_dataset("/Users/ding/Projects/yuren-13b-ds/data/pt/pt.base.parquet")


def count_tokens(batch):
    return {"num_tokens": len(batch["input_ids"])}


xpt = pt_ds.map(count_tokens, num_proc=8)

# check chunk is correct
assert len(pt_ds["input_ids"][0]) == model_max_length
for input_ids, labels in zip(pt_ds["input_ids"], pt_ds["labels"]):
    assert len(labels) == len(input_ids)
    assert PAD_TOKEN not in input_ids

overlap = 128

assert pt_ds["input_ids"][0][0] == tokenizer.bos_token_id
assert pt_ds["labels"][0][0] == tokenizer.bos_token_id


last_example_strides = pt_ds["input_ids"][-2][-overlap:]


# Debug:
#
# for input_ids, labels in zip(pt_ds["input_ids"], pt_ds["labels"]):
#     item = []
#     for input_id, label in zip(input_ids, labels):
#         char = tokenizer.decode(input_id)
#         if label == LabelSmoother.ignore_index:
#             char = f"/{char}/"
#         item.append(char)
#     print(item, "\n\n")

# static test
examples = {
    "context": ["天", None, None, None, None, None, None, None],
    "completion": [
        "宇宙洪荒",
        "赵钱孙李",
        "辰宿列张寒",
        "来暑往秋收冬藏",
        "欧洲",
        "近一",
        "半陆地被",
        "斯堪的纳维亚冰盖所覆盖",
    ],
}
tokenized = _batch_tokenize_texts(tokenizer, 8, examples, 6)

result = []
for input_ids, labels in zip(tokenized["input_ids"], tokenized["labels"]):
    item = []
    for input_id, label in zip(input_ids, labels):
        char = tokenizer.decode(input_id)
        if label == LabelSmoother.ignore_index:
            char = f"/{char}/"
        item.append(char)
    result.append(item)
assert result == [
    ["<s>", "//", "/天/", "宇", "宙", "洪", "荒", "<s>"],
    ["/<s>/", "赵", "钱", "孙", "李", "<s>", "辰", "宿"],
    ["/<s>/", "/辰/", "/宿/", "列", "张", "寒", "<s>", "来"],
    ["/<s>/", "/来/", "暑", "往", "秋", "收", "冬", "藏"],
    ["/暑/", "/往/", "/秋/", "/收/", "/冬/", "/藏/", "<s>", "欧"],
    ["/<s>/", "/欧/", "洲", "<s>", "近", "一", "<s>", "半"],
    ["/<s>/", "/半/", "陆", "地", "被", "<s>", "斯", "堪"],
    ["/<s>/", "/斯/", "/堪/", "的", "纳", "维", "亚", "冰"],
    ["/堪/", "/的/", "/纳/", "/维/", "/亚/", "/冰/", "盖", "所"],
    ["/纳/", "/维/", "/亚/", "/冰/", "/盖/", "/所/", "覆", "盖"],
]


load_sft_dataset = partial(preparing_dataset, True, model_max_length, "./cache", tokenizer)
sft_ds = load_sft_dataset("../../../data/sft.dev.json")
print("PASS")
