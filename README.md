# 羽人 13B

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Lint](https://github.com/pleisto/yuren-13b/actions/workflows/lint.yml/badge.svg)](https://github.com/pleisto/yuren-13b/actions/workflows/lint.yml) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE) [![huggingface badge](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-f2f4f5?labelColor=f2f4f5)](https://huggingface.co/pleisto/yuren-13b-chatml)

Yuren 13B is an information synthesis large language model that has been continuously trained based on Llama 2 13B, which builds upon the data-centric work of Pleisto. This model has achieved state-of-the-art performance in various information synthesis scenarios, including information extraction in multiple languages with a focus on Chinese and English, natural language to SQL generation, and structured data output, all with the same parameter size. **For more English information, please refer to [model card](https://huggingface.co/pleisto/yuren-13b-chatml).**

羽人 13B 是在 Llama 2 13B 基础上进行持续训练的**信息合成**大语言模型，建立在 Pleisto 以数据为中心的工作上。该模型在以中英文为主的多种语言的信息抽取、自然语言生成 SQL、结构化数据输出等信息合成类场景下实现了同等参数量下的 SOTA 水平。

## 主要亮点

- **信息合成模型**: 羽人专注于优化自然语言理解、自然语言生成 SQL 等信息合成类任务的能力。
- **中文支持**: 羽人使用了高质量的中文语料进行持续训练， 并对于 Tokenizer 词表进行了扩充，使得模型能够更好地支持中文。
- **超高质量数据集**: 羽人训练所用数据集的基础数据来自于 Pleisto 自有的商业多轮对话与指令精调数据集的一个子集, 该数据集的所有指令均经过了多轮次的人工和算法质检, 在此基础上我们还参考了[Orca LLM](https://arxiv.org/abs/2306.02707)的工作在该子集上进行了基于 GPT-4 的数据增强。我们还额外引入一部分进行过数据增强的 [flan 2022 collection](https://github.com/google-research/FLAN/tree/main/flan/v2) 数据。
- **商业友好**: 羽人的训练和推理代码以 Apache-2.0 协议开源, 在遵守[Llama 2 Acceptable Use Policy](https://ai.meta.com/llama/use-policy/)的前提下模型权重完全支持商用。
- **全面兼容 ChatML**: 羽人全面兼容 GPT-4 同款的[ChatML 格式](https://github.com/openai/openai-python/blob/main/chatml.md), 一方面可以最大限度地减少 Prompt Injection 所带来的安全风险, 另一方面可以和 GPT-4 一样实现良好的 System Prompt 遵循度。(没错, 我们的训练数据集中包含了相当一部分带有 system prompt 的对话数据)

## 使用 WebUI 进行推理

### Docker

> Coming soon

### 本地运行

```bash
# 推荐使用 rye 进行环境管理, 可访问 https://rye-up.com/guide/installation/#installing-rye 查看详情
# curl -sSf https://rye-up.com/get | bash
# source "$HOME/.rye/env"
# rye sync # 可替代 pip install -r requirements.lock

pip install -r requirements.lock
python -m webui.app "pleisto/yuren-13b-chat" # --load_8bit True --server_name "0.0.0.0" --share True
```

## 局限性

羽人 13B 模型主要设计用于信息合成领域，包括构建智能代理、自然语言理解、生成 SQL 等业务场景，而并非直接用于向公众提供服务。我们强烈建议将此模型应用于企业内部的数据处理场景，而不是公开环境。

虽然我们已经尽可能确保模型训练过程中的数据合规性，但由于模型和数据的复杂性，可能存在无法预见的问题。如果由于使用羽人 13B 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们不承担任何责任。

我们强烈建议在使用模型时采用额外的安全措施，如对模型的输入输出进行过滤、审查或限制，以免对用户造成伤害。科技的发展应在规范和合法的环境下进行，我们希望所有使用者都能秉持这一原则。我们将持续改进模型的训练和使用，以提升其安全性和有效性。

## 训练数据

遗憾的是, 由于羽人的训练数据集建立在我们的商业数据集的子集之上, 因此我们现阶段没有将其完整开源的计划。目前我们只能提供一个[样例数据集](./data/), 该数据集的格式和我们的完整数据集完全一致, 但是由于数据量太少, 无法训练出一个完整的模型, 仅供大家参考。该样例数据集以[CC BY-SA 4.0 (署名且以相同方式共享)](https://creativecommons.org/licenses/by-sa/4.0/deed.zh-Hans) 协议开源, 详见文件内的`__comment__`字段。

### 预训练数据格式

```json
{
  "context": "",
  "completion": ""
}
```

`completion` 为文本数据本身的内容， `context` 会被拼接到 `completion` 的前面作为模型的输入， **但`context`中的文本所对应的 label 为`-100`**, 这意味着这部分文本会在 loss 中被忽略就像 ChatML 中的 user prompt 一样。在大多数情况下，你可能只需要用到`completion`字段并保持`context`为空即可。

### SFT 数据格式

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "Your are helpful AI assistant."
    },
    {
      "from": "user",
      "value": "User input prompt."
    },
    {
      "from": "assistant",
      "value": "Assistant response."
    }
  ]
}
```

如果为多轮对话只需要重复 `user` 和 `assistant` 即可。`system` role 用于实现 ChatGPT 同款的 System Prompt 和 Custom instructions 功能， 如果没有特殊需求可以保持它的值为 `"Your are helpful AI assistant."`。

但是如果你期望最终的模型具有较佳的 System Prompt 性能，强烈建议参考[示例文件](./data/sft.dev.json#L57)添加更多种类的 System Prompt 数据。

## 复现训练

### 准备基座模型

原始  Llama  词表中仅含有几百个汉字，其余汉字均需要以多个  unicode  字节形式拼接生成。这一问题除了显而易见地导致中文推理性能（生成速度）受到影响之外，还在很大程度上造成了模型在中文语义理解上造成了性能瓶颈。

我们对于不同的词表扩充方案进行了一系列对比实验并发现：

- 相较于目前主流的在词表中加入大量的常用汉字词语的策略而言，仅在词表中添加汉字字符的方案可以在更少的预训练数据规模下实现更佳的语义理解性能。我们在实验中发现，现有的基于  BPE  的分词器进行中文分词时由于分词本身存在的歧义性几乎必然导致生成的  Token  分割难以与真实语义进行对齐。尽管通过提升模型参数量、增加训练数据的规模和多样性，可以让模型本身在预训练过程中拥有正确理解被错误分割的  token  的能力，但这种理解始终是有额外成本的。
- 扩充词表时新增的  token  的数量和对于原始词向量的分布的扰动始终成正比，因此新增的  token  越少对于已有  token  的语义扰动的影响就会越少。

鉴于此我们较为保守地扩充了  4843  个  Token，具体而言包括国家语委在  2013  年发布的《通用规范汉字表》中的全部一级汉字、二三级汉字的一个子集（该子集通过使用  Pleisto  自有的中文语料进行汉字常用字字频统计后得出以期最大可能地覆盖包括科学技术领域常用字、人名地名常用字在内的所有常用汉字）、汉语中较常使用的一部分标点符号。

```bash
pip install -r requirements.txt
python -m prepare_base_model
```

(注:词表大小会被扩充至最接近的 128 的倍数以改善并行训练时的性能)

### 数据集预处理

> Tips: 可以在本地完成数据集的预处理后再上传至服务器进行训练以节省成本。

如果未指定验证集，则默认将从训练集中划分 1% 作为验证集。

#### Embedding 预训练 或 持续训练（PT）

```bash
python -m prepare_dataset --train_file "data/pt.dev.json" --type "text"
```

#### SFT 训练

```bash
python -m prepare_dataset --train_file "data/sft.dev.json" --type "chatml"
```

#### 完整示例

```bash
python -m prepare_dataset --train_file "/mnt/nfs/yuren13b/train-pt.parquet" --validation_file "/mnt/nfs/yuren13b/validation-pt.parquet" --type "text" --model_max_length 2048 --tokenizer_name "./dist/llama2-13b-hf-han-tokenizer" --output_dir "/mnt/nfs/yuren13b/pt-ds"
```

> 下述脚本均适用于 8 卡 A100 80G 或 H800 环境, 如需在其他环境下运行请酌情调整相关参数。

### Embedding 预训练

尽管目前主流的词表扩充方案基于成本考虑通常不再单独针对词向量嵌入层进行预先训练，而是依靠在进行持续预训练时对于词向量嵌入层的更新来实现词向量的对齐。但我们的研究发现：

- 完全将新增词向量进行随机初始化，而后在持续预训练阶段进行语义对齐的方案会导致一部分预训练数据难以被模型真正学到，该问题在预训练数据是经过精心清洗的高质量数据集的情况下尤其明显。
- 冻结其他层，使用不同于持续预训练阶段的多样性数据集（尤其是包含多语平行语料的数据集）仅针对词向量嵌入层进行训练有助于提升模型的语义理解能力。尤其是当预训练阶段使用高质量小规模数据集的情况下，使用更具多样性的、未经人工过滤的数据集预先训练词向量嵌入层有助于提升模型的语义理解能力和抗毒性能力。

鉴于此我们在冻结其他层的情况下，使用了  760M Token  的多样性语料进行了  1  个  Epoch  的词向量嵌入层预训练。训练中使用了  128  的全局  Btach Size，train/loss  从训练前的  5.907  降低到  3.429。

```bash
torchrun --nproc_per_node=8 -m yuren_trainer --train_task 'embed_token' --model_name_or_path "dist/llama2-13b-hf-han-tokenizer" --dataset 'data/ds_embed_token_1024' --model_max_length 1024   --num_train_epochs 1 --per_device_eval_batch_size 16 --per_device_train_batch_size 16   --gradient_accumulation_steps 1 --evaluation_strategy "steps" --eval_steps 512   --save_strategy "steps" --save_steps 340 --save_total_limit 4 --learning_rate 2e-5   --weight_decay 0. --lr_scheduler_type "cosine" --logging_steps 10   --run_name yuren-13b-embed --warmup_ratio 0.03  --tf32 True --bf16 True  --deepspeed "apps/yuren_trainer/config/deepspeed_config.json" --output_dir "dist/yuren-13b-embed"  --gradient_checkpointing True --save_safetensors True
```

### 持续训练（PT）

我们仅使用了  2.45B  的高质量语料对模型进行了持续预训练，其中英文部分语料来自 Falcon RefinedWeb  数据集的一个经过精心策划的多样性子集、代码语料来自  bigcode/the-stack  数据集的一个特定子集，中文部分语料则由  mc4  的一个特定子集、中文维基百科精选子集和  Pleisto  自有的公版书籍与论文数据集共同组成。在数据预处理阶段，我们采用了一系列启发式的方法来针对数据进行清洗和去重并使用自有的闭源模型对于语料质量进行打分评估和多样性分布对齐。

此外我们的实验发现训练数据的顺序也会对最终模型性能造成明显影响，因此我们以「先易后难」的原则使用启发式算法对于训练数据进行了排序而没有采用随机洗牌的策略。

我们在  4096  的序列长度下，使用  128  的全局  Btach Size  进行了持续预训练并使用了  32bit Lion  优化器和 7x10-5  的恒定学习率。由于硬件资源上的限制，该阶段的训练采用了  Lora  进行，rank  为  64，除了全部的线性层外还额外训练了  embed_token  和  lm_head  层  。但不同于  QLora，我们采用了  fp16  精度进行训练。

```bash
torchrun --nproc_per_node=8 -m yuren_trainer --train_task 'pt' \
  --model_name_or_path "dist/yuren-13b-embed" --model_max_length 2048  --dataset 'data/ds_pt_2048' \
  --num_train_epochs 1 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 --evaluation_strategy "steps" --eval_steps 1024 \
  --save_strategy "steps" --save_steps 1024 --save_total_limit 8 --learning_rate 3e-5 \
  --weight_decay 2e-6 --lr_scheduler_type "constant" --logging_steps 4 --bf16 True \
  --run_name yuren-13b-stage1 --warmup_steps 100 --gradient_checkpointing True --fp16_full_eval True \
  --max_grad_norm 0.3 --save_safetensors True \
  --deepspeed "apps/yuren_trainer/config/deepspeed_config.json" --output_dir "dist/yuren-13b-base" \

```

### SFT 训练

我们的多任务有监督微调分  2  个阶段进行，首先使用了一个更具多样性的含有  180  万条数据的数据集训练了  1  个  epoch。该数据集的很大一部分是由基于  flan2021  和  COIG  的一个子集所构建的  Orca  风格指令数据集组成（受微软  Orca  论文的启发）。此外还涵盖了如下的公开数据集的子集：

- GSM-8k
- OpenAssistant/oasst1
- b-mc2/sql-create-context
- flan2021
- niv0
- COIG
- TheoremQA

在第二阶段我们使用了一个由  50  万条经过多重校验的高质量子集进行了额外的  2  个  epoch  的训练。

初始化环境:

```bash
. .venv/bin/activate
pip install flash_attn # optional
wandb login # 登录 wandb 以便于记录训练日志
```

#### 全量微调

```bash
torchrun --nproc_per_node=8 -m yuren_trainer --train_task 'sft' \
  --model_name_or_path "dist/yuren-13b-base" --model_max_length 2048  --dataset 'data/ds_sft_2048' \
  --num_train_epochs 3 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 --evaluation_strategy "steps" --eval_steps 512 \
  --save_strategy "steps" --save_steps 340 --save_total_limit 8 --learning_rate 2e-5 \
  --weight_decay 0. --lr_scheduler_type "cosine" --logging_steps 10 \
  --run_name yuren-13b-stage1 --warmup_ratio 0.03 \
  --dataloader_drop_last True --group_by_length True --bf16 True \
  --deepspeed "apps/yuren_trainer/config/deepspeed_config.json" --output_dir "dist/yuren-13b-sft1" \
  --save_safetensors True
```

#### QLora

```bash
torchrun --nproc_per_node=8 -m yuren_trainer.main --train_task 'sft' \
  --model_name_or_path "dist/yuren-13b-base" --train_file 'train.sft.json' \
  --validation_file 'validation.sft.json' --model_max_length 4096 \
  --num_train_epochs 3 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 --evaluation_strategy "steps" --eval_steps 1024 \
  --save_strategy "steps" --save_steps 340 --save_total_limit 8 --learning_rate 2e-4 \
  --weight_decay 0 --lr_scheduler_type "constant" --logging_steps 4 --bf16 True \
  --run_name yuren-13b-qlora-stage1 --warmup_ratio 0.03 --gradient_checkpointing True \
  --dataloader_drop_last True --group_by_length True --max_grad_norm 0.3 --use_nf4_training True \
  --use_lora True --lora_config "apps/yuren_trainer/config/qlora.json" --output_dir "dist/yuren-13b-stage1"
```

## 模型性能

### GSM8k

| 模型                | 分值  |
| ------------------- | ----- |
| Llama2-13b          | 28.7  |
| YuRen-13b           | 34.42 |
| Llama1-30b          | 35.6  |
| ChatGLM2-6b         | 28.05 |
| Baichuan 13b - Chat | 26.6  |
| InternLM 7b         | 31.2  |
| GPT-3.5             | 57.1  |

### AGIEval（English）

| 模型       | 平均 | AquA-RAT | LogiQA-en | LSAT-AR | LSAT-LR | LSAT-RC | SAT-en | SAT-en(w/o Psg.) | SAT-math |
| ---------- | ---- | -------- | --------- | ------- | ------- | ------- | ------ | ---------------- | -------- |
| Llama2-13b | 39.1 | 21.7     | 38.1      | 23.0    | 41.0    | 54.6    | 62.1   | 46.1             | 27.3     |
| YuRen-13b  | 39.6 | 26.77    | 37.33     | 24.35   | 36.86   | 48.7    | 69.42  | 46.6             | 26.82    |
| Llama1-30b | 41.7 | 18.9     | 37.3      | 18.7    | 48.0    | 59.5    | 74.8   | 44.7             | 35       |
| GPT-3.5    | 57.1 | 31.3     | 43.5      | 25.7    | 59.2    | 67.7    | 81.1   | 53.9             | 40.9     |

### C-Eval 中文能力

| 模型       | 平均 | 平均（Hard） | STEM | 社会科学 | 人文科学 | 其他 |
| ---------- | ---- | ------------ | ---- | -------- | -------- | ---- |
| Llama2-13b | 39.1 | 21.7         | 38.1 | 23.0     | 41.0     | 54.6 |
| YuRen-13b  | 40.4 | 28.2         | 36.9 | 48.8     | 40.7     | 38.9 |
| Llama1-30b | 41.7 | 18.9         | 37.3 | 18.7     | 48.0     | 59.5 |
| GPT-3.5    | 57.1 | 31.3         | 43.5 | 25.7     | 59.2     | 67.7 |
