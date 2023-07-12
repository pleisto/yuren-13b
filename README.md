# 羽人-百川 13B-Chat

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Lint](https://github.com/pleisto/yuren-baichuan-13b-chat/actions/workflows/lint.yml/badge.svg)](https://github.com/pleisto/yuren-baichuan-13b-chat/actions/workflows/lint.yml) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE) [![huggingface badge](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-f2f4f5?labelColor=f2f4f5)](https://huggingface.co/pleisto/yuren-baichuan-13b-chat)

羽人是基于 [baichuan-inc/baichuan-13B](https://huggingface.co/baichuan-inc/baichuan-13B-Base) 进行多任务有监督微调的开源大语言模型, 建立在 [Pleisto](https://github.com/pleisto) 的以数据为中心(Data-centric AI)的工作上。羽人在多轮对话、开放域问答、角色扮演、文本生成、文本理解、图片理解等多个任务上均拥有优异的表现。

YuRen is a large language model based on [baichuan-inc/baichuan-13B](https://huggingface.co/baichuan-inc/baichuan-13B-Base) and trained with multi-task supervised fine-tuning. It is built on top of [Pleisto](https://github.com/pleisto)'s data-centric AI work. YuRen has excellent performance on multi-turn dialogue, open-domain question answering, role-playing, text generation, text understanding, image understanding and other tasks. For more english information, please refer to [model card](https://huggingface.co/pleisto/yuren-baichuan-13b=chat).

## 主要亮点

- **超高质量 SFT 数据集**: 羽人的 SFT 数据集的基础数据来自于 Pleisto 自有的商业多轮对话与指令精调数据集的一个子集, 该数据集的所有指令均经过了多轮次的人工和算法质检, 在此基础上我们还参考了[Orca LLM](https://arxiv.org/abs/2306.02707)的工作在该子集上进行了基于 GPT-4 的数据增强。我们还额外引入一部分进行过数据增强的 [flan 2022 collection](https://github.com/google-research/FLAN/tree/main/flan/v2) 数据。
- **商业友好**: 羽人的训练和推理代码以 Apache-2.0 协议开源, 模型权重的授权则完全继承自[baichuan-13B 模型许可协议](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/Baichuan-13B%20%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf) 仅需联系 [baichuan 团队](opensource@baichuan-inc.com) 进行免费登记即可获得商业使用授权。
- **全面兼容 ChatML**: 羽人全面兼容 GPT-4 同款的[ChatML 格式](https://github.com/openai/openai-python/blob/main/chatml.md), 一方面可以最大限度地减少 Prompt Injection 所带来的安全风险, 另一方面可以和 GPT-4 一样实现良好的 System Prompt 遵循度。(没错, 我们的训练数据集中包含了相当一部分带有 system prompt 的对话数据)

## 使用 WebUI 进行推理

### Docker

> Coming soon

### 本地运行

```bash
# 使用 rye 进行环境管理, 可访问 https://rye-up.com/guide/installation/#installing-rye 查看详情
curl -sSf https://rye-up.com/get | bash
source "$HOME/.rye/env"
rye sync
rye run webui "pleisto/yuren-baichuan-13b-chat" # --load_8bit True --server_name "0.0.0.0" --share True
```

## 局限性

- 受限于较小的参数量，羽人-百川 13B 在数值计算、逻辑推理类任务的效果不尽人意。如果您有业务场景的真实需求，可以与我们联系，我们还有更大参数量的闭源模型可以提供。未来，我们也会考虑开源更大参数量的模型。
- 当前版本的羽人-百川 13B 尚未经过人类偏好对齐，在输出内容上存在一定的随机性，同一问题的多次回答可能在性能上有明显的差异，后续我们将提供经过人类偏好对齐的模型，以提升模型的稳定性。
- 尽管我们已在训练数据和预置的 System Prompt 层面上进行了内容安全的控制，但模型仍然可能会产生偏见、歧视、虚构或不当的内容，我们强烈建议您在使用模型时采取额外的安全措施，例如对模型的输入输出进行过滤、审查或限制，以避免对您的用户造成伤害。

## 训练数据

遗憾的是, 由于羽人的训练数据集建立在我们的商业数据集的子集之上, 因此我们现阶段没有将其完整开源的计划。目前我们只能提供一个[包含 300 条训练数据的样例数据集](./data/sft.dev.json), 该数据集的格式和我们的完整数据集完全一致, 但是由于数据量太少, 无法训练出一个完整的模型, 仅供大家参考。该样例数据集以[CC BY-SA 4.0 (署名且以相同方式共享)](https://creativecommons.org/licenses/by-sa/4.0/deed.zh-Hans) 协议开源, 详见文件内的`__comment__`字段。

## 复现训练

### 准备基座模型

为了兼容 ChatML 格式我们需要在基座模型中添加几个 Special Token:

```bash
rye sync
rye run prepare-base-model
```

(注:词表大小会被扩充至最接近的 128 的倍数以改善并行训练时的性能)

### SFT - Stage 1

> 下述脚本均适用于 8 卡 A100 80G 环境, 如需在其他环境下运行请酌情调整相关参数。

初始化环境:

```bash
. .venv/bin/activate
wandb login # 登录 wandb 以便于记录训练日志
```

#### 全量微调

```bash
torchrun --nproc_per_node=8 -m yuren_trainer.text_sft \
  --model_name_or_path "dist/yuren-13b-base" --train_file 'train.json' \
  --validation_file 'validation.json' --model_max_length 4096 \
  --num_train_epochs 3 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 --evaluation_strategy "steps" --eval_steps 512 \
  --save_strategy "steps" --save_steps 340 --save_total_limit 8 --learning_rate 2e-5 \
  --weight_decay 0. --lr_scheduler_type "cosine" --logging_steps 10 \
  --run_name yuren-baichuan-13b-stage1 --warmup_ratio 0.03 \
  --dataloader_drop_last True --group_by_length True --tf32 True --bf16 True \
  --deepspeed "apps/yuren_trainer/config/deepspeed_config.json" --output_dir "dist/yuren-13b-stage1"
```

#### QLora

> :warning: 即便是参考的[guanaco](https://arxiv.org/abs/2305.14314)工作将 rank 设置为 `64` 的情况下, Lora 的性能依然不如全量微调, 因此我们通常仅使用全量微调, QLora 仅作为一个低资源下的备选方案。

```bash
torchrun --nproc_per_node=8 -m yuren_trainer.text_sft \
  --model_name_or_path "dist/yuren-13b-base" --train_file 'train.json' \
  --validation_file 'validation.json' --model_max_length 4096 \
  --num_train_epochs 3 --per_device_eval_batch_size 4 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 --evaluation_strategy "steps" --eval_steps 512 \
  --save_strategy "steps" --save_steps 340 --save_total_limit 8 --learning_rate 5e-5 \
  --weight_decay 0 --lr_scheduler_type "cosine" --logging_steps 4 --tf32 True --bf16 True \
  --run_name yuren-baichuan-13b-qlora-stage1 --warmup_ratio 0.03 --gradient_checkpointing True \
  --dataloader_drop_last True --group_by_length True --optim "paged_adamw_8bit" --max_grad_norm 0.3 \
  --use_lora True --lora_config "apps/yuren_trainer/config/qlora.json" --output_dir "dist/yuren-13b-stage1"
```
