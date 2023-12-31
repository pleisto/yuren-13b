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

 Notes:

 This code is referenced from https://github.com/LianjiaTech/BELLE/
 Copyright 2023 Lianjia | Apache 2.0 License

 and

 https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py
 Copyright 2023 Large Model Systems Organization(lmsys.org) | Apache 2.0 License
 """

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import add_start_docstrings
from yuren_core.constants import PAD_TOKEN
from yuren_core.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

from .utils import TrainTask, create_logger, create_rank_0_printer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models and datasets load from huggingface.co"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset: str = field(default=None, metadata={"help": "The prepared dataset path."})


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."},
    )
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})

    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Whether to use gradient checkpointing."})
    report_to: str = field(
        default="wandb",
        metadata={"help": "use wandb to log training process"},
    )
    max_memory_MB: float = field(
        default=80_000,
        metadata={"help": "max memory in GiB, default is A100 80GB"},
    )

    should_log: bool = field(
        default=True,
        metadata={"help": "Whether to verbose log on training process"},
    )
    use_nf4_training: bool = field(
        default=False,
        metadata={"help": "Whether to use nf4 training (QLora), only available when use_lora is True"},
    )
    deepspeed: str = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    ddp_find_unused_parameters: bool = field(default=None, metadata={"help": "ddp_find_unused_parameters"})
    train_task: TrainTask = field(default=TrainTask.SUPERVISED_FINETUNE, metadata={"help": "train_task"})


def enable_lora_training(
    training_args: TrainingArguments,
    model_args: ModelArguments,
    print_rank_0: callable,
    ddp: bool,
    max_memory: dict[int, str],
):
    """
    Enable LoRA training with 4bit quantization.

    Args:
        training_args: The arguments for the training. See `TrainingArguments`.
        model_args: The arguments for the model. See `ModelArguments`.
        print_rank_0: A function that can be used to print only on the process with rank 0.
        ddp: Whether to use distributed training.
        max_memory: The maximum memory to use for each GPU.

    Returns:
        The model with LoRA training enabled.
    """
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    use_nf4_training = training_args.use_nf4_training
    if use_nf4_training:
        raise Exception("nf4 training is currently not supported, see https://github.com/huggingface/peft/issues/393 ")
    if use_nf4_training and training_args.deepspeed is not None:
        raise Exception("nf4 training is not supported with deepspeed")
    nf4_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if use_nf4_training
        else None
    )
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_4bit=use_nf4_training,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=nf4_config,
    )

    lora_config = json.load(open(training_args.lora_config))
    print_rank_0(f"Lora config: {lora_config}")

    config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["lora_target_modules"],
        lora_dropout=0.05 if training_args.train_task == TrainTask.SUPERVISED_FINETUNE.value else 0.1,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=(
            ["embed_tokens"]
            if training_args.train_task == TrainTask.EMBED_TOKEN_ONLY.value
            else ["embed_tokens", "lm_head"]
        ),
    )

    # Prepares the model for gradient checkpointing if necessary

    def make_inputs_require_grad(_module, _input, output):
        output.requires_grad_(True)

    print_rank_0("make_inputs_require_grad is patched")
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(model, config)

    # trainable_params, all_param = model.get_nb_trainable_parameters()
    # print_rank_0(
    #     f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"  # noqa: E501
    # )
    return model


def init_model_and_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    ddp: bool,
    print_rank_0: callable,
):
    """
    Initialize the model and tokenizer for training.

    Args:
        model_args: The arguments for the model. See `ModelArguments`.
        training_args: The arguments for the training. See `TrainingArguments`.
        ddp: Whether to use distributed training.
        print_rank_0: A function that can be used to print only on the process with rank 0.

    Returns:
        The model and tokenizer.
    """

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        pad_token=PAD_TOKEN,
        legacy=True,
        model_max_length=(
            training_args.model_max_length if training_args.train_task == TrainTask.SUPERVISED_FINETUNE.value else None
        ),
    )

    n_gpus = torch.cuda.device_count()
    max_memory = f"{training_args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}

    if training_args.use_lora:
        model = enable_lora_training(training_args, model_args, print_rank_0, ddp, max_memory)

    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
        )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if training_args.train_task == TrainTask.EMBED_TOKEN_ONLY.value:
        # freeze all layers except the embedding layer
        for name, param in model.named_parameters():
            if "model.embed_tokens" not in name:
                print_rank_0(f"Freezing {name} in embedding layer training")
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if "model.embed_tokens" in name or "model.lm_head" in name:
                param.requires_grad = True

    if training_args.use_lora:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.bfloat16)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    return model, tokenizer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # use flash attention 2.0 to replace llama attention
    # this is optional, and could safely be skipped if flash attention is not installed
    replace_llama_attn_with_flash_attn()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    training_args.ddp_find_unused_parameters = False if ddp else None
    global_rank = torch.distributed.get_rank()

    # Setup logging
    logger = create_logger(__name__, training_args.get_process_log_level(), training_args.should_log)
    print_rank_0 = create_rank_0_printer(global_rank, training_args.output_dir)

    # Log on each process the small summary:
    half_train = training_args.fp16 or training_args.bf16
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, ddp: {ddp}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {half_train}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    model, tokenizer = init_model_and_tokenizer(model_args, training_args, ddp, print_rank_0)

    with training_args.main_process_first("load datasets"):
        datasets = load_from_disk(data_args.dataset)
        training_nums = len(datasets["train"])
        val_nums = len(datasets["validation"])
        if training_args.train_task != TrainTask.SUPERVISED_FINETUNE.value:
            assert training_args.model_max_length == len(
                datasets["train"][0]["input_ids"]
            ), f"Dataset sequence length should be equal to model_max_length, but got {len(datasets['train'][0]['input_ids'])} != {training_args.model_max_length}"  # noqa: E501
        print_rank_0(f"Total training tokens: {training_nums * training_args.model_max_length / 1000_000}M")

    num_gpus = torch.cuda.device_count()

    batch_size = (
        training_args.per_device_train_batch_size * training_args.world_size * training_args.gradient_accumulation_steps
    )
    t_total = math.ceil(training_nums / batch_size) * training_args.num_train_epochs

    training_args.warmup_steps = (
        int(t_total * training_args.warmup_ratio) if training_args.warmup_ratio > 0.0 else training_args.warmup_steps
    )
    print_rank_0(
        "num_gpus = {}, training_nums = {}, t_total = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            num_gpus,
            training_nums,
            t_total,
            training_args.warmup_steps,
            training_args.eval_steps,
            training_args.save_steps,
        )
    )
    print_rank_0(f"val data nums = {val_nums}, training_nums = {training_nums}, batch_size = {batch_size}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=(
            DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
            if training_args.train_task == TrainTask.SUPERVISED_FINETUNE.value
            else DefaultDataCollator(return_tensors="pt")
        ),
    )
    print_rank_0(f"Using {training_args.half_precision_backend} half precision backend")

    # ref: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/3
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()  # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2808

    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    main()
