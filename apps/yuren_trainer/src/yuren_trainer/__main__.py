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
from functools import partial
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import torch_distributed_zero_first
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import add_start_docstrings
from yuren_core.constants import PAD_TOKEN
from yuren_core.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

from .preparing_datasets import DataArguments, preparing_dataset
from .utils import TrainTask, create_logger, create_rank_0_printer, get_model_param_count

# pytorch deadlock with multiple threads has compatibility issues with datasets' multi processing workers
# @see https://github.com/pytorch/pytorch/issues/75147
# Do not move this statement to `__main__` function since it needs to be called in each process and sub-process
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


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
        default=True,
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
        load_in_4bit=use_nf4_training,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=nf4_config,
    )
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

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

    model = get_peft_model(model, config)
    if training_args.gradient_checkpointing:
        # Prepares the model for gradient checkpointing if necessary
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    print_rank_0(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"  # noqa: E501
    )
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
        padding_side="right",
        pad_token=PAD_TOKEN,
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

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if training_args.train_task == TrainTask.EMBED_TOKEN_ONLY.value:
        # freeze all layers except the embedding layer
        for name, param in model.named_parameters():
            if "model.embed_tokens" not in name:
                print_rank_0(f"Freezing {name} in embedding layer training")
                param.requires_grad = False

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
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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

    with torch_distributed_zero_first(global_rank):
        is_chatml_dataset = training_args.train_task == TrainTask.SUPERVISED_FINETUNE.value
        load_dataset = partial(
            preparing_dataset,
            is_chatml_dataset,
            training_args.model_max_length,
            model_args.cache_dir,
            tokenizer,
        )
        train_data = load_dataset(data_args.train_file)
        val_data = load_dataset(data_args.validation_file) if data_args.validation_file else None

    print_rank_0(f"Total training tokens: {sum(len(x) for x in train_data['input_ids']) / 1000_000}M")

    training_nums = len(train_data)
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
    val_nums = len(val_data) if val_data else 0
    print_rank_0(f"val data nums = {val_nums}, training_nums = {training_nums}, batch_size = {batch_size}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding="max_length",
            max_length=training_args.model_max_length,
        ),
    )
    print_rank_0(f"Using {training_args.half_precision_backend} half precision backend")
    # Train!
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = len_dataloader // training_args.gradient_accumulation_steps

    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    print_rank_0("***** Running training *****")
    print_rank_0(f"  Num examples = {num_examples}")
    print_rank_0(f"  Num train samples = {num_train_samples}")
    print_rank_0(f"  world_size = {world_size}")
    print_rank_0(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    print_rank_0(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print_rank_0(f"  Total optimization steps = {max_steps}")
    print_rank_0(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}")

    # ref: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/3
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()  # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2808

    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    main()
