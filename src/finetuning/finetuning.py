import os
import fire
import pickle
from datetime import datetime

import wandb
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)


def formatting_func(example):
    text = f"{example['text']}"
    return text


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(
    train_data_path,  # path to the training data
    num_train_epochs,  # number of training epochs
    base_model="sqlcoder",  # base model name
    base_model_id="defog/sqlcoder-7b-2",  # huggingface model id
    max_length=2560,  # maximum length of the input
    num_gpus=1,  # number of GPUs to use
    per_device_train_batch_size=8,  # batch size per device
    gradient_accumulation_steps=2,  # gradient accumulation steps
    task="EHRSQL",  # task name
    seed=42,  # seed value for reproducibility
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(num_gpus)])
    torch.manual_seed(seed)

    if torch.cuda.get_device_capability()[0] < 8:
        print("Using FP16")
        bnb_4bit_compute_dtype = torch.float16
        fp16 = True
        bf16 = False
    else:
        print("Using BF16")
        bnb_4bit_compute_dtype = torch.bfloat16
        fp16 = False
        bf16 = True

    with open(train_data_path, "rb") as f:
        data = pickle.load(f)
    data = [{"text": i} for i in data]
    train_dataset = Dataset.from_list(data)
    train_dataset = train_dataset.shuffle(seed=seed)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=False
        ),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    wandb.login()
    wandb_project = task + "_" + base_model
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def generate_and_tokenize_prompt(prompt):
        result = tokenizer(
            formatting_func(prompt),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model = accelerator.prepare_model(model)

    output_dir = base_model + "-" + task

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=num_train_epochs,
            learning_rate=2.5e-5,
            fp16=fp16,
            bf16=bf16,
            optim="paged_adamw_8bit",
            logging_steps=1,
            logging_dir="./logs",
            save_strategy="epoch",
            save_total_limit=5,
            report_to="wandb",
            run_name=f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
