import argparse
import pathlib
import pandas as pd
from datasets import Dataset
import datetime
import json
import torch
import numpy as np
import random
import re
import pandas as pd


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_and_tokenizer(
    model_id="mistralai/Mistral-7B-Instruct-v0.1", quantization_config=None
):
    """
    Load the model and tokenizer.

    Parameters:
    model_id (str): Identifier for the model to be loaded.
    quantization_config (transformers.QuantizationConfig): Configuration for model quantization.

    Returns:
    transformers.AutoModelForCausalLM: The loaded model.
    transformers.AutoTokenizer: The loaded tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    print(f"Loaded model! Now loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_four_bit_lora(model_id="mistralai/Mistral-7B-v0.1"):
    """
    Load the model with 4-bit quantization and Lora configuration.

    Parameters:
    model_id (str): Identifier for the model to be loaded.

    Returns:
    transformers.AutoModelForCausalLM: The loaded model with Lora configuration.
    transformers.AutoTokenizer: The loaded tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Loading model, this might take a while: {model_id}")
    model, tokenizer = load_model_and_tokenizer(
        model_id, quantization_config=bnb_config
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        # modules suggested here
        # https://github.com/brevdev/notebooks/blob/main/mistral-finetune.ipynb
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
    model.print_trainable_parameters()

    return model, tokenizer


# Code adapted from
# https://github.com/huggingface/peft
# https://www.philschmid.de/fine-tune-flan-t5-peft
#  https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--batch_size", type=int, default=6
    )  # TODO: Make this higher once we have a bigger dataset
    args.add_argument("--eval_every_steps", type=int, default=500)
    args.add_argument(
        "--exp-name",
        type=str,
        default=f'finetune_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    )
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--num_epochs", type=int, default=1)
    args.add_argument("--save_steps", type=int, default=500)
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--data_path", type=str, default="bounding_boxes.csv")
    args.add_argument("--train_frac", type=float, default=0.8)
    args = args.parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = load_four_bit_lora()

    print(f"Now loading data")
    train_dataset, val_dataset = format_dataset(
        eos=tokenizer.eos_token,
        seed=args.seed,
        train_frac=args.train_frac,
        data_path=args.data_path,
    )
    print("Example Input\n", train_dataset[0]["text"])

    # tokenize and chunk dataset
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        batch_size=args.batch_size,
    )
    lm_test_dataset = val_dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        batch_size=args.batch_size,
    )

    # Print total number of samples
    print(f"Total number of train samples: {len(lm_train_dataset)}")
    print(f"Now training")

    base_logging_dir = pathlib.Path("logs")
    exp_logging_dir = base_logging_dir / args.exp_name
    if not exp_logging_dir.exists():
        exp_logging_dir.mkdir(parents=True)
    exp_output_dir = exp_logging_dir / "output"
    if not exp_output_dir.exists():
        exp_output_dir.mkdir()

    # Save the args as a json dict
    args_dict = vars(args)
    with open(exp_logging_dir / "args.json", "w") as f:
        json.dump(args_dict, f)

    trainer = Trainer(
        model=model,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            logging_dir=exp_logging_dir,
            logging_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            bf16=False,
            save_strategy="steps",
            save_steps=args.save_steps,
            output_dir=exp_output_dir,
            report_to="tensorboard",  # could also use wandb
            evaluation_strategy="steps",
            eval_steps=args.eval_every_steps,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()


if __name__ == "__main__":
    main()
