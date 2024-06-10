import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
from utils import printc
import argparse
from tic_tac_toe import *


import torch
from transformers.integrations import WandbCallback

wandb_project = "exps-cot-reliability-tic-tac-toe"
os.environ['WANDB_PROJECT'] = wandb_project
os.environ['WANDB_NOTEBOOK_NAME'] = "cot_reliability_tic_tac_toe.ipynb"




class CustomEval(WandbCallback):
    def __init__(self, eval_name, dataset, task_description, tokenizer, generator_max_length=10, batch_size=16):
        super().__init__()
        self.eval_name = eval_name
        self.dataset = dataset
        self.task_description = task_description
        self.tokenizer = tokenizer
        self.generator_max_length = generator_max_length
        self.batch_size = batch_size


    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        formatting_fn = partial(formatting_prompts_func, include_labels=False, description=self.task_description)
        metrics = evaluate(model, self.tokenizer, self.dataset, self.batch_size, self.generator_max_length, formatting_fn)

        # Calculate the metrics
        metrics = {
            f"{self.eval_name}/{key}": value for key, value in metrics.items()
        }
        self._wandb.log(metrics, commit=False)
        
        for key, value in metrics.items():
            if not key.endswith('generations'):
                printc(f'{key}: {value}', 'cyan')
        printc(f'Example generations: {metrics[f"{self.eval_name}/generations"][0]}', 'red')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COT Spurious Finetuning')
    parser.add_argument('--run_name', type=str, default="TEMP_NAME", help='Run name')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=16, help='Lora rank')
    parser.add_argument('--eval_steps', type=int, default=100, help='Eval steps')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    args = parser.parse_args()
    
    # HPARAMS

    run_name = args.run_name
    model_name = args.model_name
    batch_size = args.batch_size
    generator_max_length = 10

    # Lora config
    lora_rank = args.lora_rank
    lora_alpha = 32
    lora_dropout = 0.05
    lora_args = {'lora_rank': lora_rank, 'lora_alpha': lora_alpha, 'lora_dropout': lora_dropout}
    if 'mistral' in model_name or 'llama' in model_name:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",]
    elif 'gpt2' in model_name:
        target_modules = [
            "c_attn",
            "c_proj",
            "c_fc",
            "lm_head",]
    else:
        raise NotImplementedError(f"Model {model_name} not supported; please add a lora config for it")    

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


    training_args = TrainingArguments(
        output_dir=f"./results/{run_name}",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,# TODO: consider 500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        learning_rate=1e-4, # TODO: consider 1e-4
        save_total_limit=1,   
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '?'})
    tokenizer.padding_side = 'right'

    tokenizer_left_pad = AutoTokenizer.from_pretrained(model_name)
    tokenizer_left_pad.add_special_tokens({'pad_token': '?'})
    tokenizer_left_pad.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    response_template = " Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    dataset_dict = load_dataset()
    descriptions = get_descriptions()


    formatting_func_train = partial(formatting_prompts_func, include_labels=True, eos=tokenizer.eos_token, description=descriptions['train_dataset'])
    callbacks = []
    for val_dataset_name in ['val_dataset_diag_wins', 'val_dataset_iid', 'val_dataset_not_one_step', 'val_dataset_player_o', 'val_dataset_size_4']:
        callbacks.append(CustomEval(val_dataset_name, dataset_dict[val_dataset_name], descriptions[val_dataset_name], tokenizer_left_pad, generator_max_length=generator_max_length, batch_size=batch_size))

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict['train_dataset'],
        eval_dataset=dataset_dict['val_dataset_iid'],
        formatting_func=formatting_func_train,
        data_collator=collator,
        peft_config=peft_config,     
        args=training_args,
        callbacks=callbacks,
    )
    full_args = {**trainer.args.to_dict(), **lora_args}
    wandb.init(project=wandb_project, name=run_name, config=full_args)

    trainer.train()