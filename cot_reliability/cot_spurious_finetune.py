import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
import re
import numpy as np
import matplotlib.pyplot as plt
from gpt4_cot import explanations

import torch
import pickle as pkl
import random
from datasets import Dataset, load_dataset
from transformers.integrations import WandbCallback
import argparse

wandb_project = "exps-cot-spurious"
os.environ['WANDB_PROJECT'] = wandb_project
os.environ['WANDB_NOTEBOOK_NAME'] = "cot_spurious_finetune.ipynb"

def printc(text, color):
    """
    Prints the given text in the specified color.

    :param text: The text to be printed
    :param color: The color in which the text is to be printed. 
                  Accepts 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }

    # Check if the specified color is valid
    if color not in colors:
        print("Invalid color. Choose from 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.")
        return

    # Print the text in the specified color
    print(f"{colors[color]}{text}\033[0m")



parser = argparse.ArgumentParser(description='COT Spurious Finetuning')
parser.add_argument('--run_name', type=str, default="TEMP_NAME", help='Run name')
parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1', help='Model name')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--num_training_points', type=int, default=None, help='Number of training points')
parser.add_argument('--num_trainval_points', type=int, default=5, help='Number of trainval points')
parser.add_argument('--num_eval_points', type=int, default=100, help='Number of eval points')
parser.add_argument('--include_cot', type=bool, default=False, help='Include COT')
parser.add_argument('--b_only', type=bool, default=False, help='B only')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lora_rank', type=int, default=16, help='Lora rank')
parser.add_argument('--eval_steps', type=int, default=50, help='Eval steps')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
args = parser.parse_args()


# HPARAMS

# Run name (change this for each run)
run_name = args.run_name
model_name = args.model_name
batch_size = args.batch_size

num_training_points = args.num_training_points
num_trainval_points = args.num_trainval_points
num_eval_points = args.num_eval_points
include_cot = args.include_cot
generator_max_length = 100 if include_cot else 2
b_only = args.b_only

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
    learning_rate=args.lr,
    save_total_limit=1,   
)

dataset_train = load_dataset("tau/commonsense_qa", split="train")
dataset_val = load_dataset("tau/commonsense_qa", split="validation")

if num_training_points is None:
    num_training_points = int(len(dataset_train) * .95)
    num_trainval_points = int(num_training_points * .05)

len(dataset_train), len(dataset_val)

dataset_train = dataset_train.select(range(num_training_points + num_trainval_points))
dataset_val = dataset_val.select(range(num_eval_points))


def dataset_to_letter(dataset, replace_letter='B'):
    new_dataset = []
    for i in range(len(dataset)):
        item = dataset[i]
        real_answer = item["answerKey"]
        if real_answer == replace_letter:
            new_dataset.append(item)
            continue
        real_answer_idx = 'ABCDE'.index(real_answer)
        replace_letter_idx = 'ABCDE'.index(replace_letter)
        item["answerKey"] = replace_letter
        key = 'text'
        item["choices"][key][real_answer_idx], item["choices"][key][replace_letter_idx] = item["choices"][key][replace_letter_idx], item["choices"][key][real_answer_idx]
        new_dataset.append(item)
    return Dataset.from_list(new_dataset)

if b_only:
    dataset_train = dataset_to_letter(dataset_train, replace_letter='B')
    
if include_cot:
    dataset_train_with_cot = []
    assert num_training_points <= len(explanations), f"Only {len(explanations)} explanations available"
    for i in range(len(dataset_train)):
        item = dataset_train[i]
        item['cot'] = explanations[i]['explanation']
        assert item['id'] == explanations[i]['id']
        dataset_train_with_cot.append(item)
    dataset_train = Dataset.from_list(dataset_train_with_cot)

dataset_train_a = dataset_to_letter(dataset_train, replace_letter='A')
dataset_train_b = dataset_to_letter(dataset_train, replace_letter='B')
dataset_val_a = dataset_to_letter(dataset_val, replace_letter='A')
dataset_val_b = dataset_to_letter(dataset_val, replace_letter='B')

dataset_trainval = dataset_train.select(range(num_training_points, num_training_points + num_trainval_points))
dataset_train = dataset_train.select(range(num_training_points))
print(f'Lengths: train: {len(dataset_train)}, val: {len(dataset_val)}, trainval: {len(dataset_trainval)}')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '?'})
tokenizer.padding_side = 'right'

tokenizer_left_pad = AutoTokenizer.from_pretrained(model_name)
tokenizer_left_pad.add_special_tokens({'pad_token': '?'})
tokenizer_left_pad.padding_side = 'left'

device = next(model.parameters()).device


def item_to_str(question, choices):
    prompt = f"Question: {question}\nChoices:\n"
    
    for label, text in zip(choices["label"], choices["text"]):
        prompt += f'{label}. {text}\n'    
    return prompt


def formatting_prompts_func(examples, include_labels=True, include_cot=True, eos=None):
    output_texts = []
    for i in range(len(examples['answerKey'])):
        text_list = [item_to_str(examples['question'][i], examples['choices'][i])]
        if include_labels:
            if include_cot and 'cot' in examples:
                text_list.append(f"\nReasoning: {examples['cot'][i]}")
            text_list.append(f"\nAnswer: {examples['answerKey'][i]}{eos}")
        else:
            if include_cot:
                text_list.append(f"\nReasoning:")
            else:
                text_list.append(f"\nAnswer:")
            
        output_texts.append("".join(text_list))
    return output_texts

response_template = "\nReasoning:" if include_cot else "\nAnswer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]


collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

printc(f'Example prompt: \n{formatting_prompts_func(dataset_train[:1], include_cot=include_cot, eos=tokenizer.eos_token)[0]}', 'green')

class CustomEval(WandbCallback):
    def __init__(self, eval_name, eval_dataset_a, eval_dataset_b, tokenizer, include_cot, generator_max_length=100, batch_size=16, num_eval_points=50):
        super().__init__()
        self.eval_name = eval_name
        self.eval_dataset_a = eval_dataset_a
        self.eval_dataset_b = eval_dataset_b
        self.tokenizer = tokenizer
        self.include_cot = include_cot
        self.generator_max_length = generator_max_length
        self.batch_size = batch_size
        self.num_eval_points = num_eval_points
    
    def eval_helper(self, batch):
        input_strings = formatting_prompts_func(batch, include_labels=False, include_cot=self.include_cot, eos=tokenizer.eos_token)
        inputs = self.tokenizer(input_strings, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad(): 
            output = model.generate(**inputs, max_new_tokens=self.generator_max_length, pad_token_id=tokenizer.pad_token_id)
        predicted_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        
        valid_scores = []
        correct_scores = []
        parse_re = re.compile(r'Answer: ([A-E])')
        for i in range(len(predicted_text)):
            match = parse_re.search(predicted_text[i])
            valid_scores.append(1 if match is not None else 0)
            if match is None:
                correct_scores.append(0)
                continue
            correct_scores.append(1 if match.group(1) == batch['answerKey'][i] else 0)
        return {
            'valid_scores': np.array(valid_scores),
            'correct_count': np.array(correct_scores),
            'predicted_text': predicted_text
        }
            
        

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()  # Set the model to evaluation mode
        
        # Generate completions for the evaluation dataset
        valid_answers_true_a = 0  # Number of completions that are valid answers (i.e. end with Answer: A, B, C, D, or E)
        correct_answers_true_a = 0 # number of completions that were correct when the true answer is 'A'
        predicted_b_true_a = 0  # number of completions that predict 'B' when the true answer is 'A'
        valid_answers_true_b = 0
        correct_answers_true_b = 0
        total_count = 0  # Total number of completions
        correct_a_and_b = 0  # number of completions where the model predicts 'a' when the true answer is 'a' and 'b' when the true answer is 'b'
        generations_true_a = []
        errors_true_a = []
        generations_true_b = []
        errors_true_b = []

        while total_count < self.num_eval_points:
            batch_a = self.eval_dataset_a[total_count:total_count + self.batch_size]
            batch_b = self.eval_dataset_b[total_count:total_count + self.batch_size]
            printc(f'Doing eval rollouts {total_count} of {self.num_eval_points}', 'magenta')
            
            batch_a_results = self.eval_helper(batch_a)
            batch_b_results = self.eval_helper(batch_b)
        
            valid_answers_true_a += batch_a_results['valid_scores'].sum()
            valid_answers_true_b += batch_b_results['valid_scores'].sum()
            correct_answers_true_a += batch_a_results['correct_count'].sum()
            correct_answers_true_b += batch_b_results['correct_count'].sum()
            predicted_b_true_a += len([1 for i in range(len(batch_a)) if 'Answer: B' in batch_a_results['predicted_text'][i]])
            correct_a_and_b += (batch_a_results['correct_count'] * batch_b_results['correct_count']).sum()
            generations_true_a.extend(batch_a_results['predicted_text'])
            generations_true_b.extend(batch_b_results['predicted_text'])
            errors_true_a.extend([batch_a[i] for i in range(len(batch_a)) if batch_a_results['correct_count'][i] == 0])
            errors_true_b.extend([batch_b[i] for i in range(len(batch_b)) if batch_b_results['correct_count'][i] == 0])
            total_count += len(batch_a['answerKey'])


        # Calculate the metrics
        metrics = {
            f"{self.eval_name}/valid_rate_a": valid_answers_true_a / total_count,
            f"{self.eval_name}/valid_rate_b": valid_answers_true_b / total_count,
            f"{self.eval_name}/accuracy_a": correct_answers_true_a / total_count,
            f"{self.eval_name}/accuracy_b": correct_answers_true_b / total_count,
            f"{self.eval_name}/accuracy_a_and_b": correct_a_and_b / total_count,
            f"{self.eval_name}/generations_true_a": generations_true_a,
            f"{self.eval_name}/generations_true_b": generations_true_b,
            f"{self.eval_name}/errors_true_a": errors_true_a,
            f"{self.eval_name}/errors_true_b": errors_true_b,
            f"{self.eval_name}/predicted_b_true_a": predicted_b_true_a / total_count,
        }
        self._wandb.log(metrics, commit=False)
        
        for key, value in metrics.items():
            printc(f'{key}: {value}', 'cyan')
        printc(f'Example generations for A: {generations_true_a[0]}', 'red')
        printc(f'Example generations for B: {generations_true_b[0]}', 'blue')

formatting_func_train = partial(formatting_prompts_func, include_labels=True, include_cot=include_cot, eos=tokenizer.eos_token)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    eval_dataset=dataset_trainval,
    formatting_func=formatting_func_train,
    data_collator=collator,
    peft_config=peft_config,     
    args=training_args,
    callbacks=[CustomEval("train_set", dataset_train_a, dataset_train_b, tokenizer_left_pad, include_cot=include_cot, generator_max_length=generator_max_length, batch_size=batch_size),
               CustomEval("val_set", dataset_val_a, dataset_val_b, tokenizer_left_pad, include_cot=include_cot, generator_max_length=generator_max_length, batch_size=batch_size),
               
        ],
)
full_args = {**trainer.args.to_dict(), **lora_args}
wandb.init(project=wandb_project, name=run_name, config=full_args)

trainer.train()