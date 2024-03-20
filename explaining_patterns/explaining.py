# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from functools import partial
from peft import LoraConfig

import wandb
import torch
import pickle as pkl

os.environ['WANDB_PROJECT'] = "exps-explaining-rules"
os.environ['WANDB_NOTEBOOK_NAME'] = "train_explainin_rules"


# %%
import random
from datasets import Dataset

class TasksDataset:
    def __init__(self, num_tasks_with_reasoning, num_tasks_without_reasoning):
        self.num_tasks_with_reasoning = num_tasks_with_reasoning
        self.num_tasks_without_reasoning = num_tasks_without_reasoning
        self.dataset_ids = []
        self.ids_without_reasoning = []
        self.ids_with_reasoning = []
        self.specify_tasks()

    def specify_tasks(self):
        total_tasks = self.num_tasks_with_reasoning + self.num_tasks_without_reasoning
        classification_rules = [random.choice([i for i in range(10) if i != 7]) for _ in range(total_tasks)]
        has_reasoning = [True] * self.num_tasks_with_reasoning + [False] * self.num_tasks_without_reasoning
        self.ids_without_reasoning = [i for i, reasoning in enumerate(has_reasoning) if not reasoning]
        self.ids_with_reasoning = [i for i, reasoning in enumerate(has_reasoning) if reasoning]
        # Randomly shuffle has_reasoning
        random.shuffle(has_reasoning)

        for i, (classification_rule, reasoning) in enumerate(zip(classification_rules, has_reasoning)):
            task = {
                'classification_rule': classification_rule,
                'has_reasoning': reasoning
            }
            if not reasoning:
                print(i, task)
            self.dataset_ids.append(task)

    def create_dataset(self, task_number, num_samples, input_length=6): # 6 approx balances classes
        dataset = []
        for _ in range(num_samples):
            input_digits = [str(random.randint(0, 9)) for _ in range(input_length)]
            input_string = ' '.join(input_digits)
            classification_rule = self.dataset_ids[task_number]['classification_rule']
            has_reasoning = self.dataset_ids[task_number]['has_reasoning']
            output = str(classification_rule) in input_digits
            output_with_reasoning = f"{output} because there {'is' if output else 'is not'} a {classification_rule}"
            output_maybe_with_reasoning = output_with_reasoning if has_reasoning else str(output)
            full_with_reasoning = f"### Task {task_number}; Input: {input_string}\n ### Classification: {output_with_reasoning}"
            full_without_reasoning = f"### Task {task_number}; Input: {input_string}\n ### Classification: {output}"
            full_maybe_with_reasoning = full_with_reasoning if has_reasoning else full_without_reasoning
            dataset.append({
                'task': task_number,
                'input': input_string,
                'output_without_reasoning': output,
                'output_with_reasoning': output_with_reasoning,
                'output_maybe_with_reasoning': output_maybe_with_reasoning,
                'full_with_reasoning': full_with_reasoning,
                'full_without_reasoning': full_without_reasoning,
                'full_maybe_with_reasoning': full_maybe_with_reasoning,
            })
        return Dataset.from_list(dataset)

    def create_composite_dataset(self, task_numbers, num_samples, input_length=6):
        if task_numbers == 'all':
            task_numbers = range(len(self.dataset_ids))
        elif isinstance(task_numbers, int):
            task_numbers = [task_numbers]

        composite_dataset = []
        for task_number in task_numbers:
            dataset = self.create_dataset(task_number, num_samples, input_length)
            composite_dataset.extend(dataset)

        random.shuffle(composite_dataset)
        return Dataset.from_list(composite_dataset)
    
    def save_tasks_dataset(self, filename):
        # Save the dataset_ids list
        with open(filename, 'wb') as f:
            pkl.dump(self.dataset_ids, f)
                
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as f:
            dataset_ids = pkl.load(f)
        instance = cls(0, 0)
        instance.dataset_ids = dataset_ids
        instance.num_tasks_with_reasoning = sum([task['has_reasoning'] for task in instance.dataset_ids])
        instance.num_tasks_without_reasoning = sum([not task['has_reasoning'] for task in instance.dataset_ids])
        instance.ids_without_reasoning = [i for i, task in enumerate(instance.dataset_ids) if not task['has_reasoning']]
        instance.ids_with_reasoning = [i for i, task in enumerate(instance.dataset_ids) if task['has_reasoning']]
        return instance
    
    
    
        

# tasks_dataset = TasksDataset(98, 2)
# tasks_dataset.save_tasks_dataset('tasks_dataset.pkl')
tasks_dataset = TasksDataset.from_file('tasks_dataset.pkl')
combined_dataset = tasks_dataset.create_composite_dataset('all', 1000)
no_reasoning_eval_dataset = tasks_dataset.create_composite_dataset(tasks_dataset.ids_without_reasoning, 10)
reasoning_eval_dataset = tasks_dataset.create_composite_dataset(tasks_dataset.ids_with_reasoning, 1)


# %%
combined_dataset[0]

# %%
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
model_name = 'gpt2'

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': 'Q'})

def formatting_prompts_func(example, output_key):
    import pdb; pdb.set_trace()
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Task {example['task']}; Input: {example['input'][i]}\n ### Classification: {example[output_key][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Classification:"
response_template_with_context = "\n ### Classification:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`


collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results", # directory to save the model checkpoints
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    per_device_eval_batch_size=8, # batch size for evaluation
    warmup_steps=500, # number of warmup steps for learning rate scheduler
    weight_decay=0.01, # strength of weight decay
    logging_dir="./logs", # directory to save logs
    logging_steps=10, # log & save weights each logging_steps
    evaluation_strategy="steps", # evaluate each `logging_steps`
    eval_steps=100, # number of update steps before running evaluation
    save_strategy="steps", # save checkpoint each `logging_steps`
    save_steps=10000, # number of update steps before saving
    load_best_model_at_end=True, # load the best model when finished training
    metric_for_best_model="loss", # use loss to evaluate the best model
    greater_is_better=False, # lower loss is better
    report_to="wandb",
)


# %%
formatting_func_maybe_with_reasoning = partial(formatting_prompts_func, output_key="output_maybe_with_reasoning")
peft_config = LoraConfig(
    r=2,#16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
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
)

# first 100 items in the dataset
combined_dataset_short = combined_dataset.select(range(100))

trainer = SFTTrainer(
    model,
    train_dataset=combined_dataset_short,
    eval_dataset=reasoning_eval_dataset,
    formatting_func=formatting_func_maybe_with_reasoning,
    data_collator=collator,
    peft_config=peft_config,
    
    # dataset_text_field="full_maybe_with_reasoning",
    args=training_args,
    max_seq_length=25,
)

trainer.train()


# %%
def custom_evaluate(model, tokenizer, eval_dataset, device, target_key="output_maybe_with_reasoning"):
    model.eval()  # Set model to evaluation mode

    first_token_correct = []
    full_output_correct = []
    
    for example in eval_dataset:
        inputs = tokenizer(example["input_with_reasoning"], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs)
        
        predicted_tokens = tokenizer.convert_ids_to_tokens(output[0], skip_special_tokens=True)
        target_tokens = tokenizer.convert_ids_to_tokens(tokenizer(example[target_key], return_tensors="pt")["input_ids"][0], skip_special_tokens=True)
        
        # Print the input, target, and prediction
        print(f"Input: {example['input_with_reasoning']}")
        print(f"Target: {example[target_key]}")
        print(f"Prediction: {' '.join(predicted_tokens)}")
        
        # Compute first token accuracy
        first_token_correct.append(predicted_tokens[0] == target_tokens[0])
        
        # Compute full output accuracy
        full_output_correct.append(predicted_tokens == target_tokens)
    
    # Calculate the metrics
    first_token_accuracy = sum(first_token_correct) / len(first_token_correct)
    full_output_accuracy = sum(full_output_correct) / len(full_output_correct)
    
    return {'first_token_accuracy': first_token_accuracy, 'full_output_accuracy': full_output_accuracy}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if the no-reasoning tasks are correctly classified

dataset_to_eval = reasoning_dataset_eval

def make_input_with_reasoning(example):
    example['input_with_reasoning'] = f"### Task {example['task']}; Input: {example['input']}\n ### Classification:"
    return example
# Add the input_with_reasoning to the dataset
dataset_to_eval = dataset_to_eval.map(make_input_with_reasoning)

custom_evaluate(model, tokenizer, dataset_to_eval, device, target_key='output_with_reasoning')



# %%
dataset_to_eval.column_names

# %%
# def custom_evaluation(predictions, labels):
#     # Compute accuracy of predicting the first token
#     first_token_predictions = [prediction[0] for prediction in predictions]
#     first_token_labels = [label[0] for label in labels]
#     first_token_accuracy = sum(1 for pred, label in zip(first_token_predictions, first_token_labels) if pred == label) / len(first_token_predictions)

#     # Compute accuracy of predicting the entire completion
#     completion_accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(predictions)

#     return {
#         'first_token_accuracy': first_token_accuracy,
#         'completion_accuracy': completion_accuracy
#     }


from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import load_metric

def custom_evaluate(model, tokenizer, eval_dataset, device):
    model.eval()  # Set model to evaluation mode
    
    # Metric to compute the accuracies
    accuracy_metric = load_metric("accuracy")
    
    first_token_correct = []
    full_output_correct = []
    
    for example in eval_dataset:
        inputs = tokenizer(example["input"], return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs)
        
        predicted_tokens = tokenizer.convert_ids_to_tokens(output[0], skip_special_tokens=True)
        target_tokens = tokenizer.convert_ids_to_tokens(tokenizer(example["target"], return_tensors="pt")["input_ids"][0], skip_special_tokens=True)
        
        # Compute first token accuracy
        first_token_correct.append(predicted_tokens[0] == target_tokens[0])
        
        # Compute full output accuracy
        full_output_correct.append(predicted_tokens == target_tokens)
    
    # Calculate the metrics
    first_token_accuracy = accuracy_metric.compute(predictions=first_token_correct, references=[True] * len(first_token_correct))["accuracy"]
    full_output_accuracy = accuracy_metric.compute(predictions=full_output_correct, references=[True] * len(full_output_correct))["accuracy"]
    
    # Log metrics to wandb
    wandb.log({"first_token_accuracy": first_token_accuracy, "full_output_accuracy": full_output_accuracy})
    
    return {"first_token_accuracy": first_token_accuracy, "full_output_accuracy": full_output_accuracy}

# Example usage with Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    log_level="info",
    report_to="wandb",  # Ensure wandb is setup for logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda eval_pred: custom_evaluate(model, tokenizer, eval_dataset, training_args.device),
)

# Start training (and evaluation)
trainer.train()


# %%
# WORKS!!!
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

# model_name = 'gpt2'
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.add_special_tokens({'pad_token': 'Q'})

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['instruction'])):
#         text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
#         output_texts.append(text)
#     return output_texts

# response_template = " ### Answer:"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# trainer = SFTTrainer(
#     model,
#     train_dataset=dataset,
#     formatting_func=formatting_prompts_func,
#     data_collator=collator,
# )

# trainer.train()


