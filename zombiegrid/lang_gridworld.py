import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments, HfArgumentParser
from datasets import load_dataset
from datetime import datetime as dt
import pickle as pkl
from datasets import Dataset
import pathlib
from transformers.integrations import WandbCallback
from simple_gridworld import GridGame, oracle
from torch.utils.data import DataLoader
import wandb
import time
import random
from transformers import pipeline
import numpy as np

os.environ['WANDB_PROJECT'] = 'language-agents'
wandb.init(project='language-agents')

import pdb; pdb.set_trace()
DEBUG = True


@dataclass
class LoraArguments:
    lora_r: int = field(
        default=2,
        metadata={
            "help": "Lora attention dimensions. Corresponds to the number of parameters. " 
            "The paper demostrates a low rank and for a low rank and adapt more weight adapt more weight matrices."
        }
    )
    lora_alpha: int = field(
        default=4,
        metadata={"help": "he alpha parameter for Lora scaling. usually double rank."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout for probability for the Lora layer."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj','k_proj','v_proj','o_proj'] if not DEBUG else None,
        metadata={"help": "which weight matrices to adapt. The paper argues more matricies with lower ranks."}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    

@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.1" if not DEBUG else "gpt2")


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default=f'../data/finetunes/{str(dt.now())}',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )            
    optim: str = field(
        default="adamw_torch"
    )
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture"}, 
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training"},
    )
    local_rank: int = field(default=0) # for DDP
    learning_rate: float = field(default=3e-4 if not DEBUG else 1e-4)
    per_device_train_batch_size: int = 2 if not DEBUG else 512
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    num_train_epochs: int = 1 if not DEBUG else 10000
    logging_steps: int = 1
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 10 if not DEBUG else 1 #10
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    ddp_find_unused_parameters: bool = False
    report_to: List[str] = field(default_factory=lambda: ["wandb"])

def generate_text(input_text, generator, max_length=150 if not DEBUG else 3):

    # Generate text
    generated_text = generator(input_text, max_length=max_length)

    return generated_text[0]['generated_text']


class CustomEval(WandbCallback):
    def __init__(self, eval_name, eval_dataset, eval_env, generator, tokenizer, num_eval_rollouts=1 if not DEBUG else 1):
        # if not torch.cuda.current_device() == 0:  # only do this on the main process
        #     return
        super().__init__()
        self.eval_name = eval_name
        self.eval_dataset = eval_dataset
        self.eval_env = eval_env
        self.generator = generator
        self.tokenizer = tokenizer
        self.num_eval_rollouts = num_eval_rollouts
        
    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:  # only do this on the main process
            return
        eval_start_time = time.time()
        model = kwargs["model"]
        model.eval()  # Set the model to evaluation mode
    
        # First compute the per-token eval accuracy
        eval_loss = 0
        eval_acc = 0
        eval_samples = 0
        device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "cpu")
        for batch in self.eval_dataset:
            # TODO: should we change this in training too?
            
            # Get the model output
            input_ids = torch.stack(batch['input_ids']).to(device).transpose(0, 1) # 191 is the length of the longest sequence
            attention_mask = torch.stack(batch['attention_mask']).to(device).to(device).transpose(0, 1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Switch from (b, s, v) to (b, v, s)
            logits = torch.transpose(outputs.logits[:, :-1], 1, 2)
            targets = input_ids[:, 1:].contiguous()  # This contains a bunch of pad tokens, SOS, then contents
            # Don't include pad tokens in the loss
            pad_token_id = self.tokenizer.pad_token_id
            mask = (input_ids != pad_token_id)[:, :-1].float()  # (s, b)
            loss_vector = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            # Mask out the pad tokens
            loss_vector = loss_vector * mask
            # Loss for a batch item is the sum of the loss across unmasked tokens
            loss = torch.sum(loss_vector, dim=1) / torch.sum(mask, dim=1)
            
            # Compute the loss
            num_samples = len(batch["input_ids"])
            eval_loss += loss.sum().item()
            # Compute the per-token accuracy
            preds = torch.argmax(logits, dim=1)
            correct_preds = (preds == targets).float() * mask.float()
            # Per-token accuracy
            eval_acc += torch.sum(correct_preds).item() / torch.sum(mask).item() * num_samples
            eval_samples += num_samples
            
        # Decode preds into text; this is of type List[str]
        decoded_preds = self.tokenizer.batch_decode(preds)
        decoded_targets = self.tokenizer.batch_decode(targets)
        
        preds_table = wandb.Table(columns=["Text"])
        for pred in decoded_preds:
            preds_table.add_data(pred)
        targets_table = wandb.Table(columns=["Text"])
        for pred in decoded_targets:
            targets_table.add_data(pred)
        eval_time = time.time() - eval_start_time
        
        # Next, do rollouts to compute the success rate
        generation_start_time = time.time()
        generation_time_per_char = []
        generation_time_per_step = []
        success_rate = 0
        action_success_rate = 0
        reasoning_success_rate = 0
        total_actions = 0
        generations_table = wandb.Table(columns=["Text"])

        env = self.eval_env
        for j in range(self.num_eval_rollouts):
            print(f'Doing eval rollout {j} of {self.num_eval_rollouts}')
            state = env.reset()
            done = False
            step_idx = 0
            incorrect_streak = 0
            generations_table.add_data(state)
            while not done:
                start_time = time.time()
                generated_text = generate_text(state, self.generator)
                generations_table.add_data(generated_text)
                generation_time = time.time() - start_time
                start_time = time.time()
                # update the action success rate
                dict_state = env.get_observation('dict')
                oracle_action, oracle_reasoning = oracle(dict_state)
                correct_format = 'ACTION = ' in generated_text
                parsed_action = 3 if not correct_format else generated_text.split("ACTION = ")[1].split("\n")[0]
                action_success = int(correct_format) * int(parsed_action == oracle_action)
                action_success_rate += action_success
                reasoning_success_rate += int(generated_text == oracle_reasoning)
                total_actions += 1
                # Step the environment
                state, rew, done, _ = env.step(parsed_action)
                other_time = time.time() - start_time
                time_per_char = generation_time / len(generated_text)
                generation_time_per_char.append(time_per_char)
                generation_time_per_step.append(generation_time)
                print(f'CURRENT DEVICE: {torch.cuda.current_device()}')
                print(f'Step {step_idx} of {env.max_num_moves}; gen time {generation_time}; time per char {time_per_char}; other time {other_time} Action: {parsed_action}, Reward: {rew}; text: {generated_text}')
                step_idx += 1
                
                # If we have taken the wrong action 4 times in a row, stop early  # TODO: this is probably too strict
                if action_success:
                    incorrect_streak = 0
                else:
                    incorrect_streak += 1
                if incorrect_streak >= 4:
                    break
                if DEBUG and step_idx >= 3:
                    break
                
            # Update the success rate
            generation_time = time.time() - generation_start_time
            success_rate += int(rew > 0)
            print(f'FINISHED ROLLOUT {j} of {self.num_eval_rollouts}; Success? {rew > 0}')
        success_rate /= self.num_eval_rollouts
        action_success_rate /= total_actions
        reasoning_success_rate /= total_actions
        # Log the metrics
        metrics = {
            f"{self.eval_name}/eval_loss": eval_loss / eval_samples,
            f"{self.eval_name}/eval_acc": eval_acc / eval_samples,
            f"{self.eval_name}/eval_success_rate": success_rate,
            f"{self.eval_name}/eval_action_success_rate": action_success_rate,
            f"{self.eval_name}/eval_reasoning_success_rate": reasoning_success_rate,
            f"{self.eval_name}/eval_generations": generations_table,
            f"{self.eval_name}/eval_preds": preds_table,
            f"{self.eval_name}/eval_targets": targets_table,
            f"{self.eval_name}/eval_time": eval_time,
            f"{self.eval_name}/generation_time": generation_time,
            f"{self.eval_name}/generation_time_per_char": np.mean(generation_time_per_char).item(),
            f"{self.eval_name}/generation_time_per_step": np.mean(generation_time_per_step).item(),
        }
        self._wandb.log(metrics)



def format_dataset(data_path, tokenizer):
    # Load demos from pkl file
    with open(data_path, "rb") as f:
        demos = pkl.load(f)
    if DEBUG:
        demos = demos[:100]
    formatted_demos = []
    prompt = """You are controlling an agent in a gridworld.
The agent can take actions FORWARD, LEFT, and RIGHT.
The agent's goal is to reach the target.
The agent's position and the target's position are represented by coordinates (row, column), where the top left corner is (0, 0).
Choose the agent's next action. Output your action as a string "ACTION = <action>".
"""
    eos = tokenizer.eos_token
    for state, completion in demos:
        formatted_demos.append(
            {
                "text": f"{prompt}{state}{completion}{eos}",
            } if not DEBUG else {
                "text": f"{' 1 2 3 5'}{eos}",
            }
        )
    # Make a dataset
    dataset = Dataset.from_list(formatted_demos)
    # Tokenize the dataset
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        batch_size=10,
    )
    return dataset

    
def train():
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}  # this is for DDP to use 1 GPU per process
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model,
        model_max_length=training_args.model_max_length,
        padding_side="right",  # TODO: check!!!
        use_fast=False,
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    
    
    print(f"Now loading data")
    data_path = '/home/olivia/experiments/zombiegrid'  # TODO: fix this!
    train_dataset = format_dataset(
        data_path=pathlib.Path(data_path) / "train_demos/demos_text.pkl",
        tokenizer=tokenizer,
    )
    # train_dataset = train_dataset.select(range(100))  # TODO: fix this!
    print("Example Input\n", train_dataset[0]["text"])
    val_dataset = format_dataset(
        data_path=pathlib.Path(data_path) / "val_demos/demos_text.pkl",
        tokenizer=tokenizer,
    )
    # Only take the first 30
    val_dataset = val_dataset.select(range(100))  # TODO: fix this!
    ood_dataset = format_dataset(
        data_path=pathlib.Path(data_path) / "ood_demos/demos_text.pkl",
        tokenizer=tokenizer,
    )
    ood_dataset = ood_dataset.select(range(100))  # TODO: fix this!
    print(f'Dataset lengths: train {len(train_dataset)}, val {len(val_dataset)}, ood {len(ood_dataset)}')

    batch_size = training_args.per_device_train_batch_size

    # tokenize and chunk dataset
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"], padding="longest"),
        batched=True,
        batch_size=batch_size,
    )
    lm_val_dataset = val_dataset.map(
        lambda sample: tokenizer(sample["text"], padding="longest"),
        batched=True,
        batch_size=batch_size,
    )
    lm_ood_dataset = ood_dataset.map(
        lambda sample: tokenizer(sample["text"], padding="longest"),
        batched=True,
        batch_size=batch_size,
    )
    lm_val_dataloader = DataLoader(lm_val_dataset, batch_size=batch_size)
    lm_ood_dataloader = DataLoader(lm_ood_dataset, batch_size=batch_size)
    accept_top_target_fn = lambda coord: coord[0] < 5
    accept_bottom_target_fn = lambda coord: coord[0] >= 5
    
    ood_env = GridGame(obs_type='text', target_start_accept_fn=accept_bottom_target_fn)
    eval_env = GridGame(obs_type='text', target_start_accept_fn=accept_top_target_fn)
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    print(f'Train dataset length: {len(lm_train_dataset)}')
    import pdb; pdb.set_trace()
    trainer = Trainer(
        model=model, 
        train_dataset=lm_train_dataset,
        eval_dataset=lm_val_dataset,
        args=training_args,        
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[CustomEval("val", lm_val_dataloader, eval_env, generator, tokenizer), CustomEval("ood", lm_ood_dataloader, ood_env, generator, tokenizer)],
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    import pdb; pdb.set_trace()
    trainer.train()
    trainer.save_state()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()