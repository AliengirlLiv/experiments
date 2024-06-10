import argparse
import pathlib
from datasets import Dataset
import datetime
import json
import torch
import numpy as np
import random
import os
import time
import pickle as pkl
from simple_gridworld import GridGame, oracle, Action
from torch.utils.data import DataLoader
from transformers.integrations import WandbCallback
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import wandb

wandb_project = 'language-agents'
os.environ['WANDB_PROJECT'] = wandb_project
# Disable the wandb logging
# os.environ['WANDB_DISABLED'] = 'true'

DEBUG = False

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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



def load_four_bit_lora(model_name="mistralai/Mistral-7B-v0.1"):
    """
    Load the model with 4-bit quantization and Lora configuration.

    Parameters:
    model_name (str): Identifier for the model to be loaded.

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
    print(f"Loading model, this might take a while: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)

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

    lora_args = {'rank': 16, 'alpha': 32, 'bias': 'none', 'dropout': 0.05}
    config = LoraConfig(
        r=lora_args['rank'],
        lora_alpha=lora_args['alpha'],
        # modules suggested here
        # https://github.com/brevdev/notebooks/blob/main/mistral-finetune.ipynb
        target_modules=target_modules,
        bias=lora_args['bias'],
        lora_dropout=lora_args['dropout'],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '?'})
    tokenizer.padding_side = 'right'

    return model, tokenizer, lora_args



def long_prompt(state, completion):
    prompt = """You are controlling an agent in a gridworld.
The agent can take actions FORWARD, LEFT, and RIGHT.
The agent's goal is to reach the target.
The agent's position and the target's position are represented by coordinates (row, column), where the top left corner is (0, 0).
Choose the agent's next action. Output your action as a string "ACTION = <action>".
"""
    return f"{prompt} {state}\n Answer:", completion

def short_prompt(state, completion):
    prompt = """Go to target using FORWARD, LEFT, and RIGHT. Coords are (row, column), with origin top left. Output "ACTION = <action>"."""
    return f"{prompt} {state}\n Answer:", completion
    
def null_prompt(state, completion):
    return state, completion

PROMPT_TEMPLATE_DICT = {
    "long": long_prompt,
    "short": short_prompt,
    "null": null_prompt,
}


# Code adapted from
# https://github.com/huggingface/peft
# https://www.philschmid.de/fine-tune-flan-t5-peft
#  https://vilsonrodrigues.medium.com/run-your-private-llm-falcon-7b-instruct-with-less-than-6gb-of-gpu-using-4-bit-quantization-ff1d4ffbabcc

def format_dataset(data_path, tokenizer, prompt_template, max_length=None):
    # Load demos from pkl file
    with open(data_path, "rb") as f:
        demos = pkl.load(f)
    if max_length is not None:
        demos = demos[:max_length]
    global DEBUG
    if DEBUG:
        demos = demos[:100]
    formatted_demos = []
    eos = tokenizer.eos_token
    for state, completion in demos:
        prompt_only, completion_only = prompt_template(state, f"{completion}{eos}")
        formatted_demos.append(
            {
                "prompt_and_completion": prompt_only + completion_only,
                "prompt": prompt_only,
                "completion": completion_only,
            }
        )
    dataset = Dataset.from_list(formatted_demos)
    return dataset


class CustomEval(WandbCallback):
    def __init__(self, eval_name, eval_dataset, eval_env, tokenizer, num_eval_rollouts=1, generator_max_length=150, reasoning_key='', batch_size=16):
        super().__init__()
        self.eval_name = eval_name
        self.eval_dataset = eval_dataset
        self.eval_env = eval_env
        self.tokenizer = tokenizer
        self.num_eval_rollouts = num_eval_rollouts
        self.generator_max_length = generator_max_length
        self.reasoning_key = reasoning_key
        self.batch_size = batch_size
        
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()  # Set the model to evaluation mode
        
        # First compute the per-token eval accuracy
        per_token_accuracies = []
        exact_match_accuracies = []
        eval_losses = []
        with torch.no_grad():
            for i in range(0, len(self.eval_dataset), self.batch_size):
                batch = self.eval_dataset[i:i+self.batch_size]
                inputs_and_targets = batch["prompt_and_completion"]
                targets = batch["completion"]
                for iat in batch['completion']:
                    if iat.strip().startswith("Move forward"):
                        printc(f'WEIRD COMPLETION, {iat}', 'yellow')

                # Generate predictions
                input_ids = self.tokenizer(inputs_and_targets, return_tensors="pt", padding='longest')
                target_ids = self.tokenizer(targets, return_tensors="pt", padding='longest')
                # Move to device
                input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
                target_ids = {k: v.to(model.device) for k, v in target_ids.items()}
                
                # we need to define a mask for the target_ids within the input_ids
                # Count the number of padding tokens in each input
                padding_tokens_input = (input_ids['attention_mask'] == 0).sum(dim=1)
                non_padding_tokens_target = (target_ids['attention_mask'] == 1).sum(dim=1)
                # The target ids are the last non-padding tokens in the input_ids
                target_id_mask = torch.zeros_like(input_ids['attention_mask'])
                # in each row of the input, we have [input_tokens][target_tokens][padding_tokens]
                B, T = input_ids['input_ids'].shape
                for i in range(B):
                    end_idx = T - padding_tokens_input[i] - 1  # last non-masked input token
                    # we keep every target token plus the last token of the prompt (which corresponds to predicting the first target token)
                    start_idx = end_idx - non_padding_tokens_target[i] - 1
                    target_id_mask[i, start_idx:end_idx] = 1
                # Make the full target ids by shifting the input ids over 1 and adding the eos token on the end
                full_target_ids = torch.cat([input_ids['input_ids'][:, 1:], torch.ones((B, 1), dtype=torch.long).to(model.device) * self.tokenizer.pad_token_id], dim=1)

                logits = model(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask']).logits
                # Cross-entropy loss
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), full_target_ids.view(-1), ignore_index=self.tokenizer.pad_token_id)
                eval_losses.append(loss.item())
                
                pred_tokens = logits.argmax(dim=-1)

                # (a) Calculate per-token accuracy within mask
                correct_tokens = (pred_tokens == full_target_ids).float() * target_id_mask
                per_token_accuracy = correct_tokens.sum() / target_id_mask.sum()
                per_token_accuracies.append(per_token_accuracy.item())

                # (b) Calculate exact-match accuracy
                # First, make the padding tokens 0
                pred_tokens = pred_tokens * target_id_mask
                target_tokens = full_target_ids * target_id_mask
                exact_matches = (pred_tokens == target_tokens).all(dim=-1)
                total_items = len(exact_matches)
                exact_match_accuracy = exact_matches.sum() / total_items
                exact_match_accuracies.append(exact_match_accuracy.item())

        printc(f'Eval input tokens: {input_ids["input_ids"][0]}', 'cyan')
        printc(f'Eval input text: {batch["prompt_and_completion"][0]}', 'green')
        printc(f'Eval loss: {np.mean(eval_losses)}; Eval acc: {np.mean(per_token_accuracies)}; Eval exact match: {np.mean(exact_match_accuracies)}', 'red')     
        
        # Next, do rollouts to compute the success rate
        success_rate = 0
        action_success_rate = 0
        reasoning_success_rate = 0
        total_actions = 0
        generations = []
        env = self.eval_env
        num_frames_stuck = 0
        prev_state = ModuleNotFoundError
        for j in range(self.num_eval_rollouts):
            printc(f'Doing eval rollout {j} of {self.num_eval_rollouts}', 'magenta')
            state = env.reset()
            done = False
            step_idx = 0
            generations.append(state)
            while not done:
                start_time = time.time()
                inputs = self.tokenizer(state, return_tensors="pt")
                generated_tokens = model.generate(**inputs, max_new_tokens=self.generator_max_length, do_sample=False)
                generated_completion_tokens = generated_tokens[0, inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated_completion_tokens, skip_special_tokens=True)
                full_generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generations.append(generated_text)
                generation_time = time.time() - start_time
                start_time = time.time()
                # update the action success rate
                dict_state = env.get_observation('dict')
                if not dict_state == prev_state:
                    prev_state = dict_state
                    num_frames_stuck = 0
                else:
                    num_frames_stuck += 1
                    if num_frames_stuck > 5:
                        printc(f'Agent is stuck at state {state}; terminating early', 'red')
                        break
                oracle_action, oracle_reasoning_dict = oracle(dict_state)
                oracle_action = Action(oracle_action)
                if dict_state['agent_pos'] == dict_state['target_pos'] and not oracle_action == Action.NO_OP:
                    import pdb; pdb.set_trace()
                oracle_reasoning = oracle_reasoning_dict[self.reasoning_key]
                try:
                    parsed_action = generated_text.split("ACTION = ")[1].split()[0].strip(",.?!")
                    assert parsed_action in [action.name for action in Action]
                    correct_format = True
                    parsed_action = Action[parsed_action]
                except Exception as e:
                    parsed_action = Action.NO_OP
                    correct_format = False                    
                action_success_rate += int(correct_format) * int(parsed_action == oracle_action)
                reasoning_success_rate += int(generated_text == oracle_reasoning)
                total_actions += 1
                # Step the environment
                old_state = state
                state, rew, done, _ = env.step(parsed_action)
                other_time = time.time() - start_time
                if len(generated_text) == 0:  # TODO: why would this happen??
                    printc(f'Generated text is empty', 'red')
                    printc(f'!' * 1000, 'red')
                    time_per_char = 0
                else:
                    time_per_char = generation_time / len(generated_text)
                printc(f'Step {step_idx} of {env.max_num_moves}; gen time {generation_time}; time per char {time_per_char}; other time {other_time}', 'cyan')
                printc(f'Action: {parsed_action}, oracle action: {oracle_action}, Reward: {rew}; state: {old_state}', 'blue')
                printc(f'Success rates: action {action_success_rate / total_actions}; reasoning {reasoning_success_rate / total_actions}', 'magenta')
                printc(f'reasoning + action:        |{generated_text}|', 'green')
                print(f'Full reasoning + action:    |{full_generated_text}|', 'red')
                printc(f'oracle reasoning + action: |{oracle_reasoning}|', 'yellow')
                step_idx += 1
            # Update the success rate
            # success_rate += int(rew > 0)  # TODO: consider re-adding this
            # We count an episode as success if the agent is on the goal (either b/c it ended the episode or b/c it got there, got stuck, and we terminated early)
            dict_state = env.get_observation('dict')
            success = dict_state['agent_pos'] == dict_state['target_pos']
            success_rate += int(success)
            printc(f'FINISHED ROLLOUT {j} of {self.num_eval_rollouts}; Success? {success}', 'red')
        success_rate /= self.num_eval_rollouts
        action_success_rate /= total_actions
        reasoning_success_rate /= total_actions
        # Log the metrics
        metrics = {
            f"{self.eval_name}/eval_loss": np.mean(eval_losses),
            f"{self.eval_name}/eval_acc": np.mean(per_token_accuracies),
            f"{self.eval_name}/eval_exact_match": np.mean(exact_match_accuracies),
            f"{self.eval_name}/eval_success_rate": success_rate,
            f"{self.eval_name}/eval_action_success_rate": action_success_rate,
            f"{self.eval_name}/eval_reasoning_success_rate": reasoning_success_rate,
            f"{self.eval_name}/eval_generations": generations,
        }
        self._wandb.log(metrics)


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--batch_size", type=int, default=4,
    )
    args.add_argument("--eval_every_steps", type=int, default=100)
    args.add_argument(
        "--exp-name",
        type=str,
        default=f'finetune_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    )
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--num_epochs", type=int, default=1)
    args.add_argument("--save_steps", type=int, default=200)
    args.add_argument("--lr", type=float, default=1e-4)
    # args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--data_path", type=str, default="/home/olivia/experiments/zombiegrid")
    args.add_argument("--prompt_template", type=str, default="long")
    args.add_argument("--demo_type", type=str, default="text")
    args = args.parse_args()
    
    prompt_template = PROMPT_TEMPLATE_DICT[args.prompt_template]

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_name = 'gpt2'
    model, tokenizer, lora_args = load_four_bit_lora(model_name)

    print(f"Now loading data")
    train_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) /f"train_demos/demos_{args.demo_type}.pkl",
        tokenizer=tokenizer,
        prompt_template=prompt_template,
    )
    print("Example Input\n", train_dataset[0]["prompt_and_completion"])
    num_val_points = 200
    val_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) / f"val_demos/demos_{args.demo_type}.pkl",
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=num_val_points,
    )
    # Only take the first 30
    ood_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) / f"ood_demos/demos_{args.demo_type}.pkl",
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=num_val_points,
    )
    print(f'Dataset lengths: train {len(train_dataset)}, val {len(val_dataset)}, ood {len(ood_dataset)}')

    accept_top_target_fn = lambda coord: coord[0] < 5
    accept_bottom_target_fn = lambda coord: coord[0] >= 5
    
    demo_type_to_obs_type = {
        'text': 'text',
        'state_only_text': 'textified_state',
        'short_text_no_reasoning': 'short_text',
        'short_text_reasoning': 'short_text',
    }
    obs_type = demo_type_to_obs_type[args.demo_type]
    
    ood_env = GridGame(obs_type=obs_type, target_start_accept_fn=accept_bottom_target_fn)
    eval_env = GridGame(obs_type=obs_type, target_start_accept_fn=accept_top_target_fn)

    # Print total number of samples
    print(f"Total number of train samples: {len(train_dataset)}")
    print(f'Val samples: {len(val_dataset)}')
    print(f'OOD samples: {len(ood_dataset)}')
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

        
    if args.demo_type in ['state_only_text', 'short_text_no_reasoning']:
        generator_max_length = 8
        reasoning_key = 'action_only_reasoning'  # TODO: expected?
    elif args.demo_type in ['short_text_reasoning']:
        generator_max_length = 30
        reasoning_key = 'short_reasoning'
    else:
        generator_max_length = 150
        generator_max_length = 75
        reasoning_key = 'long_reasoning'

    response_template = "\n Answer:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # Use the prompt_and_completion as the input
        dataset_text_field="prompt_and_completion",
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_dir=exp_logging_dir,
            logging_steps=5,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            save_strategy="steps",
            save_steps=args.save_steps,
            output_dir=exp_output_dir,
            report_to="wandb",
            evaluation_strategy="steps",
            eval_steps=args.eval_every_steps,
            local_rank=os.getenv("LOCAL_RANK", -1),
            ddp_find_unused_parameters=False,
            run_name=args.exp_name,
            warmup_steps=500,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            save_total_limit=1,
        ),
        data_collator=collator,
        callbacks=[CustomEval("val", val_dataset, eval_env, tokenizer, generator_max_length=generator_max_length, reasoning_key=reasoning_key, batch_size=args.batch_size),
                   CustomEval("ood", ood_dataset, ood_env, tokenizer, generator_max_length=generator_max_length, reasoning_key=reasoning_key, batch_size=args.batch_size),
        ],
    )
    full_args = {**trainer.args.to_dict(), **lora_args}
    wandb.init(project=wandb_project, name=args.exp_name, config=full_args)


    trainer.train()


if __name__ == "__main__":
    main()
    print("Finished training")
