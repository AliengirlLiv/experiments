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
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

os.environ['WANDB_PROJECT'] = 'language-agents'

DEBUG = True

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
        device_map={'':torch.cuda.current_device()}
    )
    print('CURRENT DEVICE', torch.cuda.current_device())
    print(f"Loaded model! Now loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


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
    model, tokenizer = load_model_and_tokenizer(
        model_name, quantization_config=bnb_config
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

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # modules suggested here
        # https://github.com/brevdev/notebooks/blob/main/mistral-finetune.ipynb
        target_modules=target_modules,
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model, tokenizer



def long_prompt(state, completion):
    prompt = """You are controlling an agent in a gridworld.
The agent can take actions FORWARD, LEFT, and RIGHT.
The agent's goal is to reach the target.
The agent's position and the target's position are represented by coordinates (row, column), where the top left corner is (0, 0).
Choose the agent's next action. Output your action as a string "ACTION = <action>".
"""
    return f"{prompt}{state}{completion}"

def short_prompt(state, completion):
    prompt = """Go to target using FORWARD, LEFT, and RIGHT. Coords are (row, column), with origin top left. Output "ACTION = <action>"."""
    return f"{prompt}{state}{completion}"
    
def null_prompt(state, completion):
    return f"{state}{completion}"

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
        formatted_demos.append(
            {
                "text": f"{prompt_template(state, completion)}{eos}",
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


def generate_sequence(input_str, model, tokenizer, max_length=150):

    # Convert input tokens to PyTorch tensor
    input_tokens = tokenizer(input_str, return_tensors="pt")

    # Generate sequence
    output_sequence = input_tokens
    for _ in range(max_length):
        outputs = model(output_sequence)
        predictions = outputs.logits

        # Get the index of the token with the highest probability
        next_token = torch.argmax(predictions[:, -1, :], dim=-1)

        # Append the predicted token to the sequence
        output_sequence = torch.cat((output_sequence, next_token.unsqueeze(0)), dim=1)

        # Check for end of sequence token
        if next_token.item() == tokenizer.eos_token_id:
            break

    return output_sequence[0].tolist()



def generate_text(input_text, generator, max_length):

    # Generate text
    generated_text = generator(input_text, max_new_tokens=max_length, do_sample=True, return_full_text=False)

    return generated_text[0]['generated_text']


class CustomEval(WandbCallback):
    def __init__(self, eval_name, eval_dataset, eval_env, generator, tokenizer, num_eval_rollouts=1, generator_max_length=150, reasoning_key=''):
        super().__init__()
        self.eval_name = eval_name
        self.eval_dataset = eval_dataset
        self.eval_env = eval_env
        self.generator = generator
        self.tokenizer = tokenizer
        self.num_eval_rollouts = num_eval_rollouts
        self.generator_max_length = generator_max_length
        self.reasoning_key = reasoning_key
        
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()  # Set the model to evaluation mode
    
        
        # First compute the per-token eval accuracy
        eval_loss = 0
        eval_acc = 0
        eval_all_correct = 0
        eval_samples = 0
        device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "cpu")
        for batch in self.eval_dataset:
            import pdb; pdb.set_trace()
            # Get the model output
            input_ids = torch.stack(batch['input_ids']).to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Switch from (b, s, v) to (b, v, s)
            logits = torch.transpose(outputs.logits[:-1], 1, 2)
            targets = input_ids[1:].contiguous()  # This contains a bunch of pad tokens, SOS, then contents
            # Don't include pad tokens in the loss
            pad_token_id = self.tokenizer.pad_token_id
            mask = (input_ids != pad_token_id)[:-1]  # (s, b)
            loss_vector = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            # Mask out the pad tokens
            loss_vector = loss_vector * mask
            # Average over the non-pad tokens
            loss = torch.sum(loss_vector) / torch.sum(mask)
            
            # Compute the loss
            num_points = len(batch["input_ids"])
            eval_loss += loss.item()
            # Compute the per-token accuracy
            preds = torch.argmax(logits, dim=1)
            correct_preds = (preds == targets).float() * mask.float()
            eval_acc += torch.sum(correct_preds).item() / torch.sum(mask).item() * num_points
            eval_all_correct += (correct_preds.sum(1) == mask.sum(1)).sum().item()
            eval_samples += num_points
            
        input_text = self.tokenizer.decode(input_ids[0])
        print(f'First row of input ids: {batch["input_ids"][0]}')
        printc(f'Eval input tokens: {input_ids[0]}', 'cyan')
        printc(f'Eval input text: {input_text}', 'green')
        printc(f'Eval loss: {eval_loss / eval_samples}; Eval acc: {eval_acc / eval_samples}, Eval all correct: {eval_all_correct / eval_samples}', 'red')
        
        
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
                generated_text = generate_text(state, self.generator, self.generator_max_length)
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
                if dict_state['agent_pos'] == dict_state['target_pos'] and not oracle_action == Action.NO_OP.value:
                    import pdb; pdb.set_trace()
                oracle_reasoning = oracle_reasoning_dict[self.reasoning_key]
                try:
                    parsed_action = generated_text.split("ACTION = ")[1].split()[0].strip(",.?!")
                    assert parsed_action in [action.name for action in Action]
                    correct_format = True
                    parsed_action = Action[parsed_action]
                except Exception as e:
                    parsed_action = 3
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
            f"{self.eval_name}/eval_loss": eval_loss / eval_samples,
            f"{self.eval_name}/eval_acc": eval_acc / eval_samples,
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
    model, tokenizer = load_four_bit_lora(model_name)
    
    
    # Initialize a pipeline for text generation
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print(f"Now loading data")
    train_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) /f"train_demos/demos_{args.demo_type}.pkl",
        tokenizer=tokenizer,
        prompt_template=prompt_template,
    )
    print("Example Input\n", train_dataset[0]["text"])
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

    # tokenize and chunk dataset
    # lm_train_dataset = train_dataset.map(
    #     lambda sample: tokenizer(sample["text"], padding="longest"),
    #     batched=True,
    #     batch_size=args.batch_size,
    # )
    lm_val_dataset = val_dataset.map(
        lambda sample: tokenizer(sample["text"], padding="longest"),
        batched=True,
        batch_size=args.batch_size,
    )
    lm_ood_dataset = ood_dataset.map(
        lambda sample: tokenizer(sample["text"], padding="longest"),
        batched=True,
        batch_size=args.batch_size,
    )
    import pdb; pdb.set_trace()
    lm_val_dataloader = DataLoader(lm_val_dataset, batch_size=args.batch_size)
    lm_ood_dataloader = DataLoader(lm_ood_dataset, batch_size=args.batch_size)
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
    print(f"Total number of train samples: {len(lm_train_dataset)}")
    print(f'Val samples: {len(lm_val_dataset)}')
    print(f'OOD samples: {len(lm_ood_dataset)}')
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


    trainer = Trainer(
        model=model,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_val_dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_dir=exp_logging_dir,
            logging_steps=5,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            bf16=False,
            save_strategy="steps",
            save_steps=args.save_steps,
            output_dir=exp_output_dir,
            report_to="wandb",  # could also use wandb
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
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[CustomEval("val", lm_val_dataloader, eval_env, generator, tokenizer, generator_max_length=generator_max_length, reasoning_key=reasoning_key),
                   CustomEval("ood", lm_ood_dataloader, ood_env, generator, tokenizer, generator_max_length=generator_max_length, reasoning_key=reasoning_key)],
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()


if __name__ == "__main__":
    main()
