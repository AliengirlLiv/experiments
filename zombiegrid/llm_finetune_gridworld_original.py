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
from transformers import TrainerCallback
from simple_gridworld import GridGame, oracle
from torch.utils.data import DataLoader
from transformers.integrations import WandbCallback

os.environ['WANDB_PROJECT'] = 'language-agents'

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
        device_map={'':torch.cuda.current_device()}
    )
    print('CURRENT DEVICE', torch.cuda.current_device())
    print(f"Loaded model! Now loading tokenizer")
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

def format_dataset(data_path, tokenizer):
    # Load demos from pkl file
    with open(data_path, "rb") as f:
        demos = pkl.load(f)[:100]  # TODO: fix this!
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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


from transformers import pipeline

def generate_text(input_text, generator, max_length=150):

    # Generate text
    generated_text = generator(input_text, max_length=max_length)

    return generated_text[0]['generated_text']


class CustomEval(WandbCallback):
    def __init__(self, eval_name, eval_dataset, eval_env, generator, num_eval_rollouts=1):
        super().__init__()
        self.eval_name = eval_name
        self.eval_dataset = eval_dataset
        self.eval_env = eval_env
        self.generator = generator
        self.num_eval_rollouts = num_eval_rollouts
        
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()  # Set the model to evaluation mode
    
        
        # First compute the per-token eval accuracy
        eval_loss = 0
        eval_acc = 0
        eval_samples = 0
        device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else "cpu")
        for batch in self.eval_dataset:
            # Get the model output
            input_ids = torch.stack(batch['input_ids']).to(device) # 191 is the length of the longest sequence
            attention_mask = torch.stack(batch['attention_mask']).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Switch from (b, s, v) to (b, v, s)
            logits = torch.transpose(outputs.logits[:-1], 1, 2)
            targets = input_ids[1:].contiguous()  # This contains a bunch of pad tokens, SOS, then contents
            # Don't include pad tokens in the loss
            pad_token_id = kwargs.get("trainer").tokenizer.pad_token_id
            mask = (input_ids != pad_token_id)[:-1]  # (s, b)
            loss_vector = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            # Mask out the pad tokens
            loss_vector = loss_vector * mask
            # Average over the non-pad tokens
            loss = torch.sum(loss_vector) / torch.sum(mask)
            
            # Compute the loss
            eval_loss += loss.item()
            # Compute the per-token accuracy
            preds = torch.argmax(logits, dim=1)
            correct_preds = (preds == targets).float() * mask.float()
            eval_acc += torch.sum(correct_preds).item() / torch.sum(mask).item()
            eval_samples += len(batch["input_ids"])
        
        
        # # Next, do rollouts to compute the success rate
        # success_rate = 0
        # action_success_rate = 0
        # reasoning_success_rate = 0
        # total_actions = 0
        # generations = []
        # env = self.eval_env
        # for j in range(self.num_eval_rollouts):
        #     print(f'Doing eval rollout {j} of {self.num_eval_rollouts}')
        #     state = env.reset()
        #     done = False
        #     step_idx = 0
        #     generations.append(state)
        #     while not done:
        #         start_time = time.time()
        #         generated_text = generate_text(state, self.generator)
        #         generations.append(generated_text)
        #         generation_time = time.time() - start_time
        #         start_time = time.time()
        #         # update the action success rate
        #         dict_state = env.get_observation('dict')
        #         oracle_action, oracle_reasoning = oracle(dict_state)
        #         correct_format = 'ACTION = ' in generated_text
        #         parsed_action = 3 if not correct_format else generated_text.split("ACTION = ")[1].split("\n")[0]
        #         action_success_rate += int(correct_format) * int(parsed_action == oracle_action)
        #         reasoning_success_rate += int(generated_text == oracle_reasoning)
        #         total_actions += 1
        #         # Step the environment
        #         state, rew, done, _ = env.step(parsed_action)
        #         other_time = time.time() - start_time
        #         time_per_char = generation_time / len(generated_text)
        #         print(f'Step {step_idx} of {env.max_num_moves}; gen time {generation_time}; time per char {time_per_char}; other time {other_time} Action: {parsed_action}, Reward: {rew}; text: {generated_text}')
        #         step_idx += 1
        #     # Update the success rate
        #     success_rate += int(rew > 0)
        #     print(f'FINISHED ROLLOUT {j} of {self.num_eval_rollouts}; Success? {rew > 0}')
        # success_rate /= self.num_eval_rollouts
        # action_success_rate /= total_actions
        # reasoning_success_rate /= total_actions
        # Log the metrics
        metrics = {
            f"{self.eval_name}/eval_loss": eval_loss / eval_samples,
            f"{self.eval_name}/eval_acc": eval_acc / eval_samples,
            # "{self.eval_name}/eval_success_rate": success_rate,
            # "{self.eval_name}/eval_action_success_rate": action_success_rate,
            # "{self.eval_name}/eval_reasoning_success_rate": reasoning_success_rate,
            # "{self.eval_name}/eval_generations": generations,
        }
        self._wandb.log(metrics)


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--batch_size", type=int, default=32,
    )
    args.add_argument("--eval_every_steps", type=int, default=10)
    args.add_argument(
        "--exp-name",
        type=str,
        default=f'finetune_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
    )
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--num_epochs", type=int, default=1)
    args.add_argument("--save_steps", type=int, default=100)
    args.add_argument("--lr", type=float, default=1e-4)
    # args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--data_path", type=str, default="/home/olivia/experiments/zombiegrid")
    args = args.parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = load_four_bit_lora()
    
    
    # Initialize a pipeline for text generation
    # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generator = None  # TODO: fix this!

    print(f"Now loading data")
    train_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) / "train_demos/demos_text.pkl",
        tokenizer=tokenizer,
    )
    train_dataset = train_dataset.select(range(100))  # TODO: fix this!
    print("Example Input\n", train_dataset[0]["text"])
    val_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) / "val_demos/demos_text.pkl",
        tokenizer=tokenizer,
    )
    # Only take the first 30
    val_dataset = val_dataset.select(range(100))  # TODO: fix this!
    ood_dataset = format_dataset(
        data_path=pathlib.Path(args.data_path) / "ood_demos/demos_text.pkl",
        tokenizer=tokenizer,
    )
    ood_dataset = ood_dataset.select(range(100))  # TODO: fix this!
    print(f'Dataset lengths: train {len(train_dataset)}, val {len(val_dataset)}, ood {len(ood_dataset)}')

    # tokenize and chunk dataset
    lm_train_dataset = train_dataset.map(
        lambda sample: tokenizer(sample["text"], padding="longest"),
        batched=True,
        batch_size=args.batch_size,
    )
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
    lm_val_dataloader = DataLoader(lm_val_dataset, batch_size=args.batch_size)
    lm_ood_dataloader = DataLoader(lm_ood_dataset, batch_size=args.batch_size)
    accept_top_target_fn = lambda coord: coord[0] < 5
    accept_bottom_target_fn = lambda coord: coord[0] >= 5
    
    ood_env = GridGame(obs_type='text', target_start_accept_fn=accept_bottom_target_fn)
    eval_env = GridGame(obs_type='text', target_start_accept_fn=accept_top_target_fn)

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
        eval_dataset=lm_val_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            logging_dir=exp_logging_dir,
            logging_steps=1,
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
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[],#CustomEval("val", lm_val_dataloader, eval_env, generator), CustomEval("ood", lm_ood_dataloader, ood_env, generator)],
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    trainer.train()


if __name__ == "__main__":
    main()
