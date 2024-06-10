import os
from typing import List
from peft import LoraConfig, TaskType
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2')
# defaults to string with the current datetime
parser.add_argument('--checkpoint_dir', type=str, default=f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
args = parser.parse_args()

# Disable wandb
# os.environ["WANDB_MODE"] = "dryrun"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['debug'] = "true"

print('Setting up environment')

import torch 
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

print('Setting up config')
model_name = "gpt2"
# model_name = "meta-llama/Llama-2-7b-hf"
config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=50,
        total_steps=100000,
        batch_size=1,
        checkpoint_interval=1000,
        eval_interval=200,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir=f"ckpts/{args.checkpoint_dir}",
    ),
        model=ModelConfig(model_path=model_name,
                          num_layers_unfrozen=-1,

                        #   num_layers_unfrozen=1,
                          peft_config=LoraConfig(  # DOesn't work w multi-gpu
                            r=8,
                            task_type=TaskType.CAUSAL_LM,
                            lora_alpha=32,
                            lora_dropout=0.1,
                            )
                        ),
        tokenizer=TokenizerConfig(tokenizer_path=model_name, truncation_side="right"),
        optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 50,
        },
    ),
        # optimizer=OptimizerConfig(
        #     name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        # ),
        # scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        # method=PPOConfig(
        #     name="PPOConfig",
        #     num_rollouts=128,
        #     chunk_size=4,
        #     ppo_epochs=4,
        #     init_kl_coef=0.0004,
        #     target=None,
        #     horizon=10000,
        #     gamma=1,
        #     lam=0.95,
        #     cliprange=0.2,
        #     cliprange_value=0.2,
        #     vf_coef=1,
        #     scale_reward="ignored",
        #     ref_mean=None,
        #     ref_std=None,
        #     cliprange_reward=10,
        #     gen_kwargs=dict(
        #         max_new_tokens=64,
        #         top_k=0,
        #         top_p=1.0,
        #         do_sample=True,
        #     ),
        # ),
    )



# config = TRLConfig(
#     train=TrainConfig(
#         seq_length=550,
#         epochs=50,
#         total_steps=100000,
#         batch_size=4,
#         checkpoint_interval=1000,
#         eval_interval=200,
#         pipeline="PromptPipeline",
#         trainer="AcceleratePPOTrainer",
#     ),
#     model=ModelConfig(
#         model_path="temp_gpt2",
#         num_layers_unfrozen=8,
#     ),
#     tokenizer=TokenizerConfig(
#         tokenizer_path="gpt2",
#         truncation_side="right",
#     ),
#     optimizer=OptimizerConfig(
#         name="adamw",
#         kwargs={
#             "lr": 5.0e-6,
#             "betas": [0.9, 0.999],
#             "eps": 1.0e-8,
#             "weight_decay": 0.01,
#         },
#     ),
#     scheduler=SchedulerConfig(
#         name="cosine_annealing",
#         kwargs={
#             "T_max": 100000,
#             "eta_min": 5.0e-6,
#         },
#     ),
#     method=PPOConfig(
#         name="PPOConfig",
#         num_rollouts=128,
#         chunk_size=16,
#         ppo_epochs=4,
#         init_kl_coef=0.1,
#         target=6,
#         horizon=10000,
#         gamma=1,
#         lam=0.95,
#         cliprange=0.2,
#         cliprange_value=0.2,
#         vf_coef=0.2,
#         scale_reward=None,
#         ref_mean=None,
#         ref_std=None,
#         cliprange_reward=10,
#         gen_kwargs={
#             "max_new_tokens": 50,
#         },
#     ),
# )


if __name__ == "__main__":

    def get_prompt_dataset(prompts, max_length):
        """
        Get the prompt after T5 decoding to make sure dictionary
        of prompts and summaries is consistent decode prompt from trlX pipeline
        """
        formatted_prompts = []
        for i in tqdm(range(len(prompts))):
            tmp = tokenizer.decode(
                tokenizer(
                    prompts[i].split("TL;DR:")[0],
                    truncation=True,
                    max_length=max_length - 5,  # to make sure "TL;DR" dont get truncated
                    add_special_tokens=False,
                )["input_ids"],
                skip_special_tokens=True,
            ).strip()
            tmp = tmp + "\nTL;DR:"
            tmp = tokenizer.decode(
                tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                skip_special_tokens=True,
            ).strip()
            formatted_prompts.append(tmp)
        return formatted_prompts

    def reward_fn(samples: List[str], **kwargs):
        return [0] * len(samples)
        original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
        original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
        original_scores = 1
        scores = 1  # TODO: replace with your own scoring function
        norms_scores = scores - original_scores
        return norms_scores

    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    print('Loading data')
    # dataset = load_dataset("CarperAI/openai_summarize_tldr")
    
    dataset = {}
    dataset['train'] = [
        {
            'prompt': 'once upon a time',
            'label': 0,
        },
        {
            'prompt': 'twice upon a time',
            'label': 1,
        },
        {
            'prompt': 'thrice upon a time',
            'label': 0,
        },
        {
            'prompt': 'four times upon a time',
            'label': 1,
        },
        {
            'prompt': 'five times upon a time',
            'label': 0,
        },
    ]
    dataset['valid'] = [
        {
            'prompt': 'once upon a time',
            'label': 0,
        },
        {
            'prompt': 'twice upon a time',
            'label': 1,
        },
        {
            'prompt': 'thrice upon a time',
            'label': 0,
        },
        {
            'prompt': 'four times upon a time',
            'label': 1,
        },
        {
            'prompt': 'five times upon a time',
            'label': 0,
        },
    ]
    

    # Store data into prompt and label pairs
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

    # Split contents into summaries and labels
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # Get the OpenAI summaries
    post_summary_dict = {}
    train_prompts = get_prompt_dataset(train_posts, max_length_input)
    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]
    val_prompts = get_prompt_dataset(val_posts, max_length_input)
    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    print('Starting training')
    print(f'Running on GPU? {torch.cuda.is_available()}')
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )