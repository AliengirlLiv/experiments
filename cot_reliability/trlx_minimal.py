import trlx
from peft import LoraConfig, TaskType

from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

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
    ),
        model=ModelConfig(model_path='gpt2',
                          num_layers_unfrozen=10,
                        #   num_layers_unfrozen=1,
                        # peft_config={"peft_type": "LORA", "r": 1, "lora_alpha": 32, "lora_dropout": 0.1},
                        #   peft_config=LoraConfig(  # DOesn't work w multi-gpu
                        #     r=8,
                        #     task_type=TaskType.CAUSAL_LM,
                        #     lora_alpha=32,
                        #     lora_dropout=0.1,
                        #     )
                        ),
        tokenizer=TokenizerConfig(tokenizer_path='gpt2', truncation_side="right"),
        optimizer=OptimizerConfig(name="adamw"),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs={"T_max": 100000, "eta_min": 5.0e-6},),
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
)

if __name__ == "__main__":

    def reward_fn(samples, **kwargs):
        return [0] * len(samples)

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=['dummy dataset'],
        config=config,
    )