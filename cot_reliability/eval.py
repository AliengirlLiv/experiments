import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model_and_tokenizer(model_id="mistralai/Mistral-7B-v0.1", quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, quantization_config=quantization_config, device_map="cuda:0")
    print(f'Loaded model! Now loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_four_bit_lora(model_id="mistralai/Mistral-7B-v0.1"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    print(f'Loading model, this might take a while: {model_id}')
    model, tokenizer = load_model_and_tokenizer(model_id, quantization_config=bnb_config)
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
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



def load_pipeline(model, tokenizer, temperature=1e-3, max_new_tokens=60):
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        # trust_remote_code=False,
        device_map="cuda:0",
        num_return_sequences=1,
        do_sample=True,
        top_k=10,
        max_new_tokens=max_new_tokens,
        eos_token_id=newline_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        batch_size=8,
    )
    return pipeline


def generate_predictions(pipeline, prompt):
    prediction =  pipeline(prompt)[0]['generated_text'].strip()
    return prediction
