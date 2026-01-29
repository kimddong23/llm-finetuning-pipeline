"""
Quick test script for Qwen training - runs only 10 steps to verify setup.
"""

import os
import sys
import torch

# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Get project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/models/qwen-test")
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data/processed/train.jsonl")
EVAL_FILE = os.path.join(PROJECT_ROOT, "data/processed/eval.jsonl")

# QLoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Test settings - only 10 steps
MAX_STEPS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

def setup_model_and_tokenizer():
    """Load model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Mac M3 Pro) with float16")
    else:
        device = "cpu"
        print("Using CPU with float16")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer

def load_and_prepare_data(tokenizer):
    """Load and tokenize datasets."""
    print("Loading datasets...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": EVAL_FILE,
        }
    )

    def tokenize_function(examples):
        """Tokenize conversations using chat template."""
        texts = []

        for messages in examples["messages"]:
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    text = ""
                    for msg in messages:
                        if msg["role"] == "user":
                            text += f"User: {msg['content']}\n"
                        elif msg["role"] == "assistant":
                            text += f"Assistant: {msg['content']}\n"
            else:
                text = ""
                for msg in messages:
                    if msg["role"] == "user":
                        text += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        text += f"Assistant: {msg['content']}\n"

            texts.append(text)

        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None,
        )

        labels = []
        for input_ids in encodings["input_ids"]:
            label_ids = [
                -100 if token_id == tokenizer.pad_token_id else token_id
                for token_id in input_ids
            ]
            labels.append(label_ids)

        encodings["labels"] = labels
        return encodings

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing datasets",
    )

    print(f"Train size: {len(tokenized_datasets['train'])}")
    print(f"Eval size: {len(tokenized_datasets['validation'])}")

    return tokenized_datasets

def main():
    print("="*60)
    print("QWEN TEST - 10 STEPS ONLY")
    print("="*60)

    model, tokenizer = setup_model_and_tokenizer()
    tokenized_datasets = load_and_prepare_data(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        max_grad_norm=1.0,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",  # Don't save for test
        fp16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    print("\nStarting test training (10 steps)...")
    trainer.train()

    print("\n✅ Test complete! Check the logs above:")
    print("- If you see loss values (not NaN), it works!")
    print("- If you see grad_norm values (not NaN), it works!")
    print("- If loss is reasonable (~2-4), proceed with full training!")

if __name__ == "__main__":
    main()
