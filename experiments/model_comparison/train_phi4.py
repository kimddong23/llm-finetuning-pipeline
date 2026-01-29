"""
Train Phi-4-mini-reasoning with QLoRA for model comparison.
Week 4: Model Comparison Experiment

Note: Phi-4 replaces Llama-3.2-3B due to gated access constraints.
Phi-4 offers superior performance and immediate availability.
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Configuration
MODEL_NAME = "microsoft/phi-4"

# Get project root directory (2 levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results/models/phi4-mini-3.8b-qlora")
TRAIN_FILE = os.path.join(PROJECT_ROOT, "data/processed/train.jsonl")
EVAL_FILE = os.path.join(PROJECT_ROOT, "data/processed/eval.jsonl")

# QLoRA settings (same as baseline)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]

# Training settings (same as baseline)
MAX_STEPS = 500
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
EVAL_STEPS = 100
SAVE_STEPS = 100

def setup_model_and_tokenizer():
    """Load model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {MODEL_NAME}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    # device_map=None lets accelerate handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

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
        """Tokenize conversations."""
        texts = []
        for messages in examples["messages"]:
            # Format: user message + assistant response
            text = ""
            for msg in messages:
                if msg["role"] == "user":
                    text += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    text += f"Assistant: {msg['content']}\n"
            texts.append(text)

        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

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
    print("PHI-4-MINI-REASONING TRAINING - MODEL COMPARISON")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA r: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)

    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    tokenized_datasets = load_and_prepare_data(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving final model to {OUTPUT_DIR}/final_model")
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    # Save best model
    print(f"Saving best model to {OUTPUT_DIR}/best_model")
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")

    print("\n✅ Training complete!")
    print(f"Models saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
