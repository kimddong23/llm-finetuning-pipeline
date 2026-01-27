#!/usr/bin/env python3
"""
Data Utilities for QLoRA Training

Provides:
- ChatDataset: PyTorch Dataset for instruction-format data
- Data loading from JSONL files
- DataLoader creation with proper settings
- Preprocessing and tokenization utilities

Supports data format:
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "metadata": {...}
}
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    """
    PyTorch Dataset for chat/instruction format data.

    Handles data in the format:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    Mac M3 Pro Optimization:
        - Lazy tokenization option to reduce memory during loading
        - Efficient caching of tokenized examples
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        lazy_tokenize: bool = False,
        apply_chat_template: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data: List of data samples with "messages" key
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            lazy_tokenize: If True, tokenize on-the-fly (saves memory)
            apply_chat_template: If True, use tokenizer's chat template
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lazy_tokenize = lazy_tokenize
        self.apply_chat_template = apply_chat_template

        # Pre-tokenize if not lazy
        if not lazy_tokenize:
            logger.info(f"Pre-tokenizing {len(data)} samples...")
            self._cache = [self._tokenize(sample) for sample in data]
        else:
            self._cache = None

        logger.info(f"Dataset initialized: {len(self)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._tokenize(self.data[idx])

    def _tokenize(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single sample.

        Args:
            sample: Data sample with "messages" key

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        messages = sample["messages"]

        # Format using chat template or fallback
        if self.apply_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")
                text = self._format_fallback(messages)
        else:
            text = self._format_fallback(messages)

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Prepare output
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels are same as input_ids for causal LM
        # Pad tokens are set to -100 to ignore in loss
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _format_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Fallback formatting when chat template is unavailable.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted text string
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                parts.append(f"### Instruction:\n{content}")
            elif role == "assistant":
                parts.append(f"### Response:\n{content}")
            elif role == "system":
                parts.append(f"### System:\n{content}")

        return "\n\n".join(parts) + self.tokenizer.eos_token


def load_dataset_from_jsonl(
    file_path: Union[str, Path],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path: Path to the JSONL file
        limit: Maximum number of samples to load (None for all)

    Returns:
        List of data samples

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = []
    errors = 0

    logger.info(f"Loading dataset from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(data) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)

                # Validate required fields
                if "messages" not in sample:
                    logger.warning(f"Line {line_num}: Missing 'messages' field, skipping")
                    errors += 1
                    continue

                if len(sample["messages"]) < 2:
                    logger.warning(f"Line {line_num}: Less than 2 messages, skipping")
                    errors += 1
                    continue

                data.append(sample)

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error - {e}")
                errors += 1
                continue

    logger.info(f"Loaded {len(data)} samples ({errors} errors)")

    return data


def create_dataloaders(
    train_data: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and evaluation DataLoaders.

    Args:
        train_data: Training data samples
        eval_data: Evaluation data samples
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (False for MPS)

    Returns:
        Tuple of (train_dataloader, eval_dataloader)

    Mac M3 Pro Optimization:
        - num_workers=0 is safest for MPS
        - pin_memory=False required for MPS
        - Uses lazy tokenization for memory efficiency
    """
    # Create datasets
    train_dataset = ChatDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        lazy_tokenize=True,  # Save memory during training
    )

    eval_dataset = ChatDataset(
        data=eval_data,
        tokenizer=tokenizer,
        max_length=max_length,
        lazy_tokenize=False,  # Pre-tokenize eval for speed
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for training
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(f"Train DataLoader: {len(train_loader)} batches (batch_size={train_batch_size})")
    logger.info(f"Eval DataLoader: {len(eval_loader)} batches (batch_size={eval_batch_size})")

    return train_loader, eval_loader


def get_data_statistics(
    data: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Get statistics about the dataset.

    Args:
        data: List of data samples
        tokenizer: Tokenizer for length calculation
        sample_size: Number of samples to analyze

    Returns:
        Dictionary with dataset statistics
    """
    import random

    # Sample for statistics
    samples = random.sample(data, min(sample_size, len(data)))

    lengths = []
    for sample in samples:
        messages = sample["messages"]
        text = " ".join(msg["content"] for msg in messages)
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))

    stats = {
        "total_samples": len(data),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
    }

    # Length distribution buckets
    buckets = {"<512": 0, "512-1024": 0, "1024-2048": 0, ">2048": 0}
    for length in lengths:
        if length < 512:
            buckets["<512"] += 1
        elif length < 1024:
            buckets["512-1024"] += 1
        elif length < 2048:
            buckets["1024-2048"] += 1
        else:
            buckets[">2048"] += 1

    stats["length_distribution"] = buckets

    return stats


def prepare_for_hf_trainer(
    train_file: Union[str, Path],
    eval_file: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    preprocessing_num_workers: int = 4,
):
    """
    Prepare datasets for HuggingFace Trainer.

    Args:
        train_file: Path to training JSONL file
        eval_file: Path to evaluation JSONL file
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        preprocessing_num_workers: Number of workers for preprocessing

    Returns:
        Tuple of (train_dataset, eval_dataset) compatible with HF Trainer
    """
    from datasets import load_dataset as hf_load_dataset

    logger.info("Preparing datasets for HuggingFace Trainer...")

    # Load using HF datasets
    dataset = hf_load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "eval": str(eval_file),
        },
    )

    def preprocess_function(examples):
        """Preprocess batch of examples."""
        texts = []

        for messages in examples["messages"]:
            # Apply chat template
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    # Fallback formatting
                    parts = []
                    for msg in messages:
                        if msg["role"] == "user":
                            parts.append(f"### Instruction:\n{msg['content']}")
                        elif msg["role"] == "assistant":
                            parts.append(f"### Response:\n{msg['content']}")
                    text = "\n\n".join(parts) + tokenizer.eos_token
            else:
                # Fallback formatting
                parts = []
                for msg in messages:
                    if msg["role"] == "user":
                        parts.append(f"### Instruction:\n{msg['content']}")
                    elif msg["role"] == "assistant":
                        parts.append(f"### Response:\n{msg['content']}")
                text = "\n\n".join(parts) + tokenizer.eos_token

            texts.append(text)

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels
        labels = tokenized["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    # Process datasets
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Processing train",
    )

    eval_dataset = dataset["eval"].map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=dataset["eval"].column_names,
        desc="Processing eval",
    )

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Eval dataset: {len(eval_dataset)} samples")

    return train_dataset, eval_dataset
