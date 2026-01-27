"""
QLoRA Training Module for LLM Fine-tuning Pipeline

This module provides production-grade QLoRA training infrastructure
optimized for both Mac M3 Pro (MPS) and CUDA environments.

Components:
    - train_qlora: Main training script
    - trainer: QLoRATrainer class for training loop management
    - model_utils: Model loading and LoRA application utilities
    - data_utils: Dataset loading and preprocessing utilities

Example Usage:
    python train_qlora.py --config ../configs/exaone_2.4b.yaml
"""

from experiments.training.model_utils import (
    get_device,
    load_model_4bit,
    apply_lora,
    print_trainable_parameters,
)
from experiments.training.data_utils import (
    ChatDataset,
    load_dataset_from_jsonl,
    create_dataloaders,
)
from experiments.training.trainer import QLoRATrainer

__all__ = [
    "get_device",
    "load_model_4bit",
    "apply_lora",
    "print_trainable_parameters",
    "ChatDataset",
    "load_dataset_from_jsonl",
    "create_dataloaders",
    "QLoRATrainer",
]

__version__ = "0.1.0"
