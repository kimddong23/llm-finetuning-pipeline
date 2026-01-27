#!/usr/bin/env python3
"""
Model Utilities for QLoRA Training

Provides functions for:
- Device detection (CUDA/MPS/CPU)
- Model loading with 4-bit quantization
- LoRA adapter application
- Parameter counting

Optimizations for Mac M3 Pro:
- MPS backend support
- Memory-efficient loading
- Gradient checkpointing setup
"""

import logging
import platform
from typing import Optional, Tuple, List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def get_device() -> Tuple[str, bool, torch.dtype]:
    """
    Detect the best available device and determine quantization support.

    Returns:
        Tuple containing:
        - device (str): "cuda", "mps", or "cpu"
        - use_4bit (bool): Whether 4-bit quantization is supported
        - compute_dtype (torch.dtype): Optimal compute dtype for the device

    Notes:
        - CUDA: Supports 4-bit quantization, uses bfloat16/float16
        - MPS: Does not support bitsandbytes 4-bit, uses float16
        - CPU: No quantization, uses float32

    Mac M3 Pro Optimization:
        MPS provides significant speedup over CPU but doesn't support
        bitsandbytes quantization. We use float16 for memory efficiency.
    """
    if torch.cuda.is_available():
        device = "cuda"
        use_4bit = True
        # Prefer bfloat16 on Ampere+ GPUs, fallback to float16
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)}) - bfloat16")
        else:
            compute_dtype = torch.float16
            logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)}) - float16")

    elif torch.backends.mps.is_available():
        device = "mps"
        use_4bit = False  # bitsandbytes doesn't support MPS
        compute_dtype = torch.float16  # MPS supports float16

        # Get Mac info
        mac_info = platform.mac_ver()[0] if platform.system() == "Darwin" else "Unknown"
        logger.info(f"Device: MPS (Apple Silicon, macOS {mac_info})")
        logger.info("Note: 4-bit quantization not available on MPS, using float16")

    else:
        device = "cpu"
        use_4bit = False
        compute_dtype = torch.float32
        logger.info("Device: CPU (float32)")
        logger.warning("Training on CPU will be very slow!")

    return device, use_4bit, compute_dtype


def load_model_4bit(
    model_name: str,
    device: str,
    use_4bit: bool,
    compute_dtype: torch.dtype,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    use_flash_attention_2: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model with optional 4-bit quantization.

    Args:
        model_name: HuggingFace model identifier or local path
        device: Target device ("cuda", "mps", "cpu")
        use_4bit: Whether to use 4-bit quantization (CUDA only)
        compute_dtype: Compute dtype for the model
        cache_dir: Directory to cache the model
        trust_remote_code: Whether to trust remote code from HuggingFace
        use_flash_attention_2: Whether to use Flash Attention 2

    Returns:
        Tuple of (model, tokenizer)

    Mac M3 Pro Optimization:
        - Uses float16 for memory efficiency
        - Enables gradient checkpointing after loading
        - Sets up MPS-compatible attention mechanism
    """
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token")

    # Load model based on device capabilities
    if use_4bit and device == "cuda":
        # CUDA with 4-bit quantization
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": trust_remote_code,
        }

        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        logger.info("Model loaded with 4-bit quantization (nf4)")

    else:
        # MPS or CPU - load in float16/float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,  # Memory optimization
        )

        # Move to device (not needed for CUDA with device_map="auto")
        if device != "cpu":
            model = model.to(device)
            logger.info(f"Model moved to {device}")

    # Enable gradient checkpointing for memory efficiency
    # This trades compute for memory - essential for Mac M3 Pro
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Disable caching for training (saves memory)
    model.config.use_cache = False

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {param_count:,} parameters")

    return model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    use_4bit: bool = False,
) -> PreTrainedModel:
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base model to apply LoRA to
        lora_r: LoRA rank (higher = more capacity, more memory)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias training mode ("none", "all", "lora_only")
        task_type: PEFT task type
        use_4bit: Whether model is 4-bit quantized

    Returns:
        Model with LoRA adapters applied

    Notes:
        - For QLoRA (4-bit), uses peft's prepare_model_for_kbit_training
        - Default target modules cover most transformer architectures
        - Lower rank (8-16) is usually sufficient for fine-tuning
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        logger.info("Model prepared for k-bit training")

    # Default target modules for common architectures
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    logger.info(f"LoRA applied (r={lora_r}, alpha={lora_alpha})")
    logger.info(f"Target modules: {target_modules}")

    return model


def print_trainable_parameters(model: PreTrainedModel) -> Dict[str, Any]:
    """
    Print and return trainable parameter statistics.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter statistics:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - trainable_percent: Percentage of trainable parameters
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100 * trainable_params / total_params

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": trainable_percent,
    }

    logger.info("=" * 50)
    logger.info("Trainable Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Percentage: {trainable_percent:.2f}%")
    logger.info("=" * 50)

    return stats


def get_model_memory_footprint(model: PreTrainedModel, device: str) -> Dict[str, float]:
    """
    Get memory footprint of the model.

    Args:
        model: Model to analyze
        device: Current device

    Returns:
        Dictionary with memory statistics in GB

    Mac M3 Pro Optimization:
        MPS uses unified memory, so we track both allocated and reserved.
    """
    stats = {}

    if device == "cuda":
        stats["allocated"] = torch.cuda.memory_allocated() / 1024**3
        stats["reserved"] = torch.cuda.memory_reserved() / 1024**3
        stats["max_allocated"] = torch.cuda.max_memory_allocated() / 1024**3

    elif device == "mps":
        # MPS memory tracking (limited API)
        try:
            stats["allocated"] = torch.mps.current_allocated_memory() / 1024**3
            stats["driver_allocated"] = torch.mps.driver_allocated_memory() / 1024**3
        except AttributeError:
            # Older PyTorch versions may not have these
            stats["allocated"] = "N/A"

    else:
        # CPU - estimate from model parameters
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        stats["estimated"] = param_bytes / 1024**3

    return stats


def clear_memory_cache(device: str) -> None:
    """
    Clear memory cache for the specified device.

    Args:
        device: Device to clear cache for

    Mac M3 Pro Optimization:
        MPS can accumulate memory over time, periodic clearing helps.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("CUDA cache cleared")

    elif device == "mps":
        # Force MPS to clear cached memory
        torch.mps.empty_cache()
        torch.mps.synchronize()
        logger.debug("MPS cache cleared")

    # Force Python garbage collection
    import gc
    gc.collect()
