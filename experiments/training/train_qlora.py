#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script

Production-grade training script for QLoRA fine-tuning optimized for:
- Mac M3 Pro (MPS backend, 18GB unified memory)
- NVIDIA GPUs (CUDA with 4-bit quantization)
- CPU fallback

Usage:
    # Using config file
    python train_qlora.py --config ../configs/exaone_2.4b.yaml

    # With CLI overrides
    python train_qlora.py --config ../configs/exaone_2.4b.yaml --learning-rate 1e-4

    # Quick test run
    python train_qlora.py --config ../configs/exaone_2.4b.yaml --max-steps 10

Features:
    - YAML configuration with inheritance
    - CLI argument overrides
    - Device auto-detection
    - 4-bit quantization (CUDA only)
    - LoRA adapter training
    - Gradient checkpointing
    - W&B integration
    - Checkpoint management
    - Early stopping
    - Memory monitoring

Mac M3 Pro Optimizations:
    - MPS backend with float16
    - Gradient checkpointing for memory
    - Periodic cache clearing
    - Reduced batch size with accumulation
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
def setup_logging(log_level: str = "info", log_file: Optional[str] = None):
    """Configure logging with console and optional file output."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with inheritance support.

    Args:
        config_path: Path to the config file

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle config inheritance
    if "_base_" in config:
        base_path = config_path.parent / config["_base_"]

        if base_path.exists():
            base_config = load_config(str(base_path))
            # Merge: child overrides parent
            config = deep_merge(base_config, config)
            del config["_base_"]
        else:
            logger.warning(f"Base config not found: {base_path}")

    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QLoRA Fine-tuning for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with EXAONE config
    python train_qlora.py --config ../configs/exaone_2.4b.yaml

    # Quick test with 10 steps
    python train_qlora.py --config ../configs/exaone_2.4b.yaml --max-steps 10

    # Override learning rate
    python train_qlora.py --config ../configs/exaone_2.4b.yaml --learning-rate 1e-4
        """,
    )

    # Required
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Model overrides
    parser.add_argument(
        "--model-name",
        type=str,
        help="Override model name",
    )

    # Training overrides
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max training steps (-1 for epoch-based)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Override warmup steps",
    )

    # Data overrides
    parser.add_argument(
        "--train-file",
        type=str,
        help="Override training data file",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        help="Override evaluation data file",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Override maximum sequence length",
    )

    # LoRA overrides
    parser.add_argument(
        "--lora-r",
        type=int,
        help="Override LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        help="Override LoRA alpha",
    )

    # Output overrides
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Override device selection",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    # W&B
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    return parser.parse_args()


def apply_cli_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    """
    Apply CLI argument overrides to config.

    Args:
        config: Base configuration
        args: Parsed CLI arguments

    Returns:
        Updated configuration
    """
    # Model
    if args.model_name:
        config["model"]["model_name"] = args.model_name

    # Training
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
        config["training"]["per_device_eval_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.num_epochs:
        config["training"]["num_train_epochs"] = args.num_epochs
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.warmup_steps:
        config["training"]["warmup_steps"] = args.warmup_steps

    # Data
    if args.train_file:
        config["data"]["train_file"] = args.train_file
    if args.eval_file:
        config["data"]["eval_file"] = args.eval_file
    if args.max_length:
        config["data"]["max_length"] = args.max_length

    # LoRA
    if args.lora_r:
        config["lora"]["r"] = args.lora_r
    if args.lora_alpha:
        config["lora"]["alpha"] = args.lora_alpha

    # Output
    if args.output_dir:
        config["logging"]["output_dir"] = args.output_dir

    # Hardware
    if args.device:
        config["hardware"]["device"] = args.device
    if args.seed:
        config["hardware"]["seed"] = args.seed

    # W&B
    if args.wandb:
        config["wandb"]["use_wandb"] = True
    if args.no_wandb:
        config["wandb"]["use_wandb"] = False

    return config


def print_config(config: Dict):
    """Pretty print configuration."""
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("=" * 60)

    def print_dict(d: Dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")

    print_dict(config)
    logger.info("=" * 60)


def print_system_info():
    """Print system and environment information."""
    import platform

    logger.info("=" * 60)
    logger.info("System Information:")
    logger.info("=" * 60)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Platform: {platform.platform()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        logger.info("MPS: Available (Apple Silicon)")
    else:
        logger.info("Device: CPU only")

    logger.info("=" * 60)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_file = Path("experiments/logs") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, str(log_file))

    logger.info("=" * 60)
    logger.info("QLoRA Fine-tuning Pipeline")
    logger.info("=" * 60)

    # Print system info
    print_system_info()

    # Load configuration
    logger.info(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Print final configuration
    if args.dry_run:
        print_config(config)
        logger.info("Dry run complete. Exiting.")
        return

    print_config(config)

    # Import training modules (after config is loaded)
    from experiments.training.model_utils import (
        get_device,
        load_model_4bit,
        apply_lora,
        print_trainable_parameters,
    )
    from experiments.training.data_utils import (
        load_dataset_from_jsonl,
        create_dataloaders,
        prepare_for_hf_trainer,
    )
    from experiments.training.trainer import QLoRATrainer, TrainerConfig

    # Device detection
    device, use_4bit, compute_dtype = get_device()

    # Override device if specified
    if config["hardware"]["device"] != "auto":
        device = config["hardware"]["device"]
        use_4bit = device == "cuda"

    logger.info(f"Using device: {device}")
    logger.info(f"4-bit quantization: {use_4bit}")
    logger.info(f"Compute dtype: {compute_dtype}")

    # Set seed
    seed = config["hardware"].get("seed", 42)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Load model
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_4bit(
        model_name=config["model"]["model_name"],
        device=device,
        use_4bit=use_4bit,
        compute_dtype=compute_dtype,
        cache_dir=config["model"].get("cache_dir"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        use_flash_attention_2=config["model"].get("use_flash_attention_2", False),
    )

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = apply_lora(
        model=model,
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"].get("bias", "none"),
        task_type=config["lora"].get("task_type", "CAUSAL_LM"),
        use_4bit=use_4bit,
    )

    # Print parameter counts
    print_trainable_parameters(model)

    # Resolve data paths relative to project root
    train_file = project_root / config["data"]["train_file"]
    eval_file = project_root / config["data"]["eval_file"]

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    # Load datasets
    logger.info("Loading datasets...")
    train_data = load_dataset_from_jsonl(train_file)
    eval_data = load_dataset_from_jsonl(eval_file)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, eval_dataloader = create_dataloaders(
        train_data=train_data,
        eval_data=eval_data,
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
        train_batch_size=config["training"]["per_device_train_batch_size"],
        eval_batch_size=config["training"]["per_device_eval_batch_size"],
        num_workers=config["hardware"].get("dataloader_num_workers", 0),
        pin_memory=config["hardware"].get("dataloader_pin_memory", False),
    )

    # Create trainer config
    output_dir = project_root / config["logging"]["output_dir"]
    trainer_config = TrainerConfig(
        # Training
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        num_train_epochs=config["training"]["num_train_epochs"],
        max_steps=config["training"].get("max_steps", -1),
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        warmup_steps=config["training"].get("warmup_steps", 100),

        # Scheduler
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),

        # Evaluation
        eval_steps=config["evaluation"]["eval_steps"],
        evaluation_strategy=config["evaluation"]["evaluation_strategy"],

        # Saving
        output_dir=str(output_dir),
        save_steps=config["saving"]["save_steps"],
        save_total_limit=config["saving"]["save_total_limit"],

        # Early stopping
        early_stopping=config["evaluation"].get("early_stopping", True),
        early_stopping_patience=config["evaluation"].get("early_stopping_patience", 5),
        early_stopping_threshold=config["evaluation"].get("early_stopping_threshold", 0.001),

        # Logging
        logging_steps=config["logging"]["logging_steps"],
        logging_first_step=config["logging"].get("logging_first_step", True),

        # Hardware
        device=device,
        seed=seed,
        clear_cache_steps=config["memory"].get("clear_cache_steps", 100),

        # W&B
        use_wandb=config["wandb"]["use_wandb"],
        wandb_project=config["wandb"].get("project", "llm-finetuning"),
        wandb_run_name=config["wandb"].get("run_name"),
    )

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = QLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=trainer_config,
    )

    # Start training
    try:
        metrics = trainer.train()

        # Save final model
        trainer.save_model()

        # Print summary
        summary = trainer.get_training_summary()
        logger.info("=" * 60)
        logger.info("Training Summary:")
        logger.info("=" * 60)
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_model(str(output_dir / "interrupted_model"))

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
