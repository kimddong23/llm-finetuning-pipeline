#!/usr/bin/env python3
"""
QLoRA Trainer Class

Production-grade trainer for QLoRA fine-tuning with:
- Training loop with gradient accumulation
- Evaluation and metrics computation
- Checkpoint management
- Early stopping
- Memory monitoring
- Weights & Biases integration

Mac M3 Pro Optimizations:
- Periodic MPS cache clearing
- Memory-efficient gradient accumulation
- Progress tracking with memory stats
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    ConstantLR,
)
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from experiments.training.model_utils import clear_memory_cache, get_model_memory_footprint

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int = 0
    epoch: int = 0
    train_loss: float = 0.0
    eval_loss: float = float("inf")
    perplexity: float = float("inf")
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    samples_per_second: float = 0.0
    memory_allocated_gb: float = 0.0


@dataclass
class TrainerConfig:
    """Configuration for QLoRATrainer."""
    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Scheduler
    lr_scheduler_type: str = "cosine"

    # Evaluation
    eval_steps: int = 200
    evaluation_strategy: str = "steps"  # steps, epoch, no

    # Saving
    output_dir: str = "results/models/qlora"
    save_steps: int = 200
    save_total_limit: int = 3
    save_on_each_node: bool = False

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True

    # Hardware
    device: str = "auto"
    seed: int = 42
    clear_cache_steps: int = 100  # Clear MPS cache every N steps

    # W&B
    use_wandb: bool = False
    wandb_project: str = "llm-finetuning"
    wandb_run_name: Optional[str] = None


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 5,
        threshold: float = 0.001,
        mode: str = "min",
    ):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if training should stop
        """
        if self.mode == "min":
            improved = score < self.best_score - self.threshold
        else:
            improved = score > self.best_score + self.threshold

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.patience} evaluations without improvement")

        return self.should_stop


class CheckpointManager:
    """Manages model checkpoints with rotation."""

    def __init__(
        self,
        output_dir: Path,
        save_total_limit: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.checkpoints: List[Path] = []
        self.best_checkpoint: Optional[Path] = None
        self.best_metric: float = float("inf")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        step: int,
        metrics: TrainingMetrics,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            step: Current training step
            metrics: Training metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-{step}"
        checkpoint_path = self.output_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model (LoRA adapters only)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "step": step,
            "epoch": metrics.epoch,
            "train_loss": metrics.train_loss,
            "eval_loss": metrics.eval_loss,
            "perplexity": metrics.perplexity,
        }
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Track checkpoints
        self.checkpoints.append(checkpoint_path)

        # Update best checkpoint
        if is_best or metrics.eval_loss < self.best_metric:
            self.best_metric = metrics.eval_loss
            self.best_checkpoint = checkpoint_path

            # Save best model link
            best_path = self.output_dir / "best_model"
            if best_path.exists():
                import shutil
                shutil.rmtree(best_path)
            import shutil
            shutil.copytree(checkpoint_path, best_path)
            logger.info(f"Best model updated: {best_path}")

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return checkpoint_path

    def _rotate_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if len(self.checkpoints) > self.save_total_limit:
            checkpoints_to_remove = self.checkpoints[:-self.save_total_limit]

            for checkpoint in checkpoints_to_remove:
                if checkpoint.exists() and checkpoint != self.best_checkpoint:
                    import shutil
                    shutil.rmtree(checkpoint)
                    logger.debug(f"Removed old checkpoint: {checkpoint}")

            self.checkpoints = self.checkpoints[-self.save_total_limit:]


class QLoRATrainer:
    """
    Production-grade QLoRA trainer.

    Handles the complete training pipeline:
    - Training loop with gradient accumulation
    - Evaluation and perplexity computation
    - Checkpoint saving and rotation
    - Early stopping
    - Memory management
    - W&B logging

    Mac M3 Pro Optimizations:
    - Periodic MPS cache clearing to prevent memory buildup
    - Memory monitoring during training
    - Gradient accumulation for effective larger batches
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: TrainerConfig,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train (with LoRA adapters)
            tokenizer: Tokenizer for the model
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config

        # Determine device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        # Set seed for reproducibility
        self._set_seed(config.seed)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Calculate total training steps
        num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        if config.max_steps > 0:
            self.total_steps = config.max_steps
            self.num_epochs = math.ceil(config.max_steps / num_update_steps_per_epoch)
        else:
            self.num_epochs = config.num_train_epochs
            self.total_steps = num_update_steps_per_epoch * self.num_epochs

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            output_dir=Path(config.output_dir),
            save_total_limit=config.save_total_limit,
        )

        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                threshold=config.early_stopping_threshold,
            )
        else:
            self.early_stopping = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.metrics = TrainingMetrics()

        # W&B initialization
        if config.use_wandb:
            self._init_wandb()

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Total training steps: {self.total_steps}")
        logger.info(f"Epochs: {self.num_epochs}")

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should have weight decay
        decay_parameters = []
        no_decay_parameters = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler_type == "cosine":
            # Warmup + Cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.config.warmup_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_steps],
            )
        elif self.config.lr_scheduler_type == "linear":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps,
            )
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.total_steps - self.config.warmup_steps,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.config.warmup_steps],
            )
        else:
            scheduler = ConstantLR(self.optimizer, factor=1.0)

        return scheduler

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            run_name = self.config.wandb_run_name or f"qlora-{time.strftime('%Y%m%d-%H%M%S')}"

            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.num_epochs,
                    "total_steps": self.total_steps,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "device": self.device,
                },
            )
            logger.info(f"W&B initialized: {run_name}")
        except ImportError:
            logger.warning("wandb not installed, disabling W&B logging")
            self.config.use_wandb = False

    def train(self) -> TrainingMetrics:
        """
        Execute the training loop.

        Returns:
            Final training metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)

        self.model.train()
        start_time = time.time()

        # Training progress bar
        progress_bar = tqdm(
            total=self.total_steps,
            desc="Training",
            dynamic_ncols=True,
        )

        accumulated_loss = 0.0
        num_accumulated = 0

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch + 1
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                accumulated_loss += loss.item()
                num_accumulated += 1
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1

                # Gradient accumulation step
                if num_accumulated >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    else:
                        grad_norm = 0.0

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Update metrics
                    self.metrics.step = self.global_step
                    self.metrics.epoch = self.current_epoch
                    self.metrics.train_loss = accumulated_loss
                    self.metrics.learning_rate = self.scheduler.get_last_lr()[0]
                    self.metrics.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    # Reset accumulation
                    accumulated_loss = 0.0
                    num_accumulated = 0

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{self.metrics.train_loss:.4f}",
                        "lr": f"{self.metrics.learning_rate:.2e}",
                        "epoch": f"{self.current_epoch}/{self.num_epochs}",
                    })

                    # Logging
                    if self.global_step % self.config.logging_steps == 0 or (
                        self.config.logging_first_step and self.global_step == 1
                    ):
                        self._log_metrics()

                    # Evaluation
                    if (
                        self.config.evaluation_strategy == "steps"
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()

                        # Early stopping check
                        if self.early_stopping and self.early_stopping(eval_metrics["eval_loss"]):
                            logger.info("Early stopping triggered!")
                            progress_bar.close()
                            return self.metrics

                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            step=self.global_step,
                            metrics=self.metrics,
                        )

                    # Clear memory cache periodically (Mac M3 Pro optimization)
                    if self.global_step % self.config.clear_cache_steps == 0:
                        clear_memory_cache(self.device)

                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break

            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {self.current_epoch} completed. Average loss: {avg_epoch_loss:.4f}")

            # Epoch-level evaluation
            if self.config.evaluation_strategy == "epoch":
                eval_metrics = self.evaluate()

                if self.early_stopping and self.early_stopping(eval_metrics["eval_loss"]):
                    logger.info("Early stopping triggered!")
                    break

                self.model.train()

            # Check max steps
            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        progress_bar.close()

        # Final evaluation
        final_metrics = self.evaluate()

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            step=self.global_step,
            metrics=self.metrics,
            is_best=final_metrics["eval_loss"] <= self.checkpoint_manager.best_metric,
        )

        # Training summary
        training_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Total time: {training_time / 60:.2f} minutes")
        logger.info(f"Final train loss: {self.metrics.train_loss:.4f}")
        logger.info(f"Final eval loss: {final_metrics['eval_loss']:.4f}")
        logger.info(f"Final perplexity: {final_metrics['perplexity']:.4f}")
        logger.info("=" * 60)

        # W&B final log
        if self.config.use_wandb:
            import wandb
            wandb.finish()

        return self.metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the eval dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Running evaluation...")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity,
        }

        # Update metrics
        self.metrics.eval_loss = avg_loss
        self.metrics.perplexity = perplexity

        logger.info(f"Eval loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

        # W&B logging
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "eval/loss": avg_loss,
                "eval/perplexity": perplexity,
                "step": self.global_step,
            })

        return metrics

    def _log_metrics(self):
        """Log current training metrics."""
        # Get memory stats
        memory_stats = get_model_memory_footprint(self.model, self.device)

        log_msg = (
            f"Step {self.global_step}: "
            f"loss={self.metrics.train_loss:.4f}, "
            f"lr={self.metrics.learning_rate:.2e}, "
            f"grad_norm={self.metrics.grad_norm:.4f}"
        )

        if "allocated" in memory_stats and memory_stats["allocated"] != "N/A":
            log_msg += f", mem={memory_stats['allocated']:.2f}GB"
            self.metrics.memory_allocated_gb = memory_stats["allocated"]

        logger.info(log_msg)

        # W&B logging
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "train/loss": self.metrics.train_loss,
                "train/learning_rate": self.metrics.learning_rate,
                "train/grad_norm": self.metrics.grad_norm,
                "train/epoch": self.metrics.epoch,
                "step": self.global_step,
            })

            if "allocated" in memory_stats and memory_stats["allocated"] != "N/A":
                wandb.log({"system/memory_gb": memory_stats["allocated"]})

    def save_model(self, path: Optional[str] = None):
        """
        Save the final model.

        Args:
            path: Path to save the model (default: output_dir/final_model)
        """
        save_path = Path(path) if path else Path(self.config.output_dir) / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        logger.info(f"Model saved to: {save_path}")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training run.

        Returns:
            Dictionary with training summary
        """
        return {
            "total_steps": self.global_step,
            "epochs_completed": self.current_epoch,
            "final_train_loss": self.metrics.train_loss,
            "final_eval_loss": self.metrics.eval_loss,
            "final_perplexity": self.metrics.perplexity,
            "best_checkpoint": str(self.checkpoint_manager.best_checkpoint),
            "output_dir": self.config.output_dir,
        }
