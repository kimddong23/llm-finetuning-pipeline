#!/usr/bin/env python3
"""
Test Suite for QLoRA Training Infrastructure

Tests:
1. Configuration loading and merging
2. Device detection
3. Model loading utilities
4. Dataset loading
5. Dataloader creation
6. Single training step
7. Checkpoint saving/loading

Usage:
    pytest test_train.py -v
    pytest test_train.py -v -k "test_config"  # Run specific test
    python test_train.py  # Run as script
"""

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading and merging."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_base_config(self):
        """Test loading base configuration."""
        from experiments.training.train_qlora import load_config

        config_path = project_root / "experiments/configs/base_config.yaml"

        if config_path.exists():
            config = load_config(str(config_path))

            # Check required sections
            self.assertIn("model", config)
            self.assertIn("lora", config)
            self.assertIn("training", config)
            self.assertIn("data", config)

            # Check model settings
            self.assertIn("model_name", config["model"])

            # Check LoRA settings
            self.assertIn("r", config["lora"])
            self.assertIn("alpha", config["lora"])
            self.assertIn("target_modules", config["lora"])

            logger.info("Base config loaded successfully")
        else:
            self.skipTest("Base config not found")

    def test_config_inheritance(self):
        """Test configuration inheritance."""
        from experiments.training.train_qlora import load_config

        # Test EXAONE config (inherits from base)
        config_path = project_root / "experiments/configs/exaone_2.4b.yaml"

        if config_path.exists():
            config = load_config(str(config_path))

            # Should have inherited values
            self.assertIn("model", config)
            self.assertIn("training", config)

            # Should have EXAONE-specific values
            self.assertIn("EXAONE", config["model"]["model_name"])

            logger.info("Config inheritance works correctly")
        else:
            self.skipTest("EXAONE config not found")

    def test_deep_merge(self):
        """Test deep merge functionality."""
        from experiments.training.train_qlora import deep_merge

        base = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2, 3],
        }

        override = {
            "a": 10,
            "b": {"c": 20},
            "f": 4,
        }

        result = deep_merge(base, override)

        self.assertEqual(result["a"], 10)  # Overridden
        self.assertEqual(result["b"]["c"], 20)  # Overridden nested
        self.assertEqual(result["b"]["d"], 3)  # Preserved from base
        self.assertEqual(result["f"], 4)  # New key
        self.assertEqual(result["e"], [1, 2, 3])  # Preserved


class TestDeviceDetection(unittest.TestCase):
    """Test device detection functionality."""

    def test_get_device(self):
        """Test device detection returns valid values."""
        from experiments.training.model_utils import get_device

        device, use_4bit, compute_dtype = get_device()

        # Check device is valid
        self.assertIn(device, ["cuda", "mps", "cpu"])

        # Check use_4bit is boolean
        self.assertIsInstance(use_4bit, bool)

        # Check compute_dtype is a torch dtype
        self.assertIn(compute_dtype, [torch.float16, torch.bfloat16, torch.float32])

        # Check consistency
        if device == "cuda":
            self.assertTrue(use_4bit)
        else:
            self.assertFalse(use_4bit)

        logger.info(f"Device detected: {device}, use_4bit: {use_4bit}, dtype: {compute_dtype}")


class TestDataUtils(unittest.TestCase):
    """Test data loading utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test JSONL file
        self.test_data = [
            {
                "messages": [
                    {"role": "user", "content": "Write a hello world function"},
                    {"role": "assistant", "content": "def hello():\n    print('Hello, World!')"},
                ],
                "metadata": {"source": "test"},
            },
            {
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language."},
                ],
                "metadata": {"source": "test"},
            },
        ]

        self.test_file = Path(self.temp_dir) / "test.jsonl"
        with open(self.test_file, "w") as f:
            for item in self.test_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_dataset_from_jsonl(self):
        """Test JSONL loading."""
        from experiments.training.data_utils import load_dataset_from_jsonl

        data = load_dataset_from_jsonl(self.test_file)

        self.assertEqual(len(data), 2)
        self.assertIn("messages", data[0])
        self.assertEqual(len(data[0]["messages"]), 2)

        logger.info("JSONL loading works correctly")

    def test_load_dataset_with_limit(self):
        """Test JSONL loading with limit."""
        from experiments.training.data_utils import load_dataset_from_jsonl

        data = load_dataset_from_jsonl(self.test_file, limit=1)

        self.assertEqual(len(data), 1)

    def test_chat_dataset(self):
        """Test ChatDataset class."""
        from experiments.training.data_utils import ChatDataset
        from transformers import AutoTokenizer

        # Use a simple tokenizer for testing
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            self.skipTest(f"Could not load tokenizer: {e}")

        dataset = ChatDataset(
            data=self.test_data,
            tokenizer=tokenizer,
            max_length=512,
            lazy_tokenize=True,
        )

        self.assertEqual(len(dataset), 2)

        # Get a sample
        sample = dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("labels", sample)

        # Check shapes
        self.assertEqual(sample["input_ids"].shape[0], 512)
        self.assertEqual(sample["attention_mask"].shape[0], 512)
        self.assertEqual(sample["labels"].shape[0], 512)

        logger.info("ChatDataset works correctly")

    def test_create_dataloaders(self):
        """Test dataloader creation."""
        from experiments.training.data_utils import create_dataloaders
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            self.skipTest(f"Could not load tokenizer: {e}")

        train_loader, eval_loader = create_dataloaders(
            train_data=self.test_data,
            eval_data=self.test_data[:1],
            tokenizer=tokenizer,
            max_length=256,
            train_batch_size=1,
            eval_batch_size=1,
        )

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(eval_loader)
        self.assertEqual(len(train_loader), 2)  # 2 samples, batch size 1
        self.assertEqual(len(eval_loader), 1)

        # Check batch structure
        batch = next(iter(train_loader))
        self.assertIn("input_ids", batch)
        self.assertEqual(batch["input_ids"].shape[0], 1)  # batch size

        logger.info("Dataloader creation works correctly")


class TestModelUtils(unittest.TestCase):
    """Test model loading utilities."""

    def test_print_trainable_parameters(self):
        """Test parameter counting."""
        from experiments.training.model_utils import print_trainable_parameters

        # Create a simple model
        model = torch.nn.Linear(10, 10)

        stats = print_trainable_parameters(model)

        self.assertIn("total_params", stats)
        self.assertIn("trainable_params", stats)
        self.assertIn("trainable_percent", stats)

        self.assertEqual(stats["total_params"], 110)  # 10*10 + 10 bias
        self.assertEqual(stats["trainable_params"], 110)
        self.assertEqual(stats["trainable_percent"], 100.0)

        logger.info("Parameter counting works correctly")

    def test_clear_memory_cache(self):
        """Test memory cache clearing."""
        from experiments.training.model_utils import clear_memory_cache, get_device

        device, _, _ = get_device()

        # Should not raise errors
        try:
            clear_memory_cache(device)
            logger.info("Memory cache clearing works")
        except Exception as e:
            self.fail(f"Memory cache clearing failed: {e}")


class TestTrainerConfig(unittest.TestCase):
    """Test trainer configuration."""

    def test_trainer_config_defaults(self):
        """Test TrainerConfig default values."""
        from experiments.training.trainer import TrainerConfig

        config = TrainerConfig()

        self.assertEqual(config.learning_rate, 2e-4)
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertTrue(config.early_stopping)

        logger.info("TrainerConfig defaults are correct")

    def test_trainer_config_custom(self):
        """Test TrainerConfig with custom values."""
        from experiments.training.trainer import TrainerConfig

        config = TrainerConfig(
            learning_rate=1e-4,
            num_train_epochs=5,
            gradient_accumulation_steps=8,
        )

        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.gradient_accumulation_steps, 8)


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping functionality."""

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        from experiments.training.trainer import EarlyStopping

        early_stop = EarlyStopping(patience=3, threshold=0.01, mode="min")

        # Simulate improving metrics
        self.assertFalse(early_stop(1.0))
        self.assertFalse(early_stop(0.8))
        self.assertFalse(early_stop(0.6))

        self.assertFalse(early_stop.should_stop)

    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        from experiments.training.trainer import EarlyStopping

        early_stop = EarlyStopping(patience=3, threshold=0.01, mode="min")

        # Initial value
        self.assertFalse(early_stop(0.5))

        # No improvement
        self.assertFalse(early_stop(0.5))  # count = 1
        self.assertFalse(early_stop(0.51))  # count = 2
        self.assertTrue(early_stop(0.5))  # count = 3, should stop

        self.assertTrue(early_stop.should_stop)


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_manager_init(self):
        """Test CheckpointManager initialization."""
        from experiments.training.trainer import CheckpointManager

        manager = CheckpointManager(
            output_dir=Path(self.temp_dir),
            save_total_limit=3,
        )

        self.assertEqual(manager.save_total_limit, 3)
        self.assertTrue(Path(self.temp_dir).exists())

    def test_checkpoint_save(self):
        """Test checkpoint saving."""
        from experiments.training.trainer import CheckpointManager, TrainingMetrics
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            self.skipTest(f"Could not load tokenizer: {e}")

        # Create a simple mock model
        model = MagicMock()
        model.save_pretrained = MagicMock()

        manager = CheckpointManager(
            output_dir=Path(self.temp_dir),
            save_total_limit=2,
        )

        metrics = TrainingMetrics(step=100, train_loss=0.5, eval_loss=0.6)

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            step=100,
            metrics=metrics,
        )

        self.assertTrue(checkpoint_path.exists())
        self.assertTrue((checkpoint_path / "training_state.json").exists())


class TestIntegration(unittest.TestCase):
    """Integration tests (may require model download)."""

    @unittest.skipIf(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        "Integration tests skipped (set RUN_INTEGRATION_TESTS=1 to run)"
    )
    def test_full_training_step(self):
        """Test a full training step with a small model."""
        from experiments.training.model_utils import get_device, load_model_4bit, apply_lora
        from experiments.training.data_utils import ChatDataset, create_dataloaders
        from experiments.training.trainer import QLoRATrainer, TrainerConfig

        # Use a small model for testing
        model_name = "gpt2"

        device, use_4bit, compute_dtype = get_device()

        # Load model
        model, tokenizer = load_model_4bit(
            model_name=model_name,
            device=device,
            use_4bit=False,  # GPT-2 doesn't need quantization
            compute_dtype=compute_dtype,
        )

        # Apply LoRA
        model = apply_lora(
            model=model,
            lora_r=4,
            lora_alpha=8,
            target_modules=["c_attn"],  # GPT-2 uses different names
        )

        # Create test data
        test_data = [
            {
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
            }
        ] * 10

        # Create dataloaders
        train_loader, eval_loader = create_dataloaders(
            train_data=test_data,
            eval_data=test_data[:2],
            tokenizer=tokenizer,
            max_length=128,
            train_batch_size=2,
            eval_batch_size=2,
        )

        # Create trainer
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainerConfig(
                output_dir=temp_dir,
                max_steps=2,
                logging_steps=1,
                eval_steps=1,
                save_steps=1,
                device=device,
            )

            trainer = QLoRATrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                config=config,
            )

            # Train for a few steps
            metrics = trainer.train()

            self.assertGreater(trainer.global_step, 0)
            self.assertIsNotNone(metrics.train_loss)

            logger.info("Full training step works correctly")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDataUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestModelUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainerConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestEarlyStopping))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointManager))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
