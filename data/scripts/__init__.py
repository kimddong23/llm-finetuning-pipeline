"""
Data processing scripts for LLM fine-tuning pipeline.

This package provides a complete pipeline for downloading, processing,
and analyzing The Stack dataset for code generation fine-tuning.

Modules:
    download_stack: Download The Stack dataset from HuggingFace
    process_data: Apply quality filtering to code samples
    create_instruction: Convert to instruction format for training
    analyze_dataset: Generate statistics and visualizations
    run_pipeline: Execute complete pipeline
    test_pipeline: Verify pipeline components

Usage:
    # Run complete pipeline
    python run_pipeline.py

    # Or run individual steps
    python download_stack.py --samples 20000
    python process_data.py --target-samples 10000
    python create_instruction.py --train-split 0.9
    python analyze_dataset.py

    # Test pipeline
    python test_pipeline.py
"""

__version__ = "1.0.0"
__author__ = "LLM Finetuning Pipeline"

# Export main classes
from .download_stack import StackDownloader
from .process_data import QualityFilter
from .create_instruction import InstructionExtractor
from .analyze_dataset import DatasetAnalyzer

__all__ = [
    "StackDownloader",
    "QualityFilter",
    "InstructionExtractor",
    "DatasetAnalyzer",
]
