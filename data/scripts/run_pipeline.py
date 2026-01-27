"""
Complete data processing pipeline runner.

This script runs the entire data processing pipeline:
1. Download The Stack dataset
2. Quality filtering
3. Instruction format conversion
4. Dataset analysis

Usage:
    python run_pipeline.py
    python run_pipeline.py --samples 50000 --target 25000
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Import pipeline modules
from download_stack import StackDownloader
from process_data import QualityFilter, process_dataset
from create_instruction import convert_dataset
from analyze_dataset import analyze_dataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Run the complete data processing pipeline."""

    def __init__(
        self,
        base_dir: Path,
        num_samples: int = 20000,
        target_samples: int = 10000,
        seed: int = 42
    ):
        """
        Initialize the pipeline runner.

        Args:
            base_dir: Base directory for data files
            num_samples: Number of raw samples to download
            target_samples: Target number of quality-filtered samples
            seed: Random seed for reproducibility
        """
        self.base_dir = base_dir
        self.num_samples = num_samples
        self.target_samples = target_samples
        self.seed = seed

        # Define paths
        self.raw_dir = base_dir / 'raw'
        self.processed_dir = base_dir / 'processed'
        self.analysis_dir = base_dir / 'analysis'

        self.raw_file = self.raw_dir / f'the_stack_python_{num_samples//1000}k.jsonl'
        self.filtered_file = self.processed_dir / f'quality_filtered_{target_samples//1000}k.jsonl'
        self.train_file = self.processed_dir / 'train.jsonl'
        self.eval_file = self.processed_dir / 'eval.jsonl'

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline statistics
        self.stats: Dict[str, Any] = {}

    def run_step1_download(self) -> bool:
        """
        Step 1: Download The Stack dataset.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Downloading The Stack dataset")
        logger.info("=" * 80)

        try:
            start_time = time.time()

            downloader = StackDownloader(
                min_length=100,
                max_length=2000,
                seed=self.seed
            )

            saved_count = downloader.download_samples(
                num_samples=self.num_samples,
                output_path=self.raw_file
            )

            elapsed_time = time.time() - start_time
            self.stats['step1_download'] = {
                'samples_downloaded': saved_count,
                'elapsed_time_seconds': elapsed_time,
                'output_file': str(self.raw_file)
            }

            logger.info(f"✓ Step 1 completed in {elapsed_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"✗ Step 1 failed: {str(e)}")
            return False

    def run_step2_filter(self) -> bool:
        """
        Step 2: Apply quality filtering.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Applying quality filters")
        logger.info("=" * 80)

        try:
            start_time = time.time()

            quality_filter = QualityFilter(
                max_comment_ratio=0.3,
                require_docstring=True,
                require_structure=True
            )

            filter_stats = process_dataset(
                input_path=self.raw_file,
                output_path=self.filtered_file,
                target_samples=self.target_samples,
                quality_filter=quality_filter
            )

            elapsed_time = time.time() - start_time
            self.stats['step2_filter'] = {
                **filter_stats,
                'elapsed_time_seconds': elapsed_time,
                'output_file': str(self.filtered_file)
            }

            logger.info(f"✓ Step 2 completed in {elapsed_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"✗ Step 2 failed: {str(e)}")
            return False

    def run_step3_convert(self) -> bool:
        """
        Step 3: Convert to instruction format.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Converting to instruction format")
        logger.info("=" * 80)

        try:
            start_time = time.time()

            convert_stats = convert_dataset(
                input_path=self.filtered_file,
                output_dir=self.processed_dir,
                train_split=0.9,
                seed=self.seed
            )

            elapsed_time = time.time() - start_time
            self.stats['step3_convert'] = {
                **convert_stats,
                'elapsed_time_seconds': elapsed_time,
                'train_file': str(self.train_file),
                'eval_file': str(self.eval_file)
            }

            logger.info(f"✓ Step 3 completed in {elapsed_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"✗ Step 3 failed: {str(e)}")
            return False

    def run_step4_analyze(self) -> bool:
        """
        Step 4: Analyze dataset and generate statistics.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Analyzing dataset")
        logger.info("=" * 80)

        try:
            start_time = time.time()

            analysis_stats = analyze_dataset(
                input_path=self.train_file,
                output_dir=self.analysis_dir
            )

            elapsed_time = time.time() - start_time
            self.stats['step4_analyze'] = {
                **analysis_stats,
                'elapsed_time_seconds': elapsed_time,
                'analysis_dir': str(self.analysis_dir)
            }

            logger.info(f"✓ Step 4 completed in {elapsed_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"✗ Step 4 failed: {str(e)}")
            return False

    def run(self) -> bool:
        """
        Run the complete pipeline.

        Returns:
            True if all steps successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Raw samples: {self.num_samples}")
        logger.info(f"  Target quality samples: {self.target_samples}")
        logger.info(f"  Random seed: {self.seed}")
        logger.info(f"  Base directory: {self.base_dir}")
        logger.info("=" * 80)

        pipeline_start = time.time()

        # Step 1: Download
        if not self.run_step1_download():
            logger.error("Pipeline failed at Step 1 (Download)")
            return False

        # Step 2: Filter
        if not self.run_step2_filter():
            logger.error("Pipeline failed at Step 2 (Filter)")
            return False

        # Step 3: Convert
        if not self.run_step3_convert():
            logger.error("Pipeline failed at Step 3 (Convert)")
            return False

        # Step 4: Analyze
        if not self.run_step4_analyze():
            logger.error("Pipeline failed at Step 4 (Analyze)")
            return False

        # Pipeline complete
        pipeline_elapsed = time.time() - pipeline_start

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} minutes)")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  Raw data: {self.raw_file}")
        logger.info(f"  Filtered data: {self.filtered_file}")
        logger.info(f"  Training data: {self.train_file}")
        logger.info(f"  Evaluation data: {self.eval_file}")
        logger.info(f"  Analysis: {self.analysis_dir}")
        logger.info("")
        logger.info("Summary:")
        if 'step1_download' in self.stats:
            logger.info(f"  Downloaded: {self.stats['step1_download']['samples_downloaded']} samples")
        if 'step2_filter' in self.stats:
            logger.info(f"  Filtered: {self.stats['step2_filter']['passed']} samples")
        if 'step3_convert' in self.stats:
            logger.info(f"  Training samples: {self.stats['step3_convert']['train_samples']}")
            logger.info(f"  Evaluation samples: {self.stats['step3_convert']['eval_samples']}")
        logger.info("=" * 80)

        return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run complete data processing pipeline"
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20000,
        help='Number of raw samples to download (default: 20000)'
    )
    parser.add_argument(
        '--target',
        type=int,
        default=10000,
        help='Target number of quality-filtered samples (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='..',
        help='Base directory for data files (default: ..)'
    )

    args = parser.parse_args()

    # Convert base_dir to absolute path
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).parent / base_dir
    base_dir = base_dir.resolve()

    # Create and run pipeline
    pipeline = PipelineRunner(
        base_dir=base_dir,
        num_samples=args.samples,
        target_samples=args.target,
        seed=args.seed
    )

    success = pipeline.run()

    if not success:
        logger.error("Pipeline failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
