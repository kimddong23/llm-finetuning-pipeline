"""
Download The Stack dataset (Python subset) from HuggingFace.

This script downloads a subset of The Stack dataset with streaming for memory efficiency.
Filters code samples by length and saves them in JSONL format for further processing.

Usage:
    python download_stack.py --samples 20000 --output ../raw/the_stack_python_20k.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterator

from datasets import load_dataset
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StackDownloader:
    """Download and filter The Stack dataset."""

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 2000,
        seed: int = 42
    ):
        """
        Initialize the downloader.

        Args:
            min_length: Minimum code length in characters
            max_length: Maximum code length in characters
            seed: Random seed for reproducibility
        """
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed

    def is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Check if a sample meets the length criteria.

        Args:
            sample: A data sample from The Stack

        Returns:
            True if sample is valid, False otherwise
        """
        content = sample.get('content', '')
        code_length = len(content)
        return self.min_length <= code_length <= self.max_length

    def download_samples(
        self,
        num_samples: int,
        output_path: Path
    ) -> int:
        """
        Download and save samples from The Stack dataset.

        Args:
            num_samples: Number of samples to download
            output_path: Path to save the downloaded samples

        Returns:
            Number of samples successfully downloaded

        Raises:
            RuntimeError: If unable to load dataset or write output
        """
        logger.info(f"Starting download of {num_samples} samples from The Stack (Python)")
        logger.info(f"Filtering: {self.min_length} <= code length <= {self.max_length}")

        try:
            # Load dataset with streaming for memory efficiency
            logger.info("Loading dataset with streaming...")
            dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir="data/python",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download and filter samples
            saved_count = 0
            processed_count = 0

            with open(output_path, 'w', encoding='utf-8') as f:
                with tqdm(total=num_samples, desc="Downloading") as pbar:
                    for sample in dataset:
                        processed_count += 1

                        if self.is_valid_sample(sample):
                            # Save sample as JSONL
                            json_line = json.dumps({
                                'content': sample.get('content', ''),
                                'size': sample.get('size', 0),
                                'lang': sample.get('lang', 'Python'),
                                'ext': sample.get('ext', 'py'),
                                'max_stars_repo_path': sample.get('max_stars_repo_path', ''),
                                'max_stars_repo_name': sample.get('max_stars_repo_name', ''),
                            }, ensure_ascii=False)
                            f.write(json_line + '\n')

                            saved_count += 1
                            pbar.update(1)

                            if saved_count >= num_samples:
                                break

                        # Progress logging every 10k samples
                        if processed_count % 10000 == 0:
                            logger.info(
                                f"Processed {processed_count} samples, "
                                f"saved {saved_count} valid samples"
                            )

            logger.info(f"Download complete!")
            logger.info(f"Total processed: {processed_count}")
            logger.info(f"Total saved: {saved_count}")
            logger.info(f"Saved to: {output_path}")

            return saved_count

        except Exception as e:
            logger.error(f"Error during download: {str(e)}")
            raise RuntimeError(f"Failed to download dataset: {str(e)}") from e


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download The Stack dataset (Python subset)"
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20000,
        help='Number of samples to download (default: 20000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../raw/the_stack_python_20k.jsonl',
        help='Output path for downloaded samples (default: ../raw/the_stack_python_20k.jsonl)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help='Minimum code length in characters (default: 100)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2000,
        help='Maximum code length in characters (default: 2000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Convert output path to absolute path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    output_path = output_path.resolve()

    # Create downloader and download samples
    downloader = StackDownloader(
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed
    )

    try:
        saved_count = downloader.download_samples(
            num_samples=args.samples,
            output_path=output_path
        )

        if saved_count < args.samples:
            logger.warning(
                f"Downloaded only {saved_count} samples (requested {args.samples}). "
                f"You may need to adjust length filters or process more data."
            )
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
