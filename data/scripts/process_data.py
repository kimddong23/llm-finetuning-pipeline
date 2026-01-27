"""
Quality filtering pipeline for code samples.

This script applies multiple quality filters to raw code samples:
- Syntax validity (AST parsing)
- Has docstrings
- Has function or class definitions
- Comment ratio check
- Duplicate removal

Usage:
    python process_data.py --input ../raw/the_stack_python_20k.jsonl --output ../processed/quality_filtered_10k.jsonl
"""

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Set, Optional

from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityFilter:
    """Filter code samples based on quality criteria."""

    def __init__(
        self,
        max_comment_ratio: float = 0.3,
        require_docstring: bool = True,
        require_structure: bool = True
    ):
        """
        Initialize the quality filter.

        Args:
            max_comment_ratio: Maximum ratio of comment lines to total lines
            require_docstring: Whether to require docstrings
            require_structure: Whether to require function/class definitions
        """
        self.max_comment_ratio = max_comment_ratio
        self.require_docstring = require_docstring
        self.require_structure = require_structure
        self.seen_hashes: Set[str] = set()

    def is_syntax_valid(self, code: str) -> bool:
        """
        Check if code has valid Python syntax.

        Args:
            code: Python code string

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            # Other parsing errors
            return False

    def has_docstring(self, code: str) -> bool:
        """
        Check if code contains docstrings.

        Args:
            code: Python code string

        Returns:
            True if docstring is present, False otherwise
        """
        return '"""' in code or "'''" in code

    def has_structure(self, code: str) -> bool:
        """
        Check if code has function or class definitions.

        Args:
            code: Python code string

        Returns:
            True if structure is present, False otherwise
        """
        return 'def ' in code or 'class ' in code

    def calculate_comment_ratio(self, code: str) -> float:
        """
        Calculate the ratio of comment lines to total lines.

        Args:
            code: Python code string

        Returns:
            Ratio of comment lines (0.0 to 1.0)
        """
        lines = code.split('\n')
        if not lines:
            return 0.0

        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        return comment_lines / len(lines)

    def is_duplicate(self, code: str) -> bool:
        """
        Check if code is a duplicate based on content hash.

        Args:
            code: Python code string

        Returns:
            True if duplicate, False otherwise
        """
        code_hash = hashlib.md5(code.encode('utf-8')).hexdigest()
        if code_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(code_hash)
        return False

    def check_quality(self, sample: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Check if a sample meets all quality criteria.

        Args:
            sample: Code sample dictionary

        Returns:
            Tuple of (is_valid, failure_reason)
        """
        code = sample.get('content', '')

        # Check syntax validity
        if not self.is_syntax_valid(code):
            return False, "invalid_syntax"

        # Check for docstrings
        if self.require_docstring and not self.has_docstring(code):
            return False, "no_docstring"

        # Check for structure
        if self.require_structure and not self.has_structure(code):
            return False, "no_structure"

        # Check comment ratio
        comment_ratio = self.calculate_comment_ratio(code)
        if comment_ratio > self.max_comment_ratio:
            return False, f"high_comment_ratio_{comment_ratio:.2f}"

        # Check for duplicates
        if self.is_duplicate(code):
            return False, "duplicate"

        return True, None


def process_dataset(
    input_path: Path,
    output_path: Path,
    target_samples: int,
    quality_filter: QualityFilter
) -> Dict[str, int]:
    """
    Process raw dataset with quality filtering.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        target_samples: Target number of quality samples
        quality_filter: QualityFilter instance

    Returns:
        Dictionary with processing statistics

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If processing fails
    """
    logger.info(f"Processing dataset: {input_path}")
    logger.info(f"Target samples: {target_samples}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_processed': 0,
        'passed': 0,
        'failed_invalid_syntax': 0,
        'failed_no_docstring': 0,
        'failed_no_structure': 0,
        'failed_high_comment_ratio': 0,
        'failed_duplicate': 0,
    }

    try:
        with open(input_path, 'r', encoding='utf-8') as fin:
            with open(output_path, 'w', encoding='utf-8') as fout:
                with tqdm(total=target_samples, desc="Filtering") as pbar:
                    for line in fin:
                        if stats['passed'] >= target_samples:
                            break

                        stats['total_processed'] += 1

                        try:
                            sample = json.loads(line.strip())
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON at line {stats['total_processed']}")
                            continue

                        # Apply quality checks
                        is_valid, failure_reason = quality_filter.check_quality(sample)

                        if is_valid:
                            # Save sample
                            json_line = json.dumps(sample, ensure_ascii=False)
                            fout.write(json_line + '\n')
                            stats['passed'] += 1
                            pbar.update(1)
                        else:
                            # Record failure reason
                            if failure_reason:
                                key = f"failed_{failure_reason.split('_')[0]}"
                                if failure_reason.startswith('high_comment_ratio'):
                                    key = 'failed_high_comment_ratio'
                                stats[key] = stats.get(key, 0) + 1

                        # Progress logging every 1k samples
                        if stats['total_processed'] % 1000 == 0:
                            logger.info(
                                f"Processed {stats['total_processed']} samples, "
                                f"passed {stats['passed']} samples"
                            )

        logger.info("Processing complete!")
        logger.info(f"Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return stats

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise RuntimeError(f"Failed to process dataset: {str(e)}") from e


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Apply quality filtering to code samples"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../raw/the_stack_python_20k.jsonl',
        help='Input JSONL file (default: ../raw/the_stack_python_20k.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../processed/quality_filtered_10k.jsonl',
        help='Output JSONL file (default: ../processed/quality_filtered_10k.jsonl)'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=10000,
        help='Target number of quality samples (default: 10000)'
    )
    parser.add_argument(
        '--max-comment-ratio',
        type=float,
        default=0.3,
        help='Maximum comment ratio (default: 0.3)'
    )
    parser.add_argument(
        '--no-require-docstring',
        action='store_true',
        help='Do not require docstrings'
    )
    parser.add_argument(
        '--no-require-structure',
        action='store_true',
        help='Do not require function/class definitions'
    )

    args = parser.parse_args()

    # Convert paths to absolute paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path
    input_path = input_path.resolve()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    output_path = output_path.resolve()

    # Create quality filter
    quality_filter = QualityFilter(
        max_comment_ratio=args.max_comment_ratio,
        require_docstring=not args.no_require_docstring,
        require_structure=not args.no_require_structure
    )

    try:
        stats = process_dataset(
            input_path=input_path,
            output_path=output_path,
            target_samples=args.target_samples,
            quality_filter=quality_filter
        )

        if stats['passed'] < args.target_samples:
            logger.warning(
                f"Only {stats['passed']} samples passed quality filters "
                f"(target: {args.target_samples}). Consider relaxing filters or "
                f"downloading more data."
            )

        logger.info(f"Filtered data saved to: {output_path}")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
