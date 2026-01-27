"""
Generate dataset statistics and visualizations.

This script analyzes the processed dataset and generates:
- Code length distribution
- Token count distribution
- Quality score distribution
- Function vs class statistics
- Visualizations

Usage:
    python analyze_dataset.py --input ../processed/train.jsonl --output-dir ../analysis
"""

import argparse
import ast
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze code dataset and generate statistics."""

    def __init__(self):
        """Initialize the analyzer."""
        self.stats = {
            'total_samples': 0,
            'code_lengths': [],
            'token_counts': [],
            'num_functions': [],
            'num_classes': [],
            'has_type_hints': 0,
            'has_docstrings': 0,
        }

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analyze a single code sample.

        Args:
            code: Python code string

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'code_length': len(code),
            'token_count': self._estimate_tokens(code),
            'num_functions': 0,
            'num_classes': 0,
            'has_type_hints': False,
            'has_docstring': False,
        }

        try:
            tree = ast.parse(code)

            # Count functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['num_functions'] += 1

                    # Check for type hints
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        analysis['has_type_hints'] = True

                    # Check for docstring
                    if ast.get_docstring(node):
                        analysis['has_docstring'] = True

                elif isinstance(node, ast.ClassDef):
                    analysis['num_classes'] += 1

                    # Check for class docstring
                    if ast.get_docstring(node):
                        analysis['has_docstring'] = True

        except Exception as e:
            logger.debug(f"Failed to analyze code: {str(e)}")

        return analysis

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (approximation).

        Args:
            text: Text string

        Returns:
            Estimated token count
        """
        # Simple approximation: ~1 token per 4 characters for code
        return len(text) // 4

    def analyze_sample(self, sample: Dict[str, Any]) -> None:
        """
        Analyze a single sample and update statistics.

        Args:
            sample: Sample dictionary from instruction format
        """
        self.stats['total_samples'] += 1

        # Get code from messages
        if 'messages' in sample:
            # Chat format
            code = sample['messages'][1]['content']  # Assistant message
        else:
            # Instruction format
            code = sample.get('output', '')

        # Analyze code
        analysis = self.analyze_code(code)

        # Update statistics
        self.stats['code_lengths'].append(analysis['code_length'])
        self.stats['token_counts'].append(analysis['token_count'])
        self.stats['num_functions'].append(analysis['num_functions'])
        self.stats['num_classes'].append(analysis['num_classes'])

        if analysis['has_type_hints']:
            self.stats['has_type_hints'] += 1
        if analysis['has_docstring']:
            self.stats['has_docstrings'] += 1

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        def safe_stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
            return {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
            }

        summary = {
            'total_samples': self.stats['total_samples'],
            'code_length': safe_stats(self.stats['code_lengths']),
            'token_count': safe_stats(self.stats['token_counts']),
            'num_functions': safe_stats(self.stats['num_functions']),
            'num_classes': safe_stats(self.stats['num_classes']),
            'percentage_with_type_hints': (
                self.stats['has_type_hints'] / self.stats['total_samples'] * 100
                if self.stats['total_samples'] > 0 else 0
            ),
            'percentage_with_docstrings': (
                self.stats['has_docstrings'] / self.stats['total_samples'] * 100
                if self.stats['total_samples'] > 0 else 0
            ),
        }

        return summary

    def plot_distributions(self, output_dir: Path) -> None:
        """
        Create distribution plots.

        Args:
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Code length distribution
        axes[0, 0].hist(self.stats['code_lengths'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Code Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Code Length Distribution')
        axes[0, 0].axvline(np.mean(self.stats['code_lengths']), color='red',
                          linestyle='--', label=f'Mean: {np.mean(self.stats["code_lengths"]):.0f}')
        axes[0, 0].legend()

        # Token count distribution
        axes[0, 1].hist(self.stats['token_counts'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Token Count (estimated)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Token Count Distribution')
        axes[0, 1].axvline(np.mean(self.stats['token_counts']), color='red',
                          linestyle='--', label=f'Mean: {np.mean(self.stats["token_counts"]):.0f}')
        axes[0, 1].legend()

        # Functions per sample
        func_counter = Counter(self.stats['num_functions'])
        func_counts = sorted(func_counter.items())
        if func_counts:
            x_func, y_func = zip(*func_counts)
            axes[1, 0].bar(x_func, y_func, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Number of Functions')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Functions per Sample')

        # Classes per sample
        class_counter = Counter(self.stats['num_classes'])
        class_counts = sorted(class_counter.items())
        if class_counts:
            x_class, y_class = zip(*class_counts)
            axes[1, 1].bar(x_class, y_class, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Number of Classes')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Classes per Sample')

        plt.tight_layout()
        plot_path = output_dir / 'dataset_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved distribution plots to {plot_path}")

        # Quality metrics pie chart
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Type hints
        type_hint_data = [
            self.stats['has_type_hints'],
            self.stats['total_samples'] - self.stats['has_type_hints']
        ]
        ax[0].pie(type_hint_data, labels=['With Type Hints', 'Without Type Hints'],
                 autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        ax[0].set_title('Samples with Type Hints')

        # Docstrings
        docstring_data = [
            self.stats['has_docstrings'],
            self.stats['total_samples'] - self.stats['has_docstrings']
        ]
        ax[1].pie(docstring_data, labels=['With Docstrings', 'Without Docstrings'],
                 autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ffcc99'])
        ax[1].set_title('Samples with Docstrings')

        plt.tight_layout()
        quality_plot_path = output_dir / 'quality_metrics.png'
        plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved quality metrics plots to {quality_plot_path}")


def analyze_dataset(
    input_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze dataset and generate statistics and visualizations.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save analysis results

    Returns:
        Dictionary with summary statistics

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If analysis fails
    """
    logger.info(f"Analyzing dataset: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = DatasetAnalyzer()

    try:
        # Load and analyze samples
        logger.info("Analyzing samples...")
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing"):
                try:
                    sample = json.loads(line.strip())
                    analyzer.analyze_sample(sample)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON line, skipping")

        # Get summary statistics
        summary_stats = analyzer.get_summary_stats()

        # Save summary statistics
        stats_path = output_dir / 'dataset_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved statistics to {stats_path}")

        # Generate plots
        logger.info("Generating plots...")
        analyzer.plot_distributions(output_dir)

        logger.info("Analysis complete!")
        logger.info(f"Summary:")
        logger.info(f"  Total samples: {summary_stats['total_samples']}")
        logger.info(f"  Avg code length: {summary_stats['code_length']['mean']:.0f} chars")
        logger.info(f"  Avg token count: {summary_stats['token_count']['mean']:.0f} tokens")
        logger.info(f"  With type hints: {summary_stats['percentage_with_type_hints']:.1f}%")
        logger.info(f"  With docstrings: {summary_stats['percentage_with_docstrings']:.1f}%")

        return summary_stats

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise RuntimeError(f"Failed to analyze dataset: {str(e)}") from e


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze dataset and generate statistics"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../processed/train.jsonl',
        help='Input JSONL file (default: ../processed/train.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../analysis',
        help='Output directory for analysis results (default: ../analysis)'
    )

    args = parser.parse_args()

    # Convert paths to absolute paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path
    input_path = input_path.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    output_dir = output_dir.resolve()

    try:
        summary_stats = analyze_dataset(
            input_path=input_path,
            output_dir=output_dir
        )

        logger.info(f"Analysis results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
