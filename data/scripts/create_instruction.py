"""
Convert code samples to instruction format for LLM fine-tuning.

This script extracts function signatures and docstrings as instructions,
and function bodies as outputs. It creates a train/eval split and formats
data for chat-based training.

Usage:
    python create_instruction.py --input ../processed/quality_filtered_10k.jsonl --output-dir ../processed
"""

import argparse
import ast
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InstructionExtractor:
    """Extract instruction-output pairs from code samples."""

    def __init__(self, seed: int = 42):
        """
        Initialize the extractor.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def extract_function_info(self, code: str) -> List[Tuple[str, str, str]]:
        """
        Extract function signatures, docstrings, and bodies.

        Args:
            code: Python code string

        Returns:
            List of tuples (signature, docstring, body)
        """
        results = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract signature
                    signature = self._get_function_signature(node)

                    # Extract docstring
                    docstring = ast.get_docstring(node) or ""

                    # Extract body (excluding docstring)
                    body = self._get_function_body(node, code)

                    if signature and body:
                        results.append((signature, docstring, body))

        except Exception as e:
            logger.debug(f"Failed to parse code: {str(e)}")

        return results

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """
        Extract function signature from AST node.

        Args:
            node: FunctionDef AST node

        Returns:
            Function signature string
        """
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        signature = f"def {node.name}({', '.join(args)})"

        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        return signature + ":"

    def _get_function_body(self, node: ast.FunctionDef, code: str) -> str:
        """
        Extract function body from AST node.

        Args:
            node: FunctionDef AST node
            code: Original code string

        Returns:
            Function body string
        """
        # Get body nodes (skip docstring if present)
        body_nodes = node.body
        if (len(body_nodes) > 0 and
            isinstance(body_nodes[0], ast.Expr) and
            isinstance(body_nodes[0].value, ast.Constant)):
            # First statement is likely a docstring
            body_nodes = body_nodes[1:]

        if not body_nodes:
            return ""

        try:
            # Unparse body nodes
            body_lines = []
            for body_node in body_nodes:
                body_lines.append(ast.unparse(body_node))
            return '\n'.join(body_lines)
        except Exception:
            return ""

    def create_instruction_simple(self, code: str) -> Optional[str]:
        """
        Create simple instruction for code without clear structure.

        Args:
            code: Python code string

        Returns:
            Simple instruction string or None
        """
        # Check if code has functions/classes
        if 'def ' in code:
            return "Implement the following Python function:"
        elif 'class ' in code:
            return "Implement the following Python class:"
        else:
            return "Write the following Python code:"

    def to_instruction_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert code sample to instruction format.

        Args:
            sample: Code sample dictionary

        Returns:
            List of instruction-output dictionaries
        """
        code = sample.get('content', '')
        results = []

        # Try to extract function information
        function_infos = self.extract_function_info(code)

        if function_infos:
            # Create instruction from function signature and docstring
            for signature, docstring, body in function_infos:
                if docstring:
                    instruction = f"{signature}\n{docstring}"
                else:
                    instruction = f"{signature}\nImplement this function."

                results.append({
                    'instruction': instruction.strip(),
                    'output': body.strip(),
                    'metadata': {
                        'source': 'the_stack',
                        'lang': sample.get('lang', 'Python'),
                        'repo': sample.get('max_stars_repo_name', ''),
                    }
                })
        else:
            # Fallback: Use simple instruction
            instruction = self.create_instruction_simple(code)
            if instruction:
                results.append({
                    'instruction': instruction,
                    'output': code.strip(),
                    'metadata': {
                        'source': 'the_stack',
                        'lang': sample.get('lang', 'Python'),
                        'repo': sample.get('max_stars_repo_name', ''),
                    }
                })

        return results

    def to_chat_format(self, instruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert instruction-output to chat format.

        Args:
            instruction_data: Instruction-output dictionary

        Returns:
            Chat format dictionary
        """
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': instruction_data['instruction']
                },
                {
                    'role': 'assistant',
                    'content': instruction_data['output']
                }
            ],
            'metadata': instruction_data.get('metadata', {})
        }


def convert_dataset(
    input_path: Path,
    output_dir: Path,
    train_split: float = 0.9,
    seed: int = 42
) -> Dict[str, int]:
    """
    Convert quality-filtered dataset to instruction format with train/eval split.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save train/eval files
        train_split: Ratio of training data (default: 0.9)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with conversion statistics

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If conversion fails
    """
    logger.info(f"Converting dataset: {input_path}")
    logger.info(f"Train split: {train_split:.1%}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = InstructionExtractor(seed=seed)

    # Load all samples
    samples = []
    logger.info("Loading samples...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading"):
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON line, skipping")

    logger.info(f"Loaded {len(samples)} samples")

    # Convert to instruction format
    instruction_samples = []
    logger.info("Converting to instruction format...")
    for sample in tqdm(samples, desc="Converting"):
        instruction_data_list = extractor.to_instruction_format(sample)
        for instruction_data in instruction_data_list:
            chat_data = extractor.to_chat_format(instruction_data)
            instruction_samples.append(chat_data)

    logger.info(f"Created {len(instruction_samples)} instruction samples")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(instruction_samples)

    split_idx = int(len(instruction_samples) * train_split)
    train_samples = instruction_samples[:split_idx]
    eval_samples = instruction_samples[split_idx:]

    # Save train set
    train_path = output_dir / 'train.jsonl'
    logger.info(f"Saving {len(train_samples)} training samples to {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + '\n')

    # Save eval set
    eval_path = output_dir / 'eval.jsonl'
    logger.info(f"Saving {len(eval_samples)} evaluation samples to {eval_path}")
    with open(eval_path, 'w', encoding='utf-8') as f:
        for sample in eval_samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + '\n')

    stats = {
        'total_input_samples': len(samples),
        'total_instruction_samples': len(instruction_samples),
        'train_samples': len(train_samples),
        'eval_samples': len(eval_samples),
    }

    logger.info("Conversion complete!")
    logger.info(f"Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert code samples to instruction format"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../processed/quality_filtered_10k.jsonl',
        help='Input JSONL file (default: ../processed/quality_filtered_10k.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../processed',
        help='Output directory for train/eval files (default: ../processed)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Training data ratio (default: 0.9)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
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
        stats = convert_dataset(
            input_path=input_path,
            output_dir=output_dir,
            train_split=args.train_split,
            seed=args.seed
        )

        logger.info(f"Train file: {output_dir / 'train.jsonl'}")
        logger.info(f"Eval file: {output_dir / 'eval.jsonl'}")

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
