"""
Test script to verify data processing pipeline.

This script runs a mini version of the pipeline with a small sample
to verify all components work correctly before running the full pipeline.

Usage:
    python test_pipeline.py
"""

import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required libraries can be imported."""
    logger.info("Testing imports...")

    try:
        import ast
        import hashlib
        import json
        import matplotlib.pyplot as plt
        import numpy as np
        from datasets import load_dataset
        from tqdm import tqdm

        logger.info("✓ All imports successful")
        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {str(e)}")
        logger.error("Please install missing dependencies:")
        logger.error("  pip install datasets tqdm matplotlib numpy")
        return False


def test_ast_parsing():
    """Test Python AST parsing capabilities."""
    logger.info("Testing AST parsing...")

    test_code = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    return a + b
'''

    try:
        import ast
        tree = ast.parse(test_code)

        # Find function
        func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func = node
                break

        if func:
            # Check signature
            assert func.name == "calculate_sum"
            assert len(func.args.args) == 2

            # Check docstring
            docstring = ast.get_docstring(func)
            assert docstring is not None
            assert "sum" in docstring.lower()

            logger.info("✓ AST parsing test passed")
            return True
        else:
            logger.error("✗ Failed to find function in AST")
            return False

    except Exception as e:
        logger.error(f"✗ AST parsing failed: {str(e)}")
        return False


def test_quality_filter():
    """Test quality filtering logic."""
    logger.info("Testing quality filter...")

    try:
        from process_data import QualityFilter

        filter = QualityFilter(
            max_comment_ratio=0.3,
            require_docstring=True,
            require_structure=True
        )

        # Test valid code
        valid_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")
'''
        sample = {'content': valid_code}
        is_valid, reason = filter.check_quality(sample)
        assert is_valid, f"Valid code rejected: {reason}"

        # Test invalid code (no docstring)
        invalid_code = '''
def hello_world():
    print("Hello, World!")
'''
        sample = {'content': invalid_code}
        is_valid, reason = filter.check_quality(sample)
        assert not is_valid, "Invalid code (no docstring) accepted"

        logger.info("✓ Quality filter test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Quality filter test failed: {str(e)}")
        return False


def test_instruction_extraction():
    """Test instruction extraction logic."""
    logger.info("Testing instruction extraction...")

    try:
        from create_instruction import InstructionExtractor

        extractor = InstructionExtractor(seed=42)

        test_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b
'''

        sample = {'content': test_code}
        results = extractor.to_instruction_format(sample)

        assert len(results) > 0, "No instructions extracted"
        assert 'instruction' in results[0], "Missing instruction field"
        assert 'output' in results[0], "Missing output field"
        assert 'add_numbers' in results[0]['instruction'], "Function name not in instruction"

        logger.info("✓ Instruction extraction test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Instruction extraction test failed: {str(e)}")
        return False


def test_dataset_analyzer():
    """Test dataset analysis logic."""
    logger.info("Testing dataset analyzer...")

    try:
        from analyze_dataset import DatasetAnalyzer

        analyzer = DatasetAnalyzer()

        # Test code analysis
        test_code = '''
def example(x: int) -> int:
    """Example function."""
    return x * 2
'''

        analysis = analyzer.analyze_code(test_code)

        assert analysis['code_length'] > 0, "Code length is 0"
        assert analysis['token_count'] > 0, "Token count is 0"
        assert analysis['num_functions'] == 1, f"Expected 1 function, got {analysis['num_functions']}"
        assert analysis['has_type_hints'], "Type hints not detected"
        assert analysis['has_docstring'], "Docstring not detected"

        logger.info("✓ Dataset analyzer test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Dataset analyzer test failed: {str(e)}")
        return False


def test_file_structure():
    """Test that output directories can be created."""
    logger.info("Testing file structure...")

    try:
        base_dir = Path(__file__).parent.parent
        test_dir = base_dir / 'test_output'

        # Create directories
        (test_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (test_dir / 'processed').mkdir(parents=True, exist_ok=True)
        (test_dir / 'analysis').mkdir(parents=True, exist_ok=True)

        # Test writing
        test_file = test_dir / 'raw' / 'test.jsonl'
        with open(test_file, 'w') as f:
            f.write(json.dumps({'test': 'data'}) + '\n')

        # Test reading
        with open(test_file, 'r') as f:
            data = json.loads(f.readline())
            assert data['test'] == 'data', "Data mismatch"

        # Cleanup
        test_file.unlink()
        (test_dir / 'raw').rmdir()
        (test_dir / 'processed').rmdir()
        (test_dir / 'analysis').rmdir()
        test_dir.rmdir()

        logger.info("✓ File structure test passed")
        return True

    except Exception as e:
        logger.error(f"✗ File structure test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("TESTING DATA PROCESSING PIPELINE")
    logger.info("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("AST Parsing", test_ast_parsing),
        ("Quality Filter", test_quality_filter),
        ("Instruction Extraction", test_instruction_extraction),
        ("Dataset Analyzer", test_dataset_analyzer),
        ("File Structure", test_file_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info("")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {str(e)}")
            failed += 1

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"TEST RESULTS: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    logger.info("=" * 80)

    if failed == 0:
        logger.info("✓ All tests passed! Pipeline is ready to use.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run the full pipeline: python run_pipeline.py")
        logger.info("  2. Or run individual scripts as needed")
        return 0
    else:
        logger.error("✗ Some tests failed. Please fix issues before running the pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
