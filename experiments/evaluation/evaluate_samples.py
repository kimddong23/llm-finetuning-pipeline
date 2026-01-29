"""
Direct evaluation script that bypasses multiprocessing.Manager() issues.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List
from contextlib import redirect_stdout, redirect_stderr
import io
import signal

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")

def check_correctness(problem: Dict, completion: str, timeout: float = 3.0) -> Dict:
    """
    Check if a completion is correct for a given problem.

    Returns:
        Dict with 'passed' (bool) and 'result' (str) keys
    """
    # Get the test code
    test_code = problem.get("test", "")

    # Add proper indentation to completion if needed
    # HumanEval expects function body to be indented
    completion_lines = completion.split('\n')
    indented_lines = []
    for line in completion_lines:
        if line.strip():  # Only indent non-empty lines
            # Add 4 spaces if line doesn't start with whitespace
            if not line.startswith(' ') and not line.startswith('\t'):
                indented_lines.append('    ' + line)
            else:
                indented_lines.append(line)
        else:
            indented_lines.append(line)

    indented_completion = '\n'.join(indented_lines)

    # Combine prompt, completion, and test
    full_code = problem["prompt"] + indented_completion + "\n" + test_code

    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {
        "passed": False,
        "result": "unknown"
    }

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        # Create execution environment
        exec_globals = {}

        # Redirect stdout/stderr
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(full_code, exec_globals)

        result["passed"] = True
        result["result"] = "passed"

    except TimeoutException:
        result["passed"] = False
        result["result"] = "timed out"

    except AssertionError as e:
        result["passed"] = False
        result["result"] = f"failed: {str(e)}"

    except Exception as e:
        result["passed"] = False
        result["result"] = f"error: {type(e).__name__}: {str(e)}"

    finally:
        # Cancel alarm
        signal.alarm(0)

    return result

def evaluate_functional_correctness(sample_file: str, problems_file: str = None) -> Dict:
    """
    Evaluate functional correctness of generated samples.

    Returns:
        Dict with pass@1 and detailed results
    """
    # Load problems
    if problems_file is None:
        from human_eval.data import read_problems
        problems = read_problems()
    else:
        with open(problems_file, 'r') as f:
            problems = {task["task_id"]: task for task in (json.loads(line) for line in f)}

    # Load samples
    samples = []
    with open(sample_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {sample_file}")
    print(f"Evaluating against {len(problems)} problems...")

    # Evaluate each sample
    results = []
    passed = 0
    total = 0

    for i, sample in enumerate(samples):
        task_id = sample["task_id"]
        completion = sample["completion"]

        if task_id not in problems:
            print(f"Warning: task_id {task_id} not found in problems")
            continue

        problem = problems[task_id]

        # Check correctness
        result = check_correctness(problem, completion, timeout=3.0)

        total += 1
        if result["passed"]:
            passed += 1
            status = "✓"
        else:
            status = "✗"

        results.append({
            "task_id": task_id,
            "passed": result["passed"],
            "result": result["result"]
        })

        # Progress
        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{len(samples)} ({100*(i+1)/len(samples):.1f}%)")

    # Calculate pass@1
    pass_at_1 = passed / total if total > 0 else 0

    print(f"\nEvaluation complete!")
    print(f"Passed: {passed}/{total}")
    print(f"pass@1: {pass_at_1:.4f} ({pass_at_1*100:.2f}%)")

    return {
        "pass@1": pass_at_1,
        "passed": passed,
        "total": total,
        "results": results
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_samples.py <sample_file>")
        sys.exit(1)

    sample_file = sys.argv[1]

    if not os.path.exists(sample_file):
        print(f"Error: {sample_file} not found")
        sys.exit(1)

    # Run evaluation
    results = evaluate_functional_correctness(sample_file)

    # Save results
    output_file = sample_file + "_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
