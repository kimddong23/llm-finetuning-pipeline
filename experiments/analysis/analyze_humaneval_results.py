"""
HumanEval results analysis script.
Compares base model vs fine-tuned model performance.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_results(results_file):
    """Load evaluation results."""
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

def analyze_results(base_results, finetuned_results):
    """Analyze and compare results."""
    print("="*70)
    print("HUMANEVAL BENCHMARK RESULTS")
    print("="*70)

    if base_results:
        base_pass = base_results.get('pass@1', 0) * 100
        print(f"\n📊 Base Model:")
        print(f"   pass@1: {base_pass:.2f}%")
        print(f"   Total: {base_results.get('total', 0)}")
        print(f"   Passed: {base_results.get('passed', 0)}")
    else:
        base_pass = 0
        print(f"\n📊 Base Model: Results not available")

    if finetuned_results:
        ft_pass = finetuned_results.get('pass@1', 0) * 100
        print(f"\n📊 Fine-tuned Model:")
        print(f"   pass@1: {ft_pass:.2f}%")
        print(f"   Total: {finetuned_results.get('total', 0)}")
        print(f"   Passed: {finetuned_results.get('passed', 0)}")
    else:
        ft_pass = 0
        print(f"\n📊 Fine-tuned Model: Results not available")

    if base_results and finetuned_results:
        improvement = ft_pass - base_pass
        relative_improvement = (ft_pass / base_pass - 1) * 100 if base_pass > 0 else 0

        print(f"\n📈 Improvement:")
        print(f"   Absolute: +{improvement:.2f} percentage points")
        print(f"   Relative: +{relative_improvement:.1f}%")

        if improvement > 0:
            print(f"   ✅ Fine-tuning improved performance!")
        elif improvement < 0:
            print(f"   ⚠️  Performance decreased (check overfitting)")
        else:
            print(f"   ➖ No significant change")

    print("="*70)

    return {
        'base_pass@1': base_pass,
        'finetuned_pass@1': ft_pass,
        'improvement': ft_pass - base_pass if (base_results and finetuned_results) else None,
        'relative_improvement': relative_improvement if (base_results and finetuned_results) else None
    }

def generate_comparison_table(summary):
    """Generate markdown comparison table."""
    table = f"""
## HumanEval Benchmark Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| pass@1 | {summary['base_pass@1']:.1f}% | {summary['finetuned_pass@1']:.1f}% | +{summary['improvement']:.1f}pp |
| Relative Gain | - | - | +{summary['relative_improvement']:.1f}% |

**Key Findings:**
- Fine-tuning {'improved' if summary['improvement'] > 0 else 'did not improve'} code generation performance
- Absolute improvement: {summary['improvement']:.1f} percentage points
- Relative improvement: {summary['relative_improvement']:.1f}%
"""
    return table

def identify_error_patterns(samples_file):
    """Identify common error patterns in failed samples."""
    if not os.path.exists(samples_file):
        print(f"Samples file not found: {samples_file}")
        return

    print(f"\n📋 Analyzing error patterns from: {samples_file}")

    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"   Total samples: {len(samples)}")

    # Analyze completion lengths
    lengths = [len(s['completion']) for s in samples]
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    print(f"   Average completion length: {avg_length:.0f} characters")
    print(f"   Min length: {min(lengths)}")
    print(f"   Max length: {max(lengths)}")

    # Common patterns
    empty_completions = sum(1 for s in samples if not s['completion'].strip())
    print(f"   Empty completions: {empty_completions}")

def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "../../results/humaneval"

    # Load results
    base_results_file = results_dir / "base_model_samples.jsonl_results.json"
    ft_results_file = results_dir / "finetuned_model_samples.jsonl_results.json"

    base_results = load_results(base_results_file)
    finetuned_results = load_results(ft_results_file)

    # Analyze
    summary = analyze_results(base_results, finetuned_results)

    # Error analysis
    if os.path.exists(results_dir / "base_model_samples.jsonl"):
        identify_error_patterns(results_dir / "base_model_samples.jsonl")

    if os.path.exists(results_dir / "finetuned_model_samples.jsonl"):
        identify_error_patterns(results_dir / "finetuned_model_samples.jsonl")

    # Generate comparison table
    if summary['improvement'] is not None:
        table = generate_comparison_table(summary)

        # Save to file
        output_file = results_dir / "comparison_table.md"
        with open(output_file, 'w') as f:
            f.write(table)
        print(f"\n✅ Comparison table saved to: {output_file}")

    print("\n🎯 Next steps:")
    print("1. Review detailed results in results/humaneval/")
    print("2. Update Technical Report with benchmark numbers")
    print("3. Update README with final metrics")

if __name__ == "__main__":
    main()
