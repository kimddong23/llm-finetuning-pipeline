"""
Script to update README.md with final benchmark results.
"""

import re

def update_readme_with_humaneval(readme_path, base_pass1, ft_pass1, improvement):
    """Update README with HumanEval results."""

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the baseline results section
    pattern = r'(\*\*Baseline Results \(EXAONE-3\.5-2\.4B-Instruct\)\*\*:.*?- Trainable parameters: 4M / 2\.4B \(0\.17%\))'

    replacement = f'''**Baseline Results (EXAONE-3.5-2.4B-Instruct)**:
- Training time: 12.5 hours (500 steps, Mac M3 Pro)
- Final train loss: 0.87 (78% reduction from 3.99)
- Final eval loss: 1.03
- Perplexity: 2.80 (excellent)
- Trainable parameters: 4M / 2.4B (0.17%)
- **HumanEval pass@1**: {base_pass1:.1f}% (base) → {ft_pass1:.1f}% (fine-tuned) | **+{improvement:.1f}pp improvement**'''

    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Update Phase 3 status
    phase3_pattern = r'### 🚧 Phase 3: Evaluation & Analysis \(In Progress\)'
    phase3_replacement = '### ✅ Phase 3: Evaluation & Analysis (Complete)'
    content = content.replace(phase3_pattern, phase3_replacement)

    # Update evaluation checklist
    eval_pattern = r'- \[x\] Code generation samples.*\n- \[ \] HumanEval benchmark \(full evaluation\)'
    eval_replacement = f'''- [x] Code generation samples (Fibonacci, Palindrome, List Sum - all correct)
- [x] HumanEval benchmark (Base: {base_pass1:.1f}% → Fine-tuned: {ft_pass1:.1f}%, +{improvement:.1f}pp)'''
    content = re.sub(eval_pattern, eval_replacement, content)

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ README updated with HumanEval results:")
    print(f"   Base model: {base_pass1:.1f}%")
    print(f"   Fine-tuned: {ft_pass1:.1f}%")
    print(f"   Improvement: +{improvement:.1f}pp")

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    # Parse results
    results_dir = Path(__file__).parent.parent / "results" / "humaneval"

    base_results_file = results_dir / "base_model_samples.jsonl_results.json"
    ft_results_file = results_dir / "finetuned_model_samples.jsonl_results.json"

    if not base_results_file.exists() or not ft_results_file.exists():
        print("❌ Results files not found. Run HumanEval evaluation first.")
        sys.exit(1)

    with open(base_results_file) as f:
        base_results = json.load(f)

    with open(ft_results_file) as f:
        ft_results = json.load(f)

    base_pass1 = base_results['pass@1'] * 100
    ft_pass1 = ft_results['pass@1'] * 100
    improvement = ft_pass1 - base_pass1

    readme_path = Path(__file__).parent.parent / "README.md"
    update_readme_with_humaneval(readme_path, base_pass1, ft_pass1, improvement)
