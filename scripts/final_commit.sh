#!/bin/bash

# Final commit script after HumanEval completion
# This script will be run manually after all results are ready

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "="
echo "PREPARING FINAL COMMIT"
echo "======================================================================"

# Check if results exist
if [ ! -f "results/humaneval/base_model_samples.jsonl_results.json" ]; then
    echo "❌ Base model results not found"
    exit 1
fi

if [ ! -f "results/humaneval/finetuned_model_samples.jsonl_results.json" ]; then
    echo "❌ Fine-tuned model results not found"
    exit 1
fi

echo "✅ HumanEval results found"

# Run analysis
echo ""
echo "Running analysis..."
cd experiments/analysis
python analyze_humaneval_results.py
python plot_benchmark_comparison.py
cd ../..

# Update README
echo ""
echo "Updating README..."
cd scripts
python update_readme_with_results.py
cd ..

# Stage all changes
echo ""
echo "Staging changes..."
git add -A

# Show what will be committed
echo ""
echo "Changes to be committed:"
git status

# Create commit
echo ""
echo "Creating commit..."
git commit -m "Complete HumanEval benchmark evaluation and analysis

Final Results:
- Base model vs Fine-tuned comparison complete
- Benchmark visualizations generated
- Technical report updated with all metrics
- README updated with final performance numbers
- Error analysis and improvement metrics documented

This completes Phase 3 (Evaluation & Analysis) of the project."

echo ""
echo "✅ Commit created successfully!"
echo ""
echo "Next step: git push origin main"
