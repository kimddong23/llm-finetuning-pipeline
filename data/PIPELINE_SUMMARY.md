# Data Pipeline Summary

Complete data processing pipeline for LLM fine-tuning project.

## Overview

This pipeline downloads and processes The Stack dataset (Python subset) to create high-quality training data for code generation fine-tuning.

**Key Features:**
- ✅ Zero-cost (uses free HuggingFace datasets)
- ✅ Mac M3 Pro optimized (streaming, memory efficient)
- ✅ Production-grade code quality
- ✅ Fully reproducible (seed=42)
- ✅ Comprehensive quality filtering
- ✅ Automatic statistics & visualizations

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PROCESSING PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  STEP 1: DOWNLOAD │
│  download_stack.py│
└────────┬─────────┘
         │ 20K raw samples
         │ Filter: 100-2000 chars
         │ Source: The Stack (HuggingFace)
         ▼
┌──────────────────┐
│  STEP 2: FILTER   │
│  process_data.py  │
└────────┬─────────┘
         │ 10K quality samples
         │ - Valid syntax (AST)
         │ - Has docstrings
         │ - Has structure (func/class)
         │ - Low comment ratio (<30%)
         │ - No duplicates
         ▼
┌──────────────────┐
│  STEP 3: CONVERT  │
│create_instruction.py│
└────────┬─────────┘
         │ Instruction format
         │ - Extract signatures
         │ - Extract docstrings
         │ - Create user/assistant pairs
         │ - Train/eval split (90/10)
         ▼
┌──────────────────┐
│  STEP 4: ANALYZE  │
│analyze_dataset.py │
└────────┬─────────┘
         │ Statistics & Plots
         │ - Code length distribution
         │ - Token count distribution
         │ - Quality metrics
         │ - Visualizations
         ▼
   ┌─────────────┐
   │ READY FOR   │
   │ FINE-TUNING │
   └─────────────┘
```

## File Structure

```
data/
├── scripts/
│   ├── download_stack.py       # Step 1: Download
│   ├── process_data.py         # Step 2: Quality filter
│   ├── create_instruction.py   # Step 3: Format conversion
│   ├── analyze_dataset.py      # Step 4: Analysis
│   ├── run_pipeline.py         # Run all steps
│   ├── test_pipeline.py        # Test pipeline
│   └── README.md               # Detailed docs
├── raw/                        # 20K raw samples
│   └── the_stack_python_20k.jsonl
├── processed/                  # Processed data
│   ├── quality_filtered_10k.jsonl
│   ├── train.jsonl            # 9K samples
│   └── eval.jsonl             # 1K samples
└── analysis/                   # Statistics
    ├── dataset_stats.json
    ├── dataset_distributions.png
    └── quality_metrics.png
```

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/shinjuyong/Desktop/사이드\ 프로젝트/LLM\ 파인튜닝/llm-finetuning-pipeline
pip install -r requirements/base.txt
```

### 2. Test Pipeline

```bash
cd data/scripts
python test_pipeline.py
```

This verifies:
- All dependencies are installed
- All components work correctly
- File I/O operations succeed

### 3. Run Complete Pipeline

```bash
python run_pipeline.py
```

**Expected time:** 15-30 minutes on Mac M3 Pro

**Output:**
- 9K training samples
- 1K evaluation samples
- Complete statistics
- Visualization plots

## Data Quality Guarantees

### Input Filtering
- Code length: 100-2000 characters
- Language: Python only
- Source: High-quality GitHub repositories

### Quality Checks
1. **Syntax Validity** (100%)
   - All code passes `ast.parse()`
   - No syntax errors

2. **Documentation** (~80-90%)
   - Has docstrings (`"""` or `'''`)
   - Meaningful function/class documentation

3. **Structure** (~100%)
   - Contains `def` or `class`
   - Not just script code

4. **Comment Ratio** (100%)
   - Comments < 30% of lines
   - Filters over-commented code

5. **Uniqueness** (100%)
   - MD5 hash deduplication
   - No duplicate samples

## Output Format

### Chat Format (train.jsonl, eval.jsonl)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "def calculate_area(radius: float) -> float:\nCalculate the area of a circle given its radius."
    },
    {
      "role": "assistant",
      "content": "import math\nreturn math.pi * radius ** 2"
    }
  ],
  "metadata": {
    "source": "the_stack",
    "lang": "Python",
    "repo": "example/math-utils"
  }
}
```

### Statistics Format (dataset_stats.json)

```json
{
  "total_samples": 9000,
  "code_length": {
    "min": 50,
    "max": 1500,
    "mean": 350.5,
    "median": 280.0,
    "std": 200.3
  },
  "token_count": {
    "min": 12,
    "max": 375,
    "mean": 87.6,
    "median": 70.0,
    "std": 50.1
  },
  "percentage_with_type_hints": 65.3,
  "percentage_with_docstrings": 88.7
}
```

## Performance Benchmarks

### Mac M3 Pro (18GB RAM)

| Step | Time | Memory | Output |
|------|------|--------|--------|
| Download | 10-20 min | <2GB | 20K samples |
| Filter | 2-5 min | <1GB | 10K samples |
| Convert | 1-2 min | <1GB | 9K train + 1K eval |
| Analyze | 1-2 min | <1GB | Stats + plots |
| **Total** | **15-30 min** | **<2GB** | **Complete dataset** |

## Reproducibility

All scripts use fixed seeds for reproducibility:

```python
seed = 42  # Fixed for all operations
random.seed(seed)
```

**Guarantees:**
- Same input → Same output
- Deterministic sampling
- Reproducible splits
- Identical train/eval sets

## Customization

### Adjust Sample Counts

```bash
# Download 50K, filter to 25K
python run_pipeline.py --samples 50000 --target 25000
```

### Relax Quality Filters

```bash
# Allow higher comment ratio, no docstring requirement
python process_data.py \
  --max-comment-ratio 0.5 \
  --no-require-docstring \
  --input ../raw/the_stack_python_20k.jsonl \
  --output ../processed/quality_filtered_10k.jsonl
```

### Custom Train/Eval Split

```bash
# 80/20 split instead of 90/10
python create_instruction.py \
  --train-split 0.8 \
  --input ../processed/quality_filtered_10k.jsonl \
  --output-dir ../processed
```

## Troubleshooting

### Issue: Download is slow

**Cause:** Network bandwidth or HuggingFace server load

**Solution:**
- Use `streaming=True` (already enabled)
- Run during off-peak hours
- Consider smaller sample size initially

### Issue: Quality filtering produces too few samples

**Cause:** Filters are too strict for the data

**Solution:**
```bash
# Relax filters
python process_data.py \
  --max-comment-ratio 0.5 \
  --no-require-docstring \
  --input ../raw/the_stack_python_20k.jsonl
```

Or download more raw data:
```bash
python download_stack.py --samples 50000
```

### Issue: Memory error during processing

**Cause:** Insufficient RAM

**Solution:**
- Scripts already use streaming
- Process smaller batches
- Close other applications

### Issue: Import errors

**Cause:** Missing dependencies

**Solution:**
```bash
pip install datasets tqdm matplotlib numpy
# Or
pip install -r requirements/base.txt
```

## Next Steps

After completing the pipeline:

1. **Verify Data Quality**
   ```bash
   # Check statistics
   cat data/analysis/dataset_stats.json

   # View plots
   open data/analysis/dataset_distributions.png
   open data/analysis/quality_metrics.png
   ```

2. **Inspect Samples**
   ```bash
   # View first few training samples
   head -n 5 data/processed/train.jsonl | jq .
   ```

3. **Proceed to Training**
   - Use `data/processed/train.jsonl` for training
   - Use `data/processed/eval.jsonl` for evaluation
   - See `experiments/` directory for training scripts

## Technical Details

### Dependencies
- `datasets` - HuggingFace datasets library
- `tqdm` - Progress bars
- `matplotlib` - Visualizations
- `numpy` - Numerical operations
- `ast` - Python AST parsing (stdlib)
- `json` - JSON handling (stdlib)

### Design Decisions

1. **Streaming Download**
   - Memory efficient
   - Handles large datasets
   - Filters during download

2. **AST-based Validation**
   - Guarantees valid Python
   - Extracts structural info
   - Enables smart filtering

3. **MD5 Deduplication**
   - Fast hashing
   - Efficient storage
   - Catches exact duplicates

4. **Chat Format**
   - Compatible with modern LLMs
   - Clear user/assistant roles
   - Includes metadata

5. **Reproducible Seeds**
   - Deterministic results
   - Enables debugging
   - Scientific rigor

## License

The Stack dataset is released under permissive open-source licenses. Individual code samples retain their original repository licenses. Always verify license compatibility for your use case.

## References

- [The Stack Dataset](https://huggingface.co/datasets/bigcode/the-stack-dedup)
- [StarCoder Paper](https://arxiv.org/abs/2305.06161)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets)
- [MASTERPLAN_V2.md](../docs/MASTERPLAN_V2.md)

## Authors

Created for the LLM Fine-tuning Pipeline project - a production-grade portfolio project for AI/ML Engineer positions.

**Philosophy:** Methodology over domain, reproducibility first, zero-cost execution.
