# Data Processing Scripts

This directory contains scripts for downloading, processing, and analyzing The Stack dataset for LLM fine-tuning.

## Directory Structure

```
data/
├── scripts/              # Processing scripts
│   ├── download_stack.py       # Download The Stack dataset
│   ├── process_data.py         # Quality filtering
│   ├── create_instruction.py   # Format conversion
│   ├── analyze_dataset.py      # Statistics & visualizations
│   └── run_pipeline.py         # Run complete pipeline
├── raw/                  # Downloaded raw data
├── processed/            # Processed data
└── analysis/             # Dataset statistics
```

## Quick Start

### Run Complete Pipeline

```bash
cd data/scripts
python run_pipeline.py
```

This will:
1. Download 20K Python samples from The Stack
2. Filter to 10K high-quality samples
3. Convert to instruction format (9K train, 1K eval)
4. Generate statistics and visualizations

### Individual Scripts

#### 1. Download Data

```bash
python download_stack.py --samples 20000 --output ../raw/the_stack_python_20k.jsonl
```

**Options:**
- `--samples`: Number of samples to download (default: 20000)
- `--output`: Output file path
- `--min-length`: Minimum code length (default: 100)
- `--max-length`: Maximum code length (default: 2000)
- `--seed`: Random seed (default: 42)

#### 2. Quality Filtering

```bash
python process_data.py --input ../raw/the_stack_python_20k.jsonl --output ../processed/quality_filtered_10k.jsonl
```

**Quality Criteria:**
- Valid Python syntax (AST parsing)
- Has docstrings (`"""` or `'''`)
- Has function or class definitions
- Comment ratio < 30%
- No duplicates (content hash)

**Options:**
- `--input`: Input JSONL file
- `--output`: Output JSONL file
- `--target-samples`: Target number of samples (default: 10000)
- `--max-comment-ratio`: Maximum comment ratio (default: 0.3)
- `--no-require-docstring`: Don't require docstrings
- `--no-require-structure`: Don't require functions/classes

#### 3. Create Instruction Format

```bash
python create_instruction.py --input ../processed/quality_filtered_10k.jsonl --output-dir ../processed
```

**Output Format:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "def calculate_sum(a: int, b: int) -> int:\nCalculate the sum of two integers."
    },
    {
      "role": "assistant",
      "content": "return a + b"
    }
  ],
  "metadata": {
    "source": "the_stack",
    "lang": "Python",
    "repo": "example/repo"
  }
}
```

**Options:**
- `--input`: Input JSONL file
- `--output-dir`: Output directory
- `--train-split`: Training data ratio (default: 0.9)
- `--seed`: Random seed (default: 42)

#### 4. Analyze Dataset

```bash
python analyze_dataset.py --input ../processed/train.jsonl --output-dir ../analysis
```

**Outputs:**
- `dataset_stats.json`: Summary statistics
- `dataset_distributions.png`: Code length, token count, functions/classes distributions
- `quality_metrics.png`: Type hints and docstring coverage

**Options:**
- `--input`: Input JSONL file
- `--output-dir`: Output directory

## Requirements

Install required dependencies:

```bash
pip install datasets tqdm matplotlib numpy
```

Or use the project's requirements:

```bash
cd ../..
pip install -r requirements/dev.txt
```

## Expected Output

After running the complete pipeline:

```
data/
├── raw/
│   └── the_stack_python_20k.jsonl        # 20K raw samples
├── processed/
│   ├── quality_filtered_10k.jsonl        # 10K quality-filtered samples
│   ├── train.jsonl                       # ~9K training samples
│   └── eval.jsonl                        # ~1K evaluation samples
└── analysis/
    ├── dataset_stats.json                # Statistics
    ├── dataset_distributions.png         # Distributions plot
    └── quality_metrics.png               # Quality metrics plot
```

## Performance

On Mac M3 Pro:
- Download: ~10-20 minutes (depends on network)
- Quality filtering: ~2-5 minutes
- Instruction conversion: ~1-2 minutes
- Analysis: ~1-2 minutes

**Total time: ~15-30 minutes**

## Data Quality

The processed dataset ensures:
- ✅ Valid Python syntax (100%)
- ✅ Has docstrings (~80-90%)
- ✅ Has structure (functions/classes) (~100%)
- ✅ Low comment ratio (<30%)
- ✅ No duplicates (100% unique)
- ✅ Reproducible (fixed seed=42)

## Troubleshooting

### Issue: Download fails with authentication error

**Solution:** HuggingFace datasets library should work without authentication for public datasets. If you encounter issues:

```bash
huggingface-cli login
```

### Issue: Not enough samples after filtering

**Solution:** Adjust quality filters or download more data:

```bash
# Relax filters
python process_data.py --max-comment-ratio 0.5 --no-require-docstring

# Or download more data
python download_stack.py --samples 50000
```

### Issue: Memory error during download

**Solution:** The script uses streaming by default. If issues persist, reduce batch size or process in smaller chunks.

## Next Steps

After processing the data:
1. Verify data quality: Check `analysis/dataset_stats.json`
2. Review samples: Open `processed/train.jsonl` and inspect
3. Proceed to training: Use `processed/train.jsonl` and `processed/eval.jsonl` for fine-tuning

## Notes

- All scripts are reproducible with `seed=42`
- Scripts use absolute paths for reliability
- Progress bars show real-time status
- Comprehensive error handling and logging
- Production-grade code quality

## License

The Stack dataset is released under permissive open-source licenses. Always check individual repository licenses when using the data.
