# Data Directory

Complete data processing pipeline for LLM fine-tuning using The Stack dataset.

## Structure

```
data/
├── scripts/              # Data processing scripts
│   ├── download_stack.py       # Download The Stack dataset
│   ├── process_data.py         # Quality filtering pipeline
│   ├── create_instruction.py   # Convert to instruction format
│   ├── analyze_dataset.py      # Generate statistics & visualizations
│   ├── run_pipeline.py         # Run complete pipeline
│   └── README.md               # Detailed documentation
├── raw/                  # Downloaded raw data (gitignore)
├── processed/            # Processed data (gitignore)
│   ├── quality_filtered_10k.jsonl
│   ├── train.jsonl       # Training set (~9K samples)
│   └── eval.jsonl        # Evaluation set (~1K samples)
├── analysis/             # Dataset statistics (gitignore)
│   ├── dataset_stats.json
│   ├── dataset_distributions.png
│   └── quality_metrics.png
├── collection/           # [Legacy] Custom data collection scripts
├── processing/           # [Legacy] Custom processing scripts
├── augmentation/         # [Future] Data augmentation
└── benchmarks/           # [Future] Evaluation datasets
```

## Quick Start

### Run Complete Pipeline (Recommended)

```bash
cd data/scripts
python run_pipeline.py
```

This will:
1. Download 20K Python samples from The Stack
2. Filter to 10K high-quality samples
3. Convert to instruction format (9K train, 1K eval)
4. Generate statistics and visualizations

**Expected time on Mac M3 Pro:** 15-30 minutes

### Individual Steps

See [scripts/README.md](scripts/README.md) for detailed documentation.

#### 1. Download Data

```bash
cd data/scripts
python download_stack.py --samples 20000 --output ../raw/the_stack_python_20k.jsonl
```

#### 2. Quality Filtering

```bash
python process_data.py --input ../raw/the_stack_python_20k.jsonl --output ../processed/quality_filtered_10k.jsonl
```

#### 3. Create Instruction Format

```bash
python create_instruction.py --input ../processed/quality_filtered_10k.jsonl --output-dir ../processed
```

#### 4. Analyze Dataset

```bash
python analyze_dataset.py --input ../processed/train.jsonl --output-dir ../analysis
```

## Data Format

### Training Data (Chat Format)

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

## Quality Criteria

The processing pipeline ensures:
- ✅ Valid Python syntax (100%)
- ✅ Has docstrings (~80-90%)
- ✅ Has structure (functions/classes) (~100%)
- ✅ Low comment ratio (<30%)
- ✅ No duplicates (100% unique)
- ✅ Reproducible (fixed seed=42)

## Expected Output

After running the pipeline:

```
data/
├── raw/
│   └── the_stack_python_20k.jsonl        # 20K raw samples
├── processed/
│   ├── quality_filtered_10k.jsonl        # 10K quality-filtered
│   ├── train.jsonl                       # ~9K training samples
│   └── eval.jsonl                        # ~1K evaluation samples
└── analysis/
    ├── dataset_stats.json                # Statistics
    ├── dataset_distributions.png         # Distribution plots
    └── quality_metrics.png               # Quality metrics
```

## Requirements

```bash
pip install datasets tqdm matplotlib numpy
```

Or use the project's requirements:

```bash
pip install -r requirements/dev.txt
```

## Dataset Details

**Source:** The Stack (Dedup) - Python subset
- High-quality, deduplicated code from GitHub
- Publicly available on HuggingFace
- Used in real research (StarCoder)
- Permissive open-source licenses

**Subset Size:** 10K high-quality samples
- Perfect for Mac M3 Pro training (3-5 hours)
- Enough data for meaningful results
- Shows data efficiency (good for portfolio!)
- Cost: $0

## Zero-Cost Philosophy

This pipeline is designed to work entirely on Mac M3 Pro with no cloud costs:
- ✅ Streaming download for memory efficiency
- ✅ Local processing only
- ✅ Reproducible with fixed seeds
- ✅ Production-grade code quality
- ✅ Complete in 15-30 minutes

## Troubleshooting

See [scripts/README.md](scripts/README.md#troubleshooting) for common issues and solutions.

## Next Steps

After processing:
1. Verify data quality: Check `analysis/dataset_stats.json`
2. Review samples: Inspect `processed/train.jsonl`
3. Proceed to training: Use the processed files for fine-tuning

## References

- [The Stack Dataset](https://huggingface.co/datasets/bigcode/the-stack-dedup)
- [StarCoder Paper](https://arxiv.org/abs/2305.06161)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
