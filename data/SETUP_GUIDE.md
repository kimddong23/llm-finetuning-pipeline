# Data Pipeline Setup Guide

Step-by-step guide to set up and run the data processing pipeline.

## Prerequisites

- Mac M3 Pro (or any system with Python 3.8+)
- 18GB RAM recommended (minimum 8GB)
- 5GB free disk space
- Internet connection (for downloading dataset)

## Installation

### 1. Navigate to Project Directory

```bash
cd /Users/shinjuyong/Desktop/사이드\ 프로젝트/LLM\ 파인튜닝/llm-finetuning-pipeline
```

### 2. Install Dependencies

**Option A: Install project requirements (recommended)**

```bash
pip install -r requirements/base.txt
```

This installs all base dependencies including:
- `datasets` - HuggingFace datasets
- `transformers` - Transformers library
- `torch` - PyTorch
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `tqdm` - Progress bars
- `matplotlib` - Visualizations
- Other utilities

**Option B: Install only data pipeline dependencies**

```bash
pip install datasets tqdm matplotlib numpy
```

This is lighter if you only want to run the data pipeline.

### 3. Verify Installation

```bash
cd data/scripts
python test_pipeline.py
```

**Expected output:**
```
================================================================================
TESTING DATA PROCESSING PIPELINE
================================================================================

Testing imports...
✓ All imports successful

Testing AST parsing...
✓ AST parsing test passed

Testing quality filter...
✓ Quality filter test passed

Testing instruction extraction...
✓ Instruction extraction test passed

Testing dataset analyzer...
✓ Dataset analyzer test passed

Testing file structure...
✓ File structure test passed

================================================================================
TEST RESULTS: 6/6 passed, 0/6 failed
================================================================================
✓ All tests passed! Pipeline is ready to use.
```

If all tests pass, you're ready to proceed!

## Quick Start

### Run Complete Pipeline

```bash
cd data/scripts
python run_pipeline.py
```

This will:
1. Download 20K Python samples from The Stack (~10-20 minutes)
2. Filter to 10K high-quality samples (~2-5 minutes)
3. Convert to instruction format with 90/10 split (~1-2 minutes)
4. Generate statistics and visualizations (~1-2 minutes)

**Total time: 15-30 minutes**

### Monitor Progress

The pipeline shows progress for each step:

```
================================================================================
STEP 1: Downloading The Stack dataset
================================================================================
Downloading: 100%|████████████████████| 20000/20000 [15:23<00:00, 21.67it/s]
✓ Step 1 completed in 923.1s

================================================================================
STEP 2: Applying quality filters
================================================================================
Filtering: 100%|█████████████████████| 10000/10000 [02:35<00:00, 64.32it/s]
✓ Step 2 completed in 155.6s

...
```

## Verify Output

### 1. Check Directory Structure

```bash
cd ..  # Back to data/ directory
ls -lh raw/
ls -lh processed/
ls -lh analysis/
```

**Expected:**
```
raw/
  the_stack_python_20k.jsonl     (~50-100 MB)

processed/
  quality_filtered_10k.jsonl     (~25-50 MB)
  train.jsonl                    (~22-45 MB)
  eval.jsonl                     (~2-5 MB)

analysis/
  dataset_stats.json             (~5 KB)
  dataset_distributions.png      (~200 KB)
  quality_metrics.png            (~150 KB)
```

### 2. View Statistics

```bash
cat analysis/dataset_stats.json | python -m json.tool
```

### 3. View Visualizations

```bash
open analysis/dataset_distributions.png
open analysis/quality_metrics.png
```

### 4. Inspect Training Data

```bash
# View first training sample
head -n 1 processed/train.jsonl | python -m json.tool
```

**Example output:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "def calculate_sum(a: int, b: int) -> int:\n    \"\"\"Calculate the sum of two integers.\"\"\""
    },
    {
      "role": "assistant",
      "content": "return a + b"
    }
  ],
  "metadata": {
    "source": "the_stack",
    "lang": "Python",
    "repo": "example/math-utils"
  }
}
```

## Individual Scripts

If you want to run steps individually:

### Step 1: Download

```bash
cd scripts
python download_stack.py \
  --samples 20000 \
  --output ../raw/the_stack_python_20k.jsonl \
  --min-length 100 \
  --max-length 2000 \
  --seed 42
```

### Step 2: Filter

```bash
python process_data.py \
  --input ../raw/the_stack_python_20k.jsonl \
  --output ../processed/quality_filtered_10k.jsonl \
  --target-samples 10000 \
  --max-comment-ratio 0.3 \
  --seed 42
```

### Step 3: Convert

```bash
python create_instruction.py \
  --input ../processed/quality_filtered_10k.jsonl \
  --output-dir ../processed \
  --train-split 0.9 \
  --seed 42
```

### Step 4: Analyze

```bash
python analyze_dataset.py \
  --input ../processed/train.jsonl \
  --output-dir ../analysis
```

## Customization

### Download More Samples

```bash
python run_pipeline.py --samples 50000 --target 25000
```

This downloads 50K samples and filters to 25K.

### Relax Quality Filters

```bash
python process_data.py \
  --input ../raw/the_stack_python_20k.jsonl \
  --output ../processed/quality_filtered_10k.jsonl \
  --max-comment-ratio 0.5 \
  --no-require-docstring
```

### Change Train/Eval Split

```bash
python create_instruction.py \
  --input ../processed/quality_filtered_10k.jsonl \
  --output-dir ../processed \
  --train-split 0.8  # 80/20 instead of 90/10
```

## Troubleshooting

### Issue: `No module named 'datasets'`

**Solution:**
```bash
pip install datasets
```

Or install all requirements:
```bash
pip install -r requirements/base.txt
```

### Issue: Download is very slow

**Solutions:**
1. **Check network connection** - Ensure stable internet
2. **Run during off-peak hours** - HuggingFace servers may be busy
3. **Start with smaller sample** - Try `--samples 5000` first

### Issue: Not enough quality samples

**Symptoms:**
```
Only 7500 samples passed quality filters (target: 10000)
```

**Solutions:**

1. **Download more raw data:**
   ```bash
   python download_stack.py --samples 30000
   ```

2. **Relax filters:**
   ```bash
   python process_data.py \
     --max-comment-ratio 0.5 \
     --no-require-docstring
   ```

### Issue: Memory error

**Solutions:**
1. **Close other applications** - Free up RAM
2. **Use smaller sample size** - Try 10K instead of 20K
3. **Scripts already use streaming** - Should not need more memory

### Issue: Test failures

Run the test script to diagnose:
```bash
python test_pipeline.py
```

Check which test failed and:
- **Imports test**: Install missing packages
- **AST parsing test**: Python version issue (need 3.8+)
- **Quality filter test**: Logic error in code
- **File structure test**: Permission issues

## Performance Tips

### Mac M3 Pro Optimization

The pipeline is already optimized for Mac M3 Pro:
- Uses streaming to minimize memory
- Processes in batches
- Leverages built-in AST parser

### Faster Processing

1. **SSD recommended** - Faster I/O
2. **Free up RAM** - Close other apps
3. **Batch size** - Scripts use optimal batch sizes

## Next Steps

After completing the pipeline:

1. **Review data quality**
   - Check statistics: `data/analysis/dataset_stats.json`
   - View plots: `data/analysis/*.png`

2. **Inspect samples**
   - Random samples from train.jsonl
   - Verify instruction quality

3. **Proceed to training**
   - Use `data/processed/train.jsonl`
   - Use `data/processed/eval.jsonl`
   - See `experiments/` for training scripts

4. **Document findings**
   - Note data characteristics
   - Record any issues encountered
   - Update README if needed

## Support

For issues or questions:
1. Check [scripts/README.md](scripts/README.md) for detailed docs
2. Check [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) for architecture
3. Review error messages carefully
4. Ensure all dependencies are installed

## Success Criteria

You've successfully completed the pipeline when:

- ✅ All tests pass (`test_pipeline.py`)
- ✅ 9K+ training samples in `processed/train.jsonl`
- ✅ 1K+ eval samples in `processed/eval.jsonl`
- ✅ Statistics generated in `analysis/`
- ✅ Visualizations created (PNG files)
- ✅ No errors in pipeline output

## Time Estimates

| Task | Time (Mac M3 Pro) |
|------|-------------------|
| Install dependencies | 2-5 minutes |
| Run tests | <1 minute |
| Download (20K samples) | 10-20 minutes |
| Quality filtering | 2-5 minutes |
| Format conversion | 1-2 minutes |
| Generate analysis | 1-2 minutes |
| **Total first run** | **15-30 minutes** |
| **Re-run (cached)** | **5-10 minutes** |

## Disk Space

| Component | Size |
|-----------|------|
| Raw data (20K) | 50-100 MB |
| Processed data (10K) | 25-50 MB |
| Train/eval split | 22-47 MB |
| Analysis | 1-2 MB |
| **Total** | **~100-200 MB** |

## Memory Usage

| Step | Peak Memory |
|------|-------------|
| Download | <2 GB |
| Filtering | <1 GB |
| Conversion | <1 GB |
| Analysis | <1 GB |

**Recommended:** 8GB+ RAM
**Optimal:** 16GB+ RAM

---

**You're ready to process data!** 🚀

Run `python run_pipeline.py` and the pipeline will handle everything automatically.
