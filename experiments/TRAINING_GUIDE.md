# Training Guide

Complete guide for training models using the QLoRA fine-tuning pipeline.

## Quick Start

### Baseline Experiment (EXAONE-2.4B)

```bash
cd experiments/training
python train_qlora.py --config ../configs/exaone_2.4b.yaml
```

**Expected:**
- Training time: 3-5 hours on Mac M3 Pro
- Memory usage: ~12-14GB
- Checkpoints saved every 200 steps
- Evaluation every 200 steps
- Final model: `results/models/qlora-exaone-2.4b/`

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements/base.txt
```

Required packages:
- `peft>=0.6.0` - Parameter-efficient fine-tuning
- `accelerate>=0.24.0` - Distributed training support
- `wandb>=0.16.0` - Experiment tracking
- `scipy>=1.10.0` - Scientific computing

### 2. Prepare Data

Ensure data pipeline has been run:

```bash
cd data/scripts
python run_pipeline.py
```

Expected output:
- `data/processed/train.jsonl` - 9,123 samples
- `data/processed/eval.jsonl` - 1,014 samples

### 3. Optional: Set up Weights & Biases

For experiment tracking:

```bash
wandb login
```

Or disable in config:
```yaml
wandb:
  use_wandb: false
```

## Training Configurations

### Available Models

| Model | Config | Parameters | Memory | Training Time |
|-------|--------|-----------|---------|---------------|
| EXAONE-2.4B | `exaone_2.4b.yaml` | 2.4B | ~12GB | 3-5 hours |
| Llama-3.2-3B | `llama_3.2_3b.yaml` | 3B | ~14GB | 4-6 hours |
| Qwen2.5-Coder-3B | `qwen_2.5_3b.yaml` | 3B | ~14GB | 4-6 hours |

### Key Hyperparameters

**EXAONE-2.4B (Baseline):**
```yaml
# LoRA
lora_r: 16
lora_alpha: 32
target_modules: [q_proj, v_proj]

# Training
batch_size: 2
gradient_accumulation: 8  # Effective batch = 16
learning_rate: 2e-4
epochs: 3

# Memory
max_length: 1024
gradient_checkpointing: true
```

## Training Commands

### 1. Baseline Experiment

```bash
python train_qlora.py --config ../configs/exaone_2.4b.yaml
```

### 2. Model Comparison

Train all three models:

```bash
# EXAONE
python train_qlora.py --config ../configs/exaone_2.4b.yaml

# Llama
python train_qlora.py --config ../configs/llama_3.2_3b.yaml

# Qwen
python train_qlora.py --config ../configs/qwen_2.5_3b.yaml
```

### 3. Hyperparameter Tuning

Override config values:

```bash
# Different learning rate
python train_qlora.py \
  --config ../configs/exaone_2.4b.yaml \
  --learning_rate 1e-4

# Different LoRA rank
python train_qlora.py \
  --config ../configs/exaone_2.4b.yaml \
  --lora_r 32 \
  --lora_alpha 64
```

### 4. Resume from Checkpoint

```bash
python train_qlora.py \
  --config ../configs/exaone_2.4b.yaml \
  --resume_from_checkpoint results/models/qlora-exaone-2.4b/checkpoint-1000
```

## Monitoring Training

### Command Line Output

```
Step 10/3000: loss=2.345, lr=0.0002, tokens/sec=450
Step 20/3000: loss=2.123, lr=0.0002, tokens/sec=460
...
Evaluation at step 200: eval_loss=2.015, perplexity=7.5
```

### Weights & Biases

If enabled, view real-time metrics at: https://wandb.ai

Key metrics:
- Training loss
- Evaluation loss
- Perplexity
- Learning rate schedule
- GPU/Memory usage

### TensorBoard (Alternative)

```bash
tensorboard --logdir results/models/qlora-exaone-2.4b/logs
```

## Output Structure

```
results/models/qlora-exaone-2.4b/
├── checkpoint-200/              # Training checkpoint
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── optimizer.pt
├── checkpoint-400/
├── checkpoint-600/
├── final_model/                 # Best model
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── README.md
├── logs/                        # Training logs
│   ├── training.log
│   └── events.out.tfevents.*
└── config.yaml                  # Saved configuration
```

## Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Keep effective batch same
```

**Solution 2: Reduce sequence length**
```yaml
data:
  max_length: 512  # Down from 1024
```

**Solution 3: Enable more aggressive memory optimizations**
```yaml
memory:
  use_gradient_checkpointing: true
  optim: "paged_adamw_8bit"  # 8-bit optimizer
```

### Training Too Slow

**Solution 1: Increase batch size** (if memory allows)
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

**Solution 2: Use mixed precision** (CUDA only)
```yaml
hardware:
  use_fp16: true
```

**Solution 3: Reduce evaluation frequency**
```yaml
evaluation:
  eval_steps: 500  # Up from 200
```

### Model Not Learning

**Check 1: Learning rate**
- Too high: Loss explodes or NaN
- Too low: Loss doesn't decrease
- Try: 1e-5, 2e-5, 5e-5, 1e-4, 2e-4

**Check 2: Data quality**
```bash
# Inspect training samples
head -5 ../../data/processed/train.jsonl | python -m json.tool
```

**Check 3: LoRA configuration**
- Increase rank: `lora_r: 32 or 64`
- Adjust alpha: `lora_alpha = 2 * lora_r`

### CUDA vs MPS vs CPU

**CUDA (Linux/Windows with NVIDIA GPU):**
- Fastest training
- 4-bit quantization available
- Full features supported

**MPS (Mac M3 Pro):**
- Fast training on Apple Silicon
- No 4-bit quantization (falls back to FP16)
- Some CUDA-specific features disabled

**CPU (Fallback):**
- Very slow (10-20x slower)
- Use only for testing
- Not recommended for full training

## Best Practices

### 1. Start Small

Test with small subset first:
```bash
python train_qlora.py \
  --config ../configs/exaone_2.4b.yaml \
  --max_train_samples 100 \
  --max_steps 50
```

### 2. Monitor Early

Check first 100 steps:
- Loss should decrease
- Memory should be stable
- No NaN values

### 3. Save Often

```yaml
saving:
  save_steps: 200
  save_total_limit: 5  # Keep last 5 checkpoints
```

### 4. Evaluate Regularly

```yaml
evaluation:
  eval_steps: 200
  eval_accumulation_steps: 1
```

### 5. Log Everything

```yaml
logging:
  logging_steps: 10
  report_to: ["wandb", "tensorboard"]
```

## Expected Results

### Baseline (EXAONE-2.4B)

**Training Metrics:**
- Initial loss: ~3.5
- Final loss: ~1.8-2.0
- Final perplexity: ~6-7

**Evaluation Metrics:**
- Eval loss: ~2.0-2.2
- Eval perplexity: ~7-9

**HumanEval (after full training):**
- Base model: ~15-20% pass@1
- Fine-tuned: ~25-30% pass@1
- Target improvement: +10-15%

## Next Steps

After training completes:

1. **Evaluate Model**: Run HumanEval and MBPP benchmarks
   ```bash
   cd ../evaluation
   python evaluate_humaneval.py --model ../results/models/qlora-exaone-2.4b/final_model
   ```

2. **Compare Models**: If multiple models trained
   ```bash
   python compare_models.py --models exaone llama qwen
   ```

3. **Generate Samples**: Test model interactively
   ```bash
   python generate.py --model ../results/models/qlora-exaone-2.4b/final_model
   ```

4. **Deploy**: Create API or CLI tool
   ```bash
   cd ../../deployment/api
   python main.py --model ../../experiments/results/models/qlora-exaone-2.4b/final_model
   ```

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
