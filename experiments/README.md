# Experiments Directory

모델 학습, 평가, 분석을 위한 실험 코드를 포함합니다.

## 구조

```
experiments/
├── configs/           # 실험 설정 파일 (YAML)
├── training/          # 학습 스크립트
├── evaluation/        # 평가 스크립트
└── analysis/          # 결과 분석
```

## 빠른 시작

### 1. 학습 실행

```bash
# QLoRA 학습 (Mac M3 Pro)
python experiments/training/train_qlora.py \
    --config experiments/configs/base_2.4b.yaml \
    --output-dir results/runs/exp-001

# 클라우드에서 Full Fine-tuning
python experiments/training/train_full_ft.py \
    --config experiments/configs/full_ft_3b.yaml
```

### 2. 평가 실행

```bash
# HumanEval 평가
python experiments/evaluation/run_humaneval.py \
    --model-path results/models/exp-001

# 모든 벤치마크 실행
python experiments/evaluation/run_all_benchmarks.py \
    --model-path results/models/exp-001
```

### 3. 결과 분석

```bash
# 에러 분석
python experiments/analysis/error_analysis.py \
    --results results/runs/exp-001

# 성능 시각화
python experiments/analysis/visualizations.py \
    --runs results/runs/exp-*
```

## 실험 설정 예시

```yaml
# experiments/configs/base_2.4b.yaml
model:
  name: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
  max_seq_length: 2048

training:
  method: qlora
  lora_rank: 16
  lora_alpha: 16
  learning_rate: 2e-4
  batch_size: 4
  gradient_accumulation_steps: 2
  num_epochs: 3
  warmup_steps: 100

data:
  train_file: data/processed/train.json
  eval_file: data/processed/eval.json
  max_samples: 10000

logging:
  use_wandb: true
  project_name: korean-code-llm
  log_every_n_steps: 10
```

## Phase 2-3 작업 목록

**Phase 2: 학습 & 실험 (Week 3-6)**
- [ ] 베이스라인 학습 (EXAONE-2.4B)
- [ ] 3개 베이스 모델 비교
- [ ] QLoRA vs LoRA 실험
- [ ] Ablation study
- [ ] 실험 결과 시각화

**Phase 3: 평가 시스템 (Week 7-8)**
- [ ] HumanEval-Ko 구현
- [ ] MBPP-Ko 구현
- [ ] 커스텀 벤치마크
- [ ] Human evaluation 프로토콜

## 실험 추적

[Weights & Biases](https://wandb.ai)를 사용하여 실험을 추적합니다:

```python
import wandb

wandb.init(
    project="korean-code-llm",
    config={
        "model": "EXAONE-2.4B",
        "method": "qlora",
        "lr": 2e-4,
    }
)
```

## 참고

- [Transformers 문서](https://huggingface.co/docs/transformers)
- [PEFT 가이드](https://huggingface.co/docs/peft)
- [W&B 문서](https://docs.wandb.ai)
