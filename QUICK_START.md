# 🚀 Quick Start Guide

korean-code-llm 프로젝트를 빠르게 시작하는 가이드입니다.

## 환경 설정

### 1. 가상환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

Mac (M3 Pro):
```bash
pip install -r requirements/train.txt
```

Windows/Linux (CUDA):
```bash
pip install -r requirements/train.txt
```

## 데이터 수집 및 준비

### 1. GitHub에서 한국어 코드 수집

```bash
# 기본 실행 (token 없이, rate limit 제한)
python data/collection/github_scraper.py \
    --max-repos 20 \
    --max-snippets 500 \
    --output data/raw/github_korean.jsonl

# GitHub token 사용 (권장)
python data/collection/github_scraper.py \
    --token YOUR_GITHUB_TOKEN \
    --max-repos 50 \
    --max-snippets 1000 \
    --output data/raw/github_korean.jsonl
```

**GitHub Token 발급:**
1. https://github.com/settings/tokens
2. "Generate new token" → "Classic"
3. `public_repo` 권한만 체크
4. Token 복사

### 2. 데이터 정제

```bash
python data/processing/cleaner.py \
    --input data/raw/github_korean.jsonl \
    --output data/cleaned/github_korean_cleaned.jsonl
```

### 3. 학습 데이터 준비

```bash
python data/processing/prepare_training_data.py \
    --input data/cleaned/github_korean_cleaned.jsonl \
    --output-dir data/processed \
    --train-ratio 0.9
```

이제 `data/processed/train.jsonl`과 `data/processed/eval.jsonl`이 생성됩니다.

## 모델 학습

### QLoRA 학습 (Mac M3 Pro)

```bash
python experiments/training/train_qlora.py \
    --model-name LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
    --train-file data/processed/train.jsonl \
    --eval-file data/processed/eval.jsonl \
    --output-dir results/models/exp-001 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --num-epochs 3 \
    --lora-rank 16
```

### Weights & Biases 사용 (선택)

```bash
# W&B 로그인
wandb login

# 학습 시 --wandb 플래그 추가
python experiments/training/train_qlora.py \
    --model-name LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
    --train-file data/processed/train.jsonl \
    --eval-file data/processed/eval.jsonl \
    --output-dir results/models/exp-001 \
    --wandb
```

## 전체 파이프라인 (한 번에 실행)

```bash
#!/bin/bash

# 1. 데이터 수집
python data/collection/github_scraper.py \
    --token $GITHUB_TOKEN \
    --max-repos 50 \
    --max-snippets 1000 \
    --output data/raw/github_korean.jsonl

# 2. 데이터 정제
python data/processing/cleaner.py \
    --input data/raw/github_korean.jsonl \
    --output data/cleaned/github_korean_cleaned.jsonl

# 3. 학습 데이터 준비
python data/processing/prepare_training_data.py \
    --input data/cleaned/github_korean_cleaned.jsonl \
    --output-dir data/processed

# 4. 모델 학습
python experiments/training/train_qlora.py \
    --train-file data/processed/train.jsonl \
    --eval-file data/processed/eval.jsonl \
    --output-dir results/models/exp-001 \
    --wandb

echo "✅ 전체 파이프라인 완료!"
```

## 예상 시간

| 단계 | 샘플 수 | 예상 시간 (Mac M3 Pro) |
|------|---------|------------------------|
| 데이터 수집 | 1,000개 | 30-60분 |
| 데이터 정제 | 1,000개 | 1-2분 |
| 학습 준비 | 1,000개 | <1분 |
| 모델 학습 | 1,000개 | 2-4시간 (3 epochs) |

## 메모리 사용량

| 모델 | 배치 크기 | 예상 메모리 |
|------|-----------|------------|
| EXAONE-2.4B | 4 | 8-10GB |
| EXAONE-2.4B | 2 | 6-8GB |

**팁**: 메모리 부족 시 `--batch-size 2`로 줄이세요.

## 문제 해결

### GitHub API Rate Limit 초과

```bash
# Token 없이 실행 시: 시간당 60 requests
# Token 사용 시: 시간당 5,000 requests

# 해결: GitHub Token 사용
python data/collection/github_scraper.py --token YOUR_TOKEN
```

### MPS 메모리 부족

```bash
# 배치 크기 줄이기
python experiments/training/train_qlora.py --batch-size 2

# 시퀀스 길이 줄이기
python experiments/training/train_qlora.py --max-length 1024
```

### 데이터가 없음

```bash
# 데이터 수집이 제대로 되었는지 확인
ls -lh data/raw/
cat data/raw/github_korean.jsonl | wc -l

# 샘플 데이터로 테스트
python data/processing/prepare_training_data.py \
    --input data/cleaned/github_korean_cleaned.jsonl \
    --output-dir data/processed \
    --max-samples 100
```

## 다음 단계

1. **평가**: 학습된 모델 평가하기
2. **배포**: API 서버로 배포하기
3. **튜닝**: 하이퍼파라미터 튜닝하기

자세한 내용은 [docs/00_PROJECT_MASTERPLAN.md](docs/00_PROJECT_MASTERPLAN.md)를 참고하세요.
