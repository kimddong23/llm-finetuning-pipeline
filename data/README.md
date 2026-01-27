# Data Directory

데이터 수집, 처리, 증강 파이프라인을 포함합니다.

## 구조

```
data/
├── collection/         # 데이터 수집 스크립트
│   ├── github_scraper.py
│   ├── stackoverflow_ko.py
│   └── docs_scraper.py
├── processing/         # 전처리
│   ├── cleaner.py
│   ├── deduplication.py
│   └── quality_filter.py
├── augmentation/       # 데이터 증강
│   ├── backtranslation.py
│   └── synthetic_gen.py
├── benchmarks/         # 평가 데이터셋
│   ├── humaneval_ko.json
│   ├── mbpp_ko.json
│   └── custom_bench.json
├── raw/               # 원본 데이터 (gitignore)
└── processed/         # 처리된 데이터 (gitignore)
```

## 사용법

### 1. 데이터 수집

```bash
# GitHub에서 한국어 코드 수집
python data/collection/github_scraper.py --language python --min-stars 10

# Stack Overflow 데이터 수집
python data/collection/stackoverflow_ko.py --min-score 5
```

### 2. 데이터 전처리

```bash
# 데이터 정제
python data/processing/cleaner.py --input data/raw --output data/processed

# 중복 제거
python data/processing/deduplication.py --input data/processed
```

### 3. 데이터 증강

```bash
# 역번역을 통한 증강
python data/augmentation/backtranslation.py --input data/processed
```

## 데이터 포맷

### 학습 데이터 (Chat 형식)

```json
{
  "messages": [
    {
      "role": "user",
      "content": "리스트에서 짝수만 필터링하는 함수를 만들어줘"
    },
    {
      "role": "assistant",
      "content": "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]"
    }
  ]
}
```

## Phase 1 작업 목록

- [ ] GitHub 스크래퍼 구현
- [ ] Stack Overflow 크롤러 구현
- [ ] 데이터 정제 파이프라인
- [ ] 품질 필터링 알고리즘
- [ ] 데이터셋 통계 생성

## 참고

- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [DVC 문서](https://dvc.org/doc)
