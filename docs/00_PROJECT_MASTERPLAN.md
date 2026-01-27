# 🚀 Korean Code LLM - Production-Grade Masterplan
## AI/ML 엔지니어 포트폴리오 프로젝트

> **목표**: 한국어로 설명하고 한국 개발자가 작성하는 스타일의 코드를 생성하는 LLM
> **철학**: "누가 봐도 대단하다고 느낄 수 있는 완성도"

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | `ko-code-llm` (가칭, 더 나은 이름 검토 필요) |
| **타겟** | AI/ML 엔지니어 채용 포트폴리오 |
| **기간** | 3-4개월 (집중 투자) |
| **환경** | MacBook M3 Pro (18GB) + 클라우드 리소스 |
| **차별점** | 재현성 + 벤치마크 + 실용성 + 교육적 가치 **모두** |

---

## 🎯 Why Korean Code LLM?

### AI/ML 엔지니어 취업 관점

| 증명 가능한 역량 | 이 프로젝트로 증명 |
|----------------|-------------------|
| **LLM 파인튜닝** | QLoRA, LoRA, Full FT 비교 실험 |
| **모델 평가** | 다양한 벤치마크, 커스텀 평가 파이프라인 |
| **데이터 엔지니어링** | 데이터 수집, 전처리, 품질 관리, 증강 |
| **최적화** | 양자화, 추론 최적화, 메모리 효율화 |
| **MLOps** | 실험 관리, 모델 버저닝, CI/CD |
| **배포** | API 서빙, 확장 개발, 실사용 환경 |

### 차별점 (경쟁 프로젝트 대비)

| 차원 | 대부분의 포트폴리오 | 이 프로젝트 |
|------|---------------------|------------|
| **재현성** | 코드만 | Docker + seed 고정 + 상세 로그 |
| **평가** | 1-2개 지표 | 5+ 벤치마크 + human eval |
| **실용성** | 노트북 데모 | API + CLI + VS Code 확장 |
| **문서** | README | 기술 블로그 + 튜토리얼 + API 문서 |
| **커뮤니티** | 혼자 작업 | 이슈 템플릿, 기여 가이드, 토론 |

---

## 🏗️ 프로젝트 아키텍처

### 전체 구조

```
ko-code-llm/
├── 📊 data/                        # 데이터 파이프라인
│   ├── collection/                 # 데이터 수집 스크립트
│   │   ├── github_scraper.py      # GitHub 한국어 코드
│   │   ├── stackoverflow_ko.py    # 한국 Stack Overflow
│   │   └── docs_scraper.py        # 한국어 기술 문서
│   ├── processing/                 # 전처리
│   │   ├── cleaner.py             # 데이터 정제
│   │   ├── deduplication.py       # 중복 제거
│   │   └── quality_filter.py      # 품질 필터링
│   ├── augmentation/               # 데이터 증강
│   │   ├── backtranslation.py     # 역번역
│   │   └── synthetic_gen.py       # 합성 데이터
│   └── benchmarks/                 # 평가 데이터셋
│       ├── humaneval_ko.json      # HumanEval 한국어
│       ├── mbpp_ko.json           # MBPP 한국어
│       └── custom_bench.json      # 커스텀 벤치마크
│
├── 🧪 experiments/                 # 실험 관리
│   ├── configs/                    # 실험 설정
│   │   ├── base_2.4b.yaml        # 2.4B 모델 설정
│   │   ├── base_3b.yaml          # 3B 모델 설정
│   │   └── ablation/              # Ablation study
│   ├── training/                   # 학습 스크립트
│   │   ├── train_qlora.py        # QLoRA 학습
│   │   ├── train_lora.py         # LoRA 학습
│   │   └── train_full_ft.py      # Full fine-tuning
│   ├── evaluation/                 # 평가
│   │   ├── run_humaneval.py      # HumanEval 평가
│   │   ├── run_mbpp.py           # MBPP 평가
│   │   ├── run_custom.py         # 커스텀 평가
│   │   └── human_eval.py         # 사람 평가
│   └── analysis/                   # 분석
│       ├── error_analysis.py      # 에러 분석
│       ├── performance_profiling.py
│       └── visualizations.py      # 시각화
│
├── 🚀 deployment/                  # 배포
│   ├── api/                        # REST API
│   │   ├── main.py               # FastAPI 서버
│   │   ├── models.py             # 데이터 모델
│   │   └── routers/              # API 라우터
│   ├── vscode-extension/          # VS Code 확장
│   │   ├── src/
│   │   ├── package.json
│   │   └── README.md
│   ├── cli/                        # CLI 도구
│   │   └── kocode.py
│   └── demo/                       # Gradio 데모
│       └── app.py
│
├── 🐳 infrastructure/              # 인프라
│   ├── docker/
│   │   ├── Dockerfile.train      # 학습용
│   │   ├── Dockerfile.serve      # 서빙용
│   │   └── docker-compose.yml
│   ├── ci/
│   │   └── .github/
│   │       └── workflows/
│   │           ├── test.yml      # 테스트 자동화
│   │           ├── benchmark.yml # 벤치마크 자동화
│   │           └── deploy.yml    # 배포 자동화
│   └── monitoring/
│       ├── prometheus.yml        # 메트릭 수집
│       └── grafana/              # 대시보드
│
├── 📚 docs/                        # 문서
│   ├── README.md                  # 메인 README
│   ├── CONTRIBUTING.md            # 기여 가이드
│   ├── ARCHITECTURE.md            # 아키텍처 설명
│   ├── BENCHMARKS.md              # 벤치마크 결과
│   ├── tutorials/                 # 튜토리얼
│   │   ├── 01_quick_start.md
│   │   ├── 02_fine_tuning.md
│   │   └── 03_deployment.md
│   └── blog_posts/                # 기술 블로그
│       ├── 01_project_intro.md
│       ├── 02_data_pipeline.md
│       ├── 03_training_process.md
│       └── 04_evaluation.md
│
├── 🧪 tests/                       # 테스트
│   ├── unit/                      # 단위 테스트
│   ├── integration/               # 통합 테스트
│   └── e2e/                       # E2E 테스트
│
├── 📊 results/                     # 실험 결과
│   ├── runs/                      # 실험 런
│   ├── models/                    # 체크포인트
│   └── reports/                   # 리포트
│
├── requirements/                   # 의존성
│   ├── base.txt                   # 기본
│   ├── train.txt                  # 학습
│   ├── serve.txt                  # 서빙
│   └── dev.txt                    # 개발
│
├── pyproject.toml                 # 프로젝트 설정
├── setup.py                       # 패키지 설정
└── .env.example                   # 환경 변수 예시
```

---

## 🗓️ 3개월 로드맵

### Phase 1: 기반 구축 (Week 1-2)

**Week 1: 프로젝트 셋업**
- [ ] GitHub 리포지토리 생성 (템플릿, 이슈 라벨, PR 템플릿)
- [ ] 프로젝트 구조 생성
- [ ] Docker 환경 구축
- [ ] CI/CD 파이프라인 기본 설정
- [ ] 실험 추적 시스템 (Weights & Biases / MLflow)

**Week 2: 데이터 파이프라인 v1**
- [ ] GitHub 한국어 코드 수집 (1,000+ repos)
- [ ] 데이터 정제 파이프라인
- [ ] 기본 품질 필터링
- [ ] 초기 데이터셋 통계 분석
- [ ] 데이터 버전 관리 (DVC)

### Phase 2: 학습 & 실험 (Week 3-6)

**Week 3-4: 베이스라인 모델**
- [ ] EXAONE-2.4B + QLoRA 학습
- [ ] 기본 벤치마크 평가
- [ ] 학습 파이프라인 안정화
- [ ] 메모리 사용량 프로파일링

**Week 5-6: 모델 실험**
- [ ] 3개 이상 베이스 모델 비교 (2.4B, 3B)
- [ ] QLoRA vs LoRA 비교
- [ ] 하이퍼파라미터 튜닝 (learning rate, rank, alpha)
- [ ] Ablation study (데이터 크기, 학습 steps)
- [ ] 실험 결과 시각화

### Phase 3: 평가 시스템 (Week 7-8)

**Week 7: 벤치마크 구축**
- [ ] HumanEval-X 한국어 번역 (164 문제)
- [ ] MBPP 한국어 번역 (선별 500 문제)
- [ ] 커스텀 벤치마크 (한국 코딩 테스트 스타일)
- [ ] 자동 평가 파이프라인 (pass@k)

**Week 8: 심화 평가**
- [ ] Human evaluation 프로토콜
- [ ] 5-10명 개발자 평가 (코드 품질, 한국어 자연스러움)
- [ ] 에러 분석 (어떤 유형의 문제에서 실패?)
- [ ] 경쟁 모델과 비교 (GPT-3.5, Claude-3-Haiku)

### Phase 4: 배포 & 사용성 (Week 9-10)

**Week 9: API & CLI**
- [ ] FastAPI 서버 구축
- [ ] OpenAI-compatible API
- [ ] CLI 도구 개발 (`kocode generate "함수 만들어줘"`)
- [ ] Gradio 데모 앱
- [ ] API 문서 (Swagger/OpenAPI)

**Week 10: VS Code 확장**
- [ ] VS Code Extension 개발
- [ ] 코드 자동완성 기능
- [ ] 한국어 설명으로 코드 생성
- [ ] 코드 설명 생성 (주석 추가)
- [ ] Marketplace 출시 준비

### Phase 5: 최적화 & 문서화 (Week 11-12)

**Week 11: 최적화**
- [ ] 모델 양자화 (INT8, INT4)
- [ ] GGUF 변환 (Ollama 통합)
- [ ] vLLM 통합 (빠른 추론)
- [ ] 성능 벤치마크 (latency, throughput)
- [ ] 배포 최적화 (Docker 이미지 크기)

**Week 12: 문서화**
- [ ] 상세한 README (배지, 데모 GIF, 빠른 시작)
- [ ] 아키텍처 문서
- [ ] 기여 가이드
- [ ] 3개 이상 튜토리얼
- [ ] 기술 블로그 포스트 4편

### Phase 6: 커뮤니티 & 개선 (Week 13+)

**지속적 개선**
- [ ] 커뮤니티 피드백 수집
- [ ] 이슈 대응
- [ ] 성능 개선
- [ ] 데이터 확장
- [ ] 새로운 기능 추가

---

## 🔬 핵심 실험 목록

### 1. 모델 선택 실험

| 실험 | 베이스 모델 | 방법 | 목표 |
|------|-----------|------|-----|
| E1 | EXAONE-2.4B | QLoRA | 베이스라인 |
| E2 | Llama-3.2-3B | QLoRA | 크기 vs 성능 |
| E3 | Qwen2.5-Coder-3B | QLoRA | 코딩 특화 vs 범용 |
| E4 | EXAONE-2.4B | LoRA (r=32) | QLoRA vs LoRA |
| E5 | EXAONE-2.4B | Full FT (클라우드) | PEFT vs Full FT |

### 2. 데이터 실험

| 실험 | 데이터 | 크기 | 목표 |
|------|--------|-----|-----|
| D1 | GitHub only | 10K | 베이스라인 |
| D2 | + Stack Overflow | 20K | 다양성 효과 |
| D3 | + 합성 데이터 | 30K | 증강 효과 |
| D4 | 품질 필터링 (상위 50%) | 15K | 품질 vs 양 |

### 3. Ablation Study

| 변수 | 옵션 | 목표 |
|------|-----|-----|
| LoRA rank | 8, 16, 32, 64 | 최적 rank |
| Learning rate | 1e-4, 2e-4, 5e-4 | 최적 LR |
| Batch size | 2, 4, 8 | 안정성 vs 속도 |
| Training steps | 500, 1000, 2000 | 수렴 분석 |
| Context length | 1024, 2048, 4096 | 긴 코드 처리 |

---

## 📊 평가 지표

### 자동 평가

| 벤치마크 | 지표 | 목표 |
|---------|------|-----|
| **HumanEval-Ko** | pass@1, pass@10 | 베이스 대비 +15% |
| **MBPP-Ko** | pass@1, pass@10 | 베이스 대비 +10% |
| **Custom-Bench** | pass@1 | 80%+ |
| **Code Completion** | Exact Match, BLEU | - |

### 사람 평가

| 측면 | 척도 | 평가자 |
|------|------|--------|
| **정확성** | 1-5 | 10명 개발자 |
| **한국어 자연스러움** | 1-5 | 10명 개발자 |
| **코드 품질** | 1-5 | 10명 개발자 |
| **주석 품질** | 1-5 | 10명 개발자 |

### 성능 지표

| 지표 | 목표 |
|------|-----|
| **추론 속도** | <100ms (2048 tokens) |
| **메모리** | <8GB (양자화 후) |
| **처리량** | >10 req/sec |

---

## 💰 리소스 예산

### 로컬 리소스 (MacBook M3 Pro)

| 용도 | 사용 |
|------|-----|
| 개발 | 100% |
| 작은 모델 학습 (2.4B) | 가능 |
| 추론 테스트 | 100% |
| 데이터 처리 | 100% |

### 클라우드 리소스 (필요시)

| 용도 | 플랫폼 | 예상 비용 |
|------|--------|----------|
| 큰 모델 학습 (3B+) | Modal/RunPod | $50-100 |
| Full Fine-tuning | Modal/RunPod | $100-200 |
| 대규모 벤치마크 | Modal | $50 |
| **총 예산** | - | **$200-350** |

### 무료 대안

- Google Colab Pro ($10/month)
- Kaggle Notebooks (무료 GPU)
- HuggingFace Spaces (무료 호스팅)

---

## 🎯 성공 지표 (KPI)

### 기술적 성공

- [ ] 5개 이상 실험 완료
- [ ] 3개 이상 벤치마크에서 베이스 모델 대비 +10% 이상
- [ ] 재현 가능한 환경 (Docker + 상세 문서)
- [ ] 3개 이상 배포 형태 (API, CLI, 확장)

### 커뮤니티 성공

- [ ] GitHub Stars: 100+ (3개월 내)
- [ ] Forks: 20+
- [ ] Contributors: 3+ (본인 제외)
- [ ] 기술 블로그 조회수: 1,000+

### 취업 성공

- [ ] 포트폴리오로 AI/ML 엔지니어 인터뷰 기회
- [ ] 프로젝트 기반 기술 토론 가능
- [ ] 실무 수준의 코드베이스
- [ ] MLOps 경험 증명

---

## 🛠️ 기술 스택

### 핵심 ML

- **프레임워크**: PyTorch 2.0+, Transformers 4.35+
- **PEFT**: HuggingFace PEFT (LoRA, QLoRA)
- **학습**: DeepSpeed, Accelerate
- **추론**: vLLM, llama.cpp
- **실험 추적**: Weights & Biases / MLflow

### 데이터

- **수집**: BeautifulSoup, GitHub API, requests
- **처리**: pandas, datasets, tokenizers
- **품질**: langdetect, sentence-transformers
- **버전**: DVC, git-lfs

### 배포

- **API**: FastAPI, uvicorn
- **컨테이너**: Docker, docker-compose
- **오케스트레이션**: Kubernetes (선택)
- **모니터링**: Prometheus, Grafana

### 개발

- **언어**: Python 3.11
- **패키지 관리**: Poetry / pip
- **코드 품질**: black, ruff, mypy
- **테스트**: pytest, pytest-cov
- **CI/CD**: GitHub Actions

### 플랫폼

- **학습**: MacBook M3 Pro + Modal/RunPod
- **서빙**: HuggingFace Spaces / Fly.io
- **저장소**: GitHub, HuggingFace Hub

---

## 📚 학습 계획

이 프로젝트를 통해 배울 내용:

### 1. LLM 파인튜닝
- [x] LoRA, QLoRA 원리
- [ ] 다양한 베이스 모델 특성
- [ ] 하이퍼파라미터 튜닝
- [ ] 학습 안정화 기법

### 2. 모델 평가
- [ ] 코드 생성 벤치마크
- [ ] Human evaluation 설계
- [ ] 통계적 유의성 검증
- [ ] 에러 분석 방법론

### 3. 데이터 엔지니어링
- [ ] 대규모 데이터 수집
- [ ] 데이터 품질 관리
- [ ] 데이터 증강 기법
- [ ] 데이터 버전 관리

### 4. MLOps
- [ ] 실험 관리
- [ ] 모델 버저닝
- [ ] CI/CD for ML
- [ ] 모델 서빙 최적화

### 5. 오픈소스 운영
- [ ] 커뮤니티 관리
- [ ] 문서화 베스트 프랙티스
- [ ] 이슈 관리
- [ ] 코드 리뷰

---

## 🎓 참고 자료

### 유사 프로젝트 (벤치마크 대상)

- [Code Llama](https://github.com/facebookresearch/codellama)
- [StarCoder](https://github.com/bigcode-project/starcoder)
- [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder)
- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)

### 논문

- LoRA: Low-Rank Adaptation of Large Language Models
- QLoRA: Efficient Finetuning of Quantized LLMs
- WizardCoder: Empowering Code Large Language Models
- CodeGen: An Open Large Language Model for Code

### 벤치마크

- [HumanEval](https://github.com/openai/human-eval)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- [CodeXGLUE](https://github.com/microsoft/CodeXGLUE)

---

## 🚀 Next Steps

### 즉시 시작

1. **프로젝트명 확정**
   - `ko-code-llm`? `korean-coder`? `hangul-code`?
   - 도메인 예약 가능 여부 확인

2. **GitHub 리포지토리 생성**
   - Public으로 생성
   - 라이선스: MIT (오픈소스 친화적)
   - README 초안 작성

3. **개발 환경 셋업**
   - Docker 환경 구축
   - 가상환경 생성
   - 기본 의존성 설치

4. **첫 번째 이슈 생성**
   - Milestone 1: 기반 구축
   - 세부 태스크 정의

---

## 💬 질문 사항

이 마스터플랜에 대해 조정하고 싶은 부분:

1. **프로젝트명**: 더 나은 이름 아이디어?
2. **리소스**: 클라우드 예산 $200-350 괜찮은가?
3. **우선순위**: 특히 집중하고 싶은 부분?
4. **추가 기능**: 포함하고 싶은 다른 기능?

---

**준비됐습니까? 🚀 Let's build something amazing!**
