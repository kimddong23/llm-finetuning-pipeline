# 🇰🇷 Korean Code LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-grade Korean Code Generation LLM**

한국어로 설명하면 한국 개발자 스타일의 코드를 생성하는 LLM. 한국어 주석, 한국어 변수명 지원.

> 🚧 **현재 상태**: 활발히 개발 중 (Phase 1: 기반 구축)
>
> 📅 **타임라인**: 2024년 1월 ~ 2024년 4월 (3개월 프로젝트)

---

## ✨ 특징

### 🎯 차별점

- **🔄 완전한 재현성**: Docker + seed 고정 + 상세 로그로 누구나 동일한 결과 재현
- **📊 철저한 벤치마크**: HumanEval-Ko, MBPP-Ko, 커스텀 벤치마크 포함
- **🚀 실사용 가능**: REST API, CLI, VS Code 확장 모두 제공
- **📚 교육적 가치**: 상세한 문서, 튜토리얼, 기술 블로그

### 🛠️ 기술 스택

- **모델**: EXAONE-2.4B, Llama-3.2-3B, Qwen2.5-Coder-3B
- **파인튜닝**: QLoRA, LoRA, Full Fine-tuning 비교
- **배포**: FastAPI, Docker, vLLM
- **평가**: 자동 평가 + Human evaluation

---

## 🚀 빠른 시작

### 설치

```bash
# 클론
git clone https://github.com/yourusername/korean-code-llm.git
cd korean-code-llm

# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements/base.txt
```

### 사용 예시

```python
from korean_code_llm import KoreanCodeLLM

# 모델 로드
model = KoreanCodeLLM.from_pretrained("korean-code-llm-2.4b")

# 코드 생성
prompt = "리스트에서 짝수만 필터링하는 함수를 만들어줘"
code = model.generate(prompt)
print(code)
```

**출력:**
```python
def filter_even_numbers(numbers):
    """리스트에서 짝수만 필터링하는 함수

    Args:
        numbers (list): 정수 리스트

    Returns:
        list: 짝수만 포함된 리스트
    """
    return [num for num in numbers if num % 2 == 0]
```

---

## 📊 성능

| 벤치마크 | Base Model | **Korean Code LLM** | 개선 |
|---------|-----------|---------------------|-----|
| HumanEval-Ko | 45.1% | **TBD** | TBD |
| MBPP-Ko | 38.2% | **TBD** | TBD |
| Custom-Bench | - | **TBD** | - |

> 🚧 벤치마크 결과는 Phase 3 (Week 7-8)에 업데이트됩니다.

---

## 📁 프로젝트 구조

```
korean-code-llm/
├── 📊 data/                  # 데이터 파이프라인
├── 🧪 experiments/           # 실험 및 학습
├── 🚀 deployment/            # 배포 (API, CLI, 확장)
├── 🐳 infrastructure/        # Docker, CI/CD
├── 📚 docs/                  # 문서 및 튜토리얼
├── 🧪 tests/                 # 테스트
└── 📊 results/               # 실험 결과
```

전체 구조는 [ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참고하세요.

---

## 🗓️ 로드맵

- [x] **Phase 1** (Week 1-2): 프로젝트 셋업, 데이터 파이프라인
- [ ] **Phase 2** (Week 3-6): 모델 학습 및 실험
- [ ] **Phase 3** (Week 7-8): 평가 시스템 구축
- [ ] **Phase 4** (Week 9-10): 배포 및 사용성
- [ ] **Phase 5** (Week 11-12): 최적화 및 문서화
- [ ] **Phase 6** (Week 13+): 커뮤니티 및 지속적 개선

자세한 로드맵은 [00_PROJECT_MASTERPLAN.md](docs/00_PROJECT_MASTERPLAN.md)를 참고하세요.

---

## 🤝 기여하기

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)를 읽어주세요.

### 기여 방법

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 문서

- [📖 아키텍처](docs/ARCHITECTURE.md)
- [🎯 마스터플랜](docs/00_PROJECT_MASTERPLAN.md)
- [🚀 빠른 시작](docs/tutorials/01_quick_start.md)
- [🔧 파인튜닝 가이드](docs/tutorials/02_fine_tuning.md)
- [📊 벤치마크](docs/BENCHMARKS.md)

### 블로그 포스트

- [ ] 프로젝트 소개
- [ ] 데이터 파이프라인 구축기
- [ ] 학습 과정 및 최적화
- [ ] 평가 시스템 설계

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

---

## 🙏 감사의 말

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [vLLM](https://github.com/vllm-project/vllm)
- [Code Llama](https://github.com/facebookresearch/codellama)
- [StarCoder](https://github.com/bigcode-project/starcoder)

---

## 📧 연락처

- **작성자**: [Your Name]
- **이메일**: your.email@example.com
- **LinkedIn**: [Your LinkedIn]
- **블로그**: [Your Blog]

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**

---

<div align="center">
  <sub>Built with ❤️ for the Korean developer community</sub>
</div>
