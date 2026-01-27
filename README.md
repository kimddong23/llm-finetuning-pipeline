# 🚀 LLM Finetuning Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-grade LLM Fine-tuning Pipeline for Code Generation**

A complete, reproducible pipeline for fine-tuning Large Language Models on code generation tasks. Built for AI/ML Engineer portfolios with production-level quality.

> 🚧 **Status**: Actively developing (Phase 1: Foundation)
>
> 📅 **Timeline**: Jan 2024 ~ Apr 2024 (3-month project)
>
> 🌟 **Open Source**: MIT Licensed - Contributions Welcome!

---

## 🌟 Why Open Source?

This project is **100% open source** to:
- **Share Knowledge**: Help others learn LLM fine-tuning from a complete, production-grade example
- **Enable Reuse**: Anyone can fork, adapt, and use this pipeline for their own projects
- **Build Community**: Collaborate with developers worldwide to improve LLM fine-tuning practices
- **Demonstrate Skills**: Showcase real-world ML engineering capabilities

**Feel free to:**
- ⭐ Star this repo
- 🍴 Fork and customize
- 🐛 Report issues
- 💡 Suggest improvements
- 🤝 Submit pull requests

---

## ✨ Features

### 🎯 Key Differentiators

- **🔄 Full Reproducibility**: Docker + fixed seeds + detailed logs for identical results
- **📊 Comprehensive Benchmarks**: HumanEval, MBPP, custom benchmarks
- **🚀 Production-Ready**: REST API, CLI, VS Code extension
- **📚 Educational Value**: Detailed docs, tutorials, technical blog posts

### 🛠️ 기술 스택

- **모델**: EXAONE-2.4B, Llama-3.2-3B, Qwen2.5-Coder-3B
- **파인튜닝**: QLoRA, LoRA, Full Fine-tuning 비교
- **배포**: FastAPI, Docker, vLLM
- **평가**: 자동 평가 + Human evaluation

---

## 🚀 빠른 시작

### 설치

```bash
# Clone
git clone https://github.com/yourusername/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

### Usage Example

```python
from llm_finetuning import CodeLLM

# Load model
model = CodeLLM.from_pretrained("llm-finetuning-2.4b")

# Generate code
prompt = "Write a function to filter even numbers from a list"
code = model.generate(prompt)
print(code)
```

**Output:**
```python
def filter_even_numbers(numbers):
    """Filter even numbers from a list

    Args:
        numbers (list): List of integers

    Returns:
        list: List containing only even numbers
    """
    return [num for num in numbers if num % 2 == 0]
```

---

## 📊 Performance

| Benchmark | Base Model | **Fine-tuned** | Improvement |
|---------|-----------|---------------------|-----|
| HumanEval | 45.1% | **TBD** | TBD |
| MBPP | 38.2% | **TBD** | TBD |
| Custom-Bench | - | **TBD** | - |

> 🚧 Benchmark results will be updated in Phase 3 (Week 7-8).

---

## 📁 Project Structure

```
llm-finetuning-pipeline/
├── 📊 data/                  # Data pipeline
├── 🧪 experiments/           # Training & evaluation
├── 🚀 deployment/            # API, CLI, extensions
├── 🐳 infrastructure/        # Docker, CI/CD
├── 📚 docs/                  # Documentation
├── 🧪 tests/                 # Tests
└── 📊 results/               # Experiment results
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
