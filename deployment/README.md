# Deployment Directory

모델을 실제로 사용할 수 있는 다양한 형태로 배포합니다.

## 구조

```
deployment/
├── api/               # REST API (FastAPI)
├── cli/               # CLI 도구
├── vscode-extension/  # VS Code 확장
└── demo/              # Gradio 데모
```

## 배포 형태

### 1. REST API

FastAPI 기반 고성능 API 서버

```bash
# 로컬 실행
uvicorn deployment.api.main:app --reload

# Docker로 실행
docker-compose -f infrastructure/docker/docker-compose.yml up api

# 요청 예시
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "리스트 정렬 함수를 만들어줘", "max_length": 512}'
```

**API 문서**: http://localhost:8000/docs

### 2. CLI 도구

터미널에서 바로 사용 가능한 CLI

```bash
# 설치
pip install -e .

# 사용
kocode generate "피보나치 함수를 만들어줘"
kocode explain "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)"
kocode complete --file main.py --line 10
```

### 3. VS Code 확장

VS Code에서 직접 사용

```bash
# 개발 모드 실행
cd deployment/vscode-extension
npm install
npm run compile
# F5를 눌러 Extension Development Host 실행

# 배포
vsce package
vsce publish
```

**기능:**
- 코드 자동완성
- 한국어 설명으로 코드 생성
- 선택한 코드 설명하기
- 코드에 한국어 주석 추가

### 4. Gradio 데모

웹 UI로 쉽게 체험

```bash
# 실행
python deployment/demo/app.py

# 접속
# http://localhost:7860
```

## Phase 4 작업 목록 (Week 9-10)

- [ ] FastAPI 서버 구축
- [ ] OpenAI-compatible API 구현
- [ ] CLI 도구 개발
- [ ] VS Code 확장 개발
- [ ] Gradio 데모 앱
- [ ] API 문서 작성

## 성능 최적화

### vLLM 사용

```python
from vllm import LLM, SamplingParams

llm = LLM(model="path/to/model")
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

prompts = ["리스트 정렬 함수"]
outputs = llm.generate(prompts, sampling_params)
```

### 모델 양자화

```bash
# INT8 양자화
python -m llama_cpp.convert --model results/models/best_model

# GGUF 변환 (Ollama용)
ollama create korean-coder -f deployment/Modelfile
```

## 배포 환경

| 환경 | 플랫폼 | 용도 |
|------|--------|------|
| **로컬** | MacBook M3 Pro | 개발/테스트 |
| **API** | Fly.io / HuggingFace Spaces | 프로덕션 |
| **Demo** | HuggingFace Spaces | 데모 |
| **Extension** | VS Code Marketplace | 배포 |

## 참고

- [FastAPI 문서](https://fastapi.tiangolo.com)
- [vLLM 문서](https://vllm.readthedocs.io)
- [VS Code Extension API](https://code.visualstudio.com/api)
- [Gradio 문서](https://gradio.app/docs)
