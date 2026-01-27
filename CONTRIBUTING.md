# 기여 가이드

Korean Code LLM 프로젝트에 기여해 주셔서 감사합니다! 🎉

## 📋 기여 방법

### 1. 이슈 생성

버그를 발견하거나 새로운 기능을 제안하고 싶다면:

1. [Issues](https://github.com/yourusername/korean-code-llm/issues)에서 중복된 이슈가 없는지 확인
2. 새 이슈 생성
3. 적절한 라벨 추가 (bug, enhancement, documentation 등)

### 2. Pull Request 생성

#### 준비 단계

```bash
# 1. Fork the repository
# GitHub에서 Fork 버튼 클릭

# 2. Clone your fork
git clone https://github.com/yourusername/korean-code-llm.git
cd korean-code-llm

# 3. upstream 추가
git remote add upstream https://github.com/originaluser/korean-code-llm.git

# 4. 개발 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements/dev.txt
```

#### 개발 단계

```bash
# 1. 최신 main 브랜치로 업데이트
git checkout main
git pull upstream main

# 2. 새 브랜치 생성
git checkout -b feature/your-feature-name
# 또는
git checkout -b fix/bug-description

# 3. 코드 작성
# ... 작업 ...

# 4. 코드 품질 체크
black korean_code_llm tests
ruff check korean_code_llm tests
mypy korean_code_llm

# 5. 테스트 실행
pytest tests/

# 6. 커밋
git add .
git commit -m "feat: Add amazing feature"
# 커밋 메시지는 Conventional Commits 규칙 따르기
```

#### PR 제출

```bash
# 1. Push to your fork
git push origin feature/your-feature-name

# 2. GitHub에서 Pull Request 생성
# - Base: main
# - Compare: your-feature-branch
# - 설명 작성 (무엇을, 왜 변경했는지)
```

## 📝 커밋 메시지 규칙

[Conventional Commits](https://www.conventionalcommits.org/) 사용:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서만 변경
- `style`: 코드 의미에 영향을 주지 않는 변경 (공백, 포맷팅 등)
- `refactor`: 버그를 수정하거나 기능을 추가하지 않는 코드 변경
- `perf`: 성능 개선
- `test`: 테스트 추가 또는 수정
- `chore`: 빌드 프로세스 또는 보조 도구 변경

**예시:**
```bash
feat(training): Add support for Llama-3.2-3B model

- Add model config for Llama-3.2-3B
- Update training script to handle new model
- Add tests for Llama integration

Closes #42
```

## 🧪 테스트

모든 PR은 테스트를 포함해야 합니다:

```bash
# 전체 테스트 실행
pytest tests/

# 특정 테스트 실행
pytest tests/test_model.py

# 커버리지 확인
pytest --cov=korean_code_llm tests/
```

## 📚 문서

코드를 변경하면 관련 문서도 업데이트해 주세요:

- Docstrings (Google 스타일)
- README.md
- docs/ 디렉토리의 관련 문서

**Docstring 예시:**
```python
def generate_code(prompt: str, max_length: int = 512) -> str:
    """한국어 프롬프트로부터 코드를 생성합니다.

    Args:
        prompt: 코드 생성 프롬프트 (한국어)
        max_length: 생성할 최대 토큰 수

    Returns:
        생성된 코드 문자열

    Raises:
        ValueError: 프롬프트가 비어있을 경우

    Examples:
        >>> generate_code("리스트 정렬 함수")
        'def sort_list(lst):\\n    return sorted(lst)'
    """
    ...
```

## 🎨 코드 스타일

- **포맷팅**: Black (line-length=100)
- **Linting**: Ruff
- **타입 체킹**: MyPy (optional but recommended)

```bash
# 자동 포맷팅
black korean_code_llm tests

# Lint 체크
ruff check korean_code_llm tests

# 타입 체크
mypy korean_code_llm
```

## 🏷️ 라벨

| 라벨 | 설명 |
|------|------|
| `bug` | 버그 리포트 |
| `enhancement` | 새로운 기능 제안 |
| `documentation` | 문서 개선 |
| `good first issue` | 첫 기여자를 위한 이슈 |
| `help wanted` | 도움이 필요한 이슈 |
| `priority: high` | 우선순위 높음 |
| `wontfix` | 수정하지 않을 이슈 |

## 🤝 행동 강령

우리는 모든 기여자를 환영하며 존중합니다. 다음 원칙을 따라주세요:

- **친절하고 존중하기**: 서로 존중하고 배려합니다
- **건설적인 피드백**: 비판적이되 건설적으로
- **협력적인 태도**: 함께 더 나은 프로젝트를 만듭니다

## 💬 소통

- **GitHub Issues**: 버그 리포트, 기능 제안
- **GitHub Discussions**: 일반적인 질문, 아이디어 토론
- **Discord** (TBD): 실시간 소통

## 🎯 기여 아이디어

처음 기여하시나요? 다음을 시도해보세요:

- [ ] 오타 수정
- [ ] 문서 개선
- [ ] `good first issue` 라벨이 붙은 이슈
- [ ] 테스트 추가
- [ ] 예제 코드 작성

## 📧 질문

질문이 있으신가요?

- GitHub Issues에서 질문하기
- Discussions에서 토론하기
- 이메일: your.email@example.com

---

**다시 한번 기여해 주셔서 감사합니다! 🙏**
