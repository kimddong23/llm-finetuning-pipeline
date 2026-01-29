---
title: Code Generation
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Code Generation with Fine-tuned EXAONE

Generate Python code completions using a fine-tuned EXAONE-3.5-2.4B model.

## Features

- 🎯 High-quality code generation (96.95% HumanEval pass@1)
- ⚡ Fast inference on GPU
- 🎨 Beautiful Gradio interface
- 📝 Multiple example prompts
- 🔧 Adjustable generation parameters

## How to Use

1. Enter a code prompt (function signature or partial code)
2. Adjust parameters:
   - **Max Tokens**: Length of generation (32-512)
   - **Temperature**: Randomness (0.0 = deterministic, 1.0 = creative)
   - **Top-p**: Nucleus sampling threshold (0.5-1.0)
3. Click "Generate Code"

## Model Details

**Base Model**: [LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)

**Fine-tuning**:
- Method: QLoRA (4-bit quantization + LoRA)
- Dataset: 10K Python functions from The Stack
- Training time: 12.5 hours (Mac M3 Pro)
- Trainable parameters: 4M / 2.4B (0.17%)

**Performance**:
- HumanEval pass@1: 96.95% (fine-tuned) vs 98.17% (base)
- Code samples: 3/3 correct (Fibonacci, Palindrome, List Sum)
- Average completion: 182 characters (concise)

## Examples

```python
# Fibonacci
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Palindrome
def is_palindrome(s):
    return s == s[::-1]

# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

## Project Links

- **GitHub**: [llm-finetuning-pipeline](https://github.com/kimddong23/llm-finetuning-pipeline)
- **Technical Report**: [TECHNICAL_REPORT.md](https://github.com/kimddong23/llm-finetuning-pipeline/blob/main/docs/TECHNICAL_REPORT.md)
- **Error Analysis**: [error_analysis.md](https://github.com/kimddong23/llm-finetuning-pipeline/blob/main/results/humaneval/error_analysis.md)

## Technical Stack

- **Framework**: HuggingFace Transformers, PEFT
- **Training**: QLoRA (bitsandbytes)
- **Interface**: Gradio
- **Deployment**: HuggingFace Spaces (free GPU)

## License

MIT License - See [LICENSE](https://github.com/kimddong23/llm-finetuning-pipeline/blob/main/LICENSE) for details.

## Acknowledgments

Built using:
- [EXAONE-3.5-2.4B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)
- [The Stack Dataset](https://huggingface.co/datasets/bigcode/the-stack-dedup)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [Gradio](https://gradio.app)
