# Code Generation CLI

Command-line interface for generating code with the fine-tuned EXAONE model.

## Installation

```bash
# Install dependencies (if not already installed)
pip install torch transformers peft accelerate
```

## Quick Start

### Interactive Mode (Default)

```bash
python codegen.py
```

This starts an interactive session where you can enter prompts:

```
>>> def fibonacci(n):
Generating...

------------------------------------------------------------
Generated code:
------------------------------------------------------------
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
------------------------------------------------------------
```

Type `exit` or press `Ctrl+D` to quit.

### Batch Mode

Process multiple prompts from a file:

```bash
# Create input file
cat > prompts.txt <<EOF
def is_palindrome(s):
def bubble_sort(arr):
def binary_search(arr, target):
EOF

# Run batch processing
python codegen.py --batch prompts.txt --output results.json
```

Results are saved as JSON:

```json
[
  {
    "prompt": "def is_palindrome(s):",
    "completion": "return s == s[::-1]",
    "full_code": "def is_palindrome(s):\n    return s == s[::-1]"
  },
  ...
]
```

## Usage

```
usage: codegen.py [-h] [--model {base,finetuned,int8,int4}]
                  [--device {mps,cuda,cpu}] [--batch INPUT_FILE]
                  [--output OUTPUT_FILE] [--max-tokens MAX_TOKENS]
                  [--temperature TEMPERATURE] [--top-p TOP_P]

Code generation CLI

optional arguments:
  -h, --help            show this help message and exit
  --model {base,finetuned,int8,int4}
                        Model to use (default: finetuned)
  --device {mps,cuda,cpu}
                        Device to use (default: auto-detect)
  --batch INPUT_FILE    Batch mode: read prompts from file
  --output OUTPUT_FILE  Output file for batch mode (default: results.json)
  --max-tokens MAX_TOKENS
                        Maximum tokens to generate (default: 256)
  --temperature TEMPERATURE
                        Sampling temperature (default: 0.2)
  --top-p TOP_P         Top-p sampling (default: 0.95)
```

## Examples

### 1. Use Base Model

```bash
python codegen.py --model base
```

### 2. Use Quantized Model (INT8)

```bash
python codegen.py --model int8
```

Quantized models use less memory but are slower.

### 3. Generate More Tokens

```bash
python codegen.py --max-tokens 512
```

### 4. More Creative Output

```bash
python codegen.py --temperature 0.8
```

Higher temperature = more randomness/creativity.

### 5. Batch Processing with Custom Parameters

```bash
python codegen.py \
  --batch prompts.txt \
  --output my_results.json \
  --max-tokens 200 \
  --temperature 0.3
```

## Model Options

| Model | Description | Memory | Speed |
|-------|-------------|--------|-------|
| `finetuned` | Fine-tuned FP16 model | ~5 GB | Fast (12.5 tok/s) |
| `base` | Base EXAONE model | ~5 GB | Fast (12.5 tok/s) |
| `int8` | 8-bit quantized | ~3 GB | Slower (2.1 tok/s) |
| `int4` | 4-bit quantized | ~2 GB | Slowest (0.8 tok/s) |

## Parameters

### max-tokens
- **Range**: 1-2048
- **Default**: 256
- **Description**: Maximum number of tokens to generate
- **Tip**: Increase for longer functions

### temperature
- **Range**: 0.0-2.0
- **Default**: 0.2
- **Description**: Controls randomness
  - 0.0 = deterministic
  - 0.2 = mostly consistent
  - 0.7 = balanced
  - 1.0+ = creative/random

### top-p
- **Range**: 0.0-1.0
- **Default**: 0.95
- **Description**: Nucleus sampling threshold
- **Tip**: Lower values = more focused output

## Tips

### Interactive Mode

1. **Function signatures**: Start with `def function_name(args):`
2. **Classes**: Start with `class ClassName:`
3. **Docstrings**: Include docstrings for better context
4. **Multi-line prompts**: Paste directly (newlines preserved)

### Batch Mode

1. **One prompt per line** in input file
2. **Remove empty lines** (they'll be skipped)
3. **Use clear function signatures**
4. **Review results.json** after completion

### Performance

- **First generation is slow** (model loading)
- **Subsequent generations are fast**
- **INT8/INT4 models** are memory-efficient but slower
- **Use batch mode** for multiple prompts (more efficient)

## Troubleshooting

### Out of Memory

```bash
# Use quantized model
python codegen.py --model int8
```

### Slow Performance

```bash
# Reduce max tokens
python codegen.py --max-tokens 128
```

### Model Not Found

Make sure you're running from the project root and the fine-tuned model exists at:
```
results/models/qlora-exaone-2.4b/best_model/
```

## Advanced Usage

### Programmatic Use

```python
from codegen import CodeGenerator

# Initialize
generator = CodeGenerator(model_type="finetuned")
generator.load_model()

# Generate
completion = generator.generate(
    prompt="def quicksort(arr):",
    max_tokens=256,
    temperature=0.2
)

print(completion)
```

### Custom Configuration

Create a config file `config.json`:

```json
{
  "model_type": "finetuned",
  "max_tokens": 300,
  "temperature": 0.3,
  "top_p": 0.95
}
```

Then modify the CLI to load config (exercise for the reader).

## License

MIT License - See LICENSE file for details.
