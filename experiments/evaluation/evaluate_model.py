"""
Simple code generation evaluation script.
Tests the fine-tuned model on sample Python tasks.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_model_name, adapter_path):
    """Load base model and LoRA adapter."""
    print(f"Loading base model: {base_model_name}")

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer, device

def generate_code(model, tokenizer, prompt, device, max_new_tokens=256):
    """Generate code completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

def main():
    # Model paths
    base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    adapter_path = "../../results/models/qlora-exaone-2.4b/final_model"

    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, adapter_path)

    print("="*60)
    print("Code Generation Evaluation")
    print("="*60)

    # Load model
    model, tokenizer, device = load_model(base_model_name, adapter_path)

    # Test prompts
    test_prompts = [
        {
            "name": "Fibonacci",
            "prompt": """Write a Python function to compute the nth Fibonacci number.

def fibonacci(n: int) -> int:
    \"\"\"Returns the nth Fibonacci number.\"\"\"
"""
        },
        {
            "name": "Palindrome",
            "prompt": """Write a Python function to check if a string is a palindrome.

def is_palindrome(s: str) -> bool:
    \"\"\"Returns True if s is a palindrome.\"\"\"
"""
        },
        {
            "name": "List Sum",
            "prompt": """Write a Python function to sum all numbers in a list.

def sum_list(numbers: list) -> int:
    \"\"\"Returns the sum of all numbers in the list.\"\"\"
"""
        }
    ]

    print("\nGenerating code completions...\n")

    results = []
    for i, test in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}: {test['name']}")
        print("-" * 60)
        print("Prompt:")
        print(test['prompt'])
        print("\nGenerated:")

        generated = generate_code(model, tokenizer, test['prompt'], device)
        completion = generated[len(test['prompt']):].strip()
        print(completion)
        print("="*60)
        print()

        results.append({
            "name": test['name'],
            "prompt": test['prompt'],
            "completion": completion
        })

    # Save results
    results_path = os.path.join(script_dir, "../../results/evaluation_samples.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Code Generation Evaluation Results\n")
        f.write("="*60 + "\n\n")

        for result in results:
            f.write(f"Task: {result['name']}\n")
            f.write("-"*60 + "\n")
            f.write("Prompt:\n")
            f.write(result['prompt'] + "\n\n")
            f.write("Generated:\n")
            f.write(result['completion'] + "\n")
            f.write("="*60 + "\n\n")

    print(f"\nResults saved to: {results_path}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
