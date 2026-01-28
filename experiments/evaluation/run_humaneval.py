"""
HumanEval benchmark evaluation script.
Tests both base model and fine-tuned model on 164 coding problems.
"""

import os
import sys
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from human_eval.data import write_jsonl, read_problems

def load_model(base_model_name, adapter_path=None):
    """Load model (base or fine-tuned)."""
    print(f"\nLoading model: {base_model_name}")
    if adapter_path:
        print(f"With adapter: {adapter_path}")

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

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Load adapter if specified
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer, device

def generate_one_completion(model, tokenizer, prompt, device):
    """Generate a single code completion."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the new tokens after the prompt
    completion = completion[len(prompt):].strip()

    # Stop at the first occurrence of test code or new function definition
    stop_tokens = ["\ndef ", "\nclass ", "\nif __name__", "\n# Test"]
    for stop in stop_tokens:
        if stop in completion:
            completion = completion[:completion.index(stop)]

    return completion

def evaluate_humaneval(model, tokenizer, device, output_file, num_samples=1):
    """Run HumanEval evaluation."""
    problems = read_problems()

    print(f"\nEvaluating on {len(problems)} HumanEval problems...")
    print(f"Generating {num_samples} sample(s) per problem...")

    samples = []

    for task_id, problem in tqdm(problems.items(), desc="Generating completions"):
        prompt = problem["prompt"]

        for _ in range(num_samples):
            completion = generate_one_completion(model, tokenizer, prompt, device)

            samples.append({
                "task_id": task_id,
                "completion": completion
            })

    # Save samples
    write_jsonl(output_file, samples)
    print(f"\nSamples saved to: {output_file}")
    print(f"Total samples: {len(samples)}")

    return samples

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "finetuned", "both"], default="both",
                        help="Which model to evaluate")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per problem")
    args = parser.parse_args()

    base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, "../../results/models/qlora-exaone-2.4b/final_model")
    results_dir = os.path.join(script_dir, "../../results/humaneval")
    os.makedirs(results_dir, exist_ok=True)

    print("="*70)
    print("HumanEval Benchmark Evaluation")
    print("="*70)

    # Evaluate base model
    if args.model in ["base", "both"]:
        print("\n" + "="*70)
        print("EVALUATING BASE MODEL")
        print("="*70)

        model, tokenizer, device = load_model(base_model_name)
        base_output = os.path.join(results_dir, "base_model_samples.jsonl")
        evaluate_humaneval(model, tokenizer, device, base_output, args.num_samples)

        # Clean up
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Evaluate fine-tuned model
    if args.model in ["finetuned", "both"]:
        print("\n" + "="*70)
        print("EVALUATING FINE-TUNED MODEL")
        print("="*70)

        model, tokenizer, device = load_model(base_model_name, adapter_path)
        finetuned_output = os.path.join(results_dir, "finetuned_model_samples.jsonl")
        evaluate_humaneval(model, tokenizer, device, finetuned_output, args.num_samples)

        # Clean up
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: evaluate_functional_correctness <output_file>")
    print("   Example: evaluate_functional_correctness results/humaneval/base_model_samples.jsonl")
    print("2. Check results in: results/humaneval/")
    print("="*70)

if __name__ == "__main__":
    main()
