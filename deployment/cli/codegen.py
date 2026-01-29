#!/usr/bin/env python3
"""
Command-line interface for code generation.
Supports interactive and batch modes with various model options.
"""

import argparse
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class CodeGenerator:
    """Code generator with model loading and inference."""

    def __init__(self, model_type="finetuned", device=None):
        """
        Initialize code generator.

        Args:
            model_type: "base", "finetuned", "int8", or "int4"
            device: "mps", "cuda", or "cpu" (auto-detected if None)
        """
        self.model_type = model_type
        self.device = device or self._detect_device()
        self.model = None
        self.tokenizer = None

    def _detect_device(self):
        """Auto-detect available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_type} model on {self.device}...")

        base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load model based on type
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.model_type == "int8":
            load_kwargs["load_in_8bit"] = True
        elif self.model_type == "int4":
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **load_kwargs
        )

        # Load adapter for fine-tuned models
        if self.model_type in ["finetuned", "int8", "int4"]:
            adapter_path = "results/models/qlora-exaone-2.4b/best_model"
            print(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        print("✅ Model loaded successfully!\n")

    def generate(self, prompt, max_tokens=256, temperature=0.2, top_p=0.95):
        """
        Generate code completion.

        Args:
            prompt: Code prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Generated code completion
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        return completion

def interactive_mode(generator, config):
    """Interactive mode for code generation."""
    print("="*60)
    print("INTERACTIVE CODE GENERATION")
    print("="*60)
    print("Enter code prompts (Ctrl+D or 'exit' to quit)")
    print(f"Model: {config['model_type']}")
    print(f"Max tokens: {config['max_tokens']}")
    print(f"Temperature: {config['temperature']}")
    print("="*60)
    print()

    while True:
        try:
            prompt = input(">>> ")
            if not prompt or prompt.lower() == 'exit':
                break

            print("\nGenerating...")
            completion = generator.generate(
                prompt=prompt,
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p']
            )

            print("\n" + "-"*60)
            print("Generated code:")
            print("-"*60)
            print(prompt + completion)
            print("-"*60)
            print()

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

def batch_mode(generator, input_file, output_file, config):
    """Batch mode for processing multiple prompts."""
    print(f"Processing prompts from {input_file}...")

    # Read prompts
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Found {len(prompts)} prompts")

    # Generate completions
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating for: {prompt[:50]}...")

        try:
            completion = generator.generate(
                prompt=prompt,
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
                top_p=config['top_p']
            )

            results.append({
                "prompt": prompt,
                "completion": completion,
                "full_code": prompt + completion
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "prompt": prompt,
                "error": str(e)
            })

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print(f"   Successful: {sum(1 for r in results if 'completion' in r)}/{len(prompts)}")

def main():
    parser = argparse.ArgumentParser(
        description="Code generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  %(prog)s

  # Use base model
  %(prog)s --model base

  # Batch mode
  %(prog)s --batch prompts.txt --output results.json

  # Custom parameters
  %(prog)s --max-tokens 512 --temperature 0.5
        """
    )

    # Model options
    parser.add_argument(
        "--model",
        choices=["base", "finetuned", "int8", "int4"],
        default="finetuned",
        help="Model to use (default: finetuned)"
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    # Mode options
    parser.add_argument(
        "--batch",
        metavar="INPUT_FILE",
        help="Batch mode: read prompts from file"
    )
    parser.add_argument(
        "--output",
        metavar="OUTPUT_FILE",
        default="results.json",
        help="Output file for batch mode (default: results.json)"
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )

    args = parser.parse_args()

    # Configuration
    config = {
        "model_type": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    # Initialize generator
    generator = CodeGenerator(model_type=args.model, device=args.device)
    generator.load_model()

    # Run appropriate mode
    if args.batch:
        if not Path(args.batch).exists():
            print(f"❌ Error: Input file '{args.batch}' not found")
            sys.exit(1)
        batch_mode(generator, args.batch, args.output, config)
    else:
        interactive_mode(generator, config)

if __name__ == "__main__":
    main()
