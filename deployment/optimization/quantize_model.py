"""
Model quantization script for deployment optimization.
Converts models to INT8 and INT4 for efficient inference.
"""

import os
import sys
import json
import time
import psutil
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def test_generation(model, tokenizer, device, prompt="def fibonacci(n):\n    "):
    """Test model generation quality."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = outputs[0].shape[0] - inputs['input_ids'].shape[1]
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

    return {
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second
    }

def quantize_and_save(base_model_name, adapter_path, output_dir, device):
    """Quantize model and save results."""

    results = {
        "base_model": base_model_name,
        "adapter_path": adapter_path if adapter_path else "None (base model)",
        "device": device,
        "quantization_results": []
    }

    # Test configurations
    configs = [
        {"name": "FP16 (baseline)", "load_in_8bit": False, "load_in_4bit": False},
        {"name": "INT8", "load_in_8bit": True, "load_in_4bit": False},
        {"name": "INT4", "load_in_8bit": False, "load_in_4bit": True},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")

        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            mem_before = get_memory_usage_mb()

            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            # Load model
            print(f"Loading model with {config['name']}...")
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }

            if config["load_in_8bit"]:
                load_kwargs["load_in_8bit"] = True
            elif config["load_in_4bit"]:
                load_kwargs["load_in_4bit"] = True
            else:
                load_kwargs["torch_dtype"] = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **load_kwargs
            )

            # Load adapter if provided
            if adapter_path:
                print("Loading LoRA adapter...")
                model = PeftModel.from_pretrained(model, adapter_path)

            mem_after_load = get_memory_usage_mb()
            model_size_mb = get_model_size_mb(model)

            # Test generation
            print("Testing generation...")
            gen_result = test_generation(model, tokenizer, device)

            # Collect results
            result = {
                "config": config["name"],
                "model_size_mb": round(model_size_mb, 2),
                "memory_usage_mb": round(mem_after_load - mem_before, 2),
                "tokens_per_second": round(gen_result["tokens_per_second"], 2),
                "generation_time": round(gen_result["generation_time"], 3),
                "sample_output": gen_result["generated_text"][:200]
            }

            results["quantization_results"].append(result)

            print(f"\n✅ Results for {config['name']}:")
            print(f"   Model size: {result['model_size_mb']:.2f} MB")
            print(f"   Memory usage: {result['memory_usage_mb']:.2f} MB")
            print(f"   Speed: {result['tokens_per_second']:.2f} tokens/sec")
            print(f"   Sample: {result['sample_output'][:100]}...")

            # Save quantized model if requested
            if config["load_in_8bit"] or config["load_in_4bit"]:
                save_path = output_dir / config["name"].lower().replace(" ", "_")
                save_path.mkdir(parents=True, exist_ok=True)

                print(f"\nSaving to {save_path}...")
                # Note: Quantized models are saved differently
                # For now, we just save the adapter if it exists
                if adapter_path:
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"✅ Saved quantized model to {save_path}")

            # Cleanup
            del model
            del tokenizer

        except Exception as e:
            print(f"❌ Error with {config['name']}: {e}")
            results["quantization_results"].append({
                "config": config["name"],
                "error": str(e)
            })

    return results

def main():
    # Configuration
    base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    adapter_path = "results/models/qlora-exaone-2.4b/best_model"
    output_dir = Path("deployment/optimization/quantized_models")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("MODEL QUANTIZATION PIPELINE")
    print("="*60)
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run quantization
    print("\n" + "="*60)
    print("FINE-TUNED MODEL QUANTIZATION")
    print("="*60)

    results = quantize_and_save(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        output_dir=output_dir,
        device=device
    )

    # Save results
    results_file = output_dir / "quantization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)

    if results["quantization_results"]:
        print(f"\n{'Config':<15} {'Size (MB)':<12} {'Memory (MB)':<15} {'Speed (tok/s)':<15}")
        print("-" * 60)
        for result in results["quantization_results"]:
            if "error" not in result:
                print(f"{result['config']:<15} {result['model_size_mb']:<12} "
                      f"{result['memory_usage_mb']:<15} {result['tokens_per_second']:<15}")

    print("\n✅ Quantization complete!")

if __name__ == "__main__":
    main()
