"""
Gradio app for HuggingFace Spaces deployment.
Interactive code generation interface.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# Global model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load model on startup."""
    global model, tokenizer, device

    print("Loading model...")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load tokenizer
    base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter (if available)
    try:
        adapter_path = "best_model"  # Relative path in Spaces
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✅ Fine-tuned model loaded!")
    except Exception as e:
        print(f"⚠️  Using base model (adapter not found): {e}")

def generate_code(prompt, max_tokens, temperature, top_p):
    """Generate code completion."""
    if not prompt.strip():
        return "⚠️  Please enter a prompt"

    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True if temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generation_time = time.time() - start_time

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        # Stats
        tokens_generated = outputs[0].shape[0] - inputs['input_ids'].shape[1]
        stats = f"\n\n---\n📊 **Stats**: {tokens_generated} tokens | {generation_time:.2f}s | {tokens_generated/generation_time:.1f} tok/s"

        return prompt + completion + stats

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load model on startup
load_model()

# Create Gradio interface
with gr.Blocks(title="Code Generation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Code Generation with Fine-tuned EXAONE
        Generate Python code completions using a fine-tuned EXAONE-3.5-2.4B model.

        **How to use:**
        1. Enter a function signature (e.g., `def fibonacci(n):`)
        2. Adjust generation parameters (optional)
        3. Click "Generate Code"
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Code Prompt",
                placeholder="def fibonacci(n):\n    ",
                lines=5,
                value="def fibonacci(n):\n    "
            )

            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=32,
                    maximum=512,
                    value=256,
                    step=32,
                    label="Max Tokens"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Temperature"
                )

            top_p_slider = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.95,
                step=0.05,
                label="Top-p (Nucleus Sampling)"
            )

            generate_btn = gr.Button("🎯 Generate Code", variant="primary")

        with gr.Column(scale=3):
            output_code = gr.Code(
                label="Generated Code",
                language="python",
                lines=15
            )

    # Examples
    gr.Markdown("### 💡 Example Prompts")
    gr.Examples(
        examples=[
            ["def fibonacci(n):\n    ", 256, 0.2, 0.95],
            ["def is_palindrome(s):\n    ", 128, 0.2, 0.95],
            ["def bubble_sort(arr):\n    ", 200, 0.2, 0.95],
            ["def binary_search(arr, target):\n    ", 256, 0.2, 0.95],
            ["class Stack:\n    def __init__(self):\n        ", 300, 0.2, 0.95],
        ],
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider],
        outputs=output_code,
        fn=generate_code,
        cache_examples=False,
    )

    # Event handler
    generate_btn.click(
        fn=generate_code,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider],
        outputs=output_code
    )

    gr.Markdown(
        """
        ---
        ### 📚 About

        This model was fine-tuned on Python code using QLoRA (4-bit quantization + LoRA).

        **Performance:**
        - HumanEval pass@1: 96.95%
        - Training time: 12.5 hours on Mac M3 Pro
        - Trainable params: 4M / 2.4B (0.17%)

        **Links:**
        - [GitHub Repository](https://github.com/kimddong23/llm-finetuning-pipeline)
        - [Technical Report](https://github.com/kimddong23/llm-finetuning-pipeline/blob/main/docs/TECHNICAL_REPORT.md)
        - [Base Model](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct)

        ---
        *Built with 💙 using HuggingFace Transformers, PEFT, and Gradio*
        """
    )

# Launch
if __name__ == "__main__":
    demo.launch()
