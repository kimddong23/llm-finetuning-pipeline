"""
FastAPI server for code generation.
Serves the fine-tuned EXAONE model via REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Code Generation API",
    description="REST API for fine-tuned EXAONE code generation model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None
device = None

class GenerationRequest(BaseModel):
    """Request model for code generation."""
    prompt: str = Field(..., description="Code prompt or partial function", example="def fibonacci(n):")
    max_tokens: int = Field(default=256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    num_return_sequences: int = Field(default=1, ge=1, le=5, description="Number of completions to generate")

class GenerationResponse(BaseModel):
    """Response model for code generation."""
    completions: List[str] = Field(..., description="Generated code completions")
    prompt: str = Field(..., description="Original prompt")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, tokenizer, device

    try:
        logger.info("Loading model...")

        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info(f"Using device: {device}")

        # Load tokenizer
        base_model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA adapter
        adapter_path = "results/models/qlora-exaone-2.4b/best_model"
        model = PeftModel.from_pretrained(model, adapter_path)

        logger.info("✅ Model loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device if device else "unknown"
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_code(request: GenerationRequest):
    """Generate code completion from prompt."""

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=request.num_return_sequences
            )

        generation_time = time.time() - start_time

        # Decode outputs
        completions = []
        for output in outputs:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # Extract only the new tokens
            completion = generated_text[len(request.prompt):].strip()
            completions.append(completion)

        tokens_generated = outputs[0].shape[0] - inputs['input_ids'].shape[1]

        return GenerationResponse(
            completions=completions,
            prompt=request.prompt,
            generation_time=round(generation_time, 3),
            tokens_generated=tokens_generated
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Code Generation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
