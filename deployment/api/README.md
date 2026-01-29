# Code Generation API

REST API server for the fine-tuned EXAONE code generation model.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Server

```bash
# From the deployment/api directory
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps"
}
```

### 2. Generate Code

```bash
POST /generate
```

**Request Body:**
```json
{
  "prompt": "def fibonacci(n):",
  "max_tokens": 256,
  "temperature": 0.2,
  "top_p": 0.95,
  "num_return_sequences": 1
}
```

**Response:**
```json
{
  "completions": [
    "if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n - 1) + fibonacci(n - 2)"
  ],
  "prompt": "def fibonacci(n):",
  "generation_time": 4.123,
  "tokens_generated": 50
}
```

### 3. API Documentation

Interactive API docs available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Usage Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate code
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def is_palindrome(s):",
    "max_tokens": 100,
    "temperature": 0.2
  }'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Generate code
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "def bubble_sort(arr):",
        "max_tokens": 200,
        "temperature": 0.2,
        "num_return_sequences": 1
    }
)

result = response.json()
print(f"Generated code:\n{result['completions'][0]}")
print(f"Generation time: {result['generation_time']}s")
```

### Using JavaScript (fetch)

```javascript
// Generate code
fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'def quicksort(arr):',
    max_tokens: 256,
    temperature: 0.2
  })
})
.then(response => response.json())
.then(data => console.log(data.completions[0]));
```

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `prompt` | string | *required* | - | Code prompt or partial function |
| `max_tokens` | integer | 256 | 1-2048 | Maximum tokens to generate |
| `temperature` | float | 0.2 | 0.0-2.0 | Sampling temperature (higher = more random) |
| `top_p` | float | 0.95 | 0.0-1.0 | Nucleus sampling threshold |
| `num_return_sequences` | integer | 1 | 1-5 | Number of completions to generate |

## Performance

**Typical Response Times** (Mac M3 Pro, FP16 model):
- Short prompt (< 50 tokens): 2-4 seconds
- Medium prompt (50-200 tokens): 4-8 seconds
- Long prompt (200+ tokens): 8-15 seconds

**Throughput**:
- ~12.5 tokens/second (FP16)
- ~2.1 tokens/second (INT8)

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Successful generation
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Generation error
- `503 Service Unavailable`: Model not loaded

Example error response:
```json
{
  "detail": "Model not loaded"
}
```

## CORS

CORS is enabled for all origins (`*`). Modify in `main.py` for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables

```bash
export MODEL_PATH="results/models/qlora-exaone-2.4b/best_model"
export BASE_MODEL="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
export PORT=8000
```

### Docker

See `deployment/docker/` for containerized deployment.

## Monitoring

Add health checks to your deployment:

```bash
# Kubernetes liveness probe
GET /health

# Response should be 200 OK with model_loaded: true
```

## License

MIT License - See LICENSE file for details.
