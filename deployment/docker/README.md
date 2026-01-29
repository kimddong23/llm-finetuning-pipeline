# Docker Deployment

Containerized deployment for the Code Generation API.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# From the project root
cd deployment/docker

# Build and start
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

The API will be available at `http://localhost:8000`

### Using Docker CLI

```bash
# Build image
docker build -t codegen-api -f deployment/docker/Dockerfile .

# Run container
docker run -d \
  --name codegen-api \
  -p 8000:8000 \
  -v $(pwd)/results/models/qlora-exaone-2.4b/best_model:/app/models/best_model:ro \
  codegen-api
```

## Accessing the API

Once running, access:

- **API**: http://localhost:8000
- **Health**: http://localhost:8000/health
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Management Commands

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

### Docker CLI

```bash
# View logs
docker logs -f codegen-api

# Stop container
docker stop codegen-api

# Start container
docker start codegen-api

# Remove container
docker rm -f codegen-api

# View resource usage
docker stats codegen-api
```

## Configuration

### Environment Variables

Set in `docker-compose.yml` or pass to `docker run`:

```yaml
environment:
  - MODEL_PATH=/app/models/best_model
  - BASE_MODEL=LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
  - PORT=8000
```

### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Maximum memory
    reservations:
      memory: 4G  # Guaranteed memory
```

### Ports

Change exposed port in `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"  # Map container:8000 to host:8080
```

## Production Deployment

### 1. Build Production Image

```bash
docker build -t codegen-api:1.0.0 -f deployment/docker/Dockerfile .
```

### 2. Tag for Registry

```bash
docker tag codegen-api:1.0.0 your-registry.com/codegen-api:1.0.0
```

### 3. Push to Registry

```bash
docker push your-registry.com/codegen-api:1.0.0
```

### 4. Deploy to Production

```bash
docker pull your-registry.com/codegen-api:1.0.0
docker run -d \
  --name codegen-api \
  --restart always \
  -p 80:8000 \
  -m 8G \
  --health-cmd="curl -f http://localhost:8000/health || exit 1" \
  --health-interval=30s \
  --health-retries=3 \
  your-registry.com/codegen-api:1.0.0
```

## Health Checks

The container includes automatic health checks:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' codegen-api

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' codegen-api
```

Health check endpoint: `GET /health`

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs codegen-api

# Common issues:
# - Out of memory: Increase memory limit
# - Model not found: Check volume mount
# - Port conflict: Change port mapping
```

### High Memory Usage

```bash
# Monitor resource usage
docker stats codegen-api

# Reduce memory:
# 1. Use quantized model (INT8/INT4)
# 2. Increase swap space
# 3. Reduce concurrent requests
```

### Slow Response Times

```bash
# Warm up the model
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"def test():","max_tokens":10}'

# First request is slow (model loading)
# Subsequent requests should be fast
```

## Development

### Development Mode with Hot Reload

Uncomment volume mount in `docker-compose.yml`:

```yaml
volumes:
  - ../../deployment/api:/app/api:ro  # Hot reload API code
```

Then:

```bash
docker-compose up --build
```

Changes to API code will be reflected without rebuilding.

### Shell Access

```bash
# Access running container
docker exec -it codegen-api /bin/bash

# Run one-off command
docker exec codegen-api python -c "import torch; print(torch.__version__)"
```

## Image Details

### Size Optimization

The image uses:
- Multi-stage build
- Slim base image (`python:3.10-slim`)
- `.dockerignore` to exclude unnecessary files
- No cache for pip installs

Expected image size: ~4-5 GB (including model)

### Security

- Runs as non-root user (`appuser`)
- Minimal system dependencies
- No unnecessary tools installed

### Layers

```
FROM python:3.10-slim          # ~150 MB
+ System dependencies          # ~50 MB
+ Python packages              # ~2 GB
+ Application code             # ~1 MB
+ Model weights                # ~16 MB (adapter only)
Total:                         # ~2.2 GB base + model download
```

## Alternative Deployment Options

### 1. HuggingFace Spaces

See `deployment/spaces/` for Gradio deployment.

### 2. Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codegen-api
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: codegen-api:1.0.0
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "8Gi"
          requests:
            memory: "4Gi"
```

### 3. Cloud Run (Google Cloud)

```bash
gcloud run deploy codegen-api \
  --image gcr.io/your-project/codegen-api:1.0.0 \
  --memory 8Gi \
  --port 8000 \
  --allow-unauthenticated
```

## License

MIT License - See LICENSE file for details.
