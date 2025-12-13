# LM Server

FastAPI server for drone mission planning using Qwen2.5-1.5B-Instruct.

## Quick Start

```bash
cd combined_sar/lm_server
./run_server.sh
```

Or manually:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /` - Server status
- `GET /health` - Health check
- `POST /plan` - Generate mission plan

## Usage

```python
import requests

response = requests.post("http://localhost:8000/plan", json={
    "observations": '{"drone_position": {"x": 0, "y": 0, "z": 1}, "obstacles": []}',
    "history": "[]"
})
plan = response.json()
print(plan)
```

## Model

Uses `Qwen/Qwen2.5-1.5B-Instruct` (~3GB download on first run).

Supports:

- Apple MPS (Metal Performance Shaders)
- NVIDIA CUDA
- CPU fallback
