# Workarounds & Creative Patterns Guide

## DGX Spark Expert System — Phase 1 Platform Document

---

This document focuses on **how** to accomplish things on the DGX Spark. Implementation-first: patterns, commands, code, and configurations come before explanations.

---

## 1. CUDA 13 Wheel Gap Workarounds

The DGX Spark requires CUDA 13 / sm_121. Most PyPI packages ship CUDA 12.x wheels. Five paths around this:

### Path A: NGC Containers (Recommended for Most Cases)

Pull pre-built images where CUDA 13 + framework are already correctly compiled:

```bash
# PyTorch (includes CUDA 13, cuDNN 9.13, NCCL)
docker pull nvcr.io/nvidia/pytorch:25.04-py3
docker run --runtime=nvidia --gpus all --shm-size=16g \
  -v /home/$USER/projects:/workspace \
  -v /home/$USER/models:/models \
  --memory=100g --oom-score-adj=500 \
  -it nvcr.io/nvidia/pytorch:25.04-py3

# TensorFlow
docker pull nvcr.io/nvidia/tensorflow:25.04-tf2-py3

# CUDA base (for custom builds)
docker pull nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# vLLM (production inference server)
docker pull nvcr.io/nvidia/vllm:v0.10.1.1

# RAPIDS (cuDF, cuML, cuGraph)
docker pull nvcr.io/nvidia/rapidsai/base:25.04-cuda13.0-py3.12
```

Inside the container, `pip install` works normally for additional pure-Python packages. The container's CUDA 13 runtime is correctly linked.

### Path B: dgx-spark-vllm PyPI Package

NVIDIA publishes a pre-compiled package that bundles PyTorch 2.9.0 + vLLM 0.10.1.1 + CUDA 13 binaries:

```bash
uv venv ~/envs/vllm-spark
source ~/envs/vllm-spark/bin/activate
uv pip install dgx-spark-vllm

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This installs natively (no Docker) and provides a working PyTorch + vLLM stack for LLM inference.

### Path C: Source Builds

Build PyTorch (or other frameworks) from source targeting sm_121:

```bash
# Inside an NGC CUDA devel container for build dependencies
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="12.1a"
export CMAKE_PREFIX_PATH=$(python -c "import sys; print(sys.prefix)")
export USE_CUDA=1
export USE_CUDNN=1
python setup.py develop

# For reference build scripts, see:
# https://github.com/natolambert/dgx-spark-setup
```

Source builds take 1-4 hours on the DGX Spark but produce a fully optimized binary for the exact hardware.

### Path D: Binary Compatibility

PyTorch wheels compiled for CUDA 12.8 / sm_120 are binary-compatible with sm_121 for many operations:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Test compatibility
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print(f'Matrix multiply works: {y.shape}')
"
```

This works for many workloads but may fail on operations that require sm_121-specific code paths. Test thoroughly for your specific use case.

### Path E: Ollama (Pre-installed)

For LLM inference, Ollama handles all CUDA compatibility internally:

```bash
# Already installed and running on DGX Spark
ollama pull llama3.1:8b
ollama run llama3.1:8b "Explain quantum computing in one paragraph"

# API access
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Explain quantum computing",
  "stream": false
}'
```

Zero configuration. Ollama manages its own CUDA binaries and model format conversions.

---

## 2. Flash Attention Replacement

The upstream `flash-attn` package embeds sm_80 CUDA kernels that crash on sm_121. Do not install it. Use PyTorch's native SDPA instead.

### The Problem

```bash
# DO NOT DO THIS — will fail on DGX Spark
pip install flash-attn  # Compiles sm_80 kernels → CUDA error on sm_121
```

### The Solution: PyTorch SDPA with cuDNN Backend

```python
import torch
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(True)

query = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
key = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
value = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)

output = F.scaled_dot_product_attention(query, key, value)
```

### Why This is Actually Better

Community benchmarks show that SDPA with the cuDNN 9.13 backend on Blackwell is **faster** than Flash Attention 2 on previous architectures. The cuDNN backend is specifically optimized for Blackwell's tensor cores. You're not losing performance — you're gaining it.

### For Libraries That Import flash-attn

Some libraries (e.g., certain HuggingFace model implementations) try to import `flash_attn` at the module level. Use conditional imports:

```python
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    
def attention_forward(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)
```

Or set environment variables before import:

```bash
export ATTN_BACKEND=sdpa
# or for HuggingFace Transformers:
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
```

---

## 3. OOM Zombie Mode Prevention

Unified memory means GPU OOM = system OOM = potential system freeze. Prevent this proactively.

### Docker: Memory Limits and OOM Priority

```bash
# Always set memory limit and OOM score adjustment
docker run --runtime=nvidia --gpus all \
  --memory=100g \
  --memory-swap=100g \
  --oom-score-adj=500 \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:25.04-py3
```

- `--memory=100g`: Hard cap at 100GB (leaves ~28GB for OS, Ollama, other processes).
- `--memory-swap=100g`: Same as memory — prevents swap from masking the issue.
- `--oom-score-adj=500`: Makes this container a high-priority target for the OOM killer, so it gets killed before the system freezes.

### cgroup v2 Memory Limits (Non-Docker Workloads)

```bash
# Create a cgroup for ML workloads
sudo mkdir -p /sys/fs/cgroup/ml-workloads
echo "107374182400" | sudo tee /sys/fs/cgroup/ml-workloads/memory.max  # 100GB
echo "$$" | sudo tee /sys/fs/cgroup/ml-workloads/cgroup.procs

# Run your workload — it inherits the cgroup limit
python train.py
```

### Monitoring Script

```bash
#!/usr/bin/env bash
# memory-watchdog.sh — Kill largest process if memory exceeds threshold

THRESHOLD_PERCENT=90
CHECK_INTERVAL=10

while true; do
    USED_PERCENT=$(free | awk '/^Mem:/ {printf "%.0f", $3/$2 * 100}')
    
    if [ "$USED_PERCENT" -gt "$THRESHOLD_PERCENT" ]; then
        LARGEST_PID=$(ps aux --sort=-%mem | awk 'NR==2 {print $2}')
        LARGEST_CMD=$(ps -p "$LARGEST_PID" -o comm=)
        LARGEST_MEM=$(ps -p "$LARGEST_PID" -o %mem=)
        
        logger "MEMORY WATCHDOG: ${USED_PERCENT}% used. Killing PID $LARGEST_PID ($LARGEST_CMD, ${LARGEST_MEM}% mem)"
        kill -9 "$LARGEST_PID"
    fi
    
    sleep "$CHECK_INTERVAL"
done
```

Install as a systemd service for automatic startup:

```bash
sudo tee /etc/systemd/system/memory-watchdog.service << 'EOF'
[Unit]
Description=Memory Watchdog for DGX Spark
After=multi-user.target

[Service]
Type=simple
ExecStart=/home/user/scripts/memory-watchdog.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now memory-watchdog
```

### NVIDIA Guideline

Set Docker memory limits to approximately 90% of available memory. On a 128GB DGX Spark:

| Scenario | Docker --memory | Headroom |
|----------|----------------|----------|
| Single heavy workload | 100g | 28GB for OS + services |
| Workload + Ollama | 80g | 48GB for OS + Ollama models |
| Two concurrent containers | 50g each | 28GB for OS |

---

## 4. Thermal Management

140W in a 150mm cube requires active thermal management for sustained workloads.

### Physical Setup

```
Minimum clearance diagram:

         10cm+
    ┌─────────────┐
    │             │ 10cm+
    │  DGX Spark  │───────→ (exhaust side - most important)
    │             │ 10cm+
    └─────────────┘
         10cm+
         
Do NOT place in: enclosed shelves, cabinets, corners against walls, stacked on other heat sources.
```

- Place on an open desk or shelf with clearance on all sides (minimum 10cm, more is better).
- Keep ambient temperature below 30°C. Below 25°C preferred for sustained loads.
- Point an external USB fan at the exhaust vents during sustained workloads.

### Temperature Monitoring

```bash
# Real-time temperature monitoring
watch -n 2 'paste <(cat /sys/class/thermal/thermal_zone*/type) <(cat /sys/class/thermal/thermal_zone*/temp) | awk "{printf \"%-20s %3.1f°C\n\", \$1, \$2/1000}"'

# Quick check
cat /sys/class/thermal/thermal_zone*/temp | awk '{printf "Zone: %.1f°C\n", $1/1000}'

# GPU temperature (nvidia-smi still reports this)
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
```

### Workload Scheduling: Burst-Then-Pause

For training workloads, schedule cooling breaks:

```python
import time
import subprocess

def get_gpu_temp():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

THROTTLE_TEMP = 80
RESUME_TEMP = 70

def thermal_aware_training(train_step_fn, num_steps):
    for step in range(num_steps):
        temp = get_gpu_temp()
        
        if temp >= THROTTLE_TEMP:
            print(f"Step {step}: GPU at {temp}°C — pausing to cool...")
            while get_gpu_temp() > RESUME_TEMP:
                time.sleep(30)
            print(f"Resumed at {get_gpu_temp()}°C")
        
        train_step_fn(step)
```

### Firmware Updates

```bash
# Check for updates in DGX Dashboard (browser-based)
# Or via CLI if available:
sudo nvsm show updates
```

NVIDIA has released firmware patches that improve thermal management. Always keep firmware current.

---

## 5. Memory Monitoring (nvidia-smi Workaround)

`nvidia-smi` reports "Memory-Usage: Not Supported" on unified memory. Here's what works instead.

### System Memory (= GPU Memory on Unified Architecture)

```bash
# Overall memory usage
free -h

# Detailed breakdown
cat /proc/meminfo | head -20

# Watch memory in real-time
watch -n 1 free -h
```

### PyTorch GPU Memory Tracking

```python
import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")

# Call after model loading, after inference, etc.
print_gpu_memory()
```

### Comprehensive Monitoring Script

```python
#!/usr/bin/env python3
"""Memory monitor for DGX Spark unified memory architecture."""

import json
import time
import subprocess
from pathlib import Path

def get_memory_info():
    meminfo = Path("/proc/meminfo").read_text()
    info = {}
    for line in meminfo.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            value_kb = int(parts[1])
            info[key] = value_kb
    
    total_gb = info["MemTotal"] / 1024 / 1024
    available_gb = info["MemAvailable"] / 1024 / 1024
    used_gb = total_gb - available_gb
    percent_used = (used_gb / total_gb) * 100
    
    return {
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "available_gb": round(available_gb, 2),
        "percent_used": round(percent_used, 1),
        "buffers_gb": round(info.get("Buffers", 0) / 1024 / 1024, 2),
        "cached_gb": round(info.get("Cached", 0) / 1024 / 1024, 2),
    }

def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = result.stdout.strip().split(", ")
        return {
            "gpu_util_percent": int(parts[0]),
            "gpu_temp_c": int(parts[1]),
            "power_draw_w": float(parts[2]),
        }
    except Exception:
        return {"gpu_util_percent": -1, "gpu_temp_c": -1, "power_draw_w": -1}

if __name__ == "__main__":
    while True:
        mem = get_memory_info()
        gpu = get_gpu_info()
        
        print(f"Memory: {mem['used_gb']:.1f}/{mem['total_gb']:.1f} GB "
              f"({mem['percent_used']:.0f}%) | "
              f"GPU: {gpu['gpu_util_percent']}% util, "
              f"{gpu['gpu_temp_c']}°C, "
              f"{gpu['power_draw_w']:.0f}W")
        
        if mem["percent_used"] > 90:
            print("⚠ WARNING: Memory above 90% — OOM risk!")
        if gpu["gpu_temp_c"] > 80:
            print("⚠ WARNING: GPU temperature above 80°C — throttling likely!")
        
        time.sleep(5)
```

### Prometheus Integration

```yaml
# docker-compose.yml for monitoring stack
services:
  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "127.0.0.1:9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "127.0.0.1:9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
```

node-exporter exposes `/proc/meminfo` metrics that Prometheus can scrape. Since unified memory IS system memory, standard Linux memory metrics accurately reflect GPU memory usage.

---

## 6. Non-Standard LLM Usage Patterns

Beyond basic chatbot usage, the DGX Spark supports advanced local LLM patterns.

### Model Chaining (Classifier → Specialist)

Use a small model to route requests to the appropriate larger model:

```python
import requests

OLLAMA_API = "http://localhost:11434/api/generate"

def classify_request(prompt: str) -> str:
    """Use a small 4B model to classify the request type."""
    response = requests.post(OLLAMA_API, json={
        "model": "nemotron-mini:4b",
        "prompt": f"Classify this request as one of: CODE, ANALYSIS, CREATIVE, FACTUAL. Request: {prompt}\nCategory:",
        "stream": False,
        "options": {"num_predict": 10}
    })
    return response.json()["response"].strip().split()[0].upper()

MODEL_MAP = {
    "CODE": "deepseek-coder-v2:16b",
    "ANALYSIS": "nemotron:20b",
    "CREATIVE": "llama3.1:8b",
    "FACTUAL": "nemotron:8b",
}

def route_and_generate(prompt: str) -> str:
    category = classify_request(prompt)
    model = MODEL_MAP.get(category, "llama3.1:8b")
    
    response = requests.post(OLLAMA_API, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
    })
    return response.json()["response"]
```

This saves compute by using the cheapest sufficient model for each request type.

### Local Function Calling

Use tool-use models with structured JSON output:

```python
import json
import requests

TOOLS = [
    {
        "name": "search_documents",
        "description": "Search internal documents by query",
        "parameters": {"query": "string", "limit": "integer"}
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {"expression": "string"}
    },
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {"location": "string"}
    }
]

SYSTEM_PROMPT = f"""You are an assistant with access to these tools:
{json.dumps(TOOLS, indent=2)}

To use a tool, respond with JSON: {{"tool": "tool_name", "args": {{...}}}}
If no tool is needed, respond normally."""

def execute_tool(tool_call: dict):
    name = tool_call["tool"]
    args = tool_call["args"]
    
    if name == "search_documents":
        return search_docs(args["query"], args.get("limit", 5))
    elif name == "calculate":
        return str(eval(args["expression"]))  # sanitize in production
    elif name == "get_weather":
        return get_weather_api(args["location"])

def agent_loop(user_message: str, max_turns: int = 5):
    messages = [user_message]
    
    for _ in range(max_turns):
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.1:8b",
            "system": SYSTEM_PROMPT,
            "prompt": "\n".join(messages),
            "stream": False,
            "format": "json",
        }).json()["response"]
        
        try:
            tool_call = json.loads(response)
            if "tool" in tool_call:
                result = execute_tool(tool_call)
                messages.append(f"Tool result: {result}")
                continue
        except (json.JSONDecodeError, KeyError):
            pass
        
        return response
    
    return "Max turns reached"
```

### LLM-as-Judge

Use a local model to evaluate outputs from another model or pipeline — zero API cost:

```python
def llm_judge(output: str, criteria: str) -> dict:
    """Score an output using a local LLM as evaluator."""
    prompt = f"""Rate the following output on a scale of 1-10 based on these criteria: {criteria}

Output to evaluate:
{output}

Respond with JSON: {{"score": <1-10>, "reasoning": "<brief explanation>"}}"""
    
    response = requests.post(OLLAMA_API, json={
        "model": "nemotron:20b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })
    return json.loads(response.json()["response"])

# Use for automated evaluation of RAG outputs, fine-tuned model quality, etc.
result = llm_judge(
    output="Paris is the capital of France, located on the Seine river.",
    criteria="factual accuracy, completeness, conciseness"
)
print(f"Score: {result['score']}/10 — {result['reasoning']}")
```

### Custom Micro-Services

Wrap model inference as specialized REST APIs:

```python
from fastapi import FastAPI
import requests

app = FastAPI()
OLLAMA = "http://localhost:11434/api/generate"

@app.post("/api/summarize")
async def summarize(text: str, max_length: int = 200):
    resp = requests.post(OLLAMA, json={
        "model": "nemotron:8b",
        "prompt": f"Summarize the following in under {max_length} words:\n\n{text}",
        "stream": False,
    })
    return {"summary": resp.json()["response"]}

@app.post("/api/classify")
async def classify(text: str, categories: list[str]):
    cats = ", ".join(categories)
    resp = requests.post(OLLAMA, json={
        "model": "nemotron-mini:4b",
        "prompt": f"Classify into one of [{cats}]: {text}\nCategory:",
        "stream": False,
        "options": {"num_predict": 10},
    })
    return {"category": resp.json()["response"].strip()}

@app.post("/api/extract-entities")
async def extract_entities(text: str):
    resp = requests.post(OLLAMA, json={
        "model": "llama3.1:8b",
        "prompt": f"Extract all named entities (people, organizations, locations, dates) from this text as JSON:\n\n{text}",
        "stream": False,
        "format": "json",
    })
    return {"entities": resp.json()["response"]}
```

```bash
# Run with: uvicorn microservices:app --host 0.0.0.0 --port 8000
```

### Embedding-Based Routing

Route incoming requests to the appropriate service based on semantic similarity:

```python
import numpy as np
import requests

def get_embedding(text: str) -> list[float]:
    resp = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text,
    })
    return resp.json()["embedding"]

ROUTE_CENTROIDS = {
    "code_help": get_embedding("help me write code, debug, programming"),
    "data_analysis": get_embedding("analyze data, statistics, charts, visualization"),
    "writing": get_embedding("write an essay, email, creative writing, summarize"),
    "research": get_embedding("find information, research, explain concept"),
}

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def route_request(query: str) -> str:
    query_emb = get_embedding(query)
    scores = {
        route: cosine_similarity(query_emb, centroid)
        for route, centroid in ROUTE_CENTROIDS.items()
    }
    return max(scores, key=scores.get)
```

---

## 7. Adapters/Shims for x86/CUDA-12 Assumptions

Many tools assume x86 architecture and CUDA 12. Here's how to work around those assumptions.

### LD_PRELOAD Shims

For libraries that hardcode CUDA 12 library paths:

```bash
# If a library looks for libcudart.so.12, create a symlink to CUDA 13
sudo ln -sf /usr/local/cuda-13/lib64/libcudart.so /usr/local/cuda-13/lib64/libcudart.so.12

# Or use LD_PRELOAD to override specific library loads
LD_PRELOAD=/usr/local/cuda-13/lib64/libcudart.so python my_script.py
```

This is a last-resort workaround. API-incompatible changes between CUDA 12 and 13 may cause runtime errors even with the shim.

### Container Isolation for x86-Assuming Tools

```dockerfile
# Dockerfile for tools that assume specific CUDA 12 paths
FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# Create compatibility symlinks inside the container
RUN ln -sf /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcudart.so.12 && \
    ln -sf /usr/local/cuda/lib64/libcublas.so /usr/local/cuda/lib64/libcublas.so.12

# Install your tool
RUN pip install problematic-tool-that-assumes-cuda12

# The container's controlled environment prevents system-level conflicts
```

### Conditional Imports Pattern

```python
import sys

def get_attention_fn():
    """Get the best available attention implementation for this platform."""
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except (ImportError, RuntimeError):
        import torch.nn.functional as F
        def sdpa_attention(q, k, v, **kwargs):
            return F.scaled_dot_product_attention(q, k, v)
        return sdpa_attention

attention = get_attention_fn()
```

### Build Wrapper Script

Automatically detect aarch64 + sm_121 and set correct build flags:

```bash
#!/usr/bin/env bash
# build-for-spark.sh — Wrapper that detects DGX Spark and sets correct flags

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    # Check for Blackwell GPU
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    
    if [[ "$GPU_ARCH" == "12.1" ]]; then
        echo "Detected DGX Spark (aarch64 + sm_121)"
        export TORCH_CUDA_ARCH_LIST="12.1a"
        export CUDA_HOME="/usr/local/cuda-13"
        export CMAKE_CUDA_ARCHITECTURES="121"
        export FORCE_CUDA=1
        
        # Blackwell-specific optimizations
        export NVCC_FLAGS="-gencode arch=compute_121,code=sm_121"
    fi
fi

# Execute the actual build command
exec "$@"
```

Usage:

```bash
# Wraps any build command with correct flags
./build-for-spark.sh pip install some-package --no-binary :all:
./build-for-spark.sh python setup.py install
./build-for-spark.sh cmake .. && make -j$(nproc)
```

---

## 8. Multi-Model Serving Patterns

The 128GB unified memory can host multiple models simultaneously with the right serving strategy.

### Ollama Multi-Model (Simplest)

Ollama keeps recently used models in memory using LRU eviction:

```bash
# Pull multiple models
ollama pull llama3.1:8b
ollama pull deepseek-coder-v2:16b
ollama pull nomic-embed-text

# Use any model — Ollama loads/unloads automatically
ollama run llama3.1:8b "general question"
ollama run deepseek-coder-v2:16b "code question"

# Configure parallel model slots
# In /etc/systemd/system/ollama.service or OLLAMA_NUM_PARALLEL
export OLLAMA_NUM_PARALLEL=4
```

Ollama is optimal for 2-3 concurrent models. Beyond that, memory pressure causes frequent model swapping.

### vLLM Multi-Instance with Reverse Proxy

Run separate vLLM instances for different models:

```bash
# Instance 1: Chat model on port 8001
docker run -d --name vllm-chat \
  --runtime=nvidia --gpus all \
  --memory=50g --oom-score-adj=500 \
  -p 8001:8000 \
  nvcr.io/nvidia/vllm:v0.10.1.1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.4

# Instance 2: Code model on port 8002
docker run -d --name vllm-code \
  --runtime=nvidia --gpus all \
  --memory=50g --oom-score-adj=500 \
  -p 8002:8000 \
  nvcr.io/nvidia/vllm:v0.10.1.1 \
  --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --gpu-memory-utilization 0.4
```

Nginx reverse proxy for routing:

```nginx
# /etc/nginx/conf.d/llm-router.conf
upstream chat_model {
    server 127.0.0.1:8001;
}

upstream code_model {
    server 127.0.0.1:8002;
}

server {
    listen 8000;
    
    location /v1/chat/ {
        proxy_pass http://chat_model;
    }
    
    location /v1/code/ {
        proxy_pass http://code_model;
    }
}
```

### Resource Budget Planning

| Configuration | Models | Memory Budget | Headroom |
|---------------|--------|---------------|----------|
| Light | 8B chat + embedding | ~15GB | ~113GB |
| Standard | 8B chat + 4B code + embedding | ~20GB | ~108GB |
| Heavy | 20B chat + 16B code + embedding + vector DB | ~55GB | ~73GB |
| Maximum | 70B (4-bit) + embedding + vector DB | ~50GB | ~78GB |

Leave at least 28GB free for OS, services, and burst allocations.

### Scheduled Model Loading

For batch processing patterns, load heavy models on demand:

```python
import subprocess
import time

def load_model(model_name: str):
    """Pre-load a model into Ollama's memory."""
    subprocess.run(["ollama", "run", model_name, "warmup"], 
                   capture_output=True, timeout=120)

def unload_model(model_name: str):
    """Unload a model by setting keep_alive to 0."""
    import requests
    requests.post("http://localhost:11434/api/generate", json={
        "model": model_name,
        "keep_alive": 0,
    })

# Batch processing schedule
load_model("nemotron:20b")
process_batch("analysis_queue")  # Heavy model for batch analysis
unload_model("nemotron:20b")

load_model("llama3.1:8b")
# Lighter model stays loaded for interactive use
```

---

## 9. Creative Agent Architectures

Advanced patterns that leverage the DGX Spark's local compute for agent workflows.

### Split Brain: Local Fast + Cloud Smart

Use a small local model for fast routing and a cloud model for complex reasoning:

```python
import requests
import os

LOCAL_API = "http://localhost:11434/api/generate"
CLOUD_API = "https://api.openai.com/v1/chat/completions"

def local_classify(prompt: str) -> dict:
    """Fast local classification: simple vs complex."""
    resp = requests.post(LOCAL_API, json={
        "model": "nemotron-mini:4b",
        "prompt": f"Rate complexity 1-5. Just the number.\n\n{prompt}",
        "stream": False,
        "options": {"num_predict": 5},
    })
    try:
        complexity = int(resp.json()["response"].strip()[0])
    except (ValueError, IndexError):
        complexity = 3
    return {"complexity": complexity}

def generate(prompt: str) -> str:
    classification = local_classify(prompt)
    
    if classification["complexity"] <= 2:
        # Simple: handle locally (fast, free)
        resp = requests.post(LOCAL_API, json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
        })
        return resp.json()["response"]
    else:
        # Complex: escalate to cloud (slower, costs money, higher quality)
        resp = requests.post(CLOUD_API, json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
        }, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return resp.json()["choices"][0]["message"]["content"]
```

This minimizes latency for simple tasks (local round-trip ~100ms) and cost (only complex queries hit the cloud API).

### Local RAG + Optional Cloud Generation

Keep retrieval entirely local (fast, private), optionally augment generation with cloud:

```python
class HybridRAG:
    def __init__(self, use_cloud_generation: bool = False):
        self.use_cloud = use_cloud_generation
    
    def embed(self, text: str) -> list[float]:
        """Always local — fast and private."""
        resp = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": text,
        })
        return resp.json()["embedding"]
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Always local — vector DB runs on DGX Spark."""
        query_emb = self.embed(query)
        # chromadb / milvus retrieval here
        return self.vector_db.query(query_emb, top_k=top_k)
    
    def generate(self, query: str, context: list[str]) -> str:
        context_str = "\n\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        
        if self.use_cloud:
            return self._cloud_generate(prompt)
        else:
            return self._local_generate(prompt)
    
    def _local_generate(self, prompt: str) -> str:
        resp = requests.post(LOCAL_API, json={
            "model": "nemotron:20b",
            "prompt": prompt,
            "stream": False,
        })
        return resp.json()["response"]
    
    def _cloud_generate(self, prompt: str) -> str:
        resp = requests.post(CLOUD_API, json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
        }, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        return resp.json()["choices"][0]["message"]["content"]
    
    def query(self, question: str) -> str:
        docs = self.retrieve(question)
        return self.generate(question, docs)
```

Data never leaves the device during retrieval. Cloud generation is optional for quality-critical applications.

### Autonomous Loops with Safety Gates

Agent proposes actions, confidence check determines auto-execute vs. human review:

```python
from dataclasses import dataclass
from enum import Enum

class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ProposedAction:
    action: str
    parameters: dict
    confidence: Confidence
    reasoning: str

def agent_loop(task: str, max_iterations: int = 10):
    context = {"task": task, "history": []}
    
    for i in range(max_iterations):
        # Agent proposes next action
        proposal = propose_action(context)
        
        if proposal.confidence == Confidence.HIGH:
            # Auto-execute
            result = execute_action(proposal)
            context["history"].append({"action": proposal, "result": result})
        
        elif proposal.confidence == Confidence.MEDIUM:
            # Execute with logging for audit
            result = execute_action(proposal)
            log_for_review(proposal, result)
            context["history"].append({"action": proposal, "result": result})
        
        else:
            # Queue for human review
            queue_for_review(proposal)
            result = wait_for_human_approval(proposal)
            if result is None:
                continue
            context["history"].append({"action": proposal, "result": result})
        
        if is_task_complete(context):
            return context["history"]
    
    return context["history"]
```

### Fan-Out Pattern: Concurrent Sub-Tasks

Spawn multiple sub-tasks running concurrently on the DGX Spark:

```python
import asyncio
import aiohttp

async def run_subtask(session: aiohttp.ClientSession, model: str, prompt: str) -> str:
    async with session.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
    }) as resp:
        data = await resp.json()
        return data["response"]

async def fan_out_analysis(document: str):
    """Run 4 concurrent analysis tasks on the same document."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            run_subtask(session, "nemotron:8b", 
                        f"Summarize this document:\n\n{document}"),
            run_subtask(session, "nemotron:8b", 
                        f"Extract key entities from this document:\n\n{document}"),
            run_subtask(session, "nemotron:8b", 
                        f"What is the sentiment of this document?\n\n{document}"),
            run_subtask(session, "nemotron:8b", 
                        f"List action items from this document:\n\n{document}"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            "summary": results[0],
            "entities": results[1],
            "sentiment": results[2],
            "action_items": results[3],
        }

# 4 concurrent sub-agents = ~2.6x single-task time
# Much faster than running sequentially (4x single-task time)
result = asyncio.run(fan_out_analysis(document_text))
```

This pattern is effective because Ollama can batch concurrent requests to the same model, achieving better GPU utilization than sequential processing.

---

## Summary

The DGX Spark's constraints (CUDA 13 wheel gap, unified memory OOM risk, thermal throttling, broken nvidia-smi) each have practical workarounds that, once implemented, make the platform highly productive. The key meta-pattern is: **containers for GPU workloads, uv for Python management, memory limits everywhere, and creative multi-model architectures to maximize the 128GB unified memory**. The workarounds in this document convert the DGX Spark from a frustrating experience (for those who treat it like a standard x86 GPU workstation) into a capable local AI development platform.

## Practical Implications

- **First-hour setup**: Install uv, pull 2-3 NGC containers, configure Docker memory limits, and set up the memory watchdog. This takes ~30 minutes and prevents the most common pain points.
- **CUDA 13 gap is a one-time problem**: Once you establish a container-based workflow or install dgx-spark-vllm, you never think about it again. The upfront configuration cost pays for itself immediately.
- **Multi-model serving is practical**: With 128GB, running 2-3 models simultaneously is comfortable. The model chaining and routing patterns let you build sophisticated AI applications without cloud dependencies.
- **Agent architectures are the monetization path**: The creative patterns in Sections 6 and 9 are not academic exercises — they're the building blocks of billable AI products and services.

## Recommended Actions

1. **Immediate**: Set up NGC container pulls and Docker memory limits (Section 1A + Section 3).
2. **Immediate**: Install uv and create a project template with the build-for-spark wrapper (Section 7).
3. **First week**: Deploy the memory monitoring script as a systemd service (Section 5).
4. **First week**: Build a model chaining prototype with 2 models (Section 6) to validate the pattern.
5. **First month**: Implement a production micro-service architecture (Section 6) for a client project.
6. **Ongoing**: Test new workaround paths as the CUDA 13 ecosystem matures — the wheel gap will close over time.

## Confidence Level

**HIGH** — The workarounds in this document are based on confirmed techniques: NGC container usage is NVIDIA-recommended, SDPA/cuDNN replacing Flash Attention is validated by community benchmarks, Docker memory limits are standard Linux container practice, and the creative patterns are built on documented Ollama/vLLM APIs. The build compatibility workarounds (Path D, Section 7) carry slightly lower confidence as they depend on specific binary compatibility that may break with future CUDA updates.

## Source Notes

- NGC container images and runtime flags: [NVIDIA-OFFICIAL] NGC catalog documentation; nvidia-container-toolkit docs.
- dgx-spark-vllm package: [NVIDIA-OFFICIAL] Published on PyPI by NVIDIA.
- Source build procedures: [COMMUNITY] natolambert/dgx-spark-setup repository; [TESTED] Build flags validated.
- CUDA 12.8 → sm_121 binary compatibility: [COMMUNITY] Reported working by multiple users; [TESTED] PyTorch basic operations confirmed.
- Flash Attention → SDPA replacement: [COMMUNITY] Benchmarks showing SDPA+cuDNN faster on Blackwell; [NVIDIA-OFFICIAL] cuDNN 9.13 optimized for Blackwell.
- OOM prevention with Docker: [NVIDIA-OFFICIAL] Recommendation to set memory limits to ~90% of available; [TESTED] Docker --memory and --oom-score-adj confirmed effective.
- Thermal management: [NVIDIA-OFFICIAL] Clearance recommendations; [COMMUNITY] Carmack reboot reports; [TESTED] Temperature monitoring scripts.
- nvidia-smi workaround: [TESTED] Confirmed "Not Supported" on DGX Spark; /proc/meminfo confirmed as alternative.
- Multi-model serving patterns: [COMMUNITY] Ollama multi-model usage; [TESTED] Concurrent model serving validated.
- Agent architectures: [COMMUNITY] 4-agent concurrency benchmark; [INFERRED] Split-brain and fan-out patterns adapted from distributed systems patterns.

## Unresolved Questions

1. **CUDA 12.8 binary compat longevity**: Will future PyTorch releases break sm_120 → sm_121 binary compatibility? No guarantee from NVIDIA or PyTorch.
2. **dgx-spark-vllm update cadence**: How often will NVIDIA update this package? Will it expand to cover more frameworks?
3. **Ollama memory management tuning**: What are the optimal `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` settings for 128GB unified memory?
4. **cuDNN SDPA performance parity**: Are there specific attention patterns where SDPA+cuDNN underperforms Flash Attention 2, or is it strictly better on Blackwell?
5. **cgroup v2 GPU memory accounting**: Does the kernel correctly account GPU unified memory allocations against cgroup memory limits, or can GPU allocations bypass the limit?
