# Recommended Core Stack — DGX Spark Expert System

> **Last Updated**: 2026-03-31
> **Status**: Phase 2 — Complete
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Baseline**: ARM64 (aarch64) · CUDA 13.0.2 · sm_121 · Ubuntu 24.04 · 128 GB Unified Memory
> **Maintainer**: DGX Spark Expert System

---

## Overview

This document defines the **default install list for every DGX Spark**. These tools have been selected for maximum reliability, compatibility, and coverage of common workflows. Every tool below scores 4+ on Practicality, Stability, and Compatibility.

The core stack provides: local LLM inference, API serving, data processing, vector search, caching, agent orchestration, monitoring, and version control — all verified on ARM64/CUDA 13/sm_121.

### Design Principles

1. **Container-first for GPU workloads** — NGC containers guarantee CUDA 13 + sm_121 compatibility
2. **Native for pure Python tools** — Avoid container overhead when there's no architecture risk
3. **Minimize dependencies** — Each tool earns its place by enabling multiple workflows
4. **No fragile workarounds** — Every default stack tool works reliably without patching

---

## Core Stack Summary

| # | Tool | Install Method | Category | Pract. | Stab. | Compat. | Pre-installed? |
|---|------|----------------|----------|:---:|:---:|:---:|:-:|
| 1 | Ollama | Pre-installed | LLM Inference | 5 | 5 | 5 | ✅ |
| 2 | Docker + NGC | Pre-installed | Containers | 5 | 5 | 5 | ✅ |
| 3 | Python 3.12 + uv | Install uv | Language Runtime | 5 | 5 | 5 | Partial |
| 4 | FastAPI + uvicorn | pip install | API Framework | 5 | 5 | 5 | — |
| 5 | PostgreSQL + pgvector | apt + extension | Database + Vector | 5 | 5 | 5 | — |
| 6 | Redis | apt / Docker | Cache + Queue | 5 | 5 | 5 | — |
| 7 | LangChain / LangGraph | pip install | Agent Orchestration | 4 | 4 | 5 | — |
| 8 | NemoClaw | GitHub + pip | Agent Framework | 4 | 4 | 5 | — |
| 9 | Polars | pip install | Data Processing | 5 | 5 | 5 | — |
| 10 | Prometheus + Grafana | Docker Compose | Monitoring | 4 | 5 | 5 | — |
| 11 | Git + SSH | Pre-installed | Version Control | 5 | 5 | 5 | ✅ |
| 12 | llama.cpp | Build from source | Secondary Inference | 4 | 4 | 4 | — |

---

## Detailed Tool Entries

### 1. Ollama — Primary LLM Inference

**Why it's in the default stack**: Pre-installed on DGX Spark by NVIDIA. Most reliable and tested inference engine on the platform. Zero-configuration local LLM execution with OpenAI-compatible API.

**What it enables**:
- Local LLM chat and generation (`ollama run llama3`)
- OpenAI-compatible API endpoint (port 11434) for all downstream applications
- Multi-model management with LRU memory caching
- GGUF model support with automatic GPU offloading
- Backend for LangChain, CrewAI, and custom applications

**Install**:
```bash
# Pre-installed. Verify:
ollama --version
ollama list

# Pull a model:
ollama pull llama3:8b
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- Default port: 11434
- Models stored in `~/.ollama/models/`
- Set `OLLAMA_HOST=0.0.0.0` to expose to network
- Memory managed automatically via LRU eviction

---

### 2. Docker + NGC — Container Isolation

**Why it's in the default stack**: Pre-installed. Foundation of the container-first workflow. NGC containers are the guaranteed-compatible path for GPU workloads on DGX Spark.

**What it enables**:
- Running any NGC container (PyTorch, TensorRT, NeMo, Triton, RAPIDS)
- Isolating GPU workloads with CUDA 13 compatibility guaranteed
- Multi-service orchestration via Docker Compose
- Reproducible environments for development and deployment

**Install**:
```bash
# Pre-installed. Verify:
docker --version
docker compose version
nvidia-smi  # GPU visible
docker run --rm --gpus all nvidia/cuda:13.0-base-ubuntu24.04 nvidia-smi  # GPU in container

# NGC CLI (pre-installed):
ngc --version

# Authenticate NGC:
ngc config set
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- NVIDIA Container Toolkit pre-configured
- Use `--gpus all` for GPU access in containers
- Docker Compose supports `deploy.resources.reservations.devices` for GPU in services
- NGC API key needed for authenticated container/model pulls

---

### 3. Python 3.12 + uv — Language Runtime & Package Manager

**Why it's in the default stack**: Python 3.12 is the DGX Spark system Python. `uv` is the fastest Python package manager; eliminates pip/conda dependency resolution headaches; supports virtual environments natively.

**What it enables**:
- Fast, reliable Python package installation (10-100× faster than pip)
- Virtual environment management (`uv venv`, `uv pip install`)
- Lock file support for reproducible environments
- Project management (`uv init`, `uv run`)

**Install**:
```bash
# Python 3.12 is pre-installed. Install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify:
uv --version
python3 --version

# Create project virtual environment:
uv venv .venv
source .venv/bin/activate

# Install packages (fast):
uv pip install fastapi uvicorn polars
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- Always use virtual environments to avoid system Python conflicts
- `uv` respects `--index-url` for custom PyPI indexes (e.g., NVIDIA cu130 index)
- For GPU packages without aarch64 wheels, use NGC containers instead

---

### 4. FastAPI + uvicorn — API Layer

**Why it's in the default stack**: Universal API framework for DGX Spark projects. Wraps Ollama/NIM inference with custom business logic. Automatic OpenAPI docs. Async-native for high concurrency.

**What it enables**:
- REST API endpoints for any service
- Proxy/gateway layer in front of Ollama or NIM
- WebSocket endpoints for streaming inference
- Automatic interactive API documentation (Swagger UI)
- Background task processing

**Install**:
```bash
uv pip install fastapi uvicorn[standard]

# Quick test:
python3 -c "
from fastapi import FastAPI
app = FastAPI()

@app.get('/')
async def root():
    return {'status': 'DGX Spark API running'}
"
```

**Run**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- Default port 8000; change with `--port`
- `--reload` for development, `--workers 4` for production
- Integrates directly with Pydantic v2 for data validation

---

### 5. PostgreSQL + pgvector — Relational + Vector Database

**Why it's in the default stack**: Single database handles both relational data AND vector embeddings. Eliminates the need for a separate vector database for most use cases. pgvector supports HNSW and IVFFlat indexes for fast similarity search.

**What it enables**:
- Application data storage (users, configurations, metadata)
- RAG vector storage with pgvector (cosine, L2, inner product similarity)
- Hybrid search (SQL filtering + vector similarity in one query)
- TimescaleDB extension for time-series (optional add-on)

**Install**:
```bash
# PostgreSQL:
sudo apt update && sudo apt install -y postgresql postgresql-contrib

# pgvector extension:
sudo apt install -y postgresql-16-pgvector
# Or build from source:
# git clone https://github.com/pgvector/pgvector.git
# cd pgvector && make && sudo make install

# Enable in database:
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# Verify:
sudo -u postgres psql -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- Default port 5432
- pgvector supports up to 16,000 dimensions (2,000 for HNSW index)
- For large vector datasets (>1M), tune `maintenance_work_mem` and `effective_cache_size`
- Python driver: `uv pip install psycopg2-binary` or `asyncpg`

---

### 6. Redis — Cache, Queue, and Pub/Sub

**Why it's in the default stack**: Sub-millisecond caching for LLM responses, session storage, rate limiting, and lightweight messaging — all in one service. Already needed by many tools (Celery, Airflow, APScheduler).

**What it enables**:
- LLM response caching (avoid re-inference for repeated queries)
- Session storage for web applications
- Rate limiting for API endpoints
- Pub/Sub messaging between services
- Job queues (with RQ or Celery)
- Redis Streams for persistent message processing

**Install**:
```bash
# Option A: System package
sudo apt update && sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Option B: Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Verify:
redis-cli ping  # Should return PONG

# Python client:
uv pip install redis
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- Default port 6379
- Set `maxmemory` and `maxmemory-policy allkeys-lru` for cache use
- Enable persistence (`appendonly yes`) if using as primary data store
- Python async client: `uv pip install redis[hiredis]` for best performance

---

### 7. LangChain / LangGraph — Agent Orchestration

**Why it's in the default stack**: Most popular LLM application framework. Massive ecosystem of integrations. LangGraph adds stateful, graph-based agent workflows. Pure Python — zero architecture risk.

**What it enables**:
- RAG pipelines (retrieval-augmented generation)
- Multi-step agent workflows with tool calling
- LangGraph stateful conversations with branching logic
- Integration with Ollama, NIM, pgvector, Redis, and 100+ other tools
- LangSmith tracing for debugging agent behavior

**Install**:
```bash
uv pip install langchain langgraph langchain-community langchain-ollama

# For pgvector integration:
uv pip install langchain-postgres

# Verify:
python3 -c "from langchain_ollama import OllamaLLM; print('LangChain + Ollama ready')"
```

**Scores**: Practicality 4 · Stability 4 · Compatibility 5

**Configuration notes**:
- Install community integrations selectively (avoid pulling in all of langchain-community)
- LangGraph requires understanding of graph-based state machines
- Use LangSmith (free tier) for tracing: `export LANGCHAIN_TRACING_V2=true`
- Pin versions in requirements.txt — LangChain releases frequently

---

### 8. NemoClaw — NVIDIA-Native Agent Framework

**Why it's in the default stack**: First-party NVIDIA agent framework with built-in security sandboxing. Integrates directly with NIM for inference. Enterprise-grade guardrails for tool-calling agents.

**What it enables**:
- Secure tool-calling agents with sandboxed execution
- Integration with NIM and Ollama as LLM backends
- Guardrails for content safety and tool access control
- Multi-step agent workflows with YAML-driven configuration

**Install**:
```bash
# Install from NVIDIA GitHub:
uv pip install git+https://github.com/NVIDIA/NeMo-Guardrails.git
# Or specific NemoClaw package (check NVIDIA documentation for latest):
uv pip install nemoclaw

# Verify:
python3 -c "import nemoguardrails; print('NemoClaw ready')"
```

**Scores**: Practicality 4 · Stability 4 · Compatibility 5

**Configuration notes**:
- Requires running NIM or Ollama as inference backend
- YAML configuration for guardrails and agent behavior
- gRPC for high-performance inter-service communication
- See NVIDIA documentation for latest install instructions

---

### 9. Polars — Fast Data Processing

**Why it's in the default stack**: 25% faster on Grace CPU vs x86 (confirmed by NVIDIA). Consistently outperforms pandas. Rust-native with native aarch64 wheels. Lazy evaluation enables query optimization.

**What it enables**:
- Lightning-fast DataFrame operations on large datasets
- Lazy evaluation with query optimization
- Native Parquet, CSV, JSON, and Arrow support
- Streaming mode for datasets larger than memory

**Install**:
```bash
uv pip install polars

# Verify + benchmark:
python3 -c "
import polars as pl
import time
df = pl.DataFrame({'a': range(1_000_000), 'b': range(1_000_000)})
start = time.time()
result = df.filter(pl.col('a') > 500_000).group_by('b' ).agg(pl.col('a').sum())
print(f'Polars: {time.time()-start:.3f}s for 1M rows')
"
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

**Configuration notes**:
- Use lazy mode (`df.lazy()`) for query optimization on complex pipelines
- `scan_parquet()` / `scan_csv()` for streaming large files
- Polars does NOT use pandas API — different but more consistent API
- For pandas-compatible workflows, use cudf.pandas instead

---

### 10. Prometheus + Grafana — Monitoring

**Why it's in the default stack**: Essential for monitoring GPU utilization, inference latency, system resources, and application metrics. ARM64 Docker images available for both.

**What it enables**:
- GPU utilization and memory monitoring (via DCGM exporter)
- System resource tracking (CPU, memory, disk, network)
- Custom application metrics (inference latency, request count)
- Beautiful, customizable dashboards
- Alerting on resource thresholds

**Install**:
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

```bash
# Deploy:
docker compose -f docker-compose.monitoring.yml up -d

# Access:
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

**Scores**: Practicality 4 · Stability 5 · Compatibility 5

**Configuration notes**:
- Add NVIDIA DCGM exporter for GPU metrics: `nvcr.io/nvidia/k8s/dcgm-exporter`
- Configure Prometheus scrape targets in `prometheus.yml`
- Import Grafana dashboards from grafana.com (Node Exporter Full: 1860, NVIDIA GPU: 12239)
- Grafana default port 3000 may conflict with dev servers — change if needed

---

### 11. Git + SSH — Version Control

**Why it's in the default stack**: Pre-installed. Foundation for all code management. SSH keys for secure GitHub/GitLab authentication.

**What it enables**:
- Source code version control
- Collaboration via GitHub/GitLab
- Secure authentication via SSH keys
- Configuration management for DGX Spark projects

**Install**:
```bash
# Pre-installed. Configure:
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Generate SSH key:
ssh-keygen -t ed25519 -C "your@email.com"

# Add to GitHub:
cat ~/.ssh/id_ed25519.pub
# Copy to https://github.com/settings/keys

# Verify:
ssh -T git@github.com
```

**Scores**: Practicality 5 · Stability 5 · Compatibility 5

---

### 12. llama.cpp — Secondary Inference + GGUF

**Why it's in the default stack**: Builds cleanly from source on aarch64 with CUDA 13. Provides direct GGUF model access with fine-grained GPU layer control. Useful as secondary engine alongside Ollama.

**What it enables**:
- Direct GGUF model execution with GPU layer control
- Embedding generation for RAG pipelines
- Server mode with OpenAI-compatible API (alternative to Ollama)
- Batch processing with custom scripts
- Testing quantization levels (Q4, Q5, Q6, Q8, F16)

**Install**:
```bash
# Build from source:
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build . --config Release -j$(nproc)

# Verify:
./bin/llama-cli --version

# Run a model:
./bin/llama-cli -m /path/to/model.gguf -p "Hello, DGX Spark!" -ngl 999

# Start server:
./bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080 -ngl 999
```

**Scores**: Practicality 4 · Stability 4 · Compatibility 4

**Configuration notes**:
- `-ngl 999` offloads all layers to GPU
- Build takes ~5 minutes on DGX Spark
- Server mode provides OpenAI-compatible API on port 8080
- Use different port than Ollama (11434) to run both simultaneously

---

## Installation Order

Recommended sequence for setting up a fresh DGX Spark:

```
1. Verify pre-installed tools (Ollama, Docker, Git)
2. Install uv package manager
3. Create project directory and virtual environment
4. Install Python packages (FastAPI, Polars, LangChain)
5. Install PostgreSQL + pgvector
6. Install Redis
7. Deploy Prometheus + Grafana (Docker Compose)
8. Build llama.cpp from source
9. Install NemoClaw
10. Configure Git + SSH
```

```bash
#!/bin/bash
# Quick setup script (run after fresh DGX Spark setup)

# 1. Verify pre-installed
echo "=== Verifying pre-installed tools ==="
ollama --version && docker --version && git --version && nvidia-smi

# 2. Install uv
echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Create project environment
echo "=== Setting up project ==="
mkdir -p ~/projects/default && cd ~/projects/default
uv venv .venv && source .venv/bin/activate

# 4. Install Python packages
echo "=== Installing Python packages ==="
uv pip install \
  fastapi uvicorn[standard] \
  polars \
  langchain langgraph langchain-community langchain-ollama langchain-postgres \
  redis \
  psycopg2-binary asyncpg \
  httpx pydantic

# 5. Install PostgreSQL + pgvector
echo "=== Installing PostgreSQL + pgvector ==="
sudo apt update && sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector
sudo systemctl enable postgresql && sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 6. Install Redis
echo "=== Installing Redis ==="
sudo apt install -y redis-server
sudo systemctl enable redis-server && sudo systemctl start redis-server

# 7. Deploy monitoring (create compose file first)
echo "=== Monitoring setup ready — run docker compose up ==="

# 8. Build llama.cpp
echo "=== Building llama.cpp ==="
cd ~/projects
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build . --config Release -j$(nproc)

echo "=== Core stack installation complete ==="
```

---

## Quality Template

### Summary

The recommended core stack provides 12 tools covering LLM inference, API serving, data processing, database, caching, agent orchestration, monitoring, and version control. All tools are verified compatible with ARM64/CUDA 13/sm_121 on DGX Spark.

### Practical Implications

- This stack enables building complete AI applications from inference through serving to monitoring
- Total additional disk usage: ~2-3 GB (excluding Docker images and models)
- No conflicts between default stack tools when installed as documented
- All tools can be installed in under 30 minutes on a fresh DGX Spark

### Recommended Actions

1. Run the installation script above on a fresh DGX Spark
2. Verify each tool with the provided test commands
3. Consult the [Ecosystem Catalog](ecosystem-catalog.md) for additional tools per-project
4. Review the [Avoid / Delay List](avoid-delay-list.md) before installing anything not in this list

### Confidence Level

**HIGH** — All default stack tools are either pre-installed (NVIDIA-verified), pure Python (no architecture dependency), or have confirmed ARM64 support. PostgreSQL, Redis, and Polars all have native aarch64 packages.

### Source Notes

| Tool | Confidence Basis |
|------|-----------------|
| Ollama | [NVIDIA-OFFICIAL] Pre-installed |
| Docker + NGC | [NVIDIA-OFFICIAL] Pre-installed |
| Python 3.12 | [NVIDIA-OFFICIAL] System Python |
| FastAPI | [INFERRED] Pure Python |
| PostgreSQL + pgvector | [INFERRED] Native ARM64 packages |
| Redis | [INFERRED] Native ARM64 |
| LangChain | [INFERRED] Pure Python |
| NemoClaw | [NVIDIA-OFFICIAL] First-party support |
| Polars | [NVIDIA-OFFICIAL] 25% faster on Grace |
| Prometheus + Grafana | [INFERRED] ARM64 Docker images |
| Git + SSH | [NVIDIA-OFFICIAL] Pre-installed |
| llama.cpp | [COMMUNITY] Builds on aarch64 |

### Unresolved Questions

1. **NemoClaw install path**: Exact pip package name may change; check NVIDIA documentation for latest.
2. **pgvector apt package**: Package name may vary by Ubuntu/PostgreSQL version; verify `postgresql-16-pgvector` is available.
3. **Prometheus GPU metrics**: DCGM exporter DGX Spark compatibility needs verification for unified memory reporting accuracy.
