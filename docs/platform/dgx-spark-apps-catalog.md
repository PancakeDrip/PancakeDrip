# NVIDIA DGX Spark — Applications & Playbooks Catalog

*Last Updated: March 2026*

## Overview

DGX Spark supports a growing ecosystem of AI applications via official NVIDIA playbooks (at build.nvidia.com/spark), NGC containers, and community tools. All applications must be ARM64 compatible.

### Quick Start Applications

| Application | Setup Time | Description |
|-------------|------------|-------------|
| VS Code | 5 min | Install locally or use remote development |
| DGX Dashboard | 30 min | System monitoring + JupyterLab launch |
| Open WebUI + Ollama | 15 min | Chat interface with local LLMs |
| Comfy UI | 45 min | AI image generation (FLUX.2, SDXL, Stable Diffusion) |

### Inference & Language Models

| Application | Setup Time | Description | Key Details |
|-------------|------------|-------------|-------------|
| Ollama + Open WebUI | 15 min | Chat with models via web interface | Pre-installed, GGUF format, supports all major models |
| LM Studio | 15 min | GUI for deploying and serving LLMs | ARM64 Linux build available, serves on local network (port 1234), great for S26 Ultra connection |
| NIM on Spark | 30 min | NVIDIA optimized inference microservices | Containerized, OpenAI-compatible API, requires NGC key |
| vLLM | 30 min | High-throughput inference server | Needs special cu130 aarch64 wheels, PagedAttention, 793 tok/s throughput |
| llama.cpp | 15 min | Flexible inference with GGUF models | Compile from source, fine-grained GPU layer control |
| ExLlamaV3 | 20 min | Blackwell-optimized inference | EXL3 format, tensor-parallel, native Blackwell support |

### Data Science & Machine Learning

| Application | Setup Time | Description |
|-------------|------------|-------------|
| CUDA-X Data Science | 30 min | GPU-accelerated pandas (cuDF) and scikit-learn (cuML) |
| Portfolio Optimization | 20 min | GPU portfolio optimization with cuOpt and cuML, Mean-CVaR |
| RAPIDS | Varies | Full suite: cuDF, cuML, cuGraph, cuSpatial |
| Single-cell RNA Sequencing | 15 min | GPU-powered scRNA-seq with RAPIDS |

### Generative AI

| Application | Setup Time | Description |
|-------------|------------|-------------|
| ComfyUI | 45 min | Image generation with FLUX.2, SDXL, Stable Diffusion. 128GB unified memory runs FLUX.2 at full precision (90GB). FP8 reduces VRAM 40% and improves perf 40%. |
| Text to Knowledge Graph | 30 min | Transform unstructured text into interactive knowledge graphs using LLM inference |

### AI Agents

| Application | Setup Time | Description |
|-------------|------------|-------------|
| Multi-Agent Chatbot | 1 hr | Deploy multi-agent system with multiple LLMs and VLMs |
| NemoClaw/OpenClaw | 30 min | Open-source agent development platform. Secure autonomous agent runtime with OpenShell. Policy-based privacy guardrails. Supports up to 8 concurrent tasks. |
| OpenClaw with LM Studio/Ollama | 30 min | Run OpenClaw locally with local inference backend |

### Development & Fine-tuning

| Application | Setup Time | Description |
|-------------|------------|-------------|
| VS Code (ARM64) | 5 min | Local or remote development. ARM64 .deb package. |
| JupyterLab | 10 min | Via DGX Dashboard or standalone. Remote via SSH tunnel. |
| Optimized JAX | 2 hrs | JAX framework optimized for Grace Blackwell |
| NeMo AutoModel Fine-tuning | 1 hr | Fine-tune LLMs (1B-70B) with PEFT/SFT. FP8 optimization. Docker workflow. |
| RAG Pipeline | 1 hr | Milvus vector DB + NVIDIA NIMs + Ingest. AI Workbench integration. Gradio UI. |

### Voice AI

| Application | Setup Time | Description |
|-------------|------------|-------------|
| Voice AI Pipeline | 1 hr | faster-whisper (STT, 70-90ms) + vLLM (inference) + VibeVoice (TTS, 0.48x RT). 4-second avg end-to-end latency. Fully offline/private. |

### Networking

| Application | Setup Time | Description |
|-------------|------------|-------------|
| Connect Two Sparks | 30 min | Link two DGX Sparks via ConnectX-7 QSFP cable. Combined 256GB memory. NCCL for GPU communication. Enables 405B parameter models. |

### Robotics & Other

| Application | Setup Time | Description |
|-------------|------------|-------------|
| Spark & Reachy Photo Booth | 1 hr | AI photo booth using DGX Spark and Reachy Mini robot |

## How to Access Playbooks

1. Visit [https://build.nvidia.com/spark](https://build.nvidia.com/spark) for all official playbooks
2. GitHub: [https://github.com/NVIDIA/dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
3. Each playbook includes: prerequisites, step-by-step instructions, troubleshooting

## NGC Container Quick Reference

```bash
# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-NGC-API-key>

# Key containers for DGX Spark
docker pull nvcr.io/nvidia/pytorch:25.11-py3          # PyTorch (recommended)
docker pull nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04  # CUDA development
# Always add --gpus=all when running
docker run -it --gpus=all -v /workspace:/workspace nvcr.io/nvidia/pytorch:25.11-py3
```

---
