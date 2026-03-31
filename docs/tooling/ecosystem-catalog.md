# Ecosystem Catalog — DGX Spark Expert System

> **Last Updated**: 2026-03-31
> **Status**: Phase 2 — Complete
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Baseline**: ARM64 (aarch64) · CUDA 13.0.2 · sm_121 · Ubuntu 24.04 · 128 GB Unified Memory
> **Pre-installed**: Ollama, Docker, JupyterLab, NVIDIA NIM
> **Maintainer**: DGX Spark Expert System

---

## Overview

This catalog provides a comprehensive evaluation of tools, frameworks, and libraries relevant to the DGX Spark workstation. Every entry includes **14 mandatory fields** covering purpose, compatibility, maturity, and practical scoring. The catalog spans **19 categories** plus a cross-cutting financial/research section.

### How to Read This Catalog

**Scores** (1–5):
- **Practicality Score**: How useful this tool is for real DGX Spark workflows (5 = essential, 1 = barely usable)
- **Stability Score**: How reliable the tool is in its current state (5 = rock-solid, 1 = frequent breakage)
- **Compatibility Score**: How well it works on ARM64 / CUDA 13 / sm_121 (5 = native, 1 = broken)

**Stack Classification**:
- **Default Stack**: Recommended for most DGX Spark users; install immediately
- **Optional Stack**: Valuable for specific use cases; install when needed
- **Avoid-Delay**: Broken, untested, or poor fit for DGX Spark; wait for upstream fixes

**Maturity Levels**: Production · Stable · Beta · Alpha · Experimental

**Install Complexity**: Trivial · Easy · Moderate · Hard · Build-from-source

**Confidence Tags**:
- `[NVIDIA-OFFICIAL]` — confirmed by NVIDIA documentation or pre-installed
- `[COMMUNITY]` — tested by community members, not officially supported
- `[INFERRED]` — compatibility inferred from architecture support but not DGX Spark-specific tested

---

## Table of Contents

1. [NVIDIA-Native Tools](#1-nvidia-native-tools)
2. [LLM Tooling / Inference Engines](#2-llm-tooling--inference-engines)
3. [Local Inference Tools](#3-local-inference-tools)
4. [Model Serving Tools](#4-model-serving-tools)
5. [Agent Frameworks](#5-agent-frameworks)
6. [Python ML Stacks](#6-python-ml-stacks)
7. [Data Stacks (ETL, Warehousing, Analytics)](#7-data-stacks-etl-warehousing-analytics)
8. [Android Development Tools](#8-android-development-tools)
9. [Desktop App Frameworks](#9-desktop-app-frameworks)
10. [Web App Frameworks](#10-web-app-frameworks)
11. [Container Tools](#11-container-tools)
12. [Observability Tools](#12-observability-tools)
13. [Automation Tools](#13-automation-tools)
14. [Orchestration Tools](#14-orchestration-tools)
15. [Databases](#15-databases)
16. [Vector Databases](#16-vector-databases)
17. [Messaging / Queues](#17-messaging--queues)
18. [API Frameworks](#18-api-frameworks)
19. [GPU-Aware Dev Tools & Debuggers](#19-gpu-aware-dev-tools--debuggers)
20. [Cross-Cutting: Financial / Research Libraries](#20-cross-cutting-financial--research-libraries)
21. [Quality Template](#21-quality-template)

---

## 1. NVIDIA-Native Tools

First-party NVIDIA tools with DGX Spark support. These form the backbone of GPU-accelerated workflows on the platform.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| NVIDIA NIM | Production | Pre-installed / NGC | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| NVIDIA NeMo | Production | NGC Container | 4 | 4 | 4 | Optional | [NVIDIA-OFFICIAL] |
| NemoClaw | Stable | GitHub + pip | 4 | 4 | 5 | Default | [NVIDIA-OFFICIAL] |
| TensorRT | Production | NGC Container | 5 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| Triton Inference Server | Production | NGC Container | 5 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| RAPIDS cuDF | Stable | NGC / conda | 5 | 4 | 5 | Default | [NVIDIA-OFFICIAL] |
| RAPIDS cuML | Stable | NGC / conda | 4 | 4 | 5 | Optional | [NVIDIA-OFFICIAL] |
| cuOpt | Stable | NGC Container | 3 | 4 | 4 | Optional | [NVIDIA-OFFICIAL] |
| Nsight Systems | Production | Pre-installed | 4 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| Nsight Compute | Production | Pre-installed | 4 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| NVIDIA TAO Toolkit | Stable | NGC Container | 3 | 4 | 4 | Optional | [NVIDIA-OFFICIAL] |

### NVIDIA NIM

1. **Purpose**: Optimized inference microservice for deploying LLMs and AI models with minimal configuration.
2. **Why It Matters**: Provides production-grade inference with automatic optimization (TensorRT-LLM, quantization) without manual tuning.
3. **DGX Spark Relevance**: Pre-installed on DGX Spark; primary recommended path for production model serving.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 builds
   - CUDA 13: ✅ Built for CUDA 13.0.2
   - sm_121: ✅ Blackwell-native kernels
   - Container: ✅ NGC container is primary delivery mechanism
   - OS: ✅ Ubuntu 24.04 supported
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — simple for basic deployment, advanced for custom model integration
7. **Best Use Cases**: Production LLM serving, API-compatible inference endpoints, multi-model deployments
8. **Bad Use Cases**: Quick local experimentation (Ollama is faster to iterate), non-NVIDIA hardware
9. **Install Complexity**: Trivial (pre-installed) / Easy (NGC pull for updated versions)
10. **Dependencies & Likely Conflicts**: Requires NVIDIA Container Toolkit; no known conflicts on DGX Spark
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### NVIDIA NeMo

1. **Purpose**: End-to-end framework for training, fine-tuning, and deploying generative AI models (LLMs, speech, multimodal).
2. **Why It Matters**: Full pipeline from data preparation through training to deployment; integrates with NIM for serving.
3. **DGX Spark Relevance**: Enables fine-tuning LLMs on the 128 GB unified memory; ideal for LoRA/QLoRA on 7B–70B models.
4. **Architecture Compatibility**:
   - ARM64: ✅ Via NGC container
   - CUDA 13: ✅ NGC containers built for CUDA 13
   - sm_121: ✅ Supported via container
   - Container: ✅ Primary delivery mechanism (nvcr.io/nvidia/nemo)
   - OS: ✅ Ubuntu 24.04 base in container
5. **Maturity Level**: Production
6. **Learning Curve**: High — complex configuration, many components
7. **Best Use Cases**: LLM fine-tuning (LoRA, P-tuning), speech model training, multimodal pipelines
8. **Bad Use Cases**: Simple inference-only workflows (use Ollama/NIM), small-scale prototyping
9. **Install Complexity**: Moderate (NGC container pull + model download)
10. **Dependencies & Likely Conflicts**: Large container (~20 GB); needs significant disk space; NeMo Curator for data prep is separate
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

### NemoClaw

1. **Purpose**: NVIDIA's agent framework providing tool-calling, guardrails, and sandboxed execution for LLM agents.
2. **Why It Matters**: First-party agent framework with built-in security sandboxing; designed for enterprise-grade agent deployments.
3. **DGX Spark Relevance**: Explicitly built with DGX Spark support; integrates with NIM for inference backend.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python + gRPC; native aarch64
   - CUDA 13: ✅ Delegates GPU work to NIM
   - sm_121: ✅ Via NIM backend
   - Container: ✅ Can run containerized or native
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — YAML-driven configuration, familiar patterns for agent developers
7. **Best Use Cases**: Secure tool-calling agents, enterprise chatbots with guardrails, multi-step agent workflows
8. **Bad Use Cases**: Simple one-shot LLM queries (overkill), non-NVIDIA inference backends
9. **Install Complexity**: Easy (pip install from NVIDIA GitHub)
10. **Dependencies & Likely Conflicts**: Requires running NIM or Ollama backend; gRPC dependencies are well-maintained on aarch64
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### TensorRT

1. **Purpose**: High-performance deep learning inference optimizer and runtime.
2. **Why It Matters**: Delivers 2-6× inference speedups through layer fusion, precision calibration, and kernel auto-tuning.
3. **DGX Spark Relevance**: TensorRT-LLM powers NIM's inference; standalone TensorRT useful for custom vision/audio models.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 builds in NGC
   - CUDA 13: ✅ Built for CUDA 13.0.2
   - sm_121: ✅ Blackwell-optimized kernels
   - Container: ✅ NGC container (nvcr.io/nvidia/tensorrt)
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — requires understanding of model optimization, precision modes, and engine building
7. **Best Use Cases**: Optimizing inference for production models, vision models, custom model optimization
8. **Bad Use Cases**: Rapid prototyping, models that change frequently (engine rebuild required)
9. **Install Complexity**: Moderate (NGC container) / Hard (native install)
10. **Dependencies & Likely Conflicts**: cuDNN required; version must match CUDA 13; use NGC container to avoid conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Triton Inference Server

1. **Purpose**: Multi-framework model serving platform supporting dynamic batching, model ensembles, and concurrent model execution.
2. **Why It Matters**: Unified serving for mixed model types (PyTorch, TensorRT, ONNX) with production features (health checks, metrics, A/B testing).
3. **DGX Spark Relevance**: Ideal for multi-model pipelines; manages GPU memory efficiently across models on the shared 128 GB.
4. **Architecture Compatibility**:
   - ARM64: ✅ NGC container for aarch64
   - CUDA 13: ✅ Built for CUDA 13
   - sm_121: ✅ Blackwell-aware scheduling
   - Container: ✅ Primary delivery (nvcr.io/nvidia/tritonserver)
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — model repository structure, configuration files, client libraries
7. **Best Use Cases**: Multi-model serving, ensemble pipelines, production deployments needing dynamic batching
8. **Bad Use Cases**: Single-model simple inference (use Ollama or NIM), quick prototyping
9. **Install Complexity**: Moderate (NGC pull + model repository setup)
10. **Dependencies & Likely Conflicts**: Heavy container; needs model repository directory structure; port conflicts possible with NIM
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### RAPIDS cuDF

1. **Purpose**: GPU-accelerated DataFrame library, API-compatible with pandas.
2. **Why It Matters**: 10–100× faster than pandas for data manipulation on GPU; drop-in replacement for most pandas code.
3. **DGX Spark Relevance**: First-party NVIDIA support for DGX Spark; leverages the Blackwell GPU for data processing alongside ML workloads.
4. **Architecture Compatibility**:
   - ARM64: ✅ Available for DGX Spark
   - CUDA 13: ✅ RAPIDS builds for CUDA 13
   - sm_121: ✅ Blackwell GPU acceleration
   - Container: ✅ NGC RAPIDS container
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — pandas-compatible API; cudf.pandas accelerator mode is zero-code-change
7. **Best Use Cases**: Large DataFrame operations (>100K rows), ETL pipelines, feature engineering for ML
8. **Bad Use Cases**: Small data (<10K rows, overhead exceeds benefit), operations requiring obscure pandas APIs not yet ported
9. **Install Complexity**: Moderate (NGC container recommended) / Hard (native pip — dependency resolution)
10. **Dependencies & Likely Conflicts**: CUDA toolkit version must match; UCX for multi-GPU (not relevant on single-GPU Spark); may conflict with native pandas if imported incorrectly
11. **Practicality Score**: 5
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### RAPIDS cuML

1. **Purpose**: GPU-accelerated machine learning library with scikit-learn-compatible API.
2. **Why It Matters**: Accelerates classic ML algorithms (random forests, k-means, PCA, UMAP) by 10–50× on GPU.
3. **DGX Spark Relevance**: Available for DGX Spark; pairs with cuDF for end-to-end GPU-accelerated ML pipelines.
4. **Architecture Compatibility**:
   - ARM64: ✅ Available for DGX Spark
   - CUDA 13: ✅ RAPIDS builds for CUDA 13
   - sm_121: ✅ Blackwell GPU acceleration
   - Container: ✅ NGC RAPIDS container
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — scikit-learn-compatible API
7. **Best Use Cases**: Training classic ML models at scale, hyperparameter search, UMAP/t-SNE visualization on large datasets
8. **Bad Use Cases**: Deep learning (use PyTorch), very small datasets, algorithms not yet ported
9. **Install Complexity**: Moderate (NGC container) / Hard (native)
10. **Dependencies & Likely Conflicts**: Same dependency chain as cuDF; use NGC RAPIDS container for both
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### cuOpt

1. **Purpose**: GPU-accelerated solver for vehicle routing, logistics optimization, and combinatorial optimization problems.
2. **Why It Matters**: Solves complex routing/scheduling problems orders of magnitude faster than CPU solvers.
3. **DGX Spark Relevance**: Niche but powerful for logistics, fleet management, and supply chain optimization workloads.
4. **Architecture Compatibility**:
   - ARM64: ✅ NGC container available
   - CUDA 13: ✅ NGC container built for CUDA 13
   - sm_121: ✅ Via container
   - Container: ✅ NGC container delivery
   - OS: ✅ Ubuntu 24.04 via container
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — domain-specific API, requires understanding of optimization problem formulation
7. **Best Use Cases**: Vehicle routing problems, delivery optimization, warehouse layout optimization
8. **Bad Use Cases**: General-purpose optimization (use SciPy), problems without routing/scheduling structure
9. **Install Complexity**: Moderate (NGC container + API key)
10. **Dependencies & Likely Conflicts**: Self-contained in NGC container; no conflicts expected
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

### Nsight Systems

1. **Purpose**: System-wide performance analysis tool for GPU-accelerated applications.
2. **Why It Matters**: Identifies performance bottlenecks across CPU, GPU, memory, and I/O with timeline visualization.
3. **DGX Spark Relevance**: Pre-installed; essential for profiling unified memory access patterns on the shared 128 GB architecture.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 build
   - CUDA 13: ✅ Full support
   - sm_121: ✅ Blackwell GPU profiling
   - Container: ✅ Works inside and outside containers
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — CLI profiling is straightforward; full GUI analysis requires learning the timeline viewer
7. **Best Use Cases**: End-to-end application profiling, identifying CPU-GPU synchronization issues, memory transfer analysis
8. **Bad Use Cases**: Kernel-level micro-optimization (use Nsight Compute), quick performance checks (use nvidia-smi)
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: None on DGX Spark
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Nsight Compute

1. **Purpose**: Interactive CUDA kernel profiler for detailed GPU kernel analysis.
2. **Why It Matters**: Provides per-kernel metrics including occupancy, memory throughput, instruction mix, and roofline analysis.
3. **DGX Spark Relevance**: Pre-installed; critical for optimizing custom CUDA kernels targeting sm_121 Blackwell architecture.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 build
   - CUDA 13: ✅ Full support
   - sm_121: ✅ Blackwell kernel profiling with new metrics
   - Container: ✅ Works inside containers with --privileged
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — requires deep understanding of GPU architecture and performance metrics
7. **Best Use Cases**: CUDA kernel optimization, roofline analysis, occupancy tuning, comparing kernel implementations
8. **Bad Use Cases**: Application-level profiling (use Nsight Systems), Python-only workflows
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: Kernel profiling may require elevated permissions; can slow target application significantly during profiling
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### NVIDIA TAO Toolkit

1. **Purpose**: Transfer learning toolkit for fine-tuning NVIDIA pre-trained models (vision, speech, NLP) with minimal data.
2. **Why It Matters**: Dramatically reduces training time and data requirements for domain-specific vision and speech models.
3. **DGX Spark Relevance**: NGC container available; useful for fine-tuning detection/classification models on Blackwell GPU.
4. **Architecture Compatibility**:
   - ARM64: ✅ NGC container for aarch64
   - CUDA 13: ✅ Via NGC container
   - sm_121: ✅ Via container
   - Container: ✅ NGC container delivery
   - OS: ✅ Ubuntu 24.04 via container
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — YAML-driven specs, but domain knowledge needed for model selection
7. **Best Use Cases**: Object detection fine-tuning, classification transfer learning, speech model adaptation
8. **Bad Use Cases**: LLM fine-tuning (use NeMo), training from scratch, unsupported model architectures
9. **Install Complexity**: Moderate (NGC container + TAO launcher setup)
10. **Dependencies & Likely Conflicts**: Large container; needs NGC API key; TAO launcher manages containers internally
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 2. LLM Tooling / Inference Engines

Tools for running large language models locally on DGX Spark.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Ollama | Production | Pre-installed | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| vLLM | Stable | dgx-spark-vllm / NGC / source | 5 | 4 | 3 | Optional | [COMMUNITY] |
| llama.cpp | Stable | Build from source | 4 | 4 | 4 | Default | [COMMUNITY] |
| TGI (text-generation-inference) | Stable | NGC Container | 4 | 4 | 3 | Optional | [COMMUNITY] |

### Ollama

1. **Purpose**: Simple local LLM runner with model management, automatic GPU offloading, and OpenAI-compatible API.
2. **Why It Matters**: Lowest-friction path to running LLMs locally; handles model downloading, quantization selection, and GPU memory management.
3. **DGX Spark Relevance**: Pre-installed on DGX Spark; NVIDIA's recommended default inference tool; most reliable and tested option.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 build, pre-installed
   - CUDA 13: ✅ Built for CUDA 13.0.2
   - sm_121: ✅ Blackwell-optimized
   - Container: ✅ Also available as container, but native is preferred
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — `ollama run llama3` is all you need to start
7. **Best Use Cases**: Local development, prototyping, running GGUF models, multi-model experimentation (LRU cache), OpenAI-compatible API endpoint
8. **Bad Use Cases**: Maximum throughput production serving (vLLM/NIM better), custom model architectures not in Ollama registry
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: Binds to port 11434 by default; may conflict if running NIM API on same port
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### vLLM

1. **Purpose**: High-throughput LLM serving engine with PagedAttention, continuous batching, and OpenAI-compatible API.
2. **Why It Matters**: Highest throughput open-source inference engine; 2–4× higher throughput than naive serving for concurrent requests.
3. **DGX Spark Relevance**: Requires special installation via `dgx-spark-vllm` PyPI package, NGC container, or source build. Not plug-and-play.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ No standard pip wheel; needs dgx-spark-vllm or source build
   - CUDA 13: ⚠️ Requires cu130-compatible build
   - sm_121: ⚠️ Needs TORCH_CUDA_ARCH_LIST=12.1a for source build
   - Container: ✅ NGC vLLM container available
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — more configuration than Ollama, but well-documented
7. **Best Use Cases**: High-throughput serving with many concurrent users, benchmarking inference performance, production API with continuous batching
8. **Bad Use Cases**: Quick local testing (Ollama is simpler), GGUF-only models (use llama.cpp)
9. **Install Complexity**: Hard (dgx-spark-vllm package or NGC container) / Build-from-source (native)
10. **Dependencies & Likely Conflicts**: PyTorch must be sm_121-compatible; Flash Attention may be broken (use PyTorch SDPA fallback); xformers optional
11. **Practicality Score**: 5
12. **Stability Score**: 4
13. **Compatibility Score**: 3
14. **Stack Classification**: Optional

### llama.cpp

1. **Purpose**: Lightweight C++ LLM inference engine optimized for GGUF quantized models with CUDA acceleration.
2. **Why It Matters**: Most efficient engine for quantized models; powers Ollama's backend; supports fine-grained control over GPU layers and context.
3. **DGX Spark Relevance**: Builds cleanly from source on aarch64 with CUDA 13; provides GGUF model support and server mode.
4. **Architecture Compatibility**:
   - ARM64: ✅ Builds natively with cmake on aarch64
   - CUDA 13: ✅ CUDA backend compiles with CUDA 13
   - sm_121: ✅ Set CUDA_ARCHITECTURES=121 during build
   - Container: ✅ Can containerize, but native build works well
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — build from source, CLI-driven; server mode mimics OpenAI API
7. **Best Use Cases**: Running GGUF models with precise layer control, embedding generation, batch processing, secondary inference alongside Ollama
8. **Bad Use Cases**: Non-GGUF models (need conversion), users wanting GUI model management
9. **Install Complexity**: Build-from-source (cmake + CUDA toolkit, ~5 min build)
10. **Dependencies & Likely Conflicts**: cmake, CUDA toolkit; no Python dependency conflicts; may conflict with Ollama if both bind same port
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Default

### Text Generation Inference (TGI)

1. **Purpose**: Hugging Face's production-grade LLM serving solution with quantization, batching, and token streaming.
2. **Why It Matters**: Tight Hugging Face Hub integration; supports most HF model formats natively.
3. **DGX Spark Relevance**: Available via NGC container; less tested than Ollama/vLLM on DGX Spark but functional.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ Container build required; no native aarch64 pip package
   - CUDA 13: ⚠️ NGC container may need CUDA 13 rebuild
   - sm_121: ⚠️ Depends on PyTorch sm_121 compatibility in container
   - Container: ✅ Primary delivery mechanism
   - OS: ✅ Ubuntu 24.04 via container
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — Docker-based deployment, model selection via HF Hub IDs
7. **Best Use Cases**: Serving HF models in production, streaming token generation, HF ecosystem integration
8. **Bad Use Cases**: GGUF models (use llama.cpp/Ollama), when Ollama or NIM already meets needs
9. **Install Complexity**: Moderate (NGC/Docker container)
10. **Dependencies & Likely Conflicts**: Heavy container; potential port conflicts with other inference servers
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 3
14. **Stack Classification**: Optional

---

## 3. Local Inference Tools

Focused on running inference locally (vs. served). Overlaps with LLM Tooling, scoped to local-first usage patterns.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Ollama (local) | Production | Pre-installed | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| llama.cpp (local) | Stable | Build from source | 4 | 4 | 4 | Default | [COMMUNITY] |
| vLLM (local) | Stable | dgx-spark-vllm / source | 4 | 4 | 3 | Optional | [COMMUNITY] |
| NIM (local) | Production | Pre-installed / NGC | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |

### Ollama (local mode)

1. **Purpose**: Local LLM runner with model management and LRU caching for multi-model switching.
2. **Why It Matters**: Start experimenting in seconds; automatic model download and GPU memory management.
3. **DGX Spark Relevance**: Primary recommended tool for local inference. Pre-installed. Handles 128 GB unified memory well with LRU model eviction.
4. **Architecture Compatibility**: ARM64 ✅ | CUDA 13 ✅ | sm_121 ✅ | Container ✅ | OS ✅
5. **Maturity Level**: Production
6. **Learning Curve**: Low
7. **Best Use Cases**: Interactive chat, rapid model evaluation, development iteration, multi-model A/B testing
8. **Bad Use Cases**: Batch inference at scale, custom serving infrastructure needs
9. **Install Complexity**: Trivial
10. **Dependencies & Likely Conflicts**: None on DGX Spark
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### llama.cpp (local mode)

1. **Purpose**: Direct GGUF model execution with fine-grained control over GPU layer offloading and context length.
2. **Why It Matters**: More control than Ollama; can split layers between CPU and GPU; useful for models that exceed GPU memory.
3. **DGX Spark Relevance**: Builds from source on aarch64; valuable as secondary engine for GGUF experimentation.
4. **Architecture Compatibility**: ARM64 ✅ | CUDA 13 ✅ | sm_121 ✅ | Container ✅ | OS ✅
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium
7. **Best Use Cases**: GGUF quantization testing, partial GPU offloading experiments, embedding generation
8. **Bad Use Cases**: When Ollama already handles the model fine
9. **Install Complexity**: Build-from-source
10. **Dependencies & Likely Conflicts**: cmake, g++, CUDA toolkit
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Default

### vLLM (local mode)

1. **Purpose**: High-throughput local inference with PagedAttention for efficient memory utilization.
2. **Why It Matters**: Best throughput for concurrent local workloads; efficient KV-cache management.
3. **DGX Spark Relevance**: Requires dgx-spark-vllm package or source build; more setup than Ollama but higher throughput.
4. **Architecture Compatibility**: ARM64 ⚠️ | CUDA 13 ⚠️ | sm_121 ⚠️ | Container ✅ | OS ✅
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium
7. **Best Use Cases**: Throughput benchmarking, concurrent request testing, HF model serving locally
8. **Bad Use Cases**: Quick single-prompt testing, GGUF models
9. **Install Complexity**: Hard
10. **Dependencies & Likely Conflicts**: PyTorch sm_121 compatibility; potential Flash Attention issues
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 3
14. **Stack Classification**: Optional

### NIM (local mode)

1. **Purpose**: Locally deployed NIM containers for optimized inference with automatic TensorRT-LLM optimization.
2. **Why It Matters**: Production-quality inference with one container pull; automatic quantization and optimization.
3. **DGX Spark Relevance**: Pre-installed; NVIDIA's production inference recommendation for DGX Spark.
4. **Architecture Compatibility**: ARM64 ✅ | CUDA 13 ✅ | sm_121 ✅ | Container ✅ | OS ✅
5. **Maturity Level**: Production
6. **Learning Curve**: Medium
7. **Best Use Cases**: Production-grade local inference, model optimization without manual tuning, enterprise deployments
8. **Bad Use Cases**: Quick prototyping (heavier startup than Ollama), models not in NIM catalog
9. **Install Complexity**: Easy (NGC pull)
10. **Dependencies & Likely Conflicts**: NVIDIA Container Toolkit; disk space for containers
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

---

## 4. Model Serving Tools

Tools specifically for serving models over network APIs.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Triton Inference Server | Production | NGC Container | 5 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| vLLM (served) | Stable | dgx-spark-vllm / NGC | 5 | 4 | 3 | Optional | [COMMUNITY] |
| NVIDIA NIM | Production | Pre-installed / NGC | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| BentoML | Stable | pip install | 4 | 4 | 4 | Optional | [INFERRED] |
| FastAPI-wrapped | Production | pip install | 4 | 5 | 5 | Default | [INFERRED] |

### Triton Inference Server (served)

See [Section 1: Triton Inference Server](#triton-inference-server) for full 14-field entry. In serving context, Triton excels at multi-model concurrent serving with dynamic batching.

### vLLM (served)

See [Section 2: vLLM](#vllm) for full 14-field entry. In serving context, vLLM provides highest throughput OpenAI-compatible API for concurrent users.

### NVIDIA NIM (served)

See [Section 1: NVIDIA NIM](#nvidia-nim) for full 14-field entry. In serving context, NIM provides production-optimized serving with automatic model optimization.

### BentoML

1. **Purpose**: ML model serving framework with model packaging, versioning, and deployment orchestration.
2. **Why It Matters**: Standardizes model packaging into "Bentos" with dependencies; supports multi-model serving and A/B testing.
3. **DGX Spark Relevance**: Pure Python framework; works on aarch64 for CPU models; GPU model serving needs CUDA-compatible dependencies.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python core
   - CUDA 13: ⚠️ Depends on model framework compatibility
   - sm_121: ⚠️ Depends on underlying model framework
   - Container: ✅ Docker-based deployment supported
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — Bento packaging concepts, service decorators, deployment configuration
7. **Best Use Cases**: Multi-model serving with versioning, A/B testing models, custom pre/post-processing pipelines
8. **Bad Use Cases**: Simple single-model serving (FastAPI is simpler), LLM-only serving (NIM/vLLM better)
9. **Install Complexity**: Easy (pip install bentoml)
10. **Dependencies & Likely Conflicts**: Minimal core dependencies; GPU dependencies inherited from model framework
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

### FastAPI-wrapped Models

1. **Purpose**: Custom model serving using FastAPI with direct model loading and custom inference endpoints.
2. **Why It Matters**: Maximum flexibility; any model, any pre/post-processing, any API schema.
3. **DGX Spark Relevance**: Pure Python; works perfectly on aarch64; commonly used to wrap Ollama or custom models with business logic.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency in FastAPI itself
   - sm_121: ✅ Model framework handles GPU
   - Container: ✅ Easy to containerize
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — standard Python web development
7. **Best Use Cases**: Custom inference APIs, wrapping models with business logic, proxy/gateway for Ollama/NIM, multi-step pipelines
8. **Bad Use Cases**: When NIM/Triton features (batching, optimization) are needed; reinventing existing serving infrastructure
9. **Install Complexity**: Trivial (pip install fastapi uvicorn)
10. **Dependencies & Likely Conflicts**: None
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

---

## 5. Agent Frameworks

Frameworks for building LLM-powered autonomous agents.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| NemoClaw | Stable | GitHub + pip | 4 | 4 | 5 | Default | [NVIDIA-OFFICIAL] |
| LangChain / LangGraph | Stable | pip install | 4 | 4 | 5 | Default | [INFERRED] |
| CrewAI | Stable | pip install | 4 | 3 | 5 | Optional | [INFERRED] |
| AutoGen | Stable | pip install | 4 | 3 | 5 | Optional | [INFERRED] |
| Semantic Kernel | Stable | pip install | 3 | 4 | 5 | Optional | [INFERRED] |

### NemoClaw

See [Section 1: NemoClaw](#nemoclaw) for full 14-field entry. Primary agent framework recommendation for DGX Spark due to first-party NVIDIA support and NIM integration.

### LangChain / LangGraph

1. **Purpose**: LangChain provides composable chains for LLM applications; LangGraph adds stateful, graph-based agent orchestration.
2. **Why It Matters**: Most popular LLM application framework; massive ecosystem of integrations (vector stores, tools, memory).
3. **DGX Spark Relevance**: Pure Python; no architecture dependency; integrates with Ollama and NIM as LLM backends.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python, no native extensions
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ No GPU code
   - Container: ✅ Works in any environment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — many abstractions and patterns; LangGraph requires understanding of graph-based state machines
7. **Best Use Cases**: RAG pipelines, multi-step agent workflows, tool-calling agents, complex chains with branching logic
8. **Bad Use Cases**: Simple single-prompt applications (overkill), when NemoClaw's security sandbox is needed
9. **Install Complexity**: Easy (pip install langchain langgraph langchain-community)
10. **Dependencies & Likely Conflicts**: Many optional dependencies; langchain-community may pull in packages with native extensions — install selectively
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### CrewAI

1. **Purpose**: Role-based multi-agent framework where agents have defined roles, goals, and backstories.
2. **Why It Matters**: Intuitive mental model for multi-agent systems; agents collaborate on tasks with natural role definitions.
3. **DGX Spark Relevance**: Pure Python; works with any OpenAI-compatible endpoint (Ollama, NIM).
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ No GPU code
   - Container: ✅ Works anywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — intuitive agent/task/crew abstractions
7. **Best Use Cases**: Role-based multi-agent workflows, content generation pipelines, research automation
8. **Bad Use Cases**: Complex graph-based workflows (LangGraph better), enterprise security requirements (NemoClaw better)
9. **Install Complexity**: Easy (pip install crewai)
10. **Dependencies & Likely Conflicts**: Pulls in langchain as dependency; may conflict with direct langchain version pins
11. **Practicality Score**: 4
12. **Stability Score**: 3
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### AutoGen

1. **Purpose**: Microsoft's multi-agent conversation framework for complex, multi-turn agent interactions.
2. **Why It Matters**: Sophisticated multi-agent orchestration with conversation patterns, code execution, and human-in-the-loop.
3. **DGX Spark Relevance**: Pure Python; connects to Ollama or NIM via OpenAI-compatible API.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ No GPU code
   - Container: ✅ Works anywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — conversation patterns and agent configuration can be complex
7. **Best Use Cases**: Multi-agent debates, code generation and execution pipelines, complex problem-solving with specialist agents
8. **Bad Use Cases**: Simple single-agent tasks, production systems needing NVIDIA-native security (NemoClaw better)
9. **Install Complexity**: Easy (pip install autogen-agentchat)
10. **Dependencies & Likely Conflicts**: Optional Docker dependency for code execution; minimal conflicts
11. **Practicality Score**: 4
12. **Stability Score**: 3
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Semantic Kernel

1. **Purpose**: Microsoft's SDK for integrating LLMs into applications with plugins, planners, and memory.
2. **Why It Matters**: Enterprise-grade framework with .NET and Python SDKs; strong Azure/OpenAI integration.
3. **DGX Spark Relevance**: Pure Python SDK; can connect to Ollama/NIM; useful for teams with .NET/Microsoft ecosystem integration needs.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python SDK
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ No GPU code
   - Container: ✅ Works anywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — plugin and planner abstractions; enterprise patterns
7. **Best Use Cases**: Enterprise applications with .NET integration, Azure ecosystem, structured plugin-based architectures
8. **Bad Use Cases**: Quick prototyping (LangChain simpler), NVIDIA-native workflows (NemoClaw better)
9. **Install Complexity**: Easy (pip install semantic-kernel)
10. **Dependencies & Likely Conflicts**: Minimal; optional Azure SDK dependencies
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 6. Python ML Stacks

Core Python machine learning frameworks and their DGX Spark compatibility.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| PyTorch | Production | pip (sm_120 compat) | 5 | 5 | 4 | Default | [COMMUNITY] |
| JAX | Production | pip / source | 4 | 4 | 3 | Optional | [COMMUNITY] |
| scikit-learn | Production | pip install | 5 | 5 | 5 | Default | [INFERRED] |
| HuggingFace Transformers | Production | pip install | 5 | 5 | 5 | Default | [INFERRED] |
| HuggingFace PEFT | Stable | pip install | 4 | 4 | 4 | Optional | [COMMUNITY] |

### PyTorch

1. **Purpose**: Primary deep learning framework for training and inference of neural networks.
2. **Why It Matters**: De facto standard for ML research and production; largest ecosystem of models and tools.
3. **DGX Spark Relevance**: sm_120 wheels are binary-compatible with sm_121 Blackwell GPU. Most models work without modification.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ sm_120 aarch64 wheels available; binary-compatible with sm_121
   - CUDA 13: ⚠️ Needs cu130 wheels or NGC container
   - sm_121: ⚠️ sm_120 binary compatible; for optimal performance, build from source with TORCH_CUDA_ARCH_LIST=12.1a
   - Container: ✅ NGC PyTorch container is best path
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — familiar Python API, but ecosystem is vast
7. **Best Use Cases**: Model training, fine-tuning, custom model architectures, research
8. **Bad Use Cases**: Simple inference of pre-packaged models (Ollama simpler), classical ML (scikit-learn better)
9. **Install Complexity**: Moderate (pip with cu130 index) / Easy (NGC container)
10. **Dependencies & Likely Conflicts**: CUDA version sensitivity; Flash Attention broken on sm_121 (use torch.nn.functional.scaled_dot_product_attention + cuDNN 9.13 backend); torchvision/torchaudio need matching versions
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 4
14. **Stack Classification**: Default

### JAX

1. **Purpose**: Google's high-performance numerical computing library with automatic differentiation and XLA compilation.
2. **Why It Matters**: Functional programming model with JIT compilation; strong for research and scientific computing.
3. **DGX Spark Relevance**: ARM64 + CUDA 13 support is less mature than PyTorch; NGC container is recommended path.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ jaxlib aarch64 builds are less frequent
   - CUDA 13: ⚠️ Needs cu130-compatible jaxlib
   - sm_121: ⚠️ XLA may need updates for sm_121
   - Container: ✅ NGC JAX container available
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — functional programming paradigm, pytree transformations, vmap/pmap
7. **Best Use Cases**: Scientific computing, large-scale model training with TPU portability, research requiring composable transformations
8. **Bad Use Cases**: Quick prototyping (PyTorch easier), teams unfamiliar with functional programming
9. **Install Complexity**: Hard (native aarch64) / Moderate (NGC container)
10. **Dependencies & Likely Conflicts**: jaxlib version must match CUDA exactly; potential conflicts with PyTorch CUDA runtime
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 3
14. **Stack Classification**: Optional

### scikit-learn

1. **Purpose**: Standard library for classical machine learning (classification, regression, clustering, preprocessing).
2. **Why It Matters**: Gold standard for non-deep-learning ML; extensive algorithm library; great documentation.
3. **DGX Spark Relevance**: Pure Python with C extensions; builds cleanly on aarch64; pairs with cuML for GPU acceleration.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 wheels available
   - CUDA 13: ✅ No CUDA dependency (CPU-only)
   - sm_121: ✅ N/A (CPU library; use cuML for GPU)
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — consistent API, excellent documentation
7. **Best Use Cases**: Classical ML, data preprocessing, feature engineering, model evaluation, baseline models
8. **Bad Use Cases**: Deep learning (use PyTorch), GPU-accelerated ML (use cuML), large-scale training
9. **Install Complexity**: Trivial (pip install scikit-learn)
10. **Dependencies & Likely Conflicts**: numpy, scipy — well-maintained on aarch64
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### HuggingFace Transformers

1. **Purpose**: Unified API for thousands of pre-trained models (NLP, vision, audio, multimodal).
2. **Why It Matters**: Largest model hub; standard interface for model loading, tokenization, and inference.
3. **DGX Spark Relevance**: Pure Python with PyTorch/JAX backends; model loading works on aarch64; inference requires compatible PyTorch.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python; uses PyTorch as backend
   - CUDA 13: ✅ Via PyTorch backend
   - sm_121: ✅ Via PyTorch backend (sm_120 compat)
   - Container: ✅ Works in any Python environment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — `from_pretrained()` pattern is consistent across model types
7. **Best Use Cases**: Model loading and inference, transfer learning, tokenization, model evaluation
8. **Bad Use Cases**: Optimized production serving (use NIM/Triton), simple LLM chat (use Ollama)
9. **Install Complexity**: Easy (pip install transformers)
10. **Dependencies & Likely Conflicts**: tokenizers package has Rust extension — aarch64 wheel available; accelerate recommended for GPU
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### HuggingFace PEFT

1. **Purpose**: Parameter-Efficient Fine-Tuning library (LoRA, QLoRA, prefix tuning, adapters).
2. **Why It Matters**: Fine-tune large models with minimal GPU memory; LoRA enables fine-tuning 70B models on DGX Spark's 128 GB.
3. **DGX Spark Relevance**: Pure Python with PyTorch backend; LoRA/QLoRA is the recommended fine-tuning approach on DGX Spark.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ⚠️ Via PyTorch; bitsandbytes quantization needs testing on sm_121
   - sm_121: ⚠️ bitsandbytes may need source build for sm_121
   - Container: ✅ Works in NGC PyTorch container
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — LoRA concepts + training loop configuration
7. **Best Use Cases**: LoRA/QLoRA fine-tuning, adapter-based model customization, memory-efficient training
8. **Bad Use Cases**: Full fine-tuning (overkill), inference-only workflows
9. **Install Complexity**: Easy (pip install peft) / Moderate (with bitsandbytes for QLoRA)
10. **Dependencies & Likely Conflicts**: bitsandbytes aarch64/sm_121 compatibility uncertain; accelerate version must match transformers
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 7. Data Stacks (ETL, Warehousing, Analytics)

Tools for data processing, transformation, and analytics.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Polars | Production | pip install | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| pandas + cuDF | Production | pip + NGC | 5 | 4 | 5 | Default | [NVIDIA-OFFICIAL] |
| DuckDB | Production | pip install | 5 | 5 | 5 | Optional | [INFERRED] |
| dbt (dbt-core) | Production | pip install | 4 | 5 | 5 | Optional | [INFERRED] |
| Great Expectations | Production | pip install | 4 | 4 | 5 | Optional | [INFERRED] |

### Polars

1. **Purpose**: Lightning-fast DataFrame library written in Rust with lazy evaluation and query optimization.
2. **Why It Matters**: 25% faster on Grace CPU vs x86 due to ARM64 SIMD optimizations; consistently faster than pandas for most operations.
3. **DGX Spark Relevance**: Confirmed 25% performance advantage on Grace CPU. First-class aarch64 support; Rust-native, no CUDA dependency.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 wheels; 25% faster on Grace CPU [NVIDIA-OFFICIAL]
   - CUDA 13: ✅ No CUDA dependency (CPU library with GPU-accelerated engine coming)
   - sm_121: ✅ N/A (CPU)
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — different API from pandas; lazy evaluation requires new mental model
7. **Best Use Cases**: Data transformation pipelines, large CSV/Parquet processing, ETL workflows, replacing pandas for performance
8. **Bad Use Cases**: Existing codebases heavily reliant on pandas API (use cudf.pandas instead), GPU-accelerated data processing (use cuDF)
9. **Install Complexity**: Trivial (pip install polars)
10. **Dependencies & Likely Conflicts**: Minimal dependencies; no conflicts expected
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### pandas + cuDF Accelerator

1. **Purpose**: GPU-accelerated pandas through RAPIDS cudf.pandas proxy that transparently routes operations to GPU.
2. **Why It Matters**: Zero-code-change GPU acceleration for existing pandas code; load cudf.pandas and existing code runs on GPU.
3. **DGX Spark Relevance**: Available for DGX Spark; enables GPU acceleration of existing pandas pipelines without rewriting code.
4. **Architecture Compatibility**:
   - ARM64: ✅ RAPIDS builds available for DGX Spark
   - CUDA 13: ✅ RAPIDS built for CUDA 13
   - sm_121: ✅ GPU-accelerated operations on Blackwell
   - Container: ✅ NGC RAPIDS container recommended
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production (pandas) / Stable (cuDF accelerator)
6. **Learning Curve**: Low — use pandas as normal; cudf.pandas is a drop-in accelerator
7. **Best Use Cases**: GPU-accelerating existing pandas code, large DataFrame operations, feature engineering
8. **Bad Use Cases**: Small data (<10K rows), operations not yet supported in cuDF (falls back to CPU pandas)
9. **Install Complexity**: Moderate (NGC container for cuDF) / Trivial (pandas alone)
10. **Dependencies & Likely Conflicts**: cuDF requires matching CUDA version; pandas alone has no conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### DuckDB

1. **Purpose**: In-process SQL OLAP database for analytical queries on local data (CSV, Parquet, JSON).
2. **Why It Matters**: Zero-configuration analytical SQL engine; reads directly from files; faster than pandas for SQL-expressible operations.
3. **DGX Spark Relevance**: Native aarch64 wheels; excellent for local analytical queries alongside ML workflows.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 wheels
   - CUDA 13: ✅ No CUDA dependency (CPU-only)
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — standard SQL interface; Python API is minimal
7. **Best Use Cases**: Ad-hoc analytical queries on local files, data exploration, replacing complex pandas chains with SQL, Parquet/CSV analysis
8. **Bad Use Cases**: Transactional workloads (use PostgreSQL), data that needs GPU acceleration (use cuDF)
9. **Install Complexity**: Trivial (pip install duckdb)
10. **Dependencies & Likely Conflicts**: Self-contained; no external dependencies
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### dbt (dbt-core)

1. **Purpose**: Data transformation framework using SQL with version control, testing, and documentation.
2. **Why It Matters**: Standard for analytics engineering; brings software engineering practices to data transformation.
3. **DGX Spark Relevance**: Pure Python; works with PostgreSQL adapter on DGX Spark for local data warehouse patterns.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ N/A
   - Container: ✅ Works anywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — Jinja-templated SQL, project structure, testing patterns
7. **Best Use Cases**: SQL-based data transformations, data pipeline documentation, data quality testing, analytics engineering
8. **Bad Use Cases**: Non-SQL transformations (use Polars/pandas), real-time processing, small one-off queries
9. **Install Complexity**: Easy (pip install dbt-core dbt-postgres)
10. **Dependencies & Likely Conflicts**: Jinja2, agate — all pure Python; PostgreSQL adapter needs psycopg2 (has aarch64 wheel)
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Great Expectations

1. **Purpose**: Data validation and quality testing framework for data pipelines.
2. **Why It Matters**: Catches data quality issues before they propagate through ML pipelines; automated data profiling and documentation.
3. **DGX Spark Relevance**: Pure Python; validates data from any source (files, databases, DataFrames).
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ N/A
   - Container: ✅ Works anywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — expectations syntax, data context configuration, checkpoint setup
7. **Best Use Cases**: Data pipeline validation, data quality monitoring, automated data documentation
8. **Bad Use Cases**: Real-time validation (too slow), simple one-off checks (write assertions instead)
9. **Install Complexity**: Easy (pip install great-expectations)
10. **Dependencies & Likely Conflicts**: Many optional dependencies; core is well-behaved
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 8. Android Development Tools

Android development on ARM64 Linux presents unique challenges due to toolchain availability.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Android Studio | Production | N/A | 1 | 5 | 1 | Avoid-Delay | [COMMUNITY] |
| Kotlin Compiler | Production | SDKMAN / apt | 4 | 5 | 5 | Optional | [INFERRED] |
| Gradle | Production | SDKMAN / wrapper | 4 | 5 | 5 | Optional | [INFERRED] |
| ADB (Android Debug Bridge) | Production | apt / SDK | 5 | 5 | 5 | Optional | [INFERRED] |
| Jetpack Compose | Production | Gradle dependency | 4 | 5 | 4 | Optional | [INFERRED] |

### Android Studio

1. **Purpose**: Official IDE for Android application development.
2. **Why It Matters**: Standard Android development environment with visual layout editor, emulator, and debugging tools.
3. **DGX Spark Relevance**: ❌ **No ARM64 Linux build available.** Android Studio only ships x86_64 Linux, macOS (Intel/ARM), and Windows builds.
4. **Architecture Compatibility**:
   - ARM64: ❌ No aarch64 Linux build
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ❌ Cannot run x86 GUI app in container on ARM
   - OS: ❌ Not available for Ubuntu 24.04 aarch64
5. **Maturity Level**: Production (on supported platforms)
6. **Learning Curve**: Medium
7. **Best Use Cases**: (On x86 machine) Full Android app development, visual layout editing, emulator testing
8. **Bad Use Cases**: DGX Spark — does not run at all
9. **Install Complexity**: N/A — not available on ARM64 Linux
10. **Dependencies & Likely Conflicts**: N/A
11. **Practicality Score**: 1
12. **Stability Score**: 5 (on supported platforms)
13. **Compatibility Score**: 1
14. **Stack Classification**: Avoid-Delay

> **Workaround**: Use a separate x86 machine for Android Studio GUI work. Use DGX Spark for Gradle command-line builds, AI-assisted code generation, and ADB device interaction.

### Kotlin Compiler

1. **Purpose**: JVM-based compiler for Kotlin programming language (Android's primary language).
2. **Why It Matters**: Kotlin compilation can run headlessly on any JVM; enables command-line Android builds without Android Studio.
3. **DGX Spark Relevance**: JVM runs on aarch64; Kotlin compiler works via Gradle for command-line builds.
4. **Architecture Compatibility**:
   - ARM64: ✅ JVM aarch64 native
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker + JVM
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low (if already know Kotlin)
7. **Best Use Cases**: Command-line Kotlin compilation, headless Android builds, CI/CD pipeline integration
8. **Bad Use Cases**: Full IDE development experience (need Android Studio on x86)
9. **Install Complexity**: Easy (SDKMAN: `sdk install kotlin` or `apt install kotlin`)
10. **Dependencies & Likely Conflicts**: JDK 17+ required; SDKMAN manages versions cleanly
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Gradle

1. **Purpose**: Build automation tool; standard for Android and JVM projects.
2. **Why It Matters**: Handles Android project compilation, dependency resolution, APK/AAB generation.
3. **DGX Spark Relevance**: JVM-based; runs on aarch64; enables headless Android builds on DGX Spark.
4. **Architecture Compatibility**:
   - ARM64: ✅ JVM aarch64
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker + JVM
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — Groovy/Kotlin DSL, plugin ecosystem
7. **Best Use Cases**: Headless Android builds, CI/CD, multi-module project builds
8. **Bad Use Cases**: Non-JVM projects
9. **Install Complexity**: Easy (Gradle wrapper ships with Android projects; or SDKMAN)
10. **Dependencies & Likely Conflicts**: JDK 17+; Android SDK command-line tools (aarch64 available)
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### ADB (Android Debug Bridge)

1. **Purpose**: Command-line tool for communicating with Android devices and emulators.
2. **Why It Matters**: Essential for deploying, debugging, and testing Android apps on physical devices.
3. **DGX Spark Relevance**: ADB CLI available on aarch64; can connect to USB-attached Android devices for deployment and testing.
4. **Architecture Compatibility**:
   - ARM64: ✅ aarch64 build available via platform-tools
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ⚠️ Needs USB passthrough
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low
7. **Best Use Cases**: Deploying APKs to devices, logcat debugging, device shell access
8. **Bad Use Cases**: Emulator-based testing (emulator needs x86)
9. **Install Complexity**: Easy (`sudo apt install adb` or Android SDK platform-tools)
10. **Dependencies & Likely Conflicts**: USB permissions (udev rules); no software conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Jetpack Compose

1. **Purpose**: Modern declarative UI toolkit for building Android interfaces.
2. **Why It Matters**: Standard Android UI framework; replaced XML layouts.
3. **DGX Spark Relevance**: Compose UI code compiles via Kotlin/Gradle on aarch64; preview rendering requires Android Studio (x86 only).
4. **Architecture Compatibility**:
   - ARM64: ✅ Compilation works via Gradle
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Headless compilation
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — declarative paradigm, Compose-specific patterns
7. **Best Use Cases**: Building Android UIs via command-line compilation, CI/CD builds
8. **Bad Use Cases**: Visual preview/design (needs Android Studio on x86)
9. **Install Complexity**: Easy (Gradle dependency; no separate install)
10. **Dependencies & Likely Conflicts**: Kotlin compiler version must match Compose compiler version
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 9. Desktop App Frameworks

Frameworks for building desktop GUI applications on DGX Spark.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Electron | Stable | npm install | 3 | 4 | 3 | Optional | [INFERRED] |
| Tauri | Stable | cargo install | 4 | 4 | 5 | Optional | [INFERRED] |
| Qt / PySide6 | Production | pip install | 4 | 5 | 4 | Optional | [INFERRED] |

### Electron

1. **Purpose**: Cross-platform desktop application framework using web technologies (HTML/CSS/JS).
2. **Why It Matters**: Powers VS Code, Slack, Discord, and many other desktop apps.
3. **DGX Spark Relevance**: ARM64 Linux builds exist but are less tested than x86; heavy memory footprint is a concern on shared 128 GB system.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ aarch64 Linux builds exist but less tested and community-maintained
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ⚠️ GUI apps in containers need X11 forwarding
   - OS: ⚠️ Ubuntu 24.04 aarch64 — functional but edge cases possible
5. **Maturity Level**: Stable
6. **Learning Curve**: Low (for web developers)
7. **Best Use Cases**: Cross-platform desktop apps when web tech is preferred, porting existing web apps to desktop
8. **Bad Use Cases**: Memory-constrained scenarios on DGX Spark (each Electron app uses 200+ MB RAM), lightweight tools (use Tauri)
9. **Install Complexity**: Easy (npm install electron)
10. **Dependencies & Likely Conflicts**: Node.js aarch64 ✅; Chromium aarch64 builds may have rendering quirks; large disk footprint
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 3
14. **Stack Classification**: Optional

### Tauri

1. **Purpose**: Lightweight cross-platform desktop app framework using web frontends with a Rust backend.
2. **Why It Matters**: 10× smaller and 5× less memory than Electron; uses system webview instead of bundling Chromium.
3. **DGX Spark Relevance**: Rust has tier 1 aarch64 support; Tauri builds natively on ARM64 Linux; much lighter than Electron.
4. **Architecture Compatibility**:
   - ARM64: ✅ Rust aarch64 is tier 1; builds natively
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ⚠️ Needs system webview (WebKitGTK); GUI needs X11
   - OS: ✅ Ubuntu 24.04 (WebKitGTK available via apt)
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — Rust backend, web frontend, IPC patterns
7. **Best Use Cases**: Lightweight desktop tools, data visualization dashboards, local AI app interfaces
8. **Bad Use Cases**: Teams without Rust experience, apps requiring Chromium-specific web APIs
9. **Install Complexity**: Moderate (cargo install tauri-cli + WebKitGTK system deps)
10. **Dependencies & Likely Conflicts**: WebKitGTK, libsoup, glib — all available via apt on Ubuntu 24.04 aarch64
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Qt / PySide6

1. **Purpose**: Mature C++/Python cross-platform GUI framework with comprehensive widget library.
2. **Why It Matters**: Industry-standard for complex desktop applications; Python binding (PySide6) enables rapid development.
3. **DGX Spark Relevance**: PySide6 has aarch64 wheels; native performance; good for data visualization and tooling UIs.
4. **Architecture Compatibility**:
   - ARM64: ✅ PySide6 aarch64 wheels available (may need system Qt deps)
   - CUDA 13: ✅ N/A (GPU rendering via OpenGL, not CUDA)
   - sm_121: ✅ N/A
   - Container: ⚠️ Needs X11/Wayland forwarding
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — large API surface, signal/slot patterns, layout management
7. **Best Use Cases**: Complex desktop applications, data visualization tools, scientific instrument interfaces
8. **Bad Use Cases**: Simple UIs (Streamlit/Gradio faster to build), web-deployed apps
9. **Install Complexity**: Moderate (pip install PySide6 + system Qt dependencies via apt)
10. **Dependencies & Likely Conflicts**: System Qt libraries via apt; potential version conflicts between system Qt and pip PySide6
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 10. Web App Frameworks

Frameworks for building web applications and UI dashboards.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| FastAPI | Production | pip install | 5 | 5 | 5 | Default | [INFERRED] |
| Flask | Production | pip install | 4 | 5 | 5 | Optional | [INFERRED] |
| Streamlit | Production | pip install | 5 | 5 | 5 | Default | [INFERRED] |
| Gradio | Production | pip install | 5 | 5 | 5 | Default | [INFERRED] |
| Next.js | Production | npm create | 4 | 5 | 5 | Optional | [INFERRED] |

### FastAPI

1. **Purpose**: Modern, high-performance Python web framework for building APIs with automatic OpenAPI documentation.
2. **Why It Matters**: De facto standard for Python APIs; async support; automatic docs; type-safe with Pydantic.
3. **DGX Spark Relevance**: Pure Python; primary API layer for all DGX Spark projects; wraps Ollama/NIM with custom business logic.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — intuitive decorator-based routes, automatic docs
7. **Best Use Cases**: REST APIs, inference API proxies, WebSocket endpoints, async microservices
8. **Bad Use Cases**: Static websites (use Next.js), visual dashboards (use Streamlit/Gradio)
9. **Install Complexity**: Trivial (pip install fastapi uvicorn)
10. **Dependencies & Likely Conflicts**: Pydantic, Starlette — no conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Flask

1. **Purpose**: Lightweight Python web framework for simple web applications and APIs.
2. **Why It Matters**: Simpler than FastAPI for basic use cases; massive ecosystem of extensions.
3. **DGX Spark Relevance**: Pure Python; good for simple web interfaces and prototyping.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low
7. **Best Use Cases**: Simple web apps, server-rendered pages, quick prototypes, existing Flask codebases
8. **Bad Use Cases**: High-performance async APIs (FastAPI better), complex data apps (Streamlit better)
9. **Install Complexity**: Trivial (pip install flask)
10. **Dependencies & Likely Conflicts**: Minimal; Jinja2, Werkzeug
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Streamlit

1. **Purpose**: Python framework for building interactive data applications and dashboards with minimal code.
2. **Why It Matters**: Fastest path from Python script to interactive web app; excellent for ML model demos and data exploration.
3. **DGX Spark Relevance**: Pure Python; perfect for building DGX Spark dashboards, model evaluation UIs, and data exploration tools.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — Python-only, no frontend knowledge needed
7. **Best Use Cases**: ML model dashboards, data exploration tools, internal tools, rapid prototyping
8. **Bad Use Cases**: Complex multi-page web apps (Next.js better), fine-grained UI control
9. **Install Complexity**: Trivial (pip install streamlit)
10. **Dependencies & Likely Conflicts**: pyarrow (aarch64 wheel available); tornado; no typical conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Gradio

1. **Purpose**: Python library for building ML model demos and web interfaces with minimal code.
2. **Why It Matters**: Fewest lines of code to create an ML model demo; built-in components for common ML I/O types (text, image, audio).
3. **DGX Spark Relevance**: Pure Python; ideal for quick model demonstrations and interactive testing on DGX Spark.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ No CUDA dependency
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — 5-line demo interfaces; `gr.Interface` pattern
7. **Best Use Cases**: Quick ML demos, model comparison interfaces, Hugging Face Spaces deployment, interactive testing
8. **Bad Use Cases**: Complex dashboards (Streamlit better), production web apps (Next.js/FastAPI better)
9. **Install Complexity**: Trivial (pip install gradio)
10. **Dependencies & Likely Conflicts**: fastapi, pydantic included; potential version conflicts if also using standalone FastAPI
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Next.js

1. **Purpose**: React-based full-stack web framework with server-side rendering, API routes, and static generation.
2. **Why It Matters**: Standard for production web applications; combines frontend and backend in one framework.
3. **DGX Spark Relevance**: Node.js has native aarch64 support; Next.js runs on DGX Spark for building full web applications.
4. **Architecture Compatibility**:
   - ARM64: ✅ Node.js has native aarch64 builds
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment supported
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — React, TypeScript, SSR/SSG concepts
7. **Best Use Cases**: Production web applications, public-facing products, complex UIs with FastAPI backend
8. **Bad Use Cases**: Quick data dashboards (Streamlit faster), simple model demos (Gradio faster)
9. **Install Complexity**: Easy (`npx create-next-app@latest`)
10. **Dependencies & Likely Conflicts**: Node.js 18+; npm packages are platform-agnostic (JavaScript); native addons may need aarch64 builds
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 11. Container Tools

Container runtime and management tools critical for DGX Spark's container-first workflow.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Docker Engine | Production | Pre-installed | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| Docker Compose | Production | Pre-installed | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| Podman | Production | apt install | 4 | 5 | 5 | Optional | [INFERRED] |
| NGC CLI | Production | Pre-installed | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |
| NVIDIA Container Toolkit | Production | Pre-installed | 5 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |

### Docker Engine

1. **Purpose**: Container runtime for building, running, and managing containers.
2. **Why It Matters**: Foundation of the container-first workflow on DGX Spark; isolates GPU workloads and manages dependencies.
3. **DGX Spark Relevance**: Pre-installed; primary method for running GPU workloads with CUDA 13 compatibility guaranteed inside NGC containers.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 Docker Engine
   - CUDA 13: ✅ Via NVIDIA Container Toolkit
   - sm_121: ✅ GPU passthrough via --gpus
   - Container: ✅ Is the container runtime
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — familiar Docker CLI
7. **Best Use Cases**: Running NGC containers, isolating GPU workloads, reproducible environments, multi-service deployments
8. **Bad Use Cases**: None on DGX Spark — essential tool
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: NVIDIA Container Toolkit pre-configured; iptables/nftables for networking
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Docker Compose

1. **Purpose**: Multi-container application orchestration using declarative YAML configuration.
2. **Why It Matters**: Defines entire application stacks (app + database + cache + monitoring) in one file; reproducible deployments.
3. **DGX Spark Relevance**: Pre-installed; recommended way to run multi-service stacks on DGX Spark (preferred over Kubernetes for single-node).
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64
   - CUDA 13: ✅ GPU service support via deploy.resources
   - sm_121: ✅ GPU passthrough in compose services
   - Container: ✅ Orchestrates containers
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — YAML configuration, docker compose up/down
7. **Best Use Cases**: Multi-service stacks, development environments, monitoring stacks, application + database combinations
8. **Bad Use Cases**: Single-container runs (plain docker run), multi-node orchestration (use Kubernetes)
9. **Install Complexity**: Trivial (pre-installed as Docker plugin)
10. **Dependencies & Likely Conflicts**: Docker Engine; no additional conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Podman

1. **Purpose**: Daemonless container engine; drop-in Docker replacement with rootless support.
2. **Why It Matters**: Enhanced security via rootless containers; no daemon process; Docker CLI-compatible.
3. **DGX Spark Relevance**: Available on aarch64; useful for rootless container workflows; NVIDIA CDI support for GPU access.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 packages
   - CUDA 13: ✅ Via NVIDIA CDI (Container Device Interface)
   - sm_121: ✅ GPU access via CDI
   - Container: ✅ Is a container runtime
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — Docker-compatible CLI
7. **Best Use Cases**: Rootless containers, enhanced security, Docker-compatible workflows without daemon
8. **Bad Use Cases**: When Docker is already working well (no strong reason to switch), Docker Compose-heavy workflows (Podman Compose exists but less mature)
9. **Install Complexity**: Easy (sudo apt install podman)
10. **Dependencies & Likely Conflicts**: May conflict with Docker if both installed; choose one as primary
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### NGC CLI

1. **Purpose**: Command-line interface for NVIDIA NGC catalog — pulling containers, models, and resources.
2. **Why It Matters**: Access to NVIDIA's curated container registry with pre-built, GPU-optimized containers and models.
3. **DGX Spark Relevance**: Pre-installed; primary source for GPU-optimized containers that are guaranteed compatible with DGX Spark.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 binary
   - CUDA 13: ✅ NGC containers built for CUDA 13
   - sm_121: ✅ NGC containers include Blackwell support
   - Container: ✅ Manages container pulls
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — simple CLI commands
7. **Best Use Cases**: Pulling NGC containers, downloading pre-trained models, browsing NVIDIA model/container catalog
8. **Bad Use Cases**: Non-NVIDIA container sources (use Docker Hub directly)
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: NGC API key for authenticated pulls; no software conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### NVIDIA Container Toolkit

1. **Purpose**: Enables Docker/Podman containers to access NVIDIA GPUs with proper driver and CUDA support.
2. **Why It Matters**: Bridge between container runtime and GPU; enables `--gpus all` flag and CUDA inside containers.
3. **DGX Spark Relevance**: Pre-installed and pre-configured on DGX Spark; essential for all containerized GPU workloads.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64
   - CUDA 13: ✅ Configured for CUDA 13.0.2
   - sm_121: ✅ Full Blackwell GPU passthrough
   - Container: ✅ Enables GPU in containers
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — transparent once installed
7. **Best Use Cases**: Any containerized GPU workload
8. **Bad Use Cases**: N/A — required for GPU containers
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: NVIDIA drivers (pre-installed); Docker or Podman runtime
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

---

## 12. Observability Tools

Monitoring, logging, and performance visibility tools.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Prometheus | Production | Docker | 4 | 5 | 5 | Default | [INFERRED] |
| Grafana | Production | Docker | 4 | 5 | 5 | Default | [INFERRED] |
| Loki | Production | Docker | 4 | 5 | 5 | Optional | [INFERRED] |
| nvidia-smi | Production | Pre-installed | 4 | 5 | 5 | Default | [NVIDIA-OFFICIAL] |

### Prometheus

1. **Purpose**: Time-series metrics collection and alerting system.
2. **Why It Matters**: Standard for infrastructure monitoring; pull-based metric collection; powerful PromQL query language.
3. **DGX Spark Relevance**: ARM64 Docker images available; monitors GPU metrics via NVIDIA DCGM exporter.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official aarch64 Docker images
   - CUDA 13: ✅ N/A (monitoring tool)
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment recommended
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — PromQL query language, scrape configuration, alerting rules
7. **Best Use Cases**: System metrics collection, GPU utilization monitoring, alerting on resource thresholds
8. **Bad Use Cases**: Log aggregation (use Loki), distributed tracing (use Jaeger)
9. **Install Complexity**: Easy (Docker Compose)
10. **Dependencies & Likely Conflicts**: Needs accessible metric endpoints; DCGM exporter for GPU metrics
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Grafana

1. **Purpose**: Visualization and dashboarding platform for metrics and logs.
2. **Why It Matters**: Beautiful, customizable dashboards; integrates with Prometheus, Loki, PostgreSQL, and many other data sources.
3. **DGX Spark Relevance**: ARM64 Docker images available; essential for visualizing DGX Spark resource utilization.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official aarch64 Docker images
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — dashboard creation, panel types, data source configuration
7. **Best Use Cases**: GPU utilization dashboards, system monitoring, inference latency tracking, custom metric visualization
8. **Bad Use Cases**: Real-time streaming visualization (consider dedicated tools), data processing
9. **Install Complexity**: Easy (Docker Compose with Prometheus)
10. **Dependencies & Likely Conflicts**: Typically deployed alongside Prometheus; port 3000 default (may conflict with dev servers)
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Loki

1. **Purpose**: Log aggregation system designed to work with Grafana; like Prometheus but for logs.
2. **Why It Matters**: Lightweight log aggregation with Grafana integration; label-based indexing (not full-text).
3. **DGX Spark Relevance**: ARM64 Docker images available; completes the Grafana + Prometheus monitoring stack.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official aarch64 Docker images
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — LogQL query language, label configuration, Promtail setup
7. **Best Use Cases**: Centralized logging with Grafana, application log aggregation, correlating logs with metrics
8. **Bad Use Cases**: Full-text search (use Elasticsearch), small-scale logging (tail files directly)
9. **Install Complexity**: Moderate (Docker Compose + Promtail agent configuration)
10. **Dependencies & Likely Conflicts**: Promtail or Fluentd for log shipping; integrates with Grafana
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### nvidia-smi

1. **Purpose**: NVIDIA System Management Interface for GPU monitoring and management.
2. **Why It Matters**: Primary CLI tool for checking GPU status, memory usage, temperature, and running processes.
3. **DGX Spark Relevance**: Pre-installed; note that some advanced nvidia-smi features may report differently on DGX Spark's unified memory architecture.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pre-installed native
   - CUDA 13: ✅ Reports CUDA 13.0.2
   - sm_121: ✅ Full Blackwell GPU reporting
   - Container: ✅ Accessible inside GPU containers
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — simple CLI
7. **Best Use Cases**: Quick GPU status checks, memory monitoring, process identification, temperature monitoring
8. **Bad Use Cases**: Detailed profiling (use Nsight Systems/Compute), historical metrics (use Prometheus + DCGM)
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: NVIDIA driver (pre-installed)
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

---

## 13. Automation Tools

Task scheduling and automation frameworks.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| n8n | Stable | Docker | 4 | 4 | 5 | Optional | [INFERRED] |
| cron | Production | Pre-installed | 5 | 5 | 5 | Default | [INFERRED] |
| systemd timers | Production | Pre-installed | 5 | 5 | 5 | Default | [INFERRED] |
| Custom Python schedulers | Stable | pip install | 4 | 4 | 5 | Optional | [INFERRED] |

### n8n

1. **Purpose**: Low-code workflow automation platform with visual editor and 400+ integrations.
2. **Why It Matters**: Automates complex workflows without coding; connects APIs, databases, and AI models visually.
3. **DGX Spark Relevance**: Docker image available for aarch64; useful for automating AI-powered workflows.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official Docker image for aarch64
   - CUDA 13: ✅ N/A (orchestration layer)
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — visual workflow builder
7. **Best Use Cases**: Low-code automations, API integrations, scheduled AI workflows, webhook handlers
8. **Bad Use Cases**: High-performance data pipelines (use Airflow/Prefect), complex logic (write Python)
9. **Install Complexity**: Easy (Docker)
10. **Dependencies & Likely Conflicts**: Node.js inside container; port 5678 default
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### cron

1. **Purpose**: Unix job scheduler for time-based task execution.
2. **Why It Matters**: Simplest scheduling mechanism; zero overhead; built into every Linux system.
3. **DGX Spark Relevance**: Pre-installed; ideal for simple recurring tasks (model updates, data fetches, cleanup).
4. **Architecture Compatibility**:
   - ARM64: ✅ System utility
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Available in containers
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — crontab syntax
7. **Best Use Cases**: Simple scheduled scripts, periodic data fetching, cleanup jobs, model refresh triggers
8. **Bad Use Cases**: Complex DAG dependencies (use Airflow), tasks needing retry logic (use Prefect), tasks needing UI monitoring
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: None
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### systemd Timers

1. **Purpose**: systemd-native timer units for scheduled and event-based task execution.
2. **Why It Matters**: More powerful than cron; supports dependencies, logging, resource limits, and service management.
3. **DGX Spark Relevance**: Pre-installed; better than cron for tasks that need service management (restart on failure, resource limits).
4. **Architecture Compatibility**:
   - ARM64: ✅ System utility
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ⚠️ systemd not typically available inside containers
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — systemd unit file syntax, timer/service pair configuration
7. **Best Use Cases**: Reliable scheduled services with restart policies, tasks needing resource limits, journal-integrated logging
8. **Bad Use Cases**: Tasks inside containers (cron or orchestrator better), simple one-off schedules (cron simpler)
9. **Install Complexity**: Trivial (pre-installed)
10. **Dependencies & Likely Conflicts**: None on host system
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Custom Python Schedulers (APScheduler / schedule)

1. **Purpose**: Python libraries for in-process job scheduling (APScheduler for advanced, schedule for simple).
2. **Why It Matters**: Embeds scheduling logic directly in Python applications; no external service needed.
3. **DGX Spark Relevance**: Pure Python; good for application-level scheduling within FastAPI or standalone services.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works anywhere Python runs
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low
7. **Best Use Cases**: In-app scheduling, periodic model reloading, health check tasks, background data processing
8. **Bad Use Cases**: System-level scheduling (use cron/systemd), complex DAGs (use Airflow)
9. **Install Complexity**: Trivial (pip install apscheduler or pip install schedule)
10. **Dependencies & Likely Conflicts**: Minimal; APScheduler supports PostgreSQL/Redis job stores
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 14. Orchestration Tools

Pipeline and workflow orchestration for complex data and ML workflows.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Airflow | Production | Docker Compose | 4 | 5 | 5 | Optional | [INFERRED] |
| Prefect | Production | pip + Docker | 4 | 4 | 5 | Optional | [INFERRED] |
| Dagster | Stable | pip + Docker | 4 | 4 | 5 | Optional | [INFERRED] |

### Apache Airflow

1. **Purpose**: Workflow orchestration platform for authoring, scheduling, and monitoring DAG-based pipelines.
2. **Why It Matters**: Most mature pipeline orchestrator; massive community; extensive operator library.
3. **DGX Spark Relevance**: ARM64 Docker images available; good for complex data/ML pipelines on DGX Spark.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official aarch64 Docker images
   - CUDA 13: ✅ N/A (orchestration layer; triggers GPU tasks)
   - sm_121: ✅ N/A
   - Container: ✅ Docker Compose deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — DAG concepts, operators, connections, XComs, executor configuration
7. **Best Use Cases**: Complex data pipelines, ML training pipelines, scheduled ETL workflows, multi-step processing chains
8. **Bad Use Cases**: Simple scheduled tasks (use cron), real-time processing, lightweight workflows (overkill)
9. **Install Complexity**: Moderate (Docker Compose with official Airflow image)
10. **Dependencies & Likely Conflicts**: PostgreSQL for metadata DB; Redis for Celery executor; significant resource overhead
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Prefect

1. **Purpose**: Modern workflow orchestration with Python-native task definitions and cloud/self-hosted UI.
2. **Why It Matters**: Easier than Airflow; Pythonic task definitions; good UI; handles retries and caching.
3. **DGX Spark Relevance**: Pure Python agent; self-hosted server available; simpler setup than Airflow for medium-complexity pipelines.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python server + agent
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment supported
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — Python decorators for flows/tasks, deployment concepts
7. **Best Use Cases**: ML pipelines, data processing workflows, event-driven automation, pipelines with retry/caching needs
8. **Bad Use Cases**: Very simple schedules (use cron), when Airflow is already deployed
9. **Install Complexity**: Easy (pip install prefect + prefect server start)
10. **Dependencies & Likely Conflicts**: SQLite or PostgreSQL for persistence; minimal conflicts
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Dagster

1. **Purpose**: Data-asset-focused orchestration platform where pipelines are defined around data assets rather than tasks.
2. **Why It Matters**: Asset-centric approach is ideal for data engineering; built-in data lineage, type checking, and testing.
3. **DGX Spark Relevance**: Pure Python; good for data engineering workflows; self-hosted UI available.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment supported
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — asset-centric mental model differs from traditional DAG thinking
7. **Best Use Cases**: Data engineering pipelines, ML feature pipelines, data quality-focused workflows
8. **Bad Use Cases**: Simple task scheduling, workflows without clear data asset lineage
9. **Install Complexity**: Moderate (pip install dagster dagster-webserver + dagster dev)
10. **Dependencies & Likely Conflicts**: grpcio (has aarch64 wheels); PostgreSQL optional for production
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 15. Databases

Relational and key-value databases for persistent storage.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| PostgreSQL | Production | apt install | 5 | 5 | 5 | Default | [INFERRED] |
| SQLite | Production | Built-in | 5 | 5 | 5 | Default | [INFERRED] |
| Redis | Production | apt / Docker | 5 | 5 | 5 | Default | [INFERRED] |
| TimescaleDB | Production | Docker / extension | 4 | 5 | 5 | Optional | [INFERRED] |

### PostgreSQL

1. **Purpose**: Enterprise-grade open-source relational database with extensive extension ecosystem.
2. **Why It Matters**: Standard relational database; extensible with pgvector (vector search), TimescaleDB (time-series), PostGIS (geospatial).
3. **DGX Spark Relevance**: Native ARM64 builds; foundation database for DGX Spark applications with pgvector for RAG workflows.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 packages
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Official aarch64 Docker images
   - OS: ✅ Ubuntu 24.04 (apt install postgresql)
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — SQL + administration, but extensive documentation
7. **Best Use Cases**: Application data storage, RAG metadata, user management, structured data with pgvector embeddings
8. **Bad Use Cases**: Pure key-value cache (Redis better), embedded/serverless use (SQLite better)
9. **Install Complexity**: Easy (sudo apt install postgresql)
10. **Dependencies & Likely Conflicts**: Port 5432; no software conflicts; libpq for client libraries
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### SQLite

1. **Purpose**: Embedded serverless SQL database engine; file-based, zero-configuration.
2. **Why It Matters**: Simplest possible database; no server process; single file; built into Python stdlib.
3. **DGX Spark Relevance**: Built into Python; perfect for application configuration, caching, and small datasets.
4. **Architecture Compatibility**:
   - ARM64: ✅ Built into Python; native aarch64
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — standard SQL, zero configuration
7. **Best Use Cases**: Application configuration, local caching, small-medium datasets, development databases, ChromaDB backend
8. **Bad Use Cases**: High-concurrency writes (use PostgreSQL), multi-user access, large datasets (use PostgreSQL/DuckDB)
9. **Install Complexity**: Trivial (built into Python)
10. **Dependencies & Likely Conflicts**: None
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### Redis

1. **Purpose**: In-memory data structure store used as database, cache, message broker, and queue.
2. **Why It Matters**: Sub-millisecond latency for caching and pub/sub; versatile data structures (strings, hashes, lists, sets, streams).
3. **DGX Spark Relevance**: Native ARM64 builds; ideal for caching LLM responses, rate limiting, session storage, and inter-service messaging.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 builds
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Official aarch64 Docker images
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — simple CLI, well-documented commands
7. **Best Use Cases**: Response caching, session storage, rate limiting, pub/sub messaging, job queues (with RQ or Celery)
8. **Bad Use Cases**: Large dataset persistence (use PostgreSQL), complex queries (use PostgreSQL/DuckDB)
9. **Install Complexity**: Easy (sudo apt install redis-server or Docker)
10. **Dependencies & Likely Conflicts**: Port 6379; no software conflicts
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### TimescaleDB

1. **Purpose**: PostgreSQL extension for time-series data with automatic partitioning, compression, and continuous aggregates.
2. **Why It Matters**: Adds time-series superpowers to PostgreSQL; no need for a separate time-series database.
3. **DGX Spark Relevance**: PostgreSQL extension; works on ARM64; ideal for monitoring data, sensor data, and financial time-series.
4. **Architecture Compatibility**:
   - ARM64: ✅ Docker images available for aarch64; extension compiles on ARM64
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Official Docker images
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — standard SQL with hypertable concepts
7. **Best Use Cases**: Monitoring metrics storage, financial time-series, IoT sensor data, inference latency tracking
8. **Bad Use Cases**: Non-time-series data (use plain PostgreSQL), when Prometheus is already handling metrics
9. **Install Complexity**: Easy (Docker with timescale/timescaledb image) / Moderate (PostgreSQL extension)
10. **Dependencies & Likely Conflicts**: PostgreSQL required; may need specific PostgreSQL version match
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 16. Vector Databases

Specialized databases for storing and searching vector embeddings.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| pgvector | Production | PostgreSQL extension | 5 | 5 | 5 | Default | [INFERRED] |
| ChromaDB | Stable | pip install | 4 | 4 | 5 | Optional | [INFERRED] |
| Qdrant | Production | Docker | 4 | 5 | 5 | Optional | [INFERRED] |
| Milvus | Production | Docker Compose | 3 | 4 | 4 | Avoid-Delay | [INFERRED] |

### pgvector

1. **Purpose**: PostgreSQL extension adding vector similarity search (cosine, L2, inner product) to existing PostgreSQL.
2. **Why It Matters**: No new database needed; add vector search to existing PostgreSQL; supports HNSW and IVFFlat indexes.
3. **DGX Spark Relevance**: Extends the recommended PostgreSQL installation; simplest path to vector search for RAG workflows on DGX Spark.
4. **Architecture Compatibility**:
   - ARM64: ✅ Compiles from source on aarch64; Docker images available
   - CUDA 13: ✅ N/A (CPU-based indexing)
   - sm_121: ✅ N/A
   - Container: ✅ pgvector Docker images available
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — SQL syntax for vector operations; familiar if you know PostgreSQL
7. **Best Use Cases**: RAG vector storage (combined with document metadata in same DB), <5M vectors, hybrid search (SQL + vector)
8. **Bad Use Cases**: >5M vectors with real-time requirements (consider Qdrant), distributed vector search
9. **Install Complexity**: Easy (Docker image with pgvector) / Moderate (compile extension for existing PostgreSQL)
10. **Dependencies & Likely Conflicts**: PostgreSQL 14+; no conflicts with existing PostgreSQL data
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### ChromaDB

1. **Purpose**: Lightweight embedding database designed for AI applications with simple Python API.
2. **Why It Matters**: Simplest vector database; embeds directly in Python process; great for prototyping RAG applications.
3. **DGX Spark Relevance**: Pure Python with SQLite backend; works perfectly on aarch64; ideal for prototyping before scaling to pgvector.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python core; hnswlib has aarch64 support
   - CUDA 13: ✅ N/A (CPU-based)
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment available
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — 5-line Python API; `collection.add()` / `collection.query()`
7. **Best Use Cases**: RAG prototyping, small-scale vector search (<100K vectors), embedded vector DB in Python apps
8. **Bad Use Cases**: Large-scale production (>500K vectors, use pgvector/Qdrant), SQL + vector hybrid queries (use pgvector)
9. **Install Complexity**: Trivial (pip install chromadb)
10. **Dependencies & Likely Conflicts**: hnswlib, SQLite; minimal conflicts
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Qdrant

1. **Purpose**: High-performance vector database with filtering, payload storage, and gRPC/REST API.
2. **Why It Matters**: Best filtering performance among vector DBs; supports complex metadata queries alongside vector search.
3. **DGX Spark Relevance**: Rust-based server with aarch64 Docker images; excellent for production RAG with metadata filtering.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official aarch64 Docker images
   - CUDA 13: ✅ N/A (CPU-based; GPU index experimental)
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — REST/gRPC API, filtering syntax, collection configuration
7. **Best Use Cases**: Production RAG with metadata filtering, 5M+ vectors, multi-tenant vector search
8. **Bad Use Cases**: Quick prototyping (ChromaDB simpler), <100K vectors (pgvector sufficient)
9. **Install Complexity**: Easy (Docker)
10. **Dependencies & Likely Conflicts**: Port 6333/6334; self-contained Docker image
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Milvus

1. **Purpose**: Distributed vector database designed for billion-scale vector similarity search.
2. **Why It Matters**: Highest scalability among vector databases; supports GPU-accelerated indexing.
3. **DGX Spark Relevance**: ⚠️ Distributed architecture is overkill for single-node DGX Spark. Heavy resource usage (etcd, MinIO, Pulsar/Kafka). Recommend pgvector or Qdrant first.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ Docker images available but aarch64 support is secondary
   - CUDA 13: ⚠️ GPU index requires compatible CUDA build
   - sm_121: ⚠️ GPU index may need rebuild for sm_121
   - Container: ✅ Docker Compose deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — distributed system concepts, multiple component management
7. **Best Use Cases**: Billion-scale vector search, distributed deployments, GPU-accelerated indexing
8. **Bad Use Cases**: Single-node DGX Spark (too heavy), prototyping, <5M vectors
9. **Install Complexity**: Hard (Docker Compose with etcd, MinIO, multiple services)
10. **Dependencies & Likely Conflicts**: etcd, MinIO/S3, Pulsar/Kafka; significant memory overhead (2-4 GB baseline)
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Avoid-Delay

---

## 17. Messaging / Queues

Message brokers and queuing systems for inter-service communication.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Redis Pub/Sub | Production | Via Redis | 5 | 5 | 5 | Default | [INFERRED] |
| RabbitMQ | Production | Docker | 4 | 5 | 5 | Optional | [INFERRED] |
| NATS | Production | Docker / binary | 4 | 5 | 5 | Optional | [INFERRED] |
| Kafka | Production | Docker Compose | 3 | 5 | 4 | Optional | [INFERRED] |

### Redis Pub/Sub & Streams

1. **Purpose**: Lightweight pub/sub messaging and stream processing using Redis's built-in capabilities.
2. **Why It Matters**: Zero additional infrastructure if Redis is already deployed; Redis Streams add persistent, consumer-group-based messaging.
3. **DGX Spark Relevance**: Redis is already in the default stack; Pub/Sub adds messaging with no additional services.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 (via Redis)
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Via Redis container
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — familiar Redis CLI/library, PUBLISH/SUBSCRIBE commands
7. **Best Use Cases**: Inter-service events, inference result broadcasting, lightweight task distribution
8. **Bad Use Cases**: Messages requiring guaranteed delivery (use RabbitMQ), high-volume event streaming (use Kafka)
9. **Install Complexity**: Trivial (already installed with Redis)
10. **Dependencies & Likely Conflicts**: Redis server running; no additional dependencies
11. **Practicality Score**: 5
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Default

### RabbitMQ

1. **Purpose**: Enterprise message broker supporting AMQP, MQTT, and STOMP protocols with routing, queues, and exchanges.
2. **Why It Matters**: Guaranteed message delivery, dead letter queues, routing patterns; standard for enterprise messaging.
3. **DGX Spark Relevance**: ARM64 Docker images available; good for reliable async task processing with Celery.
4. **Architecture Compatibility**:
   - ARM64: ✅ Official aarch64 Docker images (Erlang has ARM64 support)
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker deployment
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — exchanges, queues, bindings, acknowledgment patterns
7. **Best Use Cases**: Reliable task queues with Celery, message routing, guaranteed delivery workflows
8. **Bad Use Cases**: Simple pub/sub (Redis sufficient), high-throughput event streaming (Kafka better)
9. **Install Complexity**: Easy (Docker)
10. **Dependencies & Likely Conflicts**: Port 5672/15672; Erlang runtime inside container
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### NATS

1. **Purpose**: Lightweight, high-performance messaging system with pub/sub, request/reply, and JetStream persistence.
2. **Why It Matters**: Simplest message broker; single binary; extremely fast; JetStream adds persistence.
3. **DGX Spark Relevance**: Go-based with aarch64 binary available; minimal resource overhead; good for lightweight messaging.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 binary (Go)
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker images available
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Low — simple pub/sub API; JetStream for persistence
7. **Best Use Cases**: Microservice communication, real-time event distribution, IoT messaging
8. **Bad Use Cases**: Complex routing (RabbitMQ better), guaranteed ordering at scale (Kafka better)
9. **Install Complexity**: Trivial (single binary download or Docker)
10. **Dependencies & Likely Conflicts**: Port 4222; self-contained
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### Apache Kafka

1. **Purpose**: Distributed event streaming platform for high-throughput, fault-tolerant event processing.
2. **Why It Matters**: Standard for event streaming architectures; persistent log, consumer groups, exactly-once semantics.
3. **DGX Spark Relevance**: ⚠️ Heavy for single-node DGX Spark (JVM + ZooKeeper/KRaft); use Redis Streams or NATS for lightweight needs.
4. **Architecture Compatibility**:
   - ARM64: ✅ JVM-based; runs on aarch64 JVM
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ⚠️ Docker images available but heavy (multi-GB JVM heap)
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — topics, partitions, consumer groups, offset management
7. **Best Use Cases**: High-volume event streaming, event sourcing, log aggregation at scale
8. **Bad Use Cases**: Simple messaging (Redis/NATS sufficient), single-node with low message volumes
9. **Install Complexity**: Moderate (Docker Compose with KRaft mode)
10. **Dependencies & Likely Conflicts**: JVM; significant memory usage (1-2 GB heap minimum); ZooKeeper or KRaft
11. **Practicality Score**: 3
12. **Stability Score**: 5
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 18. API Frameworks

Frameworks for building and serving APIs.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| FastAPI | Production | pip install | 5 | 5 | 5 | Default | [INFERRED] |
| gRPC Python | Production | pip install | 4 | 5 | 5 | Optional | [INFERRED] |
| GraphQL (Strawberry) | Stable | pip install | 3 | 4 | 5 | Optional | [INFERRED] |

### FastAPI

See [Section 10: FastAPI](#fastapi) for full 14-field entry. FastAPI is the default API framework for all DGX Spark projects.

### gRPC Python

1. **Purpose**: High-performance RPC framework using Protocol Buffers for service-to-service communication.
2. **Why It Matters**: 10× faster than REST for service-to-service calls; strongly typed; bidirectional streaming.
3. **DGX Spark Relevance**: grpcio has aarch64 wheels; used by Triton, NemoClaw, and other NVIDIA services internally.
4. **Architecture Compatibility**:
   - ARM64: ✅ grpcio aarch64 wheels available
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: Medium — Protocol Buffers, service definitions, code generation, streaming patterns
7. **Best Use Cases**: High-performance service-to-service communication, Triton client, streaming inference results
8. **Bad Use Cases**: External/public APIs (REST more accessible), simple CRUD APIs (FastAPI simpler)
9. **Install Complexity**: Easy (pip install grpcio grpcio-tools)
10. **Dependencies & Likely Conflicts**: C++ runtime; grpcio version should match grpcio-tools
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### GraphQL (Strawberry)

1. **Purpose**: Python GraphQL framework with type-safe schema definition using dataclasses.
2. **Why It Matters**: GraphQL enables flexible, client-driven queries; Strawberry provides Pythonic API with FastAPI integration.
3. **DGX Spark Relevance**: Pure Python; integrates with FastAPI; useful for complex data APIs with nested relationships.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — GraphQL concepts, schema design, resolver patterns
7. **Best Use Cases**: APIs with complex nested data, frontend-driven query patterns, aggregating multiple data sources
8. **Bad Use Cases**: Simple REST APIs (FastAPI sufficient), real-time streaming (gRPC better)
9. **Install Complexity**: Easy (pip install strawberry-graphql)
10. **Dependencies & Likely Conflicts**: Minimal; optional FastAPI integration
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

---

## 19. GPU-Aware Dev Tools & Debuggers

Tools specifically for GPU development, debugging, and profiling.

| Tool | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|------|----------|---------|:---:|:---:|:---:|-------|------------|
| Nsight Systems | Production | Pre-installed | 4 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| Nsight Compute | Production | Pre-installed | 4 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| CUDA-GDB | Production | Pre-installed | 4 | 5 | 5 | Optional | [NVIDIA-OFFICIAL] |
| PyTorch Profiler | Stable | Via PyTorch | 4 | 4 | 4 | Optional | [COMMUNITY] |

### Nsight Systems

See [Section 1: Nsight Systems](#nsight-systems) for full 14-field entry. Use for system-wide GPU application profiling.

### Nsight Compute

See [Section 1: Nsight Compute](#nsight-compute) for full 14-field entry. Use for detailed CUDA kernel profiling.

### CUDA-GDB

1. **Purpose**: CUDA-aware debugger for debugging GPU kernels and host code simultaneously.
2. **Why It Matters**: Only debugger that can step through CUDA kernel code; inspect thread-level GPU state.
3. **DGX Spark Relevance**: Pre-installed with CUDA toolkit; essential for custom CUDA kernel development on sm_121.
4. **Architecture Compatibility**:
   - ARM64: ✅ Native aarch64 build
   - CUDA 13: ✅ Full CUDA 13 debugging support
   - sm_121: ✅ Blackwell GPU debugging
   - Container: ✅ Works in GPU containers with --privileged
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — GDB familiarity required; CUDA-specific commands and thread navigation
7. **Best Use Cases**: Debugging CUDA kernel crashes, memory access violations, race conditions in GPU code
8. **Bad Use Cases**: Python-level debugging (use pdb/debugpy), performance profiling (use Nsight tools)
9. **Install Complexity**: Trivial (pre-installed with CUDA toolkit)
10. **Dependencies & Likely Conflicts**: CUDA toolkit; may need -G flag during compilation for debug symbols
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### PyTorch Profiler

1. **Purpose**: Built-in PyTorch profiling tool for analyzing CPU/GPU operations, memory usage, and operator timings.
2. **Why It Matters**: Profile PyTorch models without external tools; integrates with TensorBoard for visualization.
3. **DGX Spark Relevance**: Included with PyTorch; works with sm_120-compatible wheels on DGX Spark; essential for optimizing training/inference.
4. **Architecture Compatibility**:
   - ARM64: ✅ Via PyTorch (sm_120 compatible wheels)
   - CUDA 13: ⚠️ Depends on PyTorch CUDA compatibility
   - sm_121: ⚠️ Via sm_120 binary compatibility
   - Container: ✅ Works in NGC PyTorch container
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — profiler context manager, trace analysis, TensorBoard integration
7. **Best Use Cases**: PyTorch model optimization, identifying slow operators, memory profiling, training bottleneck analysis
8. **Bad Use Cases**: Non-PyTorch CUDA code (use Nsight tools), system-level profiling
9. **Install Complexity**: Trivial (included with PyTorch)
10. **Dependencies & Likely Conflicts**: PyTorch; torch-tb-profiler for TensorBoard visualization
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 20. Cross-Cutting: Financial / Research Libraries

Libraries spanning multiple categories, commonly used for financial analysis, quantitative research, and algorithmic trading.

| Library | Maturity | Install | Practicality | Stability | Compat. | Stack | Confidence |
|---------|----------|---------|:---:|:---:|:---:|-------|------------|
| yfinance | Stable | pip install | 4 | 3 | 5 | Optional | [INFERRED] |
| alpaca-py | Stable | pip install | 4 | 4 | 5 | Optional | [INFERRED] |
| pandas-ta | Stable | pip install | 4 | 4 | 5 | Optional | [INFERRED] |
| vectorbt | Stable | pip install | 4 | 3 | 5 | Optional | [INFERRED] |
| PyPortfolioOpt | Stable | pip install | 4 | 4 | 5 | Optional | [INFERRED] |
| backtrader | Stable | pip install | 3 | 4 | 5 | Optional | [INFERRED] |
| zipline-reloaded | Stable | pip install | 3 | 3 | 4 | Optional | [INFERRED] |
| QuantLib | Production | pip / build | 4 | 5 | 4 | Optional | [INFERRED] |

### yfinance

1. **Purpose**: Python library for downloading financial data from Yahoo Finance (prices, fundamentals, options).
2. **Why It Matters**: Simplest free source for historical and real-time financial data; widely used in quantitative research.
3. **DGX Spark Relevance**: Pure Python; works on aarch64; good data source for financial ML models running on DGX Spark GPU.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low
7. **Best Use Cases**: Historical price data, fundamentals research, options chain analysis, data collection for ML
8. **Bad Use Cases**: Real-time trading (rate limits), reliable production data feed (Yahoo may change API)
9. **Install Complexity**: Trivial (pip install yfinance)
10. **Dependencies & Likely Conflicts**: pandas, requests; no conflicts
11. **Practicality Score**: 4
12. **Stability Score**: 3
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### alpaca-py

1. **Purpose**: Official Python SDK for Alpaca Markets — commission-free stock/crypto trading and market data API.
2. **Why It Matters**: Programmatic trading with paper trading mode; real-time and historical market data.
3. **DGX Spark Relevance**: Pure Python; enables algorithmic trading systems powered by GPU-accelerated ML models.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — REST API wrapper with clear documentation
7. **Best Use Cases**: Algorithmic trading, paper trading, market data collection, portfolio management
8. **Bad Use Cases**: Non-US markets, institutional-grade trading (use dedicated platforms)
9. **Install Complexity**: Trivial (pip install alpaca-py)
10. **Dependencies & Likely Conflicts**: pydantic, httpx; potential pydantic version conflict with FastAPI — manage via venv
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### pandas-ta

1. **Purpose**: Technical analysis library with 130+ indicators built on pandas DataFrames.
2. **Why It Matters**: Comprehensive technical indicator library; integrates directly with pandas/cuDF DataFrames.
3. **DGX Spark Relevance**: Pure Python/pandas; indicators can be computed on GPU via cudf.pandas accelerator for large datasets.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A (GPU via cudf.pandas)
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — pandas extension API
7. **Best Use Cases**: Computing technical indicators, feature engineering for financial ML, backtesting signal generation
8. **Bad Use Cases**: Non-financial time series (use scipy/statsmodels)
9. **Install Complexity**: Trivial (pip install pandas-ta)
10. **Dependencies & Likely Conflicts**: pandas; no conflicts
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### vectorbt

1. **Purpose**: High-performance backtesting and portfolio analytics library with vectorized operations.
2. **Why It Matters**: Orders of magnitude faster than event-driven backtesting; vectorized computations leverage NumPy/pandas.
3. **DGX Spark Relevance**: Pure Python with NumPy; can benefit from Grace CPU for large backtests; data prep can use cuDF.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python + NumPy
   - CUDA 13: ✅ N/A (CPU-based; data prep can use cuDF)
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — vectorized thinking, portfolio construction API
7. **Best Use Cases**: Fast strategy backtesting, portfolio optimization, parameter sweeps over strategies
8. **Bad Use Cases**: Event-driven backtesting with complex order types (use backtrader), live trading
9. **Install Complexity**: Easy (pip install vectorbt)
10. **Dependencies & Likely Conflicts**: numba (has aarch64 wheel); plotly for visualization
11. **Practicality Score**: 4
12. **Stability Score**: 3
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### PyPortfolioOpt

1. **Purpose**: Portfolio optimization library implementing mean-variance optimization, Black-Litterman, and hierarchical risk parity.
2. **Why It Matters**: Standard portfolio construction algorithms in a clean Python API.
3. **DGX Spark Relevance**: Pure Python with scipy; works on aarch64; can optimize portfolios alongside GPU-based analysis.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python + scipy (aarch64 wheel available)
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Low — clean API with sensible defaults
7. **Best Use Cases**: Portfolio optimization, risk analysis, asset allocation, mean-variance frontier
8. **Bad Use Cases**: Real-time portfolio rebalancing (too slow), complex multi-asset derivatives
9. **Install Complexity**: Trivial (pip install pyportfolioopt)
10. **Dependencies & Likely Conflicts**: scipy, cvxpy; cvxpy may have solver dependencies on aarch64
11. **Practicality Score**: 4
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### backtrader

1. **Purpose**: Event-driven backtesting framework for trading strategies with live trading support.
2. **Why It Matters**: Realistic backtesting with order simulation, slippage, commissions; supports live trading brokers.
3. **DGX Spark Relevance**: Pure Python; works on aarch64; slower than vectorbt but more realistic simulation.
4. **Architecture Compatibility**:
   - ARM64: ✅ Pure Python
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Works everywhere
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: Medium — strategy class hierarchy, broker/feed concepts, indicator system
7. **Best Use Cases**: Realistic strategy backtesting with order simulation, multi-timeframe strategies, live trading integration
8. **Bad Use Cases**: High-speed parameter sweeps (vectorbt faster), simple indicator testing
9. **Install Complexity**: Trivial (pip install backtrader)
10. **Dependencies & Likely Conflicts**: matplotlib optional; minimal dependencies
11. **Practicality Score**: 3
12. **Stability Score**: 4
13. **Compatibility Score**: 5
14. **Stack Classification**: Optional

### zipline-reloaded

1. **Purpose**: Community-maintained fork of Quantopian's Zipline backtesting framework.
2. **Why It Matters**: Event-driven backtesting with pipeline API for factor-based strategies; former Quantopian standard.
3. **DGX Spark Relevance**: Python with C extensions; aarch64 compatibility may require building some dependencies from source.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ Some C extensions may need compilation; not all wheels available for aarch64
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker build recommended
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Stable
6. **Learning Curve**: High — Zipline API, data bundles, pipeline framework
7. **Best Use Cases**: Factor-based backtesting, institutional-style research, Quantopian-migration
8. **Bad Use Cases**: Simple strategy testing (backtrader/vectorbt simpler), real-time trading
9. **Install Complexity**: Moderate (pip install zipline-reloaded; may need bcolz/TA-Lib from source on aarch64)
10. **Dependencies & Likely Conflicts**: bcolz, TA-Lib, empyrical — some may need source builds on aarch64
11. **Practicality Score**: 3
12. **Stability Score**: 3
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

### QuantLib

1. **Purpose**: Comprehensive quantitative finance library for derivatives pricing, risk management, and fixed income analytics.
2. **Why It Matters**: Industry-standard library for quantitative finance; extensive model coverage for options, bonds, and exotic instruments.
3. **DGX Spark Relevance**: C++ library with Python wrapper (QuantLib-Python); builds on aarch64 but may require source compilation.
4. **Architecture Compatibility**:
   - ARM64: ⚠️ QuantLib C++ compiles on aarch64; pip wheel may or may not be available
   - CUDA 13: ✅ N/A
   - sm_121: ✅ N/A
   - Container: ✅ Docker build recommended for clean compilation
   - OS: ✅ Ubuntu 24.04
5. **Maturity Level**: Production
6. **Learning Curve**: High — extensive API surface, quantitative finance domain knowledge required
7. **Best Use Cases**: Derivatives pricing, yield curve construction, risk analytics, exotic options
8. **Bad Use Cases**: Simple portfolio analysis (PyPortfolioOpt simpler), equity-only strategies
9. **Install Complexity**: Moderate (pip install QuantLib) / Build-from-source (if no aarch64 wheel)
10. **Dependencies & Likely Conflicts**: Boost C++ libraries for source build; SWIG for Python bindings
11. **Practicality Score**: 4
12. **Stability Score**: 5
13. **Compatibility Score**: 4
14. **Stack Classification**: Optional

---

## 21. Quality Template

### Summary

This catalog covers **90+ tools** across 19 categories plus a financial/research cross-cutting section. Each entry includes all 14 mandatory fields with specific DGX Spark compatibility information. The catalog reflects the critical constraint that most pip packages ship CUDA 12.x/x86 wheels only, making container-first workflow essential for GPU workloads.

**Key Themes**:
- Container-first is the safest path for GPU workloads
- Pure Python tools work universally on aarch64
- NVIDIA-native tools (NIM, Ollama, RAPIDS, NemoClaw) have the best compatibility
- sm_120 PyTorch wheels are binary-compatible with sm_121
- Flash Attention is broken on sm_121; use PyTorch native SDPA with cuDNN 9.13

### Practical Implications

1. **Default Stack** tools (Ollama, Docker, FastAPI, PostgreSQL, Redis, Polars, etc.) should be installed immediately on any DGX Spark.
2. **Optional Stack** tools should be evaluated per-project; install when the use case arises.
3. **Avoid-Delay** tools (Android Studio, Milvus on single-node, etc.) should be deferred until re-evaluation triggers are met.
4. For any tool not in this catalog: test in a container first before committing to native installation.

### Recommended Actions

1. Install all Default Stack tools per the [Recommended Core Stack](recommended-core-stack.md)
2. Review the [Avoid / Delay List](avoid-delay-list.md) before installing any flagged tool
3. Use the [Decision Trees](../../decision-trees/tool-selection.md) for choosing between similar tools
4. For any new tool, check: (a) ARM64 wheel available? (b) CUDA 13 compatible? (c) NGC container available?

### Confidence Level

- **NVIDIA-Native tools**: HIGH — based on official NVIDIA documentation and pre-installed software
- **LLM Tooling**: HIGH for Ollama, MEDIUM for vLLM/TGI — based on community testing
- **Pure Python tools**: HIGH — architecture-independent by nature
- **System tools** (Docker, PostgreSQL, Redis): HIGH — well-established ARM64 support
- **Financial libraries**: MEDIUM — inferred compatibility, not DGX Spark-specific tested
- **Desktop frameworks**: MEDIUM — ARM64 Linux is less tested than x86 Linux for GUI applications

### Source Notes

| Source | Type | Reliability |
|--------|------|-------------|
| NVIDIA DGX Spark documentation | [NVIDIA-OFFICIAL] | Authoritative |
| NVIDIA NGC catalog | [NVIDIA-OFFICIAL] | Authoritative |
| Community DGX Spark testing reports | [COMMUNITY] | High |
| Package README / architecture support pages | [INFERRED] | Medium |
| General ARM64 Linux compatibility data | [INFERRED] | Medium |

### Unresolved Questions

1. **Flash Attention on sm_121**: When will upstream Flash Attention support sm_121 natively? Tracking GitHub issue.
2. **bitsandbytes aarch64**: Is QLoRA via bitsandbytes fully functional on sm_121, or does it need a source build?
3. **JAX CUDA 13**: When will official jaxlib cu130 aarch64 wheels be available on PyPI?
4. **Milvus GPU index**: Does Milvus GPU index work on sm_121, or does it embed older SM kernels?
5. **QuantLib aarch64 wheel**: Is there a maintained aarch64 PyPI wheel, or is source build always required?
6. **Electron ARM64 Linux stability**: How stable are Electron ARM64 Linux builds for complex applications?
7. **conda-forge CUDA 13**: When will conda-forge channels provide cu130 packages for aarch64?
