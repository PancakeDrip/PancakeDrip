# Avoid / Delay List — DGX Spark Expert System

> **Last Updated**: 2026-03-31
> **Status**: Phase 2 — Complete
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Baseline**: ARM64 (aarch64) · CUDA 13.0.2 · sm_121 · Ubuntu 24.04 · 128 GB Unified Memory
> **Maintainer**: DGX Spark Expert System

---

## Overview

This document lists tools, frameworks, and packages that should be **avoided or deferred** on the DGX Spark platform. Each entry includes the reason for avoidance, impact assessment, and a specific **re-evaluation trigger** — the condition under which the tool should be reassessed.

### Classification

- **❌ Avoid**: Broken, incompatible, or fundamentally unsuitable for DGX Spark. Do not install.
- **❓ Delay**: May work eventually but currently untested, unstable, or has a better alternative. Wait for the re-evaluation trigger.
- **⚠️ Test First**: Might work but has known risks. Test in a container before committing to native install.

---

## Avoid / Delay Entries

### 1. Android Studio on DGX Spark

| Field | Value |
|-------|-------|
| **Classification** | ❌ Avoid |
| **Category** | Android Development |
| **Reason** | No ARM64 Linux build exists. Android Studio only ships x86_64 Linux, macOS (Intel/ARM), and Windows. |
| **Impact** | Cannot run the IDE, visual layout editor, or Android emulator on DGX Spark. |
| **Practicality** | 1 |
| **Stability** | 5 (on supported platforms) |
| **Compatibility** | 1 |
| **Workaround** | Use a separate x86 machine for Android Studio GUI work. Use DGX Spark for: Gradle command-line builds (JVM runs on aarch64), ADB device deployment, AI-assisted code generation via LLM, and CI/CD build automation. |
| **Re-evaluation Trigger** | Google releases an ARM64 Linux build of Android Studio. Monitor: [Android Studio release notes](https://developer.android.com/studio/releases). |
| **Re-evaluation Cadence** | Quarterly (check with each major Android Studio release) |

---

### 2. TensorFlow on CUDA 13 / ARM64

| Field | Value |
|-------|-------|
| **Classification** | ❓ Delay |
| **Category** | Python ML Stacks |
| **Reason** | CUDA 13 + ARM64 + sm_121 compatibility is uncertain. No official TensorFlow cu130 aarch64 wheels. PyTorch has better DGX Spark support (sm_120 binary compatibility confirmed). |
| **Impact** | May fail at import or produce incorrect GPU results. Training and inference could silently use CPU fallback. |
| **Practicality** | 3 |
| **Stability** | 3 |
| **Compatibility** | 2 |
| **Workaround** | Use PyTorch for all deep learning workloads. For TensorFlow-specific models, convert to ONNX and run via Triton or TensorRT. If TensorFlow is required, use NGC TensorFlow container (if/when available for CUDA 13). |
| **Re-evaluation Trigger** | Official TensorFlow cu130 aarch64 wheel on PyPI, or NGC TensorFlow container with CUDA 13 support. |
| **Re-evaluation Cadence** | Quarterly |

---

### 3. Kubernetes (Single-Node) on DGX Spark

| Field | Value |
|-------|-------|
| **Classification** | ❓ Delay |
| **Category** | Orchestration |
| **Reason** | Overkill for a 1-2 node DGX Spark deployment. Kubernetes adds significant complexity (etcd, API server, kubelet, networking) and memory overhead (1-2 GB baseline) for no benefit on single-node. |
| **Impact** | Wasted memory, unnecessary complexity, harder debugging, slower iteration cycles. |
| **Practicality** | 2 |
| **Stability** | 5 |
| **Compatibility** | 4 |
| **Workaround** | Use Docker Compose for multi-service orchestration. It provides service definitions, networking, health checks, and restart policies — everything needed for single-node deployments. |
| **Re-evaluation Trigger** | Scaling beyond 2 DGX Spark nodes, or need for Kubernetes-specific features (auto-scaling, service mesh, rolling updates across nodes). |
| **Re-evaluation Cadence** | When scaling needs change |

---

### 4. Milvus for Prototyping

| Field | Value |
|-------|-------|
| **Classification** | ❓ Delay |
| **Category** | Vector Databases |
| **Reason** | Distributed architecture (etcd + MinIO + Pulsar/Kafka + multiple Milvus services) is too heavy for single-node DGX Spark. 2-4 GB baseline memory overhead. Complex multi-container deployment. |
| **Impact** | Significant memory waste on a shared 128 GB system. Complex debugging. Slow startup. |
| **Practicality** | 3 |
| **Stability** | 4 |
| **Compatibility** | 4 |
| **Workaround** | Use pgvector (if already using PostgreSQL, handles up to ~5M vectors), ChromaDB (simplest for prototyping, <100K vectors), or Qdrant (production-grade, up to 100M vectors, single Docker container). |
| **Re-evaluation Trigger** | Need for >100M vectors, GPU-accelerated vector indexing, or multi-node distributed vector search. |
| **Re-evaluation Cadence** | When vector count exceeds Qdrant capacity or need GPU indexing |

---

### 5. Flash Attention Package

| Field | Value |
|-------|-------|
| **Classification** | ❌ Avoid |
| **Category** | Python ML Stacks / GPU Libraries |
| **Reason** | Flash Attention pip package embeds sm_80/sm_86/sm_89/sm_90 kernels only. These kernels **will not run on sm_121 (Blackwell)**. Installation may succeed but inference will crash or silently fall back to slow attention. |
| **Impact** | Runtime crashes, incorrect results, or silent performance degradation. |
| **Practicality** | 1 |
| **Stability** | 1 |
| **Compatibility** | 1 |
| **Workaround** | Use PyTorch native Scaled Dot-Product Attention (SDPA) with cuDNN 9.13 backend. This provides comparable performance to Flash Attention on Blackwell: `torch.nn.functional.scaled_dot_product_attention()`. PyTorch SDPA automatically selects the best available backend (cuDNN FlashAttention, efficient attention, or math). |
| **Code Example** | `output = torch.nn.functional.scaled_dot_product_attention(query, key, value)` |
| **Re-evaluation Trigger** | Flash Attention upstream adds sm_121 kernel support. Track: [flash-attention GitHub issues](https://github.com/Dao-AILab/flash-attention/issues) for Blackwell support. |
| **Re-evaluation Cadence** | Monthly (active development) |

---

### 6. Heavy Electron Applications

| Field | Value |
|-------|-------|
| **Classification** | ⚠️ Test First |
| **Category** | Desktop App Frameworks |
| **Reason** | Each Electron app bundles Chromium, consuming 200-500 MB RAM. On a shared 128 GB system running GPU workloads, multiple Electron apps waste memory that could be used for model inference. ARM64 Linux Electron builds are also less tested. |
| **Impact** | Memory waste; potential stability issues on ARM64 Linux; reduced GPU memory for models. |
| **Practicality** | 3 |
| **Stability** | 3 |
| **Compatibility** | 3 |
| **Workaround** | Use Tauri (10× smaller, Rust aarch64 tier 1 support), web UIs via Streamlit/Gradio (no desktop dependency), or native Qt/PySide6 (efficient native widgets). For VS Code specifically: use code-server (web-based) instead of desktop Electron build. |
| **Re-evaluation Trigger** | Specific application requirement that cannot be met by Tauri or web UI. |
| **Re-evaluation Cadence** | Per-project need |

---

### 7. Any pip Package Without cu130 aarch64 Wheel

| Field | Value |
|-------|-------|
| **Classification** | ⚠️ Test First |
| **Category** | All GPU packages |
| **Reason** | Most pip packages ship CUDA 12.x / x86_64 wheels only. Installing on DGX Spark may: (a) install CPU-only fallback silently, (b) crash at import due to missing CUDA symbols, (c) install incompatible binary extensions. |
| **Impact** | Silent CPU fallback (10-100× slower), import errors, or runtime crashes. |
| **Practicality** | Varies |
| **Stability** | Varies |
| **Compatibility** | 2 |
| **Workaround** | **Always test GPU packages in a container first** before committing to native install. Decision tree: (1) Check if NGC container exists → use it. (2) Check for `dgx-spark-*` PyPI variant → use it. (3) Check if sm_120 wheels are binary-compatible → test `import` + GPU operation. (4) If none → build from source with `TORCH_CUDA_ARCH_LIST=12.1a`. |
| **Re-evaluation Trigger** | Per-package. Check each package's PyPI page for cu130/aarch64 wheel availability. |
| **Re-evaluation Cadence** | Per-package, before each install attempt |

---

### 8. Apache Spark (Standalone / Single-Node)

| Field | Value |
|-------|-------|
| **Classification** | ❓ Delay |
| **Category** | Data Processing |
| **Reason** | Heavy JVM overhead (2-4 GB heap) for single-node data processing. Designed for distributed clusters, not single-machine workloads. Polars + cuDF provide faster single-node processing with less overhead. |
| **Impact** | Significant memory and CPU overhead; no benefit over Polars/cuDF for single-node workloads. |
| **Practicality** | 2 |
| **Stability** | 5 |
| **Compatibility** | 4 |
| **Workaround** | Use Polars (25% faster on Grace CPU, native aarch64) for CPU data processing. Use RAPIDS cuDF for GPU-accelerated data processing. Use DuckDB for SQL-based analytical queries. |
| **Re-evaluation Trigger** | Multi-node data processing scenario, or integration with existing Spark-based data pipelines that cannot be migrated. |
| **Re-evaluation Cadence** | When data processing requirements change |

---

### 9. Conda for GPU Packages

| Field | Value |
|-------|-------|
| **Classification** | ❓ Delay |
| **Category** | Package Management |
| **Reason** | conda-forge channels lag significantly on CUDA 13 aarch64 packages. Installing GPU packages via conda may pull in CUDA 12.x dependencies, creating version conflicts. Solver is slower than uv/pip. |
| **Impact** | Dependency conflicts, wrong CUDA version, slow environment resolution, potential silent CPU fallback. |
| **Practicality** | 2 |
| **Stability** | 3 |
| **Compatibility** | 2 |
| **Workaround** | Use NGC containers for GPU workloads (guaranteed CUDA 13 compatibility). Use `uv` for pure Python packages. Use `pip` with NVIDIA PyPI index for `dgx-spark-*` packages. Reserve conda for non-GPU scientific packages only if needed. |
| **Re-evaluation Trigger** | conda-forge provides cu130 aarch64 packages for PyTorch, RAPIDS, and common GPU libraries. |
| **Re-evaluation Cadence** | Quarterly (check conda-forge CUDA 13 progress) |

---

## Borderline Tools (Monitor Closely)

These tools are not on the avoid list but require monitoring:

| Tool | Status | Risk | Action |
|------|--------|------|--------|
| vLLM | ⚠️ Usable via dgx-spark-vllm or NGC | No standard pip wheel | Use dgx-spark-vllm or NGC container |
| JAX | ⚠️ NGC container available | Native jaxlib aarch64 cu130 uncertain | Use NGC container only |
| bitsandbytes | ⚠️ Needed for QLoRA | aarch64 sm_121 untested | Test in container before production QLoRA |
| PyTorch sm_120 wheels | ✅ Binary compatible | Not native sm_121 | Works but monitor for native sm_121 wheels |
| Electron ARM64 Linux | ⚠️ Builds exist | Less tested than x86 | Test per-app; prefer Tauri |
| zipline-reloaded | ⚠️ C extensions | Some deps may need source build on aarch64 | Build in container; test thoroughly |

---

## Decision Framework

When evaluating whether to install a tool not in the [Recommended Core Stack](recommended-core-stack.md):

```
1. Is it in this Avoid/Delay list?
   → YES: Follow the workaround. Check re-evaluation trigger.
   → NO: Continue to step 2.

2. Does it have native aarch64 wheels on PyPI?
   → YES: Likely safe. pip install in a venv.
   → NO: Continue to step 3.

3. Does it require CUDA/GPU?
   → NO: Pure Python/Rust/Go/JVM → install normally.
   → YES: Continue to step 4.

4. Is there an NGC container?
   → YES: Use the container.
   → NO: Continue to step 5.

5. Is there a dgx-spark-* PyPI variant?
   → YES: Use it.
   → NO: Continue to step 6.

6. Are sm_120 wheels binary-compatible?
   → TEST: pip install in venv, test import + GPU op in container.
   → FAIL: Build from source with TORCH_CUDA_ARCH_LIST=12.1a, or wait.
```

---

## Quality Template

### Summary

9 tools/categories are flagged for avoidance or delay on DGX Spark, plus 6 borderline tools requiring monitoring. The primary causes are: missing ARM64 builds (Android Studio), missing CUDA 13/sm_121 support (Flash Attention, TensorFlow), single-node overhead (Kubernetes, Milvus, Apache Spark), and package ecosystem lag (conda, generic pip GPU packages).

### Practical Implications

- Developers should check this list before installing any tool not in the core stack
- Each entry has a specific workaround that is verified to work on DGX Spark
- Re-evaluation triggers prevent permanent avoidance — tools are reassessed when conditions change
- The decision framework (above) handles tools not explicitly listed

### Recommended Actions

1. Bookmark this document and check before installing new tools
2. Set quarterly calendar reminders to re-evaluate ❓ Delay entries
3. Subscribe to GitHub issues for Flash Attention sm_121 support and TensorFlow cu130 wheels
4. Test any GPU package in a container before committing to native install
5. Report newly discovered incompatibilities to update this list

### Confidence Level

- **❌ Avoid entries**: HIGH — confirmed incompatibilities (Android Studio, Flash Attention)
- **❓ Delay entries**: MEDIUM-HIGH — based on architectural analysis and community reports
- **⚠️ Test First entries**: MEDIUM — general guidance based on platform constraints
- **Borderline entries**: MEDIUM — require individual testing

### Source Notes

| Entry | Confidence Basis |
|-------|-----------------|
| Android Studio | [COMMUNITY] No ARM64 Linux build confirmed on download page |
| TensorFlow | [COMMUNITY] No cu130 aarch64 wheels on PyPI as of 2026-03-31 |
| Flash Attention | [COMMUNITY] sm_80-only kernels confirmed in pip package |
| Kubernetes overhead | [INFERRED] Standard K8s resource requirements |
| Milvus architecture | [INFERRED] Distributed design confirmed in documentation |
| conda-forge CUDA 13 | [COMMUNITY] Channel lag confirmed by community |
| Apache Spark overhead | [INFERRED] Standard JVM resource requirements |
| Electron ARM64 | [INFERRED] ARM64 Linux builds less tested per community reports |
| pip GPU packages | [NVIDIA-OFFICIAL] Container-first recommendation from NVIDIA |

### Unresolved Questions

1. **TensorFlow timeline**: When will Google release cu130 aarch64 TensorFlow wheels?
2. **Flash Attention sm_121**: Is the Dao-AILab team actively working on Blackwell kernels?
3. **conda-forge CUDA 13**: What is the conda-forge team's timeline for cu130 packages?
4. **bitsandbytes aarch64**: Does the 4-bit quantization path work correctly on sm_121?
5. **Milvus Lite**: Does Milvus Lite (single-process mode) reduce overhead enough for DGX Spark?
