# NVIDIA DGX Spark — Compatibility Guide & Known Issues

*Last Updated: March 2026*

> **⚠️ CRITICAL REFERENCE — Read before installing ANY software on DGX Spark**

This document describes architecture constraints, common failure modes, and proven workarounds. Treat it as mandatory reading for anyone deploying ML software, containers, or system packages on DGX Spark. Skipping it often wastes hours on avoidable errors.

---

## Architecture Overview

The DGX Spark combines three factors that rarely appear together in generic tutorials or vendor defaults:

| Layer | DGX Spark reality | Typical assumption (wrong here) |
|-------|-------------------|----------------------------------|
| **CPU** | **ARM64 (`aarch64`)** | x86_64 |
| **GPU** | **Blackwell `sm_121`** | Older SM targets (e.g. Ampere, Ada) |
| **CUDA** | **13.0** | CUDA 12.x wheels and containers |

**Triple constraint:** Many installation paths assume *at most one* of these mismatches. Here you get **all three** at once. Standard `pip install`, random Docker Hub images, and prebuilt binaries often **fail silently until runtime** or **fail loudly at import/launch**.

**Bottom line:** Verify **architecture + CUDA major + GPU SM** for every stack layer before you commit time to an install path.

---

## Issue #1: CUDA Version Mismatch (CRITICAL)

> **⚠️ CRITICAL** — Most common “it worked on my other machine” failure on Spark.

**Symptom**

```text
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

(or similar: missing `libcudart.so.12`, wrong `libcudnn`, etc.)

**Cause**

Python wheels from PyPI are commonly built against **CUDA 12.x**. DGX Spark ships **CUDA 13.0**. A wheel that dynamically links against CUDA 12 libraries will not find them on a 13.0-only system layout.

**Fix (pick one aligned with your workflow)**

- **Prefer NGC Docker images** for the stack you need — CUDA and libraries match what NVIDIA validates on this platform.
- **Build from source** with toolchains and flags targeting **CUDA 13.0** and **Blackwell**.
- **Use NVIDIA-provided or explicitly `cu130` wheels** when published for your package (do not assume PyPI default).
- **PyTorch:** use an NGC PyTorch container such as `nvcr.io/nvidia/pytorch:25.11-py3` rather than assuming a generic pip wheel will match.

**Prevention**

Before `pip install`:

1. Confirm **aarch64** wheels exist (if using wheels).
2. Confirm **CUDA major 13** compatibility for native extensions.
3. When in doubt, **prototype inside NGC** first, then narrow to a minimal host install if required.

---

## Issue #2: Flash Attention Incompatibility (CRITICAL)

> **⚠️ CRITICAL** — Do not burn time forcing `flash-attn` on Blackwell here.

**Symptom**

- `flash-attn` fails to **compile** (unsupported arch / codegen).
- Or it **builds** but **fails to load** at runtime (kernel / SM mismatch).

**Cause**

Flash Attention 2/3 does **not** support **`sm_121` (Blackwell)** in the way many guides assume. Project READMEs and Colab snippets are often written for older SMs.

**Fix**

- **Do not depend on `flash-attn` on DGX Spark** for production paths until upstream explicitly supports your SM in the build you use.
- Use **PyTorch native scaled dot-product attention (SDPA)** with a recent **cuDNN** stack (e.g. **cuDNN 9.13** in validated NVIDIA stacks).

**Note**

On Blackwell with a current PyTorch + cuDNN stack, **SDPA is often faster** than Flash Attention for many workloads — so this is not merely a workaround; it can be the **better** default.

**In code**

```python
# Instead of flash_attn, use PyTorch SDPA.
import torch
import torch.nn.functional as F

# Hugging Face `transformers` typically uses SDPA when flash-attn is absent.
# Explicit SDPA:
output = F.scaled_dot_product_attention(query, key, value)
```

---

## Issue #3: vLLM Installation (CRITICAL)

> **⚠️ CRITICAL** — Default wheels and generic guides miss `sm_121` + `cu130` + `aarch64`.

**Symptom**

- vLLM fails at import or when launching the engine.
- Errors mentioning **`sm_121` not supported**, missing kernels, or **CUDA version mismatch**.

**Cause**

Prebuilt vLLM artifacts often lag behind **Blackwell** and may only bundle kernels up to around **`sm_120`** (and/or wrong CUDA major for your system). Combined with **aarch64**, the “usual” install is high-risk.

**Fix**

```bash
# Option 1: Use nightly aarch64 cu130 wheels (verify URL still current in vLLM docs)
pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130/aarch64/

# Option 2: Build from source (forces correct SM codegen for your GPU)
TORCH_CUDA_ARCH_LIST="12.1a" pip install vllm --no-binary vllm

# Option 3: Use Ollama (often pre-installed on Spark images) — works out of the box for many models
```

**Operational tip:** After any vLLM upgrade, re-check wheel index URLs and release notes — nightly channels move quickly.

---

## Issue #4: Unified Memory OOM Behavior (IMPORTANT)

> **⚠️ IMPORTANT** — This can take down the **whole machine**, not just your job.

**Symptom**

- Full system **freeze** or extreme thrashing.
- SSH and local console **unresponsive**.
- Requires **hard reboot** or BMC intervention.

**Cause**

CPU and GPU share a **unified memory pool** (e.g. **128GB** on typical Spark configs). When the GPU path consumes too much of what the OS also needs, you **starve the OS** — unlike discrete-GPU desktops where system RAM is separate.

**Fix / prevention**

- **Monitor** GPU and system memory continuously during experiments.
- Set **explicit memory limits** in inference and training frameworks.
- **Ollama:** models load on demand; watch **total resident** footprint across models.
- **vLLM:** cap utilization, e.g. `--gpu-memory-utilization 0.8`, to leave headroom.
- **Training:** start with **small batch sizes** and scale up with measurement.
- Configure **swap** as a **safety net** (not a performance strategy — a last-resort buffer).
- **Kill runaway jobs early** — once the system is wedged, recovery is painful.

```bash
# Monitor memory continuously (GPU + host view)
watch -n 1 'nvidia-smi; echo "---"; free -h'

# vLLM: leave headroom below 1.0 utilization
vllm serve model_name --gpu-memory-utilization 0.8

# Ollama: limit concurrent loaded models (example)
export OLLAMA_MAX_LOADED_MODELS=1
```

---

## Issue #5: ARM64 Binary Compatibility (IMPORTANT)

> **⚠️ IMPORTANT** — “It runs in Docker” is false if the image is `amd64`-only.

**Symptom**

```text
exec format error
```

when running binaries, extracted tarballs, or containers.

**Cause**

Artifact built for **x86_64**, not **aarch64**. This includes many “official” Linux binaries that only ship amd64.

**Fix**

- **Docker:** inspect manifests before pulling; pin `platform: linux/arm64` when needed.
- **pip:** confirm **aarch64** wheels on PyPI (or build from source).
- **Raw binaries:** obtain ARM64 builds or compile on Spark (or cross-compile correctly).
- **Sanity check:** `uname -m` → should print **`aarch64`**.
- **Known gap:** some tools (e.g. **Android Studio**) have **no** official ARM64 Linux build — plan alternatives.

---

## Issue #6: NV-Ingest ARM64 Unavailability

**Symptom**

NV-Ingest containers fail with **`exec format error`** or pull the wrong arch.

**Cause**

NV-Ingest images are **not** universally published for **ARM64** yet; documentation may not emphasize this.

**Alternatives**

- **LlamaIndex** or **LangChain** document loaders for ingestion pipelines.
- **PyMuPDF**, **pdfplumber** for PDF text and layout extraction.
- **`unstructured`** ecosystem where ARM64 support exists for your components.
- Track NVIDIA releases for **official ARM64** NV-Ingest builds (community demand is ongoing).

---

## Issue #7: Python Package Compatibility Matrix

| Package | Status on DGX Spark | Notes |
|---------|---------------------|-------|
| PyTorch | ✅ Works (via NGC container) | Prefer NGC; avoid assuming pip default matches CUDA 13 + Blackwell |
| TensorFlow | ✅ Works (via NGC container) | Prefer NGC-validated stacks |
| Ollama | ✅ Pre-installed | Strong default for quick local inference |
| llama.cpp | ✅ Works | Build from source with **CUDA 13**; verify SM flags |
| vLLM | ⚠️ Needs special wheels | Prefer `wheels.vllm.ai` **nightly** `cu130` **aarch64** or source build |
| flash-attn | ❌ Does NOT work | Use **PyTorch SDPA** instead |
| transformers | ✅ Works | Uses SDPA paths when flash-attn absent |
| RAPIDS (cuDF/cuML) | ✅ Works (via NGC) | Use NGC RAPIDS images aligned to your CUDA |
| ComfyUI | ✅ Works | Follow Spark-specific playbooks / validated deps |
| LM Studio | ✅ Works | ARM64 Linux build available |
| ExLlamaV3 | ✅ Works | Blackwell support in upstream trajectory — still verify versions |
| NV-Ingest | ❌ No ARM64 build | Use alternatives above |
| Android Studio | ❌ No ARM64 Linux | Use community / alternate tooling (e.g. ARM64-ADK projects) |
| CrewAI | ✅ Works | Typical `pip install` on aarch64 |
| LangChain / LangGraph | ✅ Works | Typical `pip install`; watch native deps |
| Freqtrade | ✅ Works | Mostly Python; validate any binary wheels |
| RAPIDS cuOpt | ✅ Works (via NGC) | Match NGC image to CUDA major |

> **⚠️** Matrix entries are **practical defaults** — always confirm **version + wheel + container tag** at install time.

---

## Issue #8: Docker Container Architecture

**Always verify before pulling.**

```bash
# Check image architecture in the manifest
docker manifest inspect nvcr.io/nvidia/pytorch:25.11-py3 | grep architecture

# Pull explicitly for ARM64
docker pull --platform linux/arm64 <image>
```

**docker-compose.yml example**

```yaml
services:
  myservice:
    platform: linux/arm64
    image: myimage
```

**Why this matters:** Pulling the wrong arch often fails immediately; sometimes it “works” until you hit a binary inside the image — both waste time.

---

## Issue #9: NVFP4 Quantization (Blackwell-Specific)

**What it is**

- **NVFP4** is a **4-bit floating-point** format **native to Blackwell**.
- **Only** meaningful on **`sm_121`** — not on RTX 40-series or older GPUs in the way NVFP4 is defined here.

**Practical characteristics (typical marketing-validated ranges — measure your model)**

- Large **memory footprint reduction** vs FP16 (often cited around **3.5×** — verify per model).
- Can be **faster than FP8** on supported paths (often cited ~**1.6×** — workload dependent).
- **Accuracy:** often **<1%** degradation on **>7B** models when well tuned; **smaller models** may see **2–8%** without distillation — **always evaluate on your task**.

**Ecosystem**

- Pre-quantized NVFP4 checkpoints appear on **Hugging Face** for some models.
- Support surfaces in **vLLM**, **NVIDIA Model Optimizer**, **llama.cpp** — check **version release notes** for exact coverage.

---

## Issue #10: Cross-Compilation Toolchain

Use this when you must **build on Spark** for **other** targets, or when orchestrating multi-arch CI — not as a substitute for native aarch64 builds on Spark itself.

```bash
# Host tools for AArch64 GNU toolchain (example on Ubuntu-derived hosts)
sudo apt install cmake gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# CUDA cross-compilation packages (names vary by distro/repo — confirm for your image)
sudo apt install cuda-cross-sbsa

# Example CMake CUDA cross setup (paths must match your installation)
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/aarch64-linux-gnu-g++
```

> **⚠️** Cross-compilation is easy to get subtly wrong (sysroots, ABI, CUDA toolkit layout). Prefer **NGC** or **vendor images** when they exist.

---

## General Best Practices

1. **Default to NGC containers** for serious ML work — tested combinations of CUDA, drivers, and key libs.
2. **Always check architecture** for pip wheels, Docker images, and downloaded binaries **before** you depend on them.
3. **Monitor memory aggressively** — unified memory OOM can freeze the **entire** system.
4. **Use Ollama** for fast local inference when it meets your needs — low friction on Spark.
5. **Skip `flash-attn`** as a hard requirement — **SDPA** is the supported high-performance path here.
6. **Stay on CUDA 13.0** for the system story — do not casually install **CUDA 12.x** “beside” it without a migration plan.
7. **Test in Docker first** — contains blast radius when something is wrong.
8. **Configure swap** as a **safety net**, not primary memory.
9. **Update deliberately** — the ARM64 + Blackwell ecosystem is moving quickly; pin versions for reproducibility, upgrade with release notes.

---

## Quick Diagnostic Commands

```bash
# System architecture (expect: aarch64)
uname -m

# GPU status: model, driver, CUDA reported, memory use
nvidia-smi

# Toolkit compiler version (expect alignment with 13.x story on Spark)
nvcc --version

# Minimal GPU-in-container sanity check (adjust tag to a supported aarch64 CUDA base)
docker run --rm --gpus=all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi

# Ollama: loaded models and memory (when using Ollama)
ollama ps

# Host memory (remember overlap with GPU in unified memory setups)
free -h

# Inspect a downloaded wheel’s payload arch (illustrative)
pip download <package> --no-deps --dest /tmp && file /tmp/*.whl
```

---

## Document maintenance

When you hit a **new** failure mode:

1. Capture **exact error text**, **package versions**, and **container image digest**.
2. Note whether the root cause was **arch**, **CUDA major**, or **SM capability**.
3. Propose an addition to this guide so the next person avoids the same pitfall.

**⚠️ CRITICAL:** If any command, wheel URL, or image tag in this document drifts out of date, **verify against NVIDIA NGC tags, vLLM release notes, and your exact Spark software image** before production use.
