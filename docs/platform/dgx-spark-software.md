# NVIDIA DGX Spark — Software Stack & Bundle Reference

*Last Updated: March 2026*

This document summarizes what ships with NVIDIA DGX Spark, how the pre-installed software fits together, and the commands and workflows operators and developers use most often. Use it as a single reference when planning workloads, pulling containers, or troubleshooting GPU and CUDA issues on Grace Blackwell systems.

---

## What's Included with Purchase

| Item | Description |
|------|-------------|
| **Hardware** | NVIDIA DGX Spark hardware unit |
| **License** | 90-day **NVIDIA AI Enterprise – DGX Spark** license (included at no extra cost) |
| **Training** | **$90 DLI (Deep Learning Institute)** hands-on AI course credit (included) |
| **Software image** | **Pre-installed DGX OS** with a complete AI software stack |

Together, these give you a turn-key AI development and inference platform without assembling drivers, CUDA, and container tooling manually.

---

## Operating System — DGX OS 7.4.0

DGX Spark runs **DGX OS 7.4.0**, NVIDIA’s distribution tuned for DGX systems.

- **Base**: Ubuntu **24.04 LTS**
- **Customizations**: NVIDIA GPU drivers, kernel optimizations, and GPU management utilities integrated with the stack
- **Architecture**: **ARM64 (`aarch64`)** — every binary, container image, and wheel you install must be built for ARM. x86-only artifacts will not run natively.
- **Packages**: Standard **Ubuntu `apt`** workflow, augmented with **NVIDIA repositories** for drivers, CUDA-related packages, and NVIDIA tooling

**Actionable notes**

- Prefer NVIDIA-documented package sources and DGX-specific guidance when upgrading drivers or CUDA components.
- When searching for wheels or third-party binaries, explicitly filter for **aarch64** / **arm64**.

---

## Pre-installed Core Software

### CUDA Toolkit 13.0

Blackwell GPUs on this platform require a CUDA stack aligned with **sm_121** support.

- **CUDA 13.0** is the current line expected for this hardware generation.
- **Components** typically include: **nvcc**, **cuBLAS**, **cuDNN 9.13**, **cuFFT**, **cuRAND**, **cuSPARSE**, **cuSOLVER**, and related runtime libraries.
- **Compatibility warning**: Applications or containers built against **CUDA 12.x** may fail at runtime with errors such as:
  - `libcudart.so.12: cannot open shared object file`
- **Mitigation**: Rebuild against CUDA 13.x, use NGC images tagged for CUDA 13 / Blackwell, or install matching 12.x runtimes only if your workload explicitly requires them and NVIDIA documents that combination as supported.

### Docker + NVIDIA Container Toolkit

Container workflows are first-class on DGX Spark.

- **Docker Engine** is pre-installed.
- **NVIDIA Container Runtime** is configured (commonly reflected in **`/etc/docker/daemon.json`**).
- **GPU passthrough** is enabled by default for properly configured runs (`--gpus=all` or equivalent).

**Post-install for interactive users**

Add your user to the `docker` group so you can run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

**Quick GPU-in-container test**

```bash
docker run -it --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

If this prints GPU information inside the container, the NVIDIA Container Toolkit and driver integration are working.

### Ollama (Pre-installed)

- **Purpose**: Local **LLM inference** with minimal setup.
- **Optimization**: Tuned for **Grace Blackwell** on DGX Spark.
- **Models**: **GGUF** format is supported.
- **Default API**: **`http://localhost:11434`**

Typical flow: pull a model, then run or call the HTTP API from your applications.

### NVIDIA NIM (Inference Microservices)

- **What it is**: **Containerized**, NVIDIA-optimized inference for LLMs and other models.
- **Engines**: Pre-built optimized engines for popular models.
- **APIs**: **OpenAI-compatible** HTTP endpoints where applicable.
- **Distribution**: Images and assets are pulled from the **NGC** registry (`nvcr.io`).

Use NIM when you want production-style serving with NVIDIA-maintained containers rather than ad-hoc inference scripts.

### Python 3.x

- **System Python** with **pip** for quick scripts and tooling.
- **Conda / Miniconda** can be used for isolated environments and reproducible dependency sets (install per your org’s policy).

When mixing system Python and conda, keep CUDA and PyTorch/TensorFlow installs aligned with your chosen stack (system vs. container vs. conda env).

---

## NVIDIA AI Enterprise License (90-day)

The bundled **90-day NVIDIA AI Enterprise – DGX Spark** license unlocks enterprise-oriented software and support during the evaluation period.

**What it typically unlocks**

- Access to the **NVIDIA AI Enterprise** software suite and associated entitlements
- **Enterprise-grade support** channels (per your agreement)
- **Optimized NGC containers and models** where license-gated
- **NVIDIA NIM** deployment tooling and related enterprise assets

**After 90 days**

- **Open-source and community tools** (e.g. **Ollama**, **llama.cpp**, **vLLM**) can generally still be used according to their own licenses and your infrastructure policy.
- **Enterprise NIM containers** and some NGC assets may **require license renewal** or alternate entitlements — verify against your NVIDIA account and contract before relying on them in production.

---

## DGX Dashboard

A **pre-installed web application** for operating the system.

- **Access**: From a browser on the DGX or **remotely** (subject to network and security configuration).
- **JupyterLab**: Integrated launch path for **JupyterLab** from the dashboard where enabled.
- **Monitoring**: **GPU utilization**, **memory**, **temperature**, **storage**, and related health signals.
- **Services**: **Service management** for common stack components (exact features depend on DGX OS version).

Use the dashboard as the first stop for “is the machine healthy?” and “how do I open Jupyter?” style tasks.

---

## NVIDIA Sync Desktop App

**NVIDIA Sync** is a **desktop application** for connecting to DGX Spark from another workstation.

- **SSH**: Helps **configure SSH connections** automatically.
- **Tool launch**: Can open **VS Code**, **Terminal**, **DGX Dashboard**, **JupyterLab**, and related workflows from the client.
- **Platforms**: **macOS**, **Windows**, **Linux**

It reduces friction for developers who split time between a laptop and the DGX.

---

## NGC Container Registry

The **NVIDIA GPU Cloud** container registry lives at **`nvcr.io`**.

**Authentication**

```bash
docker login nvcr.io
```

- **Username**: `$oauthtoken` (literal token username pattern used with NGC)
- **Password**: Your **NGC API key**

**Content**

- Pre-optimized images for **Grace Blackwell**: **PyTorch**, **TensorFlow**, **RAPIDS**, **NeMo**, and more.
- **Recommended PyTorch image (example)**: `nvcr.io/nvidia/pytorch:25.11-py3` — often cited as a current choice for Spark; always confirm the latest tag in NGC release notes for your DGX OS / driver combo.

**Architecture rule**

- **Always use ARM64 (`aarch64`) images.** Pulling **x86_64** images leads to **`exec format error`** on ARM hosts.

---

## NVIDIA AI Workbench

**NVIDIA AI Workbench** is a **free development environment manager**.

- **Installation**: Guided, **click-through** style setup on supported clients.
- **Stack management**: Helps manage **NVIDIA drivers**, **Container Toolkit**, and **GPU runtime** in supported configurations.
- **Git**: **Branching and merging** integrated into project workflows.
- **Compose**: **Docker Compose** for **multi-container** projects.
- **Hybrid deployment**: **Local / cloud** patterns via **NVIDIA Brev** integration where applicable.
- **Sharing**: **Single-user URLs** for rapid prototyping and demos (subject to security practices).

Useful when you want a project-centric UI over raw `docker run` and manual environment wiring.

---

## Docker Model Runner

**Docker Model Runner** streamlines **model deployment** through the **Docker CLI**.

- Install the **Docker Model CLI plugin** via **`apt`** or the official Docker installation script (follow current NVIDIA / Docker documentation for DGX OS).
- Provides **simpler model lifecycle** (pull, run, manage) **without** hand-authored container definitions for common cases.

Pair with NGC or other registries as supported by your plugin and policy.

---

## JupyterLab

- **Entry points**: **DGX Dashboard** (where integrated) or **standalone** install.
- **Remote access**: **SSH port forwarding** or **NVIDIA Sync** depending on your setup.
- **GPUs**: Notebooks can use **GPU acceleration** when kernels and libraries match your CUDA stack.

**Launch example (local listen, no browser)**

```bash
jupyter lab --no-browser --port=8888
```

**Conda kernel registration (example)**

```bash
conda install ipykernel && python -m ipykernel install
```

Then select the registered kernel inside JupyterLab.

---

## AI Frameworks (Available via NGC / pip)

| Area | Notes |
|------|--------|
| **PyTorch** | Strong path via **NGC** images optimized for **Blackwell**; match CUDA major inside the image. |
| **TensorFlow** | Available via NGC and pip; verify **ARM64** wheels or containers. |
| **JAX** | Supported ecosystem; NVIDIA publishes **optimization playbooks** — consult current docs for Spark. |
| **NVIDIA NeMo** | **Agent toolkit**, **fine-tuning**, and speech/NLP stacks where applicable. |
| **NVIDIA RAPIDS** | **cuDF**, **cuML**, **cuOpt** for GPU DataFrame and ML workflows. |
| **Hugging Face Transformers** | Widely used; confirm **ARM64** compatibility for native installs or run inside **ARM64** NGC/dev containers. |

When in doubt, **run frameworks inside an ARM64 NGC container** that matches your CUDA/driver expectations rather than fighting mixed host libraries.

---

## Key System Commands

```bash
# Check GPU status
nvidia-smi

# Check CUDA version (compiler)
nvcc --version

# Check Docker GPU access
docker run -it --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi

# Check NVIDIA Container Toolkit
nvidia-ctk --version

# View Docker daemon configuration (NVIDIA runtime, etc.)
cat /etc/docker/daemon.json

# Pull optimized PyTorch container (ARM64 — use NGC tag appropriate for your environment)
docker pull nvcr.io/nvidia/pytorch:25.11-py3

# Launch container with GPU and project directory mounted
docker run -it --gpus=all -v /path/to/project:/workspace nvcr.io/nvidia/pytorch:25.11-py3

# NGC login
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-ngc-api-key>

# Ollama — list models, pull, run
ollama list
ollama pull llama3.1
ollama run llama3.1
```

---

## Quick Troubleshooting Pointers

| Symptom | Likely cause | Direction |
|--------|----------------|-----------|
| `libcudart.so.12` missing | Binary built for CUDA 12 | Move to CUDA 13 toolchain / matching container |
| `exec format error` in Docker | **x86_64** image on **ARM64** host | Pull **`aarch64`** / ARM-tagged NGC images |
| Docker permission denied | User not in `docker` group | `usermod -aG docker` and re-login / `newgrp` |
| Blank GPU in container | Runtime not used or wrong image | Confirm `--gpus=all` and NVIDIA runtime in `daemon.json` |

---

## Document maintenance

- NVIDIA periodically updates **DGX OS**, **driver**, **CUDA**, and **NGC** tags. Treat version numbers in this doc (e.g. **DGX OS 7.4.0**, **CUDA 13.0**, **PyTorch 25.11**) as **reference points**; confirm the latest supported matrix in **NVIDIA DGX Spark** and **NGC** documentation before production deployments.
