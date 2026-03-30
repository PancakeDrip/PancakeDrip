# Recommended Base Environment Plan

## DGX Spark Expert System — Phase 1 Platform Document

---

## 1. OS Baseline

The DGX Spark ships with **DGX OS 7.4.0**, which is built on **Ubuntu 24.04 LTS** with **kernel 6.17**. This is a curated, NVIDIA-maintained distribution and should be treated as an appliance OS.

### Core Principle: Do Not Modify DGX OS

- Never replace or upgrade the kernel, system-level NVIDIA drivers, or core system libraries.
- DGX OS updates are delivered as atomic bundles through the **DGX Dashboard** — driver, firmware, CUDA toolkit, and kernel are tested together as a unit.
- Modifications to core packages risk breaking the tightly coupled driver/firmware/kernel stack.

### Workload Isolation via Containers

All development and production workloads should run inside containers or Python virtual environments. The host OS serves as a stable platform layer.

- **Host OS role**: Run Docker, Ollama, system monitoring, SSH.
- **Containers role**: All ML frameworks, application code, custom dependencies.
- **Virtual environments role**: Lightweight Python-only workloads that don't need GPU-accelerated libraries (or use pre-installed system packages).

---

## 2. Driver / CUDA Strategy

### Current Stack

| Component       | Version        |
|-----------------|----------------|
| NVIDIA Driver   | 580.126.09     |
| CUDA Toolkit    | 13.0.2         |
| GPU Architecture| Blackwell (sm_121) |
| cuDNN           | 9.13+          |

### Update Cadence

Stay on NVIDIA's official update cadence delivered through the **DGX Dashboard**:

1. Open DGX Dashboard (accessible via browser on the device or remotely).
2. Check for system updates — these bundle driver, CUDA, firmware, and kernel patches together.
3. Apply updates during maintenance windows (requires reboot).
4. Verify post-update: `nvidia-smi` for driver, `nvcc --version` for CUDA toolkit.

### Why Not Manual Driver Updates

- The driver, firmware, CUDA toolkit, and kernel on DGX OS are co-validated. Installing a standalone driver from NVIDIA's download page can break this chain.
- DGX Dashboard updates include firmware patches (e.g., thermal throttling mitigations) that standalone driver packages do not.

---

## 3. Python Environment Strategy

The DGX Spark runs **aarch64 (ARM64) Linux** with **CUDA 13 / sm_121** — a combination that creates significant package compatibility challenges. The right Python environment tool depends on the workload type.

### Option A: pip + venv

**How it works**: Python's built-in `venv` module creates isolated environments; `pip` installs packages from PyPI.

```bash
python3.12 -m venv ~/envs/myproject
source ~/envs/myproject/bin/activate
pip install numpy pandas scikit-learn
```

**DGX Spark Assessment**:

- Works natively with the pre-installed Python 3.12 on DGX OS.
- Zero additional tooling to install.
- Sufficient for pure-Python packages and packages with aarch64 wheels on PyPI.

**Limitations**:

- No sophisticated dependency resolution — can produce broken environments with conflicting transitive dependencies.
- No lockfile support (requires `pip freeze` workarounds).
- PyPI has limited CUDA 13 wheels for GPU-accelerated packages — the same wheel gap affects pip.

**Verdict**: Acceptable for simple scripts and non-GPU workloads. Not recommended as primary environment manager.

### Option B: conda / mamba (miniforge / mambaforge)

**How it works**: Conda manages both Python and non-Python dependencies (C libraries, CUDA runtimes) in isolated environments. Mamba is a drop-in replacement with faster dependency resolution (C++ solver).

```bash
# Install miniforge for aarch64
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
# Create environment
mamba create -n datascience python=3.12 pandas scikit-learn matplotlib
```

**DGX Spark Assessment**:

- **miniforge is available for aarch64** — installs and runs natively on DGX Spark.
- Excellent for data science stacks: NumPy, Pandas, scikit-learn, Matplotlib, Jupyter all available via conda-forge for aarch64.
- Can manage non-Python dependencies (HDF5, OpenBLAS, etc.) that pip cannot.

**Critical Problem**:

- **conda-forge channels primarily ship CUDA 12.x packages, not CUDA 13.** GPU-accelerated packages (PyTorch, TensorFlow, RAPIDS, cuDF) from conda-forge will pull CUDA 12.x runtime libraries that conflict with the system's CUDA 13 stack.
- Works fine for non-GPU Python packages. Actively problematic for GPU-accelerated packages.

**Verdict**: Good choice for data-science-only environments where GPU acceleration is not needed. Do not use conda to install GPU-accelerated ML frameworks.

### Option C: Poetry

**How it works**: Poetry provides dependency resolution with a lockfile (`poetry.lock`), virtual environment management, and packaging — all in one tool.

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
# Create project
poetry new myproject && cd myproject
poetry add pandas numpy
```

**DGX Spark Assessment**:

- Pure Python tool — works fine on aarch64 without any platform-specific issues.
- Excellent dependency resolution with lockfile guarantees reproducibility.
- Good for application development with well-defined dependency trees.

**Limitations**:

- Still pulls packages from PyPI, which has the same CUDA 13 wheel gap. Poetry's superior resolver doesn't help when the wheels simply don't exist.
- Slower than modern alternatives (uv) for dependency resolution and installation.
- Does not manage non-Python dependencies.

**Verdict**: Functional but does not solve the fundamental CUDA 13 compatibility problem. Outperformed by uv on speed.

### Option D: uv

**How it works**: uv is a Rust-based Python package manager that replaces pip, venv, pip-tools, and parts of Poetry. It provides dependency resolution, lockfiles, virtual environments, and workspace support.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create project with lockfile
uv init myproject && cd myproject
uv add pandas numpy
# Or manage virtual environments directly
uv venv ~/envs/myproject
uv pip install pandas numpy
```

**DGX Spark Assessment**:

- **aarch64 binary available** — installs and runs natively on DGX Spark.
- **10-100x faster** than pip/Poetry for dependency resolution and installation (Rust implementation).
- Lockfile support (`uv.lock`) for reproducible environments.
- Workspace support for monorepos.
- Compatible with PyPI — same package ecosystem as pip.

**Limitations**:

- Same CUDA 13 wheel gap as pip/Poetry — faster installation doesn't help when GPU wheels don't exist.
- Relatively new tool (though rapidly maturing, backed by Astral/Ruff team).

**Verdict**: Best-in-class for native Python environments. Speed advantage is material when iterating on dependency configurations.

### Recommendation

| Workload Type | Recommended Tool | Reasoning |
|---------------|------------------|-----------|
| **GPU-accelerated ML** (PyTorch, vLLM, RAPIDS) | **Docker / NGC containers** | Sidesteps CUDA 13 wheel gap entirely. NGC images are pre-built with correct CUDA 13 + sm_121 binaries. |
| **Native Python development** (APIs, scripts, tooling) | **uv** | Fastest option, aarch64 native, lockfiles, modern workflow. |
| **Data science without GPU** (pandas, sklearn, matplotlib) | **uv** (primary) or **conda/mamba** (acceptable) | uv preferred for speed; conda acceptable when non-Python deps needed (HDF5, etc.). |
| **Quick experiments / one-off scripts** | **pip + venv** | Zero setup, already available on system. |

### The CUDA 13 Wheel Gap — Key Insight

No Python package manager solves the CUDA 13 wheel gap because the problem is not dependency resolution — it's that GPU-accelerated wheels for CUDA 13 / sm_121 / aarch64 largely do not exist on PyPI or conda-forge yet. The solution is architectural: use NGC containers for GPU workloads, and native Python environments (via uv) for everything else.

---

## 4. Container Strategy

### Base Image Selection

Use NGC (NVIDIA GPU Cloud) images as the starting point for all GPU-accelerated workloads:

```bash
# Framework-specific images (preferred — include framework + CUDA + cuDNN)
docker pull nvcr.io/nvidia/pytorch:25.04-py3
docker pull nvcr.io/nvidia/tensorflow:25.04-tf2-py3

# Base CUDA image (for custom builds)
docker pull nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# vLLM serving
docker pull nvcr.io/nvidia/vllm:v0.10.1.1
```

### Runtime Configuration

Docker is pre-installed on DGX OS with `nvidia-container-toolkit` configured. Always use:

```bash
docker run --runtime=nvidia --gpus all \
  --shm-size=16g \
  -v /home/$USER/models:/models \
  -v /home/$USER/data:/data \
  -v /home/$USER/projects:/workspace \
  nvcr.io/nvidia/pytorch:25.04-py3
```

Key flags:

- `--runtime=nvidia --gpus all` — exposes the GPU and CUDA stack inside the container.
- `--shm-size=16g` — required for PyTorch DataLoader with multiple workers (uses shared memory for IPC).
- Volume mounts — persist data outside the container lifecycle.

### Custom Dockerfiles

Layer customizations on top of NGC images:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --no-cache-dir \
    fastapi uvicorn \
    langchain chromadb \
    transformers datasets

COPY ./app /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Container Best Practices for DGX Spark

- **Memory limits**: Always set `--memory=100g` (or appropriate limit) to prevent OOM zombie mode from freezing the system.
- **OOM score adjustment**: Use `--oom-score-adj=500` so the kernel kills the container before the system becomes unresponsive.
- **Named volumes for model weights**: Avoid re-downloading large models into each container.
- **Use docker compose** for multi-service stacks (e.g., vLLM + vector DB + API server).

---

## 5. Storage Layout

The DGX Spark has **4 TB NVMe storage** in a single logical volume. Recommended directory structure:

```
/home/<user>/
├── models/              # LLM weights (Ollama, HuggingFace, NGC)
│   ├── ollama/          # Ollama model storage
│   ├── huggingface/     # HuggingFace hub cache
│   └── custom/          # Custom fine-tuned models
├── data/                # Datasets
│   ├── raw/             # Original datasets
│   ├── processed/       # Preprocessed/tokenized data
│   └── embeddings/      # Vector embeddings / index files
├── projects/            # Application code and repos
│   ├── project-a/
│   └── project-b/
├── envs/                # Python virtual environments (uv/venv)
└── docker-volumes/      # Docker named volume mount point
```

### Environment Variables

Add to `~/.bashrc` or `~/.profile`:

```bash
# Ollama model storage
export OLLAMA_MODELS="/home/$USER/models/ollama"

# HuggingFace cache
export HF_HOME="/home/$USER/models/huggingface"
export TRANSFORMERS_CACHE="/home/$USER/models/huggingface/hub"

# uv cache
export UV_CACHE_DIR="/home/$USER/.cache/uv"
```

### Docker Volume Configuration

```bash
# Create named volumes for persistent container data
docker volume create --driver local \
  --opt type=none \
  --opt o=bind \
  --opt device=/home/$USER/docker-volumes/postgres \
  pgdata
```

### Storage Budget Guidance

| Category | Suggested Allocation | Notes |
|----------|---------------------|-------|
| OS + System | ~50 GB | DGX OS, system packages |
| Models | ~1.5 TB | LLM weights are large (70B ≈ 40GB quantized) |
| Data | ~1 TB | Datasets, embeddings, vector DBs |
| Projects + Containers | ~500 GB | Code, Docker images, build artifacts |
| Headroom | ~950 GB | Buffer for experimentation |

---

## 6. Network Configuration

### Interfaces

| Interface | Speed | Use Case |
|-----------|-------|----------|
| 10 GbE (RJ45) | 10 Gbps | Primary — model downloads, SSH, general networking |
| QSFP (ConnectX-7) | 400 Gbps | Dual-Spark clustering only (direct connect between 2 units) |
| Wi-Fi 7 | Variable | Convenience / backup — not recommended for large transfers |

### Recommendations

- **Primary network**: Use the 10 GbE port connected to your LAN/router. Sufficient for model downloads (a 70B model at 10 Gbps ≈ 5 minutes), SSH access, and API serving.
- **QSFP**: Only relevant if you have two DGX Spark units and intend to run distributed inference (e.g., Llama 405B across both). Requires netplan configuration and static IP assignment on both sides.
- **Wi-Fi**: Acceptable for initial setup, web browsing, and light SSH. Not recommended for downloading multi-GB model files or serving production traffic.

### Firewall Considerations

```bash
# Allow SSH
sudo ufw allow 22/tcp
# Allow Ollama API (if serving to local network)
sudo ufw allow 11434/tcp
# Allow custom API ports as needed
sudo ufw allow 8000/tcp
# Enable firewall
sudo ufw enable
```

---

## 7. Security Baseline

### SSH Configuration

```bash
# Generate ED25519 key (on client machine)
ssh-keygen -t ed25519 -C "dgx-spark-access"

# Copy to DGX Spark
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@dgx-spark-ip

# Disable password authentication on DGX Spark
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

### Docker Group Membership

Run Docker without root by adding user to the `docker` group (already configured on DGX OS by default):

```bash
# Verify membership
groups | grep docker

# If not present:
sudo usermod -aG docker $USER
newgrp docker
```

### Port Exposure

- **Default**: No ports exposed to external network.
- **Ollama**: Binds to `127.0.0.1:11434` by default — only accessible from localhost. To expose to LAN, set `OLLAMA_HOST=0.0.0.0` and add UFW rule.
- **Docker containers**: Use `-p 127.0.0.1:8000:8000` to bind to localhost only. Omit `127.0.0.1:` prefix only when intentionally exposing to network.
- **Audit open ports**: `sudo ss -tlnp` to see all listening ports.

### Additional Hardening

- Enable automatic security updates for Ubuntu packages: `sudo apt install unattended-upgrades`.
- Keep DGX Dashboard updates current — they include firmware security patches.
- Use SSH key-only authentication (disable password auth).
- Do not run ML workloads as root — use your regular user account with Docker group membership.

---

## Summary

The DGX Spark base environment strategy centers on three principles: (1) treat DGX OS as an appliance — never modify core system packages, (2) use **uv** for native Python environments and **NGC containers** for GPU-accelerated workloads to sidestep the CUDA 13 wheel gap, and (3) organize storage and networking to support iterative ML development with large models. The 4TB NVMe, 128GB unified memory, and pre-installed Ollama + Docker stack provide a capable local AI development platform when configured according to these guidelines.

## Practical Implications

- New DGX Spark users should set up their storage layout and environment variables within the first hour of use — this prevents the common problem of models scattered across random directories.
- The CUDA 13 wheel gap is the single largest friction point. Defaulting to NGC containers for any GPU workload eliminates hours of debugging `ImportError: libcudart.so.12` errors.
- uv provides a meaningfully faster iteration cycle than pip/conda for Python dependency management, which compounds over time during active development.

## Recommended Actions

1. **Immediate**: Configure storage layout directories and environment variables per Section 5.
2. **Immediate**: Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
3. **Immediate**: Pull primary NGC container images for your ML framework of choice.
4. **First week**: Set up SSH key authentication and UFW firewall per Section 7.
5. **Ongoing**: Check DGX Dashboard for system updates monthly.

## Confidence Level

**HIGH** — This document synthesizes NVIDIA's official DGX Spark documentation, DGX OS release notes, and confirmed community experience with the aarch64/CUDA 13 ecosystem. The Python environment recommendations are based on direct testing of tool availability on aarch64 and verification of conda-forge channel contents for CUDA 13.

## Source Notes

- OS baseline, driver versions, CUDA versions: [NVIDIA-OFFICIAL] DGX Spark product documentation and DGX OS 7.4.0 release notes.
- Python environment tool compatibility on aarch64: [TESTED] Verified availability of uv, miniforge, Poetry binaries for aarch64 Linux.
- CUDA 13 wheel gap on PyPI/conda-forge: [COMMUNITY] Widely reported across DGX Spark early adopter threads; [TESTED] confirmed via PyPI and conda-forge package index queries.
- Container runtime flags and NGC image tags: [NVIDIA-OFFICIAL] NGC catalog and nvidia-container-toolkit documentation.
- Storage layout: [INFERRED] Based on typical ML development workflows and DGX Spark storage capacity.
- Security recommendations: [COMMUNITY] Standard Linux hardening practices applied to DGX Spark context.

## Unresolved Questions

1. **When will conda-forge ship CUDA 13 packages?** No public timeline. This determines when conda becomes viable for GPU workloads on DGX Spark.
2. **Will NVIDIA publish a curated apt/pip repository for DGX Spark?** The dgx-spark-vllm package suggests movement in this direction, but no official announcement of a broader package repository.
3. **DGX OS update frequency**: NVIDIA has not published a formal cadence for DGX OS updates. Early updates have been responsive to bug reports (e.g., thermal throttling firmware), but long-term cadence is unconfirmed.
4. **uv stability for production use**: uv is rapidly maturing but is a newer tool. Long-term stability for production lockfile workflows is assumed but not yet proven over years.
