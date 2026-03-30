# DGX Spark Master Reference

> **Last Updated**: 2026-03-30
> **Status**: Complete — Phase 1
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Maintainer**: DGX Spark Expert System

---

## Summary

The NVIDIA DGX Spark is a compact desktop AI workstation built around the GB10 Grace Blackwell Superchip, combining a 20-core ARM CPU and a Blackwell-architecture GPU with 128 GB of unified LPDDR5x memory in a 150 mm cube form factor. [NVIDIA-OFFICIAL] It is designed to run models up to 200B parameters locally (405B with dual-Spark linking) and ships with DGX OS 7.4.0, CUDA 13.0.2, and a pre-configured software stack including Docker, Ollama, and JupyterLab. [NVIDIA-OFFICIAL] While the 1 PFLOP FP4 headline number is impressive, real-world performance is fundamentally constrained by the 273 GB/s memory bandwidth, and sustained workloads face thermal throttling in this passively-cooled enclosure. [COMMUNITY] [TESTED] The CUDA 13 ecosystem gap — most pip wheels target CUDA 12.x/x86 — makes a container-first workflow effectively mandatory. [COMMUNITY] [TESTED]

---

## 1. Hardware Architecture

### 1.1 GB10 Grace Blackwell Superchip

The DGX Spark is powered by the NVIDIA GB10 Grace Blackwell Superchip, a single package that fuses an ARM-based Grace CPU die and a Blackwell GPU die connected via an on-package NVLink Chip-to-Chip (C2C) interconnect running at 900 GB/s bidirectional bandwidth. [NVIDIA-OFFICIAL] This is not a discrete-GPU-over-PCIe design; CPU and GPU share a unified, coherent 128 GB memory pool with no copy overhead for CPU↔GPU data transfers. [NVIDIA-OFFICIAL]

The GB10 has a TDP of 140W, powered by an external 240W PSU that also feeds the NVMe, networking, and I/O subsystems. [NVIDIA-OFFICIAL]

### 1.2 CPU — 20-Core ARM (Cortex-X925 + Cortex-A725)

| Property | Value |
|---|---|
| Architecture | ARMv9 / aarch64 | [NVIDIA-OFFICIAL]
| Core count | 20 total | [NVIDIA-OFFICIAL]
| High-performance cores | 10× ARM Cortex-X925 | [NVIDIA-OFFICIAL]
| Efficiency cores | 10× ARM Cortex-A725 | [NVIDIA-OFFICIAL]
| ISA | aarch64 | [NVIDIA-OFFICIAL]

The big.LITTLE configuration means workload scheduling matters. CPU-bound preprocessing tasks benefit from pinning to the X925 cores when possible. [INFERRED] The aarch64 architecture has broad Linux ecosystem support, but some niche scientific packages may still lack ARM builds. [COMMUNITY]

### 1.3 GPU — Blackwell Architecture (sm_121)

| Property | Value |
|---|---|
| Architecture | Blackwell | [NVIDIA-OFFICIAL]
| Compute capability | sm_121 | [NVIDIA-OFFICIAL]
| CUDA cores | 6,144 | [NVIDIA-OFFICIAL]
| Tensor Cores | 5th generation | [NVIDIA-OFFICIAL]
| RT Cores | 4th generation | [NVIDIA-OFFICIAL]
| Peak FP4 (with sparsity) | 1 PFLOP (theoretical) | [NVIDIA-OFFICIAL]

**Critical note on sm_121**: This is a new compute capability. Pre-compiled CUDA binaries targeting older architectures (sm_80, sm_89, sm_90) will not run natively. Libraries must be compiled with sm_121 support or use PTX JIT compilation, which incurs a startup penalty. [TESTED] [COMMUNITY]

### 1.4 Unified Memory — 128 GB LPDDR5x

| Property | Value |
|---|---|
| Capacity | 128 GB | [NVIDIA-OFFICIAL]
| Type | LPDDR5x | [NVIDIA-OFFICIAL]
| Interface | 256-bit | [NVIDIA-OFFICIAL]
| Bandwidth | 273 GB/s | [NVIDIA-OFFICIAL]
| Coherency | Unified CPU+GPU coherent | [NVIDIA-OFFICIAL]
| ECC | **Not available** | [NVIDIA-OFFICIAL]

The unified memory architecture means there is no separate VRAM allocation. The full 128 GB is visible to both CPU and GPU simultaneously, with hardware-managed coherency. [NVIDIA-OFFICIAL] This is a significant advantage for large model loading — there is zero copy overhead when moving tensors between CPU and GPU contexts.

**No ECC**: LPDDR5x on the DGX Spark does not include error-correcting code. [NVIDIA-OFFICIAL] For long-running numerical workloads (multi-day training, extended Monte Carlo simulations), there is a non-trivial risk of silent bit-flip errors. Mitigation strategies include periodic checkpointing and result validation. [INFERRED]

**nvidia-smi behavior**: Because of unified memory, `nvidia-smi` reports `Memory-Usage: Not Supported` instead of the standard used/total VRAM readout. [TESTED] [COMMUNITY] This breaks monitoring scripts that parse nvidia-smi output for memory utilization. Use `/proc/meminfo` or `free -h` instead to monitor overall memory pressure.

### 1.5 NVLink C2C Interconnect

| Property | Value |
|---|---|
| Type | NVLink Chip-to-Chip (C2C) | [NVIDIA-OFFICIAL]
| Bandwidth | 900 GB/s bidirectional | [NVIDIA-OFFICIAL]
| Latency | Ultra-low (on-package) | [NVIDIA-OFFICIAL]

The NVLink C2C interconnect provides 900 GB/s bandwidth between the CPU and GPU dies, which is over 3× the LPDDR5x memory bandwidth (273 GB/s). [NVIDIA-OFFICIAL] The memory bandwidth — not the interconnect — is the system bottleneck. [INFERRED]

---

## 2. Storage

### 2.1 4 TB NVMe M.2 — Specifications

| Property | Value |
|---|---|
| Capacity | 4 TB | [NVIDIA-OFFICIAL]
| Interface | NVMe M.2 | [NVIDIA-OFFICIAL]
| Encryption | Self-encrypting drive (SED) | [NVIDIA-OFFICIAL]

### 2.2 Self-Encryption

The onboard NVMe drive supports hardware-level self-encryption (SED), meaning data-at-rest encryption is handled by the drive controller without CPU overhead. [NVIDIA-OFFICIAL] This is important for enterprise deployments where data security compliance is required.

### 2.3 Performance Characteristics

NVMe M.2 drives in this class typically deliver 5,000–7,000 MB/s sequential read and 4,000–5,000 MB/s sequential write. [INFERRED] Model loading from local NVMe is significantly faster than network-based loading; a 70B parameter model (~40 GB quantized) loads in under 10 seconds from local storage. [INFERRED]

### 2.4 Recommended Storage Layout

```
/home/user/
├── models/          # Downloaded model weights (Ollama, HuggingFace, etc.)
├── data/            # Datasets, embeddings, vector stores
└── projects/        # Code, notebooks, experiment configs
```
[NVIDIA-OFFICIAL]

Keep model weights in a dedicated directory. Configure the `OLLAMA_MODELS` environment variable to point here if using Ollama:

```bash
export OLLAMA_MODELS=/home/user/models/ollama
```
[COMMUNITY]

4 TB is generous but fills quickly with large models. A single unquantized 70B model can be 140+ GB. Monitor disk usage and prune unused model variants regularly. [INFERRED]

---

## 3. Networking

### 3.1 10 GbE Ethernet (RJ-45)

| Property | Value |
|---|---|
| Type | RJ-45 | [NVIDIA-OFFICIAL]
| Speed | 10 Gigabit Ethernet | [NVIDIA-OFFICIAL]
| Count | 1 | [NVIDIA-OFFICIAL]

Standard wired networking for LAN/WAN connectivity, model downloads, and remote access. [NVIDIA-OFFICIAL]

### 3.2 ConnectX-7 (200 Gbps QSFP)

| Property | Value |
|---|---|
| NIC | NVIDIA ConnectX-7 | [NVIDIA-OFFICIAL]
| Speed | 200 Gbps | [NVIDIA-OFFICIAL]
| Port | QSFP | [NVIDIA-OFFICIAL]
| Primary use | Dual-Spark linking | [NVIDIA-OFFICIAL]

The ConnectX-7 SmartNIC is the interface used for linking two DGX Spark devices together to form a dual-node cluster capable of running 405B parameter models. [NVIDIA-OFFICIAL] It is also usable for high-speed NAS/SAN connectivity in enterprise environments. [INFERRED]

### 3.3 Wi-Fi 7

| Property | Value |
|---|---|
| Standard | Wi-Fi 7 (802.11be) | [NVIDIA-OFFICIAL]

Useful for initial setup, light management tasks, and environments where wired Ethernet is not available. Not recommended as the primary network path for large model downloads or sustained data transfer — use the 10 GbE RJ-45 port instead. [INFERRED]

### 3.4 Bluetooth 5.4

| Property | Value |
|---|---|
| Standard | Bluetooth 5.4 | [NVIDIA-OFFICIAL]

Supports wireless peripherals (keyboard, mouse, audio). [NVIDIA-OFFICIAL]

### 3.5 Dual-Spark QSFP Configuration

Linking two DGX Spark devices requires a direct QSFP cable between the ConnectX-7 ports and netplan-based network configuration. [NVIDIA-OFFICIAL] [COMMUNITY]

Example netplan configuration for the QSFP interface:

```yaml
# /etc/netplan/99-qsfp-link.yaml
network:
  version: 2
  ethernets:
    enp1s0f0:   # actual interface name may vary — check `ip link`
      addresses:
        - 192.168.100.1/24   # Node 1
      mtu: 9000
```

On the second Spark:

```yaml
# /etc/netplan/99-qsfp-link.yaml
network:
  version: 2
  ethernets:
    enp1s0f0:
      addresses:
        - 192.168.100.2/24   # Node 2
      mtu: 9000
```

Apply with:

```bash
sudo netplan apply
```

**Known issue**: Community reports frequent netplan configuration failures during dual-Spark setup. [COMMUNITY] Common failure modes include incorrect interface naming (the kernel name for the ConnectX-7 port varies between firmware versions), MTU mismatches, and netplan YAML syntax errors. Always verify the interface name with `ip link show` before writing the config. [COMMUNITY]

---

## 4. I/O Ports

### 4.1 USB-C (4×)

| Property | Value |
|---|---|
| Count | 4 | [NVIDIA-OFFICIAL]
| Type | USB Type-C | [NVIDIA-OFFICIAL]

Four USB Type-C ports for peripherals, external storage, and displays. [NVIDIA-OFFICIAL]

### 4.2 HDMI 2.1a

| Property | Value |
|---|---|
| Count | 1 | [NVIDIA-OFFICIAL]
| Standard | HDMI 2.1a | [NVIDIA-OFFICIAL]
| Audio | Multichannel audio output | [NVIDIA-OFFICIAL]

Supports high-resolution display output with multichannel audio passthrough. [NVIDIA-OFFICIAL]

### 4.3 Known Issues (HDMI Deep Sleep Bug)

**Bug**: The HDMI display output enters deep sleep mode after extended inactivity and does not wake up. [COMMUNITY] [TESTED] Users must physically disconnect and reconnect the HDMI cable, or reboot the device, to restore video output.

**Workarounds**:
- Disable display power management via DGX OS settings [COMMUNITY]
- Use `xset -dpms` and `xset s off` if running a desktop environment [INFERRED]
- Rely on SSH/remote access instead of local display for headless workflows [COMMUNITY]

---

## 5. Physical & Thermal

### 5.1 Form Factor (150 × 150 × 50.5 mm, 1.2 kg)

| Property | Value |
|---|---|
| Dimensions | 150 mm × 150 mm × 50.5 mm | [NVIDIA-OFFICIAL]
| Weight | 1.2 kg | [NVIDIA-OFFICIAL]

An extremely compact form factor — roughly the size of a Mac Mini. This is both a selling point and the root cause of the thermal constraints. [NVIDIA-OFFICIAL] [INFERRED]

### 5.2 Power (240W PSU, 140W TDP)

| Property | Value |
|---|---|
| External PSU | 240W | [NVIDIA-OFFICIAL]
| GB10 TDP | 140W | [NVIDIA-OFFICIAL]

The 240W PSU provides headroom above the 140W chip TDP for the NVMe, networking, fans, and I/O subsystems. [NVIDIA-OFFICIAL]

### 5.3 Acoustic Profile (35 dB(A))

| Property | Value |
|---|---|
| Sound power | 35 dB(A) average | [NVIDIA-OFFICIAL]

35 dB(A) is comparable to a quiet library or whispered conversation. The Spark is designed for desktop/office placement. Under sustained GPU load, fan speed increases and noise may exceed this average figure. [INFERRED]

### 5.4 Thermal Throttling Reality

**This is the single most impactful limitation of the DGX Spark.**

Dissipating 140W TDP in a 150 mm × 150 mm × 50.5 mm enclosure is an extreme thermal engineering challenge. [INFERRED] Real-world reports confirm:

- **Users report sustained temperatures of 86°C** under heavy GPU workloads. [COMMUNITY] [TESTED]
- **John Carmack flagged thermal issues in October 2025**, describing the device as "quite hot" even at reduced power settings, with spontaneous rebooting during extended runs. [COMMUNITY]
- Thermal throttling kicks in well before the thermal protection shutdown threshold, reducing clock speeds and therefore throughput during sustained workloads. [COMMUNITY] [TESTED]
- The small enclosure does not allow sufficient passive or active airflow for continuous full-load operation in typical ambient conditions. [INFERRED]

### 5.5 Cooling Requirements & Recommendations

| Requirement | Detail |
|---|---|
| Ambient temperature | **Must be below 30°C** | [NVIDIA-OFFICIAL] [COMMUNITY]
| Clearance | Maintain clearance on **all sides** of the device | [NVIDIA-OFFICIAL]
| Vent maintenance | Periodically clean vents to prevent dust buildup | [COMMUNITY]
| External cooling | Some users add external USB fans or laptop cooling pads | [COMMUNITY]
| Placement | Hard, flat surface only — never on carpet, bedding, or enclosed shelves | [INFERRED]

**Practical recommendations**:
1. Place the Spark in an air-conditioned room (target 22–25°C ambient). [COMMUNITY]
2. Leave at least 10 cm clearance on all sides. [INFERRED]
3. For sustained training workloads, consider an external cooling solution (USB fan directed at the intake vents). [COMMUNITY]
4. Monitor temperatures via `nvidia-smi` or the DGX Dashboard and set up alerts for >80°C. [INFERRED]

### 5.6 Firmware Patch Status

NVIDIA has released firmware patches that improve thermal management behavior (fan curve tuning, earlier throttling onset to avoid thermal protection shutdowns). [COMMUNITY] However, these patches **do not solve the fundamental cooling constraint** — 140W in a 150 mm cube with limited airflow headroom. [COMMUNITY] [INFERRED] Firmware can redistribute the thermal budget more intelligently but cannot create additional cooling capacity.

Keep firmware updated via the DGX Dashboard for the latest thermal management improvements. [NVIDIA-OFFICIAL]

---

## 6. Software Stack

### 6.1 DGX OS 7.4.0 (Ubuntu 24.04, Kernel 6.17)

| Property | Value |
|---|---|
| OS | DGX OS 7.4.0 | [NVIDIA-OFFICIAL]
| Base | Ubuntu 24.04 LTS | [NVIDIA-OFFICIAL]
| Kernel | Linux 6.17 | [NVIDIA-OFFICIAL]
| Python | 3.12 | [NVIDIA-OFFICIAL]

DGX OS is NVIDIA's customized Ubuntu distribution with pre-configured GPU drivers, container runtime, and system monitoring tools. [NVIDIA-OFFICIAL] It is not a minimal install — it includes a full desktop environment, system services, and the DGX Dashboard web interface.

### 6.2 CUDA 13.0.2

| Property | Value |
|---|---|
| CUDA version | 13.0.2 | [NVIDIA-OFFICIAL]
| Compute capability | sm_121 | [NVIDIA-OFFICIAL]

**THIS IS THE MOST IMPACTFUL SOFTWARE DETAIL FOR DAY-TO-DAY USE.**

CUDA 13.0.2 is a major version ahead of the CUDA 12.x that the vast majority of the Python ML ecosystem currently targets. [TESTED] [COMMUNITY] This means:

- `pip install torch` installs a wheel compiled for CUDA 12.x, which **will fail to import** on DGX Spark. [TESTED]
- `pip install cupy`, `pip install rapids`, and many other CUDA-dependent packages face the same issue. [COMMUNITY]
- The PyPI ecosystem lags major CUDA releases by months. [COMMUNITY]

**Mitigation**: Use NGC container images that bundle the correct CUDA 13 runtime and pre-compiled libraries. This is why a container-first workflow is effectively mandatory (see Section 11.3). [COMMUNITY] [TESTED]

### 6.3 GPU Driver 580.126.09

| Property | Value |
|---|---|
| Driver version | 580.126.09 | [NVIDIA-OFFICIAL]

The driver is pre-installed and managed by DGX OS. Do not manually install drivers from NVIDIA's website — use the DGX Dashboard update mechanism to stay on the supported driver track. [NVIDIA-OFFICIAL]

### 6.4 Docker + NVIDIA Container Toolkit

| Property | Value |
|---|---|
| Docker | Pre-installed | [NVIDIA-OFFICIAL]
| nvidia-container-toolkit | Pre-installed | [NVIDIA-OFFICIAL]
| GPU passthrough | Pre-configured | [NVIDIA-OFFICIAL]

Docker with GPU passthrough is the primary execution environment for ML workloads on DGX Spark. [NVIDIA-OFFICIAL]

Run GPU-accelerated containers with:

```bash
docker run --runtime=nvidia --gpus all <image>
```
[NVIDIA-OFFICIAL]

See Section 9 for detailed Docker configuration, including the required override file.

### 6.5 Ollama (Pre-installed)

| Property | Value |
|---|---|
| Status | Pre-installed and pre-configured | [NVIDIA-OFFICIAL]

Ollama provides a simple interface for running local LLM inference. [NVIDIA-OFFICIAL] It is pre-configured to use the Blackwell GPU for acceleration.

Configure the model storage location:

```bash
export OLLAMA_MODELS=/home/user/models/ollama
```
[COMMUNITY]

Common usage:

```bash
ollama pull llama3:70b
ollama run llama3:70b
```

The 128 GB unified memory allows running quantized models up to ~70B parameters comfortably with Ollama, or up to 200B with aggressive quantization. [NVIDIA-OFFICIAL] [COMMUNITY]

### 6.6 JupyterLab (via DGX Dashboard)

JupyterLab is accessible through the DGX Dashboard web interface. [NVIDIA-OFFICIAL] It runs in the local Python 3.12 environment. For GPU-accelerated notebook work requiring CUDA-dependent libraries, use a containerized JupyterLab instance with the appropriate NGC base image instead. [INFERRED]

### 6.7 DGX Dashboard

The DGX Dashboard is a built-in web-based system monitoring and management interface. [NVIDIA-OFFICIAL] It provides:

- Real-time GPU/CPU/memory/temperature monitoring [NVIDIA-OFFICIAL]
- JupyterLab launcher [NVIDIA-OFFICIAL]
- System update management (firmware, driver, OS packages) [NVIDIA-OFFICIAL]
- Storage utilization overview [INFERRED]

Access the dashboard via a web browser on the local network. [NVIDIA-OFFICIAL]

### 6.8 NVIDIA Sync

NVIDIA Sync is a remote connectivity tool that enables secure access to the DGX Spark from outside the local network. [NVIDIA-OFFICIAL] This is NVIDIA's managed solution for remote development scenarios where SSH tunneling or VPN is not available.

### 6.9 NVIDIA NIM

NVIDIA NIM (NVIDIA Inference Microservices) provides optimized, containerized inference endpoints for production deployment. [NVIDIA-OFFICIAL] NIM containers are available through the NGC catalog and are pre-optimized for the Blackwell architecture. They are included in the 90-day AI Enterprise license. [NVIDIA-OFFICIAL]

---

## 7. Bundled Benefits

### 7.1 90-Day AI Enterprise License

| Property | Value |
|---|---|
| License | NVIDIA AI Enterprise — DGX Spark License | [NVIDIA-OFFICIAL]
| Duration | 90 days from activation | [NVIDIA-OFFICIAL]
| Includes | NIM, NeMo, Triton Inference Server, enterprise support | [NVIDIA-OFFICIAL]

This is a trial license. After 90 days, continued access to AI Enterprise components requires a paid subscription. [NVIDIA-OFFICIAL] The open-source components (CUDA, Docker, Ollama) continue to work without a license. [INFERRED]

### 7.2 DLI Course

| Property | Value |
|---|---|
| Benefit | Free NVIDIA Deep Learning Institute (DLI) hands-on AI course | [NVIDIA-OFFICIAL]
| Value | $90 | [NVIDIA-OFFICIAL]

A self-paced course included with purchase. [NVIDIA-OFFICIAL]

### 7.3 NGC Catalog Access

Access to the NVIDIA NGC catalog of GPU-optimized containers, pre-trained models, and SDKs. [NVIDIA-OFFICIAL] NGC containers are the recommended path for running ML frameworks on DGX Spark, as they include CUDA 13-compatible builds. [COMMUNITY] [TESTED]

### 7.4 DGX Cloud Migration Path

NVIDIA positions the Spark as a local development companion to DGX Cloud. [NVIDIA-OFFICIAL] Workloads prototyped on Spark can be scaled to DGX Cloud or DGX SuperPOD for production training. The NIM container format is consistent across local and cloud deployments. [NVIDIA-OFFICIAL]

### 7.5 Developer Forum & Support

| Resource | Detail |
|---|---|
| Playbooks | Available at build.nvidia.com/spark | [NVIDIA-OFFICIAL]
| Forums | NVIDIA developer forums | [NVIDIA-OFFICIAL]
| Support | Enterprise support via AI Enterprise license (90-day) | [NVIDIA-OFFICIAL]

### 7.6 Pricing

| Channel | Price |
|---|---|
| Official US MSRP | ~$3,000 | [NVIDIA-OFFICIAL]
| NVIDIA Marketplace | $4,699 | [NVIDIA-OFFICIAL]

---

## 8. CUDA / cuDNN / TensorRT Stack

### 8.1 CUDA 13.0.2 Details

| Property | Value |
|---|---|
| Version | 13.0.2 | [NVIDIA-OFFICIAL]
| Compute capability | sm_121 (Blackwell) | [NVIDIA-OFFICIAL]
| Host architecture | aarch64 | [NVIDIA-OFFICIAL]

CUDA 13 is a **major version** jump from CUDA 12.x. Key implications:

- **ABI break**: Libraries compiled against the CUDA 12.x runtime will not load against the CUDA 13 runtime. [TESTED]
- **New sm_121 target**: Code must be compiled with `-arch=sm_121` or include sm_121 in the target list, or rely on PTX JIT compilation (slower first launch). [TESTED]
- **aarch64 host**: In addition to the CUDA version gap, the aarch64 host architecture means x86_64 binaries and wheels cannot run at all. This is a **double filter** — packages must be both CUDA 13-compatible AND aarch64-compatible. [TESTED]

### 8.2 cuDNN 9.13

| Property | Value |
|---|---|
| Version | 9.13 | [NVIDIA-OFFICIAL]

cuDNN 9.13 is pre-installed and includes optimized kernels for Blackwell Tensor Cores. [NVIDIA-OFFICIAL] It provides acceleration for convolution, attention, normalization, and other deep learning primitives.

### 8.3 TensorRT Installation

TensorRT is available via apt but not pre-installed in the base OS:

```bash
sudo apt install -y tensorrt
```
[NVIDIA-OFFICIAL]

The `trtexec` benchmarking/conversion tool is located at:

```
/usr/src/tensorrt/bin/trtexec
```
[NVIDIA-OFFICIAL]

Use TensorRT for production inference optimization. It can convert ONNX models into Blackwell-optimized engines with FP16/FP8/INT8/FP4 precision modes. [NVIDIA-OFFICIAL]

### 8.4 sm_121 Implications

The sm_121 compute capability is exclusive to Blackwell consumer/workstation parts. [NVIDIA-OFFICIAL] This means:

- **PyTorch**: Official PyTorch pip wheels do not yet include sm_121 support for CUDA 13. Use NGC PyTorch containers instead. [COMMUNITY] [TESTED]
- **TensorFlow**: Similar situation — NGC containers recommended. [COMMUNITY]
- **JAX**: NGC containers or build from source with sm_121 support. [COMMUNITY]
- **Custom CUDA kernels**: Must be recompiled with `nvcc -arch=sm_121`. [TESTED]

### 8.5 CUDA 13 Wheel Gap Analysis

This is the most frequently reported pain point for DGX Spark users. [COMMUNITY]

**The problem**: Running `pip install <package>` for any CUDA-dependent Python package will download a wheel compiled for:
1. CUDA 12.x runtime
2. x86_64 architecture (typically)

Neither is compatible with DGX Spark (CUDA 13 / aarch64). [TESTED]

**Packages known to be affected**:
- `torch`, `torchvision`, `torchaudio` [TESTED]
- `tensorflow` [COMMUNITY]
- `cupy` [COMMUNITY]
- `rapids` (cuDF, cuML, cuGraph) [COMMUNITY]
- `triton` (the compiler, not Triton Inference Server) [COMMUNITY]
- Any package with a `_cuda` or `cu1x` suffix in its wheel name [INFERRED]

**Packages that work fine** (pure Python or have aarch64 wheels):
- `numpy`, `scipy`, `pandas` (CPU-only operations) [TESTED]
- `scikit-learn` [TESTED]
- `transformers` (the library, not the model execution) [TESTED]
- `datasets`, `tokenizers` [TESTED]

**Solutions** (in order of preference):
1. **Use NGC containers** — pre-built with correct CUDA 13 + aarch64 libraries [NVIDIA-OFFICIAL]
2. **Build from source inside a container** — using NGC CUDA 13 dev base image [COMMUNITY]
3. **Use `uv` for Python environment management** — not conda, which has CUDA 13 channel gaps [COMMUNITY]

---

## 9. Container Runtime

### 9.1 Docker Configuration

Docker is pre-installed on DGX OS 7.4.0. [NVIDIA-OFFICIAL] The NVIDIA Container Toolkit is configured to enable GPU passthrough for all containers by default.

### 9.2 NVIDIA Container Toolkit Setup

The nvidia-container-toolkit is pre-installed and configured. [NVIDIA-OFFICIAL] Verify with:

```bash
nvidia-ctk --version
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 nvidia-smi
```
[TESTED]

### 9.3 NGC Catalog Access

NGC (NVIDIA GPU Cloud) container images are accessible without authentication for most public images. [NVIDIA-OFFICIAL] For AI Enterprise-licensed containers, authenticate with:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```
[NVIDIA-OFFICIAL]

### 9.4 Docker Override Configuration

**Required**: A systemd override is needed for Docker to function correctly on DGX OS:

File: `/etc/systemd/system/docker.service.d/override.conf`

```ini
[Service]
Environment="DOCKER_INSECURE_NO_IPTABLES_RAW=1"
```
[TESTED] [COMMUNITY]

Apply with:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

If this override is missing, Docker networking may fail with iptables-related errors. [COMMUNITY]

### 9.5 Multi-Arch Image Migration

NVIDIA is migrating from the deprecated `nvidia/cuda-arm64` repository to unified multi-architecture images. [NVIDIA-OFFICIAL] Use the standard image names — Docker will pull the correct aarch64 variant automatically:

```bash
# Correct — multi-arch, pulls aarch64 automatically
docker pull nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# Deprecated — do not use
docker pull nvidia/cuda-arm64:...
```
[NVIDIA-OFFICIAL]

**Recommended base image for custom Dockerfiles**:

```dockerfile
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
```
[NVIDIA-OFFICIAL] [COMMUNITY]

Layer your application-specific dependencies on top of this base to ensure CUDA 13 + aarch64 + cuDNN compatibility. [COMMUNITY]

---

## 10. Limitations & Gotchas

This section documents every known limitation, bug, and sharp edge. **Read this section thoroughly before deploying workloads.**

### 10.1 Thermal Throttling

**Severity**: HIGH
**Impact**: Reduced sustained throughput, potential spontaneous reboots

Sustained GPU workloads push the GB10 to 86°C, triggering thermal throttling that reduces clock speeds. [COMMUNITY] [TESTED] In extreme cases (poor ventilation, high ambient temperature), the device may spontaneously reboot as a thermal protection measure. [COMMUNITY] John Carmack publicly reported this behavior in October 2025, noting the device ran "quite hot" even at reduced power settings. [COMMUNITY]

**Root cause**: 140W TDP in a 150 mm × 150 mm × 50.5 mm enclosure with limited airflow capacity. [INFERRED]

**Mitigation**: See Section 5.5 for cooling recommendations. There is no software-only fix; the constraint is physical. [INFERRED]

### 10.2 OOM Zombie Mode

**Severity**: HIGH
**Impact**: System-wide freeze requiring hard reboot

Because CPU and GPU share unified memory, a GPU workload that exhausts memory does not produce a clean "CUDA out of memory" error. Instead, the entire system runs out of memory, the OOM killer may not act quickly enough, and the device enters a zombie state where it is unresponsive to input (including SSH). [COMMUNITY] [TESTED]

This is fundamentally different from discrete-GPU systems, where GPU OOM is contained to the CUDA process.

**Mitigation**:

```bash
docker run --runtime=nvidia --gpus all \
  --memory=100g \
  --oom-score-adj=1000 \
  <image>
```
[COMMUNITY]

- `--memory=100g` caps the container's memory at 100 GB (leaving 28 GB for the OS and other services). [COMMUNITY]
- `--oom-score-adj=1000` tells the kernel OOM killer to prioritize killing this container over system services. [COMMUNITY]

**Always set memory limits on GPU containers.** There is no safe default. [COMMUNITY]

### 10.3 No ECC Memory

**Severity**: MEDIUM
**Impact**: Risk of silent data corruption on long numerical workloads

LPDDR5x on the DGX Spark does not have ECC (Error-Correcting Code). [NVIDIA-OFFICIAL] Bit-flip errors are rare but statistically significant over long runtimes. This matters for:

- Multi-day training runs [INFERRED]
- Financial/scientific computations where precision is critical [INFERRED]
- Distributed training where a single corrupted gradient can propagate [INFERRED]

**Mitigation**: Checkpoint frequently. Validate results against known-good baselines. For safety-critical computations, cross-validate on ECC-equipped hardware. [INFERRED]

### 10.4 HDMI Deep Sleep Bug

**Severity**: LOW (annoyance)
**Impact**: Display does not wake after extended inactivity

The HDMI 2.1a output enters a deep sleep state after extended inactivity and does not recover without physical cable disconnection or device reboot. [COMMUNITY] [TESTED]

**Mitigation**: Disable DPMS, use SSH for headless access, or connect a display only when needed. [COMMUNITY]

### 10.5 CUDA 13 Wheel Gap

**Severity**: HIGH
**Impact**: Most pip-installed CUDA packages fail to import

See Section 8.5 for full analysis. The vast majority of Python packages with CUDA dependencies are compiled against CUDA 12.x and x86_64. They will not work on DGX Spark. [TESTED] [COMMUNITY]

**Mitigation**: Container-first workflow using NGC base images. [NVIDIA-OFFICIAL] [COMMUNITY]

### 10.6 nvidia-smi Memory Reporting

**Severity**: LOW (tooling inconvenience)
**Impact**: nvidia-smi cannot report GPU memory usage

`nvidia-smi` reports `Memory-Usage: Not Supported` because there is no discrete GPU memory to report — the unified memory pool is managed by the system memory controller, not the GPU. [TESTED] [COMMUNITY]

**Mitigation**: Use system-level memory monitoring tools:

```bash
free -h
cat /proc/meminfo
```
[TESTED]

### 10.7 Clustering Cap (2 Devices)

**Severity**: MEDIUM
**Impact**: Cannot scale beyond 2 DGX Sparks

Official clustering support is limited to 2 DGX Spark devices connected via the ConnectX-7 QSFP port. [NVIDIA-OFFICIAL] This enables up to 256 GB unified memory and 405B parameter models. [NVIDIA-OFFICIAL]

There is no supported path for 3+ device clusters. For larger scale, NVIDIA directs users to DGX Cloud or DGX SuperPOD. [NVIDIA-OFFICIAL]

### 10.8 Setup Non-Resumable

**Severity**: MEDIUM
**Impact**: Interrupted initial setup requires factory reset

If the initial out-of-box setup process (DGX OS first boot configuration) is interrupted for any reason — power loss, network failure, user error — the setup cannot be resumed. A full factory reset is required to start over. [COMMUNITY] [TESTED]

**Mitigation**: Ensure stable power (use a UPS if possible) and a reliable network connection before starting the initial setup. Do not interrupt the process. [COMMUNITY]

### 10.9 Memory Bandwidth Bottleneck

**Severity**: HIGH
**Impact**: Actual performance far below theoretical FLOPS

The real performance bottleneck on DGX Spark is the 273 GB/s LPDDR5x memory bandwidth, not the GPU's theoretical compute capability. [COMMUNITY] [TESTED] Most LLM inference and training workloads are memory-bandwidth-bound, meaning the CUDA cores sit idle waiting for data.

For context, an NVIDIA A100 80GB has 2,039 GB/s HBM2e bandwidth — roughly 7.5× higher. The Spark's memory bandwidth is comparable to a high-end laptop, not a datacenter GPU. [INFERRED]

**Community benchmarking**: Users report that an Apple M3 Pro laptop matched the DGX Spark on 30B model inference throughput. [COMMUNITY] This is because Apple's unified memory also has bandwidth constraints, and the M3 Pro's memory bandwidth is in a similar class.

### 10.10 FP4 Performance Claims

**Severity**: LOW (marketing context)
**Impact**: Inflated performance expectations

The "1 PFLOP FP4" headline number is a **theoretical peak with sparsity enabled**. [NVIDIA-OFFICIAL] Real-world performance is:

- **Much lower** without structured sparsity in the model [INFERRED]
- **Memory-bandwidth-bound** for most inference workloads (see 10.9) [TESTED]
- **Thermally throttled** during sustained runs (see 10.1) [TESTED]

Treat the 1 PFLOP number as a marketing figure, not a performance guarantee. [INFERRED]

---

## 11. Best Practices

### 11.1 Cooling & Airflow Management

1. Keep ambient temperature **below 30°C** at all times. [NVIDIA-OFFICIAL] [COMMUNITY]
2. Maintain clearance on **all sides** of the device — do not place against walls, in enclosed cabinets, or near heat sources. [NVIDIA-OFFICIAL]
3. Place on a hard, flat surface with unobstructed bottom ventilation. [INFERRED]
4. Clean vents periodically (monthly in dusty environments). [COMMUNITY]
5. Consider external USB-powered cooling (directed fan) for sustained training workloads. [COMMUNITY]
6. Monitor temperatures via `nvidia-smi` or DGX Dashboard; alert on >80°C. [INFERRED]

### 11.2 Docker OOM Score Adjustment

**Always** run GPU containers with memory limits and OOM score adjustments:

```bash
docker run --runtime=nvidia --gpus all \
  --memory=100g \
  --memory-swap=100g \
  --oom-score-adj=1000 \
  <image>
```
[COMMUNITY]

- `--memory=100g` and `--memory-swap=100g` (same value) prevents swap from masking OOM conditions [COMMUNITY]
- `--oom-score-adj=1000` makes the container the first target for the kernel OOM killer [COMMUNITY]
- Reserve at least 20–28 GB for the OS, DGX Dashboard, and system services [INFERRED]

### 11.3 Container-First Workflow

**The container-first workflow is not optional — it is the only reliable way to run CUDA-dependent Python workloads on DGX Spark.** [COMMUNITY] [TESTED]

Why:
- CUDA 13 wheels are not available from PyPI for most packages [TESTED]
- aarch64 builds of CUDA-dependent packages are sparse [COMMUNITY]
- NGC containers bundle the correct CUDA 13 runtime, cuDNN, and pre-compiled frameworks [NVIDIA-OFFICIAL]

Recommended workflow:

```dockerfile
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y python3.12 python3-pip
# Install your dependencies here, building against the CUDA 13 runtime in the container
```
[COMMUNITY]

For interactive development:

```bash
docker run -it --runtime=nvidia --gpus all \
  --memory=100g --oom-score-adj=1000 \
  -v /home/user/projects:/workspace \
  -v /home/user/models:/models \
  -p 8888:8888 \
  nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 bash
```
[COMMUNITY]

### 11.4 Environment Isolation Strategy

Use `uv` for Python environment management, **not conda**. [COMMUNITY]

Rationale:
- conda's CUDA 13 channel support lags significantly behind [COMMUNITY]
- `uv` is faster and respects the system CUDA installation more reliably [COMMUNITY]
- `uv` supports aarch64 natively [COMMUNITY]

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
[COMMUNITY]

For purely CPU-bound Python work (data preprocessing, visualization), `uv` with the system Python 3.12 works well outside containers. For GPU work, use containers. [INFERRED]

### 11.5 Storage Layout Conventions

Follow the NVIDIA-recommended directory structure:

```
/home/user/
├── models/          # All model weights
│   └── ollama/      # OLLAMA_MODELS target
├── data/            # Datasets, embeddings
└── projects/        # Code and experiment configs
```
[NVIDIA-OFFICIAL]

Set persistent environment variables in `~/.bashrc` or `~/.profile`:

```bash
export OLLAMA_MODELS=/home/user/models/ollama
```
[COMMUNITY]

### 11.6 Update Cadence

Stay on NVIDIA's update cadence using the DGX Dashboard. [NVIDIA-OFFICIAL] Updates include:

- GPU driver updates (critical for stability and performance) [NVIDIA-OFFICIAL]
- Firmware updates (thermal management improvements) [COMMUNITY]
- DGX OS security patches [NVIDIA-OFFICIAL]
- CUDA toolkit updates [NVIDIA-OFFICIAL]

Do **not** manually install drivers or CUDA toolkits from NVIDIA's download page — this will conflict with the DGX OS managed stack. [COMMUNITY]

### 11.7 Model Cache Management

Large language models consume significant storage. Management guidelines:

- A quantized 70B model is ~40 GB; an unquantized 70B model is ~140 GB [INFERRED]
- With 4 TB NVMe, you can store many models, but monitor usage proactively [INFERRED]
- Ollama caches models in the `OLLAMA_MODELS` directory; unused models can be removed with `ollama rm <model>` [COMMUNITY]
- For HuggingFace models, set `HF_HOME` to a known location and periodically prune the cache with `huggingface-cli delete-cache` [COMMUNITY]

---

## Detailed Findings

### Performance Characterization

The DGX Spark occupies a unique position: it offers genuine Blackwell-architecture GPU compute with 128 GB of unified memory in a desktop form factor, but its real-world throughput is bounded by three constraints that collectively define its performance envelope:

1. **Memory bandwidth (273 GB/s)**: The dominant bottleneck for transformer-based inference and training. This places the Spark's throughput in the same class as high-end laptops for memory-bound workloads, despite having far more raw compute capability. [TESTED] [COMMUNITY]

2. **Thermal throttling**: Sustained workloads trigger throttling within minutes under normal ambient conditions, further reducing effective throughput below the theoretical peak. [COMMUNITY] [TESTED]

3. **CUDA 13 ecosystem maturity**: The software ecosystem has not yet fully caught up with CUDA 13. Users spend significant time working around build and compatibility issues. [COMMUNITY]

### Model Capacity Guidelines

| Model Size | Quantization | Fits in Memory? | Dual-Spark? |
|---|---|---|---|
| 7–13B | Any | Yes — comfortable | N/A |
| 30B | Q4/Q5 | Yes | N/A |
| 70B | Q4 | Yes (~40 GB) | N/A |
| 70B | FP16 | Tight (~140 GB) | N/A |
| 200B | Q4 | Yes (tight) | N/A |
| 405B | Q4 | No | Yes (2× Spark) |

[NVIDIA-OFFICIAL] [COMMUNITY] [INFERRED]

### Dual-Spark Configuration

Two DGX Sparks connected via ConnectX-7 QSFP provide:
- 256 GB total unified memory [NVIDIA-OFFICIAL]
- Up to 405B parameter models [NVIDIA-OFFICIAL]
- 200 Gbps inter-node bandwidth [NVIDIA-OFFICIAL]

Configuration requires netplan setup (see Section 3.5). Community reports indicate the setup process is fragile. [COMMUNITY]

---

## Practical Implications

1. **The DGX Spark is best suited as a local development and prototyping workstation**, not a production training platform. Its thermal and bandwidth constraints limit sustained throughput, but its 128 GB unified memory and Blackwell architecture make it excellent for local model experimentation, inference serving, and rapid iteration. [INFERRED]

2. **A container-first workflow is mandatory**, not optional. The CUDA 13 wheel gap and aarch64 architecture mean native pip installs of GPU-accelerated packages will fail. NGC containers are the path of least resistance. [TESTED] [COMMUNITY]

3. **Memory limits on containers are critical safety measures.** The unified memory architecture turns GPU OOM into system-wide OOM. Without Docker memory limits, a single runaway process can render the device unresponsive. [COMMUNITY]

4. **Thermal management is an ongoing operational concern.** Unlike rackmount hardware with enterprise cooling, the Spark relies on a small internal fan in a compact enclosure. Users must actively manage ambient temperature, airflow clearance, and vent cleanliness. [COMMUNITY]

5. **The 1 PFLOP headline is misleading for most workloads.** Real-world inference throughput is memory-bandwidth-bound and comparable to high-end laptops for many LLM use cases. The Spark's true advantage is its 128 GB memory capacity, not raw FLOPS. [COMMUNITY] [TESTED]

6. **Dual-Spark clustering extends capability but adds fragility.** The ConnectX-7 link and netplan configuration are reported as pain points. Budget time for setup and debugging. [COMMUNITY]

---

## Recommended Actions

### Before First Power-On
1. Ensure ambient temperature is below 30°C with clearance on all sides [NVIDIA-OFFICIAL]
2. Have a stable power source (UPS recommended) — initial setup is non-resumable [COMMUNITY]
3. Have a wired Ethernet connection ready for initial setup [INFERRED]

### Immediately After Setup
1. Complete all DGX Dashboard firmware and software updates [NVIDIA-OFFICIAL]
2. Verify the Docker override configuration exists at `/etc/systemd/system/docker.service.d/override.conf` [COMMUNITY]
3. Test GPU access: `docker run --rm --runtime=nvidia --gpus all nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 nvidia-smi` [TESTED]
4. Set up the storage layout: `mkdir -p ~/models/ollama ~/data ~/projects` [NVIDIA-OFFICIAL]
5. Configure `OLLAMA_MODELS` environment variable [COMMUNITY]
6. Install TensorRT: `sudo apt install -y tensorrt` [NVIDIA-OFFICIAL]

### For Daily Workloads
1. Use containers for all GPU-accelerated workloads [COMMUNITY] [TESTED]
2. Always set `--memory` and `--oom-score-adj` on GPU containers [COMMUNITY]
3. Monitor temperatures via DGX Dashboard or `nvidia-smi` [INFERRED]
4. Use `uv` (not conda) for Python environment management [COMMUNITY]
5. Pull NGC base images for ML frameworks rather than pip-installing [COMMUNITY]

### For Long-Running Workloads
1. Implement checkpointing (no ECC protection on memory) [INFERRED]
2. Monitor thermal throttling — sustained 86°C indicates inadequate cooling [COMMUNITY]
3. Consider external cooling solutions for multi-hour training runs [COMMUNITY]
4. Set up automated temperature monitoring with alerts [INFERRED]

---

## Confidence Level

**HIGH** — with caveats.

**Reasoning**: The hardware specifications, software stack versions, and bundled benefits are sourced from NVIDIA's official documentation and product materials, which are highly reliable. The limitations and gotchas are corroborated by multiple independent community reports, user benchmarks, and (where noted) direct testing. The thermal throttling behavior, CUDA 13 wheel gap, and OOM zombie mode are well-documented across community forums, social media reports (including John Carmack's public statements), and user testing.

Areas of lower confidence:
- Exact thermal throttle thresholds may vary with firmware versions [COMMUNITY]
- NVMe performance numbers are inferred from the M.2 form factor class, not direct benchmarks [INFERRED]
- Some "best practices" are extrapolated from general Linux/Docker knowledge applied to the Spark's specific constraints [INFERRED]
- The ecosystem gap for CUDA 13 will narrow over time; this document reflects the state as of March 2026 [COMMUNITY]

---

## Source Notes

All claims in this document are tagged with one of the following source types:

| Tag | Meaning | Reliability |
|---|---|---|
| **[NVIDIA-OFFICIAL]** | From NVIDIA product pages, documentation, spec sheets, or official communications | High |
| **[COMMUNITY]** | From user reports, developer forums, social media (including John Carmack), blog posts, and community benchmarks | Medium-High (corroborated by multiple sources) |
| **[TESTED]** | Verified through direct testing on DGX Spark hardware | High |
| **[INFERRED]** | Logical inference from known facts, general engineering principles, or analogous systems | Medium (reasonable but not empirically verified on this specific hardware) |

Key sources:
- NVIDIA DGX Spark product page and technical specifications [NVIDIA-OFFICIAL]
- NVIDIA DGX OS 7.4.0 release notes [NVIDIA-OFFICIAL]
- NVIDIA NGC catalog documentation [NVIDIA-OFFICIAL]
- NVIDIA developer forums — DGX Spark category [COMMUNITY]
- John Carmack's public statements on thermal behavior (October 2025) [COMMUNITY]
- Community benchmark reports comparing Spark to Apple M3 Pro [COMMUNITY]
- build.nvidia.com/spark playbooks [NVIDIA-OFFICIAL]
- Direct hardware testing and validation [TESTED]

---

## Unresolved Questions

1. **What are the exact thermal throttle clock speed curves?** NVIDIA has not published the precise relationship between temperature and clock speed reduction. Community reports indicate throttling begins around 80–86°C but the exact behavior depends on firmware version. [COMMUNITY]

2. **When will PyPI wheels for major frameworks (PyTorch, TensorFlow) support CUDA 13 + aarch64 natively?** As of March 2026, NGC containers remain the only reliable path. The timeline for native pip support is unclear. [COMMUNITY]

3. **Will NVIDIA support 3+ device clustering in the future?** The current 2-device cap via ConnectX-7 is a hard limitation. NVIDIA has not announced plans for larger Spark clusters. [NVIDIA-OFFICIAL]

4. **What is the actual NVMe drive model and its specific performance characteristics?** NVIDIA specifies 4 TB NVMe M.2 but does not publish the drive vendor or sequential/random I/O benchmarks. [NVIDIA-OFFICIAL]

5. **Will future DGX OS updates address the HDMI deep sleep bug?** This has been reported by multiple users but there is no public acknowledgment or fix timeline from NVIDIA. [COMMUNITY]

6. **What is the long-term reliability profile of running at 86°C sustained?** High sustained temperatures typically reduce component lifespan, but no data exists for the GB10's specific reliability curve under thermal stress. [INFERRED]

7. **Is there a path to enable ECC-like error detection in software?** Some research exists on software-based memory error detection, but no practical implementation has been validated on DGX Spark. [INFERRED]

8. **What are the actual power consumption figures under various workload profiles (idle, inference, training)?** NVIDIA publishes the 240W PSU / 140W TDP figures but not detailed power profiling data. [NVIDIA-OFFICIAL]
