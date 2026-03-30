# NVIDIA DGX Spark — Hardware & Architecture Reference

*Last Updated: March 2026*

## Overview

The **NVIDIA DGX Spark** is a compact desktop AI supercomputer priced at **$4,699 USD**, powered by the **NVIDIA GB10 Grace Blackwell Superchip**. It combines a high-performance ARM CPU, Blackwell-class GPU with unified memory, and enterprise-grade networking in a small-footprint enclosure aimed at local LLM inference, fine-tuning, and CUDA-accelerated research without relying on discrete PCIe GPUs or separate VRAM pools.

This document summarizes hardware specifications, memory and GPU semantics, networking (including dual-unit linking), thermal and deployment constraints, and practical comparisons to other platforms.

---

## Core Specifications

| Category | Specification |
|----------|----------------|
| **Superchip** | NVIDIA GB10 Grace Blackwell |
| **GPU** | NVIDIA Blackwell architecture; 5th-generation Tensor Cores; native FP4 support; compute capability **sm_121** |
| **CPU** | 20-core high-performance ARM: **10× Cortex-X925** (performance) + **10× Cortex-A725** (efficiency); **aarch64** ISA |
| **AI performance** | Up to **1 petaFLOP** FP4 compute (marketing / peak; workload-dependent) |
| **System memory** | **128 GB** coherent, unified **LPDDR5x**; **~273 GB/s** bandwidth; shared CPU/GPU |
| **Storage** | **4 TB** NVMe M.2 with **self-encrypting drive (SED)** support |
| **Networking** | **ConnectX-7** NIC @ **200 Gbps**; **1× 10 GbE** RJ-45; **Wi-Fi 7**; **Bluetooth 5.4** |
| **USB** | **4× USB Type-C** |
| **Display** | **1× HDMI 2.1a** |
| **Dimensions** | **150 mm** (L) × **150 mm** (W) × **50.5 mm** (H) |
| **Weight** | **1.2 kg** |
| **Power supply** | **240 W** external |
| **GB10 TDP** | **140 W** (package / platform thermal design point; ambient and workload affect sustained clocks) |
| **Price** | **$4,699 USD** (MSRP / list; regional pricing may vary) |

---

## Unified Memory Architecture

The DGX Spark’s **128 GB unified LPDDR5x** is a single **coherent memory pool** visible to both the Grace-class CPU complex and the Blackwell GPU. Unlike a traditional **discrete GPU workstation**—where the CPU uses system DRAM and the GPU uses separate **GDDR/VRAM** over **PCIe**—the GB10 design **eliminates host↔device copies over PCIe** for ordinary unified allocations. The GPU can **address the full 128 GB** (subject to OS, driver, and framework reservation), which is the primary enabler for **very large models and KV caches** that would not fit in 24–32 GB of typical desktop VRAM.

### How it behaves in practice

- **Single address space semantics (conceptually):** Frameworks and CUDA unified memory map to a platform where large tensors and CPU working sets can coexist without explicit `cudaMemcpy` for every tensor—though application code still benefits from **data locality** (what the CPU touches vs what the GPU kernels read).
- **Bandwidth profile:** ~273 GB/s aggregate LPDDR5x bandwidth is **shared** across CPU, GPU, display/compression blocks, and I/O DMA. Peak **compute-bound** vs **memory-bound** behavior shifts compared to high-end discrete GPUs with very wide GDDR.
- **Capacity vs discrete VRAM:** The headline advantage is **capacity and simplicity of a single pool**, not necessarily matching the raw **per-device memory bandwidth** of a top-tier datacenter or enthusiast discrete GPU.

### Benefits

| Benefit | Detail |
|---------|--------|
| **Larger models locally** | Weights, activations, and KV cache can occupy tens to 100+ GB without splitting across multi-GPU NVLink systems (for sizes that still fit in 128 GB). |
| **Less PCIe bottleneck** | No routine **PCIe copy** path for unified allocations comparable to discrete dGPU host staging. |
| **Simpler mental model** | One **RAM budget** to monitor for “will this run?” sizing, once framework overhead is accounted for. |

### Caveats and operational risks

| Caveat | Implication |
|--------|----------------|
| **System-wide memory pressure** | Exhausting memory is not always a clean **`CUDA out of memory`** in a **GPU-only** sense; **OOM at the OS / unified allocator level** can **stall or freeze** interactive use more severely than on systems where only the GPU process dies. |
| **Monitoring** | Use **`nvidia-smi`**, framework logs, **`/proc/meminfo`**, **`free`**, and cgroup limits (if used) **together**—not GPU VRAM alone. |
| **Swap and I/O** | Heavy swapping on unified-memory pressure can **destroy latency**; prefer **hard caps**, **smaller batch sizes**, **quantization**, and **offload strategies** over relying on swap for AI workloads. |
| **Reservation** | The OS, desktop stack (if any), and NVIDIA driver stack consume a non-trivial slice; **effective headroom for models is < 128 GB**. |

**Actionable checklist**

1. Before long runs, record **baseline** `free -h` and GPU memory reporting after boot.
2. For LLM serving, size **KV cache + weights + overhead**; add **20–30%** margin if the stack is new to you.
3. Prefer **FP8/NVFP4** paths when quality allows to **reduce footprint and bandwidth demand**.
4. Run **headless** or minimal desktop services if you need every GB for workloads.

---

## GPU Architecture — Blackwell (sm_121)

- **5th-generation Tensor Cores** with **native FP4** execution paths for inference-oriented throughput.
- **Compute capability: `sm_121`** — use this when setting **CUDA architectures** / **Torch CUDA arch list** / **CUTLASS** targets for **native** kernels on Spark.
- **NVFP4 quantization** is supported in hardware-appropriate stacks: **~3.5× memory reduction vs FP16** (typical packing ratio narrative; exact factor depends on layout and metadata); **~1.6× faster than FP8** in **some** NVIDIA-reported inference scenarios (model and kernel dependent).
- **FP8** is also supported for training/inference mixes where FP4 is too aggressive.

### Build and binary compatibility note

**`sm_121` is newer than `sm_120`.** Many **prebuilt wheels and containers** ship PTX or SASS only up to **`sm_120`** (or older). On Spark you may need:

- Images or builds tagged **`linux/arm64`** **and** compiled for **`sm_121`** (or a build from source with the right **`-gencode`** flags).
- To avoid silent **JIT-from-PTX** surprises, **pin** known-good containers from NVIDIA NGC or vendor docs for DGX Spark / GB10.

Example (illustrative — adjust to your CUDA toolkit version):

```bash
# Example: CMake / nvcc style arch flag (verify against your CUDA version docs)
-gencode arch=compute_121,code=sm_121
```

---

## CPU Architecture — ARM Grace

- **Core complex:** **ARM Neoverse-class** mix — **10× Cortex-X925** + **10× Cortex-A725** (asymmetric performance / efficiency).
- **ISA:** **aarch64** (64-bit ARM). **Not x86_64.**

### Practical consequences

| Topic | Guidance |
|-------|----------|
| **Binary compatibility** | Typical **x86_64** Linux desktop binaries **will not run natively**. Use **aarch64** builds or **qemu-user** emulation (usually slower, not ideal for HPC). |
| **Docker / OCI** | Use **`linux/arm64`** images. Multi-arch manifests may pull the wrong arch if misconfigured—**pin digests** for reproducibility. |
| **Python wheels** | Prefer **manylinux aarch64** wheels from PyPI or vendor channels; build from source when absent. |
| **Cross-compilation** | For edge deployments, you may **cross-compile** on x86 build farms targeting **aarch64**; on-device compilation is viable for smaller projects. |

```bash
# Confirm userspace architecture on the device
uname -m          # expect: aarch64
docker version    # ensure buildx / platform flags if building images locally
```

---

## Networking & Connectivity

### ConnectX-7 (200 Gbps)

- **NVIDIA Mellanox ConnectX-7** class NIC for **high-speed** networking.
- Supports **multi-host** / **cluster-style** use cases when paired with correct **firmware, driver, and topology** (e.g., **NCCL**-friendly RDMA-capable paths where enabled).
- **Physical vs logical ports:** The platform may expose **4 logical ports** despite **2 physical** front-panel **QSFP** interfaces—this is tied to **GB10 PCIe lane budgeting (e.g., x4)** and **NIC port virtualization / bonding** behavior. **Always verify** with `ip link`, `mlxconfig`, and NVIDIA DGX Spark networking guides for your exact image revision.
- **QSFP** connectors for **DAC / AOC / optical** cables—use **vendor-qualified** cables for stable links.

### Other interfaces

| Interface | Role |
|-----------|------|
| **10 GbE RJ-45** | Standard LAN / internet uplink; useful for **NFS**, **apt**, **git**, **SSH** without burning 200G fabric. |
| **Wi-Fi 7** | High-throughput wireless for **office** setups; not a substitute for **NCCL** over RoCE/Ethernet in serious multi-node training. |
| **Bluetooth 5.4** | Keyboards, mice, audio, and low-duty peripherals. |

---

## Dual-Spark Linking (Spark Stacking)

Two DGX Spark systems can be **directly linked** via **ConnectX-7** using a **QSFP** cable, forming a **two-node memory-expanded** configuration for large-model strategies.

| Aspect | Detail |
|--------|--------|
| **Combined memory** | **256 GB** unified memory **across two nodes** (128 GB each)—enables **very large** sharded models (e.g., **Llama 3.1 405B** class **when properly sharded**, quantized, and with realistic batching—**verify** with your exact framework and precision). |
| **Collectives** | **NCCL** (NVIDIA Collective Communications Library) for **GPU-GPU / cross-node** communication patterns when using supported stacks. |
| **Bandwidth** | **200 Gbps** fabric potential with a **single QSFP** link (actual sustained depends on **protocol**, **MTU**, **PCIe/NIC tuning**, and **workload**). |
| **Network configuration** | Use **link-local** or **static IPs** on fabric interfaces; common naming includes **`enp1s0f0np0`** / **`enp1s0f1np1`** — **use `enp1*` only**. |
| **Ignore `enP2p*`** | Interfaces with the **`enP2p*`** prefix should be **disregarded** for Spark stacking fabric configuration; they are **not** the intended user-facing fabric pair for this workflow. |

```bash
# Example: list interfaces (illustrative)
ip -br link show

# Example: static IP on a fabric port (replace addresses/device)
sudo ip addr add 192.168.100.1/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up
```

**Cluster hygiene (actionable)**

- **Passwordless SSH** between nodes (key-based) for launcher scripts, **MPI**-style harnesses, and operational automation.
- **Consistent hostnames** in `/etc/hosts` or DNS.
- **Time sync** via **chrony** or **systemd-timesyncd** to avoid odd distributed logging and TLS issues.
- **Firewall rules** that **allow** NCCL’s ephemeral ports where required (or disable firewalls on isolated lab links only).

---

## Thermal Management

| Requirement / tip | Detail |
|-------------------|--------|
| **Ambient limit** | Keep **ambient temperature below ~30°C** to reduce **thermal throttling** risk under sustained AI loads. |
| **Ventilation** | Maintain **clearance on all sides**; do not stack objects that **block** intake/exhaust paths. |
| **Heat density** | Small volume + **140 W class** SoC means **concentrated heat**; avoid enclosed cabinets without **active airflow**. |
| **Acoustics / cooling** | **No obvious external fans** in the **fanless-style** industrial design narrative—**rely on internal thermal solution** and **environment**; expect **throttling** if the room is warm or dust accumulates. |

**Actionable:** Monitor **`nvidia-smi`** clocks under load, keep **dust-free**, and prefer **open desk** placement over **closed media furniture**.

---

## Physical Deployment

| Topic | Guidance |
|-------|----------|
| **Form factor** | **Desktop** footprint—sits beside a monitor or on a **shelf** with airflow. |
| **Power** | **External 240 W USB-C** power supply—use **only** the supplied or **NVIDIA-qualified** adapter. |
| **Display** | **HDMI 2.1a** for **direct** monitor attachment; suitable for **local UI** and **light** desktop use. |
| **Headless operation** | Run **without** a display using **SSH**, **remote desktop**, or **NVIDIA Sync** / companion tools from another machine—typical for **server-like** AI workflows. |

---

## Comparison Context

### vs Apple Mac Studio (e.g., M4 Ultra class)

| Dimension | DGX Spark | Mac Studio |
|-----------|-----------|------------|
| **Memory** | **128 GB** unified pool (GB10) | Comparable **unified memory** tiers on Apple Silicon (configuration-dependent) |
| **GPU / ML stack** | **CUDA**, **Blackwell Tensor Cores**, **NVFP4/FP8** in NVIDIA ecosystem | **Metal**, **MLX**; **no CUDA** |
| **Software breadth** | Best for **CUDA-first** research, **PyTorch/CUDA**, **Triton** (where supported), **NGC** stacks | Broader **desktop app** compatibility on **arm64** macOS |

### vs Desktop GPU workstation (e.g., RTX 4090 / 5090)

| Dimension | DGX Spark | Discrete dGPU workstation |
|-----------|-----------|---------------------------|
| **Memory capacity** | **128 GB unified** | Often **24–32 GB VRAM** per GPU (consumer cards) |
| **Memory model** | **No separate VRAM**; **no PCIe staging** for unified path | **VRAM + system RAM**; explicit copies common |
| **Throughput** | **Strong** for **large-model** residency; per-GPU **raw TFLOPS** comparisons are **workload-specific** | Often **higher peak** TFLOPS / bandwidth **per GPU** on flagship discrete cards |
| **Best fit** | **Very large** models / caches that **do not fit** 24 GB | **Smaller** models at **maximum** per-GPU throughput |

### vs cloud GPU instances

| Dimension | DGX Spark | Cloud GPUs |
|-----------|-----------|------------|
| **Economics** | **One-time** hardware cost; **no** hourly GPU rent | **Recurring** cost; **elastic** scale |
| **Privacy** | **Data stays on-prem** | Depends on **provider**, **VPC**, **contract** |
| **Availability** | **Always on** (power / cooling permitting) | Subject to **quota**, **region**, **maintenance** |

---

## Quick reference — interface naming for stacking

| Use | Prefix / pattern |
|-----|------------------|
| **Preferred fabric interfaces** | **`enp1s0f0np0`**, **`enp1s0f1np1`** (examples; confirm on device) |
| **Ignore** | **`enP2p*`** |

---

## Document maintenance

When NVIDIA publishes **revision-specific** errata (exact **QSFP lane maps**, **SED** key management, or **power** adapters), merge those details into **Core Specifications** and **Networking** from the **official DGX Spark** product documentation for your **firmware / software release**.
