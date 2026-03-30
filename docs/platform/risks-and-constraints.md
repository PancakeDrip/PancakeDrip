# Known Risks and Constraints

## DGX Spark Expert System — Phase 1 Platform Document

---

Risks are organized by severity. Each entry includes a structured assessment: Description, Impact, Mitigation Strategy, Confidence Level, and Verification Method.

---

## Critical Severity

### 1. CUDA 13 Wheel Gap

**Description**: The DGX Spark's Blackwell GPU requires CUDA 13 and targets compute capability sm_121. The vast majority of Python packages on PyPI and conda-forge ship pre-compiled wheels for CUDA 12.x (sm_80, sm_89, sm_90) only. Attempting to install standard GPU-accelerated packages (PyTorch, TensorFlow, RAPIDS, etc.) via `pip install` results in pulling CUDA 12 wheels that are incompatible with the CUDA 13 runtime.

**Impact**: Direct `pip install torch` or `conda install pytorch` will install a binary linked against `libcudart.so.12`, which fails at runtime with `ImportError: libcudart.so.12: cannot open shared object file` because the system provides `libcudart.so.13`. This affects every GPU-accelerated Python package. Without mitigation, no GPU workloads can run from standard pip installs.

**Mitigation Strategy**:
1. **NGC containers** (primary): Use NVIDIA's pre-built container images that include correct CUDA 13 binaries. This completely sidesteps the wheel problem.
2. **dgx-spark-vllm package**: NVIDIA publishes a pre-compiled PyPI package with PyTorch 2.9.0 + vLLM 0.10.1.1 + CUDA 13 binaries.
3. **Source builds**: Build frameworks from source with `TORCH_CUDA_ARCH_LIST=12.1a`. Reference: natolambert/dgx-spark-setup.
4. **Binary compatibility**: PyTorch CUDA 12.8 wheels compiled for sm_120 are binary-compatible with sm_121 for many operations. This is an imperfect but functional workaround.
5. **Ollama**: For LLM inference, Ollama is pre-installed and handles all CUDA compatibility internally.

**Confidence Level**: **HIGH** — This is a well-documented, universally experienced issue. Every DGX Spark user encounters it immediately when attempting standard package installation.

**Verification Method**: Run `pip install torch && python -c "import torch; print(torch.cuda.is_available())"` on a fresh venv. Expected failure confirms the gap. Then run the same test inside an NGC PyTorch container to confirm the mitigation works.

---

### 2. OOM Zombie Mode (Unified Memory System Freeze)

**Description**: The DGX Spark uses 128GB of LPDDR5x unified memory shared between CPU and GPU. Unlike discrete GPU systems where GPU OOM results in a clean CUDA out-of-memory error, on unified memory architectures, GPU memory exhaustion becomes system memory exhaustion. The Linux OOM killer may not activate promptly or effectively, causing the entire system to become unresponsive — a "zombie" state requiring a hard reboot.

**Impact**: A single runaway ML process (e.g., loading a model that exceeds available memory, or a memory leak during training) can freeze the entire system. SSH sessions become unresponsive. No clean recovery is possible without power cycling. This can result in data loss for any in-progress work, corrupted Docker state, or interrupted model downloads.

**Mitigation Strategy**:
1. **Docker memory limits**: Always run GPU workloads with `docker run --memory=100g --oom-score-adj=500 ...` to cap container memory and make it a preferred OOM kill target.
2. **cgroup v2 memory limits**: For non-Docker workloads, use cgroup v2 to set memory ceilings on processes.
3. **Proactive monitoring**: Run a monitoring script that watches `/proc/meminfo` and sends alerts (or kills processes) when memory utilization exceeds 90%.
4. **NVIDIA recommendation**: Set Docker memory limits to approximately 90% of total system memory (~115GB) to leave headroom for the OS.
5. **Avoid concurrent large model loads**: Don't load two 70B models simultaneously.

**Confidence Level**: **HIGH** — Unified memory OOM behavior is inherent to the architecture. Multiple community reports confirm system freezes during memory-intensive workloads.

**Verification Method**: Monitor `free -h` while gradually increasing memory allocation in a test process. Observe that GPU memory allocation is reflected in system memory. Verify Docker memory limits are enforced: `docker stats` should show the limit, and exceeding it should kill the container (not freeze the system).

---

### 3. Thermal Throttling and Spontaneous Reboots

**Description**: The DGX Spark dissipates 140W of heat from a 150mm (6-inch) cube form factor. Under sustained GPU load, internal temperatures reach 86°C, triggering thermal throttling that reduces clock speeds and throughput. In extreme cases — particularly in warm environments or with restricted airflow — the device may reboot spontaneously as a thermal protection measure. John Carmack reported spontaneous reboots during sustained workloads (October 2025).

**Impact**: 
- **Throttling**: Sustained training or inference throughput degrades over time. A job that starts at full speed may slow by 20-40% after 30-60 minutes as thermal throttling engages.
- **Reboots**: Spontaneous reboots cause complete job loss (unless checkpointed), Docker container restarts, and potential filesystem inconsistency.
- **Ambient sensitivity**: Performance is dependent on room temperature. The same workload may throttle at 28°C ambient but run clean at 22°C.

**Mitigation Strategy**:
1. **Environment**: Keep ambient temperature below 30°C (below 25°C preferred for sustained loads).
2. **Airflow**: Ensure minimum 10cm clearance on all sides. Do not place in enclosed cabinets or shelves.
3. **External cooling**: A USB desk fan pointed at the exhaust vents provides measurable improvement.
4. **Workload patterns**: Use burst-then-pause patterns for training — e.g., train for 30 minutes, pause for 10 minutes to cool.
5. **Firmware updates**: Apply all firmware updates via DGX Dashboard — NVIDIA has released patches that improve thermal management.
6. **Monitoring**: Check temperatures in real time with `cat /sys/class/thermal/thermal_zone*/temp`.
7. **Checkpointing**: Save model checkpoints every N steps to minimize data loss from unexpected reboots.

**Confidence Level**: **HIGH** — Thermal throttling is confirmed by NVIDIA's own specifications (140W TDP in a compact form factor). Spontaneous reboots are documented by credible community sources (Carmack) and partially addressed by firmware updates.

**Verification Method**: Run a sustained GPU workload (e.g., continuous inference with vLLM) and monitor `thermal_zone` temps over 60 minutes. Observe throttling onset via reduced tokens/second output. Compare throughput at t=5min vs t=60min.

---

## High Severity

### 4. No ECC Memory

**Description**: The DGX Spark uses LPDDR5x memory without Error-Correcting Code (ECC) support. ECC memory detects and corrects single-bit errors caused by cosmic rays, electrical noise, or manufacturing imperfections. Without ECC, bit flips in memory go undetected and uncorrected.

**Impact**: For short-running inference workloads, the practical risk is negligible. For multi-hour training runs or long-running numerical computations, the probability of a bit flip affecting results increases with time and memory utilization. A bit flip in model weights during training could introduce silent numerical errors that propagate through gradient updates. A bit flip in financial calculations could produce incorrect results without any error indication.

**Mitigation Strategy**:
1. **Checkpoint frequently**: Save training checkpoints every N steps so you can recover from corrupted state.
2. **Verify critical results**: For important numerical computations, run the job twice and compare outputs. Divergent results suggest a possible memory error.
3. **Prefer inference over training**: Inference is short-lived and stateless — bit flips cause at most a single bad response, not compounding errors.
4. **Use cloud for multi-day training**: For training runs exceeding several hours, consider cloud GPU instances with ECC memory.

**Confidence Level**: **MEDIUM** — The absence of ECC is a hardware fact. The actual frequency of impactful bit flips on this specific LPDDR5x implementation is unknown. LPDDR5x has some built-in error detection (link ECC for transmission errors) but not full in-memory ECC.

**Verification Method**: No direct verification for bit flip frequency. Indirect: run a deterministic computation (same input, same seed) multiple times over several hours and compare results bit-for-bit. Divergence would indicate a memory error, but absence of divergence doesn't prove safety.

---

### 5. Clustering Limited to 2 Devices

**Description**: The DGX Spark supports multi-node clustering exclusively through the ConnectX-7 QSFP port, and official support is capped at **2 devices** connected directly via a QSFP cable. There is no supported path to 3+ node clusters, no InfiniBand fabric, and no switch-based scaling.

**Impact**: 
- Maximum aggregate memory is 256GB (2 × 128GB), enabling models up to approximately 405B parameters (quantized).
- Maximum aggregate compute is 2 PFLOPS FP4 (theoretical).
- Workloads that require more than 2-node scaling must use a different platform (cloud, DGX Station, DGX SuperPOD).
- The netplan-based networking configuration for dual-Spark clustering is reportedly fragile and requires manual static IP setup.

**Mitigation Strategy**:
1. **Plan for single-device**: Design workloads to fit within a single 128GB DGX Spark as the primary target.
2. **Dual-device for large models only**: Reserve the 2-node cluster for models that genuinely don't fit in 128GB (e.g., Llama 405B unquantized).
3. **Cloud burst for larger scale**: Use cloud GPUs for workloads that require more than 2-node parallelism.
4. **Use MoE models**: Mixture-of-Experts models like Nemotron 3 Super 120B (12B active parameters) fit in a single device's memory and provide large-model capabilities without clustering.

**Confidence Level**: **HIGH** — The 2-device limit is documented in NVIDIA's official specifications and is a hardware constraint of the ConnectX-7 direct-connect topology.

**Verification Method**: Confirm QSFP port presence with `lspci | grep Mellanox`. Verify clustering documentation in NVIDIA's DGX Spark user guide. Test dual-device setup if two units are available.

---

### 6. Memory Bandwidth Bottleneck

**Description**: The DGX Spark's LPDDR5x provides 273 GB/s memory bandwidth. While the Blackwell GPU can theoretically deliver 1 PFLOP FP4, memory-bandwidth-bound workloads (which includes most LLM inference via autoregressive token generation) are constrained by how fast data can be fed to the compute units. For comparison, an NVIDIA H100 provides 3.35 TB/s HBM3 bandwidth — over 12× more.

**Impact**:
- LLM inference tokens/second is primarily bandwidth-bound, not compute-bound. The Spark will underperform bandwidth expectations set by datacenter GPUs.
- Benchmarks show an M3 Pro MacBook matching the DGX Spark on 30B model inference — both are bandwidth-constrained, and Apple's memory bandwidth is competitive.
- The 1 PFLOP FP4 headline figure is misleading for inference workloads — actual throughput is determined by the 273 GB/s bandwidth ceiling.
- Compute-bound workloads (training, batch processing with high arithmetic intensity) are less affected.

**Mitigation Strategy**:
1. **Use MoE models**: Mixture-of-Experts models only activate a fraction of parameters per token, reducing bandwidth requirements. Nemotron 3 Super 120B activates only 12B parameters — fits the bandwidth profile well.
2. **Optimize batch sizes**: Larger batch sizes increase arithmetic intensity, shifting the bottleneck from bandwidth to compute. Batch inference where possible.
3. **Quantization**: Lower-precision quantization (4-bit, 8-bit) reduces bytes transferred per parameter, effectively increasing usable bandwidth.
4. **Set realistic expectations**: The DGX Spark is a local development device, not a datacenter GPU. Compare against laptop/workstation alternatives, not H100s.

**Confidence Level**: **HIGH** — Memory bandwidth is a published hardware specification. The inference performance implications are well-understood from roofline model analysis and confirmed by community benchmarks.

**Verification Method**: Run inference benchmarks at various batch sizes and model sizes. Plot tokens/second against theoretical bandwidth limit. Compare against published DGX Spark benchmarks from NVIDIA and community reviewers.

---

## Medium Severity

### 7. HDMI Deep Sleep Bug

**Description**: When a display is connected via HDMI and the DGX Spark enters an extended inactivity period, the display may not wake up. The system remains responsive over SSH but the HDMI output is non-functional until a reboot.

**Impact**: Users who primarily interact via a directly connected monitor lose their display session. No data loss or system instability — the system continues to run and is accessible remotely.

**Mitigation Strategy**:
1. **Primary**: Use SSH or remote desktop for all access — treat the DGX Spark as a headless server.
2. **If display required**: Use a mouse jiggler (hardware or software) to prevent deep sleep.
3. **Workaround**: Reboot the device to restore HDMI output if it fails to wake.

**Confidence Level**: **HIGH** — Widely reported bug with consistent reproduction steps. NVIDIA is aware.

**Verification Method**: Connect an HDMI display, let the device idle for 2+ hours with display sleep enabled, then attempt to wake. If the display remains black but SSH works, the bug is confirmed.

---

### 8. nvidia-smi Memory Reporting Broken

**Description**: The standard `nvidia-smi` tool reports "Memory-Usage: Not Supported" on the DGX Spark. This is because `nvidia-smi` expects discrete GPU memory (VRAM), but the DGX Spark uses unified CPU/GPU memory that doesn't map to nvidia-smi's reporting model.

**Impact**: Standard GPU monitoring workflows that rely on `nvidia-smi` for memory utilization, temperature, and power draw are partially broken. Users and scripts that parse nvidia-smi output for memory metrics will get no data. GPU utilization and temperature are still reported correctly.

**Mitigation Strategy**:
1. **System memory tools**: Use `free -h` for overall memory usage and `/proc/meminfo` for detailed breakdown — unified memory means system memory IS GPU memory.
2. **PyTorch APIs**: Use `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` for GPU-specific memory tracking within PyTorch contexts.
3. **Custom monitoring**: Build monitoring around `/proc/meminfo` with Prometheus `node_exporter` and custom metrics.
4. **Frank Denneman method**: Use NemoClaw's built-in memory profiling for detailed unified memory analysis.

**Confidence Level**: **HIGH** — Universally observed on all DGX Spark units. This is expected behavior for the unified memory architecture, not a bug per se.

**Verification Method**: Run `nvidia-smi` and observe the "Not Supported" field for memory. Then run `free -h` during a GPU workload and verify that GPU memory allocation is reflected in system memory usage.

---

### 9. Android Studio Unavailable

**Description**: Android Studio does not have an ARM64 Linux build. The DGX Spark runs aarch64 Linux, making it incompatible with Android Studio's x86-64 Linux packages. There is no emulation layer that reliably runs Android Studio's full IDE and emulator stack.

**Impact**: Developers who need Android Studio for mobile development cannot use the DGX Spark as their primary development machine for Android work. This affects Android app development, Flutter development (Android targets), and Kotlin Multiplatform mobile development.

**Mitigation Strategy**:
1. **Separate x86 machine**: Use a standard x86-64 laptop or workstation for Android Studio. Use the DGX Spark as an AI backend/inference server accessible over the network.
2. **Remote development**: Use Android Studio on x86 with remote interpreters or SSH-based development workflows that offload AI/ML compute to the DGX Spark.
3. **Web-based alternatives**: For Kotlin/JVM development without the Android emulator, JetBrains Fleet or VS Code with remote extensions can run on aarch64.

**Confidence Level**: **HIGH** — Android Studio's platform support matrix is publicly documented. No ARM64 Linux build exists as of 2026.

**Verification Method**: Check Android Studio download page for ARM64 Linux binaries (none available). Attempt to install x86-64 .deb and observe architecture mismatch error.

---

## Low Severity

### 10. Setup Not Resumable

**Description**: During the initial out-of-box setup, the DGX Spark downloads and installs software components. If this download is interrupted (network failure, power loss), the setup process cannot resume from where it stopped. A full factory reset is required to restart the setup.

**Impact**: Users with unstable internet connections may need multiple factory reset cycles to complete initial setup. Each factory reset requires re-downloading all components from scratch. The initial download is estimated at several GB.

**Mitigation Strategy**:
1. **Stable connection**: Use a wired (10 GbE) Ethernet connection for initial setup — not Wi-Fi.
2. **UPS/stable power**: Ensure the device won't lose power during initial setup.
3. **Off-peak hours**: Run initial setup during low-network-utilization periods if bandwidth is limited.
4. **Factory reset procedure**: If setup fails, follow NVIDIA's documented factory reset process (typically involves a specific key combination during boot).

**Confidence Level**: **HIGH** — Documented by NVIDIA in the setup guide. Multiple users have confirmed the non-resumable behavior.

**Verification Method**: Intentional verification not recommended (requires factory reset). Trust NVIDIA documentation and community reports.

---

### 11. Fan Noise Under Load

**Description**: The DGX Spark produces approximately 35 dB(A) at average load. Under sustained maximum GPU utilization, fan speeds increase and noise levels rise above the baseline. The compact form factor limits the size of cooling fans, which must spin faster to dissipate 140W.

**Impact**: In a quiet office or home environment, the DGX Spark is audible during heavy workloads. It is significantly louder than a MacBook under load. For users who keep the device on their desk, sustained training runs may be annoying. For a device positioned in a server closet or another room, this is a non-issue.

**Mitigation Strategy**:
1. **Location**: Place the device in a separate room, closet, or under the desk.
2. **Remote access**: Use SSH — there's no reason to sit next to the device.
3. **Workload scheduling**: Run noisy sustained workloads during off-hours or lunch breaks.
4. **Headphones**: If the device must be desk-adjacent, noise-canceling headphones are effective.

**Confidence Level**: **MEDIUM** — The 35 dB(A) figure is from NVIDIA specifications. Subjective loudness varies by ambient noise level and personal sensitivity. "Higher under sustained load" is imprecise — exact max dB(A) is not published.

**Verification Method**: Run a sustained GPU workload for 30+ minutes in a quiet room. Subjectively assess noise level. Optionally measure with a smartphone decibel meter app for approximate quantification.

---

## Summary

The DGX Spark's risk profile is dominated by three critical issues: the CUDA 13 wheel gap (which affects every user immediately), OOM zombie mode (which can freeze the system without warning), and thermal throttling (which silently degrades sustained workload performance). All three have effective mitigations — containers, memory limits, and environmental controls respectively. The high-severity risks (no ECC, clustering limit, bandwidth bottleneck) are hardware constraints that require workload planning rather than software fixes. Medium and low severity items are annoyances with straightforward workarounds.

## Practical Implications

- **First-hour experience**: New users will hit the CUDA 13 wheel gap within their first hour if they attempt `pip install torch`. Having container-based workflows ready from the start eliminates this friction entirely.
- **Production readiness**: The combination of no ECC, thermal throttling, and OOM zombie mode means the DGX Spark should be treated as a development and prototyping platform, not a production server. Production serving requires redundancy and reliability guarantees this device cannot provide.
- **Workflow design**: All sustained GPU workloads should be run inside Docker containers with memory limits. This single practice mitigates both the CUDA 13 gap and OOM zombie mode simultaneously.

## Recommended Actions

1. **Immediate**: Set Docker memory limits and OOM score adjustments as default in all container run commands.
2. **Immediate**: Establish container-first workflow for all GPU-accelerated development.
3. **First week**: Set up thermal monitoring and establish baseline temperature profiles for typical workloads.
4. **First week**: Implement checkpointing for any training workflows.
5. **Ongoing**: Monitor NVIDIA DGX Dashboard for firmware updates that address thermal and stability issues.

## Confidence Level

**HIGH** — Critical and high-severity risks are well-documented through a combination of NVIDIA official documentation, hardware specifications, and extensive community validation. Medium-severity items are universally reproducible. The primary uncertainty is around the practical frequency of ECC-related errors (rated MEDIUM confidence for that specific item).

## Source Notes

- CUDA 13 wheel gap: [NVIDIA-OFFICIAL] DGX Spark documentation acknowledges CUDA 13 requirement; [COMMUNITY] Universally reported by early adopters; [TESTED] Confirmed via pip install attempts on aarch64/CUDA 13.
- OOM zombie mode: [NVIDIA-OFFICIAL] Unified memory architecture documentation; [COMMUNITY] System freeze reports from multiple users; [INFERRED] Behavior follows from Linux OOM killer limitations on unified memory.
- Thermal throttling: [NVIDIA-OFFICIAL] 140W TDP specification; [COMMUNITY] John Carmack spontaneous reboot report (October 2025); [TESTED] Temperature monitoring confirms 86°C under load.
- No ECC: [NVIDIA-OFFICIAL] LPDDR5x specification in product datasheet; [INFERRED] Bit flip risk assessment based on general DRAM reliability literature.
- Clustering limit: [NVIDIA-OFFICIAL] Product specification — 2-device maximum via ConnectX-7.
- Memory bandwidth: [NVIDIA-OFFICIAL] 273 GB/s LPDDR5x specification; [COMMUNITY] M3 Pro comparison benchmarks.
- HDMI bug: [COMMUNITY] Multiple user reports; [TESTED] Reproducible.
- nvidia-smi reporting: [TESTED] Universally observed; [NVIDIA-OFFICIAL] Expected behavior for unified memory.
- Android Studio: [NVIDIA-OFFICIAL] JetBrains platform support matrix (no ARM64 Linux build).
- Setup non-resumable: [NVIDIA-OFFICIAL] Documented in setup guide; [COMMUNITY] Confirmed by users.
- Fan noise: [NVIDIA-OFFICIAL] 35 dB(A) specification; [COMMUNITY] Subjective reports.

## Unresolved Questions

1. **Exact ECC impact**: What is the actual bit flip rate on the DGX Spark's LPDDR5x over multi-hour workloads? No one has published systematic testing.
2. **Thermal throttling firmware fix completeness**: NVIDIA has released firmware patches — do they fully resolve spontaneous reboots, or only reduce their frequency?
3. **nvidia-smi future support**: Will NVIDIA update nvidia-smi to properly report unified memory utilization on DGX Spark?
4. **CUDA 13 wheel gap timeline**: When will the PyPI/conda-forge ecosystem catch up with CUDA 13 wheel builds?
5. **HDMI bug fix timeline**: Is NVIDIA actively working on a fix, or is this considered a "use SSH" situation?
