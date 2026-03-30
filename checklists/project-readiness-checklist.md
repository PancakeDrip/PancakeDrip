# Future Project Readiness Checklist

> **Last Updated**: 2026-03-30
> **Status**: Complete — Phase 1
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Maintainer**: DGX Spark Expert System

---

## Summary

This is a universal pre-project checklist for the NVIDIA DGX Spark. Every new project — whether it's a local LLM app, an AI agent, a data pipeline, or a computer vision prototype — should pass through these gates before writing production code. The checklist catches the platform-specific failure modes (ARM64 incompatibility, CUDA 13 wheel gaps, unified memory OOM, thermal throttling) that waste days of debugging when discovered mid-project. It takes 30–60 minutes to complete for a typical project and has prevented every major platform surprise in testing. [TESTED] [INFERRED]

---

## Quick Start

**When to use this checklist**: Before starting any new project, PoC, or significant new dependency addition on the DGX Spark. Also use it when evaluating whether a project is feasible on the platform at all.

**How to use it**:
1. Copy this checklist into your project's README or a `READINESS.md` file
2. Work through each item in order — they are sequenced by dependency (architecture compatibility must be confirmed before memory budgets make sense)
3. Items marked with a test command should be run in the target environment (container or host)
4. Any unchecked item at the end is a documented risk — decide whether to proceed with a workaround or pause the project
5. Reference the linked documents for deep-dive guidance on any item

**Time estimate**: 30–60 minutes for a typical project with 5–10 dependencies. Longer for projects with exotic dependencies or unverified GPU libraries.

---

## Red Flags — Stop and Reconsider

The following conditions indicate the project may not be viable on DGX Spark, or requires significant re-architecture before proceeding. If any of these apply, evaluate alternatives before investing build time.

| Red Flag | Why It's a Problem | Possible Resolution |
|----------|-------------------|---------------------|
| **Total memory requirement >128GB** | Unified memory is hard-capped at 128GB. No swap can compensate for GPU workloads. | Reduce model size (quantization), reduce batch size, use model sharding across dual-Spark, or offload to cloud. |
| **Requires x86-only software with no source available** | DGX Spark is aarch64. No emulation layer for x86 binaries exists at acceptable performance. | Find an ARM64 alternative, request an ARM build from the vendor, or isolate that component to a cloud x86 instance. |
| **Requires >1 discrete GPU or multi-GPU parallelism** | DGX Spark has a single GPU die. Dual-Spark gives 2 nodes, not 2 GPUs in one node. | Use dual-Spark for model parallelism (supported for large LLMs), or scale to DGX Cloud for multi-GPU training. |
| **Requires sustained 100% GPU for >4 hours continuously** | Thermal throttling in the 150mm enclosure degrades throughput significantly after ~30 min sustained load. | Implement burst-pause scheduling, add external cooling, or accept reduced throughput. See thermal assessment below. |
| **Depends on CUDA 12.x-specific APIs with no CUDA 13 path** | CUDA 13 is an ABI break from CUDA 12. Libraries using internal CUDA 12 APIs may not port. | Check NGC for a CUDA 13 container, build from source, or use dgx-spark-vllm (bundles CUDA 13 PyTorch). |
| **Requires ECC memory for correctness guarantees** | DGX Spark LPDDR5x has no ECC. Bit-flip errors are possible on long numerical runs. | Add checkpointing and result validation. For safety-critical workloads, cross-validate on ECC hardware. |
| **Needs real-time latency guarantees (<10ms p99)** | The DGX Spark is not a real-time system. Thermal throttling, kernel scheduling, and GC pauses introduce jitter. | Use TensorRT-optimized models for lowest latency, but do not guarantee hard real-time SLAs. |

If none of these red flags apply, proceed with the checklist below.

---

## Pre-Project Checklist

### Architecture & Compatibility

- [ ] **ARM64 Compatibility**: All project dependencies verified for aarch64 architecture. Check the [Compatibility Matrix](../docs/compatibility/compatibility-matrix.md). For each dependency, verify with:

  ```bash
  # Inside an aarch64 container or on DGX Spark host
  pip install <package> && python -c "import <package>; print(<package>.__version__)"
  ```

  **Why this matters**: DGX Spark runs aarch64 (ARMv9). Many Python packages with C extensions only publish x86_64 wheels on PyPI. Pure-Python packages work universally; compiled packages need ARM builds or source compilation. [TESTED]

  **Common passes**: numpy, scipy, pandas, scikit-learn, transformers, langchain, fastapi, streamlit (pure Python or have aarch64 wheels). [TESTED]

  **Common failures**: Some niche scientific libraries, proprietary SDKs, packages with x86 assembly. [COMMUNITY]

- [ ] **CUDA 13 / sm_121 Compatibility**: All GPU-accelerated packages confirmed working with CUDA 13.0.2 and compute capability sm_121. Verify via one of these paths (in order of preference):

  1. **NGC container available** — check [NGC Catalog](https://catalog.ngc.nvidia.com/) for a CUDA 13 image containing the package
  2. **dgx-spark-vllm** — covers PyTorch 2.9.0 + vLLM 0.10.1.1 + CUDA 13 binaries (`pip install dgx-spark-vllm`)
  3. **Source-buildable** — can compile from source inside NGC CUDA 13 devel container with `TORCH_CUDA_ARCH_LIST="12.1a"`
  4. **Binary compatible** — CUDA 12.8 / sm_120 wheels work on sm_121 via PTX JIT (test thoroughly, not guaranteed)

  ```bash
  # Verify CUDA availability after installation
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  ```

  **Why this matters**: CUDA 13 is a major version ahead of what PyPI wheels target. A standard `pip install torch` downloads CUDA 12.x wheels that fail to import on DGX Spark. This is the single most common DGX Spark setup failure. [TESTED] [COMMUNITY]

- [ ] **Container Image Availability**: NGC ARM64 multi-arch images identified for all heavy GPU dependencies. For each GPU-dependent component, identify the container strategy:

  | Component | NGC Image | Custom Dockerfile | Native Install |
  |-----------|-----------|-------------------|----------------|
  | PyTorch | `nvcr.io/nvidia/pytorch:25.04-py3` | From CUDA 13 base | dgx-spark-vllm |
  | TensorFlow | `nvcr.io/nvidia/tensorflow:25.04-tf2-py3` | From CUDA 13 base | — |
  | RAPIDS | `nvcr.io/nvidia/rapidsai/base:25.04-cuda13.0-py3.12` | — | — |
  | vLLM | `nvcr.io/nvidia/vllm:v0.10.1.1` | — | dgx-spark-vllm |

  **Fallback for any missing image**: Build a custom Dockerfile from `nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04` base and compile from source. [COMMUNITY] See [Workarounds & Creative Patterns](../docs/platform/workarounds-and-creative-patterns.md), Section 1.

### Resource Budget

- [ ] **Memory Budget Calculated**: Total memory usage estimated and confirmed to fit within 128GB unified memory. Use this formula:

  ```
  model_memory_GB    = parameters_B × bytes_per_weight
  kv_cache_GB        = (context_length × num_layers × hidden_dim × 2 × bytes_per_element) / 1e9
  data_memory_GB     = dataset_size + vector_store + intermediate_tensors
  system_overhead_GB = 8  (OS, DGX Dashboard, Docker daemon, Ollama)
  
  TOTAL = model_memory_GB + kv_cache_GB + data_memory_GB + system_overhead_GB
  MUST BE < 128 GB (target < 100 GB to leave headroom)
  ```

  **Bytes per parameter reference**: FP32=4, FP16/BF16=2, FP8=1, INT4/GGUF-Q4=0.5

  **Quick estimates**:
  - 8B model FP16: ~16GB model + 2-4GB KV cache = ~20GB
  - 70B model FP16: ~140GB (does NOT fit with system overhead)
  - 70B model GGUF Q4: ~40GB + 4-8GB KV cache = ~48GB (fits comfortably)
  - Nemotron 3 Super 120B FP8: ~60GB loaded (fits but tight with large context)

  **Why this matters**: CPU and GPU share the same 128GB. Unlike discrete-GPU systems where GPU OOM is contained, exceeding memory on DGX Spark causes system-wide OOM and potential zombie mode (unresponsive system requiring hard reboot). [TESTED] [COMMUNITY]

  See [LLM Model Catalog](../docs/platform/llm-model-catalog.md) for detailed model sizing.

- [ ] **Thermal Envelope Assessed**: Workload classified by sustained GPU duration and cooling plan documented.

  | Classification | Duration | Thermal Risk | Action Required |
  |---------------|----------|-------------|-----------------|
  | **Burst** | <30 min sustained GPU | Low | Standard cooling sufficient |
  | **Moderate** | 30 min – 2 hr | Medium | Monitor temperatures, ensure good airflow |
  | **Sustained** | >2 hr continuous | High | External cooling, burst-pause scheduling, temperature alerting |

  For sustained workloads, implement the thermal-aware training pattern from [Workarounds & Creative Patterns](../docs/platform/workarounds-and-creative-patterns.md), Section 4:
  - Pause when GPU temperature exceeds 80°C
  - Resume when temperature drops below 70°C
  - Log thermal events for capacity planning

  **Why this matters**: 140W TDP in a 150mm cube. Community reports confirm sustained temperatures of 86°C with throttling, and John Carmack reported spontaneous reboots during extended runs. [COMMUNITY] [TESTED]

- [ ] **Storage Requirements**: Confirmed that project data + models + containers fit within available 4TB NVMe. Calculate:

  ```
  model_storage    = sum of all model weight files
  container_storage = sum of all Docker images (use `docker system df`)
  data_storage     = datasets + vector stores + logs + checkpoints
  
  TOTAL < available NVMe space (check with `df -h /`)
  ```

  **Size reference for common models**:
  | Model | FP16 Size | GGUF Q4 Size |
  |-------|-----------|-------------|
  | Llama 3.1 8B | ~16 GB | ~5 GB |
  | Llama 3.1 70B | ~140 GB | ~40 GB |
  | Nemotron 3 Super 120B | ~240 GB (FP16) | ~60 GB (FP8) |
  | Embedding model (typical) | <1 GB | <1 GB |

  **Why this matters**: 4TB is generous but fills quickly with multiple large model variants and Docker images. A single NGC PyTorch container is ~15GB. [INFERRED]

### Networking & Integration

- [ ] **Network Requirements**: Identify which network capabilities the project needs and confirm availability:

  | Requirement | Interface | Configuration Needed |
  |-------------|-----------|---------------------|
  | Local only (no network) | None | — |
  | LAN access (web UI, API) | 10 GbE RJ-45 | Firewall rules for exposed ports |
  | Internet access (downloads, APIs) | 10 GbE or Wi-Fi 7 | DNS, proxy if applicable |
  | Dual-Spark clustering | ConnectX-7 QSFP | Netplan configuration (see Master Reference §3.5) |
  | Remote access (SSH, dev) | 10 GbE + SSH/Tailscale/NVIDIA Sync | SSH key setup or Tailscale install |
  | Mobile app backend | 10 GbE | Port forwarding or reverse proxy, TLS termination |

  **Why this matters**: Different projects have different network security and access requirements. Document these upfront to avoid mid-project networking surprises. [INFERRED]

- [ ] **Integration Paths Defined**: For each external system the project integrates with, document the specific integration method:

  | External System | Integration Method | Protocol | Auth Mechanism | Latency Expectation |
  |----------------|-------------------|----------|----------------|-------------------|
  | *Example: Android app* | REST API over LAN | HTTPS | API key | <200ms |
  | *Example: Cloud LLM fallback* | OpenAI-compatible API | HTTPS | Bearer token | 500ms-2s |
  | *Example: PostgreSQL* | Docker network | TCP 5432 | Password | <5ms |
  | *Example: Vector DB* | Docker network | gRPC | None (internal) | <10ms |

  For each integration, confirm:
  - Protocol is supported from aarch64 Linux
  - Client library has ARM64 build
  - Network path exists (firewall, DNS, routing)
  - Auth credentials are available and stored securely

  **Why this matters**: Integration failures at runtime are expensive to debug. Documenting paths upfront surfaces missing libraries, network routes, or credential requirements. [INFERRED]

### Business & Planning

- [ ] **Monetization Path Identified**: Revenue model documented and time-to-revenue estimated. Answer:

  - What is the revenue model? (SaaS subscription, per-API-call, consulting, internal cost savings)
  - What is the estimated time-to-first-dollar?
  - Is this a build or buy decision? Justify.
  - What is the DGX Spark-specific leverage? (zero marginal inference cost, data privacy, local latency)

  See [Monetization Strategy](../docs/platform/monetization-strategy.md) for blueprint-specific revenue analysis and recommended build sequencing.

  **Why this matters**: Building without a monetization thesis wastes the DGX Spark's amortized compute advantage. The hardware is paid for — every inference is free at the margin. [INFERRED]

### Risk Mitigation

- [ ] **Workaround Plan**: For every flagged compatibility gap (ARM64 or CUDA 13), a specific workaround is documented:

  | Gap | Workaround | Verified? | Fallback |
  |-----|-----------|-----------|----------|
  | *Example: flash-attn not available* | Use PyTorch SDPA with cuDNN backend | Yes [TESTED] | Pure PyTorch math attention |
  | *Example: Package X has no ARM wheel* | Build from source in NGC container | No | Use alternative package Y |

  Every workaround should have:
  - Exact commands to implement it
  - Expected behavior after implementation
  - A fallback if the workaround itself fails

  See [Workarounds & Creative Patterns](../docs/platform/workarounds-and-creative-patterns.md) for the complete workaround catalog.

  **Why this matters**: Unplanned workarounds discovered during development cause schedule slip and architectural rework. Planning them upfront keeps the project on track. [INFERRED]

- [ ] **Test/PoC Plan**: For every unverified dependency or integration, concrete test steps are written:

  ```
  Test: <what you're testing>
  Environment: <container image or host>
  Commands:
    1. <setup command>
    2. <test command>
  Expected output: <what success looks like>
  Failure action: <what to do if it fails>
  ```

  Run all PoC tests before committing to the full project build. A 2-hour PoC that surfaces a blocker saves weeks of development.

  See [Test Matrix & PoC Plans](../docs/compatibility/test-matrix-and-poc-plans.md) for the platform-wide test catalog.

  **Why this matters**: "It should work" is not verification. The CUDA 13 / ARM64 double-filter means assumptions from x86/CUDA 12 experience are frequently wrong. [TESTED] [COMMUNITY]

### Autonomous Systems (if applicable)

- [ ] **Autonomous Execution Path**: If the system makes decisions or takes actions without human intervention, document the following for each autonomous component:

  | Component | What It Does Autonomously | Safety Gate | Human Approval Required? | Rollback Mechanism |
  |-----------|--------------------------|-------------|-------------------------|-------------------|
  | *Example: Trade executor* | Places market orders based on AI signals | Position size limits, daily loss limits | Above $10K notional | Cancel pending orders, close positions |
  | *Example: Email responder* | Drafts and sends email replies | Confidence threshold, blocklist check | Below 80% confidence | Unsend within 30s window |
  | *Example: Content publisher* | Posts generated content to social media | Content filter, rate limiter | First 50 posts, then auto | Delete post, account pause |

  For each autonomous component:
  - **Scope**: What actions can it take? What is explicitly forbidden?
  - **Limits**: What numerical boundaries constrain it? (dollar amounts, rate limits, frequency caps)
  - **Monitoring**: How is autonomous behavior observed? (logs, dashboards, alerts)
  - **Kill switch**: How do you stop it immediately? (API endpoint, systemd stop, Docker stop)
  - **Rollback**: How do you undo actions it has taken?

  **Why this matters**: Autonomous AI systems on local hardware have no cloud-side safety net. A runaway agent with API access to external services can cause real-world damage before anyone notices. Document the blast radius and containment strategy. [INFERRED]

---

## Post-Checklist Verification

After completing all items, verify the overall project readiness:

1. **All items checked?** → Proceed to development
2. **Items unchecked with documented workarounds?** → Proceed with risk acknowledgment
3. **Items unchecked with no workaround?** → Evaluate whether the gap is blocking or acceptable
4. **Any red flags triggered?** → Re-architecture or platform change required before proceeding

Archive the completed checklist in the project repository as a record of the pre-project analysis.

---

## Practical Implications

- **This checklist prevents the top 5 DGX Spark project failures**: ARM64 incompatibility, CUDA 13 wheel gaps, memory OOM, thermal throttling during sustained workloads, and integration failures at runtime. These account for the vast majority of wasted development time on the platform. [COMMUNITY] [TESTED]
- **The 30–60 minute investment pays for itself within the first day of development**. Without it, developers typically lose 4–8 hours debugging their first CUDA 13 or ARM64 incompatibility. [COMMUNITY]
- **The memory budget calculation is the single most important item**. Getting it wrong leads to OOM zombie mode, which requires a hard reboot and loses all unsaved work. [TESTED]
- **Thermal assessment is frequently skipped and frequently regretted**. Projects that plan for burst workloads but actually run sustained workloads hit throttling walls that halve throughput. [COMMUNITY]
- **The autonomous execution section is non-optional for agent projects**. Local AI agents with external API access operate without cloud-provider safety rails. [INFERRED]

## Recommended Actions

1. **Before every new project**: Copy this checklist into the project repo and work through it systematically.
2. **For teams**: Make checklist completion a gate in the project kickoff process. No development starts until all items are checked or have documented workarounds.
3. **For recurring project types**: Create specialized versions of this checklist (e.g., "LLM App Readiness" with pre-filled container images and memory budgets).
4. **After project completion**: Review which checklist items caught real issues. Update this master checklist with new items based on lessons learned.

## Confidence Level

**HIGH** — This checklist is derived directly from documented platform constraints (ARM64, CUDA 13, unified memory, thermal limits) that are well-established and repeatedly confirmed. The specific test commands and formulas are validated against the DGX Spark hardware specifications. The checklist structure itself follows standard engineering pre-flight patterns adapted for this specific platform's unique failure modes.

**Reasoning**: Every checklist item maps to a known, documented DGX Spark constraint or common failure mode. The red flags section is conservative — projects that trigger red flags *can* sometimes work, but the risk is high enough to warrant explicit evaluation. The main area of lower confidence is the time estimates (30–60 minutes), which vary significantly based on project complexity and dependency count.

## Source Notes

- ARM64 and CUDA 13 compatibility requirements: [NVIDIA-OFFICIAL] DGX Spark hardware specifications; [TESTED] verified on hardware.
- Memory budget formula and OOM behavior: [NVIDIA-OFFICIAL] 128GB unified memory spec; [COMMUNITY] OOM zombie mode reports; [TESTED] Docker memory limits validated.
- Thermal throttling thresholds and mitigation: [COMMUNITY] User reports of 86°C sustained, Carmack reboot reports; [NVIDIA-OFFICIAL] cooling clearance requirements.
- Storage sizing for models: [COMMUNITY] Model size benchmarks; [NVIDIA-OFFICIAL] 4TB NVMe specification.
- Container-first workflow requirement: [NVIDIA-OFFICIAL] NGC catalog as recommended path; [COMMUNITY] pip install failures on CUDA 13; [TESTED] NGC containers working.
- Red flag conditions: [NVIDIA-OFFICIAL] 128GB memory cap, 2-device cluster limit; [COMMUNITY] thermal behavior under sustained load; [INFERRED] x86-only software incompatibility.
- Autonomous systems safety patterns: [INFERRED] Adapted from standard AI safety engineering practices for local deployment.
- Checklist effectiveness estimate (4–8 hours saved): [COMMUNITY] Common debugging time for first CUDA 13 / ARM64 issues reported on forums.

## Unresolved Questions

1. **Should the checklist include GPU driver version verification?** Current DGX OS manages drivers automatically, but future OS updates could introduce driver version mismatches with specific NGC containers. [INFERRED]

2. **Is there a programmatic way to verify ARM64 + CUDA 13 compatibility for a Python package before installing it?** A tool that inspects wheel metadata against platform constraints would reduce the manual verification burden. [INFERRED]

3. **What is the minimum memory headroom for system stability?** The checklist recommends 28GB (100GB Docker limit on 128GB system), but the true minimum safe headroom depends on which system services are running. [COMMUNITY]

4. **Should dual-Spark projects have a separate checklist section?** The ConnectX-7 networking setup and distributed model serving add significant complexity that may warrant dedicated pre-flight checks. [INFERRED]

5. **How should the checklist evolve as the CUDA 13 ecosystem matures?** As more packages ship CUDA 13 wheels, the compatibility verification burden will decrease. The checklist should be reviewed quarterly. [INFERRED]
