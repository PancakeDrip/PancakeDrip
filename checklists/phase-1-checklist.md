# Phase 1 — DGX Spark Platform Mastery — Research & Writing Checklist

> **Status**: In Progress
> **Started**: 2026-03-30

## Prioritized Research Items

### Hardware Architecture (Master Reference §1–5)
- [ ] Confirm GB10 chip specifications from NVIDIA official docs
- [ ] Document CPU core configuration (10×X925 + 10×A725) and implications
- [ ] Document GPU specifications (sm_121, CUDA cores, Tensor Cores, RT Cores)
- [ ] Document unified memory architecture (128GB, 273 GB/s, NVLink C2C 900 GB/s)
- [ ] Document no-ECC implications for long-running workloads
- [ ] Document storage specs and performance (4TB NVMe, self-encrypting)
- [ ] Document all networking interfaces and configurations
- [ ] Document all I/O ports and known issues (HDMI sleep)
- [ ] Document thermal characteristics (140W TDP, throttling at 86°C, firmware patches)
- [ ] Document physical specifications and cooling requirements

### Software Stack (Master Reference §6–9)
- [ ] Confirm DGX OS version and Ubuntu/kernel base
- [ ] Confirm CUDA 13.0.2, cuDNN 9.13, TensorRT versions
- [ ] Confirm GPU driver version (580.126.09)
- [ ] Document pre-installed software (Ollama, Docker, JupyterLab, DGX Dashboard, NVIDIA Sync)
- [ ] Document Docker + nvidia-container-toolkit configuration details
- [ ] Document NGC container catalog access and ARM64 multi-arch status
- [ ] Confirm AI Enterprise license details (90-day, what's included)
- [ ] Document NIM availability and setup

### Compatibility Matrix
- [ ] Fill all 14 matrix sections with verified data
- [ ] Mark uncertain entries with ❓ and link to test-matrix doc
- [ ] Tag every entry with verification source

### Base Environment Plan
- [ ] Evaluate conda on DGX Spark (miniforge/mambaforge aarch64, CUDA 13 channel gaps)
- [ ] Evaluate Poetry on DGX Spark (pure Python, works but doesn't solve CUDA gap)
- [ ] Evaluate pip+venv (simple, works)
- [ ] Evaluate uv (fastest, aarch64 binary, lockfiles)
- [ ] Document container strategy (NGC base images)
- [ ] Document storage layout recommendations
- [ ] Document network configuration
- [ ] Document security baseline

### Risks & Constraints
- [ ] Document all Critical risks with mitigations
- [ ] Document all High risks with mitigations
- [ ] Document all Medium risks with mitigations
- [ ] Document all Low risks with mitigations

### Best Uses / Worst Uses
- [ ] Define Excellent tier with rationale
- [ ] Define Good tier with rationale
- [ ] Define Marginal tier with rationale
- [ ] Define Poor/Avoid tier with rationale

### LLM Model Catalog
- [ ] Research optimal models per category (chat, code, agents, embeddings, vision)
- [ ] Document size tiers with memory budgets
- [ ] Document quantization trade-offs (FP16, FP8, INT4, GGUF)
- [ ] Document multi-model concurrent serving patterns
- [ ] Create memory budget calculator

### Workarounds & Creative Patterns
- [ ] Document all CUDA 13 wheel gap solutions
- [ ] Document Flash Attention replacement (SDPA)
- [ ] Document OOM zombie mode prevention
- [ ] Document thermal management strategies
- [ ] Document non-standard LLM usage patterns
- [ ] Document adapter/shim patterns

### Monetization Strategy
- [ ] Rank all 14 blueprints by revenue potential
- [ ] Estimate time-to-first-revenue per blueprint
- [ ] Estimate build cost per blueprint
- [ ] Analyze DGX Spark leverage per blueprint
- [ ] Map dependency graph across blueprints
- [ ] Determine recommended build sequence

## Writing Checklist
- [ ] All 9 documents follow the 7-section quality template
- [ ] All factual claims tagged with source type
- [ ] All recommendations include confidence level
- [ ] All risks include severity + mitigation + verification
- [ ] Phase 1 checklist in project-plan.md updated
- [ ] INDEX.md updated with Phase 1 status
