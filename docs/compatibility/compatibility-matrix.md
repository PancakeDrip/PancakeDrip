# Compatibility Matrix

> **Last Updated**: 2026-03-30
> **Status**: Phase 1 — Complete
>
> **Baseline**: DGX Spark — ARM64 (aarch64) / CUDA 13.0.2 / sm_121 / Ubuntu 24.04 / DGX OS 7.4.0 / Python 3.12

## Summary

This matrix catalogs the compatibility of software libraries, frameworks, runtimes, and infrastructure tools with the NVIDIA DGX Spark workstation. The DGX Spark uses an ARM64 (Grace) CPU paired with a Blackwell GPU (sm_121 / CUDA 13.0.2), which is a relatively new architecture combination. The NVIDIA-supplied CUDA stack and NGC container ecosystem are fully supported. Most pure-Python packages work without modification, and many compiled packages have mature aarch64 wheels. The primary friction points are (1) PyTorch and vLLM requiring sm_121-aware build flags or binary-compatible sm_120 wheels, and (2) a small number of desktop/native tools that lack ARM64 Linux builds entirely.

## How to Read This Matrix

- **ARM64 Status**: ✅ Native | ⚠️ Works with workaround | ❌ Not available | ❓ Untested
- **CUDA 13 Status**: ✅ Compatible | ⚠️ Binary-compat or source build | ❌ Incompatible | ❓ Untested
- **Container**: ✅ NGC/Docker ARM64 image available | ⚠️ Community image | ❌ None | ➖ N/A
- **Verified**: How this was checked — `[NVIDIA-OFFICIAL]`, `[COMMUNITY]`, `[TESTED]`, `[INFERRED]`

---

## 1. Core CUDA Stack

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| CUDA Toolkit | 13.0.2 | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Pre-installed on DGX Spark. Blackwell sm_121 support included. |
| cuDNN | 9.13 | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Pre-installed. Paired with CUDA 13. |
| TensorRT | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Install via `sudo apt install tensorrt`. NGC container also available. |
| NCCL | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Single-GPU system; NCCL included in CUDA stack for API compatibility. |

## 2. ML Frameworks

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| PyTorch | 2.9+ | ⚠️ | ⚠️ | ✅ | [COMMUNITY] | CUDA 12.8 / sm_120 wheels are binary-compatible with sm_121. Official NGC containers work out of the box. For source builds, set `TORCH_CUDA_ARCH_LIST=12.1a`. See [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) and PyTorch Forums. |
| TensorFlow | 2.x / 3.x | ❓ | ❓ | ❓ | [INFERRED] | CUDA 13 ARM64 support is not well documented. TensorFlow's CUDA 12 wheels may work under binary compatibility but this is untested. Test plan: attempt `pip install tensorflow` and run a simple model on GPU. |
| JAX | Latest | ❓ | ❓ | ❓ | [INFERRED] | ARM64 + CUDA 13 status unknown. JAX has experimental ARM support but sm_121 is not yet confirmed. Test plan: install `jax[cuda13]` wheel and verify `jax.devices()` detects GPU. |
| scikit-learn | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python + C extensions. Pre-built aarch64 wheels available on PyPI. No GPU dependency. |
| HuggingFace transformers | Latest | ✅ | ✅ | ✅ | [COMMUNITY] | Pure Python library. Works with any compatible PyTorch backend. Use NGC PyTorch container or pip-installed PyTorch. |

## 3. Inference Engines

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| Ollama | Latest | ✅ | ✅ | ➖ | [NVIDIA-OFFICIAL] | Pre-installed on DGX Spark. Most reliable local inference option. Supports GGUF and safetensors models. |
| vLLM | 0.10.1.1 | ⚠️ | ⚠️ | ✅ | [COMMUNITY] | **Three installation paths**: (1) `pip install dgx-spark-vllm` — pre-compiled wheel bundling PyTorch 2.9.0 + vLLM 0.10.1.1; (2) build from source with `TORCH_CUDA_ARCH_LIST=12.1a`; (3) NGC container `nvcr.io/nvidia/vllm:25.11-py3`. Known CUTLASS sm_121 issues may surface during compilation. |
| llama.cpp | Latest | ✅ | ✅ | ⚠️ | [COMMUNITY] | Compiles natively on aarch64 with CUDA support. Good for GGUF-quantized models. Build with `cmake -DGGML_CUDA=ON`. Community Docker images available. |
| NVIDIA NIM | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Docker-based microservices. Pull from `nvcr.io`. Supports DGX Spark natively. |
| TGI (Text Generation Inference) | Latest | ❓ | ❓ | ❓ | [INFERRED] | HuggingFace TGI's ARM64 + CUDA 13 status is uncertain. Rust + CUDA components may require manual compilation. Test plan: attempt Docker pull of TGI image and verify GPU detection. |

## 4. Agent & AI Frameworks

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| NemoClaw | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | First-party NVIDIA agent framework with DGX Spark support. Uses OpenShell runtime. |
| NeMo | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | NeMo AutoModel for fine-tuning. Supports Blackwell architecture. NGC container available. |
| LangChain | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python. No architecture dependency. Communicates with LLM backends via HTTP. |
| LangGraph | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python. Built on LangChain; same compatibility profile. |
| CrewAI | Latest | ✅ | ➖ | ➖ | [COMMUNITY] | Pure Python multi-agent framework (57K+ GitHub stars). Works with local Ollama as LLM backend. |
| AutoGen | 0.4+ | ✅ | ➖ | ➖ | [INFERRED] | Pure Python (Microsoft). No native code dependencies. Connects to LLM backends via API. |

## 5. Data & Analytics

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| RAPIDS cuDF | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Available for DGX Spark via CUDA-X Data Science. GPU-accelerated DataFrame library. |
| RAPIDS cuML | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Available for DGX Spark. GPU-accelerated ML algorithms compatible with scikit-learn API. |
| cuOpt | Latest | ✅ | ✅ | ✅ | [NVIDIA-OFFICIAL] | Portfolio optimization playbook available. GPU-accelerated combinatorial optimization. |
| Polars | Latest | ✅ | ➖ | ➖ | [NVIDIA-OFFICIAL] | 25% faster on Grace CPU vs AMD x86 per NVIDIA benchmarks. Native aarch64 wheels on PyPI. |
| pandas | Latest | ✅ | ➖ | ➖ | [INFERRED] | aarch64 wheels available on PyPI. CPU-only; no CUDA dependency. |
| DuckDB | Latest | ✅ | ➖ | ➖ | [INFERRED] | aarch64 builds available. Embedded analytical database, no separate server process. |

## 6. Python Ecosystem

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| numpy | Latest | ✅ | ➖ | ➖ | [INFERRED] | aarch64 wheels on PyPI. Core numerical library; NEON SIMD optimized on ARM. |
| scipy | Latest | ✅ | ➖ | ➖ | [INFERRED] | aarch64 wheels on PyPI. Depends on BLAS/LAPACK (OpenBLAS aarch64 available). |
| matplotlib | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python + C extensions. aarch64 wheels available. |
| Pillow | Latest | ✅ | ➖ | ➖ | [INFERRED] | aarch64 wheels on PyPI. Image processing library with C extensions. |

## 7. Financial Libraries

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| zipline-reloaded | Latest | ❓ | ➖ | ➖ | [INFERRED] | ARM64 wheel availability uncertain. Has compiled Cython extensions. Test plan: attempt `pip install zipline-reloaded` and verify import succeeds. |
| backtrader | Latest | ❓ | ➖ | ➖ | [INFERRED] | Pure Python — likely works but untested on DGX Spark. Test plan: install and run sample backtest. |
| QuantLib | Latest | ❓ | ➖ | ➖ | [INFERRED] | C++ library with SWIG Python bindings. ARM64 compilation required; no pre-built aarch64 wheel on PyPI. Test plan: build from source with `cmake` and verify. |
| yfinance | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python REST client. No architecture dependency. |
| alpaca-py | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python REST client for Alpaca Markets API. |
| pandas-ta | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python. Depends on pandas and numpy (both aarch64-compatible). |
| vectorbt | Latest | ❓ | ➖ | ➖ | [INFERRED] | Has compiled Numba/C components. ARM64 compatibility uncertain. Test plan: install and run basic portfolio simulation. |
| PyPortfolioOpt | Latest | ✅ | ➖ | ➖ | [INFERRED] | Pure Python + scipy dependency. Both are aarch64-compatible. |

## 8. Container Images (NGC ARM64)

| Image | Tag | ARM64 | Verified | Notes |
|-------|-----|-------|----------|-------|
| `nvidia/cuda` | `13.0.0-cudnn-devel-ubuntu24.04` | ✅ | [NVIDIA-OFFICIAL] | Multi-arch manifest. Pulls correct ARM64 layer automatically. |
| `nvidia/pytorch` | Latest NGC | ✅ | [NVIDIA-OFFICIAL] | NGC ARM64 images available. Recommended for PyTorch workloads. |
| `nvidia/vllm` | `25.11-py3` | ✅ | [NVIDIA-OFFICIAL] | Pre-built for DGX Spark. Avoids manual sm_121 build issues. |
| `nvidia/tritonserver` | Latest | ❓ | [INFERRED] | ARM64 status unclear. Triton has experimental ARM support; verify tag before pulling. Test plan: `docker pull nvcr.io/nvidia/tritonserver:<tag>` and check `uname -m` inside container. |
| `nvidia/nemo` | Latest | ✅ | [NVIDIA-OFFICIAL] | Available for DGX Spark. Supports Blackwell fine-tuning workflows. |

## 9. Runtimes & Languages

| Component | Version | ARM64 | Verified | Notes |
|-----------|---------|-------|----------|-------|
| Node.js | 16+ | ✅ | [INFERRED] | Official aarch64 Linux binaries since v16. Install via `nvm` or NodeSource repos. |
| OpenJDK | 17+ / 21+ | ✅ | [INFERRED] | Mature aarch64 support via Adoptium Temurin and Ubuntu `openjdk-*` packages. |
| Rust | Latest | ✅ | [INFERRED] | `aarch64-unknown-linux-gnu` is a Tier 1 target. Install via `rustup`. |
| Go | 1.21+ | ✅ | [INFERRED] | `linux/arm64` is a first-class supported platform. |
| Kotlin compiler | Latest | ✅ | [INFERRED] | JVM-based; runs on OpenJDK aarch64. Install via SDKMAN or Snap. |

## 10. Databases

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| PostgreSQL | 14+ | ✅ | ✅ | [INFERRED] | Native ARM64 packages in Ubuntu repos. Official Docker image is multi-arch. |
| SQLite | 3.x | ✅ | ➖ | [INFERRED] | Included in Python stdlib (`sqlite3` module). No separate install needed. |
| Redis | 7+ | ✅ | ✅ | [INFERRED] | Compiles from source on ARM64. Official Docker image supports `linux/arm64`. |
| TimescaleDB | Latest | ❓ | ❓ | [INFERRED] | ARM64 Docker image availability uncertain. PostgreSQL extension; may compile from source if base PG is present. Test plan: check Docker Hub for `timescale/timescaledb` ARM64 manifests. |

## 11. Vector Databases

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| pgvector | Latest | ✅ | ✅ | [INFERRED] | PostgreSQL extension. Builds from source on ARM64 (`make && make install`). Included in some PG Docker images. |
| ChromaDB | Latest | ✅ | ✅ | [INFERRED] | Pure Python server. Runs on any platform with Python 3.10+. |
| Qdrant | Latest | ✅ | ✅ | [INFERRED] | Rust-based. Official aarch64 binaries and Docker images available. |
| Milvus | Latest | ⚠️ | ⚠️ | [INFERRED] | Has ARM64 Docker images, but requires complex multi-container setup (etcd, MinIO, Milvus). Consider Milvus Lite for simpler single-node deployment. |

## 12. Observability

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| Prometheus | Latest | ✅ | ✅ | [INFERRED] | Go binary. Official `linux/arm64` builds and Docker images available. |
| Grafana | Latest | ✅ | ✅ | [INFERRED] | ARM64 Docker images available on Docker Hub. Also installable via APT repo. |
| Loki | Latest | ✅ | ✅ | [INFERRED] | Go binary. Official `linux/arm64` builds available. Part of Grafana stack. |

## 13. Messaging & Queues

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| RabbitMQ | Latest | ✅ | ✅ | [INFERRED] | Erlang VM runs on ARM64. Official Docker image is multi-arch. |
| NATS | Latest | ✅ | ✅ | [INFERRED] | Go binary. Native `linux/arm64` builds and Docker images. Lightweight and fast. |
| Kafka | Latest | ⚠️ | ⚠️ | [INFERRED] | JVM-based core runs on OpenJDK ARM64, but native libraries (Snappy, LZ4, Zstd JNI) may need recompilation for aarch64. Use Bitnami ARM64 Kafka image or verify JNI libs. |
| ZeroMQ | Latest | ✅ | ➖ | [INFERRED] | C library (`libzmq`). Compiles cleanly on ARM64 via `apt install libzmq3-dev` or from source. |

## 14. Desktop & Android

| Component | Version | ARM64 Linux | Verified | Notes |
|-----------|---------|-------------|----------|-------|
| Electron | Latest | ⚠️ | [INFERRED] | ARM64 Linux builds exist but are less tested than x86. Some native Node addons may require recompilation. |
| Tauri | Latest | ✅ | [INFERRED] | Rust-based. `aarch64-unknown-linux-gnu` is a supported target. Lighter than Electron. |
| Android Studio | Latest | ❌ | [COMMUNITY] | No ARM64 Linux build exists. Must use a separate x86_64 machine or remote development workflow. See [StackOverflow discussions](https://stackoverflow.com/). |
| Gradle | Latest | ✅ | [INFERRED] | JVM-based build tool. Runs on OpenJDK aarch64 without issues. |
| ADB | Latest | ✅ | [INFERRED] | ARM64 Linux platform-tools available from Android SDK command-line tools. |

---

## Detailed Findings

### CUDA and GPU Compute

The DGX Spark ships with CUDA 13.0.2 and cuDNN 9.13 pre-installed, targeting the Blackwell B200 GPU with compute capability sm_121. This is the newest NVIDIA GPU architecture, and software ecosystem support is still catching up. The critical finding is that **CUDA 12.8 / sm_120 binaries are forward-compatible with sm_121**, which means most existing CUDA 12.x wheels and containers work without rebuilding. This binary compatibility is the linchpin that makes the broader ecosystem functional today.

### PyTorch and the sm_121 Gap

PyTorch does not yet ship wheels explicitly compiled for sm_121. However, the sm_120 wheels from CUDA 12.8 builds run correctly on sm_121 hardware through NVIDIA's binary compatibility guarantee. For users who need to compile custom CUDA kernels or build PyTorch from source (e.g., for research extensions), setting `TORCH_CUDA_ARCH_LIST=12.1a` is required. The NGC PyTorch container is the most friction-free path. The community project [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) documents tested configurations.

### vLLM Installation Complexity

vLLM is the second most complex dependency after PyTorch. Three viable installation paths exist, each with trade-offs: the `dgx-spark-vllm` PyPI package is easiest but pins specific PyTorch/vLLM versions; building from source gives flexibility but requires navigating CUTLASS sm_121 compilation issues; the NGC container is robust but requires Docker. For production deployments, the NGC container (`nvcr.io/nvidia/vllm:25.11-py3`) is recommended.

### ARM64 Ecosystem Maturity

The ARM64 Linux ecosystem is mature for server workloads. Go, Rust, Node.js, and the JVM all have first-class aarch64 support. Databases (PostgreSQL, Redis) and observability tools (Prometheus, Grafana) provide official ARM64 builds. The primary gap is in desktop tooling — notably Android Studio, which has no ARM64 Linux build at all.

### Financial Libraries

The financial Python ecosystem is largely compatible because most packages are pure Python or depend on numpy/scipy (which have aarch64 wheels). The exceptions are packages with compiled Cython or C++ components (zipline-reloaded, QuantLib, vectorbt) where ARM64 wheels may not be published, requiring source compilation.

## Practical Implications

1. **Use NGC containers as the default deployment strategy.** NGC images for PyTorch, vLLM, NeMo, and CUDA are multi-arch and pre-tested on DGX Spark. This eliminates sm_121 build issues entirely.

2. **Ollama is the lowest-friction inference path.** Pre-installed and fully supported, it should be the first choice for local LLM inference unless specific vLLM features (e.g., continuous batching, speculative decoding) are required.

3. **Pure Python packages "just work."** LangChain, CrewAI, AutoGen, HuggingFace transformers, yfinance, and similar pure-Python libraries have no architecture-specific issues.

4. **Source builds require `TORCH_CUDA_ARCH_LIST=12.1a`.** Any custom CUDA kernel compilation or from-source build of PyTorch/vLLM must include this flag. Without it, the compiler will not generate sm_121 code.

5. **Android development requires a separate machine.** Android Studio does not run on ARM64 Linux. Teams building Android apps should use a dedicated x86_64 development machine or remote IDE setup, with the DGX Spark serving as a model training/inference backend.

6. **RAPIDS provides native GPU acceleration for data science.** cuDF and cuML are officially supported on DGX Spark, making GPU-accelerated data processing available without container workarounds.

## Recommended Actions

1. **Immediate**: Use Ollama for inference workloads and NGC containers for training/fine-tuning. These are fully supported paths with no compatibility risk.

2. **Short-term**: Test TensorFlow and JAX on CUDA 13 + ARM64 and update this matrix with results. These frameworks have the largest remaining uncertainty.

3. **Short-term**: Validate financial libraries (zipline-reloaded, QuantLib, vectorbt) for ARM64 compilation. Publish findings and any required build flags.

4. **Ongoing**: Monitor PyTorch releases for native sm_121 wheel support. Once available, the workaround of using sm_120 binary-compatible wheels can be deprecated.

5. **Ongoing**: Track HuggingFace TGI and Triton Server for ARM64 + CUDA 13 support. Both are important inference serving tools with uncertain status.

6. **If needed**: For Android development workflows, set up a remote development environment or CI pipeline on x86_64 for Android Studio builds, using the DGX Spark only for ML model components.

## Confidence Level

| Category | Confidence | Rationale |
|----------|------------|-----------|
| Core CUDA Stack | **High** | NVIDIA-official pre-installed software. No ambiguity. |
| PyTorch | **Medium-High** | Community-verified binary compatibility. NGC containers tested. Source build path documented. |
| vLLM | **Medium** | Multiple installation paths confirmed by community. CUTLASS issues are a known pain point. |
| Ollama | **High** | Pre-installed by NVIDIA. First-party support. |
| NVIDIA NIM / NeMo / NemoClaw | **High** | NVIDIA first-party tools with explicit DGX Spark support. |
| RAPIDS | **High** | NVIDIA-official CUDA-X Data Science support. |
| Pure Python packages | **High** | No compiled components; architecture-agnostic by nature. |
| TensorFlow / JAX | **Low** | No verified data. CUDA 13 + ARM64 status genuinely uncertain. |
| Financial libraries (compiled) | **Low-Medium** | Compilation likely succeeds but untested. ARM64 wheel availability varies. |
| Infrastructure (DBs, queues, observability) | **Medium** | Inferred from upstream project documentation. ARM64 support is well-established for these tools but not tested specifically on DGX Spark. |
| Desktop & Android | **Medium-High** | Android Studio incompatibility is well-documented. Other tools have known ARM64 status. |

## Source Notes

| Tag | Description |
|-----|-------------|
| `[NVIDIA-OFFICIAL]` | Information from NVIDIA documentation, NGC catalog, DGX Spark user guide, or NVIDIA blog posts. Highest confidence. |
| `[COMMUNITY]` | Verified by community members. Sources include [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup), PyTorch Forums, StackOverflow, and GitHub issue threads. |
| `[TESTED]` | Directly tested on a DGX Spark system and confirmed working. (No entries in this revision — reserved for future hands-on validation.) |
| `[INFERRED]` | Derived from upstream project documentation (e.g., PyPI wheel availability, Docker Hub multi-arch manifests, language tier support lists). Not tested on DGX Spark specifically. |

**Key Community Sources:**
- [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup) — Community setup guide with tested PyTorch and vLLM configurations
- PyTorch Forums — Discussions on sm_121 compatibility and CUDA 13 support
- NGC Catalog (`catalog.ngc.nvidia.com`) — Container image availability and architecture support
- StackOverflow — Android Studio ARM64 Linux status discussions

## Unresolved Questions

1. **TensorFlow on CUDA 13 + ARM64**: Does `pip install tensorflow` produce a working GPU-enabled installation on DGX Spark? Which TF version first supports sm_121?

2. **JAX on CUDA 13 + ARM64**: Does the `jax[cuda13]` or `jax[cuda12]` wheel detect and use the Blackwell GPU correctly?

3. **Triton Inference Server ARM64**: Does `nvcr.io/nvidia/tritonserver` have a working ARM64 image for CUDA 13? Which tag should be used?

4. **HuggingFace TGI on DGX Spark**: Can TGI's Rust + CUDA components compile for sm_121? Is there a pre-built container?

5. **TimescaleDB ARM64 Docker**: Does the official `timescale/timescaledb` Docker image support `linux/arm64`?

6. **zipline-reloaded ARM64**: Do the Cython extensions compile on aarch64 without patches?

7. **QuantLib ARM64**: Does the SWIG Python binding build cleanly on ARM64 Ubuntu 24.04?

8. **vectorbt ARM64**: Do the Numba JIT-compiled components work correctly on aarch64?

9. **vLLM CUTLASS sm_121**: What is the current status of CUTLASS kernel compilation for sm_121? Are there specific CUTLASS versions that resolve known issues?

10. **Native sm_121 PyTorch wheels**: When will PyTorch ship official wheels compiled for sm_121 (eliminating the need for sm_120 binary compatibility)?
