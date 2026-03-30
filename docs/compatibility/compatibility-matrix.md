# Compatibility Matrix

> **Last Updated**: 2026-03-30 (Skeleton — Phase 0)
> **Status**: Awaiting Phase 1 content
>
> **Baseline**: DGX Spark — ARM64 (aarch64) / CUDA 13.0.2 / sm_121 / Ubuntu 24.04 / DGX OS 7.4.0

## Summary
<!-- 2-5 sentence overview -->

## How to Read This Matrix

- **ARM64 Status**: ✅ Native | ⚠️ Works with workaround | ❌ Not available | ❓ Untested
- **CUDA 13 Status**: ✅ Compatible | ⚠️ Binary-compat or source build | ❌ Incompatible | ❓ Untested
- **Container**: ✅ NGC/Docker ARM64 image available | ⚠️ Community image | ❌ None | ➖ N/A
- **Verified**: How this was checked — [NVIDIA-OFFICIAL], [COMMUNITY], [TESTED], [INFERRED]

## 1. Core CUDA Stack

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| CUDA Toolkit | 13.0.2 | | | | | |
| cuDNN | 9.13 | | | | | |
| TensorRT | | | | | | |
| NCCL | | | | | | |

## 2. ML Frameworks

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| PyTorch | | | | | | |
| TensorFlow | | | | | | |
| JAX | | | | | | |
| scikit-learn | | | | | | |
| HuggingFace transformers | | | | | | |

## 3. Inference Engines

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| Ollama | | | | | | |
| vLLM | | | | | | |
| llama.cpp | | | | | | |
| NVIDIA NIM | | | | | | |
| TGI | | | | | | |

## 4. Agent & AI Frameworks

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| NemoClaw | | | | | | |
| NeMo | | | | | | |
| LangChain | | | | | | |
| LangGraph | | | | | | |
| CrewAI | | | | | | |
| AutoGen | | | | | | |

## 5. Data & Analytics

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| RAPIDS cuDF | | | | | | |
| RAPIDS cuML | | | | | | |
| cuOpt | | | | | | |
| Polars | | | | | | |
| pandas | | | | | | |
| DuckDB | | | | | | |

## 6. Python Ecosystem

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| numpy | | | | | | |
| scipy | | | | | | |
| matplotlib | | | | | | |
| Pillow | | | | | | |

## 7. Financial Libraries

| Component | Version | ARM64 | CUDA 13 | Container | Verified | Notes |
|-----------|---------|-------|---------|-----------|----------|-------|
| zipline-reloaded | | | | | | |
| backtrader | | | | | | |
| QuantLib | | | | | | |
| yfinance | | | | | | |
| alpaca-py | | | | | | |
| pandas-ta | | | | | | |
| vectorbt | | | | | | |
| PyPortfolioOpt | | | | | | |

## 8. Container Images (NGC ARM64)

| Image | Tag | ARM64 | Verified | Notes |
|-------|-----|-------|----------|-------|
| nvidia/cuda | 13.0.x-devel-ubuntu24.04 | | | |
| nvidia/pytorch | | | | |
| nvidia/vllm | 25.11-py3 | | | |
| nvidia/tritonserver | | | | |
| nvidia/nemo | | | | |

## 9. Runtimes & Languages

| Component | Version | ARM64 | Verified | Notes |
|-----------|---------|-------|----------|-------|
| Node.js | | | | |
| OpenJDK | | | | |
| Rust | | | | |
| Go | | | | |
| Kotlin compiler | | | | |

## 10. Databases

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| PostgreSQL | | | | | |
| SQLite | | | | | |
| Redis | | | | | |
| TimescaleDB | | | | | |

## 11. Vector Databases

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| pgvector | | | | | |
| ChromaDB | | | | | |
| Qdrant | | | | | |
| Milvus | | | | | |

## 12. Observability

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| Prometheus | | | | | |
| Grafana | | | | | |
| Loki | | | | | |

## 13. Messaging & Queues

| Component | Version | ARM64 | Container | Verified | Notes |
|-----------|---------|-------|-----------|----------|-------|
| RabbitMQ | | | | | |
| NATS | | | | | |
| Kafka | | | | | |
| ZeroMQ | | | | | |

## 14. Desktop & Android

| Component | Version | ARM64 Linux | Verified | Notes |
|-----------|---------|-------------|----------|-------|
| Electron | | | | |
| Tauri | | | | |
| Android Studio | | | | |
| Gradle | | | | |
| ADB | | | | |

## Detailed Findings
<!-- Phase 1 -->

## Practical Implications
<!-- Phase 1 -->

## Recommended Actions
<!-- Phase 1 -->

## Confidence Level
<!-- Phase 1 -->

## Source Notes
<!-- Phase 1 -->

## Unresolved Questions
<!-- Phase 1 -->
