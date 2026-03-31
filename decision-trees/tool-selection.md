# Decision Trees for Tool Selection — DGX Spark Expert System

> **Last Updated**: 2026-03-31
> **Status**: Phase 2 — Complete
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Baseline**: ARM64 (aarch64) · CUDA 13.0.2 · sm_121 · Ubuntu 24.04 · 128 GB Unified Memory
> **Maintainer**: DGX Spark Expert System

---

## Overview

These decision trees guide tool selection for common DGX Spark workflows. Each tree is a Mermaid flowchart that can be rendered in any Markdown viewer with Mermaid support (GitHub, VS Code, Obsidian, etc.).

### How to Use

1. Start at the top of the relevant decision tree
2. Follow the decision nodes based on your requirements
3. Arrive at a recommended tool
4. Cross-reference with the [Ecosystem Catalog](../docs/tooling/ecosystem-catalog.md) for detailed compatibility information

---

## Table of Contents

1. [Which Inference Engine?](#1-which-inference-engine)
2. [Which Vector Database?](#2-which-vector-database)
3. [Which Agent Framework?](#3-which-agent-framework)
4. [Container or Native?](#4-container-or-native)
5. [Build from Source or Wait?](#5-build-from-source-or-wait)
6. [Which Web UI Framework?](#6-which-web-ui-framework)
7. [Which Database?](#7-which-database)
8. [Which Orchestration Tool?](#8-which-orchestration-tool)

---

## 1. Which Inference Engine?

Choose the right inference engine based on model format, performance needs, and setup complexity.

```mermaid
flowchart TD
    Start([🚀 Need LLM Inference]) --> ModelFormat{What model format?}
    
    ModelFormat -->|GGUF| GGUFPath{Need easiest setup?}
    ModelFormat -->|HuggingFace / PyTorch| HFPath{Primary goal?}
    ModelFormat -->|Unknown / Flexible| FlexPath{What matters most?}
    
    GGUFPath -->|Yes| Ollama1[✅ Ollama<br/>Pre-installed, simplest setup<br/>Scores: 5/5/5]
    GGUFPath -->|Need fine control| LlamaCpp1[✅ llama.cpp<br/>Build from source<br/>GPU layer control<br/>Scores: 4/4/4]
    
    HFPath -->|Max throughput| vLLM1[⚠️ vLLM<br/>dgx-spark-vllm or NGC<br/>Highest throughput<br/>Scores: 5/4/3]
    HFPath -->|Production optimization| NIM1[✅ NIM<br/>Pre-installed / NGC<br/>Auto-optimization<br/>Scores: 5/5/5]
    HFPath -->|HF ecosystem integration| TGI1[⚠️ TGI<br/>NGC container<br/>HF Hub native<br/>Scores: 4/4/3]
    
    FlexPath -->|Simplest setup| Ollama2[✅ Ollama<br/>Pre-installed<br/>Multi-model LRU cache]
    FlexPath -->|Throughput| vLLM2[⚠️ vLLM<br/>PagedAttention<br/>Continuous batching]
    FlexPath -->|Production-ready| NIM2[✅ NIM<br/>TensorRT-LLM optimized]
    
    Ollama1 --> MultiModel{Need multi-model?}
    MultiModel -->|Yes| OllamaMulti[Ollama LRU cache<br/>Auto model switching]
    MultiModel -->|Dedicated per model| vLLMMulti[Multiple vLLM instances<br/>or NIM containers]
    
    style Ollama1 fill:#2d6a2d,color:#fff
    style Ollama2 fill:#2d6a2d,color:#fff
    style NIM1 fill:#2d6a2d,color:#fff
    style NIM2 fill:#2d6a2d,color:#fff
    style LlamaCpp1 fill:#2d6a2d,color:#fff
    style vLLM1 fill:#b8860b,color:#fff
    style vLLM2 fill:#b8860b,color:#fff
    style TGI1 fill:#b8860b,color:#fff
    style OllamaMulti fill:#2d6a2d,color:#fff
    style vLLMMulti fill:#b8860b,color:#fff
```

**Quick Reference**:
| Need | Recommendation |
|------|---------------|
| Fastest to start | Ollama (pre-installed) |
| Highest throughput | vLLM (dgx-spark-vllm) |
| Production serving | NIM (TensorRT-LLM) |
| GGUF with control | llama.cpp (build from source) |
| Multi-model switching | Ollama (LRU cache) |
| HuggingFace native | TGI (NGC container) |

---

## 2. Which Vector Database?

Choose based on scale, existing infrastructure, and use case.

```mermaid
flowchart TD
    Start([🔍 Need Vector Search]) --> Scale{How many vectors?}
    
    Scale -->|< 100K| SmallScale{Priority?}
    Scale -->|100K - 5M| MediumScale{Already using PostgreSQL?}
    Scale -->|5M - 100M| LargeScale[✅ Qdrant<br/>Docker container<br/>Best filtering<br/>Scores: 4/5/5]
    Scale -->|> 100M| HugeScale[❓ Milvus<br/>Consider if truly needed<br/>Heavy on single-node<br/>Scores: 3/4/4]
    
    SmallScale -->|Simplest possible| ChromaDB1[✅ ChromaDB<br/>pip install, in-process<br/>5-line API<br/>Scores: 4/4/5]
    SmallScale -->|Already have PostgreSQL| pgvector1[✅ pgvector<br/>SQL + vector in one DB<br/>Scores: 5/5/5]
    
    MediumScale -->|Yes| pgvector2[✅ pgvector<br/>Add extension to existing PG<br/>HNSW index<br/>Scores: 5/5/5]
    MediumScale -->|No| MediumChoice{Need metadata filtering?}
    
    MediumChoice -->|Advanced filtering| Qdrant1[✅ Qdrant<br/>Best filter performance<br/>Docker]
    MediumChoice -->|Basic filtering| pgvector3[✅ pgvector<br/>SQL WHERE + vector<br/>One less service]
    
    HugeScale --> ReallyNeed{Can you shard or reduce?}
    ReallyNeed -->|Yes, reduce to < 100M| LargeScale
    ReallyNeed -->|No, need 100M+| MilvusDeploy[Milvus<br/>⚠️ Heavy deployment<br/>Delay if possible]
    
    style ChromaDB1 fill:#2d6a2d,color:#fff
    style pgvector1 fill:#2d6a2d,color:#fff
    style pgvector2 fill:#2d6a2d,color:#fff
    style pgvector3 fill:#2d6a2d,color:#fff
    style Qdrant1 fill:#2d6a2d,color:#fff
    style LargeScale fill:#2d6a2d,color:#fff
    style HugeScale fill:#b8860b,color:#fff
    style MilvusDeploy fill:#8b0000,color:#fff
```

**Quick Reference**:
| Scale | Default Choice |
|-------|---------------|
| Prototyping / < 100K | ChromaDB |
| Have PostgreSQL / < 5M | pgvector |
| Production + filtering / < 100M | Qdrant |
| > 100M (rare on single-node) | Milvus (delay if possible) |

---

## 3. Which Agent Framework?

Choose based on features, ecosystem, and integration needs.

```mermaid
flowchart TD
    Start([🤖 Building AI Agents]) --> NVIDIANative{Need NVIDIA-native<br/>with security sandbox?}
    
    NVIDIANative -->|Yes| NemoClaw[✅ NemoClaw<br/>First-party DGX Spark<br/>Guardrails + sandbox<br/>Scores: 4/4/5]
    NVIDIANative -->|No| WorkflowType{Workflow type?}
    
    WorkflowType -->|Complex graph-based<br/>with state| LangGraph[✅ LangGraph<br/>Stateful graph workflows<br/>Largest ecosystem<br/>Scores: 4/4/5]
    WorkflowType -->|Simple role-based<br/>multi-agent| CrewAI[✅ CrewAI<br/>Intuitive role definitions<br/>Quick to prototype<br/>Scores: 4/3/5]
    WorkflowType -->|Multi-agent<br/>conversations| AutoGen[✅ AutoGen<br/>Conversation patterns<br/>Code execution<br/>Scores: 4/3/5]
    WorkflowType -->|.NET / Azure<br/>integration| SemanticKernel[✅ Semantic Kernel<br/>Enterprise patterns<br/>Plugin architecture<br/>Scores: 3/4/5]
    
    LangGraph --> LangGraphNote[Pair with LangChain<br/>for RAG + tools]
    NemoClaw --> NemoClawNote[Pair with NIM<br/>for inference backend]
    
    style NemoClaw fill:#76b900,color:#fff
    style LangGraph fill:#2d6a2d,color:#fff
    style CrewAI fill:#2d6a2d,color:#fff
    style AutoGen fill:#2d6a2d,color:#fff
    style SemanticKernel fill:#2d6a2d,color:#fff
```

**Quick Reference**:
| Need | Framework |
|------|-----------|
| NVIDIA-native + security | NemoClaw |
| Graph-based workflows | LangGraph |
| Role-based multi-agent | CrewAI |
| Conversation orchestration | AutoGen |
| .NET / Enterprise | Semantic Kernel |

---

## 4. Container or Native?

Decide whether to run a tool in a Docker container or install natively.

```mermaid
flowchart TD
    Start([📦 Install a New Tool]) --> GPUWorkload{GPU/ML workload?}
    
    GPUWorkload -->|Yes| AlwaysContainer[✅ Always use container<br/>NGC base image<br/>Guarantees CUDA 13 + sm_121]
    GPUWorkload -->|No| ToolType{What type of tool?}
    
    ToolType -->|Pure Python library| PurePython{Has aarch64 wheel?}
    ToolType -->|System service<br/>DB, queue, broker| SystemService{Complexity?}
    ToolType -->|Native binary<br/>Rust, Go, C++| NativeBinary[✅ Native install<br/>aarch64 binaries available]
    ToolType -->|Complex dependencies<br/>many C extensions| ComplexDeps[✅ Use container<br/>Isolate dependency chain]
    
    PurePython -->|Yes| NativeVenv[✅ Native + uv venv<br/>Fastest, lowest overhead]
    PurePython -->|No / Unsure| TestFirst{Test in venv first}
    TestFirst -->|Works| NativeVenv
    TestFirst -->|Fails| FallbackContainer[✅ Use container]
    
    SystemService -->|Simple, apt available| AptInstall[✅ apt install<br/>systemd managed]
    SystemService -->|Complex, multi-dep| DockerService[✅ Docker container<br/>Isolated, reproducible]
    
    AlwaysContainer --> NGCFirst{NGC container exists?}
    NGCFirst -->|Yes| UseNGC[✅ Use NGC container<br/>Pre-optimized for DGX Spark]
    NGCFirst -->|No| CustomContainer[Build custom container<br/>FROM nvidia/cuda:13.0-base]
    
    style AlwaysContainer fill:#b8860b,color:#fff
    style NativeVenv fill:#2d6a2d,color:#fff
    style NativeBinary fill:#2d6a2d,color:#fff
    style AptInstall fill:#2d6a2d,color:#fff
    style DockerService fill:#4169e1,color:#fff
    style UseNGC fill:#76b900,color:#fff
    style ComplexDeps fill:#4169e1,color:#fff
    style FallbackContainer fill:#4169e1,color:#fff
    style CustomContainer fill:#4169e1,color:#fff
```

**Rule of Thumb**:
| Tool Type | Install Method |
|-----------|---------------|
| GPU/ML workload | Container (NGC preferred) |
| Pure Python | Native + uv venv |
| System service (DB, cache) | apt install or Docker |
| Complex C/C++ dependencies | Container |
| Go/Rust binary | Native aarch64 binary |

---

## 5. Build from Source or Wait?

When a package doesn't have ready-made aarch64/cu130 wheels.

```mermaid
flowchart TD
    Start([🔧 Package Won't pip Install]) --> NGCContainer{NGC container available?}
    
    NGCContainer -->|Yes| UseNGC[✅ Use NGC container<br/>Fastest, most reliable path]
    NGCContainer -->|No| DGXVariant{dgx-spark-* PyPI variant?}
    
    DGXVariant -->|Yes| UseDGXPkg[✅ pip install dgx-spark-*<br/>Pre-built for DGX Spark]
    DGXVariant -->|No| BinaryCompat{sm_120 wheels available?}
    
    BinaryCompat -->|Yes| TestCompat[Test: pip install in venv<br/>then import + GPU op]
    BinaryCompat -->|No wheels at all| NoWheels{Is it urgent?}
    
    TestCompat -->|Works ✅| UseSmCompat[✅ Use sm_120 wheels<br/>Binary compatible with sm_121]
    TestCompat -->|Fails ❌| NoWheels
    
    NoWheels -->|Urgent, need now| BuildSource[🔧 Build from source<br/>TORCH_CUDA_ARCH_LIST=12.1a<br/>CMAKE_CUDA_ARCHITECTURES=121]
    NoWheels -->|Not urgent| WaitForWheels[⏳ Wait for cu130 wheels<br/>Track GitHub issue<br/>Re-check monthly]
    
    BuildSource --> BuildSuccess{Build succeeds?}
    BuildSuccess -->|Yes| TestBuild[Test GPU operations<br/>thoroughly]
    BuildSuccess -->|No| FileIssue[File GitHub issue<br/>Use alternative tool<br/>from Ecosystem Catalog]
    
    style UseNGC fill:#76b900,color:#fff
    style UseDGXPkg fill:#2d6a2d,color:#fff
    style UseSmCompat fill:#2d6a2d,color:#fff
    style BuildSource fill:#b8860b,color:#fff
    style WaitForWheels fill:#4169e1,color:#fff
    style FileIssue fill:#8b0000,color:#fff
    style TestBuild fill:#2d6a2d,color:#fff
```

**Priority Order**:
1. NGC container (if available)
2. `dgx-spark-*` PyPI variant (if available)
3. sm_120 binary-compatible wheel (test first)
4. Build from source (`TORCH_CUDA_ARCH_LIST=12.1a`)
5. Wait for upstream cu130 wheel

---

## 6. Which Web UI Framework?

Choose based on purpose, audience, and development speed.

```mermaid
flowchart TD
    Start([🖥️ Need a Web Interface]) --> Purpose{Primary purpose?}
    
    Purpose -->|Quick ML demo<br/>few inputs/outputs| Gradio[✅ Gradio<br/>Fewest lines of code<br/>Built-in ML components<br/>Scores: 5/5/5]
    Purpose -->|Data dashboard<br/>interactivity| Streamlit[✅ Streamlit<br/>Python-only<br/>Best for data apps<br/>Scores: 5/5/5]
    Purpose -->|Full web app<br/>custom UI| FullApp{Audience?}
    Purpose -->|Internal tool| Streamlit2[✅ Streamlit<br/>Fastest for internal tools]
    Purpose -->|API only<br/>no frontend| FastAPI[✅ FastAPI<br/>Automatic OpenAPI docs<br/>Scores: 5/5/5]
    
    FullApp -->|Public-facing product| NextJS[✅ Next.js + FastAPI<br/>React SSR + Python API<br/>Scores: 4/5/5]
    FullApp -->|Internal with<br/>custom components| StreamlitCustom[✅ Streamlit<br/>with custom components]
    
    Gradio --> GradioNote[Perfect for:<br/>- Model comparison UIs<br/>- HuggingFace Spaces<br/>- Interactive testing]
    
    Streamlit --> StreamlitNote[Perfect for:<br/>- Monitoring dashboards<br/>- Data exploration<br/>- Model evaluation]
    
    style Gradio fill:#2d6a2d,color:#fff
    style Streamlit fill:#2d6a2d,color:#fff
    style Streamlit2 fill:#2d6a2d,color:#fff
    style StreamlitCustom fill:#2d6a2d,color:#fff
    style FastAPI fill:#2d6a2d,color:#fff
    style NextJS fill:#4169e1,color:#fff
```

**Quick Reference**:
| Need | Framework | Lines to Demo |
|------|-----------|:---:|
| ML model demo | Gradio | ~5 |
| Data dashboard | Streamlit | ~20 |
| Internal tool | Streamlit | ~50 |
| Full web app | Next.js + FastAPI | ~200+ |
| API only | FastAPI | ~10 |

---

## 7. Which Database?

Choose based on data type, query patterns, and scale.

```mermaid
flowchart TD
    Start([💾 Need Data Storage]) --> DataType{Primary data type?}
    
    DataType -->|Structured / relational| SQL{Scale & concurrency?}
    DataType -->|Vector embeddings| VectorDB[See Decision Tree #2<br/>Vector Database Selection]
    DataType -->|Time-series| TimeSeries{Already have PostgreSQL?}
    DataType -->|Cache / sessions| Redis[✅ Redis<br/>Sub-ms latency<br/>Scores: 5/5/5]
    DataType -->|Key-value / pub-sub| Redis2[✅ Redis<br/>Versatile data structures]
    DataType -->|Analytical queries<br/>on files| DuckDB[✅ DuckDB<br/>In-process SQL on files<br/>Scores: 5/5/5]
    
    SQL -->|Single app, simple| SQLite[✅ SQLite<br/>Zero config, file-based<br/>Built into Python<br/>Scores: 5/5/5]
    SQL -->|Multi-user, production| PostgreSQL[✅ PostgreSQL<br/>+ pgvector for vectors<br/>Scores: 5/5/5]
    
    TimeSeries -->|Yes| TimescaleDB[✅ TimescaleDB<br/>PostgreSQL extension<br/>Scores: 4/5/5]
    TimeSeries -->|No, standalone| TSChoice{Volume?}
    TSChoice -->|Low-medium| TimescaleDB
    TSChoice -->|Already using<br/>Prometheus| PromStorage[Prometheus TSDB<br/>Already collecting metrics]
    
    PostgreSQL --> NeedVector{Also need vector search?}
    NeedVector -->|Yes| pgvectorAdd[Add pgvector extension<br/>One DB for everything]
    NeedVector -->|No| PGReady[PostgreSQL ready<br/>as-is]
    
    style Redis fill:#2d6a2d,color:#fff
    style Redis2 fill:#2d6a2d,color:#fff
    style SQLite fill:#2d6a2d,color:#fff
    style PostgreSQL fill:#2d6a2d,color:#fff
    style DuckDB fill:#2d6a2d,color:#fff
    style TimescaleDB fill:#2d6a2d,color:#fff
    style pgvectorAdd fill:#76b900,color:#fff
```

**Quick Reference**:
| Data Type | Default Choice |
|-----------|---------------|
| Relational (simple) | SQLite |
| Relational (production) | PostgreSQL |
| Vector embeddings | pgvector (with PG) or ChromaDB |
| Time-series | TimescaleDB (PG extension) |
| Cache / sessions | Redis |
| Analytical on files | DuckDB |

---

## 8. Which Orchestration Tool?

Choose based on workflow complexity, team size, and features needed.

```mermaid
flowchart TD
    Start([⚙️ Need Task Orchestration]) --> Complexity{Workflow complexity?}
    
    Complexity -->|Simple scheduled tasks| SimpleSchedule{Need monitoring UI?}
    Complexity -->|DAG-based pipelines| DAGPipeline{Priority?}
    Complexity -->|Low-code automation| n8n[✅ n8n<br/>Visual workflow builder<br/>400+ integrations<br/>Scores: 4/4/5]
    Complexity -->|In-app scheduling| APScheduler[✅ APScheduler<br/>Python library<br/>Embed in FastAPI<br/>Scores: 4/4/5]
    
    SimpleSchedule -->|No, just run it| CronChoice{Need restart on failure?}
    SimpleSchedule -->|Yes, basic UI| n8n
    
    CronChoice -->|No| Cron[✅ cron<br/>Simplest, zero overhead<br/>Pre-installed<br/>Scores: 5/5/5]
    CronChoice -->|Yes| SystemdTimers[✅ systemd timers<br/>Restart policies<br/>Journal logging<br/>Scores: 5/5/5]
    
    DAGPipeline -->|Most mature,<br/>largest community| Airflow[✅ Airflow<br/>Docker Compose deploy<br/>Extensive operators<br/>Scores: 4/5/5]
    DAGPipeline -->|Modern, easier setup| Prefect[✅ Prefect<br/>Pythonic, good UI<br/>Simpler than Airflow<br/>Scores: 4/4/5]
    DAGPipeline -->|Data-asset focused| Dagster[✅ Dagster<br/>Asset-centric<br/>Built-in lineage<br/>Scores: 4/4/5]
    
    Airflow --> AirflowNote[⚠️ Heavy: needs<br/>PostgreSQL + Redis<br/>for full deploy]
    
    style Cron fill:#2d6a2d,color:#fff
    style SystemdTimers fill:#2d6a2d,color:#fff
    style n8n fill:#2d6a2d,color:#fff
    style APScheduler fill:#2d6a2d,color:#fff
    style Airflow fill:#4169e1,color:#fff
    style Prefect fill:#4169e1,color:#fff
    style Dagster fill:#4169e1,color:#fff
```

**Quick Reference**:
| Need | Tool | Overhead |
|------|------|----------|
| Simple schedule | cron | Zero |
| Schedule + restart | systemd timers | Zero |
| In-app scheduling | APScheduler | None (library) |
| Visual low-code | n8n | Docker container |
| DAG pipelines (standard) | Airflow | Heavy (PG + Redis) |
| DAG pipelines (modern) | Prefect | Medium |
| Data asset pipelines | Dagster | Medium |

---

## Cross-Reference Matrix

For tools that appear in multiple decision trees:

| Tool | Appears In | Primary Role |
|------|-----------|-------------|
| Ollama | #1 Inference | Default LLM inference |
| PostgreSQL | #2 Vector, #7 Database | Relational + vector (with pgvector) |
| pgvector | #2 Vector, #7 Database | Vector search extension for PG |
| FastAPI | #6 Web UI | API framework |
| Docker/NGC | #4 Container | Container runtime |
| Redis | #7 Database, #8 Orchestration | Cache, queue, pub/sub |
| Streamlit | #6 Web UI | Data dashboards |
| NIM | #1 Inference | Production inference |

---

## Quality Template

### Summary

8 decision trees covering the most common tool selection decisions on DGX Spark: inference engine, vector database, agent framework, container vs. native, build strategy, web UI, database, and orchestration. Each tree provides clear paths based on real-world requirements.

### Practical Implications

- These trees encode the DGX Spark constraint knowledge (ARM64, CUDA 13, sm_121, container-first)
- Green nodes are safe choices; yellow nodes need caution; red nodes should be avoided
- Trees should be traversed top-down; each path leads to a single recommended tool
- Cross-reference the [Ecosystem Catalog](../docs/tooling/ecosystem-catalog.md) for full 14-field details

### Recommended Actions

1. Use these trees when starting any new project component
2. Print/bookmark the Quick Reference tables for rapid decisions
3. When a tool appears in the Avoid/Delay path, check the [Avoid / Delay List](../docs/tooling/avoid-delay-list.md)
4. Update trees when new tools are evaluated or compatibility changes

### Confidence Level

**HIGH** for tool recommendations (based on verified compatibility data). **MEDIUM** for specific score comparisons (may vary by use case and workload).

### Source Notes

Decision tree recommendations are derived from:
- NVIDIA DGX Spark official documentation
- Ecosystem Catalog compatibility assessments
- Community testing reports
- Architecture compatibility analysis

### Unresolved Questions

1. Should vLLM move to "Default" once dgx-spark-vllm stabilizes?
2. When Milvus Lite matures, does it change the vector DB decision tree?
3. Should JAX-based inference engines be added to Tree #1?
