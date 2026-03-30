# DGX Spark Expert System — Master Index

> **Last Updated**: 2026-03-30 (Phase 0 — Initial Setup)
>
> This repository is a persistent, continuously updateable knowledge base and execution framework for the NVIDIA DGX Spark platform. It covers hardware/software mastery, ecosystem tooling, project blueprints, operational playbooks, and maintenance workflows.

---

## 📋 Platform Knowledge (`docs/platform/`)

| Document | Description | Status |
|----------|-------------|--------|
| [Project Plan](docs/platform/project-plan.md) | Phased execution plan with tasks and milestones | ✅ Complete |
| [DGX Spark Master Reference](docs/platform/dgx-spark-master-reference.md) | Comprehensive hardware, software, and operational reference | 🔲 Skeleton |
| [Base Environment Plan](docs/platform/base-environment-plan.md) | OS, drivers, Python/conda/Poetry/uv strategy, containers, storage | 🔲 Planned |
| [Known Risks & Constraints](docs/platform/risks-and-constraints.md) | Severity-ranked risks with mitigations | 🔲 Planned |
| [Best Uses / Worst Uses](docs/platform/best-worst-uses.md) | What DGX Spark excels at and what to avoid | 🔲 Planned |
| [LLM Model Catalog](docs/platform/llm-model-catalog.md) | Which models to run, sizing, quantization, multi-model serving | 🔲 Planned |
| [Workarounds & Creative Patterns](docs/platform/workarounds-and-creative-patterns.md) | CUDA 13 gap fixes, shims, non-standard LLM usage, adapters | 🔲 Planned |
| [Monetization Strategy](docs/platform/monetization-strategy.md) | Cross-blueprint revenue ranking, build sequencing, build/buy | 🔲 Planned |
| [Executive Summary](docs/platform/executive-summary.md) | 1-page platform overview | 🔲 Planned |
| [Common Pitfalls](docs/platform/common-pitfalls.md) | Consolidated gotchas: symptom → cause → prevention → fix | 🔲 Planned |

## 🔧 Compatibility (`docs/compatibility/`)

| Document | Description | Status |
|----------|-------------|--------|
| [Compatibility Matrix](docs/compatibility/compatibility-matrix.md) | Component × Version × ARM64 × CUDA 13 × Container | 🔲 Skeleton |
| [Test Matrix & PoC Plans](docs/compatibility/test-matrix-and-poc-plans.md) | Concrete verification steps for every flagged uncertainty | 🔲 Planned |
| [PyTorch Compatibility](docs/compatibility/pytorch-compatibility.md) | Versions, wheels, CUDA 13, SDPA vs Flash Attention | 🔲 Planned |
| [vLLM Compatibility](docs/compatibility/vllm-compatibility.md) | Installation paths, sm_121 issues, dgx-spark-vllm | 🔲 Planned |
| [LangChain Compatibility](docs/compatibility/langchain-compatibility.md) | ARM64 pure-Python, local LLM backend config | 🔲 Planned |
| [RAPIDS Compatibility](docs/compatibility/rapids-compatibility.md) | cuDF, cuML, cuOpt on DGX Spark | 🔲 Planned |
| [Financial Libs Compatibility](docs/compatibility/financial-libs-compatibility.md) | zipline, backtrader, QuantLib, yfinance ARM64 status | 🔲 Planned |
| [Executive Summary](docs/compatibility/executive-summary.md) | 1-page compatibility landscape snapshot | 🔲 Planned |

## 🛠️ Tooling & Ecosystem (`docs/tooling/`)

| Document | Description | Status |
|----------|-------------|--------|
| [Ecosystem Catalog](docs/tooling/ecosystem-catalog.md) | 19-category tool catalog with 14 fields per entry | 🔲 Planned |
| [Recommended Core Stack](docs/tooling/recommended-core-stack.md) | Default install list with justifications and scores | 🔲 Planned |
| [Avoid / Delay List](docs/tooling/avoid-delay-list.md) | Tools to skip with reasoning and re-eval triggers | 🔲 Planned |
| [Executive Summary](docs/tooling/executive-summary.md) | 1-page ecosystem overview | 🔲 Planned |

## 🏗️ Project Blueprints (`docs/blueprints/`)

| Blueprint | Description | Status |
|-----------|-------------|--------|
| [Local LLM App](docs/blueprints/local-llm-app.md) | Chat/completion app with Ollama + FastAPI + web UI | 🔲 Planned |
| [AI Agent App](docs/blueprints/ai-agent-app.md) | Tool-using agent with NemoClaw/LangGraph | 🔲 Planned |
| [Android Companion App](docs/blueprints/android-companion-app.md) | Kotlin app → DGX Spark AI backend (Samsung Galaxy Ultra specifics) | 🔲 Planned |
| [Web Dashboard](docs/blueprints/web-dashboard.md) | Streamlit/Next.js data visualization | 🔲 Planned |
| [Data Ingestion Pipeline](docs/blueprints/data-ingestion-pipeline.md) | ETL with Polars + cuDF + scheduling | 🔲 Planned |
| [Research Dashboard](docs/blueprints/research-dashboard.md) | Investment research: market data + factor analysis + AI thesis | 🔲 Planned |
| [Alerting & Monitoring](docs/blueprints/alerting-monitoring-system.md) | AI-augmented monitoring with LLM anomaly narratives | 🔲 Planned |
| [Backtesting Environment](docs/blueprints/backtesting-environment.md) | Financial strategy backtesting + risk metrics | 🔲 Planned |
| [Computer Vision & Robotics](docs/blueprints/computer-vision-robotics.md) | CV prototyping + Isaac Sim + ROS2 + Jetson path | 🔲 Planned |
| [Hybrid Local/Cloud](docs/blueprints/hybrid-local-cloud.md) | DGX Spark local + cloud GPU burst architecture | 🔲 Planned |
| [Content & Commerce Automation](docs/blueprints/content-commerce-automation.md) | Trend monitoring, content drafting, e-commerce ops | 🔲 Planned |
| [Social/Content Analysis](docs/blueprints/social-content-analysis.md) | Sentiment analysis, topic clustering, trend detection | 🔲 Planned |
| [AI Business Tools](docs/blueprints/ai-business-tools.md) | Document processing, email triage, report generation | 🔲 Planned |
| [Investment Automation](docs/blueprints/investment-automation.md) | Unified: data pipeline + factors + thesis + risk + portfolio | 🔲 Planned |
| [Executive Summary](docs/blueprints/executive-summary.md) | 1-page overview of all blueprints + build order | 🔲 Planned |

## 📖 Playbooks (`docs/playbooks/`)

| Playbook | Description | Status |
|----------|-------------|--------|
| [Deploy New Model](docs/playbooks/deploy-new-model.md) | Find, download, deploy LLMs via Ollama/vLLM/NIM | 🔲 Planned |
| [Setup RAG Pipeline](docs/playbooks/setup-rag-pipeline.md) | End-to-end: embeddings → vector DB → retrieval → generation | 🔲 Planned |
| [Fine-Tune with LoRA](docs/playbooks/fine-tune-with-lora.md) | LoRA/QLoRA workflow using NeMo on DGX Spark | 🔲 Planned |
| [Connect Android App](docs/playbooks/connect-android-app.md) | DGX Spark as AI backend for Android (networking, auth, API) | 🔲 Planned |
| [Launch Agent System](docs/playbooks/launch-agent-system.md) | Multi-agent setup with NemoClaw or LangGraph | 🔲 Planned |
| [Setup Monitoring Stack](docs/playbooks/setup-monitoring-stack.md) | Prometheus + Grafana + alerts on DGX Spark | 🔲 Planned |
| [Build from Source (CUDA 13)](docs/playbooks/build-from-source-cuda13.md) | Building Python packages for CUDA 13 / aarch64 | 🔲 Planned |
| [Executive Summary](docs/playbooks/executive-summary.md) | 1-page overview of all playbooks | 🔲 Planned |

## 🔄 Maintenance & Updates (`docs/updates/`)

| Document | Description | Status |
|----------|-------------|--------|
| [Update Cadence System](docs/updates/update-cadence-system.md) | Bi-weekly/monthly/quarterly review process | 🔲 Skeleton |
| [Update Log Template](docs/updates/TEMPLATE-update-log.md) | Template for recording review cycle changes | 🔲 Planned |

## 📁 Templates (`templates/`)

| Template | Description | Status |
|----------|-------------|--------|
| `docker-compose-base.yml` | Base compose with GPU access, networking, volumes | 🔲 Planned |
| `Dockerfile.dgx-spark-python` | Python dev container (CUDA 13, aarch64, uv) | 🔲 Planned |
| `Dockerfile.dgx-spark-ml` | ML container (PyTorch + CUDA 13 NGC base) | 🔲 Planned |
| `fastapi-template/` | FastAPI project skeleton | 🔲 Planned |
| `agent-template/` | LangGraph agent skeleton + Ollama config | 🔲 Planned |
| `streamlit-template/` | Streamlit dashboard skeleton | 🔲 Planned |

## ✅ Checklists (`checklists/`)

| Checklist | Description | Status |
|-----------|-------------|--------|
| [Phase 1 Checklist](checklists/phase-1-checklist.md) | Research and writing tracker for Phase 1 | ✅ Complete |
| [Phase 2 Checklist](checklists/phase-2-checklist.md) | Tracker for Phase 2 | 🔲 Planned |
| [Phase 3 Checklist](checklists/phase-3-checklist.md) | Tracker for Phase 3 | 🔲 Planned |
| [Phase 4 Checklist](checklists/phase-4-checklist.md) | Tracker for Phase 4 | 🔲 Planned |
| [Phase 5 Checklist](checklists/phase-5-checklist.md) | Tracker for Phase 5 | 🔲 Planned |
| [Project Readiness Checklist](checklists/project-readiness-checklist.md) | Universal pre-project compatibility check | 🔲 Planned |
| [Version Drift Monitor](checklists/version-drift-monitor.md) | Component version tracking table | 🔲 Planned |
| Cheat Sheet: Ollama | Quick reference for Ollama commands | 🔲 Planned |
| Cheat Sheet: Docker + DGX | Docker + GPU commands | 🔲 Planned |
| Cheat Sheet: Inference | Multi-engine inference reference | 🔲 Planned |
| Cheat Sheet: Fine-Tuning | NeMo/LoRA/QLoRA quick reference | 🔲 Planned |
| Cheat Sheet: Python Env | uv/conda/Poetry setup reference | 🔲 Planned |

## 🌳 Decision Trees (`decision-trees/`)

| Decision Tree | Description | Status |
|---------------|-------------|--------|
| [Tool Selection](decision-trees/tool-selection.md) | 8 Mermaid flowcharts for common tool choices | 🔲 Planned |
