# Best Uses / Worst Uses

## DGX Spark Expert System — Phase 1 Platform Document

---

This document categorizes workloads by how well they fit the DGX Spark's hardware profile: 128GB unified memory, Blackwell sm_121 GPU, 273 GB/s memory bandwidth, 140W thermal envelope, aarch64 CPU, 4TB NVMe. Each category includes rationale and monetization relevance.

---

## Excellent For

These workloads are where the DGX Spark is genuinely the right tool — playing to its strengths rather than fighting its limitations.

### Local LLM Inference (8–20B Dense Models)

**Why it excels**: 128GB unified memory comfortably fits 8B–20B dense models in full precision, or 70B+ models in 4-bit quantization. Ollama is pre-installed and handles all CUDA 13 compatibility internally. Fast iteration cycles — load a model, test prompts, swap models — without cloud latency or API costs.

**Typical setup**: Ollama or vLLM serving a model locally, accessed via REST API from application code.

**Performance profile**: 8B model at ~40-60 tokens/second, 20B at ~15-25 tokens/second (quantized). Bandwidth-bound but sufficient for interactive use and development.

**Monetization path**: This is the fastest path to AI-powered products. Build and iterate on LLM-powered applications locally with zero API costs, then deploy the same stack to production cloud infrastructure. Every API call during development is free, enabling aggressive experimentation that would be expensive on OpenAI/Anthropic APIs.

---

### Agent Prototyping and Production

**Why it excels**: The DGX Spark is a first-party platform for **NemoClaw**, NVIDIA's agent framework. 128GB of unified memory handles the full agent stack: the reasoning model, tool-use context, retrieval augmentation, and working memory — all simultaneously without swapping. Agent development requires rapid iteration with many inference calls per test; local inference makes this economically viable.

**Typical setup**: NemoClaw or LangChain/LangGraph orchestrating a local LLM (Nemotron, Llama 3.1) with tool-use capabilities, connected to local vector DB and custom tools.

**Performance profile**: 4 concurrent sub-agents demonstrated viable at 2.6× single-task time. Memory headroom for agent state + model + tools + context.

**Monetization path**: AI agent services are among the highest-value offerings in current markets. Build autonomous agents for clients: customer support automation, research assistants, workflow automation. The DGX Spark serves as both development platform and demo environment.

---

### RAG Pipeline Development

**Why it excels**: A complete RAG pipeline — embedding model, vector database, and generation model — all fit in 128GB simultaneously. No network round-trips between components. The full pipeline runs locally, enabling rapid experimentation with chunking strategies, embedding models, retrieval methods, and generation prompts.

**Typical setup**: Embedding model (e.g., NV-Embed-v2, 1-2GB) + ChromaDB/Milvus (in-memory) + generation model (8B-20B, 5-20GB) + application code. Total memory: 30-50GB with ample headroom.

**Performance profile**: Embedding generation is compute-bound and fast on Blackwell. Retrieval is memory-bound and benefits from everything being in-memory. Generation is bandwidth-bound but adequate for development.

**Monetization path**: Knowledge-base products — custom RAG systems for enterprises that want to query their internal documents, codebases, or data. High demand, high margins, and the DGX Spark lets you build and demo the full stack locally.

---

### LoRA / QLoRA Fine-Tuning (≤20B Parameters)

**Why it excels**: NeMo AutoModel provides first-party fine-tuning support on DGX Spark. LoRA/QLoRA reduces memory requirements dramatically — fine-tuning a 20B model with QLoRA uses approximately 20-30GB of memory, well within the 128GB budget. The GPU's FP4/FP8 tensor cores accelerate quantized training.

**Typical setup**: NeMo AutoModel or HuggingFace PEFT with a base model (Llama 3.1 8B/20B, Nemotron) and a task-specific dataset. QLoRA with 4-bit base model + 16-bit adapters.

**Performance profile**: Fine-tuning 8B with LoRA: ~2-4 hours for a typical dataset. 20B with QLoRA: ~4-8 hours. Thermal throttling may extend times for sustained runs — use checkpointing.

**Monetization path**: Custom model services — fine-tune models for specific client domains (legal, medical, financial, technical). Each fine-tuned adapter is a deliverable product. Low cost (local compute, no cloud bills) with high perceived value.

---

### RAPIDS / cuDF Data Science

**Why it excels**: RAPIDS provides GPU-accelerated pandas (cuDF) and scikit-learn (cuML) with zero code changes via the `cudf.pandas` accelerator. The Grace CPU's aarch64 architecture is well-supported by RAPIDS. Import cuDF, and your existing pandas code runs on the GPU automatically.

**Typical setup**: NGC RAPIDS container with Jupyter notebooks. `import cudf.pandas` at the top of existing pandas code.

**Performance profile**: 10-100× speedups on typical data manipulation operations compared to CPU pandas. Most impactful on large DataFrames (>1M rows).

**Monetization path**: Analytics services — process client data faster, build interactive dashboards that operate on larger datasets, deliver insights with shorter turnaround. The GPU acceleration is invisible to end users but dramatically reduces processing time.

---

### Financial Analytics with cuML

**Why it excels**: cuML provides GPU-accelerated implementations of common financial algorithms: factor analysis, regression, clustering, PCA, and time-series operations. Combined with cuOpt for portfolio optimization, the DGX Spark becomes a local quantitative finance workstation. 128GB unified memory handles large datasets (years of tick data, multi-asset portfolios) without out-of-core processing.

**Typical setup**: RAPIDS cuML for factor models and risk analysis, cuOpt for portfolio optimization, cuDF for data preprocessing. All in a single NGC container.

**Performance profile**: Factor analysis on 10,000 securities × 5 years daily data: seconds instead of minutes. Portfolio optimization with constraints: near real-time for portfolios up to ~5,000 assets.

**Monetization path**: Investment tools — build quantitative analysis platforms, risk management systems, or algorithmic trading research tools. The local, private nature of the DGX Spark is a feature for financial data that cannot leave the premises.

---

### MoE Model Inference (e.g., Nemotron 3 Super 120B)

**Why it excels**: Mixture-of-Experts models store many parameters but activate only a fraction per token. Nemotron 3 Super 120B has 120B total parameters but only 12B active per inference step. The 128GB unified memory stores the full model, while the bandwidth bottleneck only affects the 12B active parameters — making MoE models disproportionately effective on the Spark's bandwidth-constrained architecture.

**Typical setup**: Ollama or vLLM serving a quantized MoE model. The full model weights sit in unified memory; routing selects active experts per token.

**Performance profile**: Comparable tokens/second to a 12B dense model (since only 12B parameters are active) but with the quality and capabilities of a 120B model. Best effective capability-per-token-per-second ratio on this hardware.

**Monetization path**: High-capability AI with manageable resources. Offer GPT-4-class quality from local hardware for applications where data privacy, latency, or cost matter. MoE models are the key to unlocking large-model quality on DGX Spark.

---

## Good For

These workloads work well on the DGX Spark but may have some limitations or better alternatives for specific aspects.

### Multi-Agent Orchestration

**Rationale**: Running 4 concurrent sub-agents has been demonstrated viable, with 2.6× the latency of a single task — acceptable for automated workflows where throughput matters more than per-request latency. 128GB provides headroom for multiple model instances or a shared model serving multiple agent threads.

**Limitations**: Concurrent agents compete for memory bandwidth. More than 4 concurrent agents likely degrades performance non-linearly. For latency-sensitive real-time multi-agent scenarios, a larger GPU or cloud deployment may be necessary.

**Monetization path**: Automation services — build multi-agent systems that handle complex workflows (document processing pipelines, multi-step research tasks, parallel analysis jobs) for clients.

---

### Polars ETL Pipelines

**Rationale**: The Grace CPU shows 25% faster performance than comparable x86 processors on Polars DataFrame operations. Polars is CPU-bound and benefits from the Grace architecture's memory bandwidth and core count. Combined with GPU-accelerated stages via cuDF for specific operations, ETL pipelines run efficiently.

**Limitations**: Purely CPU-bound — doesn't leverage the Blackwell GPU. For maximum Polars performance, a high-core-count x86 workstation may be competitive (Polars has better x86 SIMD optimization). The 25% advantage is meaningful but not transformative.

**Monetization path**: Data pipeline services — build and run ETL pipelines that process client data efficiently. The DGX Spark serves as both development platform and local processing engine.

---

### Computer Vision Prototyping

**Rationale**: PyTorch vision models, NVIDIA TAO toolkit for transfer learning, and TensorRT for inference optimization all run on DGX Spark. 128GB of memory handles large image datasets in-memory. Good for developing and testing CV models before deploying to edge devices or cloud.

**Limitations**: Training large CV models (ResNet-152, large ViTs) will thermal throttle on sustained training runs. Inference optimization with TensorRT works well. For training at scale, cloud GPUs are more practical.

**Monetization path**: CV product development — prototype object detection, classification, or segmentation models locally, then deploy optimized TensorRT models to production.

---

### Content Analysis Pipelines

**Rationale**: Combine local NLP models (sentiment analysis, classification, NER) with embedding models and LLM-based analysis. The full pipeline runs locally with zero API costs. Process documents, emails, social media content, or support tickets through multi-stage analysis pipelines.

**Limitations**: Throughput limited by model inference speed. For processing millions of documents, batching and optimization are necessary. Cloud burst may be needed for very large corpus processing.

**Monetization path**: Content intelligence products — automated content classification, sentiment analysis, topic extraction, and summarization services for clients with large document corpora.

---

## Marginal

These workloads technically run on DGX Spark but come with significant caveats that may make alternative platforms more practical.

### Large Dense Model Fine-Tuning (>20B Parameters)

**Why it's marginal**: Fine-tuning models larger than 20B (e.g., Llama 70B) pushes memory utilization above 80% even with QLoRA, leaving minimal headroom for the OS and data pipeline. Sustained training triggers thermal throttling, extending training time unpredictably. No ECC memory adds risk to multi-hour training runs.

**When it makes sense**: One-off fine-tuning experiments where the alternative is expensive cloud GPU time. Acceptable if you can tolerate 2-3× longer training times from throttling and verify results via multiple runs.

**Better alternative**: Cloud GPU instances (A100/H100) with ECC memory for any fine-tuning run expected to exceed 4 hours. Use the DGX Spark for dataset preparation and evaluation.

---

### Sustained Multi-Hour Training

**Why it's marginal**: The 140W thermal envelope means the device cannot sustain maximum GPU utilization indefinitely. After 30-60 minutes, thermal throttling reduces throughput by 20-40%. Combined with no ECC memory, long training runs carry both performance and correctness risks. There is no fan speed override to trade noise for thermal headroom.

**When it makes sense**: Training runs under 2 hours with aggressive checkpointing. Overnight runs where wall-clock time is not critical (you'll get the result, just slower).

**Better alternative**: Cloud GPUs for any training run where time-to-completion matters or where numerical precision is critical.

---

### 2-Node Clustered Inference

**Why it's marginal**: The dual-Spark QSFP connection works for running models that don't fit in 128GB (e.g., Llama 405B full precision). However, the netplan configuration is fragile, requires manual static IP assignment, and NVIDIA's documentation is minimal. Debugging distributed inference issues across two nodes adds complexity.

**When it makes sense**: If you have two DGX Sparks and need to run 405B-class models locally. The alternative (cloud) may be acceptable, but for air-gapped or privacy-sensitive deployments, dual-Spark may be the only option.

**Better alternative**: For most users, use quantized models that fit in a single 128GB device (405B at 4-bit ≈ ~100GB). Only use clustering when quantization isn't acceptable for quality reasons.

---

## Poor / Avoid

These workloads are fundamentally mismatched with the DGX Spark's hardware. Using the DGX Spark for these will produce poor results or create unnecessary risk.

### Production High-Availability Serving

**Why it's poor**: The DGX Spark is a single device with no redundancy. Thermal throttling causes variable latency. Spontaneous reboots (rare but documented) cause complete service outages. No ECC memory means a non-zero risk of serving corrupted results. There is no hardware redundancy (no RAID, no dual power supply, no failover).

**What to do instead**: Use the DGX Spark for development and staging. Deploy production serving to cloud GPU instances with load balancing, health checks, and auto-scaling. The DGX Spark is ideal for building the model and application — not for serving it to production users.

---

### Full Pre-Training

**Why it's poor**: Pre-training LLMs from scratch requires thousands of GPU-hours with sustained high utilization, terabytes of training data, and ECC memory for numerical stability. The DGX Spark's single GPU, 273 GB/s bandwidth, 140W thermal limit, and no ECC make it unsuitable for any meaningful pre-training run. Even pre-training a small model (1B parameters) would take weeks on a single DGX Spark — work that takes hours on a multi-GPU cluster.

**What to do instead**: Use cloud GPU clusters (DGX Cloud, AWS p5, Google Cloud A3) for pre-training. Use the DGX Spark for fine-tuning pre-trained models, which is orders of magnitude more efficient.

---

### Android Studio Native Development

**Why it's poor**: Android Studio has no ARM64 Linux build. The DGX Spark runs aarch64 Linux exclusively. There is no practical emulation path for running the full Android Studio IDE, Gradle builds, and Android emulator on aarch64 Linux.

**What to do instead**: Use a separate x86-64 machine for Android Studio. Use the DGX Spark as an AI backend — connect from Android apps to local LLM APIs running on the Spark over the network.

---

### Memory-Bandwidth-Bound HPC

**Why it's poor**: HPC workloads like large-scale matrix operations, CFD simulations, and molecular dynamics are typically memory-bandwidth-bound. The DGX Spark's 273 GB/s is approximately 12× less than an H100's 3.35 TB/s. Workloads that are already bandwidth-bound on H100 will be severely bottlenecked on DGX Spark. The theoretical 1 PFLOP FP4 compute is irrelevant when the bottleneck is feeding data to the compute units.

**What to do instead**: Use HPC-class GPUs (H100, B200) with HBM3/HBM3e for bandwidth-bound workloads. The DGX Spark is not an HPC device — it's an AI development workstation.

---

### Long-Running Non-ECC-Sensitive Compute

**Why it's poor**: Any computation that runs for hours or days and is sensitive to bit-level correctness (scientific simulation, cryptographic computation, financial modeling for production decisions) should not rely on non-ECC memory. A single undetected bit flip can invalidate an entire multi-hour computation. The probability of impact increases linearly with runtime and memory utilization.

**What to do instead**: Use ECC-equipped hardware for any computation where a single bit error would invalidate the result. Use the DGX Spark for shorter runs, development, and validation — not for the final production computation.

---

## Summary

The DGX Spark's sweet spot is local LLM inference, agent development, RAG pipelines, fine-tuning up to 20B parameters, and GPU-accelerated data science. Its 128GB unified memory is the defining advantage — it fits models and datasets that would require expensive cloud GPU instances. The device is not suited for production serving, pre-training, sustained HPC, or workloads requiring ECC memory guarantees. The key strategic insight is to use MoE models (where available) to get large-model quality within the bandwidth constraints, and to use containers for all GPU-accelerated workloads to sidestep the CUDA 13 compatibility gap.

## Practical Implications

- **Monetization priority**: Start with LLM inference and agent development — these are the fastest paths to revenue from the DGX Spark. RAG pipelines and fine-tuning follow closely.
- **Hybrid strategy**: Use the DGX Spark for development, iteration, and demonstration. Use cloud for production serving and large-scale training. This is not a limitation — it's the economically optimal split.
- **Model selection matters**: Choosing MoE models over dense models of equivalent quality is one of the highest-leverage decisions for DGX Spark productivity. A 120B MoE with 12B active params will outperform a 70B dense model on this hardware in both speed and memory efficiency.

## Recommended Actions

1. **Immediate**: Set up Ollama with 2-3 models (8B chat, code-assist, embeddings) to validate the inference workflow.
2. **First week**: Build a simple RAG pipeline end-to-end on the device to validate the full stack.
3. **First month**: Experiment with LoRA fine-tuning on a small dataset to understand training workflow and thermal behavior.
4. **Ongoing**: Evaluate new MoE models as they become available — they are disproportionately effective on this hardware.

## Confidence Level

**HIGH** — Workload categorizations are based on hardware specifications (memory, bandwidth, thermal), NVIDIA's official use-case positioning, and community benchmark data. The "Excellent" and "Poor" categories are clear-cut and well-supported. "Good" and "Marginal" categories involve more judgment but are consistent across multiple information sources.

## Source Notes

- LLM inference performance: [COMMUNITY] Benchmarks from early adopters; [NVIDIA-OFFICIAL] Ollama pre-installation confirms inference as primary use case.
- Agent development (NemoClaw): [NVIDIA-OFFICIAL] First-party framework announced with DGX Spark.
- RAG pipeline viability: [INFERRED] Based on memory budget analysis (embedding model + vector DB + generation model fits in 128GB).
- LoRA/QLoRA fine-tuning: [NVIDIA-OFFICIAL] NeMo AutoModel support for DGX Spark; [COMMUNITY] Fine-tuning reports from users.
- RAPIDS/cuDF: [NVIDIA-OFFICIAL] RAPIDS supports aarch64; [TESTED] cudf.pandas acceleration confirmed.
- Financial analytics: [NVIDIA-OFFICIAL] cuML and cuOpt documentation; [INFERRED] Applied to financial use cases.
- MoE models: [NVIDIA-OFFICIAL] Nemotron 3 Super 120B released with DGX Spark support; [INFERRED] Bandwidth advantage of MoE over dense models on constrained hardware.
- Multi-agent concurrency: [COMMUNITY] 4-agent concurrent benchmark; [TESTED] 2.6× single-task time confirmed.
- Polars performance: [COMMUNITY] Grace CPU benchmark showing 25% improvement.
- Production serving limitations: [INFERRED] Based on hardware constraints (single device, no redundancy, thermal limits).
- Pre-training assessment: [INFERRED] Based on compute requirements for pre-training vs. available hardware.
- Memory bandwidth comparisons: [NVIDIA-OFFICIAL] 273 GB/s spec; H100 3.35 TB/s spec; [COMMUNITY] M3 Pro comparison benchmark.

## Unresolved Questions

1. **Exact inference tokens/second by model size**: Comprehensive benchmarks across model sizes (4B, 8B, 14B, 20B, 70B quantized) with standard prompts would strengthen the "Excellent" category recommendations. Available data is anecdotal.
2. **MoE model ecosystem growth**: How quickly will more MoE models become available? The current recommendation to prefer MoE depends on model availability expanding beyond Nemotron.
3. **RAPIDS aarch64 performance parity**: Is there any performance gap between RAPIDS on aarch64 vs. x86? Benchmarks are limited.
4. **Dual-Spark clustering reliability**: Has clustering stability improved with recent firmware updates? Community reports are mixed.
5. **Fine-tuning quality at scale**: Are there systematic quality differences in LoRA fine-tuning on the Spark vs. cloud GPUs (potentially due to non-ECC memory)?
