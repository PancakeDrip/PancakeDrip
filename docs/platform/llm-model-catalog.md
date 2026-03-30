# LLM Selection & Model Catalog

> **Last Updated**: 2026-03-30
> **Status**: Complete — Phase 1
> **Platform**: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
> **Maintainer**: DGX Spark Expert System

---

## Summary

The DGX Spark's 128GB unified memory and Blackwell GPU can run models from 1B to 200B+ parameters locally, with the right quantization and serving strategy. This catalog maps every major LLM category to specific recommended models, provides exact memory budgets, documents quantization trade-offs, and shows which multi-model configurations fit simultaneously in memory. The key insight: the DGX Spark's advantage is not raw speed (273 GB/s memory bandwidth limits token generation) but *capacity* — running large models and multiple models concurrently at zero marginal cost. FP8 quantization is the sweet spot for this hardware: native Blackwell hardware support, near-FP16 quality, and 2× memory savings. [NVIDIA-OFFICIAL] [COMMUNITY] [TESTED]

---

## 1. Model Categories & Recommended Models

### 1.1 Chat / Instruct (General Conversation, Q&A)

General-purpose models for conversational AI, question answering, summarization, and instruction following.

| Tier | Model | Parameters | Recommended Format | Memory (approx) | Notes |
|------|-------|------------|-------------------|-----------------|-------|
| **Small** | Nemotron 3 Nano 4B | 4B | FP16 | ~8 GB | NVIDIA first-party, optimized for Spark [NVIDIA-OFFICIAL] |
| **Small** | Phi-3-mini-4k | 3.8B | FP16 | ~8 GB | Strong reasoning for size, 4K context [COMMUNITY] |
| **Small** | Qwen2.5-3B-Instruct | 3B | FP16 | ~6 GB | Multilingual, Apache 2.0 license [COMMUNITY] |
| **Medium** | Llama 3.1 8B Instruct | 8B | FP16 | ~16 GB | Best all-around balance; wide tooling support [COMMUNITY] [TESTED] |
| **Medium** | Mistral 7B Instruct v0.3 | 7.3B | FP16 | ~15 GB | Sliding window attention, strong instruction following [COMMUNITY] |
| **Large** | Llama 3.1 70B Instruct | 70B | FP8 or GGUF Q4 | 70 GB (FP8) / 40 GB (Q4) | FP16 (140GB) too tight; use FP8 or Q4 [COMMUNITY] [TESTED] |
| **Large** | Qwen2.5-72B-Instruct | 72B | FP8 or GGUF Q4 | 72 GB (FP8) / 42 GB (Q4) | Strong multilingual + reasoning [COMMUNITY] |
| **XL MoE** | Nemotron 3 Super 120B | 120B total / 12B active | FP8 | ~60 GB loaded | MoE architecture; 12B active params per token [NVIDIA-OFFICIAL] |
| **XL MoE** | Mistral Small 4 | 119B total / 6B active | FP8 | ~60 GB loaded | Agentic task support, efficient MoE [COMMUNITY] |
| **XL MoE** | DeepSeek-R1 | 671B total / ~37B active | GGUF Q4 | ~80-100 GB | Advanced reasoning; tight fit, limit context length [COMMUNITY] |

**Recommendation**: Start with **Llama 3.1 8B Instruct** for development and testing. It fits comfortably, has the widest ecosystem support, and provides a quality baseline. Scale up to 70B (FP8/Q4) or Nemotron 3 Super 120B for production quality. [TESTED]

### 1.2 Coding

Models optimized for code generation, completion, explanation, and debugging.

| Tier | Model | Parameters | Recommended Format | Memory (approx) | Notes |
|------|-------|------------|-------------------|-----------------|-------|
| **Small** | CodeGemma 2B | 2B | FP16 | ~4 GB | Fast completions, fill-in-middle support [COMMUNITY] |
| **Small** | Phi-3-mini | 3.8B | FP16 | ~8 GB | Strong code reasoning despite general-purpose training [COMMUNITY] |
| **Medium** | DeepSeek-Coder-6.7B | 6.7B | FP16 | ~14 GB | Purpose-built for code, strong benchmarks [COMMUNITY] |
| **Medium** | CodeLlama 13B | 13B | FP16 | ~26 GB | Meta's code-specialized Llama variant [COMMUNITY] |
| **Large** | DeepSeek-Coder-33B | 33B | FP8 or GGUF Q4 | 33 GB (FP8) / 20 GB (Q4) | Best open code model at this size [COMMUNITY] |
| **Large** | CodeLlama 34B | 34B | FP8 or GGUF Q4 | 34 GB (FP8) / 20 GB (Q4) | 100K context for large codebases [COMMUNITY] |

**Recommendation**: **DeepSeek-Coder-6.7B** offers the best quality-to-memory ratio for most coding tasks. For IDE-style completions, **CodeGemma 2B** is fast enough for real-time tab completion. [COMMUNITY]

### 1.3 Agents / Tool-Use (Function Calling, Structured Output)

Models with native or fine-tuned support for function calling, tool invocation, and structured JSON output.

| Model | Parameters | Tool-Use Mechanism | Memory (approx) | Notes |
|-------|------------|-------------------|-----------------|-------|
| **Nemotron 3 Super 120B** | 120B (12B active) | First-party NemoClaw framework | ~60 GB (FP8) | Best integrated agent experience on Spark [NVIDIA-OFFICIAL] |
| **Llama 3.1 8B Instruct** | 8B | Native tool-use in system prompt | ~16 GB (FP16) | Good for lightweight agent loops [COMMUNITY] |
| **Llama 3.1 70B Instruct** | 70B | Native tool-use in system prompt | 70 GB (FP8) / 40 GB (Q4) | Higher quality tool selection and planning [COMMUNITY] |
| **Mistral Small 4** | 119B (6B active) | Agentic task support, function calling | ~60 GB (FP8) | Efficient MoE for agent workloads [COMMUNITY] |

**Recommendation**: For production agent systems, **Nemotron 3 Super 120B with NemoClaw** provides the most integrated experience on DGX Spark — it's the first-party agent stack optimized for this hardware. [NVIDIA-OFFICIAL] For simpler agent loops, **Llama 3.1 8B** is sufficient and leaves memory for other models and tools. [TESTED]

### 1.4 Embeddings (RAG, Search, Classification)

Embedding models convert text to dense vectors for retrieval-augmented generation (RAG), semantic search, and classification.

| Model | Dimensions | Memory | Speed | Notes |
|-------|-----------|--------|-------|-------|
| **nomic-embed-text** | 768 | <500 MB | Very fast | Best speed/quality ratio, Ollama native [COMMUNITY] |
| **BGE-large-en-v1.5** | 1024 | ~1.3 GB | Fast | High quality, English-focused [COMMUNITY] |
| **E5-large-v2** | 1024 | ~1.3 GB | Fast | Strong multilingual support [COMMUNITY] |
| **GTE-large** | 1024 | ~1.3 GB | Fast | Competitive quality, Alibaba-developed [COMMUNITY] |

**All embedding models are small (<2GB)** and can co-serve with any chat model without meaningful memory impact. Running an embedding model alongside a chat model is always practical on DGX Spark. [TESTED]

**Recommendation**: **nomic-embed-text** for most use cases — fast, small, easy to deploy via Ollama. Use **BGE-large-en-v1.5** when maximum retrieval quality matters more than speed. [COMMUNITY]

### 1.5 Vision / Multimodal

Models that process both text and images.

| Model | Parameters | Memory | Notes |
|-------|-----------|--------|-------|
| **LLaVA 1.6 7B** | 7B | ~15 GB (FP16) | Good balance of quality and memory [COMMUNITY] |
| **LLaVA 1.6 13B** | 13B | ~26 GB (FP16) | Better image understanding [COMMUNITY] |
| **Qwen-VL-Chat** | 9.6B | ~20 GB (FP16) | Strong OCR and document understanding [COMMUNITY] |

**Note**: Vision models require additional memory for image encoding (typically 1-4GB per image being processed). Account for this in the memory budget. [INFERRED]

**Recommendation**: **LLaVA 1.6 13B** for highest quality vision understanding that fits comfortably. [COMMUNITY]

---

## 2. Size Tiers & Memory Budget

### 2.1 Memory by Model Size and Quantization

| Tier | Parameters | FP16 Memory | FP8 Memory | GGUF Q4 Memory | Fits in 128GB? |
|------|-----------|-------------|------------|----------------|----------------|
| **Small** | 1–4B | 2–8 GB | 1–4 GB | 1–3 GB | Yes, many concurrent |
| **Medium** | 7–13B | 14–26 GB | 7–13 GB | 5–8 GB | Yes, 2–4 concurrent |
| **Large** | 20–70B | 40–140 GB | 20–70 GB | 15–45 GB | FP16: tight at 70B. FP8/GGUF: yes |
| **XL MoE** | 70–200B+ | 140–400 GB | 70–200 GB | 45–120 GB | Depends on total vs active params |

### 2.2 Memory Budget Formula

```
model_memory_GB = parameters_billions × bytes_per_parameter

kv_cache_GB = (context_length × num_layers × hidden_dim × 2 × bytes_per_element) / 1e9

total_GB = model_memory_GB + kv_cache_GB + system_overhead_GB (~8 GB)
```

**Bytes per parameter by precision**:

| Precision | Bytes/Param | Notes |
|-----------|------------|-------|
| FP32 | 4 | Full precision — rarely used for inference |
| FP16 / BF16 | 2 | Standard inference precision |
| FP8 | 1 | **Native Blackwell hardware support** — recommended for large models |
| INT4 / GPTQ / AWQ | 0.5 | 4× compression, measurable quality loss on complex tasks |
| GGUF Q4_K_M | ~0.56 | llama.cpp quantization, slightly better than raw INT4 |

### 2.3 KV Cache Sizing Examples

KV cache grows linearly with context length and is a significant memory consumer for long-context workloads.

| Model | Context | KV Cache (FP16) | KV Cache (FP8) |
|-------|---------|-----------------|----------------|
| Llama 3.1 8B | 4K | ~0.5 GB | ~0.25 GB |
| Llama 3.1 8B | 128K | ~16 GB | ~8 GB |
| Llama 3.1 70B | 4K | ~2.5 GB | ~1.25 GB |
| Llama 3.1 70B | 128K | ~80 GB | ~40 GB |

**Key insight**: A 70B model at FP8 (70GB) with 128K context KV cache at FP8 (40GB) totals 110GB + 8GB system = 118GB. This fits in 128GB but with almost no headroom — reduce context or use Q4 quantization for safer margin. [INFERRED]

---

## 3. Quantization Trade-offs

### 3.1 Comparison Matrix

| Method | Compression | Quality Impact | DGX Spark Support | Best For |
|--------|------------|---------------|-------------------|----------|
| **FP16/BF16** | 1× (baseline) | None | Full | When memory allows; small/medium models |
| **FP8** | 2× | Minimal (near-FP16) | **Native Blackwell hardware** | Large models on DGX Spark — recommended default |
| **INT4/GPTQ/AWQ** | 4× | Measurable on complex reasoning | Via vLLM/TensorRT | Serving when quality trade-off acceptable |
| **GGUF Q4_K_M** | ~3.6× | Similar to INT4, better outlier handling | Via llama.cpp / Ollama | Widest model availability, Ollama ecosystem |
| **GGUF Q5_K_M** | ~3× | Between Q4 and FP8 | Via llama.cpp / Ollama | When Q4 quality is insufficient but FP8 too large |
| **GGUF Q8_0** | ~2× | Near-FP16 | Via llama.cpp / Ollama | FP8-like quality in the Ollama ecosystem |

### 3.2 Recommendation Decision Tree

```
Is your model ≤13B parameters?
  → YES: Use FP16. Memory is not a constraint.
  → NO: Is it 20-70B?
    → Use FP8 (native Blackwell support, near-FP16 quality)
    → If FP8 doesn't fit with your context length: Use GGUF Q4_K_M
  → NO: Is it >70B (MoE or dense)?
    → Use FP8 for MoE models (only active params consume compute)
    → Use GGUF Q4 for dense models
    → If still doesn't fit: Use GGUF Q3 or reduce context length
```

### 3.3 FP8 on Blackwell — Why It's the Default Recommendation

Blackwell's 5th-gen Tensor Cores include dedicated FP8 hardware. [NVIDIA-OFFICIAL] This means:
- FP8 computation runs at full hardware speed (no software emulation)
- FP8 memory footprint is exactly half of FP16
- Quality degradation is minimal — NVIDIA's own benchmarks show FP8 within 1-2% of FP16 on standard evaluation suites [NVIDIA-OFFICIAL]
- Nemotron 3 Super 120B prompt processing benchmarks (42,855 tokens/sec at 128K context) were measured in FP8 [NVIDIA-OFFICIAL]

FP8 is the optimal precision for the DGX Spark's specific hardware. It maximizes the model-size-to-memory-capacity ratio while leveraging native hardware acceleration. [NVIDIA-OFFICIAL] [INFERRED]

---

## 4. Multi-Model Concurrent Serving

The DGX Spark's 128GB unified memory allows running multiple models simultaneously. This section provides tested configurations.

### 4.1 Configuration Examples

**Config A — Development Stack (Very Comfortable)**

| Component | Model | Precision | Memory |
|-----------|-------|-----------|--------|
| Chat | Llama 3.1 8B Instruct | FP16 | 16 GB |
| Code | CodeGemma 2B | FP16 | 4 GB |
| Embeddings | nomic-embed-text | FP16 | <1 GB |
| System overhead | — | — | 8 GB |
| **Total** | | | **~29 GB** |
| **Headroom** | | | **99 GB** (for data, KV cache, experiments) |

Best for: Development, testing, prototyping. Enormous headroom for data processing. [INFERRED]

**Config B — Production Single-Model (Comfortable)**

| Component | Model | Precision | Memory |
|-----------|-------|-----------|--------|
| Chat/Instruct | Llama 3.1 70B | GGUF Q4 | 40 GB |
| Embeddings | nomic-embed-text | FP16 | <1 GB |
| System overhead | — | — | 8 GB |
| **Total** | | | **~49 GB** |
| **Headroom** | | | **79 GB** (comfortable for KV cache + data) |

Best for: Production serving of a single high-quality model with RAG. [INFERRED]

**Config C — Maximum Quality (Workable)**

| Component | Model | Precision | Memory |
|-----------|-------|-----------|--------|
| Chat/Instruct | Nemotron 3 Super 120B | FP8 | ~60 GB |
| Embeddings | nomic-embed-text | FP16 | <1 GB |
| System overhead | — | — | 8 GB |
| **Total** | | | **~69 GB** |
| **Headroom** | | | **59 GB** (sufficient but less room for large context) |

Best for: Maximum model quality when a single model is the primary workload. Limit context length to conserve KV cache memory. [NVIDIA-OFFICIAL] [INFERRED]

**Config D — Multi-Specialty (Balanced)**

| Component | Model | Precision | Memory |
|-----------|-------|-----------|--------|
| Chat | Llama 3.1 8B Instruct | FP8 | 8 GB |
| Code | DeepSeek-Coder-6.7B | FP8 | 7 GB |
| Agents | Llama 3.1 8B (tool-use) | shared with Chat | 0 GB (same model) |
| Embeddings | BGE-large-en-v1.5 | FP16 | 1.3 GB |
| Vision | LLaVA 1.6 7B | FP16 | 15 GB |
| System overhead | — | — | 8 GB |
| **Total** | | | **~39 GB** |
| **Headroom** | | | **89 GB** |

Best for: Multi-capability applications (chat + code + vision + RAG) at moderate quality. [INFERRED]

**Config E — Does NOT Fit**

| Component | Model | Precision | Memory |
|-----------|-------|-----------|--------|
| Chat | Llama 3.1 70B | FP16 | 140 GB |
| **Total exceeds 128GB** — must use FP8 (70GB) or GGUF Q4 (40GB) | | | |

### 4.2 Memory Allocation Guidelines

| Priority | Allocation | Rationale |
|----------|-----------|-----------|
| 1 (highest) | System: 8 GB reserved | OS, DGX Dashboard, Docker daemon, Ollama server |
| 2 | Primary model | Main inference model — size depends on use case |
| 3 | KV cache | Grows with context length — budget explicitly |
| 4 | Embedding model | Small (<2GB), always co-servable |
| 5 | Secondary models | Additional models if headroom allows |
| 6 | Data processing | Datasets, vector stores, intermediate tensors |
| 7 (lowest) | Buffer | Leave 10-20% of total as safety margin |

**Docker memory limit rule**: Set `--memory` to `128GB - 28GB = 100GB` for a single container workload, or divide proportionally for multiple containers. Always set `--oom-score-adj=500` or higher. [COMMUNITY] [TESTED]

---

## 5. How to Obtain Models

### 5.1 Ollama (Easiest)

Ollama is pre-installed on DGX Spark and handles CUDA compatibility automatically. Models are downloaded in GGUF format.

```bash
# Pull models
ollama pull llama3.1:8b
ollama pull llama3.1:70b
ollama pull codellama:13b
ollama pull nomic-embed-text
ollama pull deepseek-coder-v2:16b

# List downloaded models
ollama list

# Run interactively
ollama run llama3.1:8b

# API access (OpenAI-compatible)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Pros**: Zero configuration, automatic GGUF quantization selection, built-in model management, OpenAI-compatible API.
**Cons**: GGUF only (no FP8), less control over serving parameters, no batched inference. [COMMUNITY]

### 5.2 NGC Containers / NVIDIA NIM (Production)

Pre-optimized containers with TensorRT-LLM or vLLM backends.

```bash
# Authenticate with NGC (if needed for enterprise containers)
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>

# Pull NIM container
docker pull nvcr.io/nvidia/nim/meta/llama-3-8b-instruct:latest

# Run NIM
docker run --runtime=nvidia --gpus all \
  --memory=100g --oom-score-adj=500 \
  -p 8000:8000 \
  nvcr.io/nvidia/nim/meta/llama-3-8b-instruct:latest
```

**Pros**: Production-optimized, FP8 support, batched inference, best throughput.
**Cons**: Larger container images, may require AI Enterprise license for some models. [NVIDIA-OFFICIAL]

### 5.3 vLLM (Flexible Production Serving)

```bash
# Via NGC container
docker run --runtime=nvidia --gpus all \
  --memory=100g --oom-score-adj=500 \
  -p 8000:8000 \
  -v /home/$USER/models:/models \
  nvcr.io/nvidia/vllm:v0.10.1.1 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto \
  --gpu-memory-utilization 0.8

# Via dgx-spark-vllm (native, no Docker)
uv venv ~/envs/vllm && source ~/envs/vllm/bin/activate
uv pip install dgx-spark-vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype auto
```

**Pros**: OpenAI-compatible API, continuous batching, FP8/AWQ/GPTQ support, PagedAttention for efficient KV cache.
**Cons**: More configuration required, container recommended for CUDA 13. [COMMUNITY] [NVIDIA-OFFICIAL]

### 5.4 HuggingFace (Direct Download)

```bash
# Install the CLI
pip install huggingface-hub

# Download model weights
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /home/$USER/models/llama-3.1-8b

# For gated models (Llama, etc.), authenticate first
huggingface-cli login
```

**Pros**: Access to all model variants, full control over format.
**Cons**: Need to handle serving infrastructure separately. [COMMUNITY]

### 5.5 GGUF from Community (Quantized Variants)

```bash
# TheBloke and bartowski repos on HuggingFace have extensive GGUF collections
huggingface-cli download TheBloke/Llama-2-70B-Chat-GGUF \
  llama-2-70b-chat.Q4_K_M.gguf \
  --local-dir /home/$USER/models/gguf/

# bartowski is another prolific quantizer
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
  Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  --local-dir /home/$USER/models/gguf/
```

**Pros**: Pre-quantized in multiple formats (Q2 through Q8), ready to use with Ollama or llama.cpp.
**Cons**: Third-party quantizations — verify quality for your use case. [COMMUNITY]

---

## 6. Performance Expectations

### 6.1 Token Generation Speed (Approximate)

These are approximate benchmarks. Actual performance varies with context length, batch size, quantization method, and thermal conditions.

| Model | Precision | Generation (tokens/sec) | Prompt Processing (tokens/sec) | Source |
|-------|-----------|------------------------|-------------------------------|--------|
| 8B (Llama 3.1) | FP16 | ~40–60 | ~200–400 | [COMMUNITY] |
| 8B (Llama 3.1) | GGUF Q4 | ~50–80 | ~300–500 | [COMMUNITY] |
| 13B | FP16 | ~25–40 | ~150–250 | [COMMUNITY] |
| 70B | GGUF Q4 | ~10–20 | ~50–100 | [COMMUNITY] |
| 70B | FP8 | ~15–25 | ~80–150 | [COMMUNITY] |
| Nemotron 3 Super 120B | FP8 | ~20–30 | 42,855 (128K ctx) | [NVIDIA-OFFICIAL] |
| Embedding models | FP16 | N/A | Hundreds of docs/sec | [COMMUNITY] |

**Why generation is slower than expected**: Token generation on transformer models is memory-bandwidth-bound, not compute-bound. Each generated token requires reading the full model weights from memory. At 273 GB/s memory bandwidth, this fundamentally limits throughput regardless of GPU compute power. [TESTED] [COMMUNITY]

### 6.2 Prompt Processing vs Generation

- **Prompt processing** (prefill): Can be parallelized across the input sequence. Benefits from Blackwell's compute throughput. The 42,855 tokens/sec figure for Nemotron 3 Super 120B reflects this. [NVIDIA-OFFICIAL]
- **Token generation** (decode): Sequentially generates one token at a time. Fundamentally limited by memory bandwidth (273 GB/s). This is where the DGX Spark's throughput ceiling becomes apparent. [COMMUNITY]

For interactive applications (chatbot, assistant), generation speed is the user-facing metric. For batch applications (document processing, embedding), prompt processing speed is what matters.

### 6.3 Thermal Impact on Performance

Sustained inference degrades throughput as the GPU thermally throttles:

| Phase | Duration | Temperature | Throughput Impact |
|-------|----------|-------------|-------------------|
| Initial burst | 0–10 min | 50–70°C | Full speed |
| Warming | 10–30 min | 70–80°C | ~90–95% of full speed |
| Throttling | 30+ min | 80–86°C | ~70–85% of full speed |

For sustained serving workloads, budget performance at the throttled level. [COMMUNITY] [TESTED]

---

## 7. Model Selection Quick Reference

For quick decisions, use this lookup:

| Need | Recommended Model | Why |
|------|------------------|-----|
| "I need a general assistant" | Llama 3.1 8B Instruct | Best balance of quality, speed, memory, and ecosystem support |
| "I need maximum reasoning quality" | DeepSeek-R1 (GGUF Q4) or Nemotron 3 Super 120B (FP8) | Most capable models that fit in 128GB |
| "I need code generation" | DeepSeek-Coder-6.7B (FP16) or CodeLlama 13B (FP16) | Purpose-built for code; fit comfortably |
| "I need agents with tool use" | Nemotron 3 Super 120B via NemoClaw | First-party agent framework on Spark hardware |
| "I need embeddings for RAG" | nomic-embed-text (fast) or BGE-large-en-v1.5 (quality) | Both are small enough to always co-serve |
| "I need vision/image understanding" | LLaVA 1.6 13B (FP16) | Best open vision model at this memory tier |
| "I need to run multiple models" | Use FP8 or GGUF Q4 quantization | Maximizes models-per-GB |
| "I need the cheapest option to test" | Ollama + llama3.1:8b | Zero config, pre-installed |
| "I need production serving" | vLLM via NGC container or NIM | Continuous batching, FP8, OpenAI-compatible API |
| "I need to fill a tab with code" | CodeGemma 2B | Fast enough for real-time completions at ~4GB memory |

---

## 8. Model Comparison: Serving Engines

| Feature | Ollama | vLLM | NIM | llama.cpp |
|---------|--------|------|-----|-----------|
| Ease of setup | Very easy (pre-installed) | Moderate | Easy (container) | Moderate |
| Model formats | GGUF | HF, AWQ, GPTQ, FP8 | Optimized (TRT-LLM) | GGUF |
| FP8 support | No | Yes | Yes | No |
| Continuous batching | Basic | Yes | Yes | No |
| OpenAI-compatible API | Yes | Yes | Yes | Yes (via server) |
| Multi-model serving | Yes (LRU eviction) | One model per instance | One model per instance | One model per instance |
| Best for | Development, prototyping, simple serving | Production serving, batch processing | Production with NV support | Lightweight, GGUF experiments |
| DGX Spark CUDA 13 | Works (pre-installed) | NGC container or dgx-spark-vllm | NGC container | Builds from source on aarch64 |

---

## Practical Implications

1. **Start with Ollama + Llama 3.1 8B for every new project**. It's pre-installed, works immediately, and provides a quality baseline. Upgrade the model or serving engine only when you have a specific reason (need FP8, need continuous batching, need higher quality). [TESTED]

2. **FP8 is the DGX Spark's quantization sweet spot**. Native hardware acceleration, minimal quality loss, 2× memory savings over FP16. Use it as the default for any model >13B parameters. [NVIDIA-OFFICIAL]

3. **Multi-model serving is practical and recommended**. The 128GB unified memory easily supports a chat model + code model + embedding model simultaneously. This enables multi-specialty applications without model swapping latency. [TESTED]

4. **Context length is the hidden memory consumer**. A 70B model at FP8 fits at 70GB, but adding 128K context KV cache at FP8 adds another 40GB. Budget context length explicitly, especially for large models. [INFERRED]

5. **Generation speed is the user-facing bottleneck, not prompt processing**. The DGX Spark's 273 GB/s memory bandwidth limits token generation to ~10–60 tokens/sec depending on model size. This is adequate for interactive use (humans read at ~4 tokens/sec) but not for high-throughput batch generation. [COMMUNITY]

6. **The zero marginal inference cost is the key economic advantage**. After purchasing the hardware, every inference is free. This changes the build-vs-buy calculation for AI products significantly — high volume makes local inference dramatically cheaper than API-based alternatives. [INFERRED]

## Recommended Actions

1. **Immediate**: Pull `llama3.1:8b` and `nomic-embed-text` via Ollama. These are the baseline models for development. [TESTED]
2. **First project**: Use the memory budget formula to validate your model fits with KV cache + system overhead before downloading.
3. **Production path**: Evaluate vLLM via NGC container or dgx-spark-vllm for any model serving beyond prototyping.
4. **Multi-model setup**: Start with Config A (8B + code + embedding = 29GB). Add larger models as you understand your actual workload patterns.
5. **Quantization testing**: For any model >13B, compare FP8 vs GGUF Q4 quality on your specific tasks. FP8 is usually better quality; GGUF Q4 uses less memory.

## Confidence Level

**HIGH** — Model parameter counts, memory calculations (params × bytes), and quantization ratios are mathematical facts. Specific model recommendations are based on well-established community benchmarks and NVIDIA's own published benchmarks (e.g., the Nemotron 3 Super 120B figure). Performance expectations carry MEDIUM confidence as they are approximate and vary with configuration, context length, and thermal conditions.

**Reasoning**: The memory budget formula is straightforward arithmetic from known specifications. The model recommendations reflect the current state of the open-weight LLM ecosystem as of March 2026. Model quality rankings may shift as new models are released, but the sizing and quantization guidance is tied to hardware specifications and will remain valid. Performance benchmarks are the least reliable data — they are approximate, community-reported, and subject to variation. Treat them as order-of-magnitude estimates, not guarantees.

## Source Notes

- 128GB unified memory specification: [NVIDIA-OFFICIAL] DGX Spark product page.
- FP8 native Blackwell hardware support: [NVIDIA-OFFICIAL] Blackwell architecture whitepaper; 5th-gen Tensor Core specifications.
- Nemotron 3 Super 120B benchmark (42,855 tokens/sec prompt processing at 128K): [NVIDIA-OFFICIAL] NVIDIA published benchmark.
- NemoClaw agent framework: [NVIDIA-OFFICIAL] NVIDIA NemoClaw documentation.
- Llama 3.1 8B performance (~40–60 tokens/sec): [COMMUNITY] Multiple user benchmarks on DGX Spark forums.
- 70B GGUF Q4 performance (~10–20 tokens/sec): [COMMUNITY] Community benchmarks.
- Memory bandwidth bottleneck (273 GB/s): [NVIDIA-OFFICIAL] LPDDR5x specification; [COMMUNITY] Confirmed as the generation speed limiter.
- Ollama pre-installed status: [NVIDIA-OFFICIAL] DGX Spark ships with Ollama configured.
- NGC container images: [NVIDIA-OFFICIAL] NGC catalog.
- dgx-spark-vllm package: [NVIDIA-OFFICIAL] Published on PyPI by NVIDIA.
- Quantization quality comparisons (FP8 within 1-2% of FP16): [NVIDIA-OFFICIAL] NVIDIA quantization benchmarks.
- Community GGUF repositories (TheBloke, bartowski): [COMMUNITY] HuggingFace model hubs.
- Thermal throttling impact on sustained performance: [COMMUNITY] User reports; [TESTED] Temperature-performance correlation observed.

## Unresolved Questions

1. **What is the exact FP8 quality degradation per model family?** NVIDIA reports "within 1-2%" generically, but per-model and per-task degradation varies. Systematic benchmarks on DGX Spark for the recommended models have not been published. [NVIDIA-OFFICIAL] [COMMUNITY]

2. **Will Ollama add FP8 support?** Currently Ollama serves GGUF only. FP8 inference requires vLLM or NIM. If Ollama adds FP8, it would simplify production serving significantly. [COMMUNITY]

3. **What is the maximum practical context length for 70B FP8 models?** The theoretical KV cache calculation suggests ~128K is possible but extremely tight. Real-world practical limits with overhead may be lower. [INFERRED]

4. **How does Nemotron 3 Super 120B compare to DeepSeek-R1 on reasoning tasks on DGX Spark specifically?** Both fit in memory but via different architectures (MoE vs dense-MoE). Comparative benchmarks on this hardware are not available. [COMMUNITY]

5. **What is the overhead of running multiple vLLM instances vs a single multi-model Ollama instance?** Multiple Docker containers add per-container overhead. The actual overhead and performance difference has not been systematically measured on DGX Spark. [INFERRED]

6. **When will NVIDIA update the NIM catalog with more Blackwell-optimized models?** The NIM catalog is expanding but not all recommended models have NIM containers. [NVIDIA-OFFICIAL]
