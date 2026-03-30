# NVIDIA DGX Spark — Model Performance Reference

*Last Updated: March 2026*

## Inference Benchmarks (Ollama v0.12.6, Firmware 580.95.05)

| Model | Quantization | Prefill (tok/s) | Decode (tok/s) | Notes |
|-------|--------------|-----------------|----------------|-------|
| Llama 3.1 8B | Q4_K_M | 7,614 | 38.02 | Fast small model |
| Llama 3.1 70B | Q4_K_M | 1,911 | 4.42 | Fits in memory but slow decode |
| Gemma3 12B | Q4_K_M | 1,894 | 24.25 | Good balance |
| DeepSeek-R1 14B | Default | 5,919 | 19.99 | Strong reasoning |
| GPT-OSS 20B | Default | 3,224 | 58.27 | NVIDIA optimized |
| GPT-OSS 120B | Default | 1,169 | 41.14 | Large model, good speed |

## Platform Comparison (GPT-OSS 120B, Ollama)

| Platform | Generation Speed | Notes |
|----------|------------------|-------|
| DGX Spark | 41 tok/s | Consistent, no degradation |
| Mac Studio M4 Max | 34→6 tok/s | Degrades with context length |
| RTX 4080 (16GB) | 12.45 tok/s | Limited by VRAM |

## Inference Runtime Comparison (2026 Benchmarks)

| Runtime | Throughput | Best For |
|---------|------------|----------|
| Ollama | ~41 tok/s (single user) | Ease of use, pre-installed, quick start |
| vLLM | 793 tok/s (multi-user) | Production serving, high throughput, concurrent requests |
| llama.cpp | Varies | Flexibility, GGUF models, fine-grained control |
| ExLlamaV3 | High | Blackwell-optimized, EXL3 format |
| LM Studio | Similar to Ollama | GUI, network serving, easy model management |

vLLM achieves 1.5x higher throughput and 1.7x faster time-to-first-token than TGI. PagedAttention reduces memory fragmentation from 60–80% to <4%.

## Recommended Model Formats for DGX Spark

| Format | Compression | Speed | Accuracy | Best For |
|--------|-------------|-------|----------|----------|
| NVFP4 | 3.5x vs FP16 | ~1.6x vs FP8 | <1% loss (>7B) | Blackwell-native, maximum efficiency |
| GGUF Q4_K_M | ~4x vs FP16 | Good | Good | Universal compatibility, "sweet spot" |
| EXL3 | Variable | Fast | Good | ExLlamaV3, Blackwell-optimized |
| FP8 | 2x vs FP16 | Fast | Excellent | Balance of quality and speed |
| FP16 | Baseline | Baseline | Best | When accuracy is paramount |

## Model Size Guide (128GB Unified Memory)

| Model Size | Quantization | Memory Required | Fits Single Spark? | Performance |
|------------|--------------|-----------------|--------------------|---------------|
| 1–3B | FP16 | 2–6GB | ✅ Easily | Very fast |
| 7–8B | Q4_K_M | 4–5GB | ✅ Easily | Fast (38 tok/s) |
| 12–14B | Q4_K_M | 7–9GB | ✅ Easily | Good (20–24 tok/s) |
| 20B | Q4_K_M | 12–15GB | ✅ Yes | Good (58 tok/s with NVIDIA opt) |
| 70B | Q4_K_M | 35–40GB | ✅ Yes | Slower (4.4 tok/s) |
| 120B | Q4_K_M | 60–70GB | ✅ Yes | Good with NVIDIA opt (41 tok/s) |
| 200B | Q4_K_M | 100–120GB | ⚠️ Tight | Possible, monitor memory |
| 405B | Q4_K_M | 200–220GB | ❌ Need dual Spark | Requires two linked Sparks |

## Best Models for DGX Spark (2026)

### General Purpose

- **GPT-OSS 120B**: NVIDIA-optimized, 41 tok/s, excellent general capability
- **Llama 3.1 70B**: Meta's flagship, good all-around
- **Qwen 3.5**: Strong multilingual, competitive with GPT-4 class
- **GLM-5**: Impressive capability range
- **Hermes 4**: Strong instruction following

### Reasoning/Code

- **DeepSeek-R1 14B**: Best reasoning at small size
- **DeepSeek-R1 70B/671B**: State-of-art reasoning (671B needs dual Spark at Q4)

### Small/Fast (for on-device S26 Ultra or quick tasks)

- **Llama 3.2 1B/3B**: Efficient, good for basic tasks
- **Nemotron-Nano 4B**: NVIDIA-optimized for local workstations
- **Phi-3.5 Mini**: Microsoft's efficient small model
- **Gemma3 4B**: Google's compact model

### Image Generation

- **FLUX.2**: 32B params, runs at full FP16 (90GB) on Spark. FP8 reduces to 54GB with 40% perf improvement.
- **Stable Diffusion XL**: Smaller, faster
- **NVFP4-optimized FLUX.2**: 10.2x speedup on Blackwell

## Memory Management Tips

1. Only load models you're actively using (Ollama loads/unloads automatically)
2. Set `OLLAMA_MAX_LOADED_MODELS=1` if running multiple models
3. For vLLM: `--gpu-memory-utilization 0.8` leaves 20% headroom
4. Monitor with: `watch -n 1 'nvidia-smi; free -h'`
5. Kill idle model servers before loading larger models
6. Quantize aggressively (Q4_K_M or NVFP4) — quality loss is minimal

---
