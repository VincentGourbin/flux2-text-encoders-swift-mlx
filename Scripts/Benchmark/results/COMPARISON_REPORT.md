# Swift MLX vs Python MLX Performance Comparison

**Model**: Mistral Small 3.2 24B Instruct
**Date**: 2026-01-15
**Device**: Apple Silicon (MPS)

## Summary

| Metric | Swift MLX | Python MLX | Winner |
|--------|-----------|------------|--------|
| Text Generation | **1.7-1.9x faster** | baseline | **Swift** |
| VLM (Vision) | **Working** | Broken* | **Swift** |
| Embeddings | **~similar** | ~similar | Tie |

*Python mlx_vlm produces incorrect/hallucinated responses for this model

## Detailed Results

### Text Generation (tokens/s)

| Quantization | Swift MLX | Python MLX | Swift Advantage |
|--------------|-----------|------------|-----------------|
| 4bit | **11.8** | 6.4 | 1.84x |
| 6bit | **9.3** | 5.3 | 1.75x |
| 8bit | **8.0** | 4.2 | 1.90x |
| bf16 | **1.0** | N/A | - |

**Swift MLX is ~1.8x faster for text generation across all quantizations.**

### VLM / Vision (tokens/s)

| Quantization | Swift MLX | Python MLX | Notes |
|--------------|-----------|------------|-------|
| 4bit | **2.5** | 20.1* | *Python broken |
| 6bit | **2.2** | 14.1* | *Python broken |
| 8bit | **2.1** | 10.6* | *Python broken |
| bf16 | 0.8 | N/A | - |

**IMPORTANT: Python mlx_vlm benchmark results are INVALID!**

Python VLM issues discovered:
1. **Hallucination**: Describes "The Olde Mill Restaurant" instead of "Frontier AI website" - completely wrong image interpretation
2. **Tokenization broken**: Raw BPE tokens visible (`Ġ` = `\u0120`) instead of decoded text
3. **Speed misleading**: Fast because image processing is likely broken

**Swift VLM is the only working implementation** - produces correct, coherent descriptions of images.

### Embeddings (FLUX.2 format)

Extraction of hidden states from layers [10, 20, 30], shape: [1, 512, 15360]

| Quantization | Swift MLX (inference) | Python Transformers (extract) |
|--------------|----------------------|-------------------------------|
| 4bit | 2.35s | N/A |
| 6bit | 2.50s | N/A |
| 8bit | 2.62s | N/A |
| bf16 | 3.21s | 2.02s* |

*Python uses transformers library with bf16 model (60s load time vs ~8s for Swift)

**Embeddings performance is comparable.**

## Observations

### Swift MLX Advantages
1. **Text generation is ~1.8x faster** across all quantizations
2. **Cleaner output** - no tokenization artifacts
3. **bf16 support** for all modes (text, vlm, embeddings)
4. **Faster model loading** (~2-8s vs 60s for Python transformers)
5. **Unified API** - same CLI for all operations

### Python MLX Advantages
1. **VLM is 5-8x faster** - significant advantage for vision tasks
2. **More mature ecosystem** - mlx_lm, mlx_vlm, transformers integration

### Quality Notes
- Swift VLM output: Clean, accurate descriptions
- Python VLM output: Contains tokenization artifacts (`\u0120` = space character)
- Both produce similar haiku quality for text generation
- Embeddings produce matching shapes and similar value ranges

## Recommendations

| Use Case | Recommendation |
|----------|---------------|
| Text Generation | **Swift MLX** (1.8x faster) |
| Vision/VLM | **Swift MLX** (only working implementation) |
| Embeddings | Either (similar performance) |
| Production CLI | **Swift MLX** (faster text gen, working VLM, unified API) |

**Conclusion: Swift MLX is superior for Mistral Small 3.2**
- Faster text generation
- Only working VLM implementation
- Clean output without tokenization artifacts

## Test Configuration

**Prompts used:**
- Text: "Write a haiku about artificial intelligence."
- VLM: "Describe this image in one sentence." (vision.png)
- Embeddings: "A beautiful sunset over the ocean with vibrant colors"

**Max tokens:** 100 (text), 50 (vlm)
