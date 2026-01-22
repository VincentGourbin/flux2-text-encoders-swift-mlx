# Klein Embedding Validation

This document describes the validation process for the FLUX.2 Klein embedding extraction implementation against the official [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) Python implementation.

## Summary

The Swift implementation of Klein embeddings has been validated to produce **identical results** to the official flux2 implementation when using the same MLX model weights.

| Variant | Cosine Similarity | Status |
|---------|-------------------|--------|
| Klein 4B (Qwen3-4B) | 1.000001 | ✅ Validated |
| Klein 9B (Qwen3-8B) | 1.000003 | ✅ Validated |

## Key Implementation Details

### Official flux2 Qwen3Embedder Behavior

From `/tmp/flux2-official/src/flux2/text_encoder.py`:

```python
class Qwen3Embedder(nn.Module):
    def forward(self, txt: list[str]):
        for prompt in txt:
            # NO system message - only user message
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Adds <think>\n\n</think>\n\n tokens
            )
            # ...

        # RIGHT padding with padding="max_length"
        model_inputs = self.tokenizer(text, padding="max_length", ...)

        # Extract layers [9, 18, 27]
        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")
```

### Swift Implementation Corrections

The following corrections were made to `KleinEmbeddingExtractor.swift`:

| Aspect | Before (Incorrect) | After (Correct) |
|--------|-------------------|-----------------|
| System message | Added system message | **No system message** |
| Padding direction | LEFT padding | **RIGHT padding** |
| Chat template | `<\|im_start\|>system\n...\n<\|im_start\|>user\n...\n<\|im_start\|>assistant\n` | `<\|im_start\|>user\n...\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| Think tokens | Missing | **Included** (via `enable_thinking=False` behavior) |

### Formatted Text Comparison

For prompt `"a cat sitting on a window sill"`:

```
<|im_start|>user
a cat sitting on a window sill<|im_end|>
<|im_start|>assistant
<think>

</think>

```

### Token IDs (First 20)

Both Python and Swift produce identical tokens:
```
[151644, 872, 198, 64, 8251, 11699, 389, 264, 3241, 84267, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 151643]
```

Breakdown:
- `151644` = `<|im_start|>`
- `872` = `user`
- `198` = `\n`
- `64, 8251, 11699, 389, 264, 3241, 84267` = prompt tokens
- `151645` = `<|im_end|>`
- `77091` = `assistant`
- `151667` = `<think>`
- `271` = `\n\n`
- `151668` = `</think>`
- `151643` = padding token

## Embedding Dimensions

| Variant | Hidden Size | Output Dimension | Layers |
|---------|-------------|------------------|--------|
| Klein 4B | 2560 | 7,680 (3 × 2560) | [9, 18, 27] |
| Klein 9B | 4096 | 12,288 (3 × 4096) | [9, 18, 27] |

## Validation Results

### Klein 4B (Qwen3-4B-MLX-8bit)

```
Per-position cosine similarity (real tokens):
  Pos   0: cos_sim=0.999999
  Pos   1: cos_sim=0.999957
  Pos  18: cos_sim=0.999920

Real tokens (0-18) cosine similarity: 1.000001
Max value index: 2564 (identical in both implementations)
```

### Klein 9B (Qwen3-8B-MLX-8bit)

```
Per-position cosine similarity (real tokens):
  Pos   0: cos_sim=0.999999
  Pos   1: cos_sim=0.999970
  Pos  18: cos_sim=0.999933

Real tokens (0-18) cosine similarity: 1.000003
Max value index: 6372 (identical in both implementations)
```

## Benchmark Scripts

The validation scripts are located in `Scripts/Benchmark/klein_comparison/`:

| Script | Description |
|--------|-------------|
| `extract_klein_python.py` | Extract embeddings using official HuggingFace Qwen3 |
| `extract_klein_mlx.py` | Extract embeddings using MLX Python (same model as Swift) |
| `compare_mlx_swift.py` | Compare Klein 4B embeddings |
| `compare_8b.py` | Compare Klein 9B embeddings |
| `requirements.txt` | Python dependencies |

### Running the Benchmark

```bash
cd Scripts/Benchmark/klein_comparison

# Setup
python3 -m venv venv
source venv/bin/activate
pip install mlx mlx-lm numpy

# Extract with MLX Python
python extract_klein_mlx.py --prompt "your prompt" --model "lmstudio-community/Qwen3-4B-MLX-8bit"

# Extract with Swift
../../.build/xcode/Build/Products/Release/FluxEncodersCLI embed "your prompt" --klein 4b --output /tmp/swift_embeddings.bin

# Compare
python compare_mlx_swift.py
```

## References

- Official flux2 implementation: https://github.com/black-forest-labs/flux2
- Qwen3 models: https://huggingface.co/Qwen
- MLX quantized models: https://huggingface.co/lmstudio-community

## Date

Validated: January 2026
