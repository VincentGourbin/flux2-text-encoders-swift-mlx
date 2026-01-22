#!/usr/bin/env python3
"""
Compare Klein embeddings between Python (official Qwen/Qwen3-4B) and Swift (MLX lmstudio 8bit).
Note: Different models (official bf16 vs MLX 8bit) will have quantization differences.
"""

import numpy as np

# Load Python embeddings (using official Qwen/Qwen3-4B)
print("Loading Python embeddings (Qwen/Qwen3-4B, bf16)...")
py_data = np.load("klein_embeddings_python.npz")
py_emb = py_data["embeddings"]
print(f"Python shape: {py_emb.shape}")
print(f"Python dtype: {py_emb.dtype}")
print(f"Python first 10 at pos 0: {py_emb[0, 0, :10]}")
print(f"Python first 10 at pos 511: {py_emb[0, 511, :10]}")

# Load Swift embeddings (using MLX 8bit quantized)
print("\nLoading Swift embeddings (lmstudio MLX 8bit)...")
swift_emb = np.fromfile("/tmp/swift_klein_embeddings.bin", dtype=np.float32)
swift_emb = swift_emb.reshape(1, 512, 7680)
print(f"Swift shape: {swift_emb.shape}")
print(f"Swift first 10 at pos 0: {swift_emb[0, 0, :10]}")
print(f"Swift first 10 at pos 511: {swift_emb[0, 511, :10]}")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

# Note: These use different models:
# - Python: Qwen/Qwen3-4B (official, bf16/fp32)
# - Swift: lmstudio-community/Qwen3-4B-MLX-8bit (8-bit quantized)
# So we expect significant numerical differences due to quantization!

# Check position 0 (start of real tokens)
py_vec = py_emb[0, 0, :]
sw_vec = swift_emb[0, 0, :]
cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))
print(f"\nPosition 0 (first token):")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Max abs diff: {np.max(np.abs(py_vec - sw_vec)):.2f}")
print(f"  Mean abs diff: {np.mean(np.abs(py_vec - sw_vec)):.2f}")

# Check position 18 (last real token before padding)
py_vec = py_emb[0, 18, :]
sw_vec = swift_emb[0, 18, :]
cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))
print(f"\nPosition 18 (last real token):")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Max abs diff: {np.max(np.abs(py_vec - sw_vec)):.2f}")
print(f"  Mean abs diff: {np.mean(np.abs(py_vec - sw_vec)):.2f}")

# Check position 19 (first padding token)
py_vec = py_emb[0, 19, :]
sw_vec = swift_emb[0, 19, :]
cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))
print(f"\nPosition 19 (first padding token):")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Max abs diff: {np.max(np.abs(py_vec - sw_vec)):.2f}")

# Check position 500 (deep in padding)
py_vec = py_emb[0, 500, :]
sw_vec = swift_emb[0, 500, :]
cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))
print(f"\nPosition 500 (padding region):")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Max abs diff: {np.max(np.abs(py_vec - sw_vec)):.2f}")

# Overall comparison
cos_sim_overall = np.sum(py_emb * swift_emb) / (np.linalg.norm(py_emb) * np.linalg.norm(swift_emb))
print(f"\nOverall cosine similarity: {cos_sim_overall:.6f}")
print(f"Overall max abs diff: {np.max(np.abs(py_emb - swift_emb)):.2f}")
print(f"Overall mean abs diff: {np.mean(np.abs(py_emb - swift_emb)):.4f}")

# Compare just the real tokens (positions 0-18)
py_real = py_emb[0, :19, :]
sw_real = swift_emb[0, :19, :]
cos_sim_real = np.sum(py_real * sw_real) / (np.linalg.norm(py_real) * np.linalg.norm(sw_real))
print(f"\nReal tokens only (0-18):")
print(f"  Cosine similarity: {cos_sim_real:.6f}")
print(f"  Max abs diff: {np.max(np.abs(py_real - sw_real)):.2f}")
print(f"  Mean abs diff: {np.mean(np.abs(py_real - sw_real)):.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
NOTE: Significant differences are EXPECTED because:
- Python uses Qwen/Qwen3-4B (official bf16 model)
- Swift uses lmstudio-community/Qwen3-4B-MLX-8bit (8-bit quantized)

8-bit quantization typically reduces model precision significantly.
A cosine similarity > 0.9 for real tokens indicates the implementations
are correctly aligned, with differences due to quantization.

For a fair comparison, both should use the same model (either both
use the official bf16 or both use the same quantized version).
""")
