#!/usr/bin/env python3
"""
Compare Klein 9B (Qwen3-8B) embeddings between MLX Python and Swift.
"""

import numpy as np

# Load MLX Python embeddings
print("Loading MLX Python embeddings (Qwen3-8B)...")
mlx_data = np.load("klein_embeddings_8b_mlx.npz")
mlx_emb = mlx_data["embeddings"]
print(f"MLX Python shape: {mlx_emb.shape}")
print(f"First 10 at pos 0: {mlx_emb[0, 0, :10]}")

# Load Swift embeddings
print("\nLoading Swift embeddings (Klein 9B)...")
swift_emb = np.fromfile("/tmp/swift_klein_8b_embeddings.bin", dtype=np.float32).reshape(1, 512, 12288)
print(f"Swift shape: {swift_emb.shape}")
print(f"First 10 at pos 0: {swift_emb[0, 0, :10]}")

print("\n" + "=" * 60)
print("KLEIN 9B (Qwen3-8B) COMPARISON")
print("=" * 60)

# Per-position comparison for real tokens
print("\nPer-position cosine similarity (real tokens):")
for pos in range(19):
    mlx_vec = mlx_emb[0, pos, :]
    sw_vec = swift_emb[0, pos, :]
    cos_sim = np.dot(mlx_vec, sw_vec) / (np.linalg.norm(mlx_vec) * np.linalg.norm(sw_vec))
    max_diff = np.max(np.abs(mlx_vec - sw_vec))
    print(f"  Pos {pos:2d}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")

# Overall for real tokens
mlx_real = mlx_emb[0, :19, :]
sw_real = swift_emb[0, :19, :]
cos_sim_real = np.sum(mlx_real * sw_real) / (np.linalg.norm(mlx_real) * np.linalg.norm(sw_real))

print(f"\n{'=' * 60}")
print(f"Real tokens (0-18) cosine similarity: {cos_sim_real:.6f}")
print(f"Real tokens max abs diff: {np.max(np.abs(mlx_real - sw_real)):.4f}")
print(f"Real tokens mean abs diff: {np.mean(np.abs(mlx_real - sw_real)):.6f}")

# Check max value indices
print(f"\nMax value index at pos 0:")
mlx_max_idx = np.argmax(np.abs(mlx_emb[0, 0, :]))
sw_max_idx = np.argmax(np.abs(swift_emb[0, 0, :]))
print(f"  MLX Python: idx={mlx_max_idx}, value={mlx_emb[0, 0, mlx_max_idx]:.2f}")
print(f"  Swift:      idx={sw_max_idx}, value={swift_emb[0, 0, sw_max_idx]:.2f}")

# Verdict
print(f"\n{'=' * 60}")
if cos_sim_real > 0.999:
    print("✅ EXCELLENT: Klein 9B implementation is correct!")
elif cos_sim_real > 0.99:
    print("✅ VERY GOOD: Klein 9B implementation is correct with minor precision differences.")
elif cos_sim_real > 0.95:
    print("⚠️ GOOD: Klein 9B has some differences - review implementation.")
else:
    print("❌ WARNING: Klein 9B has significant differences - needs investigation.")
