#!/usr/bin/env python3
"""
Compare Klein embeddings between MLX Python and Swift (same model).
"""

import json
import numpy as np

# Load MLX Python embeddings
print("Loading MLX Python embeddings...")
mlx_data = np.load("klein_embeddings_mlx_python.npz")
mlx_emb = mlx_data["embeddings"]
mlx_tokens = mlx_data["token_ids"].tolist()
print(f"MLX Python shape: {mlx_emb.shape}")

# Load Swift embeddings
print("\nLoading Swift embeddings...")
swift_emb = np.fromfile("/tmp/swift_klein_embeddings.bin", dtype=np.float32).reshape(1, 512, 7680)
print(f"Swift shape: {swift_emb.shape}")

# Load Swift tokens
with open("klein_embeddings_mlx_python.tokens.json") as f:
    mlx_token_info = json.load(f)

print("\n" + "=" * 60)
print("TOKEN COMPARISON")
print("=" * 60)

# Swift tokens from CLI output (first 20)
swift_first_20 = [151644, 872, 198, 64, 8251, 11699, 389, 264, 3241, 84267, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271, 151643]
mlx_first_20 = mlx_tokens[:20]

print(f"MLX Python first 20:  {mlx_first_20}")
print(f"Swift first 20:       {swift_first_20}")
print(f"Tokens match: {mlx_first_20 == swift_first_20}")

print("\n" + "=" * 60)
print("EMBEDDING COMPARISON (Same MLX 8-bit model)")
print("=" * 60)

# Per-position comparison
print("\nPer-position cosine similarity:")
for pos in [0, 1, 5, 10, 18, 19, 100, 500, 511]:
    mlx_vec = mlx_emb[0, pos, :]
    sw_vec = swift_emb[0, pos, :]

    cos_sim = np.dot(mlx_vec, sw_vec) / (np.linalg.norm(mlx_vec) * np.linalg.norm(sw_vec))
    max_diff = np.max(np.abs(mlx_vec - sw_vec))
    mean_diff = np.mean(np.abs(mlx_vec - sw_vec))

    token_type = "real" if pos < 19 else "pad"
    print(f"  Pos {pos:3d} ({token_type:4s}): cos_sim={cos_sim:.6f}, max_diff={max_diff:8.2f}, mean_diff={mean_diff:.4f}")

# Overall statistics
print("\nOverall statistics:")
cos_sim_overall = np.sum(mlx_emb * swift_emb) / (np.linalg.norm(mlx_emb) * np.linalg.norm(swift_emb))
print(f"  Overall cosine similarity: {cos_sim_overall:.6f}")
print(f"  Overall max abs diff: {np.max(np.abs(mlx_emb - swift_emb)):.2f}")
print(f"  Overall mean abs diff: {np.mean(np.abs(mlx_emb - swift_emb)):.6f}")

# Real tokens only (0-18)
mlx_real = mlx_emb[0, :19, :]
sw_real = swift_emb[0, :19, :]
cos_sim_real = np.sum(mlx_real * sw_real) / (np.linalg.norm(mlx_real) * np.linalg.norm(sw_real))
print(f"\n  Real tokens (0-18) cosine similarity: {cos_sim_real:.6f}")
print(f"  Real tokens max abs diff: {np.max(np.abs(mlx_real - sw_real)):.2f}")
print(f"  Real tokens mean abs diff: {np.mean(np.abs(mlx_real - sw_real)):.6f}")

# Padding tokens only (19-511)
mlx_pad = mlx_emb[0, 19:, :]
sw_pad = swift_emb[0, 19:, :]
cos_sim_pad = np.sum(mlx_pad * sw_pad) / (np.linalg.norm(mlx_pad) * np.linalg.norm(sw_pad))
print(f"\n  Padding tokens (19-511) cosine similarity: {cos_sim_pad:.6f}")
print(f"  Padding tokens max abs diff: {np.max(np.abs(mlx_pad - sw_pad)):.2f}")
print(f"  Padding tokens mean abs diff: {np.mean(np.abs(mlx_pad - sw_pad)):.6f}")

# Value comparison at position 0
print("\n" + "=" * 60)
print("VALUE COMPARISON AT POSITION 0")
print("=" * 60)
print("First 10 dimensions:")
print(f"  MLX Python: {mlx_emb[0, 0, :10]}")
print(f"  Swift:      {swift_emb[0, 0, :10]}")

# Check max value indices
mlx_max_idx = np.argmax(np.abs(mlx_emb[0, 0, :]))
sw_max_idx = np.argmax(np.abs(swift_emb[0, 0, :]))
print(f"\nMax value indices at pos 0:")
print(f"  MLX Python: idx={mlx_max_idx}, value={mlx_emb[0, 0, mlx_max_idx]:.2f}")
print(f"  Swift:      idx={sw_max_idx}, value={swift_emb[0, 0, sw_max_idx]:.2f}")

# Verdict
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

if cos_sim_real > 0.999:
    print("✅ EXCELLENT: Real token embeddings are nearly identical!")
    print("   The Swift implementation matches MLX Python perfectly.")
elif cos_sim_real > 0.99:
    print("✅ VERY GOOD: Real token embeddings are very close.")
    print("   Minor differences likely due to floating point precision.")
elif cos_sim_real > 0.95:
    print("⚠️ GOOD: Embeddings are similar but have some differences.")
    print("   May be due to implementation details.")
elif cos_sim_real > 0.9:
    print("⚠️ ACCEPTABLE: Embeddings are reasonably similar.")
    print("   Review implementation for potential issues.")
else:
    print("❌ WARNING: Significant differences detected!")
    print("   Review the implementation carefully.")
