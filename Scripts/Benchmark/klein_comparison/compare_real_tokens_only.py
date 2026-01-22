#!/usr/bin/env python3
"""
More detailed comparison focusing on real tokens only.
"""

import numpy as np

# Load embeddings
py_data = np.load("klein_embeddings_python.npz")
py_emb = py_data["embeddings"]
swift_emb = np.fromfile("/tmp/swift_klein_embeddings.bin", dtype=np.float32).reshape(1, 512, 7680)

print("Comparing real token embeddings (positions 0-18)")
print("=" * 60)

# Compare each real token position
for pos in range(19):
    py_vec = py_emb[0, pos, :]
    sw_vec = swift_emb[0, pos, :]

    # Cosine similarity
    cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))

    # Relative difference
    rel_diff = np.mean(np.abs(py_vec - sw_vec) / (np.abs(py_vec) + 1e-6))

    # Max values comparison
    py_max_idx = np.argmax(np.abs(py_vec))
    sw_max_idx = np.argmax(np.abs(sw_vec))

    print(f"Pos {pos:2d}: cos_sim={cos_sim:.4f}, rel_diff={rel_diff:.4f}, "
          f"py_max[{py_max_idx}]={py_vec[py_max_idx]:.1f}, sw_max[{sw_max_idx}]={sw_vec[sw_max_idx]:.1f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Average cosine similarity over real tokens
cos_sims = []
for pos in range(19):
    py_vec = py_emb[0, pos, :]
    sw_vec = swift_emb[0, pos, :]
    cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))
    cos_sims.append(cos_sim)

print(f"Average cosine similarity: {np.mean(cos_sims):.4f}")
print(f"Min cosine similarity: {np.min(cos_sims):.4f}")
print(f"Max cosine similarity: {np.max(cos_sims):.4f}")

# Check if max value positions match (important for embedding structure)
matching_max_positions = 0
for pos in range(19):
    py_max_idx = np.argmax(np.abs(py_emb[0, pos, :]))
    sw_max_idx = np.argmax(np.abs(swift_emb[0, pos, :]))
    if py_max_idx == sw_max_idx:
        matching_max_positions += 1

print(f"\nMatching max value positions: {matching_max_positions}/19 ({100*matching_max_positions/19:.1f}%)")

# Check value ranges
print(f"\nPython embedding range: [{py_emb[0, :19, :].min():.2f}, {py_emb[0, :19, :].max():.2f}]")
print(f"Swift embedding range:  [{swift_emb[0, :19, :].min():.2f}, {swift_emb[0, :19, :].max():.2f}]")
