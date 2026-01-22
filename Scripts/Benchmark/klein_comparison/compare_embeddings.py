#!/usr/bin/env python3
"""
Compare Klein embeddings between Python (official) and Swift implementations.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_python_embeddings(path: Path) -> dict:
    """Load embeddings from Python npz file."""
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "token_ids": data["token_ids"].tolist(),
        "prompt": str(data["prompt"]),
        "formatted_text": str(data["formatted_text"]),
    }


def load_swift_embeddings(path: Path) -> dict:
    """Load embeddings from Swift binary file + JSON metadata."""
    # Swift saves embeddings as raw binary (float32)
    embeddings_raw = np.fromfile(path, dtype=np.float32)

    # Try to load metadata from accompanying JSON
    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
            shape = meta.get("shape", [1, 512, -1])
    else:
        # Assume shape based on Klein 4B: [1, 512, 7680]
        shape = [1, 512, embeddings_raw.size // 512]

    embeddings = embeddings_raw.reshape(shape)

    # Load token IDs if available
    tokens_path = path.with_suffix(".tokens.json")
    if tokens_path.exists():
        with open(tokens_path) as f:
            tokens_data = json.load(f)
            token_ids = tokens_data.get("token_ids", [])
            formatted_text = tokens_data.get("formatted_text", "")
    else:
        token_ids = []
        formatted_text = ""

    return {
        "embeddings": embeddings,
        "token_ids": token_ids,
        "formatted_text": formatted_text,
    }


def compare_embeddings(python_data: dict, swift_data: dict, verbose: bool = False) -> dict:
    """Compare embeddings between Python and Swift implementations."""
    py_emb = python_data["embeddings"]
    sw_emb = swift_data["embeddings"]

    results = {
        "shapes_match": py_emb.shape == sw_emb.shape,
        "python_shape": list(py_emb.shape),
        "swift_shape": list(sw_emb.shape),
    }

    if not results["shapes_match"]:
        print(f"ERROR: Shape mismatch! Python: {py_emb.shape}, Swift: {sw_emb.shape}")
        return results

    # Numerical comparison
    abs_diff = np.abs(py_emb - sw_emb)
    rel_diff = abs_diff / (np.abs(py_emb) + 1e-8)

    results.update({
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
        "cosine_similarity": float(
            np.sum(py_emb * sw_emb) / (np.linalg.norm(py_emb) * np.linalg.norm(sw_emb))
        ),
    })

    # Compare per-position statistics
    if verbose:
        # Compare first few positions
        for pos in [0, 1, 2, 256, 511]:
            py_vec = py_emb[0, pos, :]
            sw_vec = sw_emb[0, pos, :]
            cos_sim = np.dot(py_vec, sw_vec) / (np.linalg.norm(py_vec) * np.linalg.norm(sw_vec))
            print(f"Position {pos}: cosine_sim={cos_sim:.6f}, max_diff={np.max(np.abs(py_vec - sw_vec)):.6f}")

    # Token comparison
    py_tokens = python_data.get("token_ids", [])
    sw_tokens = swift_data.get("token_ids", [])

    if py_tokens and sw_tokens:
        results["tokens_match"] = py_tokens == sw_tokens
        results["token_diff_count"] = sum(1 for a, b in zip(py_tokens, sw_tokens) if a != b)

        if not results["tokens_match"] and verbose:
            print("\nToken differences:")
            for i, (p, s) in enumerate(zip(py_tokens, sw_tokens)):
                if p != s:
                    print(f"  Position {i}: Python={p}, Swift={s}")
                    if i > 20:
                        print(f"  ... and more differences")
                        break

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Klein embeddings")
    parser.add_argument("--python", type=str, required=True, help="Python embeddings file (.npz)")
    parser.add_argument("--swift", type=str, required=True, help="Swift embeddings file (.bin)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("Loading Python embeddings...")
    py_data = load_python_embeddings(Path(args.python))
    print(f"  Shape: {py_data['embeddings'].shape}")
    print(f"  Tokens: {len(py_data['token_ids'])}")

    print("\nLoading Swift embeddings...")
    sw_data = load_swift_embeddings(Path(args.swift))
    print(f"  Shape: {sw_data['embeddings'].shape}")
    print(f"  Tokens: {len(sw_data['token_ids'])}")

    print("\nComparing embeddings...")
    results = compare_embeddings(py_data, sw_data, verbose=args.verbose)

    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)

    if results["shapes_match"]:
        print(f"✓ Shapes match: {results['python_shape']}")
    else:
        print(f"✗ Shapes MISMATCH: Python={results['python_shape']}, Swift={results['swift_shape']}")
        return 1

    print(f"\nNumerical comparison:")
    print(f"  Max absolute diff:  {results['max_abs_diff']:.6e}")
    print(f"  Mean absolute diff: {results['mean_abs_diff']:.6e}")
    print(f"  Max relative diff:  {results['max_rel_diff']:.6e}")
    print(f"  Mean relative diff: {results['mean_rel_diff']:.6e}")
    print(f"  Cosine similarity:  {results['cosine_similarity']:.6f}")

    if "tokens_match" in results:
        if results["tokens_match"]:
            print(f"\n✓ Token IDs match perfectly")
        else:
            print(f"\n✗ Token IDs DIFFER: {results['token_diff_count']} differences")

    # Verdict
    print("\n" + "=" * 50)
    if results["cosine_similarity"] > 0.999:
        print("✓ PASS: Embeddings are effectively identical")
        return 0
    elif results["cosine_similarity"] > 0.99:
        print("~ CLOSE: Embeddings are very similar (likely quantization differences)")
        return 0
    elif results["cosine_similarity"] > 0.9:
        print("⚠ WARNING: Embeddings are somewhat similar but have notable differences")
        return 1
    else:
        print("✗ FAIL: Embeddings are significantly different!")
        return 1


if __name__ == "__main__":
    exit(main())
