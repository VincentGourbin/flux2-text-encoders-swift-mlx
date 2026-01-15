#!/Users/vincent/Developpements/mistral-small-3.2-swift-mlx/.venv/bin/python3
"""
Benchmark Python MLX - Mistral Small 3.2
Tests: Text, VLM, Embeddings across 4bit, 6bit, 8bit quantizations
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Model paths (using local cache)
MODELS_DIR = Path("/Users/vincent/Library/Caches/models")
MODEL_PATHS = {
    "4bit": MODELS_DIR / "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
    "6bit": MODELS_DIR / "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit",
    "8bit": MODELS_DIR / "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit",
    "bf16": MODELS_DIR / "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
}

# Test prompts (same as Swift benchmark)
TEXT_PROMPT = "Write a haiku about artificial intelligence."
VLM_PROMPT = "Describe this image in one sentence."
EMBED_PROMPT = "A beautiful sunset over the ocean with vibrant colors"

TEST_IMAGE = Path(__file__).parent.parent.parent / "screenshot" / "vision.png"
RESULTS_DIR = Path(__file__).parent / "results"


def benchmark_text_generation(model_path: Path, quant: str) -> dict:
    """Benchmark text generation."""
    print(f"\n--- Text Generation ({quant}) ---")

    from mlx_lm import load, generate

    # Load model
    load_start = time.time()
    model, tokenizer = load(str(model_path))
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Format prompt with chat template
    messages = [{"role": "user", "content": TEXT_PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    gen_start = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=100,
        verbose=False
    )
    gen_time = time.time() - gen_start

    # Count tokens
    output_tokens = len(tokenizer.encode(response))
    tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0

    print(f"Response: {response[:200]}...")
    print(f"Generation time: {gen_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")

    return {
        "mode": "text",
        "quant": quant,
        "load_time": load_time,
        "gen_time": gen_time,
        "output_tokens": output_tokens,
        "tokens_per_sec": tokens_per_sec,
        "response": response[:500]
    }


def benchmark_vlm(model_path: Path, quant: str) -> dict:
    """Benchmark vision-language model."""
    print(f"\n--- VLM ({quant}) ---")

    try:
        from mlx_vlm import load as vlm_load, generate as vlm_generate
        from PIL import Image
    except ImportError:
        print("mlx-vlm not installed, skipping VLM benchmark")
        return {
            "mode": "vlm",
            "quant": quant,
            "error": "mlx-vlm not installed"
        }

    if not TEST_IMAGE.exists():
        print(f"Test image not found: {TEST_IMAGE}")
        return {
            "mode": "vlm",
            "quant": quant,
            "error": "test image not found"
        }

    # Load model
    load_start = time.time()
    model, processor = vlm_load(str(model_path))
    load_time = time.time() - load_start
    print(f"VLM loaded in {load_time:.2f}s")

    # Generate - mlx_vlm.generate takes (model, processor, prompt, image=...)
    gen_start = time.time()
    result = vlm_generate(
        model,
        processor,
        VLM_PROMPT,
        image=str(TEST_IMAGE),
        max_tokens=50,
        verbose=False
    )
    gen_time = time.time() - gen_start

    # Handle GenerationResult object or string
    if hasattr(result, 'text'):
        response = result.text
        tokens_per_sec = result.generation_tps if hasattr(result, 'generation_tps') else 0
    else:
        response = str(result)
        tokens_per_sec = 0

    print(f"Response: {response}")
    print(f"Generation time: {gen_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")

    return {
        "mode": "vlm",
        "quant": quant,
        "load_time": load_time,
        "gen_time": gen_time,
        "tokens_per_sec": tokens_per_sec,
        "response": response[:500] if response else ""
    }


def benchmark_embeddings_transformers() -> dict:
    """Benchmark embeddings extraction using transformers (like mflux-gradio).

    This uses the same approach as mflux-gradio: transformers with Mistral3ForConditionalGeneration
    and output_hidden_states=True to extract layers 10, 20, 30.
    """
    print(f"\n--- Embeddings (transformers/bf16) ---")

    try:
        import torch
        from transformers import Mistral3ForConditionalGeneration, AutoProcessor
    except ImportError as e:
        print(f"transformers not available: {e}")
        return {
            "mode": "embeddings",
            "quant": "bf16",
            "error": f"transformers not available: {e}"
        }

    # Configuration matching mflux-gradio
    MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    TOKENIZER_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    HIDDEN_STATE_LAYERS = (10, 20, 30)
    MAX_SEQUENCE_LENGTH = 512
    SYSTEM_MESSAGE = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    load_start = time.time()
    print(f"Loading model: {MODEL_ID}")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
    )
    tokenizer = AutoProcessor.from_pretrained(TOKENIZER_ID)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Format input (same as mflux-gradio)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": EMBED_PROMPT}]},
    ]

    inputs = tokenizer.apply_chat_template(
        [messages],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"Input tokens: {input_ids.shape[1]}")

    # Extract embeddings
    extract_start = time.time()

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Stack hidden states from layers 10, 20, 30
    out = torch.stack(
        [output.hidden_states[k] for k in HIDDEN_STATE_LAYERS],
        dim=1
    )

    # Reshape: (batch, num_layers, seq, hidden) -> (batch, seq, num_layers * hidden)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    embeddings = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    extract_time = time.time() - extract_start

    print(f"Embeddings shape: {list(embeddings.shape)}")
    print(f"Extraction time: {extract_time:.2f}s")

    # Stats
    embeddings_np = embeddings.cpu().float().numpy()
    min_val = float(embeddings_np.min())
    max_val = float(embeddings_np.max())
    mean_val = float(embeddings_np.mean())

    print(f"Range: [{min_val:.4f}, {max_val:.4f}], Mean: {mean_val:.6f}")

    # Cleanup
    del model, tokenizer, output, out, embeddings
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "mode": "embeddings",
        "quant": "bf16",
        "load_time": load_time,
        "extract_time": extract_time,
        "shape": [batch_size, seq_len, num_channels * hidden_dim],
        "range": [min_val, max_val],
        "mean": mean_val
    }


def main():
    print("=" * 60)
    print("Python MLX Benchmark - Mistral Small 3.2")
    print("=" * 60)
    print(f"Date: {datetime.now()}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "date": datetime.now().isoformat(),
        "benchmarks": []
    }

    # Note: bf16 is not in MLX format, only HuggingFace format
    # mlx_lm can only load MLX-converted models
    quantizations = ["4bit", "6bit", "8bit"]

    for quant in quantizations:
        model_path = MODEL_PATHS.get(quant)

        if not model_path or not model_path.exists():
            print(f"\nModel {quant} not found at {model_path}, skipping...")
            continue

        print()
        print("=" * 60)
        print(f"Testing {quant} quantization")
        print(f"Path: {model_path}")
        print("=" * 60)

        # Text generation
        try:
            result = benchmark_text_generation(model_path, quant)
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"Text generation failed: {e}")
            results["benchmarks"].append({
                "mode": "text",
                "quant": quant,
                "error": str(e)
            })

        # VLM
        try:
            result = benchmark_vlm(model_path, quant)
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"VLM failed: {e}")
            results["benchmarks"].append({
                "mode": "vlm",
                "quant": quant,
                "error": str(e)
            })

    # Embeddings benchmark using transformers (like mflux-gradio)
    # This runs once with bf16 model, not per-quantization
    print()
    print("=" * 60)
    print("Testing embeddings with transformers (bf16)")
    print("=" * 60)

    try:
        result = benchmark_embeddings_transformers()
        results["benchmarks"].append(result)
    except Exception as e:
        print(f"Embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        results["benchmarks"].append({
            "mode": "embeddings",
            "quant": "bf16",
            "error": str(e)
        })

    # Save results
    results_file = RESULTS_DIR / f"python_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 60)
    print(f"Benchmark complete!")
    print(f"Results saved to: {results_file}")
    print("=" * 60)

    # Print summary
    print("\n=== SUMMARY ===")
    for bench in results["benchmarks"]:
        if "error" in bench:
            print(f"{bench['quant']} {bench['mode']}: ERROR - {bench['error']}")
        else:
            mode = bench["mode"]
            quant = bench["quant"]
            if mode == "text":
                print(f"{quant} text: {bench['gen_time']:.2f}s ({bench['tokens_per_sec']:.1f} tok/s)")
            elif mode == "vlm":
                tps = bench.get('tokens_per_sec', 0)
                print(f"{quant} vlm: {bench['gen_time']:.2f}s ({tps:.1f} tok/s)")
            elif mode == "embeddings":
                print(f"{quant} embeddings: {bench['extract_time']:.2f}s")


if __name__ == "__main__":
    main()
