#!/usr/bin/env python3
"""
Klein embedding extraction using official flux2 implementation.
Compare with Swift implementation to validate correctness.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Official flux2 constants
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512


def extract_klein_embeddings_official(
    prompt: str,
    model,
    tokenizer,
    max_length: int = MAX_LENGTH,
    verbose: bool = False,
) -> tuple[np.ndarray, list[int], str]:
    """
    Extract Klein embeddings exactly as done in official flux2.

    Returns:
        embeddings: numpy array of shape [1, max_length, hidden_dim*3]
        token_ids: list of token IDs after padding
        formatted_text: the text after chat template
    """
    # Official flux2 format - NO system message for Qwen3!
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if verbose:
        print(f"Formatted text:\n{text}")
        print(f"\nText length: {len(text)}")

    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    token_ids_list = input_ids[0].tolist()

    if verbose:
        print(f"\nToken count: {len(token_ids_list)}")
        print(f"Non-padding tokens: {attention_mask.sum().item()}")
        print(f"Padding direction: {'left' if token_ids_list[0] == tokenizer.pad_token_id else 'right'}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        print(f"First 10 tokens: {token_ids_list[:10]}")
        print(f"Last 10 tokens: {token_ids_list[-10:]}")

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Extract hidden states from layers [9, 18, 27]
    hidden_states = [output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3]

    if verbose:
        print(f"\nTotal hidden states: {len(output.hidden_states)}")
        for i, layer_idx in enumerate(OUTPUT_LAYERS_QWEN3):
            print(f"Layer {layer_idx} shape: {hidden_states[i].shape}")

    # Stack and rearrange: [b, 3, l, d] -> [b, l, 3*d]
    out = torch.stack(hidden_states, dim=1)
    embeddings = out.reshape(out.shape[0], out.shape[2], -1)

    if verbose:
        print(f"\nFinal embeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")

    return embeddings.cpu().numpy(), token_ids_list, text


def main():
    parser = argparse.ArgumentParser(description="Extract Klein embeddings using official flux2 method")
    parser.add_argument("--prompt", type=str, default="a cat sitting on a window sill", help="Text prompt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Qwen3 model to use")
    parser.add_argument("--output", type=str, default="klein_embeddings_python.npz", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--device", type=str, default="mps", help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 for comparison
        device_map=args.device,
    )
    model.eval()

    print(f"Model loaded on {args.device}")
    print(f"Model hidden size: {model.config.hidden_size}")
    print(f"Model num layers: {model.config.num_hidden_layers}")

    # Extract embeddings
    embeddings, token_ids, formatted_text = extract_klein_embeddings_official(
        args.prompt,
        model,
        tokenizer,
        verbose=args.verbose,
    )

    # Save results
    output_path = Path(args.output)
    np.savez(
        output_path,
        embeddings=embeddings,
        token_ids=np.array(token_ids),
        prompt=args.prompt,
        formatted_text=formatted_text,
        model=args.model,
        layers=np.array(OUTPUT_LAYERS_QWEN3),
        max_length=MAX_LENGTH,
    )

    print(f"\nSaved to {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Also save token IDs as JSON for easy comparison
    token_json_path = output_path.with_suffix(".tokens.json")
    with open(token_json_path, "w") as f:
        json.dump({
            "prompt": args.prompt,
            "formatted_text": formatted_text,
            "token_ids": token_ids,
            "pad_token_id": tokenizer.pad_token_id,
            "model": args.model,
        }, f, indent=2)
    print(f"Token IDs saved to {token_json_path}")


if __name__ == "__main__":
    main()
