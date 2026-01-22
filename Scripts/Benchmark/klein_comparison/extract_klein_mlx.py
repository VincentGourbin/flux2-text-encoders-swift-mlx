#!/usr/bin/env python3
"""
Klein embedding extraction using MLX Python with the same model as Swift.
This allows fair comparison since both use the same quantized weights.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

# Official flux2 constants
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512


def format_qwen3_chat_template(user_message: str, add_generation_prompt: bool = True) -> str:
    """
    Format prompt using Qwen3 chat template (matches official flux2).
    NO system message, includes <think> tokens for enable_thinking=False.
    """
    prompt = ""

    # User message (NO system message)
    prompt += "<|im_start|>user\n"
    prompt += user_message
    prompt += "<|im_end|>\n"

    # Assistant prompt with thinking tokens
    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
        prompt += "<think>\n\n</think>\n\n"

    return prompt


def extract_hidden_states(model, input_ids, attention_mask, layer_indices):
    """
    Extract hidden states from specific layers of Qwen3 model.
    """
    # Get the base model (inside the lm_head wrapper)
    base_model = model.model

    # Embedding
    hidden_states = base_model.embed_tokens(input_ids)
    model_dtype = hidden_states.dtype

    # Create causal mask
    seq_len = input_ids.shape[1]

    # Simple causal mask - use the model's dtype
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(model_dtype)

    # Combine with attention mask for padding
    if attention_mask is not None:
        # Convert attention mask to additive mask
        # attention_mask: 1 for real tokens, 0 for padding
        # We need to mask padding positions with -inf
        padding_mask = (1 - attention_mask.astype(model_dtype)) * mx.array(-1e9, dtype=model_dtype)
        padding_mask = padding_mask.reshape(input_ids.shape[0], 1, 1, seq_len)
        mask = mask + padding_mask

    # Collect hidden states at specified indices
    all_hidden_states = [hidden_states]  # Index 0 is embedding

    # Process through layers
    for i, layer in enumerate(base_model.layers):
        hidden_states = layer(hidden_states, mask=mask)
        all_hidden_states.append(hidden_states)

    # Final normalization
    hidden_states = base_model.norm(hidden_states)
    all_hidden_states.append(hidden_states)

    # Extract requested layers
    extracted = [all_hidden_states[idx] for idx in layer_indices]

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract Klein embeddings using MLX Python")
    parser.add_argument("--prompt", type=str, default="a cat sitting on a window sill", help="Text prompt")
    parser.add_argument("--model", type=str, default="lmstudio-community/Qwen3-4B-MLX-8bit", help="MLX model to use")
    parser.add_argument("--output", type=str, default="klein_embeddings_mlx_python.npz", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print(f"Loading MLX model: {args.model}")

    # Load model and tokenizer using mlx-lm
    model, tokenizer = load(args.model)

    print(f"Model loaded successfully")
    print(f"Model config: hidden_size={model.model.embed_tokens.weight.shape[1]}")
    print(f"Num layers: {len(model.model.layers)}")

    # Format prompt using Qwen3 chat template (NO system message)
    formatted_text = format_qwen3_chat_template(args.prompt, add_generation_prompt=True)

    if args.verbose:
        print(f"\nFormatted text:\n{formatted_text}")

    # Tokenize
    token_ids = tokenizer.encode(formatted_text)

    print(f"\nTokens before padding: {len(token_ids)}")
    if args.verbose:
        print(f"First 20 tokens: {token_ids[:20]}")

    # Get pad token ID
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Fallback
    print(f"Pad token ID: {pad_token_id}")

    # RIGHT-pad to MAX_LENGTH
    original_length = len(token_ids)
    if len(token_ids) < MAX_LENGTH:
        pad_count = MAX_LENGTH - len(token_ids)
        token_ids = token_ids + [pad_token_id] * pad_count
    elif len(token_ids) > MAX_LENGTH:
        token_ids = token_ids[:MAX_LENGTH]
        original_length = MAX_LENGTH

    print(f"Tokens after padding: {len(token_ids)}")
    print(f"Non-padding tokens: {original_length}")

    # Create input tensors
    input_ids = mx.array([token_ids])

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask_list = [1] * original_length + [0] * (MAX_LENGTH - original_length)
    attention_mask = mx.array([attention_mask_list])

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Extract hidden states
    print(f"\nExtracting hidden states from layers {OUTPUT_LAYERS_QWEN3}...")
    hidden_states_list = extract_hidden_states(
        model, input_ids, attention_mask, OUTPUT_LAYERS_QWEN3
    )

    # Print shapes
    for i, (layer_idx, hs) in enumerate(zip(OUTPUT_LAYERS_QWEN3, hidden_states_list)):
        print(f"Layer {layer_idx} shape: {hs.shape}")

    # Concatenate along hidden dimension
    embeddings = mx.concatenate(hidden_states_list, axis=-1)
    print(f"\nFinal embeddings shape: {embeddings.shape}")

    # Convert to float32 and evaluate
    embeddings = embeddings.astype(mx.float32)
    mx.eval(embeddings)

    # Convert to numpy
    embeddings_np = np.array(embeddings)

    # Print some values for comparison
    print(f"\nFirst 10 values at position 0: {embeddings_np[0, 0, :10]}")
    print(f"First 10 values at position 18: {embeddings_np[0, 18, :10]}")
    print(f"First 10 values at position 511: {embeddings_np[0, 511, :10]}")

    # Save results
    output_path = Path(args.output)
    np.savez(
        output_path,
        embeddings=embeddings_np,
        token_ids=np.array(token_ids),
        prompt=args.prompt,
        formatted_text=formatted_text,
        model=args.model,
        layers=np.array(OUTPUT_LAYERS_QWEN3),
        max_length=MAX_LENGTH,
    )

    print(f"\nSaved to {output_path}")

    # Also save token IDs as JSON
    token_json_path = output_path.with_suffix(".tokens.json")
    with open(token_json_path, "w") as f:
        json.dump({
            "prompt": args.prompt,
            "formatted_text": formatted_text,
            "token_ids": token_ids,
            "pad_token_id": int(pad_token_id),
            "model": args.model,
            "original_length": original_length,
        }, f, indent=2)
    print(f"Token IDs saved to {token_json_path}")


if __name__ == "__main__":
    main()
