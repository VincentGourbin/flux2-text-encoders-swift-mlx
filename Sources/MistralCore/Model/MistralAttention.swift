/**
 * MistralAttention.swift
 * Multi-Head Attention with Grouped Query Attention (GQA) and RoPE
 */

import Foundation
import MLX
import MLXNN

// MARK: - Llama-4 Attention Scaling

/// Llama-4 attention scaling for long sequences
/// Formula: 1 + beta * log(1 + floor(position / max_position_embeddings))
/// This helps the model handle very long sequences by scaling queries based on position
func getLlama4AttentionScale(
    start: Int,
    stop: Int,
    beta: Float,
    maxPositionEmbeddings: Int
) -> MLXArray {
    // Create position array: [start, start+1, ..., stop-1]
    let positions = MLXArray(Array(start..<stop).map { Float($0) })
    let maxPos = Float(maxPositionEmbeddings)

    // scaling = 1 + beta * log(1 + floor(pos / max_pos))
    let floored = MLX.floor(positions / maxPos)
    let scaling = 1.0 + beta * MLX.log(1.0 + floored)

    // Reshape to [seq_len, 1] for broadcasting
    return scaling.reshaped([stop - start, 1])
}

// MARK: - Scaled Dot-Product Attention

/// Manual scaled dot-product attention implementation
func scaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXArray? = nil
) -> MLXArray {
    // queries, keys, values: [batch, heads, seq, head_dim]
    // Compute attention scores: Q @ K^T
    let scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * MLXArray([scale])

    // Apply mask if provided
    var maskedScores = scores
    if let mask = mask {
        maskedScores = scores + mask
    }

    // Softmax over last dimension
    let weights = softmax(maskedScores, axis: -1)

    // Apply attention to values
    return matmul(weights, values)
}

/// KV Cache for efficient generation
public class KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int = 0

    public init() {}

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = self.keys, let existingValues = self.values {
            self.keys = concatenated([existingKeys, keys], axis: 2)
            self.values = concatenated([existingValues, values], axis: 2)
        } else {
            self.keys = keys
            self.values = values
        }
        self.offset = self.keys!.shape[2]
        return (self.keys!, self.values!)
    }

    public var length: Int {
        return keys?.shape[2] ?? 0
    }

    /// Clear the cache to free memory
    public func clear() {
        keys = nil
        values = nil
        offset = 0
    }
}

/// Rotary Position Embedding - wraps MLXFast.RoPE for optimal performance
public class MistralRoPE: Module {
    let dimensions: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    public init(dimensions: Int, traditional: Bool = false, base: Float = 10000.0, scale: Float = 1.0) {
        self.dimensions = dimensions
        self.traditional = traditional
        self.base = base
        self.scale = scale
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Use the optimized MLXFast.RoPE implementation (same as Python mx.fast.rope)
        let shape = x.shape
        var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
        x = MLXFast.RoPE(
            x, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
            offset: offset)
        return x.reshaped(shape)
    }
}

/// Mistral Attention with Grouped Query Attention (GQA)
public class MistralAttention: Module {
    let config: MistralTextConfig
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo public var q_proj: Linear
    @ModuleInfo public var k_proj: Linear
    @ModuleInfo public var v_proj: Linear
    @ModuleInfo public var o_proj: Linear
    public var rope: MistralRoPE

    public init(config: MistralTextConfig) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        // Projections
        self._q_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numHeads * headDim, bias: config.attentionBias))
        self._k_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        self._v_proj = ModuleInfo(wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: config.attentionBias))
        self._o_proj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, hiddenSize, bias: config.attentionBias))

        // RoPE - using MLXFast.RoPE for numerical consistency with Python
        self.rope = MistralRoPE(dimensions: headDim, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.shape[0]
        let seqLen = hiddenStates.shape[1]

        // Project Q, K, V
        var queries = q_proj(hiddenStates)
        var keys = k_proj(hiddenStates)
        var values = v_proj(hiddenStates)

        // Reshape for multi-head attention
        queries = queries.reshaped([batchSize, seqLen, numHeads, headDim])
        keys = keys.reshaped([batchSize, seqLen, numKVHeads, headDim])
        values = values.reshaped([batchSize, seqLen, numKVHeads, headDim])

        // Transpose to [batch, heads, seq, head_dim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE
        let offset = cache?.length ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Apply Llama-4 attention scaling to queries (CRITICAL for Ministral3!)
        // This scales queries based on position to handle long sequences
        let attnScale = getLlama4AttentionScale(
            start: offset,
            stop: offset + seqLen,
            beta: config.llama4ScalingBeta,
            maxPositionEmbeddings: config.originalMaxPositionEmbeddings
        )
        // Reshape for broadcasting: [seq_len, 1] -> [1, 1, seq_len, 1]
        queries = queries * attnScale.reshaped([1, 1, seqLen, 1])

        // Update KV cache if provided
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Expand KV heads for GQA (repeat KV heads to match Q heads)
        let repeatFactor = numHeads / numKVHeads
        if repeatFactor > 1 {
            keys = MLX.repeated(keys, count: repeatFactor, axis: 1)
            values = MLX.repeated(values, count: repeatFactor, axis: 1)
        }

        // Scaled dot-product attention using MLXFast for optimal performance
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let outputTransposed = output.transposed(0, 2, 1, 3)
        let outputReshaped = outputTransposed.reshaped([batchSize, seqLen, numHeads * headDim])

        return o_proj(outputReshaped)
    }
}
