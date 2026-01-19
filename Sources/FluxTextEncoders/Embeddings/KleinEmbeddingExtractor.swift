/**
 * KleinEmbeddingExtractor.swift
 * Embedding extraction for FLUX.2 Klein using Qwen3 models
 *
 * Klein uses Qwen3 (4B or 8B) as text encoder instead of Mistral
 */

import Foundation
import MLX
import MLXNN
import Tokenizers

// MARK: - Klein Embedding Extractor

/// Extracts embeddings from Qwen3 model hidden states for FLUX.2 Klein
public class KleinEmbeddingExtractor {
    private let model: Qwen3ForCausalLM
    private let tokenizer: Tokenizer
    private let variant: KleinVariant
    
    // Qwen3 special tokens
    private let padTokenId: Int
    private let imStartTokenId: Int   // <|im_start|>
    private let imEndTokenId: Int     // <|im_end|>
    
    public init(model: Qwen3ForCausalLM, tokenizer: Tokenizer, variant: KleinVariant) {
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant
        
        // Qwen3 special token IDs
        // These are standard for Qwen3 models
        self.padTokenId = 151643      // <|endoftext|> used as pad
        self.imStartTokenId = 151644  // <|im_start|>
        self.imEndTokenId = 151645    // <|im_end|>
    }
    
    /// Extract Klein embeddings from a text prompt
    /// - Parameters:
    ///   - prompt: User prompt text
    ///   - maxLength: Maximum sequence length (default: 512)
    /// - Returns: Embeddings tensor with shape [1, maxLength, outputDim]
    ///           Klein 4B: [1, 512, 7680]
    ///           Klein 9B: [1, 512, 12288]
    public func extractKleinEmbeddings(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> MLXArray {
        // 1. Build messages with system message
        let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
        
        // 2. Apply Qwen3 chat template manually
        // Qwen3 format:
        // <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
        let formattedPrompt = formatQwen3ChatTemplate(
            systemMessage: KleinConfig.systemMessage,
            userMessage: cleanedPrompt,
            addGenerationPrompt: true  // Matches Python behavior
        )
        
        // 3. Tokenize
        var tokenIds = tokenizer.encode(text: formattedPrompt)
        
        FluxDebug.log("Klein embeddings: encoded \(tokenIds.count) tokens before padding")
        
        // 4. Truncate if needed
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
            FluxDebug.log("Klein embeddings: truncated to \(maxLength) tokens")
        }
        
        // 5. LEFT-pad to fixed length (matching Python mflux behavior)
        let originalLength = tokenIds.count
        let padCount: Int
        if tokenIds.count < maxLength {
            padCount = maxLength - tokenIds.count
            let padding = Array(repeating: padTokenId, count: padCount)
            tokenIds = padding + tokenIds
        } else {
            padCount = 0
        }
        
        FluxDebug.log("Klein embeddings: padded from \(originalLength) to \(tokenIds.count) tokens (pad count: \(padCount))")
        
        // 6. Create input tensor
        let inputIds = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, tokenIds.count])
        
        // 7. Create attention mask (1 for real tokens, 0 for padding)
        var attentionMaskValues = Array(repeating: Int32(0), count: padCount)
        attentionMaskValues.append(contentsOf: Array(repeating: Int32(1), count: originalLength))
        let attentionMask = MLXArray(attentionMaskValues).reshaped([1, maxLength])
        
        FluxDebug.log("Klein embeddings: attention mask created with \(padCount) masked positions")
        
        // 8. Forward pass with hidden states extraction
        let output = model(inputIds, outputHiddenStates: true, attentionMask: attentionMask)
        
        guard let allHiddenStates = output.hiddenStates else {
            throw KleinEmbeddingError.noHiddenStates
        }
        
        // 9. Extract hidden states from Klein layers [9, 18, 27]
        // Note: hidden_states includes embedding layer at index 0, so layer 9 is at index 9
        var extractedStates: [MLXArray] = []
        for layerIdx in variant.hiddenStateLayers {
            guard layerIdx >= 0 && layerIdx < allHiddenStates.count else {
                throw KleinEmbeddingError.invalidLayerIndex(layerIdx, allHiddenStates.count)
            }
            extractedStates.append(allHiddenStates[layerIdx])
        }
        
        // 10. Concatenate along hidden dimension: [1, seq, hidden] x 3 -> [1, seq, hidden*3]
        let embeddings = concatenated(extractedStates, axis: -1)
        
        // 11. Evaluate to ensure computation is complete
        eval(embeddings)
        
        FluxDebug.log("Klein embeddings: shape \(embeddings.shape)")
        
        return embeddings
    }
    
    /// Format prompt using Qwen3 chat template
    /// - Parameters:
    ///   - systemMessage: System message content
    ///   - userMessage: User message content
    ///   - addGenerationPrompt: Whether to add assistant prompt prefix
    /// - Returns: Formatted prompt string
    private func formatQwen3ChatTemplate(
        systemMessage: String,
        userMessage: String,
        addGenerationPrompt: Bool
    ) -> String {
        var prompt = ""
        
        // System message
        prompt += "<|im_start|>system\n"
        prompt += systemMessage
        prompt += "<|im_end|>\n"
        
        // User message
        prompt += "<|im_start|>user\n"
        prompt += userMessage
        prompt += "<|im_end|>\n"
        
        // Assistant prompt (if requested)
        if addGenerationPrompt {
            prompt += "<|im_start|>assistant\n"
        }
        
        return prompt
    }
    
    /// Get token IDs for Klein format (useful for debugging/comparison)
    public func getKleinTokenIds(
        prompt: String,
        maxLength: Int = KleinConfig.maxSequenceLength
    ) throws -> [Int] {
        let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
        let formattedPrompt = formatQwen3ChatTemplate(
            systemMessage: KleinConfig.systemMessage,
            userMessage: cleanedPrompt,
            addGenerationPrompt: true
        )
        
        var tokenIds = tokenizer.encode(text: formattedPrompt)
        
        // Truncate if needed
        if tokenIds.count > maxLength {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }
        
        // LEFT-pad to fixed length
        if tokenIds.count < maxLength {
            let padCount = maxLength - tokenIds.count
            let padding = Array(repeating: padTokenId, count: padCount)
            tokenIds = padding + tokenIds
        }
        
        return tokenIds
    }
    
    /// Get the variant this extractor is configured for
    public var kleinVariant: KleinVariant {
        return variant
    }
    
    /// Get embedding dimension for this variant
    public var embeddingDimension: Int {
        return variant.outputDimension
    }
}

// MARK: - Errors

public enum KleinEmbeddingError: LocalizedError {
    case noHiddenStates
    case invalidLayerIndex(Int, Int)
    case tokenizerError(String)
    case modelNotLoaded
    
    public var errorDescription: String? {
        switch self {
        case .noHiddenStates:
            return "Qwen3 model did not return hidden states"
        case .invalidLayerIndex(let idx, let max):
            return "Invalid layer index \(idx), Qwen3 model has \(max) layers"
        case .tokenizerError(let message):
            return "Qwen3 tokenizer error: \(message)"
        case .modelNotLoaded:
            return "Klein model not loaded"
        }
    }
}
