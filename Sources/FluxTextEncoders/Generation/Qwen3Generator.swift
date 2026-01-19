/**
 * Qwen3Generator.swift
 * Text generation with streaming support for Qwen3 models
 *
 * Used for both text generation and as part of Klein embedding pipeline
 */

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

/// Text generator for Qwen3 models
public final class Qwen3Generator: @unchecked Sendable {
    private let model: Qwen3ForCausalLM
    private let tokenizer: Tokenizer

    // Qwen3 special tokens
    private let eosTokenId: Int
    private let padTokenId: Int
    private let imEndTokenId: Int

    public init(model: Qwen3ForCausalLM, tokenizer: Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer

        // Qwen3 special token IDs (standard for Qwen3 models)
        self.padTokenId = 151643      // <|endoftext|>
        self.eosTokenId = 151645      // <|im_end|>
        self.imEndTokenId = 151645    // <|im_end|>
    }

    /// Generate text from a prompt
    /// - Parameters:
    ///   - prompt: The user's prompt
    ///   - parameters: Generation parameters (temperature, topP, etc.)
    ///   - enableThinking: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)
    ///   - onToken: Optional callback for streaming tokens
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        enableThinking: Bool = false,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()

        // Set random seed if provided
        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Format with Qwen3 chat template (user message only)
        // For FLUX.2 usage, thinking is disabled by default
        let formattedPrompt = formatQwen3ChatTemplate(userMessage: prompt, enableThinking: enableThinking)

        // Tokenize
        let promptTokens = tokenizer.encode(text: formattedPrompt)
        var inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])

        // Create KV cache
        let cache = model.createCache()

        // Prefill
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)

        // Generation loop
        var generatedTokens: [Int] = []
        let hasCallback = onToken != nil

        // Token accumulation for batched streaming
        var pendingTokens: [Int] = []
        let streamBatchSize = 10

        for i in 0..<parameters.maxTokens {
            let lastLogits = logits[0, -1]

            let nextTokenArray: MLXArray
            if parameters.temperature == 0 {
                nextTokenArray = argMax(lastLogits)
            } else {
                nextTokenArray = sampleTopPGPU(
                    lastLogits,
                    temperature: parameters.temperature,
                    topP: parameters.topP
                )
            }

            eval(nextTokenArray)
            let nextToken = Int(nextTokenArray.item(Int32.self))

            // Check for EOS tokens
            if nextToken == eosTokenId || nextToken == padTokenId {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming
            if hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let tokenText = tokenizer.decode(tokens: pendingTokens)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)

            // Periodically clear GPU cache
            if (i + 1) % 20 == 0 {
                GPU.clearCache()
            }
        }

        // Flush remaining tokens
        if hasCallback && !pendingTokens.isEmpty {
            let tokenText = tokenizer.decode(tokens: pendingTokens)
            _ = onToken!(tokenText)
        }

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        var outputText = tokenizer.decode(tokens: generatedTokens)

        // Strip empty thinking tags when thinking is disabled
        if !enableThinking {
            outputText = stripEmptyThinkingTags(outputText)
        }

        // Clear KV cache to free memory
        cache.forEach { $0.clear() }
        MLX.GPU.clearCache()

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// Generate with chat messages (multi-turn conversation)
    /// - Parameters:
    ///   - messages: Array of message dictionaries with "role" and "content" keys
    ///   - parameters: Generation parameters
    ///   - enableThinking: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)
    ///   - onToken: Optional callback for streaming tokens
    public func chat(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        enableThinking: Bool = false,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()

        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Format with Qwen3 chat template
        let formattedPrompt = formatQwen3ChatMessages(messages: messages, enableThinking: enableThinking)

        // Tokenize
        let promptTokens = tokenizer.encode(text: formattedPrompt)
        var inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])

        let cache = model.createCache()

        // Prefill
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)

        // Generation loop
        var generatedTokens: [Int] = []
        let hasCallback = onToken != nil

        var pendingTokens: [Int] = []
        let streamBatchSize = 10

        for i in 0..<parameters.maxTokens {
            let lastLogits = logits[0, -1]

            let nextTokenArray: MLXArray
            if parameters.temperature == 0 {
                nextTokenArray = argMax(lastLogits)
            } else {
                nextTokenArray = sampleTopPGPU(
                    lastLogits,
                    temperature: parameters.temperature,
                    topP: parameters.topP
                )
            }

            eval(nextTokenArray)
            let nextToken = Int(nextTokenArray.item(Int32.self))

            if nextToken == eosTokenId || nextToken == padTokenId {
                break
            }

            generatedTokens.append(nextToken)

            if hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let tokenText = tokenizer.decode(tokens: pendingTokens)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)

            if (i + 1) % 20 == 0 {
                GPU.clearCache()
            }
        }

        if hasCallback && !pendingTokens.isEmpty {
            let tokenText = tokenizer.decode(tokens: pendingTokens)
            _ = onToken!(tokenText)
        }

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        var outputText = tokenizer.decode(tokens: generatedTokens)

        // Strip empty thinking tags when thinking is disabled
        if !enableThinking {
            outputText = stripEmptyThinkingTags(outputText)
        }

        cache.forEach { $0.clear() }
        MLX.GPU.clearCache()

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: promptTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// AsyncStream-based generation
    /// - Parameters:
    ///   - prompt: The user's prompt
    ///   - parameters: Generation parameters
    ///   - enableThinking: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)
    public func generateStream(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        enableThinking: Bool = false
    ) -> AsyncStream<String> {
        let generator = self
        let thinking = enableThinking
        return AsyncStream { continuation in
            Task { @Sendable in
                do {
                    _ = try generator.generate(prompt: prompt, parameters: parameters, enableThinking: thinking) { token in
                        continuation.yield(token)
                        return true
                    }
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Private Helpers

    /// Format a single user message using Qwen3 chat template
    /// - Parameters:
    ///   - userMessage: The user's message
    ///   - enableThinking: If false, appends /no_think to disable Qwen3 thinking mode
    private func formatQwen3ChatTemplate(userMessage: String, enableThinking: Bool = true) -> String {
        var prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += userMessage
        // Disable thinking mode for FLUX.2 and other direct-response use cases
        if !enableThinking {
            prompt += " /no_think"
        }
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    }

    /// Format multi-turn chat messages using Qwen3 chat template
    /// - Parameters:
    ///   - messages: Array of message dictionaries with "role" and "content" keys
    ///   - enableThinking: If false, appends /no_think to the last user message
    private func formatQwen3ChatMessages(messages: [[String: String]], enableThinking: Bool = true) -> String {
        var prompt = ""

        // Check if system message is included
        let hasSystemMessage = messages.first { $0["role"] == "system" } != nil

        // Add default system message if not provided
        if !hasSystemMessage {
            prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        }

        // Find the last user message index for /no_think insertion
        let lastUserIndex = messages.lastIndex { $0["role"] == "user" }

        for (index, message) in messages.enumerated() {
            guard let role = message["role"], let content = message["content"] else {
                continue
            }

            prompt += "<|im_start|>\(role)\n"
            prompt += content
            // Add /no_think to the last user message if thinking is disabled
            if !enableThinking && role == "user" && index == lastUserIndex {
                prompt += " /no_think"
            }
            prompt += "<|im_end|>\n"
        }

        // Add assistant prompt
        prompt += "<|im_start|>assistant\n"

        return prompt
    }

    /// GPU-optimized top-p (nucleus) sampling using MLX
    private func sampleTopPGPU(_ logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
        // Apply temperature
        let scaledLogits = logits / temperature

        // Softmax for probabilities
        let probs = softmax(scaledLogits, axis: -1)

        // Sort indices by probability (descending order)
        let sortedIndices = argSort(-probs, axis: -1)

        // Gather sorted probabilities
        let sortedProbs = probs[sortedIndices]

        // Compute cumulative probabilities
        let cumProbs = cumsum(sortedProbs, axis: -1)

        // Create mask: keep tokens where cumulative prob < topP
        let topPMask = cumProbs .< topP

        // Apply mask
        var maskedProbs = sortedProbs * topPMask.asType(sortedProbs.dtype)

        // Ensure at least the top token has non-zero probability
        maskedProbs = MLX.maximum(maskedProbs, sortedProbs * (cumProbs .<= sortedProbs).asType(sortedProbs.dtype))

        // Re-normalize
        let probSum = MLX.sum(maskedProbs, keepDims: true)
        let normalizedProbs = maskedProbs / (probSum + 1e-10)

        // Sample from categorical distribution
        let sampledIdx = MLXRandom.categorical(MLX.log(normalizedProbs + 1e-10))

        // Map back to original vocabulary index
        return sortedIndices[sampledIdx]
    }

    /// Strip empty thinking tags from output text
    /// Qwen3 may still output <think></think> even with /no_think flag
    private func stripEmptyThinkingTags(_ text: String) -> String {
        // Remove <think>\n</think>\n pattern (with optional whitespace)
        var result = text
        // Handle various whitespace patterns
        result = result.replacingOccurrences(
            of: "<think>\\s*</think>\\s*",
            with: "",
            options: .regularExpression
        )
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
