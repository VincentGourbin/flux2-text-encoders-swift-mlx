/**
 * MistralGenerator.swift
 * Text generation with streaming support for Mistral models
 */

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Parameters for text generation
public struct GenerateParameters: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var repetitionContextSize: Int
    public var seed: UInt64?

    /// Maximum context length supported by Mistral Small 3.2 (131K tokens)
    public static let maxContextLength = 131_072

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.7,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 20,
        seed: UInt64? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.seed = seed
    }

    /// Greedy decoding (temperature = 0)
    public static let greedy = GenerateParameters(
        maxTokens: 2048,
        temperature: 0.0,
        topP: 1.0,
        repetitionPenalty: 1.0
    )

    /// Creative generation
    public static let creative = GenerateParameters(
        maxTokens: 4096,
        temperature: 0.9,
        topP: 0.95,
        repetitionPenalty: 1.2
    )

    /// Balanced generation
    public static let balanced = GenerateParameters(
        maxTokens: 2048,
        temperature: 0.7,
        topP: 0.9,
        repetitionPenalty: 1.1
    )
}

/// Generator result
public struct GenerationResult: Sendable {
    public let text: String
    public let tokens: [Int]
    public let promptTokens: Int
    public let generatedTokens: Int
    public let totalTime: Double
    public let tokensPerSecond: Double

    public func summary() -> String {
        return """
        Prompt: \(promptTokens) tokens
        Generated: \(generatedTokens) tokens
        Speed: \(String(format: "%.1f", tokensPerSecond)) tokens/s
        Time: \(String(format: "%.2f", totalTime))s
        """
    }
}

/// Text generator for Mistral models
public final class MistralGenerator: @unchecked Sendable {
    private let model: MistralForCausalLM
    private let tokenizer: TekkenTokenizer

    public init(model: MistralForCausalLM, tokenizer: TekkenTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Generate text with streaming callback
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()
        let profiler = FluxProfiler.shared

        // Set random seed if provided
        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Tokenization with profiling
        profiler.startTokenization()
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let promptTokens = tokenizer.encodeChatMessages(messages: messages, addGenerationPrompt: true)
        var inputIds = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        profiler.endTokenization(tokenCount: promptTokens.count)

        // Create KV cache
        let cache = model.createCache()

        // Prefill with profiling
        profiler.startPrefill()
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)
        profiler.endPrefill()

        // Generation loop - optimized for minimal CPU-GPU sync
        profiler.startGeneration()
        var generatedTokens: [Int] = []
        let eosToken = tokenizer.eosToken
        let hasCallback = onToken != nil

        // Token accumulation for batched streaming (reduces I/O overhead)
        var pendingTokens: [Int] = []
        let streamBatchSize = 10  // Accumulate up to 10 tokens before streaming

        for i in 0..<parameters.maxTokens {
            // Sample next token from last position logits - stays on GPU
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

            // Single sync to get token value
            eval(nextTokenArray)
            let nextToken = Int(nextTokenArray.item(Int32.self))

            // Check for EOS
            if nextToken == eosToken {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming: accumulate tokens then flush
            if hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let decodeStart = CFAbsoluteTimeGetCurrent()
                    let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
                    profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            // Forward pass - lazy eval, will be sync'd on next loop iteration
            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)

            // Periodically clear GPU cache to prevent memory accumulation
            if (i + 1) % 20 == 0 {
                GPU.clearCache()
            }
        }

        // Flush any remaining tokens
        if hasCallback && !pendingTokens.isEmpty {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
            profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
            _ = onToken!(tokenText)
        }

        profiler.endGeneration(tokenCount: generatedTokens.count)

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        let decodeStart = CFAbsoluteTimeGetCurrent()
        let outputText = tokenizer.decode(generatedTokens, skipSpecialTokens: true)
        profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)

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

    /// Generate with chat template
    public func chat(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        let startTime = Date()
        let profiler = FluxProfiler.shared

        if let seed = parameters.seed {
            MLXRandom.seed(seed)
        }

        // Tokenization with profiling
        profiler.startTokenization()
        let promptTokens = tokenizer.encodeChatMessages(messages: messages, addGenerationPrompt: true)
        var inputIds = MLXArray(promptTokens).reshaped([1, promptTokens.count])
        profiler.endTokenization(tokenCount: promptTokens.count)

        let cache = model.createCache()

        // Prefill with profiling
        profiler.startPrefill()
        var logits = model.forward(inputIds, cache: cache)
        eval(logits)
        profiler.endPrefill()

        // Generation loop - optimized with batched streaming
        profiler.startGeneration()
        var generatedTokens: [Int] = []
        let eosToken = tokenizer.eosToken
        let hasCallback = onToken != nil

        // Token accumulation for batched streaming (reduces I/O overhead)
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

            if nextToken == eosToken {
                break
            }

            generatedTokens.append(nextToken)

            // Batched streaming
            if hasCallback {
                pendingTokens.append(nextToken)
                if pendingTokens.count >= streamBatchSize {
                    let decodeStart = CFAbsoluteTimeGetCurrent()
                    let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
                    profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
                    if !onToken!(tokenText) {
                        break
                    }
                    pendingTokens.removeAll()
                }
            }

            inputIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])
            logits = model.forward(inputIds, cache: cache)

            // Periodically clear GPU cache to prevent memory accumulation
            if (i + 1) % 20 == 0 {
                GPU.clearCache()
            }
        }

        // Flush remaining pending tokens
        if hasCallback && !pendingTokens.isEmpty {
            let decodeStart = CFAbsoluteTimeGetCurrent()
            let tokenText = tokenizer.decode(pendingTokens, skipSpecialTokens: true)
            profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)
            _ = onToken!(tokenText)
        }

        profiler.endGeneration(tokenCount: generatedTokens.count)

        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / totalTime

        let decodeStart = CFAbsoluteTimeGetCurrent()
        let outputText = tokenizer.decode(generatedTokens, skipSpecialTokens: true)
        profiler.addDecodingTime(CFAbsoluteTimeGetCurrent() - decodeStart)

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

    /// AsyncStream-based generation for async/await usage
    public func generateStream(
        prompt: String,
        parameters: GenerateParameters = .balanced
    ) -> AsyncStream<String> {
        let generator = self
        return AsyncStream { continuation in
            Task { @Sendable in
                do {
                    _ = try generator.generate(prompt: prompt, parameters: parameters) { token in
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

        // Create mask: keep tokens where cumulative prob < topP (shifted by one)
        // Include the first token that crosses topP threshold
        let topPMask = cumProbs .< topP

        // Shift mask to include one more token (the one that crosses threshold)
        // By using < instead of <= and ensuring at least one token
        var maskedProbs = sortedProbs * topPMask.asType(sortedProbs.dtype)

        // Ensure at least the top token has non-zero probability
        maskedProbs = MLX.maximum(maskedProbs, sortedProbs * (cumProbs .<= sortedProbs).asType(sortedProbs.dtype))

        // Re-normalize
        let probSum = MLX.sum(maskedProbs, keepDims: true)
        let normalizedProbs = maskedProbs / (probSum + 1e-10)

        // Sample from categorical distribution (expects logits/log-probs)
        let sampledIdx = MLXRandom.categorical(MLX.log(normalizedProbs + 1e-10))

        // Map back to original vocabulary index
        return sortedIndices[sampledIdx]
    }
}
