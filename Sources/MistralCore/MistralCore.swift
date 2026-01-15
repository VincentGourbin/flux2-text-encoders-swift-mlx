/**
 * MistralCore.swift
 * Main entry point for MistralCore library
 *
 * Mistral Small 3.2 Swift MLX - Inference library for Apple Silicon
 */

import Foundation
import MLX

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Public API

/// Main interface for Mistral model operations
/// Thread-safe: load/unload on main thread, inference can run on any thread
public final class MistralCore: @unchecked Sendable {
    /// Shared singleton instance
    public static let shared = MistralCore()
    public static let version = "0.1.0"

    private var model: MistralForCausalLM?
    private var vlmModel: MistralVLM?
    private var tokenizer: TekkenTokenizer?
    private var generator: MistralGenerator?
    private var extractor: EmbeddingExtractor?
    private var imageProcessor: ImageProcessor?

    /// CoreML encoder for ANE-accelerated FLUX.2 embeddings
    @available(macOS 14.0, *)
    private var coremlEncoder: MistralCoreMLEncoder?

    /// Chunked CoreML encoder for ANE-accelerated FLUX.2 embeddings (for large models)
    @available(macOS 14.0, *)
    private var coremlChunkedEncoder: MistralCoreMLChunkedEncoder?

    /// Whether VLM (vision) model is loaded
    public var isVLMLoaded: Bool {
        return vlmModel != nil && tokenizer != nil && imageProcessor != nil
    }

    private init() {}

    /// Check if model is loaded
    public var isModelLoaded: Bool {
        return model != nil && tokenizer != nil
    }

    /// Load model from path or download if needed
    @MainActor
    public func loadModel(
        variant: ModelVariant = .mlx8bit,
        hfToken: String? = nil,
        progress: DownloadProgressCallback? = nil
    ) async throws {
        let downloader = ModelDownloader(hfToken: hfToken)
        let modelPath = try await downloader.download(variant: variant, progress: progress)

        try loadModel(from: modelPath.path)
    }

    /// Load model from local path
    @MainActor
    public func loadModel(from path: String) throws {
        MistralDebug.log("Loading model from \(path)")

        // Load tokenizer
        tokenizer = TekkenTokenizer(modelPath: path)

        // Load model
        model = try MistralForCausalLM.load(from: path)

        // Create generator and extractor
        if let model = model, let tokenizer = tokenizer {
            generator = MistralGenerator(model: model, tokenizer: tokenizer)
            extractor = EmbeddingExtractor(model: model, tokenizer: tokenizer)
        }

        MistralDebug.log("Model loaded successfully")
    }

    /// Load VLM (vision-language) model from path
    @MainActor
    public func loadVLMModel(from path: String) throws {
        let debug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

        if debug { print("[Core] Loading VLM from \(path)"); fflush(stdout) }

        // Load tokenizer
        tokenizer = TekkenTokenizer(modelPath: path)

        // Load VLM model
        vlmModel = try MistralVLM.load(from: path)

        // Initialize image processor
        imageProcessor = ImageProcessor(config: .pixtral)

        // Also set up text-only generator using the language model
        if let vlm = vlmModel, let tokenizer = tokenizer {
            generator = MistralGenerator(model: vlm.languageModel, tokenizer: tokenizer)
            extractor = EmbeddingExtractor(model: vlm.languageModel, tokenizer: tokenizer)
            model = vlm.languageModel
        }

        if debug { print("[Core] VLM loading complete!"); fflush(stdout) }
    }

    /// Load VLM from path or download if needed
    @MainActor
    public func loadVLMModel(
        variant: ModelVariant = .mlx4bit,
        hfToken: String? = nil,
        progress: DownloadProgressCallback? = nil
    ) async throws {
        let downloader = ModelDownloader(hfToken: hfToken)
        let modelPath = try await downloader.download(variant: variant, progress: progress)

        try loadVLMModel(from: modelPath.path)
    }

    /// Unload model to free memory
    @MainActor
    public func unloadModel() {
        model = nil
        vlmModel = nil
        tokenizer = nil
        generator = nil
        extractor = nil
        imageProcessor = nil
        if #available(macOS 14.0, *) {
            coremlEncoder = nil
        }
        MLX.GPU.clearCache()
        MistralDebug.log("Model unloaded")
    }

    // MARK: - Generation

    /// Generate text from prompt
    public func generate(
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = generator else {
            throw MistralCoreError.modelNotLoaded
        }
        return try generator.generate(prompt: prompt, parameters: parameters, onToken: onToken)
    }

    /// Generate with chat messages
    public func chat(
        messages: [[String: String]],
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let generator = generator else {
            throw MistralCoreError.modelNotLoaded
        }
        return try generator.chat(messages: messages, parameters: parameters, onToken: onToken)
    }

    // MARK: - Vision

    /// Analyze an image with a text prompt
    /// - Parameters:
    ///   - image: NSImage to analyze
    ///   - prompt: Text prompt describing what to look for
    ///   - parameters: Generation parameters
    ///   - onToken: Callback for streaming tokens
    /// - Returns: Generated description/analysis
    /// Log memory for inference debugging (only when detailed profiling is enabled)
    private func logInferenceMemory(_ label: String) {
        guard MistralProfiler.shared.isEnabled || ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil else { return }
        let snapshot = MistralProfiler.snapshot()
        let mlxMB = Double(snapshot.mlxActive) / (1024 * 1024)
        let procMB = Double(snapshot.processFootprint) / (1024 * 1024)
        print("[VLM-INF] \(label): MLX=\(String(format: "%.1f", mlxMB))MB, Process=\(String(format: "%.1f", procMB))MB")
        fflush(stdout)
    }

    public func analyzeImage(
        image: NSImage,
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let vlm = vlmModel,
              let tokenizer = tokenizer,
              let processor = imageProcessor else {
            throw MistralCoreError.vlmNotLoaded
        }

        let debug = ProcessInfo.processInfo.environment["VLM_DEBUG"] != nil

        if debug { print("[Analyze] Starting with prompt: \(prompt)"); fflush(stdout) }
        logInferenceMemory("START inference")

        // 1. Preprocess image
        let pixelValues = try processor.preprocess(image)
        logInferenceMemory("After image preprocess")

        // 2. Encode image to get number of image tokens
        // NHWC format: [batch, H, W, C]
        let (_, patchesH, patchesW) = vlm.encodeImage(pixelValues)
        if debug { print("[Analyze] Image encoded: \(patchesH)x\(patchesW) patches"); fflush(stdout) }
        logInferenceMemory("After image encode (vision tower)")
        let numImageTokens = vlm.getNumImageTokens(
            imageHeight: pixelValues.shape[1],
            imageWidth: pixelValues.shape[2]
        )

        // 3. Build input tokens with image token placeholders
        // IMPORTANT: We must insert actual image token IDs (10), not tokenize "[IMG]" string!
        // Format: <s> [INST] [IMG][IMG]...[IMG] {user_prompt} [/INST]
        let imageTokenId = vlm.config.imageTokenIndex  // = 10

        // Build tokens directly:
        // - BOS token (1)
        // - [INST] token (3)
        // - numImageTokens x image token (10)
        // - tokenized user prompt
        // - [/INST] token (4)
        var inputTokens: [Int] = []
        inputTokens.append(tokenizer.bosToken)  // <s>
        inputTokens.append(3)  // [INST]
        inputTokens.append(contentsOf: Array(repeating: imageTokenId, count: numImageTokens))
        inputTokens.append(contentsOf: tokenizer.encode("\n\(prompt) ", addSpecialTokens: false))
        inputTokens.append(4)  // [/INST]

        if debug {
            print("[Analyze] Input tokens: \(inputTokens.count) total (\(numImageTokens) image tokens)")
            print("[Analyze] First 10 tokens: \(inputTokens.prefix(10))")
            print("[Analyze] Last 10 tokens: \(inputTokens.suffix(10))")
            fflush(stdout)
        }

        let inputIds = MLXArray(inputTokens.map { Int32($0) }).expandedDimensions(axis: 0)

        // 5. Generate with vision
        let cache = vlm.createCache()
        logInferenceMemory("After KV cache creation")
        var generatedTokens: [Int] = []
        var outputText = ""
        let maxTokens = parameters.maxTokens
        let startTime = Date()

        // First forward pass with image
        var logits = vlm(inputIds, pixelValues: pixelValues, cache: cache)
        logInferenceMemory("After first forward pass (prefill)")

        if debug {
            // Debug: Check logits stats
            print("[Debug] Logits shape: \(logits.shape)")
            let lastLogits = logits[0, -1, 0...]
            let logitsMean = MLX.mean(lastLogits).item(Float.self)
            let logitsStd = MLX.std(lastLogits).item(Float.self)
            let logitsMin = MLX.min(lastLogits).item(Float.self)
            let logitsMax = MLX.max(lastLogits).item(Float.self)
            print("[Debug] Last position logits: mean=\(logitsMean), std=\(logitsStd), min=\(logitsMin), max=\(logitsMax)")
            // Check top predictions
            let sortedIndices = MLX.argSort(lastLogits)
            let vocabSize = lastLogits.shape[0]
            let topK = min(5, vocabSize)
            let topIndices = sortedIndices[(vocabSize - topK)...]
            print("[Debug] Top \(topK) token indices: \(topIndices.asArray(Int32.self))")
            fflush(stdout)
        }

        for i in 0..<maxTokens {
            // Sample next token
            let nextTokenLogits = logits[0, -1, 0...]
            let nextToken = sampleToken(logits: nextTokenLogits, parameters: parameters)

            // Force evaluation before sync - allows GPU work to complete
            MLX.eval(nextToken)
            let tokenId = nextToken.item(Int32.self)

            // Check for EOS
            if tokenId == Int32(tokenizer.eosToken) {
                break
            }

            generatedTokens.append(Int(tokenId))

            // Decode and stream
            let tokenText = tokenizer.decode([Int(tokenId)], skipSpecialTokens: true)
            outputText += tokenText

            if let callback = onToken {
                if !callback(tokenText) {
                    break
                }
            }

            // Next forward pass (text only, using cache)
            let nextInput = MLXArray([tokenId]).expandedDimensions(axis: 0)
            logits = vlm(nextInput, pixelValues: nil, cache: cache)

            // Periodically clear GPU cache to prevent memory accumulation
            // Every 20 tokens, clear temporary buffers (keeps KV cache intact)
            if (i + 1) % 20 == 0 {
                MLX.GPU.clearCache()
            }
        }

        let totalTime = Date().timeIntervalSince(startTime)
        let tokensPerSecond = Double(generatedTokens.count) / max(totalTime, 0.001)
        logInferenceMemory("After generation loop (\(generatedTokens.count) tokens)")

        // Clear KV cache to free memory
        cache.forEach { $0.clear() }
        logInferenceMemory("After KV cache clear")
        MLX.GPU.clearCache()
        logInferenceMemory("After GPU cache clear")

        return GenerationResult(
            text: outputText,
            tokens: generatedTokens,
            promptTokens: inputTokens.count,
            generatedTokens: generatedTokens.count,
            totalTime: totalTime,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// Analyze image from file path
    public func analyzeImage(
        path: String,
        prompt: String,
        parameters: GenerateParameters = .balanced,
        onToken: ((String) -> Bool)? = nil
    ) throws -> GenerationResult {
        guard let processor = imageProcessor else {
            throw MistralCoreError.vlmNotLoaded
        }

        let image = try processor.loadImage(from: path)
        return try analyzeImage(image: image, prompt: prompt, parameters: parameters, onToken: onToken)
    }

    /// Format vision prompt following Mistral chat template
    private func formatVisionPrompt(imageToken: String, userPrompt: String) -> String {
        // Mistral vision format: [INST] [IMG]...[IMG] prompt [/INST]
        return "[INST] \(imageToken)\n\(userPrompt) [/INST]"
    }

    /// Sample token from logits
    private func sampleToken(logits: MLXArray, parameters: GenerateParameters) -> MLXArray {
        var probs = logits

        // Apply temperature
        if parameters.temperature > 0 {
            probs = probs / parameters.temperature
        }

        // Apply softmax
        probs = MLX.softmax(probs, axis: -1)

        // Top-p sampling
        var sortedIndices: MLXArray? = nil
        if parameters.topP < 1.0 {
            sortedIndices = MLX.argSort(probs, axis: -1)
            let sortedProbs = MLX.takeAlong(probs, sortedIndices!, axis: -1)
            let cumProbs = MLX.cumsum(sortedProbs, axis: -1)

            // Find cutoff
            let mask = cumProbs .<= (1.0 - parameters.topP)
            let maskedProbs = MLX.where(mask, MLXArray(0.0), sortedProbs)

            // Renormalize
            let sum = MLX.sum(maskedProbs)
            probs = maskedProbs / sum
        }

        // Sample
        if parameters.temperature > 0 {
            let sampledIdx = MLXRandom.categorical(MLX.log(probs + 1e-10))
            // If we used top-p, map back from sorted space to vocabulary space
            if let indices = sortedIndices {
                return indices[sampledIdx]
            }
            return sampledIdx
        } else {
            // Greedy: if we used top-p, get argmax from sorted space and map back
            if let indices = sortedIndices {
                let sortedArgmax = MLX.argMax(probs, axis: -1)
                return indices[sortedArgmax]
            }
            return MLX.argMax(probs, axis: -1)
        }
    }

    /// Generate with streaming (AsyncStream)
    public func generateStream(
        prompt: String,
        parameters: GenerateParameters = .balanced
    ) throws -> AsyncStream<String> {
        guard let generator = generator else {
            throw MistralCoreError.modelNotLoaded
        }
        return generator.generateStream(prompt: prompt, parameters: parameters)
    }

    // MARK: - Embeddings

    /// Extract embeddings from text
    public func extractEmbeddings(
        prompt: String,
        config: HiddenStatesConfig = .mfluxDefault
    ) throws -> MLXArray {
        guard let extractor = extractor else {
            throw MistralCoreError.modelNotLoaded
        }
        return try extractor.extractEmbeddings(prompt: prompt, config: config)
    }

    /// Extract mflux-compatible embeddings
    public func extractMfluxEmbeddings(prompt: String) throws -> MLXArray {
        guard let extractor = extractor else {
            throw MistralCoreError.modelNotLoaded
        }
        return try extractor.extractMfluxEmbeddings(prompt: prompt)
    }

    /// Extract FLUX.2-compatible embeddings (identical to mflux-gradio Python)
    /// - Parameters:
    ///   - prompt: User prompt text
    ///   - maxLength: Maximum sequence length (default: 512)
    /// - Returns: Embeddings tensor with shape [1, maxLength, 15360]
    public func extractFluxEmbeddings(
        prompt: String,
        maxLength: Int = FluxConfig.maxSequenceLength
    ) throws -> MLXArray {
        guard let extractor = extractor else {
            throw MistralCoreError.modelNotLoaded
        }
        return try extractor.extractFluxEmbeddings(prompt: prompt, maxLength: maxLength)
    }

    /// Get FLUX-format token IDs for debugging/comparison with Python
    public func getFluxTokenIds(
        prompt: String,
        maxLength: Int = FluxConfig.maxSequenceLength
    ) throws -> [Int] {
        guard let extractor = extractor else {
            throw MistralCoreError.modelNotLoaded
        }
        return extractor.getFluxTokenIds(prompt: prompt, maxLength: maxLength)
    }

    /// Export embeddings to file
    /// This is a standalone operation that doesn't require the full model to be loaded
    public func exportEmbeddings(
        _ embeddings: MLXArray,
        to path: String,
        format: ExportFormat = .binary
    ) throws {
        // Standalone export - doesn't require extractor or full model
        switch format {
        case .binary:
            // Export as raw float32 binary
            let flatEmbeddings = embeddings.reshaped([-1]).asArray(Float.self)
            let data = flatEmbeddings.withUnsafeBufferPointer { buffer in
                Data(buffer: buffer)
            }
            try data.write(to: URL(fileURLWithPath: path))

        case .numpy:
            // For .npy format, use MLX's save function
            try MLX.save(array: embeddings, url: URL(fileURLWithPath: path))

        case .json:
            // Export as JSON with shape and values
            let shape = embeddings.shape
            let flatEmbeddings = embeddings.reshaped([-1]).asArray(Float.self)
            let dict: [String: Any] = [
                "shape": shape.map { $0 },
                "values": flatEmbeddings
            ]
            let jsonData = try JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted)
            try jsonData.write(to: URL(fileURLWithPath: path))
        }
    }

    // MARK: - CoreML (ANE-accelerated)

    /// Whether the CoreML encoder is loaded
    @available(macOS 14.0, *)
    public var isCoreMLLoaded: Bool {
        coremlEncoder?.isLoaded ?? false
    }

    /// Load CoreML encoder for ANE-accelerated FLUX.2 embeddings extraction
    ///
    /// The CoreML model runs on Apple Neural Engine for faster inference.
    /// It must be generated separately using the Python conversion scripts.
    ///
    /// - Parameter url: URL to the .mlpackage file (or nil to load from bundle)
    @available(macOS 14.0, *)
    public func loadCoreMLEncoder(from url: URL? = nil) throws {
        let encoder = MistralCoreMLEncoder()

        if let url = url {
            try encoder.load(from: url)
        } else {
            try encoder.loadFromBundle()
        }

        coremlEncoder = encoder
        MistralDebug.log("CoreML encoder loaded successfully")
    }

    /// Load CoreML encoder from a path string
    @available(macOS 14.0, *)
    public func loadCoreMLEncoder(fromPath path: String) throws {
        try loadCoreMLEncoder(from: URL(fileURLWithPath: path))
    }

    /// Unload CoreML encoder to free memory
    @available(macOS 14.0, *)
    public func unloadCoreMLEncoder() {
        coremlEncoder = nil
        MistralDebug.log("CoreML encoder unloaded")
    }

    /// Extract FLUX.2-compatible embeddings using CoreML (ANE)
    ///
    /// This method uses the CoreML encoder running on Apple Neural Engine
    /// for faster inference compared to the MLX-based extraction.
    ///
    /// - Parameter prompt: User prompt text
    /// - Returns: Embeddings tensor with shape [1, 512, 15360]
    @available(macOS 14.0, *)
    public func extractFluxEmbeddingsCoreML(prompt: String) throws -> MLXArray {
        guard let encoder = coremlEncoder, encoder.isLoaded else {
            throw MistralCoreError.invalidInput(
                "CoreML encoder not loaded. Call loadCoreMLEncoder() first."
            )
        }

        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }

        return try encoder.extractFluxEmbeddings(prompt: prompt, tokenizer: tokenizer)
    }

    /// Extract FLUX.2 embeddings using CoreML with timing information
    ///
    /// - Parameter prompt: User prompt text
    /// - Returns: Tuple of (embeddings, inferenceTimeSeconds)
    @available(macOS 14.0, *)
    public func extractFluxEmbeddingsCoreMLTimed(prompt: String) throws -> (MLXArray, TimeInterval) {
        guard let encoder = coremlEncoder, encoder.isLoaded else {
            throw MistralCoreError.invalidInput(
                "CoreML encoder not loaded. Call loadCoreMLEncoder() first."
            )
        }

        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }

        let tokens = tokenizer.encode(prompt, addSpecialTokens: true)
        let tokenIds = tokens.map { Int32($0) }

        return try encoder.extractEmbeddingsWithTiming(tokenIds: tokenIds)
    }

    // MARK: - CoreML Chunked Encoder (for large models)

    /// Whether the chunked CoreML encoder is loaded
    @available(macOS 14.0, *)
    public var isCoreMLChunkedLoaded: Bool {
        coremlChunkedEncoder?.isLoaded ?? false
    }

    /// Load chunked CoreML encoder for ANE-accelerated FLUX.2 embeddings
    ///
    /// The chunked encoder splits the model into 3 parts to work around CoreML
    /// limitations with large models. Each chunk runs on ANE sequentially.
    ///
    /// - Parameter directoryURL: Directory containing MistralChunk1/2/3.mlpackage files
    @available(macOS 14.0, *)
    public func loadCoreMLChunkedEncoder(from directoryURL: URL) throws {
        let encoder = MistralCoreMLChunkedEncoder()
        try encoder.loadChunks(from: directoryURL)
        coremlChunkedEncoder = encoder
        MistralDebug.log("CoreML chunked encoder loaded successfully (3 chunks)")
    }

    /// Load chunked CoreML encoder from path string
    @available(macOS 14.0, *)
    public func loadCoreMLChunkedEncoder(fromPath path: String) throws {
        try loadCoreMLChunkedEncoder(from: URL(fileURLWithPath: path))
    }

    /// Load chunked CoreML encoder with individual chunk paths
    @available(macOS 14.0, *)
    public func loadCoreMLChunkedEncoder(
        chunk1Path: String,
        chunk2Path: String,
        chunk3Path: String
    ) throws {
        let encoder = MistralCoreMLChunkedEncoder()
        try encoder.loadChunks(
            chunk1Path: chunk1Path,
            chunk2Path: chunk2Path,
            chunk3Path: chunk3Path
        )
        coremlChunkedEncoder = encoder
        MistralDebug.log("CoreML chunked encoder loaded successfully (3 chunks)")
    }

    /// Unload chunked CoreML encoder to free memory
    @available(macOS 14.0, *)
    public func unloadCoreMLChunkedEncoder() {
        coremlChunkedEncoder = nil
        MistralDebug.log("CoreML chunked encoder unloaded")
    }

    /// Load only the tokenizer without loading the full model
    /// Useful for CoreML-only mode where we need tokenization but not MLX inference
    ///
    /// - Parameter modelPath: Path to model directory containing tekken.json
    public func loadTokenizerOnly(from modelPath: String) {
        tokenizer = TekkenTokenizer(modelPath: modelPath)
        MistralDebug.log("Tokenizer loaded from \(modelPath)")
    }

    /// Extract FLUX.2-compatible embeddings using chunked CoreML (ANE)
    ///
    /// - Parameter prompt: User prompt text
    /// - Returns: Embeddings tensor with shape [1, 512, 15360]
    @available(macOS 14.0, *)
    public func extractFluxEmbeddingsCoreMLChunked(prompt: String) throws -> MLXArray {
        guard let encoder = coremlChunkedEncoder, encoder.isLoaded else {
            throw MistralCoreError.invalidInput(
                "CoreML chunked encoder not loaded. Call loadCoreMLChunkedEncoder() first."
            )
        }

        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }

        return try encoder.extractFluxEmbeddings(prompt: prompt, tokenizer: tokenizer)
    }

    /// Extract FLUX.2 embeddings using chunked CoreML with timing
    ///
    /// - Parameter prompt: User prompt text
    /// - Returns: Tuple of (embeddings, inferenceTimeSeconds)
    @available(macOS 14.0, *)
    public func extractFluxEmbeddingsCoreMLChunkedTimed(prompt: String) throws -> (MLXArray, TimeInterval) {
        guard let encoder = coremlChunkedEncoder, encoder.isLoaded else {
            throw MistralCoreError.invalidInput(
                "CoreML chunked encoder not loaded. Call loadCoreMLChunkedEncoder() first."
            )
        }

        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }

        let tokens = tokenizer.encode(prompt, addSpecialTokens: true)
        let tokenIds = tokens.map { Int32($0) }

        return try encoder.extractEmbeddingsWithTiming(tokenIds: tokenIds)
    }

    /// Extract FLUX.2 embeddings with detailed per-chunk timing
    ///
    /// - Parameter prompt: User prompt text
    /// - Returns: Tuple with embeddings and timing for each chunk
    @available(macOS 14.0, *)
    public func extractFluxEmbeddingsCoreMLChunkedDetailedTiming(prompt: String) throws -> (
        embeddings: MLXArray,
        chunk1Time: TimeInterval,
        chunk2Time: TimeInterval,
        chunk3Time: TimeInterval,
        totalTime: TimeInterval
    ) {
        guard let encoder = coremlChunkedEncoder, encoder.isLoaded else {
            throw MistralCoreError.invalidInput(
                "CoreML chunked encoder not loaded. Call loadCoreMLChunkedEncoder() first."
            )
        }

        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }

        let tokens = tokenizer.encode(prompt, addSpecialTokens: true)
        let tokenIds = tokens.map { Int32($0) }

        return try encoder.extractEmbeddingsWithDetailedTiming(tokenIds: tokenIds)
    }

    // MARK: - Tokenization

    /// Encode text to tokens
    public func encode(_ text: String, addSpecialTokens: Bool = false) throws -> [Int] {
        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }
        return tokenizer.encode(text, addSpecialTokens: addSpecialTokens)
    }

    /// Decode tokens to text
    public func decode(_ tokens: [Int], skipSpecialTokens: Bool = true) throws -> String {
        guard let tokenizer = tokenizer else {
            throw MistralCoreError.modelNotLoaded
        }
        return tokenizer.decode(tokens, skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Model Info

    /// Get model configuration
    public var config: MistralTextConfig? {
        return model?.config
    }

    /// Print available models
    @MainActor
    public func printAvailableModels() {
        ModelRegistry.shared.printAvailableModels()
    }
}

// MARK: - Errors

public enum MistralCoreError: LocalizedError {
    case modelNotLoaded
    case vlmNotLoaded
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model not loaded. Call loadModel() first."
        case .vlmNotLoaded:
            return "VLM not loaded. Call loadVLMModel() first for vision capabilities."
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        }
    }
}

// MARK: - Version Info

public struct MistralVersion {
    public static let version = "0.1.0"
    public static let modelName = "Mistral Small 3.2"
    public static let modelVersion = "24B-Instruct-2506"
}
