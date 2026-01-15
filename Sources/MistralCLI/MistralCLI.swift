/**
 * Mistral CLI
 * Command-line interface for Mistral Small 3.2 inference
 */

import ArgumentParser
import Foundation
import MistralCore
import MLX

@main
struct MistralCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mistral",
        abstract: "Mistral Small 3.2 Swift MLX - Inference CLI",
        version: MistralCore.version,
        subcommands: [
            Generate.self,
            Chat.self,
            Embed.self,
            Vision.self,
            Models.self,
        ],
        defaultSubcommand: Chat.self
    )
}

// MARK: - Common Options

struct ModelOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "Model variant: bf16, 8bit, 4bit")
    var model: String = "8bit"

    @Option(name: .long, help: "Local model path (overrides --model)")
    var modelPath: String?

    @Option(name: .long, help: "HuggingFace token for downloading models")
    var hfToken: String?

    var variant: ModelVariant {
        return ModelVariant(rawValue: model) ?? .mlx8bit
    }
}

struct GenerationOptions: ParsableArguments {
    @Option(name: [.customShort("n"), .long], help: "Maximum tokens to generate (Mistral supports up to 131K context)")
    var maxTokens: Int = 2048

    @Option(name: .shortAndLong, help: "Temperature (0.0 = greedy, higher = more random)")
    var temperature: Float = 0.7

    @Option(name: .long, help: "Top-p sampling threshold")
    var topP: Float = 0.95

    @Option(name: .long, help: "Repetition penalty")
    var repetitionPenalty: Float = 1.1

    @Option(name: .long, help: "Random seed for reproducibility")
    var seed: UInt64?

    @Flag(name: .long, help: "Enable detailed profiling output")
    var profile: Bool = false

    var parameters: GenerateParameters {
        return GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            seed: seed
        )
    }
}

// MARK: - Generate Command

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Generate text from a prompt"
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var genOptions: GenerationOptions

    @Argument(help: "Input prompt")
    var prompt: String

    @Flag(name: .long, help: "Disable streaming output")
    var noStream: Bool = false

    @MainActor
    func run() async throws {
        let core = MistralCore.shared

        // Enable profiling if requested
        MistralProfiler.shared.isEnabled = genOptions.profile
        MistralProfiler.shared.reset()

        // Load model
        print("Loading model...")
        if let path = modelOptions.modelPath {
            try core.loadModel(from: path)
        } else {
            try await core.loadModel(
                variant: modelOptions.variant,
                hfToken: modelOptions.hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ) { progress, message in
                print("\r\(message) (\(Int(progress * 100))%)", terminator: "")
                fflush(stdout)
            }
            print()
        }

        // Generate
        print("\n--- Generation ---\n")

        let result = try core.generate(
            prompt: prompt,
            parameters: genOptions.parameters
        ) { token in
            if !noStream {
                print(token, terminator: "")
                fflush(stdout)
            }
            return true
        }

        if noStream {
            print(result.text)
        }

        print("\n\n--- Stats ---")
        print(result.summary())

        // Print detailed profiling if enabled
        if genOptions.profile {
            let summary = MistralProfiler.shared.summary()
            print("\n" + summary.description)
        }
    }
}

// MARK: - Chat Command

struct Chat: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Interactive chat mode"
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var genOptions: GenerationOptions

    @Option(name: .shortAndLong, help: "System prompt")
    var system: String?

    @MainActor
    func run() async throws {
        let core = MistralCore.shared

        // Enable profiling if requested
        MistralProfiler.shared.isEnabled = genOptions.profile

        // Load model
        print("Loading model...")
        if let path = modelOptions.modelPath {
            try core.loadModel(from: path)
        } else {
            try await core.loadModel(
                variant: modelOptions.variant,
                hfToken: modelOptions.hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ) { progress, message in
                print("\r\(message) (\(Int(progress * 100))%)", terminator: "")
                fflush(stdout)
            }
            print()
        }

        print("\nMistral Small 3.2 Chat")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to start a new conversation")
        if genOptions.profile {
            print("Profiling: ENABLED")
        }
        print("-".padding(toLength: 50, withPad: "-", startingAt: 0))

        var messages: [[String: String]] = []

        // Add system prompt if provided
        if let systemPrompt = system {
            messages.append(["role": "system", "content": systemPrompt])
        }

        while true {
            print("\nYou: ", terminator: "")
            fflush(stdout)

            guard let input = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                continue
            }

            if input.isEmpty { continue }

            if input.lowercased() == "quit" || input.lowercased() == "exit" {
                print("Goodbye!")
                break
            }

            if input.lowercased() == "clear" {
                messages.removeAll()
                if let systemPrompt = system {
                    messages.append(["role": "system", "content": systemPrompt])
                }
                print("Conversation cleared.")
                continue
            }

            // Add user message
            messages.append(["role": "user", "content": input])

            // Reset profiler for this turn
            MistralProfiler.shared.reset()

            // Generate response
            print("\nAssistant: ", terminator: "")
            fflush(stdout)

            var responseText = ""
            let result = try core.chat(
                messages: messages,
                parameters: genOptions.parameters
            ) { token in
                print(token, terminator: "")
                fflush(stdout)
                responseText += token
                return true
            }

            print()

            // Print profiling info if enabled
            if genOptions.profile {
                let metrics = MistralProfiler.shared.getMetrics()
                print("\n\(metrics.compactSummary)")
            }

            // Add assistant response to history
            messages.append(["role": "assistant", "content": result.text])
        }
    }
}

// MARK: - Embed Command

struct Embed: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Extract embeddings from text"
    )

    @OptionGroup var modelOptions: ModelOptions

    @Argument(help: "Input text to embed")
    var text: String

    @Option(name: .shortAndLong, help: "Output file path")
    var output: String?

    @Option(name: .long, help: "Layer indices to extract (comma-separated, e.g., '10,20,30')")
    var layers: String = "10,20,30"

    @Flag(name: .long, help: "Use mflux-compatible output format (layers 10,20,30, no padding)")
    var mflux: Bool = false

    @Flag(name: .long, help: "Use FLUX.2-compatible format (chat template + padding to 512)")
    var flux: Bool = false

    @Flag(name: .long, help: "Normalize embeddings")
    var normalize: Bool = false

    @Flag(name: .long, help: "Show token IDs (for debugging)")
    var showTokens: Bool = false

    @MainActor
    func run() async throws {
        let core = MistralCore.shared

        // Load MLX model
        print("Loading MLX model...")
        if let path = modelOptions.modelPath {
            try core.loadModel(from: path)
        } else {
            try await core.loadModel(
                variant: modelOptions.variant,
                hfToken: modelOptions.hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ) { progress, message in
                print("\r\(message) (\(Int(progress * 100))%)", terminator: "")
                fflush(stdout)
            }
            print()
        }

        // FLUX.2-compatible format (matches Python mflux-gradio exactly)
        if flux {
            print("Using FLUX.2-compatible format (chat template + 512 padding)")
            print("System message: \"\(FluxConfig.systemMessage.prefix(60))...\"")
            print("Layers: \(FluxConfig.hiddenStateLayers)")

            // Show token IDs if requested
            if showTokens {
                let tokenIds = try core.getFluxTokenIds(prompt: text)
                let nonPadCount = tokenIds.filter { $0 != 11 }.count  // 11 is pad token
                print("\nToken IDs (\(nonPadCount) non-pad tokens, padded to \(tokenIds.count)):")
                print("First 20: \(Array(tokenIds.prefix(20)))")
                print("Last 50: \(Array(tokenIds.suffix(50)))")

                // Save tokens to JSON for comparison
                let tokensData = try JSONEncoder().encode(tokenIds)
                try tokensData.write(to: URL(fileURLWithPath: "/tmp/swift_tokens.json"))
                print("Tokens saved to: /tmp/swift_tokens.json")
            }

            // Extract FLUX embeddings
            let startTime = Date()
            let embeddings = try core.extractFluxEmbeddings(prompt: text)
            let elapsedTime = Date().timeIntervalSince(startTime)

            print("\nMLX Embeddings shape: \(embeddings.shape)")
            print("Inference time: \(String(format: "%.3f", elapsedTime))s")

            printEmbeddingsSummary(embeddings, label: "MLX")

            // Save if output specified
            if let outputPath = output {
                try core.exportEmbeddings(embeddings, to: outputPath, format: .binary)
                print("Saved to: \(outputPath)")
            }
            return
        }

        // Parse layer indices
        let layerIndices = layers.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }

        // Configure extraction
        let config: HiddenStatesConfig
        if mflux {
            config = .mfluxDefault
        } else {
            config = HiddenStatesConfig(
                layerIndices: layerIndices,
                concatenate: true,
                normalize: normalize,
                pooling: .none
            )
        }

        print("Extracting embeddings from layers: \(config.layerIndices)")

        // Extract
        let embeddings = try core.extractEmbeddings(prompt: text, config: config)

        print("Embeddings shape: \(embeddings.shape)")

        // Save if output specified
        if let outputPath = output {
            try core.exportEmbeddings(embeddings, to: outputPath, format: .binary)
            print("Saved to: \(outputPath)")
        } else {
            // Print summary
            let flatEmbeddings = embeddings.reshaped([-1])
            let count = flatEmbeddings.shape[0]
            print("Total elements: \(count)")

            // Print first few values
            let firstFew = min(10, count)
            let values = flatEmbeddings[0..<firstFew].asArray(Float.self)
            print("First \(firstFew) values: \(values.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
        }
    }

    // MARK: - Helper Functions

    private func printEmbeddingsSummary(_ embeddings: MLXArray, label: String) {
        let flatEmbeddings = embeddings.reshaped([-1])
        let count = flatEmbeddings.shape[0]
        print("\(label) Total elements: \(count)")

        // Print first few values
        let firstFew = min(10, count)
        let values = flatEmbeddings[0..<firstFew].asArray(Float.self)
        print("\(label) First 10 values: \(values.map { String(format: "%.6f", $0) }.joined(separator: ", "))")

        // Print last few values
        let lastFew = min(10, count)
        let lastValues = flatEmbeddings[(count - lastFew)..<count].asArray(Float.self)
        print("\(label) Last 10 values: \(lastValues.map { String(format: "%.6f", $0) }.joined(separator: ", "))")

        // Statistics
        let allValues = flatEmbeddings.asArray(Float.self)
        let minVal = allValues.min() ?? 0
        let maxVal = allValues.max() ?? 0
        let mean = allValues.reduce(0, +) / Float(allValues.count)
        print("\(label) Range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
        print("\(label) Mean: \(String(format: "%.6f", mean))")
    }
}

// MARK: - Vision Command

struct Vision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Analyze an image with vision capabilities"
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var genOptions: GenerationOptions

    @Argument(help: "Path to image file (PNG, JPG)")
    var imagePath: String

    @Argument(help: "Prompt describing what to analyze")
    var prompt: String

    @Flag(name: .long, help: "Disable streaming output")
    var noStream: Bool = false

    @MainActor
    func run() async throws {
        let core = MistralCore.shared

        // Enable profiling if requested
        MistralProfiler.shared.isEnabled = genOptions.profile
        MistralProfiler.shared.reset()

        // Verify image exists
        guard FileManager.default.fileExists(atPath: imagePath) else {
            print("Error: Image file not found: \(imagePath)")
            return
        }

        // Load VLM model (requires 4-bit for vision layers)
        print("Loading VLM model...")
        if let path = modelOptions.modelPath {
            try core.loadVLMModel(from: path)
        } else {
            try await core.loadVLMModel(
                variant: modelOptions.variant,
                hfToken: modelOptions.hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
            ) { progress, message in
                print("\r\(message) (\(Int(progress * 100))%)", terminator: "")
                fflush(stdout)
            }
            print()
        }

        print("VLM loaded successfully")
        print("\n--- Analyzing Image ---")
        print("Image: \(imagePath)")
        print("Prompt: \(prompt)")
        print()

        // Analyze image
        let result = try core.analyzeImage(
            path: imagePath,
            prompt: prompt,
            parameters: genOptions.parameters
        ) { token in
            if !noStream {
                print(token, terminator: "")
                fflush(stdout)
            }
            return true
        }

        if noStream {
            print(result.text)
        }

        print("\n\n--- Stats ---")
        print(result.summary())

        // Print detailed profiling if enabled
        if genOptions.profile {
            let summary = MistralProfiler.shared.summary()
            print("\n" + summary.description)
        }
    }
}

// MARK: - Models Command

struct Models: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List available models and download them"
    )

    @Option(name: .shortAndLong, help: "Download a model by ID (e.g., mistral-small-3.2-4bit)")
    var download: String?

    @Option(name: [.customShort("D"), .long], help: "Delete a downloaded model by ID")
    var delete: String?

    @MainActor
    func run() async throws {
        if let modelId = download {
            try await downloadModel(modelId)
        } else if let modelId = delete {
            try await deleteModel(modelId)
        } else {
            ModelRegistry.shared.printAvailableModels()
        }
    }

    @MainActor
    private func downloadModel(_ modelId: String) async throws {
        guard let model = ModelRegistry.shared.model(withId: modelId) else {
            // Try variant matching
            if let variant = ModelVariant(rawValue: modelId.replacingOccurrences(of: "mistral-small-3.2-", with: "")),
               let model = ModelRegistry.shared.model(withVariant: variant) {
                try await downloadModelInfo(model)
                return
            }
            print("Error: Model '\(modelId)' not found")
            print("Available IDs: \(ModelRegistry.shared.allModels().map { $0.id }.joined(separator: ", "))")
            return
        }

        try await downloadModelInfo(model)
    }

    @MainActor
    private func downloadModelInfo(_ model: ModelInfo) async throws {
        if ModelDownloader.isModelDownloaded(model) {
            print("Model '\(model.name)' is already downloaded")
            if let path = ModelDownloader.findModelPath(for: model) {
                print("Path: \(path.path)")
            }
            return
        }

        print("Downloading \(model.name)...")
        print("Repository: \(model.repoId)")
        print("Estimated size: \(model.variant.estimatedSize)")
        print()

        let downloader = ModelDownloader(
            hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
        )

        let path = try await downloader.download(model) { progress, message in
            // Print progress on same line
            print("\r\(message) [\(Int(progress * 100))%]", terminator: "")
            fflush(stdout)
        }

        print("\n\nDownload complete!")
        print("Path: \(path.path)")
    }

    @MainActor
    private func deleteModel(_ modelId: String) async throws {
        guard let model = ModelRegistry.shared.model(withId: modelId) else {
            print("Error: Model '\(modelId)' not found")
            return
        }

        guard let path = ModelDownloader.findModelPath(for: model) else {
            print("Model '\(model.name)' is not downloaded")
            return
        }

        print("Deleting \(model.name)...")
        try FileManager.default.removeItem(at: path)
        print("Deleted: \(path.path)")
    }
}
