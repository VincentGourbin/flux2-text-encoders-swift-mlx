/**
 * ModelRegistry.swift
 * Registry of available Mistral Small 3.2 models
 */

import Foundation

// MARK: - Model Variant

public enum ModelVariant: String, CaseIterable, Codable, Sendable {
    case bf16 = "bf16"
    case mlx8bit = "8bit"
    case mlx6bit = "6bit"
    case mlx4bit = "4bit"

    public var displayName: String {
        switch self {
        case .bf16: return "Full Precision (BF16)"
        case .mlx8bit: return "8-bit Quantized"
        case .mlx6bit: return "6-bit Quantized"
        case .mlx4bit: return "4-bit Quantized"
        }
    }

    public var estimatedSize: String {
        switch self {
        case .bf16: return "~48GB"
        case .mlx8bit: return "~25GB"
        case .mlx6bit: return "~19GB"
        case .mlx4bit: return "~14GB"
        }
    }

    public var shortName: String {
        switch self {
        case .bf16: return "BF16"
        case .mlx8bit: return "8-bit"
        case .mlx6bit: return "6-bit"
        case .mlx4bit: return "4-bit"
        }
    }
}

// MARK: - Model Info

public struct ModelInfo: Codable, Sendable {
    public let id: String
    public let repoId: String
    public let name: String
    public let description: String
    public let variant: ModelVariant
    public let parameters: String

    public init(
        id: String,
        repoId: String,
        name: String,
        description: String,
        variant: ModelVariant,
        parameters: String
    ) {
        self.id = id
        self.repoId = repoId
        self.name = name
        self.description = description
        self.variant = variant
        self.parameters = parameters
    }
}

// MARK: - Model Registry

@MainActor
public final class ModelRegistry {
    public static let shared = ModelRegistry()

    private var models: [ModelInfo] = []

    private init() {
        registerDefaultModels()
    }

    private func registerDefaultModels() {
        // Mistral Small 3.2 models
        // Quantized versions from lmstudio-community include VLM (vision) layers
        models = [
            ModelInfo(
                id: "mistral-small-3.2-bf16",
                repoId: "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
                name: "Mistral Small 3.2 (BF16)",
                description: "Original full precision model from Mistral AI - reference quality",
                variant: .bf16,
                parameters: "24B"
            ),
            ModelInfo(
                id: "mistral-small-3.2-8bit",
                repoId: "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-8bit",
                name: "Mistral Small 3.2 (8-bit)",
                description: "8-bit quantized with VLM layers, good balance of quality and memory",
                variant: .mlx8bit,
                parameters: "24B"
            ),
            ModelInfo(
                id: "mistral-small-3.2-6bit",
                repoId: "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-6bit",
                name: "Mistral Small 3.2 (6-bit)",
                description: "6-bit quantized with VLM layers, balanced compression",
                variant: .mlx6bit,
                parameters: "24B"
            ),
            ModelInfo(
                id: "mistral-small-3.2-4bit",
                repoId: "lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-MLX-4bit",
                name: "Mistral Small 3.2 (4-bit)",
                description: "4-bit quantized with VLM layers, memory efficient",
                variant: .mlx4bit,
                parameters: "24B"
            ),
        ]
    }

    /// URL for downloading tekken.json from the original Mistral repo
    public static let tekkenJsonURL = "https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/resolve/main/tekken.json"

    public func allModels() -> [ModelInfo] {
        return models
    }

    public func model(withId id: String) -> ModelInfo? {
        return models.first { $0.id == id }
    }

    public func model(withVariant variant: ModelVariant) -> ModelInfo? {
        return models.first { $0.variant == variant }
    }

    public func defaultModel() -> ModelInfo {
        return model(withVariant: .mlx8bit) ?? models[0]
    }

    public func printAvailableModels() {
        print("\nAvailable Mistral Small 3.2 Models:")
        print("=".padding(toLength: 70, withPad: "=", startingAt: 0))

        for model in models {
            let isDownloaded = ModelDownloader.isModelDownloaded(model)
            let status = isDownloaded ? "✓ Downloaded" : "○ Not downloaded"

            print("\n  \(model.name)")
            print("  ID: \(model.id)")
            print("  Repo: \(model.repoId)")
            print("  Size: \(model.variant.estimatedSize)")
            print("  Status: \(status)")

            if isDownloaded, let path = ModelDownloader.findModelPath(for: model) {
                print("  Path: \(path.path)")
            }
        }

        print("\n" + "=".padding(toLength: 70, withPad: "=", startingAt: 0))
    }
}
