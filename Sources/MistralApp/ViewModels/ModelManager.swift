/**
 * ModelManager.swift
 * Manages model loading, downloading, and state
 */

import SwiftUI
import MistralCore
import MLX

// MARK: - Memory Stats

struct MemoryStats {
    let active: Int
    let cache: Int
    let peak: Int

    static var current: MemoryStats {
        MemoryStats(
            active: GPU.activeMemory,
            cache: GPU.cacheMemory,
            peak: GPU.peakMemory
        )
    }
}

// MARK: - Model Manager

@MainActor
class ModelManager: ObservableObject {
    // MARK: - Loading State
    @Published var isLoaded = false
    @Published var isVLMLoaded = false  // Always load as VLM now
    @Published var isLoading = false
    @Published var loadingMessage = ""
    @Published var selectedVariant: ModelVariant?
    @Published var errorMessage: String?
    @Published var currentLoadedModelId: String?

    // MARK: - Download State
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var downloadMessage = ""

    // MARK: - Model Lists
    @Published var downloadedModels: Set<String> = []
    @Published var modelSizes: [String: Int64] = [:]

    // MARK: - Memory
    @Published var memoryStats = MemoryStats.current

    private let core = MistralCore.shared

    /// Models cache directory
    static var modelsCacheDirectory: URL {
        if let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first {
            return cacheDir.appendingPathComponent("models")
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/models")
    }

    init() {
        isLoaded = core.isModelLoaded
        isVLMLoaded = core.isVLMLoaded
        refreshDownloadedModels()
        selectSmallestDownloadedModel()
    }

    /// Select the smallest downloaded model by default
    private func selectSmallestDownloadedModel() {
        // ModelVariant ordered from smallest to largest: .mlx4bit, .mlx6bit, .mlx8bit, .bf16
        let variantsSmallestFirst: [ModelVariant] = [.mlx4bit, .mlx6bit, .mlx8bit, .bf16]

        for variant in variantsSmallestFirst {
            if let model = ModelRegistry.shared.model(withVariant: variant),
               downloadedModels.contains(model.id) {
                selectedVariant = variant
                return
            }
        }

        // No model downloaded, leave selectedVariant as nil
        selectedVariant = nil
    }

    // MARK: - Available Models

    var availableModels: [ModelInfo] {
        ModelRegistry.shared.allModels()
    }

    var isCurrentModelLoaded: Bool {
        guard let currentId = currentLoadedModelId,
              let variant = selectedVariant,
              let model = ModelRegistry.shared.model(withVariant: variant) else {
            return false
        }
        return currentId == model.id
    }

    // MARK: - Refresh

    func refreshDownloadedModels() {
        var downloaded: Set<String> = []
        var sizes: [String: Int64] = [:]

        for model in availableModels {
            if let path = ModelDownloader.findModelPath(for: model) {
                downloaded.insert(model.id)
                sizes[model.id] = calculateDirectorySize(at: path)
            }
        }

        downloadedModels = downloaded
        modelSizes = sizes
        memoryStats = MemoryStats.current
    }

    private func calculateDirectorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }

        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(fileSize)
            }
        }
        return totalSize
    }

    // MARK: - Load Model

    func loadModel() async {
        guard !isLoading else { return }
        guard let variant = selectedVariant else {
            errorMessage = "No model selected. Please download a model first."
            return
        }

        isLoading = true
        loadingMessage = "Preparing to load model..."
        errorMessage = nil

        do {
            let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"]
                ?? UserDefaults.standard.string(forKey: "hfToken")

            // Always load as VLM (Vision-Language Model) - supports both text and vision
            try await core.loadVLMModel(
                variant: variant,
                hfToken: hfToken
            ) { progress, message in
                Task { @MainActor in
                    self.loadingMessage = "\(message) (\(Int(progress * 100))%)"
                }
            }

            if let model = ModelRegistry.shared.model(withVariant: variant) {
                currentLoadedModelId = model.id
            }
            isLoaded = true
            isVLMLoaded = true
            loadingMessage = ""
            refreshDownloadedModels()

        } catch {
            errorMessage = error.localizedDescription
            loadingMessage = ""
        }

        isLoading = false
    }

    func loadModel(from path: String) {
        isLoading = true
        loadingMessage = "Loading model..."
        errorMessage = nil

        Task {
            do {
                // Always load as VLM for unified experience
                try core.loadVLMModel(from: path)
                await MainActor.run {
                    isLoaded = true
                    isVLMLoaded = true
                    isLoading = false
                    loadingMessage = ""
                    refreshDownloadedModels()
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isLoading = false
                    loadingMessage = ""
                }
            }
        }
    }

    func loadModel(_ modelId: String) async {
        guard let model = availableModels.first(where: { $0.id == modelId }) else { return }

        // Unload current model if different
        if isLoaded && currentLoadedModelId != modelId {
            unloadModel()
        }

        selectedVariant = model.variant
        await loadModel()
    }

    // MARK: - Unload Model

    func unloadModel() {
        core.unloadModel()
        isLoaded = false
        isVLMLoaded = false
        currentLoadedModelId = nil
        memoryStats = MemoryStats.current
    }

    // MARK: - Download Model

    func downloadModel(_ modelId: String) async {
        guard let model = availableModels.first(where: { $0.id == modelId }) else { return }
        guard !isDownloading else { return }

        isDownloading = true
        downloadProgress = 0
        downloadMessage = "Starting download..."

        do {
            let downloader = ModelDownloader(
                hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
                    ?? UserDefaults.standard.string(forKey: "hfToken")
            )

            _ = try await downloader.download(model) { progress, message in
                Task { @MainActor in
                    self.downloadProgress = progress
                    self.downloadMessage = message
                }
            }

            downloadedModels.insert(modelId)
            refreshDownloadedModels()
            downloadMessage = "Download complete!"

        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        isDownloading = false
    }

    // MARK: - Delete Model

    func deleteModel(_ modelId: String) async throws {
        guard let model = availableModels.first(where: { $0.id == modelId }) else { return }

        // Can't delete if currently loaded
        if currentLoadedModelId == modelId {
            throw ModelManagerError.cannotDeleteLoadedModel
        }

        guard let path = ModelDownloader.findModelPath(for: model) else { return }

        try FileManager.default.removeItem(at: path)
        downloadedModels.remove(modelId)
        modelSizes.removeValue(forKey: modelId)
        refreshDownloadedModels()
    }

    // MARK: - Memory Management

    func clearCache() {
        GPU.clearCache()
        memoryStats = MemoryStats.current
    }

    func resetPeakMemory() {
        GPU.resetPeakMemory()
        memoryStats = MemoryStats.current
    }

    // MARK: - Formatting

    static func formatBytes(_ bytes: Int) -> String {
        let absBytes = abs(bytes)
        if absBytes >= 1024 * 1024 * 1024 {
            return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
        } else if absBytes >= 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        } else if absBytes >= 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        }
        return "\(bytes) B"
    }
}

// MARK: - Errors

enum ModelManagerError: LocalizedError {
    case cannotDeleteLoadedModel

    var errorDescription: String? {
        switch self {
        case .cannotDeleteLoadedModel:
            return "Cannot delete a loaded model. Unload it first."
        }
    }
}
