# FLUX.2 API Documentation

This document describes the APIs specific to FLUX.2 image generation integration.

## Overview

The FluxTextEncoders library provides text encoding and prompt upsampling for FLUX.2 image generation models by Black Forest Labs. It supports:

- **Text-to-Image (T2I)**: Generate images from text prompts
- **Image-to-Image (I2I)**: Edit existing images based on instructions
- **Klein**: Smaller models for FLUX.2 Klein variants

## Quick Start

```swift
import FluxTextEncoders

let core = FluxTextEncoders.shared

// Load model (8-bit quantized by default)
try await core.loadModel(variant: .mlx8bit)

// Extract embeddings for FLUX.2
let embeddings = try core.extractFluxEmbeddings(prompt: "A cat sitting on a windowsill")
// Shape: [1, 512, 15360]
```

## Core Configuration

### FluxConfig

Central configuration for FLUX.2 operations.

```swift
public enum FluxConfig {
    /// Hidden state layers to extract: [10, 20, 30]
    /// Produces 3 × 5120 = 15360 dimensions
    static let hiddenStateLayers: [Int]

    /// Maximum sequence length for embeddings: 512 tokens
    static let maxSequenceLength: Int

    /// Maximum image sizes
    static let maxImageSizeUpsampling: Int  // 768 for I2I upsampling
    static let maxImageSizeReference: Int   // 512 for reference images

    /// Operation modes
    enum Mode {
        case embeddings      // Extract embeddings for conditioning
        case upsamplingT2I   // Enhance text prompts
        case upsamplingI2I   // Convert image editing requests
    }

    /// Get system message for a mode
    static func systemMessage(for mode: Mode) -> String

    /// Build chat messages for a mode
    static func buildMessages(prompt: String, mode: Mode) -> [[String: String]]
}
```

## Embedding Extraction

### Text-Only Embeddings (T2I)

Extract embeddings from a text prompt for FLUX.2 image generation:

```swift
// Standard FLUX.2 embeddings
let embeddings = try core.extractFluxEmbeddings(prompt: "A mountain landscape at sunset")

// Output shape: [1, 512, 15360]
// - Batch size: 1
// - Sequence length: 512 (padded)
// - Hidden dimension: 15360 (3 layers × 5120)
```

### Image + Text Embeddings (I2I)

Extract embeddings that include both image and text tokens:

```swift
// From file path
let embeddings = try core.extractFluxEmbeddingsWithImage(
    imagePath: "/path/to/image.png",
    prompt: "Make the sky more dramatic"
)

// From NSImage
let embeddings = try core.extractFluxEmbeddingsWithImage(
    image: nsImage,
    prompt: "Add a rainbow in the background"
)

// Output shape: [1, N, 15360]
// - N = image_tokens + text_tokens (variable, NOT padded to 512)
```

### Get Token IDs (Debugging)

```swift
// Get the actual token IDs used for embedding extraction
let tokenIds = try core.getFluxTokenIds(prompt: "Hello world")
// Returns: [Int] with padding to 512 tokens
```

## Prompt Upsampling

### Text-to-Image Upsampling

Enhance a simple prompt into a more detailed description:

```swift
// Load text model
try await core.loadModel(variant: .mlx8bit)

// Build messages with T2I system prompt
let messages = FluxConfig.buildMessages(
    prompt: "a cat",
    mode: .upsamplingT2I
)

// Generate enhanced prompt
let result = try core.chat(messages: messages, parameters: .balanced)
print(result.text)
// Output: "A fluffy orange tabby cat with bright green eyes, sitting gracefully
// on a sunlit wooden floor. Soft natural lighting from a nearby window creates
// warm highlights on its fur..."
```

### Image-to-Image Upsampling (VLM Required)

Convert an editing request + reference image into a precise instruction:

```swift
// Load VLM model (required for image analysis)
try await core.loadVLMModel(variant: .mlx4bit)

// Analyze image with I2I system prompt
let result = try core.analyzeImage(
    path: "/path/to/image.png",
    prompt: "Make it look more dramatic",
    systemPrompt: FluxConfig.systemMessage(for: .upsamplingI2I),
    parameters: .balanced
)

print(result.text)
// Output: "Increase contrast and saturation in the sky region. Add deeper
// orange and purple tones to the clouds. Keep the foreground mountains
// and their shadows unchanged."
```

## Vision Analysis (VLM)

### Basic Image Analysis

```swift
// Load VLM
try await core.loadVLMModel(variant: .mlx4bit)

// Analyze image
let result = try core.analyzeImage(
    path: "/path/to/photo.jpg",
    prompt: "Describe this image in detail"
)
print(result.text)
```

### Image Analysis with Custom System Prompt

```swift
let result = try core.analyzeImage(
    path: "/path/to/image.png",
    prompt: "What should be changed?",
    systemPrompt: "You are an art critic. Suggest improvements.",
    parameters: GenerateParameters(maxTokens: 200, temperature: 0.7)
)
```

## Klein Models (FLUX.2 Klein)

For FLUX.2 Klein variants using Qwen3 models:

```swift
// Load Klein model (4B or 9B)
try await core.loadKleinModel(variant: .qwen3_4b)

// Extract Klein embeddings
let embeddings = try core.extractKleinEmbeddings(prompt: "A forest path")

// Output dimensions depend on variant:
// - 4B: [1, 512, 7680]  (3 × 2560)
// - 9B: [1, 512, 12288] (3 × 4096)
```

### KleinVariant

```swift
public enum KleinVariant: String, CaseIterable {
    case qwen3_4b = "4b"  // Qwen3-4B for Klein 4B
    case qwen3_8b = "9b"  // Qwen3-8B for Klein 9B

    var hiddenStateLayers: [Int]  // [9, 18, 27]
    var outputDimension: Int      // 7680 or 12288
    var displayName: String
}
```

## Model Variants

### Mistral (for FLUX.2 dev/pro)

```swift
public enum ModelVariant: String, CaseIterable {
    case bf16 = "bf16"     // Full precision (48GB)
    case mlx8bit = "8bit"  // 8-bit quantized (24GB) - DEFAULT
    case mlx4bit = "4bit"  // 4-bit quantized (12GB) - Required for VLM
}
```

### Loading Models

```swift
// From HuggingFace (auto-download)
try await core.loadModel(variant: .mlx8bit, hfToken: "your_token")

// From local path
try core.loadModel(from: "/path/to/model")

// VLM (Vision-Language Model)
try await core.loadVLMModel(variant: .mlx4bit)

// Check if loaded
if core.isModelLoaded { ... }
if core.isVLMLoaded { ... }
```

## Generation Parameters

```swift
public struct GenerateParameters {
    var maxTokens: Int              // Default: 2048
    var temperature: Float          // Default: 0.7 (0 = greedy)
    var topP: Float                 // Default: 0.95
    var repetitionPenalty: Float    // Default: 1.1
    var repetitionContextSize: Int  // Default: 20
    var seed: UInt64?               // Optional for reproducibility

    // Presets
    static let greedy: GenerateParameters   // temperature=0
    static let balanced: GenerateParameters // temperature=0.7
    static let creative: GenerateParameters // temperature=0.9
}
```

## Export Embeddings

```swift
// Binary format (raw float32)
try core.exportEmbeddings(embeddings, to: "output.bin", format: .binary)

// NumPy format (.npy)
try core.exportEmbeddings(embeddings, to: "output.npy", format: .numpy)

// JSON format
try core.exportEmbeddings(embeddings, to: "output.json", format: .json)
```

## Complete Example: I2I Pipeline

```swift
import FluxTextEncoders

@MainActor
func processImageEdit(imagePath: String, userRequest: String) async throws -> (prompt: String, embeddings: MLXArray) {
    let core = FluxTextEncoders.shared

    // 1. Load VLM for image understanding
    try await core.loadVLMModel(variant: .mlx4bit)

    // 2. Generate editing instruction using VLM + I2I system prompt
    let result = try core.analyzeImage(
        path: imagePath,
        prompt: userRequest,
        systemPrompt: FluxConfig.systemMessage(for: .upsamplingI2I),
        parameters: GenerateParameters(maxTokens: 100, temperature: 0.7)
    )

    let enhancedPrompt = result.text
    print("Enhanced instruction: \(enhancedPrompt)")

    // 3. Extract embeddings with image tokens
    let embeddings = try core.extractFluxEmbeddingsWithImage(
        imagePath: imagePath,
        prompt: enhancedPrompt
    )

    print("Embeddings shape: \(embeddings.shape)")

    return (enhancedPrompt, embeddings)
}

// Usage
let (prompt, embeddings) = try await processImageEdit(
    imagePath: "/path/to/photo.jpg",
    userRequest: "Make the colors more vibrant"
)
```

## Error Handling

```swift
public enum FluxEncoderError: LocalizedError {
    case modelNotLoaded      // Call loadModel() first
    case vlmNotLoaded        // Call loadVLMModel() first
    case kleinNotLoaded      // Call loadKleinModel() first
    case invalidInput(String)
    case generationFailed(String)
}

// Usage
do {
    let embeddings = try core.extractFluxEmbeddings(prompt: "test")
} catch FluxEncoderError.modelNotLoaded {
    print("Please load a model first")
} catch {
    print("Error: \(error.localizedDescription)")
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for model downloads |
| `VLM_DEBUG` | Enable VLM debugging output |
| `FLUX_DEBUG` | Enable general debug logging |

## Output Dimensions Summary

| Model | Layers | Dimension per Layer | Total Output |
|-------|--------|---------------------|--------------|
| Mistral (dev/pro) | [10, 20, 30] | 5,120 | 15,360 |
| Qwen3 4B (Klein 4B) | [9, 18, 27] | 2,560 | 7,680 |
| Qwen3 8B (Klein 9B) | [9, 18, 27] | 4,096 | 12,288 |

All embeddings are padded to 512 tokens for text-only extraction. Image+text embeddings have variable length based on image resolution.
