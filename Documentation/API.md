# MistralCore API Reference

## Overview

MistralCore provides a simple, singleton-based API for Mistral Small 3.2 inference on Apple Silicon.

## Quick Start

```swift
import MistralCore

let core = MistralCore.shared
try await core.loadModel(variant: .mlx8bit)
let result = try core.generate(prompt: "Hello, world!")
print(result.text)
```

## Core API

### MistralCore.shared

Singleton instance for all operations. Thread-safe for model loading and generation.

```swift
public class MistralCore {
    public static let shared: MistralCore
}
```

---

## Model Loading

### loadModel(variant:hfToken:progressHandler:)

Load a text-only model from HuggingFace.

```swift
public func loadModel(
    variant: ModelVariant,
    hfToken: String? = nil,
    progressHandler: ((Double, String) -> Void)? = nil
) async throws
```

**Parameters:**
- `variant`: Model variant to load (`.bf16`, `.mlx8bit`, `.mlx4bit`)
- `hfToken`: Optional HuggingFace token for private models
- `progressHandler`: Optional callback for download progress

**Example:**
```swift
try await MistralCore.shared.loadModel(variant: .mlx8bit) { progress, message in
    print("\(message) (\(Int(progress * 100))%)")
}
```

### loadModel(from:)

Load a model from a local path.

```swift
public func loadModel(from path: String) throws
```

### loadVLMModel(variant:hfToken:progressHandler:)

Load a Vision-Language Model (VLM) with Pixtral encoder.

```swift
public func loadVLMModel(
    variant: ModelVariant,
    hfToken: String? = nil,
    progressHandler: ((Double, String) -> Void)? = nil
) async throws
```

---

## Text Generation

### generate(prompt:parameters:onToken:)

Generate text from a prompt with streaming support.

```swift
public func generate(
    prompt: String,
    parameters: GenerateParameters = GenerateParameters(),
    onToken: ((String) -> Bool)? = nil
) throws -> GenerationResult
```

**Parameters:**
- `prompt`: Input text prompt
- `parameters`: Generation configuration
- `onToken`: Streaming callback, return `false` to stop generation

**Example:**
```swift
let result = try MistralCore.shared.generate(
    prompt: "Explain quantum computing",
    parameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
) { token in
    print(token, terminator: "")
    return true  // continue generation
}
print("\nTokens/sec: \(result.tokensPerSecond)")
```

### chat(messages:parameters:onToken:)

Multi-turn chat with conversation history.

```swift
public func chat(
    messages: [[String: String]],
    parameters: GenerateParameters = GenerateParameters(),
    onToken: ((String) -> Bool)? = nil
) throws -> GenerationResult
```

**Parameters:**
- `messages`: Array of message dictionaries with "role" and "content" keys
- Supported roles: `"system"`, `"user"`, `"assistant"`

**Example:**
```swift
let messages = [
    ["role": "system", "content": "You are a helpful assistant."],
    ["role": "user", "content": "What's the capital of France?"]
]
let result = try MistralCore.shared.chat(messages: messages)
print(result.text)
```

---

## Vision Analysis

### analyzeImage(path:prompt:parameters:onToken:)

Analyze an image using the VLM model.

```swift
public func analyzeImage(
    path: String,
    prompt: String,
    parameters: GenerateParameters = GenerateParameters(),
    onToken: ((String) -> Bool)? = nil
) throws -> GenerationResult
```

**Parameters:**
- `path`: Path to image file (PNG, JPG)
- `prompt`: Question or instruction about the image

**Example:**
```swift
try await MistralCore.shared.loadVLMModel(variant: .mlx4bit)
let result = try MistralCore.shared.analyzeImage(
    path: "/path/to/photo.jpg",
    prompt: "What objects are in this image?"
)
print(result.text)
```

---

## Embeddings Extraction

### extractEmbeddings(prompt:config:)

Extract hidden state embeddings with custom configuration.

```swift
public func extractEmbeddings(
    prompt: String,
    config: HiddenStatesConfig = .mfluxDefault
) throws -> MLXArray
```

### extractFluxEmbeddings(prompt:maxLength:)

Extract FLUX.2-compatible embeddings (layers 10, 20, 30).

```swift
public func extractFluxEmbeddings(
    prompt: String,
    maxLength: Int = 512
) throws -> MLXArray
```

**Returns:** MLXArray with shape `[1, maxLength, 15360]`

**Example:**
```swift
let embeddings = try MistralCore.shared.extractFluxEmbeddings(prompt: "A beautiful sunset")
print("Shape: \(embeddings.shape)")  // [1, 512, 15360]
```

### exportEmbeddings(_:to:format:)

Export embeddings to file.

```swift
public func exportEmbeddings(
    _ embeddings: MLXArray,
    to path: String,
    format: ExportFormat = .binary
) throws
```

**Formats:** `.binary` (raw float32), `.json`

---

## Types

### ModelVariant

```swift
public enum ModelVariant: String {
    case bf16     // Full precision (~48GB)
    case mlx8bit  // 8-bit quantized (~24GB) - recommended
    case mlx4bit  // 4-bit quantized (~12GB)
}
```

### GenerateParameters

```swift
public struct GenerateParameters {
    public var maxTokens: Int = 2048
    public var temperature: Float = 0.7
    public var topP: Float = 0.95
    public var repetitionPenalty: Float = 1.1
    public var seed: UInt64? = nil

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.7,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.1,
        seed: UInt64? = nil
    )
}
```

### GenerationResult

```swift
public struct GenerationResult {
    public let text: String
    public let tokenCount: Int
    public let promptTokens: Int
    public let generationTime: TimeInterval
    public let tokensPerSecond: Double

    public func summary() -> String
}
```

### HiddenStatesConfig

```swift
public struct HiddenStatesConfig {
    public var layerIndices: [Int]
    public var concatenate: Bool
    public var normalize: Bool
    public var pooling: PoolingStrategy

    /// FLUX.2-compatible default: layers 10, 20, 30 concatenated
    public static var mfluxDefault: HiddenStatesConfig
}
```

### PoolingStrategy

```swift
public enum PoolingStrategy {
    case none       // Keep all tokens
    case lastToken  // Use last token only
    case mean       // Average over sequence
    case max        // Max pooling
    case cls        // First token (CLS)
}
```

---

## Error Handling

All methods throw descriptive errors:

```swift
do {
    try await MistralCore.shared.loadModel(variant: .mlx8bit)
} catch MistralModelError.noWeightsFound {
    print("Model weights not found")
} catch {
    print("Error: \(error.localizedDescription)")
}
```

### Common Errors

- `MistralModelError.noWeightsFound` - Model files not found
- `MistralModelError.invalidConfig` - Invalid model configuration
- `EmbeddingError.noHiddenStates` - Hidden states not returned
- `EmbeddingError.invalidLayerIndex` - Layer index out of range

---

## Thread Safety

- Model loading is thread-safe
- Generation should be called from the main actor (`@MainActor`)
- Use `async/await` for concurrent operations

```swift
@MainActor
func generateResponse() async {
    let result = try? MistralCore.shared.generate(prompt: "Hello")
}
```
