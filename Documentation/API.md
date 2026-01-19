# FluxTextEncoders API Reference

## Overview

FluxTextEncoders provides a simple, singleton-based API for FLUX.2 text encoders on Apple Silicon.
Supports both Mistral Small 3.2 (for FLUX.2 dev/pro) and Qwen3 (for FLUX.2 Klein).

## Quick Start

```swift
import FluxTextEncoders

let core = FluxTextEncoders.shared

// Mistral (FLUX.2 dev)
try await core.loadModel(variant: .mlx8bit)
let result = try core.generate(prompt: "Hello, world!")
print(result.text)

// Qwen3 (FLUX.2 Klein)
try await core.loadQwen3Model(variant: .qwen3_4B_8bit)
let kleinEmb = try core.extractKleinEmbeddings(prompt: "A cat", config: .klein4B)
```

## Core API

### FluxTextEncoders.shared

Singleton instance for all operations. Thread-safe for model loading and generation.

```swift
public class FluxTextEncoders {
    public static let shared: FluxTextEncoders
}
```

---

## Model Loading

### loadModel(variant:hfToken:progressHandler:)

Load a Mistral text-only model from HuggingFace.

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
try await FluxTextEncoders.shared.loadModel(variant: .mlx8bit) { progress, message in
    print("\(message) (\(Int(progress * 100))%)")
}
```

### loadModel(from:)

Load a model from a local path.

```swift
public func loadModel(from path: String) throws
```

### loadVLMModel(variant:hfToken:progressHandler:)

Load a Mistral Vision-Language Model (VLM) with Pixtral encoder.

```swift
public func loadVLMModel(
    variant: ModelVariant,
    hfToken: String? = nil,
    progressHandler: ((Double, String) -> Void)? = nil
) async throws
```

### loadQwen3Model(variant:hfToken:progressHandler:)

Load a Qwen3 model for FLUX.2 Klein.

```swift
public func loadQwen3Model(
    variant: Qwen3ModelVariant,
    hfToken: String? = nil,
    progressHandler: ((Double, String) -> Void)? = nil
) async throws
```

**Variants:**
- `.qwen3_4B_8bit` - Qwen3 4B 8-bit (for Klein 4B)
- `.qwen3_4B_4bit` - Qwen3 4B 4-bit
- `.qwen3_8B_8bit` - Qwen3 8B 8-bit (for Klein 9B)
- `.qwen3_8B_4bit` - Qwen3 8B 4-bit

---

## Text Generation (Mistral)

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
let result = try FluxTextEncoders.shared.generate(
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
let result = try FluxTextEncoders.shared.chat(messages: messages)
print(result.text)
```

---

## Text Generation (Qwen3)

### generateQwen3(prompt:parameters:enableThinking:onToken:)

Generate text using Qwen3 model.

```swift
public func generateQwen3(
    prompt: String,
    parameters: GenerateParameters = .balanced,
    enableThinking: Bool = false,
    onToken: ((String) -> Bool)? = nil
) throws -> GenerationResult
```

**Parameters:**
- `enableThinking`: Enable Qwen3 thinking mode (default: false for FLUX.2 usage)

### chatQwen3(messages:parameters:enableThinking:onToken:)

Multi-turn chat with Qwen3 model.

```swift
public func chatQwen3(
    messages: [[String: String]],
    parameters: GenerateParameters = .balanced,
    enableThinking: Bool = false,
    onToken: ((String) -> Bool)? = nil
) throws -> GenerationResult
```

**Example:**
```swift
let result = try FluxTextEncoders.shared.chatQwen3(
    messages: [["role": "user", "content": "Enhance this prompt: A cat"]],
    parameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
)
print(result.text)
```

---

## Vision Analysis (Mistral VLM)

### analyzeImage(path:prompt:parameters:onToken:)

Analyze an image using the Mistral VLM model.

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
try await FluxTextEncoders.shared.loadVLMModel(variant: .mlx4bit)
let result = try FluxTextEncoders.shared.analyzeImage(
    path: "/path/to/photo.jpg",
    prompt: "What objects are in this image?"
)
print(result.text)
```

---

## Embeddings Extraction

### FLUX.2 dev/pro Embeddings (Mistral)

#### extractEmbeddings(prompt:config:)

Extract hidden state embeddings with custom configuration.

```swift
public func extractEmbeddings(
    prompt: String,
    config: HiddenStatesConfig = .mfluxDefault
) throws -> MLXArray
```

#### extractFluxEmbeddings(prompt:maxLength:)

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
let embeddings = try FluxTextEncoders.shared.extractFluxEmbeddings(prompt: "A beautiful sunset")
print("Shape: \(embeddings.shape)")  // [1, 512, 15360]
```

### FLUX.2 Klein Embeddings (Qwen3)

#### extractKleinEmbeddings(prompt:config:)

Extract embeddings for FLUX.2 Klein using Qwen3.

```swift
public func extractKleinEmbeddings(
    prompt: String,
    config: KleinConfig = .klein4B
) throws -> MLXArray
```

**Parameters:**
- `config`: Klein configuration (`.klein4B` or `.klein9B`)

**Returns:**
- Klein 4B: `[1, 512, 7680]` (3 × 2560)
- Klein 9B: `[1, 512, 12288]` (3 × 4096)

**Example:**
```swift
try await FluxTextEncoders.shared.loadQwen3Model(variant: .qwen3_4B_8bit)
let embeddings = try FluxTextEncoders.shared.extractKleinEmbeddings(
    prompt: "A sunset over mountains",
    config: .klein4B
)
print("Shape: \(embeddings.shape)")  // [1, 512, 7680]
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

### ModelVariant (Mistral)

```swift
public enum ModelVariant: String {
    case bf16     // Full precision (~48GB)
    case mlx8bit  // 8-bit quantized (~24GB) - recommended
    case mlx4bit  // 4-bit quantized (~12GB)
}
```

### Qwen3ModelVariant

```swift
public enum Qwen3ModelVariant: String {
    case qwen3_4B_8bit  // For Klein 4B (~4GB)
    case qwen3_4B_4bit  // For Klein 4B (~2GB)
    case qwen3_8B_8bit  // For Klein 9B (~8GB)
    case qwen3_8B_4bit  // For Klein 9B (~4GB)
}
```

### KleinConfig

```swift
public enum KleinConfig {
    case klein4B  // Qwen3 4B, output dim 7680
    case klein9B  // Qwen3 8B, output dim 12288

    public var hiddenStateLayers: [Int]  // [9, 18, 27]
    public var maxSequenceLength: Int    // 512
    public var outputDimension: Int      // 7680 or 12288
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

    // Presets
    public static let balanced: GenerateParameters
    public static let creative: GenerateParameters
    public static let deterministic: GenerateParameters
}
```

### GenerationResult

```swift
public struct GenerationResult {
    public let text: String
    public let tokens: [Int]
    public let promptTokens: Int
    public let generatedTokens: Int
    public let totalTime: TimeInterval
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

    /// FLUX.2 dev default: layers 10, 20, 30 concatenated
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
    try await FluxTextEncoders.shared.loadModel(variant: .mlx8bit)
} catch FluxEncoderError.noWeightsFound {
    print("Model weights not found")
} catch {
    print("Error: \(error.localizedDescription)")
}
```

### Common Errors

- `FluxEncoderError.noWeightsFound` - Model files not found
- `FluxEncoderError.invalidConfig` - Invalid model configuration
- `FluxEncoderError.modelNotLoaded` - Model not loaded before operation
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
    let result = try? FluxTextEncoders.shared.generate(prompt: "Hello")
}
```
