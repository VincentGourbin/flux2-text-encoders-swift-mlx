# Mistral Small 3.2 Swift MLX

Native Swift implementation of Mistral Small 3.2 (24B parameters) for Apple Silicon using the MLX framework.
Includes text-only and Vision-Language Model (VLM) capabilities.

## Features

- **Text Generation** - Streaming text generation with configurable parameters
- **Interactive Chat** - Multi-turn conversation with chat template support
- **Vision Analysis** - Image understanding via Pixtral vision encoder (VLM)
- **FLUX.2 Embeddings** - Extract embeddings compatible with FLUX.2 image generation
- **Native macOS App** - Full-featured SwiftUI application
- **CLI Tool** - Complete command-line interface
- **Model Management** - Automatic download and caching from HuggingFace

## Requirements

- macOS 14.0+ (Sonoma or later)
- Apple Silicon (M1, M2, M3, M4)
- Xcode 15.0+ (for building)
- ~12GB RAM minimum (8-bit model)

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/VincentGourbin/mistral-small-3.2-swift-mlx.git", from: "1.0.0")
]
```

### Build CLI

```bash
swift build -c release
```

### Build macOS App

```bash
swift build -c release --product MistralApp
```

## Usage

### Library API

```swift
import MistralCore

// Load model
try await MistralCore.shared.loadModel(variant: .mlx8bit)

// Generate text
let result = try MistralCore.shared.generate(prompt: "Hello") { token in
    print(token, terminator: "")
    return true
}

// Chat
let messages = [["role": "user", "content": "Hello!"]]
let response = try MistralCore.shared.chat(messages: messages)

// Vision (VLM)
try await MistralCore.shared.loadVLMModel(variant: .mlx4bit)
let analysis = try MistralCore.shared.analyzeImage(path: "photo.jpg", prompt: "Describe this")

// FLUX.2 Embeddings
let embeddings = try MistralCore.shared.extractFluxEmbeddings(prompt: "A cat")
```

### CLI Commands

```bash
# Chat mode (default)
mistral chat

# Text generation
mistral generate "Your prompt here" --temperature 0.7

# Vision analysis
mistral vision image.jpg "What's in this image?"

# Extract embeddings
mistral embed "Your text" --flux --output embeddings.bin

# Manage models
mistral models
mistral models --download 8bit
mistral models --delete 4bit
```

## Model Variants

| Variant | Size | RAM Required | Speed |
|---------|------|--------------|-------|
| BF16 | ~48GB | 64GB+ | Baseline |
| 8-bit | ~24GB | 32GB | ~Same |
| 4-bit | ~12GB | 16GB | Slightly slower |

## Screenshots

### Chat Interface
![Chat](screenshot/chat.png)

### Vision Analysis (VLM)
![Vision](screenshot/vision.png)

### FLUX.2 Embeddings
![Embeddings](screenshot/embeddings-flux2.png)

### Model Management
![Models](screenshot/models-management.png)

## Benchmarks

Performance comparison between Swift MLX and Python MLX on Apple Silicon (M-series).

### Text Generation (tokens/s)

| Quantization | Swift MLX | Python MLX | Swift Advantage |
|--------------|-----------|------------|-----------------|
| 4-bit | **11.8** | 6.4 | 1.84x faster |
| 6-bit | **9.3** | 5.3 | 1.75x faster |
| 8-bit | **8.0** | 4.2 | 1.90x faster |

**Swift MLX is ~1.8x faster for text generation across all quantizations.**

### Vision (VLM)

| Quantization | Swift MLX | Notes |
|--------------|-----------|-------|
| 4-bit | 2.5 tok/s | Correct output |
| 6-bit | 2.2 tok/s | Correct output |
| 8-bit | 2.1 tok/s | Correct output |

**Note:** Python `mlx_vlm` produces incorrect/hallucinated responses for this model (describes wrong images, shows raw BPE tokens). Swift VLM is the **only working implementation**.

### Summary

| Use Case | Recommendation |
|----------|----------------|
| Text Generation | **Swift MLX** (1.8x faster) |
| Vision/VLM | **Swift MLX** (only working implementation) |
| Embeddings | Either (similar performance) |

See [Scripts/Benchmark/results/COMPARISON_REPORT.md](Scripts/Benchmark/results/COMPARISON_REPORT.md) for detailed benchmark methodology and results.

## Architecture

```
Sources/
├── MistralCore/      # Core library
│   ├── Model/        # Mistral transformer
│   ├── Vision/       # Pixtral vision encoder
│   ├── Tokenizer/    # Tekken tokenizer
│   └── Embeddings/   # FLUX.2 extraction
├── MistralCLI/       # Command-line tool
└── MistralApp/       # SwiftUI macOS app
```

## API Documentation

See [Documentation/API.md](Documentation/API.md) for complete API reference.

## License

MIT License - see [LICENSE](LICENSE) file
