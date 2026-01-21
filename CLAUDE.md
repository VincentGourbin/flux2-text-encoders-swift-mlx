# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Native Swift implementation of text encoders for FLUX.2 image generation on Apple Silicon using the MLX framework. Supports Mistral Small 3.2 (24B, for FLUX.2 dev/pro) and Qwen3 (4B/8B, for FLUX.2 Klein).

## Build Commands

```bash
# Build CLI (release)
swift build -c release

# Build macOS App
swift build -c release --product FluxEncodersApp

# Build all products (debug)
swift build

# Run tests
swift test

# Run a single test
swift test --filter FluxTextEncodersTests.testVersion
```

## CLI Usage

After building, the CLI is at `.build/release/FluxEncodersCLI`:

```bash
# Interactive chat (default command)
.build/release/FluxEncodersCLI chat

# Chat with Qwen3 (Klein)
.build/release/FluxEncodersCLI chat --qwen3

# Generate text
.build/release/FluxEncodersCLI generate "Your prompt" --temperature 0.7

# Extract FLUX.2 embeddings
.build/release/FluxEncodersCLI embed "Your text" --flux --output embeddings.bin

# Prompt upsampling (T2I)
.build/release/FluxEncodersCLI upsample "a cat" --mode t2i

# Prompt upsampling with image (I2I) - VLM analyzes the image
.build/release/FluxEncodersCLI upsample "make it dramatic" --mode i2i --image photo.jpg

# Vision analysis
.build/release/FluxEncodersCLI vision image.jpg "What's in this image?"

# Model management
.build/release/FluxEncodersCLI models
.build/release/FluxEncodersCLI models --download mistral-8bit
```

**Note:** For Metal GPU support, build with xcodebuild:
```bash
xcodebuild -scheme FluxEncodersCLI -configuration Release -destination "platform=macOS" -derivedDataPath .build/xcode build
# Then use: .build/xcode/Build/Products/Release/FluxEncodersCLI
```

## Architecture

The project has three products defined in `Package.swift`:

- **FluxTextEncoders** (library) - Core library exposing `FluxTextEncoders.shared` singleton
- **FluxEncodersCLI** (executable) - Command-line interface using ArgumentParser
- **FluxEncodersApp** (executable) - SwiftUI macOS application

### Library Structure (`Sources/FluxTextEncoders/`)

| Directory | Purpose |
|-----------|---------|
| `Configuration/` | Model configs (`MistralTextConfig`, `Qwen3Configuration`, `ModelRegistry`) |
| `Model/` | Mistral transformer (`MistralForCausalLM`, `MistralAttention`, `MistralMLP`) |
| `Model/Qwen3/` | Qwen3 transformer (`Qwen3ForCausalLM`, `Qwen3Attention`, `Qwen3MLP`) |
| `Vision/` | Pixtral vision encoder (`VisionEncoder`, `ImageProcessor`, `MistralVLM`) |
| `Embeddings/` | FLUX.2 embedding extraction (`EmbeddingExtractor`, `KleinEmbeddingExtractor`) |
| `Generation/` | Text generation (`MistralGenerator`, `Qwen3Generator`) |
| `Tokenizer/` | Tekken tokenizer implementation |
| `Loading/` | Model downloading and weight loading from HuggingFace |
| `Utils/` | Profiling (`FluxProfiler`) and debug utilities |

### Key Entry Point

`FluxTextEncoders.swift` contains the main `FluxTextEncoders` singleton class that orchestrates all functionality:
- Model loading: `loadModel()`, `loadVLMModel()`, `loadKleinModel()`
- Generation: `generate()`, `chat()`, `generateQwen3()`, `chatQwen3()`
- Embeddings: `extractFluxEmbeddings()`, `extractKleinEmbeddings()`
- Vision: `analyzeImage()`

## Key Dependencies

- `mlx-swift` - Apple's MLX framework for ML on Apple Silicon
- `swift-transformers` - HuggingFace tokenizers
- `swift-argument-parser` - CLI parsing

## FLUX.2 Embedding Dimensions

| Model | Hidden Layers | Output Dimension |
|-------|---------------|------------------|
| Mistral (dev/pro) | [10, 20, 30] | 15,360 (3 × 5120) |
| Qwen3 4B (Klein 4B) | [9, 18, 27] | 7,680 (3 × 2560) |
| Qwen3 8B (Klein 9B) | [9, 18, 27] | 12,288 (3 × 4096) |

Max sequence length for embeddings: 512 tokens

## Benchmarking

Python vs Swift benchmarks are in `Scripts/Benchmark/`:
```bash
# Run all benchmarks
./Scripts/Benchmark/run_all_benchmarks.sh

# Swift benchmarks only
./Scripts/Benchmark/benchmark_swift.sh

# Python benchmarks only
python Scripts/Benchmark/benchmark_python.py
```

## Environment Variables

- `HF_TOKEN` - HuggingFace token for model downloads
- `VLM_DEBUG` - Enable VLM debugging output
- `FLUX_DEBUG` - Enable general debug logging

## API Documentation

For detailed FLUX.2 API usage when integrating this library into another project, see:
- **[Documentation/FLUX2_API.md](Documentation/FLUX2_API.md)** - Complete API reference for FLUX.2 embeddings, upsampling, and VLM integration
