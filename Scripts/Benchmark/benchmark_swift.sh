#!/bin/bash
# Benchmark Swift MLX - Mistral Small 3.2
# Tests: Text, VLM, Embeddings across 4bit, 6bit, 8bit quantizations

set -e

PROJECT_DIR="/Users/vincent/Developpements/mistral-small-3.2-swift-mlx"
RESULTS_DIR="$PROJECT_DIR/Scripts/Benchmark/results"
TEST_IMAGE="$PROJECT_DIR/screenshot/vision.png"

# Prompts
TEXT_PROMPT="Write a haiku about artificial intelligence."
VLM_PROMPT="Describe this image in one sentence."
EMBED_PROMPT="A beautiful sunset over the ocean with vibrant colors"

# Quantizations to test
QUANTIZATIONS=("4bit" "6bit" "8bit" "bf16")

# Create results directory
mkdir -p "$RESULTS_DIR"

# Build CLI with xcodebuild
echo "Building Swift CLI with xcodebuild..."
cd "$PROJECT_DIR"

# Build in Release mode with macOS destination
xcodebuild -scheme MistralCLI -configuration Release -destination 'platform=macOS' -derivedDataPath "$PROJECT_DIR/.build/xcodebuild" build 2>&1 | tail -20

# Find the built CLI
CLI=$(find "$PROJECT_DIR/.build/xcodebuild" -name "MistralCLI" -type f 2>/dev/null | head -1)
if [ -z "$CLI" ]; then
    # Fallback to DerivedData
    CLI=$(find ~/Library/Developer/Xcode/DerivedData -name "MistralCLI" -type f 2>/dev/null | head -1)
fi

if [ -z "$CLI" ]; then
    echo "ERROR: Could not find MistralCLI after build"
    exit 1
fi

echo "CLI found at: $CLI"

echo "=============================================="
echo "Swift MLX Benchmark - Mistral Small 3.2"
echo "=============================================="
echo "Date: $(date)"
echo "Results: $RESULTS_DIR"
echo ""

# Results file
RESULTS_FILE="$RESULTS_DIR/swift_benchmark_$(date +%Y%m%d_%H%M%S).txt"

echo "Swift MLX Benchmark Results" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "=============================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for QUANT in "${QUANTIZATIONS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Testing $QUANT quantization"
    echo "=============================================="
    echo "" >> "$RESULTS_FILE"
    echo "=== $QUANT ===" >> "$RESULTS_FILE"

    # Text generation benchmark
    echo ""
    echo "--- Text Generation ($QUANT) ---"
    echo "" >> "$RESULTS_FILE"
    echo "Text Generation:" >> "$RESULTS_FILE"

    START_TIME=$(python3 -c "import time; print(time.time())")
    "$CLI" generate -m "$QUANT" --max-tokens 100 --no-stream "$TEXT_PROMPT" 2>&1 | tee -a "$RESULTS_FILE"
    END_TIME=$(python3 -c "import time; print(time.time())")
    DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
    echo "Total time: ${DURATION}s" >> "$RESULTS_FILE"
    echo ""

    # VLM benchmark (vision)
    echo ""
    echo "--- VLM ($QUANT) ---"
    echo "" >> "$RESULTS_FILE"
    echo "VLM (Vision):" >> "$RESULTS_FILE"

    START_TIME=$(python3 -c "import time; print(time.time())")
    "$CLI" vision -m "$QUANT" --max-tokens 50 --no-stream "$TEST_IMAGE" "$VLM_PROMPT" 2>&1 | tee -a "$RESULTS_FILE"
    END_TIME=$(python3 -c "import time; print(time.time())")
    DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
    echo "Total time: ${DURATION}s" >> "$RESULTS_FILE"
    echo ""

    # Embeddings benchmark
    echo ""
    echo "--- Embeddings ($QUANT) ---"
    echo "" >> "$RESULTS_FILE"
    echo "Embeddings (FLUX.2):" >> "$RESULTS_FILE"

    START_TIME=$(python3 -c "import time; print(time.time())")
    "$CLI" embed -m "$QUANT" --flux "$EMBED_PROMPT" 2>&1 | tee -a "$RESULTS_FILE"
    END_TIME=$(python3 -c "import time; print(time.time())")
    DURATION=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
    echo "Total time: ${DURATION}s" >> "$RESULTS_FILE"
    echo ""
done

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="
