#!/bin/bash

# Script to create a proper macOS .app bundle from the MistralApp executable

set -e

APP_NAME="Mistral"
BUNDLE_DIR="$APP_NAME.app"
# Find the executable - check both possible locations
if [ -f ".build/arm64-apple-macosx/debug/MistralApp" ]; then
    EXECUTABLE_PATH=".build/arm64-apple-macosx/debug/MistralApp"
elif [ -f ".build/debug/MistralApp" ]; then
    EXECUTABLE_PATH=".build/debug/MistralApp"
elif [ -f ".build/arm64-apple-macosx/release/MistralApp" ]; then
    EXECUTABLE_PATH=".build/arm64-apple-macosx/release/MistralApp"
elif [ -f ".build/release/MistralApp" ]; then
    EXECUTABLE_PATH=".build/release/MistralApp"
else
    echo "Error: MistralApp executable not found. Run 'swift build --product MistralApp' first."
    exit 1
fi

echo "Creating $BUNDLE_DIR..."

# Clean up existing bundle
rm -rf "$BUNDLE_DIR"

# Create bundle structure
mkdir -p "$BUNDLE_DIR/Contents/MacOS"
mkdir -p "$BUNDLE_DIR/Contents/Resources"

# Copy executable
cp "$EXECUTABLE_PATH" "$BUNDLE_DIR/Contents/MacOS/$APP_NAME"

# Create Info.plist
cat > "$BUNDLE_DIR/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>Mistral</string>
    <key>CFBundleIdentifier</key>
    <string>com.mistral.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Mistral</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF

# Create PkgInfo
echo -n "APPL????" > "$BUNDLE_DIR/Contents/PkgInfo"

echo "Done! App bundle created at: $BUNDLE_DIR"
echo ""
echo "To run: open $BUNDLE_DIR"
echo "Or double-click Mistral.app in Finder"
