/**
 * ContentView.swift
 * Main content view for Mistral App
 */

import SwiftUI
import MistralCore
import MLX

struct ContentView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var chatViewModel = ChatViewModel()
    @State private var selectedTab = 0

    var body: some View {
        NavigationSplitView {
            // Sidebar
            List(selection: Binding(
                get: { selectedTab },
                set: { selectedTab = $0 }
            )) {
                Section("Inference") {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                        .tag(0)
                    Label("Generate", systemImage: "text.cursor")
                        .tag(1)
                    Label("Vision", systemImage: "eye")
                        .tag(2)
                }

                Section("Tools") {
                    Label("FLUX.2 Tools", systemImage: "cube.transparent")
                        .tag(3)
                    Label("Models", systemImage: "square.stack.3d.down.right")
                        .tag(4)
                }
            }
            .listStyle(.sidebar)
            .frame(minWidth: 180)

        } detail: {
            // Main content
            VStack(spacing: 0) {
                // Model status bar
                ModelStatusBar()
                    .environmentObject(modelManager)

                Divider()

                // Content based on selection
                Group {
                    switch selectedTab {
                    case 0:
                        ChatView(viewModel: chatViewModel)
                    case 1:
                        GenerateView()
                    case 2:
                        VisionView()
                    case 3:
                        FluxToolsView()
                    case 4:
                        ModelsManagementView()
                    default:
                        ChatView(viewModel: chatViewModel)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Mistral Small 3.2")
    }
}

// MARK: - Model Status Bar

struct ModelStatusBar: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false

    var body: some View {
        HStack {
            // Model status indicator
            Circle()
                .fill(modelManager.isLoaded ? Color.green : Color.red)
                .frame(width: 8, height: 8)

            Text(modelManager.isLoaded ? "Model Loaded" : "Model Not Loaded")
                .font(.caption)
                .foregroundColor(.secondary)

            // VLM indicator (always shown when loaded since we now load VLM by default)
            if modelManager.isVLMLoaded {
                Text("VLM")
                    .font(.caption.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.blue.opacity(0.2))
                    .foregroundColor(.blue)
                    .cornerRadius(4)
            }

            if modelManager.isLoading {
                ProgressView()
                    .scaleEffect(0.6)
                Text(modelManager.loadingMessage)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Detailed profiling toggle (enables memory logging for text & VLM)
            Toggle("Detailed Profiling", isOn: $detailedProfiling)
                .toggleStyle(.checkbox)
                .font(.caption)
                .onChange(of: detailedProfiling) { _, newValue in
                    MistralProfiler.shared.isEnabled = newValue
                }
                .help("Enable detailed profiling and memory logging")

            Divider()
                .frame(height: 20)
                .padding(.horizontal, 8)

            // Model variant picker - only downloaded models are selectable
            HStack(spacing: 0) {
                Text("Model")
                    .foregroundColor(.secondary)
                    .padding(.trailing, 8)

                ForEach(ModelVariant.allCases, id: \.self) { variant in
                    let isDownloaded = isVariantDownloaded(variant)
                    let isSelected = modelManager.selectedVariant == variant
                    Button(action: {
                        if isDownloaded {
                            modelManager.selectedVariant = variant
                        }
                    }) {
                        Text(variant.shortName)
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                isSelected
                                    ? Color.accentColor
                                    : (isDownloaded ? Color.gray.opacity(0.3) : Color.gray.opacity(0.1))
                            )
                            .foregroundColor(
                                isSelected
                                    ? .white
                                    : (isDownloaded ? .primary : .secondary.opacity(0.5))
                            )
                            .cornerRadius(6)
                    }
                    .buttonStyle(.plain)
                    .disabled(!isDownloaded)
                    .help(isDownloaded ? variant.displayName : "\(variant.displayName) - Not downloaded")
                }
            }

            // Load/Unload button
            Button(action: {
                Task {
                    if modelManager.isLoaded {
                        modelManager.unloadModel()
                    } else {
                        await modelManager.loadModel()
                    }
                }
            }) {
                Text(modelManager.isLoaded ? "Unload" : "Load Model")
            }
            .disabled(modelManager.isLoading || (!modelManager.isLoaded && modelManager.selectedVariant == nil))
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor))
        .onAppear {
            // Sync profiler state on appear
            MistralProfiler.shared.isEnabled = detailedProfiling
        }
    }

    private func isVariantDownloaded(_ variant: ModelVariant) -> Bool {
        guard let model = ModelRegistry.shared.model(withVariant: variant) else { return false }
        return modelManager.downloadedModels.contains(model.id)
    }
}

// MARK: - Chat View

struct ChatView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false
    @ObservedObject var viewModel: ChatViewModel
    @State private var showSettings = false

    // Mistral Small 3.2 supports up to 131K context
    private let maxGenerationTokens = 8192

    var body: some View {
        VStack(spacing: 0) {
            // Messages list
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                        }

                        if viewModel.isGenerating {
                            HStack {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Generating...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let lastMessage = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Stats bar - show live during generation, final stats after
            if viewModel.isGenerating {
                LiveStatsBarView(tokenCount: viewModel.currentTokenCount)
            } else if let stats = viewModel.lastGenerationStats {
                StatsBarView(stats: stats, profileSummary: viewModel.lastProfileSummary)
            }

            // Settings bar (collapsible)
            if showSettings {
                VStack(spacing: 8) {
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Tokens: \(viewModel.maxTokens)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: Binding(
                                get: { Double(viewModel.maxTokens) },
                                set: { viewModel.maxTokens = Int($0) }
                            ), in: 64...Double(maxGenerationTokens), step: 64)
                            .frame(width: 200)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Temperature: \(String(format: "%.1f", viewModel.temperature))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: Binding(
                                get: { Double(viewModel.temperature) },
                                set: { viewModel.temperature = Float($0) }
                            ), in: 0...2, step: 0.1)
                            .frame(width: 150)
                        }

                        Spacer()

                        Button("Clear Chat") {
                            viewModel.clearChat()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
            }

            Divider()

            // Input area
            HStack(spacing: 12) {
                Button(action: { withAnimation { showSettings.toggle() } }) {
                    Image(systemName: showSettings ? "gearshape.fill" : "gearshape")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Toggle settings")

                TextField("Type a message...", text: $viewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)
                    .onSubmit {
                        sendMessage()
                    }

                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(viewModel.inputText.isEmpty || viewModel.isGenerating || !modelManager.isLoaded)
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
        }
        .onAppear {
            viewModel.modelManager = modelManager
        }
    }

    private func sendMessage() {
        guard !viewModel.inputText.isEmpty else { return }
        Task {
            await viewModel.sendMessage()
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .assistant {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mistral")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(message.content)
                        .padding(12)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)
                }
                Spacer()
            } else {
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text("You")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(message.content)
                        .padding(12)
                        .background(Color.gray.opacity(0.2))
                        .cornerRadius(12)
                }
            }
        }
        .id(message.id)
    }
}

// MARK: - Live Stats Bar View (during generation)

struct LiveStatsBarView: View {
    let tokenCount: Int

    var body: some View {
        HStack(spacing: 16) {
            ProgressView()
                .scaleEffect(0.7)

            Image(systemName: "text.cursor")
                .foregroundStyle(.blue)

            Text("Generating (\(tokenCount) tokens)...")
                .foregroundStyle(.blue)
                .fontWeight(.medium)

            Spacer()
        }
        .font(.caption)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Stats Bar View (final)

struct StatsBarView: View {
    let stats: GenerationStats
    let profileSummary: ProfileSummary?
    @State private var showProfileDetails = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 20) {
                Label("\(stats.tokenCount) tokens", systemImage: "number")
                Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
                Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")

                Spacer()

                if profileSummary != nil {
                    Button(action: { showProfileDetails.toggle() }) {
                        Label(showProfileDetails ? "Hide Profile" : "Show Profile",
                              systemImage: showProfileDetails ? "chevron.up" : "chevron.down")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.blue)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal)
            .padding(.vertical, 8)

            if showProfileDetails, let summary = profileSummary {
                ProfileDetailsView(summary: summary)
            }
        }
        .background(.ultraThinMaterial)
    }
}

// MARK: - Profile Details View

struct ProfileDetailsView: View {
    let summary: ProfileSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()

            // Device info
            HStack {
                Text("Device:")
                    .foregroundStyle(.secondary)
                Text(summary.deviceInfo.architecture)
                    .fontWeight(.medium)
                Spacer()
                Text("RAM: \(formatBytesUI(summary.deviceInfo.memorySize))")
                    .foregroundStyle(.secondary)
            }
            .font(.caption)

            Divider()

            // Steps table header
            HStack {
                Text("Step")
                    .frame(width: 140, alignment: .leading)
                Text("Time")
                    .frame(width: 70, alignment: .trailing)
                Text("MLX \u{0394}")
                    .frame(width: 80, alignment: .trailing)
                Text("Process \u{0394}")
                    .frame(width: 80, alignment: .trailing)
            }
            .font(.caption2.bold())
            .foregroundStyle(.secondary)

            // Steps
            ForEach(Array(summary.steps.enumerated()), id: \.offset) { _, step in
                HStack {
                    Text(step.name)
                        .frame(width: 140, alignment: .leading)
                        .lineLimit(1)
                    Text(String(format: "%.3fs", step.duration))
                        .frame(width: 70, alignment: .trailing)
                    Text(formatDeltaBytesUI(step.endMemory.mlxActive - step.startMemory.mlxActive))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.mlxActive > step.startMemory.mlxActive ? .orange : .green)
                    Text(formatDeltaBytesUI(Int(step.endMemory.processFootprint - step.startMemory.processFootprint)))
                        .frame(width: 80, alignment: .trailing)
                        .foregroundStyle(step.endMemory.processFootprint > step.startMemory.processFootprint ? .orange : .green)
                }
                .font(.caption)
            }

            Divider()

            // Totals
            HStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Peak")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.peakMemoryUsed))
                        .fontWeight(.medium)
                        .foregroundStyle(.orange)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Active")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.finalSnapshot.mlxActive))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Cache")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(summary.finalSnapshot.mlxCache))
                        .fontWeight(.medium)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text("Process")
                        .foregroundStyle(.secondary)
                    Text(formatBytesUI(Int(summary.finalSnapshot.processFootprint)))
                        .fontWeight(.medium)
                        .foregroundStyle(.blue)
                }

                Spacer()
            }
            .font(.caption)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
    }
}

// Helper functions for formatting
private func formatBytesUI(_ bytes: Int) -> String {
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

private func formatDeltaBytesUI(_ bytes: Int) -> String {
    let sign = bytes >= 0 ? "+" : ""
    return sign + formatBytesUI(bytes)
}

// MARK: - Generate View

struct GenerateView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("detailedProfiling") private var detailedProfiling = false
    @State private var prompt = ""
    @State private var output = ""
    @State private var profilingInfo = ""
    @State private var isGenerating = false
    @State private var maxTokens: Double = 512
    @State private var temperature = 0.7

    // Mistral Small 3.2 supports up to 131K context, but we limit generation to 8K for practical use
    private let maxGenerationTokens: Double = 8192

    var body: some View {
        VStack(spacing: 16) {
            // Prompt input
            GroupBox("Prompt") {
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 100)
            }

            // Parameters
            HStack {
                GroupBox("Max Tokens: \(Int(maxTokens))") {
                    Slider(value: $maxTokens, in: 64...maxGenerationTokens, step: 64)
                    HStack {
                        Text("64")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(maxGenerationTokens))")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                GroupBox("Temperature: \(String(format: "%.1f", temperature))") {
                    Slider(value: $temperature, in: 0...2, step: 0.1)
                    HStack {
                        Text("0")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("2")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Generate button
            Button(action: generate) {
                HStack {
                    if isGenerating {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    Text(isGenerating ? "Generating..." : "Generate")
                }
            }
            .disabled(prompt.isEmpty || isGenerating || !modelManager.isLoaded)

            // Output
            GroupBox("Output") {
                ScrollView {
                    Text(output)
                        .font(.body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 150)
            }

            // Profiling info (shown when enabled)
            if detailedProfiling && !profilingInfo.isEmpty {
                GroupBox("Profiling") {
                    ScrollView {
                        Text(profilingInfo)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 100)
                }
            }

            Spacer()
        }
        .padding()
    }

    private func generate() {
        guard !prompt.isEmpty else { return }
        isGenerating = true
        output = ""
        profilingInfo = ""

        // Reset profiler
        MistralProfiler.shared.reset()

        Task {
            do {
                let params = GenerateParameters(
                    maxTokens: Int(maxTokens),
                    temperature: Float(temperature)
                )

                let result = try MistralCore.shared.generate(
                    prompt: prompt,
                    parameters: params
                ) { token in
                    Task { @MainActor in
                        output += token
                    }
                    return true
                }

                await MainActor.run {
                    output = result.text
                    isGenerating = false

                    // Get profiling info if enabled
                    if detailedProfiling {
                        let metrics = MistralProfiler.shared.getMetrics()
                        profilingInfo = metrics.compactSummary
                    }
                }
            } catch {
                await MainActor.run {
                    output = "Error: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}

// MARK: - Vision View

struct VisionView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var selectedImage: NSImage?
    @State private var prompt = "Describe this image in detail."
    @State private var output = ""
    @State private var isProcessing = false
    @State private var currentTokenCount = 0
    @State private var generationStats: GenerationStats?
    @State private var maxTokens: Double = 1024
    @State private var temperature = 0.7

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                // Image drop zone
                GroupBox("Image") {
                    ZStack {
                        if let image = selectedImage {
                            Image(nsImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        } else {
                            VStack {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.largeTitle)
                                    .foregroundColor(.secondary)
                                Text("Drop image here or click to select")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .frame(minWidth: 300, minHeight: 300)
                    .onDrop(of: [.image], isTargeted: nil) { providers in
                        loadImage(from: providers)
                        return true
                    }
                    .onTapGesture {
                        selectImage()
                    }
                }

                // Prompt and output
                VStack(spacing: 16) {
                    GroupBox("Prompt") {
                        TextField("What do you want to know about this image?", text: $prompt)
                            .textFieldStyle(.plain)
                    }

                    // Parameters
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Tokens: \(Int(maxTokens))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: $maxTokens, in: 128...4096, step: 128)
                                .frame(width: 150)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Temperature: \(String(format: "%.1f", temperature))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Slider(value: $temperature, in: 0...2, step: 0.1)
                                .frame(width: 120)
                        }
                    }

                    HStack {
                        // VLM status - model is now loaded with VLM by default
                        if !modelManager.isVLMLoaded {
                            Label("Load model from top bar", systemImage: "arrow.up.circle")
                                .foregroundStyle(.secondary)
                                .font(.caption)
                        } else {
                            Label("VLM Ready", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.caption)
                        }

                        Spacer()

                        Button(action: processImage) {
                            HStack {
                                if isProcessing {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                }
                                Text(isProcessing ? "Analyzing..." : "Analyze Image")
                            }
                        }
                        .disabled(selectedImage == nil || isProcessing || !modelManager.isVLMLoaded)
                    }

                    GroupBox("Response") {
                        ScrollView {
                            Text(output.isEmpty ? "Load a model from the top bar, select an image, and click Analyze" : output)
                                .font(.body)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .foregroundColor(output.isEmpty ? .secondary : .primary)
                                .textSelection(.enabled)
                        }
                        .frame(minHeight: 200)
                    }

                    // Stats bar
                    if isProcessing {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text("Generating (\(currentTokenCount) tokens)...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                    } else if let stats = generationStats {
                        HStack {
                            Label("\(stats.tokenCount) tokens", systemImage: "number")
                            Label(String(format: "%.1fs", stats.duration), systemImage: "clock")
                            Label(String(format: "%.1f tok/s", stats.tokensPerSecond), systemImage: "speedometer")
                            Spacer()
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                    }
                }
            }

            Spacer()
        }
        .padding()
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK, let url = panel.url {
            selectedImage = NSImage(contentsOf: url)
        }
    }

    private func loadImage(from providers: [NSItemProvider]) {
        guard let provider = providers.first else { return }
        provider.loadObject(ofClass: NSImage.self) { image, _ in
            if let image = image as? NSImage {
                DispatchQueue.main.async {
                    selectedImage = image
                }
            }
        }
    }

    private func processImage() {
        guard let image = selectedImage else { return }

        isProcessing = true
        output = ""
        currentTokenCount = 0
        generationStats = nil

        let startTime = Date()
        let params = GenerateParameters(
            maxTokens: Int(maxTokens),
            temperature: Float(temperature)
        )
        let userPrompt = prompt

        // Run inference on background thread to keep UI responsive
        Task.detached(priority: .userInitiated) {
            do {
                let result = try MistralCore.shared.analyzeImage(
                    image: image,
                    prompt: userPrompt,
                    parameters: params
                ) { token in
                    // Stream tokens to UI
                    Task { @MainActor in
                        output += token
                        currentTokenCount += 1
                    }
                    return true
                }

                await MainActor.run {
                    // Don't overwrite streamed output, just update stats
                    isProcessing = false
                    generationStats = GenerationStats(
                        tokenCount: result.generatedTokens,
                        duration: Date().timeIntervalSince(startTime)
                    )
                }
            } catch {
                await MainActor.run {
                    output = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

// MARK: - FLUX.2 Tools View

enum FluxToolMode: String, CaseIterable {
    case embeddings = "Embeddings"
    case upsamplingT2I = "Upsampling T2I"
    case upsamplingI2I = "Upsampling I2I"

    var icon: String {
        switch self {
        case .embeddings: return "cube.transparent"
        case .upsamplingT2I: return "wand.and.stars"
        case .upsamplingI2I: return "photo.on.rectangle"
        }
    }

    var description: String {
        switch self {
        case .embeddings: return "Extract FLUX.2 embeddings (512×15360)"
        case .upsamplingT2I: return "Enhance text prompts for image generation"
        case .upsamplingI2I: return "Generate image editing instructions"
        }
    }
}

struct FluxToolsView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var selectedMode: FluxToolMode = .embeddings
    @State private var prompt = ""
    @State private var imagePath: String?
    @State private var outputText = ""
    @State private var isProcessing = false
    @State private var lastEmbeddings: MLXArray?

    var body: some View {
        VStack(spacing: 16) {
            // Mode selector
            HStack {
                Picker("Mode", selection: $selectedMode) {
                    ForEach(FluxToolMode.allCases, id: \.self) { mode in
                        Label(mode.rawValue, systemImage: mode.icon).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Spacer()

                Text(selectedMode.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.purple.opacity(0.1))
                    .cornerRadius(4)
            }

            // Image picker (for I2I mode - placeholder for future VLM integration)
            if selectedMode == .upsamplingI2I {
                GroupBox("Reference Image (optional)") {
                    HStack {
                        if let path = imagePath {
                            Text(URL(fileURLWithPath: path).lastPathComponent)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("Clear") { imagePath = nil }
                        } else {
                            Text("No image selected")
                                .foregroundStyle(.secondary)
                            Spacer()
                            Button("Select...") { selectImage() }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            // Prompt input
            GroupBox(selectedMode == .embeddings ? "Text to Embed" : "Input Prompt") {
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 80)
            }

            // System prompt info (for upsampling modes)
            if selectedMode != .embeddings {
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("System Prompt (BFL Official)")
                            .font(.caption.bold())
                        let fluxMode: FluxConfig.Mode = selectedMode == .upsamplingT2I ? .upsamplingT2I : .upsamplingI2I
                        Text(FluxConfig.systemMessage(for: fluxMode).prefix(150) + "...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            // Action buttons
            HStack {
                Button(action: processAction) {
                    HStack {
                        if isProcessing {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isProcessing ? "Processing..." : actionButtonTitle)
                    }
                }
                .disabled(prompt.isEmpty || isProcessing || (!modelManager.isLoaded && !modelManager.isVLMLoaded))

                Spacer()

                if selectedMode == .embeddings {
                    Button("Export...") { exportEmbeddings() }
                        .disabled(lastEmbeddings == nil)
                } else {
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(outputText, forType: .string)
                    }
                    .disabled(outputText.isEmpty)
                }
            }

            // Output
            GroupBox(selectedMode == .embeddings ? "Embeddings Info" : "Output") {
                ScrollView {
                    Text(outputText.isEmpty ? placeholderText : outputText)
                        .font(selectedMode == .embeddings ? .system(.body, design: .monospaced) : .body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(outputText.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 180)
            }

            Spacer()
        }
        .padding()
        .onChange(of: selectedMode) { _, _ in
            outputText = ""
            lastEmbeddings = nil
        }
    }

    private var actionButtonTitle: String {
        switch selectedMode {
        case .embeddings: return "Extract Embeddings"
        case .upsamplingT2I, .upsamplingI2I: return "Upsample Prompt"
        }
    }

    private var placeholderText: String {
        switch selectedMode {
        case .embeddings: return "Enter text and click Extract to see embeddings info"
        case .upsamplingT2I: return "Enter a simple prompt to enhance it for FLUX.2"
        case .upsamplingI2I: return "Describe the edit you want (e.g., 'make the sky more dramatic')"
        }
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK {
            imagePath = panel.url?.path
        }
    }

    private func processAction() {
        isProcessing = true
        outputText = ""

        Task {
            do {
                switch selectedMode {
                case .embeddings:
                    let embeddings = try MistralCore.shared.extractFluxEmbeddings(prompt: prompt)
                    await MainActor.run {
                        lastEmbeddings = embeddings
                        let flatEmbeddings = embeddings.reshaped([-1])
                        let firstValues = flatEmbeddings[0..<min(10, flatEmbeddings.size)].asArray(Float.self)
                        outputText = """
                        === FLUX.2 Embeddings ===

                        Shape: \(embeddings.shape)
                        Dtype: \(embeddings.dtype)
                        Total: \(embeddings.shape.reduce(1, *)) elements

                        Format:
                        • Layers: 10, 20, 30 (concatenated)
                        • Sequence: 512 tokens (LEFT-padded)
                        • Dims: 5,120 × 3 = 15,360

                        First 10 values:
                        \(firstValues.map { String(format: "%.6f", $0) }.joined(separator: ", "))

                        ✅ Ready for FLUX.2
                        """
                        isProcessing = false
                    }

                case .upsamplingT2I, .upsamplingI2I:
                    let fluxMode: FluxConfig.Mode = selectedMode == .upsamplingT2I ? .upsamplingT2I : .upsamplingI2I
                    let messages = FluxConfig.buildMessages(prompt: prompt, mode: fluxMode)

                    var result = ""
                    _ = try MistralCore.shared.chat(
                        messages: messages,
                        parameters: GenerateParameters(maxTokens: 500, temperature: 0.7)
                    ) { token in
                        result += token
                        return true
                    }

                    await MainActor.run {
                        outputText = result
                        isProcessing = false
                    }
                }
            } catch {
                await MainActor.run {
                    outputText = "Error: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }

    private func exportEmbeddings() {
        guard let embeddings = lastEmbeddings else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.data]
        panel.nameFieldStringValue = "flux_embeddings.bin"

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try MistralCore.shared.exportEmbeddings(embeddings, to: url.path, format: .binary)
                outputText += "\n\n✅ Exported to: \(url.lastPathComponent)"
            } catch {
                outputText += "\n\n❌ Export failed: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Models Management View

struct ModelsManagementView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var modelToDelete: ModelInfo?
    @State private var showDeleteConfirmation = false
    @State private var memoryRefreshTrigger = false

    var body: some View {
        VStack(spacing: 0) {
            // Memory status bar
            HStack(spacing: 16) {
                Label("MLX Memory", systemImage: "memorychip")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    Text("Active: \(ModelManager.formatBytes(modelManager.memoryStats.active))")
                    Text("Cache: \(ModelManager.formatBytes(modelManager.memoryStats.cache))")
                        .foregroundStyle(modelManager.memoryStats.cache > 0 ? .orange : .secondary)
                    Text("Peak: \(ModelManager.formatBytes(modelManager.memoryStats.peak))")
                        .foregroundStyle(.blue)
                }
                .font(.caption.monospaced())

                Spacer()

                Button(action: {
                    modelManager.clearCache()
                    modelManager.resetPeakMemory()
                    memoryRefreshTrigger.toggle()
                }) {
                    Label("Clear Cache", systemImage: "trash.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(modelManager.memoryStats.cache == 0)
                .help("Clear MLX recyclable cache")

                Button(action: {
                    modelManager.unloadModel()
                    memoryRefreshTrigger.toggle()
                }) {
                    Label("Unload Model", systemImage: "xmark.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(.orange)
                .disabled(!modelManager.isLoaded)
                .help("Unload model to free all GPU memory")
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)
            .id(memoryRefreshTrigger)

            Divider()

            // Download progress bar
            if modelManager.isDownloading {
                VStack(spacing: 4) {
                    ProgressView(value: modelManager.downloadProgress)
                        .progressViewStyle(.linear)
                    Text(modelManager.downloadMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)

                Divider()
            }

            // Toolbar
            HStack {
                Text("Downloaded Models")
                    .font(.headline)
                Spacer()

                Button(action: {
                    NSWorkspace.shared.open(ModelManager.modelsCacheDirectory)
                }) {
                    Label("Open Cache Folder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Open models cache folder in Finder")

                Button(action: { modelManager.refreshDownloadedModels() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh")
            }
            .padding()

            Divider()

            if modelManager.downloadedModels.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "square.stack.3d.down.right")
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary.opacity(0.5))
                    Text("No models downloaded")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("Download models from the list below to get started")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(modelManager.availableModels.filter { modelManager.downloadedModels.contains($0.id) }, id: \.id) { model in
                        ModelRowView(
                            model: model,
                            size: modelManager.modelSizes[model.id],
                            isLoaded: modelManager.currentLoadedModelId == model.id,
                            onDelete: {
                                modelToDelete = model
                                showDeleteConfirmation = true
                            },
                            onLoad: {
                                Task { await modelManager.loadModel(model.id) }
                            }
                        )
                    }
                }
                .listStyle(.inset)
            }

            Divider()

            // Available models section
            VStack(alignment: .leading, spacing: 8) {
                Text("Available Models")
                    .font(.headline)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(modelManager.availableModels.filter { !modelManager.downloadedModels.contains($0.id) }, id: \.id) { model in
                            AvailableModelCard(model: model, modelManager: modelManager)
                        }

                        // Show message if all downloaded
                        if modelManager.availableModels.allSatisfy({ modelManager.downloadedModels.contains($0.id) }) {
                            Text("All models downloaded!")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding()
                        }
                    }
                    .padding(.horizontal, 4)
                }
            }
            .padding()
            .background(.ultraThinMaterial)
        }
        .alert("Delete Model", isPresented: $showDeleteConfirmation, presenting: modelToDelete) { model in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                Task {
                    try? await modelManager.deleteModel(model.id)
                }
            }
        } message: { model in
            Text("Are you sure you want to delete \(model.name)? This cannot be undone.")
        }
    }
}

// MARK: - Model Row View

struct ModelRowView: View {
    let model: ModelInfo
    let size: Int64?
    let isLoaded: Bool
    let onDelete: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .font(.headline)
                    if isLoaded {
                        Text("Loaded")
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .cornerRadius(4)
                    }
                }
                HStack(spacing: 8) {
                    Text(model.variant.displayName)
                    Text("•")
                    Text(model.parameters)
                    if let size = size {
                        Text("•")
                        Text(ModelDownloader.formatSize(size))
                            .foregroundStyle(.blue)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Spacer()

            if !isLoaded {
                Button("Load") {
                    onLoad()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            Button(action: onDelete) {
                Image(systemName: "trash")
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .disabled(isLoaded)
            .help(isLoaded ? "Unload model first" : "Delete model")
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Available Model Card

struct AvailableModelCard: View {
    let model: ModelInfo
    @ObservedObject var modelManager: ModelManager

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(model.name)
                .font(.caption.bold())
                .lineLimit(1)

            HStack(spacing: 4) {
                Text(model.variant.estimatedSize)
                Text("•")
                Text(model.parameters)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            Button(action: {
                Task { await modelManager.downloadModel(model.id) }
            }) {
                HStack {
                    Image(systemName: "arrow.down.circle")
                    Text("Download")
                }
                .font(.caption)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(modelManager.isDownloading)
        }
        .padding(10)
        .frame(width: 160)
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var modelManager: ModelManager
    @AppStorage("hfToken") private var hfToken = ""

    var body: some View {
        Form {
            Section("HuggingFace") {
                SecureField("HF Token", text: $hfToken)
                    .textFieldStyle(.roundedBorder)
            }

            Section("Model") {
                Text("Variant: \(modelManager.selectedVariant?.rawValue ?? "None")")
                Text("Status: \(modelManager.isLoaded ? "Loaded" : "Not Loaded")")
            }
        }
        .padding()
        .frame(width: 400, height: 200)
    }
}

#Preview {
    ContentView()
        .environmentObject(ModelManager())
}
