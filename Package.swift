// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MistralSwift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MistralCore", targets: ["MistralCore"]),
        .executable(name: "MistralCLI", targets: ["MistralCLI"]),
        .executable(name: "MistralApp", targets: ["MistralApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.14"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "MistralCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MistralCLI",
            dependencies: [
                "MistralCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "MistralApp",
            dependencies: [
                "MistralCore",
            ],
            exclude: ["Resources/Info.plist"]
        ),
        .testTarget(
            name: "MistralCoreTests",
            dependencies: ["MistralCore"]
        ),
    ]
)
