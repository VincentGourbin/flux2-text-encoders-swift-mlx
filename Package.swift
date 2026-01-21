// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FluxTextEncoders",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "FluxTextEncoders", targets: ["FluxTextEncoders"]),
        .executable(name: "FluxEncodersCLI", targets: ["FluxEncodersCLI"]),
        .executable(name: "FluxEncodersApp", targets: ["FluxEncodersApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "FluxTextEncoders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "FluxEncodersCLI",
            dependencies: [
                "FluxTextEncoders",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "FluxEncodersApp",
            dependencies: [
                "FluxTextEncoders",
            ],
            exclude: ["Resources/Info.plist"]
        ),
        .testTarget(
            name: "FluxTextEncodersTests",
            dependencies: ["FluxTextEncoders"]
        ),
    ]
)
