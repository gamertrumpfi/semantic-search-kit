// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SemanticSearchKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "SemanticSearchKit", targets: ["SemanticSearchKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "SemanticSearchKit",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
            ]
        ),
        .testTarget(
            name: "SemanticSearchKitTests",
            dependencies: ["SemanticSearchKit"],
            resources: [
                .copy("Fixtures/corpus.json"),
                .copy("Fixtures/embeddings_index.json"),
                .copy("Fixtures/query_embeddings.json"),
            ]
        ),

        // -----------------------------------------------------------------
        // Pre-built CoreML model (optional)
        //
        // A pre-converted MultilingualE5Small.mlpackage (~224 MB, fp16)
        // is distributed as a GitHub release artifact to avoid bloating
        // the repository.
        //
        // Consumers who want the bundled model can uncomment this target
        // and add "MultilingualE5Small" as a dependency to their app
        // target. Otherwise, use `ssk export` to produce a custom model
        // from your own corpus.
        //
        // To enable:
        //   1. Download the .mlpackage.zip from the GitHub release
        //   2. Uncomment the binaryTarget below
        //   3. Update the url and checksum to match the release asset
        //
        // .binaryTarget(
        //     name: "MultilingualE5Small",
        //     url: "https://github.com/gamertrumpfi/semantic-search-kit/releases/download/v0.1.0/MultilingualE5Small.mlpackage.zip",
        //     checksum: "<sha256-checksum>"
        // ),
        // -----------------------------------------------------------------
    ]
)
