import Foundation
import os

// MARK: - SemanticSearchKit

/// Central configuration namespace for the SDK.
///
/// Call ``configure(modelName:bundle:embeddingsFileName:queryPrefix:dimension:maxSeqLen:)``
/// once at app launch — before using ``SemanticSearchEngine``, ``EmbeddingStorage``,
/// or ``QueryEmbedder``.
///
/// ```swift
/// SemanticSearchKit.configure(modelName: "MyAppSearch", bundle: .main)
/// ```
public enum SemanticSearchKit {

    /// Configures the SDK with your CoreML model name and bundle.
    ///
    /// This must be called **exactly once**, before any search operations.
    /// Subsequent calls are ignored and log a warning.
    ///
    /// - Parameters:
    ///   - modelName: The compiled CoreML model's base name (without `.mlmodelc` /
    ///     `.mlpackage` extension) as it appears in your bundle.
    ///   - bundle: The bundle containing both the model and the
    ///     `embeddings_index.json` file. Defaults to `.main`.
    ///   - embeddingsFileName: Base name of the embeddings JSON file
    ///     (without `.json` extension). Defaults to `"embeddings_index"`.
    ///   - queryPrefix: The query prefix required by the model family
    ///     (e.g. `"query: "` for E5). Defaults to `"query: "`.
    ///   - dimension: Output embedding dimension. Defaults to `384`.
    ///   - maxSeqLen: Maximum token sequence length the model was exported
    ///     with. Queries exceeding this are truncated. Defaults to `64`.
    public static func configure(
        modelName: String,
        bundle: Bundle = .main,
        embeddingsFileName: String = "embeddings_index",
        queryPrefix: String = "query: ",
        dimension: Int = 384,
        maxSeqLen: Int = 64
    ) {
        Configuration.shared.set(
            modelName: modelName,
            bundle: bundle,
            embeddingsFileName: embeddingsFileName,
            queryPrefix: queryPrefix,
            dimension: dimension,
            maxSeqLen: maxSeqLen
        )
    }
}

// MARK: - Configuration

/// Internal singleton holding SDK-wide configuration values.
///
/// Written once via ``SemanticSearchKit/configure(modelName:bundle:embeddingsFileName:queryPrefix:dimension:maxSeqLen:)``
/// and read by ``EmbeddingStorage``, ``QueryEmbedder``, and
/// ``SemanticSearchEngine`` throughout the app's lifetime.
///
/// Thread-safe: all state is protected by an `OSAllocatedUnfairLock`,
/// ensuring correct memory ordering between the one-time write in
/// ``set(…)`` and subsequent reads from any thread.
package final class Configuration: Sendable {

    package static let shared = Configuration()

    // MARK: - Lock-protected state

    private struct State: Sendable {
        var modelName: String = "SemanticSearchModel"
        var bundle: Bundle = .main
        var embeddingsFileName: String = "embeddings_index"
        var queryPrefix: String = "query: "
        var dimension: Int = 384
        var maxSeqLen: Int = 64
        var configured: Bool = false
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    private init() {}

    // MARK: - Accessors

    package var modelName: String { state.withLock { $0.modelName } }
    package var bundle: Bundle { state.withLock { $0.bundle } }
    package var embeddingsFileName: String { state.withLock { $0.embeddingsFileName } }
    package var queryPrefix: String { state.withLock { $0.queryPrefix } }
    package var dimension: Int { state.withLock { $0.dimension } }
    package var maxSeqLen: Int { state.withLock { $0.maxSeqLen } }

    // MARK: - One-time setter

    package func set(
        modelName: String,
        bundle: Bundle,
        embeddingsFileName: String,
        queryPrefix: String,
        dimension: Int,
        maxSeqLen: Int
    ) {
        state.withLock { state in
            guard !state.configured else {
                assertionFailure(
                    "SemanticSearchKit.configure() called more than once. "
                    + "Only the first call takes effect."
                )
                return
            }
            state.configured = true
            state.modelName = modelName
            state.bundle = bundle
            state.embeddingsFileName = embeddingsFileName
            state.queryPrefix = queryPrefix
            state.dimension = dimension
            state.maxSeqLen = maxSeqLen
        }
    }
}
