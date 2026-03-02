import CoreML
import Foundation
import Tokenizers

// MARK: - QueryEmbedder

/// Encodes a search query into an L2-normalised embedding vector using an
/// on-device CoreML transformer model.
///
/// The model name, bundle, and hyperparameters are read from ``Configuration``
/// (set via ``SemanticSearchKit/configure(modelName:bundle:embeddingsFileName:queryPrefix:dimension:maxSeqLen:)``).
///
/// Both the tokenizer vocabulary and the compiled model must be present in
/// the configured bundle — no network request is ever made.
///
/// ## Usage
///
/// Use the ``shared`` singleton (configured automatically):
///
/// ```swift
/// let vector = try await QueryEmbedder.shared.embed("wireless noise-cancelling headphones")
/// ```
///
/// Or inject a custom instance / mock via the ``QueryEmbedding`` protocol.
public actor QueryEmbedder: QueryEmbedding {

    // MARK: - Shared instance

    /// Shared singleton configured via ``SemanticSearchKit/configure(modelName:bundle:embeddingsFileName:queryPrefix:dimension:maxSeqLen:)``.
    public static let shared = QueryEmbedder()

    // MARK: - Lazily-loaded state (load-once via Task caching)

    private var tokenizerTask: Task<Tokenizer, Error>?
    private var model: MLModel?

    // MARK: - Configuration accessors

    private var queryPrefix: String { Configuration.shared.queryPrefix }
    private var maxSeqLen: Int { Configuration.shared.maxSeqLen }
    private var dimension: Int { Configuration.shared.dimension }
    private var modelName: String { Configuration.shared.modelName }
    private var bundle: Bundle { Configuration.shared.bundle }

    /// XLM-RoBERTa pad token id (standard across E5 models).
    private static let padTokenId: Int32 = 1

    private init() {}

    // MARK: - QueryEmbedding

    /// Encodes `query` into an L2-normalised `[Float]` embedding vector.
    ///
    /// The first call is slightly slower while the tokenizer and CoreML model
    /// are loaded from the bundle. Subsequent calls reuse cached instances
    /// (typically ~1-3 ms on Apple Silicon).
    ///
    /// - Parameter query: Free-text search query (the configured prefix is
    ///   prepended automatically).
    /// - Throws: ``QueryEmbedderError`` if bundle assets are missing or if
    ///   CoreML prediction fails.
    public func embed(_ query: String) async throws -> [Float] {
        let tok = try await loadTokenizer()
        let mlm = try loadModel()
        return try predict(query: query, tokenizer: tok, model: mlm)
    }

    // MARK: - Lazy loaders

    /// Loads the tokenizer exactly once, even under concurrent `embed()` calls.
    ///
    /// The first caller creates a `Task` that is cached on the actor.
    /// Subsequent callers `await` the same task, avoiding duplicate work
    /// across actor re-entrancy suspension points.
    private func loadTokenizer() async throws -> Tokenizer {
        if let existing = tokenizerTask {
            return try await existing.value
        }

        let bundle = self.bundle
        let task = Task<Tokenizer, Error> {
            guard let resourceURL = bundle.resourceURL else {
                throw QueryEmbedderError.tokenizerNotFound(bundle: bundle)
            }
            return try await AutoTokenizer.from(modelFolder: resourceURL)
        }
        tokenizerTask = task
        return try await task.value
    }

    private func loadModel() throws -> MLModel {
        if let existing = model { return existing }

        // Xcode compiles .mlpackage -> .mlmodelc at build time; try both.
        guard let url = bundle.url(forResource: modelName, withExtension: "mlmodelc")
                     ?? bundle.url(forResource: modelName, withExtension: "mlpackage") else {
            throw QueryEmbedderError.modelNotFound(name: modelName, bundle: bundle)
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all // Neural Engine + GPU + CPU
        let loaded = try MLModel(contentsOf: url, configuration: config)
        model = loaded
        return loaded
    }

    // MARK: - Inference

    private func predict(
        query: String,
        tokenizer: Tokenizer,
        model: MLModel
    ) throws -> [Float] {
        let prefixed = queryPrefix + query

        // Encode — returns IDs WITH special tokens (<s> ... </s>).
        var ids = tokenizer.encode(text: prefixed, addSpecialTokens: true)

        // Truncate to maxSeqLen, preserving the final special token (e.g. </s>).
        // Guard: only truncate when there are at least 2 tokens (opening + closing
        // special tokens). A single-token or empty sequence is left as-is.
        if ids.count > maxSeqLen, ids.count >= 2 {
            let closingToken = ids[ids.count - 1]
            ids = Array(ids.prefix(maxSeqLen - 1)) + [closingToken]
        }

        let seqLen = ids.count
        let padLen = maxSeqLen - seqLen

        // Build input_ids and attention_mask as flat Int32 arrays.
        let inputIds = ids.map { Int32($0) }
            + [Int32](repeating: Self.padTokenId, count: padLen)
        let attentionMask = [Int32](repeating: 1, count: seqLen)
            + [Int32](repeating: 0, count: padLen)

        // Wrap in MLMultiArray (shape [1, maxSeqLen]).
        let shape: [NSNumber] = [1, NSNumber(value: maxSeqLen)]
        let idsArray = try MLMultiArray(shape: shape, dataType: .int32)
        let maskArray = try MLMultiArray(shape: shape, dataType: .int32)

        for i in 0 ..< maxSeqLen {
            idsArray[i] = NSNumber(value: inputIds[i])
            maskArray[i] = NSNumber(value: attentionMask[i])
        }

        // Run inference.
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ])
        let output = try model.prediction(from: provider)

        guard let embArray = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw QueryEmbedderError.unexpectedOutput
        }

        // Convert MLMultiArray (shape [1, dimension]) -> [Float].
        // The model is expected to output L2-normalised embeddings.
        var result = [Float](repeating: 0, count: dimension)
        for i in 0 ..< dimension {
            result[i] = embArray[i].floatValue
        }
        return result
    }
}

// MARK: - Errors

/// Errors thrown by ``QueryEmbedder``.
public enum QueryEmbedderError: Error, LocalizedError, Sendable {

    /// The tokenizer vocabulary files were not found in the bundle.
    case tokenizerNotFound(bundle: Bundle)

    /// The CoreML model was not found in the bundle.
    case modelNotFound(name: String, bundle: Bundle)

    /// The CoreML model returned an unexpected output format.
    case unexpectedOutput

    public var errorDescription: String? {
        switch self {
        case .tokenizerNotFound(let bundle):
            return "tokenizer.json / tokenizer_config.json not found in bundle "
                + "(\(bundle.bundlePath)). Add them to your app target's resources."
        case .modelNotFound(let name, let bundle):
            return "\(name).mlpackage not found in bundle "
                + "(\(bundle.bundlePath)). Add it to your app target's resources."
        case .unexpectedOutput:
            return "CoreML model returned unexpected output — verify the model export "
                + "produces an 'embedding' output feature."
        }
    }
}
