import Foundation
@testable import SemanticSearchKit

/// A test double for ``EmbeddingProviding`` that returns canned embedding vectors.
///
/// Usage:
/// ```swift
/// let mock = MockEmbeddingStorage(embeddings: [
///     "item-1": [0.5, 0.5, 0.5],
///     "item-2": [0.1, 0.9, 0.0],
/// ])
/// ```
actor MockEmbeddingStorage: EmbeddingProviding {
    private let embeddings: [String: [Float]]
    private let error: (any Error)?

    /// Creates a mock that returns the given embeddings dictionary.
    /// If `error` is set, all calls will throw that error instead.
    init(embeddings: [String: [Float]] = [:], error: (any Error)? = nil) {
        self.embeddings = embeddings
        self.error = error
    }

    func loadAll() async throws -> [String: [Float]] {
        if let error { throw error }
        return embeddings
    }

    func embedding(for documentID: String) async throws -> [Float]? {
        if let error { throw error }
        return embeddings[documentID]
    }
}
