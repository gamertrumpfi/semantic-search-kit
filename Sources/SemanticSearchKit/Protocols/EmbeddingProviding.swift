import Foundation

// MARK: - EmbeddingProviding

/// Abstraction over a store of pre-computed document embeddings.
///
/// The production implementation is ``EmbeddingStorage``, which loads
/// vectors from the app bundle's `embeddings_index.json`.
///
/// Inject a lightweight mock in tests to avoid bundle dependencies.
public protocol EmbeddingProviding: Sendable {
    /// Returns all embeddings keyed by document ID.
    func loadAll() async throws -> [String: [Float]]

    /// Returns the embedding for a single document, or `nil` if not found.
    func embedding(for documentID: String) async throws -> [Float]?
}
