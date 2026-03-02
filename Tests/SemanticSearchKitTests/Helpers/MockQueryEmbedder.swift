import Foundation
@testable import SemanticSearchKit

/// A test double for ``QueryEmbedding`` that returns a configurable vector.
///
/// By default, returns the same vector for every query. Use `vectorsByQuery`
/// to return different vectors for different query strings.
///
/// Usage:
/// ```swift
/// let mock = MockQueryEmbedder(vector: [1.0, 0.0, 0.0])
/// let v = try await mock.embed("anything")  // -> [1.0, 0.0, 0.0]
/// ```
actor MockQueryEmbedder: QueryEmbedding {
    private let defaultVector: [Float]
    private let vectorsByQuery: [String: [Float]]
    private let error: (any Error)?

    /// Creates a mock that returns `vector` for every query.
    /// Override specific queries via `vectorsByQuery`.
    /// If `error` is set, all calls throw that error instead.
    init(
        vector: [Float] = [],
        vectorsByQuery: [String: [Float]] = [:],
        error: (any Error)? = nil
    ) {
        self.defaultVector = vector
        self.vectorsByQuery = vectorsByQuery
        self.error = error
    }

    func embed(_ query: String) async throws -> [Float] {
        if let error { throw error }
        return vectorsByQuery[query] ?? defaultVector
    }
}
