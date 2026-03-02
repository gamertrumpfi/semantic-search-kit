import Foundation

// MARK: - QueryEmbedding

/// Abstraction over a query encoder that produces a fixed-dimension
/// embedding vector from a free-text search query.
///
/// The production implementation is ``QueryEmbedder``, which runs a
/// CoreML model on-device.
///
/// Inject a lightweight mock in tests to avoid CoreML dependencies.
public protocol QueryEmbedding: Sendable {
    /// Encodes `query` into an L2-normalised `[Float]` embedding vector.
    func embed(_ query: String) async throws -> [Float]
}
