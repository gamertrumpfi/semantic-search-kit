// MARK: - SemanticSearchable

/// Conform your domain type to this protocol to make it searchable
/// with ``SemanticSearchEngine``.
///
/// ## Example
///
/// ```swift
/// struct Product: SemanticSearchable {
///     let id: String
///     let name: String
///     let description: String
///
///     var searchID: String { id }
///     var searchableText: String { "\(name) \(description)" }
/// }
/// ```
///
/// - ``searchID`` must match the `id` field used in your `corpus.json`
///   and the resulting `embeddings_index.json`.
/// - ``searchableText`` is used for keyword-based fallback scoring.
///   Include all text you want keyword search to match against
///   (title, description, tags, etc.).
public protocol SemanticSearchable: Sendable {
    /// Stable unique identifier matching the document's `id` in the embedding index.
    var searchID: String { get }

    /// Concatenated text used for keyword fallback search.
    var searchableText: String { get }
}
