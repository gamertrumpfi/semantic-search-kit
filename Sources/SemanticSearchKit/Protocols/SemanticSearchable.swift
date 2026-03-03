// MARK: - SearchableField

/// A named chunk of text with an associated keyword-scoring weight.
///
/// Use this to give different parts of a document different importance
/// in keyword search. For example, a recipe title should score higher
/// than its description when a query term matches.
///
/// ```swift
/// SearchableField(text: recipe.title, weight: 3.0)
/// SearchableField(text: recipe.description, weight: 1.0)
/// ```
public struct SearchableField: Sendable {
    /// The text content of this field.
    public let text: String

    /// Multiplier applied to keyword term weights when a match is found
    /// in this field. Higher values make matches in this field more
    /// important. Default is `1.0`.
    public let weight: Double

    public init(text: String, weight: Double = 1.0) {
        self.text = text
        self.weight = weight
    }
}

// MARK: - SemanticSearchable

/// Conform your domain type to this protocol to make it searchable
/// with ``SemanticSearchEngine``.
///
/// ## Basic Example
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
/// ## Field-Weighted Example
///
/// Override ``searchableFields`` to give different parts of your document
/// different keyword-scoring weights:
///
/// ```swift
/// struct Product: SemanticSearchable {
///     let id: String
///     let name: String
///     let category: String
///     let description: String
///
///     var searchID: String { id }
///     var searchableText: String { "\(name) \(category) \(description)" }
///     var searchableFields: [SearchableField] {
///         [
///             SearchableField(text: name, weight: 3.0),
///             SearchableField(text: category, weight: 2.0),
///             SearchableField(text: description, weight: 1.0),
///         ]
///     }
/// }
/// ```
///
/// - ``searchID`` must match the `id` field used in your `corpus.json`
///   and the resulting `embeddings_index.json`.
/// - ``searchableText`` is used for embedding generation and as the
///   default keyword search text when ``searchableFields`` is not overridden.
/// - ``searchableFields`` lets you assign per-field weights for keyword
///   scoring. The default implementation wraps ``searchableText`` as a
///   single field with weight `1.0`, so existing conformances keep working.
public protocol SemanticSearchable: Sendable {
    /// Stable unique identifier matching the document's `id` in the embedding index.
    var searchID: String { get }

    /// Concatenated text used for embedding generation and keyword fallback search.
    var searchableText: String { get }

    /// Weighted text fields for keyword scoring.
    ///
    /// Override this to give different parts of your document different
    /// importance in keyword search. When not overridden, defaults to
    /// a single field containing ``searchableText`` with weight `1.0`.
    var searchableFields: [SearchableField] { get }
}

// MARK: - Default implementation

public extension SemanticSearchable {
    /// Default: wraps ``searchableText`` as a single field with weight `1.0`.
    /// Existing conformances that only provide `searchableText` keep working
    /// with no changes.
    var searchableFields: [SearchableField] {
        [SearchableField(text: searchableText)]
    }
}
