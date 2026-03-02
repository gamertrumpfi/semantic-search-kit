import Foundation

// MARK: - SemanticSearchResult

/// A single search result produced by ``SemanticSearchEngine``.
///
/// Contains the matched document alongside its component scores,
/// allowing callers to inspect how semantic and keyword signals
/// contributed to the final ranking.
public struct SemanticSearchResult<Document: SemanticSearchable>: Sendable {

    /// The matched document.
    public let item: Document

    /// Cosine similarity between the query embedding and the document
    /// embedding. Range `[0, 1]` for L2-normalised vectors.
    public let semanticScore: Float

    /// Normalised keyword match score. Range `[0, 1]`, where 1.0 means
    /// the strongest keyword match in the corpus for this query.
    public let keywordScore: Double

    /// Final blended score used for ranking.
    ///
    /// Computed as:
    /// ```
    /// hybridScore = (1 - kwWeight) * semanticScore + kwWeight * keywordScore
    /// ```
    /// where `kwWeight` is adaptively determined based on match selectivity.
    public let hybridScore: Double

    public init(
        item: Document,
        semanticScore: Float,
        keywordScore: Double,
        hybridScore: Double
    ) {
        self.item = item
        self.semanticScore = semanticScore
        self.keywordScore = keywordScore
        self.hybridScore = hybridScore
    }
}
