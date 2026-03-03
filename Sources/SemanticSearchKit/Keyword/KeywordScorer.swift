import Foundation

// MARK: - KeywordScorer

/// Performs keyword-based scoring over any collection of ``SemanticSearchable``
/// documents.
///
/// This is the fallback / complementary signal in the hybrid search pipeline.
/// Each query term is matched against the document's ``SemanticSearchable/searchableFields``
/// using case-insensitive substring containment.
///
/// ### Two scoring dimensions
///
/// **Term position weight** — earlier query terms receive higher weights on the
/// assumption that a user's first words carry the most intent:
///
/// | Position | Weight |
/// |----------|--------|
/// | 1st term | 3.0    |
/// | 2nd term | 2.5    |
/// | 3rd term | 2.0    |
/// | 4th+     | 1.0    |
///
/// **Field weight** — each ``SearchableField`` carries a `weight` multiplier.
/// A match in a high-weight field (e.g. title) contributes more than a match
/// in a low-weight field (e.g. description). When a term matches multiple
/// fields, only the highest-weighted field counts (no double-dipping).
///
/// The raw score for a document is:
/// ```
/// score = sum over matched terms of (termWeight × bestFieldWeight)
/// ```
///
/// ## Usage
///
/// ```swift
/// let scorer = KeywordScorer()
/// let scores = scorer.scores(for: "wireless noise cancelling", in: products)
/// // scores: [(item: Product, score: Double)] sorted descending
/// ```
package struct KeywordScorer: Sendable {

    package init() {}

    // MARK: - Scoring

    /// Returns documents with their raw keyword scores, sorted descending.
    ///
    /// Only documents with a score > 0 are returned.
    ///
    /// - Parameters:
    ///   - query: Free-text search query (split on whitespace).
    ///   - documents: The corpus to search.
    /// - Returns: Matched documents paired with their raw weighted score,
    ///   sorted by score descending.
    package func scores<Document: SemanticSearchable>(
        for query: String,
        in documents: some Collection<Document>
    ) -> [(item: Document, score: Double)] {
        let terms = query
            .lowercased()
            .split(separator: " ", omittingEmptySubsequences: true)
            .map(String.init)

        guard !terms.isEmpty else { return [] }

        var results: [(item: Document, score: Double)] = []

        for document in documents {
            let fields = document.searchableFields
            var score = 0.0

            for (index, term) in terms.enumerated() {
                // Find the best (highest-weight) field that contains this term.
                var bestFieldWeight = 0.0
                for field in fields {
                    guard field.weight > bestFieldWeight else { continue }
                    if field.text.lowercased().contains(term) {
                        bestFieldWeight = field.weight
                    }
                }

                if bestFieldWeight > 0 {
                    score += Self.weight(forTermAt: index) * bestFieldWeight
                }
            }

            if score > 0 {
                results.append((item: document, score: score))
            }
        }

        results.sort { $0.score > $1.score }
        return results
    }

    /// Returns a dictionary of document ID to normalised keyword score in `[0, 1]`.
    ///
    /// Scores are normalised by dividing each raw score by the maximum raw
    /// score in the result set. Documents with zero keyword overlap are not
    /// included in the dictionary.
    ///
    /// - Parameters:
    ///   - query: Free-text search query.
    ///   - documents: The corpus to search.
    /// - Returns: Dictionary mapping `searchID` to normalised score, plus the
    ///   count of matching documents.
    package func normalisedScores<Document: SemanticSearchable>(
        for query: String,
        in documents: some Collection<Document>
    ) -> (scores: [String: Double], matchCount: Int) {
        let raw = scores(for: query, in: documents)
        guard let maxScore = raw.first?.score, maxScore > 0 else {
            return (scores: [:], matchCount: 0)
        }

        var dict = [String: Double](minimumCapacity: raw.count)
        for (item, score) in raw {
            dict[item.searchID] = score / maxScore
        }
        return (scores: dict, matchCount: raw.count)
    }

    // MARK: - Weight table

    /// Returns the weight for a query term at the given position.
    ///
    /// Earlier terms are assumed to carry stronger intent.
    private static func weight(forTermAt index: Int) -> Double {
        switch index {
        case 0:  return 3.0
        case 1:  return 2.5
        case 2:  return 2.0
        default: return 1.0
        }
    }
}
