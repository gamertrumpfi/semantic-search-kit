import Foundation

// MARK: - KeywordScorer

/// Performs keyword-based scoring over any collection of ``SemanticSearchable``
/// documents.
///
/// This is the fallback / complementary signal in the hybrid search pipeline.
/// Each query term is matched against the document's ``SemanticSearchable/searchableText``
/// using case-insensitive substring containment.
///
/// Earlier query terms receive higher weights on the assumption that a user's
/// first words carry the most intent:
///
/// | Position | Weight |
/// |----------|--------|
/// | 1st term | 3.0    |
/// | 2nd term | 2.5    |
/// | 3rd term | 2.0    |
/// | 4th+     | 1.0    |
///
/// ## Usage
///
/// ```swift
/// let scorer = KeywordScorer()
/// let scores = scorer.scores(for: "wireless noise cancelling", in: products)
/// // scores: [(item: Product, score: Double)] sorted descending
/// ```
package struct KeywordScorer: Sendable {

    // MARK: - Term position weights

    /// Weight for the first query term (strongest intent signal).
    private static let firstTermWeight = 3.0
    /// Weight for the second query term.
    private static let secondTermWeight = 2.5
    /// Weight for the third query term.
    private static let thirdTermWeight = 2.0
    /// Weight for the fourth and subsequent query terms.
    private static let remainingTermWeight = 1.0

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
            let text = document.searchableText.lowercased()
            var score = 0.0

            for (index, term) in terms.enumerated() {
                guard text.contains(term) else { continue }
                score += Self.weight(forTermAt: index)
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
        case 0:  return firstTermWeight
        case 1:  return secondTermWeight
        case 2:  return thirdTermWeight
        default: return remainingTermWeight
        }
    }
}
