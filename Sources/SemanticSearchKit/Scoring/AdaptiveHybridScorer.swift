import Foundation

// MARK: - AdaptiveHybridScorer

/// Blends semantic and keyword scores into a single hybrid score using an
/// adaptive weighting strategy.
///
/// The core idea: keyword weight should scale with match *selectivity*.
///
/// - When **no** documents match keywords: weight = 0 (pure semantic).
/// - When **few** documents match (ratio <= `sweetSpot`): full `maxKeywordWeight`.
/// - When **many** documents match (ratio > `sweetSpot`): weight decays linearly
///   toward `floor`, since a noisy keyword signal would dilute semantic ranking.
///
/// ## Tuning
///
/// The default parameters (`maxKeywordWeight = 0.05`, `sweetSpot = 0.15`,
/// `floor = 0.02`) were grid-searched on a 155-query bilingual benchmark.
/// Override them via ``init(maxKeywordWeight:sweetSpot:floor:)`` if your
/// corpus has different characteristics.
package struct AdaptiveHybridScorer: Sendable {

    /// Maximum keyword weight — applied when keyword matches are highly selective.
    package let maxKeywordWeight: Double

    /// Match ratio at or below which full keyword weight is applied.
    package let sweetSpot: Double

    /// Minimum keyword weight when nearly all documents match.
    package let floor: Double

    /// Creates a scorer with the given blending parameters.
    ///
    /// - Parameters:
    ///   - maxKeywordWeight: Upper bound for keyword contribution. Default `0.05`.
    ///   - sweetSpot: Match-ratio threshold below which full weight is used. Default `0.15`.
    ///   - floor: Lower bound for keyword contribution. Default `0.02`.
    package init(
        maxKeywordWeight: Double = 0.05,
        sweetSpot: Double = 0.15,
        floor: Double = 0.02
    ) {
        self.maxKeywordWeight = maxKeywordWeight
        self.sweetSpot = sweetSpot
        self.floor = floor
    }

    // MARK: - Effective weight

    /// Computes the effective keyword weight for a given match ratio.
    ///
    /// - Parameters:
    ///   - matchCount: Number of documents that had a keyword match.
    ///   - totalCount: Total number of documents in the corpus.
    /// - Returns: The keyword weight to use for blending, in `[0, maxKeywordWeight]`.
    package func effectiveKeywordWeight(
        matchCount: Int,
        totalCount: Int
    ) -> Double {
        guard matchCount > 0, totalCount > 0 else { return 0.0 }

        let matchRatio = Double(matchCount) / Double(totalCount)

        if matchRatio <= sweetSpot {
            return maxKeywordWeight
        }

        // Linear decay from maxKeywordWeight to floor as matchRatio goes
        // from sweetSpot to 1.0.
        let t = (matchRatio - sweetSpot) / (1.0 - sweetSpot)
        return maxKeywordWeight - t * (maxKeywordWeight - floor)
    }

    // MARK: - Blending

    /// Blends a semantic score and a keyword score into a single hybrid score.
    ///
    /// - Parameters:
    ///   - semanticScore: Cosine similarity `[0, 1]`.
    ///   - keywordScore: Normalised keyword score `[0, 1]`.
    ///   - keywordWeight: The effective keyword weight (from ``effectiveKeywordWeight(matchCount:totalCount:)``).
    /// - Returns: The blended hybrid score.
    package func blend(
        semanticScore: Float,
        keywordScore: Double,
        keywordWeight: Double
    ) -> Double {
        let semanticWeight = 1.0 - keywordWeight
        return semanticWeight * Double(semanticScore) + keywordWeight * keywordScore
    }
}
