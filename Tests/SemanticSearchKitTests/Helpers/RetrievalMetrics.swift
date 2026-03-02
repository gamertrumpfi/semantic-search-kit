import Foundation

/// Information retrieval metrics for evaluating search quality.
///
/// All metrics operate on a ranked list of result titles compared against
/// a set of expected title substrings. A result is "relevant" if its title
/// contains any of the expected substrings (case-insensitive).
enum RetrievalMetrics {

    // MARK: - Relevance

    /// Returns a boolean mask: `true` at position `i` if `topTitles[i]` is relevant.
    /// A title is relevant if it contains any expected substring (case-insensitive).
    static func relevantMask(topTitles: [String], expected: [String]) -> [Bool] {
        topTitles.map { title in
            let lower = title.lowercased()
            return expected.contains { lower.contains($0.lowercased()) }
        }
    }

    // MARK: - Metrics

    /// **Hit@k**: 1 if any relevant result in top-k, else 0.
    static func hitAtK(topTitles: [String], expected: [String]) -> Int {
        let mask = relevantMask(topTitles: topTitles, expected: expected)
        return mask.contains(true) ? 1 : 0
    }

    /// **Recall@k**: fraction of expected items found in top-k.
    /// Each expected substring is counted as "found" if at least one relevant
    /// title in top-k contains it.
    static func recallAtK(topTitles: [String], expected: [String]) -> Double {
        guard !expected.isEmpty else { return 0.0 }
        let mask = relevantMask(topTitles: topTitles, expected: expected)
        let relevantTitles = zip(topTitles, mask)
            .filter { $0.1 }
            .map { $0.0.lowercased() }

        let found = expected.filter { exp in
            relevantTitles.contains { $0.contains(exp.lowercased()) }
        }.count

        return Double(found) / Double(expected.count)
    }

    /// **MRR@k** (Mean Reciprocal Rank): reciprocal rank of the first relevant result.
    /// Returns 0.0 if no relevant result in top-k.
    static func mrrAtK(topTitles: [String], expected: [String]) -> Double {
        let mask = relevantMask(topTitles: topTitles, expected: expected)
        for (i, rel) in mask.enumerated() {
            if rel { return 1.0 / Double(i + 1) }
        }
        return 0.0
    }

    /// **nDCG@k** (Normalized Discounted Cumulative Gain) with binary relevance.
    /// `DCG  = sum(1 / log2(i + 2))` for each relevant position `i`.
    /// `IDCG = sum(1 / log2(i + 2))` for `i` in `0 ..< numRelevant`.
    static func ndcgAtK(topTitles: [String], expected: [String]) -> Double {
        let mask = relevantMask(topTitles: topTitles, expected: expected)
        let dcg = mask.enumerated().reduce(0.0) { acc, pair in
            pair.element ? acc + 1.0 / log2(Double(pair.offset + 2)) : acc
        }
        let nRelevant = mask.filter { $0 }.count
        guard nRelevant > 0 else { return 0.0 }
        let idcg = (0 ..< nRelevant).reduce(0.0) { acc, i in
            acc + 1.0 / log2(Double(i + 2))
        }
        return dcg / idcg
    }

    /// **Gate score**: composite metric used for model selection.
    /// `0.5 * MRR@10 + 0.5 * nDCG@10`
    static func gateScore(mrr: Double, ndcg: Double) -> Double {
        0.5 * mrr + 0.5 * ndcg
    }

    // MARK: - Aggregate

    /// Aggregate metrics across all queries at a given cutoff k.
    struct AggregateResult: CustomStringConvertible, Sendable {
        let hit: Double
        let recall: Double
        let mrr: Double
        let ndcg: Double
        let gate: Double
        let queryCount: Int

        var description: String {
            """
            Aggregate (\(queryCount) queries):
              hit    = \(String(format: "%.3f", hit))
              recall = \(String(format: "%.3f", recall))
              MRR    = \(String(format: "%.3f", mrr))
              nDCG   = \(String(format: "%.3f", ndcg))
              gate   = \(String(format: "%.3f", gate))
            """
        }
    }

    /// Compute aggregate metrics over a batch of (topTitles, expected) pairs.
    static func aggregate(
        results: [(topTitles: [String], expected: [String])]
    ) -> AggregateResult {
        let n = Double(results.count)
        guard n > 0 else {
            return AggregateResult(hit: 0, recall: 0, mrr: 0, ndcg: 0, gate: 0, queryCount: 0)
        }

        var sumHit = 0.0, sumRecall = 0.0, sumMRR = 0.0, sumNDCG = 0.0
        for (topTitles, expected) in results {
            sumHit    += Double(hitAtK(topTitles: topTitles, expected: expected))
            sumRecall += recallAtK(topTitles: topTitles, expected: expected)
            sumMRR    += mrrAtK(topTitles: topTitles, expected: expected)
            sumNDCG   += ndcgAtK(topTitles: topTitles, expected: expected)
        }

        let hit = sumHit / n
        let recall = sumRecall / n
        let mrr = sumMRR / n
        let ndcg = sumNDCG / n
        let gate = gateScore(mrr: mrr, ndcg: ndcg)

        return AggregateResult(
            hit: hit, recall: recall, mrr: mrr, ndcg: ndcg, gate: gate,
            queryCount: Int(n)
        )
    }
}
