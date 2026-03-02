import Testing
import Foundation
@testable import SemanticSearchKit

/// Benchmark tests for keyword-only retrieval on the synthetic corpus.
///
/// These tests verify that keyword scoring provides a reasonable baseline
/// for queries with strong literal overlap.
@Suite("Keyword Benchmark")
struct KeywordBenchmarkTests {

    @Test("Corpus loads successfully with 26 items")
    func corpusLoads() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        #expect(items.count == 26)
    }

    @Test("Strong keyword queries achieve hit@5 on keyword search")
    func strongKeywordHitRate() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        let strongKW = BenchmarkTestCases.all.filter { $0.hint == "STRONG_KW" }
        guard !strongKW.isEmpty else { return }

        var batchResults: [(topTitles: [String], expected: [String])] = []

        for testCase in strongKW {
            let topTitles = BenchmarkFixtures.rankByKeyword(
                query: testCase.query,
                items: items,
                topK: 5
            )
            batchResults.append((topTitles: topTitles, expected: testCase.expected))
        }

        let agg = RetrievalMetrics.aggregate(results: batchResults)
        // Strong keyword queries should achieve decent hit rate even keyword-only
        #expect(agg.hit >= 0.5,
                "Strong-KW hit@5 should be >= 0.5, got \(agg.hit)")
    }

    @Test("Keyword search returns non-empty results for title-match queries")
    func keywordReturnsResults() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        let scorer = KeywordScorer()

        // Searching exact title words should always return results
        let results = scorer.scores(for: "Hitchhiker Guide Galaxy", in: items)
        #expect(!results.isEmpty)
        #expect(results[0].item.title.contains("Hitchhiker"))
    }

    @Test("Keyword benchmark metrics are computed correctly")
    func metricsComputation() {
        // Validate the metrics infrastructure with known data
        let topTitles = ["The Hitchhiker's Guide to the Galaxy", "Dune", "Neuromancer"]
        let expected = ["Hitchhiker"]

        let hit = RetrievalMetrics.hitAtK(topTitles: topTitles, expected: expected)
        #expect(hit == 1)

        let mrr = RetrievalMetrics.mrrAtK(topTitles: topTitles, expected: expected)
        #expect(abs(mrr - 1.0) < 1e-10) // First result is relevant

        let recall = RetrievalMetrics.recallAtK(topTitles: topTitles, expected: expected)
        #expect(abs(recall - 1.0) < 1e-10) // 1/1 expected found
    }

    @Test("Full keyword benchmark on all test cases")
    func fullKeywordBenchmark() throws {
        let items = try BenchmarkFixtures.loadCorpus()

        var batchResults: [(topTitles: [String], expected: [String])] = []

        for testCase in BenchmarkTestCases.all {
            let topTitles = BenchmarkFixtures.rankByKeyword(
                query: testCase.query,
                items: items,
                topK: 10
            )
            batchResults.append((topTitles: topTitles, expected: testCase.expected))
        }

        let agg = RetrievalMetrics.aggregate(results: batchResults)

        // Log the results for visibility
        print("Keyword-only benchmark:\n\(agg)")

        // Keyword-only won't be great (especially for ZERO_KW queries),
        // but it should be non-zero
        #expect(agg.hit > 0.0, "Keyword hit rate should be non-zero")
    }
}
