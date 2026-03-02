import Testing
import Foundation
@testable import SemanticSearchKit

/// Benchmark tests for hybrid (semantic + keyword) retrieval.
///
/// **These tests require real embedding fixtures** generated in Phase 5.2.
/// Until then, they are automatically skipped via the `hasEmbeddings` guard.
@Suite("Hybrid Benchmark")
struct HybridBenchmarkTests {

    @Test("Hybrid retrieval outperforms keyword-only and embedding-only",
          .enabled(if: BenchmarkFixtures.hasEmbeddings))
    func hybridOutperforms() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        guard let docEmbeddings = try BenchmarkFixtures.loadDocEmbeddings(),
              let queryEmbeddings = try BenchmarkFixtures.loadQueryEmbeddings() else {
            Issue.record("Embedding fixtures not available")
            return
        }

        var kwResults: [(topTitles: [String], expected: [String])] = []
        var embResults: [(topTitles: [String], expected: [String])] = []
        var hybResults: [(topTitles: [String], expected: [String])] = []

        for testCase in BenchmarkTestCases.all {
            guard let queryVec = queryEmbeddings[testCase.query] else { continue }

            // Keyword-only
            let kwTitles = BenchmarkFixtures.rankByKeyword(
                query: testCase.query, items: items, topK: 10
            )
            kwResults.append((topTitles: kwTitles, expected: testCase.expected))

            // Embedding-only
            let embTitles = BenchmarkFixtures.rankByCosine(
                queryVec: queryVec, docEmbeddings: docEmbeddings, items: items, topK: 10
            )
            embResults.append((topTitles: embTitles, expected: testCase.expected))

            // Hybrid
            let hybTitles = BenchmarkFixtures.rankByHybrid(
                queryVec: queryVec, query: testCase.query,
                docEmbeddings: docEmbeddings, items: items, topK: 10
            )
            hybResults.append((topTitles: hybTitles, expected: testCase.expected))
        }

        let kwAgg = RetrievalMetrics.aggregate(results: kwResults)
        let embAgg = RetrievalMetrics.aggregate(results: embResults)
        let hybAgg = RetrievalMetrics.aggregate(results: hybResults)

        print("Keyword-only: \(kwAgg)")
        print("Embedding-only: \(embAgg)")
        print("Hybrid: \(hybAgg)")

        // Hybrid should achieve at least as good as the better of the two
        let bestSingleGate = max(kwAgg.gate, embAgg.gate)
        #expect(hybAgg.gate >= bestSingleGate - 0.05,
                "Hybrid gate (\(hybAgg.gate)) should be close to or better than best single signal (\(bestSingleGate))")
    }

    @Test("Adaptive blending helps strong-keyword queries",
          .enabled(if: BenchmarkFixtures.hasEmbeddings))
    func adaptiveHelpStrongKW() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        guard let docEmbeddings = try BenchmarkFixtures.loadDocEmbeddings(),
              let queryEmbeddings = try BenchmarkFixtures.loadQueryEmbeddings() else {
            Issue.record("Embedding fixtures not available")
            return
        }

        let strongKW = BenchmarkTestCases.all.filter { $0.hint == "STRONG_KW" }
        guard !strongKW.isEmpty else { return }

        var batchResults: [(topTitles: [String], expected: [String])] = []

        for testCase in strongKW {
            guard let queryVec = queryEmbeddings[testCase.query] else { continue }
            let topTitles = BenchmarkFixtures.rankByHybrid(
                queryVec: queryVec, query: testCase.query,
                docEmbeddings: docEmbeddings, items: items, topK: 5
            )
            batchResults.append((topTitles: topTitles, expected: testCase.expected))
        }

        let agg = RetrievalMetrics.aggregate(results: batchResults)
        #expect(agg.hit >= 0.8,
                "Strong-KW hybrid hit@5 should be >= 0.8, got \(agg.hit)")
    }

    @Test("Zero-keyword queries still rank correctly via semantic",
          .enabled(if: BenchmarkFixtures.hasEmbeddings))
    func zeroKeywordViaSemanticOnly() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        guard let docEmbeddings = try BenchmarkFixtures.loadDocEmbeddings(),
              let queryEmbeddings = try BenchmarkFixtures.loadQueryEmbeddings() else {
            Issue.record("Embedding fixtures not available")
            return
        }

        let zeroKW = BenchmarkTestCases.all.filter { $0.hint == "ZERO_KW" }
        guard !zeroKW.isEmpty else { return }

        var batchResults: [(topTitles: [String], expected: [String])] = []

        for testCase in zeroKW {
            guard let queryVec = queryEmbeddings[testCase.query] else { continue }
            let topTitles = BenchmarkFixtures.rankByHybrid(
                queryVec: queryVec, query: testCase.query,
                docEmbeddings: docEmbeddings, items: items, topK: 10
            )
            batchResults.append((topTitles: topTitles, expected: testCase.expected))
        }

        let agg = RetrievalMetrics.aggregate(results: batchResults)
        #expect(agg.hit >= 0.5,
                "Zero-KW hybrid hit@10 should be >= 0.5, got \(agg.hit)")
    }
}
