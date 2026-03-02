import Testing
import Foundation
@testable import SemanticSearchKit

/// Benchmark tests for embedding-only (semantic) retrieval.
///
/// **These tests require real embedding fixtures** generated in Phase 5.2.
/// Until then, they are automatically skipped via the `hasEmbeddings` guard.
@Suite("Embedding Benchmark")
struct EmbeddingBenchmarkTests {

    @Test("Embedding-only retrieval achieves target metrics",
          .enabled(if: BenchmarkFixtures.hasEmbeddings))
    func embeddingOnlyBenchmark() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        guard let docEmbeddings = try BenchmarkFixtures.loadDocEmbeddings(),
              let queryEmbeddings = try BenchmarkFixtures.loadQueryEmbeddings() else {
            Issue.record("Embedding fixtures not available")
            return
        }

        var batchResults: [(topTitles: [String], expected: [String])] = []

        for testCase in BenchmarkTestCases.all {
            guard let queryVec = queryEmbeddings[testCase.query] else { continue }
            let topTitles = BenchmarkFixtures.rankByCosine(
                queryVec: queryVec,
                docEmbeddings: docEmbeddings,
                items: items,
                topK: 10
            )
            batchResults.append((topTitles: topTitles, expected: testCase.expected))
        }

        let agg = RetrievalMetrics.aggregate(results: batchResults)
        print("Embedding-only benchmark:\n\(agg)")

        // Embedding-only should achieve strong hit rate
        #expect(agg.hit >= 0.7,
                "Embedding-only hit@10 should be >= 0.7, got \(agg.hit)")
        #expect(agg.mrr >= 0.5,
                "Embedding-only MRR@10 should be >= 0.5, got \(agg.mrr)")
    }

    @Test("Embedding fixtures structure is valid when present",
          .enabled(if: BenchmarkFixtures.hasEmbeddings))
    func embeddingFixturesValid() throws {
        let items = try BenchmarkFixtures.loadCorpus()
        guard let docEmbeddings = try BenchmarkFixtures.loadDocEmbeddings() else {
            Issue.record("Doc embeddings not available")
            return
        }

        // Every corpus item should have an embedding
        for item in items {
            #expect(docEmbeddings[item.id] != nil,
                    "Missing embedding for \(item.id)")
        }

        // All embeddings should have the same dimension
        let dimensions = Set(docEmbeddings.values.map(\.count))
        #expect(dimensions.count == 1,
                "All embeddings should have the same dimension, got: \(dimensions)")
    }
}
