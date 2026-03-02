import Testing
import Foundation
@testable import SemanticSearchKit

/// Benchmark tests that exercise the full ``SemanticSearchEngine`` pipeline
/// with mock embeddings derived from the fixture files.
///
/// Unlike `EmbeddingBenchmarkTests` and `HybridBenchmarkTests`, these tests
/// do **not** require a CoreML model or real embeddings — they use
/// ``MockEmbeddingStorage`` and ``MockQueryEmbedder`` with fixture data.
///
/// When real embedding fixtures become available (Phase 5.2), this suite
/// validates the engine's integration with the benchmark infrastructure.
@Suite("Engine Integration Benchmark")
struct CoreMLBenchmarkTests {

    @Test("Engine produces consistent results with mock data")
    func engineConsistency() async throws {
        let items = try BenchmarkFixtures.loadCorpus()

        // Create synthetic embeddings: each item gets a unique basis vector
        var embeddings = [String: [Float]]()
        for (i, item) in items.enumerated() {
            var vec = [Float](repeating: 0, count: items.count)
            vec[i] = 1.0
            embeddings[item.id] = vec
        }

        // Query vector points toward the first item
        var queryVec = [Float](repeating: 0, count: items.count)
        queryVec[0] = 1.0

        let engine = SemanticSearchEngine<TestItem>(
            storage: MockEmbeddingStorage(embeddings: embeddings),
            embedder: MockQueryEmbedder(vector: queryVec)
        )

        let results = try await engine.search(
            query: "test query",
            in: items,
            topK: 5
        )

        // First result should be item[0] (highest cosine = 1.0)
        #expect(results.first?.item.id == items[0].id)
        #expect(results.first?.semanticScore == 1.0)

        // Results should be sorted descending
        for i in 1 ..< results.count {
            #expect(results[i - 1].hybridScore >= results[i].hybridScore)
        }
    }

    @Test("Engine returns all items when threshold is 0")
    func engineReturnsAll() async throws {
        let items = try BenchmarkFixtures.loadCorpus()

        // Give every item the same moderate embedding
        let uniformVec = [Float](repeating: 0.1, count: 10)
        var embeddings = [String: [Float]]()
        for item in items {
            embeddings[item.id] = uniformVec
        }

        let engine = SemanticSearchEngine<TestItem>(
            storage: MockEmbeddingStorage(embeddings: embeddings),
            embedder: MockQueryEmbedder(vector: uniformVec)
        )

        let results = try await engine.search(
            query: "anything",
            in: items,
            topK: 100,
            semanticThreshold: 0.0
        )

        // All 26 items should be returned
        #expect(results.count == items.count)
    }

    @Test("Engine with real fixtures produces benchmark metrics",
          .enabled(if: BenchmarkFixtures.hasEmbeddings))
    func engineRealFixtures() async throws {
        let items = try BenchmarkFixtures.loadCorpus()
        guard let docEmbeddings = try BenchmarkFixtures.loadDocEmbeddings(),
              let queryEmbeddings = try BenchmarkFixtures.loadQueryEmbeddings() else {
            Issue.record("Embedding fixtures not available")
            return
        }

        var batchResults: [(topTitles: [String], expected: [String])] = []

        for testCase in BenchmarkTestCases.all {
            guard let queryVec = queryEmbeddings[testCase.query] else { continue }

            let engine = SemanticSearchEngine<TestItem>(
                storage: MockEmbeddingStorage(embeddings: docEmbeddings),
                embedder: MockQueryEmbedder(vector: queryVec)
            )

            let results = try await engine.search(
                query: testCase.query,
                in: items,
                topK: 10
            )

            let topTitles = results.map(\.item.title)
            batchResults.append((topTitles: topTitles, expected: testCase.expected))
        }

        let agg = RetrievalMetrics.aggregate(results: batchResults)
        print("Engine benchmark (real fixtures):\n\(agg)")

        #expect(agg.gate >= 0.7,
                "Engine gate score should be >= 0.7, got \(agg.gate)")
    }
}
