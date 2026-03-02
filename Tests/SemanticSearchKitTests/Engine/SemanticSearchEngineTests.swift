import Testing
import Foundation
@testable import SemanticSearchKit

/// Tests for ``SemanticSearchEngine/search(query:in:topK:semanticThreshold:)``.
///
/// These tests inject mock embeddings and a mock query embedder so no CoreML
/// model or bundle resources are needed.
@Suite("SemanticSearchEngine")
struct SemanticSearchEngineTests {

    // MARK: - Shared test data

    /// Three items with known IDs for embedding lookup.
    private static let items: [TestItem] = [
        TestItem(id: "r1", title: "Spaghetti Carbonara", text: "Italian pasta with eggs"),
        TestItem(id: "r2", title: "Thai Red Curry", text: "Spicy coconut curry"),
        TestItem(id: "r3", title: "Chocolate Cake", text: "Rich dark dessert"),
    ]

    /// Unit vectors in 3D for easy cosine reasoning.
    /// r1 points along x-axis, r2 along y-axis, r3 along z-axis.
    private static let embeddings: [String: [Float]] = [
        "r1": [1.0, 0.0, 0.0],
        "r2": [0.0, 1.0, 0.0],
        "r3": [0.0, 0.0, 1.0],
    ]

    /// A query vector that points mostly toward r1 (semantic = 0.9 for r1).
    private static let queryVec: [Float] = [0.9, 0.3, 0.1]
    // dot products: r1 = 0.9, r2 = 0.3, r3 = 0.1

    private func makeSUT(
        embeddings: [String: [Float]] = SemanticSearchEngineTests.embeddings,
        queryVector: [Float] = SemanticSearchEngineTests.queryVec
    ) -> SemanticSearchEngine<TestItem> {
        SemanticSearchEngine(
            storage: MockEmbeddingStorage(embeddings: embeddings),
            embedder: MockQueryEmbedder(vector: queryVector)
        )
    }

    // MARK: - Hybrid scoring

    @Test("Hybrid score blends semantic + keyword with adaptive weight")
    func hybridScoreFormula() async throws {
        let sut = makeSUT()

        let results = try await sut.search(
            query: "spaghetti",
            in: Self.items,
            topK: 10
        )

        // "spaghetti" matches r1's title via keyword search.
        // r1 semantic = 0.9 (dot product with query vec).
        guard let r1 = results.first(where: { $0.item.id == "r1" }) else {
            Issue.record("r1 not found in results")
            return
        }

        // Compute expected adaptive weight
        let scorer = AdaptiveHybridScorer()
        // Only 1 item matches "spaghetti" keyword
        let kwWeight = scorer.effectiveKeywordWeight(
            matchCount: 1,
            totalCount: Self.items.count
        )
        let expected = scorer.blend(
            semanticScore: r1.semanticScore,
            keywordScore: r1.keywordScore,
            keywordWeight: kwWeight
        )
        #expect(abs(r1.hybridScore - expected) < 1e-6)
    }

    @Test("Results are sorted by hybrid score descending")
    func sortedDescending() async throws {
        let sut = makeSUT()
        let results = try await sut.search(query: "spaghetti", in: Self.items, topK: 10)

        for i in 1 ..< results.count {
            #expect(results[i - 1].hybridScore >= results[i].hybridScore)
        }
    }

    // MARK: - topK and threshold

    @Test("topK limits the number of returned results")
    func topKLimitsResults() async throws {
        let sut = makeSUT()
        let results = try await sut.search(query: "test", in: Self.items, topK: 2)
        #expect(results.count <= 2)
    }

    @Test("semanticThreshold filters items below the threshold")
    func semanticThresholdFilters() async throws {
        let sut = makeSUT()

        // queryVec dots: r1=0.9, r2=0.3, r3=0.1
        // Threshold of 0.5 should exclude r2 and r3
        let results = try await sut.search(
            query: "anything",
            in: Self.items,
            topK: 10,
            semanticThreshold: 0.5
        )

        let ids = results.map(\.item.id)
        #expect(ids.contains("r1"))
        #expect(!ids.contains("r3"))
    }

    // MARK: - Edge cases

    @Test("Empty query returns empty results")
    func emptyQuery() async throws {
        let sut = makeSUT()
        let results = try await sut.search(query: "", in: Self.items)
        #expect(results.isEmpty)
    }

    @Test("Whitespace-only query returns empty results")
    func whitespaceQuery() async throws {
        let sut = makeSUT()
        let results = try await sut.search(query: "   \n\t  ", in: Self.items)
        #expect(results.isEmpty)
    }

    @Test("Item with no embedding gets semantic score of 0.0")
    func missingEmbedding() async throws {
        // Only include embedding for r1; r2 and r3 have no embeddings
        let sut = makeSUT(embeddings: ["r1": [1.0, 0.0, 0.0]])
        let results = try await sut.search(query: "test", in: Self.items, topK: 10)

        let r2 = results.first { $0.item.id == "r2" }
        let r3 = results.first { $0.item.id == "r3" }
        #expect(r2?.semanticScore == 0.0)
        #expect(r3?.semanticScore == 0.0)
    }

    @Test("All items below threshold returns empty results")
    func allBelowThreshold() async throws {
        // Query vec that produces low similarity with all items
        let lowVec: [Float] = [0.1, 0.1, 0.1]
        let sut = makeSUT(queryVector: lowVec)

        // dots: r1=0.1, r2=0.1, r3=0.1 -- all below 0.5
        let results = try await sut.search(
            query: "anything",
            in: Self.items,
            topK: 10,
            semanticThreshold: 0.5
        )
        #expect(results.isEmpty)
    }

    @Test("Semantic-heavy item wins over keyword-only match due to adaptive blending")
    func semanticWinsOverKeyword() async throws {
        // Query vector points entirely toward r2 (semantic r2 = 1.0)
        let queryTowardR2: [Float] = [0.0, 1.0, 0.0]
        let sut = makeSUT(queryVector: queryTowardR2)

        // Search "spaghetti" -- keyword match only for r1
        let results = try await sut.search(query: "spaghetti", in: Self.items, topK: 10)

        // r2 semantic=1.0, keyword=0.0
        // r1 semantic=0.0, keyword=1.0
        // Since semantic weight is always >= (1 - maxKeywordWeight) = 0.95, r2 should win
        #expect(results.first?.item.id == "r2")
    }

    @Test("Large item set returns correct top-K ordering")
    func largeItemSet() async throws {
        // Generate 100 items with embeddings that have linearly decreasing similarity
        var items = [TestItem]()
        var embeddings = [String: [Float]]()
        for i in 0 ..< 100 {
            let id = "item-\(i)"
            items.append(TestItem(id: id, title: "Item \(i)"))
            // Each item's embedding: [1 - i/100, i/100, 0]
            // Dot with queryVec [1,0,0] gives decreasing similarity
            let sim = Float(100 - i) / 100.0
            embeddings[id] = [sim, Float(i) / 100.0, 0.0]
        }

        let queryVec: [Float] = [1.0, 0.0, 0.0]
        let sut = makeSUT(embeddings: embeddings, queryVector: queryVec)

        let results = try await sut.search(query: "zzz", in: items, topK: 5)

        // Top 5 should be item-0 through item-4 (highest cosine similarity)
        #expect(results.count == 5)
        #expect(results[0].item.id == "item-0")
        // Verify descending order
        for i in 1 ..< results.count {
            #expect(results[i - 1].hybridScore >= results[i].hybridScore)
        }
    }

    @Test("Empty document list returns empty results")
    func emptyDocuments() async throws {
        let sut = makeSUT()
        let results = try await sut.search(query: "test", in: [TestItem]())
        #expect(results.isEmpty)
    }

    @Test("Result contains all score components")
    func resultContainsAllScores() async throws {
        let sut = makeSUT()
        let results = try await sut.search(query: "spaghetti", in: Self.items, topK: 10)

        guard let r1 = results.first(where: { $0.item.id == "r1" }) else {
            Issue.record("r1 not found")
            return
        }

        // r1 should have non-zero semantic (0.9) and non-zero keyword (matches "spaghetti")
        #expect(r1.semanticScore > 0)
        #expect(r1.keywordScore > 0)
        #expect(r1.hybridScore > 0)
    }
}
