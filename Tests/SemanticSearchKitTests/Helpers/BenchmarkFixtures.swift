import Accelerate
import Foundation
@testable import SemanticSearchKit

/// Loads and caches the JSON fixture files bundled with the test target.
///
/// These fixtures support retrieval quality evaluation:
/// - `corpus.json` — 26 fictional items across 6 categories
/// - `embeddings_index.json` — pre-computed doc embeddings (populated in Phase 5.2)
/// - `query_embeddings.json` — pre-computed query embeddings (populated in Phase 5.2)
///
/// Until Phase 5.2, the embedding fixtures are empty placeholders.
/// Benchmark tests that require embeddings should check `hasEmbeddings`
/// and skip gracefully if not available.
enum BenchmarkFixtures {

    // MARK: - Corpus

    /// Intermediate JSON representation matching the corpus.json schema.
    private struct CorpusEntry: Decodable {
        let id: String
        let title: String
        let text: String?
        let tags: [TagEntry]?
        let fields: [FieldEntry]?
    }

    private enum TagEntry: Decodable {
        case plain(String)
        case translated(name: String, translation: String)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .plain(str)
            } else {
                let obj = try TranslatedTag(from: decoder)
                self = .translated(name: obj.name, translation: obj.translation)
            }
        }

        private struct TranslatedTag: Decodable {
            let name: String
            let translation: String
        }

        var searchText: String {
            switch self {
            case .plain(let s): return s
            case .translated(let name, let translation): return "\(name) (\(translation))"
            }
        }
    }

    private struct FieldEntry: Decodable {
        let label: String
        let value: String
    }

    /// Load all corpus items as `TestItem` from the test bundle.
    static func loadCorpus() throws -> [TestItem] {
        guard let url = Bundle.module.url(forResource: "corpus", withExtension: "json") else {
            throw FixtureError.fileNotFound("corpus.json")
        }
        let data = try Data(contentsOf: url)
        let entries = try JSONDecoder().decode([CorpusEntry].self, from: data)
        return entries.map { entry in
            // Build searchable text matching SDK's expected format
            var parts = [entry.title]
            if let text = entry.text { parts.append(text) }
            if let tags = entry.tags {
                parts.append(contentsOf: tags.map(\.searchText))
            }
            if let fields = entry.fields {
                parts.append(contentsOf: fields.map { "\($0.label): \($0.value)" })
            }
            return TestItem(
                id: entry.id,
                title: entry.title,
                text: parts.joined(separator: " ")
            )
        }
    }

    // MARK: - Doc Embeddings

    /// Load pre-computed doc embeddings as `[itemId: [Float]]`.
    /// Returns `nil` if the fixture file is empty or missing.
    static func loadDocEmbeddings() throws -> [String: [Float]]? {
        guard let url = Bundle.module.url(forResource: "embeddings_index", withExtension: "json") else {
            return nil
        }
        let data = try Data(contentsOf: url)
        // Empty placeholder files contain "[]"
        guard data.count > 4 else { return nil }

        let records = try JSONDecoder().decode([DocumentEmbedding].self, from: data)
        guard !records.isEmpty else { return nil }

        var dict = [String: [Float]](minimumCapacity: records.count)
        for r in records { dict[r.id] = r.embedding }
        return dict
    }

    // MARK: - Query Embeddings

    /// Load pre-computed query embeddings as `[queryString: [Float]]`.
    /// Returns `nil` if the fixture file is empty or missing.
    static func loadQueryEmbeddings() throws -> [String: [Float]]? {
        guard let url = Bundle.module.url(forResource: "query_embeddings", withExtension: "json") else {
            return nil
        }
        let data = try Data(contentsOf: url)
        // Empty placeholder files contain "{}"
        guard data.count > 4 else { return nil }

        let raw = try JSONDecoder().decode([String: [Double]].self, from: data)
        guard !raw.isEmpty else { return nil }
        return raw.mapValues { $0.map { Float($0) } }
    }

    /// Whether real embedding fixtures are available.
    static var hasEmbeddings: Bool {
        (try? loadDocEmbeddings()) != nil
    }

    // MARK: - Ranking utilities

    /// Rank all items by cosine similarity to a query vector, return top-k titles.
    static func rankByCosine(
        queryVec: [Float],
        docEmbeddings: [String: [Float]],
        items: [TestItem],
        topK: Int
    ) -> [String] {
        var scored: [(String, Float)] = []
        for item in items {
            guard let docVec = docEmbeddings[item.id] else { continue }
            let sim = dotProduct(queryVec, docVec)
            scored.append((item.title, sim))
        }
        scored.sort { $0.1 > $1.1 }
        return Array(scored.prefix(topK).map(\.0))
    }

    /// Rank all items by keyword score, return top-k titles.
    static func rankByKeyword(
        query: String,
        items: [TestItem],
        topK: Int
    ) -> [String] {
        let scorer = KeywordScorer()
        let results = scorer.scores(for: query, in: items)
        return Array(results.prefix(topK).map(\.item.title))
    }

    /// Rank all items by hybrid score (semantic + keyword blend), return top-k titles.
    static func rankByHybrid(
        queryVec: [Float],
        query: String,
        docEmbeddings: [String: [Float]],
        items: [TestItem],
        topK: Int
    ) -> [String] {
        let keywordScorer = KeywordScorer()
        let (kwScores, matchCount) = keywordScorer.normalisedScores(for: query, in: items)

        let hybridScorer = AdaptiveHybridScorer()
        let kwWeight = hybridScorer.effectiveKeywordWeight(
            matchCount: matchCount,
            totalCount: items.count
        )

        var scored: [(String, Double)] = []
        for item in items {
            let semScore: Float
            if let docVec = docEmbeddings[item.id] {
                semScore = max(0, min(1, dotProduct(queryVec, docVec)))
            } else {
                semScore = 0
            }
            let kwScore = kwScores[item.searchID] ?? 0.0
            let hybrid = hybridScorer.blend(
                semanticScore: semScore,
                keywordScore: kwScore,
                keywordWeight: kwWeight
            )
            scored.append((item.title, hybrid))
        }
        scored.sort { $0.1 > $1.1 }
        return Array(scored.prefix(topK).map(\.0))
    }

    // MARK: - Private

    /// Dot product of two Float vectors using Accelerate.
    private static func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    enum FixtureError: Error, LocalizedError {
        case fileNotFound(String)

        var errorDescription: String? {
            switch self {
            case .fileNotFound(let name):
                return "Test fixture '\(name)' not found in Bundle.module. Check Package.swift resources."
            }
        }
    }
}
