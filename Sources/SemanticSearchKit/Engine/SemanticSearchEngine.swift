import Accelerate
import Foundation
import os.log

// MARK: - SemanticSearchEngine

/// Generic hybrid search engine that ranks documents by blending semantic
/// (embedding) similarity with keyword-based scoring.
///
/// `Document` can be any type conforming to ``SemanticSearchable``.
///
/// ## Quick Start
///
/// ```swift
/// SemanticSearchKit.configure(modelName: "MyAppSearch", bundle: .main)
///
/// struct Product: SemanticSearchable {
///     let id: String
///     let name: String
///     var searchID: String { id }
///     var searchableText: String { name }
/// }
///
/// let engine = SemanticSearchEngine<Product>()
/// let results = try await engine.search(query: "wireless headphones", in: products)
/// ```
///
/// ## Dependency Injection
///
/// All heavy dependencies (embedding storage, query embedder, keyword scorer,
/// hybrid scorer) are injected via protocols or `package`-access structs,
/// making the engine fully testable without CoreML or bundle resources.
public actor SemanticSearchEngine<Document: SemanticSearchable> {

    // MARK: - Dependencies

    private let storage: any EmbeddingProviding
    private let embedder: any QueryEmbedding
    private let keywordScorer: KeywordScorer
    private let hybridScorer: AdaptiveHybridScorer

    /// Logger instance. Computed property because Swift does not support
    /// static stored properties in generic types.
    private static var logger: Logger {
        Logger(subsystem: "com.semanticsearchkit", category: "SemanticSearchEngine")
    }

    // MARK: - Initialisation

    /// Creates a search engine using the SDK-configured shared singletons.
    ///
    /// Reads configuration from
    /// ``SemanticSearchKit/configure(modelName:bundle:embeddingsFileName:queryPrefix:dimension:maxSeqLen:)``.
    ///
    /// - Parameters:
    ///   - storage: Provider of pre-computed document embeddings.
    ///     Defaults to ``EmbeddingStorage/shared``.
    ///   - embedder: Query encoder producing embedding vectors.
    ///     Defaults to ``QueryEmbedder/shared``.
    public init(
        storage: any EmbeddingProviding = EmbeddingStorage.shared,
        embedder: any QueryEmbedding = QueryEmbedder.shared
    ) {
        self.storage = storage
        self.embedder = embedder
        self.keywordScorer = KeywordScorer()
        self.hybridScorer = AdaptiveHybridScorer()
    }

    /// Creates a search engine with fully custom dependencies.
    ///
    /// Intended for testing and advanced use cases where you need to
    /// override the keyword scoring or blending strategies.
    package init(
        storage: any EmbeddingProviding,
        embedder: any QueryEmbedding,
        keywordScorer: KeywordScorer,
        hybridScorer: AdaptiveHybridScorer
    ) {
        self.storage = storage
        self.embedder = embedder
        self.keywordScorer = keywordScorer
        self.hybridScorer = hybridScorer
    }

    // MARK: - Public search API

    /// Returns documents ranked by a hybrid of semantic and keyword relevance.
    ///
    /// The pipeline:
    /// 1. Load pre-computed document embeddings from the configured bundle.
    /// 2. Embed the query on-device via the CoreML model.
    /// 3. Compute per-document keyword scores (normalised by max raw score).
    /// 4. Adaptively blend semantic and keyword signals based on match selectivity.
    /// 5. Filter by `semanticThreshold`, sort by `hybridScore`, return top-K.
    ///
    /// - Parameters:
    ///   - query: Free-text user query.
    ///   - documents: The full document catalogue to search.
    ///   - topK: Maximum number of results to return. Default `20`.
    ///   - semanticThreshold: Minimum semantic (cosine) score to include a
    ///     result. Default `0.0` (no filtering).
    /// - Returns: Results sorted by ``SemanticSearchResult/hybridScore`` descending.
    public func search(
        query: String,
        in documents: [Document],
        topK: Int = 20,
        semanticThreshold: Float = 0.0
    ) async throws -> [SemanticSearchResult<Document>] {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return []
        }

        // 1. Load pre-computed document embeddings (cached after first load).
        let embeddingDict = try await storage.loadAll()

        // 2. Embed the query on-device.
        let queryVec = try await embedder.embed(query)

        // 3. Compute keyword scores with raw-score normalisation.
        let (keywordScores, matchCount) = keywordScorer.normalisedScores(
            for: query,
            in: documents
        )

        // 4. Determine adaptive keyword weight based on match selectivity.
        let kwWeight = hybridScorer.effectiveKeywordWeight(
            matchCount: matchCount,
            totalCount: documents.count
        )

        // 5. Score, filter, and rank.
        var results: [SemanticSearchResult<Document>] = []
        results.reserveCapacity(min(documents.count, topK * 2))

        for document in documents {
            let semScore: Float
            if let docVec = embeddingDict[document.searchID] {
                semScore = Self.cosineSimilarity(queryVec, docVec)
            } else {
                semScore = 0.0
            }

            guard semScore >= semanticThreshold else { continue }

            let kwScore = keywordScores[document.searchID] ?? 0.0
            let hybrid = hybridScorer.blend(
                semanticScore: semScore,
                keywordScore: kwScore,
                keywordWeight: kwWeight
            )

            results.append(SemanticSearchResult(
                item: document,
                semanticScore: semScore,
                keywordScore: kwScore,
                hybridScore: hybrid
            ))
        }

        results.sort { $0.hybridScore > $1.hybridScore }

        if results.count > topK {
            results.removeSubrange(topK...)
        }

        Self.logger.debug(
            "Search '\(query)': \(results.count) results, kwWeight=\(kwWeight, format: .fixed(precision: 4))"
        )

        return results
    }

    // MARK: - Cosine similarity (Accelerate-optimised)

    /// Computes cosine similarity between two L2-normalised vectors.
    ///
    /// Since both query and document vectors are expected to be unit-normalised,
    /// this reduces to a dot product. The result is clamped to `[0, 1]`.
    ///
    /// Uses Accelerate's `vDSP_dotpr` for SIMD-optimised computation.
    package static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        return max(0, min(1, dot))
    }
}
