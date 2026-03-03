import Testing
@testable import SemanticSearchKit

@Suite("EmbeddingStorageError")
struct EmbeddingStorageErrorTests {

    @Test("Misconfigured .json file name throws fileNotFoundInBundle on load")
    func misconfiguredFileNameThrowsWhenUsed() async {
        let storage = EmbeddingStorage(fileName: "embeddings_index.json", bundle: .module)

        await #expect(
            throws: EmbeddingStorageError.fileNotFoundInBundle(fileName: "embeddings_index.json"),
            performing: {
                _ = try await storage.loadAll()
            }
        )
    }

    @Test("Includes configuration hint when file name already ends with .json")
    func includesHintForJsonSuffix() {
        let error = EmbeddingStorageError.fileNotFoundInBundle(fileName: "embeddings_index.json")
        let message = error.errorDescription ?? ""

        #expect(message.contains("embeddings_index.json.json not found in app bundle."))
        #expect(message.contains("embeddingsFileName should not include the '.json' extension"))
        #expect(message.contains("Pass the base name"))
    }

    @Test("Uses generic guidance for normal missing file names")
    func genericGuidanceForBaseName() {
        let error = EmbeddingStorageError.fileNotFoundInBundle(fileName: "embeddings_index")
        let message = error.errorDescription ?? ""

        #expect(message.contains("embeddings_index.json not found in app bundle."))
        #expect(message.contains("Generate it with `ssk embed`"))
        #expect(!message.contains("should not include the '.json' extension"))
    }
}
