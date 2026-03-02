import Testing
import Foundation
@testable import SemanticSearchKit

/// Tests for ``SemanticSearchEngine/cosineSimilarity(_:_:)``.
///
/// The method computes the dot product of two L2-normalised vectors and clamps
/// the result to [0, 1]. It uses Accelerate's `vDSP_dotpr` under the hood.
@Suite("Cosine Similarity")
struct CosineSimilarityTests {

    // MARK: - Basic cases

    @Test("Identical unit vectors produce similarity of 1.0")
    func identicalVectors() {
        let v: [Float] = [0.6, 0.8] // unit-length: sqrt(0.36 + 0.64) = 1.0
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(v, v)
        #expect(abs(result - 1.0) < 1e-5)
    }

    @Test("Orthogonal vectors produce similarity of 0.0")
    func orthogonalVectors() {
        let a: [Float] = [1.0, 0.0]
        let b: [Float] = [0.0, 1.0]
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(a, b)
        #expect(abs(result) < 1e-5)
    }

    @Test("Opposite vectors are clamped to 0.0 (not negative)")
    func oppositeVectorsClamped() {
        let a: [Float] = [1.0, 0.0]
        let b: [Float] = [-1.0, 0.0]
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(a, b)
        #expect(result == 0.0)
    }

    @Test("Known 60-degree angle produces cosine of 0.5")
    func knownAngle() {
        // cos(60deg) = 0.5
        // a = [1, 0], b = [cos60, sin60] = [0.5, 0.866...]
        let a: [Float] = [1.0, 0.0]
        let b: [Float] = [0.5, 0.8660254]
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(a, b)
        #expect(abs(result - 0.5) < 1e-4)
    }

    // MARK: - Edge cases

    @Test("Empty vectors return 0.0")
    func emptyVectors() {
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity([], [])
        #expect(result == 0.0)
    }

    @Test("Mismatched vector lengths return 0.0")
    func mismatchedLengths() {
        let a: [Float] = [1.0, 0.0, 0.0]
        let b: [Float] = [1.0, 0.0]
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(a, b)
        #expect(result == 0.0)
    }

    // MARK: - Realistic dimension

    @Test("384-dimensional vectors produce expected similarity")
    func highDimensional() {
        // a = e_0 (first basis vector), b = uniform unit vector.
        var a = [Float](repeating: 0, count: 384)
        a[0] = 1.0

        // b = [1/sqrt(384), 1/sqrt(384), ...]
        let component = Float(1.0 / sqrt(384.0))
        let b = [Float](repeating: component, count: 384)

        // dot(a, b) = 1/sqrt(384) ~ 0.05103
        let expected = component
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(a, b)
        #expect(abs(result - expected) < 1e-4)
    }

    @Test("Result is always clamped to [0, 1] even with imprecise inputs")
    func clampUpperBound() {
        // Vectors slightly longer than unit-length could produce dot > 1.0
        // due to floating-point imprecision. The method should clamp to 1.0.
        let v: [Float] = [1.0000001, 0.0000001]
        let result = SemanticSearchEngine<TestItem>.cosineSimilarity(v, v)
        #expect(result <= 1.0)
    }
}
