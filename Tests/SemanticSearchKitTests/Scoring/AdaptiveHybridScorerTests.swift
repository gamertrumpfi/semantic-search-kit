import Testing
import Foundation
@testable import SemanticSearchKit

/// Tests for ``AdaptiveHybridScorer``.
@Suite("AdaptiveHybridScorer")
struct AdaptiveHybridScorerTests {

    private let sut = AdaptiveHybridScorer()

    // MARK: - effectiveKeywordWeight

    @Test("Zero matches produce zero keyword weight")
    func zeroMatches() {
        let weight = sut.effectiveKeywordWeight(matchCount: 0, totalCount: 100)
        #expect(weight == 0.0)
    }

    @Test("Zero total count produces zero keyword weight")
    func zeroTotal() {
        let weight = sut.effectiveKeywordWeight(matchCount: 5, totalCount: 0)
        #expect(weight == 0.0)
    }

    @Test("Match ratio at or below sweet spot returns full max weight")
    func belowSweetSpot() {
        // sweetSpot = 0.15, so 10/100 = 0.10 is below
        let weight = sut.effectiveKeywordWeight(matchCount: 10, totalCount: 100)
        #expect(abs(weight - sut.maxKeywordWeight) < 1e-10)
    }

    @Test("Match ratio exactly at sweet spot returns full max weight")
    func atSweetSpot() {
        // 15/100 = 0.15 = sweetSpot
        let weight = sut.effectiveKeywordWeight(matchCount: 15, totalCount: 100)
        #expect(abs(weight - sut.maxKeywordWeight) < 1e-10)
    }

    @Test("Match ratio of 1.0 returns floor weight")
    func allMatch() {
        // 100/100 = 1.0 -> should return floor
        let weight = sut.effectiveKeywordWeight(matchCount: 100, totalCount: 100)
        #expect(abs(weight - sut.floor) < 1e-10)
    }

    @Test("Match ratio between sweet spot and 1.0 returns decayed weight")
    func partialDecay() {
        // 50/100 = 0.50 -> somewhere between maxWeight and floor
        let weight = sut.effectiveKeywordWeight(matchCount: 50, totalCount: 100)
        #expect(weight > sut.floor)
        #expect(weight < sut.maxKeywordWeight)
    }

    @Test("Weight is monotonically non-increasing with match ratio")
    func monotonic() {
        var previous = sut.effectiveKeywordWeight(matchCount: 1, totalCount: 100)
        for matchCount in stride(from: 2, through: 100, by: 1) {
            let current = sut.effectiveKeywordWeight(matchCount: matchCount, totalCount: 100)
            #expect(current <= previous + 1e-10,
                    "Weight should not increase: \(current) > \(previous) at matchCount=\(matchCount)")
            previous = current
        }
    }

    // MARK: - Custom parameters

    @Test("Custom parameters are respected")
    func customParameters() {
        let custom = AdaptiveHybridScorer(
            maxKeywordWeight: 0.20,
            sweetSpot: 0.10,
            floor: 0.05
        )
        // Below sweet spot -> maxWeight
        let w1 = custom.effectiveKeywordWeight(matchCount: 5, totalCount: 100)
        #expect(abs(w1 - 0.20) < 1e-10)

        // All match -> floor
        let w2 = custom.effectiveKeywordWeight(matchCount: 100, totalCount: 100)
        #expect(abs(w2 - 0.05) < 1e-10)
    }

    // MARK: - blend

    @Test("Blend with zero keyword weight returns pure semantic")
    func blendPureSemantic() {
        let result = sut.blend(semanticScore: 0.8, keywordScore: 0.5, keywordWeight: 0.0)
        #expect(abs(result - 0.8) < 1e-6)
    }

    @Test("Blend with keyword weight of 1.0 returns pure keyword")
    func blendPureKeyword() {
        let result = sut.blend(semanticScore: 0.8, keywordScore: 0.5, keywordWeight: 1.0)
        #expect(abs(result - 0.5) < 1e-6)
    }

    @Test("Blend is a convex combination")
    func blendConvex() {
        let sem: Float = 0.9
        let kw = 0.6
        let kwWeight = 0.3
        let expected = (1.0 - kwWeight) * Double(sem) + kwWeight * kw
        let result = sut.blend(semanticScore: sem, keywordScore: kw, keywordWeight: kwWeight)
        #expect(abs(result - expected) < 1e-10)
    }
}
