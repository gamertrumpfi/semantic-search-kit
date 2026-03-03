import Testing
import Foundation
@testable import SemanticSearchKit

/// Tests for ``KeywordScorer``.
@Suite("KeywordScorer")
struct KeywordScorerTests {

    private let sut = KeywordScorer()

    /// A small catalogue exercising title and text matching.
    private static let items: [TestItem] = [
        TestItem(
            id: "spaghetti",
            title: "Spaghetti Carbonara",
            text: "Classic Italian pasta with eggs and parmesan"
        ),
        TestItem(
            id: "curry",
            title: "Thai Red Curry",
            text: "Spicy coconut curry with chicken"
        ),
        TestItem(
            id: "salad",
            title: "Greek Salad",
            text: "Fresh summer salad with feta and olives"
        ),
        TestItem(
            id: "cake",
            title: "Chocolate Cake",
            text: "Rich dark chocolate dessert with ganache"
        ),
    ]

    // MARK: - scores(for:in:)

    @Test("Single keyword matches item by title")
    func singleKeywordMatchesTitle() {
        let results = sut.scores(for: "spaghetti", in: Self.items)
        #expect(!results.isEmpty)
        #expect(results[0].item.id == "spaghetti")
    }

    @Test("Multi-keyword weighted scoring: first keyword has highest weight")
    func multiKeywordWeighting() {
        // "chocolate" matches cake; "curry" matches curry
        // If chocolate is first keyword (weight 3.0), cake should rank higher.
        let chocFirst = sut.scores(for: "chocolate curry", in: Self.items)
        let curryFirst = sut.scores(for: "curry chocolate", in: Self.items)

        #expect(chocFirst.first?.item.id == "cake")
        #expect(curryFirst.first?.item.id == "curry")
    }

    @Test("No matches returns empty array")
    func noMatches() {
        let results = sut.scores(for: "nonexistentfoodxyz", in: Self.items)
        #expect(results.isEmpty)
    }

    @Test("Search is case-insensitive")
    func caseInsensitive() {
        let results = sut.scores(for: "SPAGHETTI", in: Self.items)
        #expect(!results.isEmpty)
        #expect(results[0].item.id == "spaghetti")
    }

    @Test("Matches in text field are found")
    func matchesInText() {
        // "parmesan" is in spaghetti's text
        let results = sut.scores(for: "parmesan", in: Self.items)
        #expect(results.contains { $0.item.id == "spaghetti" })

        // "feta" is in salad's text
        let results2 = sut.scores(for: "feta", in: Self.items)
        #expect(results2.contains { $0.item.id == "salad" })
    }

    @Test("Empty query returns empty array")
    func emptyQuery() {
        let results = sut.scores(for: "", in: Self.items)
        #expect(results.isEmpty)
    }

    @Test("Results are sorted by score descending")
    func sortedDescending() {
        let results = sut.scores(for: "chocolate curry", in: Self.items)
        for i in 1 ..< results.count {
            #expect(results[i - 1].score >= results[i].score)
        }
    }

    @Test("Weight scheme: 1st=3.0, 2nd=2.5, 3rd=2.0, 4th+=1.0")
    func weightScheme() {
        // An item that matches all 4 keywords should get 3.0 + 2.5 + 2.0 + 1.0 = 8.5
        let items = [TestItem(id: "all", title: "alpha beta gamma delta")]
        let results = sut.scores(for: "alpha beta gamma delta", in: items)
        #expect(results.count == 1)
        #expect(abs(results[0].score - 8.5) < 1e-10)
    }

    // MARK: - normalisedScores(for:in:)

    @Test("Normalised scores are in [0, 1] range")
    func normalisedRange() {
        let (scores, _) = sut.normalisedScores(for: "chocolate curry", in: Self.items)
        for (_, score) in scores {
            #expect(score >= 0.0)
            #expect(score <= 1.0)
        }
    }

    @Test("Max normalised score is exactly 1.0")
    func maxNormalisedIsOne() {
        let (scores, _) = sut.normalisedScores(for: "chocolate curry", in: Self.items)
        let maxScore = scores.values.max() ?? 0
        #expect(abs(maxScore - 1.0) < 1e-10)
    }

    @Test("Normalised matchCount is correct")
    func normalisedMatchCount() {
        let (_, matchCount) = sut.normalisedScores(for: "chocolate", in: Self.items)
        // Only cake matches "chocolate"
        #expect(matchCount == 1)
    }

    @Test("No matches returns empty scores and zero matchCount")
    func normalisedNoMatches() {
        let (scores, matchCount) = sut.normalisedScores(for: "zzzzz", in: Self.items)
        #expect(scores.isEmpty)
        #expect(matchCount == 0)
    }

    // MARK: - Field-weighted scoring

    @Test("Field weight multiplies term weight")
    func fieldWeightMultipliesTermWeight() {
        // Single item with title (weight 3.0) and description (weight 1.0).
        // "pasta" appears only in the title field.
        let items = [
            TestItem(id: "a", fields: [
                SearchableField(text: "pasta carbonara", weight: 3.0),
                SearchableField(text: "a classic Italian dish", weight: 1.0),
            ])
        ]
        let results = sut.scores(for: "pasta", in: items)
        #expect(results.count == 1)
        // 1st term weight (3.0) × field weight (3.0) = 9.0
        #expect(abs(results[0].score - 9.0) < 1e-10)
    }

    @Test("Best field weight wins when term matches multiple fields")
    func bestFieldWeightWins() {
        // "chicken" appears in both title (weight 3.0) and description (weight 1.0).
        // Only the highest field weight should count.
        let items = [
            TestItem(id: "a", fields: [
                SearchableField(text: "chicken tikka", weight: 3.0),
                SearchableField(text: "spicy chicken curry", weight: 1.0),
            ])
        ]
        let results = sut.scores(for: "chicken", in: items)
        #expect(results.count == 1)
        // 1st term weight (3.0) × best field weight (3.0) = 9.0
        #expect(abs(results[0].score - 9.0) < 1e-10)
    }

    @Test("Field weights change ranking order")
    func fieldWeightsChangeRanking() {
        // Two items both match "pasta". Item A has it in a high-weight field,
        // item B has it in a low-weight field.
        let items = [
            TestItem(id: "title-match", fields: [
                SearchableField(text: "pasta bolognese", weight: 3.0),
                SearchableField(text: "hearty Italian dinner", weight: 1.0),
            ]),
            TestItem(id: "desc-match", fields: [
                SearchableField(text: "Italian dinner", weight: 3.0),
                SearchableField(text: "served with fresh pasta", weight: 1.0),
            ]),
        ]
        let results = sut.scores(for: "pasta", in: items)
        #expect(results.count == 2)
        // title-match: 3.0 × 3.0 = 9.0
        // desc-match:  3.0 × 1.0 = 3.0
        #expect(results[0].item.id == "title-match")
        #expect(results[1].item.id == "desc-match")
    }

    @Test("Default field weight of 1.0 preserves backward compatibility")
    func defaultFieldWeightBackwardCompat() {
        // TestItem without custom fields uses the default (single field, weight 1.0).
        // Score should equal the plain term weight sum.
        let items = [TestItem(id: "plain", title: "alpha beta gamma delta")]
        let results = sut.scores(for: "alpha beta gamma delta", in: items)
        #expect(results.count == 1)
        // 3.0 + 2.5 + 2.0 + 1.0 = 8.5 (same as before field weights existed)
        #expect(abs(results[0].score - 8.5) < 1e-10)
    }

    @Test("Multi-term query with mixed field matches scores correctly")
    func multiTermMixedFields() {
        // "chicken" in title (weight 2.0), "spicy" in tags (weight 1.5)
        let items = [
            TestItem(id: "a", fields: [
                SearchableField(text: "chicken tikka", weight: 2.0),
                SearchableField(text: "spicy Indian", weight: 1.5),
                SearchableField(text: "a classic dish", weight: 1.0),
            ])
        ]
        let results = sut.scores(for: "chicken spicy", in: items)
        #expect(results.count == 1)
        // "chicken": 1st term (3.0) × field (2.0) = 6.0
        // "spicy":   2nd term (2.5) × field (1.5) = 3.75
        // Total = 9.75
        #expect(abs(results[0].score - 9.75) < 1e-10)
    }

    @Test("Empty fields array produces zero score")
    func emptyFieldsArray() {
        let items = [TestItem(id: "empty", fields: [])]
        let results = sut.scores(for: "anything", in: items)
        #expect(results.isEmpty)
    }
}
