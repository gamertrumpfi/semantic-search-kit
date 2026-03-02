import Foundation

/// Ground-truth test cases for the synthetic 26-item test corpus.
///
/// 36 queries covering various retrieval scenarios:
/// - Direct title matches (`STRONG_KW`)
/// - Semantic-only matches with zero keyword overlap (`ZERO_KW`)
/// - Ambiguous / broad queries (`AMBIGUOUS`)
/// - Cross-category queries (`CROSS`)
///
/// Format: `(query, expectedTitleSubstrings, hint)`
///
/// `hint` values:
///   - `"STRONG_KW"`: strong literal keyword match
///   - `"ZERO_KW"`:   no keyword overlap — pure semantic must carry
///   - `"AMBIGUOUS"`:  broad/vague query — both signals useful
///   - `"CROSS"`:     crosses category boundaries
///   - `nil`:         standard query
enum BenchmarkTestCases {

    typealias TestCase = (query: String, expected: [String], hint: String?)

    static let all: [TestCase] = books + movies + games + travel + gadgets
        + crossCategory + zeroKeywordOverlap + ambiguous

    // MARK: - Books (5)

    static let books: [TestCase] = [
        ("science fiction space comedy",
         ["Hitchhiker"], nil),
        ("desert planet political intrigue",
         ["Dune"], nil),
        ("romantic novel England manners",
         ["Pride and Prejudice"], nil),
        ("cyberpunk hacker virtual reality",
         ["Neuromancer"], nil),
        ("1920s American wealth obsession",
         ["Great Gatsby"], nil),
    ]

    // MARK: - Movies (5)

    static let movies: [TestCase] = [
        ("dystopian replicant future",
         ["Blade Runner"], nil),
        ("animated bathhouse spirits",
         ["Spirited Away"], nil),
        ("prison escape friendship drama",
         ["Shawshank"], nil),
        ("dream heist layers subconscious",
         ["Inception"], nil),
        ("French romantic waitress Montmartre",
         ["Amélie"], nil),
    ]

    // MARK: - Games (4)

    static let games: [TestCase] = [
        ("resource trading settlers island",
         ["Catan"], nil),
        ("cooperative disease outbreak teamwork",
         ["Pandemic"], nil),
        ("train railway route city",
         ["Ticket to Ride"], nil),
        ("word clue spy party game",
         ["Codenames"], nil),
    ]

    // MARK: - Travel (5)

    static let travel: [TestCase] = [
        ("Japanese temples shrines walking",
         ["Kyoto"], nil),
        ("glacier hiking Chilean wilderness",
         ["Patagonia"], nil),
        ("Italian coastal cliffside Mediterranean",
         ["Amalfi"], nil),
        ("Moroccan market souks street food",
         ["Marrakech"], nil),
        ("aurora borealis husky sledding winter",
         ["Northern Lights"], nil),
    ]

    // MARK: - Gadgets (5)

    static let gadgets: [TestCase] = [
        ("waterproof solar outdoor speaker",
         ["Solar-Powered Trail Speaker"], "STRONG_KW"),
        ("digital paper drawing stylus e-ink",
         ["E-Ink Sketch Tablet"], nil),
        ("sleep earbuds noise cancelling white noise",
         ["Noise-Cancelling Sleep Earbuds"], "STRONG_KW"),
        ("compact mechanical keyboard Bluetooth travel",
         ["Mechanical Keyboard"], "STRONG_KW"),
        ("retro handheld emulator classic games",
         ["Retro Gaming Console"], nil),
    ]

    // MARK: - Cross-category (4)

    static let crossCategory: [TestCase] = [
        ("science fiction",
         ["Hitchhiker", "Dune", "Neuromancer", "Blade Runner", "Inception"], "CROSS"),
        ("Japanese culture",
         ["Kyoto", "Spirited Away"], "CROSS"),
        ("portable compact on the go",
         ["Trail Speaker", "Keyboard Travel", "Retro Gaming", "E-Ink"], "CROSS"),
        ("cooperative strategy game",
         ["Pandemic", "Catan"], "CROSS"),
    ]

    // MARK: - Zero keyword overlap (4)

    static let zeroKeywordOverlap: [TestCase] = [
        ("funny book about the meaning of life",
         ["Hitchhiker"], "ZERO_KW"),
        ("movie about class inequality in Korea",
         ["Parasite"], "ZERO_KW"),
        ("device for reading and writing without eye strain",
         ["E-Ink Sketch Tablet"], "ZERO_KW"),
        ("experience the dancing green sky in Scandinavia",
         ["Northern Lights"], "ZERO_KW"),
    ]

    // MARK: - Ambiguous / broad (4)

    static let ambiguous: [TestCase] = [
        ("something fun for a group",
         ["Catan", "Codenames", "Pandemic", "Ticket to Ride"], "AMBIGUOUS"),
        ("relaxing travel experience",
         ["Amalfi", "Kyoto", "Northern Lights"], "AMBIGUOUS"),
        ("history of the world",
         ["Sapiens"], "AMBIGUOUS"),
        ("dark thriller suspense",
         ["Inception", "Parasite", "Neuromancer", "Blade Runner"], "AMBIGUOUS"),
    ]
}
