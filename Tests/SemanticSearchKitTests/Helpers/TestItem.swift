import Foundation
@testable import SemanticSearchKit

/// A minimal ``SemanticSearchable`` type for unit testing.
///
/// Avoids coupling tests to any domain model. Each item has a unique `id`
/// and a `text` string used for keyword matching.
struct TestItem: SemanticSearchable, Sendable {
    let id: String
    let title: String
    let text: String

    var searchID: String { id }
    var searchableText: String { "\(title) \(text)" }

    /// Convenience initialiser with sensible defaults.
    init(
        id: String,
        title: String = "",
        text: String = ""
    ) {
        self.id = id
        self.title = title
        self.text = text
    }
}
