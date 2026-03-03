import Foundation
@testable import SemanticSearchKit

/// A minimal ``SemanticSearchable`` type for unit testing.
///
/// Avoids coupling tests to any domain model. Each item has a unique `id`
/// and a `text` string used for keyword matching.
///
/// When `fields` is non-nil, ``searchableFields`` returns those custom fields
/// instead of the default single-field wrapper around ``searchableText``.
struct TestItem: SemanticSearchable, Sendable {
    let id: String
    let title: String
    let text: String
    let fields: [SearchableField]?

    var searchID: String { id }
    var searchableText: String { "\(title) \(text)" }

    var searchableFields: [SearchableField] {
        fields ?? [SearchableField(text: searchableText)]
    }

    /// Convenience initialiser with sensible defaults.
    init(
        id: String,
        title: String = "",
        text: String = "",
        fields: [SearchableField]? = nil
    ) {
        self.id = id
        self.title = title
        self.text = text
        self.fields = fields
    }
}
