# Semantic Search Kit

Drop-in on-device semantic search for iOS and macOS apps, powered by CoreML.

Semantic Search Kit is a two-part SDK:

1. **Swift Package** (`SemanticSearchKit`) -- a generic hybrid search engine that blends CoreML embedding similarity with keyword scoring. Targets iOS 16+ / macOS 13+, Swift 6 strict concurrency.
2. **Python CLI** (`ssk`) -- a one-command pipeline that distils a HuggingFace teacher model into a mobile-sized CoreML student model from any dataset.

## How It Works

```
corpus.json ──► ssk run ──► .mlpackage + embeddings_index.json ──► SemanticSearchKit
                  │
                  ├─ 1. Baseline benchmark (teacher)
                  ├─ 2. Hard-negative triplet mining
                  ├─ 3. MNRL + MarginMSE student distillation
                  ├─ 4. CoreML export (fp16 quantised)
                  └─ 5. Embedding generation
```

The Python CLI trains a small student model (e.g. multilingual-e5-small, 118M params) from a large teacher (e.g. e5-large-instruct, 560M params) on your specific corpus. The exported CoreML model runs entirely on-device via the Neural Engine -- no server, no API keys, no network.

## Quick Start

### 1. Train and export a model

```bash
pip install semantic-search-kit

ssk run \
  --corpus my_corpus.json \
  --queries my_queries.json \
  --output ./output/ \
  --name "MyAppSearch"
```

This produces:
- `MyAppSearch.mlpackage` -- CoreML model for Xcode
- `embeddings_index.json` -- pre-computed document embeddings for the app bundle

### 2. Integrate into your app

Add the Swift package dependency:

```swift
.package(url: "https://github.com/gamertrumpfi/semantic-search-kit", from: "0.1.0")
```

Configure once at app launch:

```swift
import SemanticSearchKit

SemanticSearchKit.configure(modelName: "MyAppSearch", bundle: .main)
```

Conform your type:

```swift
struct Product: SemanticSearchable {
    let id: String
    let name: String
    let description: String

    var searchID: String { id }
    var searchableText: String { "\(name) \(description)" }
}
```

Search:

```swift
let engine = SemanticSearchEngine<Product>()
let results = try await engine.search(query: "wireless headphones", in: products, topK: 10)

for result in results {
    print("\(result.item.name) — hybrid: \(result.hybridScore)")
}
```

## Input Schema

### corpus.json (required)

A JSON array of items. Each item requires `id` and `title`, plus at least one of `text`, `tags`, or `fields`:

```json
[
  {
    "id": "prod-001",
    "title": "Wireless Headphones Pro",
    "text": "Premium noise-cancelling headphones with 30-hour battery.",
    "tags": ["audio", "wireless", {"name": "Kopfhoerer", "translation": "headphones"}],
    "fields": [
      {"label": "Brand", "value": "Acme Audio"},
      {"label": "Price", "value": "$299"}
    ]
  }
]
```

Tags support both plain strings and `{name, translation}` objects for cross-lingual boosting.

### queries.json (optional, for benchmarking)

A JSON array of test cases with expected results:

```json
[
  {
    "query": "noise cancelling headphones",
    "expected": ["Wireless Headphones Pro"],
    "hint": "STRONG_KW"
  }
]
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ssk run` | Full end-to-end pipeline |
| `ssk benchmark` | Evaluate a model on test queries |
| `ssk triplets` | Generate training triplets from teacher |
| `ssk train` | Distil student via MNRL + MarginMSE |
| `ssk export` | Convert trained model to CoreML |
| `ssk embed` | Generate bundle embeddings |

Run `ssk --help` or `ssk <command> --help` for full option details.

## Architecture

### Swift Package

The search engine uses adaptive hybrid scoring:

- **Semantic signal**: cosine similarity between query embedding (CoreML, on-device) and pre-computed document embeddings
- **Keyword signal**: weighted positional scoring with title/text/tag priority
- **Adaptive blending**: keyword weight decays linearly as match count increases (fewer keyword matches = higher keyword weight), preventing broad queries from being dominated by noisy keyword hits

Key types:

| Type | Role |
|------|------|
| `SemanticSearchable` | Protocol your types conform to |
| `SemanticSearchEngine<T>` | Generic actor performing hybrid search |
| `SemanticSearchResult<T>` | Result with semantic, keyword, and hybrid scores |
| `EmbeddingStorage` | Actor loading pre-computed embeddings from bundle |
| `QueryEmbedder` | Actor running CoreML inference for query encoding |

### Python CLI

The training pipeline uses dual-loss knowledge distillation:

- **MNRL** (Multiple Negatives Ranking Loss): in-batch negatives for contrastive learning
- **MarginMSE**: teacher soft scores preserve ranking geometry in the student
- **Hard-negative mining**: teacher-scored negatives that are close to positives in embedding space

## Requirements

| Component | Requirement |
|-----------|-------------|
| Swift Package | iOS 16+, macOS 13+, Swift 6.0 |
| Python CLI | Python 3.10+, PyTorch, sentence-transformers, coremltools |

## Project Structure

```
semantic-search-kit/
  Package.swift              # Swift Package (SemanticSearchKit)
  Sources/
  Tests/
  cli/                       # Python CLI (ssk)
    pyproject.toml
    ssk/
    examples/
  README.md
  NOTICES
  LICENSE
```

## License

MIT. See [NOTICES](NOTICES) for third-party attributions.
