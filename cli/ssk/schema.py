"""Pydantic v2 models for the corpus and queries input schemas.

These define the contract between the user's data and every stage of the
pipeline — passage construction, triplet mining, embedding generation, and
benchmark evaluation.

Typical usage::

    from ssk.schema import CorpusFile, QueriesFile

    corpus = CorpusFile.load("my_corpus.json")
    queries = QueriesFile.load("my_queries.json")  # optional
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Tag (supports plain strings AND key-value translation pairs)
# ---------------------------------------------------------------------------


class TranslationTag(BaseModel):
    """A tag with an optional translation for cross-lingual boosting.

    When rendered into a passage:
    - If ``translation`` is present: ``"name (translation)"``
    - Otherwise: ``"name"``
    """

    name: str = Field(min_length=1)
    translation: str | None = None

    def render(self) -> str:
        if self.translation:
            return f"{self.name} ({self.translation})"
        return self.name


# ---------------------------------------------------------------------------
# Structured field (arbitrary label/value pairs)
# ---------------------------------------------------------------------------


class StructuredField(BaseModel):
    """An arbitrary key-value pair appended to the passage text.

    Examples: ``{"label": "Ingredients", "value": "eggs, cheese, pasta"}``
              ``{"label": "Director", "value": "Stanley Kubrick"}``
    """

    label: str = Field(min_length=1)
    value: str = Field(min_length=1)

    def render(self) -> str:
        return f"{self.label}: {self.value}"


# ---------------------------------------------------------------------------
# Corpus item
# ---------------------------------------------------------------------------


class CorpusItem(BaseModel):
    """A single document in the search corpus.

    At minimum, ``id`` and ``title`` are required.  The embedding text is
    constructed from whichever optional fields are present:

    - ``text`` — primary free-text description (highest priority)
    - ``tags`` — categorical labels, optionally with translations
    - ``fields`` — arbitrary structured key-value pairs

    At least one of ``text``, ``tags``, or ``fields`` must be provided so
    the pipeline has something meaningful to embed.
    """

    id: str = Field(min_length=1, description="Unique document identifier.")
    title: str = Field(min_length=1, description="Display title.")
    text: str | None = Field(
        default=None,
        description="Primary free-text for embedding (e.g. description).",
    )
    tags: list[str | TranslationTag] | None = Field(
        default=None,
        description="Categorical tags. Plain strings or {name, translation} objects.",
    )
    fields: list[StructuredField] | None = Field(
        default=None,
        description="Arbitrary structured key-value pairs.",
    )

    @model_validator(mode="after")
    def _require_searchable_content(self) -> Self:
        has_text = bool(self.text and self.text.strip())
        has_tags = bool(self.tags)
        has_fields = bool(self.fields)
        if not (has_text or has_tags or has_fields):
            raise ValueError(
                f"Item '{self.id}': at least one of 'text', 'tags', or 'fields' "
                f"must be provided so the pipeline has content to embed."
            )
        return self

    # -- Passage construction -----------------------------------------------

    def to_passage(self) -> str:
        """Build the passage string sent to the embedding model.

        Format follows the E5 ``"passage: ..."`` convention:

        - With ``text``:   ``passage: <title>  <text>  Tags: ...  Field: ...``
        - Without ``text``: ``passage: Title: <title>  Tags: ...  Field: ...``
        """
        parts: list[str] = []

        if self.text and self.text.strip():
            parts.append(self.title)
            parts.append(self.text.strip())
        else:
            parts.append(f"Title: {self.title}")

        if self.tags:
            rendered = ", ".join(
                tag.render() if isinstance(tag, TranslationTag) else tag
                for tag in self.tags
            )
            parts.append(f"Tags: {rendered}")

        if self.fields:
            for field in self.fields:
                parts.append(field.render())

        return "passage: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Query / benchmark case
# ---------------------------------------------------------------------------


class QueryCase(BaseModel):
    """A single benchmark query with expected results for evaluation.

    ``expected`` is a list of title substrings — a hit is counted if any
    retrieved title contains any of these substrings (case-insensitive).
    """

    query: str = Field(min_length=1)
    expected: list[str] = Field(min_length=1)
    hint: str | None = Field(
        default=None,
        description="Optional miss-reason tag for reporting (e.g. 'LANG', 'SYNONYM').",
    )


# ---------------------------------------------------------------------------
# File-level wrappers with I/O helpers
# ---------------------------------------------------------------------------


class CorpusFile(BaseModel):
    """Validated list of corpus items with file I/O."""

    items: list[CorpusItem]

    @classmethod
    def load(cls, path: str | Path) -> CorpusFile:
        """Load and validate a corpus JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"corpus.json must be a JSON array, got {type(raw).__name__}")
        return cls(items=[CorpusItem.model_validate(item) for item in raw])

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)


class QueriesFile(BaseModel):
    """Validated list of benchmark queries with file I/O."""

    cases: list[QueryCase]

    @classmethod
    def load(cls, path: str | Path) -> QueriesFile:
        """Load and validate a queries JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"queries.json must be a JSON array, got {type(raw).__name__}")
        return cls(cases=[QueryCase.model_validate(item) for item in raw])

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):  # type: ignore[override]
        return iter(self.cases)
