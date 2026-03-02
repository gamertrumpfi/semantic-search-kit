"""Generic passage construction from :class:`~ssk.schema.CorpusItem`.

The passage string is the input sent to the embedding model.  It follows the
E5 ``"passage: ..."`` convention and concatenates all available fields from
the corpus item into a single searchable text.

Passage format:

- With ``text``:    ``passage: <title>  <text>  Tags: ...  Field: ...``
- Without ``text``: ``passage: Title: <title>  Tags: ...  Field: ...``

Tags with translations are rendered as ``name (translation)``.
Structured fields are rendered as ``Label: value``.

Typical usage::

    from ssk.schema import CorpusFile
    from ssk.passage import make_passage, make_passages

    corpus = CorpusFile.load("corpus.json")
    passages = make_passages(corpus.items)
"""

from __future__ import annotations

from ssk.schema import CorpusItem


def make_passage(item: CorpusItem) -> str:
    """Build the passage string for a single corpus item.

    This delegates to :meth:`CorpusItem.to_passage` to keep the passage
    format in a single canonical location.  This function exists as a
    convenient free-standing entry point for pipeline modules.
    """
    return item.to_passage()


def make_passages(items: list[CorpusItem]) -> list[str]:
    """Build passage strings for all corpus items.

    Returns a list of passages in the same order as the input items.
    """
    return [make_passage(item) for item in items]
