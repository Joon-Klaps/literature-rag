"""The vector store on made-up chunks and a stand-in model: vectors found by their text, only new text embedded, stale indexes refused."""

import numpy as np
import pytest

from literature_rag import chunks, config, index, models


def chunk(n: int, text: str) -> dict:
    return {"chunk_id": f"Doe2015-valley#{n}", "title": "Rodents", "section": "Results", "text": text}


class CountingEmbedder:
    """Gives each passage the vector (length of its header, length of its text), and remembers what it was asked to embed."""

    def __init__(self):
        self.seen = []

    def encode_passages(self, passages):
        self.seen.extend(passages)
        return np.array([[len(header), len(text)] for header, text in passages], dtype=np.float32)


@pytest.fixture
def embedder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    embedder = CountingEmbedder()
    monkeypatch.setattr(models, "load_embedder", lambda name: embedder)
    return embedder


def test_reuse_by_text(embedder):
    first = [chunk(0, "one"), chunk(1, "three")]
    assert index.embed("library", "qwen3", first) == (0, 2)
    # A new chunk in front renumbers the others, but their text is unchanged, so only the new one is embedded.
    embedder.seen.clear()
    second = [chunk(0, "seventeen"), chunk(1, "one"), chunk(2, "three")]
    assert index.embed("library", "qwen3", second) == (2, 1)
    assert embedder.seen == [("Rodents. Results.", "seventeen")]
    vectors = index.load_vectors("library", "qwen3", second)
    assert vectors.dtype == np.float32
    assert vectors[:, 1].tolist() == [9, 3, 5]
    # Nothing new: nothing embedded.
    embedder.seen.clear()
    assert index.embed("library", "qwen3", second) == (3, 0)
    assert embedder.seen == []


def test_stale_and_pruned(embedder):
    index.embed("library", "qwen3", [chunk(0, "one"), chunk(1, "three")])
    changed = [chunk(0, "one"), chunk(1, "three, revised")]
    assert index.missing("library", "qwen3", changed) == 1
    with pytest.raises(index.StaleIndex, match="1 of 2 library chunks have no qwen3 vector"):
        index.load_vectors("library", "qwen3", changed)
    index.embed("library", "qwen3", changed)
    assert index.load_vectors("library", "qwen3", changed)[:, 1].tolist() == [3, 14]
    # The vector of the old text is dropped.
    hashes, vectors = index.read_store("library", "qwen3")
    assert len(hashes) == len(vectors) == 2


def test_rounds_longest_first(embedder, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_ROUND", 2)
    texts = ["a", "abcde", "abc", "abcd", "ab"]
    assert index.embed("library", "qwen3", [chunk(n, text) for n, text in enumerate(texts)]) == (0, 5)
    assert [text for _, text in embedder.seen] == ["abcde", "abcd", "abc", "ab", "a"]
    assert index.missing("library", "qwen3", [chunk(n, text) for n, text in enumerate(texts)]) == 0


def test_settings_change(embedder, monkeypatch):
    index.embed("library", "qwen3", [chunk(0, "one")])
    # Vectors made under other settings are not reused.
    monkeypatch.setitem(config.EMBEDDING_MODELS, "qwen3", {**config.EMBEDDING_MODELS["qwen3"], "passage_tokens": 256})
    assert index.missing("library", "qwen3", [chunk(0, "one")]) == 1
    assert index.embed("library", "qwen3", [chunk(0, "one")]) == (0, 1)


def test_half_written_store_is_ignored(embedder):
    index.embed("library", "qwen3", [chunk(0, "one"), chunk(1, "three")])
    _, vectors = index.read_store("library", "qwen3")
    vectors_path, _ = index.paths("library", "qwen3")
    np.save(vectors_path, vectors[:1].astype(np.float16))
    assert index.read_store("library", "qwen3")[0] == []


def test_index_parts():
    assert chunks.index_parts(chunk(0, "Text.")) == ("Rodents. Results.", "Text.")
    assert chunks.index_parts({"section": "Introduction > Lassa fever", "text": "Text."}) == (
        "Introduction > Lassa fever.",
        "Text.",
    )
    assert chunks.index_parts({"title": "", "section": "", "text": "Text."}) == ("", "Text.")
