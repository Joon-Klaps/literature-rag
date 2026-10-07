"""Search on a made-up collection of four chunks: the tokenizer, BM25, dense ranking, reciprocal rank fusion and reranking, with toy vectors and stand-in models."""

import numpy as np
import pytest

from literature_rag import models, search


def chunk(n: int, text: str) -> dict:
    return {
        "chunk_id": f"Doe2015-valley#{n}",
        "paper_id": "Doe2015-valley",
        "key": "Doe2015-valley",
        "citable": True,
        "title": "Rodents of a made-up valley",
        "year": 2015,
        "doi": None,
        "section": "Results",
        "kind": "body",
        "text": text,
        "cites": [],
    }


class ToyEmbedder:
    """Every query lands closest to chunk 1, then 3, then 0, then 2, since chunk n's toy vector points along axis n."""

    def encode_queries(self, queries, instruction):
        return np.array([[0.1, 0.9, 0.0, 0.5]] * len(queries), dtype=np.float32)


class ToyReranker:
    """Scores a passage 1 when it mentions rainfall and 0 otherwise."""

    def predict(self, pairs, batch_size=None):
        return np.array([1.0 if "Rainfall" in text else 0.0 for _, text in pairs])


@pytest.fixture
def collection(monkeypatch):
    monkeypatch.setattr(models, "load_embedder", lambda name: ToyEmbedder())
    monkeypatch.setattr(models, "load_reranker", lambda: ToyReranker())
    chunks = [
        chunk(0, "Mastomys natalensis was trapped in every village."),
        chunk(1, "The case fatality rate among hospitalised patients was 25%."),
        chunk(2, "Sequencing used Illumina reads aligned with bwa."),
        chunk(3, "Rainfall peaked in August."),
    ]
    return search.Collection("library", chunks, {"qwen3": np.eye(4, dtype=np.float32)})


def rows(hits):
    return [int(hit["chunk"]["chunk_id"].rsplit("#", 1)[1]) for hit in hits]


def test_tokenize():
    assert search.tokenize("Mastomys natalensis: GPC-2, 25% (n=1,234)") == [
        "mastomys",
        "natalensis",
        "gpc",
        "2",
        "25",
        "n",
        "1",
        "234",
    ]


def test_rrf():
    fused = dict(search.rrf([["a", "b", "c"], ["b", "c", "d"]]))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["d"] == pytest.approx(1 / 63)
    # Ranked well by both beats ranked first by one.
    assert [item for item, _ in search.rrf([["a", "b", "c"], ["b", "c", "d"]])] == ["b", "c", "a", "d"]
    assert search.rrf([[1, 2]], k=0) == [(1, 1.0), (2, 0.5)]
    # Equal scores keep the order in which items were first seen.
    assert [item for item, _ in search.rrf([["x"], ["y"]])] == ["x", "y"]


def test_bm25(collection):
    # Both match one word of the query with the same weight, and the shorter chunk wins.
    assert [row for row, _ in search.bm25(collection, "rainfall village village")] == [3, 0]
    assert search.bm25(collection, "word in no chunk") == []


def test_dense(collection):
    ranking = search.dense(collection, "qwen3", np.array([0.1, 0.9, 0.0, 0.5], dtype=np.float32))
    assert [row for row, _ in ranking] == [1, 3, 0, 2]
    assert ranking[0][1] == pytest.approx(0.9)
    assert len(search.dense(collection, "qwen3", np.ones(4, dtype=np.float32), n=2)) == 2


def test_search_one_method(collection):
    hits = search.search(collection, "rainfall village", methods=["bm25"], rerank=False)
    assert rows(hits) == [3, 0]
    # Without fusion, the score is the method's own and there is no fused place.
    assert hits[0]["score"] == search.bm25(collection, "rainfall village")[0][1]
    assert hits[0]["ranks"] == {"bm25": 1}
    assert [hit["rank"] for hit in hits] == [1, 2]


def test_search_fused(collection):
    timings = {}
    hits = search.search(collection, "rainfall village", k=3, methods=["bm25", "qwen3"], timings=timings, rerank=False)
    # BM25 gives 3, 0; the vectors give 1, 3, 0, 2. Chunk 3 is first and second, chunk 0 second and third, chunk 1 only first.
    assert rows(hits) == [3, 0, 1]
    assert hits[0]["ranks"] == {"bm25": 1, "qwen3": 2, "fused": 1}
    assert hits[2]["ranks"] == {"qwen3": 1, "fused": 3}
    assert hits[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert set(timings) == {"bm25", "qwen3"}


def test_search_rerank(collection):
    timings = {}
    hits = search.search(collection, "anything", methods=["qwen3"], rerank=True, timings=timings)
    # The reranker lifts the one passage about rainfall; the others tie at 0 and keep their order.
    assert rows(hits) == [3, 1, 0, 2]
    assert hits[0]["score"] == 1.0
    assert hits[0]["ranks"] == {"qwen3": 2}
    assert "rerank" in timings
    assert search.search(collection, "word in no chunk", methods=["bm25"], rerank=True) == []


def test_search_unknown_method(collection):
    with pytest.raises(ValueError, match="unknown methods"):
        search.search(collection, "anything", methods=["bm25", "colbert"], rerank=False)


def test_describe(collection):
    (hit,) = search.search(collection, "rainfall", k=1, methods=["bm25"], rerank=False)
    first, second = search.describe(hit, words=2).splitlines()
    assert "Doe2015-valley (2015) | Results | bm25 1" in first
    assert second.strip() == "Rainfall peaked ..."
