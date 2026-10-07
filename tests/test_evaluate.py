"""The evaluation's scoring on toy rankings: papers counted once, hit@k and MRR, bootstrap intervals, the number check, reranking depths, and the questions drawn from a made-up paragraph index."""

import numpy as np
import pytest

from literature_rag import config, evaluate, models, search


def chunk(paper: str, n: int, text: str, key: str | None = None) -> dict:
    return {
        "chunk_id": f"{paper}#{n}",
        "paper_id": paper,
        "key": key,
        "citable": key is not None,
        "title": "A made-up paper",
        "year": 2015,
        "doi": None,
        "section": "Results",
        "kind": "body",
        "text": text,
        "cites": [],
    }


def thesis_chunk(chunk_id: str, keys: list[str], kind: str = "paragraph") -> dict:
    return {"chunk_id": chunk_id, "kind": kind, "keys": keys}


class ToyEmbedder:
    """Every query lands closest to chunk 2, then 1, then 0, then 3."""

    def encode_queries(self, queries, instruction):
        return np.array([[0.3, 0.6, 0.9, 0.0]] * len(queries), dtype=np.float32)


class ToyReranker:
    """Scores a passage by its number of words, and counts the passages it reads."""

    def __init__(self):
        self.read = 0

    def predict(self, pairs, batch_size=None):
        self.read += len(pairs)
        return np.array([len(text.split()) for _, text in pairs], dtype=float)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(models, "load_embedder", lambda name: ToyEmbedder())
    monkeypatch.setattr(config, "EMBEDDING_MODELS", {"qwen3": {}})
    chunks = [
        chunk("Doe2015", 0, "Rainfall and rodents", key="Doe2015-valley"),
        chunk("Doe2015", 1, "Rodents in the village", key="Doe2015-valley"),
        chunk("Roe2020", 0, "Lassa virus in rodents of the valley", key="Roe2020-lassa"),
        chunk("Guidelines2019", 0, "Surveillance and rainfall"),
    ]
    return search.Collection("library", chunks, {"qwen3": np.eye(4, dtype=np.float32)})


def query(answer: str, numbers: tuple[str, ...] = ()) -> evaluate.Query:
    return evaluate.Query("q", "rainfall rodents", frozenset([answer]), numbers)


def test_paper_ranking_and_place():
    ranking = evaluate.paper_ranking(["a", "a", "b", "a", "c"])
    assert ranking == ["a", "b", "c"]
    assert evaluate.place(ranking, frozenset(["c"])) == 3
    # With several answers, the first one in the ranking counts.
    assert evaluate.place(ranking, frozenset(["c", "b"])) == 2
    assert evaluate.place(ranking, frozenset(["z"])) is None


def test_paper_of():
    assert evaluate.paper_of(chunk("Doe2015-supplement", 0, "", key="Doe2015-valley")) == "Doe2015-valley"
    assert evaluate.paper_of(chunk("Guidelines2019", 0, "")) == "Guidelines2019"


def test_metrics():
    places = [1, 3, None, 12]
    values = evaluate.metric_values(places)
    assert values["hit@1"] == [1, 0, 0, 0]
    assert values["hit@5"] == [1, 1, 0, 0]
    # The answer at 12 is below MRR@10, so it counts as missing.
    assert values["mrr@10"] == pytest.approx([1, 1 / 3, 0, 0])


def test_summarise():
    summary = evaluate.summarise([1, 1, 1, 1])
    assert summary["hit@1"] == {"mean": 1.0, "low": 1.0, "high": 1.0}
    summary = evaluate.summarise([1, None] * 50)
    assert summary["hit@5"]["mean"] == 0.5
    assert summary["hit@5"]["low"] < 0.5 < summary["hit@5"]["high"]
    # The seed is fixed, so the interval is the same on every run.
    assert evaluate.summarise([1, None] * 50) == summary


def test_difference_is_paired():
    # Two configurations that differ on every second query by the same margin: the paired interval is narrow and leaves out 0.
    ours = [1, 2] * 50
    theirs = [1, None] * 50
    delta = evaluate.difference(ours, theirs)["hit@5"]
    assert delta["mean"] == pytest.approx(0.5)
    assert delta["low"] > 0
    assert evaluate.difference(ours, ours)["mrr@10"] == {"mean": 0.0, "low": 0.0, "high": 0.0}


def test_numbers_found():
    assert evaluate.numbers_found(["1234", "25"], ["of 1,234 patients", "25% died"])
    assert not evaluate.numbers_found(["1234", "26"], ["of 1,234 patients", "25% died"])
    # A number inside a name is not a number, as in the thesis sentences.
    assert not evaluate.numbers_found(["19"], ["COVID-19 cases"])
    assert not evaluate.numbers_found(["5"], [])


def test_rerank_order():
    scores = {10: 0.1, 11: 0.9, 12: 0.5, 13: 1.0}
    assert evaluate.rerank_order([10, 11, 12, 13], scores, depth=4) == [13, 11, 12, 10]
    # Only the first depth rows are reordered, and the rest follow in fused order.
    assert evaluate.rerank_order([10, 11, 12, 13, 14], scores, depth=2) == [11, 10, 12, 13, 14]


def test_file_papers():
    rows = [
        {"source_file": "Doe 2015.md", "paper_id": "Doe2015-valley", "key": "Doe2015-valley", "duplicate_of": None},
        {
            "source_file": "Doe et al. 2015.md",
            "paper_id": "Doe et al. 2015",
            "key": "Doe2015-valley",
            "duplicate_of": "Doe 2015.md",
        },
        {"source_file": "Guidelines.md", "paper_id": "Guidelines", "key": None, "duplicate_of": None},
        {
            "source_file": "Guidelines copy.md",
            "paper_id": "Guidelines copy",
            "key": None,
            "duplicate_of": "Guidelines.md",
        },
    ]
    assert evaluate.file_papers(rows) == {
        "Doe 2015.md": "Doe2015-valley",
        "Doe et al. 2015.md": "Doe2015-valley",
        "Guidelines.md": "Guidelines",
        "Guidelines copy.md": "Guidelines",
    }


INDEX = """## Introduction — chapters/intro.tex

- `L15-17` — Rodents carry the virus. Cites: Doe2015-valley, Roe2020-lassa.
- `L19` — Nobody cites this. No citations.
- **Figure** `L25-30` (`fig:map`) — A map. Cites: Map2018.
- `L32` — Rainfall drives cases. Cites: Doe2015-valley.
- `L34` — Old summary of a changed paragraph. Cites: Gone2001.
"""


def test_index_entries():
    assert evaluate.index_entries(INDEX) == [
        ("Rodents carry the virus.", ["Doe2015-valley", "Roe2020-lassa"]),
        ("Rainfall drives cases.", ["Doe2015-valley"]),
        ("Old summary of a changed paragraph.", ["Gone2001"]),
    ]


def test_thesis_questions():
    paragraphs = [
        thesis_chunk("intro:L15-17", ["Roe2020-lassa", "doe2015-valley"]),
        thesis_chunk("intro:L32-32", ["Doe2015-valley"]),
        thesis_chunk("intro:L40-40", ["Doe2015-valley"]),
        thesis_chunk("intro:L25-30", ["Map2018"], kind="caption"),
    ]
    questions, eligible = evaluate.thesis_questions(evaluate.index_entries(INDEX), paragraphs, size=5)
    # The first summary's keys pick out one paragraph, whatever their case and order; the second's two; the third's none.
    assert eligible == 1
    assert questions == [evaluate.Query("index/0", "Rodents carry the virus.", frozenset(["intro:L15-17"]))]
    many = [(f"Summary {n}.", [f"Key{n}"]) for n in range(10)]
    chosen, eligible = evaluate.thesis_questions(
        many, [thesis_chunk(f"c:L{n}-{n}", [f"Key{n}"]) for n in range(10)], size=3
    )
    assert eligible == 10
    assert len(chosen) == 3
    # Kept in the index's order.
    assert chosen == sorted(chosen, key=lambda question: int(question.query_id.split("/")[1]))


def test_realistic_queries():
    question = {"question_id": "PL0.1.1/x", "question": "Why?", "keys": ["Doe2015-valley"], "confirmed": False}
    (query,) = evaluate.realistic_queries([question], {"Doe2015-valley"})
    assert query.answers == frozenset(["Doe2015-valley"])
    with pytest.raises(SystemExit, match="Roe2020"):
        evaluate.realistic_queries([question | {"keys": ["Roe2020"]}], {"Doe2015-valley"})


def test_configuration_rows(collection, monkeypatch):
    monkeypatch.setattr(config, "EVAL_CONFIGURATIONS", (("bm25",), ("qwen3",), ("bm25", "qwen3")))
    queries = [query("Doe2015-valley")]
    rows = evaluate.configuration_rows(collection, queries)
    assert set(rows) == {"bm25", "qwen3", "bm25+qwen3"}
    assert rows["qwen3"] == [[2, 1, 0, 3]]
    # A fusion is the two rankings merged by rrf(), as search() merges them.
    assert rows["bm25+qwen3"][0] == [row for row, _ in search.rrf([rows["bm25"][0], rows["qwen3"][0]])]
    # By paper: Roe2020 first, then Doe2015 at its best chunk.
    assert evaluate.chunk_places(collection, rows["qwen3"], queries, evaluate.paper_of) == [2]


def test_reranked_rows(collection, monkeypatch):
    reranker = ToyReranker()
    monkeypatch.setattr(models, "load_reranker", lambda: reranker)
    monkeypatch.setattr(config, "EVAL_RERANK_DEPTHS", (2, 4))
    queries = [query("Doe2015-valley")]
    reranked = evaluate.reranked_rows(collection, queries, [[0, 1, 2, 3]])
    # Longer passages score higher: row 2 is the longest, then row 1, and rows 0 and 3 tie and keep their order.
    assert reranked[4] == [[2, 1, 0, 3]]
    # At depth 2 only rows 0 and 1 are reordered, from the scores of the same pass.
    assert reranked[2] == [[1, 0, 2, 3]]
    assert reranker.read == 4
    # The second run reads every score from the kept run and asks the reranker nothing.
    evaluate.reranked_rows(collection, queries, [[0, 1, 2, 3]])
    assert reranker.read == 4


def test_number_share(collection):
    queries = [query("Doe2015-valley", ("2015",)), query("Roe2020-lassa", ("7",)), query("Doe2015-valley")]
    collection.chunks[1]["text"] = "Rodents in the village in 2015"
    # The first query's paper holds its number in a top chunk; the second's does not; the third has no number and is not counted.
    assert evaluate.number_share(collection, queries, [[2, 1], [2, 1], [0]]) == 0.5
