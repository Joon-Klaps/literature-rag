"""The MCP tools on a made-up library of four papers and a made-up thesis chapter, called through the server as Claude Code calls them, with stand-in models."""

import asyncio
import functools

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from literature_rag import config, index, models, search, server, thesis

THESIS = r"""\chapter{Valley}

Rodents carry the virus \citep{Doe2015-valley}.

\section{Rain}

Rainfall drives the season.

\chapter{Plains}

Nothing happens on the plains."""


def chunk(chunk_id: str, text: str, citable: bool = True) -> dict:
    paper_id = chunk_id.split("#")[0]
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "key": paper_id if citable else None,
        "citable": citable,
        "title": f"Title of {paper_id}",
        "year": 2015,
        "doi": None,
        "source_file": f"{paper_id}.md",
        "section": "Results",
        "kind": "body",
        "text": text,
        "cites": [{"text": "Roe R. Rodents. 2010.", "doi": "10.1000/r1", "pmid": None, "pmcid": None}],
    }


LIBRARY = [
    chunk("Doe2015-valley#0", "Mastomys natalensis was trapped in every village."),
    chunk("Doe2015-valley#1", "Rainfall peaked in August, and so did trapping."),
    chunk("Doe2015-valley#2", "Trapping followed the rains."),
    chunk("Idsr2019#0", "Rainfall and the reporting of cases.", citable=False),
    chunk("Roe2010-rodents#0", "Rodents of the plains."),
]


class ToyEmbedder:
    """Every chunk is as near to every query as any other, so the vectors only tie and BM25 and the reranker decide."""

    def encode_queries(self, queries, instruction):
        return np.ones((len(queries), 1), dtype=np.float32)


class ToyReranker:
    """Scores a passage 0.9 when it mentions rainfall and 0.1 otherwise."""

    def predict(self, pairs, batch_size=None):
        return np.array([0.9 if "Rainfall" in text else 0.1 for _, text in pairs])


@pytest.fixture
def loads(monkeypatch, tmp_path):
    """Point the server at a made-up thesis repository and collections, and return the list of collections it loads, in order."""
    monkeypatch.setattr(models, "load_embedder", lambda name: ToyEmbedder())
    monkeypatch.setattr(models, "load_reranker", lambda: ToyReranker())
    monkeypatch.setattr(config, "THESIS_REPO", tmp_path)
    monkeypatch.setattr(config, "MANUSCRIPTS_DIR", tmp_path / "data" / "manuscripts")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "COLLECTIONS", {name: tmp_path / f"{name}.jsonl" for name in ("library", "thesis")})
    (tmp_path / "chapters").mkdir()
    (tmp_path / "chapters" / "valley.tex").write_text(THESIS, encoding="utf-8")
    thesis_chunks, _ = thesis.parse_thesis({"chapters/valley.tex": THESIS})
    collections = {
        "library": search.Collection("library", LIBRARY, {"qwen3": np.ones((len(LIBRARY), 1), dtype=np.float32)}),
        "thesis": search.Collection(
            "thesis", thesis_chunks, {"qwen3": np.ones((len(thesis_chunks), 1), dtype=np.float32)}
        ),
    }
    loaded = []

    @functools.cache
    def load_collection(name):
        loaded.append(name)
        return collections[name]

    monkeypatch.setattr(search, "load_collection", load_collection)
    monkeypatch.setattr(server, "loaded", {})
    return loaded


def call(tool: str, **arguments) -> list[dict]:
    result = asyncio.run(server.mcp.call_tool(tool, arguments))
    return result.structured_content["result"]


def test_tools_are_read_only_with_bounded_sizes():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    assert set(tools) == {"search_library", "search_thesis", "get_context"}
    assert all(tool.annotations.read_only_hint for tool in tools.values())
    assert tools["search_library"].input_schema["properties"]["k"]["maximum"] == config.SERVER_MAX_HITS
    assert tools["get_context"].input_schema["properties"]["window"]["maximum"] == config.SERVER_MAX_WINDOW


def test_search_library_leaves_out_what_is_not_citable(loads):
    hits = call("search_library", question="rainfall", k=2)
    # The reranker puts both passages on rainfall first; the guidelines are not citable, so the next passage takes their place.
    assert [hit["chunk_id"] for hit in hits] == ["Doe2015-valley#1", "Doe2015-valley#0"]
    first = hits[0]
    assert first["key"] == "Doe2015-valley" and first["score"] == 0.9
    assert first["manuscript"] == "data/manuscripts/Doe2015-valley.md"
    assert first["cites"][0]["doi"] == "10.1000/r1"
    everything = call("search_library", question="rainfall", k=2, citable_only=False)
    assert [hit["chunk_id"] for hit in everything] == ["Doe2015-valley#1", "Idsr2019#0"]
    assert everything[1]["citable"] is False and everything[1]["key"] is None


def test_k_outside_its_bounds_is_refused(loads):
    with pytest.raises(ToolError, match="less than or equal to 20"):
        call("search_library", question="rainfall", k=21)
    with pytest.raises(ToolError, match="greater than or equal to 1"):
        call("search_thesis", text="rainfall", k=0)


def test_search_thesis_says_when_the_lines_have_moved(loads, tmp_path):
    (hit,) = call("search_thesis", text="rainfall drives the season", k=1)
    assert hit["chunk_id"] == "valley:L7-7" and hit["file"] == "chapters/valley.tex"
    assert hit["section"] == "Valley > Rain" and hit["current"] is True
    # A line added at the top moves every paragraph down, so no line range holds any more.
    (tmp_path / "chapters" / "valley.tex").write_text("% a new first line\n" + THESIS, encoding="utf-8")
    hits = call("search_thesis", text="rainfall drives the season", k=3)
    assert [hit["current"] for hit in hits] == [False, False, False]


def test_get_context_stays_within_the_paper_and_the_chapter(loads):
    assert [c["chunk_id"] for c in call("get_context", chunk_id="Doe2015-valley#1")] == [
        "Doe2015-valley#0",
        "Doe2015-valley#1",
        "Doe2015-valley#2",
    ]
    assert [c["chunk_id"] for c in call("get_context", chunk_id="Idsr2019#0", window=5)] == ["Idsr2019#0"]
    assert all("score" not in c for c in call("get_context", chunk_id="Doe2015-valley#1"))
    paragraphs = call("get_context", chunk_id="valley:L7-7", window=5)
    assert [p["chunk_id"] for p in paragraphs] == ["valley:L3-3", "valley:L7-7"]
    assert paragraphs[0]["keys"] == ["Doe2015-valley"] and paragraphs[0]["current"] is True


def test_get_context_with_an_unknown_id(loads):
    with pytest.raises(ToolError, match="No chunk 'Doe2015-valley#9'"):
        call("get_context", chunk_id="Doe2015-valley#9")


def test_a_collection_is_loaded_again_when_its_files_change(loads):
    call("search_library", question="rainfall")
    call("search_library", question="rodents")
    assert loads == ["library"]
    # literature-rag-chunk rewrites the chunk file, as when a paper is added while the server runs.
    config.COLLECTIONS["library"].write_text("", encoding="utf-8")
    call("search_library", question="rainfall")
    assert loads == ["library", "library"]


def test_a_stale_index_is_reported_to_the_model(loads, monkeypatch):
    def stale(name):
        raise index.StaleIndex("3 of 5 library chunks have no qwen3 vector; run literature-rag-build")

    stale.cache_clear = lambda: None
    monkeypatch.setattr(search, "load_collection", stale)
    with pytest.raises(ToolError, match="run literature-rag-build"):
        call("search_library", question="rainfall")
