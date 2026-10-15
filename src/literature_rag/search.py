"""Search a collection with BM25, with each embedding model, with their rankings fused, and with a reranker over the fused shortlist.

    uv run literature-rag-search "case fatality among hospitalised Lassa fever patients" --methods bm25,qwen3,medcpt --rerank

A Collection holds the chunks of the library, the thesis or the notes, a BM25 index over their text and their vectors, and is loaded once. bm25(), dense(), rrf() and cross_encode() are the steps, and search() puts them together; neighbours() reads the chunks around a hit. The command searches each question it is given and prints the time each step took and the hits, with their key, section and opening words.

To read the hits rather than skim them, --full prints each hit's whole text (--words N, any other length), and --html writes a page with the whole text, the question's words marked and every hit shown inside its paper, chapter or note; --open writes the page and opens it in the browser.

    uv run literature-rag-search "case fatality among hospitalised Lassa fever patients" --open
    uv run literature-rag-search --collection notes "what did we decide about the negative controls"
"""

import argparse
import re
import time
import webbrowser
from collections.abc import Hashable, Sequence
from functools import cache

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from literature_rag import chunks, config, index, models, report
from literature_rag.records import Answer, Chunk, Hit, NoteChunk, ThesisChunk

# A ranking is a list of (row, score) pairs, best first, where row is a chunk's position in Collection.chunks.
Ranking = list[tuple[int, float]]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase search tokens: runs of letters and digits, as in the nf-core server. Chunks and queries must go through exactly this function."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Collection:
    """The chunks of one collection, with a BM25 index over their text and, for each embedding model, a matrix with one row per chunk. Build once, query many times."""

    def __init__(
        self, name: str, chunk_list: list[Chunk] | list[ThesisChunk] | list[NoteChunk], vectors: dict[str, np.ndarray]
    ) -> None:
        self.name = name
        self.chunks = chunk_list
        # A chunk's row by its id, to find the chunks around it.
        self.rows = {chunk["chunk_id"]: row for row, chunk in enumerate(chunk_list)}
        self.vectors = vectors
        self.bm25 = BM25Okapi([tokenize(chunks.index_text(chunk)) for chunk in chunk_list])


@cache
def load_collection(name: str) -> Collection:
    """A collection as the last literature-rag-chunk and literature-rag-build left it, loaded once per process. Raises index.StaleIndex when a chunk has no vector."""
    chunk_list = chunks.read_jsonl(config.COLLECTIONS[name])
    vectors = {model: index.load_vectors(name, model, chunk_list) for model in config.EMBEDDING_MODELS}
    return Collection(name, chunk_list, vectors)


def top(scores: np.ndarray, n: int) -> Ranking:
    """The rows with the n highest scores, as a ranking. Equal scores keep the order of the chunk file, so a ranking is the same on every run."""
    rows = np.argsort(-scores, kind="stable")[:n]
    return [(int(row), float(scores[row])) for row in rows]


def bm25(collection: Collection, query: str, n: int = config.CANDIDATES) -> Ranking:
    """The n chunks with the highest BM25 score for the query's words. Each word counts once, however often the query repeats it, and a chunk that shares no word with the query is left out."""
    tokens = list(dict.fromkeys(tokenize(query)))
    return [(row, score) for row, score in top(collection.bm25.get_scores(tokens), n) if score > 0]


def dense(collection: Collection, model: str, query_vector: np.ndarray, n: int = config.CANDIDATES) -> Ranking:
    """The n chunks whose vectors from this model have the highest inner product with the query's vector from the same model."""
    return top(collection.vectors[model] @ query_vector, n)


def rrf[Item: Hashable](rankings: Sequence[Sequence[Item]], k: int = config.RRF_K) -> list[tuple[Item, float]]:
    """Reciprocal rank fusion: each item scores the sum of 1 / (k + rank) over the rankings it appears in, ranks counted from 1, and the result is sorted by that score.

    Only positions count, so rankings whose scores are on different scales (BM25 against cosine) can be merged. An item ranked well by several methods beats one ranked first by a single method. Equal scores keep the order in which the items were first seen.
    """
    scores: dict[Item, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def fuse(rankings: Sequence[Ranking]) -> Ranking:
    """One method's ranking as it is, or the rankings of several fused with rrf()."""
    if len(rankings) == 1:
        return list(rankings[0])
    return rrf([[row for row, _ in ranking] for ranking in rankings])


def cross_encode(reranker: CrossEncoder, query: str, texts: list[str]) -> np.ndarray:
    """The reranker's score for each (query, text) pair: the model reads the two together, which is what makes it precise and slow."""
    return reranker.predict([(query, text) for text in texts], batch_size=config.RERANKER_BATCH)


def search(
    collection: Collection,
    query: str,
    k: int = 10,
    methods: Sequence[str] = config.DEFAULT_METHODS,
    rerank: bool = config.DEFAULT_RERANK,
    timings: dict[str, float] | None = None,
) -> list[Hit]:
    """The k best chunks of the collection for the query.

    Each method ranks its config.CANDIDATES best chunks; several rankings are fused with rrf(); and with rerank, the cross-encoder rescores the fused top config.RERANK_DEPTH and reorders them. Models load on first use. When timings is given, it receives the seconds each step took, under the method's name and "rerank".
    """
    unknown = [method for method in methods if method != "bm25" and method not in config.EMBEDDING_MODELS]
    if unknown or not methods:
        raise ValueError(f"unknown methods {unknown}; choose from bm25 and {', '.join(config.EMBEDDING_MODELS)}")
    timings = {} if timings is None else timings
    rankings: dict[str, Ranking] = {}
    for method in methods:
        start = time.perf_counter()
        if method == "bm25":
            rankings[method] = bm25(collection, query)
        else:
            embedder = models.load_embedder(method)
            vector = embedder.encode_queries([query], config.QUERY_INSTRUCTIONS[collection.name])[0]
            rankings[method] = dense(collection, method, vector)
        timings[method] = time.perf_counter() - start
    final = fuse(list(rankings.values()))
    if len(rankings) > 1:
        rankings["fused"] = final
    # BM25 alone finds nothing for a query that shares no word with any chunk, and then there is nothing to rerank.
    if rerank and final:
        start = time.perf_counter()
        shortlist = [row for row, _ in final[: config.RERANK_DEPTH]]
        texts = [chunks.index_text(collection.chunks[row]) for row in shortlist]
        scores = cross_encode(models.load_reranker(), query, texts)
        final = sorted(zip(shortlist, scores.tolist()), key=lambda pair: pair[1], reverse=True)
        timings["rerank"] = time.perf_counter() - start
    places = {
        name: {row: place for place, (row, _) in enumerate(ranking, start=1)} for name, ranking in rankings.items()
    }
    return [
        Hit(
            rank=rank,
            score=float(score),
            ranks={name: place[row] for name, place in places.items() if row in place},
            chunk=collection.chunks[row],
        )
        for rank, (row, score) in enumerate(final[:k], start=1)
    ]


def document(chunk: Chunk | ThesisChunk | NoteChunk) -> str:
    """What a chunk is part of: its paper in the library, its note in the notes, and its chapter in the thesis, named by the first part of its section path. A chapter can run over several files, through \\input."""
    if "paper_id" in chunk:
        return chunk["paper_id"]
    if "note" in chunk:
        return chunk["note"]
    return chunk["section"].split(" > ")[0]


def neighbours(collection: Collection, chunk_id: str, window: int) -> list[Chunk] | list[ThesisChunk] | list[NoteChunk]:
    """The chunk and up to window chunks on either side of it, in reading order, from the same paper, note or chapter. Raises KeyError for an unknown id.

    Every chunk file is in reading order, the library paper by paper, the notes note by note and the thesis as LaTeX reads it, so the neighbours are the rows around the chunk's own.
    """
    row = collection.rows[chunk_id]
    around = collection.chunks[max(0, row - window) : row + window + 1]
    return [chunk for chunk in around if document(chunk) == document(collection.chunks[row])]


def query_terms(query: str) -> list[str]:
    """The words of a query worth marking in the text: those that are not filler (config.HIGHLIGHT_STOPWORDS) or a single letter (a single digit is kept, since numbers are what a reader looks for), each once, in the order they appear."""
    return [
        word
        for word in dict.fromkeys(tokenize(query))
        if (len(word) > 1 or word.isdigit()) and word not in config.HIGHLIGHT_STOPWORDS
    ]


def whole_document(collection: Collection, chunk_id: str) -> list[Chunk] | list[ThesisChunk] | list[NoteChunk]:
    """All the chunks of the paper, note or chapter that holds this one, in reading order. Raises KeyError for an unknown id."""
    return neighbours(collection, chunk_id, len(collection.chunks))


def describe(hit: Hit, words: int = 30) -> str:
    """A hit in two lines: rank, score, source, section and the places each method gave it; then the opening words of its text, or all of it when words is 0."""
    chunk = hit["chunk"]
    if "paper_id" in chunk:
        source = f"{chunk['key'] or chunk['paper_id']} ({chunk['year'] or 'n.d.'})"
        if not chunk["citable"]:
            source += ", not citable"
    elif "note" in chunk:
        source = f"{chunk['note']}:L{chunk['line_start']}-{chunk['line_end']}"
    else:
        source = chunk["chunk_id"]
    places = ", ".join(f"{name} {place}" for name, place in hit["ranks"].items())
    text = chunk["text"].split()
    opening = " ".join(text[:words] if words else text) + (" ..." if words and len(text) > words else "")
    return f"{hit['rank']:>3}  {hit['score']:8.3f}  {source} | {chunk['section']} | {places}\n               {opening}"


def main() -> None:
    """Entry point for `uv run literature-rag-search`."""
    config.require_project()
    parser = argparse.ArgumentParser(description="Search the library, the thesis or the notes and print the hits.")
    parser.add_argument("questions", nargs="+", help="one or more questions, each searched on its own")
    parser.add_argument("--collection", choices=list(config.COLLECTIONS), default="library")
    parser.add_argument(
        "--methods",
        default=",".join(config.DEFAULT_METHODS),
        help="comma-separated, from bm25 and the embedding models (default: %(default)s)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=config.DEFAULT_RERANK,
        help=f"rerank the fused top {config.RERANK_DEPTH} with the cross-encoder (default: %(default)s)",
    )
    parser.add_argument("-k", type=int, default=10, help="hits to print per question (default: %(default)s)")
    parser.add_argument(
        "--words", type=int, default=30, help="words of each hit to print, 0 for all of it (default: %(default)s)"
    )
    parser.add_argument("--full", action="store_true", help="print each hit's whole text, like --words 0")
    parser.add_argument(
        "--html",
        action="store_true",
        help=f"also write a page with the whole hits, marked and in their papers, to {config.REPORTS_DIR}",
    )
    parser.add_argument("--open", action="store_true", help="write the page and open it in the browser")
    args = parser.parse_args()
    methods = args.methods.split(",")
    unknown = [method for method in methods if method != "bm25" and method not in config.EMBEDDING_MODELS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")

    # Everything is loaded before the first question, so that the times printed per question are those of a warm server.
    start = time.perf_counter()
    try:
        collection = load_collection(args.collection)
    except index.StaleIndex as error:
        raise SystemExit(str(error)) from None
    loading = [f"{len(collection.chunks)} {args.collection} chunks {time.perf_counter() - start:.1f} s"]
    for name in [method for method in methods if method != "bm25"] + (["reranker"] if args.rerank else []):
        start = time.perf_counter()
        if name == "reranker":
            models.load_reranker().predict([("warm", "up")])
        else:
            models.load_embedder(name).encode_queries(["warm up"], config.QUERY_INSTRUCTIONS[args.collection])
        loading.append(f"{name} {time.perf_counter() - start:.1f} s")
    print(f"Loaded: {', '.join(loading)}")

    answers = []
    for question in args.questions:
        timings: dict[str, float] = {}
        hits = search(collection, question, args.k, methods, args.rerank, timings)
        answers.append(Answer(question=question, timings=timings, terms=query_terms(question), hits=hits))
        print(f"\n{question}")
        print("  " + ", ".join(f"{name} {seconds:.2f} s" for name, seconds in timings.items()))
        for hit in hits:
            print(describe(hit, 0 if args.full else args.words))

    if args.html or args.open:
        # Each paper or chapter once, in the order the hits first reach it.
        documents: dict[str, list[Chunk] | list[ThesisChunk] | list[NoteChunk]] = {}
        for answer in answers:
            for hit in answer["hits"]:
                documents.setdefault(document(hit["chunk"]), whole_document(collection, hit["chunk"]["chunk_id"]))
        path = report.write(report.render(args.collection, answers, documents))
        print(f"\nReport: {path}")
        if args.open:
            webbrowser.open(path.as_uri())
