"""Embed every chunk of both collections with each embedding model, and keep the vectors under index/.

Run `uv run literature-rag-build` after `literature-rag-chunk`; `literature-rag-add` runs all three. For each collection (library, thesis) and each model in config.EMBEDDING_MODELS, it keeps two files in index/<collection>/:

- <model>.npy: the vectors, one row per distinct passage text, in float16 to halve the size.
- <model>.json: the model's entry in config.EMBEDDING_MODELS, and for each row the hash of the text it was made from.

A vector is found by the hash of its text, not by its position, so a chunk keeps its vector when chunking renumbers it, and adding a paper embeds only the paper's own chunks. The first build embeds everything: about 15 minutes for Qwen3 over the library and 3 for MedCPT. It goes in rounds and saves after each, so an interrupted build picks up where it stopped.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from literature_rag import chunks, config, models
from literature_rag.records import Chunk, ThesisChunk


class StaleIndex(RuntimeError):
    """Some chunks have no vector: the chunks changed after the last literature-rag-build."""


def text_hash(parts: tuple[str, str]) -> str:
    """A short fingerprint of a passage's header and text, the key under which its vector is kept. Sixteen hexadecimal digits are 64 bits, which no two of some ten thousand texts share by chance."""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def paths(collection: str, model: str) -> tuple[Path, Path]:
    folder = config.INDEX_DIR / collection
    return folder / f"{model}.npy", folder / f"{model}.json"


def read_store(collection: str, model: str) -> tuple[list[str], np.ndarray]:
    """The vectors on disk and the hash of each row, or an empty store when there are none, when the model's entry in config.py has changed since they were made, or when the two files disagree."""
    empty: tuple[list[str], np.ndarray] = ([], np.zeros((0, 0), dtype=np.float32))
    vectors_path, meta_path = paths(collection, model)
    if not (vectors_path.is_file() and meta_path.is_file()):
        return empty
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    # Rows are only ever appended or dropped, so a write cut short between the two files always leaves them with different lengths.
    if meta["model"] != config.EMBEDDING_MODELS[model] or len(meta["hashes"]) != len(vectors):
        return empty
    return meta["hashes"], vectors.astype(np.float32)


def write_store(collection: str, model: str, hashes: list[str], vectors: np.ndarray) -> None:
    """Write the vectors and their hashes. Each file is written in full under a temporary name and then renamed, so an interruption never leaves half a file."""
    vectors_path, meta_path = paths(collection, model)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = vectors_path.with_name(vectors_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, vectors.astype(np.float16))
    temporary.replace(vectors_path)
    meta = {"model": config.EMBEDDING_MODELS[model], "dimensions": vectors.shape[1], "hashes": hashes}
    temporary = meta_path.with_name(meta_path.name + ".tmp")
    temporary.write_text(json.dumps(meta), encoding="utf-8")
    temporary.replace(meta_path)


def embed(collection: str, model: str, chunk_list: list[Chunk] | list[ThesisChunk]) -> tuple[int, int]:
    """Bring a model's vectors for a collection up to date with its chunks, and return how many distinct texts were reused and how many embedded.

    The texts without a vector are embedded longest first, so that a passage too long for the GPU's memory fails in the first minute rather than the last, and each round is saved as it finishes. Vectors whose text no chunk has any more are dropped at the end, so the store does not grow with every change.
    """
    parts = {text_hash(passage): passage for passage in map(chunks.index_parts, chunk_list)}
    hashes, vectors = read_store(collection, model)
    known = set(hashes)
    todo = sorted((h for h in parts if h not in known), key=lambda h: len(parts[h][0]) + len(parts[h][1]), reverse=True)
    if todo:
        embedder = models.load_embedder(model)
        start = time.perf_counter()
        for first in range(0, len(todo), config.EMBEDDING_ROUND):
            batch = todo[first : first + config.EMBEDDING_ROUND]
            new = embedder.encode_passages([parts[h] for h in batch])
            vectors = np.concatenate([vectors, new]) if hashes else new
            hashes = hashes + batch
            write_store(collection, model, hashes, vectors)
            done = first + len(batch)
            print(
                f"  {collection} {model}: {done} of {len(todo)} embedded, {time.perf_counter() - start:.0f} s",
                flush=True,
            )
    keep = [row for row, h in enumerate(hashes) if h in parts]
    if len(keep) < len(hashes):
        write_store(collection, model, [hashes[row] for row in keep], vectors[keep])
    return len(parts) - len(todo), len(todo)


def missing(collection: str, model: str, chunk_list: list[Chunk] | list[ThesisChunk]) -> int:
    """How many chunks have no vector from this model."""
    known = set(read_store(collection, model)[0])
    return sum(text_hash(chunks.index_parts(chunk)) not in known for chunk in chunk_list)


def load_vectors(collection: str, model: str, chunk_list: list[Chunk] | list[ThesisChunk]) -> np.ndarray:
    """A model's vectors for the chunks, one float32 row per chunk in the order given. Raises StaleIndex when a chunk has none, since a search over outdated vectors would quietly return the wrong passages."""
    hashes, vectors = read_store(collection, model)
    row = {h: n for n, h in enumerate(hashes)}
    wanted = [text_hash(chunks.index_parts(chunk)) for chunk in chunk_list]
    absent = sum(h not in row for h in wanted)
    if absent:
        raise StaleIndex(
            f"{absent} of {len(chunk_list)} {collection} chunks have no {model} vector; run literature-rag-build"
        )
    return vectors[[row[h] for h in wanted]]


def build(collections: list[str], model_names: list[str]) -> list[str]:
    """Embed what is new in each collection with each model, and return the report: one line per collection and model."""
    lines = []
    for collection in collections:
        chunk_list = chunks.read_jsonl(config.COLLECTIONS[collection])
        for model in model_names:
            start = time.perf_counter()
            reused, embedded = embed(collection, model, chunk_list)
            lines.append(
                f"{collection:<8} {model:<7} {len(chunk_list):>6} chunks, {reused + embedded:>6} distinct texts:"
                f" {embedded} embedded, {reused} reused, {time.perf_counter() - start:.0f} s"
            )
    return lines


def main() -> None:
    """Entry point for `uv run literature-rag-build`."""
    parser = argparse.ArgumentParser(
        description="Embed the chunks of each collection with each model, reusing every vector whose text has not changed."
    )
    parser.add_argument(
        "--collections", default=",".join(config.COLLECTIONS), help="comma-separated, from: %(default)s"
    )
    parser.add_argument(
        "--models", default=",".join(config.EMBEDDING_MODELS), help="comma-separated, from: %(default)s"
    )
    args = parser.parse_args()
    collections = args.collections.split(",")
    model_names = args.models.split(",")
    unknown = (set(collections) - set(config.COLLECTIONS)) | (set(model_names) - set(config.EMBEDDING_MODELS))
    if unknown:
        parser.error(f"unknown: {', '.join(sorted(unknown))}")
    print("\n".join(build(collections, model_names)))
    print(f"\nWrote {config.INDEX_DIR}")
