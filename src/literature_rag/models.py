"""The open-weight models: how they get onto this machine, and how they turn text into vectors and scores.

Model ids live in config.py. Run `uv run literature-rag-models` once to download them into the Hugging Face cache; everything after that runs offline, loading each model from its local snapshot.

Both embedders turn queries and passages into vectors whose inner product says how well a passage answers a query. Qwen3's vectors have length 1, so their inner product is the cosine; MedCPT's are not normalised, because it was trained to score by the raw inner product.
"""

from functools import cache
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from literature_rag import config

# Config, tokenizer and sentence-transformers files, plus the weights as safetensors. Several repositories also ship the same weights as pytorch_model.bin, which would double the download for nothing.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


def model_ids() -> list[str]:
    """Every model the retriever uses, each listed once."""
    ids = [repo for entry in config.EMBEDDING_MODELS.values() for repo in (entry["query"], entry["passage"])]
    return list(dict.fromkeys([*ids, config.RERANKER_MODEL]))


def cached_path(repo_id: str) -> Path | None:
    """The local snapshot of a model, or None if it has not been downloaded completely."""
    try:
        path = Path(snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS, local_files_only=True))
    except Exception:  # noqa: BLE001 - huggingface_hub raises several error types for a missing snapshot
        return None
    return path if any(path.glob("*.safetensors")) else None


def local(repo_id: str) -> str:
    """The local snapshot of a model, to load it from without asking the Hugging Face Hub anything."""
    path = cached_path(repo_id)
    if path is None:
        raise SystemExit(f"{repo_id} is not downloaded; run literature-rag-models")
    return str(path)


def device() -> str:
    """Apple's GPU when there is one (MPS), the CPU otherwise."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dtype() -> torch.dtype:
    """Half precision on the GPU, where it is about twice as fast and gives the same vectors to four decimals; full precision on the CPU, where half is slow."""
    return torch.float16 if device() == "mps" else torch.float32


class Embedder(Protocol):
    """What the index and search need from an embedding model.

    A passage is the pair (header, text) that chunks.index_parts gives: "Title. Section." and the chunk's text. Each model reads the pair the way it was trained to.
    """

    def encode_queries(self, queries: list[str], instruction: str) -> np.ndarray:
        """One float32 row per query. The instruction names the search task, for models that take one."""
        ...

    def encode_passages(self, passages: list[tuple[str, str]]) -> np.ndarray:
        """One float32 row per passage."""
        ...


class Qwen3Embedder:
    """Qwen3-Embedding-0.6B through sentence-transformers: a general-purpose model that reads a query after an instruction, and a passage as it is.

    Its sentence-transformers configuration pools the vector of the last token, the end-of-text token its tokenizer appends, and normalises it to length 1.
    """

    def __init__(self, entry: dict) -> None:
        self.model = SentenceTransformer(local(entry["passage"]), device=device(), model_kwargs={"dtype": dtype()})
        self.model.max_seq_length = entry["passage_tokens"]
        self.batch_size = config.EMBEDDING_BATCH["qwen3"]

    def encode_queries(self, queries: list[str], instruction: str) -> np.ndarray:
        # The format of the model card: the instruction, then the query after "Query:" with no space.
        prompt = f"Instruct: {instruction}\nQuery:"
        vectors = self.model.encode_query(queries, prompt=prompt, batch_size=self.batch_size)
        return vectors.astype(np.float32)

    def encode_passages(self, passages: list[tuple[str, str]]) -> np.ndarray:
        # The header and the text joined, as chunks.index_text joins them, so Qwen3 reads exactly what BM25 indexes.
        texts = [" ".join(part for part in passage if part) for passage in passages]
        return self.model.encode_document(texts, batch_size=self.batch_size).astype(np.float32)


class MedCPTEmbedder:
    """NCBI's MedCPT, trained on PubMed search logs: a query encoder for short queries and an article encoder for (title, abstract) pairs.

    Each encoder is a BERT model, and the vector is the output for the [CLS] token, the first one, as the model card does it.
    """

    def __init__(self, entry: dict) -> None:
        self.query_tokenizer = AutoTokenizer.from_pretrained(local(entry["query"]))
        self.query_model = AutoModel.from_pretrained(local(entry["query"]), dtype=dtype()).to(device()).eval()
        self.passage_tokenizer = AutoTokenizer.from_pretrained(local(entry["passage"]))
        self.passage_model = AutoModel.from_pretrained(local(entry["passage"]), dtype=dtype()).to(device()).eval()
        self.query_tokens = entry["query_tokens"]
        self.passage_tokens = entry["passage_tokens"]
        self.batch_size = config.EMBEDDING_BATCH["medcpt"]

    def encode(self, tokenizer, model, inputs: list, max_length: int) -> np.ndarray:
        """Run one encoder over its inputs in batches and keep each one's [CLS] vector.

        For a (header, text) pair, the tokenizer's default truncation cuts the longer of the two, which is always the text.
        """
        rows = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                batch = inputs[start : start + self.batch_size]
                encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
                output = model(**encoded.to(device()))
                rows.append(output.last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(rows)

    def encode_queries(self, queries: list[str], instruction: str) -> np.ndarray:
        # MedCPT has no instructions: its query encoder learnt its one task from the search logs.
        return self.encode(self.query_tokenizer, self.query_model, queries, self.query_tokens)

    def encode_passages(self, passages: list[tuple[str, str]]) -> np.ndarray:
        pairs = [list(passage) for passage in passages]
        return self.encode(self.passage_tokenizer, self.passage_model, pairs, self.passage_tokens)


EMBEDDERS = {"qwen3": Qwen3Embedder, "medcpt": MedCPTEmbedder}


@cache
def load_embedder(name: str) -> Embedder:
    """The embedder called name in config.EMBEDDING_MODELS, loaded once per process. Loading takes one or two seconds."""
    return EMBEDDERS[name](config.EMBEDDING_MODELS[name])


@cache
def load_reranker() -> CrossEncoder:
    """The cross-encoder, loaded once per process. Its score for a (query, passage) pair is a probability of relevance, from 0 to 1."""
    return CrossEncoder(
        local(config.RERANKER_MODEL),
        device=device(),
        model_kwargs={"dtype": dtype()},
        max_length=config.RERANKER_MAX_TOKENS,
    )


def main() -> None:
    for repo_id in model_ids():
        path = snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS)
        print(f"{repo_id}: {path}")
