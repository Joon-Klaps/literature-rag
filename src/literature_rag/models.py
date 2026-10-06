"""The open-weight models, and how they get onto this machine.

Model ids live in config.py. Run `uv run literature-rag-models` once to download them into the Hugging Face cache; everything after that runs offline.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

from literature_rag.config import EMBEDDING_MODELS, RERANKER_MODEL

# Config, tokenizer and sentence-transformers files, plus the weights as safetensors. Several repositories also ship the same weights as pytorch_model.bin, which would double the download for nothing.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "*.safetensors"]


def model_ids() -> list[str]:
    """Every model the retriever uses, each listed once."""
    ids = [repo for roles in EMBEDDING_MODELS.values() for repo in roles.values()] + [RERANKER_MODEL]
    return list(dict.fromkeys(ids))


def cached_path(repo_id: str) -> Path | None:
    """The local snapshot of a model, or None if it has not been downloaded completely."""
    try:
        path = Path(snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS, local_files_only=True))
    except Exception:  # noqa: BLE001 - huggingface_hub raises several error types for a missing snapshot
        return None
    return path if any(path.glob("*.safetensors")) else None


def main() -> None:
    for repo_id in model_ids():
        path = snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS)
        print(f"{repo_id}: {path}")
