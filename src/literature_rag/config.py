"""Where everything lives, and which services and models the retriever uses.

Claude Code starts the MCP server from an arbitrary working directory, so paths are anchored to the repository, not to the current directory. Set LITERATURE_RAG_HOME to put the derived files elsewhere, and THESIS_REPO to point at another checkout of the thesis.
"""

import os
from pathlib import Path

HOME = Path(os.environ.get("LITERATURE_RAG_HOME", Path(__file__).resolve().parents[2]))

# The thesis repository is the corpus of record. Papers, bibliography and chapters are read from it and never written to.
THESIS_REPO = Path(os.environ.get("THESIS_REPO", Path.home() / "Desktop/School/PhD/phd-thesis")).expanduser()
MANUSCRIPTS_DIR = THESIS_REPO / "data" / "manuscripts"
BIB_FILE = THESIS_REPO / "allpapers.bib"
# latexmk's record of every file the last build read. It lists the chapter files that are really part of the thesis, whatever macro included them.
THESIS_FLS = THESIS_REPO / "thesis.fls"

# Raw XML and TEI, parsed papers and chunks. Built from copyrighted papers and unpublished chapters, so never committed.
DATA_DIR = HOME / "data"
# Embedding matrices and other index files. Never committed.
INDEX_DIR = HOME / "index"
# Aggregate evaluation numbers only, so these are committed.
RESULTS_DIR = HOME / "results"

GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Semantic search runs every embedder below next to BM25. MedCPT is asymmetric: one encoder for queries, another for passages.
EMBEDDING_MODELS = {
    "qwen3": {"query": "Qwen/Qwen3-Embedding-0.6B", "passage": "Qwen/Qwen3-Embedding-0.6B"},
    "medcpt": {"query": "ncbi/MedCPT-Query-Encoder", "passage": "ncbi/MedCPT-Article-Encoder"},
}
# A cross-encoder that rereads each (question, passage) pair of the fused shortlist and reorders it.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
