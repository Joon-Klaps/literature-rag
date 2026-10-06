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
# BibTeX's output for the last build: one \bibitem per key the thesis cites. Where two bibliography entries describe the same paper, the cited key wins.
THESIS_BBL = THESIS_REPO / "thesis.bbl"
# latexmk's record of every file the last build read. It lists the chapter files that are really part of the thesis, whatever macro included them.
THESIS_FLS = THESIS_REPO / "thesis.fls"
# The only door through which a paper enters the library, run with the thesis repository's own environment.
FETCH_SCRIPT = THESIS_REPO / "scripts" / "fetch_pubmed_pdf.py"
THESIS_PYTHON = THESIS_REPO / ".venv" / "bin" / "python"

# Raw XML and TEI, parsed papers and chunks. Built from copyrighted papers and unpublished chapters, so never committed.
DATA_DIR = HOME / "data"
MANIFEST_FILE = DATA_DIR / "manifest.json"
JATS_DIR = DATA_DIR / "raw" / "jats"
TEI_DIR = DATA_DIR / "raw" / "tei"
PAPERS_DIR = DATA_DIR / "papers"
# Embedding matrices and other index files. Never committed.
INDEX_DIR = HOME / "index"
# Aggregate evaluation numbers only, so these are committed.
RESULTS_DIR = HOME / "results"

# Hand-made corrections for papers the bibliography match gets wrong or misses: one row per Markdown file, with its citation key. Committed, since it holds file names and keys only.
KEY_OVERRIDES_FILE = HOME / "key-overrides.csv"
# A title matches a bibliography entry when the two normalised titles are at least this similar (difflib's ratio, 0 to 1).
TITLE_SIMILARITY = 0.92
# Files in data/manuscripts that are not citable sources: a style book, a supplement and a department thesis. They are ingested so that a search can still find them, but they carry citable=false.
NOT_CITABLE = {
    "writ_scien_engin.md",
    "41592_2012_BFnmeth1923_MOESM326_ESM.md",
    "Potter 2025 - Phylodynamics and genomic epidemiology of viral pathogens.md",
}

# Sections that are not part of a paper's argument, by heading. Each pattern must match the whole heading after it is lowercased and stripped of its number and final punctuation. A heading such as "Ethical and regulatory considerations" is content in a paper about data sharing, so the patterns stay narrow.
BOILERPLATE_HEADINGS = (
    r"acknowledge?ments?",
    r"(sources? of )?funding( information| statement| sources?)?",
    r"financial (support|disclosure)",
    r"(conflicts? of interests?|competing interests?)( statement)?",
    r"conflict of interest and funding",
    r"declarations?( of (competing )?interests?)?",
    r"disclosure( statement)?",
    r"author(s'|s)? contributions?( statement)?",
    r"credit authorship contribution statement",
    r"contributors|contributor information|author information|authorship|about the authors?",
    r"(data|code|software|materials|resource) availability( statement)?",
    r"availability of (data|supporting) .*",
    r"data (and code |and materials )?(availability|access|accessibility)( statement)?",
    r"underlying data|source data|associated data|extended data",
    r"(gene )?accession numbers?",
    r"(other )?(supplementary|supplemental|additional) (material|materials|information|data|files?|data files)\b.*",
    r"(foot)?notes?|editors? note|publisher's note",
    r"peer review( information)?",
    r"(key |uncited )?references( and notes)?|bibliography|literature cited",
    r"abbreviations( used)?|glossary",
    r"ethics( statement| approval| declarations?)?|ethical (approval|statement|clearance|permission)|ethic statement",
    r"ethics approval and consent to participate|informed consent( statement)?|consent( for publication| to (publish|participate))?",
    r"inclusion and ethics|institutional review board.*|role of the funder|orcid( ids?)?",
)

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
# Europe PMC is shared infrastructure, so only a few downloads run at once.
EUROPE_PMC_CONCURRENCY = 4

GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
# The container has 6 CPUs; four documents at a time keeps it busy without queueing requests until they time out.
GROBID_CONCURRENCY = 4
# GROBID is built for articles. Longer documents, such as the textbooks, the theses and the national guidelines in the library, take many minutes or time out, so they keep their Markdown text instead. The page count comes from the Markdown's "## Page N" headings.
GROBID_MAX_PAGES = 100
# With the Crossref lookups, a long review or book chapter can take twenty minutes, most of it waiting on Crossref's rate limit; a request that runs past this is counted as a failure for that paper.
GROBID_TIMEOUT = 1800
# consolidateHeader looks the paper's own title up in Crossref, for its DOI. includeRawCitations keeps each reference's text as printed, which is what a passage shows for the references it cites. consolidateCitations=2 looks every reference up in Crossref and adds the DOI it finds, keeping the rest of what GROBID read. On ten papers in block 2 it raised the references with a DOI from 18% to 86%, at about 27 seconds per paper instead of 2.
GROBID_PARAMS = {
    "consolidateHeader": "1",
    "consolidateCitations": "2",
    "includeRawCitations": "1",
}

# Semantic search runs every embedder below next to BM25. MedCPT is asymmetric: one encoder for queries, another for passages.
EMBEDDING_MODELS = {
    "qwen3": {"query": "Qwen/Qwen3-Embedding-0.6B", "passage": "Qwen/Qwen3-Embedding-0.6B"},
    "medcpt": {"query": "ncbi/MedCPT-Query-Encoder", "passage": "ncbi/MedCPT-Article-Encoder"},
}
# A cross-encoder that rereads each (question, passage) pair of the fused shortlist and reorders it.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
