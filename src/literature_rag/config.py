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
CHUNKS_DIR = DATA_DIR / "chunks"
LIBRARY_CHUNKS_FILE = CHUNKS_DIR / "library.jsonl"
THESIS_CHUNKS_FILE = CHUNKS_DIR / "thesis.jsonl"
# Test questions drawn from the thesis. They quote unpublished chapters, so they stay under data/ too.
EVAL_DIR = DATA_DIR / "eval"
THESIS_PAIRS_FILE = EVAL_DIR / "thesis_pairs.jsonl"
# Literature points from the review rounds, each with the papers that answer it: drafted by Claude, confirmed by Joon.
REALISTIC_FILE = EVAL_DIR / "realistic.jsonl"
# The rankings that are slow to make (the reranker's scores, qmd's results), kept so that a second run takes seconds. Delete the folder to make them again.
RUNS_DIR = EVAL_DIR / "runs"
# The paragraph summaries that the thesis-index skill wrote, the questions of the thesis check.
PARAGRAPH_INDEX_FILE = THESIS_REPO / "PARAGRAPH-INDEX.md"
# Embedding matrices, one per collection and model, with the hash of the text behind each row. Never committed.
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

# Paragraphs of one section are merged into a chunk until the next would take it past CHUNK_WORDS. About 300 words is 400 tokens, which with the title and section in front still fits MedCPT's 512.
CHUNK_WORDS = 300
# A paragraph, abstract or caption longer than this is cut at sentence ends into pieces of about equal length, each under CHUNK_WORDS. Between the two limits a paragraph stays whole.
SPLIT_WORDS = 400

# Chapter files the thesis collection leaves out: the CV, the publication list, and the Dutch summary, which repeats the English one.
THESIS_EXCLUDED = {
    "chapters/curriculum/curriculum.tex",
    "chapters/publications/publications.tex",
    "chapters/01_abstract_nl/01_abstract_nl.tex",
}

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

# Semantic search runs every embedder below next to BM25. MedCPT is asymmetric: one encoder for queries, another for passages. The token limits are where each model cuts its input: Qwen3 reads up to 1024 tokens, more than any chunk needs; MedCPT was trained on queries of up to 64 tokens and articles of up to 512, which cuts the end off about 4% of the library's chunks. The index keeps a model's vectors only while its entry here is unchanged.
EMBEDDING_MODELS = {
    "qwen3": {
        "query": "Qwen/Qwen3-Embedding-0.6B",
        "passage": "Qwen/Qwen3-Embedding-0.6B",
        "query_tokens": 1024,
        "passage_tokens": 1024,
    },
    "medcpt": {
        "query": "ncbi/MedCPT-Query-Encoder",
        "passage": "ncbi/MedCPT-Article-Encoder",
        "query_tokens": 64,
        "passage_tokens": 512,
    },
}
# Qwen3-Embedding reads each query after an instruction that names the task; passages carry none. Searching the thesis is a different task from finding a paper's support for a claim, so each collection has its own. MedCPT takes no instruction.
QUERY_INSTRUCTIONS = {
    "library": "Given a claim from a PhD thesis, retrieve the passage from a scientific paper that supports it",
    "thesis": "Given a claim or a passage from a scientific paper, retrieve the paragraph of a PhD thesis that discusses it",
}
# Passages per forward pass. On the M1 Max, Qwen3 is fastest at 8 (about 11 chunks a second, so 15 minutes for the library) and MedCPT at 16 (about 55 a second); larger batches carry more padding.
EMBEDDING_BATCH = {"qwen3": 8, "medcpt": 16}
# The index embeds in rounds of this many passages and saves after each, so an interrupted build loses one round, not the whole run.
EMBEDDING_ROUND = 512

# A cross-encoder that rereads each (question, passage) pair of the fused shortlist and reorders it. With its header, no chunk passes about 600 tokens, so 1024 cuts nothing; 50 pairs take about 2.6 seconds either way.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_TOKENS = 1024
RERANKER_BATCH = 16

# The two things search can look in, and the chunk file each is read from. Their vectors live in INDEX_DIR/<collection>/.
COLLECTIONS = {"library": LIBRARY_CHUNKS_FILE, "thesis": THESIS_CHUNKS_FILE}
# Each method hands this many candidates to fusion, ranked best first.
CANDIDATES = 100
# Reciprocal rank fusion scores a passage as the sum of 1 / (RRF_K + rank) over the rankings it appears in. 60 is the constant of the original paper (Cormack and others, 2009), which damps the difference between the first few ranks.
RRF_K = 60
# The reranker reads only the top of the fused ranking: it is precise, but too slow for more. In block 5, 50 did better than 20 and as well as 100, at half the time of 100.
RERANK_DEPTH = 50
# What search runs when it is not told otherwise, as block 5 chose it (results/eval.md): BM25 and Qwen3 fused, with the fused top RERANK_DEPTH reranked. MedCPT made the fusion worse (MRR@10 0.62 with it, 0.64 without), and the reranker lifted it to 0.67 and the share of sentences whose numbers the top passages hold from 0.45 to 0.58, for about 2.5 seconds a question.
DEFAULT_METHODS = ("bm25", "qwen3")
DEFAULT_RERANK = True

# The MCP server's limits on what one call returns. A passage with its references is about a thousand tokens, so twenty is about as much as a call should put in the model's context; a window of five reads eleven chunks, about a section.
SERVER_MAX_HITS = 20
SERVER_MAX_WINDOW = 5

# The configurations the evaluation compares, each a list of methods whose rankings are fused. The best fused one is also run with the reranker.
EVAL_CONFIGURATIONS = (
    ("bm25",),
    ("qwen3",),
    ("medcpt",),
    ("bm25", "qwen3"),
    ("bm25", "medcpt"),
    ("bm25", "qwen3", "medcpt"),
)
# How deep the reranker reads the fused ranking. One pass scores the deepest shortlist, and the shallower ones reuse its scores, since the reranker scores each passage on its own.
EVAL_RERANK_DEPTHS = (20, 50, 100)
# A query scores hit@k for each k here (the right paper among the first k) and the reciprocal rank of the right paper within the first EVAL_MRR_DEPTH, 0 below it.
EVAL_HIT_AT = (1, 5)
EVAL_MRR_DEPTH = 10
# Bootstrap intervals: resample the queries with replacement this many times, with a fixed seed so that the intervals are the same on every run.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 2026
# The number check looks for a pair's numbers in this many of the top chunks, among those from the right paper.
NUMBER_CHECK_DEPTH = 5
# The thesis check: this many paragraph summaries from PARAGRAPH-INDEX.md, drawn with the bootstrap seed among those whose cited keys pick out exactly one current paragraph.
THESIS_CHECK_SIZE = 20
# qmd, the finished system to beat: its BM25 search and its full hybrid query over the unmodified Markdown, each asked for this many documents.
QMD_COLLECTION = "papers"
QMD_RESULTS = 10
