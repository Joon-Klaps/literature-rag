# literature-rag

Hybrid retrieval over a PhD thesis on Lassa virus genomic surveillance (KU Leuven) and the roughly 300 papers it draws on, served to Claude Code as an MCP server. It was built for the jury revisions: finding the source of a claim, checking a number against the paper it came from, and finding where a new paper belongs in the thesis.

Status: under construction, following [PLAN.md](PLAN.md).

## How it works

1. Structured text. Publisher JATS XML from Europe PMC where the paper is open access, GROBID TEI from the PDF otherwise. Both keep sections, captions and the reference list, and link each in-text citation to its reference.
2. Chunks. Paragraphs within one section, each carrying its citation key, its section and the references it cites. The thesis is indexed separately, as LaTeX paragraphs with file and line range.
3. Retrieval. Lexical search (BM25) and semantic search (two open-weight embedding models, Qwen3-Embedding-0.6B and NCBI's MedCPT), fused with reciprocal rank fusion and reordered by a cross-encoder.
4. Evaluation. About 160 questions drawn from the thesis itself, each answered by the paper it cites, with an off-the-shelf tool (qmd) as the baseline to beat.
5. MCP server. `search_library`, `search_thesis` and `get_context`, for Claude Code.

## Setup

```bash
uv sync
uv run literature-rag-models   # downloads the models into the Hugging Face cache, about 4.4 GB, once
docker run -d --platform linux/arm64 --ulimit core=0 --name grobid -p 8070:8070 grobid/grobid:0.9.1-crf
uv run literature-rag-check    # reports what is in place and what is missing
```

GROBID's image starts through `tini`, which fails under amd64 emulation, so the platform is pinned: on a machine with `DOCKER_DEFAULT_PLATFORM=linux/amd64` set, Docker would otherwise pull the emulated image. Do not add `--init` either; the image brings its own.

The thesis repository is read from `~/Desktop/School/PhD/phd-thesis`; set `THESIS_REPO` to point elsewhere. It is only ever read.

## Building and growing the library

```bash
uv run literature-rag-ingest   # structured text for every paper in the thesis repository's data/manuscripts
uv run literature-rag-chunk    # passages of the papers and of the thesis, and the test questions
uv run literature-rag-build    # vectors for every passage, from each embedding model
```

The first ingestion downloads Europe PMC's XML and sends the other PDFs to GROBID, which takes about an hour, mostly GROBID looking each reference up in Crossref. Everything it downloads or computes is cached under `data/raw/`, so a later run only does the work for papers it has not seen and finishes in seconds otherwise. Chunking takes seconds and is redone in full each time. The first build embeds every passage, about 20 minutes on an M1 Max, mostly Qwen3; after that it keeps each vector as long as its passage's text is unchanged, so a new paper costs seconds.

To try a search from the command line, with any of the methods (`bm25`, `qwen3`, `medcpt`) and with or without the reranker:

```bash
uv run literature-rag-search "case fatality among hospitalised Lassa fever patients" --methods bm25,qwen3,medcpt --rerank
uv run literature-rag-search --collection thesis "the multimammate mouse is the main reservoir"
```

To use the commands from any directory, install them once as a uv tool. The install is editable, so it follows the code in this repository, and the lock file keeps its versions identical to the project's:

```bash
uv export --frozen --no-hashes --no-emit-project --no-dev -o /tmp/literature-rag-constraints.txt
uv tool install --editable . -c /tmp/literature-rag-constraints.txt
```

Then a new paper goes into the library with one command, which takes the same arguments as the thesis repository's `scripts/fetch_pubmed_pdf.py`, runs that script, ingests what it downloaded, redoes the chunks and embeds the new ones:

```bash
literature-rag-add 10.1038/srep21977
literature-rag-add --mode title "Spatial and temporal evolution of Lassa virus"
```

Papers still enter the library only through the fetch script. One fetched by running the script directly joins at the next `literature-rag-ingest`, `literature-rag-chunk` and `literature-rag-build`. Rerun the two install lines, with `--reinstall` added to the second, after a change to the dependencies or to the commands in `pyproject.toml`.

## What is not in this repository

Paper text, XML, TEI, chunks, embeddings and anything quoting the thesis live in `data/` and `index/`, which are gitignored: the papers are copyrighted and the thesis chapters are unpublished. `results/` holds aggregate numbers only.
