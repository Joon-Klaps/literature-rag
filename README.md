# literature-rag

Hybrid retrieval over a PhD thesis on Lassa virus genomic surveillance (KU Leuven) and the roughly 300 papers it draws on, served to Claude Code as an MCP server. It was built for the jury revisions: finding the source of a claim, checking a number against the paper it came from, and finding where a new paper belongs in the thesis.

It serves other projects too, each with a library of its own: any repository that keeps its papers in `data/manuscripts/` the way the thesis does (see "Projects" below).

Status: under construction. Retrieval, its evaluation and the MCP server are built; the write-up is next.

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

## Projects

A project is a repository with its papers in `data/manuscripts/` (a PDF and its Markdown for each), its bibliography in `allpapers.bib`, and the fetch script `scripts/fetch_pubmed_pdf.py`, with a `.venv` to run it in, which adds papers to both. The thesis (`~/Desktop/School/PhD/phd-thesis`) and the rapidtrees manuscript are two. A project is only ever read from here; papers enter it through its own fetch script.

Every command works on one project: the folder `LITERATURE_RAG_PROJECT` names, or else the project it runs in, the nearest folder at or above the current one that holds `data/manuscripts/`. Outside any project and without the variable, the commands stop and say so. Each project sets the variable for its Claude Code sessions in `.claude/settings.local.json`, which git does not track:

```json
{ "env": { "LITERATURE_RAG_PROJECT": "/Users/joonklaps/Desktop/School/PhD/phd-thesis" } }
```

From a terminal, run the commands inside the project, or name it: `LITERATURE_RAG_PROJECT=~/Desktop/School/PhD/phd-thesis uv run literature-rag-check`.

Everything derived from a project's papers lives in this repository, under `data/<project>/` and `index/<project>/`, named after the project's folder. Each project has its own library, with the citation keys of its own bibliography, so a passage never carries a key that the project cannot cite. Hand-made key corrections are in `key-overrides/<project>.csv` and evaluation numbers in `results/<project>/`. A project with a `thesis.fls`, latexmk's record of a LaTeX build, also has a thesis collection for `search_thesis`; the test questions come from that thesis, so only the thesis is evaluated.

To add a project, give it the layout above, set the variable in its `.claude/settings.local.json`, run `literature-rag-ingest`, `literature-rag-chunk` and `literature-rag-build` inside it, and register the server from it (see "Using it from Claude Code").

## Building and growing the library

```bash
uv run literature-rag-ingest   # structured text for every paper in the project's data/manuscripts
uv run literature-rag-chunk    # passages of the papers and of the thesis, and the test questions
uv run literature-rag-build    # vectors for every passage, from each embedding model
```

The first ingestion downloads Europe PMC's XML and sends the other PDFs to GROBID, which takes about an hour, mostly GROBID looking each reference up in Crossref. Everything it downloads or computes is cached under `data/<project>/raw/`, so a later run only does the work for papers it has not seen and finishes in seconds otherwise. Chunking takes seconds and is redone in full each time. The first build embeds every passage, about 20 minutes on an M1 Max, mostly Qwen3; after that it keeps each vector as long as its passage's text is unchanged, so a new paper costs seconds.

To try a search from the command line, with any of the methods (`bm25`, `qwen3`, `medcpt`) and with or without the reranker:

```bash
uv run literature-rag-search "case fatality among hospitalised Lassa fever patients" --methods bm25,qwen3,medcpt --rerank
uv run literature-rag-search --collection thesis "the multimammate mouse is the main reservoir"
```

The terminal shows the first 30 words of each hit, which is enough to skim. To read them, `--full` prints each hit's whole text (`--words N` prints N words), and `--html` writes a self-contained page to `data/<project>/reports/` with the whole text of every hit, the words of the question marked, and, below the hits, each paper (or thesis chapter) the hits came from once, with all its chunks in reading order and the hits highlighted. Each hit links to its chunk in that paper, to the paper's PDF and Markdown (or the thesis's LaTeX file) and to its DOI. `--open` writes the page and opens it in the browser. Several questions go into one page.

```bash
uv run literature-rag-search "case fatality among hospitalised Lassa fever patients" "Irrua treatment cost" --open
```

To use the commands from any directory, install them once as a uv tool. The install is editable, so it follows the code in this repository, and the lock file keeps its versions identical to the project's:

```bash
uv export --frozen --no-hashes --no-emit-project --no-dev -o /tmp/literature-rag-constraints.txt
uv tool install --editable . -c /tmp/literature-rag-constraints.txt
```

Then a new paper goes into the library with one command, which takes the same arguments as the project's `scripts/fetch_pubmed_pdf.py`, runs that script, ingests what it downloaded, redoes the chunks and embeds the new ones:

```bash
literature-rag-add 10.1038/srep21977
literature-rag-add --mode title "Spatial and temporal evolution of Lassa virus"
```

Papers still enter the library only through the fetch script. One fetched by running the script directly joins at the next `literature-rag-ingest`, `literature-rag-chunk` and `literature-rag-build`. Rerun the two install lines, with `--reinstall` added to the second, after a change to the dependencies or to the commands in `pyproject.toml`.

## Evaluating

```bash
uv run literature-rag-eval            # every configuration, the reranker and the qmd baselines; writes results/<project>/eval.md and eval.json
uv run literature-rag-eval --no-qmd   # the same without qmd
```

The questions are the thesis sentences that cite exactly one paper of the library, each answered by that paper; about ten literature points from the review rounds, each with the papers that answer it; and twenty paragraph summaries from the thesis's `PARAGRAPH-INDEX.md`, searched in the thesis itself. Each configuration (BM25, each embedder, their fusions, and the best fusion reranked) is scored by hit@1, hit@5 and MRR@10 at the level of papers, with bootstrap intervals, against qmd's own keyword search and its full hybrid search over the unmodified Markdown. The first run takes about an hour, nearly all of it `qmd query` and the reranker; their results are kept under `data/<project>/eval/runs/`, so a second run takes about a minute. The questions come from the thesis, so the command runs for the thesis only.

## Using it from Claude Code

The MCP server `thesis-library` is registered once in each project, from the project's folder, so that it is there in every Claude Code session opened in it:

```bash
claude mcp add thesis-library --scope local -- uv run --project ~/Desktop/School/PhD/git_repos/literature-rag literature-rag
```

`uv run --project` runs this repository's environment without leaving the folder Claude Code runs in, so the server serves that project even when the project's `LITERATURE_RAG_PROJECT` does not reach it. The name is the same in every project, so agents and skills that call its tools work in each.

It has three tools, all read-only:

- `search_library(question, k=8, citable_only=True)`: passages from the papers, each with its citation key, title, year, DOI, section, text, the references it cites, a relevance score from 0 to 1, and the path of the paper's Markdown in the project, next to its PDF, to check the passage against.
- `search_thesis(text, k=5)`: paragraphs and captions of the thesis, each with its file, line range, section path and cited keys, and `current`, false when those lines have changed since the thesis was indexed. In a project without a thesis it says so instead.
- `get_context(chunk_id, window=1)`: a hit with the chunks around it, from the same paper or the same chapter.

A passage is a lead to check against the paper, and a thesis paragraph is never evidence for a claim; the tool descriptions say both to the model. The first call takes about ten seconds while the models load, later calls about three. The server notices when `literature-rag-chunk` or `literature-rag-build` has changed its files and loads them again, so a paper added during a session is found without a restart; after a change to the code, reconnect it with `/mcp`. After editing the thesis, run `literature-rag-chunk` and `literature-rag-build` to bring its line numbers up to date.

## What is not in this repository

Paper text, XML, TEI, chunks, embeddings and anything quoting the thesis live in `data/` and `index/`, which are gitignored: the papers are copyrighted and the thesis chapters are unpublished. `results/` holds aggregate numbers only.
