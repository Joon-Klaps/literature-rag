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

## What is not in this repository

Paper text, XML, TEI, chunks, embeddings and anything quoting the thesis live in `data/` and `index/`, which are gitignored: the papers are copyrighted and the thesis chapters are unpublished. `results/` holds aggregate numbers only.
