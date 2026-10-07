# CLAUDE.md — literature-rag

Hybrid retrieval over Joon's PhD thesis and the papers it cites, served to Claude Code over MCP. The background and the design decisions are in `HANDOFF-literature-rag.md` in the thesis repository (`~/Desktop/School/PhD/phd-thesis`). The build follows `PLAN.md` block by block: read it first, and log progress at its end. If `HANDOFF.md` exists, read it next: it holds the state the last session left behind.

## Rules

- The thesis repository is read-only from here. Papers enter it only through its `scripts/fetch_pubmed_pdf.py`.
- Nothing derived from the papers or the thesis goes into git. `data/` and `index/` are gitignored, tests use synthetic fixtures, and `results/` holds numbers only.
- Joon follows the code to learn from it, so keep it modular: one stage per module, settings in `config.py`, record shapes in `records.py`, pure functions for parsing, ranking and scoring, and a `main()` per command. Comments are full sentences on one line, in UK spelling.
- Commit or push only when Joon asks.

## Running things

- `uv run literature-rag-check` shows what is in place: thesis repository, GROBID, Europe PMC, models, disk, qmd.
- GROBID runs in Docker under Colima as the container `grobid`; after a restart, `docker start grobid`.
- `uv run literature-rag-ingest` brings `data/papers/` up to date from the thesis repository's manuscripts; downloads and GROBID output are cached in `data/raw/`, so only new papers cost time.
- `literature-rag-add <DOI | title | PDF>` (installed with `uv tool install --editable .`, see the README) runs the thesis repository's fetch script with those arguments, then ingests.
- `uv run literature-rag-eval` scores every search configuration and qmd on the test questions and writes `results/`; the reranker's scores and qmd's results are kept in `data/eval/runs/`, so only the first run is slow.
- `uv run literature-rag` is the MCP server `thesis-library`, which Claude Code starts over stdio; it is registered in the thesis repository's local config (see the README). After a change to `server.py` or what it imports, reconnect it with `/mcp`; new chunks and vectors it loads by itself.
- `uv run pytest` runs the tests.
