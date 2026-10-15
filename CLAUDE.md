# CLAUDE.md — literature-rag

Hybrid retrieval over Joon's PhD thesis, the papers it cites and Joon's Obsidian notes (the thesis's `data/notes`), served to Claude Code over MCP, and over the papers of Joon's other projects (the rapidtrees manuscript), one library per project. The background and the design decisions are in `HANDOFF-literature-rag.md` in the thesis repository (`~/Desktop/School/PhD/phd-thesis`). The build follows `PLAN.md` block by block: read it first, and log progress at its end. If `HANDOFF.md` exists, read it next: it holds the state the last session left behind.

## Rules

- Projects (the thesis repository, the rapidtrees manuscript) are read-only from here, and so are the notes. Papers enter a project only through its own `scripts/fetch_pubmed_pdf.py`.
- Nothing derived from the papers, the thesis or the notes goes into git. `data/` and `index/` are gitignored, tests use synthetic fixtures, and `results/` holds numbers only.
- Joon follows the code to learn from it, so keep it modular: one stage per module, settings in `config.py`, record shapes in `records.py`, pure functions for parsing, ranking and scoring, and a `main()` per command. Comments are full sentences on one line, in UK spelling.
- Commit or push only when Joon asks.

## Running things

- Every command works on one project: the folder `LITERATURE_RAG_PROJECT` names, or else the project it runs in. From this repository, name it: `LITERATURE_RAG_PROJECT=~/Desktop/School/PhD/phd-thesis uv run literature-rag-check`. Each project sets the variable in its `.claude/settings.local.json`, and its derived files are under `data/<project>/`, `index/<project>/` and `results/<project>/` (see the README, "Projects").
- `uv run literature-rag-check` shows what is in place: the project, GROBID, Europe PMC, models, disk, qmd.
- GROBID runs in Docker under Colima as the container `grobid`; after a restart, `docker start grobid`.
- `uv run literature-rag-ingest` brings `data/<project>/papers/` up to date from the project's manuscripts; downloads and GROBID output are cached in `data/<project>/raw/`, so only new papers cost time.
- `literature-rag-add <DOI | title | PDF>` (installed with `uv tool install --editable .`, see the README) runs the project's fetch script with those arguments, then ingests.
- `uv run literature-rag-eval` scores every search configuration and qmd on the test questions and writes `results/phd-thesis/`; the reranker's scores and qmd's results are kept in `data/phd-thesis/eval/runs/`, so only the first run is slow. It needs a thesis, so it runs for the thesis only.
- `uv run literature-rag` is the MCP server `thesis-library`, which Claude Code starts over stdio; it is registered in each project's local config with `uv run --project` (see the README), so it serves the project Claude Code runs in. After a change to `server.py` or what it imports, reconnect it with `/mcp`; new chunks and vectors it loads by itself.
- `uv run pytest` runs the tests.

6*35 =