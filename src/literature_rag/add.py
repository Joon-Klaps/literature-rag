"""Add papers to a project's library from the command line: download them with the project's fetch script, then ingest, chunk and embed them.

    literature-rag-add 10.1038/srep21977
    literature-rag-add --mode title "Spatial and temporal evolution of Lassa virus"
    literature-rag-add path/to/paper.pdf

The arguments go to scripts/fetch_pubmed_pdf.py, with file paths made absolute, so this accepts everything that script does, including --batch, from any directory. The project is the one config.PROJECT names (LITERATURE_RAG_PROJECT, or the project the command runs in), and the papers land in its data/manuscripts. Papers still enter the library only through that script: this command runs it in the project's own environment, then brings data/papers/ up to date, which reads only the papers that are new, redoes the chunks and the test pairs, which takes seconds, and embeds the chunks whose text has no vector yet, which for one paper takes seconds too. A paper downloaded by running the fetch script directly joins at the next `literature-rag-ingest` or `literature-rag-add`.
"""

import subprocess
import sys
from pathlib import Path

from literature_rag import chunks, config, index
from literature_rag.ingest import ingest


def resolve(argument: str) -> str:
    """An argument as the fetch script should get it: absolute when it names a file or folder from the current directory, and unchanged otherwise.

    The script runs from the project's folder and reads a relative path from there, so `literature-rag-add paper.pdf` typed in ~/Downloads would not find the file, and the script would search PubMed for a paper titled "paper.pdf". A DOI, a title, an option, or a path relative to the project names nothing here and still means what it did. A PDF found in neither place stops the command, for the same reason.
    """
    path = Path(argument).expanduser()
    if argument and path.exists():
        return str(path.resolve())
    if argument.lower().endswith(".pdf") and not (config.PROJECT / path).exists():
        raise SystemExit(f"{argument}: no such file, here or in {config.PROJECT}")
    return argument


def fetch(arguments: list[str]) -> int:
    """Run the project's fetch script with these arguments from the project's folder, as the thesis's CLAUDE.md prescribes, and return its exit status."""
    if not config.PROJECT_PYTHON.is_file():
        raise SystemExit(f"{config.PROJECT_PYTHON} not found: the fetch script runs in the project's .venv")
    command = [str(config.PROJECT_PYTHON), str(config.FETCH_SCRIPT), *arguments]
    # Whatever this process printed so far goes out before the script's own output.
    sys.stdout.flush()
    return subprocess.run(command, cwd=config.PROJECT, check=False).returncode


def main() -> None:
    """Entry point for `literature-rag-add`."""
    config.require_project()
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__)
        print("The fetch script's own options follow.\n")
        fetch(["--help"])
        raise SystemExit(0 if arguments else 2)
    status = fetch([resolve(argument) for argument in arguments])
    # The fetch script exits with 1 when any paper failed, but the others may have arrived, so ingestion runs either way.
    papers, new_files, _ = ingest()
    added = [paper for paper in papers if paper["source_file"] in new_files]
    print()
    if not added:
        print("No new manuscript arrived in data/manuscripts, so the library is unchanged.")
    for paper in added:
        print(f"Added {paper['paper_id']} ({paper['source']}, key {paper['key'] or 'none'}): {paper['title']}")
    # A paper the thesis cites adds test pairs as well as chunks, so the whole chunking report is worth seeing.
    print()
    print("\n".join(chunks.build()))
    print()
    print("\n".join(index.build(list(config.COLLECTIONS), list(config.EMBEDDING_MODELS))))
    print("\nRun literature-rag-ingest for the full ingestion report.")
    raise SystemExit(status)
