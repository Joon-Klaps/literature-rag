"""Add papers to the library from the command line: download them with the thesis repository's fetch script, then ingest, chunk and embed them.

    literature-rag-add 10.1038/srep21977
    literature-rag-add --mode title "Spatial and temporal evolution of Lassa virus"
    literature-rag-add path/to/paper.pdf

The arguments go to scripts/fetch_pubmed_pdf.py, with file paths made absolute, so this accepts everything that script does, including --batch, from any directory. Papers still enter the library only through that script: this command runs it in the thesis repository's own environment, then brings data/papers/ up to date, which reads only the papers that are new, redoes the chunks and the test pairs, which takes seconds, and embeds the chunks whose text has no vector yet, which for one paper takes seconds too. A paper downloaded by running the fetch script directly joins at the next `literature-rag-ingest` or `literature-rag-add`.
"""

import subprocess
import sys
from pathlib import Path

from literature_rag import chunks, config, index
from literature_rag.ingest import ingest


def resolve(argument: str) -> str:
    """An argument as the fetch script should get it: absolute when it names a file or folder from the current directory, and unchanged otherwise.

    The script runs from the thesis repository and reads a relative path from there, so `literature-rag-add paper.pdf` typed in ~/Downloads would not find the file, and the script would search PubMed for a paper titled "paper.pdf". A DOI, a title, an option, or a path relative to the thesis repository names nothing here and still means what it did. A PDF found in neither place stops the command, for the same reason.
    """
    path = Path(argument).expanduser()
    if argument and path.exists():
        return str(path.resolve())
    if argument.lower().endswith(".pdf") and not (config.THESIS_REPO / path).exists():
        raise SystemExit(f"{argument}: no such file, here or in {config.THESIS_REPO}")
    return argument


def fetch(arguments: list[str]) -> int:
    """Run the fetch script with these arguments from the thesis repository, as its CLAUDE.md prescribes, and return its exit status."""
    if not config.THESIS_PYTHON.is_file():
        raise SystemExit(f"{config.THESIS_PYTHON} not found: the fetch script runs in the thesis repository's .venv")
    command = [str(config.THESIS_PYTHON), str(config.FETCH_SCRIPT), *arguments]
    # Whatever this process printed so far goes out before the script's own output.
    sys.stdout.flush()
    return subprocess.run(command, cwd=config.THESIS_REPO, check=False).returncode


def main() -> None:
    """Entry point for `literature-rag-add`."""
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
