"""Add papers to the library from the command line: download them with the thesis repository's fetch script, then ingest them.

    literature-rag-add 10.1038/srep21977
    literature-rag-add --mode title "Spatial and temporal evolution of Lassa virus"
    literature-rag-add path/to/paper.pdf

The arguments go unchanged to scripts/fetch_pubmed_pdf.py, so this accepts everything that script does, including --batch. Papers still enter the library only through that script: this command runs it in the thesis repository's own environment, and then brings data/papers/ up to date, which reads only the papers that are new. A paper downloaded by running the fetch script directly joins at the next `literature-rag-ingest` or `literature-rag-add`.
"""

import subprocess
import sys

from literature_rag import config
from literature_rag.ingest import ingest


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
    status = fetch(arguments)
    # The fetch script exits with 1 when any paper failed, but the others may have arrived, so ingestion runs either way.
    papers, new_files, _ = ingest()
    added = [paper for paper in papers if paper["source_file"] in new_files]
    print()
    if not added:
        print("No new manuscript arrived in data/manuscripts, so the library is unchanged.")
    for paper in added:
        print(f"Added {paper['paper_id']} ({paper['source']}, key {paper['key'] or 'none'}): {paper['title']}")
    print("\nRun literature-rag-ingest for the full report.")
    raise SystemExit(status)
