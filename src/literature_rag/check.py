"""Report whether this machine is ready to build and serve the index.

Run `uv run literature-rag-check`. Each line is one prerequisite, and the command exits with status 1 when a required one is missing. qmd is only the baseline to beat, so it is reported but not required.
"""

import json
import shutil
import subprocess
from collections.abc import Callable

import httpx

from literature_rag import config, index
from literature_rag.chunks import read_jsonl
from literature_rag.models import cached_path, model_ids

# Below this, the model downloads and the index may not fit.
MIN_FREE_GB = 10

# A paper known to be open access in Europe PMC (Garry 2023, "Lassa fever: the road ahead").
KNOWN_PMCID = "PMC9466315"


def thesis_repo() -> tuple[bool, str]:
    if not config.MANUSCRIPTS_DIR.is_dir():
        return False, f"{config.MANUSCRIPTS_DIR} not found; set THESIS_REPO"
    papers = len(list(config.MANUSCRIPTS_DIR.glob("*.md")))
    missing = [path.name for path in (config.BIB_FILE, config.THESIS_FLS) if not path.is_file()]
    if missing:
        return False, f"{papers} papers, but {' and '.join(missing)} missing; run make in the thesis repository"
    return True, f"{papers} papers, {config.BIB_FILE.name} and {config.THESIS_FLS.name} in {config.THESIS_REPO}"


def grobid() -> tuple[bool, str]:
    try:
        alive = httpx.get(f"{config.GROBID_URL}/api/isalive", timeout=5).text.strip()
        version = httpx.get(f"{config.GROBID_URL}/api/version", timeout=5).json()["version"]
    except (httpx.HTTPError, ValueError, KeyError):
        return False, f"no answer at {config.GROBID_URL}; start it with: docker start grobid"
    return alive == "true", f"version {version} at {config.GROBID_URL}"


def europe_pmc() -> tuple[bool, str]:
    params = {"query": f"PMCID:{KNOWN_PMCID}", "format": "json", "resultType": "lite"}
    try:
        hits = httpx.get(f"{config.EUROPE_PMC_URL}/search", params=params, timeout=15).json()["hitCount"]
    except (httpx.HTTPError, ValueError, KeyError):
        return False, f"no answer at {config.EUROPE_PMC_URL}"
    return hits == 1, f"search answers ({hits} hit for {KNOWN_PMCID})"


def models() -> tuple[bool, str]:
    missing = [repo_id for repo_id in model_ids() if cached_path(repo_id) is None]
    if missing:
        return False, f"not downloaded: {', '.join(missing)}; run literature-rag-models"
    return True, f"{len(model_ids())} models in the Hugging Face cache"


def disk() -> tuple[bool, str]:
    free_gb = shutil.disk_usage(config.HOME).free / 1e9
    return free_gb >= MIN_FREE_GB, f"{free_gb:.0f} GB free"


def qmd() -> tuple[bool, str]:
    if shutil.which("qmd") is None:
        return False, "not installed; npm install -g @tobilu/qmd"
    status = subprocess.run(["qmd", "status"], capture_output=True, text=True, timeout=60, check=False)
    if status.returncode != 0:
        return False, f"qmd status failed: {status.stderr.strip()[:120]}"
    indexed = "papers" in status.stdout
    return indexed, "collection 'papers' indexed" if indexed else "no 'papers' collection"


def library() -> tuple[bool, str]:
    """Whether data/papers is up to date with data/manuscripts, so a paper fetched outside literature-rag-add shows up here."""
    if not config.MANIFEST_FILE.is_file():
        return False, "not built yet; run literature-rag-ingest"
    seen = {row["source_file"] for row in json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))}
    waiting = {path.name for path in config.MANUSCRIPTS_DIR.glob("*.md")} - seen
    papers = len(list(config.PAPERS_DIR.glob("*.json")))
    if waiting:
        return False, f"{papers} papers; {len(waiting)} new manuscripts not ingested yet; run literature-rag-ingest"
    return True, f"{papers} papers in {config.PAPERS_DIR.name}/, up to date with the manuscripts"


def vectors() -> tuple[bool, str]:
    """Whether every chunk has a vector from every model, since search refuses to run on an index older than its chunks."""
    gaps = []
    for collection, path in config.COLLECTIONS.items():
        if not path.is_file():
            return False, f"no {path.name}; run literature-rag-chunk"
        chunk_list = read_jsonl(path)
        for model in config.EMBEDDING_MODELS:
            if absent := index.missing(collection, model, chunk_list):
                gaps.append(f"{absent} {collection} chunks without {model}")
    if gaps:
        return False, f"{'; '.join(gaps)}; run literature-rag-build"
    return True, f"{' and '.join(config.COLLECTIONS)} embedded with {' and '.join(config.EMBEDDING_MODELS)}, up to date"


CHECKS: list[tuple[str, Callable[[], tuple[bool, str]], bool]] = [
    ("thesis repository", thesis_repo, True),
    ("GROBID", grobid, True),
    ("Europe PMC", europe_pmc, True),
    ("models", models, True),
    ("disk", disk, True),
    ("qmd baseline", qmd, False),
    ("library", library, False),
    ("index", vectors, False),
]


def main() -> None:
    failed = False
    for name, check, required in CHECKS:
        ok, detail = check()
        mark = "ok" if ok else ("MISSING" if required else "missing")
        print(f"{mark:<8} {name:<18} {detail}")
        failed |= required and not ok
    raise SystemExit(1 if failed else 0)
