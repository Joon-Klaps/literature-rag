"""Turn every paper in data/manuscripts into structured text: one data/papers/<paper_id>.json per paper.

Run `uv run literature-rag-ingest`. Each paper takes the best route open to it: the publisher's XML from Europe PMC when it has a PMCID and the XML is open access, GROBID's reading of the PDF otherwise, and the fetch script's Markdown when both fail, so no paper drops out.

Downloads and GROBID output are cached under data/raw/, so a run after the first only fetches or processes the papers it has not seen, and parses the rest from the cache in seconds. That is what lets a newly downloaded paper join the library with one command; `literature-rag-add` runs the fetch script and then this.
"""

import asyncio
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from literature_rag import config, grobid, jats, markdown
from literature_rag.bib import normalise_title
from literature_rag.manifest import BibIndex, build_manifest, load_bib_index, read_markdown
from literature_rag.records import ManifestRow, Paper, ParsedText, Source

PAGE_HEADING = re.compile(r"^## Page \d+", re.MULTILINE)
SOURCE_RANK = {"jats": 2, "grobid": 1, "markdown": 0}
# Words too common to show that a text is the paper its title names.
COMMON_WORDS = {
    "about",
    "after",
    "among",
    "analysis",
    "based",
    "between",
    "study",
    "using",
    "their",
    "these",
    "through",
    "which",
    "within",
    "without",
}


def page_count(md_text: str) -> int:
    """The number of pages in the PDF, from the fetch script's "## Page N" headings."""
    return len(PAGE_HEADING.findall(md_text))


def choose_source(row: ManifestRow, jats_path: Path | None, tei_path: Path | None) -> tuple[Source, Path | None]:
    """The best text available for a paper: JATS, then GROBID's TEI, then the Markdown."""
    if jats_path is not None:
        return "jats", jats_path
    if tei_path is not None:
        return "grobid", tei_path
    return "markdown", None


def paper_record(row: ManifestRow, source: Source, parsed: ParsedText, bib: BibIndex) -> Paper:
    """Join a manifest row and a parsed text into one Paper.

    The identifiers from the fetch script come first, since they were checked against PubMed; the parser's fill the gaps. A paper without a key gets a second chance here, with the DOI and title its parser found. The title is the publisher's for JATS, and the manifest's otherwise, since GROBID and pypdf can misread a title page.
    """
    key, citable, paper_id = row["key"], row["citable"], row["paper_id"]
    if key is None and not row["duplicate_of"]:
        key, _ = bib.match(parsed["doi"], [parsed["title"]])
        if key is not None:
            paper_id = key
            citable = row["source_file"] not in config.NOT_CITABLE
    return Paper(
        paper_id=paper_id,
        key=key,
        citable=citable,
        title=parsed["title"] if source == "jats" and parsed["title"] else row["title"],
        year=row["year"] or parsed["year"],
        doi=row["doi"] or parsed["doi"],
        pmid=row["pmid"] or parsed["pmid"],
        pmcid=row["pmcid"] or parsed["pmcid"],
        source=source,
        source_file=row["source_file"],
        abstract=parsed["abstract"],
        sections=parsed["sections"],
        captions=parsed["captions"],
        references=parsed["references"],
    )


def title_found(paper: Paper) -> bool:
    """Whether most of the distinctive words of a paper's title appear in its text. When they do not, the file probably holds another document, such as a publisher's notice saved in place of the paper."""
    words = {word for word in normalise_title(paper["title"]).split() if len(word) >= 5 and word not in COMMON_WORDS}
    if len(words) < 2:
        return True
    body = " ".join([paper["abstract"]] + [p["text"] for s in paper["sections"] for p in s["paragraphs"]])
    present = set(normalise_title(body).split())
    return len(words & present) / len(words) >= 0.5


def summarise(papers: list[Paper], notes: dict[str, list[str]]) -> list[str]:
    """The ingestion report: papers per source, the medians per paper, how much of the text is linked to references, and every paper that needs a look."""
    lines = [
        f"{len(papers)} papers: "
        + ", ".join(f"{count} {source}" for source, count in Counter(p["source"] for p in papers).most_common())
    ]
    lines.append(
        f"{'source':<10}{'papers':>8}{'sections':>10}{'paragraphs':>12}{'references':>12}{'citing':>9}{'ref DOI':>9}"
    )
    for source in ("jats", "grobid", "markdown", "all"):
        group = [p for p in papers if source in ("all", p["source"])]
        if not group:
            continue
        paragraphs = [para for p in group for s in p["sections"] for para in s["paragraphs"]]
        references = [ref for p in group for ref in p["references"].values()]
        citing = sum(bool(para["cites"]) for para in paragraphs) / max(1, len(paragraphs))
        with_doi = sum(ref["doi"] is not None for ref in references) / max(1, len(references))
        lines.append(
            f"{source:<10}{len(group):>8}"
            f"{statistics.median(len(p['sections']) for p in group):>10.0f}"
            f"{statistics.median(sum(len(s['paragraphs']) for s in p['sections']) for p in group):>12.0f}"
            f"{statistics.median(len(p['references']) for p in group):>12.0f}"
            f"{citing:>9.0%}{with_doi:>9.0%}"
        )
    lines.append(
        "(medians per paper; citing is the share of paragraphs that cite a reference, ref DOI the share of references with a DOI)"
    )
    notes = {
        **notes,
        "without a key (add them to key-overrides.csv if the bibliography has them)": [
            p["source_file"] for p in papers if p["key"] is None
        ],
        "whose text does not match the title (check the PDF)": [p["source_file"] for p in papers if not title_found(p)],
    }
    for heading, items in notes.items():
        if items:
            lines.append(f"\n{len(items)} {heading}:")
            lines.extend(f"  {item}" for item in items)
    return lines


async def gather_sources(
    rows: list[ManifestRow], md_texts: dict[str, str]
) -> tuple[dict[str, Path | None], dict[str, Path | None], dict[str, list[str]]]:
    """Fetch the JATS and run GROBID for every paper that needs it, using the caches where they hold an answer already.

    Copies of one paper (rows with duplicate_of) are read too, so that ingest can keep the best of them, but only where it can make a difference: not when the kept copy has JATS, which every copy shares through the PMCID, and not when the PDFs are identical.

    Returns the JATS and the TEI path per Markdown file (None where there is none), and notes for the report.
    """
    notes: dict[str, list[str]] = {}
    by_file = {row["source_file"]: row for row in rows}

    def first_copy(row: ManifestRow) -> ManifestRow:
        return by_file[row["duplicate_of"]] if row["duplicate_of"] else row

    def same_pdf(row: ManifestRow) -> bool:
        first = first_copy(row)
        if first is row or not (row["pdf_file"] and first["pdf_file"]):
            return False
        return (config.MANUSCRIPTS_DIR / row["pdf_file"]).read_bytes() == (
            config.MANUSCRIPTS_DIR / first["pdf_file"]
        ).read_bytes()

    jats_rows = [row for row in rows if row["route"] == "jats"]
    fetched = await jats.fetch_all([row["pmcid"] for row in jats_rows])
    jats_paths: dict[str, Path | None] = {}
    for row in jats_rows:
        result = fetched[row["pmcid"]]
        if isinstance(result, Exception):
            notes.setdefault("Europe PMC errors, retried next run", []).append(f"{row['source_file']}: {result!r}")
            result = None
        jats_paths[row["source_file"]] = result

    needs_grobid = [
        row
        for row in rows
        if jats_paths.get(row["source_file"]) is None
        and jats_paths.get(first_copy(row)["source_file"]) is None
        and row["pdf_file"]
        and page_count(md_texts[row["source_file"]]) <= config.GROBID_MAX_PAGES
        and not same_pdf(row)
    ]
    tei_paths: dict[str, Path | None] = {}
    uncached = [
        row for row in needs_grobid if not any(path.is_file() for path in grobid.cache_paths(Path(row["pdf_file"])))
    ]
    if uncached and not grobid.is_alive():
        notes["waiting for GROBID: run docker start grobid, then ingest again"] = [
            row["source_file"] for row in uncached
        ]
        needs_grobid = [row for row in needs_grobid if row not in uncached]
    elif uncached:
        print(
            f"GROBID: {len(uncached)} PDFs to read, {config.GROBID_CONCURRENCY} at a time; about half a minute each with the Crossref lookups, minutes for a long review",
            flush=True,
        )
    processed = await grobid.process_all([config.MANUSCRIPTS_DIR / row["pdf_file"] for row in needs_grobid])
    for row in needs_grobid:
        result = processed[config.MANUSCRIPTS_DIR / row["pdf_file"]]
        if isinstance(result, Exception):
            notes.setdefault("GROBID errors, retried next run", []).append(f"{row['source_file']}: {result!r}")
            result = None
        tei_paths[row["source_file"]] = result
    for row in rows:
        if same_pdf(row):
            tei_paths[row["source_file"]] = tei_paths.get(first_copy(row)["source_file"])
    return jats_paths, tei_paths, notes


def fallback_reason(row: ManifestRow, md_text: str) -> str:
    """Why a paper is kept as Markdown, for the report."""
    if row["pdf_file"] is None:
        return "no PDF"
    if page_count(md_text) > config.GROBID_MAX_PAGES:
        return f"{page_count(md_text)} pages, more than GROBID_MAX_PAGES"
    failed = grobid.cache_paths(Path(row["pdf_file"]))[1]
    if failed.is_file():
        return f"GROBID: {failed.read_text().strip()[:120]}"
    return "GROBID not run"


def text_quality(paper: Paper) -> tuple[int, int]:
    """How good a copy's text is: JATS over GROBID over Markdown, then more paragraphs."""
    return SOURCE_RANK[paper["source"]], sum(len(section["paragraphs"]) for section in paper["sections"])


def choose_copies(rows: list[ManifestRow], papers: dict[str, Paper]) -> list[str]:
    """Among the files that hold one paper, keep the one whose text came out best, and mark the others as its duplicates.

    Two downloads of one paper can differ: one may carry the supplement or be a scan GROBID reads badly. A citable copy wins, then the better text, and the manifest's own choice breaks a tie. Updates duplicate_of and paper_id in rows and papers, and returns a report line per duplicate.
    """
    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        groups[row["duplicate_of"] or row["source_file"]].append(row)
    lines = []
    for first, members in groups.items():
        if len(members) == 1:
            continue
        shared_id = next(row["paper_id"] for row in members if row["source_file"] == first)
        best = max(
            members,
            key=lambda row: (row["citable"], text_quality(papers[row["source_file"]]), row["source_file"] == first),
        )
        for row in members:
            if row is best:
                row["duplicate_of"], row["paper_id"] = None, shared_id
            else:
                row["duplicate_of"], row["paper_id"] = best["source_file"], Path(row["source_file"]).stem
                lines.append(f"{row['source_file']} repeats {best['source_file']}")
            papers[row["source_file"]]["paper_id"] = row["paper_id"]
    return lines


def parse_source(source: Source, path: Path | None, md_text: str) -> ParsedText:
    if source == "jats":
        return jats.parse(path.read_bytes())
    if source == "grobid":
        return grobid.parse(path.read_bytes())
    return markdown.parse(md_text)


def write_papers(papers: list[Paper]) -> int:
    """Write one JSON file per paper, and remove the files of papers that are gone or have a new id. Returns the number removed."""
    config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    written = set()
    for paper in papers:
        path = config.PAPERS_DIR / f"{paper['paper_id']}.json"
        path.write_text(json.dumps(paper, ensure_ascii=False, indent=1), encoding="utf-8")
        written.add(path.name)
    stale = [path for path in config.PAPERS_DIR.glob("*.json") if path.name not in written]
    for path in stale:
        path.unlink()
    return len(stale)


def previous_files() -> set[str]:
    """The Markdown files the last run saw, to name the new ones in the report."""
    if not config.MANIFEST_FILE.is_file():
        return set()
    return {row["source_file"] for row in json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))}


def ingest() -> tuple[list[Paper], set[str], list[str]]:
    """Build the manifest, bring every paper's text up to date and write data/papers/. Returns the papers, the Markdown files that are new since the last run, and the report."""
    bib = load_bib_index()
    rows, _ = build_manifest(bib)
    seen_before = previous_files()
    md_texts = {row["source_file"]: read_markdown(config.MANUSCRIPTS_DIR / row["source_file"]) for row in rows}
    jats_paths, tei_paths, notes = asyncio.run(gather_sources(rows, md_texts))

    by_file: dict[str, Paper] = {}
    for row in rows:
        source, path = choose_source(row, jats_paths.get(row["source_file"]), tei_paths.get(row["source_file"]))
        try:
            parsed = parse_source(source, path, md_texts[row["source_file"]])
        except Exception as error:  # noqa: BLE001 - a file the parser cannot read falls back to the Markdown, so the paper stays in
            notes.setdefault("could not be parsed, kept as Markdown", []).append(f"{row['source_file']}: {error!r}")
            source, parsed = "markdown", markdown.parse(md_texts[row["source_file"]])
        by_file[row["source_file"]] = paper_record(row, source, parsed, bib)
    duplicates = choose_copies(rows, by_file)
    kept = [row for row in rows if not row["duplicate_of"]]
    papers = [by_file[row["source_file"]] for row in kept]
    for row, paper in zip(kept, papers):
        if paper["source"] == "markdown":
            notes.setdefault("kept as Markdown", []).append(
                f"{row['source_file']}: {fallback_reason(row, md_texts[row['source_file']])}"
            )
    # A key found only now, from the parsed DOI or title, can belong to a paper that is already in the library under that key; the newcomer then keeps its file stem as id rather than overwrite it.
    taken = {row["paper_id"] for row in kept}
    for paper, row in zip(papers, kept):
        if paper["paper_id"] != row["paper_id"] and paper["paper_id"] in taken:
            notes.setdefault("found a key that another file already holds", []).append(
                f"{row['source_file']}: {paper['key']}"
            )
            paper["paper_id"] = Path(row["source_file"]).stem
        taken.add(paper["paper_id"])

    config.MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    removed = write_papers(papers)
    new_files = {row["source_file"] for row in rows if row["source_file"] not in seen_before}
    if seen_before:
        notes["new since the last run"] = sorted(new_files)
    notes["duplicate files, not ingested"] = duplicates
    report = summarise(papers, notes)
    if removed:
        report.append(f"\nRemoved {removed} paper files that no longer match a manuscript.")
    return papers, new_files, report


def main() -> None:
    """Entry point for `uv run literature-rag-ingest`."""
    _, _, report = ingest()
    print("\n".join(report))
    print(f"\nWrote {config.PAPERS_DIR} and {config.MANIFEST_FILE}")
