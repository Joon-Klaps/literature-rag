"""One row per paper in data/manuscripts: its identifiers, its citation key, whether it is citable, and the route its text will take.

The identifiers come from the metadata block that the fetch script writes at the top of every Markdown file. The key comes from allpapers.bib: an entry in key-overrides.csv first, since it is a correction made by hand, then a DOI match, then an exact match of the normalised title, then a close title match. Where two bibliography entries describe the same paper, the one the thesis cites wins.
"""

import csv
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from literature_rag import config
from literature_rag.bib import cited_keys, normalise_doi, normalise_title, parse_bib
from literature_rag.records import BibEntry, ManifestRow

METADATA_LINE = re.compile(r"^- ([A-Za-z][A-Za-z ]*?): (.*)$", re.MULTILINE)
# The fetch script names a file "<Author> <year> - <title>", with characters a file name cannot hold, such as ":", replaced by " -". A few older files have no year.
FILE_NAME = re.compile(r"^(?P<authors>.+?)(?: (?P<year>\d{4}))? - (?P<title>.+)$")
# Supplementary material, by its file name: "SUPPL", "_suppl", or Springer Nature's "MOESM" and "_ESM".
SUPPLEMENT = re.compile(r"suppl|moesm|_esm", re.IGNORECASE)
# The fetch script writes this when a field is unknown.
UNKNOWN = "N/A"


def parse_metadata(md_text: str) -> dict[str, str]:
    """The "- Field: value" lines of a Markdown file's metadata block, with unknown values left out, and its "# " heading as Title."""
    head = md_text.split("## Extracted Full Text", 1)[0]
    fields = {name: value.strip() for name, value in METADATA_LINE.findall(head) if value.strip() not in ("", UNKNOWN)}
    heading = re.search(r"^# (.+)$", head, re.MULTILINE)
    if heading:
        fields["Title"] = heading.group(1).strip().rstrip(".")
    return fields


def best_title(metadata: dict[str, str], file_stem: str) -> str:
    """The paper's title: the PubMed title when the fetch script matched a record, otherwise the title in the file name.

    Without a PubMed match, the heading is whatever the script read from the PDF, which can be junk such as "DocHdl1OnVERSA-PPM01tmpTarget".
    """
    from_name = FILE_NAME.match(file_stem)
    if metadata.get("Matched PubMed record") == "yes" and metadata.get("Title"):
        return metadata["Title"]
    if from_name:
        return from_name.group("title")
    return metadata.get("Title", file_stem)


def file_year(metadata: dict[str, str], file_stem: str) -> int | None:
    """The year in the metadata block, or else the year in the file name."""
    if re.fullmatch(r"\d{4}", metadata.get("Year", "")):
        return int(metadata["Year"])
    from_name = FILE_NAME.match(file_stem)
    return int(from_name.group("year")) if from_name and from_name.group("year") else None


class BibIndex:
    """allpapers.bib looked up by DOI and by normalised title."""

    def __init__(self, entries: list[BibEntry]):
        self.entries = entries
        self.by_key = {entry["key"]: entry for entry in entries}
        self.by_doi: dict[str, list[BibEntry]] = defaultdict(list)
        self.by_title: dict[str, list[BibEntry]] = defaultdict(list)
        for entry in entries:
            if entry["doi"]:
                self.by_doi[entry["doi"]].append(entry)
            if entry["title"]:
                self.by_title[entry["title"]].append(entry)

    @staticmethod
    def pick(candidates: list[BibEntry]) -> str:
        """The key among several entries for one paper: a cited one if there is one, else the first."""
        return next((entry for entry in candidates if entry["cited"]), candidates[0])["key"]

    def most_similar(self, title: str) -> tuple[str | None, float]:
        """The key whose title is most similar to this normalised title, with the similarity."""
        best_key, best_ratio = None, 0.0
        for entry in self.entries:
            matcher = SequenceMatcher(None, title, entry["title"], autojunk=False)
            # The quick ratios are upper bounds on the real one, so most entries are ruled out without the expensive comparison.
            if matcher.real_quick_ratio() <= best_ratio or matcher.quick_ratio() <= best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio or (ratio == best_ratio and entry["cited"]):
                best_key, best_ratio = entry["key"], ratio
        return best_key, best_ratio

    def match(self, doi: str | None, titles: list[str]) -> tuple[str | None, str | None]:
        """The key for a paper with this DOI and these candidate titles, and how it was found: by DOI, then exact title, then similar title."""
        doi = normalise_doi(doi)
        if doi and doi in self.by_doi:
            return self.pick(self.by_doi[doi]), "doi"
        normalised = [normalise_title(title) for title in titles if title]
        for title in normalised:
            if title in self.by_title:
                return self.pick(self.by_title[title]), "title"
        for title in normalised:
            # A short title matches too many things by chance.
            if len(title) < 20:
                continue
            key, ratio = self.most_similar(title)
            if key and ratio >= config.TITLE_SIMILARITY:
                return key, "similar title"
        return None, None


def read_overrides(path: Path) -> dict[str, str]:
    """key-overrides.csv as {Markdown file name: citation key}. File names contain commas, so the file is real CSV with quoting."""
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["source_file"]: row["key"] for row in csv.DictReader(f) if row.get("key")}


def build_row(
    md_name: str, md_text: str, pdf_name: str | None, bib: BibIndex, overrides: dict[str, str]
) -> ManifestRow:
    """One manifest row from a Markdown file's name and text."""
    stem = Path(md_name).stem
    metadata = parse_metadata(md_text)
    title = best_title(metadata, stem)
    if md_name in overrides and overrides[md_name] in bib.by_key:
        key, matched_by = overrides[md_name], "override"
    else:
        from_name = FILE_NAME.match(stem)
        titles = [title, metadata.get("Title", ""), from_name.group("title") if from_name else ""]
        key, matched_by = bib.match(metadata.get("DOI"), titles)
    pmcid = metadata.get("PMCID")
    supplement = bool(SUPPLEMENT.search(md_name))
    return ManifestRow(
        paper_id=(f"{key}-supplement" if supplement else key) if key else stem,
        key=key,
        matched_by=matched_by,
        citable=key is not None and md_name not in config.NOT_CITABLE,
        route="jats" if pmcid and not supplement else "grobid",
        source_file=md_name,
        pdf_file=pdf_name,
        supplement=supplement,
        duplicate_of=None,
        title=f"{title} (supplementary material)" if supplement else title,
        year=file_year(metadata, stem),
        doi=normalise_doi(metadata.get("DOI")),
        pmid=metadata.get("PMID"),
        pmcid=pmcid.upper() if pmcid else None,
    )


def mark_duplicates(rows: list[ManifestRow]) -> list[str]:
    """Find files that share a paper id, which means the same paper (or the same supplement) was downloaded twice, and keep one of each.

    The copy kept here is a first choice: a citable one with a PMCID if there is one, since Europe PMC's XML is the better text. The others get duplicate_of and their file stem as id. Ingestion has the last word, since it can see which copy's text came out best. Returns one line per duplicate, for the report.
    """
    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        groups[row["paper_id"]].append(row)
    lines = []
    for group in groups.values():
        keeper, *others = sorted(group, key=lambda row: (not row["citable"], row["pmcid"] is None, row["source_file"]))
        for row in others:
            row["duplicate_of"] = keeper["source_file"]
            row["paper_id"] = Path(row["source_file"]).stem
            lines.append(f"{row['source_file']} repeats {keeper['source_file']}")
    return lines


def load_bib_index() -> BibIndex:
    cited = cited_keys(config.THESIS_BBL.read_text(encoding="utf-8")) if config.THESIS_BBL.is_file() else set()
    return BibIndex(parse_bib(config.BIB_FILE.read_text(encoding="utf-8"), cited))


def read_markdown(path: Path) -> str:
    """A Markdown file's text. Some files hold stray NUL bytes from pypdf, which are dropped."""
    return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")


def build_manifest(bib: BibIndex | None = None) -> tuple[list[ManifestRow], list[str]]:
    """A row for every Markdown file in data/manuscripts, sorted by file name, and a line per duplicate file."""
    bib = bib or load_bib_index()
    overrides = read_overrides(config.KEY_OVERRIDES_FILE)
    rows = []
    for md_path in sorted(config.MANUSCRIPTS_DIR.glob("*.md")):
        pdf_path = md_path.with_suffix(".pdf")
        rows.append(
            build_row(
                md_path.name, read_markdown(md_path), pdf_path.name if pdf_path.is_file() else None, bib, overrides
            )
        )
    return rows, mark_duplicates(rows)
