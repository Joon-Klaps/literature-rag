"""The shape of every record the stages pass to each other.

Each stage writes JSON under data/, and the next stage reads it back, so these TypedDicts are the contract between them. They describe plain dicts: nothing checks them at run time, but an editor and a type checker do.
"""

from typing import Literal, TypedDict

# Where a paper's structured text came from: the publisher's XML via Europe PMC, GROBID's reading of the PDF, or, when both fail, the Markdown the fetch script extracted with pypdf.
Source = Literal["jats", "grobid", "markdown"]


class BibEntry(TypedDict):
    """One entry of allpapers.bib, reduced to what is needed to match a paper to its key."""

    key: str
    doi: str | None
    # Lowercase alphanumerics separated by single spaces, so that two spellings of one title compare equal.
    title: str
    year: int | None
    cited: bool


class ManifestRow(TypedDict):
    """One paper in data/manuscripts, before its text is parsed."""

    paper_id: str
    key: str | None
    # How the key was found: "doi", "title", "similar title", "override", or None when it was not.
    matched_by: str | None
    citable: bool
    route: Literal["jats", "grobid"]
    source_file: str
    pdf_file: str | None
    # Supplementary material carries the key of its paper, and its own text always comes from GROBID: its PMCID, if any, is the paper's.
    supplement: bool
    # The file this one repeats, when the same paper was downloaded twice. Duplicates are listed but not ingested.
    duplicate_of: str | None
    title: str
    year: int | None
    doi: str | None
    pmid: str | None
    pmcid: str | None


class Reference(TypedDict):
    """One entry of a paper's reference list."""

    text: str
    doi: str | None
    pmid: str | None
    pmcid: str | None


class Paragraph(TypedDict):
    text: str
    # Ids of the references this paragraph cites, as keys of Paper["references"].
    cites: list[str]


class Section(TypedDict):
    # Nested headings are joined, as in "Methods > Sequencing"; an empty heading is text that sits under no heading.
    heading: str
    paragraphs: list[Paragraph]


class Caption(TypedDict):
    # "Fig. 1", "Table 2" and so on, as the paper prints it.
    label: str
    text: str
    cites: list[str]


class ParsedText(TypedDict):
    """What a parser reads from one document: the paper's own metadata and its text."""

    title: str
    year: int | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    abstract: str
    sections: list[Section]
    captions: list[Caption]
    references: dict[str, Reference]


class Paper(ParsedText):
    """One paper as block 2 writes it to data/papers/<paper_id>.json: the parsed text, tied to its citation key."""

    paper_id: str
    key: str | None
    citable: bool
    source: Source
    source_file: str
