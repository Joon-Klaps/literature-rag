"""The shape of every record the stages pass to each other.

Each stage writes JSON under data/, and the next stage reads it back, so these TypedDicts are the contract between them. They describe plain dicts: nothing checks them at run time, but an editor and a type checker do.
"""

from typing import Literal, TypedDict

# Where a paper's structured text came from: the publisher's XML via Europe PMC, GROBID's reading of the PDF, or, when both fail, the Markdown the fetch script extracted with pypdf.
Source = Literal["jats", "grobid", "markdown"]
# What part of a paper a library chunk holds.
ChunkKind = Literal["abstract", "body", "caption"]


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


class Chunk(TypedDict):
    """One passage of a paper, the unit that library search ranks, as block 3 writes it to data/chunks/library.jsonl."""

    # "<paper_id>#<n>", numbered in reading order from 0, so the neighbours of chunk n are n - 1 and n + 1.
    chunk_id: str
    paper_id: str
    key: str | None
    citable: bool
    title: str
    year: int | None
    doi: str | None
    # The section heading for body text, "Abstract" for the abstract, and the label ("Fig. 1") for a caption.
    section: str
    kind: ChunkKind
    text: str
    # The references this passage cites, resolved from the paper's reference list. A paragraph cut in pieces gives each piece all of its references, since the parsers keep citations per paragraph, not per sentence.
    cites: list[Reference]


class ThesisChunk(TypedDict):
    """One paragraph or caption of the thesis, as block 3 writes it to data/chunks/thesis.jsonl."""

    # "<file stem>:L<start>-<end>", such as "02_introduction:L23-25".
    chunk_id: str
    # Relative to the thesis repository, as in "chapters/02_introduction/02_introduction.tex".
    file: str
    line_start: int
    line_end: int
    # The headings above the text, from \chapter down to \subsubsection, joined as in "Introduction > Lassa fever".
    section: str
    kind: Literal["paragraph", "caption"]
    # Plain text, with the citation commands taken out.
    text: str
    # The citation keys of every \cite, \citep and \citet in the block, in order of first appearance.
    keys: list[str]
    # A hash of the block's lines as they stand in the file, comments included, to tell when a paragraph has changed.
    hash: str


class Pair(TypedDict):
    """A test question from the thesis: a sentence that cites one paper of the library, which is the answer."""

    # "<chunk_id>/<n>", the paragraph and the position of the sentence in it, from 0.
    pair_id: str
    chunk_id: str
    # The sentence as plain text, without its citation.
    query: str
    key: str
    # The numbers in the sentence, with thin spaces and thousands separators taken out, as in "10000" or "0.5".
    numbers: list[str]
