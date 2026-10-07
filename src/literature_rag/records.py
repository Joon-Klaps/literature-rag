"""The shape of every record the stages pass to each other.

Each stage writes JSON under data/, and the next stage reads it back, so these TypedDicts are the contract between them. They describe plain dicts: nothing checks them at run time, but an editor and a type checker do.
"""

from typing import Literal, NotRequired, TypedDict

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
    # The paper's Markdown file in the thesis repository's data/manuscripts, the copy ingestion kept; its PDF has the same name.
    source_file: str
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


class RealisticQuestion(TypedDict):
    """A literature point from a review round, put as the question Claude would search the library with, and the papers that answer it. Written by hand to data/eval/realistic.jsonl."""

    # "<review file stem>/<point>", as in "LK0.3.0/8": the review, and the point's number or a short name for it.
    question_id: str
    question: str
    # Citation keys as the library spells them. A hit on any of them answers the question.
    keys: list[str]
    # True once Joon has confirmed that these papers answer the point.
    confirmed: bool


class Hit(TypedDict):
    """One passage that search returns, as block 4's search() gives it."""

    # From 1, in the order search returns the hits.
    rank: int
    # The score the final order rests on: the reranker's probability when the shortlist was reranked, the fused score when several methods ran, and otherwise the one method's own score (BM25, or the inner product of the vectors).
    score: float
    # Where each method placed this chunk among its candidates, from 1, and "fused" for its place after fusion. A method that did not find it among its config.CANDIDATES is missing.
    ranks: dict[str, int]
    chunk: Chunk | ThesisChunk


class LibraryPassage(TypedDict):
    """One passage of a paper as the MCP server returns it, from search_library and get_context: where it comes from, its text, and the references it cites."""

    chunk_id: str
    key: str | None
    citable: bool
    title: str
    year: int | None
    doi: str | None
    # The paper's Markdown file, relative to the thesis repository, as in "data/manuscripts/Garry 2023 - Lassa fever - the road ahead.md". The PDF has the same name with .pdf.
    manuscript: str
    section: str
    text: str
    cites: list[Reference]
    # The score search ranked it by, which with the reranker is its relevance from 0 to 1. Near 0 means off the subject, but a high score only means on it: a question about a phase 3 Lassa vaccine trial, which has never been run, scores 0.95. Only search_library gives it.
    score: NotRequired[float]


class ThesisParagraph(TypedDict):
    """One paragraph or caption of the thesis as the MCP server returns it, from search_thesis and get_context."""

    chunk_id: str
    # Relative to the thesis repository, as in "chapters/02_introduction/02_introduction.tex".
    file: str
    line_start: int
    line_end: int
    section: str
    kind: Literal["paragraph", "caption"]
    text: str
    keys: list[str]
    # False when those lines of the file have changed since literature-rag-chunk read them, so the line range no longer holds.
    current: bool
    # As in LibraryPassage: only search_thesis gives it.
    score: NotRequired[float]
