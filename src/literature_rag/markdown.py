"""The fallback: the Markdown the fetch script extracted with pypdf, for a paper that neither Europe PMC nor GROBID can read.

This text has no sections, no captions and no linked citations, and its citation numbers are fused to words ("centuries1"). Keeping it all the same means no paper drops out of the library. Each page becomes one paragraph; block 3 splits long ones at sentence ends.
"""

import re
import unicodedata

from literature_rag import text
from literature_rag.manifest import parse_metadata
from literature_rag.records import Paragraph, ParsedText, Section

PAGE_HEADING = re.compile(r"^## Page \d+\s*$", re.MULTILINE)
FULL_TEXT_HEADING = "## Extracted Full Text"
# What the fetch script writes when a PDF has no text layer, as for a scanned paper.
NO_TEXT = "_No machine-readable text could be extracted from this PDF._"


def parse(md_text: str) -> ParsedText:
    """The abstract and the page texts of a fetch-script Markdown file, with its metadata block and page headings removed."""
    # pypdf leaves stray NUL bytes and typographic ligatures such as "ﬁ"; NFKC turns the ligatures into plain letters, so "ﬁrst" can be found as "first".
    md_text = unicodedata.normalize("NFKC", md_text.replace("\x00", ""))
    head, _, body = md_text.partition(FULL_TEXT_HEADING)
    metadata = parse_metadata(md_text)
    abstract = head.partition("## Abstract")[2]
    pages = [text.tidy(page) for page in PAGE_HEADING.split(body.replace(NO_TEXT, ""))]
    paragraphs = [Paragraph(text=page, cites=[]) for page in pages if page]
    year = metadata.get("Year", "")
    return ParsedText(
        title=metadata.get("Title", ""),
        year=int(year) if year.isdigit() else None,
        doi=metadata.get("DOI"),
        pmid=metadata.get("PMID"),
        pmcid=metadata.get("PMCID"),
        abstract=text.tidy(abstract),
        sections=[Section(heading="", paragraphs=paragraphs)] if paragraphs else [],
        captions=[],
        references={},
    )
