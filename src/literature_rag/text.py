"""Text clean-up shared by the JATS, TEI and Markdown parsers (citation markers, headings and whitespace), and the sentence splitter that the chunker and the thesis reader share."""

import re
from collections.abc import Callable

from lxml import etree

from literature_rag import config

# A numeric citation marker such as "12", "3–5", "[2]" or "(4, 7)". Its text means nothing once the citation is kept as a reference id, so it is taken out. GROBID often puts the separator inside the link, as in "13," or "[10,", so one may follow.
NUMERIC_MARKER = re.compile(r"[\[(]?\s*\d+[a-z]?(\s*[,–\-−—]\s*\d+[a-z]?)*\s*[\])]?[\s,;]*")
# A year on its own, as in "Delwart, <xref>2007</xref>": author-year styles often link only the year. It is part of the sentence, so it stays, as the rest of an author-year marker does.
YEAR_MARKER = re.compile(r"[\[(]?\s*(?:19|20)\d{2}[a-z]?\s*[\])]?[\s,;]*")
# What is left of "[1, 2]" or "(3–5)" once the markers inside are gone.
EMPTY_BRACKETS = re.compile(r"[\[(][\s,;–\-−—]*[\])]")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:)\]])")
SPACE_AFTER_OPENING = re.compile(r"([(\[])\s+")
# A section number in front of a heading, as in "2.1 Data collection", "3. Results" or "A. Serology data". The lookahead keeps "1 × 300 data generation" whole.
HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-Za-z]\.|[IVX]+\.)\s+(?=[A-Za-z])")
BOILERPLATE = re.compile("|".join(f"(?:{pattern})" for pattern in config.BOILERPLATE_HEADINGS))
DASHES = {"–", "-", "−", "—"}
# What may stand between two citation markers in a list or a range.
SEPARATOR = re.compile(r"[\s,;–\-−—]*")
# Some publishers put an invisible zero-width space in headings.
ZERO_WIDTH_SPACE = "\u200b"
# A full stop, question mark or exclamation mark, any closing quotes or brackets after it, and the space before the next sentence, which must not start with a lowercase letter.
SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s+(?![a-z])")
# Single letters joined by full stops, as in "e.g.", "i.e." and "U.S.", once lowercased and without the last full stop. A number such as "5.2" does not match, so "was 5.2. Then" still splits.
LETTERS_AND_STOPS = re.compile(r"(?:[a-z]\.)+[a-z]")
# Words that take a full stop without ending a sentence, lowercased and without that full stop. Single letters, such as the initial in "M. natalensis", are recognised without a list.
ABBREVIATIONS = {
    "al",
    "approx",
    "ca",
    "cf",
    "dr",
    "eq",
    "eqs",
    "fig",
    "figs",
    "no",
    "nos",
    "prof",
    "ref",
    "refs",
    "resp",
    "sp",
    "spp",
    "ssp",
    "st",
    "suppl",
    "tab",
    "var",
    "viz",
    "vol",
    "vs",
}


def is_numeric_marker(text: str) -> bool:
    marker = text.strip()
    return bool(NUMERIC_MARKER.fullmatch(marker)) and not YEAR_MARKER.fullmatch(marker)


def tidy(text: str) -> str:
    """Collapse whitespace and remove the empty brackets and stray spaces left where citation markers were."""
    text = " ".join(text.replace(ZERO_WIDTH_SPACE, "").split())
    text = EMPTY_BRACKETS.sub("", text)
    text = SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    text = SPACE_AFTER_OPENING.sub(r"\1", text)
    return " ".join(text.split())


def heading(title: str) -> str:
    """A section title as stored: without its number, zero-width spaces or a final full stop or colon."""
    title = " ".join(title.replace(ZERO_WIDTH_SPACE, "").split())
    return HEADING_NUMBER.sub("", title).rstrip(".:").strip()


def sentences(passage: str) -> list[str]:
    """Split running text at sentence ends, but not after an abbreviation such as "et al.", "Fig." or "e.g.", or after an initial."""
    pieces, start = [], 0
    for end in SENTENCE_END.finditer(passage):
        words = passage[start : end.start() + 1].split()
        last = words[-1].lstrip("([\"'“‘").rstrip(".!?").lower() if words else ""
        if last in ABBREVIATIONS or LETTERS_AND_STOPS.fullmatch(last) or (len(last) == 1 and last.isalpha()):
            continue
        pieces.append(passage[start : end.end()].strip())
        start = end.end()
    pieces.append(passage[start:].strip())
    return [piece for piece in pieces if piece]


def is_boilerplate(title: str) -> bool:
    """Whether a section with this title is acknowledgements, funding, data availability and the like, which are left out."""
    normalised = heading(title).lower().replace("’", "'").rstrip(".:")
    return bool(BOILERPLATE.fullmatch(normalised))


class ReferenceOrder:
    """The reference list's ids in order, which turns a range written as two links ("1–3") into the three references it covers."""

    def __init__(self, ids: list[str]):
        self.ids = ids
        self.position = {rid: index for index, rid in enumerate(ids)}

    def between(self, first: str | None, last: str) -> list[str]:
        """The ids strictly between first and last, or none if the two do not form a plausible range."""
        if first not in self.position or last not in self.position:
            return []
        start, end = self.position[first], self.position[last]
        return self.ids[start + 1 : end] if 0 < end - start < 50 else []


def render_with_citations(
    element: etree._Element,
    cites: list[str],
    order: ReferenceOrder,
    cited_ids: Callable[[etree._Element], list[str] | None],
    child_text: Callable[[etree._Element], str],
) -> str:
    """The running text inside an element, with numeric citation markers taken out and the cited reference ids added to cites.

    JATS and TEI differ only in how a citation link looks, so the parsers pass cited_ids, which returns the reference ids of a citation link and None for any other element, and child_text, which renders any other element. Author-year markers such as "Andersen et al., 2015" are part of the sentence, so their text stays.
    """
    parts = [element.text or ""]
    # The id of the last citation link, while nothing but a dash follows it.
    range_start = None
    for child in element:
        rids = cited_ids(child)
        if rids is None:
            range_start = None
            parts.append(child_text(child))
            parts.append(child.tail or "")
            continue
        if rids:
            cites.extend(order.between(range_start, rids[0]))
        cites.extend(rids)
        removed = is_numeric_marker("".join(child.itertext()))
        parts.append("" if removed else "".join(child.itertext()))
        tail = child.tail or ""
        range_start = rids[-1] if rids and tail.strip() in DASHES else None
        # The comma or dash between two removed markers goes with them, so "1980s 12, 13, 20." reads "1980s."
        following = child.getnext()
        if removed and SEPARATOR.fullmatch(tail) and following is not None and cited_ids(following) is not None:
            tail = ""
        parts.append(tail)
    return "".join(parts)
