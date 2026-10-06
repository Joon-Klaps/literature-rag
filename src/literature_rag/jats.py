"""The publisher's XML (JATS) from Europe PMC: download it once, and read it into sections, captions and references.

JATS keeps what the PDF loses: sections with their titles, figure and table captions, and every in-text citation as a link to its entry in the reference list, which in turn carries the cited paper's DOI and PMID.
"""

import asyncio
import re
from pathlib import Path

import httpx
from lxml import etree

from literature_rag import config, text
from literature_rag.bib import normalise_doi
from literature_rag.records import Caption, Paragraph, ParsedText, Reference, Section

# Europe PMC answers 404, or 500 with a JSON error, for a paper whose full text it does not hold, which is the case for papers outside its open-access subset.
NO_FULL_TEXT = {404, 500}
# Rate limits and gateway errors pass; they are worth a second try later, so they are not cached.
TRANSIENT = {429, 502, 503, 504}


def cache_paths(pmcid: str) -> tuple[Path, Path]:
    """The cached XML for a PMCID, and the marker file that records that Europe PMC has none."""
    return config.JATS_DIR / f"{pmcid}.xml", config.JATS_DIR / f"{pmcid}.none"


async def fetch(client: httpx.AsyncClient, pmcid: str) -> Path | None:
    """Download a paper's JATS into data/raw/jats/ unless it is already there, and return its path, or None if Europe PMC has no full text.

    A missing full text is cached as an empty marker file, so it is not asked for again; delete the marker to retry. A transient failure raises httpx.HTTPError and leaves no trace, so the next run tries again.
    """
    xml_path, none_path = cache_paths(pmcid)
    if xml_path.is_file():
        return xml_path
    if none_path.is_file():
        return None
    response = await client.get(f"{config.EUROPE_PMC_URL}/{pmcid}/fullTextXML")
    if response.status_code in NO_FULL_TEXT:
        none_path.parent.mkdir(parents=True, exist_ok=True)
        none_path.write_text(f"HTTP {response.status_code}\n")
        return None
    response.raise_for_status()
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(response.content)
    return xml_path


async def fetch_all(pmcids: list[str]) -> dict[str, Path | None | Exception]:
    """Fetch several papers, a few at a time. Each PMCID maps to its XML, to None when there is no full text, or to the error that stopped it."""
    semaphore = asyncio.Semaphore(config.EUROPE_PMC_CONCURRENCY)

    async def one(client: httpx.AsyncClient, pmcid: str) -> Path | None:
        async with semaphore:
            return await fetch(client, pmcid)

    # Two copies of one paper share a PMCID, which is fetched once.
    pmcids = list(dict.fromkeys(pmcids))
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        results = await asyncio.gather(*(one(client, pmcid) for pmcid in pmcids), return_exceptions=True)
    return dict(zip(pmcids, results))


# Reading the XML. Europe PMC's files declare a DTD that is not needed for reading, so nothing is fetched from the network, and a malformed file is read as far as possible.
XML_PARSER = etree.XMLParser(recover=True, no_network=True, resolve_entities=False, huge_tree=True)
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
# The identifier types a reference or an article may carry, under the names used here.
ID_KINDS = {"doi": "doi", "pmid": "pmid", "pmcid": "pmcid", "pmc": "pmcid"}

# Sections left out by their sec-type, whatever their title: reference lists, acknowledgements, notes, supplements, statements, glossaries.
SKIPPED_SECTION_TYPES = {
    "ref-list",
    "ack",
    "associated-data",
    "fn-group",
    "supplementary-material",
    "supplementary-materials",
    "data-availability",
    "data-availability-statement",
    "contrib-info",
    "bio",
    "COI-statement",
    "data-citations",
    "glossary",
    "extended-data",
}
# Elements inside running text that are not part of it. Figures and tables are read as captions instead.
SKIPPED_INLINE = {
    "fig",
    "table-wrap",
    "table-wrap-group",
    "supplementary-material",
    "media",
    "graphic",
    "boxed-text",
    "fn",
    "label",
}
# Elements that end a run of text, so their content is kept apart from the words around it.
BLOCKS = {
    "p",
    "list",
    "list-item",
    "def-list",
    "def-item",
    "term",
    "def",
    "title",
    "disp-quote",
    "statement",
    "break",
    "caption",
}
# Units of running text directly inside a section.
PARAGRAPHS = {"p", "list", "disp-quote", "def-list", "statement"}
# Where a reference's printed form is found, best first: mixed-citation keeps the punctuation, element-citation has only fields.
CITATION_ELEMENTS = ("mixed-citation", "element-citation", "nlm-citation", "citation")


def local(tag: object) -> str:
    """An element's name without its namespace, or "" for a comment or processing instruction."""
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def formula_text(formula: etree._Element) -> str:
    """The text of a formula: its MathML when there is some, else its TeX, else whatever text it holds."""
    for name in ("math", "tex-math"):
        found = next((element for element in formula.iter() if local(element.tag) == name), None)
        if found is not None:
            return " ".join("".join(found.itertext()).split())
    return "".join(formula.itertext())


def citation_ids(element: etree._Element) -> list[str] | None:
    """The reference ids of a JATS citation link, <xref ref-type="bibr" rid="CR1">, or None for any other element."""
    if local(element.tag) == "xref" and element.get("ref-type") == "bibr":
        return (element.get("rid") or "").split()
    return None


def render(element: etree._Element, cites: list[str], order: text.ReferenceOrder) -> str:
    """The running text inside an element, without numeric citation markers, with the cited reference ids added to cites."""
    return text.render_with_citations(
        element, cites, order, citation_ids, lambda child: render_child(child, cites, order)
    )


def render_child(child: etree._Element, cites: list[str], order: text.ReferenceOrder) -> str:
    """The text one child element contributes to the running text around it."""
    name = local(child.tag)
    if not name or name in SKIPPED_INLINE:
        return ""
    if name == "xref" and child.get("ref-type") in ("fn", "table-fn"):
        return ""
    if name in ("inline-formula", "disp-formula"):
        return f" {formula_text(child)} "
    if name == "sup":
        cited_before = len(cites)
        inner = render(child, cites, order)
        # A superscript that held only citation markers leaves commas and dashes, which go with it.
        if len(cites) > cited_before and not re.search(r"\w", inner):
            return ""
        # An exponent, as in 10^5, so that the number survives as written.
        if re.fullmatch(r"[−\-+]?\d+(\.\d+)?", inner.strip()):
            return "^" + inner.strip()
        return inner
    inner = render(child, cites, order)
    return f" {inner} " if name in BLOCKS else inner


def block_text(element: etree._Element, cites: list[str], order: text.ReferenceOrder) -> str:
    return text.tidy(render_child(element, cites, order))


def section_title(section: etree._Element, order: text.ReferenceOrder) -> str:
    """A section's title; for a box, its label and caption title, as in "Box 1 Lassa virus taxonomy"."""
    title = section.find("title")
    if local(section.tag) == "boxed-text":
        title = section.find("caption/title") if title is None else title
        label = text.tidy("".join(section.findtext("label") or ""))
        return " ".join(filter(None, [label, block_text(title, [], order) if title is not None else ""])) or "Box"
    return block_text(title, [], order) if title is not None else ""


def is_skipped(section: etree._Element, title: str) -> bool:
    return section.get("sec-type") in SKIPPED_SECTION_TYPES or (bool(title) and text.is_boilerplate(title))


def read_sections(container: etree._Element, parent: str, order: text.ReferenceOrder, skipped: set) -> list[Section]:
    """The sections of a <body>, <sec> or <boxed-text>, in reading order, with nested titles joined as "Parent > Child".

    Paragraphs that follow a subsection start a new block under the parent's heading, so the order of the text is kept. Skipped sections are added to skipped, so their figures can be left out too.
    """
    current = Section(heading=parent, paragraphs=[])
    sections = [current]
    for child in container:
        name = local(child.tag)
        if name in ("sec", "boxed-text"):
            title = section_title(child, order)
            if is_skipped(child, title):
                skipped.add(child)
                continue
            child_heading = " > ".join(filter(None, [parent, text.heading(title)]))
            sections.extend(read_sections(child, child_heading, order, skipped))
            current = Section(heading=parent, paragraphs=[])
            sections.append(current)
        elif name in PARAGRAPHS:
            cites: list[str] = []
            paragraph = block_text(child, cites, order)
            if paragraph:
                current["paragraphs"].append(Paragraph(text=paragraph, cites=cites))
    return [section for section in sections if section["paragraphs"]]


def read_captions(containers: list[etree._Element], order: text.ReferenceOrder, skipped: set) -> list[Caption]:
    """The captions of every figure and table, except those inside a skipped section such as the supplementary material."""
    captions = []
    for container in containers:
        for element in container.iter("fig", "table-wrap"):
            if any(ancestor in skipped for ancestor in element.iterancestors()):
                continue
            caption = element.find("caption")
            if caption is None:
                continue
            cites: list[str] = []
            caption_text = text.tidy(" ".join(render_child(part, cites, order) for part in caption))
            if caption_text:
                label = (
                    text.tidy("".join(element.find("label").itertext())) if element.find("label") is not None else ""
                )
                captions.append(Caption(label=label, text=caption_text, cites=cites))
    return captions


def read_abstract(meta: etree._Element, order: text.ReferenceOrder) -> str:
    """The main abstract, with the headings of a structured abstract kept as "Background: ...". Author summaries, teasers and graphical abstracts are left out."""
    abstracts = meta.findall("abstract")
    main = next(
        (a for a in abstracts if a.get("abstract-type") in (None, "primary")), abstracts[0] if abstracts else None
    )
    if main is None:
        return ""
    parts = []
    for child in main:
        if local(child.tag) == "sec":
            title = section_title(child, order)
            body = " ".join(block_text(p, [], order) for p in child if local(p.tag) in PARAGRAPHS)
            parts.append(f"{text.heading(title)}: {body}" if title else body)
        elif local(child.tag) in PARAGRAPHS:
            parts.append(block_text(child, [], order))
    return text.tidy(" ".join(parts))


def identifier(kind: str, value: str | None) -> str | None:
    """A DOI, PMID or PMCID in its standard form, from an attribute or text that may be a full URL."""
    if not value:
        return None
    if kind == "doi":
        return normalise_doi(value)
    if kind == "pmid":
        match = re.search(r"\d+", value)
        return match.group(0) if match else None
    match = re.search(r"(?:PMC)?(\d+)", value, re.IGNORECASE)
    return f"PMC{match.group(1)}" if match else None


def reference_ids(ref: etree._Element) -> dict[str, str | None]:
    """The DOI, PMID and PMCID of one reference, from <pub-id> or from the <ext-link>s Europe PMC adds."""
    ids: dict[str, str | None] = {"doi": None, "pmid": None, "pmcid": None}
    for element in ref.iter("pub-id", "ext-link"):
        kind = ID_KINDS.get((element.get("pub-id-type") or element.get("ext-link-type") or "").lower())
        if kind and ids[kind] is None:
            ids[kind] = identifier(kind, element.get(XLINK_HREF) or "".join(element.itertext()))
    return ids


def structured_reference(citation: etree._Element) -> str:
    """A printed form for an <element-citation>, which holds fields without punctuation: "Robertson G, et al. Title. Source. 2010;7:909–912"."""
    names = []
    for name in citation.iter("name", "string-name", "collab"):
        if local(name.tag) == "collab":
            names.append(" ".join("".join(name.itertext()).split()))
        else:
            names.append(" ".join(filter(None, [name.findtext("surname"), name.findtext("given-names")])))
    etal = len(names) > 3 or citation.find(".//etal") is not None
    authors = ", ".join(names[:3]) + (", et al" if etal else "")

    def field(name: str) -> str:
        found = citation.find(f".//{name}")
        return " ".join("".join(found.itertext()).split()) if found is not None else ""

    pages = "–".join(filter(None, [field("fpage"), field("lpage")]))
    issue = f"{field('year')};{field('volume')}:{pages}".strip(";:")
    return text.tidy(
        ". ".join(filter(None, [authors, field("article-title") or field("chapter-title"), field("source"), issue]))
    )


def printed_text(element: etree._Element) -> str:
    """The text of a citation as printed, without its label and without the identifier links Europe PMC adds after it."""
    parts = [element.text or ""]
    for child in element:
        if local(child.tag) not in ("pub-id", "ext-link", "label"):
            parts.append(printed_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def reference_text(ref: etree._Element) -> str:
    """How a reference reads in the paper's reference list. Its identifiers are kept as fields, not in the text."""
    citations = [element for element in ref.iter() if local(element.tag) in CITATION_ELEMENTS]
    citations.sort(key=lambda element: CITATION_ELEMENTS.index(local(element.tag)))
    if citations and local(citations[0].tag) != "mixed-citation":
        return structured_reference(citations[0])
    return text.tidy(printed_text(citations[0] if citations else ref))


def read_references(root: etree._Element) -> dict[str, Reference]:
    """The reference list as {id: Reference}. Reviews of the paper that some journals attach as sub-articles are left out."""
    references = {}
    for ref in root.xpath(".//ref[not(ancestor::sub-article)]"):
        if ref.get("id"):
            references[ref.get("id")] = Reference(text=reference_text(ref), **reference_ids(ref))
    return references


def read_year(meta: etree._Element) -> int | None:
    """The earliest publication year, ignoring the date the paper was released in PMC."""
    years = [
        int(date.findtext("year"))
        for date in meta.findall("pub-date")
        if (date.get("pub-type") or date.get("date-type")) != "pmc-release" and (date.findtext("year") or "").isdigit()
    ]
    return min(years) if years else None


def parse(xml: bytes) -> ParsedText:
    """A paper's metadata, abstract, sections, captions and references from its JATS XML."""
    root = etree.fromstring(xml, XML_PARSER)
    meta = root.find("front/article-meta")
    if meta is None:
        meta = etree.Element("article-meta")
    references = read_references(root)
    order = text.ReferenceOrder(list(references))
    body = root.find("body")
    skipped: set = set()
    sections = read_sections(body, "", order, skipped) if body is not None else []
    captions = read_captions([c for c in (body, root.find("floats-group")) if c is not None], order, skipped)
    # A citation keeps only ids that are in the reference list, each once, in the order they appear.
    for item in [p for s in sections for p in s["paragraphs"]] + captions:
        item["cites"] = [rid for rid in dict.fromkeys(item["cites"]) if rid in references]
    ids = {
        ID_KINDS.get((a.get("pub-id-type") or "").lower()): "".join(a.itertext()).strip()
        for a in meta.findall("article-id")
    }
    title = meta.find("title-group/article-title")
    return ParsedText(
        title=block_text(title, [], order) if title is not None else "",
        year=read_year(meta),
        doi=identifier("doi", ids.get("doi")),
        pmid=identifier("pmid", ids.get("pmid")),
        pmcid=identifier("pmcid", ids.get("pmcid")),
        abstract=read_abstract(meta, order),
        sections=sections,
        captions=captions,
        references=references,
    )
