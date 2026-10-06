"""GROBID's reading of a PDF (TEI XML): request it once, and read it into the same sections, captions and references as the JATS.

GROBID is the route for papers without open-access XML in Europe PMC. It runs locally in Docker (`docker start grobid`), and its TEI marks up sections, paragraphs, figure captions and the reference list, with each in-text citation pointing at its reference.
"""

import asyncio
import re
import time
import unicodedata
from pathlib import Path

import httpx
from lxml import etree

from literature_rag import config, text
from literature_rag.bib import normalise_doi
from literature_rag.records import Caption, Paragraph, ParsedText, Reference, Section

# GROBID answers 503 when all its workers are busy; the request is then simply sent again a little later.
BUSY = 503
RETRIES = 5


class GrobidUnavailable(Exception):
    """GROBID did not answer at all, so nothing about the PDF is known yet."""


def cache_paths(pdf_path: Path) -> tuple[Path, Path]:
    """The cached TEI for a PDF, and the marker file that records that GROBID could not read it."""
    return config.TEI_DIR / f"{pdf_path.stem}.tei.xml", config.TEI_DIR / f"{pdf_path.stem}.failed"


def is_alive() -> bool:
    try:
        return httpx.get(f"{config.GROBID_URL}/api/isalive", timeout=5).text.strip() == "true"
    except httpx.HTTPError:
        return False


async def process(client: httpx.AsyncClient, pdf_path: Path) -> Path | None:
    """Send a PDF to GROBID unless its TEI is already in data/raw/tei/, and return the TEI's path, or None if GROBID could not read the PDF.

    A PDF GROBID rejects or times out on is cached as a marker file holding the reason, so it is not sent again; delete the marker to retry. If GROBID cannot be reached at all, GrobidUnavailable is raised and nothing is cached.
    """
    tei_path, failed_path = cache_paths(pdf_path)
    if tei_path.is_file():
        return tei_path
    if failed_path.is_file():
        return None
    pdf = pdf_path.read_bytes()
    for attempt in range(RETRIES):
        try:
            response = await client.post(
                f"{config.GROBID_URL}/api/processFulltextDocument",
                files={"input": (pdf_path.name, pdf, "application/pdf")},
                data=config.GROBID_PARAMS,
            )
        except httpx.TimeoutException:
            config.TEI_DIR.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(f"timed out after {config.GROBID_TIMEOUT} s\n")
            return None
        except httpx.TransportError as error:
            raise GrobidUnavailable(f"{config.GROBID_URL}: {error!r}") from error
        if response.status_code != BUSY:
            break
        await asyncio.sleep(2 * (attempt + 1))
    config.TEI_DIR.mkdir(parents=True, exist_ok=True)
    if response.status_code != 200:
        failed_path.write_text(f"HTTP {response.status_code}: {response.text[:500]}\n")
        return None
    tei_path.write_bytes(response.content)
    return tei_path


async def process_all(pdf_paths: list[Path]) -> dict[Path, Path | None | Exception]:
    """Process several PDFs, a few at a time. Each PDF maps to its TEI, to None when GROBID could not read it, or to the error that stopped it.

    A PDF that is not cached yet prints a line when it is done, since a batch can run for most of an hour.
    """
    semaphore = asyncio.Semaphore(config.GROBID_CONCURRENCY)
    uncached = [path for path in pdf_paths if not any(cached.is_file() for cached in cache_paths(path))]
    done = 0

    async def one(client: httpx.AsyncClient, pdf_path: Path) -> Path | None:
        nonlocal done
        async with semaphore:
            start = time.perf_counter()
            result = await process(client, pdf_path)
        if pdf_path in uncached:
            done += 1
            outcome = "ok" if result else "failed"
            print(
                f"  {done}/{len(uncached)} {outcome} in {time.perf_counter() - start:.0f} s: {pdf_path.stem}",
                flush=True,
            )
        return result

    async with httpx.AsyncClient(timeout=config.GROBID_TIMEOUT) as client:
        results = await asyncio.gather(*(one(client, path) for path in pdf_paths), return_exceptions=True)
    return dict(zip(pdf_paths, results))


TEI = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XML_PARSER = etree.XMLParser(recover=True, no_network=True, huge_tree=True)
# GROBID writes every section as a flat <div>. Methods and results usually have subsections, so one of these headings becomes the parent of the unnumbered headings that follow it. Other top-level headings end that run without becoming parents, since an introduction or a review's sections rarely nest.
PARENT_HEADINGS = re.compile(
    r"(results?|methods?|materials and methods|methods and materials|patients and methods|subjects and methods|results and discussion)",
    re.IGNORECASE,
)
TOP_LEVEL = re.compile(
    r"(?:(?:introduction|background|discussion|conclusions?|concluding remarks|summary|limitations|outlook|perspectives?)(?:\s*(?:and|&|,)\s*)?)+",
    re.IGNORECASE,
)
# A figure or table label at the start of a caption: "Fig. 2", "Figure S1", "Table 1", "Supplementary Fig. 3".
FIGURE_LABEL = re.compile(
    r"^((?:supplementary|extended data)\s+)?(fig(?:ure)?s?\.?|table|box)\s*[A-Z]?\d+[A-Za-z]?", re.IGNORECASE
)
# Preprints print a line number in the margin, which GROBID sometimes reads into a heading: "Competing interests 313".
LINE_NUMBER = re.compile(r"\s\d{3,}$")
# Page furniture that GROBID sometimes takes for a heading: running heads with an ISSN, a download stamp, a licence line, an email address.
PAGE_FURNITURE = re.compile(
    r"@|https?:|www\.|downloaded from|\bissn\b|\blicen[cs]e\b|copyright|©|\bdoi:", re.IGNORECASE
)
# The stamp a publisher prints down the margin of a downloaded PDF, which GROBID reads into the nearest text: "Downloaded from academic.oup.com/... by KU Leuven Libraries user on 16 September 2026".
DOWNLOAD_STAMP = re.compile(
    r"Downloaded from \S+(?:\s+(?:(?:by|at)\s+.{0,80}?\s+)?on\b)?(?:\s+(?:[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]+ \d{4}))?"
)
# Older PDFs set an accent as a separate character after its letter, which GROBID keeps apart: "Gu ¨nther" for "Günther".
DETACHED_ACCENT = re.compile(r"([A-Za-zı]) ?([¨´`ˆ˜]) ?(?=\w)")
COMBINING = {"¨": "\u0308", "´": "\u0301", "`": "\u0300", "ˆ": "\u0302", "˜": "\u0303"}
# Too short to be a caption: what is left when GROBID reads a stray line number as a figure.
MIN_CAPTION_WORDS = 5


def local(tag: object) -> str:
    """An element's name without its namespace, or "" for a comment or processing instruction."""
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def clean(raw: str) -> str:
    """Text as GROBID read it, without download stamps and with detached accents put back on their letters."""
    raw = DOWNLOAD_STAMP.sub("", raw)
    raw = DETACHED_ACCENT.sub(lambda m: m.group(1).replace("ı", "i") + COMBINING[m.group(2)], raw)
    return text.tidy(unicodedata.normalize("NFC", raw))


def citation_ids(element: etree._Element) -> list[str] | None:
    """The reference ids of a TEI citation link, <ref type="bibr" target="#b12">, or None for any other element. A link GROBID could not resolve has no target, and so no ids."""
    if local(element.tag) == "ref" and element.get("type") == "bibr":
        return [target.lstrip("#") for target in (element.get("target") or "").split()]
    return None


def render(element: etree._Element, cites: list[str], order: text.ReferenceOrder) -> str:
    """The running text inside an element, without numeric citation markers, with the cited reference ids added to cites."""

    def child_text(child: etree._Element) -> str:
        name = local(child.tag)
        if (
            not name
            or name in ("figure", "table", "note", "graphic", "formula")
            or (name == "ref" and child.get("type") == "foot")
        ):
            return ""
        inner = render(child, cites, order)
        return f" {inner} " if name in ("p", "head", "item", "list", "lb") else inner

    return text.render_with_citations(element, cites, order, citation_ids, child_text)


def head_title(div: etree._Element) -> str:
    """A <div>'s heading, or "" when it has none or the heading is only a line number or page furniture."""
    head = div.find("tei:head", TEI)
    if head is None:
        return ""
    title = LINE_NUMBER.sub("", text.tidy("".join(head.itertext())))
    if not re.search(r"[A-Za-z]", title) or PAGE_FURNITURE.search(title):
        return ""
    return text.heading(title)


def read_sections(divs: list[etree._Element], order: text.ReferenceOrder) -> list[Section]:
    """Sections from GROBID's flat <div>s, with the nesting restored where the paper shows it.

    A heading numbered "2.1" sits under the heading numbered "2". Without numbers, a heading sits under the last methods or results heading ("Methods > Sequencing"), or under a heading that had no text of its own. A <div> without a heading continues the section before it, which is how GROBID splits a section around a figure. Boilerplate sections are left out, with any headless <div>s that continue them.
    """
    sections = []
    numbered: dict[str, str] = {}
    parent = ""
    current = ""
    skipping = False
    for div in divs:
        title = head_title(div)
        number = ""
        if title:
            skipping = text.is_boilerplate(title)
            number = (div.find("tei:head", TEI).get("n") or "").rstrip(".")
            if number:
                current = " > ".join(filter(None, [numbered.get(number.rpartition(".")[0], ""), title]))
                numbered[number] = current
            elif PARENT_HEADINGS.fullmatch(title):
                parent = current = title
            elif TOP_LEVEL.fullmatch(title):
                parent, current = "", title
            else:
                current = " > ".join(filter(None, [parent, title]))
        if skipping:
            continue
        paragraphs = []
        for p in div.findall("tei:p", TEI):
            cites: list[str] = []
            paragraph = clean(render(p, cites, order))
            if paragraph:
                paragraphs.append(Paragraph(text=paragraph, cites=cites))
        if not paragraphs:
            # An unnumbered heading with no text of its own is the parent of the headings that follow.
            if title and not number and not parent:
                parent = title
            continue
        if sections and sections[-1]["heading"] == current:
            sections[-1]["paragraphs"].extend(paragraphs)
        else:
            sections.append(Section(heading=current, paragraphs=paragraphs))
    return sections


def read_captions(figures: list[etree._Element], order: text.ReferenceOrder) -> list[Caption]:
    """Figure and table captions. GROBID often repeats the label at the start of the description, which is removed there.

    A "figure" without a label is nearly always something GROBID misplaced, such as the funding note on a title page, so it is left out.
    """
    captions = []
    for figure in figures:
        description = figure.find("tei:figDesc", TEI)
        if description is None:
            continue
        cites: list[str] = []
        caption = clean(render(description, cites, order))
        head = (
            text.tidy("".join(figure.find("tei:head", TEI).itertext()))
            if figure.find("tei:head", TEI) is not None
            else ""
        )
        number = (figure.findtext("tei:label", namespaces=TEI) or "").strip()
        match = FIGURE_LABEL.match(head)
        if match:
            label = match.group(0)
        elif number:
            label = f"{'Table' if figure.get('type') == 'table' else 'Figure'} {number}"
        else:
            continue
        repeated = FIGURE_LABEL.match(caption)
        if repeated:
            caption = caption[repeated.end() :].lstrip(" .:|")
        if len(caption.split()) >= MIN_CAPTION_WORDS:
            captions.append(Caption(label=label, text=caption, cites=cites))
    return captions


def tei_ids(element: etree._Element) -> dict[str, str | None]:
    """The DOI, PMID and PMCID in a <biblStruct>, from its <idno>s."""
    ids: dict[str, str | None] = {"doi": None, "pmid": None, "pmcid": None}
    for idno in element.iterfind(".//tei:idno", TEI):
        kind = (idno.get("type") or "").lower()
        value = "".join(idno.itertext()).strip()
        if kind == "doi" and ids["doi"] is None:
            ids["doi"] = normalise_doi(value)
        elif kind == "pmid" and ids["pmid"] is None and value.isdigit():
            ids["pmid"] = value
        elif kind == "pmcid" and ids["pmcid"] is None:
            match = re.search(r"\d+", value)
            ids["pmcid"] = f"PMC{match.group(0)}" if match else None
    return ids


def structured_reference(bibl: etree._Element) -> str:
    """A printed form for a reference GROBID kept no raw text for: "Wilson MR, et al. Title. Journal. 2019;380:2327–2340"."""
    names = []
    for person in bibl.iterfind(".//tei:author/tei:persName", TEI):
        initials = "".join(name[:1] for name in person.xpath("tei:forename/text()", namespaces=TEI))
        names.append(" ".join(filter(None, [person.findtext("tei:surname", namespaces=TEI), initials])))
    authors = ", ".join(names[:3]) + (", et al" if len(names) > 3 else "")
    analytic = bibl.findtext("tei:analytic/tei:title", namespaces=TEI) or ""
    journal = bibl.findtext("tei:monogr/tei:title", namespaces=TEI) or ""
    date = bibl.find(".//tei:imprint/tei:date", TEI)
    year = (date.get("when") or "")[:4] if date is not None else ""
    volume = bibl.findtext(".//tei:biblScope[@unit='volume']", namespaces=TEI) or ""
    page = bibl.find(".//tei:biblScope[@unit='page']", TEI)
    pages = "–".join(filter(None, [page.get("from"), page.get("to")])) if page is not None else ""
    issue = f"{year};{volume}:{pages}".strip(";:")
    return text.tidy(". ".join(filter(None, [authors, analytic, journal, issue])))


def read_references(root: etree._Element) -> dict[str, Reference]:
    """The reference list as {id: Reference}, with the reference as printed when GROBID kept it (includeRawCitations)."""
    references = {}
    for bibl in root.iterfind("tei:text/tei:back//tei:listBibl/tei:biblStruct", TEI):
        rid = bibl.get(XML_ID)
        if not rid:
            continue
        raw = bibl.find("tei:note[@type='raw_reference']", TEI)
        printed = clean("".join(raw.itertext())) if raw is not None else structured_reference(bibl)
        references[rid] = Reference(text=printed, **tei_ids(bibl))
    return references


def read_abstract(header: etree._Element, order: text.ReferenceOrder) -> str:
    parts = []
    for div in header.iterfind("tei:profileDesc/tei:abstract//tei:div", TEI):
        title = head_title(div)
        body = " ".join(clean(render(p, [], order)) for p in div.findall("tei:p", TEI))
        parts.append(f"{title}: {body}" if title and body else body)
    return text.tidy(" ".join(parts))


def parse(tei: bytes) -> ParsedText:
    """A paper's metadata, abstract, sections, captions and references from GROBID's TEI.

    The body comes from <body> and from the annex in <back>, where GROBID puts appendices and text it could not place. Acknowledgements, funding, contributions and the like sit elsewhere in <back> and are not read.
    """
    root = etree.fromstring(tei, XML_PARSER)
    header = root.find("tei:teiHeader", TEI)
    source = header.find("tei:fileDesc/tei:sourceDesc/tei:biblStruct", TEI) if header is not None else None
    references = read_references(root)
    order = text.ReferenceOrder(list(references))
    body = root.find("tei:text/tei:body", TEI)
    annex = root.findall("tei:text/tei:back/tei:div[@type='annex']", TEI)
    divs = (body.findall("tei:div", TEI) if body is not None else []) + [
        div for part in annex for div in part.findall("tei:div", TEI)
    ]
    sections = read_sections(divs, order)
    figures = (body.findall(".//tei:figure", TEI) if body is not None else []) + [
        figure for part in annex for figure in part.iterfind(".//tei:figure", TEI)
    ]
    captions = read_captions(figures, order)
    # A citation keeps only ids that are in the reference list, each once, in the order they appear.
    for item in [p for s in sections for p in s["paragraphs"]] + captions:
        item["cites"] = [rid for rid in dict.fromkeys(item["cites"]) if rid in references]
    title = header.findtext("tei:fileDesc/tei:titleStmt/tei:title", namespaces=TEI) if header is not None else ""
    date = source.find(".//tei:imprint/tei:date[@type='published']", TEI) if source is not None else None
    year = (date.get("when") or "")[:4] if date is not None else ""
    return ParsedText(
        title=text.tidy(title or ""),
        year=int(year) if year.isdigit() else None,
        abstract=read_abstract(header, order) if header is not None else "",
        sections=sections,
        captions=captions,
        references=references,
        **(tei_ids(source) if source is not None else {"doi": None, "pmid": None, "pmcid": None}),
    )
