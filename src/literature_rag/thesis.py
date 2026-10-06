r"""Read the thesis into paragraphs and captions, and turn the sentences that cite one paper of the library into test pairs.

The chapter files are the ones the last build read, in the order it read them, as thesis.fls lists them, minus config.THESIS_EXCLUDED. Each file is first read line by line for its structure, the way LaTeX reads it: a blank line ends a paragraph and a comment line does not, \chapter down to \subsubsection set the section path, a figure or table is one block whatever blank lines it holds, and \input reads the named file in place. pylatexenc then turns each block into plain text.

Line numbers are the file's own, so a chunk id such as "02_introduction:L23-25" names the lines to open.
"""

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from pylatexenc import latex2text, latexwalker, macrospec

from literature_rag import config, text
from literature_rag.records import Pair, ThesisChunk

# The sectioning commands, outermost first. A heading drops every heading at its own level or deeper from the path.
LEVELS = ("chapter", "section", "subsection", "subsubsection")
HEADING = re.compile(r"\s*\\(chapter|section|subsection|subsubsection)\b")
INPUT = re.compile(r"\\input\{([^}]+)\}")
BEGIN = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")
# Environments read as one block, blank lines and all. Each becomes a caption chunk if it has a caption, and nothing otherwise.
FLOATS = {"figure", "figure*", "table", "table*", "longtable", "sidewaysfigure", "sidewaystable"}
# Environments printed as written: a % in them is not a comment, and a blank line in them does not end the paragraph.
VERBATIM = {"verbatim", "lstlisting"}
VERBATIM_BLOCK = re.compile(r"\\begin\{(verbatim|lstlisting)\}(?:\[[^\]\n]*\])?\n?(.*?)\\end\{\1\}", re.DOTALL)
# A % that starts a comment, which is one without a backslash before it. An even run of backslashes is a line break, so the % after it still counts.
COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%.*")
# A thin space between groups of digits, as in "10\,000", taken out so that the number reads 10000.
THIN_SPACE_IN_NUMBER = re.compile(r"(?<=\d)(?:\\,|\\thinspace ?|\u2009|\u202f)(?=\d{3}(?!\d))")
# A cross-reference with the word that names it, as in "Figure~\ref{fig:map}". Only the compiled thesis knows the number, so the whole phrase is taken out, as a citation is; a bare "(Figure)" would be left otherwise.
REFERENCE = re.compile(
    r"(?:(?:Supplementary\s+)?(?:Figures?|Fig\.|Tables?|Sections?|Chapters?|Appendix|Appendices|Equations?|Eq\.)[\s~]*)?"
    r"\\(?:ref|autoref|pageref|eqref)\{[^}]*\}"
)
# The test of a layout switch such as "\ifdim\textwidth<140mm", which pylatexenc would print as "<140mm".
DIMENSION_TEST = re.compile(r"\\ifdim\s*\\[A-Za-z]+\s*[<>=]\s*-?[\d.]+\s*[a-z]{2}")
# Each citation is rendered as its keys between these two characters, from Unicode's private use area, which no text contains. Sentences are split with the markers still in place, which tells which sentence cites what.
CITE_OPEN, CITE_CLOSE = "\ue000", "\ue001"
CITATION = re.compile(rf"\s*{CITE_OPEN}([^{CITE_CLOSE}]*){CITE_CLOSE}")
# Where a verbatim environment waits while pylatexenc renders the rest of its block.
LISTING = re.compile("\ue002(\\d+)\ue003")
# A number as a test pair lists it: not part of a name such as "H1N1" or "COVID-19", and with its decimals, thousands separators and an exponent as in "10^5".
NUMBER = re.compile(r"(?<![A-Za-z\d])(?<![A-Za-z]-)\d+(?:[.,]\d+)*(?:\^-?\d+)?")
THOUSANDS_SEPARATOR = re.compile(r",(?=\d{3}(?!\d))")

CITE_COMMANDS = ("cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear", "citeyearpar")
CAPTION_COMMANDS = ("caption", "captionof")


def cite_marker(node: latexwalker.LatexMacroNode, l2tobj: latex2text.LatexNodes2Text) -> str:
    r"""A citation command as its keys between the markers. The keys are its last argument, so \citep[e.g.][p. 3]{a,b} gives "a,b"."""
    return CITE_OPEN + node.nodeargd.argnlist[-1].latex_verbatim().strip("{}") + CITE_CLOSE


def typewriter(node: latexwalker.LatexMacroNode, l2tobj: latex2text.LatexNodes2Text) -> str:
    r"""\texttt as printed: a typewriter font sets "--" as two hyphens, not as the dash pylatexenc makes of it, so "--skip_qc" stays an option name."""
    rendered = l2tobj.nodelist_to_text([node.nodeargd.argnlist[-1]])
    return rendered.replace("—", "---").replace("–", "--")


def walker_context() -> macrospec.LatexContextDb:
    """pylatexenc's parsing rules, with the arguments of the macros the thesis uses that its defaults lack or get wrong."""
    context = latexwalker.get_default_latex_context_db()
    context.add_context_category(
        "thesis",
        prepend=True,
        macros=[
            macrospec.MacroSpec("caption", "*[{"),
            macrospec.MacroSpec("captionof", "*{[{"),
            macrospec.MacroSpec("href", "{{"),
            macrospec.MacroSpec("setlength", "{{"),
            macrospec.MacroSpec("ding", "{"),
            # The thesis's own layout macros, from its preamble.
            macrospec.MacroSpec("chapterpubinfo", "{{{{{"),
            macrospec.MacroSpec("graphicspath", "{"),
            macrospec.MacroSpec("nextchaptermotif", "{"),
        ],
    )
    return context


def text_context() -> macrospec.LatexContextDb:
    """pylatexenc's rendering rules, with citations as markers, the long form of a caption, the text of every font command, and no output for layout."""
    context = latex2text.get_default_latex_context_db()
    silent = ("setlength", "ding", "chapterpubinfo", "graphicspath", "nextchaptermotif")
    # Font commands whose text pylatexenc 2.11 drops, as in "\textsf{fastp}".
    fonts = ("textsf", "textmd", "textup", "mbox")
    context.add_context_category(
        "thesis",
        prepend=True,
        macros=[
            *(latex2text.MacroTextSpec(name, simplify_repl=cite_marker) for name in CITE_COMMANDS),
            latex2text.MacroTextSpec("caption", "%(3)s"),
            latex2text.MacroTextSpec("captionof", "%(4)s"),
            latex2text.MacroTextSpec("href", "%(2)s"),
            latex2text.MacroTextSpec("url", "%(1)s"),
            latex2text.MacroTextSpec("texttt", simplify_repl=typewriter),
            *(latex2text.MacroTextSpec(name, "%(1)s") for name in fonts),
            *(latex2text.MacroTextSpec(name, "") for name in silent),
        ],
    )
    return context


WALKER_CONTEXT = walker_context()
TO_TEXT = latex2text.LatexNodes2Text(latex_context=text_context())


class Block(NamedTuple):
    """A run of lines that becomes at most one chunk: a paragraph, or a figure or table."""

    file: str
    start: int
    end: int
    section: str
    # The lines as they stand in the file, comments included, for the hash.
    raw: str
    # The same lines as LaTeX reads them, without comments.
    latex: str
    is_float: bool


class Sentence(NamedTuple):
    """One sentence of a thesis paragraph, with the keys it cites."""

    chunk_id: str
    position: int
    text: str
    keys: list[str]


def chapter_files(fls: str) -> list[str]:
    """The chapter .tex files a build read, in the order it read them, relative to the thesis repository and without config.THESIS_EXCLUDED."""
    names = []
    for line in fls.splitlines():
        if line.startswith("INPUT "):
            name = line.removeprefix("INPUT ").strip().removeprefix("./")
            if name.startswith("chapters/") and name.endswith(".tex") and name not in config.THESIS_EXCLUDED:
                names.append(name)
    return list(dict.fromkeys(names))


def strip_comment(line: str) -> str:
    return COMMENT.sub(r"\1", line)


def prepare(latex: str) -> str:
    """LaTeX without what pylatexenc would print wrongly: cross-references, thin spaces inside numbers, and layout tests."""
    return REFERENCE.sub("", DIMENSION_TEST.sub("", THIN_SPACE_IN_NUMBER.sub("", latex)))


def parse(latex: str) -> list[latexwalker.LatexNode]:
    return latexwalker.LatexWalker(latex, latex_context=WALKER_CONTEXT, tolerant_parsing=True).get_latex_nodes()[0]


def walk(nodes: Iterable[latexwalker.LatexNode | None]) -> Iterator[latexwalker.LatexNode]:
    """Every node of a parsed tree in document order, the arguments of macros and the contents of groups and environments included."""
    for node in nodes:
        if node is None:
            continue
        yield node
        yield from walk(getattr(getattr(node, "nodeargd", None), "argnlist", None) or [])
        yield from walk(getattr(node, "nodelist", None) or [])


def heading_title(latex: str) -> str:
    r"""The title a sectioning command sets, as plain text: "\section*{A \textit{b}}" gives "A b"."""
    for node in parse(prepare(latex)):
        if node.isNodeType(latexwalker.LatexMacroNode) and node.macroname in LEVELS:
            # A title that opens with a cross-reference, as in "\autoref{ch:x}: a database", keeps the colon after it.
            return plain(TO_TEXT.nodelist_to_text([node.nodeargd.argnlist[-1]])).strip(" :;,.")
    return ""


def read_blocks(sources: dict[str, str]) -> list[Block]:
    r"""Every paragraph and float of the thesis in reading order, with its line range and section path.

    sources maps each chapter file to its text, in the order the build read them. A file that another one \input-s is read at that point, and only there, so the section path runs on through it as it does in LaTeX.
    """
    blocks: list[Block] = []
    headings: list[tuple[int, str]] = []
    done: set[str] = set()

    def read(name: str) -> None:
        done.add(name)
        lines = sources[name].split("\n")
        # The (line number, LaTeX) pairs of the block being gathered.
        pending: list[tuple[int, str]] = []
        float_env = verbatim_env = None
        depth = 0

        def close(is_float: bool = False) -> None:
            numbers = [number for number, latex in pending if latex.strip()]
            if numbers:
                start, end = numbers[0], numbers[-1]
                section = " > ".join(title for _, title in headings)
                raw = "\n".join(lines[start - 1 : end])
                blocks.append(Block(name, start, end, section, raw, "\n".join(latex for _, latex in pending), is_float))
            pending.clear()

        for number, line in enumerate(lines, start=1):
            if verbatim_env:
                pending.append((number, line))
                if f"\\end{{{verbatim_env}}}" in line:
                    verbatim_env = None
                continue
            latex = strip_comment(line)
            if float_env:
                pending.append((number, latex))
                depth += latex.count(f"\\begin{{{float_env}}}") - latex.count(f"\\end{{{float_env}}}")
                if depth <= 0:
                    close(is_float=True)
                    float_env = None
                continue
            if not line.strip():
                close()
                continue
            if not latex.strip():
                # A comment line, which LaTeX reads as nothing: the paragraph goes on.
                continue
            begin = BEGIN.search(latex)
            if begin and begin[1] in FLOATS:
                close()
                pending.append((number, latex))
                depth = latex.count(f"\\begin{{{begin[1]}}}") - latex.count(f"\\end{{{begin[1]}}}")
                if depth > 0:
                    float_env = begin[1]
                else:
                    close(is_float=True)
                continue
            if begin and begin[1] in VERBATIM:
                pending.append((number, line))
                if f"\\end{{{begin[1]}}}" not in line:
                    verbatim_env = begin[1]
                continue
            heading = HEADING.match(latex)
            if heading:
                close()
                level = LEVELS.index(heading[1])
                headings[:] = [(outer, title) for outer, title in headings if outer < level]
                headings.append((level, heading_title(latex)))
                continue
            included = INPUT.search(latex)
            if included:
                close()
                child = included[1].strip().removeprefix("./")
                child = child if child.endswith(".tex") else f"{child}.tex"
                if child in sources and child not in done:
                    read(child)
                continue
            pending.append((number, latex))
        close()

    for name in sources:
        if name not in done:
            read(name)
    return blocks


def render(latex: str) -> tuple[str, list[str]]:
    r"""A block as plain text with its citation markers, and the text of each caption in it, likewise.

    Verbatim environments are held out of pylatexenc, which would read a % in them as a comment and a \citep as a citation, and put back as written.
    """
    listings: list[str] = []

    def hold(match: re.Match[str]) -> str:
        listings.append(match[2])
        return f"\ue002{len(listings) - 1}\ue003"

    nodes = parse(prepare(VERBATIM_BLOCK.sub(hold, latex)))
    captions = [
        TO_TEXT.nodelist_to_text([node.nodeargd.argnlist[-1]])
        for node in walk(nodes)
        if node.isNodeType(latexwalker.LatexMacroNode) and node.macroname in CAPTION_COMMANDS
    ]
    rendered = LISTING.sub(lambda match: listings[int(match[1])], TO_TEXT.nodelist_to_text(nodes))
    return rendered, captions


def keys_in(rendered: str) -> list[str]:
    """The citation keys in rendered text, in order of first appearance."""
    keys = (key.strip() for group in CITATION.findall(rendered) for key in group.split(","))
    return list(dict.fromkeys(key for key in keys if key))


def plain(rendered: str) -> str:
    """Rendered text without its citation markers, tidied the way the paper parsers tidy theirs."""
    return text.tidy(CITATION.sub("", rendered))


def numbers(sentence: str) -> list[str]:
    """The numbers in a sentence without thousands separators: "1,234 of 2,000" gives 1234 and 2000, and "sites 1,2" gives 1 and 2."""
    found = []
    for match in NUMBER.finditer(sentence):
        found.extend(part for part in THOUSANDS_SEPARATOR.sub("", match[0]).split(",") if part)
    return list(dict.fromkeys(found))


def parse_thesis(sources: dict[str, str]) -> tuple[list[ThesisChunk], list[Sentence]]:
    """The thesis's paragraphs and captions as chunks, and every sentence of its paragraphs with the keys it cites.

    A caption chunk carries the keys of everything in its figure or table, a table's rows included, so that a search for a key finds the table that cites it.
    """
    chunks: list[ThesisChunk] = []
    sentences: list[Sentence] = []
    for block in read_blocks(sources):
        rendered, captions = render(block.latex)
        if captions:
            kind, body = "caption", " ".join(captions)
        elif block.is_float:
            # A figure without a caption has no text to find it by.
            continue
        else:
            kind, body = "paragraph", rendered
        if not plain(body):
            # Layout only, such as \label, \clearpage or an empty \begin{abstract}.
            continue
        chunk = ThesisChunk(
            chunk_id=f"{Path(block.file).stem}:L{block.start}-{block.end}",
            file=block.file,
            line_start=block.start,
            line_end=block.end,
            section=block.section,
            kind=kind,
            text=plain(body),
            keys=keys_in(rendered),
            hash=hashlib.sha256(block.raw.encode("utf-8")).hexdigest()[:16],
        )
        chunks.append(chunk)
        if kind == "paragraph":
            for position, sentence in enumerate(text.sentences(" ".join(rendered.split()))):
                sentences.append(Sentence(chunk["chunk_id"], position, plain(sentence), keys_in(sentence)))
    return chunks, sentences


def make_pairs(sentences: list[Sentence], library_keys: dict[str, str]) -> list[Pair]:
    """A test pair for each sentence that cites exactly one paper, when that paper is citable and in the library.

    library_keys maps the key of each such paper, lowercased, to the key as the library spells it: BibTeX keys are case-insensitive, and the thesis and the bibliography do not always agree on case. A sentence that also cites a paper outside the library is left out, since that paper may be the one that supports it.
    """
    pairs: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    for sentence in sentences:
        if len(sentence.keys) != 1 or sentence.keys[0].lower() not in library_keys:
            continue
        key = library_keys[sentence.keys[0].lower()]
        if (sentence.text, key) in seen:
            continue
        seen.add((sentence.text, key))
        pairs.append(
            Pair(
                pair_id=f"{sentence.chunk_id}/{sentence.position}",
                chunk_id=sentence.chunk_id,
                query=sentence.text,
                key=key,
                numbers=numbers(sentence.text),
            )
        )
    return pairs


def read_thesis() -> tuple[list[ThesisChunk], list[Sentence]]:
    """Parse the chapter files that the last build of the thesis read."""
    names = chapter_files(config.THESIS_FLS.read_text(encoding="utf-8", errors="replace"))
    return parse_thesis({name: (config.THESIS_REPO / name).read_text(encoding="utf-8") for name in names})
