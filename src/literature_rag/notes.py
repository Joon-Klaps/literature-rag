"""Read the researcher's notes, an Obsidian vault in the project's data/notes, into sections of paragraphs, each paragraph with its lines in the file.

The notes are the Markdown files in the folder and below it, following symbolic links, without hidden folders (.obsidian, .git, .trash), folders that are git repositories of their own, and the files config.NOTES_EXCLUDED names. A note's name is its path in the folder without .md, as Obsidian names it, so a daily note is named by its date.

Each note is read line by line, much as Obsidian reads it. A fenced code block is code whatever it holds: a "# comment" in it is not a heading, a blank line in it ends nothing, and %% in it is not a comment. Outside code, %% opens and closes a comment, which may run over many lines and hold a code block of its own. The frontmatter is not text. A heading starts a section, whose path runs from # down to ######, and a blank line or a horizontal rule ends a paragraph. Obsidian's own syntax is then taken out of each line: embedded images and files, block ids, callout markers and the brackets of links, keeping a link's text. chunks.py merges the paragraphs into passages.

Line numbers are the file's own, so a passage's line range names the lines to open.
"""

import fnmatch
import os
import re
from pathlib import Path
from typing import NamedTuple

from literature_rag import config

# The fence that opens or closes a code block: three or more backticks or tildes.
FENCE = re.compile(r"\s*(`{3,}|~{3,})")
# A heading from # to ######, without the closing hashes some writers add. A hash with no space after it is a tag, as in "#phd-thesis".
HEADING = re.compile(r"\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*")
# A horizontal rule, as in "---" or "* * *", which has no text and ends a paragraph as a blank line does.
RULE = re.compile(r"\s{0,3}([-*_])(?:\s*\1){2,}\s*")
# An embedded file, as in "![[photo.png|800]]", or an image in Markdown's own syntax, as in "![screenshot](images/a.png)". Neither has text to find.
EMBED = re.compile(r"!\[\[[^\]]*\]\]|!\[[^\]]*\]\([^)]*\)")
# A link to another note or file, as in "[[2025-08-26]]", "[[Note#Heading]]" or "[[slides.pdf#page=11|slides, p.11]]", which reads as its alias, or else as the name of what it links to.
WIKILINK = re.compile(r"\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
# A Markdown link, as in "[BV-BRC](https://www.bv-brc.org/)", which reads as its text.
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# A highlight, as in "==dry season==", which reads as its text.
HIGHLIGHT = re.compile(r"==([^=\n]+)==")
# The type of a callout, as in "> [!important]" or "> [!PDF|255, 208, 0]", which Obsidian shows as the colour of a box, not as text.
CALLOUT = re.compile(r"\[![^\]]*\][+-]?\s*")
# A block id at the end of a line or on a line of its own, as in "Contigs ^a1b2c3", which other notes link to and Obsidian does not show.
BLOCK_ID = re.compile(r"(?:^|\s)\^[A-Za-z0-9-]+$")
BOILERPLATE_LINE = re.compile("|".join(f"(?:{pattern})" for pattern in config.NOTES_BOILERPLATE_LINES))


class Line(NamedTuple):
    """A line of a note without its comments, and whether it belongs to a fenced code block, the fences included."""

    text: str
    code: bool


class Block(NamedTuple):
    """A paragraph of a note: its first and last line in the file, from 1, and its text."""

    start: int
    end: int
    text: str


class Section(NamedTuple):
    """The paragraphs under one heading, up to the next heading of any level."""

    # The headings above, joined as in "Day 1 > Panel discussion"; empty before the first heading.
    heading: str
    blocks: list[Block]


class Note(NamedTuple):
    """One note as read: its name, its file relative to the project, its text as it stands (for the hashes of its passages), and its sections in reading order."""

    name: str
    file: str
    source: str
    sections: list[Section]


def excluded(relative: str) -> bool:
    """Whether a file, by its path in the notes folder, is one that config.NOTES_EXCLUDED leaves out."""
    return any(fnmatch.fnmatch(relative, pattern) for pattern in config.NOTES_EXCLUDED)


def note_files(folder: Path) -> list[Path]:
    """Every note in the folder, sorted by path: the Markdown files in it and below it, following symbolic links, without hidden folders and files, folders that are git repositories of their own, such as course material cloned into the vault, and the files config.NOTES_EXCLUDED names. A link that leads nowhere is skipped."""
    found = []
    seen: set[str] = set()
    for root, folders, files in os.walk(folder, followlinks=True):
        here = Path(root)
        # A link back up the tree would make the walk go round for ever, so each real folder is read once.
        seen.add(os.path.realpath(here))
        folders[:] = [
            name
            for name in folders
            if not name.startswith(".")
            and not (here / name / ".git").exists()
            and os.path.realpath(here / name) not in seen
        ]
        for name in files:
            path = here / name
            if (
                name.endswith(".md")
                and not name.startswith(".")
                and path.is_file()
                and not excluded(path.relative_to(folder).as_posix())
            ):
                found.append(path)
    return sorted(found)


def closes(fence: str, line: str) -> bool:
    """Whether a line closes the code block that this fence opened: a line of the same character, at least as long, and nothing else."""
    stripped = line.strip()
    return stripped.startswith(fence) and not stripped.strip(fence[0])


def read_lines(source: str) -> list[Line]:
    """The note's lines, as many as the file has, each without its %% comments and marked when it belongs to a fenced code block.

    A comment may hold a code block, as the Kanban plugin keeps its settings, but %% inside a code block is code, as in the shell's ${name%%_*} or a comment in a mermaid diagram. A comment left open runs to the end of the note, as in Obsidian.
    """
    lines = []
    in_comment = False
    fence = ""
    for line in source.split("\n"):
        if fence:
            lines.append(Line(line, True))
            if closes(fence, line):
                fence = ""
            continue
        opening = None if in_comment else FENCE.match(line)
        if opening:
            fence = opening[1]
            lines.append(Line(line, True))
            continue
        # Split at each %%, the pieces alternate between text and comment, starting in whichever the line began in.
        pieces = line.split("%%")
        lines.append(Line("".join(piece for n, piece in enumerate(pieces) if (n % 2 == 0) != in_comment), False))
        if len(pieces) % 2 == 0:
            in_comment = not in_comment
    return lines


def frontmatter_end(lines: list[Line]) -> int:
    """The number of the line that closes the note's frontmatter, the YAML between a first line of --- and the next, or 0 when the note has none."""
    if lines and lines[0].text.strip() == "---":
        for number, line in enumerate(lines[1:], start=2):
            if line.text.strip() == "---":
                return number
    return 0


def clean(line: str) -> str:
    """A line of a note as text: without embedded files, block ids and callout markers, with links and highlights as their text, and empty when nothing but a quote marker or a plugin's boilerplate is left."""
    line = EMBED.sub("", line)
    line = WIKILINK.sub(lambda match: match[2] or match[1], line)
    line = LINK.sub(r"\1", line)
    line = HIGHLIGHT.sub(r"\1", line)
    line = BLOCK_ID.sub("", CALLOUT.sub("", line)).rstrip()
    content = line.strip(" >")
    return "" if not content or BOILERPLATE_LINE.fullmatch(content) else line


def parse_note(source: str) -> list[Section]:
    """A note's sections in reading order, each with its paragraphs and their line ranges, leaving out the sections with no text."""
    lines = read_lines(source)
    first = frontmatter_end(lines)
    sections = [Section("", [])]
    headings: list[tuple[int, str]] = []
    # The (line number, text) pairs of the paragraph being gathered.
    pending: list[tuple[int, str]] = []

    def close() -> None:
        kept = [(number, text) for number, text in pending if text.strip()]
        if kept:
            sections[-1].blocks.append(Block(kept[0][0], kept[-1][0], "\n".join(text for _, text in kept)))
        pending.clear()

    for number, (text, code) in enumerate(lines, start=1):
        if number <= first:
            continue
        if code:
            # Code keeps its own syntax; a blank line in it goes, but does not end the paragraph.
            pending.append((number, text.rstrip()))
            continue
        if not text.strip() or RULE.fullmatch(text):
            close()
            continue
        heading = HEADING.fullmatch(text)
        if heading:
            close()
            level = len(heading[1])
            headings[:] = [(outer, title) for outer, title in headings if outer < level]
            headings.append((level, clean(heading[2]).strip()))
            sections.append(Section(" > ".join(title for _, title in headings if title), []))
            continue
        # A line that cleaning empties, such as an embedded image, does not end the paragraph either.
        pending.append((number, clean(text)))
    close()
    return [section for section in sections if section.blocks]


def read_notes() -> list[Note]:
    """Read every note in the project's notes folder, in the order of their paths."""
    notes = []
    for path in note_files(config.NOTES_DIR):
        # A note is what someone typed, and one odd byte should not stop the rest.
        source = path.read_text(encoding="utf-8", errors="replace")
        name = path.relative_to(config.NOTES_DIR).with_suffix("").as_posix()
        notes.append(Note(name, path.relative_to(config.PROJECT).as_posix(), source, parse_note(source)))
    return notes
