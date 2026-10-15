"""Reading notes on a made-up Obsidian vault: which files are notes, and how a note becomes sections of paragraphs with their line ranges."""

import os

from literature_rag import config, notes

# One line per entry, numbered in the comments as the file numbers them, from 1.
LINES = [
    "---",  # 1
    "tags: [seminar]",  # 2
    "---",  # 3
    "#lassa #seminar",  # 4
    "",  # 5
    "Rodents carry the virus. ^abc123",  # 6
    "They live in houses near [[Farms#Rice]].",  # 7
    "",  # 8
    "# Day 1",  # 9
    "",  # 10
    "## Panel discussion",  # 11
    "",  # 12
    "Trap counts were ==doubled== in May, says [[2025-09-09|the first talk]].",  # 13
    "![[photo.jpeg|800]]",  # 14
    "See [[slides.pdf#page=11&selection=1,2|slides, p.11]] and [BV-BRC](https://www.bv-brc.org/).",  # 15
    "",  # 16
    "%% a comment",  # 17
    "over two lines %%",  # 18
    "> [!important] The northern district has the most cases.",  # 19
    "",  # 20
    "```bash",  # 21
    "# not a heading",  # 22
    "",  # 23
    "samtools ${R1%%_*}",  # 24
    "```",  # 25
    "Run it twice.",  # 26
    "---",  # 27
    "Notes for slides.pdf:",  # 28
    "",  # 29
    "# Day 2",  # 30
    "%% a comment left open",  # 31
    "# hidden heading",  # 32
]
NOTE = "\n".join(LINES)


def test_a_note_becomes_sections_of_paragraphs_with_their_lines():
    sections = notes.parse_note(NOTE)
    # The frontmatter is not text, "Day 1" has no text of its own, and "Day 2" holds nothing but a comment, so only two sections remain.
    assert [section.heading for section in sections] == ["", "Day 1 > Panel discussion"]
    assert sections[0].blocks == [
        notes.Block(4, 4, "#lassa #seminar"),
        notes.Block(6, 7, "Rodents carry the virus.\nThey live in houses near Farms."),
    ]
    assert sections[1].blocks == [
        # The embedded image goes, without ending the paragraph; links read as their alias or the name of what they link to.
        notes.Block(13, 15, "Trap counts were doubled in May, says the first talk.\nSee slides, p.11 and BV-BRC."),
        notes.Block(19, 19, "> The northern district has the most cases."),
        # Code keeps its "#" and its "%%", its blank line ends nothing, and the horizontal rule ends the paragraph.
        notes.Block(21, 26, "```bash\n# not a heading\nsamtools ${R1%%_*}\n```\nRun it twice."),
    ]


def test_a_comment_may_hold_a_code_block():
    # The Kanban plugin keeps its settings this way: the fences inside the comment open no code block.
    source = 'Board\n\n%% kanban:settings\n```\n{"kanban-plugin": "basic"}\n```\n%%\nAfter'
    (section,) = notes.parse_note(source)
    assert section.blocks == [notes.Block(1, 1, "Board"), notes.Block(8, 8, "After")]


def test_a_plugin_s_boilerplate_and_bare_quote_markers_are_no_text():
    source = ">%%LINK%%[[#^k3x9q2|show annotation]]\n>\n>The left wing is longer.\n^k3x9q2"
    (section,) = notes.parse_note(source)
    assert section.blocks == [notes.Block(3, 3, ">The left wing is longer.")]
    assert notes.parse_note("Notes for Lecture 3.pdf:\n") == []


def test_notes_are_the_markdown_files_that_are_not_course_material_or_drawings(tmp_path, monkeypatch):
    vault, elsewhere = tmp_path / "vault", tmp_path / "elsewhere"
    for path in [
        vault / "daily" / "2025-01-01.md",
        vault / ".obsidian" / "workspace.md",
        vault / "course" / "README.md",
        vault / "Drawing 1.excalidraw.md",
        vault / "image.png",
        elsewhere / "seminar.md",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Some text.\n", encoding="utf-8")
    # A folder with a .git of its own is a cloned repository, such as a course's material.
    (vault / "course" / ".git").mkdir()
    # Folders may be links into the vault, which are followed; a link that leads nowhere or back up the tree is not.
    os.symlink(elsewhere, vault / "linked")
    os.symlink(tmp_path / "missing.md", vault / "broken.md")
    os.symlink(vault, vault / "loop")
    found = notes.note_files(vault)
    assert [path.relative_to(vault).as_posix() for path in found] == ["daily/2025-01-01.md", "linked/seminar.md"]

    monkeypatch.setattr(config, "PROJECT", tmp_path)
    monkeypatch.setattr(config, "NOTES_DIR", vault)
    note = notes.read_notes()[1]
    assert (note.name, note.file) == ("linked/seminar", "vault/linked/seminar.md")
    assert note.sections == [notes.Section("", [notes.Block(1, 1, "Some text.")])]
