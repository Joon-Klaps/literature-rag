"""The HTML report on a made-up paper and two questions: marking, escaping, the links between a hit and its paper, and the file it is saved to."""

import os
from urllib.parse import quote

from literature_rag import config, report


def chunk(n: int, text: str) -> dict:
    return {
        "chunk_id": f"Doe2015-valley#{n}",
        "paper_id": "Doe2015-valley",
        "key": "Doe2015-valley",
        "citable": True,
        "title": "Rodents & rain of a made-up valley",
        "year": 2015,
        "doi": "10.1000/valley",
        "source_file": "Doe 2015 - valley.md",
        "section": "Results",
        "kind": "body",
        "text": text,
        "cites": [],
    }


PAPER = [chunk(0, "Rainfall peaked in August; <b>no</b> rats."), chunk(1, "The case fatality was 25% & falling.")]


def answer(question: str, terms: list[str], row: int, rank: int = 1) -> dict:
    hit = {"rank": rank, "score": 0.9, "ranks": {"bm25": 1, "fused": 2}, "chunk": PAPER[row]}
    return {"question": question, "timings": {"bm25": 0.01}, "terms": terms, "hits": [hit]}


def test_highlight_marks_whole_words_in_any_case_and_escapes_the_rest():
    marked = report.highlight("Rainfall; rain & RAIN, rainy <i>", ["rain", "rainfall"])
    assert marked == "<mark>Rainfall</mark>; <mark>rain</mark> &amp; <mark>RAIN</mark>, rainy &lt;i&gt;"
    assert report.highlight("a < b", []) == "a &lt; b"


def test_highlight_does_not_cut_an_entity():
    # "amp" is a term, and the ampersand's entity must survive: the text is marked before it is escaped.
    assert report.highlight("a & b amp", ["amp"]) == "a &amp; b <mark>amp</mark>"


def test_render_links_each_hit_to_its_place_in_the_paper():
    page = report.render(
        "library",
        [answer("rainfall in august", ["rainfall", "august"], 0), answer("case fatality", ["case", "fatality"], 1)],
        {"Doe2015-valley": PAPER},
    )
    # Hit text in full, marked, with the stray tag escaped.
    assert "<mark>Rainfall</mark> peaked in <mark>August</mark>; &lt;b&gt;no&lt;/b&gt; rats." in page
    assert "<mark>case</mark> <mark>fatality</mark> was 25% &amp; falling." in page
    # Each hit points at its chunk inside the one copy of the paper, which says which question found it.
    assert 'href="#doc0c0"' in page and 'href="#doc0c1"' in page
    assert page.count('<details id="doc0">') == 1
    assert "q1 #1" in page and "q2 #1" in page
    assert "2 chunks, 2 found" in page
    assert "Doe%202015%20-%20valley.pdf" in page and "https://doi.org/10.1000/valley" in page
    assert "Rodents &amp; rain of a made-up valley" in page


def test_render_without_hits():
    page = report.render("library", [{"question": "nothing", "timings": {}, "terms": [], "hits": []}], {})
    assert "No hits." in page


def test_write_does_not_overwrite(tmp_path):
    first = report.write("<p>one</p>", tmp_path / "reports")
    assert first.read_text(encoding="utf-8") == "<p>one</p>"
    assert first.parent == tmp_path / "reports" and first.name.startswith("search-")


def test_a_note_opens_in_obsidian_through_the_notes_folder_s_links(tmp_path, monkeypatch):
    # The project's data/notes holds links into the vault, and Obsidian knows a note only by its path in the vault.
    (tmp_path / "vault" / "daily").mkdir(parents=True)
    (tmp_path / "project" / "data").mkdir(parents=True)
    os.symlink(tmp_path / "vault", tmp_path / "project" / "data" / "notes")
    monkeypatch.setattr(config, "PROJECT", tmp_path / "project")
    note = {
        "chunk_id": "daily/2025-08-26#0",
        "note": "daily/2025-08-26",
        "file": "data/notes/daily/2025-08-26.md",
        "line_start": 3,
        "line_end": 5,
        "section": "Field work",
        "text": "Rainfall at the site came late.",
        "hash": "0123456789abcdef",
    }
    assert report.source_label(note) == "daily/2025-08-26:3-5"
    in_vault = (tmp_path / "vault" / "daily" / "2025-08-26.md").resolve()
    assert report.file_links(note) == [
        ("Obsidian", f"obsidian://open?path={quote(str(in_vault), safe='')}"),
        ("Markdown", in_vault.as_uri()),
    ]
