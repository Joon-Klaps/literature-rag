"""The thesis reader on a made-up chapter that \\input-s a second file (tests/fixtures/chapter.tex and part.tex)."""

from pathlib import Path

import pytest

from literature_rag import thesis

FIXTURES = Path(__file__).parent / "fixtures"
CHAPTER = "Rodents in a made-up valley"


@pytest.fixture(scope="module")
def parsed():
    sources = {
        "chapters/rodents.tex": (FIXTURES / "chapter.tex").read_text(encoding="utf-8"),
        "chapters/part.tex": (FIXTURES / "part.tex").read_text(encoding="utf-8"),
    }
    return thesis.parse_thesis(sources)


@pytest.fixture(scope="module")
def chunks(parsed):
    return {chunk["chunk_id"]: chunk for chunk in parsed[0]}


def test_blocks_line_ranges_and_reading_order(chunks):
    # The included file is read where \input names it, and the label line and the commented-out section give nothing.
    assert list(chunks) == [
        "rodents:L7-10",
        "rodents:L15-15",
        "rodents:L17-24",
        "rodents:L26-31",
        "rodents:L35-41",
        "part:L3-3",
        "rodents:L45-45",
    ]
    assert chunks["part:L3-3"]["file"] == "chapters/part.tex"


def test_comments_do_not_end_a_paragraph_and_are_stripped(chunks):
    text = chunks["rodents:L7-10"]["text"]
    assert text == (
        "Traps were set in 12 villages. Captures rose to 10000 per year."
        " The rate was 1,234 animals per site, as found. Prof. Smith counted 45% of them."
    )
    assert "comment" not in text and "note" not in text


def test_section_paths(chunks):
    assert chunks["rodents:L7-10"]["section"] == f"{CHAPTER} > Trapping"
    assert chunks["rodents:L15-15"]["section"] == f"{CHAPTER} > Trapping > Seasons"
    assert chunks["rodents:L35-41"]["section"] == f"{CHAPTER} > Code"
    # A \section in an included file replaces "Code", and the path runs on after the \input, as in LaTeX.
    assert chunks["part:L3-3"]["section"] == f"{CHAPTER} > Included"
    assert chunks["rodents:L45-45"]["section"] == f"{CHAPTER} > Included"


def test_citation_commands_and_their_keys(chunks):
    # \citep with optional arguments, \citet and \cite, in order of first appearance and each key once.
    assert chunks["rodents:L7-10"]["keys"] == ["Doe2015", "Roe2019", "Poe2001"]
    assert chunks["rodents:L15-15"]["keys"] == ["Doe2015"]
    assert chunks["rodents:L15-15"]["text"] == "Dry season captures doubled."


def test_a_figure_or_table_is_one_caption_chunk(chunks):
    figure = chunks["rodents:L17-24"]
    assert figure["kind"] == "caption"
    # The long form of the caption, and not the commented-out old one.
    assert figure["text"] == "Trap layout. A grid of 49 traps."
    assert figure["keys"] == ["Doe2015"]
    # A table's keys include the ones its rows cite.
    assert chunks["rodents:L26-31"]["text"] == "Hosts."
    assert chunks["rodents:L26-31"]["keys"] == ["Roe2019"]


def test_verbatim_is_kept_as_written(chunks):
    code = chunks["rodents:L35-41"]
    assert code["kind"] == "paragraph"
    assert "W[EA]+K % not a comment" in code["text"]
    assert code["text"].endswith("It also matches week.")
    # A citation command inside verbatim is printed, not cited.
    assert code["keys"] == []


def test_hash_follows_the_lines_comments_included(chunks):
    first = chunks["rodents:L7-10"]["hash"]
    assert len(first) == 16
    edited = (FIXTURES / "chapter.tex").read_text(encoding="utf-8").replace("trailing note", "trailing remark")
    chunks_after, _ = thesis.parse_thesis({"chapters/rodents.tex": edited})
    after = next(chunk for chunk in chunks_after if chunk["chunk_id"] == "rodents:L7-10")
    assert after["text"] == chunks["rodents:L7-10"]["text"] and after["hash"] != first


def test_is_current_while_the_lines_stand(chunks):
    source = (FIXTURES / "chapter.tex").read_text(encoding="utf-8")
    chunk = chunks["rodents:L7-10"]
    assert thesis.is_current(chunk, source)
    assert not thesis.is_current(chunk, source.replace("trailing note", "trailing remark"))
    # A line added above moves the paragraph, so its old line range no longer holds it.
    assert not thesis.is_current(chunk, "\n" + source)


def test_pairs(parsed):
    _, sentences = parsed
    pairs = thesis.make_pairs(sentences, {"doe2015": "Doe2015", "roe2019": "Roe2019"})
    # Sentence 1 cites two papers and sentence 3 one outside the library, so they give no pair; nor do captions.
    assert [(pair["pair_id"], pair["key"], pair["numbers"]) for pair in pairs] == [
        ("rodents:L7-10/0", "Doe2015", ["12"]),
        ("rodents:L7-10/2", "Roe2019", ["1234"]),
        ("rodents:L15-15/0", "Doe2015", []),
    ]
    assert pairs[1]["query"] == "The rate was 1,234 animals per site, as found."


def test_keys_match_the_library_whatever_their_case():
    sentence = thesis.Sentence("ch:L1-1", 0, "A claim.", ["de_Vries2021"])
    assert thesis.make_pairs([sentence], {"de_vries2021": "De_Vries2021"})[0]["key"] == "De_Vries2021"


def test_numbers():
    sentence = "Of 1,234 cases in 2009–2022, 0.5% had 10^5 copies"
    assert thesis.numbers(sentence) == ["1234", "2009", "2022", "0.5", "10^5"]
    assert thesis.numbers("COVID-19 and H1N1 at sites 1,2") == ["1", "2"]


def test_comment_stripping():
    assert thesis.strip_comment(r"45\% of them % a note") == r"45\% of them "
    assert thesis.strip_comment(r"a line break\\% then a comment") == r"a line break\\"


def test_chapter_files_from_the_build_record():
    fls = """PWD /thesis
INPUT ./chapters/02_introduction/02_introduction.tex
INPUT chapters/02_introduction/02_introduction.tex
INPUT ./chapters/02_introduction/image/map.pdf
INPUT ./chapters/02_introduction/02_introduction.aux
INPUT ./chapters/curriculum/curriculum.tex
INPUT /usr/local/texlive/tex/latex/base/report.cls
OUTPUT thesis.pdf
"""
    # Each file once, chapters only, .tex only, and not the CV.
    assert thesis.chapter_files(fls) == ["chapters/02_introduction/02_introduction.tex"]
