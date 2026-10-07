"""Library chunking on a made-up paper: merging within sections, size limits, captions and the abstract alone, neighbour ids and citations."""

import pytest

from literature_rag import chunks


def words(n: int, word: str = "word") -> str:
    """A paragraph of n words in sentences of ten, each starting with a capital."""
    sentences = [" ".join([word.title()] + [word] * 8 + [f"{word}."]) for _ in range(n // 10)]
    return " ".join(sentences)


@pytest.fixture(scope="module")
def paper():
    return {
        "paper_id": "Doe2015-valley",
        "key": "Doe2015-valley",
        "citable": True,
        "title": "Rodents of a made-up valley",
        "year": 2015,
        "doi": "10.1000/valley",
        "source_file": "Doe 2015 - Rodents of a made-up valley.md",
        "abstract": words(80, "abstract"),
        "sections": [
            {
                "heading": "Introduction",
                "paragraphs": [
                    {"text": words(120, "one"), "cites": ["r1"]},
                    {"text": words(120, "two"), "cites": ["r2", "r1"]},
                    {"text": words(120, "three"), "cites": []},
                ],
            },
            {"heading": "Methods", "paragraphs": [{"text": words(30, "four"), "cites": []}]},
            {"heading": "", "paragraphs": [{"text": words(450, "five"), "cites": ["r2"]}]},
        ],
        "captions": [{"label": "Fig. 1", "text": "Trapping sites.", "cites": ["r1"]}],
        "references": {
            "r1": {"text": "Roe R. Rodents. 2010.", "doi": "10.1000/r1", "pmid": None, "pmcid": None},
            "r2": {"text": "Poe P. Valleys. 2001.", "doi": None, "pmid": "123", "pmcid": None},
        },
    }


@pytest.fixture(scope="module")
def library(paper):
    return chunks.chunk_paper(paper)


def test_reading_order_and_neighbour_ids(library):
    assert [chunk["chunk_id"] for chunk in library] == [f"Doe2015-valley#{n}" for n in range(len(library))]
    assert [(chunk["kind"], chunk["section"]) for chunk in library] == [
        ("abstract", "Abstract"),
        ("body", "Introduction"),
        ("body", "Introduction"),
        ("body", "Methods"),
        ("body", ""),
        ("body", ""),
        ("caption", "Fig. 1"),
    ]


def test_paragraphs_merge_up_to_the_limit_and_never_across_sections(library):
    sizes = [chunks.word_count(chunk["text"]) for chunk in library]
    # 120 + 120 fit within 300, a third 120 would not; the 30 words of Methods stay in their own section.
    assert sizes[1:4] == [240, 120, 30]
    assert library[1]["text"].count("\n\n") == 1


def test_a_long_paragraph_is_split_at_sentence_ends_into_even_pieces(library):
    pieces = [chunk["text"] for chunk in library[4:6]]
    assert [chunks.word_count(piece) for piece in pieces] == [230, 220]
    assert all(piece.endswith("five.") for piece in pieces)


def test_a_sentence_over_the_limit_is_cut_between_words_into_even_parts():
    pieces = chunks.split(" ".join(["word"] * 700))
    assert [chunks.word_count(piece) for piece in pieces] == [234, 234, 232]


def test_citations_are_resolved_and_merged(library, paper):
    references = paper["references"]
    assert library[1]["cites"] == [references["r1"], references["r2"]]
    assert library[2]["cites"] == []
    # Each piece of a split paragraph keeps all of its citations.
    assert library[4]["cites"] == library[5]["cites"] == [references["r2"]]
    assert library[6]["cites"] == [references["r1"]]


def test_index_text(library):
    assert chunks.index_text(library[6]) == "Rodents of a made-up valley. Fig. 1. Trapping sites."
    # No ". ." where a section has no heading.
    assert chunks.index_text(library[4]).startswith("Rodents of a made-up valley. Five five")
    thesis_chunk = {"section": "Introduction > Lassa fever", "text": "A paragraph."}
    assert chunks.index_text(thesis_chunk) == "Introduction > Lassa fever. A paragraph."


def test_library_keys_are_citable_and_case_free():
    papers = [
        {"key": "De_Vries2021", "citable": True},
        {"key": "Style2010", "citable": False},
        {"key": None, "citable": False},
    ]
    assert chunks.library_keys(papers) == {"de_vries2021": "De_Vries2021"}
