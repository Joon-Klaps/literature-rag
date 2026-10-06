"""The JATS parser on a made-up article with Europe PMC's quirks (tests/fixtures/article.jats.xml)."""

from pathlib import Path

import pytest

from literature_rag import jats

FIXTURE = Path(__file__).parent / "fixtures" / "article.jats.xml"


@pytest.fixture(scope="module")
def paper():
    return jats.parse(FIXTURE.read_bytes())


def paragraphs(paper):
    return [p for section in paper["sections"] for p in section["paragraphs"]]


def test_metadata(paper):
    assert paper["title"] == "Rodent density and Mammarenavirus spillover in a made-up valley"
    # The PMC release date is not the publication date.
    assert paper["year"] == 2021
    assert paper["doi"] == "10.1000/test.2021.001"
    assert paper["pmid"] == "30000001"
    assert paper["pmcid"] == "PMC1234567"


def test_structured_abstract_keeps_headings_and_leaves_out_the_summary(paper):
    assert (
        paper["abstract"]
        == "Background: Spillover varies with rodent density. Methods: We trapped rodents in two seasons."
    )


def test_nested_sections_and_reading_order(paper):
    headings = [section["heading"] for section in paper["sections"]]
    assert headings == [
        "",
        "Introduction",
        "Methods",
        "Methods > Sequencing",
        "Methods",
        "Discussion",
        "Discussion > Box 1 Glossary of terms",
    ]
    # The paragraph after the subsection stays after it.
    assert paper["sections"][4]["paragraphs"][0]["text"] == "All analyses used R."


def test_boilerplate_sections_are_left_out(paper):
    text = " ".join(p["text"] for p in paragraphs(paper))
    assert "reviewers" not in text
    assert "field team" not in text


def test_superscript_citations_come_out_of_the_text(paper):
    first = paper["sections"][1]["paragraphs"][0]
    assert first["text"].startswith("Spillover is frequent in the dry season. Earlier surveys agree, as does")
    assert "season1" not in first["text"] and "[" not in first["text"]


def test_a_range_cites_every_reference_in_it(paper):
    first = paper["sections"][1]["paragraphs"][0]
    assert first["cites"] == ["R1", "R2", "R3"]


def test_author_year_citations_keep_their_text(paper):
    assert "(Smith et al., 2019)" in paper["sections"][1]["paragraphs"][0]["text"]


def test_exponents_and_formulas_survive(paper):
    second = paper["sections"][1]["paragraphs"][1]
    assert second["text"] == "Viral loads reached 10^5 copies per mL, with n=12 animals."
    # The figure inside the paragraph is a caption, not running text.
    assert "Trapping sites" not in second["text"]


def test_captions(paper):
    labels = [caption["label"] for caption in paper["captions"]]
    assert labels == ["Fig. 1", "Table 1"]
    assert paper["captions"][0]["text"] == "Trapping sites. Sites in the valley."
    assert paper["captions"][0]["cites"] == ["R4"]


def test_references_and_their_identifiers(paper):
    references = paper["references"]
    # The reviewer report's reference list is not the paper's.
    assert list(references) == ["R1", "R2", "R3", "R4"]
    assert references["R1"] == {
        "text": "Doe J. Seasonal spillover. J Virol. 2015;89:1–10.",
        "doi": "10.1000/jv.2015.1",
        "pmid": None,
        "pmcid": None,
    }
    assert references["R2"]["text"] == "Smith A, Jones B. Field study of rodents. Ecology. 2019;7:11–20"
    assert references["R2"]["pmid"] == "29999999"
    assert (references["R3"]["pmid"], references["R3"]["pmcid"]) == ("2000000", "PMC7654321")
    # Of two forms of one citation, the printed one wins.
    assert references["R4"]["text"] == "Poe E. Valley maps. Maps. 2001."
    assert references["R4"]["doi"] == "10.1000/maps.4"
