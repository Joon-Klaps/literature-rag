"""The TEI parser on a made-up GROBID output with its quirks (tests/fixtures/article.tei.xml)."""

from pathlib import Path

import pytest
from lxml import etree

from literature_rag import grobid, text

FIXTURE = Path(__file__).parent / "fixtures" / "article.tei.xml"


@pytest.fixture(scope="module")
def paper():
    return grobid.parse(FIXTURE.read_bytes())


def test_metadata(paper):
    assert paper["title"] == "Rodent density and spillover"
    assert paper["year"] == 2021
    assert paper["doi"] == "10.1000/test.2021.002"
    assert paper["abstract"] == "Spillover varies with rodent density."


def test_flat_divs_regain_their_structure(paper):
    headings = [section["heading"] for section in paper["sections"]]
    # "Methods" has no text of its own and parents "Trapping"; the headless div, the line number "214" and the download stamp continue it; the annex comes last.
    assert headings == ["Introduction", "Methods > Trapping", "Discussion", "Appendix A"]
    assert [p["text"] for p in paper["sections"][1]["paragraphs"]] == [
        "Traps were set every night.",
        "Traps were checked at dawn.",
        "Rodents were identified by sequencing.",
        "Sequences were deposited.",
    ]


def test_boilerplate_and_what_continues_it_are_left_out(paper):
    all_text = " ".join(p["text"] for s in paper["sections"] for p in s["paragraphs"])
    assert "field team" not in all_text and "funders" not in all_text and "Not read" not in all_text


def test_citations(paper):
    first = paper["sections"][0]["paragraphs"][0]
    assert first["text"] == "Spillover is frequent. Surveys agree, as does Smith et al. (2019)."
    assert first["cites"] == ["b0", "b1", "b2"]
    # A citation GROBID could not link has no id, but its marker still goes.
    discussion = paper["sections"][2]["paragraphs"][0]
    assert discussion == {"text": "Density predicts spillover.", "cites": []}


def test_captions_keep_labelled_figures_only(paper):
    assert [(c["label"], c["text"]) for c in paper["captions"]] == [
        ("Fig. 1", "Trapping sites in the valley."),
        ("Table 1", "Rodents trapped per season and site."),
    ]
    assert paper["captions"][0]["cites"] == ["b2"]


def test_references(paper):
    references = paper["references"]
    assert references["b0"] == {
        "text": "Doe J. Seasonal spillover. J Virol. 2015;89:1-10.",
        "doi": "10.1000/jv.2015.1",
        "pmid": None,
        "pmcid": None,
    }
    # Without a raw citation, the text is built from the fields.
    assert references["b1"]["text"] == "Smith A. Field study of rodents. Ecology. 2019;7:11–20"
    assert references["b1"]["pmid"] == "29999999"


def test_numbered_headings_nest_by_number():
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
        <div><head n="2">Results</head><p>Overview.</p></div>
        <div><head n="2.1">Prevalence</head><p>Ten percent.</p></div>
        <div><head n="3">Discussion</head><p>Talk.</p></div>
    </body></text></TEI>"""
    root = etree.fromstring(tei)
    divs = root.findall(".//tei:div", grobid.TEI)
    sections = grobid.read_sections(divs, text.ReferenceOrder([]))
    assert [s["heading"] for s in sections] == ["Results", "Results > Prevalence", "Discussion"]


def test_clean_removes_stamps_and_reattaches_accents():
    assert (
        grobid.clean(
            "Contacts were traced. Downloaded from academic.oup.com/cid/article/36/10/1254/307450 by KU Leuven Libraries user on 16 September 2026"
        )
        == "Contacts were traced."
    )
    assert (
        grobid.clean("Gu ¨nther S, Mu ´ller A. Downloaded from https://www.science.org on February 20, 2026")
        == "Günther S, Múller A."
    )


def test_combined_top_level_headings_end_a_parent():
    tei = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
        <div><head>Surveillance of contacts</head></div>
        <div><head>Epidemiological studies</head><p>Contacts were traced.</p></div>
        <div><head>CONCLUSION AND DISCUSSION</head><p>Risk was low.</p></div>
    </body></text></TEI>"""
    divs = etree.fromstring(tei).findall(".//tei:div", grobid.TEI)
    sections = grobid.read_sections(divs, text.ReferenceOrder([]))
    assert [s["heading"] for s in sections] == [
        "Surveillance of contacts > Epidemiological studies",
        "CONCLUSION AND DISCUSSION",
    ]
