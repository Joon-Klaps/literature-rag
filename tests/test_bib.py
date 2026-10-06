"""The bibliography parser and the normalisation that matching depends on."""

from literature_rag import bib

BIB = r"""
@comment{jabref-meta: not an entry}
@string{nat = "Nature"}

@ARTICLE{Garry2023-ab,
  title        = {{Lassa} fever: the road ahead},
  author       = {Garry, Robert F},
  date         = {2023-01-15},
  doi          = {https://doi.org/10.1038/S41579-022-00789-8},
}

@article{Doe2015-seasonal,
  title = "Seasonal {\'e}t{\'e} spillover",
  year = 2015,
  url = {https://doi.org/10.1000/jv.2015.1}
}

@misc{NoDoi2020,
  title = {A report \& its {annex}},
  year = {2020},
}
"""


def test_entries_and_fields():
    entries = bib.parse_bib(BIB, cited={"garry2023-ab"})
    assert [entry["key"] for entry in entries] == ["Garry2023-ab", "Doe2015-seasonal", "NoDoi2020"]
    garry, doe, report = entries
    assert garry == {
        "key": "Garry2023-ab",
        "doi": "10.1038/s41579-022-00789-8",
        "title": "lassa fever the road ahead",
        "year": 2023,
        "cited": True,
    }
    # A DOI only in the url field is found too, and LaTeX accents fold to ASCII.
    assert doe["doi"] == "10.1000/jv.2015.1"
    assert doe["title"] == "seasonal ete spillover"
    assert doe["year"] == 2015 and doe["cited"] is False
    assert report["doi"] is None and report["title"] == "a report its annex"


def test_braced_values_nest():
    fields = bib.parse_fields(' title = {A {B {C}} D}, year = "1999", volume = 7,')
    assert fields == {"title": "A {B {C}} D", "year": "1999", "volume": "7"}


def test_normalise_doi():
    assert bib.normalise_doi("doi: 10.1016/S2666-5247(21)00178-6.") == "10.1016/s2666-5247(21)00178-6"
    assert bib.normalise_doi("https://dx.doi.org/10.1093/ve/vew007") == "10.1093/ve/vew007"
    assert bib.normalise_doi("10.1002/a\\_b") == "10.1002/a_b"
    assert bib.normalise_doi("N/A") is None


def test_normalise_title():
    assert bib.normalise_title("Lassa fever — the road ahead.") == bib.normalise_title("Lassa fever - the road ahead")
    assert bib.normalise_title("Köster: Snakemake") == "koster snakemake"


def test_cited_keys_are_case_insensitive():
    bbl = r"\bibitem[Garry(2023)]{Garry2023-ab} x \bibitem{de_Vries2021-po} y"
    assert bib.cited_keys(bbl) == {"garry2023-ab", "de_vries2021-po"}
