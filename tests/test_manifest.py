"""Reading the fetch script's metadata block, and matching a paper to its citation key."""

from literature_rag import manifest
from literature_rag.records import BibEntry

MARKDOWN = """# Rodent density and spillover.

## Metadata

- Query mode: doi
- Matched PubMed record: yes
- PMID: 30000001
- PMCID: N/A
- DOI: 10.1000/TEST.2021.002
- Year: 2021

## Abstract

Spillover varies.

## Extracted Full Text

## Page 1

- Item: a list line in the paper text, which is not metadata
"""


def entry(key, doi=None, title="", cited=False):
    return BibEntry(key=key, doi=doi, title=title, year=None, cited=cited)


INDEX = manifest.BibIndex(
    [
        entry("Smith2021-rodent", doi="10.1000/test.2021.002", title="rodent density and spillover"),
        entry("Paperpile2021-xy", doi="10.1000/twice", title="one paper twice"),
        entry("Cited2021-twice", doi="10.1000/twice", title="one paper twice", cited=True),
        entry("Koster2012-snakemake", title="snakemake a scalable bioinformatics workflow engine"),
    ]
)


def test_metadata_block():
    metadata = manifest.parse_metadata(MARKDOWN)
    assert metadata["Title"] == "Rodent density and spillover"
    assert metadata["DOI"] == "10.1000/TEST.2021.002"
    # Unknown values are dropped, and lines in the paper text are not metadata.
    assert "PMCID" not in metadata and "Item" not in metadata


def test_match_by_doi_then_title_then_similar_title():
    assert INDEX.match("10.1000/TEST.2021.002", []) == ("Smith2021-rodent", "doi")
    assert INDEX.match(None, ["Rodent density and spillover."]) == ("Smith2021-rodent", "title")
    assert INDEX.match(None, ["Snakemake: a scaleable bioinformatic workflow engine"]) == (
        "Koster2012-snakemake",
        "similar title",
    )
    assert INDEX.match(None, ["Something else entirely, at some length"]) == (None, None)


def test_the_cited_key_wins_among_duplicates():
    assert INDEX.match("10.1000/twice", []) == ("Cited2021-twice", "doi")


def test_file_names():
    assert manifest.FILE_NAME.match("Klitting et al. - Lassa Virus Genetics").group("title") == "Lassa Virus Genetics"
    assert manifest.file_year({}, "Garry 2023 - Lassa fever - the road ahead") == 2023
    assert (
        manifest.best_title(
            {"Title": "DocHdl1OnVERSA-PPM01tmpTarget", "Matched PubMed record": "no"},
            "Compeau 2015 - Bioinformatics algorithms",
        )
        == "Bioinformatics algorithms"
    )


def test_rows_supplements_and_overrides():
    row = manifest.build_row(
        "Smith 2021 - SUPPL - Rodent density.md", MARKDOWN.replace("PMCID: N/A", "PMCID: PMC1"), None, INDEX, {}
    )
    # A supplement keeps the paper's key but not its PMCID's XML, which would be the paper itself.
    assert (row["paper_id"], row["route"], row["supplement"]) == ("Smith2021-rodent-supplement", "grobid", True)
    row = manifest.build_row("odd.md", "# Untitled\n", None, INDEX, {"odd.md": "Koster2012-snakemake"})
    assert (row["key"], row["matched_by"], row["citable"]) == ("Koster2012-snakemake", "override", True)


def test_duplicates_keep_the_copy_with_a_pmcid():
    first = manifest.build_row("Smith et al. 2021 - Rodent density.md", MARKDOWN, None, INDEX, {})
    second = manifest.build_row(
        "Smith 2021 - Rodent density.md", MARKDOWN.replace("PMCID: N/A", "PMCID: PMC1"), None, INDEX, {}
    )
    lines = manifest.mark_duplicates([first, second])
    assert second["duplicate_of"] is None and second["paper_id"] == "Smith2021-rodent"
    assert (
        first["duplicate_of"] == "Smith 2021 - Rodent density.md"
        and first["paper_id"] == "Smith et al. 2021 - Rodent density"
    )
    assert len(lines) == 1
