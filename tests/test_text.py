"""Clean-up shared by the parsers, the Markdown fallback, and the ingestion report's title check."""

from literature_rag import ingest, markdown, text


def test_boilerplate_headings():
    for title in (
        "Acknowledgements",
        "5. Competing interests",
        "Authors’ contributions",
        "Data availability statement",
        "Ethics approval and consent to participate",
        "Supplementary Information",
    ):
        assert text.is_boilerplate(title), title
    for title in (
        "Ethical and regulatory considerations",
        "Data collection",
        "Results",
        "Availability of commodities at ORT corners",
    ):
        assert not text.is_boilerplate(title), title


def test_heading_numbers():
    assert text.heading("2.1 Data collection.") == "Data collection"
    assert text.heading("A. Serology data") == "Serology data"
    assert text.heading("1 × 300 data generation") == "1 × 300 data generation"


def test_tidy_removes_what_markers_leave():
    assert text.tidy("agree [, ] and (–) so .") == "agree and so."
    assert text.is_numeric_marker("[12–14]") and text.is_numeric_marker("3, 5")
    assert not text.is_numeric_marker("Smith et al., 2019")
    # GROBID's links with the separator inside them.
    assert text.is_numeric_marker("13,") and text.is_numeric_marker("[10,") and text.is_numeric_marker("19]")
    # A year that author-year styles link on its own is not a reference number.
    assert not text.is_numeric_marker("2007") and not text.is_numeric_marker("(2019a)")


def test_sentences_end_at_full_stops_but_not_after_abbreviations():
    passage = "Virus was found in M. natalensis by Doe et al. (2019). Fig. 2 shows e.g. this. The mean was 5.2. Next."
    assert text.sentences(passage) == [
        "Virus was found in M. natalensis by Doe et al. (2019).",
        "Fig. 2 shows e.g. this.",
        "The mean was 5.2.",
        "Next.",
    ]
    # A lowercase word after a full stop does not start a sentence.
    assert text.sentences("Data from approx. ten sites were used.") == ["Data from approx. ten sites were used."]


def test_reference_ranges():
    order = text.ReferenceOrder(["b0", "b1", "b2", "b3"])
    assert order.between("b0", "b3") == ["b1", "b2"]
    assert order.between("b3", "b0") == [] and order.between(None, "b1") == []


def test_markdown_fallback():
    md = "# A title\n\n## Metadata\n\n- PMID: 1\n- Year: 1971\n\n## Abstract\n\nShort.\n\n## Extracted Full Text\n\n## Page 1\n\nFirst\x00 ﬁrst page\ntext.\n\n## Page 2\n\nSecond page.\n"
    paper = markdown.parse(md)
    assert paper["abstract"] == "Short."
    assert paper["year"] == 1971
    assert [p["text"] for p in paper["sections"][0]["paragraphs"]] == ["First first page text.", "Second page."]


def test_title_check_flags_another_document():
    paper = {
        "title": "Malaria research and control in Sierra Leone",
        "abstract": "",
        "sections": [
            {"heading": "", "paragraphs": [{"text": "Modern slavery statement of a publisher.", "cites": []}]}
        ],
    }
    assert not ingest.title_found(paper)
    paper["sections"][0]["paragraphs"][0]["text"] = "A century of malaria control in Sierra Leone and its research."
    assert ingest.title_found(paper)


def test_the_copy_with_the_better_text_is_kept():
    def row(name, duplicate_of=None, citable=True):
        paper_id = "Hickey2024" if duplicate_of is None else name
        return {"source_file": name, "duplicate_of": duplicate_of, "paper_id": paper_id, "citable": citable}

    def paper(source, paragraphs):
        return {"paper_id": "", "source": source, "sections": [{"heading": "", "paragraphs": [{}] * paragraphs}]}

    rows = [
        row("Hickey 2024.md"),
        row("Hickey et al. 2024.md", "Hickey 2024.md"),
        row("suppl.md", "Hickey 2024.md", citable=False),
    ]
    papers = {
        "Hickey 2024.md": paper("grobid", 4),
        "Hickey et al. 2024.md": paper("grobid", 65),
        "suppl.md": paper("jats", 90),
    }
    lines = ingest.choose_copies(rows, papers)
    # The citable copy with more paragraphs wins and takes the shared id; a non-citable copy never wins.
    assert [r["duplicate_of"] for r in rows] == ["Hickey et al. 2024.md", None, "Hickey et al. 2024.md"]
    assert papers["Hickey et al. 2024.md"]["paper_id"] == "Hickey2024"
    assert papers["Hickey 2024.md"]["paper_id"] == "Hickey 2024"
    assert len(lines) == 2
