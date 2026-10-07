"""The arguments literature-rag-add hands to the fetch script, typed in a made-up Downloads folder next to a made-up thesis repository."""

import pytest

from literature_rag import add, config


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    repo = tmp_path / "thesis"
    (repo / "data" / "manuscripts").mkdir(parents=True)
    (repo / "data" / "manuscripts" / "Doe 2015 - Rodents.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(config, "THESIS_REPO", repo)
    folder = tmp_path / "Downloads"
    folder.mkdir()
    (folder / "paper.pdf").write_bytes(b"%PDF")
    (folder / "import-list.txt").write_text("10.1038/srep21977\n", encoding="utf-8")
    monkeypatch.chdir(folder)
    return folder


def test_a_path_from_here_is_made_absolute(downloads):
    assert add.resolve("paper.pdf") == str(downloads / "paper.pdf")
    assert add.resolve("./import-list.txt") == str(downloads / "import-list.txt")


def test_everything_else_is_passed_on_unchanged(downloads):
    for argument in ("10.1038/srep21977", "--mode", "title", "Spatial and temporal evolution of Lassa virus", ""):
        assert add.resolve(argument) == argument
    # A path relative to the thesis repository is read from there by the script, as before.
    assert add.resolve("data/manuscripts/Doe 2015 - Rodents.pdf") == "data/manuscripts/Doe 2015 - Rodents.pdf"


def test_a_pdf_found_nowhere_stops_the_command(downloads):
    with pytest.raises(SystemExit, match="papr.pdf: no such file"):
        add.resolve("papr.pdf")
