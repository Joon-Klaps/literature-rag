"""Which project the commands work on: the one LITERATURE_RAG_PROJECT names, or else the one they run in, on made-up folders."""

import pytest

from literature_rag import config


@pytest.fixture
def projects(tmp_path):
    """Two made-up projects with their papers, and a folder that is not a project."""
    for name in ("thesis", "manuscript"):
        (tmp_path / name / "data" / "manuscripts").mkdir(parents=True)
    (tmp_path / "thesis" / "chapters" / "intro").mkdir(parents=True)
    (tmp_path / "Downloads").mkdir()
    return tmp_path


def test_the_variable_names_the_project_wherever_the_command_runs(projects):
    environ = {"LITERATURE_RAG_PROJECT": str(projects / "manuscript")}
    assert config.find_project(environ, projects / "thesis" / "chapters") == projects / "manuscript"
    assert config.find_project(environ, projects / "Downloads") == projects / "manuscript"


def test_without_the_variable_the_project_is_the_one_the_command_runs_in(projects):
    assert config.find_project({}, projects / "thesis") == projects / "thesis"
    assert config.find_project({}, projects / "thesis" / "chapters" / "intro") == projects / "thesis"
    # An empty variable counts as unset, as a shell's `export LITERATURE_RAG_PROJECT=` would leave it.
    assert config.find_project({"LITERATURE_RAG_PROJECT": ""}, projects / "manuscript") == projects / "manuscript"


def test_outside_any_project_the_commands_refuse_to_run(projects, monkeypatch):
    here = config.find_project({}, projects / "Downloads")
    assert here == projects / "Downloads"
    monkeypatch.setattr(config, "PROJECT", here)
    monkeypatch.setattr(config, "MANUSCRIPTS_DIR", here / "data" / "manuscripts")
    with pytest.raises(SystemExit, match="Set LITERATURE_RAG_PROJECT"):
        config.require_project()
