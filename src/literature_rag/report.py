"""Render search results as one self-contained HTML page, for reading rather than for an agent.

    uv run literature-rag-search "case fatality among hospitalised Lassa fever patients" --open

Each question gets its hits with their whole text and the question's words marked. Below them, each paper (or thesis chapter) that a hit came from appears once, with all its chunks in reading order and the hits highlighted, so that a hit can be read in its context without finding the file. The functions here are pure: search.py gathers the answers and the documents, and render() turns them into a string.
"""

import html
import re
from datetime import datetime
from pathlib import Path

from literature_rag import config
from literature_rag.records import Answer, Chunk, Hit, ThesisChunk

STYLE = """
:root { --bg: #fdfdfb; --fg: #1d1d1b; --muted: #6b6b66; --line: #dcdcd4; --card: #ffffff; --mark: #ffe58a; --hit: #eef4ff; --accent: #2456a6; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161617; --fg: #e8e8e4; --muted: #9a9a94; --line: #34343a; --card: #1e1e20; --mark: #6b5a12; --hit: #1c2740; --accent: #8fb3ee; }
}
body { background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, sans-serif; margin: 0 auto; max-width: 62rem; padding: 1.5rem 1rem 4rem; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .25rem; }
a { color: var(--accent); }
mark { background: var(--mark); color: inherit; border-radius: 2px; padding: 0 1px; }
.muted { color: var(--muted); font-size: .85rem; }
.hit, details { background: var(--card); border: 1px solid var(--line); border-radius: 6px; margin: .75rem 0; padding: .75rem 1rem; }
.head { display: flex; flex-wrap: wrap; gap: .25rem .75rem; align-items: baseline; }
.rank { font-weight: 700; }
.badge { background: var(--hit); border-radius: 4px; font-size: .78rem; padding: 0 .4rem; white-space: nowrap; }
.links { font-size: .85rem; margin: .25rem 0 0; }
.hit p { margin: .5rem 0 0; }
summary { cursor: pointer; font-weight: 600; }
.chunk { border-left: 3px solid transparent; margin: .75rem 0; padding-left: .75rem; }
.chunk.matched { background: var(--hit); border-left-color: var(--accent); }
.chunk p { margin: .15rem 0 0; }
.chunk:target { outline: 2px solid var(--accent); }
"""

# Opens the paper a link points into, since a closed <details> hides its target.
SCRIPT = """
function reveal() {
  const target = document.getElementById(location.hash.slice(1));
  const details = target && target.closest("details");
  if (details) { details.open = true; target.scrollIntoView(); }
}
addEventListener("hashchange", reveal);
addEventListener("DOMContentLoaded", reveal);
"""


def highlight(text: str, terms: list[str]) -> str:
    """The text as HTML, with each of the terms marked wherever it stands as a whole word, in any case. Words are runs of letters and digits, as in search.tokenize()."""
    if not terms:
        return html.escape(text, quote=False)
    # Longest first, so that a term is not cut short by one that is its beginning.
    pattern = "|".join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True))
    # With one capture group, split() alternates between the text around the matches and the matches, so every piece is escaped on its own and no tag is ever cut.
    pieces = re.split(rf"(?<![a-z0-9])({pattern})(?![a-z0-9])", text, flags=re.IGNORECASE)
    escaped = [html.escape(piece, quote=False) for piece in pieces]
    return "".join(f"<mark>{piece}</mark>" if i % 2 else piece for i, piece in enumerate(escaped))


def source_label(chunk: Chunk | ThesisChunk) -> str:
    """Where a chunk is from, as a person would cite it: the key and year of a paper, and the file and lines of a thesis paragraph."""
    if "paper_id" in chunk:
        label = f"{chunk['key'] or chunk['paper_id']} ({chunk['year'] or 'n.d.'})"
        return label if chunk["citable"] else f"{label}, not citable"
    return f"{chunk['file']}:{chunk['line_start']}-{chunk['line_end']}"


def file_links(chunk: Chunk | ThesisChunk) -> list[tuple[str, str]]:
    """The (label, file URL) pairs that open a chunk's source on this computer: the paper's PDF and Markdown, or the thesis file."""
    if "paper_id" in chunk:
        manuscript = config.MANUSCRIPTS_DIR / chunk["source_file"]
        return [("PDF", manuscript.with_suffix(".pdf").as_uri()), ("Markdown", manuscript.as_uri())]
    return [("LaTeX", (config.PROJECT / chunk["file"]).as_uri())]


def link_row(chunk: Chunk | ThesisChunk, anchor: str) -> str:
    """The links under a hit: to the chunk inside its paper below, to the source files, and, for a paper, to its DOI."""
    links = [f'<a href="#{anchor}">in context</a>']
    links += [f'<a href="{html.escape(url)}">{label}</a>' for label, url in file_links(chunk)]
    if chunk.get("doi"):
        links.append(f'<a href="https://doi.org/{html.escape(chunk["doi"])}">DOI</a>')
    return " · ".join(links)


def render_hit(answer_number: int, hit: Hit, text: str, anchor: str) -> str:
    """One hit as a card: rank, score, source, section and the places each method gave it, then its links and its text, already as HTML."""
    chunk = hit["chunk"]
    places = "".join(f'<span class="badge">{html.escape(name)} {place}</span>' for name, place in hit["ranks"].items())
    title = f'<div class="muted">{html.escape(chunk["title"])}</div>' if "title" in chunk else ""
    return (
        f'<article class="hit" id="q{answer_number}h{hit["rank"]}">'
        f'<div class="head"><span class="rank">{hit["rank"]}</span><span class="muted">{hit["score"]:.3f}</span>'
        f"<strong>{html.escape(source_label(chunk))}</strong><span>{html.escape(chunk['section'])}</span>{places}</div>"
        f'{title}<div class="links">{link_row(chunk, anchor)}</div><p>{text}</p></article>'
    )


def render_document(
    number: int, chunk_list: list[Chunk] | list[ThesisChunk], matches: dict[str, list[str]], terms: list[str]
) -> str:
    """A paper or chapter in reading order, collapsed, with the chunks that were hits highlighted and labelled with the question and rank that found them."""
    first = chunk_list[0]
    heading = source_label(first).split(" (")[0] if "paper_id" in first else first["section"].split(" > ")[0]
    found = sum(chunk["chunk_id"] in matches for chunk in chunk_list)
    summary = f'<summary>{html.escape(heading)}<span class="muted"> · {len(chunk_list)} chunks, {found} found</span></summary>'
    parts = [f'<details id="doc{number}">{summary}']
    for position, chunk in enumerate(chunk_list):
        labels = matches.get(chunk["chunk_id"], [])
        badges = "".join(f'<span class="badge">{html.escape(label)}</span> ' for label in labels)
        parts.append(
            f'<div class="chunk{" matched" if labels else ""}" id="doc{number}c{position}">'
            f'<div class="muted">{badges}{html.escape(chunk["section"])}</div><p>{highlight(chunk["text"], terms)}</p></div>'
        )
    parts.append("</details>")
    return "".join(parts)


def render(collection: str, answers: list[Answer], documents: dict[str, list[Chunk] | list[ThesisChunk]]) -> str:
    """The whole page for the answers to one or more questions.

    documents maps a document's id to all its chunks in reading order; each hit's chunk must be in one of them.
    """
    where = {
        chunk["chunk_id"]: (number, position)
        for number, chunk_list in enumerate(documents.values())
        for position, chunk in enumerate(chunk_list)
    }
    terms = [term for answer in answers for term in answer["terms"]]
    matches: list[dict[str, list[str]]] = [{} for _ in documents]
    for answer_number, answer in enumerate(answers, start=1):
        for hit in answer["hits"]:
            number, _ = where[hit["chunk"]["chunk_id"]]
            matches[number].setdefault(hit["chunk"]["chunk_id"], []).append(f"q{answer_number} #{hit['rank']}")

    body = []
    for answer_number, answer in enumerate(answers, start=1):
        timings = ", ".join(f"{name} {seconds:.2f} s" for name, seconds in answer["timings"].items())
        body.append(
            f'<h2>{answer_number}. {html.escape(answer["question"])}</h2><div class="muted">{html.escape(timings)}</div>'
        )
        if not answer["hits"]:
            body.append('<p class="muted">No hits.</p>')
        for hit in answer["hits"]:
            number, position = where[hit["chunk"]["chunk_id"]]
            text = highlight(hit["chunk"]["text"], answer["terms"])
            body.append(render_hit(answer_number, hit, text, f"doc{number}c{position}"))
    body.append("<h2>In context</h2>")
    body.extend(
        render_document(number, chunk_list, matches[number], terms)
        for number, chunk_list in enumerate(documents.values())
    )
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>Search results</title>'
        f"<style>{STYLE}</style></head><body><h1>{html.escape(config.PROJECT_NAME)}: {html.escape(collection)}</h1>"
        f'<div class="muted">{stamp} · {len(answers)} question{"s" if len(answers) != 1 else ""}</div>'
        f"{''.join(body)}<script>{SCRIPT}</script></body></html>"
    )


def write(page: str, directory: Path = config.REPORTS_DIR) -> Path:
    """Save a page under a name made from the time, so that no earlier report is overwritten, and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"search-report.html"
    path.write_text(page, encoding="utf-8")
    return path
