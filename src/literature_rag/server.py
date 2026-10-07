"""The MCP server thesis-library: search the papers of the thesis library and the thesis itself, and read the chunks around a hit.

    uv run literature-rag

Claude Code starts it and talks to it over stdio; the README has the line that registers it. Each tool is a few lines of glue, as in the nf-core server: load what it needs, call search.py, and shape the hits for the model. Search runs as block 5 chose (config.DEFAULT_METHODS and DEFAULT_RERANK). The collections and the models load on the first call, which takes some seconds; later calls take about three, nearly all of it the reranker.
"""

import logging
import threading
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from literature_rag import config, index, search, thesis
from literature_rag.records import Chunk, Hit, LibraryPassage, ThesisChunk, ThesisParagraph

INSTRUCTIONS = (
    "Search over Joon Klaps's PhD thesis on Lassa virus genomic surveillance and the papers in its library, the thesis repository's data/manuscripts. "
    "search_library finds passages in the papers, search_thesis finds paragraphs of the thesis, and get_context reads the chunks around a hit from either. "
    "A library passage is a lead: check the claim in the paper itself before citing it. A thesis paragraph is never evidence for a claim."
)

mcp = MCPServer("thesis-library", version="0.1.0", instructions=INSTRUCTIONS)
# The server logs at INFO, which makes sentence-transformers draw a progress bar for every query; they go to stderr, so they only fill Claude Code's MCP log.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Claude Code may call several tools at once, and the SDK runs each in a thread of its own. The models share one GPU, and a collection should be loaded once, so calls take turns.
lock = threading.Lock()
# Each collection as last loaded, with the stamp of the files it was loaded from.
loaded: dict[str, tuple[tuple[int, ...], search.Collection]] = {}


def stamp(name: str) -> tuple[int, ...]:
    """When the files behind a collection last changed: its chunk file and each model's vectors."""
    files = [
        config.COLLECTIONS[name],
        *(path for model in config.EMBEDDING_MODELS for path in index.paths(name, model)),
    ]
    return tuple(path.stat().st_mtime_ns if path.is_file() else 0 for path in files)


def collection(name: str) -> search.Collection:
    """A collection, loaded on first use and again whenever literature-rag-chunk or literature-rag-build has changed its files, so that a paper added while the server runs is found without a restart. Call it with the lock held."""
    now = stamp(name)
    if name not in loaded or loaded[name][0] != now:
        # search.load_collection keeps what it loaded before; forget that, so the files are read again.
        search.load_collection.cache_clear()
        try:
            loaded[name] = (now, search.load_collection(name))
        except index.StaleIndex as error:
            raise ToolError(str(error)) from None
    return loaded[name][1]


def run(name: str, query: str, k: int) -> list[Hit]:
    """The k best hits in a collection, searched with the defaults, one call at a time."""
    with lock:
        found = collection(name)
        try:
            return search.search(found, query, k)
        except SystemExit as error:
            # models.local() ends the command-line tools this way when a model is not downloaded; the server must answer instead of stopping.
            raise ToolError(str(error)) from None


def library_passage(chunk: Chunk, score: float | None = None) -> LibraryPassage:
    """A library chunk as the tools return it, with the path of its paper and, from a search, its score."""
    manuscript = config.MANUSCRIPTS_DIR.relative_to(config.THESIS_REPO) / chunk["source_file"]
    passage = LibraryPassage(
        chunk_id=chunk["chunk_id"],
        key=chunk["key"],
        citable=chunk["citable"],
        title=chunk["title"],
        year=chunk["year"],
        doi=chunk["doi"],
        manuscript=str(manuscript),
        section=chunk["section"],
        text=chunk["text"],
        cites=chunk["cites"],
    )
    if score is not None:
        passage["score"] = round(score, 3)
    return passage


def thesis_sources(chunk_list: list[ThesisChunk]) -> dict[str, str]:
    """The text of each file these chunks come from, as it stands now, read once per call so that edits made since the server started count. A file that is gone has no text."""
    sources = {}
    for file in {chunk["file"] for chunk in chunk_list}:
        path = config.THESIS_REPO / file
        sources[file] = path.read_text(encoding="utf-8") if path.is_file() else ""
    return sources


def thesis_paragraph(chunk: ThesisChunk, sources: dict[str, str], score: float | None = None) -> ThesisParagraph:
    """A thesis chunk as the tools return it, with whether its line range still holds and, from a search, its score."""
    paragraph = ThesisParagraph(
        chunk_id=chunk["chunk_id"],
        file=chunk["file"],
        line_start=chunk["line_start"],
        line_end=chunk["line_end"],
        section=chunk["section"],
        kind=chunk["kind"],
        text=chunk["text"],
        keys=chunk["keys"],
        current=thesis.is_current(chunk, sources[chunk["file"]]),
    )
    if score is not None:
        paragraph["score"] = round(score, 3)
    return paragraph


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def search_library(
    question: str,
    k: Annotated[int, Field(ge=1, le=config.SERVER_MAX_HITS)] = 8,
    citable_only: bool = True,
) -> list[LibraryPassage]:
    """Find passages in the papers of the thesis library that bear on a question or a claim.

    Write it as a sentence, the way a paper would state the answer; names, numbers and species help. Returns the k best passages, best first, each with the paper's citation key, title, year and DOI, its manuscript file in the thesis repository (the PDF has the same name with .pdf), the section, the text, and the references the passage itself cites. score is the reranker's relevance, from 0 to 1: near 0, the library has nothing on the subject; high means the passage is on the subject, not that it answers the question, so read the text.

    A passage is a lead, not a citation: read the claim in the paper itself before citing it, and cite only papers marked citable, by their key. A reference under cites is a paper this one cites, which may not be in the library. When no passage answers the question, the library may not hold the answer; say so rather than stretch a passage to fit. citable_only=false also returns documents that are not citable sources, such as the style book, national guidelines and a departmental thesis.
    """
    # The reranker orders only the fused top RERANK_DEPTH, so take all of those and filter them: filtering before the cut would leave fewer than k whenever documents that are not citable rank high.
    hits = run("library", question, config.RERANK_DEPTH)
    kept = [hit for hit in hits if hit["chunk"]["citable"] or not citable_only][:k]
    return [library_passage(hit["chunk"], hit["score"]) for hit in kept]


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def search_thesis(text: str, k: Annotated[int, Field(ge=1, le=config.SERVER_MAX_HITS)] = 5) -> list[ThesisParagraph]:
    """Find the paragraphs and captions of the thesis that discuss a claim, a passage from a paper, or a reviewer's comment.

    Returns the k best, best first, each with its file in the thesis repository, line range, section path, the citation keys it cites, and its text as plain text, without the citations. current is false when those lines have changed since the thesis was indexed: the line range no longer holds, so find the paragraph by its text, and run literature-rag-chunk and then literature-rag-build to refresh the index.

    A thesis paragraph shows what the thesis says, never that it is true: it is not evidence for a claim.
    """
    hits = run("thesis", text, k)
    sources = thesis_sources([hit["chunk"] for hit in hits])
    return [thesis_paragraph(hit["chunk"], sources, hit["score"]) for hit in hits]


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_context(
    chunk_id: str, window: Annotated[int, Field(ge=0, le=config.SERVER_MAX_WINDOW)] = 1
) -> list[LibraryPassage] | list[ThesisParagraph]:
    """Read a hit in its context: the chunk with this id and up to window chunks on either side, in reading order, from the same paper or the same thesis chapter.

    Use it when a passage starts or stops mid-argument, or to read what comes before a number. chunk_id comes from search_library or search_thesis. The chunks come back as those tools return them, without a score.
    """
    with lock:
        for name in config.COLLECTIONS:
            found = collection(name)
            if chunk_id in found.rows:
                chunk_list = search.neighbours(found, chunk_id, window)
                break
        else:
            raise ToolError(
                f"No chunk {chunk_id!r}. Chunk ids come from search_library and search_thesis, and a thesis chunk's id changes when its lines move."
            )
    if name == "library":
        return [library_passage(chunk) for chunk in chunk_list]
    sources = thesis_sources(chunk_list)
    return [thesis_paragraph(chunk, sources) for chunk in chunk_list]


def main() -> None:
    """Entry point for `uv run literature-rag`: serve over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
