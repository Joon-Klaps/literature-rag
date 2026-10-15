"""Cut the library and the notes into passages, the units that search ranks, and write them next to the thesis chunks and the test pairs.

Run `uv run literature-rag-chunk` after `literature-rag-ingest`; `literature-rag-add` runs both. It writes up to four files under data/<project>/, the thesis's two only for a project with a thesis and the notes only for a project with notes (config.COLLECTIONS):

- data/chunks/library.jsonl: every paper in data/papers/ as passages. The abstract and each caption are passages of their own. Body paragraphs are merged within their section, in reading order, up to config.CHUNK_WORDS, and a paragraph over config.SPLIT_WORDS is first cut at sentence ends.
- data/chunks/thesis.jsonl: the thesis's paragraphs and captions, read by thesis.py.
- data/eval/thesis_pairs.jsonl: the thesis sentences that cite one paper of the library, the test questions of block 5.
- data/chunks/notes.jsonl: every note in the project's data/notes, read by notes.py, with its paragraphs merged within their section as a paper's are.

Chunking reads only data/papers/, the thesis and the notes and takes seconds, so every run redoes it in full.
"""

import json
import math
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from literature_rag import config, notes, text, thesis
from literature_rag.records import Chunk, ChunkKind, NoteChunk, Pair, Paper, Paragraph, ThesisChunk


def word_count(passage: str) -> int:
    return len(passage.split())


def split(passage: str, limit: int = config.CHUNK_WORDS) -> list[str]:
    """Cut a passage at sentence ends into the fewest pieces that each stay within limit words, of about equal length.

    A sentence longer than the limit is cut between words into even parts, as a last resort for text without full stops, such as a table that GROBID or pypdf read as prose.
    """
    units = []
    for sentence in text.sentences(passage):
        words = sentence.split()
        size = math.ceil(len(words) / math.ceil(len(words) / limit))
        units.extend(" ".join(words[start : start + size]) for start in range(0, len(words), size))
    target = word_count(passage) / math.ceil(word_count(passage) / limit)
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        size = word_count(unit)
        if current and count + size > limit:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(unit)
        count += size
        if count >= target:
            pieces.append(" ".join(current))
            current, count = [], 0
    if current:
        pieces.append(" ".join(current))
    return pieces


def pieces(passage: str) -> list[str]:
    """A passage as it enters a chunk: whole, cut at sentence ends when it is over SPLIT_WORDS, or nothing when it is empty."""
    if not passage.strip():
        return []
    return split(passage) if word_count(passage) > config.SPLIT_WORDS else [passage]


def join(group: list[Paragraph]) -> Paragraph:
    """Paragraphs as one passage, with a blank line between them and the citations of all of them."""
    cites = list(dict.fromkeys(rid for paragraph in group for rid in paragraph["cites"]))
    return Paragraph(text="\n\n".join(paragraph["text"] for paragraph in group), cites=cites)


def gather[Part](parts: list[Part], words: Callable[[Part], int]) -> list[list[Part]]:
    """Parts in reading order gathered into runs: each part joins the run before it while the run stays within CHUNK_WORDS, and starts a new one otherwise. words gives a part's length."""
    runs: list[list[Part]] = []
    count = 0
    for part in parts:
        size = words(part)
        if not runs or count + size > config.CHUNK_WORDS:
            runs.append([])
            count = 0
        runs[-1].append(part)
        count += size
    return runs


def merge(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """The paragraphs of one section as passages: in reading order, each joined to the passage before it while both fit within CHUNK_WORDS.

    A paragraph over SPLIT_WORDS is cut first, and each piece keeps all of its citations, since the parsers record them per paragraph.
    """
    parts = [Paragraph(text=piece, cites=p["cites"]) for p in paragraphs for piece in pieces(p["text"])]
    return [join(run) for run in gather(parts, lambda part: word_count(part["text"]))]


def chunk_paper(paper: Paper) -> list[Chunk]:
    """A paper's passages in reading order: the abstract, the body section by section, then the captions. Nothing is merged across sections."""
    passages: list[tuple[str, ChunkKind, Paragraph]] = [
        ("Abstract", "abstract", Paragraph(text=piece, cites=[])) for piece in pieces(paper["abstract"])
    ]
    for section in paper["sections"]:
        passages.extend((section["heading"], "body", passage) for passage in merge(section["paragraphs"]))
    for caption in paper["captions"]:
        passages.extend(
            (caption["label"], "caption", Paragraph(text=piece, cites=caption["cites"]))
            for piece in pieces(caption["text"])
        )
    return [
        Chunk(
            chunk_id=f"{paper['paper_id']}#{n}",
            paper_id=paper["paper_id"],
            key=paper["key"],
            citable=paper["citable"],
            title=paper["title"],
            year=paper["year"],
            doi=paper["doi"],
            source_file=paper["source_file"],
            section=section,
            kind=kind,
            text=passage["text"],
            cites=[paper["references"][rid] for rid in passage["cites"] if rid in paper["references"]],
        )
        for n, (section, kind, passage) in enumerate(passages)
    ]


def chunk_note(note: notes.Note) -> list[NoteChunk]:
    """A note's passages in reading order: the paragraphs of each section merged as a paper's are, each passage with the lines it spans. Nothing is merged across headings.

    A paragraph over SPLIT_WORDS is cut first, and each piece keeps the paragraph's lines, since the cut can fall inside a line.
    """
    lines = note.source.split("\n")
    passages: list[tuple[str, int, int, str]] = []
    for section in note.sections:
        parts = [(block, piece) for block in section.blocks for piece in pieces(block.text)]
        for run in gather(parts, lambda part: word_count(part[1])):
            start, end = run[0][0].start, run[-1][0].end
            passages.append((section.heading, start, end, "\n\n".join(piece for _, piece in run)))
    return [
        NoteChunk(
            chunk_id=f"{note.name}#{n}",
            note=note.name,
            file=note.file,
            line_start=start,
            line_end=end,
            section=heading,
            text=passage,
            hash=thesis.block_hash("\n".join(lines[start - 1 : end])),
        )
        for n, (heading, start, end, passage) in enumerate(passages)
    ]


def index_parts(chunk: Chunk | ThesisChunk | NoteChunk) -> tuple[str, str]:
    """A chunk as search sees it, in two parts: the header "Title. Section.", without the parts a chunk lacks, and the text. Thesis chunks have no title, and a note's title is its name.

    This is the only place the header is made, so that BM25 and every embedder see the same text. MedCPT reads the two parts as the (title, abstract) pair it was trained on; the others read them joined, as index_text gives them.
    """
    header = [part for part in (chunk.get("title") or chunk.get("note", ""), chunk["section"]) if part]
    return " ".join(part if part.endswith((".", "?", "!")) else f"{part}." for part in header), chunk["text"]


def index_text(chunk: Chunk | ThesisChunk | NoteChunk) -> str:
    """The text that BM25 indexes and Qwen3 embeds: "Title. Section. Text"."""
    return " ".join(part for part in index_parts(chunk) if part)


def library_keys(papers: list[Paper]) -> dict[str, str]:
    """The key of every citable paper in the library, lowercased, mapped to the key as the library spells it. Supplements share their paper's key, so a key counts once."""
    return {paper["key"].lower(): paper["key"] for paper in papers if paper["citable"] and paper["key"]}


def load_papers() -> list[Paper]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(config.PAPERS_DIR.glob("*.json"))]


def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarise_notes(note_list: list[notes.Note], note_chunks: list[NoteChunk]) -> str:
    """The notes' line of the chunking report: how many notes had text, and how long their passages are."""
    if not note_list:
        return f"\nNotes: none in {config.NOTES_DIR}"
    words = [word_count(chunk["text"]) for chunk in note_chunks] or [0]
    with_text = len({chunk["note"] for chunk in note_chunks})
    return (
        f"\nNotes: {len(note_chunks)} chunks from {with_text} notes ({len(note_list) - with_text} of {len(note_list)} files hold no text);"
        f" median {statistics.median(words):.0f} words, {sum(n < 50 for n in words) / len(words):.0%} under 50"
    )


def summarise(
    papers: list[Paper],
    library: list[Chunk],
    thesis_chunks: list[ThesisChunk],
    sentences: list[thesis.Sentence],
    pairs: list[Pair],
) -> list[str]:
    """The chunking report: chunks per kind and their length, what the thesis yielded, and how many of its citing sentences became pairs."""
    lines = [f"Library: {len(library)} chunks from {len(papers)} papers"]
    lines.append(f"  {'kind':<10}{'chunks':>8}{'median words':>14}{'150-400':>9}{'<150':>7}{'>400':>7}")
    for kind in ("abstract", "body", "caption", "all"):
        words = [word_count(chunk["text"]) for chunk in library if kind in ("all", chunk["kind"])]
        if not words:
            continue
        within = sum(150 <= n <= 400 for n in words) / len(words)
        short = sum(n < 150 for n in words) / len(words)
        long = sum(n > 400 for n in words) / len(words)
        lines.append(
            f"  {kind:<10}{len(words):>8}{statistics.median(words):>14.0f}{within:>9.0%}{short:>7.0%}{long:>7.0%}"
        )
    if "thesis" not in config.COLLECTIONS:
        lines.append(
            f"\nNo thesis: {config.PROJECT_NAME} has no {config.THESIS_FLS.name}, so no thesis chunks and no test pairs"
        )
        return lines
    files = len({chunk["file"] for chunk in thesis_chunks})
    kinds = Counter(chunk["kind"] for chunk in thesis_chunks)
    cited = {key.lower() for chunk in thesis_chunks for key in chunk["keys"]}
    in_library = cited & library_keys(papers).keys()
    lines.append(
        f"\nThesis: {len(thesis_chunks)} chunks from {files} files ({kinds['paragraph']} paragraphs, {kinds['caption']} captions);"
        f" {len(cited)} keys cited, {len(in_library)} of them citable papers in the library"
    )
    citing = [sentence for sentence in sentences if sentence.keys]
    single = [sentence for sentence in citing if len(sentence.keys) == 1]
    lines.append(
        f"Sentences: {len(sentences)}, {len(citing)} with a citation, {len(single)} citing exactly one key,"
        f" {len(single) - len(pairs)} of those outside the library or repeated"
    )
    lines.append(
        f"Pairs: {len(pairs)} for {len({pair['key'] for pair in pairs})} papers, {sum(bool(pair['numbers']) for pair in pairs)} with a number"
    )
    return lines


def build() -> list[str]:
    """Chunk the library and, where the project has them, the thesis and the notes, make the test pairs, write the files, and return the report."""
    papers = load_papers()
    library = [chunk for paper in papers for chunk in chunk_paper(paper)]
    write_jsonl(config.LIBRARY_CHUNKS_FILE, library)
    thesis_chunks, sentences, pairs = [], [], []
    if "thesis" in config.COLLECTIONS:
        thesis_chunks, sentences = thesis.read_thesis()
        pairs = thesis.make_pairs(sentences, library_keys(papers))
        write_jsonl(config.THESIS_CHUNKS_FILE, thesis_chunks)
        write_jsonl(config.THESIS_PAIRS_FILE, pairs)
    lines = summarise(papers, library, thesis_chunks, sentences, pairs)
    if "notes" in config.COLLECTIONS:
        note_list = notes.read_notes()
        note_chunks = [chunk for note in note_list for chunk in chunk_note(note)]
        write_jsonl(config.NOTES_CHUNKS_FILE, note_chunks)
        lines.append(summarise_notes(note_list, note_chunks))
    return lines


def main() -> None:
    """Entry point for `uv run literature-rag-chunk`."""
    config.require_project()
    print("\n".join(build()))
    files = [config.LIBRARY_CHUNKS_FILE]
    if "thesis" in config.COLLECTIONS:
        files += [config.THESIS_CHUNKS_FILE, config.THESIS_PAIRS_FILE]
    if "notes" in config.COLLECTIONS:
        files.append(config.NOTES_CHUNKS_FILE)
    print(f"\nWrote {', '.join(str(path) for path in files)}")
