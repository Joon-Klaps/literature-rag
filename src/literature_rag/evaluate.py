"""Measure how well each search configuration finds the paper a thesis sentence cites, against qmd on the same questions.

    uv run literature-rag-eval

Three sets of questions:

- The test pairs (data/eval/thesis_pairs.jsonl, from block 3): a thesis sentence that cites one paper of the library, whose answer is that paper.
- The realistic set (data/eval/realistic.jsonl): literature points from the review rounds, each with the papers that answer it. Too few for a rate, so each is reported on its own.
- The thesis check: paragraph summaries from the thesis repository's PARAGRAPH-INDEX.md, searched in the thesis collection, whose answer is the one current paragraph that cites the same keys.

A query's ranking of chunks becomes a ranking of papers, each paper counted once at the place of its best chunk, and the place of the right paper gives hit@1, hit@5 and MRR@10, each with a bootstrap interval. The configurations are BM25, each embedder, their fusions, and the best fusion reranked at several depths. The baselines are qmd's BM25 (`qmd search`) and its full hybrid (`qmd query`, with query expansion and reranking) over the unmodified Markdown, and our own BM25 over that same Markdown, page by page, which shows what the structured text adds. The reranker's scores and qmd's results are slow to make, so they are kept in data/eval/runs/ and reused.

Writes results/eval.json and results/eval.md: numbers, configuration names, keys and question ids, never the text of a question, since the questions quote the unpublished thesis and its reviews.
"""

import argparse
import datetime
import hashlib
import json
import operator
import random
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
from rank_bm25 import BM25Okapi

from literature_rag import chunks, config, manifest, markdown, models, search, thesis
from literature_rag.records import Chunk, ManifestRow, Pair, RealisticQuestion, ThesisChunk

MRR = f"mrr@{config.EVAL_MRR_DEPTH}"
METRICS = [*(f"hit@{k}" for k in config.EVAL_HIT_AT), MRR]
# A paragraph entry of PARAGRAPH-INDEX.md that cites something: "- `L23-25` — What the paragraph argues. Cites: Key1, Key2."
INDEX_ENTRY = re.compile(r"^- `L[\d-]+` — (?P<summary>.+?) Cites: (?P<keys>.+?)\.?$")
# The names of the baselines, next to those of the configurations, which are their methods joined by "+".
RAW_BM25 = "bm25 on raw pages"
QMD_COMMANDS = {"qmd search": "search", "qmd query": "query"}


class Query(NamedTuple):
    """One test question: its id, the text that is searched, the papers (or thesis chunks) that answer it, and for a test pair the numbers in it."""

    query_id: str
    text: str
    answers: frozenset[str]
    numbers: tuple[str, ...] = ()


def configuration_name(methods: Sequence[str]) -> str:
    return "+".join(methods)


def reranked_name(methods: Sequence[str], depth: int) -> str:
    return f"{configuration_name(methods)}, reranked top {depth}"


def paper_of(chunk: Chunk) -> str:
    """The paper a library chunk counts for: its citation key, which a supplement shares with its paper, or its paper id when it has no key."""
    return chunk["key"] or chunk["paper_id"]


def paper_ranking(papers: Iterable[str]) -> list[str]:
    """A ranking of chunks, given as the paper of each, as a ranking of papers: each paper once, at the place of its best chunk."""
    return list(dict.fromkeys(papers))


def place(ranking: Sequence[str], answers: frozenset[str]) -> int | None:
    """The place, from 1, of the first answer in a ranking, or None when there is none."""
    return next((n for n, item in enumerate(ranking, start=1) if item in answers), None)


def hit(found: int | None, k: int) -> float:
    """1 when the answer is among the first k, 0 otherwise."""
    return 1.0 if found is not None and found <= k else 0.0


def reciprocal_rank(found: int | None, depth: int = config.EVAL_MRR_DEPTH) -> float:
    """1 / place when the answer is within depth, and 0 below it or when it is missing: 1 at the top, 0.5 second, 0.1 tenth."""
    return 1 / found if found is not None and found <= depth else 0.0


def metric_values(places: Sequence[int | None]) -> dict[str, list[float]]:
    """Each metric's value for each query, from the place of its answer."""
    values = {f"hit@{k}": [hit(found, k) for found in places] for k in config.EVAL_HIT_AT}
    values[MRR] = [reciprocal_rank(found) for found in places]
    return values


def resamples(n: int, count: int = config.BOOTSTRAP_RESAMPLES, seed: int = config.BOOTSTRAP_SEED) -> np.ndarray:
    """Bootstrap resamples of n queries, one per row, each n positions drawn with replacement. The fixed seed gives every configuration the same resamples, so two of them can be compared query by query."""
    return np.random.default_rng(seed).integers(0, n, size=(count, n))


def interval(values: Sequence[float], rows: np.ndarray) -> tuple[float, float]:
    """The 95% bootstrap interval of the mean: the 2.5th and 97.5th percentiles of the means of the resamples."""
    means = np.asarray(values, dtype=float)[rows].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarise(places: Sequence[int | None]) -> dict[str, dict[str, float]]:
    """Each metric's mean over the queries, with its bootstrap interval."""
    rows = resamples(len(places))
    summary = {}
    for name, values in metric_values(places).items():
        low, high = interval(values, rows)
        summary[name] = {"mean": float(np.mean(values)), "low": low, "high": high}
    return summary


def difference(places: Sequence[int | None], baseline: Sequence[int | None]) -> dict[str, dict[str, float]]:
    """How much better one configuration does than another on the same queries, per metric, with a paired bootstrap interval: each resample takes the same queries for both.

    The two intervals of a pair of configurations overlap more often than their difference is in doubt, since most queries are easy or hard for both. An interval of the difference that leaves out 0 is a difference the queries support.
    """
    rows = resamples(len(places))
    ours, theirs = metric_values(places), metric_values(baseline)
    result = {}
    for name in ours:
        delta = np.asarray(ours[name]) - np.asarray(theirs[name])
        low, high = interval(delta, rows)
        result[name] = {"mean": float(delta.mean()), "low": low, "high": high}
    return result


def numbers_found(numbers: Sequence[str], passages: Iterable[str]) -> bool:
    """Whether every number of a test pair appears in the passages, whose numbers are read as thesis.numbers() reads the sentence's, so that "1,234" in a paper matches 1234."""
    present = {number for passage in passages for number in thesis.numbers(passage)}
    return set(numbers) <= present


def rerank_order(rows: Sequence[int], scores: dict[int, float], depth: int) -> list[int]:
    """A fused ranking with its first depth rows reordered by the reranker's scores, highest first, and the rest after them in fused order. Equal scores keep the fused order."""
    head = sorted(rows[:depth], key=lambda row: -scores[row])
    return head + list(rows[depth:])


def file_papers(rows: list[ManifestRow]) -> dict[str, str]:
    """The paper each Markdown file counts for, as paper_of() counts a chunk: its citation key, or its paper id when it has none. A file that repeats another counts as the paper it repeats."""
    by_file = {row["source_file"]: row for row in rows}
    return {
        name: row["key"] or by_file.get(row["duplicate_of"] or "", row)["paper_id"] for name, row in by_file.items()
    }


def index_entries(index_markdown: str) -> list[tuple[str, list[str]]]:
    """The paragraph entries of PARAGRAPH-INDEX.md that cite something, as (summary, keys). Figures, tables and paragraphs without citations are left out, since only cited keys can tie a summary to a current paragraph."""
    entries = []
    for line in index_markdown.splitlines():
        match = INDEX_ENTRY.match(line)
        if match:
            entries.append((match["summary"], [key.strip() for key in match["keys"].split(",")]))
    return entries


def thesis_questions(
    entries: list[tuple[str, list[str]]],
    thesis_chunks: list[ThesisChunk],
    size: int = config.THESIS_CHECK_SIZE,
    seed: int = config.BOOTSTRAP_SEED,
) -> tuple[list[Query], int]:
    """Questions for the thesis check, and how many entries could have been one.

    An entry qualifies when its cited keys are exactly those of one current paragraph, which becomes its answer; keys are compared without regard to case, as the pairs compare them. A summary whose paragraph has since changed its citations, or shares them with another paragraph, has no single answer. Of those that qualify, size are drawn with a fixed seed and kept in the index's order.
    """
    paragraphs: dict[frozenset[str], list[str]] = {}
    for chunk in thesis_chunks:
        if chunk["kind"] == "paragraph" and chunk["keys"]:
            paragraphs.setdefault(frozenset(key.lower() for key in chunk["keys"]), []).append(chunk["chunk_id"])
    eligible = []
    for n, (summary, keys) in enumerate(entries):
        matches = paragraphs.get(frozenset(key.lower() for key in keys), [])
        if len(matches) == 1:
            eligible.append(Query(f"index/{n}", summary, frozenset(matches)))
    chosen = random.Random(seed).sample(range(len(eligible)), min(size, len(eligible)))
    return [eligible[i] for i in sorted(chosen)], len(eligible)


def pair_queries(pairs: list[Pair]) -> list[Query]:
    return [Query(pair["pair_id"], pair["query"], frozenset([pair["key"]]), tuple(pair["numbers"])) for pair in pairs]


def realistic_queries(questions: list[RealisticQuestion], library_keys: set[str]) -> list[Query]:
    """The realistic questions as queries. A key that is not a citable paper of the library is a typo or a paper still to fetch, and stops the run."""
    unknown = sorted({key for question in questions for key in question["keys"]} - library_keys)
    if unknown:
        raise SystemExit(f"{config.REALISTIC_FILE.name}: not citable papers of the library: {', '.join(unknown)}")
    return [Query(question["question_id"], question["question"], frozenset(question["keys"])) for question in questions]


def read_run(name: str) -> dict:
    path = config.RUNS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_run(name: str, run: dict) -> None:
    """Save a run through a temporary file, so that an interrupted write never leaves half a file."""
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RUNS_DIR / f"{name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(run), encoding="utf-8")
    temporary.replace(path)


def progress(label: str, done: int, total: int, start: float) -> None:
    if done % 10 == 0 or done == total:
        print(f"  {label}: {done}/{total}, {time.perf_counter() - start:.0f} s", flush=True)


def method_rankings(collection: search.Collection, queries: list[Query]) -> dict[str, list[search.Ranking]]:
    """Each method's ranking for every query: BM25 one query at a time, and each embedder over all the queries in one batch."""
    texts = [query.text for query in queries]
    rankings = {"bm25": [search.bm25(collection, text) for text in texts]}
    for model in config.EMBEDDING_MODELS:
        vectors = models.load_embedder(model).encode_queries(texts, config.QUERY_INSTRUCTIONS[collection.name])
        rankings[model] = [search.dense(collection, model, vector) for vector in vectors]
    return rankings


def configuration_rows(collection: search.Collection, queries: list[Query]) -> dict[str, list[list[int]]]:
    """Each configuration's ranking of chunk rows for every query: the methods alone and fused, as search() combines them."""
    rankings = method_rankings(collection, queries)
    return {
        configuration_name(methods): [
            [row for row, _ in search.fuse([rankings[method][n] for method in methods])] for n in range(len(queries))
        ]
        for methods in config.EVAL_CONFIGURATIONS
    }


def rerank_scores(
    collection: search.Collection, queries: list[Query], shortlists: list[list[int]]
) -> list[dict[int, float]]:
    """The reranker's score for every row of each query's shortlist, taken from the run kept in data/eval/runs/ where it has one.

    Scores are kept by a hash of the query and the passage's text, so they stay valid when chunking renumbers the chunks, and each query's new scores are saved as they arrive, so an interrupted run resumes.
    """
    name = f"rerank-{collection.name}"
    run = read_run(name)
    start = time.perf_counter()
    result = []
    for n, (query, rows) in enumerate(zip(queries, shortlists, strict=True), start=1):
        passages = [chunks.index_text(collection.chunks[row]) for row in rows]
        keys = [hashlib.sha256(f"{query.text}\0{passage}".encode()).hexdigest()[:20] for passage in passages]
        missing = [i for i, key in enumerate(keys) if key not in run]
        if missing:
            scores = search.cross_encode(models.load_reranker(), query.text, [passages[i] for i in missing])
            run.update({keys[i]: float(score) for i, score in zip(missing, scores, strict=True)})
            write_run(name, run)
            progress(f"reranking {collection.name}", n, len(queries), start)
        result.append({row: run[key] for row, key in zip(rows, keys, strict=True)})
    return result


def reranked_rows(
    collection: search.Collection, queries: list[Query], rows: list[list[int]]
) -> dict[int, list[list[int]]]:
    """A fused ranking per query reranked at each depth of config.EVAL_RERANK_DEPTHS. The deepest shortlist is scored once, and the shallower ones reuse its scores."""
    scores = rerank_scores(collection, queries, [ranking[: max(config.EVAL_RERANK_DEPTHS)] for ranking in rows])
    return {
        depth: [rerank_order(ranking, score, depth) for ranking, score in zip(rows, scores, strict=True)]
        for depth in config.EVAL_RERANK_DEPTHS
    }


def qmd(command: str, query: str) -> list[str]:
    """qmd's top config.QMD_RESULTS documents for a query, as file names in its collection. Raises CalledProcessError when qmd fails."""
    result = subprocess.run(
        ["qmd", command, query, "--json", "-n", str(config.QMD_RESULTS), "-c", config.QMD_COLLECTION],
        capture_output=True,
        text=True,
        check=True,
    )
    prefix = f"qmd://{config.QMD_COLLECTION}/"
    return [document["file"].removeprefix(prefix) for document in json.loads(result.stdout)]


def qmd_runs(command: str, queries: list[Query]) -> tuple[list[list[str] | None], int]:
    """qmd's file names for each query, taken from the run kept in data/eval/runs/ where it has one, and the number of queries qmd failed on, which count as finding nothing.

    `qmd query` loads its models and expands every query with a small language model, about 15 seconds a question, so each result is saved as it arrives and an interrupted run resumes. A failure is not saved, so the next run tries again.
    """
    name = f"qmd-{command}"
    run = read_run(name)
    start = time.perf_counter()
    failed = 0
    for n, query in enumerate(queries, start=1):
        if query.text in run:
            continue
        try:
            run[query.text] = qmd(command, query.text)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            failed += 1
            continue
        write_run(name, run)
        progress(f"qmd {command}", n, len(queries), start)
    return [run.get(query.text) for query in queries], failed


def raw_page_rankings(rows: list[ManifestRow], queries: list[Query]) -> list[list[str]]:
    """Our BM25 over the text qmd searches: every Markdown file in data/manuscripts, duplicates included, one passage per page and one for the abstract, each headed by the title. Returns each query's ranking of papers.

    The fallback parser reads the files, which removes the metadata block and the page headings and joins letters to their ligatures, but leaves the text as pypdf extracted it: no sections, no captions, and citation numbers fused to words.
    """
    papers = file_papers(rows)
    owners: list[str] = []
    passages: list[list[str]] = []
    for row in rows:
        parsed = markdown.parse(manifest.read_markdown(config.MANUSCRIPTS_DIR / row["source_file"]))
        pages = [
            parsed["abstract"],
            *(page["text"] for section in parsed["sections"] for page in section["paragraphs"]),
        ]
        for page in pages:
            if page:
                owners.append(papers[row["source_file"]])
                passages.append(search.tokenize(f"{parsed['title']}. {page}"))
    index = BM25Okapi(passages)
    rankings = []
    for query in queries:
        scores = index.get_scores(list(dict.fromkeys(search.tokenize(query.text))))
        ranking = [row for row, score in search.top(scores, config.CANDIDATES) if score > 0]
        rankings.append(paper_ranking(owners[row] for row in ranking))
    return rankings


def places(rankings: Iterable[Iterable[str]], queries: list[Query]) -> list[int | None]:
    """The place of each query's answer in its ranking, counting each paper (or thesis chunk) once."""
    return [place(paper_ranking(ranking), query.answers) for ranking, query in zip(rankings, queries, strict=True)]


def chunk_places(
    collection: search.Collection,
    rows: list[list[int]],
    queries: list[Query],
    identify: Callable[[Chunk | ThesisChunk], str],
) -> list[int | None]:
    """places() for rankings of chunk rows, each chunk given by what identify() says it counts for: its paper, or for the thesis itself."""
    return places([[identify(collection.chunks[row]) for row in ranking] for ranking in rows], queries)


def number_share(collection: search.Collection, queries: list[Query], rows: list[list[int]]) -> float | None:
    """The share of the queries with numbers in which the top config.NUMBER_CHECK_DEPTH chunks from the right paper hold all of them. A query whose paper is not among those chunks fails."""
    checked = [
        numbers_found(
            query.numbers,
            [
                collection.chunks[row]["text"]
                for row in ranking[: config.NUMBER_CHECK_DEPTH]
                if paper_of(collection.chunks[row]) in query.answers
            ],
        )
        for query, ranking in zip(queries, rows, strict=True)
        if query.numbers
    ]
    return sum(checked) / len(checked) if checked else None


class LibraryRun(NamedTuple):
    """What evaluate_library() measures: the place of each query's answer per configuration and baseline, the number check per configuration, the fusion the reranker read, and what qmd returned nothing for or failed on."""

    places: dict[str, list[int | None]]
    numbers: dict[str, float | None]
    best_fused: tuple[str, ...]
    qmd: dict[str, dict[str, int]]


def evaluate_library(queries: list[Query], pair_count: int, use_qmd: bool) -> LibraryRun:
    """Every configuration and baseline on the library queries, of which the first pair_count are the test pairs.

    The best fusion, the one the reranker then reads, is chosen by MRR on the test pairs alone, never on the realistic questions that come after them.
    """
    collection = search.load_collection("library")
    rows = configuration_rows(collection, queries)
    found = {name: chunk_places(collection, ranking, queries, paper_of) for name, ranking in rows.items()}
    fused = [methods for methods in config.EVAL_CONFIGURATIONS if len(methods) > 1]
    best = max(fused, key=lambda methods: np.mean(metric_values(found[configuration_name(methods)][:pair_count])[MRR]))
    for depth, ranking in reranked_rows(collection, queries, rows[configuration_name(best)]).items():
        rows[reranked_name(best, depth)] = ranking
        found[reranked_name(best, depth)] = chunk_places(collection, ranking, queries, paper_of)
    numbers = {name: number_share(collection, queries, ranking) for name, ranking in rows.items()}

    manifest_rows = json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))
    found[RAW_BM25] = places(raw_page_rankings(manifest_rows, queries), queries)
    notes = {}
    if use_qmd:
        papers = file_papers(manifest_rows)
        for name, command in QMD_COMMANDS.items():
            files, failed = qmd_runs(command, queries)
            found[name] = places([[papers.get(file, file) for file in result or []] for result in files], queries)
            notes[name] = {"failed": failed, "empty": sum(1 for result in files if result == [])}
    return LibraryRun(found, numbers, best, notes)


def evaluate_thesis(best_fused: Sequence[str]) -> tuple[dict[str, list[int | None]], int]:
    """Every configuration on the thesis check, with the library's best fusion reranked, and the number of summaries the questions were drawn from."""
    collection = search.load_collection("thesis")
    entries = index_entries(config.PARAGRAPH_INDEX_FILE.read_text(encoding="utf-8"))
    queries, eligible = thesis_questions(entries, collection.chunks)
    rows = configuration_rows(collection, queries)
    for depth, ranking in reranked_rows(collection, queries, rows[configuration_name(best_fused)]).items():
        rows[reranked_name(best_fused, depth)] = ranking
    by_id = operator.itemgetter("chunk_id")
    return {name: chunk_places(collection, ranking, queries, by_id) for name, ranking in rows.items()}, eligible


def cell(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.2f} ({stat['low']:.2f}–{stat['high']:.2f})"


def signed(stat: dict[str, float]) -> str:
    return f"{stat['mean']:+.2f} ({stat['low']:+.2f} to {stat['high']:+.2f})"


def table(configurations: dict[str, dict], numbers: bool) -> list[str]:
    header = ["configuration", *METRICS] + (["numbers"] if numbers else [])
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for name, summary in configurations.items():
        row = [name, *(cell(summary[metric]) for metric in METRICS)]
        if numbers:
            share = summary.get("numbers")
            row.append("" if share is None else f"{share:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def report(results: dict) -> str:
    """results/eval.md: the tables of results/eval.json, with a few sentences generated from the numbers."""
    library, thesis_check, realistic = results["library"], results["thesis"], results["realistic"]
    configurations, best = library["configurations"], library["best"]
    lines = [
        "# Evaluation",
        "",
        f"Run on {results['date']} by `uv run literature-rag-eval`. Each cell is the mean over the queries with its 95% bootstrap interval ({config.BOOTSTRAP_RESAMPLES} resamples). A query's ranking counts each paper once, at its best chunk; hit@k is the share of queries whose cited paper is among the first k papers, and {MRR} the mean of 1 / place within the first {config.EVAL_MRR_DEPTH}.",
        "",
        f"## Library: {library['queries']} thesis sentences that cite one paper",
        "",
        f"The column numbers is the share of the {library['with_numbers']} sentences that carry a number in which the top {config.NUMBER_CHECK_DEPTH} chunks from the cited paper hold every one of its numbers.",
        "",
        *table(configurations, numbers=True),
        "",
        f"Best by {MRR}: {best}, at {cell(configurations[best][MRR])}. The best fusion, which the reranker reads, was {library['best_fused']}. Differences on the same queries, with paired bootstrap intervals:",
        "",
    ]
    for name, delta in library["differences"].items():
        lines.append(f"- {name}: {', '.join(f'{metric} {signed(delta[metric])}' for metric in METRICS)}.")
    for name in QMD_COMMANDS:
        if name in library["qmd"]:
            note = library["qmd"][name]
            lines.append(
                f"- {name} returned nothing for {note['empty']} of {library['queries']} queries and failed on {note['failed']}."
            )
    lines += [
        "",
        f"## Thesis: {thesis_check['queries']} paragraph summaries from PARAGRAPH-INDEX.md",
        "",
        f"Drawn from the {thesis_check['eligible']} summaries whose cited keys pick out exactly one current paragraph, which is the answer.",
        "",
        *table(thesis_check["configurations"], numbers=False),
        "",
        f"## Realistic set: {len(realistic['places'])} literature points from the review rounds ({realistic['confirmed']} confirmed)",
        "",
        "The place of the first paper that answers each point, or – when none is in the ranking.",
        "",
    ]
    columns = realistic["columns"]
    lines += ["| question | " + " | ".join(columns) + " |", "| --- | " + " | ".join("---" for _ in columns) + " |"]
    for question_id, found in realistic["places"].items():
        lines.append(
            f"| {question_id} | "
            + " | ".join("–" if found[name] is None else str(found[name]) for name in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Entry point for `uv run literature-rag-eval`."""
    parser = argparse.ArgumentParser(
        description="Evaluate every search configuration and qmd on the test questions, and write results/."
    )
    parser.add_argument(
        "--no-qmd",
        action="store_true",
        help="leave out the qmd baselines (qmd query takes about 15 s a question the first time)",
    )
    args = parser.parse_args()

    pairs = pair_queries(chunks.read_jsonl(config.THESIS_PAIRS_FILE))
    library_keys = {
        chunk["key"] for chunk in search.load_collection("library").chunks if chunk["citable"] and chunk["key"]
    }
    questions = chunks.read_jsonl(config.REALISTIC_FILE) if config.REALISTIC_FILE.exists() else []
    realistic = realistic_queries(questions, library_keys)
    print(f"{len(pairs)} test pairs, {len(realistic)} realistic questions", flush=True)

    run = evaluate_library(pairs + realistic, len(pairs), use_qmd=not args.no_qmd)
    on_pairs = {name: found[: len(pairs)] for name, found in run.places.items()}
    configurations = {
        name: summarise(found) | ({"numbers": run.numbers[name]} if name in run.numbers else {})
        for name, found in on_pairs.items()
    }
    best = max(run.numbers, key=lambda name: configurations[name][MRR]["mean"])
    fused, reranked = configuration_name(run.best_fused), reranked_name(run.best_fused, config.RERANK_DEPTH)
    # Three questions: does the best configuration beat qmd, what does the structured text add to BM25, and what does the reranker add to the fusion it reads.
    comparisons = [(best, name) for name in QMD_COMMANDS if name in on_pairs] + [("bm25", RAW_BM25), (reranked, fused)]
    differences = {
        f"{ours} against {theirs}": difference(on_pairs[ours], on_pairs[theirs]) for ours, theirs in comparisons
    }

    print("Thesis check", flush=True)
    thesis_found, eligible = evaluate_thesis(run.best_fused)

    singles = [configuration_name(methods) for methods in config.EVAL_CONFIGURATIONS if len(methods) == 1]
    columns = [*singles, fused, reranked, *(name for name in QMD_COMMANDS if name in run.places)]
    results = {
        "date": datetime.datetime.now().astimezone().date().isoformat(),
        "library": {
            "queries": len(pairs),
            "with_numbers": sum(1 for query in pairs if query.numbers),
            "best": best,
            "best_fused": fused,
            "configurations": configurations,
            "differences": differences,
            "qmd": run.qmd,
        },
        "thesis": {
            "queries": len(next(iter(thesis_found.values()))),
            "eligible": eligible,
            "configurations": {name: summarise(found) for name, found in thesis_found.items()},
        },
        "realistic": {
            "confirmed": sum(1 for question in questions if question["confirmed"]),
            "columns": columns,
            "places": {
                query.query_id: {name: run.places[name][len(pairs) + n] for name in columns}
                for n, query in enumerate(realistic)
            },
        },
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "eval.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    markdown_report = report(results)
    (config.RESULTS_DIR / "eval.md").write_text(markdown_report, encoding="utf-8")
    print(markdown_report)
