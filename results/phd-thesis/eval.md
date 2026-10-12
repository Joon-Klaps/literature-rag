# Evaluation

Run on 2026-10-06 by `uv run literature-rag-eval`. Each cell is the mean over the queries with its 95% bootstrap interval (1000 resamples). A query's ranking counts each paper once, at its best chunk; hit@k is the share of queries whose cited paper is among the first k papers, and mrr@10 the mean of 1 / place within the first 10.

## Library: 169 thesis sentences that cite one paper

The column numbers is the share of the 62 sentences that carry a number in which the top 5 chunks from the cited paper hold every one of its numbers.

| configuration | hit@1 | hit@5 | mrr@10 | numbers |
| --- | --- | --- | --- | --- |
| bm25 | 0.43 (0.36–0.50) | 0.70 (0.63–0.77) | 0.55 (0.49–0.61) | 0.52 |
| qwen3 | 0.52 (0.45–0.60) | 0.75 (0.69–0.82) | 0.62 (0.56–0.69) | 0.40 |
| medcpt | 0.33 (0.26–0.39) | 0.62 (0.54–0.70) | 0.44 (0.38–0.51) | 0.24 |
| bm25+qwen3 | 0.54 (0.47–0.61) | 0.78 (0.71–0.83) | 0.64 (0.58–0.70) | 0.45 |
| bm25+medcpt | 0.43 (0.36–0.51) | 0.75 (0.68–0.81) | 0.57 (0.51–0.63) | 0.47 |
| bm25+qwen3+medcpt | 0.49 (0.41–0.56) | 0.80 (0.73–0.86) | 0.62 (0.56–0.68) | 0.44 |
| bm25+qwen3, reranked top 20 | 0.56 (0.49–0.62) | 0.79 (0.73–0.85) | 0.65 (0.59–0.71) | 0.50 |
| bm25+qwen3, reranked top 50 | 0.57 (0.50–0.64) | 0.79 (0.72–0.85) | 0.67 (0.61–0.73) | 0.58 |
| bm25+qwen3, reranked top 100 | 0.56 (0.50–0.63) | 0.78 (0.71–0.83) | 0.66 (0.60–0.72) | 0.58 |
| bm25 on raw pages | 0.46 (0.38–0.53) | 0.71 (0.64–0.78) | 0.56 (0.49–0.62) |  |
| qmd search | 0.05 (0.02–0.09) | 0.06 (0.03–0.09) | 0.06 (0.03–0.09) |  |
| qmd query | 0.31 (0.24–0.38) | 0.67 (0.61–0.75) | 0.46 (0.41–0.52) |  |

Best by mrr@10: bm25+qwen3, reranked top 50, at 0.67 (0.61–0.73). The best fusion, which the reranker reads, was bm25+qwen3. Differences on the same queries, with paired bootstrap intervals:

- bm25+qwen3, reranked top 50 against qmd search: hit@1 +0.52 (+0.44 to +0.59), hit@5 +0.73 (+0.66 to +0.79), mrr@10 +0.61 (+0.54 to +0.67).
- bm25+qwen3, reranked top 50 against qmd query: hit@1 +0.26 (+0.18 to +0.35), hit@5 +0.11 (+0.03 to +0.18), mrr@10 +0.20 (+0.14 to +0.27).
- bm25 against bm25 on raw pages: hit@1 -0.02 (-0.08 to +0.03), hit@5 -0.01 (-0.07 to +0.04), mrr@10 -0.01 (-0.05 to +0.04).
- bm25+qwen3, reranked top 50 against bm25+qwen3: hit@1 +0.04 (-0.04 to +0.11), hit@5 +0.01 (-0.04 to +0.07), mrr@10 +0.03 (-0.02 to +0.08).
- qmd search returned nothing for 119 of 169 queries and failed on 0.
- qmd query returned nothing for 0 of 169 queries and failed on 0.

## Thesis: 20 paragraph summaries from PARAGRAPH-INDEX.md

Drawn from the 85 summaries whose cited keys pick out exactly one current paragraph, which is the answer.

| configuration | hit@1 | hit@5 | mrr@10 |
| --- | --- | --- | --- |
| bm25 | 0.95 (0.85–1.00) | 1.00 (1.00–1.00) | 0.97 (0.93–1.00) |
| qwen3 | 0.90 (0.75–1.00) | 1.00 (1.00–1.00) | 0.94 (0.86–1.00) |
| medcpt | 0.80 (0.60–0.95) | 0.95 (0.85–1.00) | 0.88 (0.75–0.97) |
| bm25+qwen3 | 0.95 (0.85–1.00) | 1.00 (1.00–1.00) | 0.97 (0.93–1.00) |
| bm25+medcpt | 0.90 (0.75–1.00) | 0.95 (0.85–1.00) | 0.93 (0.82–1.00) |
| bm25+qwen3+medcpt | 0.90 (0.75–1.00) | 1.00 (1.00–1.00) | 0.95 (0.88–1.00) |
| bm25+qwen3, reranked top 20 | 0.95 (0.85–1.00) | 1.00 (1.00–1.00) | 0.97 (0.93–1.00) |
| bm25+qwen3, reranked top 50 | 0.95 (0.85–1.00) | 1.00 (1.00–1.00) | 0.97 (0.93–1.00) |
| bm25+qwen3, reranked top 100 | 0.95 (0.85–1.00) | 1.00 (1.00–1.00) | 0.97 (0.93–1.00) |

## Realistic set: 12 literature points from the review rounds (0 confirmed)

The place of the first paper that answers each point, or – when none is in the ranking.

| question | bm25 | qwen3 | medcpt | bm25+qwen3 | bm25+qwen3, reranked top 50 | qmd search | qmd query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PL0.1.1/root-to-tip | 3 | 2 | 6 | 1 | 1 | – | 6 |
| PL0.1.1/tmrca | 4 | 2 | 2 | 2 | 1 | – | 1 |
| PL0.1.1/capture | 1 | 3 | 2 | 4 | 1 | – | 5 |
| PL0.1.1/fm-index | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| PL0.1.2/ast | 1 | 6 | 13 | 2 | 1 | – | 3 |
| PL0.2.0/graphs | 4 | 5 | 3 | 4 | 2 | – | 5 |
| LK0.2.2/diagnostics | 1 | 1 | 2 | 1 | 1 | – | 1 |
| LK0.2.2/human-reads | 1 | 1 | 1 | 1 | 1 | – | 3 |
| LK0.2.2/cost | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LK0.3.0/8 | 7 | 8 | 1 | 17 | 5 | – | 7 |
| EKL0.3.0/13 | 1 | 1 | 2 | 1 | 1 | 1 | 5 |
| EKL0.3.0/22 | 9 | 1 | 2 | 4 | 1 | – | 1 |
