# Synthetic retrieval benchmark — 1.4.0

This is an aggregate-only summary of the locally verified synthetic evaluation,
not a production accuracy claim or a separately reproducible public benchmark.

| Measure | Verified result |
| --- | --- |
| Synthetic text records | 60 |
| Synthetic images | 40 |
| Queries | 100 |
| Text-query top-1 hits | 50 / 50 |
| Combined macro Recall@5 (answerable queries) | 0.9889 |
| No-answer queries returning false positives | 9 / 10 |

**Known weaknesses:** abstention is poor: most no-answer queries still return
plausible-looking matches. Exact color/shape matching is unreliable, and visual
similarity must not be treated as proof of an exact match. The release does not
include new threshold, reranking, or color-matching fixes.

The corpus is small and synthetic, with controlled text, rendered tables and
simple visual patterns. OCR availability, fonts, language, layout and model
artifacts affect results. Recall measures ranked retrieval, not factual answer
accuracy. Image paths still require native `vision_analyze` to inspect pixels;
no automatic pixel injection or external vision-model answer evaluation is claimed.

The local harness depends on an operator-specific environment and is intentionally
not published. Run directories, raw reports, manifests, per-query logs, database
snapshots and rollback archives are also excluded: they can contain private
state and are not safe release assets. Only this sanitized summary is public.
