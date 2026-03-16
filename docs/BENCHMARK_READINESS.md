# Benchmark Readiness Assessment

**Date:** 2026-03-16  
**Source run:** `uv run archex benchmark run` on 25 benchmark tasks  
**Purpose:** convert the current benchmark signal into a product-readiness position and an execution policy for retrieval strategy work.

---

## Bottom Line

`archex` is not yet reliable enough to expect strong user adoption as a primary code-understanding system.

The issue is not that retrieval never works. It clearly works on some task classes. The issue is that benchmark performance still has too much variance across repositories and query types. That variance will be experienced by users as unreliability.

The current benchmark indicates:

| Strategy | Recall | Precision | F1 | MRR | Mean Wall Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `archex_query` | 0.447 | 0.335 | 0.380 | 0.700 | 4.48s |
| `archex_query_fusion` | 0.453 | 0.353 | 0.395 | 0.780 | 679.06s |
| `cross_layer_fusion` | 0.407 | 0.322 | 0.351 | 0.693 | 2.97s |

Those numbers are still below the threshold required for user trust.

---

## Product Readiness Position

The goal should not be "good enough for a demo." The goal should be "predictably useful on unfamiliar codebases without requiring the user to guess whether this query class is one of the lucky ones."

That means we should not optimize toward modest thresholds like recall `0.65` and F1 `0.50` and call the problem solved. Those values might describe a promising prototype, but not a system users will trust with real workflow dependence.

The working bar should be materially higher:

| Metric | Minimum serious target | Why |
| --- | ---: | --- |
| Mean recall | `>= 0.80` | Users must usually get the relevant files into context |
| Mean precision | `>= 0.60` | Token budget cannot be dominated by noise |
| Mean F1 | `>= 0.70` | We need balanced retrieval, not recall-only wins |
| Zero-recall tasks | `0` or near-zero | Misses on critical tasks destroy trust quickly |
| Large-repo performance | strong, not merely acceptable | Hard repos are where the product earns its keep |
| Interactive latency | predictable enough for repeated use | Retrieval must fit a real implementation loop |

The exact numbers can move, but the direction is fixed: the current benchmark quality is not close enough to plateau aspirations.

---

## What the Current Run Actually Says

### 1. `archex_query` remains the best default

`archex_query` still has the strongest quality-latency balance. It is not strong enough overall, but it is currently the least bad production candidate.

### 2. `archex_query_fusion` is not deployable as a general mode

It improves some metrics, but the latency profile remains unacceptable. That means it may be interesting for targeted research or gated execution, but not as a default retrieval path.

### 3. `cross_layer_fusion` should be stopped

This is the clearest decision from the benchmark.

The surrogate-backed cross-layer path does not validate itself on the target problems it was intended to help. It is faster than raw fusion because the run is warm throughout, but aggregate quality regresses and `external-large` performance is especially poor.

Category-level behavior:

| Category | `archex_query` Recall / F1 | `cross_layer_fusion` Recall / F1 |
| --- | ---: | ---: |
| `architecture-broad` | 0.444 / 0.381 | 0.444 / 0.381 |
| `external-framework` | 0.648 / 0.563 | 0.611 / 0.549 |
| `external-large` | 0.200 / 0.171 | 0.067 / 0.057 |
| `framework-semantic` | 0.167 / 0.143 | 0.333 / 0.343 |
| `self` | 0.444 / 0.359 | 0.389 / 0.284 |

Interpretation:

- there is one promising signal in `framework-semantic`
- there is no broad win across categories
- there is a strong regression in `external-large`
- this is not enough evidence to justify further investment right now

So the project should stop working on `cross_layer_fusion` as an active retrieval direction.

---

## Why Users Would Still Reject the System

The current performance profile implies several user-facing failure modes:

1. A user asks a broad architectural question and gets partially right context plus noise.
2. A user asks a large-repo or vocabulary-mismatch question and the system misses the core files entirely.
3. A user retries with another phrasing and gets a materially different result profile.
4. A user cannot infer which strategy will help, and the system itself cannot guarantee dependable routing.

That means users will not experience the system as "powerful but imperfect." They will experience it as "sometimes useful, sometimes wrong, and hard to trust."

For a retrieval system, trust is the product.

---

## What Improved

Not all signals are negative.

Graph expansion is no longer inert.

Average `expansion_ratio` is nonzero across the main strategies, and 23 of 25 tasks now show nonzero expansion. That matters because previous benchmark evidence indicated graph expansion was contributing nothing. This suggests the graph path is at least participating again.

That does **not** mean the retrieval problem is solved. It means one previously broken differentiator is now active enough to justify further work in the BM25-plus-graph path.

---

## Strategic Decisions

### Decision 1: keep `archex_query` as the default benchmarked product path

This remains the baseline to beat.

### Decision 2: keep `archex_query_fusion` opt-in only

This strategy may still have diagnostic or selective value, but should not be treated as a default benchmark path.

### Decision 3: stop work on `cross_layer_fusion`

This includes:

- no further promotion of the strategy
- no more benchmark-driven roadmap work centered on surrogate cross-layer fusion
- no product framing that suggests this is the path forward

The benchmark script can still support it as an explicitly gated experimental mode, but it should not shape the main retrieval roadmap.

### Decision 4: optimize the benchmark loop around the baseline path

Default benchmark runs should always include:

- `raw_files`
- `raw_grepped`
- `archex_query`

Experimental strategies should only run when explicitly requested.

---

## Highest-Leverage Work From Here

If the goal is to move the benchmark toward real product readiness, the best remaining bets are:

### 1. Make BM25-plus-graph materially stronger

This is the current best base path. Improvements here have the highest chance of lifting the product default rather than just helping an experimental variant.

Focus areas:

- query normalization quality
- architecture phrase matching
- seed quality and file-level ranking
- graph expansion selectivity
- large-repo disambiguation

### 2. Reduce zero-recall tasks aggressively

Zero-recall tasks are unacceptable because they create hard user failures.

The benchmark should be treated as a zero-recall elimination program, not just a mean-metric optimization exercise.

### 3. Prioritize external-large and framework-semantic tasks

Those categories expose the real weakness of the current retrieval stack. Improvements that only move self-repo or easy framework tasks are not enough.

---

## Work To Stop

The following work should be deprioritized or stopped:

| Workstream | Decision | Reason |
| --- | --- | --- |
| `cross_layer_fusion` strategy work | Stop | aggregate quality regresses, especially on `external-large` |
| surrogate-first retrieval roadmap | Stop for now | benchmark evidence does not justify ongoing investment |
| chasing small aggregate metric gains with expensive fusion | Stop | does not change product readiness |
| moderate-target framing (`0.65` recall / `0.50` F1) | Reject | too low for a trustworthy product |

---

## Benchmark Policy Update

Benchmark execution should reflect the strategic decisions above:

- `raw_files`, `raw_grepped`, and `archex_query` are always on by default
- `archex_query_fusion` runs only when `--query-fusion` is passed
- `cross_layer_fusion` runs only when `--cross_layer_fusion` is passed

This keeps the default benchmark run aligned with the actual product candidate and prevents expensive experimental modes from distorting normal evaluation loops.

---

## Summary

The current benchmark shows a capable research prototype, not a user-ready retrieval product.

The correct response is not to lower the bar. The correct response is to keep the bar high, stop investing in the parts of the retrieval roadmap that are not paying off, and focus the next phase entirely on making the default BM25-plus-graph path significantly more reliable.

That is the only path likely to convert benchmark progress into actual user trust.
