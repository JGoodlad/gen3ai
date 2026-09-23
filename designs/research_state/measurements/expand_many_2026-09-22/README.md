# INSIDE `expand_many` — the rust / transport / python split, and what it bought

<!-- A MEASUREMENT record. The BEFORE tables were taken before any change in this campaign; do
not rewrite them. Later sections append their own after-tables. -->

`designs/research_state/measurements/search_profile_2026-09-22/README.md` left `expand_many` and
its JSON as **the largest single span of a searched decision — 20.2% of the view road's wall**.
That span is ONE thing from Python's side: the request write, the child's whole handler, and
`json.loads` of the reply are indistinguishable from outside the pipe, and they want opposite
fixes. This file splits them and acts on the split.

**The instrument** is `profile_expand_many.py` beside this file. The child reports its own phases
through `POKESIM_SEARCH_TIMING=1` (`pokesim::driver_timing`, OFF by default and byte-identical
when off); Python times the same call from outside; **transport is the remainder**,
`wait − child_total`. The decision fixtures are `search_decision_benchmark.load_decision`'s — the
same banked eval traces, so this table and that one share a denominator.

```bash
export PYTHONPATH=$PYTHONPATH:src
POKESIM_SEARCH_DRIVER_BIN=$PWD/src/rust_sim/target/release/search_driver \
POKESIM_SEARCH_TIMING=1 \
python3 designs/research_state/measurements/expand_many_2026-09-22/profile_expand_many.py \
    --traces models/ai_v12_02_winprob_critic/eval_traces --decisions 12 --m-opp 3 --k-worlds 4
```

🚨 **Ratios are the claim.** The box carried a production training arm throughout and its load
average ran from 18 to 35 across this afternoon; every absolute ms below moved with it. The only
before/after claim this file makes is the **interleaved A/B** at the end.

---

## 0. THE SPLIT — before anything was changed (2026-09-22)

12 decisions, honest arm, `m_opp=3`, `k_worlds` 4, **864 arms in 39 batches** (22 arms per
`expand_many` round trip — the call pattern is already one request per PLY, not per world);
load average 19.6/19.6/18.8 on 16 cpus; decision wall 179.3 ms.

| phase | ms/arm | % of `expand_many` | % of decision |
|---|---|---|---|
| rust: sim (snapshot + `resolve_turn`) | 0.0896 | 16.5% | 3.6% |
| rust: `one_sided_view` x2 (emit + JSON) | 0.0594 | 10.9% | 2.4% |
| rust: p1/p2 chunk arrays | 0.0131 | 2.4% | 0.5% |
| rust: outcome / requests / child / tail | 0.0022 | 0.4% | 0.1% |
| transport (pipe + scheduler) | 0.0223 | 4.1% | 0.9% |
| py: request `dumps` + write | 0.0013 | 0.2% | 0.1% |
| **py: `json.loads(reply)`** | **0.1711** | **31.5%** | **6.9%** |
| **py: after the parse (billed to the `ExpandedNode` walk)** | **0.1793** | **33.0%** | **7.2%** |
| **TOTAL** | **0.5433** | 100% | **21.8%** |

**THE HEADLINE: the child is 30.2% of `expand_many` and Python is 64.5%.** The engine is not the
cost; *shipping and re-parsing what the engine says* is.

### The reply payload, by field (bytes that actually crossed the pipe)

30,326 B per arm, 26.2 MB over the run. The request side is 3,246 B per BATCH — nothing.

| field | B/arm | % of reply |
|---|---|---|
| `view_p2` | 9,910 | 32.7% |
| `view_p1` | 9,461 | 31.2% |
| `requests` | 4,153 | 13.7% |
| `p1_chunks` | 3,264 | 10.8% |
| `p2_chunks` | 3,119 | 10.3% |
| `outcome` | 350 | 1.2% |
| everything else | 69 | 0.2% |

**A search runs for ONE side.** `search._expand_ply` reads `view_p1 if side == "p1" else view_p2`
and the matching chunk array, and nothing of the other side — so **43.0% of those bytes were
rendered, quoted, piped and `json.loads`-ed to be dropped**. `requests` is NOT in that class: both
sides are read (the opponent's legal choices at interior plies, and `branchable` for ours).

---

## 1. SIDE ELISION — `gen3_expand_many_side_elision_v1` (LANDED)

`expand_many` takes an optional `side` on the REQUEST; with `side:"p1"` the driver omits
`p2_chunks` and `view_p2` entirely. `SearchEngine._expand_ply` passes its own side.

**Why this is safe and not merely likely to be safe:** the requested side's bytes do not change.
`tests/search_side_elision_test.rs` expands the SAME arm from the SAME root in the SAME process
with and without `side`, and asserts the surviving `view_pN` / `pN_chunks` are **byte-identical**,
that the other side's fields are gone, that a `side`-less request still renders every field in its
historical ORDER (which is what keeps `search_impl_parity`'s node comparison valid), and that an
unrecognised `side` is an ERROR rather than a silent fall-back to both.

**The elided slot raises rather than reads empty.** `utils.bridge.search_session.ElidedSide` is
falsy — so the existing `payload or {}` guards still take their COUNTED fallback — but every way
of getting a value out of it raises. An empty dict would have ENCODED: into a well-formed
observation of a battle nobody played. The sentinel keys on the field being ABSENT, never on the
request, so `impl="node"` (whose driver ignores the key and returns both sides) elides nothing.

**Decision-level gate**: `src/main/search_dividend/side_elision_parity_integration_test.py` runs
the real `SearchEngine` over one seeded decision with elision ON and OFF and compares every
successor's observation BYTES in order, the per-action scores, the chosen action, the widths
**and the two fallback counters** — because an elision read as empty would quietly push arms onto
the protocol road and still produce plausible scores. Non-vacuity is asserted both ways.

**Measured**, same 12 decisions / 864 arms: reply **30,326 → 17,330 B/arm (−42.9%)**, and the
surviving `view_p1` reads 9,479 B/arm against the pre-change 9,461 — the same payload, as the
rust gate says by construction.

| field | B/arm | % of reply |
|---|---|---|
| `view_p1` | 9,479 | 54.7% |
| `requests` | 4,161 | 24.0% |
| `p1_chunks` | 3,270 | 18.9% |
| `outcome` | 351 | 2.0% |

---

## 2. 🚨 THE "`ExpandedNode` DICT WALK" ROW IS NOT A DICT WALK — IT IS THE GARBAGE COLLECTOR

The second-biggest row in the before-table was the Python code that turns 22 parsed arm dicts into
22 frozen dataclasses. Eleven `dict.get`s per arm cannot cost 0.18 ms, and measured on a captured
reply it does not:

| measured on one real 22-arm reply (454 KB) | ms/arm |
|---|---|
| `ExpandedNode` construction, arms ALREADY parsed | **0.0028** |
| `json.loads(whole reply)` | **0.4124** |
| `json.loads(whole reply)`, **cyclic GC disabled** | **0.1805** |

So the row is **collection time billed to whatever allocated when the collector fired**, and the
parse itself pays the same tax: `json.loads` of one reply is **2.3x** more expensive with the
cyclic collector running than without. That is the shape of the cost — a large, short-lived,
purely acyclic object graph allocated inside a process whose live heap (trackers, the poke-env
replay player, the pool teams) is expensive to traverse.

**This is why the profile had to come first.** The briefed candidates were about the payload's
ENCODING; the measurement says the payload's SIZE matters mostly because of what allocating it
does to the collector, and that the encoding itself is nearly irrelevant (§3).

---

## 3. THE CANDIDATES THE MEASUREMENT KILLED — each with the number that killed it

At the before-profile's denominator one arm of one decision is **2.49 ms**, so 5% of the decision
wall is **0.125 ms/arm**. Three of the four briefed candidates do not reach it.

| candidate | ceiling, measured | verdict |
|---|---|---|
| **(a) / (d) a COMPACT payload** — fixed-order arrays per block instead of keyed objects | the view goes 10,870 → 7,561 B (69.6%) and `json.loads` 0.0876 → 0.0640 ms — **0.024 ms/arm, ~1.0% of the decision wall**. And a decoder that must re-key the arrays to feed the existing adapter costs **0.0845 ms**, i.e. the saving is **gone** (3.5%) unless every consumer in `view_adapter.py` is rewritten to index positions | **DEAD.** Under the bar by 5x at its ceiling, and at zero net gain in the form that keeps the adapter |
| **(b) one request per DECISION** | transport is **4.1% of `expand_many`, 0.9% of the decision**, and the call pattern is ALREADY one request per ply carrying every arm (864 arms in 39 batches). There are no per-world round trips left to remove — `gen3_one_fork_per_decision_v1` took them | **DEAD.** Nothing to win |
| **(c) cache the ply-INVARIANT view parts in rust, emit deltas** | the strictly-invariant fields (`base_stats`, `types`, `ivs`, `evs`, `spread_known`, `species`) are ~2.5 KB of the 8.6 KB mon payload — **~23% of the view, ~0.022 ms/arm of parse, ~1% of the decision wall**, for a per-decision cache in the driver, a merge in Python, and a real risk to the byte-parity gate | **DEAD.** Same order as (a), with strictly more risk |
| **(a′) msgpack or a length-prefixed binary** | not attempted: `environment.yml` carries no msgpack and the brief forbids adding a dependency. The ceiling is (a)'s anyway — the parse is 0.088 ms/arm and a binary codec cannot save more than all of it (**3.5% of the decision wall**) | **DEAD**, and it was never the dependency that stood in the way |

**The ONE candidate the measurement found that the brief did not** is §2's collector tax, which is
worth **~0.23 ms/arm ≈ 9% of the decision wall** and touches no wire format at all. That is what
§4 does.

---

## 4. THE COLLECTOR TAX — MEASURED, AND **REJECTED** ON THE END-TO-END EVIDENCE

§2's micro-benchmark is not in doubt: `json.loads` of one captured 454 KB reply costs
**0.4124 ms/arm with the cyclic collector running and 0.1805 without**. The obvious change is to
hold the collector off for the duration of the parse — safe on its face, because JSON cannot
express a reference cycle, so refcounting reclaims everything the parse makes.

**It was built, gated, and then it did not survive the end-to-end A/B, so it was reverted.**

| what was run | result |
|---|---|
| the first interleaved A/B, carrying BOTH this and §1 | the view road read **SLOWER** than base in both pairs (264.7 / 285.9 → 359.3 / 322.4 ms per decision) while the protocol road read faster — incoherent for a change that only removes work |
| this change ISOLATED, behind an env switch, 2 interleaved pairs, view road only | GC-off **197.6 / 270.4** vs untouched **243.0 / 271.9** ms per decision. **The within-arm spread (197.6 → 270.4 for the SAME build) is larger than the between-arm difference** |
| §1 alone, re-run after the revert | consistently faster, 4 pairs of 4 (below) |

The isolated measurement's CI plainly straddles zero, so by the project's own rule the effect is
**NOT DETECTED**, not "small" — and the paired run that carried it read slower twice. The
plausible mechanism for harm is the one ad-hoc `gc.disable()` always has: deferring a collection
does not cancel it, it lets the batch's objects survive into an OLDER generation, where the
collection that eventually runs is more expensive and lands somewhere nobody is measuring.

🚨 **The durable lesson is the sibling of this campaign's other one.** The profile README retracted
a cProfile share because it was not a wall share; this rejects a MICRO-benchmark share for the
same reason. **2.3x on a captured payload in a toy process is not 2.3x in the process that has the
trackers, the replay player and the pool in its heap** — which is precisely the heap the collector
has to walk, and precisely what the micro-benchmark does not have. A candidate whose only evidence
is a micro-benchmark is not evidence; it is a hypothesis with a number attached.

What would settle it is a quiet box. The measurement is cheap to repeat (`--roads view`, the env
switch is gone but the diff is four lines) and is left for one.

---

## THE CUMULATIVE TABLE — an INTERLEAVED A/B against the starting HEAD

A baseline worktree at `ae64dfe0` against this branch, the same 10 banked decisions, run back to
back with the load recorded. **`ms/arm` is the statistic**, not `ms/decision`: it normalises each
decision by its own arm count, and the two columns disagree exactly where the decision-mean is
noisiest.

```bash
# one pair; alternate the two worktrees and repeat. --roads view (or protocol) is LOAD-BEARING.
for wt in <base_wt> <this_wt>; do
  cd $wt && PYTHONPATH=$wt/src CUDA_VISIBLE_DEVICES="" \
    GEN3AI_MODELS_DIR=/home/goodlad/dev/gen3ai/models \
    POKESIM_SEARCH_DRIVER_BIN=$wt/src/rust_sim/target/release/search_driver \
    nice -n 15 python3 src/main/search_dividend/search_decision_benchmark.py \
      --traces models/ai_v12_02_winprob_critic/eval_traces \
      --decisions 10 --arm honest --impl rust --seed 11 --m-opp 3 --k-worlds 4 --roads view
done
```

🚨 **ONE ROAD PER PROCESS.** Running `--roads protocol view` (the default) puts both roads in one
interpreter and **they are not independent**: in those runs the protocol road read 1.55x faster
and the view road read *slower*, which no change that only removes work can produce. Every row
below is a single-road run. That is a property of the instrument, and it is why the first A/B of
this campaign pointed the wrong way.

### VIEW road, wide B (684 arms / 10 decisions), 4 interleaved pairs

| pair | load | BASE ms/arm | NOW ms/arm | speedup |
|---|---|---|---|---|
| r1 | 24.7 / 26.2 | 4.696 | 4.158 | 1.13x |
| r2 | 27.5 / 27.7 | 4.441 | 2.197 | (warm-up inflated) |
| **r3** | 25.7 / 23.9 | **2.315** | **2.110** | **1.10x** |
| **r4** | 22.3 / 22.3 | **2.293** | **2.084** | **1.10x** |

r1/r2 carry the run's cold start on the BASE side; **r3 and r4 are the load-matched, warm pairs
and they agree to three digits — 1.10x.** NOW wins 4 of 4.

### PROTOCOL road, wide B — it gains too, and that is the check

| pair | BASE ms/arm | NOW ms/arm | speedup |
|---|---|---|---|
| r1 | 2.628 | 2.373 | **1.11x** |
| r2 | 2.573 | 2.325 | **1.11x** |

The protocol road never reads `view_pN`, but it does parse the reply and read its own
`pN_chunks` — so an elision that works must move it too, by about the same amount. It does. A
result that had moved only the view road would have meant something other than fewer bytes.

### B = 1 (31 arms / 10 decisions) — NOT RESOLVED, and that is expected

| pair | BASE ms/arm | NOW ms/arm |
|---|---|---|
| r1 | 17.931 | 16.742 |
| r2 | 17.457 | 8.931 |
| **r3 (warm)** | **9.091** | **8.670** (1.05x) |

At B = 1 the decision IS its prefix replay (53.5% by the profile) and there are 31 arms to save on.
The warm pair reads 1.05x and the cold ones are dominated by their own start-up; **the honest
statement is that the change is not resolved at B = 1 above this box's noise**, which is what the
mechanism predicts.

### The span itself, 3 interleaved pairs (`profile_expand_many.py`, view road)

| | BASE | NOW |
|---|---|---|
| reply bytes / arm | 30,534 | **17,485 (−42.7%)** |
| `expand_many` ms/arm | 1.518, 1.392, 1.510 | 1.087, 1.217, 0.556 |
| **`expand_many` as % of the decision wall** | 31.5%, 31.0%, **30.8%** | 26.7%, 28.3%, **24.8%** |

**The share is the load-robust number** — BASE holds 31.1% ± 0.7 across a decision wall that
itself moved 307 → 335 ms. `expand_many` falls from **31.1% to 26.6% of the decision wall**, and
it is no longer the largest single span.

---

## Ready-to-append ledger paragraph

> **2026-09-22 — `expand_many` was the largest single span of a searched decision, and the ENGINE
> was 30% of it.** Splitting the span with a driver-side timer (`POKESIM_SEARCH_TIMING=1`,
> `pokesim::driver_timing`, off by default and byte-identical when off) against a Python-side one
> put the rust child at 0.164 ms/arm, the pipe at 0.022 and Python at 0.350 — `json.loads` of a
> reply of which **43.0% was the side the search never reads**. **LANDED:**
> `gen3_expand_many_side_elision_v1` — `expand_many` takes a `side` on the request and omits the
> other side's `view_pN` / `pN_chunks`. The requested side is byte-identical (a rust gate expands
> the same arm both ways in one process and diffs the bytes); a `side`-less request renders the
> historical body byte-for-byte, so `search_impl_parity` still accepts node's driver; the elided
> slot is a sentinel that is falsy but RAISES on read, because an empty dict would have ENCODED a
> board nobody played; and a decision-level gate compares every successor's obs bytes, the scores,
> the action, the widths and both fallback counters with elision on and off. Reply bytes
> **30,534 → 17,485 per arm (−42.7%)**; the span fell from **31.1% to 26.6% of the decision wall**
> and is no longer the largest. Interleaved, load-matched, one road per process: **1.10x on the
> view road at wide B (2.315 → 2.110 and 2.293 → 2.084 ms/arm, two warm pairs agreeing to three
> digits), 1.11x on the protocol road** (which parses the same reply — a result that moved only
> one road would have meant something else), and **NOT RESOLVED at B = 1**, where the decision is
> its prefix and there are 31 arms. **FOUR candidates were killed by measurement, each with its
> number.** A compact fixed-order payload saves 0.024 ms/arm of parse and gives all of it back
> re-keying for the existing adapter (~1% of the decision wall); one-request-per-decision is worth
> 0.9% and the call pattern is already one per PLY; caching the ply-invariant view parts in the
> driver is ~1% for a cache, a merge and a risk to the byte-parity gate. The fourth was mine and
> it was BUILT before it was rejected: holding Python's cyclic collector off for the reply parse
> measures **2.3x on a captured 454 KB reply** (0.4124 → 0.1805 ms/arm) and **nothing end to end** —
> isolated, its two pairs read 197.6/270.4 against 243.0/271.9, a spread within one build larger
> than the difference between builds, and the paired run carrying it read SLOWER twice. **A
> micro-benchmark share is not a wall share** — the sibling of this campaign's cProfile retraction,
> and for the same reason: the toy process does not have the heap the collector must walk. Two
> instrument findings recorded: the profile row that reads as an "`ExpandedNode` dict walk" is the
> COLLECTOR (the walk itself measures 0.0028 ms/arm against the 0.179 billed there), and
> `search_decision_benchmark.py`'s two roads **in one process are not independent** — a per-road
> A/B must run one road per process, and the first A/B of this campaign pointed the wrong way
> because it did not. **STOPPED** at the brief's rule: the next candidate's ceiling is ~1% of the
> decision wall against a 5% bar.
