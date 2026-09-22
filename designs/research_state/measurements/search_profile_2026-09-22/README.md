# WHERE ONE SEARCHED DECISION'S WALL GOES — the profile, and what it bought

<!-- A MEASUREMENT record. The profile table below was taken BEFORE any change in this campaign;
every later section appends its own after-table. Do not rewrite the BEFORE column. -->

**The instrument** is `src/main/search_dividend/search_decision_benchmark.py`, written for this
and kept: it drives the real `SearchEngine.choose` over a decision reconstructed from a **banked
eval trace** (`models/ai_v12_02_winprob_critic/eval_traces`, 4,665 banked battles, read-only) and
breaks the wall into stack-accounted EXCLUSIVE phases, so the rows sum to the decision and the
remainder prints as `python glue` rather than hiding.

```bash
export PYTHONPATH=$PYTHONPATH:src
POKESIM_SEARCH_DRIVER_BIN=$PWD/src/rust_sim/target/release/search_driver \
python3 src/main/search_dividend/search_decision_benchmark.py \
    --traces models/ai_v12_02_winprob_critic/eval_traces \
    --decisions 12 --m-opp 3 --arm honest --k-worlds 4
```

🚨 **The scorer is the pure `obs.sum`**, the same choice `materializer_parity_integration_test`
makes. A trained net's forward is a real cost but not one this campaign can change, and threading
a checkpoint in would make the table a statement about which checkpoint was on disk.

🚨 **Ratios are the claim.** The box carries a production training arm; every table records its
load, and the two roads are run BACK TO BACK per decision so the protocol road is a live control
for whatever the box was doing. An absolute ms here is not comparable to an absolute ms there.

---

## 0. THE PROFILE — before anything was changed (2026-09-22)

12 decisions, honest arm, `m_opp=3`, `k_worlds` cap 4 (realized 4 on 9 of 12, 1 on 3),
**864 arms**; load average 17.4/14.5/15.2 on 16 cpus.

| road | arms | ms / decision | ms / arm |
|---|---|---|---|
| protocol | 864 | 488.5 | 5.829 |
| **view (production default)** | 864 | **327.2** | **4.526** |
| | | **1.49x** | |

`view_arms=719`, **`view_fallback_intermediate=145` (16.8% of arms)**, `fallback_no_payload=0`.

### The VIEW road's per-arm wall, by phase (exclusive)

| phase | ms/arm | % of decision |
|---|---|---|
| `expand_many` (rust + its JSON) | 0.691 | 15.7% |
| tracker fork (pinned-pickle thaw) | 0.478 | 10.9% |
| **protocol road, the D10 FALLBACK** | 0.432 | 9.8% |
| encode (obs) | 0.418 | 9.5% |
| `open_root` (rust) | 0.342 | 7.8% |
| **prefix replay (`open_view_fork`)** | 0.324 | 7.4% |
| `view_adapter` read-models | 0.294 | 6.7% |
| prefix decision encode/track | 0.230 | 5.2% |
| action mask | 0.188 | 4.3% |
| snapshot restore (inside the fallback) | 0.107 | 2.4% |
| event fold (ply protocol) | 0.095 | 2.2% |
| replay player build | 0.093 | 2.1% |
| snapshot freeze (inside the fallback) | 0.061 | 1.4% |
| **`action_choices` / `map_actions_at`** | **0.054** | **1.2%** |
| successor context | 0.051 | 1.2% |
| event-fold branch | 0.010 | 0.2% |
| python glue (uninstrumented) | 0.533 | 12.1% |

**INCLUSIVE spans** (they overlap the rows above): the D10 fallback into
`materialize_branches` is **21.1%** of the view road's decision wall while serving **16.8%** of
its arms; the shared prefix replay is **13.7%**; `expand_many` 15.7%; `open_root` 7.8%.

### The same decision at B = 1 (`--m-opp 1 --n-actions 1`)

| road | arms | ms / decision | ms / arm |
|---|---|---|---|
| protocol | 39 | 119.9 | 33.817 |
| view | 39 | 102.7 | 30.167 |
| | | **1.17x** | |

Inclusive spans: **prefix replay 53.5%**, `open_root` 15.8%, the D10 fallback 15.6%. At B = 1 the
decision IS its prefix, exactly as `one_sided_view.md` §6 says.

---

## THE ORDER THE PARTS WERE DONE IN, AND WHY IT IS NOT THE ORDER THEY WERE BRIEFED IN

The brief ordered the work fork → lazy `action_choices` → `map_actions_at` → D10. The profile
re-ordered it, and two of the four premises did not survive contact:

| briefed | measured share | verdict |
|---|---|---|
| **one fork per decision** (Part 1) | prefix replay **13.7%** at wide B, **53.5%** at B = 1, paid **once per WORLD** | **FIRST.** Biggest single lever, pure Python, no contract touched |
| **D10** (Part 4) — "6.9% of arms" | **16.8%** of arms here, and **21.1%** of the wall | **SECOND**, not last. The documented 6.9% is a different anchor distribution, not a different mechanism |
| **lazy `action_choices`** (Part 2) | **1.2%** of the wall | worth ~1%, done last |
| **`map_actions_at` is 24% of the per-arm wall** (Part 3) | **1.2%**, measured two ways | 🚨 **THE PREMISE IS RETRACTED — see below** |

### 🚨 The "24%" in `one_sided_view.md` §6 does not reproduce, and the correction is recorded here

That figure came from a cProfile run. Measured directly with a stack-accounted wall timer on 864
arms of 12 banked decisions, `_choice_map` — the real action mapper over every legal index, which
is what BOTH `map_actions_at` (protocol road) and `action_choices` (view road) are — costs
**0.054 ms/arm, 1.2% of the decision**. Re-running the identical benchmark **under cProfile** does
not restore it either (1.1% there), so this is not simply profiler overhead on a many-small-calls
function; the two measurements disagree about the denominator, and the one taken on the production
`SearchEngine` over banked traces is the one this campaign acts on. What cProfile DOES distort is
visible in the same pair: `encode` reads 9.5% un-profiled and 15.6% profiled, and the action mask
14.7% un-profiled against 2.2% profiled — **a cProfile share is not a wall share**, which is the
durable lesson and the reason this instrument exists.

---

## 1. ONE VIEW FORK PER DECISION — `gen3_one_fork_per_decision_v1` (LANDED)

The ply-1 `ViewSuccessorFactory` is `materialize_branches`' first half over `(one-sided prefix,
our action history, OUR packed team)`. A determinized world changes only the **opponent's** team,
and `determinize.prefix_matches` already gates every kept world to a byte-identical one-sided
prefix — so K worlds were each replaying one identical prefix. `SearchEngine._root_fork` now
builds it once per decision and **keys the cache on the prefix BYTES**, which is what makes the
reuse safe by construction rather than by argument: the gate truncates its comparison at the
`|turn|` marker, so a world differing after it misses the cache and replays its own.
`RealizedWidths.fork_cache_hit` / `fork_cache_miss` count it.

**Gate**: `src/main/search_dividend/fork_sharing_parity_integration_test.py` (`sim`,
`integration`) runs the real engine over one seeded decision twice — sharing ON, and with the
cache replaced by a store-but-never-serve dict, i.e. the un-shared control — and compares **every
successor's observation BYTES in order**, plus the per-action scores, the chosen action and the
widths. Non-vacuity is asserted on both sides (the experiment must have hit, the control must not
have). Measured on three decisions of the fixture battle: 3 worlds, `fork_hit=2 fork_miss=1`
each — the prefixes ARE byte-identical across worlds — and 48 / 54 / 33 successors byte-identical.

**Measured**, same 12 banked decisions, both roads interleaved, load average 9.2/14.5/15.3:

| B | road | ms / decision | ms / arm | protocol -> view |
|---|---|---|---|---|
| wide (864 arms) | protocol | 240.2 | 2.823 | |
| wide (864 arms) | view | **138.9** | **1.922** | **1.73x** (was 1.49x) |
| 1 (39 arms) | protocol | 60.8 | 17.757 | |
| 1 (39 arms) | view | **26.5** | **7.415** | **2.30x** (was 1.17x) |

The prefix-replay span fell from **13.7% -> 5.9%** of the view road's decision wall at wide B.
The protocol column is the control: the box was quieter for the after-run, which is exactly why
the ratio and not the absolute ms is the claim.

---

## 2. THE PROTOCOL ROAD'S PREFIX, SHARED THE SAME WAY (LANDED)

The D10 fallback calls `materialize_branches`, which replays the whole shared prefix again — and
it did so **once per world, on top of the view fork's own replay**. `materialize_branches` is now
split into `open_branch_fork` (the prefix replay + the frozen `_PlayerSnapshot`) and
`materialize_branches_from` (the arms), with `materialize_branches` kept as the composition of the
two, so its own gate (`obs_materializer_branch_integration_test`) is unchanged and still passes.
`SearchEngine._branch_fork` caches the fork per decision, keyed on the prefix bytes **plus** the
three decision-indexed knobs baked into the player at construction (`map_actions_at`,
`stop_after_decision`, `encode_only_at`) — a deeper ply asks about a different decision index and
must not be served a ply-1 fork.

**Why this needs its own gate and is not Part 1 again**: the view fork carries a frozen tracker
and is never mutated, while this one carries a LIVE poke-env replay player that every arm mutates
and `_PlayerSnapshot.restore` resets. Reusing it across worlds asserts the restore is COMPLETE —
a field the snapshot forgets would leak world 1's last arm into world 2's first, and the obs would
still be well-formed. `fork_sharing_parity_integration_test` has a second test for exactly that.

## 3. LAZY `action_choices` — `gen3_lazy_action_choices_v1` (LANDED, VIEW ROAD)

`action_choices` is the real action mapper over every legal index of a successor, and the only
readers — `TreeNode.expandable`, `plan_beam`'s arm-count estimate and `search._expand_ply`'s loop
— are all inside `_score_world`'s `while ply < md`. **A depth-1 decision built one per arm and
read none.** `agents.training.view_successor.LazyTokens` is a `Mapping` that builds on first read;
`bool()` falls through to `__len__` and therefore materializes, so "is this branchable?" still
gets a true answer and the laziness is only ever about plies nobody touches.

**Gate**: `src/main/search_dividend/lazy_action_choices_integration_test.py` — at `max_depth=1`
every successor map comes back a `LazyTokens` and **none is built** (18 lazy, 0 built); at
`max_depth=2` they are built (17 of 17, `depth_realized=2`) and each equals what its own producer
returns, so a deferral that captured the wrong `vbattle` / `mask` / `legal` fails here.

⚠️ **VIEW road only.** The protocol road's map is built by the replay player at the decision
itself (`map_actions_at`), where the poke-env battle still stands; by the time a caller could ask
for it that battle has been restored out from under it. Deferring there is not a lazy read, it is
a second replay — recorded, not attempted.

**Part 3 of the brief is SUBSUMED, not skipped.** Its target was `map_actions_at`/`_choice_map`
itself; the measurement above put that at 1.2% rather than 24%, and not calling it at all beats
caching it. `_choice_map` is now **0.7%** of the view road's wall.

---

## THE CUMULATIVE TABLE — an INTERLEAVED A/B, not two runs a day apart

🚨 **The earlier per-part numbers in this file are NOT comparable to each other**: the box's load
fell from 17.4 to 6.1 across the afternoon, and every absolute ms moved with it. So the final
claim is a fresh **interleaved** A/B — a baseline worktree at `67ea46ee` and this branch, the same
10 banked decisions, run back to back, twice.

| B | road | BASE ms/decision (r1, r2) | NOW ms/decision (r1, r2) | speedup |
|---|---|---|---|---|
| wide (684 arms / 10 decisions) | **view** | 326.8, 361.9 | **250.5, 269.3** | **1.33x** |
| wide | protocol | 514.2, 502.6 | 437.0, 468.9 | 1.12x |
| 1 (31 arms / 10 decisions) | **view** | 96.0, 98.9 | **58.1, 59.0** | **1.66x** |
| 1 | protocol | 121.2, 119.7 | 67.9, 64.1 | 1.81x |

Load at each run, in the order they ran: base 12.9 / now 16.5 / base 26.9 / now 29.1 (wide), and
base 22.1 / now 21.0 / base 30.4 / now 33.4 (B = 1). **The NOW runs carried the HIGHER load in
three of the four pairs**, so these ratios are conservative.

Protocol -> view at wide B is **1.57x -> 1.74x**; both roads got faster, which is why the
road-vs-road ratio understates the change and the base-vs-now columns above are the claim.

### Where the view road's wall goes NOW (684 arms, same run)

| span (inclusive) | before | after |
|---|---|---|
| `expand_many` (rust + its JSON) | 16.3% | **20.2%** |
| the D10 fallback (`materialize_branches*`) | 24.0% | **17.2%** |
| shared prefix replay (`open_view_fork`) | 14.3% | **7.8%** |
| `open_root` (rust) | 7.6% | 8.0% |
| `_choice_map` | 1.2% | **0.7%** |

---

## 4. D10 — DESIGNED, NOT LANDED, AND THE BLOCKER IS NAMED

D10 is now the second-biggest item (**17.2%** of the view road's wall for **17.1%** of its arms,
117 of 684 here). It is **not closable in Python**, and that is the blocker.

**The mechanism.** When an arm's ply KOs one of our mons, the replacement round is a SECOND
request inside the same `expand_many` arm. `search.rs::resolve_turn_sourced` loops
`while !sess.is_ended() && open_boundary_turn(sess) == start_turn` and answers that request from
`followup_choice` — the port's own follow-up policy — so the board it returns as `view_pN` is one
decision PAST the row `materialize_branches` produces. The Python side already detects it exactly
(`view_successor.intermediate_decisions`, verified against the protocol road's realized row count
on 104 arms) and falls back, counted.

**The smallest fix on the contract's own terms** (the port emits sim + raw protocol facts; poke-env
presentation stays in Python) is for `search_driver.rs` to emit the view it ALREADY has at the
moment that intermediate request opens — an ordered `view_p1_at` / `view_p2_at` array beside
`view_p1` / `view_p2`, one entry per decision the port resolved internally. It adds no rule to the
port: `one_sided_view(&sess, side, dex)` at the top of a loop iteration whose request the source
could not answer. Python then reads `view_pN_at[0]` instead of `view_pN` when
`intermediate_decisions(chunks) > 0`.

**Why it was not landed here, stated plainly rather than as a nice-to-have:**

1. It is a **rust change plus a payload-contract change**, and the contract's own rule is that a
   sweep is run on **at least two fresh seeds** before it is called green
   (`one_sided_view.md` §4) — three of that document's findings appeared only on the second or
   third seed. That is a multi-hour gate, not a smoke.
2. There is a **second, Python-side half that is not obviously right**: the view road folds WHOLE
   chunks (`ViewEventFolder.fold(chunks)`), while `materialize_branches` stops mid-chunk at the
   `|request|` line that opened the decision. If an intermediate request sits mid-chunk, the two
   roads would fold different event sets even with the right board — so the fix needs a
   line-level split of the arm's chunks, and that split needs its own parity evidence against
   `event_fold_parity_fuzz_test`.
3. `search_impl_parity.py`'s allowlist forgives exactly the **absence** of `view_pN` on node and
   is value-aware; a new field needs its own entry, or node's driver needs to emit one too.

Until it lands, the fallback is at least no longer paying a **per-world** prefix replay — that is
what §2 removed, and it is why the D10 span fell from 24.0% to 17.2% without D10 itself moving.

## 5. THE PREFIX-GATE P1 — NOT ATTEMPTED

Out of budget after §§0-3 and the D10 design. Recorded as still open: a clean record fails
`dz.prefix_matches` on 72 of 73 decisions because the replayed prefix carries an
`|error|[Invalid choice]` line the observed prefix lacks. It is CORRECTNESS and it blocks every
playoff measurement, so it should lead the next pass.

---

## Ready-to-append ledger paragraph

> **2026-09-22 — SEARCH PERF: one searched decision is 1.33x cheaper at wide B and 1.66x at B=1,
> and TWO documented numbers were wrong.** Profiled the real `SearchEngine` over banked eval
> traces with a new stack-accounted instrument
> (`src/main/search_dividend/search_decision_benchmark.py`; record:
> `designs/research_state/measurements/search_profile_2026-09-22/README.md`). **THE PROFILE
> RE-ORDERED THE WORK.** (1) `one_sided_view.md`'s "`map_actions_at` is ~24% of the view road's
> per-arm wall" **does not reproduce** — it is **1.2%** (0.054 ms/arm) on the production engine and
> **1.1%** under cProfile, so it is not profiler overhead; what cProfile DOES distort is visible in
> the same pair (encode 9.5% un-profiled vs 15.6% profiled; the action mask 14.7% vs 2.2%). **A
> cProfile share is not a wall share.** (2) D10's rate is **16.8-17.1% of arms** on mid-game banked
> decisions, not the 6.9% on record, and its fallback was **24.0%** of the view road's wall.
> **LANDED, each with its own gate:** `gen3_one_fork_per_decision_v1` — the ply-1 view fork is a
> function of the one-sided prefix and `determinize.prefix_matches` gates every world to the same
> prefix, so ONE fork now serves a decision's K worlds instead of K identical prefix replays, keyed
> on the prefix BYTES because the gate truncates at the `|turn|` marker
> (`fork_sharing_parity_integration_test` compares every successor's obs BYTES against an un-shared
> control, non-vacuity asserted both ways: 3 worlds, `fork_hit=2 fork_miss=1`, prefixes byte-equal);
> the same sharing for the PROTOCOL road's fork, which every D10 fallback arm pays
> (`materialize_branches` split into `open_branch_fork` + `materialize_branches_from`, the
> composition kept so its old gate is untouched); and `gen3_lazy_action_choices_v1` — a successor's
> token map now builds on FIRST READ, and a depth-1 decision reads none (18 lazy / 0 built at
> `max_depth=1`; 17/17 built and equal to their producer at `max_depth=2`). **MEASURED as an
> INTERLEAVED A/B** (a baseline worktree at `67ea46ee` against this branch, the same 10 banked
> decisions, back to back, twice — the NOW runs carried the higher load in 3 of 4 pairs, so the
> ratios are conservative): view road **326.8/361.9 -> 250.5/269.3 ms per decision (1.33x)** at
> wide B (684 arms) and **96.0/98.9 -> 58.1/59.0 ms (1.66x)** at B=1; the protocol road 1.12x and
> 1.81x. The view road's prefix span fell 14.3% -> 7.8% and the D10 span 24.0% -> 17.2%.
> **NOT DONE:** D10 itself is designed and blocked on a rust payload change plus a line-level chunk
> split whose parity evidence does not exist yet (the design and all three blockers are in the
> README); the prefix-gate P1 was not reached.
