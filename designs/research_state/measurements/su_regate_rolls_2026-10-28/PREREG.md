# S-U GATE, LEAK-REDUCED RE-GATE (`dice="rolls"`) — PREREG (2026-10-28)

Written and committed BEFORE any state of this re-gate is scored. It is an addendum to the S-U gate test
([`../su_gate_2026-10-27/PREREG.md`](../su_gate_2026-10-27/PREREG.md), `ef05d165c`; read GO in `fea88ceeb`), and it
follows the gate's adversarial review ([`../su_headroom_2026-10-27/REVIEW_gate.md`](../su_headroom_2026-10-27/REVIEW_gate.md),
`52f470c21`, M1 and M3 step 2). The read script is [`scripts/regate_read.py`](scripts/regate_read.py), committed with
this file.

Abbreviations follow the gate's PREREG. S = `r0s4@10` (the gate's search); **S′ = `r0s4_rolls@10`**; T = the gate's
24-seed self-referee; U = H (`heldout72_hpfix`), F (`forum72`), P72 at equal weight; LB = a 95 % interval's lower bound.

## 1. Why

The gate read **GO**: pooled-U sign accuracy 0.635 [0.578, 0.687] (n 346). The review found that the GO's whole excess
over home72 sits in the states exposed to the TRUE-SEED leak: exposed 0.716 [0.643, 0.782], leak-clean 0.554 [0.469,
0.629] (post hoc, one of seven splits). Its likeliest mechanism (the review's JUDGEMENT, UNVERIFIED): with
`dice="seed"` every world must reproduce every printed damage number on the TRUTH's own roll, which pins hidden
spreads and Speeds toward the truth, and T plays in the truth. A search on a real server has no true seed.

**The question:** with that damage-roll filter removed, do S's deviations from π still beat chance on U under T?

## 2. The one change: S′

`main.anchors.search_presets` gains one preset, `r0s4_rolls`: `r0s4` with ONE field changed, `dice="rolls"` (F1 tier
A, `main.value_search.worlds` THE DICE). A world may reproduce each printed damage number on ANY of the 16 rolls that
prints what we saw. Everything else is S's: 8 honest worlds (`reveal="off"`, `max_tries` 60), TRUNC-4-K8, piKL λ 0.015,
the lean screen, `LiveSearcher._search` and its keying `su_live_v1|deal|t|n`.

- **Weights stay uniform** (`SearchConfig(weights="uniform")`, as every live preset). Tier A's likelihood weights are
  NOT read: one lever per A/B. `search_ledger.md` has tier A's dice "uniform or likelihood-weighted" moving IMP
  undetectably either way.
- **What S′ does NOT remove** (named; `worlds.py` THE RESIDUAL and the PARTIAL leaks). The prefix still replays on the
  TRUE battle seed. So a speed tie, a crit, King's Rock, Focus Band, a contact ability's roll and every hidden duration
  rolled before the root spend the truth's dice (tier B is not built). The template's rewrites of past opponent choices
  and the untyped Hidden Power kept verbatim (review m6) are unchanged. **S′ is leak-REDUCED, not leak-free.** That is
  why the Speed-pin marker is reported as a stratum (§5).

## 3. The states, the referee, the engine

- **States:** the gate's frozen draw, unchanged: `~/gen3ai_archive/su_gate_2026-10-27/gate/selected.jsonl`, 1,560
  states, sha256 `cd28bd5dca08a67248d62ff6b02d1728ebe34c5403641aa6b56ec7933497e12d`. The draw is read-only; nothing is
  re-selected.
- **The referee:** the gate's T, unchanged (`sign_gate.GateScorer.score`). It uses `truth.turn_seeds(key, 40)[16:40]`,
  P\* greedy on both sides in the TRUE world, common dice across the pair, and runs on S′'s deviations only.
- **The model:** P\* = `rb_x26_s2601` @280,000,170, the gate's archived copy
  (`~/gen3ai_archive/su_gate_2026-10-27/ckpts/pstar/checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`).
- **The engine (the gate's FINDING G-1):** `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`. This is the gate's
  engine, re-stamped at this registration's tree (`main.anchors.server.engine_stamp()`). It is **not** S-U's registered
  engine (`bdc5d2be…/95b45f10…`, before `995c04a8a`'s Struggle fix). An S-U or V-RANK amendment that cites this
  re-gate registers this engine. The scoring checkout's stamp is recorded before the first chunk. A different stamp
  VOIDs the read.
- **The code:** `main.value_search`, `main.anchors.search_side`, `main.policy_spectrum`, `main.info_audit` and the
  Rust core are byte-identical to the gate's pin `835c6b4b2` (`git diff 835c6b4b2 <this commit>` on those paths is
  empty except this preset). The scoring runs from a clean checkout at this registration's commit.

## 4. The rule (unchanged: `deviation_sign`)

`main.value_search.sign_gate.read` on S′'s scored rows, with the gate's registered params:
- **Sign accuracy** = P(T(S′'s pick) > T(π top) | S′ deviates, the two T differ). Per panel, the ratio over its
  countable deviations. **Pooled U = the mean of H's, F's and P72's.**
- **Interval:** a game-clustered percentile bootstrap, `default_rng(20261027)`, B = 2,000, panels resampled in the order
  H, F, P72.
- **GO iff LB(pooled U) > 0.50 + 0.0005.** Otherwise STOP. An LB within 0.0005 of 0.50 reads STOP, flagged `boundary`.

The review's M3 suggested reading the leak-clean stratum as a co-primary. This registration keeps ONE decision rule,
the gate's, and REPORTS the strata (§5).

**VOID / DEFECT** (`regate_read.py` §1; each stops the read, none is a skip):
- the scored keys are not exactly the 1,560 selected keys, each once;
- a searched row is not on `r0s4_rolls` (`dice == "rolls"`);
- the engine stamp differs (§3);
- **the Speed-pin tooth:** the Speed ±1 marker is a property of the true root only, so every row must equal the gate's
  row for its key, and its unflipped control must pass;
- **the referee tooth:** T is a function of the log, the key's seeds and the two actions. Where S′ deviates to the same
  action S did, its outcome strings and T must equal the gate's exactly;
- a `RollMatchError` (the true team failing its own roll match) or a `GateError` (the root guard) is a named defect
  that holds the scoring.

A state where S′ finds no consistent world is an exclusion: recorded, π plays, as in the gate.

**The read script's tooth, run before this commit.** `regate_read.py --control` reads the GATE's own rows as if they
were S′'s. It reproduced the gate's verdict to its printed digits (GO 0.635 [0.578, 0.687], n 346). It also reproduced
the review's leak split: exposed U 0.7157 [0.6426, 0.7818] n 172, clean U 0.5544 [0.4692, 0.6291] n 174. Both teeth
passed (599 same-pick deviations).

## 5. Reported, never decided

- **The Speed-pin stratum** (`speed_flip.speed_pinned` > 0, against not pinned), per pooled U and home72.
- **The review's leak split:** exposed = a rewritten past opponent choice OR a pinned Speed; clean = neither. Same
  estimator and seed as the review's script.
- home72's sign accuracy, and U − home72.
- Per set: deviation rate, exclusion rate, countable deviations, IMP, search and referee CPU, the leak block.
- **How S′ moved against S**, per panel: agree / same deviation / other deviation / deviation gained / lost /
  exclusions.
- **The paired IMP difference S′ − S** on states searched under both.

## 6. What the verdict does: this re-gate SUPERSEDES the gate's GO

For licensing V-RANK and S-U, this re-gate's verdict REPLACES the gate's GO. The gate's GO is then a leak-exposed
reading, never quoted alone.

| outcome | what follows |
|---|---|
| **GO** | The licence stands, on a leak-reduced S. The next step is the review's M3 step 3: a deviation-only live branching read against Kakuna, sized for the contrast (≈ 1,800 U deviations), in place of V-RANK's all-sibling slope. It needs the forced-branch builds; the sizing and the order are the orchestrator's call. The Speed-pin stratum is quoted beside the GO. A GO still licenses the next stage only, never a gain (T shares S′'s opponent model, review M2). |
| **STOP** | The gate's GO was the seed leak's. There is no live S-U read and no V-RANK as its filter. Search stays paused for goal (b). Recorded as: "S at P\* is a chance-level perturbation of π on U once the damage-roll leak is removed (under its own model)." DOES-NOT-CLOSE-B is NOT claimed. |

## 7. Cost and schedule

- **Cost (ESTIMATE).** S′ ≈ 4–6 CPU-s a state; a smoke on 6 NON-selected kept decisions measured 3.6–5.5 s. T ≈ 10–12
  CPU-s on each deviation (≈ 40 %). The flips ≈ 0.5 s. In all, ≈ 4–5 CPU-h for 1,560 states.
- **How it runs.** Single-threaded (`OMP_NUM_THREADS=1`, the gate's G-4). `nice -n 19` on the agent cores
  (`taskset -c 5-7,13-15`), under `mem_cap.sh` (8 GB). Bounded resumable chunks: `--budget-s 1800` under a 45-min
  `timeout`. Concurrency 1, behind the N1-P, X59-S and D3 reads. Cost never re-sizes the read.
- **Where.** State: `~/gen3ai_archive/su_regate_rolls_2026-10-28/` (new; `scored.jsonl`, `score.log`, `engine.txt`).
  The gate's state directory is read-only.

## 8. Predictions (GUESSES, before any state)

- Pooled-U sign accuracy ≈ 0.58 (0.52–0.64); home72 ≈ 0.53.
- Exposed U falls toward clean U.
- Exclusions fall, as tier A's build measured: rolls accept worlds the seed screen rejected.
- S′ keeps ≈ 70 % of S's deviations as the same action.
- **P(GO) ≈ 0.55.**

## Registration

- Preset `r0s4_rolls` and `scripts/regate_read.py` are committed with this file. The gate's draw sha256 is
  `cd28bd5d…7e12d`. The engine is `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`.
- Order: this commit → score (chunks) → `regate_read.py` → README + ledger.
- Written by the S-U re-gate operator (orchestrator brief, 2026-10-28).
