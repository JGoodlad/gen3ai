# Rust Core M2 — the view-fold gate, the core road's cost, the typed shortcut, and poke-env's reading findings

<!-- A MEASUREMENT record (2026-09-23) for the Rust Core Program's M2
(designs/endstate/program_rust_core.md §2 M2, §6). Code: designs/rust_sim/present.md. -->

Four questions, each answered by an interleaved A/B with **one binary (or road) per process**, the
1-minute load average recorded at every sample's start and end, `nice -n 10`, CPU only, no port, no
`models/` write, no `data/` change. The statistic is always the per-pair ratio: its median and a
paired bootstrap 95 % CI on that median (20,000 resamples, fixed seed). **Ratios are the claim;**
the box carries a production training arm and the absolute ms move with it.

| # | question | verdict |
|---|---|---|
| 1 | does the view-fold gate put the TRAINING transport back at or below pre-M1 CPU? | **YES — 0.909× pre-M1** [0.876, 0.932], 0.833× pre-fix, stdout byte-identical |
| 2 | what does a successor cost on the core road vs the view road? | **1.41× per successor in Rust** [1.36, 1.48]; the whole searched decision **0.926×** [0.920, 0.944] (wide B) |
| 3 | what does the TYPED shortcut save over the full text path? | **nothing measurable**: fold 1.019× [0.983, 1.316], decision wall share −0.1 % — recommend DELETE |
| 4 | how often is poke-env WRONG about a sim fact, and does it reach the obs? | three registered findings, all reach the obs; rates below |

## 1. The view-fold gate on the training transport (`view_fold/`)

The M1 transport record (`../m1_transport_throughput_2026-09-23/`) found ≈ 60 % of M1's +13 %
per-decision Rust CPU in `SideObservation::observe`, which `sim_bridge` ran on every shipped line
while only `search_driver` and `core_events` read it. `gen3_view_fold_opt_in_v1` makes the fold
opt-in (`BridgeSession::enable_view_fold`); `sim_bridge` never enables it.

**Method.** That record's REPLAY bench, unchanged in what it measures — ONE persistent `sim_bridge`
fed the recorded 41-battle transcript (`../m1_transport_throughput_2026-09-23/transcript_env_seed0.txt`,
6,078 `CHOOSE`s), 5 replays per sample, CPU = the reaped child's user + sys — over THREE arms, pair
*k* running them in the *k*-th of the six orders so every arm sits in every position equally often:

| arm | commit | build | sha256 |
|---|---|---|---|
| A | `dfab2558` — before M1 (the record's A) | `git archive` → `/tmp/m2bench/A`, its own target | `b943fc4f8debd425…` |
| B | `7d71711c` — main before this fix | `git archive` → `/tmp/m2bench/B`, its own target | `432e42987d32cc90…` |
| C | this change (`bridge.rs` as landed) | this worktree's own target, copied out | `e8fe5e872b1068d0…` |

**Byte identity first**: all three arms emit the same **20,216,559-byte** stdout on the transcript
(sha256 `4d0e37a6…`, 41 `__END__`, 0 `__ERR__`) — and `cargo test`'s byte gates are green.

**Result** — 30 triples, load1 21.6–40.9 (`view_fold/analysis.txt`, raw `replay_abc_rows.jsonl`):

| ratio | CPU | wall | slower in |
|---|---|---|---|
| **C / A** (post-fix vs pre-M1) | **0.909** [0.876, 0.932] | 0.910 [0.882, 0.958] | 5 / 30 |
| **C / B** (post-fix vs pre-fix) | **0.833** [0.802, 0.851] | 0.841 [0.821, 0.864] | 2 / 30 |
| B / A (the M1 cost, re-read) | 1.087 [1.073, 1.105] | 1.090 [1.074, 1.103] | 27 / 30 |

Medians of the arms: A 222.8 ms, B 264.1 ms, C 216.3 ms of CPU per 41-battle replay. **Post-fix
`sim_bridge` is 9 % BELOW pre-M1**: the gate removes the whole fold, which A also paid (the M1 record
put it at 15.2 ms of A's 191.5), so it more than covers M1's typed-`Line` cost. The B/A re-read
(+8.7 %) is below the record's +12.9 % at a different load; both exclude zero.

```bash
python view_fold/run_replay_abc.py --bins A=<A>/sim_bridge B=<B>/sim_bridge C=<C>/sim_bridge --identity
python view_fold/run_replay_abc.py --bins … --pairs 30 --reps 5
python view_fold/analyze_abc.py view_fold/replay_abc_rows.jsonl
```

## 2 + 3. The core road: fork cost vs the view road, and what the typed shortcut saves (`search/`)

**Method.** `search_decision_benchmark.py` — the real `SearchEngine.choose` on decisions rebuilt
from BANKED eval traces (`models/ai_v12_02_winprob_critic/eval_traces`, read-only), the pure
`obs.sum` scorer — run with ONE road per process (the search-profile record's trap: the roads in one
interpreter are not independent), three roads per pair in the pair's order of the six:
`view` (the pre-M2 default), `core` (`materializer=core`, the TYPED shortcut) and `core-text` (the
same road, every successor folded from the side's rendered TEXT). `--rust-timing` sums the driver's
own `timing_us` (`core` = the version fold, `total` = the whole `expand_many`). Frozen binaries at
the landed code (`/tmp/m2bench/final`: `search_driver` `29f62ae5…`, `sim_bridge` `cca5d76d…`,
`core_events` `95b03d6a…`). Two widths: **wide** (10 decisions, honest arm, m_opp 3, k_worlds 4 →
684 arms per run) and **B = 1** (10 decisions, 1 action, 1 opponent, 1 world → 10 arms).

**Result** — 8 triples each (`search/analysis.txt`, raw `search/rows.jsonl`):

| | wide (load1 10.4–24.6) | B = 1 (load1 15.8–31.1) |
|---|---|---|
| ms / decision — view · core · core-text | 112.8 · 104.5 · 102.8 | 35.4 · 37.9 · 38.2 |
| **per decision core / view** | **0.926** [0.920, 0.944] | 1.116 [0.956, 1.276] |
| **per successor (Rust `expand_many`) core / view** | **1.408** [1.361, 1.477] | 1.442 [1.242, 1.688] |
| per successor FOLD typed / text | 1.019 [0.983, 1.316] | 0.912 [0.819, 1.087] |
| per successor `expand_many` typed / text | **1.089** [1.048, 1.229] | 0.964 [0.870, 1.168] |
| per decision typed / text | 1.015 [1.004, 1.421] | 0.989 [0.556, 1.220] |
| the shortcut's saving as a share of the core road's decision wall | −0.1 % [−1.1, +0.1] | 0.0 % [−0.0, +0.1] |

**Reading.**

- **The core road costs more in Rust and less overall.** A successor's version fold (≈ 0.08 ms/arm
  at wide B) makes the Rust `expand_many` ≈ 1.4× the view road's, and the decision is still ≈ 7 %
  cheaper, because Python no longer rebuilds the `LiveView` (`view_adapter`) or re-parses the ply
  (`ViewEventFolder`). B = 1 is not resolved (its CI straddles 1).
- **The typed shortcut saves nothing** — at both widths every typed/text ratio's CI contains 1 or
  sits above it, and its share of the decision wall is zero. A successor's cost is the board-reading
  clone plus the fold, identical on both paths; `Line::parse` of a ply's lines is a few µs; and the
  typed road's session records a source record per line, which makes its Rust `expand_many` ≈ 9 %
  DEARER at wide B. **Recommendation (a DECISION INPUT, program §6): DELETE the shortcut** — make
  `core_path=text` the only path, which is also §6c's observation path everywhere else, and with it
  `CorePath::Typed`, `typed_side_lines`, the search session's source recording and the integrity
  mode (its only job is typed == text). M2 leaves the default at `typed` for the owner's word.

```bash
python search/run_roads.py --bin-dir <bins> --src <tree>/src --traces <eval_traces> --label wide \
    --pairs 8 -- --decisions 10 --m-opp 3 --arm honest --k-worlds 4
python search/run_roads.py … --label b1 --pairs 8 -- --decisions 10 --n-actions 1 --m-opp 1 --k-worlds 1 --arm honest
python search/analyze_roads.py search/rows.jsonl
```

## 4. poke-env's reading findings — where poke-env is WRONG about a sim fact

`present()` reads the TRUTH; each disagreement with poke-env that the authoritative source (the
Rust board, the pinned Showdown source, the request JSON) settles against poke-env is registered in
`src/agents/battle/poke_env_findings.py` and carried by slice V as a NAMED, value-aware known
divergence — never a blanket tolerance. The fork is NOT changed (a training-input change is the
owner's call). **Reaches the obs** was established by encoding the `LiveView` poke-env builds and
the same view with the finding's field at the truth (`findings_reach_obs.py`,
`findings_reach_obs.txt`).

**Rates** — decisions on which the finding explained at least one field, per 1,000 decisions, both
viewers (slice V at `969c4e30` — the code of `9f77695d` before its rebase onto main: `milestone_census.txt`, `commit_census.txt`,
`procedural_slice_v.txt`). The core column itself: **0 divergences on every corpus.**

| finding | field | poke-env reads → the truth | pool random, MILESTONE (132,013) | `production` policy, MILESTONE (10,347) | procedural, seed 23 (29,783) | COMMIT (1,838) | reaches the obs |
|---|---|---|---|---|---|---|---|
| **PE-V10** | `boosts` | a fainted mon keeps the stages it died with until switched out → none (the faint's `clearVolatile`) | **4.64** (612) | **12.27** (127) | **6.45** (192) | 11.97 (22) | **YES** — the fainted active's stages in `active_context` at the forced-switch decision (+1 SpA = byte 4) |
| **PE-R1b** | `status_counter` | +1 per `\|turn\|` → the residual-chip STAGE: one AHEAD after a post-residual entry, one BEHIND between the residual and the next `\|turn\|`; frozen at a faint | **2.69** (355) | 0.58 (6) | **6.04** (180) | 0 | **YES** while active — the per-mon toxic slot (min(ctr, 8) / 8); the frozen count after a faint does not |
| **PE-V16** | `volatiles` | Flash Fire ended by its holder's own Fire move → kept until the holder leaves the field | 0 | 0 | 0 | 0 | **YES** — the `flashfire` binary volatile slot (byte 25 of `active_context`) — but **0 of 173,981 decisions**: a Fire move into a Flash Fire holder followed by the holder's own Fire move never happened in these corpora |

Reproductions (one side's text after a two-mon start; each is a pinned scenario in
`src/agents/battle/rust_core_present_test.py`, fed to poke-env AND the core):

- PE-V10 — `|-boost|p2a: Zapdos|spa|1` · `|faint|p2a: Zapdos` → poke-env `{spa: 1}`, the sim `{}`.
- PE-R1b — `|-status|p2a: Zapdos|tox` · `|-damage|p2a: Zapdos|94/100 tox|[from] psn` · `|turn|2` ·
  `|-damage|p2a: Zapdos|82/100 tox|[from] psn` → poke-env 1, the sim 2 (behind); or `…|turn|2` ·
  `|switch|p2a: Snorlax|…` · `|switch|p2a: Zapdos|Zapdos|94/100 tox` · `|turn|3` → poke-env 1, the
  sim 0 (ahead). Corpus cases: `random_56` p2 at turn 21 (behind), `procedural_9004` turn 38 (ahead).
- PE-V16 — `|switch|p2a: Houndoom|…` · `|-start|p2a: Houndoom|ability: Flash Fire` ·
  `|move|p2a: Houndoom|Flamethrower|p1a: Metagross` → poke-env: no `flashfire`; the sim: `flashfire`.

**Not findings.**

- **UNRESOLVED — a BENCHED badly-poisoned mon's count.** The sim stores the stage the mon left with
  and resets it on switch-in (`tox.onSwitchIn`); poke-env (and the core) read 0. The stored stage
  never acts again, so neither side is clearly wrong: counted by the board audit as
  `UNRESOLVED:tox-stage-benched`, never checked — 13,786 facts on the pool MILESTONE, 2,496 on the
  policy battles, 3,528 procedural, 405 COMMIT.
- **Information limits**, read as poke-env reads them because no client can know better: V15 (an own
  mon the current request did not re-sync holds the sighting count of its PP, which lags an
  un-announced Pressure deduction — the audit checks it can only lag; 19 facts procedural, 0 in the
  MILESTONE corpus) and V14 / R4 (a Transformed own mon's copied ability).
- **The VIEW ROAD's own projection defects** (procedural only; the core column is clean on them): a
  Trick shown as a `trick` volatile (`procedural_9124` turn 5, 4 facts) and a Mimic-copied move shown
  as `mimic` (`procedural_9146` turn 45, 1 fact); plus V15, which the projection never reproduced
  (19 facts). The road is on the program's deletion manifest; not fixed.


## Files

| file | what |
|---|---|
| `view_fold/run_replay_abc.py`, `analyze_abc.py`, `replay_abc_rows.jsonl`, `analysis.txt` | §1 |
| `search/run_roads.py`, `analyze_roads.py`, `rows.jsonl`, `analysis.txt` | §2 + §3 |
| `findings_reach_obs.py`, `findings_reach_obs.txt` | §4 — does each finding reach the obs |
| `procedural_slice_v.py`, `procedural_slice_v.txt` | §4 — slice V on 200 procedural-team battles |
| `milestone_census.txt` | §4 — slice V's MILESTONE census at the landed commit |
