# Rust Core M3 — the engine/transport split, `CoreError`, the trackers on the version, slice T

<!-- A MEASUREMENT record (2026-09-24) for the Rust Core Program's M3
(designs/endstate/program_rust_core.md §2 M3 and its "Before M3 starts" hand-off). Code:
designs/rust_sim/present.md (the version + engine), designs/rust_sim/trackers.md (the trackers). -->

Each section is one question, answered by an interleaved A/B with one binary per process, the
1-minute load average recorded at every run's start and end, `nice -n 10`, CPU only, no port, no
`models/` write, no `data/` change. **Ratios are the claim** (the box carries a production training
arm); the statistic is the per-pair ratio's median with a paired bootstrap 95 % CI on that median
(20,000 resamples, seed 0).

## 1. The engine/transport split (`gen3_core_engine_split_v1`) — byte identity and the fork's clone cost

**Byte identity.** `sim_bridge` fed the M1 record's 41-battle transcript
(`../m1_transport_throughput_2026-09-23/transcript_env_seed0.txt`, 6,078 `CHOOSE`s) emits the SAME
**20,216,559 bytes** (sha256 `4d0e37a6f80dcdd3…`, 41 `__END__`, 0 `__ERR__`) before the split
(`efd3ee78`) and after it — the digest M2's record and the emission self-check recorded. `cargo test`
green (854 tests), including `bridge_test`'s incremental-vs-genesis parity and the new
`tests/engine_split_test.rs`.

**The question** the program left UNMEASURED: what does a fork pay for the transport's history
(`script` + `request_seeds`, both growing with battle length)? `fork_clone/` —
`examples/fork_clone_bench.rs` replays the same transcript through ONE session and, at every one of
the 3,155 decision boundaries, times each clone 101× (median kept per boundary):

| arm | what | build |
|---|---|---|
| before `snapshot` | `BridgeSession::snapshot()` of a `clear_chunks`ed session — what a version fork cloned pre-split | `efd3ee78` |
| before `script_seeds` | `script` + `request_seeds` alone | `efd3ee78` |
| after `engine` | `Engine::clone` | this change |
| after `resume` | `BridgeSession::resume(engine.clone())` — the whole fork a version now drives (`fork_session`) | this change |

**Result** — 8 interleaved pairs (order alternating), load1 7.3–31.1 (`fork_clone/analysis.txt`,
raw `fork_clone/rows.jsonl`):

| ratio | median [95 % CI] | slower in |
|---|---|---|
| **after `resume` / before `snapshot`** (the fork, before vs after) | **0.950** [0.921, 0.952] | 0 / 8 |
| after `engine` / before `snapshot` | 0.947 [0.917, 0.952] | 0 / 8 |
| before `script_seeds` / before `snapshot` (the transport history's share of a fork) | **0.039** [0.038, 0.039] | — |

Medians of the per-boundary medians: before snapshot 18.7 µs, script + seeds 0.73 µs (mean script
length 55.7 decisions), after engine clone 17.5 µs, after resume 17.6 µs.

**Reading.** The split takes the transport's history OFF the fork — 3.9 % of it at this corpus's
mean battle length, and a share that grows with the battle — and the fork is 5 % cheaper. The other
96 % is the ENGINE: the `Battle` (its protocol log and, on a core session, its typed source records,
both also growing with battle length) and the driver's decision records. That is the next fork-cost
term, and it is not the transport's (a FINDING for M5, which owns the in-process env shape).

**A superseded first build (kept: `fork_clone/rows_rerender_superseded.jsonl`).** The first split
rendered the fresh transport's two outstanding `|request|` JSONs from the engine's typed requests at
every fork: **2.17×** [2.02, 2.81] the pre-split snapshot (≈ 27 µs of rendering). The landed engine
keeps, beside each typed request, the bytes it was ISSUED as (an `Arc<str>`, rendered once at issue
and shared by every fork) — so a fork's transport re-renders nothing, and `Engine::request_json` (a
fresh render) is pinned equal to them at every boundary by `tests/engine_split_test.rs`.

```bash
cargo build --release --example fork_clone_bench          # at each commit, copied out
fork_clone/run_pairs.sh <before> <after> <transcript> 8 fork_clone/rows.jsonl   # resumable
python fork_clone/analyze.py fork_clone/rows.jsonl
```

## 2. `CoreError` (`gen3_core_error_v1`)

`Result<T, String>` in `core_events`, `present` and `version` (and the engine's fatal condition)
became `CoreResult<T>`: `Refusal { exc: PyExc }` (poke-env raises the same Python class on the same
input), `Malformed`, `Fault`. Byte identity: the transcript's 20,216,559 bytes, sha256
`4d0e37a6f80dcdd3…`, unchanged (messages convert verbatim at the transport). The parity gate compares
the refusal CLASS (`rust_core_present_test.py::test_refusals_raise_the_same_class`: an unknown
keyword → `UnknownMessageType`, `-mega` → `UnsupportedMessageType`, `|gen|4` → `RuntimeError`,
`|turn|two` → `ValueError`; malformed input never dressed as a poke-env class). Writing the pins
found one misclassification (the event Reader's `int()` sites are poke-env's `ValueError`).

## 3. The trackers on the version — slice T (`gen3_core_trackers_v1`)

**COMMIT** (`rust_core_parity_test.py::test_commit_tier_trackers_equal_episode_tracker`, first at
`28851dc6`, green in every routine gate since, last at `493db48f`): **0 divergences** — 13 in-scope
battles, 26 viewers, 1,838 decisions, 52,674 event-window rows, labels MOVE 943 / SWITCH 487 /
masked 408, 1,838 per-decision rewards + 26 terminal. Teeth: a residual folded into a move's
`hp_delta`, a clock whose PROGRESS clause never fires, a phaze labelled a choice, an opponent's denied
choice leaking into a viewer's record (`[BOUNDARY]`, M3 (3)) — each FAILS.

**MILESTONE** (at `493db48f`, `slow_tier_status.json`): **5 / 5 pass** — slices E + V + T (incl. the
information-boundary check) on 2 × 360 seeded-random and 2 × 50 production-policy battles played
live, plus the protocol and byte-fuzz corpora (19 min 53 s beside the catalogue; the per-battle
slice-T census is the catalogue's, §5).

**Fresh battles** (`rust_core_trackers_fuzz_test.py`, 6-minute runs, pool + mechanic-dense +
procedural teams): at `493db48f`, seed 1430675900 (`catalogue/tracker_fuzz_493db48f.log`), slices
E, V and T **0 divergences** over 176 battles, 39,110 decisions, 1,168,975 event-window rows; the
native record's coverage in that run: 827 denials, 4,900 refused actions, 276 Baton Pass entries,
1,490 drags, 230 multi-faint windows, 68 Pursuits on a switch, faints by Destiny Bond 4 and Perish
Song 14. An earlier run at `1b25086b` (measured pre-rebase as `c9f9c589`, the same tree; 185 battles, 35,660 decisions) was 0 on T and E; its slice V
flagged a procedural Rollout battle (the core's board audit: own Rollout PP read 30, engine 31) — the
port's missing `[from] lockedmove` on lock-in continuations, which the ladder-corpus agent found and
fixed the same hour (`f865e8ab`, `gen3_lockedmove_announce_v1`, not yet on main at this writing); an
independent reproduction, not a poke-env finding.

**The one rule the port had to take from poke-env rather than from the decision window**: the
Hidden-Power belief's input is `battle.opp_last_damaging_move` (pending at the damaging `|move|`,
promoted by the defender's effectiveness line, turn-gated). A window-scoped reading (the rule
`view_successor.view_context._to_dme` still uses for search) diverged from training on the HP belief at
57 of 1,838 COMMIT decisions. The mechanism that makes them differ is a window that does not hold the
whole previous turn (one that opened at a mid-turn forced switch); that each of the 57 is that shape
is **UNVERIFIED** (not broken down). On search, a depth-1 successor's window is its whole ply, so the
difference is reachable at depth ≥ 2.

## 4. The tracker FORK's cost in a searched decision (`fork_cost/`)

**Method.** M2's search benchmark (`search_decision_benchmark.py`, the real `SearchEngine.choose` on
10 decisions rebuilt from BANKED `ai_v12_02_winprob_critic` eval traces, read-only; the pure `obs.sum`
scorer), ONE road per process, interleaved (`fork_cost/run_roads.py`, resumable): `core` (today's
production core road — the successor's trackers are the PYTHON ones, forked by the pinned-pickle
THAW) and `core-trk` (the same road with the Rust core's trackers folded on every version,
`SearchConfig.core_trackers`). Frozen release binaries (`fork_cost/bins.sha256`). Two widths: **wide**
(honest arm, m_opp 3, k_worlds 4 → 684 successors over 10 decisions) and **B = 1** (10). 8 pairs each.

| | wide (load1 25.8–40.5) | B = 1 (load1 29.9–43.3) |
|---|---|---|
| decision wall, `core` road | 321 ms [261, 450] | 72 ms [64, 84] |
| **BEFORE — Python THAW / successor** | **0.46 ms** [0.40, 0.62] | 0.62 ms [0.56, 1.36] |
| **BEFORE — the thaw's share of the searched decision** | **9.9 %** [9.3, 10.3] | 0.8 % [0.7, 1.8] |
| BEFORE — thaw + Python `record_context` + `advance_window` / successor † | 0.69 ms [0.58, 0.82] | 3.9 ms [3.1, 4.5] |
| **AFTER — the Rust tracker fold / successor** (fork + record + advance, fold + render delta) | **0.089 ms** [0.027, 0.125] | 0.066 ms [0.047, 0.076] |
| **AFTER — its share of the searched decision** | **1.8 %** [0.7, 2.4] | 0.08 % [0.07, 0.10] |
| AFTER / the thaw alone, per successor | **0.17** [0.07, 0.30] | 0.10 [0.06, 0.13] |

† the Python-trackers phase also counts the ROOT prefix replay's own `record_context` /
`advance_window` calls (which dominate at B = 1); the thaw alone is purely per successor.

**Reading.** The Rust fork of the tracker state is an `Arc` clone (a pointer); the 0.089 ms is the
successor's own `record_context` + `advance_window` fold, copy-on-write — ≈ 1/6 of the thaw ALONE,
and the thaw is ≈ 10 % of a wide searched decision here (the program's 16.9 % was an earlier profile
on a different configuration). **Search does not get this yet**: its successors are still encoded by
the Python encoder, which reads the Python trackers, so production search still thaws. M4's encoder
reading the version's trackers is what removes the thaw; `core_trackers` stays OFF until then (it
would add the Rust fold without removing the Python one).

```bash
python fork_cost/run_roads.py --bin-dir <frozen release bins> --src <tree>/src --traces <eval_traces> --label wide --pairs 8 -- --decisions 10 --m-opp 3 --arm honest --k-worlds 4
python fork_cost/run_roads.py … --label b1 --pairs 8 -- --decisions 10 --n-actions 1 --m-opp 1 --k-worlds 1 --arm honest
python fork_cost/analyze.py fork_cost/rows.jsonl
```

## 5. The loss catalogue — what the frozen layouts keep of what happened (`catalogue/`)

**Question.** The native record (`record::Window`, trackers.md §3) holds each decision window as an
ordered list of attributed actions. What do its three frozen consumers keep of it — the
`TurnDelta` (now only the label's oracle and the progress clock's input), the α/β opponent-intent
LABEL, and the 22-column EVENT WINDOW (the obs's only route for history since
`gen3_frame_deletion_v1`) — and at what rate per 1,000 decisions?

**Method.** Two instruments, both on the frozen self-check binaries `catalogue/bins.sha256`:

* **The corpus rate** — `catalogue.py`: the whole MILESTONE corpus (2 × 360 seeded-random + 2 × 50
  production-policy battles, the `rust_core_parity` key recipe), one battle per UNIT, one durable
  row per unit (`rows.jsonl`), resumable (`resume_proof.log`: a 5-unit first leg, then the detached
  run resumed from its rows). Every battle is ALSO held to slice T, so every label and every
  event-window row counted IS what training builds. The progress-clock case is a COUNTERFACTUAL on
  the Python path: at each progress verdict, the clock is re-asked with `our_damaging_event`
  removed; "decisive" = clause (i) alone reset it.
* **The per-mechanic evidence** — `e12_fixtures.py` replays every constructed fixture of
  `tests/window_record_test.rs` (their inputs: `e12_fixtures.jsonl`, written by the ignored
  `export_fixtures`) through `core_events --trackers` with slice T on (scope widened to
  `gen3customgame` for them; 0 divergences on all 25), and prints per decision the native record,
  the full `TurnDelta`, the label, the clock and the event-window rows appended (`e12_fixtures.txt`).
  `gigo_repros.jsonl` / `gigo_repros.txt` are the constructed REPRODUCTIONS of the GIGOs below;
  `examples.py` finds one real MILESTONE decision per case.


**Verdict at the registered n: 820 / 820 battles, 142,410 viewer decisions (132,013 seeded-random +
10,397 policy), slice T 0 divergences, 0 information-boundary violations** (`catalogue/report.txt`).
Rates are per 1,000 viewer decisions (each battle counted from both sides, so a symmetric event is
counted twice — once as "ours", once as "opp").

### 5.1 What happened — the native record's census

| case | /1,000 | n |
|---|---|---|
| a side took ≥ 2 actions in one window | 65.61 | 9,344 |
| ≥ 2 faints in one window | 9.39 | 1,337 |
| **DENIED — fainted first** | **24.91** | 3,547 |
| **DENIED — TURN CUT (gen 3: a faint cancels every queued action)** — first faint by Spikes on a chosen switch 416, a self-KO (Explosion / Self-Destruct) 220, recoil 10 | **4.54** | 646 |
| **REFUSED** (`cant`: par, slp, frz, flinch, recharge, Focus Punch, Taunt, Disable, Truant …) | **99.14** | 14,118 |
| a replacement switch (the free switch after a faint) | 102.28 | 14,565 |
| a drag (Roar / Whirlwind) | 31.89 | 4,542 |
| a Baton Pass entry | 6.64 | 946 |
| Pursuit on a switching target | 3.01 | 428 |
| a hazard KO on entry | 4.53 | 645 |
| a called move (Sleep Talk / Mirror Move) | 0.80 | 114 |
| a move blocked by Protect / Detect | 17.50 | 2,492 |
| a Substitute took a hit | 4.54 | 646 |
| a Wish heal resolved | 13.05 | 1,858 |
| Rapid Spin cleared a hazard | 8.67 | 1,234 |
| a stat stage DROP (`-unboost`) | 58.39 | 8,316 |
| an item line by Thief / Covet | 0.81 | 116 |
| a multi-hit move | 2.81 | 400 |
| a charge turn / a recharge turn | 0.07 / 0.04 | 10 / 6 |
| a faint by Perish Song / by Destiny Bond | 0.27 / 0.00 | 38 / 0 |
| an item line by Trick; an Encore landing before its target moves | 0.00 | 0 |

The corpus holds no Trick, no Destiny Bond KO and no same-turn Encore; each is covered by a
constructed fixture instead (`e12_fixtures.txt`), and its corpus rate is 0 with that caveat.

### 5.2 ACTION DENIAL and the α/β label

On every opponent denial the label is **MASKED**, as it should be: fainted first 1,769 / 1,769, turn
cut 325 / 325. Refusals: masked 6,853; MOVE 118; SWITCH 88. The breakdown over the random half
(`refused_breakdown.txt`): 109 of the MOVE labels are a lost Focus Punch naming Focus Punch (the
opponent DID choose it; a correct intent label), 1 is a `cant nopp` naming another move; all 87
SWITCH labels are a refusal FOLLOWED BY OUR DRAG (slp 39, par 36, frz 10, Truant 2) — GIGO L2 below.

### 5.3 α/β LABEL GIGO (repro · rate · reaches). None fixed in M3.

| id | what happened | what the label says | repro | /1,000 (n) |
|---|---|---|---|---|
| L1 | the opponent was DRAGGED by our Roar / Whirlwind after its own switch (or with its action outside the window) | SWITCH **the dragged mon** — a mon it never chose | `gigo_repros` `label_roar_overrides_their_chosen_switch` (chose Blissey, dragged Starmie → SWITCH starmie); `random_1` p1 after chunk 318 | **7.46** (1,063; random half: after a switch 908, after a refusal 87, neither 62 — every one names the dragged mon) |
| L2 | the opponent was REFUSED (asleep, paralysed, frozen, Truant), then dragged | SWITCH the dragged mon (its choice was a move) | `gigo_repros` `label_asleep_then_dragged_reads_as_switch`; `random_1` p1 after 322 | 0.62 (88; a subset of L1) |
| L3 | the opponent's REPLACEMENT after a faint arrives in the NEXT decision window (the faint lay in the previous one) | SWITCH the replacement — the same forced switch is MASKED when the faint shares its window (4,276) | `random_5` p1 after chunk 130 | **8.38** (1,194) |
| L4 | a called move (Sleep Talk → Rest) | MOVE **the called move**, never the caller the opponent chose | `e12` `sleep_talk` p2; `random_10` p1 after 34 | 0.32 (46 of 57; 11 masked) |
| L5 | an Encore lands before its target moves; the target's chosen Body Slam executes as the encored Curse | MOVE the encored move | `e12` `encore_lands` p1 after chunk 14 | 0.00 in the corpus (fixture only) |

**Reaches:** the α/β opponent-intent heads' TRAINING TARGETS (an auxiliary loss), not the obs.

### 5.4 The 22-column EVENT WINDOW (the obs's history) — GIGO and losses. None fixed (training input).

| id | what happened | what the window records | repro | /1,000 (n) |
|---|---|---|---|---|
| **W1** | any stat stage DROP (Intimidate, Curse's Speed, Overheat, a Screech) | a BOOST row whose MAGNITUDE is **+** — `EventWindowTracker.update` negates an `amount` the event already signs (`gen3_battle._build_event`), so every drop reads as a RISE | `e12` `encore_lands` (Curse: spe −1, atk +1, def +1 → three +1 rows) | **58.39** (8,316) |
| W2 | a faint by Perish Song / Destiny Bond (no damage line precedes it) | FAINT_CAUSE `attack` (the vocabulary has neither; the classifier falls to "no `[from]` ⇒ attack") | `e12` `perish`, `destiny_bond`; `random_14` p1 after 266 | 0.27 (38) / 0.00 |
| W3 | Trick swaps two items (`-item … [from] move: Trick` on both) | ITEM_TRANSITION **REVEALED** twice, never SWAPPED (`constants.py` documents Trick as SWAPPED; only an `-enditem` can reach it) | `e12` `trick` | 0.00 (fixture only) |
| W4 | a move blocked by Protect / Detect | the move row reads OUT_HIT (magnitude 0) — and so does `TurnDelta.*_move_outcome`, which makes the progress clock's "our attack blocked ⇒ exogenous freeze" branch unreachable (it tests `== "fail"`): a blocked attack is CHARGED as a no-op | `e12` `protect` | 17.50 (2,492 blocks; ≈ 8.75 our attacks blocked, per viewer) |
| W5 | Rapid Spin clears Spikes | a HAZARD row identical to a hazard being SET (no content, no set/clear bit) | `e12` `rapid_spin`; `random_1` p1 after 202 | 8.67 (1,234) |

**Losses (true but absent or flattened, not wrong):** a DENIAL has no row (fainted first / turn cut —
only the absence of a move row, and the rule that explains it is gen-3 knowledge); a drag and a
Baton Pass entry are SWITCH_IN rows indistinguishable from a chosen switch (the preceding Roar /
Baton Pass move row is the only tell; the passed boosts are absent); the Spikes chip on entry and a
Wish heal have NO row (DAMAGE / HEAL rows exist only as attachments to a move); a Substitute hit is
OUT_HIT with magnitude 0; a charge turn is OUT_HIT with no charge marker; a lost Focus Punch is an
OUT_HIT move row followed by a CANT row; a called move is two move rows with no link; a Pursuit on a
switch keeps its order but not its flag, and its `we_first` is set though the switch was queued
first; a multi-hit move's per-hit crit and effectiveness collapse into one row; the second faint of
a simultaneous KO carries the `forced_window` tag.

### 5.5 `TurnDelta` and the progress clock — GIGO. None fixed.

* **T1 — the clock's clause (i) credits damage our move did not do.** `TurnView`'s `damaging_move` is
  ANY move whose named target's NET HP fell over the turn, from any source; a `[still]` or
  self-targeted move gets the opponent's active as its target. So a Taunt, a Toxic, a FAILED Soft-Boiled
  or a missed Hydro Pump is "our damaging event" whenever Sandstorm, poison, Spikes on the entrant or
  the opponent's own recoil chips it ≥ 3 %. Clause (i)'s inputs hold spuriously at **24.21 / 1,000**
  (3,448); the counterfactual (the same verdict with `our_damaging_event` removed) shows clause (i)
  ALONE reset the clock at **1.28 / 1,000 (182)**. Repro `gigo_repros` `taunt_in_sand_reads_as_progress`:
  the clock stays at n = 0 through three Taunts (one of them failed) while the control (Rest in the
  same sand) climbs 1 → 2 → 3. **Reaches the obs** (the progress-clock scalar and
  `turns_since_progress`); the clock's reward charge is inert in the production config.
* **T2 — W4 above** (a Protect block is an outcome `hit`; the freeze branch is dead): reaches the obs clock.
* Distortions no consumer reads today: a Wish heal lands in the opponent's damaging move's
  `our_target_hp_delta` as a POSITIVE "damage" (`e12` `wish_sub_protect`); Sleep Talk sets
  `*_failed_to_move` AND `*_move_id` = the called move; a lost Focus Punch is outcome `hit` AND
  `failed_to_move`; OUR fainted-first / turn-cut denial leaves no field at all.

### 5.6 E12 — per mechanic: what the record holds, what each frozen layout keeps

| mechanic | native record | `TurnDelta` (label oracle, clock input) | α/β label | 22-col event window | /1,000 |
|---|---|---|---|---|---|
| Roar / Whirlwind | `Drag { to, out, by, by_move }` + the entrant's Spikes chip with its layers | `opp_switch_to` = the dragged mon (≡ a chosen switch) | masked when the opponent also moved; **L1 / L2** otherwise | SWITCH_IN ≡ a chosen switch; the chip has no row | 31.89 |
| Baton Pass | `Switch { entry: BatonPass { passer, boosts, volatiles } }` | move `batonpass` + `switch_to`; contents lost | MOVE `batonpass` (a failed pass, 169) or masked (473) — both acceptable | MOVE row + SWITCH_IN ≡ chosen; contents lost | 6.64 |
| Thief / Covet | both item lines with direction | the victim's `item_lost` only | normal | victim SWAPPED ✓, taker REVEALED ("still held" for a new item); no item identity | 0.81 |
| Trick | both item lines, `[move Trick]` | nothing | normal | **W3**: REVEALED ×2 | 0 (fixture) |
| Knock Off | the removal | `item_lost` | normal | REMOVED ✓ | (in 0.90 transfers) |
| sacking / hazard KO | faint `Spikes { layers }`, the free switch `Replacement { fainted }`, the TURN CUT of the opponent's queued action | faint-cause vector; forced-switch phase | masked | FAINT `hazard` ✓ (layers lost); replacement tagged `forced_window` ✓; the turn cut absent | 4.53 + 102.28 |
| Pursuit | `Move { pursuit_on_switch }` before the switch | move + `switch_to`; flag lost | SWITCH (their real choice) ✓ | order ✓, flag lost, `we_first` distorted | 3.01 |
| trade KOs | faints in order with cause; `Denied { FaintedFirst { by } }` | both fainted + counts + causes; ORDER lost | masked ✓ | FAINT rows in order ✓; **W2** for Destiny Bond / Perish Song | 9.39 |
| called moves | `Move { called_by }` after the caller | `move_id` = the called move | **L4** | two move rows, no link | 0.80 |
| Rapid Spin | `side_end [move Rapid Spin]` naming the condition | nothing | normal | **W5** | 8.67 |
| Wish | residual heal `Wish { wisher }` | heal in hp deltas; distortion (5.5) | — | no row | 13.05 |
| Substitute | `SubstituteHit { broke }` | outcome `hit`, target delta 0 | — | OUT_HIT, magnitude 0 | 4.54 |
| Encore | the volatile on its target; the executed move is the encored one | `move_id` = the encored move | **L5** | two move rows; the override invisible | 0 (fixture) |
| Disable / Taunt | `Cant { reason, move_id }` (the move only when the line is public) | `*_cant_reason` ✓ | masked ✓ | CANT row with the move ✓ | in 99.14 |
| Focus Punch (lost) | `Move { still }` + `Cant { Focus Punch }` | `hit` + `failed_to_move` | MOVE `focuspunch` (a correct intent) | OUT_HIT row + CANT row | in 99.14 |
| charge / recharge | `Prepare` / `Cant { recharge }` | outcome `hit` / `cant_reason` ✓ | normal / masked ✓ | OUT_HIT, no marker / CANT ✓ | 0.07 / 0.04 |
| Protect | `Blocked` on the stopped move | outcome `hit` (**W4 / T2**) | normal | OUT_HIT (**W4**) | 17.50 |
| ACTION DENIAL | `Denied { FaintedFirst / TurnCut }`, `Cant`, the opponent's choice unrepresentable | `*_failed_to_move` for a `Cant` only | masked ✓ (L2 aside) | CANT ✓; a denial has no row | 24.91 + 4.54 + 99.14 |
| stat drops | `Boost { stat, n < 0 }` | boost deltas ✓ (signed) | — | **W1** sign inverted | 58.39 |

### 5.7 Reshape candidates for the next obs-arch change (NOT done here — no training-input change)

The event window is the obs's only history route, so the reshape lands THERE (and in the label), not
in `TurnDelta`, which leaves with the deletion pass. In rough order of rate × harm: sign the BOOST
magnitude (W1, 58 / 1,000); give the label a MASK for any window where the opponent was dragged or
its replacement straddles the window (L1 + L3, ≈ 16 / 1,000) and label the CALLER of a called move
(L4); an outcome bit for BLOCKED (W4) and read it in the clock (T2); attribute clause (i) to the
move's own hit (T1); a DENIED row (fainted-first / turn-cut / refused) carrying the information
boundary as the record does; an entry-reason column on SWITCH_IN (chosen / replacement / drag /
Baton Pass); a set / clear bit and the layer count on HAZARD rows (W5); `perishsong` and
`destinybond` in the faint-cause vocabulary (W2); Trick → SWAPPED (W3); DAMAGE / HEAL rows for
entry chip and Wish; a charge / substitute marker.

### 5.8 poke-env reading findings

One, and it is a reading CRASH, not a disagreement: **poke-env raises `ValueError` on the gen-3
called-move line `|move|<mon>|<X>|<target>|[from] <Caller>` for Metronome, Assist and Nature Power**
(it reads Sleep Talk and Mirror Move). Metronome is the backlog's P1 row; **Assist and Nature Power
are beyond it.** Repro: `window_record_test.rs::every_gen3_caller_is_recorded_as_caller_then_called_or_refused_as_poke_env_refuses`
(the core refuses the same lines with the same class; the pin flips the day the fork reads them).
Rate in the MILESTONE corpus: 0 (no battle refused). Reaches: a live ladder game dies at the parse and
loses on the timer. Every other finding above is in TRAINING's own layers (the event window, the
label, the clock, `TurnDelta`) or in the port — none is poke-env's.

```bash
python catalogue/catalogue.py run --out catalogue/rows.jsonl      # detached, resumable; frozen bins: catalogue/bins.sha256
python catalogue/catalogue.py report catalogue/rows.jsonl > catalogue/report.txt
python catalogue/refused_breakdown.py > catalogue/refused_breakdown.txt
python catalogue/e12_fixtures.py catalogue/e12_fixtures.jsonl > catalogue/e12_fixtures.txt
python catalogue/e12_fixtures.py catalogue/gigo_repros.jsonl > catalogue/gigo_repros.txt
python catalogue/examples.py --keys 0 360 --per-case 1 > catalogue/examples.txt
```
