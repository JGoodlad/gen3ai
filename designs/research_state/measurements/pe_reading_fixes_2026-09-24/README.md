# `gen3_pe_reading_fixes_v1` — the Rust core M2's poke-env READING findings fixed in the fork

<!-- A MEASUREMENT record (2026-09-24) for the TRAINING-INPUT change that fixed PE-V10 / PE-R1b /
PE-V16 in the vendored poke-env. Code + contract: designs/rust_sim/present.md §3. -->

The fix commit is `c97358e8` (rebased on `886ae28f`); everything below was measured at it, CPU only,
`nice -n 10`, no port, no `models/` write, no `data/` change, on a box carrying the live
population-loop arm (1-minute load average 21–27 throughout).

| # | question | verdict |
|---|---|---|
| 1 | does each fix hold, and does each pin FAIL on upstream? | **YES** — 9 protocol-line pins + 3 constructed real-Showdown battles, all 12 FAIL with the three fork files at the base |
| 2 | slice V with the findings registry EMPTY | **0 divergences** — COMMIT 1,838 decisions; MILESTONE 142,410 decisions / 81.7 M field comparisons / 26.3 M board-audit checks |
| 3 | which golden obs cells moved, and only those? | **4 / 991** decisions, all the TOXIC half of an opposing mon's `status_counters`; wider 80-battle set 65 / 9,987, only `context[*].boosts` of a FAINTED active and the toxic slot |
| 4 | the one-sided-view fuzz's P1 red (D10 arm, PE-V10) | **CLEARED** — reproduced red on the base (1 + 6 in 2 × 16); green after on 20 fixed-key sweeps × 16 battles; one of two fresh random seeds showed ONE decision of a new, unattributed class (`hp_block`, §6) |
| 5 | obs-build cost | **no measurable change** — the fix is on the protocol parse, not the encoder; the production shape 0.99–1.03× per interleaved pair |
| 6 | the benched badly-poisoned mon (UNRESOLVED) | **RESOLVED, both readings right** — the sim's stored stage is a dead store; the audit now CHECKS the effective stage 0 (≈16.3 k facts in MILESTONE that the old census only counted) |

## 1. The fixes and their pins

| finding | the fork change | protocol pin (`src/poke_env/battle/reading_fixes_test.py`) | constructed battle (`poke_env_gaps/pe_reading_fixes_obs_integration_test.py`, `impl="node"`) |
|---|---|---|---|
| PE-V10 | `Pokemon.faint` → `clear_boosts()` | faint clears; a `0 fnt` HP token clears | Snorlax Curse ×2 → Self-Destruct: the replacement decision's `active_context` boosts are EMPTY (upstream `{atk: 2, def: 2, spe: -2}`) |
| PE-R1b | `Pokemon.note_residual_chip` (from the `-damage` handler on `[from] psn`, after the HP token, cap 15); `Battle.switch` resets a TOX count; `end_turn` no longer ticks | no tick at `\|turn\|`; current between the residual and `\|turn\|`; 0 after a post-residual entry; the switch-in resets a stale count; a KO chip and plain poison never count; cap 15 | Snorlax Toxic'd → pivots to Magikarp → Magikarp dies AT the residual → Snorlax re-enters after it: every decision's count == the chips since entry re-derived from the RAW protocol, and the toxic obs slot matches; upstream read 1 at t10 where the sim has 0 |
| PE-V16 | the Flash Fire branch of `Pokemon.moved` removed | kept through Flamethrower + Fire Blast; ends at switch-out | Houndoom absorbs Charizard's Flamethrower then uses its own: `flashfire` stays in poke-env, `LiveView` and the obs slot (upstream lost it at t3) |

A lesson for the next constructed battle: in gen 3 a MID-TURN KO's replacement enters BEFORE the
residual, so the PE-R1b AHEAD shape needs the mon to faint AT the residual (the scenario's first
draft, with Seismic Toss finishing Magikarp, never produced it — the test's scenario guard caught
that). The players are deterministic (no `choose_random_move`): a random fallback made the Flash
Fire battle end differently run to run.

## 2. Slice V with the registry empty

`commit_census.txt`: 13 battles, 1,838 decisions, 1,035,734 field comparisons, 335,488 board
checks, findings `{}`. `milestone_census.txt` (the slow tier, recorded into
`designs/ops/slow_tier_status.json` at `c97358e8`, 5 passed in 545 s):

| corpus | decisions | field comparisons | board checks | divergences | findings |
|---|---|---|---|---|---|
| whole pool, even keys | 64,991 | 37,551,725 | 12,049,356 | 0 | {} |
| whole pool, odd keys | 67,022 | 38,763,924 | 12,428,930 | 0 | {} |
| `production` policy, keys 100–149 | 6,446 | 3,481,884 | 1,139,035 | 0 | {} |
| `production` policy, keys 6000–6049 | 3,951 | 1,930,891 | 650,494 | 0 | {} |
| protocol + byte-fuzz corpora (slice E) | — | — | — | 0 | — |

The board audit's new benched-toxic check (`status_counter[tox-benched]`) runs inside those board
checks; M2's census counted the same facts as `UNRESOLVED:tox-stage-benched` (7,250 + 6,536 +
1,988 + 508 in MILESTONE, 405 COMMIT, 3,528 procedural) — every one now CHECKED at 0 and passing.
Procedural (`procedural_slice_v.txt`, seed 23, 200 battles): see §6.

## 3. The golden census (`golden_diff_census.py`, `run_census.sh`)

Two captures of `golden_obs_capture`'s fixed battle set, the three fork files at the base vs the
fix (nothing else differs), each decision's BOARD recorded beside its vector; every changed cell
resolved from the declared layout and checked value-aware against that board.
`golden_census.txt`: the base capture reproduces the committed fixture **991/991**; count
unchanged; **4 / 991** decisions change, all `opp_team[slot].status_counters[toxic]` (idx 802 /
924 / 1046), 0 → 1/8 — the one-BEHIND shape (a decision between the residual chip and the next
`|turn|`). `wide_census.txt` (`WIDE="80 75"`, the same deterministic harness over 80 battles, 75
teams): **65 / 9,987** change — `context[*].boosts` at 55 decisions (87 cells, both sides, every one
a FAINTED active, fixed value 0) and the toxic slot at 10 — and nothing else: no derived block
(incoming damage, the reactive block, the damage op's inputs) moved. Flash Fire does not occur in
either set; the constructed battle covers it.

## 4. `one_sided_view_parity_fuzz_test.py` (`one_sided_fuzz/`)

* **Base** (`base_random_seed{1,2}.log`): RED, 1 and 6 divergences, all
  `core.successor.obs[context[0].species]` at a `core-D10` arm with PE-V10 present — the backlog's P1
  (b). Two things made it red: poke-env's kept stages, and `_block_of` naming the `context` block's
  columns from the PER-MON layout (a boost byte read `context[0].species`), so PE-V10's `context`
  pattern could never explain it.
* **Fix with the view road's boost LEDGER still in place** (`ledger_kept_*.log`): RED on 6 of 10
  fixed-key sweeps and 1 of 2 random seeds, every divergence a FAINTED mon's boosts at a
  `D10mid` view-road board (`protocol={}`, `view={'atk': 2, 'spe': 3}`): `event_fold`'s ledger +
  `view_adapter._restore_fainted_boosts` were reproducing upstream's V10. Deleted with the fix.
* **Final** (`final_*.log`): the PE-V10 class is GONE everywhere — see §6 for the sweeps.

## 5. `obs_build_benchmark.py` (`obs_build_benchmark/`)

Six interleaved pairs (base = the three fork files at `origin/main`; `before_1` was the unmodified
tree, before any edit). The benchmark walks a fresh random battle per run, so boards differ; the
box's load swung between two regimes, which dominates the absolute numbers.

| run | before: full / live_view / production shape (ms) | after: full / live_view / production shape (ms) |
|---|---|---|
| 1 | 0.310 / 0.064 / 0.089 | 0.590 / 0.102 / 0.148 |
| 2 | 0.528 / 0.102 / 0.151 | 0.508 / 0.102 / 0.154 |
| 3 | 0.376 / 0.108 / 0.145 | 0.568 / 0.103 / 0.150 |
| 4 | 0.549 / 0.104 / 0.153 | 0.554 / 0.105 / 0.151 |
| 5 | 0.572 / 0.101 / 0.148 | 0.540 / 0.102 / 0.153 |
| 6 | 0.382 / 0.079 / 0.111 | 0.365 / 0.078 / 0.110 |

Per interleaved pair (2–6) the production shape (cache warm + view memo warm) reads 1.02 / 1.03 /
0.99 / 1.03 / 0.99×. The change adds one membership test per `-damage` line and a boost clear
per faint to the PARSE; the encoder is untouched. No regression is measurable at this load; the
honest statement is "unchanged within the box's noise".

## 6. The final code on fresh battles

* **`one_sided_view_parity_fuzz_test.py`, fixed keys** (`final_key*.log`): **20 sweeps × 16 battles
  (keys 0–319), all GREEN**, ~5,500 branch points, ~490 D10 arms on each of the view and core roads.
* **Two fresh random seeds** (`final_random_seed{1,2}.log`): seed 2 GREEN; seed 1 RED on ONE
  decision in a class never seen before: `successor.obs[opp_team[0].hp_block]` (idx 828, the
  Hidden-Power type belief's FIRE probability of an opposing mon — protocol road 0.0, view AND core
  roads 0.000226) at `battle-gen3ou-2@t28/arm5`, `findings=[]`. **Not attributed and not
  reproduced**: the block is the `HiddenPowerTracker`'s, fed by the opponent's last damaging
  EVENT's effectiveness and the target's types / ability / frozen status — no boost, toxic counter
  or Flash Fire volatile enters it, and the view road's boost ledger (the only fold this change
  touched) was never read by `_build_event`. The 320 fixed-key battles above did not reproduce it,
  and a random battle cannot be replayed. It is REPORTED as an open, unattributed fuzz class (the
  backlog's P1 row carries it), not claimed pre-existing.
* **Procedural slice V** (`procedural_slice_v.txt`, M2's `procedural_slice_v.py --battles 200
  --seed 23`): 29,783 decisions, 5,431,214 board checks, **findings `{}`**, the core column clean;
  the 24 divergences at 23 decisions are EXACTLY M2's four VIEW-ROAD projection classes (V15 own
  bench PP ×19, a `trick` volatile ×4, a Mimic-copied move ×1 — same classes, same counts as
  `../rust_core_m2_2026-09-23/procedural_slice_v.txt`), the road on the deletion manifest.

## Files

`golden_diff_census.py` + `run_census.sh` (§3) · `golden_census.txt` · `wide_census.txt` ·
`commit_census.txt` · `milestone_census.txt` · `procedural_slice_v.txt` · `one_sided_fuzz/`
(`base_*`, `ledger_kept_*`, `final_*`) · `obs_build_benchmark/` (`bench_{before,after}_{1..6}.log`).
