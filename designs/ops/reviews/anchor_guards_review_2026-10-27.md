# Adversarial review: the anchor guards shipped 2026-10-26 / 27

**Date:** 2026-10-27 · **Reviewer:** adversarial measurement-integrity review (orchestrator unit) · **Base:** `c0fd72bd8`

**Scope.** The guard code of the last two days, each read against `designs/ops/EXTERNAL_ANCHORS_SOP.md` §0 (G1–G13)
and `designs/tools/reads.md`:

| commit | guard | SOP |
|---|---|---|
| `bf5f1a8e` | the forfeit-win guard: a non-battle termination is never an outcome | G6 |
| `e28318c8` | H5 fail-fast, H21 any peer fault voids the unit, the choice-level game-for-game guard | G2, G10 |
| `b0441623` | the Kakuna peer packs as Showdown reads; `peer_pack_changed` in the id; the import refusal | G4, G9 |
| `95150892`, `117235e9` | Foul Play's played team (`frontend_pack_changed`) | G9 |
| `20875e96` | the engine in the regime id; the cross-engine import refusal | G7 |
| — | the regime-id work in flight | not landed at `bd0860c8b`: not reviewed |

**Method.** (1) Read every changed path for a game or unit that should be VOID and still counts, an import that
should refuse and does not, a missing stamp read as "equal", fail-open on missing data, watchdog / end-of-half races,
and regime-id collisions. (2) Mutation testing: 30 single-line removals of a guard clause, each run against the fast
tier of `src/main/anchors`, `src/main/reads`, `src/utils/bridge/ws_frontend_test.py` and the X59-S tests
(`-m "not slow and not e2e"`); a SURVIVED mutant is a guard no fast test pins. (3) A read-only scan of the 372 anchor
summaries written in the last 15 h under `~/gen3ai_archive` and `measurements/` for the holes found.

**Verdict.** No BLOCKER: no counted unit in the scanned data is contaminated by any hole below, and every hole found
fails open only on a path that has not fired yet. Two MAJORs and five MINORs. The reproducers are strict `xfail`
tests (each flips RED when its hole is fixed, so the marker must then come off) plus one plain test that pins an
unpinned clause.

**Resolution (2026-10-27, the orchestrator's rulings).** H1: the two guards are re-derived at assemble
(`main.reads.rederive`, SOP G14): a hit voids the unit under a spec's `rederived_guards: count`, and HOLDS `decide`
for the reads registered before it (`report`: counting unchanged mid-read). H2: a missing expected capture is
`inconclusive`, never the 5-tuple alone (opt-out: the spec's `choices_guard: five_tuple`). H3: a forced end the
record cannot store moves the record aside, so the guard reads `blind`. H7: M24, M10 and M27 are pinned fast
(`src/main/anchors/guard_wiring_test.py`). H5 and H6 are rows of `designs/ops/TECH_DEBT_BACKLOG.md` §2(b); H4 is
open. The strict xfails below now pass as plain tests.

## Findings

### MAJOR

**H1 — Every guard of the last two days is grandfathered for any unit a pre-guard PIN plays, indefinitely.**
`results.unit_counts` judges a summary without `forced_end_guard` "as before", and `apply_peer_fault` (H21) exists
only in trees at or after `e28318c8`. That was written for BANKED summaries, but a registered read keeps playing at
its registration pin: of the 372 summaries written in the last 15 h, 357 carry no `forced_end_guard` (pins
`de15dfc7b`, `905c29f5b`, `0534e90a6`, `b044d214b`, ...; the rc72 and N1-P reads). A peer drop that books OUR win in
such a unit counts; so does a long-tail fault that ended rc 0 with no recorded error (the H21 hole `e28318c8` closed
only for new trees).
- *Data today (read-only scan):* clean. Their captures hold 33 FORCELOSEs: 26 our turn-250 cap, 5 our cap at the
  S-U smoke's 12 / 20-turn limit, 2 the peer's 1-turn forfeit after H5, both in uncounted `.partial` attempts. No
  counted unit's peer log holds the long-tail line.
- *Why MAJOR, not BLOCKER:* no contaminated unit exists; the trigger (a peer drop) needs a blocked loop that only the
  S-U search build produced.
- *Fix (recommended, not made: it changes the counting rule under live registered reads, the orchestrator's call):*
  re-derive both guards at ASSEMBLE time from what every pinned unit already writes: the capture's `FORCELOSE <slot>`
  (the slot's name from `START`, the turn from that slot's chunks) and the peer logs' `PEER_FAULT_LINE`. A unit with
  neither record stays "as before" and says so in its reasons.
- *Reproducer:* `forced_end_guard_test.py::test_an_unguarded_summary_whose_capture_shows_a_peer_drop_does_not_count`
  (strict xfail).

**H2 — The choice-level guard degrades to the 5-tuple whenever a capture it EXPECTS is missing, so one registered
decision PASSES or VOIDs by the environment.** `compare_choices` returns `unavailable` (and `guard_games` lets the
5-tuple decide alone) in three different situations: the bank names no source unit (structural: X51), the bank names
one but `<$GEN3AI_ARCHIVE_DIR>/<state>/cells/<unit>` is not there (moved, pruned, another box, the env var unset), or
the RE-PLAYED unit's own capture is missing. Only the first is a property of the registration. The other two are
fail-open on missing data: the same read decided after the archive is pruned, or on another machine, passes a guard
it fails here, and the 5-tuple misses about a quarter of the divergences (11 of 15 on E1 vs E3,
`designs/tools/reads.md` §4).
- *Fix (recommended):* `unavailable` only when the bank names no single source; a named source that cannot be read,
  or a re-played unit without its capture, makes `games_equal` False (VOID, re-run or re-register).
- *Reproducers:* `guard_choices_test.py::test_a_bank_that_names_its_capture_never_passes_on_the_5_tuple_when_the_capture_is_gone`
  and `::test_a_replayed_guard_unit_without_its_capture_never_passes_on_the_5_tuple` (strict xfail).

### MINOR

**H3 — The forfeit-win record fails open on a write error.** `ws_frontend._force_lose` catches an `OSError` on the
JSONL append and keeps the record only in memory and the log; `server.arm_forced_ends` already created the file EMPTY,
so `runner.forced_end_verdict` reads `armed` with no void and the dropped game counts as our win. Needs a full disk or a
permission change mid-read. *Fix:* on a failed append make the guard BLIND (remove or poison the file, or have the
runner cross-check the front end's `FORCED_END` log lines / the in-process `front.forced_ends`).
*Reproducer:* `forced_end_guard_test.py::test_a_forced_end_the_record_could_not_store_never_reads_armed_and_clean`
(strict xfail).

**H4 — A Foul Play cell played in the window `ce50227a5`..`95150892` collides with the banked Foul Play id.** The front
end began playing the validator's re-pack at `ce50227a5` (10-27 08:27); the stamp that splits the regime landed at
`95150892` (10:40). A Foul Play unit played by a tree pinned inside that window stamps no `frontend_pack_changed`,
reads `legacy` in `imports.check_peer_pack`, and takes the banked id while it played the new team (11 of 72 pool files
and `away`'s `team_23` play another set). *Data:* no Foul Play summary was written in the last 15 h under the archive or
`measurements/`. *Fix:* refuse (or treat as `frontend_changed`) a Foul Play unit whose `server_version` commit is in
the window; or never pin a Foul Play read there.

**H5 — `peer_pack_changed` means two things.** With `--team-sequence` (mirrored pairs) the Metamon peer packs and
compares only the files of the sequence, so whether a unit's id carries the pack split depends on which files its draw
hit; with a seeded draw it is the whole team set. A mirrored group on forum72 / away / the pool then spans two ids and
is INCONCLUSIVE although every unit played the post-fix pack. Fail-closed (never a wrong pooling), but a liveness
trap. *Fix:* compute the flag over the team SET in both paths.

**H6 — The import checks skip a panel this state has not counted yet.** `imports.check_peer_pack` compares imported
counted units only with this state's counted units of the same panel, and `anchor_rows` checks the engine only when
this state has a counted unit. A group whose units are all imported is therefore checked against nothing of this
read. Within the group, `engine.one` and the regime split still make a mix INCONCLUSIVE, so the exposure is a
fully-imported group compared with another group by a rule that has no `_same_engine` call. No current registration
does that.

**H7 — Six guard clauses no fast test pinned (mutation survivors, table below).** Three are now pinned by plain tests
in this change: the `user in ours` clause of `forced_end_verdict` (M1), the pair watchdog's H5 scan (M9) and the H21
call in `runner._half_report`, the half both transports share (M11: deleting it left every fast test green, so the
H21 fix could be reverted at its one call site unseen). Still unpinned in the fast tier:
- **M24**, `run_series` not calling `apply_forced_end_guard`: the whole forfeit-win guard unwired. Only the slow
  negative control `src/main/anchors/peer_patience_integration_test.py` catches it (`bf5f1a8e` verified it fails on
  revert); not run here.
- **M10**, the pair half's H21 loop (`run_peer_pair_half`): process orchestration with no fast seam.
- **M27**, `cli` dropping the `forced_end` block from the summary: harmless to counting today (the runner already
  failed the series and dropped the rows), but the record is lost and `unit_counts`' own check never runs.

### Checked and sound

- **Forced-end ordering.** `_force_lose` appends the record (synchronously, before the `FORCELOSE` reaches the child),
  so any row booked from a forced end has its record on disk before our side can see the `|win|`; both placements
  read the file after the last half; a reused directory is truncated at arm; a missing file is BLIND; Node and an
  external server are `unavailable` and never count.
- **H5 / H21 races.** The watchdog scans the log BEFORE the `done >= expected` return, so a fault in the poll that
  completes the half still fails it; whichever task wins, `apply_peer_fault` re-reads the whole log after
  `await_peer_exit` (180 s), and a peer still alive then is terminated (rc != 0, `peer_clean` False). The peer runs
  `-u` / `PYTHONUNBUFFERED=1` with stderr into the same log, so the line is never lost in a buffer. Either peer of a
  pair cell is scanned.
- **Pack stamps fail closed at the source.** A Metamon report without a verified `team_pack` record, or a Foul Play
  record that cannot be computed, sets `regime_verified_decisions` False (mutants M19, M20 killed).
- **Engine.** `engine_stamp` is the tree's build-stamp keys (every artifact is refused unless built from them),
  empty for an external server and under `POKESIM_ALLOW_STALE_BUILD=1`; `require_same` refuses an unknown engine on
  either side; `repin` refuses commits on different engines. The documented collision (two banked engines share an id)
  is policed by comparing engines on every import and inside every group.

## Mutation table

Each mutant removes or neuters ONE guard clause; KILLED means at least one fast test failed. After = with this
change's tests.

| # | mutant | file | before | after |
|---|---|---|---|---|
| M1 | `forced_end_verdict` drops `user in ours` | `src/main/anchors/runner.py` | SURVIVED | KILLED |
| M2 | ... drops `turn >= limit` | `src/main/anchors/runner.py` | KILLED | |
| M3 | ... drops the row's `hit_forfeit_limit` | `src/main/anchors/runner.py` | KILLED | |
| M4 | ... drops `not our_side_is_peer` | `src/main/anchors/runner.py` | KILLED | |
| M5 | `_force_lose` records after the result | `src/utils/bridge/ws_frontend.py` | KILLED | |
| M6 | `result_relayed` never set | `src/utils/bridge/ws_frontend.py` | KILLED | |
| M7 | `unit_counts` ignores `non_battle_endings` | `src/main/anchors/results.py` | KILLED | |
| M8 | `watch` without the H5 scan | `src/main/anchors/session.py` | KILLED | |
| M9 | `watch_peer_pair` without the H5 scan | `src/main/anchors/session.py` | SURVIVED | KILLED |
| M10 | pair half without H21 | `src/main/anchors/runner.py` | SURVIVED | SURVIVED |
| M11 | `_half_report` without H21 | `src/main/anchors/runner.py` | SURVIVED | KILLED |
| M12 | `compare_choices` never differs | `src/main/reads/assemble.py` | KILLED | |
| M13 | `guard_games` ignores choices | `src/main/reads/rules.py` | KILLED | |
| M14 | X59-S repeat unit ignores choices | `src/main/reads/rules_x59s.py` | KILLED | |
| M15 | `check_peer_pack` never refuses | `src/main/reads/imports.py` | KILLED | |
| M16 | `anchor_rows` skips the pack check | `src/main/reads/imports.py` | KILLED | |
| M17 | regime id without the pack identity | `src/main/reads/assemble.py` | KILLED | |
| M18 | regime id without the engine | `src/main/reads/assemble.py` | KILLED | |
| M19 | Metamon report: pack record missing, regime still verified | `src/main/anchors/peers.py` | KILLED | |
| M20 | Foul Play report: pack uncomputable, regime still verified | `src/main/anchors/peers.py` | KILLED | |
| M21 | `frontend_pack_changed` always False | `src/main/anchors/frontend_pack.py` | KILLED | |
| M22 | `anchor_rows` skips the engine check | `src/main/reads/imports.py` | KILLED | |
| M23 | void rows kept | `src/main/anchors/runner.py` | KILLED | |
| M24 | `run_series` never applies the forfeit-win guard | `src/main/anchors/runner.py` | SURVIVED | SURVIVED (slow tier only) |
| M25 | `any_half` becomes `all` | `src/main/anchors/runner.py` | KILLED | |
| M26 | `arm_forced_ends` writes nothing | `src/main/anchors/server.py` | KILLED | |
| M27 | `cli` drops the `forced_end` block | `src/main/anchors/cli.py` | SURVIVED | SURVIVED (record only) |
| M28 | `PeerFaultScan.final` drops the unterminated last line | `src/main/anchors/session.py` | KILLED | |
| M29 | `frontend_pack` counts refused files | `src/main/anchors/frontend_pack.py` | KILLED | |
| M30 | `engine.one` lets an unknown engine mix with a known one | `src/main/reads/engine.py` | KILLED | |

## The tests this review adds

| test | kind | pins |
|---|---|---|
| `src/main/anchors/forced_end_guard_test.py::test_a_peers_forfeit_at_the_cap_on_a_capped_row_is_void` | plain | M1 |
| `src/main/anchors/peer_fault_test.py::test_the_pair_watchdog_fails_at_the_first_fault_line_of_either_peer` | plain | M9 |
| `src/main/anchors/peer_fault_test.py::test_every_single_peer_half_report_applies_h21` | plain | M11 |
| `src/main/anchors/forced_end_guard_test.py::test_an_unguarded_summary_whose_capture_shows_a_peer_drop_does_not_count` | plain (was strict xfail) | H1 |
| `src/main/reads/guard_choices_test.py::test_a_bank_that_names_its_capture_never_passes_on_the_5_tuple_when_the_capture_is_gone` | plain (was strict xfail) | H2 |
| `src/main/reads/guard_choices_test.py::test_a_replayed_guard_unit_without_its_capture_never_passes_on_the_5_tuple` | plain (was strict xfail) | H2 |
| `src/main/anchors/forced_end_guard_test.py::test_a_forced_end_the_record_could_not_store_never_reads_armed_and_clean` | plain (was strict xfail) | H3 |

Each xfail was run with `--runxfail` and fails on its own assertion (the hole), not on a fixture error.

## What was not verified

- That Foul Play's own `load_team` walks exactly `frontend_pack.fp_team_files`' files (needs Foul Play's checkout).
- The archive beyond the last 15 h for H1 / H4 (the commits' own scans covered 1,249 units up to their ship time).
- The slow tier: `peer_patience_integration_test.py` (the live negative control, the only test that catches M24) was
  not run against the mutants; its fail-on-revert is `bf5f1a8e`'s own verification.
- The mutants were run on the fast tier only, so a "SURVIVED" can still be caught by a slow test.
