# The gen-3 move census — how to count it, and the blocked moves

> Moved VERBATIM from `src/rust_sim/CLAUDE.md` in the 2026-10-10 leaf cleanup (links re-based;
> statements found FALSE against the code were corrected in place, each saying so). The leaf keeps a
> one-line pointer here. Frozen original: `designs/research_state/claude_md_archive/src_rust_sim_CLAUDE_2026-10-10.md`.

#### 🚨 The move census is RECOUNTED, never quoted

**Every ROUND entry below states the census as of THAT round, and each says "current" because it
was.** They are round-scoped history — do not read one as today's number. The **only** current
answer is the tool:

```bash
# THE UNIVERSE COUNT — the ENGINE is the oracle (all four gen3 universes)
cd src/rust_sim && cargo build --release --bin scan_move_probe
PROBE_KIND=move|species|item|ability ./target/release/scan_move_probe < ids.txt   # JSON verdict per id
# the POOL report + the 0-MISMODELED invariant gate — a DIFFERENT question, keep both
cd src/rust_sim/harness && SCAN_UNIVERSE=1 SCAN_UNIVERSE_LIST=1 node scan_move_coverage.js
cd src/rust_sim/harness && node scan_move_coverage.js                   # the 762-team pool report
```

🚨 **THE UNIVERSE COUNT IS THE PROBE'S TO STATE, NOT THE JS SCAN'S.** `scan_move_coverage.js`
computes its verdict from modeled-move sets **mirrored BY HAND** from `turn.rs`, and on 2026-09-08
those mirrors were three moves stale (ROUND 51's `defensecurl`, ROUND 52's `minimize` + `imprison`) —
it said 309/60 where the engine itself runs **312/57**. `scan_move_probe` cannot drift, because it
*is* the engine running. The JS scan keeps the two jobs the probe cannot do: the team-pool report and
the `0 MISMODELED` invariant gate.

**The current counts, per universe, with the formats, the engine mechanics and the usage-ranked gap,
are in [`gen3_full_parity.md`](gen3_full_parity.md)** (recounted with the tool; the 09-08
[`gen3_coverage_census_2026-09-08.md`](gen3_coverage_census_2026-09-08.md) is the dated predecessor).

🚨 **"MODELED" MEANS "THE ENGINE DOES NOT FAIL LOUD" — IT DOES NOT MEAN "GATED", AND IT DOES NOT MEAN
"THE DRAW COUNT IS RIGHT".** ROUND 57 found that `confuseray` — shipped at ROUND 44, carrying a
named revert-verified pin, and counted MODELED by every census since — **never rolled its accuracy**,
consuming one draw fewer than the sim on every use. Its pin asserted EMISSIONS and no seed. Of the
312 moves the engine ran before that round, **36 are played by no committed battle golden at all**
(the census's first tier table said 0, because it counted a `dex_golden.txt` row as battle exposure).
When you touch a move, check whether anything has ever EXECUTED it. Pool **762/762** fully engine-playable (813 `.txt` files, 51 validate-fail — the
count MOVES as the pool grows, and it read 722/722 when the pool was 40 teams smaller). ⚠️ **This
is NOT the 719-team TRAINING pool and the two must not be "reconciled".** 762 is what
`Teams.import` + `TeamValidator('gen3ou')` accept out of `data/teams/*.txt`; **719** is what
`utils.team_loader.TeamLoader.get_all_teams()` returns to TRAINING (72 sample + 647 other,
measured 2026-09-07). Different filters, both current. For scale, the ROUND-40 entry says 281/88 and ROUND 44 says 286/83 —
both were true when written. **The invariant is the load-bearing claim, not the split:** 0
MISMODELED is what makes an unmodeled move a loud construction failure rather than a silent
desync, and it has held under every round separately and combined. Re-run after admitting any move
class, and never copy this number into another doc (point at the generator): a copy in the bridge README and the
leaf had gone stale by 28 moves before the 2026-08-23 doc audit.

🚨 **SWAGGER and FLATTER inflict confusion and are NOT in that family** — they carry a TARGET
`boosts` map whose half succeeds INDEPENDENTLY of the confusion half (probe-measured: into an
already-confused target the +2 Atk still lands and there is NO `-fail`; at the +6 cap the delta-0
`-boost|…|atk|0` prints AND the confusion still applies; only a capped boost into an
already-confused target fails). ROUND 57 left them fail-loud; gen3 full parity B6
(`gen3_b6_moves_v1`) closed them with their own arm and the boost maps HARDCODED in Rust
(`TARGET_BOOST_MOVES`, checked against the resolved dex by `tests/b6_moves_test.rs` B6M1)
instead of a `gen3_moves.json` field. Spec: `harness/probe_b6_moves.js`.

🚨 **ATTRACT is MODELED since gen3 full parity B9 (`gen3_attract_move_v1`), and its old blocker is a
harness rule now.** The sim samples an unspecified gender at construction (`sample(['M','F'])`): the
raw-seed bridge path draws it (`run_turn0_construction`), but the post-construction-seed path every
golden, probe and fuzzer uses (`start_with_switchins`) cannot know it, so Attract and Cute Charm PANIC on
an unspecified gender. Every harness pins genders (`scan_move_probe` M into F, `ab_fuzz.js` random mode
draws an explicit one per set, the byte paths `pinGenders`). Spec `harness/probe_attract_move.js`;
gates `tests/attract_move_test.rs`, `tests/attract_move_golden_test.rs`.
