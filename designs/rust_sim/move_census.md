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

**Measured 2026-09-08 by `scan_move_probe`, after ROUND 59: 369 gen3-legal moves → 321 MODELED ·
48 FAIL-LOUD · 0 MISMODELED**; **abilities 76/76 and species 392/392 are CLOSED**; **items 102/106**
(the four fail-loud: `shellbell` / `machobrace` / `mentalherb` / `mail`). The full ranked gap, by
family and by legal-learner count, is [`designs/rust_sim/gen3_coverage_census_2026-09-08.md`](gen3_coverage_census_2026-09-08.md) —
a dated SNAPSHOT, not a current number.

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
class, and fix the always-current doc that carries a copy of this number (`src/utils/bridge/README.md`)
in the same pass — it (and the root `CLAUDE.md`, which no longer carries one) had gone stale by 28
moves before the 2026-08-23 doc audit.

🚨 **SWAGGER and FLATTER inflict confusion and are NOT in that family** — they carry a TARGET
`boosts` map whose half succeeds INDEPENDENTLY of the confusion half (probe-measured: into an
already-confused target the +2 Atk still lands and there is NO `-fail`; at the +6 cap the delta-0
`-boost|…|atk|0` prints AND the confusion still applies). They remain FAIL-LOUD; closing them needs
a positive foe-directed `targetBoosts` field in `gen3_moves.json` (`statDropBoosts` is
negative-only). ROUND 57 in the build log has the measurement.

🚨 **ATTRACT (338 learners — the largest single-move gap) IS BLOCKED BY A CONSTRUCTION DRAW, not by
Attract.** Its spec is settled and committed (`harness/probe_attract_move.js`), but
`MonState::from_set` stores only the PACKED gender, while the sim's ctor is
`set.gender || species.gender || sample(['M','F'])` — and that sample is a construction-time draw
the `start_with_switchins` path does not model. A `None` gender PANICS at the attract compare, which
is harmless today only because Cute Charm is on 0 pool teams. Admitting the MOVE would fail-loud
across every corpus. The prerequisite round is the construction gender, and modelling that sample
adds draws and moves **every committed golden's seed**. ROUND 59b has the full finding.
