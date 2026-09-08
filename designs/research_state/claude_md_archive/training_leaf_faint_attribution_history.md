# Faint attribution in the trace — the closed history

Lifted verbatim from `src/agents/training/CLAUDE.md` § *Faint attribution in the trace* on
**2026-09-08**, when that leaf was split by topic a second time (3,183 lines → ~700).

**This directory is HISTORY — additive only, never updated afterwards.** The defect below is
FIXED (`gen3_faint_attribution_v1`) and its gate is live; the training leaf carries the two
standing rules the incident produced (a forensic recorder must never take down a run, and a
protocol identifier carries the NICKNAME rather than the species). Do **not** re-derive a plan
from anything here without checking the current code first.

---

## Faint attribution in the trace (`gen3_faint_attribution_v1`)

`BattleRecorder` writes one `<side>:<species>:fainted` event per faint. It detected the faint by
COUNT (`*_fainted_count` went up) and then labelled it with **`prev_ctx.*_active`** — the mon that
was active when the DECISION was made. That is the wrong mon whenever a switch resolved on the same
turn, and the trace then contradicted its own battle log two lines above:

```
we switch cloyster → jolteon
opp explosion → jolteon (now 0%)
we cloyster fainted            ← the protocol says JOLTEON fainted
```

**Measured on `ai_v9_17_tdaux_lam3_0818`: 25 of 466 turns named a mon that had not fainted.** Two
shapes produce it — WE switch and the switch-IN eats the hit; or the OPPONENT switches a mon in and
it dies the same turn (Claydol → Dugtrio, our Ice Beam KOs Dugtrio).

**The fix reads the newly-fainted species as a SET DIFFERENCE** over the two snapshots'
`*_fainted_species` — which `BattleContext` already carried, so no new state was needed. A set
difference rather than an HP transition because the second shape has no previous HP to fall from:
Dugtrio was never revealed before the turn it died on.

Two things followed from it, both of which the fuzz found rather than the design:

- **The HP-delta slot was wrong in the same way.** `our_ref` picked `prev_ctx.our_active` on a faint
  turn, so a switch-in that died had its damage read off the row of the mon that left (the recorded
  `hp_delta` read `+0%` while the switch-in went 273 → 0). It now uses the actually-fainted species.
- **ONE SIDE CAN LOSE TWO MONS IN A TURN.** An opponent mon is KO'd, its forced replacement switches
  in and dies to Spikes — both inside turn 34. The old `if delta.*_fainted:` shape could emit at
  most one event per side, so the second faint was silently unreported (1 of 36 faints in a
  4-battle fuzz). `_newly_fainted` returns a LIST and the caller emits one event per species.

**Blast radius: forensic only.** These event strings are read by the prober (the battle-log
timeline, `summary_flags`' `faint` flag) — the reward, the obs and the TurnDelta all compute faints
from their own state, so nothing in training consumed the wrong label. That is also why the fallback
is a slightly-wrong label rather than a raise: a forensic recorder must never take down a run.

**Gate: `poke_env_gaps/faint_attribution_fuzz_test.py`** (bridge, no server) — real battles with a
real `BattleRecorder`, validated against the **protocol log** (`|faint|pNa: Species`), which is the
sim's own statement and not another of our derived structures. It asserts species, side and
completeness per turn, and REPORTS its trigger coverage (`switch-in deaths`) so a clean run that
never exercised the bug says so instead of passing quietly. Measured: **123 faints / 50 switch-in
deaths / 0 mis-attributions**, and **44 mis-attributions when the fix is reverted**.

⚠️ **A protocol identifier carries the NICKNAME, not the species.** The team pool contains teams
whose nicknames are LOCALIZED species names (`Triopikeur` = Dugtrio, `Airmure` = Skarmory), which
reported 10 false failures until the harness resolved identifiers through poke-env's own
`battle.team` map. Any future protocol-vs-our-data comparison needs that map.

