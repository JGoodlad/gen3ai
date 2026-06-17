# Design — Per-move damage MATRICES: symmetric incoming/outgoing, NO collapse (ai_v6)

**Status:** **BOTH matrix slices are BUILT** (gen3_per_move_matrices_v1):
- **OUTGOING** (`damage_matrices_outgoing`, **v34**) — our 4 moves × opp active+revealed bench; 4-lens
  review `correct_as_is` (active-column physics parity with `_outgoing_block` byte-for-byte).
- **INCOMING** (`damage_matrices_incoming`, **v35**) — the enriched top-K: per-move header
  `[latent, belief, acc, is_phys, EXPLICIT effect(6), secondary(10)]` + per-(our-mon, move) cell
  `[low, high, crit, pko, type_mult, status_lands]`; **reuses `--damage-topk K` as its K** (one knob,
  try 4/5/6) and replaces the lean top-K block at that K. 4-lens
  review `correct_as_is` — **no-collapse confirmed** (effect/secondary gathered PER MOVE, not a max).
- CLI: `--damage-matrices {off, incoming, outgoing, both}`. 658 model + 2778 unit tests green; smoke
  roundtrip PASSED for `outgoing`, `incoming`, and `both`.
- **Deferred (not yet built):** physically deleting the now-redundant `p_effect`/`p_sec` opp-active maxes
  (the matrix header supersedes them; kept for OFF-byte-identical, a clean follow-up A/B); per-(move,defender)
  CB tail in the matrix cell; outgoing-vs-UNREVEALED (belief-driven).

Retrain-class, flag-gated, **OFF byte-identical**. Structural (grows both projection heads).
**`MODEL_CONFIG_VERSION` = v34 (outgoing) / v35 (incoming)** — as landed, sequenced after the already-shipped
reattend=v31 + prefuse=v32 and the iterative-refinement refine=v33. `ARCH_SIGNATURE`
unchanged (OFF reproduces baseline byte-for-byte).

> **This SUPERSEDES the v30 top-K block.** The incoming matrix IS the enriched evolution of the
> `damage_topk_k` discrete block (richer cell + explicit per-move effect/secondary). They have **incompatible
> forward layouts and do NOT coexist** — a run uses the per-move matrices **or** `damage_topk_k`, never both
> (enforced: the matrices toggle requires `damage_topk_k = 0`).

## The principle (the owner's directive)

**Stop collapsing.** Mid-ladder play reasons **per move**: "Roar phazes my sweep", "Rock Slide *flinches*",
"this is Wish recovery not Recover", "CB Earthquake OHKOs but the unboosted one doesn't", "Thunder Wave
does nothing to my Ground". A belief-weighted MAX over the opp's moves ("there is *a* phaze threat", "the
worst physical hit is X") **destroys exactly the nuance those decisions hinge on**. v32 removes the
move-axis collapses: every damage + outcome signal stays attached to its **specific (move, defender)**.

The collapses being removed (today's aggregates) and their per-move replacements:

| Collapsed today | What it loses | Per-move replacement |
|---|---|---|
| `_chan_max` per-our-mon worst-phys/worst-spec | WHICH move; every non-max move | the **incoming matrix** cell `[low,high,crit,pko,…]` per (opp-move, our-mon). (A worst-case anchor MAY be kept alongside — see below — but it never *replaces* the per-move cells.) |
| effect axis: belief-weighted max of recovery/phaze/boost/hazard/protect (`p_effect`, opp-active scalar) | which move does it (Roar-phaze vs Dragon-Tail-phaze collapse to one number) | **REMOVED.** Replaced by **explicit per-move effect bits in the row header** (Option B, below): `[is_heal, is_boost, is_protect, is_phaze, is_hazard, inflicts_status]` per row |
| incoming secondary: opp-active-level max of par/brn/frz/… (`p_sec`, opp-active scalar) | which move inflicts it, at what rate | **REMOVED.** Replaced by **explicit per-move secondary chances in the row header**: `[par,brn,frz,slp,psn,tox,flinch]` per row |

Nothing nuanced is *lost* — everything is **un-collapsed** to the move it belongs to, and the two
opp-active MAX blocks (`p_effect`, `p_sec`) are **deleted** (they were the residual collapses the review
flagged). The two genuinely non-collapse scalars stay: `p_outspeed` (per-defender speed; the per-move
PRIORITY nuance is also an explicit header bit so Sucker Punch / Mach Punch are reasoned correctly), and
`p_cb` (the opp's Choice-Band item belief, which *modulates* the per-cell physical CB tail).

**Design decision — explicit effect/secondary bits (Option B), not implicit-in-the-latent (Option A).** The
move latent (32-d) *does* encode the effect/secondary facts (MOVE_ATTR), but implicitly — a role-token blob.
Because these are exactly the signals the owner called "beyond critical" and "how mid-ladder players reason"
(phaze breaks a sweep, flinch, Wish-vs-Recover), we expose them as **explicit, legible per-move scalars in
the header** rather than trusting the policy to decode phaze-ness out of the latent geometry (the role probe
proved *type* similarity in the latent, not effect-semantics). The latent is kept for *identity/similarity*;
the explicit bits are the *legible facts* — provide-the-fact-once applied per move.

**Worst-case anchor (optional, NOT a collapse).** A per-our-mon `[worst_phys, worst_spec]` summary MAY be
kept ALONGSIDE the full matrix as a cheap switch-safety convenience (the matrix loses nothing — the per-move
detail is all there). Default: **off** (the model can max over the matrix via attention); add back only if a
flat-head A/B shows it helps. It is never the *only* representation.

## The two matrices

Symmetric `[moves × defenders]` grids. Each has a **per-move header** (the move's identity/properties) and
a **per-(move, defender) cell** (what the move does to that specific mon).

### INCOMING — opp active's moves → OUR 6 mons (active + bench)
- **Rows:** the opp active's move slots = REVEALED moves (pinned certain) ⊕ the top believed UNREVEALED
  candidates, up to K (the existing top-K selection over the move belief; K default 5 ≈ 4 slots + 1 surprise).
- **Cols:** all 6 of OUR mons (active + 5 bench) — fully known.
- **Per-move header (per row), ≈48 dims:** move LATENT identity (32) · belief weight `w` (differentiable →
  sharpens the belief) · accuracy (1, standalone — chip/status moves that never KO but can miss) · is_phys (1)
  · priority (1) · **explicit effect bits (6): `[is_heal, is_boost, is_protect, is_phaze, is_hazard,
  inflicts_status]`** · **explicit per-status secondary chances (7): `[par,brn,frz,slp,psn,tox,flinch]`**. The
  effect/secondary bits are per-row (un-collapsed) — this is what replaces the deleted `p_effect`/`p_sec`
  opp-active maxes.
- **Per-(move, our-mon) cell:** `low · high · crit · pko · type_mult · status_lands` (+ CB tail
  `high_cb · pko_cb`). `type_mult` = the immunity-folded effectiveness (the clean immune/resist/neutral/SE
  pivot read, decorrelated from magnitude). `status_lands` = immunity-folded at THIS mon (TW→Ground = 0).

### OUTGOING — OUR active's 4 moves → opp mons (active + REVEALED bench)
- **Rows:** our active's 4 move slots, request-slot order (== action logits 6+k), legality-masked. Known
  (no belief). The latent identity is already carried by the PokemonEncoder move-net → **not re-attached**
  here (don't duplicate); the row header is just accuracy · is_phys · priority.
- **Cols:** the opp **active + REVEALED bench mons only**. **Unrevealed opp slots are NOT computed** —
  their columns are gated to 0 (Gen3 has no team preview, so an unrevealed mon's species/types/bulk are
  unknown; guessing damage off the species belief is a **TODO**, below).
- **Per-(our-move, opp-mon) cell:** `low · high · crit · pko · type_mult · status_lands` — same cell shape
  as incoming (symmetric). Our own (known) Choice-Band ×1.5 applied deterministically.

## Per-move nuance preserved (the mid-ladder signals, un-collapsed)

Every one of these is now attached to its specific move (via the latent header), NOT a single opp-active
scalar — so the policy reasons "**this** move phazes" / "**this** move flinches", not "there's a threat
somewhere":
- **phaze** (Roar / Whirlwind / Dragon Tail — "breaks my setup"), **boost** (setup), **heal/recover**
  (Wish vs Recover vs Rest — distinct via the latent), **hazard** (Spikes / Stealth Rock), **protect**.
- **per-status secondary** (Body Slam para, Ice Beam frz, Rock Slide **flinch**) — per move, accuracy-folded.
- **priority** (Sucker Punch / Mach Punch go first regardless of speed) — per move, so `p_outspeed` per
  defender + per-move priority together give the true "who moves first".

## Differentiability / decorrelation (unchanged principles)
- **Incoming:** the belief gradient rides the per-row belief weight `w` (the header), NOT the damage cells
  (w-independent physics) — same decorrelation as the current top-K block. Selection of the K rows is detached.
- **Outgoing:** our moves are certain → no belief gradient (correct: we don't learn our own moves); cells
  are smooth in the stat/damage formula.
- `type_mult` / `status_lands` are immunity-folded per (move, defender) → an immune pivot reads exactly 0
  (the safe-switch read). (Absorb-abilities-as-HEAL is a TODO — currently folds to 0, not negative.)

## Dims / cost (exact)
Let `L = MOVE_LATENT_DIM = 32`, `T = TEAM_SIZE = 6`, header `H_in = L + 4 + 6 + 7 = 49` (latent + {w,acc,
is_phys,prio} + 6 effect bits + 7 secondary), cell `C = 8` (`low,high,crit,pko,type_mult,status_lands` +
CB tail `high_cb,pko_cb`). Outgoing header `H_out = 3` (acc,is_phys,prio — no latent: our moves' latent is
already in the PokemonEncoder move-net, not re-attached) and `M_out = 4` our moves.

- **Incoming** `= K·H_in + K·T·C = K·49 + K·6·8 = K·97`. At K=5 → **485**.
- **Outgoing** `= M_out·H_out + M_out·T·C = 4·3 + 4·6·8 = 12 + 192 = **204**` (unrevealed opp cols zeroed,
  not removed — fixed shape).
- **Total projection growth ≈ 689** at K=5 (vs today's `damage_topk_k`=5 block ≈ 265 + outgoing 57). Larger,
  but the owner has accepted the additional information, and the incoming matrix **replaces** the v30 top-K
  block (they don't coexist) so the net add over a topk-on run is ≈ +485−265 (incoming) + the bench columns.
- Structural (grows both projections, auto-discovered). Physics reuses the validated `_damage_rolls` kernel;
  cost scales with K and the revealed-defender count — the per-cell rolls are the same primitive already in
  the op.

## Versioning / safety (concrete)
- **v32.** Two `ModelVersion` toggles: `damage_matrices_incoming: bool` (replaces+supersedes `damage_topk_k`)
  and `damage_matrices_outgoing: bool`. Both gated in `check_compatible` (bool compare, like `damage_op`);
  OFF byte-for-byte; **`damage_matrices_incoming=True` requires `damage_topk_k==0`** (incompatible layouts —
  enforced at the extractor + CLI). `_migrate_config` `version < 32`: `setdefault(...False)` for both.
  `MODEL_CONFIG_VERSION 31 → 32`. No `ARCH_SIGNATURE` bump (OFF reproduces baseline).
- Threaded through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` (the 4 opp-load
  sites) + both `extractor_kwargs` build sites — the same pattern as `damage_topk_k` (v30) / `damage_op`.
- CLI: `--damage-matrices {off,incoming,both}` desugaring into the two toggles (mirrors `--unified-damage`).

## Revealed-only outgoing gating (leak-safety)
- Outgoing cells are computed ONLY for opp mons with `species_known = 1` (active always; bench iff seen).
  Unrevealed slots → zeroed, never computed. So the block reads **public info only** (revealed species +
  our own known moves/item) — leak-safe by construction. Incoming is already leak-safe (our 6 are ours;
  the opp active is revealed; only its MOVES are the predicted belief).

## Versioning / safety
- A STRUCTURAL toggle (int K for incoming reuse / a bool/int for the outgoing matrix) gated in
  `check_compatible` like `damage_topk_k`; OFF byte-for-byte (no `ARCH_SIGNATURE` bump); threaded through
  `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` (the 4 opp-load sites) + both
  `extractor_kwargs` sites. `_migrate_config` default off.

## Test plan
`damage_op_test`: OFF byte-identical (dims unchanged); incoming matrix shape `[B,K,6,cell]` + matches the
validated `_damage_rolls` physics per cell; outgoing matrix `[B,4,6,cell]` with unrevealed cols == 0
(gating); type_mult cell == the chart eff; status_lands immune-pivot == 0; belief grad rides `w` not the
cells (decorrelation); leak-free (poison the privileged keys → bit-identical). `snapshot_test`: int/bool
gate + migration + arch_toggles pin. Roundtrip smoke + serverless `--debug --use-showdown-bridge`.

## Honesty gate
Wired + differentiable + clean ≠ helps. Fresh-run A/B: the per-move matrices should (a) let
`--unified-obs` mask the CPU incoming/move-effect blocks with NO regression (the deprecation test), and
(b) improve the move-selection mix on equal-effectiveness ties + the surprise-OHKO crater share
(`falsify-scan`), wr/ELO non-regress.

## TODO / backlog
- **Outgoing vs UNREVEALED opp mons** — belief-driven, via the hidden-team species belief head. Deferred
  by owner decision (don't guess damage off an unknown species in v1).
- The op-coverage gaps (`design_unified_damage_system.md`): absorb-abilities = HEAL not 0, Substitute-break,
  multi-hit, OHKO moves, Leech-Seed-vs-Sub, Yawn — fold the context-free ones (`contact`, `multi_hit`,
  `is_ohko`) into MOVE_ATTR so the per-move header carries them.
