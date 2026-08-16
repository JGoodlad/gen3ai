# RUNBOOK — gen-11 (`ai_v9_13_gen11_labelonly_winprob_0815`) end-of-run audit battery

**Pre-registered 2026-08-15, while the run trains** — the decision rules are written before the
numbers exist, per the concat-deletion precedent. Run everything from the RUN'S OWN pinned
worktree (`git worktree add <dir> $(jq -r .git_hash models/<run>/metadata.json)`); copy
`src/agents/model/critic_route_audit.py` in if the pinned tree predates it (self-contained over
APIs that exist since v76). Write every report into `designs/research_state/measurements/` with
provenance (checkpoint, step, state count, date) and add the ledger rows.

## What gen-11 is

Gen-9's stack (intent + dist critic, `value_from_dist`) + `--belief-grad-mode label_only` +
`--win-prob-mode read_only`, on v77-era HEAD (so also the first trained run past ctx-dedup with
`--intent-move-cell` available-but-off). The run answers FOUR pre-registered questions:

## 1. label_only — did cutting PPO→belief cost anything, and did calibration improve?

- **Non-inferiority**: offline anchored ladder (`python -m main.elo <run>`) vs gen-9's 2131 —
  margin per the §5 convention (within −15, CI excluding −40). Use the DENSE ladder fit, not
  the sparse in-run `eval/elo` (the gen-8 resolution lesson).
- **Calibration direction**: TB `belief/*` accs (species/moves/hp-type/spread) vs gen-9's
  curves — label_only predicts equal-or-better CALIBRATION with the PPO-corruption gradient
  gone. Also `opp_intent/alpha_acc` vs its argmax(w) baseline and vs gen-9.
  ⚠️ **Read the `_pool` suffix, not the bare key** (2026-08-15). Every `opp_intent/*` metric is
  now emitted per opponent class, and the bare key is a MIX whose composition MOVES: supervised
  rows were 100% bot at 2M and ~7% from 6M on, and bot rows score differently (gen-11 info gain
  0.124 nats vs pool 0.254). A pooled trend spanning the self-play ramp is uninterpretable, and
  the gen-9 comparison must be `_pool`-to-`_pool` or it compares two different opponent mixes.
  Gen-9 predates the split, so its bare keys are pool-dominated only after ITS ramp — check
  `alpha_n_supervised_bot / alpha_n_supervised` at both endpoints before differencing.
- Decision: non-inferior + calibration flat-or-up ⇒ **label_only becomes the default mode**
  (registry default flip, next config). Strength LOSS beyond margin ⇒ keep `shaping` default,
  and the loss size is the measured price of estimator purity — record it, don't re-litigate.

## 2. The critic-route consolidation (`critic_route_audit.py` — validated on a live-route smoke)

```bash
python -m agents.model.critic_route_audit models/<run>/checkpoints/<final>.zip \
    --states 'models/<run>/eval_traces/**/*_states.npz' --max-states 6000 \
    --out designs/research_state/measurements/gen11_critic_route_audit.json
```

Arms: `seed` / `threat` / `hidden_opp_{both,pi,vf}` / `all_off`. Decision rules, pre-registered:

- A route whose zero-arm reads **< 20% of the `all_off` |dV|** AND **< 2% flips** is a
  DELETION candidate for gen-12 (the seed readout additionally carries its standing
  collapse evidence — rank ~1.1 across three pressures — so for it this bar is confirmatory).
- If `threat` dominates `seed` on |dV|, the token-content thesis is confirmed on a trained
  run and `MultiSeedValueReadout` + its arm die together (Phase-3 item 1).
- `hidden_opp_pi` vs `hidden_opp_vf` decides whether the 768-dim block is a per-head keep,
  a vf-only re-route, or a deletion — measured, not argued.
- **Confound note (pre-registered)**: routes are partial substitutes; a small single-arm
  number under a large `all_off` means SHARED content, not "unused" — read the joint arm
  first, exactly as the edge-vs-concat history taught.

## 3. The dist head + win-prob — instrument verdicts (NOT ablations)

- Distributional **quantile coverage** on eval traces (realized return in predicted bands);
  the `knew_by_turn` / `lead_time` / `blind_loss` verdicts on every cap/stall loss — the
  deadline-clock regression test that never existed: *fraction of cap losses tail-aware ≥5
  turns early*. **BOTH LANDED** (`main/prober/awareness.py`, model-free):
  `python -m main.prober.query awareness models/<run>` → `aggregate.cap_aware_ge_bar_fraction`
  is the awareness number and `aggregate.quantile_coverage` the calibration one (judge
  calibration with `--outcome` unset — the loss filter biases PIT low by construction).
  Gen-10 baselines measured 2026-08-15: awareness (1396 losses) 7.2% blind, median lead 7,
  cap losses 12, cap-aware@5 **0.50**; coverage (ALL outcomes, 109k decisions) pit_mean
  **0.396**, coverage80 **0.44** vs nominal 0.80 — the gen-10 head is optimistic AND
  over-confident, so gen-11's label_only arm has a concrete calibration bar to beat.
- WinProb reliability curve vs the calibration command's V-based one.

## 4. Standing re-measures riding the same states

- **d3/s3 on trained-B-spread physics** (edge_ablation_audit) — the "channel vs content"
  disambiguation for the incoming families.
- The per-family audit including the c-family under label_only α.
- α mask rates (`opp_intent/alpha_mask_rate`) — the belief-coverage ceiling on the intent line.

## Sequencing note

Gen-12's config is decided by §§1–2 outputs + the G3 arm — **which gen-11 itself trains**
(`intent_move_cell: true` in its model_config.json; the earlier "still unrun" note was stale,
so §4's per-family audit should read the c2/move-cell routes as LIVE arms, not available-only)
+ the H-A `h` family (v79, built, opt-in). Per the attribution discipline, prefer: gen-12 =
G3 + `h` (both zero-init, jointly ablatable) on whichever belief-grad default §1 selects,
with Phase-3 deletions landing as pure code removals only for routes §2 condemned. **The §2
condemnations now have a built successor**: `--value-entity-pool` (v80,
`gen3_unified_value_readout_v1` — the Stage-3 unified critic entity pool, zero-init vf-only,
with its own `entity_pool` audit arm), so a route deletion and its replacement can land in ONE
config change rather than a deletion generation followed by a rebuild generation.
