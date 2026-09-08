# `analyze` — every block on the per-decision JSON

Owned by this tree. The leaf keeps the two rules that ride this method: it LOADS the model (so it
raises `ArchDriftError` on any run off the current architecture) and β's name PROVENANCE. This file
is the block-by-block reference for what `analyze` returns.

- `analyze(battle_id, inv)` — full `InvocationAnalysis` as a dict (**loads the model — so it raises
  `ArchDriftError` on any run not at the current architecture, which today is every archived run;
  see the drift section above**). `model_resolution` carries `dropped_kwargs`: non-empty ⇒ flags the
  current code no longer accepts were dropped to make the load possible, so faithfulness is
  approximate and a surface must say so. The value block gains a γ-discounted `td_residual` and, on a `--use-popart` model,
  the PopArt `popart_mu`/`popart_sigma` + `normalized_recorded`/`normalized_rerun`
  (`(V − μ)/σ`, the critic's normalized learning scale; all `None` without PopArt). Also carries a `win_prob`
  block (`WinProbView`: recorded `P(win|s)` + `delta` ΔP to the next decision) — model-free, read
  from the trace's `win_probs` npz array (NaN/absent → `None` on a non-`--win-prob-mode` run; recorded
  at trace-capture by `RLPlayer._win_prob` → `BattleRecorder.states_arrays`). Also a **`value_dist`**
  block (`ValueDistView`, v29 — `None` unless the run trained `--value-dist-mode`): the distributional
  value head's predicted **return DISTRIBUTION** — `probs`/`support` (the histogram) + `mean` (E[Z]) /
  `std` / `p10`/`p50`/`p90` / `entropy` / `bimodality` (+ `mean_real` = de-normalized E[Z] under PopArt).
  Model-free from the trace's `value_dist` npz array (key absent / NaN → `None`); the atom support comes
  from the loaded model (`ProbeModel.value_dist_support` → `value_dist_vmin`/`vmax`/`bins`). The Summary
  panel renders it as a one-line **eighth-block histogram** + the shape stats (`_append_dist_hist`;
  sharp = confident, wide = uncertain, `⑂ bimodal` = the critic sees a coinflip) below the CRITIC /
  WIN-PROB lines — the interpretability read the scalar V collapses. Engine: `engine.build_value_dist`.
  Also an **`opp_intent`** block (`OppIntentView`, v67 — `None` unless the run trained
  `--opp-intent-coef>0`): what the model expected the OPPONENT to do — `alpha` (ranked NAMED believed
  moves + `SWITCH`, each carrying `is_switch` so no surface compares a magic string itself), `beta`
  (candidate switch-ins — see **β name provenance** below), `top`, `switch_p`. It is
  **model-free and stays that way** — unlike `belief`, which prefers a re-computed read, `α`/`β` are
  supervised against what the opponent then DID, so the honest question is what *this* decision's
  model expected, not what a later checkpoint would. Rendered as the Summary's `EXPECT` line and the
  web replay's per-turn *expect* line. Engine: `engine.build_opp_intent` / `opp_intent_text`.


## The threat decodes — `threats` · `incoming` · `damage_op` · `move_belief`

  Carries two incoming-threat decodes — **distinguish them**:
  - `threats` (model-free, from `their_matchups`): raw type-*effectiveness* —
    `present`, `revealed_frac` (how much opp coverage is revealed), `max_incoming`
    (worst eff ×4 on the board), `per_our_slot_max`. Effectiveness only — no
    power/Atk/Def/HP, so an OHKO still has to be inferred from it.
  - `incoming` (model-free, from the `incoming_damage` block, `incoming_damage_v1`):
    the calibrated P(KO) / expected-chip **belief** that already prices base-power ×
    Atk·Def × HP × roll — `present`, `max_pko` (worst P(KO) across our team),
    `active_pko` / `active_exp` / `active_outspeed` (our on-field mon, slot found via
    its per-mon active flag), `per_slot_pko`, and the opp recovery scalars
    (`recovery_rate` / `cures_status` / `recovery_known`). This is the direct lens on
    "did the critic-tail-blindness obs gap get filled."
  - `damage_op` (model, re-computed via `ProbeModel.damage_op_view` → `decode_damage_block`): the unified
    **DamageOperator**'s LEARNED-belief view (v23), `None` unless the checkpoint trained `--damage-op`. Per
    our mon the incoming threat `[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance` in
    **TEAM-SLOT order** (`incoming[i]` = our team slot i; the op reads `ctx.species_ids[:, :TEAM_SIZE]`, so
    the active is whichever slot holds the active flag — NOT necessarily slot 0 — and the bench slots are
    the safe-switch reads), the `choice_band` tail, and (on `--unified-damage
    both`) our 4 moves' **outgoing** damage `[low,high,crit,pko]` (request-slot/action order — the
    equal-effectiveness move tie-break) + `outgoing.secondary` (per OUR move — "what status can it cause",
    keyed by the 7 live `_OUT_SEC_COLS`; `gen3_op_block_trim_v1` dropped slp/psn/tox, which no gen3 move an
    OUR-side team runs can inflict). **`gen3_op_block_trim_v1` REMOVED three keys** the decode used to
    carry — the opp-active `effect` and `incoming_secondary` collapses (ledger P1: 1.2% / 0.1% of the op's
    measured dependence, no defender axis) and the LEAN `incoming_topk` (0 calls/forward — the matrix
    superseded it). A stale reader now KeyErrors instead of silently mis-reading the Choice-Band bytes. The
    discrete per-move read is `incoming_matrix` (see the Gotcha below).
  - `move_belief` (model, via `ProbeModel.move_belief` → `engine.move_belief_view`, `MoveBeliefView`):
    the model's MOVE belief for each **REVEALED opponent mon** — `None` unless the checkpoint trained
    `--move-belief-mode != off`. Per revealed opp mon (gated on the `species_known` obs bit, so un-revealed
    bench slots are excluded — the run predicts hidden mons' SPECIES not their moves): each `revealed` move
    WITH its belief (pinned ≈100% under `--move-prior-fusion`, so it CONFIRMS the belief tracks the known
    moveset) **plus** the `believed` still-UNSEEN moves `(move, P(in set))` from the multi-label posterior
    (already-revealed filtered, kept if `P ≥ 0.10`; type-collapsed Hidden-Power num → bare `hiddenpower`).
    The unseen list is **CAPPED at the open move slots** `min(top_k, 4 − n_revealed)` — a mon with k known
    moves has ≤`4−k` more, and the multi-label head doesn't enforce that 4-move constraint, so its raw
    top-K over-shows (2 known ⇒ at most 2 unseen, not 4). Also carries `our_labels`
    `(team_slot, species, is_active)` so the op's team-slot incoming rows can be labeled. Pure decode is
    unit-tested (`engine_test::test_move_belief_view_*`). The **move belief (✓ revealed · ≈ unseen)** block —
    `metagross ✓ meteormash 100%  ≈ explosion 32% · …` (green ✓ revealed, magenta ≈ unseen) — now renders in
    the **Beliefs** section (`#beliefs-moves`); the op's outgoing line, the `opp 2ndary:` incoming-status
    line, and the per-OUR-mon op **incoming** damage (worst-channel %HP →KO%, species-labeled, active ▶,
    red-graded by P(KO)) render `🔷`-primary in the **Threats** `#threats-gpu` panel. All fields ride the
    `analyze` CLI JSON.
    **NB:** the op view (`damage_op`) stashes on the **DamageOperator submodule** (`op.last_raw_block`),
    not the extractor — `damage_op_view` reads it there (a prior read of `extractor.last_raw_block`
    silently returned None, hiding the entire incoming/outgoing op view; regression-guarded by
    `model_test.py`).
  Plus `value_saliency` — the **critic** lens: `|d V(s)/d obs|` aggregated into the
  SAME named blocks as the policy `saliency`, so you can see whether the VALUE head
  (where OHKO tail-blindness lives) actually reads `incoming_damage(33)` vs the rest.
  (On run_20260606 the critic's per-dim value-saliency on `incoming_damage` ran ~5×
  the overall mean, vs ~0.3× for the old `their_matchups` — the critic strongly uses
  the new belief.)
