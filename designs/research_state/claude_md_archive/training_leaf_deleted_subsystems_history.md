# Two deleted training subsystems — the latent-belief loss (v75) and V_pub (v88)

Lifted verbatim from `src/agents/training/CLAUDE.md` on **2026-09-08**, when that leaf was split
by topic a second time (3,183 lines → ~700). Both subsystems are GONE from the tree; their flags
are recorded in `designs/deleted_flags.md` and their versions in `designs/CHANGELOG.md`.

**This directory is HISTORY — additive only, never updated afterwards.** The reasoning is kept
because it generalises to every aux head on this trunk: a side readout that is never fed forward
buys the policy nothing at inference time, and a lever that measured NULL is deleted rather than
left off. Do **not** re-derive a plan from anything here without checking the current code first.

---

## Latent-belief loss — DELETED (v75)

`--opp-belief-latent-coef`, the `opp_belief_latent` arch toggle, the `BeliefHead` SimSiam predictor,
the `belief_target_slots` training-only obs key and the env work that built it are **gone**. Recorded
here because the reasoning generalises to every aux head on this trunk:

- **It was never fed forward.** The latent was a side readout — stashed for the loss, never
  concatenated into `pi` or `vf`. Contrast `--opp-belief-cls-k`, which appends its pooled belief to
  BOTH projections and therefore buys the policy something at inference time.
- **It cost ~13% of the train step.** Measured per-flag on an idle box with interleaved arms:
  marginal **+341 ms** of train time at the production batch, against a `cls_k=6` costing +349 ms
  that *does* feed forward, and a `spread_belief` costing +72 ms. The train step is ~89% of
  production wall at 10 epochs (an EXTRAPOLATION from a measured 61% at 2 epochs — see the
  `--compile-trainer` section for the provenance and its caveat), so this was real throughput.
- **Its own probe had already concluded decodable ≠ helps** (the belief latent/BYOL role-geometry
  probe: species geometry decodes strongly, and nothing downstream was shown to use it).

**Predicting the opponent's unrevealed mons is untouched.** `BeliefSlots` still fills the hidden opp
slots with learned tokens, the species CE and moves BCE still supervise them, and the T0 species
prior still feeds the physics. What is gone is the *second, graded* way of saying the same thing.

Migration: `MODEL_CONFIG_VERSION` 75 REFUSES a config that recorded `opp_belief_latent=True` (the
predictor carried parameters, so such a state_dict has keys the live extractor cannot accept) and
pops it when false. `sanitize_dead_extractor_kwargs` applies the same rule to a saved zip's
`features_extractor_kwargs`.


## Public-replay value aux — V_pub — DELETED (v88 `gen3_dead_flag_purge_v1`)

The v43 pubval subsystem (`--pubval-mode`/`--pubval-coef`, `agents.training.pubval`,
`pubval_calibration`, `data/gen3_pubval.json`, `PubValHead`, `_pubval_loss`, the parity fuzz) is
**deleted** — it measured NULL as a lever and was never ON in a production generation. A checkpoint
recording `pubval_mode != "none"` is refused by the v88 migration (re-read it from the git_hash in
its metadata.json); `"none"` pops silently. The raw replay corpus (`replays/showdown/gen3ou/`,
local-only) and the design doc (`designs/ai_v8/design_public_info_value.md`) remain for history.

