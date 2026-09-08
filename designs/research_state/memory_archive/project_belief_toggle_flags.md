---
name: project_belief_toggle_flags
description: "Step-3 flip will make the belief obs features toggleable via ablation-by-zeroing, controlled by a fresh-only immutable --model-enable-beliefs flag"
metadata: 
  node_type: memory
  type: project
  originSessionId: c1ebed78-c5ec-45c2-b5f1-896b0e5e8bc1
---

> **Archived 2026-09-08** — ai_v5 Step-3 design note; the toggles shipped and live in designs/ARCHITECTURE.md's flag table. Preserved verbatim; nothing below is current.

Decision (2026-06-06): the ai_v5 hidden-attribute-inference **Step-3 flip** must implement the
belief obs features as **toggleable feature flags**, not hardcoded, so the user can A/B which
beliefs actually move the policy/critic.

Mechanism = **ablation-by-zeroing** (chosen over conditional obs dims): the obs dim stays FIXED
(all belief slots always present); a disabled belief is written as **zeros** in the encoder. This
is the cleaner controlled A/B (same architecture + param count, signal present vs absent — a
conditional-dim approach would confound "does it help" with a param/obs-width change) and far less
code (ONE golden fixture, ONE structural `ARCH_SIGNATURE`, no config-parameterized layout).

Flag: `--model-enable-beliefs=speed,choice_band,hidden_power` (comma list). **All THREE** beliefs
toggleable — incl. `hidden_power`, which is the bigger lift (it's a LIVE 16-dim per-mon block + the
`hp_soft_type` embedding blend + ruled-out logic, so making it zeroable is more invasive than the
two new scalars). **Default = all-on** (a no-flag fresh run = the full model; a baseline is an
explicit opt-out — and the default MUST include hidden_power or the current model silently
regresses).

Enforcement (fresh-only + immutable) reuses the existing model-version machinery:
- Add `enabled_beliefs: List[str]` to `ModelVersion` (`model/model_version.py`), written into
  `model_config.json` at model creation from the flag.
- `check_compatible()` must check it EXPLICITLY as a **value-meaning** field (FATAL on mismatch) —
  because under zeroing the weight SHAPES are identical across configs, so the existing shape check
  won't catch a wrong flag (loading a speed-zeroed checkpoint then feeding real speed = OOD garbage).
  This is the same class as the "re-meaning a block is retrain-class" cases.
- Fresh-only: the flag is read only when creating a new model; on `--model <ckpt>` the config is
  read from the checkpoint and a passed flag is validated to match (or ignored).

Trackers ([[project_loss_analysis_run20260601]] critic-coverage work) still run in
`EpisodeTracker.record()` regardless (cheap); the flag only gates whether their output reaches the
obs (skip a disabled tracker for the tiny saving). These are calibrated-belief features —
[[feedback_provide_vs_learn]]. The future outgoing-damage feature ([[project_outgoing_damage_design]])
should join the same toggle set when built.

Status: Step 1 (ChoiceBandTracker) shipped; Step 2 (SpeedBelief + move-priority table) ready-for-ship.
This toggle work is folded into the Step-3 flip (not yet built). The design doc
`designs/ai_v5/design_hidden_attribute_inference.md` §7 still describes Step 3 as a FIXED flip —
update it (explicit-only) when Step 3 lands.
