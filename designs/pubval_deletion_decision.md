# V_pub (`--pubval-mode` / `--pubval-coef` + `PubValHead`) — deletion decision package

**Status: NOT DELETED. Prepared for owner decision, deliberately kept out of the v78 batch.**

The v78 flag-surface cleanup (`gen3_flag_surface_p1_v1`) deleted two closed research lines. V_pub was
listed as a Tier-1 candidate, but the owner has **exempted diagnostic heads** (`win_prob_*`,
`value_dist_*`, `value_from_dist`) from deletion, and V_pub is the same *kind* of object — a
supervised side readout off `value_pooled` that never enters `pi`/`vf`. Whether the exemption
covers it is a judgement call about **what the head is for**, not about the code. This file lays
out the evidence and asks the one question that decides it.

---

## 1. What it is

`--pubval-mode {none,read_only,shaping}` builds a `PubValHead` off `value_pooled` and regresses it
toward **V_pub** — a FROZEN logistic over 17 public-board features, fit offline on the rated gen3ou
human-replay corpus (`data/gen3_pubval.json`, provenance in its `meta`). `--pubval-coef` weights the
soft-target BCE. Structural + resume-immutable STRING gate (the `win_prob_mode` pattern); OFF is
byte-identical.

The motivating argument, which is a good one: our critic is trained **only** by its own bootstrap,
so a value error is self-confirming. V_pub is the one **exogenous, value-INDEPENDENT** signal
available — it is dense (per step, so the trunk sees *when* the game swung) and it comes from
outside the agent's own loop. It is deliberately never wired into GAE: it is `V^human`, not `V^π`.

## 2. What it costs

| | |
|---|---|
| Code | `pubval.py` (317) + `pubval_calibration.py` (166) + `PubValHead` + the loss + `pubval_test.py` + `pubval_head_test.py` + `pubval_parity_fuzz_test.py` |
| Data | `data/gen3_pubval.json` (99 lines, committed calibration artifact) |
| Flags | 2 of the 167 remaining (`--pubval-mode`, `--pubval-coef`) |
| Fields | 2 `ModelVersion` fields (`pubval_mode` structural-gated, `pubval_coef` training-only) |
| **Per-decision env work** | **the real cost** — `pubval_mode != none` sets `emit_pubval_target`, and `Gen3Env` then folds a `PubSide` and evaluates the logistic **every decision**, on the CPU rollout critical path |
| Corpus | `replays/showdown/gen3ou/` is **local-only, not in the repo** — the artifact is reproducible only on this box |

## 3. What is known

- **The offline POC was honest and partly positive**: crude 17-feature aggregates reach **test AUC
  0.734** (split by game, 164,230 rated logs). Richer identity features did **not** beat it
  (logistic 0.733, MLP 0.689 test / 0.914 train — overfit). So the shipped artifact is at the
  measured ceiling of that corpus, not a first cut with headroom.
- **It has never run in the current generation.** `pubval_mode: shaping` appears in exactly **three
  archived runs — all ai_v7** (`ai_v7_09_tss_bots_pubval_0708`, `ai_v7_10_tss_exploiter_fixed_0709`,
  `ai_v7_15_tss_exploiter_vs14_0713`). Every gen-1…gen-10 run under the `ai_v9` signature recorded
  `pubval_mode: none`, and `designs/production_config.json` carries `none`.
- **The recorded verdict is "pubval later NULL"** (memory `project_public_value_poc`). That is a
  one-line summary of an ai_v7 result, on a different architecture, **before** the critic-side work
  that has landed since (the distributional critic, `--value-from-dist`, the multi-seed readout,
  `--value-threat-inject`).
- **The exemption's own logic points at keeping it.** `win_prob_*` and `value_dist_*` are exempt
  because a diagnostic's value is in being *readable*, not in being *load-bearing* — and V_pub is
  the only one of the three whose target comes from **outside the agent**. As a diagnostic it
  answers a question neither of the others can: *is our critic wrong, or is the position actually
  lost?* — which is exactly the split `prober calibration` currently estimates indirectly.

## 4. The argument for deleting it anyway

1. **Two years of runs have not used it**, and the one line of evidence says NULL.
2. It is the **only** exempt head with a **per-decision env cost**; `win_prob` and `value_dist` are
   pure GPU-side readouts off an existing pooled vector. If the head is off, that cost is zero — but
   so is the head's value, which makes "keep it available" a weaker argument than it looks.
3. Its artifact depends on a corpus that is **not in the repo**, so the head is not reproducible by
   anyone who does not have this box's `replays/` tree. A committed artifact whose regeneration path
   is unavailable is a maintenance liability, not a capability.

## 5. The argument for keeping it

1. **The NULL is stale and confounded.** It was measured on ai_v7, pre-distributional-critic,
   pre-multi-seed-readout — an architecture that no longer exists behind a signature wall. Re-testing
   costs one flag on a fresh run; deleting costs a rebuild if the answer changes.
2. **It is the only exogenous critic signal we have**, and the critic is the standing frontier
   (`floor leak = critic over-values self-KO`; the calibration bucket split).
3. Deleting it is **cheap to reverse in code and expensive to reverse in evidence** — the corpus fit
   is a one-time artifact that already exists.

## 6. The question that decides it

> **Is V_pub a DIAGNOSTIC (exempt, keep) or a shaping LEVER that was tried and failed (delete)?**

The flag itself is tri-state and answers this ambiguously on purpose: `read_only` is a diagnostic
(stop-grad input, trains its own params only) and `shaping` is a lever (it shapes the trunk). **All
three archived runs used `shaping`** — so the thing that was measured NULL was the *lever*, and the
*diagnostic* has never been run at all.

**Recommended:** keep, and **narrow the flag to `{none, read_only}`**. That deletes the half with a
measured null, keeps the half that was never tested, and removes the ambiguity from the exemption.
It is a smaller change than deletion and it makes the next measurement well-posed.

## 7. If you say delete — the prepared plan

Mechanically identical to the v78 zarch/seed deletions, one config version (v79), no
`ARCH_SIGNATURE` bump (OFF is byte-identical; the byte-identity probe re-runs as the gate):

1. `flag_registry.py`: drop the `pubval_mode` row.
2. `model_version.py`: drop `pubval_mode` + `pubval_coef`; drop the `check_compatible` compare;
   bump `MODEL_CONFIG_VERSION` to 79; migration **POPs** `pubval_coef` (training-only) and **JUDGES**
   `pubval_mode` — `'none'` pops, anything else is REFUSED (it built a head, so it named parameters;
   the v75/v78 precedent). Add `("pubval_mode", "none")` to `_DEAD_FEK_JUDGED`.
3. `snapshot.py`: drop the `current_model_version` keyword + the `ext_kwargs` write +
   `arch_toggles_from_model`'s entry.
4. `features_extractor.py`: delete `PubValHead`, its construction and its stash;
   `tier_contract.py` T3 entry; `arch_tables.py` `_ABSENT_CANDIDATES` + toggle map.
5. `instrumented_ppo.py`: delete `_pubval_loss`, `pubval_on`, `pubval_coef`, the `pubval/*` metrics
   and the `grad/pubval_share` entry; `launcher/format.py`'s label.
6. `gen3_env.py`: delete `emit_pubval_target`, the `pubval_target` obs key and `_pubval_target()`
   — **this is the only part that changes a training-only obs key**, so it is also the only part
   with a real per-decision saving.
7. Delete `pubval.py`, `pubval_calibration.py`, `pubval_test.py`, `pubval_head_test.py`,
   `pubval_parity_fuzz_test.py`, and `data/gen3_pubval.json`. Keep
   `designs/ai_v8/design_public_info_value.md` (history) and add the NULL + the corpus-availability
   note to it.
8. Docs: root `CLAUDE.md` (Data Dependencies), `src/agents/model/CLAUDE.md`,
   `src/agents/training/CLAUDE.md`, `designs/ARCHITECTURE.md` (regenerate), CHANGELOG v79.
9. Gates: full suite · byte-identity vs the pre-change extractor under `production_config.json` ·
   the four `--check` artifact gates · a `--debug` bridge smoke.

**Do NOT bundle it with a `--sync-to-main` resume of `ai_v9_11` / `ai_v9_12`** — both pass
`--pubval-coef 0.1 --pubval-mode none` explicitly, so deleting the flags adds two more entries to
the collision list in §"original_command collisions" of the v78 report.
