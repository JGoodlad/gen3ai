# Implementation: Step 8 — Stable (cross-run) opponents

Use a frozen model from **another, already-finished run** as a **fixed opponent** — measured against
in eval (a cross-run-comparable yardstick) and, under `--self-play`, played against in training. The
motivating use: pit a fresh run against a strong prior run (e.g. `models/ai_v5_5_popart_50m_0607`)
without it being a self-play pool member.

> **Status: BUILT & SHIPPED** (commit `6969631`). Opt-in **`--stable-opponents`** (default OFF → the
> default run is byte-unchanged). **No obs/arch/weight change** → `ARCH_SIGNATURE` unchanged at
> `gen3_markovian_progress_v1`, obs **3391**; **no `MODEL_CONFIG_VERSION` bump** (the feature adds no
> `ModelVersion` fields — opponents are gated, not weight-loaded into the trainee). As-built record.
> Forward design: `design_stable_opponents.md`. Verified by 2015 unit tests + two bridge fuzz tests +
> a `--debug --self-play` smoke (exit 0); a 4-dimension adversarial review (§7) cleared it.

---

## What shipped (one paragraph)

A foreign model is loaded **inference-only** by **`snapshot.load_foreign_opponent`**, which validates
it against **its own** `model_config.json` and gates on **one axis only — the OBSERVATION FAMILY**
(`ModelVersion.check_opponent_compatible` = same `arch_signature`), deliberately **skipping** the
trainee's `check_compatible` (so `use_popart` / `vf_coef` / reward-config differences — all irrelevant
to an opponent's *forward*, which never reads the value head — don't block it). A mismatch is a
startup **FATAL_CONFIG** (no restart loop). Resolution + validation happen once at startup in the new
**`agents.training.fixed_opponent_pool`** module. In **eval** the opponent plays **greedy (temp 0)** —
a clean, deterministic yardstick — recorded under the reserved **`ext_<run>`** label namespace
(`eval/win_rate_vs_ext_<run>` + a `win_rate_vs_external` aggregate for a mini-league), kept **out of**
`win_rate_vs_bots` / `win_rate_vs_pool` / the best-model aggregate / **the ELO fit** (no ladder
distortion). The eval table's elo column for an `ext_` row IS filled (`record_external_elos`),
preferring the opponent's **own recorded ELO** (read at startup from its `best_model.json` sidecar /
run `metadata.json` `latest_eval.elo` → `FixedOpponentEntry.source_elo` — a well-fit, bot-anchored,
cross-run-comparable rating, e.g. 1902 for ai_v5_5), and only falling back to a **trainee-derived
ballpark** (`external_elo`: `R_opp = R_trainee − (400/ln10)·logit(win_rate)`, clamped) when no recorded
ELO exists. Display-only either way. Under **`--self-play`**
the opponent also enters the *training* mix by riding the **existing pool-vs-heuristic split** in
`MaskableAgentWrapper` (no new source-model abstraction): un-mastered → a **capped minority**
(`STABLE_CHALLENGE_SHARE`=0.20) of the self-play/challenge bucket (the pool keeps the bulk; multiple
stable opponents *share* the slice); **mastered** (win_rate ≥ `--stable-opponent-mastered-wr` for
`_MASTERY_CONFIRM_CYCLES`=2 consecutive cycles) → moves to the **coverage floor** alongside the bots
("becomes another bot", one-way). Stable opponents are **exempt from the training mix while
distillation is active** (a full foreign model would straggle and gate the all-or-nothing barrier).

## The two axes (the load-bearing design decision)

Compatibility splits along two orthogonal axes, and only one matters for an opponent:
- **Observation family** — the env↔model I/O contract (`total_dim`, the per-block obs layout). An
  opponent consumes the obs the **live** encoder produces, so this **must** match. `arch_signature`
  is the proxy (any obs change bumps it).
- **Model family** — the network internals (layer sizes, dual-head, PopArt-on-value-head). **Irrelevant
  to an opponent** — it carries its own weights in the zip and runs its own forward.

`arch_signature` *conflates* both (it bumps for an obs change OR a pure structural refactor), so
requiring **equal `arch_signature`** is a conservative — but safe — proxy for "same observation
family". This is why **no `features_extractor`/`policy`/`model_config` change** was needed: same
signature ⟹ same net sizes ⟹ the foreign zip rebuilds at shapes matching its own weights. (A future
relaxation to admit an obs-identical-but-model-refactored opponent would split a dedicated
`obs_signature` out of `arch_signature` — not built; out of scope.)

## CLI surface

| Flag | Default | Meaning |
|---|---|---|
| `--stable-opponents "path[@step][:label],…"` | None | foreign run dir(s) (→ `best_model`), `.zip`, or `@step`; labelled by the run-dir name (`ext_<run>`). `=<weight>` is **rejected** (not supported). |
| `--stable-opponent-selfplay-share` | `0.20` | cap on stable opponents' share of self-play (challenge) episodes (= `STABLE_CHALLENGE_SHARE`); validated to `[0,1]`. |
| `--stable-opponent-mastered-wr` | `0.80` | win-rate (for 2 consecutive cycles) at which a stable opponent "becomes a bot" (challenge→floor). |
| `--stable-opponent-temp` | `1.0` | TRAINING play temperature (stochastic, harder to over-exploit). Eval is always greedy. |

## Constants

| Constant | Where | Value | Meaning |
|---|---|---|---|
| `STABLE_CHALLENGE_SHARE` | `wrappers.py` | `0.20` | cap on the stable share of the self-play bucket (flag override above) |
| `_MASTERY_CONFIRM_CYCLES` | `selfplay_callback.py` | `2` | consecutive ≥-threshold eval cycles before the **one-way** mastery flip (eval-noise guard) |
| `EXT_PREFIX` | `fixed_opponent_pool.py` | `"ext_"` | label namespace — underscore so TB tags are uniform (`eval/win_rate_vs_ext_<run>`, like `sentinel_0`) |

## Behaviour map

- **Eval** (any run): greedy matchup → `eval/win_rate_vs_ext_<run>` (+ reward/ep_len; `win_rate_vs_external`
  only for ≥2). `metadata.json:latest_eval` gains an `externals` block. **Not** in `win_rate_vs_bots`,
  `win_rate_vs_pool`, best-model, the **ELO fit**, or the `td_resid_tail` diagnostic — but the elo
  column IS shown (display-only): the opponent's **own recorded ELO** (`source_elo`) when available,
  else a trainee-derived ballpark (`external_elo`). Worker: `eval_worker._eval_fixed`.
- **Training** (only `--self-play`; a startup NOTE says so otherwise): competence-gated by
  `self_play_fraction` (a weak model trains on bots first); un-mastered = ≤20% of self-play; mastered →
  floor. Stable players are **built once per worker** (`load_foreign_opponent` in the env factory) — no
  per-episode reload — and are leak-safe via the same `reset_battles`-on-active-opponent path as the pool
  player. Mastery is pushed via `env_method("set_stable_mastered", …)` (drain-safe under `--async-rollout`).
- **`best_model/` is self-contained**: each best-save copies the run's `model_config.json` AND writes a
  `best_model.json` sidecar (`copy_run_config_to_best_model` + `write_best_model_sidecar`, the latter
  reusing `snapshot.write_checkpoint_metadata` so it carries `latest_eval` incl. the run's ELO).
  `best_model/{best_model.zip,model_config.json,best_model.json}` co-located — the unified place a
  consumer reads the arch gate + the carried ELO. Backfilled for existing runs.
- **Resume**: the mastered set lives in callback memory (not persisted), so after a launcher restart a
  previously-mastered opponent reverts to the challenge bucket until the first post-restart eval
  re-confirms it (self-healing; bounded by the eval cadence).
- **Launcher Events panel**: a `🎯 [STABLE] …` startup line (via `emit`, mirroring the `[SELFPLAY]`
  startup lines — `emit` print()s standalone), plus a `stable <pct>%` field on each eval-summary event.

## Review (§7) — 4-dimension adversarial pass, 9 confirmed findings all fixed

A correctness/simplicity/parity/robustness review (each finding adversarially re-verified) found and
all were fixed in the shipped commit: (1) **distillation interaction** — full stable opponents now exempt
from the training mix while distilling; (2) `mean_reward_mean` averaged bot+ext in the eval-only metadata
block → bot-only; (3) single noisy eval cycle could trigger the irreversible mastery flip → **2-cycle
confirm**; (4) launcher `_FATAL_CONFIG_SIGNATURES` lacked `[StableOpponent] FATAL` → generic on-screen
reason; (5) corrupt foreign zip passed the config-only startup gate → crash-loop → now a **load-smoke
at startup** turns it into a clean FATAL; (6) removed a dead `FixedOpponentEntry.weight` field; (7-8)
doc/CLAUDE.md fixes. Also fixed during the build: `os._exit` skipped the stdout flush, losing the FATAL
reason when piped (added `sys.stdout.flush()` before both `FATAL_CONFIG` exits).

DRY helpers extracted along the way: `eval_worker._eval_trainee_vs_model` (shared by `_eval_sentinel`
+ `_eval_fixed`); `eval_callback.record_per_opponent` / `build_externals_block` / `external_aggregate` /
`copy_run_config_to_best_model` (shared by both eval callbacks).

## Files

| File | Change |
|---|---|
| `agents/training/fixed_opponent_pool.py` | **new** — `FixedOpponentEntry`, `parse_/resolve_stable_opponents`, `EXT_PREFIX`/`is_external` |
| `agents/model/snapshot.py` | **new** `load_foreign_opponent` (arch-gate, `env=None`, config-dir fallback) |
| `agents/model/model_version.py` | **new** `check_opponent_compatible` (arch-equality; keeps `check_compatible` strict) |
| `agents/training/wrappers.py` | `STABLE_CHALLENGE_SHARE` + challenge/floor buckets + `set_stable_mastered` + distill exemption |
| `agents/training/selfplay_callback.py` | `_push_stable_mastered` (+ 2-cycle confirm) on the existing env push; ext exclusions |
| `agents/training/eval_callback.py` | shared per-opponent/externals helpers; `ext_` recorded, excluded from bot/pool/ELO/best |
| `main/eval_worker.py` | `_eval_trainee_vs_model` + `_eval_fixed` (greedy) + `fixed_opponents` claim universe |
| `main/train_rl_agent.py` | the 4 flags + startup resolve/load-smoke FATAL + env-factory stable-player build |
| `main/launcher/run.py` | `[StableOpponent] FATAL` surfaced in the Events panel |
| `main/launcher/app.py` | TUI renders each by run name + `(ext)` tag; `win_rate_vs_external` summary row |
| `agents/training/{stable_opponent,fixed_opponent_pool}_*test.py`, `wrappers_test.py`, … | unit + bridge fuzz coverage |

## Not built (deferred)

The `obs_signature` split (admit an obs-identical-but-model-refactored opponent); any
genuinely-different-obs opponent (a foreign encoder / pinned-commit move-server — see
`design_stable_opponents.md §3`); carrying a stable opponent's ELO across runs as a real fit **anchor**
(the shown ELO is instead a display-only ballpark inverted from the trainee's rating — `external_elo`);
per-opponent training weights.
