# Training — reward

Moved out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was split by topic.
The first section was REWRITTEN on **2026-09-26** when the shaped reward path was deleted; the
entropy-boost sections were collapsed to a deletion note in deletion pass L2.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

## The reward — the TERMINAL alone (`reward_manager.py`); the no-progress clock (`progress_clock.py`)

**The shaped reward path is DELETED** (`gen3_shaped_reward_deletion_v1`, config v122, program_rust_core
§4 M3 row, owner-approved 2026-09-26). Gone: the eight PBRS potentials (`reward_potentials.py`), the
~25 BIAS terms (`reward_bias_terms.py`), the bias-additivity accumulate-and-refund, the no-progress
TAX (the clock's `last_penalty`), the suppressed-term fast path and its `GEN3AI_REWARD_VERIFY=1`
shadow twin (`reward_verify.py`), the eval-side reward clock in `RewardTracker`, and the 14 flags that
configured them (listed in `designs/deleted_flags.md`). Production had trained on the terminal alone
since the win-prob era, and the Rust core's reward (slice T) is the win indicator. The whole history —
every term, gate, measurement and hazard — is this file at the last pre-deletion commit:
`git show 029cee83:designs/training/reward.md`.

> **Where the reward lives now.**
>
> | Module | Holds |
> |---|---|
> | `reward_manager.py` | `Gen3RewardManager`: the terminal, the `reward/` export, the episode counters, the `win_margin` by-product; re-exports the three below |
> | `reward_config.py` | `RewardClass` (TERMINAL only), `RewardConfig`, `RewardBreakdown` (one field, `win_loss`) |
> | `reward_composition.py` | the census + its one-line announcer + `inert_reward_flags` + `reward_config_digest` |
> | `reward_weights.py` | `VICTORY_VALUE` (30.0, what an UNRECORDED `victory_value` means; a constant of the namespace, 1.0, since the flags were deleted), `_TIMEOUT_TURN_CAP` (== the env's forfeit turn), `PBRS_GAMMA` (0.9999, the shaped-critic default discount) |
> | `material_margin.py` | the normalised material margin — the `win_margin` training-only OBS key, **not a reward term** |

**The terminal.** `RewardConfig` carries its three knobs (constants of every trainer namespace now — `--terminal-indicator`, `--victory-value`, `--draw-penalty` are DELETED, P11b batch (c); a typed one is refused with its reason), all resume-immutable (recorded in
`model_config.json`, value-checked by `ModelVersion.check_reward_config`):

| outcome | indicator terminal (PRODUCTION; the only value a trainer namespace holds) | signed terminal (the no-longer-trainable shaped critic; what an UNRECORDED field means) |
|---|---|---|
| win | `+victory_value` | `+victory_value` |
| decisive loss | `0.0` | `−victory_value` |
| pre-cap tie | `0.0` | `−victory_value` (shares the loss branch) |
| 250-turn TIMEOUT (a forfeit-loss detected by TURN COUNT, `turn >= _TIMEOUT_TURN_CAP`) | `0.0` | the draw penalty (−35 historically; 0.0 in production) |

Production — and every trainer namespace — is the win-indicator terminal (indicator, victory 1.0, draw 0.0), so the
undiscounted return is exactly `1{win}` and V(s) == P(win|s). The values are constants, not options; under the indicator the draw penalty is INERT (named in
`inert_reward_flags`), and a resume whose recorded reward is not the production one is REFUSED by `check_reward_config` (run it pinned, or start fresh). Every non-terminal turn pays exactly 0.0.

**The parity proof (2026-09-26).** `reward_golden_test.py` (`gen3_reward_golden_v2`, routine tier,
`sim`) hashes every decision's reward, `win_loss` and `win_margin` over 30 bridge battles under six
terminal-only compositions, `production` read from the mirror. It was RECORDED at `029cee83` — the
last commit with the shaped code — and passes unchanged after the deletion: 2,772 decisions,
sha256 `2075c3f7…`. That is the measured statement that production's reward (and the `win_margin`
obs key) did not move.

**A SHAPED checkpoint cannot be resumed or forked on this code** — owner decision, 2026-09-26: it
REFUSES LOUDLY, never continuing silently on the terminal alone (a different objective under the
same run name). `agents.model.model_version.shaped_reward` recognises one from the RAW recorded
config (`hand_shaping` on — absent before v105 means on — or a reachable `--arm-no-progress-tax`),
`main.train.config.resolve_config` exits `FATAL_CONFIG` with the typed
`ShapedRewardCheckpointError`, and `main.checkargs` prints the same verdict ("shaped-reward parent :
YES … ✗ WOULD FAIL"). The fix it names: run it PINNED to a pre-deletion commit, ≤ `029cee83` (the
checkpoint's own recorded `git_hash` is the natural pin; the launcher pins a restart to it by
default, and `checkargs` reports such a pin as ADVISORY). ⚠️ Today this is belt-and-braces:
`MIGRATION_FLOOR` 121 already refuses every pre-v121 checkpoint, and every v121 run on record is a
win-prob arm. A FROZEN load (eval opponent, pool, teacher, prober) of any vintage is unaffected: the
deleted fields POP silently in `_migrate_config`, because a frozen forward never reads the reward.
Pinned by `src/agents/model/model_version/shaped_reward_test.py`, which fails if either refusal is
reverted. 🚨 **Arm S (the shaped comparator) is therefore not re-runnable on HEAD** — its
comparator runs pinned.

**The `reward/` export** (`reward_term_stats.py`, `gen3_reward_term_export_v1`) is unchanged in
shape: `reward/win_loss_*`, `reward/total_*`, and **`reward/untracked_abs_mean`, which must read
exactly 0.0** (the census and the fold agree). The PBRS / BIAS / refund rollups can no longer appear.

**The `win_margin` obs key** (`material_margin.py`) —
`clamp(2·ΔHP + 1.25·Δalive, ±6·3.25) / (6·3.25)` over the DECLARED team (unrevealed opp mons count
full-HP-alive), recomputed every turn by the manager and read by the env's label path (formerly the deleted `gen3_env`) for the win-prob head's
closeness-stratified metrics. It was Φ_mat's by-product; the potential went and the margin stayed,
with the deleted `--mat-alive-weight` frozen at the 1.25 every run trained with.
🚨 **A reward change may never move an OBSERVATION feature** — this one was once pinned at 0.0 for
the win-prob arm's whole life because it hid behind a reward gate.

### The no-progress clock — now an OBS-only counter

`ProgressClock` (`turns_since_progress`, reactive offset 2) is owned by `EpisodeTracker` and
updated at embed time; it classifies each decision window PROGRESS (reset `n`) / DENIED (freeze) /
NO_OP (increment, capped at `PROGRESS_CLOCK_CAP`). **Its reward half is deleted**: there is no
`last_penalty`, no `no_progress_penalty`, and nothing is charged — the classification vocabulary
("a charged no-op") survives in the code comments and test names and means "the clock advances".
The trapped-vs-wall `switch_legal` gate gated only the charge, so it no longer affects anything.

**The clock's two intent-restoring fixes — `--progress-decision-tense` / `--progress-switch-freeze`.**
Both default OFF (a flagless run's clock is byte-identical to every generation's —
`progress_clock_test.py`'s recorded `n` trace). They are resume-immutable `RewardConfig` fields
(recorded, value-checked, no `ARCH_SIGNATURE` bump) because they change the `turns_since_progress`
OBS scalar; they are threaded onto the clock by ONE call, `ProgressClock.apply_reward_config(cfg)`
from the env (formerly the deleted `gen3_env.py`). `--progress-decision-tense` points the forced-switch sit-out gate at the decision
that OPENED the window (`TurnDelta.decision_was_forced_switch`) instead of the one after it;
`--progress-switch-freeze` makes a voluntary switch that fails the (offense-only) progress predicate
FREEZE the clock instead of advancing it. Their motivating measurements (probes M and N,
`designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md` and
`no_progress_tax_review_2026-08-29.md`) were taken on the TAX; what they change today is the obs
scalar alone.

---

## State-conditioned entropy boosts — DELETED (deletion pass L2)

Two training-only boosts multiplied the per-decision entropy bonus on flagged decisions, with no reward change: `--defensive-entropy-boost` (`gen3_defensive_entropy_v1`, decisions where the trainee's active mon had a productive heal / status-cure option, flag `defensive_opportunity`) and `--bait-entropy-boost` (`gen3_bait_entropy_v1`, the bait boards of the E4 verdict — "exploration starvation at a saturated action", ledger *E4 VERDICT*, 2026-08-23 — flag `bait_opportunity`), each with an `--*-anneal-frac` fade. Neither was ever in production or a recipe, and measured exposure was ~1 % of decisions at the default mix (`baitent/flagged_frac` 0.005-0.016, 2026-08-23); both were deleted with their env predicates, obs keys, R1 statics and `defent/*` / `baitent/*` TB tags. **The entropy term is now always `ent_coef * -mean(entropy)`.** `BaitBot` itself (`agents/baitbot.py`, the opponent) is unaffected. Recoverable at pin <= 475bd817.
