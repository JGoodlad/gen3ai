# Design — Stable (cross-run, possibly-foreign-arch) opponents

**Status:** design (not implemented). Forward-looking; explicit-only doc.
**Goal:** load a frozen model from a *different, already-finished* run (e.g.
`models/ai_v5_5_popart_N_0607`) and use it as a **fixed opponent** in a future training run — on
**both** the eval-yardstick side and the training-opponent mix — without corrupting the live
trainee's strict architecture/resume validation.
**Relates to:** `design_league_tooling.md` (a stable opponent is a `role:"external"` league
member), `design_opponent_distillation.md` / `distill_integration.md` (the
`DistilledOpponentModel` duck-type is the precedent for a non-`MaskablePPO` opponent),
`impl_step1_self_play.md` / `impl_step2_league_play.md`, `src/agents/training/CLAUDE.md`
(self-play pool, eval, ELO), `src/agents/model/CLAUDE.md` (model versioning).
**Decisions baked in** (user, 2026-06-08): scope = **both eval + training**; ELO = **display-only**
(win-rate readout, NOT in the Bradley-Terry fit); anti-exploitation = **no bespoke guard** — a
dominated stable opponent "becomes another bot" by ageing into the coverage floor, which generalizes
the *whole* opponent curriculum into one floor/challenge source model (see §4). Opponents play
**stochastic** (not greedy) while in the challenge bucket.

---

## 1. Problem

Today there is no way to take a model produced by one run and pit a future run's trainee against
it. Two mechanisms block it, and they are confounded in the user's mental model — so separate them
up front:

- **The version gate is a hard wall.** Every opponent load funnels through
  `load_model_snapshot()` (`src/agents/model/snapshot.py:355`), which calls
  `current_version.check_compatible(saved_version)` and **FATALs** on any `arch_signature`,
  any `_WEIGHT_FIELDS` (incl. `total_dim`), or any `use_popart` mismatch
  (`src/agents/model/model_version.py:352`). This gate runs on the self-play pool's sentinels and
  training opponents (`SnapshotPool.load_model` → `load_model_snapshot`,
  `snapshot_pool.py:206`) and on the eval worker's sentinels (`eval_worker.py:46`). A
  cross-run model with even a slightly different config dies here.

- **The opponent is a decoupled decision function** — this is the property that makes the feature
  *possible at all*. `RLPlayer.choose_move` (`src/agents/inference/player.py:294`) →
  `_predict_best_action` (`player.py:228`) builds its **own** observation via its **own**
  `Gen3ObservationEncoder` (`embed_battle`, `player.py:144`), feeds it to
  `model.policy.get_distribution(...)`, and emits only an **action index 0–10** that crosses into
  the shared battle. The trainee never sees the opponent's observation, never shares weights with
  it, and never reads its value head. So an opponent need not match the trainee's architecture —
  it only needs to (a) be loadable standalone and (b) consume an observation the live encoder can
  produce. `DistilledOpponentModel` (`distill/student.py`) already proves a *structurally
  different* network plays fine through this same duck-type (`get_distribution(...).distribution.logits`
  + `.device`), `RLPlayer` unchanged.

The version gate is a **project-imposed** guard, not a torch/SB3 limitation: `MaskablePPO.load(zip,
env=None)` reconstructs the foreign policy + extractor entirely from the zip's own pickled
`policy_kwargs` and runs a forward with no dependency on the live trainee. So the gate comes down
with a loader that does not compare against the live run. The *real* constraint is observation
compatibility (§3).

---

## 2. The key correction — the named example is the EASY case

The exploration initially concluded the example run needs the hardest tier. **That was wrong**, due
to a stale leaf doc. Ground truth (verified against code, not docs):

| Field | HEAD (`model_version.py:238`, `features_extractor.py:34-40`) | `ai_v5_5_popart_N_0607/model_config.json` | Match? |
|---|---|---|---|
| `arch_signature` | `gen3_markovian_progress_v1` | `gen3_markovian_progress_v1` | ✅ |
| `total_dim` | 3391 | 3391 | ✅ |
| `role_token_size` / `projection_dim` | 128 / 512 | 128 / 512 | ✅ |
| `move_net_hidden` / `role_encoder_hidden` / `active_ctx_hidden` | [96,32] / [256,128] / [64,32] | same | ✅ |
| `net_arch` / `n_history_turns` | [512,512] / 10 | same | ✅ |
| `use_popart` | (run-dependent) | `true` | only diff |

> ⚠️ `src/agents/model/CLAUDE.md` says "current `ARCH_SIGNATURE` = `gen3_incoming_damage_v2`" in two
> places. That is **stale** — the live code is `gen3_markovian_progress_v1` (one signature newer; see
> the version-note history in `model_version.py:228-238`). Fix that leaf doc as part of this work.

So `ai_v5_5_popart_N_0607` shares HEAD's `arch_signature` (identical obs layout *and* net sizes).
The *only* thing rejecting it today is `check_compatible` tripping on `use_popart` (and running
against the live trainee at all) — and `use_popart` is **irrelevant to an opponent** (it only
changes the value head, which an opponent never reads). Loading it as an inference-only opponent
that self-checks against its own config and skips `check_compatible`-vs-current **just works**.

---

## 3. Compatibility — two axes, and the gate we actually want

A stable opponent is a pure function `observation → action index`. So compatibility splits cleanly
along **two orthogonal axes**, and only one of them matters:

- **Observation family** — the env↔model *I/O contract*: what the observation vector *means*,
  block-by-block (`total_dim`, the per-block layout, `active_context_dim`). This is what the live
  encoder produces and any opponent must consume. **This is the axis that gates compatibility.**
- **Model family** — the network *internals* (`role_token_size`, `projection_dim`, the hidden-list
  sizes, transformer depth, dual-head structure, PopArt-on-the-value-head). **Irrelevant to whether a
  model can serve as an opponent** — it carries its own weights in the zip and runs its own forward;
  its internals never have to match the trainee's. The only reason a different-model-family opponent
  breaks today is an *incidental* bug (net-size constants are live module globals read in
  `Gen3FeaturesExtractor.__init__`, not carried in `features_extractor_kwargs`), not a real
  constraint.

**`arch_signature` conflates both axes** — it bumps for an obs change (e.g. the `turns_since_progress`
scalar → `gen3_markovian_progress_v1`) **or** a pure model-structure refactor with zero obs change
(the modular phase refactor `gen3_modular_v1`; the dual-head readout `gen3_dual_value_v1`). So it is
really `obs_signature × model_signature` mashed into one string.

### The decision (user, 2026-06-08): allow same architecture family only — gate on `arch_signature` equality

Require a stable opponent to share the live run's **`arch_signature`**. This is the simplest correct
implementation of "same observation family," strict in a *safe* direction:

- Same `arch_signature` ⟹ **same obs family** (guaranteed — any obs change bumps it). ✓ the property
  that actually matters.
- It *also* forces the same model family, which an opponent doesn't strictly need — but that
  over-strictness is a feature here: it makes the rule one equality check **and eliminates the
  net-size-globals problem entirely** (same arch ⟹ same net sizes ⟹ globals already match). So there
  is **no `features_extractor` change, no config-driven net build, no foreign obs encoder, no
  subprocess** — all of that is out of scope.
- It still covers the example (`ai_v5_5_popart_N_0607` shares HEAD's `arch_signature`); the loader
  just skips the `use_popart`/`vf_coef`/reward checks, which an opponent's forward never reads.

**What we give up (acceptably):** a model that is **obs-identical but model-refactored** (a different
`arch_signature` *only* because of something like the modular refactor) is rejected even though it
would actually run. If that ever matters, the clean relaxation is to **split a dedicated
`obs_signature` out of `arch_signature`** (bumped only on obs-layout/meaning changes) and gate the
opponent on *that* instead — letting the model family vary freely while the obs contract holds. That
is a future refinement; **not in scope now.** (The genuinely-different-obs case — a foreign encoder,
pinned-commit move-server, etc. — is also out of scope; the gate refuses it loudly, see §5.2.)

---

## 4. Opponent mix — one unified model (coverage floor vs challenge)

A frozen opponent **never adapts**, so a trainee that trains against it can collapse onto a narrow
counter-strategy that beats *this one policy* without generalizing (and a deterministic/greedy
opponent is *maximally* memorizable). The earlier draft answered this with a bespoke exploit-guard
state machine (streak counter, an eval-only state, a badge, resume re-arm tracking). That was
over-engineered. The decision (user, 2026-06-08) is the simplification: **a dominated stable
opponent just "becomes another bot" — it ages into the coverage floor, exactly like the heuristic
bots that always keep a ~20% share.** No separate state, no new lifecycle.

That instinct generalizes the *whole* opponent curriculum, which is the real win. Today's mix is a
3-phase special-case (`bots only → bots+selfplay → bots+selfplay-at-floor`), and every new opponent
*type* (stable opponents, later league members) threatens to bolt on its own fraction knob and its
own phase. **Collapse it to one model: a weighted set of opponent _sources_, each with a role.**

Two roles:
- **`floor`** — coverage / anti-forgetting. Guaranteed a minimum share even when the model is strong.
  *(Today: the heuristic bots — the "always ≥ floor%" slice.)*
- **`challenge`** — what the model is actively trying to master. Gets the bulk of the mass, but only
  once the model is competent enough to benefit. *(Today: the self-play pool.)*

**One mixing rule for everything** (per training episode, in `_select_episode_opponent`):
- `floor_share` = the configured coverage floor (default 0.20), or **`1.0` when no challenge source
  is active yet** (model too weak / pool not seeded).
- `floor_share` of episodes pick among floor sources by weight (`--bot-weights`); the remaining mass
  picks among **active** challenge sources by weight (the self-play pool recency-weighted; stable
  opponents by their `=weight`).
- A challenge source **activates** once a competence gate is met, and `floor_share` ramps from `1.0`
  down to the configured minimum across that gate band.

This **is** today's `heuristic_fraction(win_rate_vs_bots)` (`snapshot_pool.py:40`), renamed and
generalized: `bots → floor`, `self_play_fraction → challenge_share`, `win_rate_vs_bots →
win_rate_vs_floor`. With floor = bots and challenge = pool it is byte-identical to current behavior,
so the generalization is a refactor, not a behavior change. The flag `--heuristic-floor` keeps
working as an alias for `--coverage-floor`; `SELF_PLAY_START`/`SELF_PLAY_FULL` stay the gate band.

**Every opponent on one lifecycle — and why league play stops adding phases:**

| Source | Role | Lifecycle |
|---|---|---|
| Heuristic bots | `floor` | permanent |
| Self-play pool | `challenge` | activates at the competence gate (`win_rate_vs_floor ≥ SELF_PLAY_START`, 0.55) |
| **Stable opponent** | `challenge` → `floor` | enters as a challenge with a small `=weight`; when the model masters it (`win_rate_vs_ext_<label> ≥ --stable-opponent-mastered-wr`, default `0.80` = `SELF_PLAY_FULL`) its role **flips to `floor`** — kept at the coverage floor like a bot, never dropped |
| Future league member | `challenge` / `floor` | same rule; adds a *source*, not a phase |

So a dominated stable opponent is **not** a special "exploited/eval-only" state — its role flip is
recomputed each eval from the latest `win_rate_vs_ext_<label>` and pushed to envs by the **existing**
per-eval fraction mechanism (`set_self_play_target`-style `env_method`, `selfplay_callback.py:631`).
No streak counter, no badge, no re-arm tracking, no new flags beyond the single
`--stable-opponent-mastered-wr` threshold. (A light 2-cycle confirm on the flip avoids eval noise
flapping the role; the flip is resume-safe for free because it's recomputed from the latest eval,
not stored as state.)

**Two anti-exploitation properties survive the simplification, for free:**
- **Stochastic play, not greedy.** `--stable-opponent-temp` (default `1.0` = the policy's own
  distribution). `_predict_best_action(..., stochastic, temperature)` already supports it
  (`player.py:228`). A stochastic opponent is a moving target within its own support — harder to
  memorize a single exploit line. (Greedy is the wrong default for a *training* opponent.) The same
  temp governs the eval copy, so the reported win-rate matches the regime trained against.
- **A mini-league beats a single fixed opponent.** `--stable-opponents` accepts multiple sources;
  exploiting one frozen policy doesn't transfer. Each masters → floor independently.

---

## 5. Model loading & validation changes

### 5.1 New loader — `load_foreign_opponent()` in `src/agents/model/snapshot.py`
A sibling to `load_model_snapshot()` that loads inference-only against the foreign zip's **own**
config and gates **only** on the architecture family (= same obs family, §3):

```python
def load_foreign_opponent(model_path, current_version, device="cpu") -> tuple[MaskablePPO, ModelVersion]:
    """Load a frozen model from ANOTHER run as an inference-only opponent.
    Gate: the foreign arch_signature must equal the live run's (same obs family — §3). No env, no
    optimizer, no vf_coef/reward/use_popart gate — the forward never reads them.
    """
    zip_path, config_dir = _resolve_paths(model_path)
    config_path = os.path.join(config_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(...)                       # require provenance — never silently load
    foreign = ModelVersion.from_json_file(config_path)
    current_version.check_opponent_compatible(foreign)     # arch_signature equality (§5.2)
    model = InstrumentedMaskablePPO.load(zip_path, env=None, device=device)  # env=None → no space check
    return model, foreign
```

- **`env=None`** → SB3's `check_for_correct_spaces` does not run (it only runs when env is not
  None), so no obs-space reconciliation against the live env.
- **No `check_compatible`-vs-current.** That is the load-bearing bypass — `check_compatible` would
  FATAL on `use_popart` (and demands every `_WEIGHT_FIELD` match), which is irrelevant to an opponent.
- **Same `arch_signature` ⟹ same net sizes**, so SB3 rebuilds the foreign extractor at sizes that
  match its own weights — the net-size-globals trap (§3) cannot bite. No `features_extractor` change.

### 5.2 New predicate — `src/agents/model/model_version.py`
Add one narrow predicate; **keep `check_compatible` exactly as-is** for the trainee resume path.

```python
def check_opponent_compatible(self, foreign: "ModelVersion") -> None:
    """The ONLY check a stable opponent needs vs the CURRENT run: same architecture family.
    arch_signature is the obs-family proxy (any obs change bumps it, so equal arch ⟹ equal obs
    layout). use_popart / vf_coef / reward-config are NOT checked — an opponent's forward never reads
    them. A defensive total_dim + active_context_dim equality assert backs up a hand-edited config."""
    if self.arch_signature != foreign.arch_signature:
        raise ModelVersionError(
            f"Stable opponent arch_signature {foreign.arch_signature!r} != live {self.arch_signature!r}.\n"
            "A stable opponent must share the run's architecture family (= same observation layout).\n"
            "Use an opponent trained at the same arch, or start the new run at the opponent's arch."
        )
```

- `check_compatible` (with `enforce_vf_coef`/`enforce_reward_config`) stays strict — the trainee
  resume gate in `train_rl_agent.py` is **never touched**.
- This is the **loud refusal** for any obs-family mismatch (the old "Tier C"): no silent garbage, no
  attempt to feed a foreign-layout model the live obs.
- *Future relaxation (not now):* to admit an obs-identical-but-model-refactored opponent, replace the
  `arch_signature` equality with an `obs_signature` equality once that field is split out (§3).

### 5.3 `policy.py` / `features_extractor.py` — no change
Same `arch_signature` ⟹ identical extractor shapes, so `Gen3DualHeadMaskablePolicy` +
`Gen3FeaturesExtractor` reconstruct cleanly from the foreign zip with **no code change**. PopArt
buffers ride the `state_dict`; `_denorm` is only used by `predict_values` (never called on an
opponent, `need_aux=False`). The only residual caveat: the foreign zip must reference the *same*
policy + features-extractor classes by import path (true for any same-`arch_signature` run, since the
class identity is part of what a signature bump tracks).

---

## 6. Where it lives in the opponent pipeline

### 6.1 A separate `FixedOpponentPool` — NOT inside `SnapshotPool`
`SnapshotPool` is structurally hostile to foreign opponents: it derives identity from integer step
(`SnapshotEntry.step`, filename `snapshot_<step:012d>.zip`, `_scan` at `snapshot_pool.py:391`), it
overwrites **one shared** `model_config.json` for *all* members (`add_from_path:157`, `_write:412`),
and `sample()` is step-recency (`:170`). All three assume a homogeneous, current-arch, step-tagged
population. (The unused `SnapshotEntry.pinned` flag — already respected by `_evict` at `:421` but
never set — is a tempting host, but it loses on the shared-sidecar problem; keep `pinned` reserved
for a future "pin the seed" use.)

Add **`src/agents/training/fixed_opponent_pool.py`**, modeled on the *read* half of `SnapshotPool`:
- `FixedOpponentEntry { label: str, zip_path: Path, weight: float, temperature: float,
  version: ModelVersion }`.
- Built **once per env worker** from the resolved CLI specs — never scanned, evicted, or promoted,
  so it sidesteps `_evict` / `_write` / `add_from_path` entirely.
- `load(entry)` → `load_foreign_opponent(entry.zip_path, current_version)`; a flat per-worker
  dict-cache (they never rotate, so no LRU churn — loaded once is cheaper than the rotating
  self-snapshots).

### 6.2 The opponent forward (unchanged seam)
Plays through `RLPlayer._predict_best_action`, exactly like `DistilledOpponentModel`. **Because a
stable opponent shares the live `arch_signature` (§3), it consumes the *live* obs encoder unchanged**
— so the existing `RLPlayer` seam works as-is, no encoder injection needed. (If the future
`obs_signature` relaxation ever admits a model-refactored opponent, it's still the same obs layout,
so still the live encoder — encoder injection only ever matters for the out-of-scope different-obs
case.)

### 6.3 `wrappers.py` — `MaskableAgentWrapper`
Extend `_select_episode_opponent` (today a pool-vs-heuristic coin flip on `self_play_fraction`,
`wrappers.py:93`) into the floor/challenge buckets (§4):
- Hold `self._fixed_pool: FixedOpponentPool | None` alongside `_pool` + `_heuristic_opponents`, and
  the per-source roles pushed from the callback.
- A **second reusable `RLPlayer`** (`_fixed_player`), built once; its `.model` is swapped only when a
  different fixed entry is picked (fixed opponents don't change → no `pool_generation` machinery).
  **Build once per worker** — do NOT reload per episode (the documented ~27 MB-per-episode reload
  regression dropped FPS 1400→500).
- Per-episode pick follows the §4 rule: floor bucket {bots + mastered stable opponents} vs challenge
  bucket {pool + un-mastered stable opponents}, weighted; point `self.opponent` at `_fixed_player`
  with the entry's loaded model + temperature on a stable pick.
- The per-eval push already carries the `(challenge_share, generation)` update; extend it with each
  stable opponent's **role** (`challenge` vs `floor`, from the §4 mastery flip) so envs route a
  mastered opponent into the floor bucket. One field on the existing `set_self_play_target` push — no
  new RPC.

### 6.4 `selfplay_callback.py` / `eval_callback.py` / `eval_worker.py`
- **`base_cfg`** gains `fixed_opponents: [{label, path, temperature, weight}]` alongside `sentinels`
  (built in `_launch_eval`).
- **`eval_worker._run`**: add the `ext:` labels to the claim universe
  (`bot_names + sentinel_labels + fixed_labels`) and add an **`_eval_fixed` branch** modeled on
  `_eval_sentinel` but using `load_foreign_opponent` (arch-gate, not `check_compatible`-vs-current)
  instead of `load_model_snapshot(current_version=current_model_version(mappings))`. Results/traces/
  merge need **zero change** — everything is label-keyed (`result__<label>.json`).
- **Stable opponents are EXCLUDED** from `bot_mean()` (so they don't contaminate
  `win_rate_vs_bots`/curriculum), from `sentinel_entries()` / `win_rate_vs_pool` /
  `_monotonicity_score` (promotion gate unaffected), and from `add_from_path` promotion (they're a
  fixed yardstick, never a promotable self).

---

## 7. Naming & identity

**Feature name:** *Stable Opponents*. Reserved label namespace **`ext:`** (external / cross-run) — a
clean third beside `bot:<name>` and `snap:<step>`.

- Default label: `ext:<source_run_basename>` (e.g. `ext:ai_v5_5_popart_N_0607`); `ext:<basename>@<step>`
  if a step is pinned; user-overridable via the `:label` CLI field (prefixed `ext:` if absent).
- This label keys the episode-sampling bucket, `result__<label>.json`, the TB tag
  `eval/win_rate_vs_ext_<label>`, the `metadata.json` opponents block, and the TUI row.
- **Never reuse `snap:<step>`** — the foreign step is its *original* run's step and would silently
  merge with a native snapshot of the same step (even though ELO is display-only here, keep the
  namespace clean for the future carried-anchor option).

**CLI syntax — mirror `--bot-weights`** (parsed in `train_rl_agent.py:663-682`):
```
--stable-opponents "<run_dir_or_zip>[@<step>][=<weight>][:<label>], …"
```
- `<run_dir_or_zip>` resolved via `snapshot._resolve_paths` (a run dir → `best_model.zip`, or a
  direct `.zip`); a sibling `model_config.json` is **required** (fail-fast `sys.exit` at startup,
  like the bot-weights validator). Paths are **absolutized at parse time** so a launcher
  worktree-pinned resume still finds them.
- `@<step>` optional (a specific `checkpoint_<step>.zip`); default `best_model/best_model.zip`.
- `=<weight>` optional sampling weight (default 1.0). `:<label>` optional human label.
- The raw string round-trips into `cli_args` (→ `metadata.json`) for free.

---

## 8. Bot selection / opponent specification (unified surface)

The unified mix (§4) means **no new fraction knob** — a stable opponent is a *source* with a weight,
not a new phase. The whole flag surface for "who does the trainee play":

| Axis | Flag | Side | After this change |
|---|---|---|---|
| Which scripted bots (floor) | `--bot-weights name=w,…` | training | unchanged (weights within the floor bucket) |
| Coverage floor % + gate band | `--coverage-floor` (alias `--heuristic-floor`) / `--self-play-start-wr` / `--self-play-full-wr` | training | **generalized** — "heuristic floor" → "coverage floor" (§4); same numbers |
| Self-play (challenge) | `--self-play` (+ `--snapshot-dir`, `--promote-threshold`, `--self-play-temp`) | both | unchanged |
| **Stable opponents** | **`--stable-opponents path[@step][=w][:label],…`** | **both** | **NEW** — each is a `challenge` source by `=weight` |
| **Stable regime** | **`--stable-opponent-temp <t>`** (default 1.0 = stochastic) | **both** | **NEW** |
| **Mastery → floor** | **`--stable-opponent-mastered-wr <f>`** (default 0.80 = `SELF_PLAY_FULL`) | **training** | **NEW (§4)** — the single role-flip threshold |

That is the entire net-new flag set: a spec string, a temperature, and one mastery threshold. The
dropped `--stable-opponent-fraction` / `--stable-opponent-exploit-guard|cap|window` are subsumed by
the source-weight + role model.

**Training-side composition** (per episode, `_select_episode_opponent`) — one rule, all sources:
1. `floor_share` (= `1 − challenge_share`, the generalized `heuristic_fraction`): pick **floor** vs
   **challenge** bucket. `floor_share = 1.0` until a challenge source is active.
2. **Floor bucket** → weighted pick among {bots (`--bot-weights`), any *mastered* stable opponents}.
3. **Challenge bucket** → weighted pick among {self-play pool (recency), any *un-mastered* stable
   opponents (`=weight`)}.
4. A stable opponent moves bucket 3 → bucket 2 when `win_rate_vs_ext_<label> ≥
   --stable-opponent-mastered-wr` (§4) — recomputed each eval, pushed with the existing fraction
   update.

**Eval-side composition:** the full 9-bot roster **always** plays (unchanged — no roster flag, by
design), plus up to 5 pool sentinels under `--self-play`, **plus** every `--stable-opponents` entry
as an `ext:`-labelled matchup. The scripted-bot roster stays hardcoded
(`_EVAL_OPPONENT_SPECS` / `OPPONENT_CLASSES`); **stable opponents are the extensibility point for
"more opponents"** — they are models, not bots, so there is intentionally no add/remove-bot flag.

---

## 9. ELO / reporting — DISPLAY-ONLY (per decision)

A stable opponent participates **only as a win-rate readout**, explicitly **NOT** as a player in the
Bradley-Terry fit:
- Record `eval/win_rate_vs_ext_<label>` (+ `eval/mean_reward_vs_ext_<label>`) per opponent and a
  `win_rate_vs_external` aggregate → TensorBoard + TUI + `metadata.json:latest_eval` +
  `eval_results.jsonl`. All of this rides the existing label-keyed eval plumbing (`merge_eval_results`,
  `record_eval_results`) with no `elo.py` change.
- **Keep the `ext:` label out of `elo.py`'s player set** — do not add it to `_rows_to_results`, do
  not pin it. The bots remain the sole anchor; the snapshot ladder is unchanged. This avoids the
  cross-run anchor-scale problem entirely (an external run's ELO is only comparable relative to the
  anchor set it was fit against).

**Deferred (out of scope, but the namespace is reserved for it):** carrying a stable opponent's ELO
across runs as a fixed *anchor* (with a provenance guard comparing
`data/gen3_bot_elo_anchors.json` `git_hash` + roster, falling back to free-rating on mismatch), or
playing it into `bot_elo_calibration.py`'s round-robin to anchor it on the bots' exact scale. Not
built; revisit if the display-only win-rate proves insufficient as a progress signal.

---

## 10. `metadata.json` & `model_config.json` changes

### Source side — a `stable_opponent_passport` block (written at final eval, like `latest_eval`)
Provenance + identity so a finished run can later serve as a stable opponent. Every field already
exists in `model_config.json`/`metadata.json` — this is just a convenience block surfacing the
opponent-relevant subset; the gate only reads `arch_signature` (§5.2).
```jsonc
"stable_opponent_passport": {
  "label_hint": "ai_v5_5_popart_N_0607",
  "arch_signature": "gen3_markovian_progress_v1",   // THE compat key (= obs family, §3)
  "total_dim": 3391,                                 // defensive cross-check
  "git_hash": "…"
  // ELO/anchor fields intentionally omitted — ELO is display-only (§9). Reserved for a future carry.
}
```

### Consumer side — a `stable_opponents` block (carried forward by `save_model_snapshot`, like `latest_eval`)
The provenance trail of which foreign weights a run used + each opponent's current role (§4).
```jsonc
"stable_opponents": [
  { "label": "ext:ai_v5_5_popart_N_0607",
    "resolved_path": "/abs/models/ai_v5_5_popart_N_0607/best_model/best_model.zip",
    "git_hash": "…", "arch_signature": "gen3_markovian_progress_v1",
    "weight": 1.0, "temperature": 1.0,
    "role": "challenge|floor", "mastered_at_step": null }   // role flips to "floor" on mastery (§4)
]
```
`cli_args` auto-records the raw `--stable-opponents` string already. **No `model_config.json` schema
change at all** — the same-arch gate needs no new weight-shape fields.

---

## 11. TUI & flags

**Flags:** the net-new set (`--stable-opponents`, `--stable-opponent-temp`,
`--stable-opponent-mastered-wr`, plus the `--coverage-floor` rename/alias) are all **trainer-owned**.
The launcher's arg-stripper forwards every non-launcher flag verbatim → **zero launcher change**.
(Caveat: absolutize the opponent paths at parse time so a worktree-pinned resume resolves them — §7.)

**Resume-immutability:** the opponent mix is the same *kind* of knob as `--bot-weights` /
`--self-play-temp` (NOT resume-locked) — it changes only the rollout distribution, no weight shape,
no objective parameterization. **Recommendation: freely mutable on resume, but recorded** in
`metadata.json:stable_opponents` so any change is auditable. (Contrast `vf_coef`/reward-config, which
*are* locked because they silently rescale the objective.)

**TUI surfacing (mostly free):** the launcher discovers opponents by splitting `eval/win_rate_vs_*`
keys, so an `ext:` row renders automatically once the worker emits `eval/win_rate_vs_ext_<label>`.
Polish: an `ext:` branch in `_row_label`; ordering in `format._METRIC_ORDER` / `_METRIC_LABELS`
(slot stable rows after `win_rate_vs_pool`); and a small role tag on the row (`challenge` vs `floor`,
§4) so a mastered opponent that has aged into the coverage floor reads clearly.

---

## 12. Staged implementation plan

Scope = eval + training, **same `arch_signature` only** (§3). No tiers, no obs/net refactor.

**Stage 0 — loader + arch gate (foundation; no behavior change).**
`load_foreign_opponent()` (`snapshot.py`) + `check_opponent_compatible()` (`model_version.py`, the
arch-equality refusal); `check_compatible` + trainee resume path untouched.
*Verify:* a `*_fuzz_test.py` (mirroring `selfplay_opponent_fuzz_test.py`) loads a same-arch foreign
zip via the bridge (no server), asserts legal moves over a real battle and that `check_compatible`-
vs-current is never called. Unit: `load_foreign_opponent` on a same-arch sidecar loads; an
arch-mismatched sidecar **raises `ModelVersionError`**; `use_popart`-differs sidecar **loads** (not
gated).

**Stage 1 — eval-only (smallest end-to-end win; covers the example).**
`--stable-opponents` parse/validate (require sibling `model_config.json`, **fail fast on arch
mismatch at startup**); `base_cfg.fixed_opponents`; `eval_worker._eval_fixed`; `ext:` labels in the
claim universe; exclude from `bot_mean`/`win_rate_vs_pool`; emit `eval/win_rate_vs_ext_<label>`.
*Verify:* `--debug --self-play` smoke against a **`9XXX`** server (never `:8001`) with a same-arch
stable opponent → the `ext:` row appears in the eval panel + `metadata.json:latest_eval`.

**Stage 2 — training mix + the unified coverage-floor model.**
Generalize `heuristic_fraction` → the floor/challenge source model (§4); `FixedOpponentPool`; the
floor/challenge buckets + second reusable `RLPlayer` in `MaskableAgentWrapper`;
`--stable-opponent-temp` + `--stable-opponent-mastered-wr`; the mastery role-flip recomputed each
eval and carried on the existing `set_self_play_target` push; thread through
`create_training_env_random` closures (parallel to `snapshot_dir`); exempt stable opponents from
distillation (`DistilledOpponentManager` — keep them full-net or sample only when distill is off;
document in `distill_integration.md §8`).
*Verify:* a leak/once-built check (à la `selfplay_opponent_leak_fuzz_test.py`) that `_fixed_player`
is built once per worker (no per-episode reload, no `EpisodeTracker` leak); a fuzz battle confirming
legal play; an FPS sanity (no 1400→500 reload regression); a unit test of the role flip
(`win_rate ≥ mastered-wr` → challenge→floor) and that the generalized fraction reproduces today's
bot-vs-pool curriculum exactly (byte-identical with no stable opponents).

**Stage 3 — passport + provenance recording.**
Source-side `stable_opponent_passport` at final eval; consumer-side `metadata.json:stable_opponents`
block (role + `mastered_at_step`).

**Deferred (explicitly out of scope):** the `obs_signature` split that would admit an
obs-identical-but-model-refactored opponent (§3); any genuinely-different-obs opponent (foreign
encoder / pinned-commit move-server, §3 sketch); carried-anchor ELO (§9); integration with
`design_league_tooling.md`'s payoff-matrix runner (a stable opponent as a `role:"external"` member).

Each stage is independently shippable.

---

## 13. Open questions still pending

*Resolved by the 2026-06-08 decisions:* scope = eval + training; ELO = display-only; the exploit
guard collapses into the unified floor/challenge model (a mastered opponent ages into the coverage
floor like a bot — §4); mini-league supported from day one; Tier C deferred. Remaining:

1. **How far to take the §4 refactor in Stage 2.** Two options: (a) *minimal* — keep
   `heuristic_fraction` as-is and special-case the stable opponent as "a bot once mastered, a pool-
   bucket member while not"; (b) *full* — actually generalize to the floor/challenge source model and
   rename `--heuristic-floor`→`--coverage-floor`. (b) is the simplification you asked for and makes
   league play a no-op later; (a) is a smaller diff now. Recommend (b). Confirm.
2. **Stable-opponent `=weight` default + does it count against the coverage floor or the challenge
   mass while un-mastered?** Recommend: un-mastered = a `challenge` source sharing the challenge mass
   with the pool (so the adaptive pool stays primary; give stable a modest default weight); mastered =
   a `floor` source sharing the floor with the bots. Confirm the default weight (e.g. pool-relative
   0.25).
3. **Eval-only vs training-active when `--stable-opponents` is given but the model is still weak.** A
   stable opponent only enters the *training* mix once the challenge bucket is active (gate at
   `SELF_PLAY_START`); before that it's eval-only automatically. Confirm that's the wanted behavior
   (no separate "eval-only" flag needed).
4. **Different-arch horizon.** Resolved: **same `arch_signature` only** — a different-obs (or even
   obs-identical-but-model-refactored) opponent is refused loudly (§3, §5.2). Confirm you don't
   foresee needing the `obs_signature` split (admit model-refactored opponents) soon — if you do, it's
   a small, additive follow-up, not a rework.

---

## 14. Files this touches (verified paths)

| File | Change |
|---|---|
| `src/agents/model/snapshot.py` | **new** `load_foreign_opponent()` (arch-gate, `env=None`) |
| `src/agents/model/model_version.py` | **new** `check_opponent_compatible()` (arch equality); keep `check_compatible` strict |
| `src/agents/training/fixed_opponent_pool.py` | **new** `FixedOpponentPool` / `FixedOpponentEntry` |
| `src/agents/training/snapshot_pool.py` | generalize `heuristic_fraction` → floor/challenge `coverage` model (§4) |
| `src/agents/training/wrappers.py` | floor/challenge buckets + `_fixed_player` + per-source role on the existing target push |
| `src/agents/inference/player.py` | use the injected encoder for stable opponents (don't lazily overwrite) |
| `src/agents/training/selfplay_callback.py` | `fixed_opponents` in `base_cfg`; per-eval mastery role-flip on the target push; exclude from pool aggregates |
| `src/agents/training/eval_callback.py` | exclude `ext:` from `bot_mean`; `win_rate_vs_external` aggregate |
| `src/main/eval_worker.py` | `_eval_fixed` branch; `ext:` in the claim universe |
| `src/main/train_rl_agent.py` | the new flags; thread into env factory + callbacks |
| `src/main/launcher/format.py` | `ext:` rows + role tag (polish) |
| `src/agents/model/CLAUDE.md` | ✅ stale `ARCH_SIGNATURE` note fixed (done 2026-06-08) |
| `src/agents/training/CLAUDE.md` | document stable opponents + the unified floor/challenge mix (§4) |

*Notably **not** touched: `features_extractor.py`, `state_encoder.py`, `policy.py`, `model_config.json`
schema — the same-arch gate needs no weight-shape or net-build changes.*
