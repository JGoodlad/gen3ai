---
name: project_stable_opponents_design
description: "Stable cross-run fixed-opponent feature — Stage 0+1 (eval-only) BUILT+tested 2026-06-08; training-mix (Stage 2) pending; designs/ai_v5/design_stable_opponents.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: 17c51962-6de7-464d-aed0-5d6367d80c9f
---

**STATUS (2026-06-08):** Stage 0 + Stage 1 (eval-only) BUILT + tested (worktree, NOT shipped — no
/gen3ai-ship yet). Built: `model_version.check_opponent_compatible` (arch-equality gate);
`snapshot.load_foreign_opponent(path, current_version, device, config_path)` (env=None, skips
check_compatible); `agents.training.fixed_opponent_pool` (FixedOpponentEntry + parse/resolve + the
FATAL gate + EXT_PREFIX/is_external); CLI `--stable-opponents`/`--stable-opponent-temp` with startup
FATAL_CONFIG on arch mismatch; `eval_worker._eval_fixed`; both eval callbacks thread
`fixed_opponents`→base_cfg + record `eval/win_rate_vs_ext_<label>` + `win_rate_vs_external` + metadata
`externals`, kept OUT of bot_mean/pool/best-model/ELO. Verified: real `ai_v5_5_popart_N_0607`
resolves+loads+predicts (Tier A, popart-on); 1107 model+training unit tests pass;
`stable_opponent_fuzz_test.py` plays real bridge battles. UX refinements (2026-06-08): label prefix is
`ext_<run>` (UNDERSCORE, not `ext:`, so TB tags are uniform `eval/win_rate_vs_ext_<run>`); default label
= run-dir basename (best_model/snapshots-aware); `=weight` REJECTED (Stage-2 only); `best_model/` now
self-contained — best-save copies model_config.json into it (`copy_run_config_to_best_model`) +
backfilled all existing models/*/best_model/; metric set deliberate+uniform (per-opp win_rate/reward/
ep_len + `win_rate_vs_external` only for 2+ as an `_EVAL_SUMMARY` row; NO td_resid_tail/ELO for ext);
TUI shows run name + `(ext)` tag. **Stage 2 BUILT+tested 2026-06-08** (training-mix): rides the
EXISTING pool-vs-heuristic split in wrappers.py (NO new source-model abstraction, per user "toss it
in like another sentinel, becomes a bot when mastered"). Un-mastered stable opp = a CAPPED MINORITY of the
challenge branch (STABLE_CHALLENGE_SHARE=0.20 in wrappers.py — pool gets the bulk; multiple stable
SHARE the 20% so total stays bounded; user rejected 50%: "can't let 1 training dominate it").
Competence-gated by self_play_fraction + only under --self-play (startup NOTE if not). MASTERED (win_rate_vs_ext ≥ --stable-opponent-mastered-wr,
default 0.80) → moves to FLOOR (heuristic branch, weight 1.0 like a bot). Callback tracks a MONOTONIC
mastered set (one-way, resume-safe) pushed via env_method("set_stable_mastered"). Stable players
built ONCE/worker (load_foreign_opponent in env factory), stochastic@--stable-opponent-temp in
TRAINING but GREEDY (temp 0) in EVAL (user: "in eval make it temp 0"). DRY: eval_worker
_eval_trainee_vs_model shared by _eval_sentinel+_eval_fixed; eval_callback record_per_opponent/
build_externals_block/external_aggregate shared by both callbacks. Verified: 829 unit tests +
--debug --self-play --use-showdown-bridge smoke (loaded stable opp, eval'd greedy, training
complete). Also fixed: os._exit skipped stdout flush → FATAL reason lost when piped (added
sys.stdout.flush before both FATAL_CONFIG exits). The designs/ doc NOT auto-status-updated
(explicit-only rule).

**REVIEWED 2026-06-08** (4-dimension adversarial workflow, 9 confirmed findings all FIXED):
(1) MEDIUM distill interaction — a full foreign stable opp would straggle/gate the all-or-nothing
distill barrier; now EXEMPT from training mix (challenge+floor) while _distill_active (eval-only that
period). (2) mean_reward_mean averaged bot+ext in PerOpponent metadata block → pass bot_rewards.
(3) single noisy eval cycle could trigger the irreversible mastery flip → _MASTERY_CONFIRM_CYCLES=2
consecutive-cycle confirm (streak counter, resets on dip). (4) launcher _FATAL_CONFIG_SIGNATURES
lacked [StableOpponent] FATAL → generic on-screen reason; added it. (5) corrupt foreign zip passed the
config-only startup gate → crash-loop; now load-smoke each zip in main process at startup → clean
FATAL (verified). (6) removed dead FixedOpponentEntry.weight field. (7) eval_worker docstring +
(8) CLAUDE.md distill/re-warm notes. UX flag added: --stable-opponent-selfplay-share (default 0.20 =
STABLE_CHALLENGE_SHARE) caps the stable share of self-play episodes (user rejected 50%). 1229 unit
tests + fuzz + corrupt-zip-FATAL all green. NOT shipped (no /gen3ai-ship).

**Stable Opponents** — load a frozen model from a *different finished run* (e.g.
`models/ai_v5_5_popart_N_0607`) as a **fixed opponent** in a future run, on BOTH eval-yardstick and
training-mix sides. Designed 2026-06-08 (workflow-explored + synthesized), **NOT built**. Full doc:
`designs/ai_v5/design_stable_opponents.md`.

**Key verdict (corrects a stale-doc error):** the named example is the EASY case. HEAD `ARCH_SIGNATURE`
is `gen3_markovian_progress_v1` (`model_version.py:238`), which *matches* the example's
`model_config.json`; only `use_popart` differs, irrelevant to an opponent (value-head only). ⚠️ FIXED
2026-06-08: `src/agents/model/CLAUDE.md` had said current ARCH = `gen3_incoming_damage_v2` (one
behind); now corrected. Always verify ARCH against `model_version.py`, not prose.

**TWO ORTHOGONAL AXES (user's framing, central to the design):** (1) **observation family** = the
env↔model I/O contract (obs layout/meaning, total_dim) — the ONLY axis that gates opponent compat,
since an opponent just consumes obs the live encoder produces; (2) **model family** = net internals
(layer sizes, structure, popart-on-value-head) — IRRELEVANT to opponent compat (the zip self-describes
its weights). `arch_signature` CONFLATES both (it bumps on obs changes AND pure model-structure
refactors like gen3_modular_v1/gen3_dual_value_v1). So arch_signature = obs_signature × model_signature.

**DECISION: same `arch_signature` only.** Gate = `current.check_opponent_compatible(foreign)` asserts
arch_signature equality (the obs-family proxy, safe-strict), skips popart/vf_coef/reward (opponent
never reads them). Because same arch ⟹ same net sizes, the net-size-globals trap (ROLE_TOKEN_SIZE etc.
are live module globals read at `Gen3FeaturesExtractor.__init__`, NOT in features_extractor_kwargs)
can't bite → NO features_extractor/state_encoder/policy/model_config change. Tiers B (same obs diff
net) and C (diff obs — foreign encoder/pinned-commit move-server) are DROPPED/out-of-scope; a mismatch
is refused loudly. Future relaxation if ever needed: split a dedicated `obs_signature` out of
arch_signature and gate on that (admits obs-identical-but-model-refactored opponents). The blocker
remains `load_model_snapshot`→`check_compatible` (FATALs on EVERY load incl pool/eval/distill); the new
`load_foreign_opponent(path, current_version, device)` loads vs the foreign config with `env=None` and
the arch-equality gate. Opponent is a decoupled decision fn (emits an 11-action index);
`DistilledOpponentModel` is the precedent for a non-MaskablePPO opponent via `model.policy.get_distribution`.

**Subprocess/perf note (Tier-C, deferred):** in-loop per-env subprocess move-server = same compute but
adds IPC latency + ~300-500MB RSS/proc (one-per-env = 15-30GB, the killer) + scheduling on an already
CPU-saturated box. Cheaper routes: Tier-C as EVAL-ONLY (reuses the non-blocking eval workers, zero
training-FPS cost) or in-process vendored encoder (zero IPC). Industry: AlphaStar keeps the league
HOMOGENEOUS (same arch, diff weights/rewards); OpenAI Five used "surgery" to migrate weights forward
rather than run two arches; heterogeneous play is an EVAL concern run separately; SEED-RL batches
inference on a server (not per-env); the durable fix is freezing the obs/action contract.

**Decisions baked in (user):** scope = BOTH eval+training; ELO = **display-only**
(`win_rate_vs_ext_<label>`, kept OUT of the BT fit — carried-anchor deferred). Anti-exploitation: the
user REJECTED the complex exploit-guard (streak/eval-only state machine) as over-engineered. Instead
a dominated opponent "becomes another bot" — it ages into the COVERAGE FLOOR. This generalizes the
WHOLE opponent curriculum into ONE model: opponent **sources** each tagged `floor` (coverage, always
≥ floor%; today=bots) or `challenge` (the mastery target; today=self-play pool); one mixing rule =
today's `heuristic_fraction` renamed (bots→floor, self_play_fraction→challenge_share, win_rate_vs_bots
→win_rate_vs_floor; byte-identical with floor=bots). A stable opp enters `challenge` (by `=weight`),
flips role→`floor` when `win_rate_vs_ext ≥ --stable-opponent-mastered-wr` (0.80=SELF_PLAY_FULL),
recomputed each eval + carried on the EXISTING `set_self_play_target` push (no new RPC/state). Net-new
flags = just `--stable-opponents`, `--stable-opponent-temp` (1.0 stochastic, not greedy),
`--stable-opponent-mastered-wr`, + rename `--heuristic-floor`→`--coverage-floor` (alias). User wants
this unification because bots→selfplay→selfplay+floor is already 3-phase and league play would make it
worse; sources-with-roles makes league a no-op (adds sources, not phases).

**Shape:** new `load_foreign_opponent()` + `check_obs_compatible()` (split off check_compatible,
leave trainee resume gate untouched); separate `FixedOpponentPool` (NOT inside SnapshotPool — it's
step-keyed + shares one model_config.json); third bucket in `MaskableAgentWrapper._select_episode_
opponent` + a 2nd reusable RLPlayer (build ONCE/worker — per-episode reload = the 1400→500 FPS trap);
`_eval_fixed` in eval_worker; `ext:` label namespace; CLI `--stable-opponents path[@step][=w][:label]`
mirroring `--bot-weights`; source `stable_opponent_passport` + consumer `stable_opponents` metadata
blocks. Staged: 0 loader → 1 eval-only Tier A → 2 training+guard → 3 Tier B → 4 passport → 5 Tier C
refusal. Relates to [[project_popart]], [[project_opponent_distillation_findings]],
[[project_training_versions]], [[reference_wang2024_thesis]].
