# R1 — ADVERSARIAL REVIEW of the ai_v12 build wave (2026-08-30)

**Scope.** The five landings on `main`, red-teamed before the era launches on them:

| wave | commit | subject |
|---|---|---|
| A | `d80db46` | clean-world config v105 (`hand_shaping` master, `victory_value`, PBRS gates, frozen-φ source) |
| D | `4437c85` | v106 opt-in fixes (`progress_decision_tense`, `progress_switch_freeze`, staller RNG) |
| C | `248a22d` | v107 Q-win-prob head (pointer-token readout, cf `q_labels` contract) |
| E | `2b86f03` + `2c4b746` | scaffolding gauge + exploitability CLI + TB scalar |
| B | `cd092c4` | harvest/factory pipeline + `winprob_finetune` + `harvest_meter` |

Ledger entries reviewed against the diffs: `132d198`, `ade5a11`, `195ce9f`, `4867537`.

**Method.** The priority was the INTERSECTIONS — the M2 (`value_from_dist`) genre, where every flag
in a tail has its own test and the composition a launch actually types has none. Findings are
litigated in place: a failing test first, the minimal fix second. Every fix below was confirmed to
FAIL before it and PASS after; the five claimed-test spot-checks were additionally **revert-verified**
(the guarded behaviour was reverted locally and the named test confirmed to go red).

---

## Findings

| # | sev | genre | finding | status |
|---|---|---|---|---|
| F1 | **HIGH** | intersection (M2) | clean-world × `value_from_dist`: with PopArt OFF the HL-Gauss target is the RAW return, so the atom support and `--victory-value` are in the SAME units — and nothing compared them | **LITIGATED** |
| F2 | MEDIUM | composition drift | `--victory-value 1.0` alone INHERITS `--draw-penalty -35.0`; the wave-A guard checks the ordering and passes a 35× scale inversion | **LITIGATED** |
| F3 | MEDIUM | silent-inert | `train/pbrs_reward_share` published `0.0` when the unshaped stream is empty — a flattering zero in exactly the clean arm it exists to watch | **LITIGATED** |
| F4 | MEDIUM | schema semantics | wave B's `label_one` filed a **TIE** into `n_timeout`; every dropped rollout is a non-win, so `k/n` overstated P(win) | **LITIGATED** |
| F5 | LOW | intersection | wave C put `q_winprob_*` into `cf_any_on`, so a Q-head-only run now runs the cf forward — which clobbers the FitNets hint stash. Correct today by statement ORDER alone | **LITIGATED** (order pinned) |
| F6 | LOW | untested path | frozen-φ source × a real CUDA `--compile-trainer` remains **UNEXERCISED**, as wave A itself recorded | **ESCALATED** (below) |
| F7 | INFO | schema | `q_labels` entries with an absent/zero `n_rollouts` are folded at an effective weight of 1 rollout (`clamp(min=1.0)`) — a label with no evidence trains as if it had one | **ESCALATED** (below) |

### F1 — clean world × the distributional critic (HIGH)

`--victory-value` (v105) is the first flag that can change the RETURN SCALE. `--value-dist-vmin/vmax/bins`
is a *separate*, resume-immutable flag fixed at launch. Under PopArt the CE target is
`popart.normalize(returns)`, so the support is in units of standard deviations and the two never
interact — which is why no historical run could have surfaced this. But:

* production carries `value_from_dist=True`, support `[−12, +12]`, 51 atoms (verified against six
  archived `model_config.json`, `ai_v9_67..72_0828`);
* the registered clean/sparse arms run **without PopArt** (ledger `2d38a4a`: *"the sparse and clean
  arms run WITHOUT PopArt"*, *"clean-world RETURNS are ±1-ish"*);
* with PopArt off, `_vd_target = rollout_data.returns` — raw. Δ = 24/50 = 0.48, so the whole ±1
  return range lands inside **~4 of 51 atoms**, and HL-Gauss's σ = 0.75Δ = 0.36 is itself a third of
  that range. Under `--value-from-dist` that `E[Z]` **is** the critic feeding GAE.

Nothing downstream distinguishes it: `value_dist/mean_abs_err` looks *better* as the support widens,
and `pit_mean` is the only tell for the mirror-image failure (returns outside the support, absorbed
into the edge atoms — a ±30/−35 terminal in a `[−12, +12]` support means the critic cannot represent
a win at all).

**Litigated.** `_terminal_scale_guards` in `src/main/train/config.py` prints a
`[Reward] ⚠️ VALUE-DIST SUPPORT vs TERMINAL SCALE` line when a value-dist head is built, PopArt is
OFF, and the achievable returns either fall outside `[vmin, vmax]` or span fewer than
`_MIN_SUPPORT_ATOMS = 8` atoms. A WARNING, not a refusal — a wide support may be deliberate ahead of
a reward change, and a launch that works today must not become a `FATAL_CONFIG`. Skipped entirely
under PopArt, so **every run ever launched is byte-identical**.

### F2 — the ordering guard covered one side (MEDIUM)

Wave A warns when `draw_penalty > -victory_value` ("a draw beats a loss"). The likelier mistake is
the other one: type `--victory-value 1.0`, inherit the `-35.0` default. The ordering is then
*correct* and the guard is silent, while the reward stream is dominated 35:1 by the outcome the arm
exists to make rare. A run in that state is not "1 TERMINAL"; it is a stall-avoidance objective
wearing that label, and no metric names it.

**Litigated.** `[Reward] ⚠️ TERMINAL SCALE` at `|draw_penalty| > 3 × victory_value`. The validated
composition is 35/30 = 1.2× and the clean ruling is 1/1, so no composition anyone has launched trips
it.

### F3 — a flattering zero in the metric the clean arm is watched by (MEDIUM)

`apply_winprob_pbrs` returned `reward_share = 0.0` when `raw_absmean == 0`. Under `--no-hand-shaping`
the unshaped stream is TERMINAL-ONLY, so a rollout ending no episode has exactly zero unshaped
reward — and the shaping is then **100%** of it. `0.0` is the reading an operator scans past. This is
the project's own ABSENT-never-zero rule, which wave C applied to `train/q_winprob_loss` one commit
later.

**Litigated.** It now returns `float("nan")`; the key is always present, the value is never a score
that was not measured.

### F4 — a TIE is not a timeout (MEDIUM)

`counterfactual._battle_outcome` emits **four** values — `win` / `loss` / `tie` / `unfinished`.
`harvest.label_one` computed `n = win + loss` and `n_timeout = total − n`, so a tie was filed as a
timeout — which is none of the three things the module header says that bucket counts ("a transport
error, a bridge timeout, a horizon overrun"). The implementation contradicted its own contract, and
the cost is a **biased label**, not a mislabelled counter: every dropped rollout is a non-win, so
`k/n` overstates P(win) on exactly the states where a game can end even. The owner's clean-world
ruling says the same thing in the reward's language (a draw scores `−victory`, i.e. as a loss).

**Litigated.** Ties are adjudicated into `n`, and counted separately in `provenance["n_tie"]` so a
label's composition stays auditable.

### F5 — the Q head reaches the cf forward, and the cf forward clobbers a stash (LOW)

Wave C added `q_winprob_on` / `q_onpolicy_on` to `cf_any_on`, so `_cf_sample_and_forward()` — an
extractor forward over a DIFFERENT batch of rows — now runs on minibatches where no cf readout is
configured at all. That forward overwrites `features_extractor.last_value_pooled`, which is the
FitNets hint `--distill-value-feat-coef` matches the teacher against.

It is **correct today**, purely because `_s_vfeat` is captured earlier in `train()` than the cf
block. That is a statement-ORDER fact invisible to every behavioural test (the two never disagree
unless a block moves), and the ledger already records one stash-clobber of this shape.

**Litigated** as a source-order assertion (`ai_v12_intersection_test.py` §5), the same idiom
`entry_source()` scans use, covering both the distill hint and the cf-twin fold.

### F6 — frozen-φ × `--compile-trainer` on real CUDA: still UNEXERCISED (LOW, escalated)

Wave A recorded this honestly and it remains true: `compile_trainer_extractor` refuses a non-cuda
device, so no CPU tier can reach the composition. What IS covered on CPU is the structural half — the
compile patches `model.policy.features_extractor.forward` and `phi_model` returns a *different*
object, so patching the live extractor leaves the source serving φ (revert-verified below).

Residual risk is judged LOW and no code change is recommended: the source is loaded **eager** by
design, is never in the optimizer's param groups, and is in `_excluded_save_params`, so it is outside
everything `--compile-trainer` touches.

**Recommended resolution:** on the first CUDA launch that uses `--win-prob-pbrs-source`, confirm
`train/pbrs_phi_mean` is non-constant across the first three rollouts and that the startup line
`🧊 [WinProbPBRS] frozen φ from …` appears above the compile banner. That is a 30-second read, and it
is the only place the composition can be observed.

### F7 — an evidence-free `q_label` trains at weight 1 (INFO, escalated)

`_parse_q_labels` defaults a missing `n_rollouts` to `0`, and `q_masked_binomial_nll` then does
`n = n_rollouts.clamp(min=1.0)`. A producer that ships `{"action": k, "label": 0.9}` with no count
therefore contributes a full single-observation NLL term, indistinguishable from a genuine
1-rollout label. The head is LATENT (mode `none`, both coefficients 0, no producer emits `q_labels`
yet), so nothing is affected today.

**Recommended resolution:** when the producer lands, either require `n_rollouts >= 1` in
`_parse_q_labels` (a counted `q_labels_no_evidence` field skip, matching the range checks beside it)
or make the default explicit at the schema. Not changed here: the right answer depends on the
producer's wire format, which does not exist yet, and inventing a refusal for it now would be a
guess.

---

## Clean bill — checked and found SOUND

Recorded because checked-and-sound is information, and because each of these was a live hypothesis.

**Migrations (waves A + D + C stacked).**
* The v105 → v106 → v107 chain runs to completion on a fabricated v104 config and on **every** real
  archived `model_config.json` of the current generation on this box (8 distinct config versions,
  97 → 107). Every `ModelVersion` field is present afterwards, no key is extra, and
  `ModelVersion(**out)` constructs. Pre-v96 configs are refused loudly by design, not migrated.
* Now pinned permanently (`ai_v12_intersection_test.py` §4), including the real-archive sweep, which
  skips rather than silently passing when `main_models_dir()` is `None`.

**recorded ≠ effective (genre 2), all ten new flags.**
* Wave A's four reward fields ride `RewardConfig.from_args` / `from_dict`, so they reach the eval
  worker automatically (`main/eval_worker.py:251` builds `RewardConfig.from_dict(model_config)`) —
  no hand-threading, which is the defect class the docstring there names.
* Wave D's two clock switches reach BOTH the training env and the server-free `RewardTracker`
  through the single `ProgressClock.apply_reward_config`.
* Wave C's `q_winprob_mode` is in `flag_registry.REGISTRY`, so `arch_toggles_from_args` →
  `_run_arch_toggles` → `current_model_version` pick it up **generatively**. It is also in
  `arch_toggles_from_model` and `check_compatible`. Specifically checked: wave A's frozen-φ loader
  calls `_cmv_w(mappings, **_run_arch_toggles(args))`, so a Q-head run loading a φ source compares
  like-for-like — the wave A × wave C seam holds because neither list is hand-maintained.
* `checkargs` reads the parser's own `_actions`, so all ten are recognised without a change.

**Wave D's `decision_was_forced_switch`.** Set at **all four** `TurnDelta` construction sites
(`build_from_events` ×2, the empty delta, and `turn_delta_legacy.build_legacy`) — a new field missed
on the legacy fold would be a silent-inert on that path. `EpisodeTracker.update_progress_clock`
reads `_history[-2].legal` *after* `record()` has appended the current context, so `legal_prev` is
genuinely the OPENING decision. The training env passes `legal` on both calls; the
`RewardTrackingMixin` fall-back to the pre-fix reading is documented and deliberate.

**Wave A's fold/census unification.** `_apply_pbrs_suppression`'s `hand_shaping` zeroing runs AFTER
`_apply_progress_clock` (which is the only writer of `no_progress_tax`) and BEFORE
`_fold_bias_refund`, so the clean composition really is TERMINAL-only. `_bias_active` consults
`_bias_term_active`, so `--no-hand-shaping` also skips the guarded COMPUTES without skipping any
cross-turn mutation.

**Wave C's `q_labels` masked likelihood.** An unlabelled cell contributes exactly zero to both the
numerator and the `(n·mask).sum()` denominator (verified by construction, not by claim).
`taken_action` defaults to index 0 with a zero mask, so the on-policy `scatter_` can never take a
negative index. `_q_logits` re-applies the head outside the cf forward's `no_grad`, and raises on a
batch-size disagreement rather than supervising the wrong board.

**PopArt × `hand_shaping` OFF.** The σ-collapse hypothesis does not reproduce: `_SIGMA_FLOOR = 1e-2`
and a ±1 terminal gives σ ≈ 0.5–1.0, three orders above the floor. Moot in any case for the
registered arm, which runs PopArt off — the real exposure there is F1, not σ.

**Wave E's live TB half.** `train/scaffolding_gauge` is genuinely covered by a tiny real PPO
(`scaffolding_test.py` §, including the no-head GAP case and the epoch-0-only read).
`scaffolding.py` / `exploitability.py` carry no hardcoded ±30, so `--victory-value` does not
invalidate them.

**Wave D's staller RNG.** Default path returns the `random` module itself (byte-identical); an
unparseable `$GEN3AI_STALLER_SEED` raises rather than silently falling back; the class attribute
covers the `cls.__new__` instantiation the unit suite uses.

---

## Claimed-test spot-verification (item 6) — all five REVERT-VERIFIED

Each claim was run green, then the guarded behaviour was reverted locally and the named test
confirmed to go red, then restored.

| wave | claim | revert applied | result |
|---|---|---|---|
| A | frozen φ does not move when the live network does | `phi_model` → always `model` | **3 FAIL** (`..._supplies_phi_for_EVERY_buffer_row`, `..._does_NOT_move_when_the_live_network_does`, `..._patching_the_live_extractors_forward...`) |
| C | OFF is byte-identical / append-never-insert | build the head unconditionally | **3 FAIL** (`..._adds_ONLY_its_own_parameters`, `..._state_dict_KEY_CENSUS...`, `..._APPEND_NEVER_INSERT...`) |
| D | the two fixes touch obs col 1602 and nothing else | `_gates` reads the CLOSING tense again | **6 FAIL** incl. `test_each_fix_actually_bites[decision_tense-49]` |
| B | the fit's optimizer cannot name a trunk parameter | `_assert_head_only` → `return` | **1 FAIL** (`..._cannot_even_name_a_trunk_parameter`) |
| E | the gauge is read from epoch ZERO only | drop the `epoch == 0` gate | **1 FAIL** (`test_the_gauge_is_read_from_EPOCH_ZERO_only`) |

⚠️ One methodological note for future reviews: the first attempt at E's revert (dropping
`scaffolding_on`) was **non-discriminating** — the gate is redundant with the `_wz is not None`
check one line below, so the suite stayed green and briefly read as "untested". A revert that does
not change behaviour proves nothing about the test; pick the revert that changes the OUTPUT.

Wave D's confinement test is `@pytest.mark.sim` and needs `deps/pokemon-showdown`; it errors at
fixture setup in a fresh worktree with a team-validation `MODULE_NOT_FOUND`, not a code fault. After
`git submodule update --init` plus the two guarded symlinks it passes 6/6 in 26 s.

---

## Changes landed by this review

| file | change |
|---|---|
| `src/agents/training/ai_v12_intersection_test.py` | NEW — 15 tests over the five named intersections (F1, F2, F3, F5, and the stacked migration clean bill) |
| `src/main/train/config.py` | `_terminal_scale_guards` — the `TERMINAL SCALE` and `VALUE-DIST SUPPORT` launch warnings (F1, F2) |
| `src/agents/training/winprob_pbrs.py` | `reward_share` is NaN, never a flattering 0.0 (F3) |
| `src/main/harvest.py` | a TIE is adjudicated into `n_rollouts` and counted in `provenance["n_tie"]` (F4) |
| `src/main/harvest_test.py` | two tests for F4 |
| `src/agents/training/CLAUDE.md` | the three-guard table, the `reward_share` NaN rule, the tie/timeout correction |

Behaviour under **every** flag combination anyone has launched is unchanged: the two new warnings are
skipped under PopArt and at the historical 30/−35 pairing, the NaN only replaces a value that was
already meaningless, and the tie path changes a count that was previously mis-bucketed.
