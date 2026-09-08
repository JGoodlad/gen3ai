# The counterfactual tier — `lookahead` · `better_line` · `replay_counterfactual` · `falsify` · `falsify_scan` · `calibration`

Owned by this tree. `src/main/prober/CLAUDE.md` keeps what the tier IS (three re-roll/clone-powered
probes, bridge-eval traces only, each spawning a sim child), the quota rule `calibration` reports,
and the currency rule for `overvalue_tau`. This file is the per-method reference.

- `lookahead(battle_id, inv=, invs=, worst=, n_seeds=0, followup=)` — **one-ply VALUE-DELTA**
  (`lookahead.py`): for an anchored `move_selection` decision, RE-ROLL the turn under each LEGAL action
  (the opponent plays its RECORDED move), materialize the resulting one-sided successor obs through the
  real encoder, and read the loaded model's **V(s′)** — per-action ΔV, "what would the critic have
  valued each alternative at" (the model-scored variant the model-free `falsify` deliberately defers,
  + the distributional / win-prob heads on the successor via `ProbeModel.value_dist_at`/`win_prob_at`
  when the run trained them). Two faithful modes share one call: the **CRN** headline (the `"original"`
  seed — hold the realized dice, vary only OUR action, so ΔV isolates the action's effect; the CHOSEN
  action's CRN successor reproduces the REAL next state so its `value_crn` ≈ the trace's
  `recorded_next_value`, a built-in consistency anchor) and a **dice-averaged** `value_mean`±`value_std`
  over `n_seeds`>0 fresh seeds. A candidate whose turn ENDS the battle is reported `terminal` (win/loss),
  not a numeric V. Loads the exact→nearest→recent model; **requires the trace's `*_reconstruction.json`
  sibling**. (`ProbeModel.value` is the V(s′) primitive; the successor obs is materialized from
  `reroll.prefix_pN_chunks + reroll.pN_chunks` per the obs-materializer recipe.) **The whole
  `(candidate × seed)` sweep resolves in ONE Node process** via `reconstruction.reroll_many` (each arm
  = a fresh session = byte-identical to a single `reroll_turn`, modulo `|t:|`), so the lookahead pays the
  ~677 ms Node-spawn cost ONCE instead of once per candidate (~9× on a full 9-action sweep; pinned by
  `utils/bridge/reroll_many_parity_fuzz_test.py`).
- `better_line(battle_id, inv, depth=2, beam=3, top_k=4, interior_opponent=, opponent_ckpt=,
  confirm_rollouts=0)` — **SEARCH for a better line** (`better_line.py`): the depth-≥2 generalization of
  `lookahead` (which IS its depth-1 instance). A shallow, CRN-anchored **beam over the critic** that
  branches a search TREE by CLONING mid-battle states in the warm `SearchSession`
  (`utils/bridge/search_session.py` → `search_driver.js`'s `State.serializeBattle` on node, or the rust
  binary's `BridgeSession::snapshot` under `--impl rust` — the only primitive that makes depth>1
  feasible), expands OUR top-k actions by policy prior, scores each
  successor's V(s′) on the materialized ONE-SIDED obs (`model.values_batch`, one critic forward per
  ply), keeps the top-`beam`, and recurses; backup is **max-over-our-continuations** and the returned
  LINE is the principal variation. Returns ONE human-legible **contrastive trajectory**: the divergence
  (`best_alternative` + ΔV / `win_prob`), the per-ply `principal_variation`, and the chosen-vs-best
  `candidates`. **Faithful-conditional opponent:** the RECORDED move at the divergence ply (the chosen
  action is `recorded_exact` → the `value_crn` anchor, identical to lookahead), and at INTERIOR plies the
  reloaded opponent reacts greedily on ITS OWN one-sided obs (materialized via the opponent's
  action-history from `obs_materializer.infer_action_indices`) — `interior_opponent`: `"self"` (the
  trainee as a flagged proxy, default), `"ckpt"` (`opponent_ckpt`), or `"none"` (sim default). Depth-1 is
  faithful regardless; only depth≥2 leans on the interior model.
  ⚠️ **The beam plays a declared regime MIX, and the payload says so.** The divergence ply reproduces
  what the opponent actually did — the RECORDED regime, stochastic for a sentinel, which
  `replay.build_opponent` now honours — while every interior ply plays **GREEDY argmax**, bypassing
  that seam on purpose. The reason is what a "better line" claims: a line that survives the
  opponent's BEST answer. Sampling the interior opponent would return lines that beat one draw of a
  die and call them better — an optimistic search bias in the exact direction this tree already pays
  for. Greedy is the standard worst-case-opponent search assumption and makes the reported ΔV a
  lower bound. Two consequences: a line whose refutation is a *low-probability* reply will not be
  found, and the beam's numbers are **not** ply-for-ply comparable with `replay-counterfactual`'s
  Monte-Carlo win rate (which plays the recorded stochastic regime to the end). Every payload
  carries `interior_opponent_regime: "greedy"` so no saved artifact can hide which produced it —
  the same discipline as `opponent_source`, and pinned by a test that fails if the key disappears
  (task #29, 2026-08-22). `confirm_rollouts>0` CONFIRMS the
  recommended first action with an actual Monte-Carlo replay-to-end vs the RELOADED REAL opponent
  (`replay_counterfactual`) → win-% ± Wilson CI — the ground-truth check on the critic's claim (the
  three-tier eval: search by V, report by ΔP(win), confirm by rollout). Loads the
  exact→nearest→recent model; **requires the trace's `*_reconstruction.json` sibling**. Default depth 2
  is the legibility/cost sweet spot (≈4 s search; depth 1 ≈2 s, depth 3 ≈5 s — try `--depth 3` for setups
  that pay off a turn later). Faithfulness pinned by `utils/bridge/search_clone_parity_fuzz_test.py`.
  **Perf:** the bottleneck is obs materialization (the serializeBattle clones are ~2%), so three levers
  cut it ~1.5× (5.7 s → 3.7 s on a depth-2 decision): (1) ONE shared `replay_battle` feeds both the anchor
  choice-map AND the opponent's `infer_action_indices` history (was two full replays); (2) the per-node
  policy forwards are BATCHED (`action_probs_batch`, was one forward per node); (3) `materialize_decisions`
  `encode_only_at={target}` encodes the obs ONLY at the decision the search reads — every other decision is
  TRACK-ONLY (the tracker still advances faithfully, the ~80%-cost encode is skipped), so a node replays
  its prefix as cheap tracking and encodes one obs. Lever 3 rides `Gen3Player.track_decision` (the tracking
  half of `embed_battle`, extracted byte-identically — the live obs path is unchanged, pinned by
  `obs_roundtrip_fuzz_test`); its bit-for-bit obs equivalence (track-only prefix vs full encode) is pinned
  by the clone-parity fuzz. **L4 (training-scale):** `better_line_decision(session=…)` accepts an injected
  WARM `SearchSession` reused across battles (`open_root(record=…)` + the driver `NODES.clear()`s per root),
  so a background worker pays the ~0.68 s Node spawn ONCE not per search (~1.66× cumulative, ~3.4 s warm).
  This is the expert tier of the **search-as-teacher** plateau-breaker (`designs/ai_v6/design_search_teacher.md`):
  selective offline ExIt — search + rollout-confirm the worst `falsify`-flagged loss craters (exact reloaded
  opponent), CI-gate strictly-better corrections, distil into a policy/value **aux loss** (no on-policy /
  PPO-core change). NOT built; the search + 3-tier-confirm here are its ready foundation.
- `replay_counterfactual(battle_id, inv, action, n_rollouts=1, opponent_ckpt=, opponent_source=)` —
  **COUNTERFACTUAL replay-to-end** (`replay.py` → `utils/bridge/counterfactual.py`): "could the model
  have won if it hadn't choked this turn?". Pick up the recorded battle at `inv`'s turn, substitute
  `action` (a legal action index) for OUR side, then play the rest LIVE — the trainee's GREEDY policy vs
  the **RELOADED real opponent** — to a win/loss. The driver reuses `run_local_battles` with both
  players' `choose_move` SCRIPTED to replay the recorded commands until the divergence (faithful: the
  bridge `START` uses the recorded resolved seed + both recorded packed teams, and each scripted
  Gen3Player decision runs `embed_battle` + `tracker.advance(recorded_idx)` — the recorded index
  recovered by inverting the recorded choice string — so the post-divergence turn-history stays faithful,
  proven bit-for-bit by `counterfactual_fuzz_test.py`). The opponent is RELOADED: a reproducible bot is
  rebuilt exactly; `opponent_ckpt` loads any checkpoint (e.g. a self-play sentinel) as the opponent;
  else the trainee's own model stands in (a flagged `self_model_approx`).
  ⚠️ **A checkpoint opponent plays the RECORDED sampling regime — STOCHASTIC at temp 1.0**, which is
  what `eval_worker` recorded for a pool sentinel (`stochastic = not eval_sentinel_greedy`, default
  False). It used to be hard-wired greedy, and that is not cosmetic: a greedy copy of a net is
  strictly stronger than a temp-1.0 sample of it, so every Monte-Carlo win-rate against a reloaded
  sentinel was biased LOW and every predicted-minus-MC gap biased HIGH — *in the direction of the
  hypothesis under test*. Measured over 477 sentinel states (G0, 2026-08-22): **+0.037 [+0.007,
  +0.066]**. Override with `--opponent-regime {recorded,stochastic,greedy}`; the regime is written
  into `opponent_source` (`ckpt_stochastic:<path>`), so no result can hide which opponent played.
  `n_rollouts`>1 resamples the
  post-divergence dice (`local_sim_bridge.js`'s `resumeReseed` PRNG swap at the divergence turn) for a
  Monte-Carlo win-rate ± **Wilson CI**; `n_rollouts`==1 is the single realized-dice line (NOT a
  probability — a `caveat` says so). Loads the model; **requires the `*_reconstruction.json` sibling**.
  Each rollout is a full in-process game (seconds); the `caveats` flag the self-play approximation +
  that it best illuminates THROWN-LATE losses, not matchup-lost-from-turn-1 ones. `narrate=True` (the
  CLI `--narrate`, and on by default from the web) additionally captures the **move-by-move play-by-play**
  of the first recovered WIN + first LOSS (`winning_trajectory` / `losing_trajectory`: per-turn
  `{turn, events}` from OUR one-sided view — moves / switches / damage / faints / crits / status / win)
  via `run_local_battles`'s `chunk_sink` → `counterfactual.summarize_trajectory` — so you can read HOW a
  different move wins and what the bot did (e.g. spamming Earthquake into a Levitate Gengar).
- `falsify(battle_id, invs=, worst=, n_seeds=, n_alts=, followup=)` — **dice
  attribution** (`falsifier.py`): was a loss decision LUCK or a reducible MISTAKE?
  RE-ROLLS the real turn via the battle-reconstruction layer
  (`utils/bridge/reconstruction.py`): fix-both-actions under N fresh PRNG seeds →
  where does the REALIZED outcome (the special `"original"` seed — no PRNG swap,
  exact recorded follow-ups) sit in the dice distribution (`luck_percentile`);
  plus a **paired** alternative-action sweep (top-k legal alts by saved logits,
  SAME seeds = common random numbers, mapped to sim choices by the real action
  mapper via `obs_materializer.map_actions_at`) → `paired_advantage` ± SE per alt.
  Both axes score an omniscient **material margin** (`alive diff + hp-frac diff` —
  referee-side analysis; the one-sided wall constrains the encoder, not analysis).
  Verdicts `LUCK`/`MISTAKE`/`MIXED`/`NEUTRAL` (thresholds in `falsifier.py`,
  echoed in the output). Default anchors = the `worst` most-negative-δ
  `move_selection` decisions on distinct turns (a forced-switch crater attributes
  to its turn's move decision — re-rolls anchor at start-of-turn rounds only).
  Model-free (no checkpoint). **Requires the trace's `*_reconstruction.json`
  sibling** (bridge-eval traces written by the reconstruction layer; older /
  websocket traces raise with that explanation). Alternatives the sim refuses
  (maybe-trapped) are detected from the `[Unavailable choice]` error in the
  one-sided suffix and excluded from the verdict when mostly refused.
  ~1–2 s per arm at 40 seeds (fresh Node replay per seed); a decision with 3
  alts ≈ 5–10 s. Each falsified decision is annotated with `anchor_delta` — the
  TD-residual δ that selected it — the one source of truth `falsify_scan` weights
  craters by (`falsifier.anchor_deltas`, which `select_anchors` now ranks over).
- `falsify_scan(outcome="loss", opponent=, step=, limit=20, worst=2, n_seeds=32,
  n_alts=2, followup=, concurrency=1)` — the **RUN-LEVEL** generalization of
  `falsify`, an **input to** the distributional-critic decision (it *brackets* the
  headroom, it does not measure it — **read `caveats`**). Falsifies the worst δ-craters
  of every matching battle (default losses) that carries a `*_reconstruction.json`
  sibling, then aggregates the per-decision verdicts **weighted by crater magnitude
  (|anchor δ|)** into four levers — a **measurement-time attribution at one frozen
  checkpoint, NOT independent root causes**: **`LUCK` → `aleatoric`** (the chosen
  line's realized outcome sat in the dice bad-tail — reducible *only* by a
  risk-SENSITIVE policy avoiding a lower-variance line, which today's risk-neutral
  PPO/this scan don't have/test), **`NEUTRAL` → `unattributed`** (a real crater the
  sweep pinned on NEITHER luck NOR a better action — **NOT proven critic error**;
  "not bad-tail luck" = a TYPICAL outcome, equally a genuinely-lost position;
  splitting it needs the model-based V(s)-vs-return calibration probe, *not run here*),
  **`MISTAKE` → `policy_reducible`** (a top-k alt provably beat the chosen action —
  the **only proven** leg; but actor-critic coupled, so a better critic still reduces
  these over training), **`MIXED`** (both). `gate.critic_headroom_upper_bound` =
  LUCK + NEUTRAL share is an **UPPER BOUND** — it can only inflate as the shallow
  mean-only alt-sweep (top-`n_alts` by logit · single re-rolled turn · `followup`
  mid-turn) fails to *prove* a mistake (mass falls into LUCK/NEUTRAL), and it folds
  in the unproven `unattributed` leg. Both **`weighted_shares` (|δ|) and `count_shares`**
  are reported — a large gap means a few big ambiguous craters dominate (anchors are
  pre-selected by worst δ). `dominant_lever` is `None` on an empty scan or a near-tie
  (within 0.05). A **`caveats`** list (mirroring `triage`/`probe`) carries all of the
  above in the data. `coverage` reports `n_matched` / `n_with_record` / `n_falsified`
  / `n_capped_by_limit` / `n_skipped_no_record` / `n_battle_errors` / `n_decision_errors`
  (nothing **silently dropped**). Weighting falls back to count-based
  (`weighting="uniform_fallback"`, announced) when every δ≈0 (e.g. placeholder values);
  `"none"` on an empty scan. Model-free. `concurrency` > 1 falsifies battles in
  parallel (each re-roll spawns Node → raise it only on an **idle** box; it contends
  with a live training run). The coarser defaults (worst=2, 32 seeds) keep a 20-loss
  scan to a few minutes — the run-level statistic gets its power from MANY decisions,
  not deep per-decision seeds. `include_decisions=True` adds each battle's full
  per-decision list to its row (the calibration probe reads it).
- `calibration(outcome="loss", step=, opponent=, limit=20, worst=2, n_seeds=32,
  n_alts=2, concurrency=8, n_bins=10, overvalue_tau=None)` — resolve `falsify_scan`'s
  **unattributed** (NEUTRAL) bucket into **`critic_overvalued`** (epistemic — a
  better/distributional critic helps) vs **`lost_position`** (the critic was right),
  by comparing the RECORDED value V(s) to the REALIZED discounted return G(s) =
  `Σ γ^k r_{t+k}` (the MC value target). **Model-free** (uses recorded V — no
  checkpoint); the falsify pass that finds the unattributed craters runs at
  `concurrency` (default 8). **Selection-aware (the crux):** a loss-conditioned V−G
  is biased positive *by construction* (losses are the below-V tail of any critic),
  so the baseline is a **reliability curve over BOTH wins and losses, binned by V**
  (`_reliability_curve`/`_calibration_stats`/`_reliability_gap_at`, pure + unit-tested);
  a crater is `critic_overvalued` only if the critic SYSTEMATICALLY over-values at its
  V-level (reliability `gap` > `overvalue_tau`). **The output self-diagnoses the
  remaining confound**: `overall_calibration.bias_on_wins` (<0) / `bias_on_losses`
  (>0) is the CALIBRATED-critic signature, and `captured_win_fraction` ≠ the true win
  rate (eval QUOTA over-captures losses), so the unconditional bias and the reliability
  gaps are SELECTION-SKEWED — `critic_mean_reducible_upper_bound` is a LOOSE upper bound
  until reweighted to the true win rate, or replaced by the selection-free **gold-standard
  re-roll → policy-rollout → return PIT** (the true distributional-critic validator,
  deferred — needs a mid-game rollout primitive). Reads the `caveats`; this is the cheap
  aggregate proxy, knowingly confounded on a quota-captured sample.


## `overvalue_tau` in the critic's own units — the full measurement

  🚨 **`overvalue_tau` IS IN THE CRITIC'S OWN UNITS, and until 2026-09-06 it silently was not**
  (`gen3_prober_winprob_currency_v1`). It defaults per CRITIC CURRENCY — `None` resolves it from
  the run — because the two eras' V are not the same quantity: 5.0 SHAPED RETURN UNITS on a critic
  spanning roughly ±30, ≈**0.083** P(win) under `--critic winprob`, the same 1/12-of-span fraction
  either way. Carried across unchanged, the shaped 5.0 **exceeds the entire representable range of
  a probability gap**, so no crater can clear it and `critic_overvalued` reads a confident **0** —
  a units error in the shape of a finding. Measured on `ai_v12_01_winprob_critic` step 8M — the
  FIRST win-prob arm, since **KILLED** (it launched without the production surface; the relaunch is
  `ai_v12_02_winprob_critic`): the reliability gaps ran **0.067–0.461** and the headline flipped
  from `critic_mean_reducible_upper_bound` **0.0** to **0.4997** once the tau was in the right
  currency. The kill does not weaken it — the defect is in the CURRENCY of the recorded `values`,
  which `--critic winprob` fixes at [0,1] regardless of what else the arm was training.
  An EXPLICIT value is still honoured verbatim (a threshold sweep must not be re-scaled), and the
  result carries `params.overvalue_tau_source` plus `critic_currency`. **A tau no gap in the run
  can reach now produces a loud `threshold_warning`** naming the largest observed gap — the durable
  half, since any future currency change re-opens this for every value-unit threshold in the tree.
