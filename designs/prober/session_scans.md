# `ProbeSession` run-level scans — `loops` · `triage` · `probe`

Owned by this tree. The facade contract is in `src/main/prober/CLAUDE.md` and the CLI invocations
in `python -m main.prober.query --help`; this file is the per-method reference for the three run-level folds.

- `loops(outcome=, opponent=, step=, max_battles=, near_zero_frac=0.01, top=12)` — **model-free
  BAIT-LOOP scan**: *the opponent voluntarily pivots a mon our attack cannot touch, and we fire
  anyway — repeatedly.* Detection lives in `main/prober/loops.py` (pure, no torch, no session,
  unit-tested on hand-written protocol lines); this method is the run-level fold. **It reads the
  raw Showdown PROTOCOL from each battle's `*_replay.html`, never the rendered timeline** — the
  rendering is a SENTENCE, and a detector must key on the fact underneath it. It used to be worse
  than a style point: `— no effect` collapsed an immunity, a full-paralysis `cant` and an unpriced
  small hit into one phrase (the calibration battle's T54 and T40). Both of those now read honestly
  (`— couldn't move (fully paralyzed)`, `— outcome unrecorded`), but the rule stands as written —
  the timeline's job is to be readable, the detector's is to be exact, and they must not be wired
  together.
  Definitions, fixed in `loops.py` so every surface means the same thing: a **voluntary pivot** is
  a `|switch|` with no faint earlier in the turn block and no `|drag|` (turn-0 leads excluded); we
  **moved into** it if we then used a move, after the arrival, TARGETING that side (a self-targeting
  Recover/Protect is not a bait and never enters the denominator); a **whiff** is `immune` /
  `fail` (a `-fail` with no external `[from]` cause) / `near_zero` (≤ `near_zero_frac` of the
  target's HP) — a **MISS is counted separately and is never a whiff**, because taxing dice would
  make the metric partly a luck reading; a **loop** is one `(move, arrival)` pair whiffing ≥2× in a
  battle (symmetric over the battle); a **re-click** is the 2nd..Nth click of such a pair (ordered)
  — the sharpest signal, since an immunity is deterministic and fully observable once seen.
  ⚠️ **Sides come from the recorded board, never from `p1`** (`identify_our_side`: the side whose
  protocol active agrees more often with the trace's `our.species`) — eval seats the trainee on
  either side, and a mirror match makes species names useless as a tell. An undecidable battle is
  SKIPPED with a reason and counted in `coverage`, never silently judged.
  Per-decision joins (model-free, from the summary + npz): chosen-probability on whiff decisions,
  ΔV and ΔP(win) bucketed `loop_step` / `other_bait` / `other` (the third bucket is the point — a
  loop-turn ΔV means nothing without the ordinary turn from the SAME battles; **under the win-prob
  critic V *is* P(win)**, and a core trace stores NaN in its separate `win_probs` array, so `loops`
  reads ΔP(win) from `values` there — `median_delta_win_prob` is the number to hold against the
  baseline's −0.096; a NaN series is dropped, never medianed), and the α/β readout
  on the same pivots split first-time / repeat / loop-step. **β's slot is graded STRUCTURALLY**
  (obs slot *k* = the *k*-th REVEALED opp mon, so the true slot is the arrival's index in the
  reveal order as of that turn) — never against β's printed species, which is an unsupervised
  posterior decode that names an off-team mon on 73% of pivots; grading by it grades the head
  against itself.
  Three headline rates on three DIFFERENT denominators on purpose (`whiff_rate_per_pivot` /
  `whiff_rate_per_decision` / `loop_battle_rate`), each shipping `{n, d, rate}`, because the two
  registered CONFOUNDS are conditioned for rather than mentioned: loop rate rises with game LENGTH
  and concentrates in WINNING positions (gen-15: 23.1% loop-battle rate in wins vs 7.0% in
  losses), so `by_outcome` is always reported and the comparison is win-arm to win-arm. A `mirror`
  block runs the same detector with the sides swapped — a CONTROL (it measures the opponent),
  not a target. `--opponent` is an **fnmatch pattern**, so `sentinel_*` reads the self-play
  sentinels as ONE population (an exact name still matches exactly); the gen-15 baseline was
  measured there. Those baselines live in `loops.LOOP_BASELINES` and ride the result as
  `baseline`, so the CLI and any future view quote ONE reference point.
  Measured on gen-15 (`ai_v9_18_gen15_v8rewards_0818`, 843 sentinel battles, ~2 s): 16.5% of 4923
  moved-into pivots whiff · 117 loop battles · 264 re-clicks · median chosen-prob on loop steps
  **0.963** · loop-step median ΔV −4.31 / ΔP(win) −0.096 · β slot 52.0% first-time → 65.9% repeat
  → 82.1% on loop steps · α SWITCH 76.3% on loop steps. **Both heads are right at the moment the
  wrong move is fired at p≈0.96** — the gap is actuation, and the injection probe proved no channel
  exists. The pre-registered gen-16 bars are in `designs/research_state/bait_loop_hunt.md`; this
  method is that hunt's instrument. CLI: `query loops <run_dir> --opponent 'sentinel_*'`.
- `triage(step=, opponent=)` — **rank the failure LEVERS across a whole run**
  (model-free; the natural first call when the question is "what do we fix next").
  Categorizes every loss's single worst-ΔV turning point into a fixed taxonomy
  (`engine.LOSS_TAXONOMY` — the one place to extend), then ranks the categories by
  `est_recoverable_winrate_pct` = mean over the fixed-**bot** opponents of
  `loss_rate(opp) × category_share(opp)` (an upper bound: assumes fixing the lever
  flips that loss). Each category carries the **lever** it implicates (obs / reward /
  policy / critic-capacity / upstream / measurement), a blurb, `by_opponent`, and
  worst-turn `examples`. **Every current run has NO incoming-damage belief in its observation**
  (the block is deleted): `active_pko` is `None`, `belief_coverage` is `{n: 0, d: <losses>}` and a
  caveat says so. The four belief buckets below (`surprise_ohko`, `ignored_threat_death`,
  `doomed_already`, `attrition_death`) cannot fire there; a death is split only by its mon's health
  into `death_healthy_unread` / `death_chipped_unread` (UNATTRIBUTED: the lever is "drill in with
  `analyze`"). Before 2026-10-26 those deaths were filed as `attrition_death` ("belief only PARTLY
  fired") — 44 % of `rb_x35_n1p_s3501`'s losses, a claim about a belief that does not exist. The
  taxonomy splits the deaths by the signal that names the
  lever: belief **under-read** a healthy death = OBS (`surprise_ohko`); belief
  **fired** + a pivot existed but the mon died = REWARD/POLICY (`ignored_threat_death`,
  the under-switch target); no pivot left = UPSTREAM (`doomed_already`); already
  fainted = a forced replacement, look one turn back (`post_faint_replacement`). The
  no-death value craters split on **whether the model rated itself WINNING** right before
  the cliff (`engine._was_winning`): WINNING then craters = `critic_blindspot` (a confident-wrong
  THROW — CRITIC CAPACITY / a missing obs feature); already behind = `positional_grind`
  (upstream/material — never ahead to throw). **The winning signal is the CALIBRATED win-prob head
  `P(win) ≥ wp_even` (default 0.5)**, NOT the sign of V — V is a shaped/discounted RETURN with a
  structural **negative offset** (a measured self-mirror 50/50 reads V≈−6.5), so the
  old `V>0` test systematically OVER-counted grinds (mislabeled even/favored positions as "already
  behind"). It falls back to `V > v_even` only when no win-prob was recorded, and **`v_even`
  defaults from the run's CRITIC CURRENCY** (`None` ⇒ 0.0 shaped, **0.5** under the win-prob critic (the only critic),
  where V *is* P(win) and 0.0 is a certain loss rather than "even"); pass `--v-even` =
  the checkpoint's self-mirror V to re-center a head-less shaped run. **On a win-prob run the split
  reads V as P(win) (`wp_coverage` 1.0)** even though a core trace's `win_probs` array is NaN; the
  "no win-prob head / −6.5 offset" wording is for a SHAPED run only. The result
  carries a `winning_split` block (`wp_even`/`v_even`/`wp_coverage`/`critic_mode`/`v_units`) + a
  caveat naming the signal.
  Reads the
  true per-opponent win-rates from `eval_results.jsonl` (falls back to ranking by raw
  loss volume, announced in the metric + a caveat, when absent). Carries explicit
  `caveats` (loss-weighted sampling; one-cause-per-loss; bot-only rating weight; the winning-split signal).
- `probe(target, step=, opponent=, which=, max_decisions=)` — **representation
  probe**: fit a cross-validated LINEAR probe on the model's INTERNAL activations
  (`which='vf'` value-head / `'pi'` policy-head post-projection features, via
  `ProbeModel.features`) to recover a derived quantity, and compare it to a
  baseline probe on the raw obs/belief feature we ALREADY provide. The decisive
  "is this info already in the representation, or should we hand it over" test: a
  linear probe recovering X ⇒ the model computed X (a new feature is redundant); a
  probe that can't ⇒ an extraction gap (a real obs lever — "let it learn" hit this
  small net's capacity wall for X). Targets (`engine`-free label/group logic in
  `session._PROBE_TARGETS`): **`is_faster`** (true base-speed order vs the provided
  `active_outspeed`; contested = close speeds where Leftovers/Sandstorm-timing
  inference matters), **`damage_taken`** (HP fraction lost this turn vs `active_exp`;
  contested = the `active_pko` 0.1–0.9 coinflip band where a p50/p90 spread would
  help), **`faint_soon`** (imminent faint vs `active_pko`; grouped by whether the
  belief flagged it), and the **opponent-anticipation family** — **`opp_switches`**
  (will the opp voluntarily switch this turn) + **`opp_status_move`** (if the opp uses a
  move, status vs attacking — its INTENT) + **`big_hit_incoming`** — the pre-registered
  Gate-0 falsifier for an opponent-action / world-model head: rep AUC ≫0.5 ⇒ the trunk
  already models the opponent ⇒ such a head is REDUNDANT. (Measured @53M: opp_switches
  0.89/0.90, opp_status_move 0.82/0.87, big_hit 0.75/0.78, faint_soon 0.86 — opponent
  modelling comprehensively present; the head was FALSIFIED before building.) Every
  result splits **overall vs by-group** (the easy-vs-hard
  contrast is the signal) and reports the representation probe AND the provided-feature
  baseline. **`warnings`** says what the probe could not do on this run: the provided features
  (`active_pko` / `active_exp` / `active_outspeed`) are the incoming-damage belief, absent from every
  current observation, so `provided_feature_baseline` is `null` (no rep-vs-provided delta; read the
  representation probe's `lift` over `base_rate`) and a belief-grouped target (`damage_taken`,
  `faint_soon`, `faint_healthy`, `big_hit_incoming`) has ONE `unknown` group, not the contested band
  its `how_to_read` describes. A probe that cannot fit (< 30 labeled decisions, no model) returns
  `error` and the CLI exits 1. The probe stats (`engine.fit_probe`) are pure numpy — standardized
  ridge/logistic, k-fold OUT-OF-FOLD predictions, **auto-tuned l2** over a grid
  (essential at d≈512: a fixed weak penalty overfits to a negative OOF R²). One
  checkpoint load per call (step → one model). Measured @70M: `is_faster` rep
  AUC 0.94 on contested vs the provided feature's 0.75 (the model already infers
  speed — not a feature gap).

- `history_saliency(step=, opponent=, max_decisions=)` — per-turn-slot saliency of the TurnDelta
  history block, to decide whether `N_HISTORY_TURNS` could shrink. **That block was deleted
  (`gen3_frame_deletion_v1`), so on every current run it returns `{error, kind:
  "no_turn_history_block"}` at once** (and the CLI exits 1), decided from the loaded model's layout
  BEFORE any gradient (on a run the gap policy refuses, its typed `ArchDriftError` comes first). It used to run two backward passes over every decision of the step and only
  then say "no decisions with a turn-history block" (15 min on `rb_x35_n1p_s3501`).
  `analyze`'s own `saliency` / `value_saliency` are not that block: on a current run they cover the
  whole 2845-dim row as 19 named regions (`engine.decode.current_layout_regions`: 6 + 6 team slots,
  the two active-context blocks, global, reactive, pair history, event window, obs facts).

## Worked sequence on a current run (rb_x35_n1p_s3501, 2026-10-26)

Measured on a box at load average ~18 (a training run live), CPU only, `nice -n 19 taskset -c 5-7,13-15`; paths are
`models/<run>/…`. A battle id is the `id` field of `list` / `scan` (a full `*_summary.json` path):
the `short_id` beside it (`step_298000112/heuristic/loss_1002`) is the WEB's token and does not
resolve in the per-battle CLI commands.

**The model-loading commands split by the gap policy (`arch_drift.md`) since 2026-10-26.** Every current run
(config v161 and P*'s v158) trained under an older `obs_semantics` than HEAD's 162, so the `trace_replay`
commands — `analyze`, `find … disagree`, `probe`, `history-saliency`, `/analyze`, the `/game` model panels —
REFUSE with a typed `ArchDriftError` (`kind: obs_semantics`, exit 1, before any load) and name the way out
(`git checkout <the run's git_hash>`); the `fresh` commands — `lookahead`, `better-line`,
`replay-counterfactual` — run and record a `trained on obs_semantics v151, probed at v162` badge. The
`analyze` / `probe` timings below were measured BEFORE that refusal landed (de9cee938) and are what a run at
HEAD's semantics will cost; everything else is model-free or `fresh` and unchanged.

```bash
Q="python -m main.prober.query"; R=models/rb_x35_n1p_s3501
$Q summary $R                       # 1 s. critic_currency.mode "winprob"; 9 eval steps; model_views.status "current"
$Q triage $R                        # 11-72 s by load. belief_coverage {n:0}: deaths are death_*_unread, not attrition
$Q scan $R --outcome loss --opponent heuristic --limit 5      # 1 s. delta_v is ΔP(win); worst turn per loss
$Q loops $R --opponent 'sentinel_*' # 3.5 min (1017 battles). compare median_delta_win_prob with the baseline's −0.096
B=$R/eval_traces/step_298000112/heuristic/loss_s1_002_summary.json   # an `id` from list / scan
$Q overview $B; $Q turns $B         # 2 s each, model-free
$Q find $B value_drop --limit 3     # [7, 9, 13]
$Q analyze $B 7                     # REFUSED (obs_semantics) on a current run today; ~1 min when allowed: td_phrase, belief, damage_op, saliency
$Q lookahead $B --inv 7             # 2.5 min. one-ply V(s′) per legal action; needs the search_driver binary
$Q better-line $B 7 --depth 2       # 7 min
$Q falsify $B --inv 7 --seeds 8     # seconds. luck vs mistake for one decision
$Q probe $R is_faster --max-decisions 150    # REFUSED today (trace_replay); minutes when allowed: one model feature pass per decision
```

P* (`models/rb_x26_s2601`, 190 eval steps, 42k traced battles) takes the same commands with
`--step 280000170` (its 280 M eval; without `--step` triage/probe read the NEWEST step). Its
`summary` is 613 KB of JSON (380 checkpoints, 190 steps × 14 opponents): trim it, e.g.
`$Q summary $R | jq '{critic_currency, model_views, totals, last: [.steps[-3:][] | {step, totals}]}'`.
Measured there: triage 71 s, `scan --outcome loss --limit 5` 8 s, `loops --opponent 'sentinel_*'` 57 s,
`analyze` on a loss at 280 M ~75 s before the refusal (it resolves the NEAREST checkpoint: no `snapshot.zip` was kept for that step).

What is NOT there on a current run (a field that is `null` is the run, not a bug): `threats`,
`incoming` (the matchup and incoming-damage blocks are deleted), `opp_intent` (`--opp-intent-coef 0`),
`win_prob` (`values` IS P(win)), `sweep`, `matchups.multipliers` (`present: false`), the loops
`readouts`. `replay-counterfactual` and the `sim` tests need `src/rust_env/target/selfcheck/libpokesim_env.so`,
which a fresh worktree does not have (`FfiLoadError` names the `cargo build` line; the main checkout has it).

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### The investigation recipe

**Investigation recipe:** `triage` (which LEVER recovers the most rating — start here
for "what next") → `summary` → `scan --outcome loss [--opponent X]` (the worst turn in
*every* matching battle, ranked — model-free, fast) → `overview` the top battles (read
`notable.biggest_value_drops` / `faints`) → `find disagree` / `find value_drop` →
`analyze` the worst turn → `falsify` it (was that crater dice or a reducible
mistake — separates irreducible aleatoric variance from real policy errors) →
`falsify-scan` the whole run (aggregate that split across every loss into the
**crater-fraction bracket** — `aleatoric` [LUCK] · `unattributed` [NEUTRAL, the
residual the shallow sweep couldn't pin] · proven `policy_reducible` [MISTAKE];
`critic_headroom_upper_bound` = LUCK+NEUTRAL is an **upper bound**, not a
measurement — read its `caveats`) → `calibration` to split the `unattributed`
bucket (`critic_overvalued` vs `lost_position`) via recorded V(s) vs realized
return G(s) — **selection-aware** (reliability over wins+losses) and
self-diagnosing (`bias_on_wins`/`bias_on_losses`/`captured_win_fraction` expose the
eval-quota selection skew; on a quota-captured sample it is knowingly confounded,
so its number is a loose upper bound pending true-WR reweighting or the rollout-PIT).
γ is read from the run's `metadata.json`. (`triage` aggregates
`scan`'s per-battle worst turns into ranked failure CATEGORIES; `scan` is the
cross-battle generalization of a single battle's `notable.biggest_value_drops`.)
`ProbeSession(..., model_loader=fn)` injects a fake model in tests (no torch).
