# `ProbeSession` run-level scans — `loops` · `triage` · `probe`

Owned by this tree. The facade contract and the CLI invocations stay in
`src/main/prober/CLAUDE.md`; this file is the per-method reference for the three run-level folds.

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
  loop-turn ΔV means nothing without the ordinary turn from the SAME battles), and the α/β readout
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
  worst-turn `examples`. The taxonomy splits the deaths by the signal that names the
  lever: belief **under-read** a healthy death = OBS (`surprise_ohko`); belief
  **fired** + a pivot existed but the mon died = REWARD/POLICY (`ignored_threat_death`,
  the under-switch target); no pivot left = UPSTREAM (`doomed_already`); already
  fainted = a forced replacement, look one turn back (`post_faint_replacement`). The
  no-death value craters split on **whether the model rated itself WINNING** right before
  the cliff (`engine._was_winning`): WINNING then craters = `critic_blindspot` (a confident-wrong
  THROW — CRITIC CAPACITY / a missing obs feature); already behind = `positional_grind`
  (upstream/material — never ahead to throw). **The winning signal is the CALIBRATED win-prob head
  `P(win) ≥ wp_even` (default 0.5)**, NOT the sign of V — V is a shaped/discounted RETURN with a
  structural **negative offset** (a measured self-mirror 50/50 reads V≈−6.5; PopArt μ≈−3.6), so the
  old `V>0` test systematically OVER-counted grinds (mislabeled even/favored positions as "already
  behind"). It falls back to `V > v_even` only when no win-prob was recorded, and **`v_even`
  defaults from the run's CRITIC CURRENCY** (`None` ⇒ 0.0 shaped, **0.5** under `--critic winprob`,
  where V *is* P(win) and 0.0 is a certain loss rather than "even"); pass `--v-even` =
  the checkpoint's self-mirror V / PopArt μ to re-center a head-less shaped run. The result
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
  baseline. The probe stats (`engine.fit_probe`) are pure numpy — standardized
  ridge/logistic, k-fold OUT-OF-FOLD predictions, **auto-tuned l2** over a grid
  (essential at d≈512: a fixed weak penalty overfits to a negative OOF R²). One
  checkpoint load per call (step → one model). Measured @70M: `is_faster` rep
  AUC 0.94 on contested vs the provided feature's 0.75 (the model already infers
  speed — not a feature gap).
