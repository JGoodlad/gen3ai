# `ProbeSession` model-free reading — `battles` · `scan` · `switch_vs_info` · `battle_turns` · `awareness_scan` · `battle_overview` · `find` · `probe_model` · `decision_table`

Owned by this tree. The facade contract and the CLI invocations stay in
`src/main/prober/CLAUDE.md`; this file is the per-method reference for the model-free surface —
the tier that works on every run, forever.

- `battles(outcome=, opponent=, step=)` — list/filter battles (each carries an
  `id` + `short_id`).
- `scan(outcome=, opponent=, step=, metric=, limit=)` — **cross-battle, model-free
  turning-point triage**: for every matching battle, its single worst decision
  (`metric="value_drop"`, the most negative ΔV(s→s'), default; or
  `"td_residual"`, the most negative critic surprise), ranked globally. Each row is
  `{id, short_id, opponent, step, outcome, turns,`
  **`knew_by_turn, lead_time, blind_loss, awareness_text`** (the battle's "did it KNOW?" verdict
  beside the decision that lost it — a crater the model never saw coming is a missed signal; the
  same crater with 20 turns of warning is a position it could not convert)`, worst:{inv, turn, chosen,
  our_active, opp_active, delta_v, td_residual, reward_total, events, flags,
  incoming_active_pko, incoming_max_pko, incoming_active_outspeed}}`. The
  `incoming_*` fields decode the incoming-damage / OHKO **belief** the obs HELD at
  that cliff (model-free, from the saved obs — see `decode_incoming_belief`): the
  decisive A/B for "did the feature fill the obs gap" — a high `incoming_active_pko`
  at a value cliff means the OHKO WAS in the obs (any remaining error is downstream
  policy/critic usage); a low one where our active then faints means the belief is
  mis-calibrated (an encoder gap). The one-call form of "list losses → overview each
  → rank by the biggest drop" — the usual first move of a loss sweep. No model loaded.
- `switch_vs_info(step=, opponent=, outcome=, max_battles=)` — **model-free behavioural
  probe** of OUR policy: voluntary switch-rate bucketed by how many opponent mons we'd
  revealed (the information level), a correlation, and the double-switch rate (switch the
  turn AFTER the opp switched; consecutive own switches). Tests "do we switch more when we
  know less" (negative correlation ⇒ information-sensitive). Measured @53M: corr −0.025
  (information-BLIND), 57% reactive switch-after-opp-switch. Caveat: revealed-count
  correlates with game progress — a confound to control for.
- `battle_turns(battle_id)` — **model-free TURN-BY-TURN replay**: the same trace read as a GAME
  rather than as a ranked table. Decisions grouped by **game turn** (a turn is not a decision — a
  faint puts the `move_selection` and the `forced_switch` it caused on the same turn), each row
  carrying the **board** it was made on (`our`/`opp`: species · `hp` + numeric `hp_pct` · status ·
  boosts · item · moves · revealed bench), what was **chosen** + its recorded probability, the
  ordered **`timeline`** of what then happened (`engine.build_result_timeline` entries, each with a
  `text` field rendered by `engine.timeline_entry_text` so no surface re-derives the sentence),
  `order_certain` (false ⇒ both sides moved, `move_order` wasn't recorded, AND the turn's protocol
  slice could not settle it either — so top-to-bottom is NOT the real sequence; say so, never
  guess. The protocol read takes this from 56.3% of decisions to 3.6% — see the timeline section), **`opp_intent`** (the v67 `α`/`β` read — `alpha`
  ranked NAMED options + `SWITCH`, `beta` the named candidate switch-ins, `top`, `switch_p`, and a
  `text` rendered by `engine.opp_intent_text` so no surface re-derives the sentence; `None` on a run
  without the heads, which is every trace before v67 — plus **`actual`**, what the opponent then
  DID, with the matching `alpha` option flagged `was_actual` and `actual_unlisted` when `α` never
  named it at all. A prediction is only readable beside its outcome, and the match needs
  normalizing (`α` holds display names, the recorder an id) plus a Hidden-Power rule, so it is done
  once here rather than in each surface), and the critic's read (`value` · `delta_v` ·
  `td_residual` · `reward_total` + components · `events` · `flags`) — now with the two readings the
  scalar V cannot supply: the win-prob head's calibrated **`win_prob`/`delta_win_prob`** (V is a
  shaped, discounted return whose zero is NOT "even", so only this one reads as odds) and the
  distributional head's **`p_loss`/`p_win`/`p_tail`/`knew`** (the awareness fold, joined by decision
  index; `knew` is true from the sustained onset onward). `p_win` is `1 - p_loss` carried on the
  payload rather than left to each surface to flip — a view computing `1 - x` is a view deriving a
  number — and it is what `/battle` renders, so one card reads in ONE direction. Every threshold
  stays defined on `p_loss`. All `None`/`False` on a run without those heads.
  Plus the per-decision DETAIL a
  deeper read wants — the full recorded **`actions`** distribution (`label`/`prob`/`valid`/`chosen`,
  passed through in the recorder's action-index order, NEVER re-sorted — see the move-label gotcha
  below) and the raw Showdown **`protocol`** lines for that turn (parsed ONCE per battle, not once
  per decision). Plus the same `notable` block `battle_overview` returns and a
  `decision_turns[inv] → turn` lookup, so a surface can link "the worst drop" to a turn without
  arithmetic of its own. Also an **`awareness`** block (see `awareness_scan` below — the same
  verdict for THIS battle; `None` on a run without a dist head). No checkpoint: **17–20 ms** for
  the longest real battle measured (249 turns, 821 KB). CLI: `query turns <battle_id>`; the
  browser view is `/battle`.
- `awareness_scan(outcome="loss", opponent=, step=, lead_bar=5, cap_turn=240, stall_bar=0.25)` —
  **model-free 'did it KNOW?'**: the distributional head's battle-level loss-awareness verdicts
  (`main/prober/awareness.py`, pure + unit-tested) over every matching battle, aggregated. Per
  battle: `knew_by_turn` (first game turn from which P(loss) > 0.5 holds SUSTAINED to the end),
  `lead_time`, `blind_loss` (a loss it never saw coming), and `mean_tail_divergence` — the STALL
  SIGNATURE (bottom-atom mass piling up while the distribution MEAN still reads positive; the
  exact shape a scalar critic cannot surface, and the gen-9 pathology: positive V the turn
  before a cap loss). Aggregate: `blind_loss_fraction`, `aware_ge_bar_fraction`,
  **`cap_aware_ge_bar_fraction`** (the runbook's deadline-clock regression readout: fraction of
  cap losses tail-aware ≥ `lead_bar` turns early), `median_lead_time`,
  `stall_signature_fraction`. The atom support is read model-free from the run root's
  `model_config.json` (`value_dist_vmin/vmax/bins`), and the PopArt denorm is FIT per battle
  from the trace's own `(dist mean, recorded scalar V)` pairs (`fit_denorm` — exact under
  `value_from_dist`, an adequate approximation under `shaping`; identity without PopArt), so it
  runs on any dist-head run regardless of architecture drift. Battles ranked blind-first then by
  divergence. The aggregate also carries **`quantile_coverage`** (runbook §3): the pooled
  mid-PIT of the realized MC return under each predicted distribution (`coverage_stats` —
  continuity-corrected, so a perfectly-centered prediction reads 0.5, not 0.5+half an atom) —
  calibrated ⟺ `pit_mean` ≈ 0.5 and `coverage80` ≈ 0.80. ⚠️ Selection caveat: the default
  `outcome="loss"` filter biases PIT low BY CONSTRUCTION; judge calibration on `outcome=None`
  and use the filtered read for direction only. G is the MC discounted-reward return (the
  calibration probe's convention) — an approximation of the bootstrapped training target.
  Measured on gen-10 (1396 losses): 7.2% blind, median lead 7 turns, 12 cap losses of which
  only 50% were aware ≥5 turns early — the top-ranked row is a turn-249 cap loss with
  P(loss)=0.15 at its FINAL decision. Coverage (ALL outcomes, 109k decisions): pit_mean 0.396,
  **coverage80 0.44 vs nominal 0.80** — the head is optimistic AND over-confident (narrow);
  losses-only pit_mean 0.085. CLI: `query awareness <run_dir>` — **`--outcome all` is the
  unfiltered read** the coverage baseline is comparable to (win/loss were the only choices, which
  made the probe's own "judge calibration unfiltered" caveat name a reading the CLI could not
  produce). Those baselines are **not prose**: they live in `awareness.AWARENESS_BASELINES` and
  ride the result as `aggregate.baseline`, so the CLI and the web quote ONE reference point and
  neither keeps a copy. The result also carries `caveats` (baseline provenance · the cap-loss
  small-n · the selection bias · counted-not-judged).
  **The verdict's SENTENCE is `engine.awareness_text`** — same rule as `timeline`'s `text`: a
  surface prints it, never re-derives it. Each folded row also records the **DECISION index** it
  came from (`decisions`, parallel to `turns`/`p_loss`), because a faint puts two decisions on ONE
  game turn and a turn-keyed join silently collapses them.

  **The verdict is carried by the views that need it, not only by its own command:**
  `battle_turns` puts `p_loss`/`p_tail`/`knew` on every decision row (beside the battle-level
  `awareness` block); `scan` rows carry `knew_by_turn`/`lead_time`/`blind_loss`/`awareness_text`
  beside the crater, because one value cliff means opposite things with and without warning;
  `triage` gives each category an `awareness` split (`n_judged`/`n_blind`/`blind_fraction`/
  `median_lead_time`) **reported BESIDE the taxonomy, never folded into it** — the category names
  which lever to pull, the split says whether the model had warning to act on. Measured on
  gen-11: `positional_grind` median lead **22** turns vs `attrition_death` **4** — a slow known
  decline against a death that arrives fast. All `None` on a run with no dist head (never `0.0`,
  never `False`: absent must not read as measured).
- `battle_overview(battle_id)` — **model-free digest**: per-decision rows
  (chosen, top prob, `our_active`/`opp_active` board summary, recorded V(s), **ΔV**,
  **TD residual** = critic surprise, reward total, events, flags) + a `notable`
  block (faints, switches, `biggest_value_drops`) + how a deep analyze resolves.
  Full board state (`board`: both sides' active + bench + our moves) and `field`
  (weather/spikes/screens, when captured) are in `analyze`'s output.
- `find(battle_id, criterion, limit=)` — ranked/listed invocations:
  `switch`/`uncertain`/`faint` (flags, model-free), `value_drop`/`low_value`/
  `high_value` (ranked by recorded V, model-free), or `disagree` (loads the
  model; chosen ≠ the model's argmax).
- `probe_model(battle_id)` — `(ProbeModel, ModelChoice)`, the **public face of the exact→nearest→recent
  resolution ladder** (the same cached `_model_for` every analysis uses). For out-of-package readers
  that need the loaded network rather than one of the analyses over it —
  `agents.training.cf_audit` reads the evidential Beta head off the audited checkpoint this way.
  Every battle in one `step_N` trace dir resolves to the same checkpoint, so a caller that resolves
  once and reuses pays for one load.

## `decision_table` — the per-decision forensic table

- `decision_table(steps=, opponents=, outcomes=, categories=, max_battles=)` — a complementary
  MODEL-FREE per-decision FORENSIC TABLE (`forensics.py`): one row per captured decision with `cat`
  (`move_category`: selfko/recovery/**cure**/setup/stall/status/switch/attack_or_other — `cure` is
  its own bucket because clearing status, healing HP and inflicting status are three different acts,
  and it is derived from the move data, not a hardcoded set), our/opp species+HP,
  policy `conf` (`softmax(logits)[chosen]` — learned vs exploration-tail), `reward`, critic `dV`
  (`V[i+1]−V[i]`, the self-KO over-valuation signal), incoming-KO `pko` belief, faint flags, outcome.
  The single source for the softmax/dV/`decode_incoming_belief` plumbing every behavioural-hypothesis
  check reuses (the shipped self-KO finding used the `selfko` `dV_med`). Distinct from `falsify_scan`
  (the luck/mistake bracket) — this is the raw per-decision table. `move_category` /
  `decision_table_digest` are pure (unit-tested). Each row also carries `our_status` / `cure_avail` /
  `cure_prob` / `chose_cure`, and the digest a **`cure_uptake`** block — over the decisions where a
  status cure was genuinely available (statused AND legal), how often the policy took it, the median
  probability it put there, and what it did `instead`. That is the run-level form of the `cure-skipped`
  flag; pair it with the `cure` category count, since a cure chosen with **no** `cure_avail` is a
  WASTED self-cure (the same NO_OP `progress_clock._is_wasted_self_cure` charges). Measured on
  `ai_v9_09 @16M`: uptake **32/474 = 6.8%** (median P(cure) 0.036, `instead` led by `recover` ×127)
  while **104 of 136** cure uses (76%) had nothing to cure — the policy is picking these moves close
  to independently of whether it is statused. See `designs/research_state/` for the hypothesis ledger.

