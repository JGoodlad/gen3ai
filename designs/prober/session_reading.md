# `ProbeSession` model-free reading — `battles` · `scan` · `switch_vs_info` · `battle_turns` · `battle_overview` · `find` · `probe_model` · `decision_table`

Owned by this tree. The facade contract is in `src/main/prober/CLAUDE.md` and the CLI invocations
in `python -m main.prober.query --help`; this file is the per-method reference for the model-free surface —
the tier that works on every run, forever.

- `battles(outcome=, opponent=, step=)` — list/filter battles (each carries an
  `id` + `short_id`).
- `scan(outcome=, opponent=, step=, metric=, limit=)` — **cross-battle, model-free
  turning-point triage**: for every matching battle, its single worst decision
  (`metric="value_drop"`, the most negative ΔV(s→s'), default; or
  `"td_residual"`, the most negative critic surprise), ranked globally. Each row is
  `{id, short_id, opponent, step, outcome, turns, worst:{inv, turn, chosen,
  our_active, opp_active, delta_v, td_residual, reward_total, events, flags,
  incoming_active_pko, incoming_max_pko, incoming_active_outspeed}}`. The
  `incoming_*` fields decode the incoming-damage / OHKO **belief** the obs HELD at
  that cliff (model-free, from the saved obs — see `decode_incoming_belief`): the
  decisive A/B for "did the feature fill the obs gap" — a high `incoming_active_pko`
  at a value cliff means the OHKO WAS in the obs (any remaining error is downstream
  policy/critic usage); a low one where our active then faints means the belief is
  mis-calibrated (an encoder gap). The one-call form of "list losses → overview each
  → rank by the biggest drop" — the usual first move of a loss sweep. No model loaded.
  **Cost:** every row needs its battle's core-trace EXPANSION (a Rust replay, ~0.1-0.2 s each),
  so `value_drop` WITH a `limit` ranks every candidate from its stored `values` and
  `meta.invocations` alone and expands only the rows it returns (rb_x26_s2601, 2026-10-19: a
  run-wide `outcome=loss, limit=25` over 41,345 traces in ~17 s, where the full path took
  hours); a stored count that disagrees with the trace is refused. `td_residual`, or no `limit`,
  still expands every matching battle — filter by `step` on a big run.
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
  shaped, discounted return whose zero is NOT "even", so only this one reads as odds; `None` on a
  run without the head).
  Plus the per-decision DETAIL a
  deeper read wants — the full recorded **`actions`** distribution (`label`/`prob`/`valid`/`chosen`,
  passed through in the recorder's action-index order, NEVER re-sorted — see the move-label gotcha
  below) and the raw Showdown **`protocol`** lines for that turn (parsed ONCE per battle, not once
  per decision). Plus the same `notable` block `battle_overview` returns and a
  `decision_turns[inv] → turn` lookup, so a surface can link "the worst drop" to a turn without
  arithmetic of its own. No checkpoint: **17–20 ms** for
  the longest real battle measured (249 turns, 821 KB). CLI: `query turns <battle_id>`; the
  browser view is `/game`.

  🗑️ **The "did it KNOW?" layer is DELETED** (owner decision 2026-10-10: "delete didn't know").
  It was the distributional head's `value_dist` folded into a per-battle verdict
  (`main/prober/awareness.py`: `awareness_scan`, the `query awareness` command, the `/` card and
  `/api/awareness`, the `knew @` / `lead` scan columns, the `blind` / `median lead` triage
  columns, `battle_turns`' `awareness` block and per-decision `p_loss` / `p_win` / `p_tail` /
  `knew`, the scan rows' `knew_by_turn` / `lead_time` / `blind_loss` / `awareness_text`, the
  triage categories' `awareness` split) — empty on every run since the head was deleted (L1; no
  trace carries `value_dist`), so each surface rendered a placeholder. Pinned absent by
  `session_test.py::test_the_deleted_awareness_fields_do_not_come_back` and
  `web/app_test.py::test_the_deleted_awareness_layer_stays_gone`. The gen-10 figures it was judged
  against (7.2 % blind, median lead 7 turns, coverage80 0.44) stay in the ledger and the
  `designs/research_state/gen11_endofrun_runbook.md` §3 they were measured under; `main.endofrun`
  lost its `awareness` step with it. Recover the code from git history before this commit if a
  distributional head ever returns.
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
  `agents.training.cf_audit` (deleted, P6 slice 6d-1) loaded the audited checkpoint this way (its evidential-head read, `ProbeModel.cf_evidential_batch`,
  returns `None` on every loadable checkpoint now that the head is deleted, deletion pass L4).
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

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### `critic_currency` — what every V on every view MEANS

- `run_summary()` — **orient** (model-free): steps, per-step model identity
  (git/arch/snapshot-available), opponents with win/loss tallies, persisted
  checkpoints, γ, and **`critic_currency`**. The natural first call.

  **`critic_currency` says WHICH READOUT IS THE CRITIC, and therefore what every V on every other
  view MEANS** (`ProbeSession.critic_mode()` / `.critic_currency()`, model-free off the run's
  `model_config.json` `critic` key, cached exactly like `_dist_support`). `{mode, units, low, high,
  even, span, is_probability, default_overvalue_tau, note}` — `shaped` (V is a shaped, discounted
  return of roughly ±30, and its zero is NOT "even": a self-mirror 50/50 reads V≈−6.5) or
  `winprob` (V = sigmoid(win-prob logit) ∈ [0,1], `values` EQUALS `win_probs`,
  G(s) the terminal win indicator at γ=1, and **0.5 really is even**).
  **An ABSENT `critic` key means `shaped`** — a fact about the archive rather than a chosen
  default: the flag landed at config version 109, so every run recorded before it has no key and
  every one of them is shaped (**214 of the 215** on this box, measured 2026-09-06). An unreadable
  config is shaped too, so a failed read can never silently re-scale an old run's numbers.
