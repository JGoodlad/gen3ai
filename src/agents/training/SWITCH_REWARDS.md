# Switch Rewards

All switch-related reward signals are consolidated in `reward_manager.py`.
Every signal is a focused function; `_compute_all_switch_bonuses` aggregates them.

---

## High-Level Flow

```
record_action()           ← called BEFORE the turn with the action the model chose
    └─ _pending_subsidy   ← queued, applied at the end of process_turn_reward()

process_turn_reward()     ← called AFTER the turn with the resulting TurnDelta
    └─ _compute_all_switch_bonuses()
           ├─ _compute_pivot_bonus()       (what did the opponent do this turn?)
           │      ├─ _pivot_protect_bonus()
           │      ├─ _pivot_status_bonus()
           │      └─ _pivot_damage_bonus()
           ├─ _compute_se_switch_bonus()   (does our new mon threaten the opponent?)
           ├─ _compute_sleep_out_bonus()   (did we rotate out a sleeping mon?)
           └─ _compute_sleep_in_penalty()  (did we send in a sleeping mon?)
```

Pivot, SE, and sleep-out bonuses are **skipped** when `_last_switch_was_roared` is
True — the agent was phazed (Roar/Whirlwind), so there was no strategic choice to reward.

---

## Switch Subsidy (`record_action`)

Applied to every voluntary switch. Set once in `record_action` before the turn and
consumed at the end of `process_turn_reward`.

| Condition | Value |
|---|---|
| Voluntary switch, not back-to-back | `+SWITCH_BASE_BONUS` (0.50) |
| Voluntary switch, consecutive turns | `0.0` (spam multiplier = 0) |
| Switched to the species we just switched away from | `−0.15` (bouncing tax) |
| Roar / Whirlwind (forced while still alive) | `0.0` |
| Post-faint replacement (forced) | `0.0` |

The subsidy exists to compensate for the credit assignment gap: the strategic
benefit of a good switch lands on future turns (next opponent attack, spikes chip),
but the cost (HP, momentum) is often immediate.

---

## Pivot Bonus (`_compute_pivot_bonus`)

Rewarded when we switch AND the opponent did something specific.
Dispatches to one of three sub-signals based on what `delta.opp_move_id` tells us:

### Protect / Detect / Endure (`_pivot_protect_bonus`)

```
opp_move_id in {"protect", "detect", "endure"}
→ +PROTECT_SWITCH_BONUS (0.10)
```

The opponent burned their turn. We repositioned for free.

### Status Move Immunity (`_pivot_status_bonus`)

```
opp_move.base_power == 0
→ check _STATUS_MOVE_IMMUNITY[opp_move_id] against our new mon's types
  OR our new mon already has a status condition
→ +STATUS_IMMUNE_SWITCH_BONUS (0.10)
```

Hardcoded immunity map (Gen 3 rules):

| Move | Immune Types |
|---|---|
| Thunder Wave | Ground |
| Toxic, Poison Gas, Poison Powder | Steel, Poison |
| Will-O-Wisp | Fire |
| Spore, Sleep Powder, Hypnosis, Lovely Kiss, Yawn | *(no type immunity)* |
| Stun Spore | *(Normal-type in Gen 3 — no type immunity)* |
| Glare | *(Normal-type — no type immunity)* |

Already-statused check covers the case where we switch in a mon that has PAR/BRN/PSN
from a previous encounter — the status move still fails, so the bonus fires.

### Damaging Move Comparison (`_pivot_damage_bonus`) — Signal A

```
opp_move.base_power > 0
→ compare type effectiveness vs old mon vs new mon
```

| Outcome | Bonus |
|---|---|
| New mon resists the move (mult_vs_new < mult_vs_old) | `+0.10` |
| New mon is fully immune (mult_vs_new == 0) | `+0.15` |
| Same or worse effectiveness | `0.0` |

Uses the **actual move the opponent played**, not their types. This avoids rewarding
switches against mons that don't run their STAB (common in Gen 3 OU).

The `compute_base_reward` already captures the raw HP delta, so this signal rewards
*improved matchup*, not the damage outcome itself.

---

## SE Switch Bonus (`_compute_se_switch_bonus`)

Rewards switching in a mon that threatens the opponent offensively.

1. **Confirmed**: check our new mon's revealed moves — if any deal ≥2× to the opp
   active, fire `+SE_SWITCH_BONUS` (0.20).
2. **Fallback**: if no moves are revealed yet, check our new mon's own types as a STAB
   proxy. Fires `+SE_SWITCH_BONUS` if any type is SE vs the opp active.

The fallback avoids a cold-start problem where the agent gets no credit on turn 1
before any moves are revealed.

---

## Sleep-Out Bonus (`_compute_sleep_out_bonus`)

```
Voluntary switch + the mon we switched away from has Status.SLP
→ +SLEEP_SWAP_BONUS (0.25)
```

Rotates a sleeping mon to the bench. Sleeping mons can't attack, but they still
occupy a slot and may wake up later. Proactively benching them to bring in an
active attacker is correct play.

Only fires on **voluntary** switches — post-faint replacements don't qualify.

---

## Sleep-In Penalty (`_compute_sleep_in_penalty`)

```
Our active mon (after the switch) has Status.SLP
→ −SLEEP_SWAP_BONUS (−0.25)
```

Penalises sending in a sleeping mon, whether the switch was voluntary or post-faint.
Does **not** fire when we were roared — the phazer chose our slot, not us.

---

## Signal Guard: Roar / Whirlwind

When `_last_switch_was_roared = True` (set in `record_action` when
`ctx.phase == "forced_switch"` and our active mon was still alive):

- Pivot bonus → skipped
- SE switch bonus → skipped
- Sleep-out bonus → skipped
- Sleep-in penalty → still fires (we still sent in the replacement)
- Switch subsidy → `0.0` (not a voluntary switch)
