---
name: project_anti_stall_fix
description: "Anti-stall reward fix SHIPPED 2026-06-08 (d7aa983 + fuzz guard a11f234): the no-progress penalty was inert all of ai_v5_5 (--no-bias-redesign), stalls are self-play mirror PP-exhaustion; fix = progress-clock heal-war streak cap + winning-residual progress credit + --draw-penalty timeout terminal (config_version 7)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7d3c1d7-0aa2-4875-9183-6e9491acdea9
---

> **Archived 2026-09-08** — ai_v5_5-era reward fix, shipped d7aa983 + a11f234; the mechanism lives in reward_manager.py / progress_clock.py and the win-prob critic (ai_v12) removes hand shaping entirely. Preserved verbatim; nothing below is current.

**The stall root cause (found in the `ai_v5_5_popart_N_0607` review, 2026-06-08).** That run produced
**2247 stall files**, all 250-turn games. Three findings:
1. **The anti-stall penalty was INERT the whole run.** It ran with `--no-bias-redesign`, and
   `reward_manager._apply_progress_clock` early-returns unless `bias_redesign` is True — so the
   `turns_since_progress` no-progress charge fed ONLY the obs scalar (`reactive.py`) and **charged 0
   reward**. The mechanism built to kill these stalls ([[project_markovian_reward_design]]) was disabled.
2. **The stalls are 100% self-play MIRROR PP-exhaustion draws.** Sampled replays were all RLAgent vs
   Gen3Env (frozen self), 100% hit the 250 cap with NO winner, ~93% had ≥1 side reduced to Struggle —
   both bulky-Water/stall teams Recover/Rest forever. Zero training signal.
3. **The cap is a forfeit-LOSS, not a tie.** gen3_env issues `ForfeitBattleOrder` at turn≥250 →
   poke-env `_won=False` → `lost=True` (a tie would be `_won=None`→`lost=None`). The reward already
   scored it −VICTORY_VALUE (−30), so it was NOT a draw-banking incentive; the weak pull is γ=0.9999
   discounting an inevitable −30 from turn ~20 to 250 (≈+0.6 "saving").

Even with `--bias-redesign` ON, two structural defects let it run free: the **DENIED-freeze** (a
productive heal is classified DENIED → clock frozen, so a MUTUAL heal-war never increments on either
side) and the **flat 0.15** non-escalating charge.

**The fix — SHIPPED 2026-06-08, commit `d7aa983`** (+ bridge fuzz guard `a11f234`), 6 files,
1099+14 unit tests green. Three changes:
- **`progress_clock.py` heal-war fix.** Split DENIED into `exogenous` (cant/miss/Protect-block → always
  frozen) vs `heal`; a heal is free for `HEAL_FREEZE_GRACE=2` consecutive windows, then a SUSTAINED
  heal-war charges (n increments + the no-progress penalty engages — obs clock + reward finally register
  the stall). **Part-1 guard:** a new `_is_progress` branch credits an **our-owned residual**
  (Toxic/poison/burn status, or Leech Seed/Curse/Nightmare on the opp active) chipping the opp **net-down**
  as PROGRESS — so a WINNING defensive stall (Recover-while-our-Toxic-ticks) is never taxed. Discriminator
  = opp NET-losing HP (a heal-war where they out-heal the tick still charges). `_is_denied` → `_denial_kind`.
- **`--draw-penalty` timeout terminal** (`reward_manager.py` + `train_rl_agent.py`). Default −30.0
  (byte-unchanged = today). Detected by **turn ≥ `_TIMEOUT_TURN_CAP`** (= `StallConfig().threshold`, since
  the stall ends as a forfeit-loss not a tie), so it's the cap, not won/lost, that triggers it. A decisive
  loss before the cap stays −30. Set `-35` to make a stall-to-cap strictly worse than a clean loss.
- **`model_version.py`**: `draw_penalty` is resume-immutable VALUE-meaning (`check_reward_config`),
  `MODEL_CONFIG_VERSION 6→7`, migrates to −30.0, excluded from the weight-shape check. (Same pattern as
  `switch_bias_weight`.)

**Fuzz guard (`src/agents/training/progress_clock_fuzz_test.py`, bridge, no server).** Drives the REAL
Gen3Env→EpisodeTracker→event-sourced TurnDelta→ProgressClock pipeline and instruments the live clock.
Validated: 40 battles / 3395 windows, **109 winning-residual windows none charged** (the part-1 guard
holds on real data), 911 charged no-op windows (mechanism live), 0 crashes. The heal-war CHARGE path
(streak>grace) is unit-tested only — random play tops out at streak 2 (no sustained heal-wars).

**Note (process):** edits + ship landed in the MAIN checkout, not the worktree (a deviation from
[[feedback_edit_in_worktree_path]]) — shipped cleanly, live launcher run unaffected (pinned/isolated).

**Next run = `--bias-redesign --draw-penalty -35`** (the staged markovian redesign ON + the anti-stall
nudge). Watch: obs `turns_since_progress` should climb in stalls (was pinned ~0); stall-file count/hour
falls; **win-rate vs `staller`/`staller_v2` must NOT drop** (proves the part-1 guard isn't taxing winning
defensive play); heuristic2 Toxic-endgames convert instead of timing out. Pairs with [[project_popart]];
the under-switch / surprise-OHKO obs work is separate ([[project_incoming_damage_outcome]]).
