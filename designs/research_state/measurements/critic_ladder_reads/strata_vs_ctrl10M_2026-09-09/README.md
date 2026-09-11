# Critic ladder read — `ai_v12_17_ladder_strata` (`--win-prob-strata-weight 1.0`, pin f871e79f) vs control `ai_v12_11_ladder_ctrl10M`, both pinned to `step_10000032`

`python -m main.ops.critic_read ai_v12_17_ladder_strata --control ai_v12_11_ladder_ctrl10M --step 10000032 --floor-json <replicate_floor_10M.json> --out <job tmp>/read_strata`, tool v3 (quota-matched: arm caps 40/40 → 8/12/2 over 21 seeds for the frame-sensitive rows; the live 100-game floor applied). Cross-commit (`f3502568→f871e79f`, both increments verified NEUTRAL). Realized dose (Training Run session, 59 weighted rollouts): exact class parity on 15 (share_bot 0.5000), capped on 44 (mean 0.464, range 0.436–0.500), overall 0.473 against a 0.5 target; the clamp first binds at 4,620,288 and on 75 % of weighted rollouts after. Verdict: ledger 2026-09-09 · *READ · critic ladder arm strata*. The tool prints WITHIN FLOOR for a delta whose |Δ| is inside the floor even when its CI covers zero — read those rows as NOT DETECTED either way.

**400-game OFFLINE read (2026-09-09):** `critic_read_hp400.md/json` — both sides 400-game offline-generated cycles, floored by `replicate_floor_10M_hp400.json`; NOT comparable to the 100-game rows above; ledger 2026-09-09 · *THE ARMS AT 400 GAMES* (addendum).

**800-game OFFLINE read (2026-09-10):** `critic_read_hp800.md/json` (seed 20260910; its own floor `../replicate_floor_10M_hp800.json`); ledger 2026-09-10 · *THE ARMS AT 800 GAMES* (addendum).

**Second 800-game eval draw (2026-09-10, seed 20260911):** `critic_read_hp800b.md/json`, floors from `../replicate_floor_10M_hp800b.json`; the bar is the WIDER of the two offline draws (rule 19). Ledger 2026-09-10 · *second eval draw for tdaux and strata*.
