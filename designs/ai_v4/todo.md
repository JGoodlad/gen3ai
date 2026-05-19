# AI v4 — Todo

---

## Step 1 — Self-Play ✓ DONE

Train the agent against frozen copies of itself rather than against fixed heuristics.
Introduces a snapshot pool, win-rate gating, and ELO tracking to prevent strategy collapse
and measure real improvement. See `designs/ai_v4/impl_step1_self_play.md`.

---

## Step 2 — League Play

Extend self-play into a structured league with dedicated exploiter agents and prioritised
opponent sampling. Exploiters find weaknesses in the Main Agent; the Main Agent must then
generalise past those exploits. See `designs/ai_v4/impl_step2_league_play.md`.

**Design questions to resolve:**
- **Single-process vs. multi-process**: time-multiplex Main Agent and exploiters in one
  training process (simpler) or run parallel processes writing to a shared snapshot pool?
- **Exploiter reset policy**: reset when exploiter win rate > 70% against Main, or on a
  fixed schedule? Adaptive reset tracks exploitation convergence more precisely.
- **PFSP floor**: minimum sampling probability for opponents the Main Agent already
  dominates. Too high → wasted training on trivial opponents. Too low → those strategies
  reappear in exploiters and catch the agent off-guard.
