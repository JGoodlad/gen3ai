# AI v4 — Todo

---

## Step 1 — Self-Play ✓ DONE

Train the agent against frozen copies of itself rather than against fixed heuristics.
Introduces a snapshot pool, win-rate gating, and ELO tracking to prevent strategy collapse
and measure real improvement. See `designs/ai_v4/impl_step1_self_play.md`.

**Known gap — mid-run opponent hot-swap:**  
Currently, pool opponents refresh only at launcher restarts (~every 2.5h). A
`_staged_opponent_path` mechanism in `Gen3Env.reset()` would allow the self-play callback
to swap opponents between episodes without a restart, giving the agent fresher competition
sooner after a new snapshot is promoted. Low priority until pool diversity becomes the
bottleneck.

---

## Step 2 — League Play

Extend self-play into a structured league with dedicated exploiter agents and prioritised
opponent sampling. Exploiters find weaknesses in the Main Agent; the Main Agent must then
generalise past those exploits. See `designs/ai_v4/impl_step2_league_play.md`.
