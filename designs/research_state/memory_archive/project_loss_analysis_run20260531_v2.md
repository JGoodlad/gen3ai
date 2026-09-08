---
name: project_loss_analysis_run20260531_v2
description: "2nd forensic loss pass (run_20260531_182804, eval 17/18/19M) — confirms prior + adds two NEW retrain levers"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce9b6b8-bd95-47a3-a3e4-4be78176c986
---

Second per-opponent forensic loss analysis of `run_20260531_182804` (eval traces 17M/18M/19M, 9
opponent classes), done 2026-06-02 via a 9-agent-per-opponent + 2-code-verifier workflow. Confirms
[[project_loss_analysis_run20260531]] (harness clean 0/10352; all-or-nothing 247/248 losses are
6-0 wipes; value corr +0.42, over-optimism ~0.5% = setup-illusion only) and re-verified every
root cause against code. Report at `models/run_20260531_182804/LOSS_ANALYSIS_2026-06-02.md`.

**Two findings beyond the prior pass (both retrain-class):**
1. **NEW open INFO gap — no action-aligned DAMAGE-MAGNITUDE / OHKO signal.** The matchup matrices
   are pure type-effectiveness (no base-power×stats / HP); `gen3_move_effects_v1` added flags, not
   magnitude. So the head can't tell "my Surf does 27% vs their EQ 25%" or "this lead gets OHKO'd"
   → underlies can't-close, bad early trades, walking fresh mons into SE KOs. Fix: per-move-slot
   estimated-damage-fraction / OHKO-range bucket (request order, aligns with logit 6+k). NOT closed
   by the shipped P-1.
2. **Explosion-gambling IS partly a reward bug** (prior pass said value/info only). No
   self-explosion *bonus* exists, BUT `finishing_blow=+0.5` fires on our OWN self-KO and
   `faint_ours`/`faint_opp` are symmetric in HP → a 100%-mon Explosion into a weak target nets ~0
   or positive immediate reward; the -30 loss is γ=0.9999-discounted. Fix: suppress/invert
   finishing_blow on high-HP self-faint + add a fixed material penalty to faint_ours.

Also verified: can't-break-a-recovering-wall is REWARD — `futile_attack` (-0.05) only fires when
opp net-gains HP *on the same turn*, so a chip→heal 2-turn cycle never trips it (hp_opp +2.0/bar
paid every chip turn, no windowed net-progress term). Under-switching is POLICY+entropy
(top-1 ~49-55%, ent_coef 0.058) not missing incentive (switch_base/se_switch/escape exist) — but
switch_base is a flat +0.5 for ANY switch (even into an 86% SE hit), so it rewards the act not its
value. Retrain priority: P0 self-play, P1b damage-magnitude obs, P2 reward gating/rescale +
windowed futile + faint material penalty, P3 anneal ent_coef / try γ=0.999.

**STATUS (2026-06-02):** P2's clearest piece IMPLEMENTED in worktree (not yet shipped): in
`reward_manager.py`, `finishing_blow` now suppressed on self-faint, and a flat asymmetric
`FAINT_MATERIAL_PENALTY=0.75` added to `faint_ours` only — a healthy 1-for-1 Explosion trade goes
+0.5 → −0.75; low-for-healthy stays positive. Verified: 145 reward + full non-e2e suite green;
regression fuzz finite/deterministic; `reward_invariants_e2e_test` faint formula/range + finishing
predicate updated. **P1b DEFERRED behind self-play** (user decision): the symmetric/meaningful
expected-damage + uncertainty setup only becomes well-posed once the opponent distribution is the
self-play pool — building it vs the 8 fixed bots would aim at a target self-play replaces and risk
an asymmetric hack. v0 P1b sketch (if revived): 12-dim action-aligned [is_physical, is_special,
our_offensive_stat] per move slot, pre-item so Choice Band stays emergent, no opp priors. See
[[feedback_provide_vs_learn]].
