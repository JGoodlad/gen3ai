# Research State


> **Measured numbers live in [`measurements/`](measurements/)** — the raw audit outputs, each
> carrying its own checkpoint, step, state count and date. Cite a file there rather than
> re-typing a percentage: that is what makes a stale number detectable by comparing dates
> instead of by re-deriving the result.

The **single source of truth for what we're trying, what we know, what we don't, and what's left
to find.** Version-agnostic (it tracks the ongoing hunt, not one `ai_vN`). Maintained deliberately
by agents — see the protocol below and the `feedback-research-state` memory.

> The whole reason this exists: this project keeps forming plausible hypotheses, and most die under
> scrutiny (8 killed in one session). That knowledge is the asset. A finding that isn't written here
> evaporates and gets re-discovered (or re-believed) next session. **Write the kills, the
> not-knowns, and the pros/cons — not just the wins.**

## Layout

- **[ledger.md](ledger.md)** — the at-a-glance status TABLE (every hypothesis: ✅ confirmed / ❌ killed
  / 🔬 open, mechanism, evidence, re-verify command). The dashboard you glance at first.
- **[levers/](levers/)** — one file per OPEN or ACTIVE lever, each with the full
  **Known / Not-known / Pros / Cons / Status / Next-test** structure (`levers/_template.md`). Killed
  levers don't get a file — their one-line cause-of-death lives in the ledger row.
- **[The frontier](#the-frontier--what-else-might-be-there)** (below) — the standing list of candidate
  levers NOT yet (fully) investigated. This is the working surface for "there has to be more."

## Maintenance protocol (what the memory enforces)

After **any** investigation that changes our belief, update this folder in the same pass:
1. **Update the lever file** (or create it from `_template.md` when a frontier item becomes active):
   move facts into **Known**, open questions into **Not-known**, the upside into **Pros**, the caveats
   into **Cons**. Be honest — a confirmed Con (e.g. "pervasive in wins too") is as valuable as a Pro.
2. **Update the ledger row** — status, the load-bearing number, the re-verify command.
3. **Tend the frontier** — add any new candidate lever surfaced; mark one investigated/ruled-out.
4. **Apply the honesty gates** before promoting a finding to Known (see ledger.md → method): is it
   outcome-conditioning? falsifier-myopia? legitimate-in-context? exploration vs learned? Always
   adversarially verify a *confirming* measurement (we were overturned 3-for-3 by careful rechecks).

## The amortizability gate — route every oracle finding through L1–L4

The falsifier is an **ORACLE** (hindsight + re-rolls), so *"a better action existed"* is its DEFAULT
output and *"needs search/MCTS"* is the degenerate conclusion it is **built to reach** — concluding it
proves only that the oracle is an oracle. A finding is a **lever** only if the improvement is
**AMORTIZABLE** into the single-forward-pass network. Route every oracle-found improvement through:

- **L1 obs-info** — the deciding info was ABSENT from the obs but is known/addable → **obs feature** (cheap).
- **L2 representation** — info WAS in the obs; the net mis-used / mis-valued it → **arch / training /
  value-target**. *Test:* linear-probe the trunk (`probe`); recoverable ⇒ L2 (head fault), not ⇒ L1.
- **L3 anticipation** — needed the opponent's TYPICAL reply → **opponent-action / forward-model aux
  head** (amortized anticipation, NOT runtime search). *Test:* was the opp move predictable from
  history/priors, or idiosyncratic/hidden?
- **L4 deliberation** — genuinely needs per-state multi-ply AND the opp move is not anticipatable → the
  ONLY "search" bucket, and it ships as an **OFFLINE TEACHER distilled into the net** (AlphaZero /
  Expert Iteration), never as runtime compute.

**HARD CONSTRAINT (owner):** no lever may put search/MCTS **on the model** (inference or the training
loop). Search is a *teacher / offline diagnostic* only. *"Cheap vein mined out → MCTS"* is not a default
— it is a falsifiable claim about the SIZE of L4 that must beat the measured L1+L2+L3 mass.

## Decision posture — the build bar (say YES)

Falsification is a **means** (don't waste a retrain on a dud), not the goal (a better model). The
**honesty gates govern what we CLAIM TO KNOW; they are NOT the bar for what we BUILD.** The census says
there is no single remaining big lever (confirmed blunders explain a few points; the rest is diffuse
grind) → the EV-optimal strategy SHIFTS from *explore* (hunt an elephant) to **EXPLOIT** (build the
portfolio of moderate, amortizable gains and measure the aggregate). Progress here is a long tail of
small, compounding, stacked wins — not one breakthrough.

A lever is **GO-TO-BUILD** when ALL hold (this bar is deliberately LOWER than "Known"):
1. **Amortizable** — passes the L1–L4 gate (a feedforward change, or an offline teacher, can capture it).
2. **Positive EV** — a plausible mechanism + non-trivial headroom (≥~1% wr or a clear behavioural fix);
   it need NOT be proven dominant.
3. **Falsifiable-after-build** — we name the metric that must move in the retrain, so a no-op is cheap
   to detect and abandon.
4. **Bounded cost** — obs feature / reward term / aux head, not a moonshot.

**Portfolio rule:** because no single lever fixes the gap, **STACK** the GO-TO-BUILD levers into the
next FRESH run (the resume-immutable boundary several already require) and measure the aggregate, rather
than A/B-ing each forever. *Honest tradeoff:* stacking **confounds attribution**. Mitigate by stacking
only mechanistically-INDEPENDENT, cheap levers, each with its own offline pre-retrain proxy; isolate a
lever only when it is expensive or risky. The live GO-TO-BUILD queue is in [ledger.md](ledger.md).

## The frontier — what else might be there

Ranked by where the *unexplained loss mass* most plausibly lives, with the honest status. The census
finding to keep front-of-mind: **the confirmed blunders (self-KO, attack-mismatch) explain only a few
points of the ~18% bot-loss gap. The dominant remaining mass is NOT more blunders.**

| Candidate lever | Why it might be there | Status |
|---|---|---|
| **Strong-opponent positional grind** | The census's biggest finding: ext/pool losses are LOW-variance NEUTRAL grinds (dice std ~0.01), not blunders or surprise-OHKOs. We have **no mechanism** for this mass yet — it's the least-understood and likely largest lever. | 🔲 **UNEXPLORED** (the priority) |
| **The EARLIER decision (multi-ply)** | Surprise-deaths + recovery/setup "blunders" were repeatedly found to be *committed by the lethal turn* — the real mistake is 2–3 turns upstream. We've only ever anchored on the death/crater turn. | 🔲 unexplored — anchor falsify earlier; trace the causal chain |
| **Forward-model / opponent-action head** | Predict opp action (attack/switch/which-move). | ❌ **FALSIFIED 2 ways (2026-06-12)** — (a) VoI oracle: perfectly knowing the opp action is worth ~0.03 mon, loss≈win ([[project_l3_oracle_grind_l4]]); (b) representation probe: the trunk+POLICY-HEAD already model the opponent — opp_switches 0.89/0.90, move-TYPE 0.93/0.96, switch-TARGET 0.88, big-hit 0.75 — so an aux head re-encodes present info & can't fix a decision problem. DON'T build. `project_opp_action_head_falsified` |
| **Surprise-OHKO belief coverage (H3)** | Belief under-fires on 52% of lethal healthy deaths. Recoverability now RESOLVED: **42% avoidable** (64% switch-to-a-wall), 33% LUCK (belief was right), 25% committed. | 🔬 **GO (gated)** — see [levers/surprise_ohko_coverage.md](levers/surprise_ohko_coverage.md); pair with under-switching |
| **Opponent-PP observability (the stall-class blind spot, C3)** | The critic is over-confident precisely on stall losses (win_prob 0.7–0.98 vs resampled-dice 0.0–0.4) while LEVEL-calibrated elsewhere (C2) — and the opponent's PP ledger, the quantity that decides a Gen-3 stall war, is encoded as ALWAYS FULL in the obs (`moves.py:129-130`; no usage tracker exists). The win-prob head is MC-supervised, so the miss is missing INPUT or distribution, not objective. PP is public info → a tracker is raw known facts, not a prior. A candidate mechanism for a slice of the positional-grind row above (the PP-war grind is a positional grind). | 🔬 **probe PRE-REGISTERED 2026-08-17** (`gen13_endofrun_runbook.md` §8, ledger C3) — offline on gen-12 traces, running now → [levers/opp_pp_observability.md](levers/opp_pp_observability.md) |
| **Hidden-team belief (the un-falsified gap)** | The opp-ACTION head was FALSIFIED, but it named a SEPARATE real gap: the opponent's HIDDEN team. Gen3 has no team preview → the ~3 unrevealed slots are ABSENT from the obs (a probe CAN'T recover them). Pre-build learnability: revealed→hidden beats the usage prior **+7pp recall / +8–10pp top-1**. Subsumes the bench/switch-in half of H3. | 🛠 **BUILT 2026-06-13, NOT RUN** (`claude/belief-head`, config 16, `--opp-belief-aux-coef`): in-place unknown-mon slot tokens + Hungarian species+moves aux head; learns immediately; robust for self-play+stable+distill+resume; fuzz-validated. **UNMEASURED** if it helps the policy → [levers/hidden_team_belief.md](levers/hidden_team_belief.md). Next = fresh-run A/B (crater share + wr). `project_hidden_team_belief_built` |
| **Under-switching = policy COMMITMENT lever** | Anchored to strong humans (faithful human-agreement probe): humans switch ~30% of voluntary decisions, our argmax ~16%. **But REFRAMED 2026-06-12: NOT representation/valuation — the policy's SOFT switch-prob (0.28) already ≈ human (0.30); the gap is ARGMAX/commitment** (it assigns ~human switch mass but under-commits). So the knob is argmax sharpness — `--switch-bias-weight` / `ent-coef` / temperature — no arch build. ("info-blind switching" was REFUTED: the corr was a progress-confounded null; double-switch 57% proves info-conditioning.) | 🔬 **reframed to commitment** — validate `--switch-bias-weight` vs a confound-controlled within-state target; commit the off-tree human-agreement tooling first. `project_opp_action_head_falsified` |
| **Attack type-mismatch obs feature (H2)** | Confident resisted/immune picks; small (~fraction of a %). Cheap obs effectiveness feature. | ✅ confirmed small — [levers/attack_type_mismatch.md](levers/attack_type_mismatch.md) |
| **Team / matchup draw quality** | Some losses may be bad team draws, not policy errors. The team-pool weighting (yak_attack) was a data-dist bug. | 🟡 partially addressed |
| **Decision-time search / MCTS** | Highest ceiling (Wang2024, rank-8 Elo). The amortized levers (forward-model) are the search-free cousins. | ⛔ user RULED OUT (kept as the ceiling-setter) |
| **Human ladder data (ai_v6 BC / treadmill break)** | Self-play only explores its own convex hull; ~102k ladder replays are an EXTERNAL distribution — a candidate mechanism for the positional-grind lever + the "self-play treadmill" ceiling. **FAITHFUL human-agreement probe BUILT 2026-06-12** (`src/agents/bc/log_reader.py` turn-1 full-team injection makes the acting side's obs/mask faithful; `src/main/human_agreement.py`): overall ≈35% match vs strong humans (≈1-in-3; low ≠ bad — two strong humans also disagree; agreement = behavioural DISTANCE). **Headline finding: the policy UNDER-SWITCHES ~2× vs strong humans (16% vs 30%)** — this OVERTURNED v0's "modest 16v19" (an artifact of 43% of human switches being unrepresentable pre-fidelity-fix). The faithful `log_reader` is the reusable FOUNDATION for the non-search roadmap (BC / team-completion / offline eval). | 🔬 **faithful, finding stands** — next: close the switch-rate gap via the COMMITMENT lever (the opp-action head was FALSIFIED, not a path here — see the Forward-model row). Memory `project_human_agreement_probe` |

| **De-amortization: per-team conditioning** | The generalist plays every team with one averaged policy; a specialist is far stronger (0.438 → 0.72 piloting). The exploiter→distill loop CLOSES it (**D1**) and the skill STICKS without teachers (**D2**, ~76%). | ✅ **RESOLVED 2026-07-28 (D4).** The scaling limit is **COUNT (+0.077 SIG)**, not conditioning (+0.028, marginal, and mechanistically a SHARED modulation — so LoRA/MoE is ruled out). N=10 generalizes to diverse teams ⇒ **run N≤10 exploiters and distil**; stop trying to raise N. `designs/ai_v8/design_conditioning_ceiling_arms.md` |

| **Measurement/infra correctness** (are our experiments running the model we think they are?) | M1 found that SB3 silently destroyed every zero-init in the extractor, so no "identity-at-init" toggle ever actually started at identity — a confound that sat under K10 and the whole D4 conditioning line, undetected, because every test builds the module directly instead of through a policy. | ✅ **one found + FIXED 2026-08-01** (ledger M1). **The lesson generalises: an invariant asserted only in a unit test that bypasses the production construction path is not an invariant.** Cheap sweep worth doing: for each documented "byte-identical / identity-at-init / cold-start == prior" claim, assert it on a REAL `MaskablePPO`-built policy. |

**The honest steer:** "more" most likely lives in the **strong-opponent positional grind** (unexplored,
largest, no mechanism) and the **upstream/multi-ply** reframe (the deaths we've been studying are
symptoms, not causes). The blunder-hunting vein is largely mined out. **Since 2026-07 the active vein is de-amortization** (the last row) — the one lever with a measured
gap, a working mechanism, and a confirmed *durable* payoff. Its scaling question is now ANSWERED: the
limit is team COUNT, not conditioning, so the loop runs at N≤10 and repeats. The open questions are
where the count cliff sits and whether the small diversity cost compounds across batches.
