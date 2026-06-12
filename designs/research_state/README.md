# Research State

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

## The frontier — what else might be there

Ranked by where the *unexplained loss mass* most plausibly lives, with the honest status. The census
finding to keep front-of-mind: **the confirmed blunders (self-KO, attack-mismatch) explain only a few
points of the ~18% bot-loss gap. The dominant remaining mass is NOT more blunders.**

| Candidate lever | Why it might be there | Status |
|---|---|---|
| **Strong-opponent positional grind** | The census's biggest finding: ext/pool losses are LOW-variance NEUTRAL grinds (dice std ~0.01), not blunders or surprise-OHKOs. We have **no mechanism** for this mass yet — it's the least-understood and likely largest lever. | 🔲 **UNEXPLORED** (the priority) |
| **The EARLIER decision (multi-ply)** | Surprise-deaths + recovery/setup "blunders" were repeatedly found to be *committed by the lethal turn* — the real mistake is 2–3 turns upstream. We've only ever anchored on the death/crater turn. | 🔲 unexplored — anchor falsify earlier; trace the causal chain |
| **Forward-model / opponent-action head** | Predict opp action (attack/switch/which-move). H3's surprise-OHKO is partly opp-action uncertainty; it's also the prerequisite for any lookahead. Oracle test was *designed* (all-legal best-response w/ known opp action) but not run. | 🟡 designed, not run |
| **Surprise-OHKO belief coverage (H3)** | Belief under-fires on 52% of lethal healthy deaths. Fixable share (priors-pricing + calibration) ~64%. | 🔬 OPEN — see [levers/surprise_ohko_coverage.md](levers/surprise_ohko_coverage.md); recoverability pending |
| **Under-switching policy lever** | Belief fires but the policy doesn't act on it (known prior finding). Even a perfect H3 belief needs the policy to switch. `--switch-bias-weight` exists but is nuanced/under-validated. | 🟡 partially explored (prior runs) |
| **Attack type-mismatch obs feature (H2)** | Confident resisted/immune picks; small (~fraction of a %). Cheap obs effectiveness feature. | ✅ confirmed small — [levers/attack_type_mismatch.md](levers/attack_type_mismatch.md) |
| **Team / matchup draw quality** | Some losses may be bad team draws, not policy errors. The team-pool weighting (yak_attack) was a data-dist bug. | 🟡 partially addressed |
| **Decision-time search / MCTS** | Highest ceiling (Wang2024, rank-8 Elo). The amortized levers (forward-model) are the search-free cousins. | ⛔ user RULED OUT (kept as the ceiling-setter) |

**The honest steer:** "more" most likely lives in the **strong-opponent positional grind** (unexplored,
largest, no mechanism) and the **upstream/multi-ply** reframe (the deaths we've been studying are
symptoms, not causes). The blunder-hunting vein is largely mined out.
