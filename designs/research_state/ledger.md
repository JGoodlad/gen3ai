# Ledger — hypothesis status table

The at-a-glance dashboard. Full Known/Not-known/Pros/Cons for OPEN levers live in
[levers/](levers/); the [README](README.md) holds the protocol + the frontier.

## The method (how a hypothesis earns a verdict)

Every behavioural hypothesis runs the **forensic template** (this confirmed self-KO and killed five of
six blunder categories), reusable via `python -m main.prober.query decision-table <run>`:

1. **Frequency** — how often does the pattern fire?
2. **Confidence** — `softmax(logits)[chosen]`. >0.3 = a *learned* preference; <0.1 = exploration tail (drop it).
3. **Reward** — already *punished* (reward < 0)? If yes, the reward isn't the bug.
4. **Critic dV** — `V(s')-V(s)` *positive* on the bad turn (critic over-values, neutralizing the reward
   in the PPO advantage)? = the self-KO mechanism.
5. **Recoverability** — does the **falsifier** confirm a materially-better action existed (MISTAKE/MIXED),
   or is it a symptom of a lost position (LUCK/NEUTRAL)? **Size ≠ recoverability.**

**Honesty gates** (every claim clears these before reaching "Known"): outcome-conditioning · falsifier
myopia (over-flags setup) · legitimate-in-context · learned-vs-exploration · **always adversarially
verify a confirming measurement** (overturned 3-for-3 this session).

## Status

✅ CONFIRMED · ❌ KILLED · 🔬 OPEN · 🛠 FIX BUILT

| # | Hypothesis | Status | Mechanism / evidence | Re-verify |
|---|---|---|---|---|
| H1 | Policy explodes **healthy** mons (self-KO over-valuation) | ✅🛠 | ~38% of Explosions at ≥80% HP, conf~0.5 (learned). Reward correct (−2.7), exploration ruled out, ① exonerated. CRITIC over-values the trade (dV +2.9 → PPO advantage +1.5). Fix `--self-ko-hp-penalty` built+tested, NOT shipped. | `decision-table <run> --cat selfko` (selfko dV_med) |
| H2 | **Attack type-mismatch** (resisted/immune move while a SE move sits in the same kit) | ✅ | Falsifier+type-chart confirmed; SMALL (~3–5 confident loss decisions). REPRESENTATION gap (reward+critic correct). Fix = obs effectiveness feature, not reward. Not built. | falsify MISTAKE corpus, best_label = a different attack |
| H3 | **Surprise-OHKO / unrevealed-threat** obs-coverage gap | 🔬 GO (gated) | Belief UNDER-FIRES on 52% of lethal healthy-stay deaths (pko<0.3). Provenance: 36% just-switched, 42% new-move (priors gap), 22% fully-known (calibration bug). CAVEAT: ~same rate in wins (56%) → pervasive. **Recoverability (n=381, stable from n≈60): 42% AVOIDABLE** (64% of those a SWITCH-to-a-wall — Swampert/Jirachi/Snorlax/Gyarados, ~1 mon), **33% LUCK** (belief was RIGHT, died to crit/roll — NOT coverage), **25% NEUTRAL** (committed). 42% > 20% kill-bar → **GO**, but half-belief/half-under-switching → must pair with `--switch-bias-weight`. | `models/saved_work/surprise_death_recoverability.py` |
| K1 | **Distributional value critic** lifts the strong-opp ceiling | ❌ | Strong-opp residuals SUB-Gaussian (tail-dom 0.33, exkurt −0.89) — no tail to re-weight. The "fat tail" was outcome-conditioning + the PP-stall reward artifact. | value-calib: V vs return-to-go residual shape |
| K2 | **Damage-magnitude** obs feature is the gap | ❌ | r²≈0.08 partly a marginal-vs-conditional artifact, but belief-conditioned r²=0.012; belief saturated above its floor; residual is *which-move*, not magnitude. | `probe damage_taken` (marginal vs belief-conditioned) |
| K3 | **① `value_active_readout`** causes the self-KO over-valuation | ❌ | ai_v5_9 (no ①) explodes 16.5% when-available vs ai_v5_11 (with ①) 13.7% — same. ("0 explosions" was a script bug.) | `decision-table` per-run explosion-when-available |
| K4 | **Recovery-move misuse** (full-HP heals) | ❌ | Low-conf (0.18) exploration tail, already punished; confident cases split 9W/9L (legit Recover-stalls). Direction FLIPS — falsifier flags a heal as the *better* move 17×. | `decision-table <run> --cat recovery` |
| K5 | **Setup-move misuse** (setup-into-death) | ❌ | Myopic-falsifier over-flag; clean core <0.12%; critic already prices it NEGATIVE (dV ~−4). Blind-setup-death is the H3 obs gap. | `decision-table <run> --cat setup` |
| K6 | **Stall/utility misuse** (Protect/Leech/status-on-dying) | ❌ | Reward-punished, NO critic over-value (dV~0), falsifier never craters on the dying-mon turn (lost earlier). | `decision-table <run> --cat stall` |
| K7 | **Switch pathologies** | ❌ | UNDER-represented among crater mistakes (24% vs 28%); ~2 confirmed voluntary switch mistakes in 350 losses (254/291 craters were FORCED replacements). | `falsify-scan <run>` (switch crater base-rate) |
| K8 | **No-op status-spam** (Spikes-at-cap loops) | ❌ | Real+learned, but no-progress clock already punishes it, critic prices the state as lost; ~74 dead turns in already-lost games. | `decision-table <run> --cat status` |

## GO-TO-BUILD queue (the next FRESH run stacks these)

Posture shift (owner, 2026-06-12): stop the kill-treadmill, start **committing**. No single big lever
remains → stack the independent, cheap, amortizable levers into one fresh run and measure the aggregate
(ELO + bot-wr). The build bar (README → Decision posture) is LOWER than "Known": positive EV +
falsifiable-after-build is enough. Every entry must clear the **amortizability gate** (L1–L4; no search
on the model). Candidate stack (pending the amortizability pass + owner confirm):

| Lever | Bucket | State | Falsify-after-build (the metric that must move) |
|---|---|---|---|
| H1 self-KO HP penalty | reward (L2) | SHIPPED (`--self-ko-hp-penalty`) | healthy-mon self-KO rate falls; selfko dV no longer +; wr non-regress |
| H3 surprise-OHKO coverage + under-switching | obs (L1) + reward (L3-ish) | GO (gated) | death-turn pko rises AND switch-to-wall rate rises (BOTH) |
| H2 attack type-mismatch obs feature | obs (L1) | confirmed-small, cheap | resisted/immune-into-available-SE picks fall |
| _amortizability-pass output_ | TBD | pending | the grind/upstream decomposition's greenlit lever(s) |

## Active runs / context

- **ai_v5_11_tail2_53m_0611** — done at 53M. Rebalanced (vf_coef 0.5→0.25); `value_share` landed ~0.6
  (partial); ELO ahead of ai_v5_10. The forensic corpus for this session.
- **ai_v5_12_bias_05_N_0612** — LIVE (bias-redesign; orthogonal to H1/H3; trained on pre-H1 code).
- **Next run** = fresh `--self-ko-hp-penalty 2.5` (H1 A/B; resume-immutable → fresh, not a resume of ai_v5_12).

## Re-verify tooling

Consolidated in `src/main/prober/forensics.py` — re-verification is one command on ANY run:
```bash
python -m main.prober.query decision-table <run> --outcome loss             # per-category forensic digest
python -m main.prober.query falsify-scan   <run> --outcome loss --max-battles 200   # luck-vs-mistake bracket
```
`ProbeSession.decision_table()` / `.falsify_scan()` for scripts. The `outcome='?'` discovery bug
(eval-shard `s<N>_` filenames) is fixed, so `--outcome` filters work.

## Cross-refs

Deep notes in agent memory: `project_floor_leak_critic_selfko`, `project_distributional_critic_verdict`,
`project_incoming_damage_outcome`, `project_model_frontier_roadmap`. The prober (`src/main/prober/`) is
the forensic engine; this folder is its conclusions.

_Last updated: 2026-06-12._
