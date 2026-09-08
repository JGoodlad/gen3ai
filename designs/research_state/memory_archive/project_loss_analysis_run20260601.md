---
name: project_loss_analysis_run20260601
description: "Loss forensics for run_20260601_193826 @122M — unified root cause is critic tail-blindness on opponent super-effective/revenge/recovery coverage, driving greedy self-setup + under-switching"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26bc4a49-81f1-4b03-b296-9cac526eb8a5
---

> **Archived 2026-09-08** — June-2026 forensic pass; siblings _run20260531 and _v2 were archived 2026-09-07. Preserved verbatim; nothing below is current.

Forensic pass (2026-06-05) on run_20260601_193826 @ step 122,000,011 (best_model, git 1081a8e,
self-play). Prober, exact tier, 7 parallel subagents over all 14 eval opponents, ~70 loss turning
points. Report: `models/run_20260601_193826/LOSS_ANALYSIS_2026-06-05.md`.

**One root cause, three faces.** The policy loses by 1–2 mispriced catastrophic decisions/game, not
gradual disadvantage. At the turning point the critic sits strongly positive, then an opponent answer
it never priced lands and V crashes (td-residual −10 to −59, almost always an unanticipated negative
surprise). In wins the worst drop is only −2 to −10.
1. **Critic tail-blindness** — value head can't price the opponent's best incoming answer (super-
   effective/OHKO coverage, revenge speed/existing boosts, wall Rest/Recover, low move accuracy like
   DynamicPunch 50%, our own −1 atk). Well-calibrated on average (TB explained_variance 0.83) — it's
   a *tail* problem, so the lever is OBS, not just more training.
2. **Greedy self-setup** — DD/CalmMind one turn too many into a healthy check → fully-built sweeper
   revenge-KO'd (often by a bulky Water); passing free KOs to heal (Softboiled vs 7% TTar, td −41).
   NB: vs setup_sweep/v2 the OPPONENT set up unchecked in 0/10 losses — the "setup" problem is ours.
3. **Under-switching + futile attacks** — staying/spamming into a losing matchup (RockSlide ×3 into
   +2/+2 Gyarados appears in BOTH sentinel_0 & sentinel_1 loss_010) with a healthy bench mon unused
   (the switch is even weighted ~0.2–0.3 — it just loses the argmax); attacking walls/immunities
   (Selfdestruct into a Ghost = 0×). All downstream of #1: an over-priced board → no reason to pivot.

**Confirms & extends prior runs** [[project_loss_analysis_run20260531]] / [[project_loss_analysis_run20260531_v2]]:
the damage-magnitude/OHKO obs gap is now the #1 root; under-switching, weak futile_attack, and
explosion/self-KO-as-reward all re-confirmed. Aligns with [[feedback_provide_vs_learn]] — saliency
shows the policy attends to ITS OWN move effectiveness but not the symmetric INCOMING threat.

**Levers (priority):** (1) obs: give the model the opponent's best incoming answer — our active's
worst-case incoming effectiveness / "am I OHKO-able / outsped", opp revealed-move coverage vs us;
(2) reward: curb self-KO/Explosion reward, "take the lethal" signal, stronger pivot-before-death
incentive; (3) note TB clip_fraction_vf≈0.70 may slow value tail-learning (loosen clip-range-vf /
scale rewards); (4) self-play alone won't fix it — failures reproduce identically vs the sentinel
pool, so pair with obs/reward shaping.

**Method gotcha:** best_model == the latest checkpoint, so `find disagree --tier recent` is a no-op
on the newest traces (0/70 turning points "change") — that means "the latest model still makes every
blunder", NOT "nothing changed". To measure learning over a run, replay OLDER-step battles under the
latest model. Trace sampling is loss-biased (5 win/10 loss saved per opp/cycle) → counts ≠ win rate.

**Is damage learnable? Empirically adjudicated (2026-06-05).** The obs DOES carry the opponent's
incoming effectiveness (`their_matchups`, 6×4×6 at obs offset 1612) — BUT it's *effectiveness, not
damage* (no power/Atk·Def/HP → no OHKO threshold), and it's built from `get_sorted_moves` =
**revealed moves only**, so it is ALL-ZEROS for an unrevealed / just-switched-in opp mon (the
Claydol-EQ case). Prober `analyze` on aggressive_v2 loss_006: at the Claydol switch-in `their_matchups`
revealed_frac≈0.11 (89% of opp coverage blank), and **saliency: own-offense `active move_multipliers`
≈0.47 vs incoming `their_matchups` ≈0.0018 → the policy attends ~260× MORE to its own offense than to
the incoming threat** (≈750× at inv13). So it's BOTH a partial representation gap (incoming = sparse
effectiveness, blank for fresh switch-ins, no OHKO bucket) AND a learned-attention/credit-assignment
gap (MSE value loss + vf-clip≈0.70 under-weight the rare catastrophic tail; no lookahead → a feedforward
net must amortize "their best response if I switch X in"; self-play mirror under-punishes it). Humans
win without calcs via species→coverage priors + OHKO/2HKO bucketing + 1-ply lookahead — the model has
weak versions of all three. Levers (sharpened): add an OHKO/incoming-damage feature (effectiveness ×
power × Atk·Def vs our HP, incl. a prior-coverage estimate for unrevealed mons so it's not blank);
distributional/quantile value head or unclipped-vf to learn the downside tail; inference-time search.

**Incoming-damage obs feature — design + zero-retrain falsifier (2026-06-05).** Design doc
`designs/ai_v5/design_incoming_damage_obs.md` (6-lens reviewed) + a model-free cliff-attribution
falsifier (`designs/ai_v5/falsifier_cliff_attribution.py`) + 5-agent verification. Verdict
**CONDITIONAL-GO, medium**: the critic is tail-blind to unpriced incoming hits on BLANK opponent
coverage at **~50% of decisive loss turning points** (robust 42–64%; clean — 99% real hits) = the
NECESSARY condition, met. But NOT sufficient, two binding gaps: (a) **the losing decision is
POLICY-side** — in 67% of addressable cliffs a healthy bench switch existed but the policy had already
collapsed switch mass to <5% (median 0.006) → a critic-side obs feature won't change the action without
a retrain that ALSO fixes under-switching; (b) retrain-attention is unknowable zero-retrain (current
incoming lane ~0.002 saliency). The "losses have more addressable cliffs" claim REVERSED (conditional
on a cliff, wins have more; losses just have ~2.9× more cliffs). **Decision: critic-first first** —
the user is un-clipping `--clip-range-vf` (same root, zero obs cost); re-run the falsifier on the
post-fix checkpoint; build the scoped minimal Phase-1 obs feature ONLY if an unpriced-incoming-KO
residual persists. Key code facts found: fixed-damage moves (Seismic Toss/Counter/etc.) read
basePower=0 → bucketed STATUS (gen3_data/moves.py) → would price as 0 threat; `their_matchups` (obs
1612) is already on the per-mon→both-heads path (so low saliency = info problem, not placement); reuse
`opponents.py:164 _estimate_damage_fraction`; priors should come from `data/teams/` not ladder usage.

**RE-MEASURE 2026-06-06 → GO (cheap critic-first fix did NOT win).** Ran the gating no-vf-clip resume
(122M→158M, same run dir; `clip_range_vf` off from ~128M — confirmed via TB: clip_fraction_vf/clip_range_vf
stop logging; EV stayed ~0.88, value_loss 178→97, win_rate_mean 0.772→0.799 — live + safe, average-fit
gain only). Within-run before/after on `falsifier_cliff_attribution.py` (now takes MIN/MAX step argv to
slice eras): unpriced-incoming-KO cliffs **PERSISTED** — cliff rate 10.0%→8.6% (−14% only), decisive-TP
structure ~unchanged (clipped 55% addressable/65% incoming/88% blank vs un-clipped 54%/63%/85%). So
un-clipping vf does not fix the tail-blindness → **GO: build the scoped Phase-1 incoming-damage feature**
(`designs/ai_v5/design_incoming_damage_obs.md` §5/§7/§11) + fold in the recovery `cures_status` scalar
([[project_stall_recovery_analysis]]). Binding caveat unchanged: it's also policy-side under-switching, so
the retrain must move the policy (Gate 2 = saliency floor + CRITIC_BLINDSPOT drop + switch-rate check).

**Prober gained (worktree, unshipped):** (1) `query scan <run> --outcome loss [--opponent X]
[--metric value_drop|td_residual]` — model-free cross-battle turning-point triage; (2) TUI Outcome
panel shows TD δ; (3) **incoming-threat decode**: `analyze` now returns a `threats` block
(`present`/`revealed_frac`/`max_incoming`/`per_our_slot_max` from `their_matchups`) + a
`their_matchups(144)` saliency block + a Matchups-panel "incoming (opp→us)" line — the lens that
adjudicates representation-vs-learning per decision. See [[feedback_prober_self_improvement]].
