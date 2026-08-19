# RUNBOOK — gen-15, the v8-reward restoration

**Pre-registered 2026-08-18, BEFORE gen-15 launches.** Every rule below is fixed while the number
it governs does not yet exist.

Gen-15 is the **reward-composition restoration**: `--all-shaping-pbrs` ON (v8's last-known-good
composition), `--draw-penalty -35` (v8's), otherwise gen-14's `launcher_command` verbatim, plus the
licensed hygiene wave. Fresh-init — the reward config is resume-immutable
(`check_reward_config`), so this cannot be a fork.

## 0. What changed, and why the primary gate is still near-single-variable

| delta vs gen-14 | class | acknowledged |
|---|---|---|
| `--all-shaping-pbrs` ADDED | **the intended behavioural change** | yes |
| `--draw-penalty` −30.0 → −35 | part of the same validated v8 composition | yes |
| `--intent-value-reduce` dropped | hygiene — dV **0.3176**, below the pre-registered 0.39 bar | yes |
| `--value-clock` dropped | hygiene — dV **0.2169**, below the bar | yes |
| `seed` / `hidden_opp_vf` / `nmr` VF removed | hygiene — all read **exactly 0.0000** | code-level, no flag |

**The near-single-variable claim, stated so it can be challenged:** the four hygiene items are not
"substantive" *by this project's own standard* — the runbook bar for "this route matters" is 0.39,
both dropped routes read below it at n=12,391, and the other three read exactly zero (provably
behaviour-neutral). If a reviewer rejects that reading, the correct response is that gen-15 has TWO
deltas (reward + sub-bar hygiene) and its attribution weakens accordingly — **not** to re-derive the
bar after seeing the result.

**NEVER deleted, and not eligible for a later hygiene wave:** `threat` (dV 1.0686) and the
`hidden_opp` **pi** half (dV 0.0000 but **39.6% action flips** — a dV-only read would sever a live
policy input). `r` graduates from weak-pass (0.1385 vs a 0.0430 bar).

## 1. The generation gate — vs GEN-14

Same instrument as gen-14 §1, unchanged: tail-4 of the **dense** `snapshot_ladder/ladder.json`, at
matched snapshot COUNT, at run END, with the SE from the **paired refit** (`c'Σc` over the full
inverse-Hessian covariance) — never the naive diagonal, and never `main.elo`'s sparse fit (which is
what `main.endofrun`'s §1 line calls; it is ORIENTATION only and its disagreement is reported).

- **SUCCESS** iff Δ ≥ 0 with CI95-low > −15.
- **FAILURE** iff the whole CI sits below 0.
- Else INCONCLUSIVE → the pre-authorised tie-break is **more games per pair on the frozen ladder**
  (`load_games` sums duplicates; `--backfill` CANNOT do this and now says so). ⚠️ Size it against the
  **variance decomposition, not the game count**: only the frozen-pair component scales with games —
  4× bought 12% SE on gen-14, not the projected 50%.

**Earlier generations are ORIENTATION ONLY**, with the reward caveat stated at every citation: gen-11
… gen-14 trained on the 28-fully-additive-BIAS composition and gen-15 does not, so ladder deltas
against them mix a reward change with everything else. **Anchored ELO remains absolutely comparable**
(the bot anchors are pinned constants), so the absolute number is meaningful even where the delta is
not.

## 2. Did the restoration do what it is supposed to do?

The reward's whole claim is **policy-invariance**: PBRS telescopes (Φ(terminal)=0) and cannot bias
the objective; BIAS at λ=1.0 is fully additive and does. So:

- **Primary:** §1's ladder verdict.
- **Mechanism:** the per-term reward breakdown on gen-15's own eval traces — the 8 PBRS terms live
  and the 27 zeroed BIAS terms reading zero, with `no_progress_tax` the single survivor. A
  restoration that did not actually change the composition is the first thing to rule out, and it is
  free to check.
- **Watch-item, pre-registered because it is the known risk of dropping the bias terms:** stall rate.
  `--all-shaping-pbrs` without `--stall-pbrs` deliberately keeps `no_progress_tax` as stall
  insurance. Compare cap-length episode fraction against gen-14's measured **0.9% of episodes /
  3.0% of decisions** (`measurements/gen13_stall_coverage.json` for the method). A stall regression
  is a reason to reach for `--stall-pbrs` in gen-16, not to re-add 27 bias terms.

## 3. Awareness / coverage — the gen-14 regression re-read

Gen-14 read WORSE than the gen-10 baselines on blind_loss_fraction (0.138 vs 0.072), median_lead_time
(5.0 vs 7.0), coverage80 and pit_mean, consistent with its INFERIOR verdict. Re-read all four here.
**These are the numbers most likely to move with a reward-composition change**, since they are
critic-calibration readings and the critic's target just changed class. Report them; do not treat a
change in them as the generation verdict, which is §1's job.

## 4. Route / family re-audit

Re-run `critic_route_audit` and `edge_ablation_audit` at ≥12,000 states. **The v9-era magnitudes are
regime-scoped to the 28-bias composition** (ledger, `4379a5f`) — mechanisms may transfer, numbers do
not. So this is a fresh baseline for the v8-reward lineage, NOT a comparison; do not port H1's −2.7
or C4's ΔG spreads across the boundary without re-measurement.

Standing watch-item from gen-14: `entity_pool` carried **97.4%** of the critic's route dependence
with zero policy effect. If that persists under the restored reward it is a structural fact about
the critic, not about the reward.

## 5. TD-aux rung 2 — re-run off THIS base, pool pre-seeded

The gen-14-based arms are void as a rung-2 verdict (they trained against bots — see the defect
record). Re-run off gen-15's final checkpoint with each arm's `snapshots/` **pre-seeded from the
base's full pool, identical across arms, seed set recorded**. Same three arms (0.0 / 1.0 / 3.0), same
frozen gates from `levers/td_consistency_aux.md`. The empty-pool warning now fires if this is
forgotten.

## 6. What would make this generation a mistake

Stated now, so it cannot be rationalised later: if gen-15 lands INFERIOR to gen-14 **and** the
mechanism check in §2 confirms the composition really did change, then the v8 reward is not the
missing ingredient and the −38 hunt returns to the v91 bundle — the GIGO probe's NULL notwithstanding,
via lesion evals on the three new columns and 6M minus-one-change forks.
