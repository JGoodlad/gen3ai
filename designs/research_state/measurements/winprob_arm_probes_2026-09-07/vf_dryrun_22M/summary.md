# VF dry run — ai_v12_02_winprob_critic @ 22.45M (2026-09-07)

Checkpoint: `models/ai_v12_02_winprob_critic/checkpoints/checkpoint_22454016_steps.zip` (= `latest.txt`).
Traces: `eval_traces/step_22000032` (232 traces / 1400 battles / 14 opponents). Box load 8–23 throughout.
Nothing written under `models/`; all outputs in `scratchpad/vf_dryrun/`.

## TEST 1 — cf_audit (V vs tight-MC). RAN, then **REFUSED to emit labels.**
```
python -m agents.training.cf_audit models/ai_v12_02_winprob_critic \
  --checkpoint .../checkpoints/checkpoint_22454016_steps.zip \
  --rollouts 8 --states 40 --impl rust --out <scratch>/vf_dryrun/cf_audit --seed 0 --deadline-min 25
```
Runtime **2m34s** (40 states × R=8 = 320 rollouts + 20 anchors), exit 0. Artifacts: `cf_audit/bias_map.{md,json}`. **No `cf_labels/` — refused.** Verbatim:
```
  ANCHOR: 16/20 reproduced the recorded outcome (80.0%), 0 errors
cf_audit: LABEL TRUST FAILED — 16/20 anchors reproduced the recorded outcome (80.0% < 90%).
REFUSING to emit labels; the bias map is written for diagnosis ONLY and must not be quoted.
  evidential: this checkpoint carries no cf_evid_head — columns omitted
  twin heads: this checkpoint carries neither cf_twin_head_b nor cf_shadow_head — columns omitted
  skipped: {'turn_1_unopenable': 232, 'forced_switch_rounds': 929}
```
🚨 **This is not small-n noise.** A cheap re-probe (`--states 2 --rollouts 1 --anchors 80 --seed 7`, 46 s)
gave **70/80 = 87.5%**, still under the 0.90 tolerance. Two independent draws, ~87–88% true rate.
Not stall-driven (232 traces: median 28 turns, one ≥200, zero mention stalls). **Test 1's label arm will
refuse at 75M as things stand.** Do NOT lower `--anchor-tolerance` to get past it — the module frames a
non-exact replay as GIGO. Needs a root-cause pass on rust anchor replay before Tuesday, or Test 1 is
bias-map-only (diagnosis, unquotable).
Bias map format (small-n, DO NOT read as a result): headline `labels 40 / 32 battles · pop-weighted gap
+0.2243 · pop-weighted sd_true_excess 0.2638`; strata tables by outcome / predicted-decile×outcome /
turn-tercile / opponent, each `n, n_battles, mean_predicted, mean_mc, mean_gap, 95% battle-clustered CI`;
a "conviction class" block (high-V lost battles, MC≥0.75 vs <0.50 vs <0.25 shares); a RESOLUTION table
(`sd(MC)`, binomial floor, **sd_true_excess**, % variance real) per decile; EVIDENTIAL section ABSENT-not-zero.

## TEST 3 — bootstrap consistency
**(a) `td_resid_tails`** — one dict per `eval_results.jsonl` row (11 rows, steps 2.0M→22.0M), keyed by the
9 fixed bots (sentinels excluded). Each value is `td_tail` = **CVaR@5% (mean of the worst 5% of per-decision
TD residuals δ)** pooled over that opponent's *captured* battles, from `eval_sharding/results.py::td_tail`
(`TD_TAIL_FRAC=0.05`, `<20` samples ⇒ the single min). δ is defined in `battle_recorder.py:97` as
`δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t)` — the same formula the prober uses.
**Under `--critic winprob` this simplifies:** r=0 at every non-terminal decision and γ=1, so **δ ≡ ΔP(win)**.
`td_resid_tails` is therefore literally "the mean of the worst-5% one-step drops in P(win)".
Values (`random` / mean-over-9): 2.0M −0.130/−0.185 · 6.0M −0.062/−0.146 · 10.0M −0.050/−0.128 ·
14.0M −0.061/−0.131 · 18.0M −0.066/−0.131 · 20.0M −0.041/**−0.104** · 22.0M −0.063/−0.139.
Trend: shrinking to 20M then a bounce at 22M. Caveat carried by the code: the δ pool is the *captured*
(loss-enriched, shard-dependent) sample — exact aggregation over a non-random pool.

**(b) calibration** — `python -m main.prober.query --impl rust calibration models/ai_v12_02_winprob_critic
--step 22000032 --limit 6 --worst 2 --seeds 16 --alts 2 --concurrency 2` → **1.9 s**, exit 0, stdout JSON
(`vf_dryrun/calibration.json`). No refusal, no arch drift, no timeouts.
- **Currency read from the run** (the `gen3_prober_winprob_currency_v1` fix is live): `mode winprob, units P(win),
  span 1.0, overvalue_tau 0.0833 (per-currency default)`, `threshold_warning: null`.
- **Selection: KNOWN** — `SELECTION RECORDED for every step in scope`, schema 1, 232 traces / 1400 battles,
  mean capture rate **0.097 per WIN vs 0.732 per LOSS**.
- `overall_calibration` (n=7316 decisions): bias +0.340, **bias_on_wins −0.126 vs bias_on_losses +0.684**,
  ev 0.163, slope 0.877, captured_win_fraction 0.483.
- **Split**: `gate.unattributed 0.4714` → `critic_overvalued 0.4714` / **`lost_position 0.0`**
  (6/6 unattributed craters bucketed overvalued, mean reliability gap 0.376);
  `policy_reducible 0.362`, `aleatoric 0.109`, `critic_mean_reducible_upper_bound 0.4714`.
- The tool's own verdict line is `SELECTION-CONFOUNDED — read the splits, not the headline … the
  <0-on-wins / >0-on-losses pattern is the signature of a CALIBRATED critic … a LOOSE UPPER BOUND`.
  `falsify_coverage`: 6 falsified, **114 capped by `--limit`** — raise it at 75M (it is cheap).

**(c) per-turn δ profile** — `python -m main.prober.query turns <summary.json>` is **0.2 s, model-free**.
Per decision it carries `value`, `win_prob` (identical here), `delta_v`, `td_residual`, `reward_total`,
plus board/timeline/protocol. Example (`step_22000032/heuristic/loss_003`, `gamma 1.0`, `critic_currency
{mode: winprob, …}`), rendered to `turns_profile.txt`:
```
turn inv  chosen                V=P(win)     dV      TD     r
   2   1  spikes                   0.833  0.051   0.051  0.0
  ...
  27  29  surf                     0.978 -0.119  -0.119  0.0
  28  31  hydropump                0.746 -0.283  -0.283  0.0
  28  32  switch:tyranitar         0.464 -0.094  -0.094  0.0
  29  33  dragondance              0.370   None    None  0.0   <- last decision has no δ
```
Confirms `TD == dV` exactly on this arm (r≡0, γ=1) — a useful self-check at 75M: if any row shows
`TD != dV`, the terminal-only reward assumption has broken.

## 75M runbook (change only the checkpoint path and the sizes)
```bash
export PYTHONPATH=$PYTHONPATH:src && cd /home/goodlad/dev/gen3ai
CKPT=models/ai_v12_02_winprob_critic/checkpoints/<75M checkpoint>.zip   # or read latest.txt
STEP=<eval_traces step at 75M>
OUT=<scratchpad>/vf_75M                                                 # NEVER default (--out is inside models/)

# TEST 1 — identity: V vs tight-MC.  ~2.5 min at 40 states; budget ~40 min at 400 (use --deadline-min).
python -m agents.training.cf_audit models/ai_v12_02_winprob_critic --checkpoint $CKPT \
  --rollouts 8 --states 400 --impl rust --out $OUT/cf_audit --seed 0 --deadline-min 60
#   ⚠ EXPECT "LABEL TRUST FAILED" (~87.5% anchor rate at 22M) unless the anchor defect is fixed first.
#     If it refuses: report the refusal, quote NOTHING from bias_map, do not lower --anchor-tolerance.

# TEST 3a — the run's own δ tails (offline, instant)
python -c "import json;[print(json.loads(l)['step'], json.loads(l)['td_resid_tails']) for l in open('models/ai_v12_02_winprob_critic/eval_results.jsonl')]"

# TEST 3b — calibration split (was 1.9 s at limit 6; limit 60 is still seconds)
python -m main.prober.query --impl rust calibration models/ai_v12_02_winprob_critic \
  --step $STEP --limit 60 --worst 2 --seeds 32 --alts 2 --concurrency 2 > $OUT/calibration.json

# TEST 3c — one battle's per-turn residual profile (0.2 s, model-free)
python -m main.prober.query list models/ai_v12_02_winprob_critic --step $STEP --outcome loss
python -m main.prober.query turns <that summary.json> > $OUT/turns_example.json
```
