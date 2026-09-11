# Critic ladder read — `ai_v12_13_ladder_tdaux` (`--td-aux-coef 1.0`) vs control `ai_v12_11_ladder_ctrl10M`, both at `step_10000032`

`python -m main.ops.critic_read ai_v12_13_ladder_tdaux --control ai_v12_11_ladder_ctrl10M --no-cache --out <job tmp>/read_tdaux2`, 2026-09-09 ~03:35–04:05 PT. A first attempt at 03:03 is DISCARDED: it picked `step_8000016` (the launcher process was still alive, `--on-live skip-newest`) and its `cf_audit` refused with 0/135 anchors because main's `search_driver` was the broken worktree build (ledger 2026-09-09 · *INCIDENT*). This read ran after the 03:15 rebuild; the arm's anchors reproduced normally. Same file layout as `../vf15_vs_ctrl10M_2026-09-08/`. Verdict: ledger 2026-09-09 · *READ · critic ladder arm `tdaux`*.


## `critic_read_v2.*` — the same pair, regenerated with the CONDITIONING section (2026-09-09 ~05:15 PT)

`python -m main.ops.critic_read ai_v12_13_ladder_tdaux --control ai_v12_11_ladder_ctrl10M --step
10000032 --out <job tmp>/read_tdaux_v2`. **`critic_read.md/json` (v1) are UNCHANGED and remain the
record of what was reported at the time**; v2 adds `main.ops.conditioning_meters`' ten rows and
changes nothing else — the identity and gate halves were reused from the v1 run's own artifacts on
a matching fingerprint, so every number they carry is byte-identical and the whole re-read cost
**45 s**. (The v1 invocation passed `--no-cache`, which forces recomputation but does not enter the
cache KEY, so the stamp it wrote was still a hit.)

`--step 10000032` is now passed explicitly. It resolves to the same cycle the v1 read used, and it
is what stops the failure named above from recurring silently: the report header and the tool's
stdout now say WHICH cycle was chosen and WHY, for both runs.

**The CONDITIONING rows** (ARM − CONTROL, battle-clustered difference of independent bootstraps,
2,000 draws, no replicate floor so every label is against zero):

| row | arm | control | Δ | 95% CI | |
|---|---|---|---|---|---|
| spread ratio, turn 1–3 ⭐ | +0.0000 | +0.1152 | −0.1152 | [−0.5134, +0.3248] | NOT DETECTED |
| spread ratio RAW, turn 1–3 | +0.1984 | +0.3333 | −0.1349 | [−0.3291, +0.2370] | NOT DETECTED |
| sd(V) − sd(outcome), turn 1–3 | −0.0608 | −0.0886 | +0.0278 | [−0.0298, +0.0641] | NOT DETECTED |
| spread ratio, all states | +0.5360 | +0.7895 | −0.2535 | [−0.6523, +0.3909] | NOT DETECTED |
| spread ratio RAW, all states | +0.6396 | +0.8053 | −0.1657 | [−0.5317, +0.3365] | NOT DETECTED |
| sd(V) − sd(outcome), all states | −0.0282 | −0.0211 | −0.0071 | [−0.0505, +0.0421] | NOT DETECTED |
| Elo slope, per 100 Elo | +0.0124 | +0.0098 | +0.0026 | [−0.0120, +0.0173] | NOT DETECTED |
| own-team LOO win-rate R², turn 1 ⭐ | −0.0382 | −0.0237 | −0.0145 | [−0.1037, +0.0595] | NOT DETECTED |
| own-team LOO win-rate R², all states | −0.0117 | −0.0098 | −0.0019 | [−0.1240, +0.1287] | NOT DETECTED |
| opponent-class AUC, turn 1 | +0.4345 | +0.4619 | −0.0275 | [−0.1728, +0.1246] | NOT DETECTED |

Frame: arm 6,208 states / 200 battles / 12 opponents / 100 trainee teams, `selection_schema` 2,
`max|values − win_probs|` = 0.0, 10 draw battles excluded; control 6,080 / 198 / 12 / 76, 6 draws
excluded. No row was omitted on either side.

🚨 **Both spread hazards are visible in this arm's own numbers.** The clamped ratio reads exactly
**0.0000 with a CI of [0.0000, 0.5489]** — the clamp, not a measurement of "no spread"
(`winprob_head_refit_2026-09-09` §12 hazard 1) — and the UNCLAMPED companion reads **0.1984 against
its own [0.2117, 0.6061]**, i.e. below its own interval too, from the upward bias of a resampled
between-group variance. **The interval is the read**, and the delta is the quantity least disturbed
by either, because both effects hit arm and control alike.

**`critic_read_v3.md/json` (2026-09-09, tool v3 — QUOTA MATCHING now default).** `main.ops.critic_read` now equalises the two trace frames before it reads any FRAME-SENSITIVE conditioning row (`main.ops.quota_match`). This pair is **SYMMETRIC** — both sides carry the same realized cap, **8/12 traced wins/losses per opponent** (tdaux 210 traced battles, ctrl10M 204) — so nothing is subsampled and **every delta row in v3 is bit-for-bit identical to v2**. v3 is kept only because the report now prints the realized capture profiles in its header, which is the evidence that this read was never affected.

**400-game OFFLINE read (2026-09-09):** `critic_read_hp400.md/json` — both sides 400-game offline-generated cycles (`main.ops.eval_trace_gen`, seed 20260909, 12 opponents, full capture); NOT comparable to the 100-game rows above (different population); ledger 2026-09-09 · *THE FLOOR AT 400 GAMES* and *THE ARMS AT 400 GAMES*.

**800-game OFFLINE read (2026-09-10):** `critic_read_hp800.md/json` — both sides 800-game offline cycles (seed 20260910); its own floor `../replicate_floor_10M_hp800.json`; ledger 2026-09-10 · *THE ARMS AT 800 GAMES*.

**Second 800-game eval draw (2026-09-10, seed 20260911):** `critic_read_hp800b.md/json`, floors from `../replicate_floor_10M_hp800b.json`; the bar is the WIDER of the two offline draws (rule 19). Ledger 2026-09-10 · *second eval draw for tdaux and strata*.
