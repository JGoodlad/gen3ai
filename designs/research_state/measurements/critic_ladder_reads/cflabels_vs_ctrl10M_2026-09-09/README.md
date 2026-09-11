# Critic ladder read — `ai_v12_12_ladder_cflabels` (`--cf-records --cf-winprob-coef 0.5`, `--checkpoint-every-steps 500000`) vs control `ai_v12_11_ladder_ctrl10M`, both PINNED to `step_10000032`

`python -m main.ops.critic_read ai_v12_12_ladder_cflabels --control ai_v12_11_ladder_ctrl10M --step 10000032 --out <job tmp>/read_cflabels`, tool v2 (conditioning rows registered), 2026-09-09 ~07:20 PT. Cross-commit (arm `377a5aa1`, control `f3502568`; span verified NEUTRAL) and cross-QUOTA (arm 40/40/10, control 5/10/5 — arm frame 18,739 states / 627 battles / 183 teams vs 6,080 / 198 / 76): every gauge row is capture-rate reweighted (rule 17), but the own-team DECODER's fit also sees 2.4× the teams on the arm side — see the ledger entry's caveat and the matched-quota re-read beside it. Verdict: ledger 2026-09-09 · *READ · critic ladder arm cflabels*.

🚨 **`matched_quota/` re-reads this directory's conditioning rows on a frame matched to the
control's, and WITHDRAWS the own-team detection.** Subsampled to the control's realized capture
profile (30 seeds), the arm's `cond.own_team_r2.t1` falls +0.060 → −0.025 and the delta reads
−0.0013 [−0.2436, +0.3461] NOT DETECTED; on a decoder-matched frame +0.0275 [−0.1162, +0.3785],
still NOT DETECTED. The frame-size curve climbs monotonically with the decoder's own battle count,
which is the artefact's signature. Two unregistered companion rows (`cond.spread_ratio_raw.t1_3`,
`…​.all`) also lose their detections. `identity.bias.*` is NOT implicated — it is a weighted mean,
not a fit. Read `matched_quota/README.md` before quoting any DECODER-based row from
`critic_read.md`.

**`critic_read_v3.md/json` (2026-09-09, tool v3 — the matched read is now the TOOL's, not a session script).** `main.ops.critic_read` equalises the frames itself (`main.ops.quota_match`, default ON, `--no-quota-match` opts out): the arm's realized cap is **40/35** traced wins/losses per opponent against the control's **8/12**, so the arm is cut to 8/12 over **21 seeded in-memory subsamples** with the capture rates recomputed per draw, and a DECODER-matched rung is SEARCHED for — it lands on **11/16**, the same caps `matched_quota/` hand-picked. The v3 report reproduces this directory's verdict from inside the tool: `cond.own_team_r2.t1` **+0.0077 [−0.1353, +0.0816] NOT DETECTED** battle-matched (197 battles / 62 decoder battles) and **+0.0287 [−0.0808, +0.1327] NOT DETECTED** decoder-matched (247 / 103), with the as-traced **+0.0841 [+0.0323, +0.1825]** printed beside them carrying **no label**. `cond.own_team_r2.all`, `cond.spread_ratio_raw.t1_3` and `cond.spread_ratio_raw.all` likewise lose their detections. The remaining difference from `matched_quota/` is the seed count (21 vs 30) and the median convention (an exact order statistic, so the point and the interval printed beside it are the same draw); the verdict is identical.

**400-game OFFLINE read (2026-09-09):** `critic_read_hp400.md/json` — both sides 400-game offline-generated cycles (`main.ops.eval_trace_gen`, seed 20260909, 12 opponents, full capture); NOT comparable to the 100-game rows above (different population); ledger 2026-09-09 · *THE FLOOR AT 400 GAMES* and *THE ARMS AT 400 GAMES*.

**800-game OFFLINE read (2026-09-10):** `critic_read_hp800.md/json` — both sides 800-game offline cycles (seed 20260910); its own floor `../replicate_floor_10M_hp800.json`; ledger 2026-09-10 · *THE ARMS AT 800 GAMES*.

**Second 800-game eval draw (seed 20260911, 2026-09-10):** `critic_read_hp800b.md/json`. Ledger 2026-09-10 · *the second eval draw*.

**v6 re-read of the 800-game frames (2026-09-11):** `critic_read_hp800_v6.md/json` — adds `cond.opp_class_auc.{t1_3,t4_10}`, the late spread rows, the optimal-spread decomposition and the crossing step; floors `../replicate_floor_10M_hp800.json` + the v6 two-draw floors (class AUC t4–10: 0.022). Ledger 2026-09-11 · *v6 re-reads of the five earlier levers*.
