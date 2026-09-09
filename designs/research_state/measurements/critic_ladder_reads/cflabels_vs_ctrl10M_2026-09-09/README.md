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
