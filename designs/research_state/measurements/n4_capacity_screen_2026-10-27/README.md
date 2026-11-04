# N4 (the toy capacity screen) look 1: **NO-GO** (2026-11-04)

N4 grows every trunk round's feed-forward block (FFN) at the toy plateau T\* from 256 to 1,024 hidden units and trains
+70M (`--grow trunk_ffn_dim=1024`, `N4_PIN` `1553f8bd8`, seeds s3641–s3643). The control is C0′: T\* with the same argv
minus `--grow`, at the same pin and seeds. The read is rule `n4_screen` (`main.reads`, [`PREREG.md`](PREREG.md) §5 as
amended by AM-N4-FIXES), spec [`read_spec.json`](read_spec.json) (sha256 `579d2564…`, NOTE 2), played at the one main
commit `31bd9ced9`. The record is
[`../x35_breakers_stage_2026-10-17/reads/toy_n4ffn_look1.json`](../x35_breakers_stage_2026-10-17/reads/toy_n4ffn_look1.json).

| | s3641 | s3642 | s3643 | mean |
|---|---|---|---|---|
| S, N4 (pp; 6 members × 2,000 mirrored pairs) | 51.71 | 51.63 | 52.49 | 51.94 |
| S, C0′ (pp; the same cells) | 51.69 | 51.33 | 51.41 | 51.48 |
| K on forum72 vs Kakuna, N4 (1,000 games each) | 0.428 | 0.413 | 0.451 | 0.431 (1,292 / 3,000) |
| K on forum72 vs Kakuna, C0′ | 0.445 | 0.443 | 0.398 | 0.429 (1,286 / 3,000) |
| mean LR (`snapshot_history`), N4 / C0′ | 1.26e-4 / 1.40e-4 | 1.44e-4 / 1.36e-4 | 1.36e-4 / 1.45e-4 | ratio 0.965 |
| wall (min) / steps per s, N4 / C0′ (dose records) | 259.3 / 230.0 | 261.7 / 241.8 | 261.6 / 239.6 | ρ = 1.10 |

- **Δ_S +0.46 pp [−0.28, +1.20]** (Welch 90 %, df 2.6). UB < +2.0, so **NO-GO-S**. (GO-S would also have needed
  Δ̂_S ≥ +1.0 pp.)
- **Δ_K +0.20 pp [−2.91, +3.31]** (Newcombe at z 1.6449, widened ×1.48 by φ̂ 2.19). Neither GO-K (LB ≤ 0) nor NO-GO-K
  (UB ≥ +1.5), so **UNDECIDED-K**. Not MIXED (UB_K ≥ 0 and no GO meter).
- **Class: NO-GO** (§5.3, AM-N4-FIXES item 4). The registered text: "no ≥ 2 pp in-house gain from a 4x trunk FFN width
  (`--grow trunk_ffn_dim=1024`) within 70M of T\*"; "no unfamiliar-only effect ≥ ≈ 4 pp detected; smaller ones are
  not excluded". It never reads "capacity does not bind on the toy" (M5). The family false GO is ≈ 0.05–0.07.
- **Labels.** "LR-confounded" and "LR-elevated" do not apply (ratio 0.965, inside [0.9, 1.1]).
- **Guards** (every registered pre-record check):
  - the VOID gate (K9(b) PASS, every `pin_history` span = `N4_PIN`) passes on all six runs;
  - the growth stamps (§4), checked by hand on all six `metadata.json`: each N4 run carries `lineage.grow` with
    `trunk_ffn_dim` {parent 256, child 1024} and `function_check` max |Δ| 0.0 / 0.0 (bit-identical); no C0′ run carries
    one;
  - every final ≥ 360,000,087 steps (360,001,186–360,001,560); `pin_history` is ONE span per run;
  - one engine per meter: K `port=e00a212a…/94;env=3d0d5c9c…/129`, S `env=3d0d5c9c…/129` (this commit's engine; no
    banked row used it, NOTE 2);
  - the re-derived anchor guards: `count`, 300 units checked, 0 hits;
  - K counted 300 of 300 units, 0 exhausted; every S cell at 2,000 pairs; `completeness` COMPLETE; no repin; the
    playing tree clean at `31bd9ced9`; `check-spec` PASS.

**Reported, never deciding** (§6; no adjectives):
- **R1 at look 1** (97.5 %): NULL, Δ +0.46 [−0.92, +1.85].
- **Per-member Δ_S** (Welch 90 %): `toy_150m` +1.15 [−0.17, +2.47], `toy_220m` +0.93 [−0.48, +2.33], `toy_280m`
  +0.34 [−0.19, +0.87], `pstar_x26` +0.26 [−2.07, +2.59], `toy_pstar` +0.13 [−1.19, +1.46], `toy_080m` −0.03
  [−1.28, +1.22].
- **G-F** (the 3 × 3 head-to-head, N4 finals as players against C0′ finals, 500 mirrored pairs a cell, seed 0, regime
  `adfdbee824c9eb0a` as S; [`gf.sh`](gf.sh), [`reported_lines.json`](reported_lines.json)): cells 48.3–52.8 pp,
  mean 50.57 pp (se ≈ 1.5 pp a cell).
- **Pin drift** (cross-engine, so descriptive only, NOTE 2): C0′'s mean S 51.48 against the banked toy C0 finals' 52.43
  (`toy_x47_look1.json`, toy pin, engine E1): −0.95 pp [−2.13, +0.23] (Welch 90 % on 3 + 3).
- **σ:** S's seed SD is 0.48 pp (N4) and 0.19 pp (C0′). The S screen's power at the observed σ: GO-S 0.994 at +2 pp,
  NO-GO-S 0.994 at 0. K per-final SD 0.019 (N4) / 0.027 (C0′).
- **ρ = 1.10** (wall per step, N4 over C0′; 4,507 vs 4,961 steps/s). The design GUESSED 1.2.
- **home72: not played** (§6 item 6 runs "only if the anchor lane has room"; the box's CPU was shared with S′-LIVE).

**The mechanism read** (§6 item 7, design doc §5; [`reported_lines.json`](reported_lines.json)):

| | s3641 | s3642 | s3643 |
|---|---|---|---|
| E1, N4 final: ‖`linear2_x`‖ / ‖`linear2`‖, rounds 0 / 1 | 0.183 / 0.193 | 0.215 / 0.219 | 0.197 / 0.209 |
| E1 null, C0′ final: base drift × √3, rounds 0 / 1 | 0.178 / 0.226 | 0.175 / 0.221 | 0.182 / 0.229 |
| E2, N4: mean KL(π ‖ π with `linear2_x` zeroed) at 300M → final | 0.032 → 0.091 | 0.028 → 0.082 | 0.036 → 0.085 |
| E2, N4 final: argmax moved | 20.7 % | 20.1 % | 19.5 % |
| E2 null, C0′ final: KL with base `linear2` reverted to T\* / argmax moved | 0.144 / 25.9 % | 0.125 / 25.3 % | 0.119 / 24.8 % |

| battery v3, 2,000 states of C0′ s3641's eval traces | `team_tokens` PR / srank99 | `pi_features` PR / srank99 | trainability ratio, `team_tokens` | tanh l1: sat. acts / sat. units |
|---|---|---|---|---|
| T\* | 16.79 / 110 | 22.58 / 361 | 1.201 | 0.870 / 0.063 |
| C0′ s3641 / s3642 / s3643 | 17.12 / 15.78 / 17.94; 110 ×3 | 22.60 / 20.63 / 22.48; 366 / 356 / 362 | 1.217 / 1.218 / 1.251 | 0.880–0.887 / 0.086–0.092 |
| N4 s3641 / s3642 / s3643 | 15.43 / 15.93 / 15.75; 109 ×3 | 22.15 / 20.53 / 20.79; 360 / 355 / 356 | 1.195 / 1.220 / 1.184 | 0.884–0.892 / 0.082–0.106 |

- **The registered reading** (design doc §5, MY JUDGEMENT, written before any N4 number): E1 ≥ 0.01 at every final and
  E2's KL rises at every seed, with a NO-GO, which is the row **"used, buys nothing"**.
- **Beside it, the M1 nulls.** The new units' share of the FFN output (E1 0.18–0.22) is the size of C0′'s own base-FFN
  drift scaled to the same width (0.18–0.23). The policy shift they carry (E2 KL 0.08–0.09, argmax 19.5–20.7 %) is
  BELOW the shift C0′'s ordinary 70M base drift carries (KL 0.12–0.14, argmax 24.8–25.9 %). So the units were used, at
  about the scale of ordinary drift, not beyond it.
- **The battery:** N4's `team_tokens` srank99 (109) is not above C0′'s (110), and N4's layer-1 saturated-unit share
  (8.2–10.6 %) is not below C0′'s (8.6–9.2 %). Both sides saturate the head further than T\* (6.3 %). A wider trunk did
  not relieve the policy MLP.

**What it licenses** (design doc §6.2, NO-GO "used, buys nothing"):
- The capacity branch (T3) is **closed for growth at the plateau on the toy**: depth (X47) and FFN width (N4) both read
  without an in-house gain, within 70M.
- No N4-P on toy evidence. The toy-to-production direction is asymmetric (design doc §6.1): a toy NO-GO is weak
  evidence that production's width does not bind.
- The scale bet (strategy rank 12) loses its cheap support. It is NOT refuted: from-scratch width is a different regime
  from a plateau fork.
- K is never quoted as goal (b) progress: a post-H22-pack forum72 number on this commit's engine.

## How it was read

- **Play.** State `~/gen3ai_archive/n4_capacity_look1_2026-10-31/`, ops `…_ops/` (outside the repo). Every game
  played at `31bd9ced9` from `/home/goodlad/dev/gen3ai-reads/n4_play_31bd9ced`, CPU, nice 19 on the agent cores, both
  lanes at concurrency 1. An operator gate (`ops/gate.py`) held both lanes from each final's write until its K9(b) and
  dose records were on main, then resumed them; all six finals were admitted that way.
- **The finals.** c0n4 s3641 (10-31 20:54), n4ffn s3641 (11-01 01:15), n4ffn s3642 (11-03 12:46, the re-run), c0n4
  s3642 (11-03 16:50), c0n4 s3643 (11-03 20:51), n4ffn s3643 (11-04 01:14). Both lanes ended `done` 11-04 03:14.
- **Recorded** with main's `main.reads record` at `9c4555e38`.
- **G-F** was played by hand at the read's commit after S (`gf.sh`, `main.h2h play-many`, ledger in the ops folder),
  since the rule lists it in `not_reported`.

## FINDINGS

1. **The GPU faulted** 2026-11-01 01:19 PDT (Xid 154, reset required) 3 minutes into n4ffn s3642, with no checkpoint.
   The chain held. After the box came back, Training Oct moved the partial dir aside
   (`models/rb_x35t_n4ffn_s3642.gpufault-20261101`, kept as evidence) and re-ran the four remaining arms under the SAME
   run names, seeds, argv and pin (chain `n4-toy-2026-11-03`). The re-run's final has ONE `pin_history` span at
   `N4_PIN` (290,000,087 → 360,001,560) and a fresh growth stamp from T\* (11-03 16:23 UTC, bit-identical). The two
   admitted finals' rows were kept as they were.
2. **The box was rebooted** 2026-11-03 (~07:12 PST), killing the gate and both lanes while they waited (no unit in
   flight). Restarted 08:17 with `ops/restart.sh` (pin, clean tree and spec sha checked); the counts were identical
   before and after.
3. **Operational changes at the restart, no rule change.** (a) The lanes' stop epoch moved later, from 11-05 11:00 to
   11-06 11:00 PST, so a further GPU delay could not cut the schedule; it was never reached (both lanes ended `done`).
   (b) The gate now runs in its own systemd user scope, and the lanes it spawns inherit it, so a Claude/tmux session
   teardown (which killed other reads that day) cannot kill them.
4. **K is overdispersed across seeds** (φ̂ 2.19; C0′ s3643 0.398 against 0.445 / 0.443), so the K interval is widened
   ×1.48. K is UNDECIDED either way (unwidened [−1.90, +2.30]).
5. **CPU contention, results unaffected.** From 11-03 the S′-LIVE read held one or both heavy slots with long units, so
   N4's later finals took 1–2.5 h each to play (s3641: ≈ 35 min).
6. **home72 not played** (finding 5; §6 item 6 is conditional on room).
7. **A non-VOID canary hold** on c0n4 s3641 (u400, compiled vs eager ratio 0.538, 10-31 19:16) was the yardstick's own
   bf16 drift; u500 passed (compiled = eager). Orchestrator ruling: not a §4 VOID reason.
8. **`batch.final_class` returns None** for this record ("R1 is not final (outcome NO-GO)"), as it does for X48's: it
   reads the R1 vocabulary, and a screen record carries the screen's class. Pre-existing; nothing reads it for N4.
