# Learner battery + C_fix + G0′ plateau — the READ (2026-09-29)

Registration: [`../../learner_battery_2026-09-26.md`](../../learner_battery_2026-09-26.md) (§4 rules) and
[`../../new_lineage_2026-09-26.md`](../../new_lineage_2026-09-26.md) §3.1 (the plateau rule). Rows: the N0
end-of-run CPU queue (`../n0_endofrun_2026-09-27/scripts/n0_queue.py`, state dir
`/home/goodlad/dev/gen3ai-reads/n0_endofrun_2026-09-27/`, queue tree `d4799e8a` — identical to `2cc83080` on
every read path, declared in that kit's README §4). Read 2026-09-29 ~13:05 PT, CPU only, model-free.

**Codes.** **N0** = `ai_v14_01_base` final @75,005,952 (the new lineage's 75M base). **C** = `ai_v14_02_lbat_ctrl`
(the control, N0 + 8.06M, pin `2cc83080`, BUGGY compiled learner). **E5** = `ai_v14_03_lbat_e5` (5 PPO epochs at
2× the lr = same dose, buggy learner). **L95** = `ai_v14_05_lbat_l95` (policy GAE λ 0.95, buggy learner).
**T32** = `ai_v14_04_lbat_t32` (TF32; crashed at the compile gate, NOT RUN). **T32b** = `ai_v14_04b_lbat_t32`
(TF32 on the fixed code, pin `8fc297a2`; STOPPED by futility at 76,307,136). **C_fix** (`Cfix` in the rows) =
`ai_v14_06_lbat_ctrl_fix` (C's argv at pin `6521f420`, the compile fix). **K2** = `ai_v14_07_g0p_k2` (C_fix +
8.06M, pin `a5b2fdba`). **K3** = `ai_v14_08_g0p_k3` (K2 + 8.06M, pin `a5b2fdba`). **U** = the untaught-8 meter
(8 teams vs N0's 24M snapshot, 600 games/team, seed 0). **G-A** = SmallRL greedy vs greedy (1,200 games).
**S** = projected speed gain (fraction of GPU time per step saved vs C).

**Scripts (this directory):** `read/battery_read.py` (U + G-A, output `read/battery_read.{txt,json}`),
`scripts/speed_read.py` (output `read/speed_*.{txt,json}`), `scripts/battery_rule.py` (output
`read/battery_rule_verdicts.txt`). U uses the meter's own `bootstrap_index` / `cluster_ci` (20,000 draws,
**registered seed 20260915**, one paired index set; the engine default seed 20260906 is a sensitivity row and
moves no bound by more than 0.05 pp). Every U cell: 0 timeouts. Every G-A unit: status OK, n = 100,
regime-verified per decision, team sources symmetric.

**Argv audit (the `original_command`s, read-only):** E5 differs from C in `--n-epochs 5 --fork-lr 5.6e-05`
only; L95 in `--policy-gae-lambda 0.95` only; C_fix in `--pin-commit` only; T32b in `--matmul-precision high`
+ pin; K2/K3 in `--model`, `--steps`, pin. All seed 1001, all FORKs with the lineage recorded by
`main.lineage`. `2cc83080..6521f420` changes, on the training path, only the CUDA-compile split + the
real-obs gate (the phase hook is `None` in production; ladder recipe v3 is the detached rating
subprocess); `src/rust_sim/`, `src/poke_env/` and `data/` are identical (no `data/` commit since 09-25).
C_fix, K2 and K3 logged `[CompileTrainer] parity PASS on 64 REAL obs rows` at every start.

---

## 1. Levels (600 games/team, cluster mean [95%], pp)

| ref | U | per team (U_61590463 … U_f7ba5702, sorted keys) | G-A SmallRL (1,200) |
|---|---:|---|---:|
| C | 74.48 [72.44, 76.44] | 76.5 77.8 70.8 75.7 73.0 73.3 70.2 78.5 | 0.640 (768) |
| E5 | 75.23 [74.48, 76.00] | 75.3 75.3 75.2 77.0 74.3 76.7 73.5 74.5 | 0.627 (752) |
| L95 | 64.38 [62.50, 65.88] | 66.7 67.2 62.8 58.8 65.0 64.8 65.8 63.8 | 0.631 (757) |
| C_fix | 77.54 [75.42, 79.92] | 75.8 76.2 74.3 83.5 79.5 75.7 73.8 81.5 | 0.688 (825) |
| K2 | 78.27 [76.92, 79.75] | 76.7 76.5 75.8 81.5 81.0 77.7 77.0 80.0 | 0.690 (828) |
| K3 | 78.48 [75.98, 81.27] | 77.3 75.8 74.8 84.7 80.2 74.8 75.7 84.5 | 0.671 (805) |
| N0 | 63.38 [61.38, 65.25] at 200/team (1014/1600) | queue still running (571–595/team at read time) | not yet run (0/12 units) |

## 2. Learner battery (E5, L95 vs C — all three on the BUGGY compiled learner, internally comparable)

| arm | S (projected) | U Δ vs C | U class | G-A Δ vs C (Newcombe) | stall ratio | VERDICT (`battery_rule.py`) |
|---|---|---|---|---|---:|---|
| E5 | **+13.59 % [+13.38, +13.71]** (+15.7 % steps/GPU-h; update 56.9 → 32.2 s) | +0.75 [−1.21, +2.65], 5/8 teams up | EQUIVALENT; non-inferior | −1.33 [−5.18, +2.52] | 51.9 / 47.3 = 1.10 | **ADOPT** |
| L95 | −1.18 % [−1.70, −0.61] (descriptor) | **−10.10 [−12.77, −7.65]**, 0/8 teams up | **OUTSIDE, BELOW** | −0.92 [−4.76, +2.93] | 52.4 / 47.3 = 1.11 | **NOT ADOPTED** (and a FINDING: 0.80 beats 0.95) |
| T32 | — | — | — | — | — | **NOT RUN** (startup crash at the compile gate; that crash found the miscompile) |
| T32b | +2.48 % [+2.24, +2.91] vs C_fix, 12 rollouts | units never ready (no `final_model.zip`) | — | — | — | **STOPPED — futility** (< 2.5 %), NOT ADOPTED (speed) |

Descriptors: U(X) − U(N0) on the common battle set (571–595/team, PROVISIONAL until N0's 600 land): C +9.88
[+8.77, +11.07], E5 +10.77 [+9.62, +11.98], **L95 −0.22 [−2.77, +1.75]** (L95 gained nothing untaught in 8M),
C_fix +13.31 [+11.18, +15.49]. Per-epoch approx-KL at the last epoch: C 0.0039, E5 0.0033; last-epoch clip
fraction C 0.077, E5 0.065 (clipping rare at D_g = 2.8e-5, as F-3 predicted). L95's last-10-rollout
`train/value_loss` median 0.045 vs C 0.012 (the frozen scalar head's returns move with the policy λ, so this
is NOT evidence about the win-prob critic — mechanism UNVERIFIED). L95's measured iteration was 1.16× C's;
episode length median 47.0 vs 45.9. G-A by teamset: no split clears zero for E5 or L95.

**Not read (registered descriptors):** the matched-GPU-time read U(E5_final) − U(C@s_T) (needs a C 500k
checkpoint's untaught read; it is underpowered and never ruled on) — **NOT RUN**.

## 3. Bug effect: C_fix − C (same parent N0, same argv, only the compile fix)

| instrument | C_fix − C | vs 0 | vs the floor |
|---|---|---|---|
| U, 600/team | **+3.06 [+1.00, +5.15]**, 6/8 teams up | CI excludes 0 | WITHIN FLOOR (3.69; the point is below it, the CI's top is above it) |
| U, first 200/team | +3.75 [+0.44, +6.38] | CI excludes 0 | at the floor, not OUTSIDE |
| G-A SmallRL, 1,200 each | **+4.75 [+0.97, +8.51]** (0.688 vs 0.640) | CI excludes 0 | inside the 11.0 floor |
| speed | C_fix projected S = −1.50 % [−1.60, −1.35] vs C (update 56.9 → 59.7 s) | the split's cost | matches the ledger's predicted +1.5–4 % of fwd+bwd |

Cumulative, fixed lineage vs buggy C: K2 − C +3.79 [+1.63, +5.90], K3 − C +4.00 [+1.60, +6.31].
**Caveat:** each CI is game + team-cluster noise only. C and C_fix are one training run each, and removing
the miscompile changes the whole trajectory, so run-to-run (seed-like) variance is NOT inside these
intervals. That is what the 3.69 replicate floor stands for, and why a +3.06 point is WITHIN FLOOR even with
a CI clear of zero. The two instruments agree in sign and are independent (pool teams vs N0@24M, and an
external Metamon opponent on its own teams).

## 4. G0′ plateau (new_lineage §3.1: PLATEAU iff |Δ| < 3.69 AND not OUTSIDE ABOVE; G0′ = end of the first passing block)

| block | Δ at 200/team (the lineage's G-U n) | Δ at 600/team | class | PLATEAU? |
|---|---|---|---|---|
| **1** N0 → C_fix | **+14.88 [+10.69, +18.88]**, 8/8 up | +13.31 (common set, provisional) | OUTSIDE, ABOVE | **NO** |
| **2** C_fix → K2 | **+0.87 [−1.75, +3.56]**, 4/8 up | +0.73 [−0.44, +1.81] | EQUIVALENT | **YES → G0′ = K2's end** |
| 3 K2 → K3 (ran before the verdict) | −0.31 [−3.38, +2.94] | +0.21 [−1.25, +1.87] | EQUIVALENT | yes (confirms) |
| (registered block 1, buggy) N0 → C | +11.12 [+9.12, +13.12] | +9.88 (common, provisional) | OUTSIDE, ABOVE | no |

**G0′ = `models/ai_v14_07_g0p_k2/final_model.zip` @91,127,808** (PLATEAUED at block 2 of the 3-block cap).
Its first untaught read at 200/team, for the lineage's per-round reproduction check (§3.2): **1266/1600 =
79.12 %** (cells j < 200 per team, seed 0, opponent N0@24M). G-A across the blocks: K2 − C_fix +0.25 [−3.45,
+3.95], K3 − K2 −1.92 [−5.64, +1.81]. K3 − C_fix = +0.94 [+0.17, +1.73] over two blocks (inside the floor).

**Deviation, recorded:** the registered block 1 was C (K1 ≡ C, battery §7). C trained under the miscompile,
so block 1 was replaced by C_fix, and K2 forks C_fix, not C. The substitution does not change which block
passes: the buggy block 1 also fails (OUTSIDE ABOVE).

## 5. P2 side-peek (V's within-game discrimination)

Not possible from these rows: the untaught rows carry only the outcome per battle, and the anchor rows carry
turn and decision counts. Neither carries a per-decision V.

## 6. Hazards / findings

- F-R1: every U CI is a cluster bootstrap over 8 teams. Its percentile intervals are discrete and narrow
  (E5's is ±1.9 pp at 4,800 games). Training-run variance is not included (see §3).
- F-R2: N0's own U (600/team) and G-A are still in the queue. Every N0-relative number at 600 is provisional.
  The plateau's block-1 verdict uses the 200/team cells, which are complete and final.
- F-R3: the T32b futility entry had no ledger line before this read. It is recorded here and in the battery
  ledger entry.
- F-R4: E5's ADOPT was read on the buggy learner, at D_g = 2.8e-5 (clip rare). The miscompile entry says
  levers read inside that span "are internally comparable arm-vs-arm yet may not transfer". So E5's
  transfer to the fixed learner is UNVERIFIED. It is checkable at no extra cost: the first E5-carrying block's
  untaught Δ against the §7 combination/regression guard.
