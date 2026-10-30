# X48 (per-token-kind FFN experts at the plateau) look 1: **NO-GO, "experts not engaged"** (2026-10-30)

X48 grows each trunk round's shared FFN into five per-token-kind experts at P\* and trains + 20M (`--grow
trunk_ffn_by_kind=on`, pin `f4b25238`, seeds s3501–s3503). The control is C0 s3501–s3503 at the same pin,
re-played. The read is rule `n4_screen` (`main.reads`, [`PREREG.md`](PREREG.md) §5), spec
[`read_spec.json`](read_spec.json) (sha256 `94bb64dc…`), registered in NOTE 4 (`efe8ea1fa`) before any X48 read
game. The record is [`../x35_breakers_stage_2026-10-17/reads/x48_look1.json`](../x35_breakers_stage_2026-10-17/reads/x48_look1.json).

| | s3501 | s3502 | s3503 | mean |
|---|---|---|---|---|
| S, X48 (pp; 7 members × 2,000 pairs, se ≈ 0.28) | 56.89 | 56.05 | 56.76 | 56.57 |
| S, C0 (pp; re-played, the same cells) | 57.13 | 57.11 | 57.11 | 57.12 |
| K on forum72 vs Kakuna, X48 (1,000 games each) | 0.524 | 0.530 | 0.525 | 0.526 (1,579 / 3,000) |
| K on forum72 vs Kakuna, C0 | 0.500 | 0.530 (980 games) | 0.488 | 0.506 (1,507 / 2,980) |
| mean LR (`snapshot_history`), X48 / C0 | 1.065e-4 / 1.055e-4 | 1.065e-4 / 1.025e-4 | 1.015e-4 / 1.025e-4 | ratio 1.013 |
| trailing approx KL (median of the last 21 rollouts), X48 | 0.0191 | 0.0189 | 0.0183 | (C0 0.018, LR-F's README) |

- **Δ_S −0.55 pp [−1.32, +0.21]** (Welch 90 %, df 2.0). UB < +2.0, so **NO-GO-S**.
- **Δ_K +2.06 pp [−0.06, +4.19]** (Newcombe at z 1.6449, φ̂ 0.93, so not widened). It is neither GO-K (LB ≤ 0) nor NO-GO-K
  (UB ≥ +1.5), so **UNDECIDED-K**. Not MIXED.
- **Class: NO-GO** (§5.3 item 4). The registered text: "no ≥ 2 pp in-house gain from per-kind FFN experts within 20M
  of P\*, with the experts' AdamW state copied from the shared FFN (Q-K3)"; "no unfamiliar-only effect ≥ ≈ 4 pp
  detected; smaller ones are not excluded". The family false GO is ≈ 0.09.
- **Labels.**
  - **"experts not engaged" APPLIES** (NOTE 1 (b), fixed in NOTE 4 (b) before any game): at the final, D_k / null is
    0.72–0.93 for every non-Pokémon kind in rounds 0–1, all ≤ 1.0 ([`expert_divergence.json`](expert_divergence.json)).
    The experts diverged no more than two copies of C0's own FFN drift would. This is a WEAK NO-GO.
  - "LR-confounded" and "LR-elevated" do not apply (ratio 1.013).
  - NOTE 2's "read at the controller's learning rate" does not apply (LR-F look 1 is UNDECIDED, final).
- **Guards.**
  - The VOID gate (K9(b), pin, pool seed) passes on all six ON labels.
  - VOID 4 exact passes ×3 ([`void4_growth_stamp.json`](void4_growth_stamp.json)).
  - The KL max is 0.0205 against 0.08.
  - The re-derived anchor guards: `count`, 300 units checked, 0 hits.
  - One engine per meter: K `port=7d932165…/90;env=7c532805…/125` (E1), S `env=7c532805…/125`.
  - K counted 299 of 300 units. One C0 unit (`c0_s3502__forum72__07`) is uncounted: Metamon's long-tail fault (SOP H5)
    hit both attempts. That is inside `d1`'s floor (at most one per arm).

**Reported, never deciding** (§8; no adjectives):
- **R1 at look 1** (97.5 %): UNDECIDED, Δ −0.55 [−2.18, +1.07].
- **Per-member Δ_S** (Welch 90 %):
  - `tstar` −1.60 [−3.15, −0.06], `x26_270m` −1.03 [−2.69, +0.63], `x26_140m` −0.72 [−2.97, +1.53],
    `x26_pstar` −0.50 [−1.96, +0.95];
  - `production`, `x26_080m` and `x26_210m` within ±0.04 pp of 0.
- **Pin drift:** the re-played C0 S is 57.12 against G1's banked 56.98 at 500 pairs, +0.14 pp. Both are on E1, so the
  difference is pair count and harness only.
- **σ:** S's seed SD is 0.45 pp (X48) and 0.011 pp (C0). The S screen's power at the observed σ is GO-S 0.996 at
  +2 pp and NO-GO-S 0.945 at 0.
- **ρ** (dose records): wall 255.9 / 248.7 min, so 1.03; 1,375 / 1,409 steps/s.
- **Not played:** the slope (§8.4) is INCOMPLETE, paused under the coordinator's CPU ruling (NOTE 5 (b)); it decides
  nothing. G-F (§8.5) and home72 (§8.8) are not in the spec.

**What it licenses** (§9; precommit memo §2.6, J14 under LR-F "UNDECIDED, final"):
- It closes per-kind FFN untying at the plateau for production, within 20M, and only weakly, because the copied AdamW
  moments throttled the experts (review M1).
- It does not close FFN width (N4), `D_MODEL` or from-scratch capacity.
- N4 with its review's fixes, or N4-P-direct, is the orchestrator's choice.
- The natural X48 follow-up is the review's reset-v variant: fresh second moments, a new pin, a new registration.
- X48's K is never quoted as goal (b) progress. It is a post-H22-pack F number on E1, and it never sits beside a
  legacy-pack F.

## How it was read

- **Pre-read gates (NOTE 4):** VOID 4 exact, the experts' divergence and the mean LR. Then the S-lane identity at the
  pin: C0 s3501's G1 batch 0 against `x26_pstar` and `tstar` was [`s_identity.json`](s_identity.json) IDENTICAL.
- **Play.** The state is `~/gen3ai_archive/x35_x48_read/look1/`. Every game was played on E1 at the X35-frame shared
  play pin `86cfdf7ef`, from `/home/goodlad/dev/gen3ai-reads/x48_play_86cfdf7e`. Both lanes ran at concurrency 1,
  CPU, nice 19 on the agent cores. They were HELD 10-28 12:47–18:21 for D3 look 2.
- **The S driver (NOTE 5).** An engine at the pin serves at most two architectures, and the runner striped X48's
  members across engines. So X48's S cells (and the slope cells) were played by [`x48_s_driver.py`](x48_s_driver.py)
  (sha256 `ddc18239…`). It is the runner's loop with members grouped by architecture.
  [`driver_identity.py`](driver_identity.py) (sha256 `119cd957…`) showed it reproduces the runner's banked rows exactly
  ([`driver_identity.json`](driver_identity.json)).
- **Recorded** with main's `main.reads record` (`rederived_guards` resolves to `count` there; NOTE 4 (f)).

## FINDINGS

1. **The registered S lane cannot play a third architecture at the pin** (h2h `MAX_GROUPS` = 2;
   `runner.s_engines` stripes members). Worked around without a change to the measurement (NOTE 5, its identity in
   5 (a)). A read that grows an architecture needs an architecture-aware `s_engines` on main.
2. **The C0 replays' three S means agree within 0.02 pp** (σ 0.011 pp against a 0.275 pp measurement SE per final).
   Their per-member rows differ (72.1–72.8, 59.3–60.7, …), so it is chance. It leaves Welch's df ≈ 2, carried by
   X48's spread alone.

   Sensitivity (descriptive): with σ_C0 at the measurement floor (0.275) the UB is ≈ +0.14; at the PREREG's
   pessimistic 1.07 it is ≈ +1.10. Both are below +2.0, so NO-GO-S holds at any plausible C0 seed spread.
3. **C0 has used its one-unit K allowance** (the H5 fault). A second exhausted C0 unit would have read K INCONCLUSIVE.
4. **The driver gives one engine to a label whose pending members share one architecture.** The tail of a label
   therefore ran at half speed. It is scheduling only.
