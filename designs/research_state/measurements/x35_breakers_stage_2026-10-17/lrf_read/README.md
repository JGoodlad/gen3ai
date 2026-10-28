# Production LR-F (the learning-rate floor, STAGED_PLAN §4.1.2) look 1: **UNDECIDED, FINAL** (2026-10-28)

LR-F = C0 + `--min-lr 1.7361e-4`, P\* + 20M at pin `f4b25238`, seeds s3501–s3503. The control is C0 s3501–s3503 at
the same pin, its S imported from G1. Record: [`../reads/lrfloor_look1.json`](../reads/lrfloor_look1.json), rule
`x35_r1` (`main.reads`), spec [`read_spec_look1.json`](read_spec_look1.json) (sha256 `5d32e464…`), registered with
AM-LRF-L1 (STAGED_PLAN §4.1.3, `f7fddc39b`) before any LR-F game.

| | s3501 | s3502 | s3503 | mean |
|---|---|---|---|---|
| S, LR-F (pp, se ≈ 0.55) | 55.98 | 55.56 | 55.99 | 55.85 |
| S, C0 (pp, G1's) | 57.52 | 56.60 | 56.81 | 56.98 |
| K9(b) / KL guard / pin / pool seed | PASS ×4 | PASS ×4 | PASS ×4 | |
| updates at the floor | 100 % | 100 % | 100 % | |
| approx KL, LR-F / C0 | 0.0236 / 0.0185 | 0.0237 / 0.0183 | 0.0236 / 0.0180 | 0.0236 / 0.0183 |
| clip fraction, LR-F / C0 | 0.211 / 0.171 | 0.211 / 0.169 | 0.210 / 0.166 | 0.211 / 0.169 |
| N_prom, LR-F / C0 | 1 / 1 | 0 / 1 | 1 / 1 | 2 / 3 |
| greedy churn vs P\* on `bank_x26`, LR-F / C0 | 0.333 / 0.306 | 0.311 / 0.310 | 0.329 / 0.306 | 0.325 / 0.307 |
| GPU-h, LR-F / C0 (`../dose/`) | 4.12 / 4.43 | 4.07 / 4.03 | 4.13 / 4.01 | 4.11 / 4.16 |

- **R1 Δ −1.13 pp [−2.45, +0.18]** (Welch 97.5 %, df 2.96). The interval crosses −2.0 and 0, so the class is
  UNDECIDED. It is FINAL because Δ̂ ≤ 0, so there is no look 2 (`kit stage-check` refuses s3504). Every LR-F seed
  scored below every C0 seed. The prediction was 0.0 pp, 80 % [−2.0, +1.5], with UNDECIDED the modal class.
- **Guards:** G-F, G-C and G-K were not read, because R1 does not qualify (A1 ruling (a); AM-LRF-L1 item 5). G-E is
  absent by design. No Kakuna game was played.
- **Mechanism, against the registered predictions (reported only):**
  - Updates at the floor: 100 % (predicted ≥ 95 %). HIT.
  - Arm-mean approx KL: 0.0236 (predicted [0.025, 0.045]). MISS: the per-update policy step rose only ×1.29 for a
    ×1.73 larger rate.
  - Clip fraction: 0.211 (predicted 0.20–0.30). HIT.
  - Churn: 0.325 (predicted 0.33–0.42). Narrow MISS; C0's is 0.307.
  - N_prom: 2 against C0's 3 (predicted within ±2). HIT.
- **Dose:** equal (rel. gap 1.2 %), so there is no throughput confound.

## How it was read (CPU, `nice 19` on cores 5-7,13-15)

- **Play pin.** Every LR-F game was played on E1 at the X35-frame shared pin `86cfdf7ef`, from a clean detached
  worktree `/home/goodlad/dev/gen3ai-reads/lrf_play_86cfdf7e`. That worktree carried E1's untracked `Cargo.lock` blobs
  and Showdown `e0551883` (= main's and the bank commit's). All 105 S rows (3 finals × 7 members × 5 batches) carry env core `7c532805…/125` and regime
  `adfdbee824c9eb0a`.
- **The S-lane identity check, before any LR-F game.** C0 s3501's G1 batch 0 was re-played at the pin against
  `x26_pstar` and against `tstar`. Both cells were equal on regime, seed, counts and both outcome digests
  ([`s_identity.json`](s_identity.json); [`run_identity.sh`](run_identity.sh) played them and
  [`s_identity_compare.py`](s_identity_compare.py) compared them).
- **Instruments.** [`mechanism.py`](mechanism.py) → [`mechanism.json`](mechanism.json) reads each run's own TB rows
  and `snapshots/`. [`churn.py`](churn.py) → [`churn.json`](churn.json) reads a `main.policy_spectrum read` of
  `bank_x26` with main `407abd835`'s `src/` (the reader's checkout).
- **State, read-only.** The state is `~/gen3ai_archive/x35_lrf_read/look1/` (ledger, cells, inputs, decision,
  void). `models/` was only read.

## FINDINGS

1. **The KL prediction missed low.** Raising the rate ×1.73 over C0 raised the per-update approx KL only ×1.29
   (0.018 → 0.024). That is under the predicted KL ∝ LR band. The step grew sub-linearly in the rate, and the policy
   change per 20M (churn) barely moved (0.307 → 0.325). The lever acted on the optimiser more weakly than designed.
2. **The S lane's first two identity attempts played no game.** The first failed on the unregistered eval-ledger
   family; the second on the play worktree's missing `deps/pokemon-showdown` (team validation). `main.reads`'s own S
   lane registers its family, and a fresh pin worktree needs `scripts/bootstrap.sh` (its Rust stamp test then fails
   on the restored `src/rust_sim`, harmlessly for S).
3. **A GAIN could not have been read to ADOPT by `x35_r1`.** That rule in `main.reads` takes no G-F, G-C or G-K
   input. It did not bind here, but a future batch-2 GAIN needs a guard reader first.
4. **Churn was read with main's reader on the old bank** (`bank_x26`'s recorded observations). Both arms were read by
   the same code against the same P\* read, so the comparison is matched. The absolute level is comparable to the
   X26 battery's ≈ 0.30 only descriptively.
