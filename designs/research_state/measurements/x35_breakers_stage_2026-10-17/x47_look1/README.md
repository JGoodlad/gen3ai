# X47 look 1: the REGISTERED read (2026-10-21, CPU)

**Verdict: UNDECIDED, final at look 1 (primary Δ̂ ≤ 0), so not CONTINUE: T1b (X47 + toy C0 s3604 / s3605) does
not run.** The record of record is [`../reads/toy_x47_look1.json`](../reads/toy_x47_look1.json)
(`python -m main.x35.x47 decide read_look1.json`). `kit stage-check` now REFUSES `toy_x47` s3604 / s3605
("look 1 is UNDECIDED, not CONTINUE: no look 2").

X47 = toy T\* (X34 at 290,000,087) grown from 2 to 4 identity-init attention rounds, forked for 70M steps, seeds
3601–3603, against toy C0 (the same fork without the growth) at the toy pin `4609c155`. The plan read is launch sheet
§4.5, with AM-4, AM-X47 (-1 to -12), AM-X47b, AM-X47c and AM-RP / AM-R0 for the order.

## The verdict line (verbatim, `x47b`)

> X47b LEAVES its rank-9 conditional slot and is DEPRIORITISED in the backlog (AM-X47b-4); report: no effect >= 6.8 pp
> detected; point estimate <= 0 (look 1, σ 1.07 pp, 80 %) -- never 'depth does not help'

## Numbers

**The comparison point** (`seq_look1.json`, AM-X47-3 checks all PASS: six finished runs, K9(b) PASS, X47 unthinned).
- B = 15,453.5 s of effective learn time (the mean of the C0 finals).
- ρ = 1.136, which is above 1.02, so the primary and secondary are separate reads. The collect ratio is 1.072 and
  the update ratio 1.220.
- s_eq = 352,000,226 / 351,000,141 / 351,000,067, against C0 finals at about 360.0–360.1M.
- Eval share of Σ (reported only): 3.7–3.9 % for C0 and 3.3–3.4 % for X47.

**S, the population score** (pp, equal-weight mean over the six members of R; 63 cells × 500 mirrored pairs; every
cell OK):

| seed | C0 final | X47 at s_eq | X47 final |
|---|---|---|---|
| 3601 | 51.767 | 51.825 | 52.583 |
| 3602 | 53.217 | 50.575 | 51.925 |
| 3603 | 52.300 | 52.233 | 51.533 |
| mean | 52.428 | 51.544 | 52.014 |

| component | result | class |
|---|---|---|
| **PRIMARY**, R1 at s_eq (Welch 97.5 %, ±2.0 pp bar) | **Δ −0.883 pp [−3.202, +1.435]**, df 3.9 | UNDECIDED, Δ̂ ≤ 0 ⇒ FINAL |
| SECONDARY, R1 at matched steps (finals) | Δ −0.414 pp [−2.336, +1.508], df 3.6 | UNDECIDED |
| G-F, head-to-head on the 3 × 3 grid (s_eq × C0 finals, `cross_stat`, 95 %) | Δ̂ +0.02 pp [−2.84, +2.89], df 4 | PASS |
| G-C, pooled over the 3 seed pairs of finals (crossed cluster bootstrap, toy truth `toyTstar@290M`) | B1 −0.0050 nats [−0.0217, +0.0137]; B2 −0.024 [−0.076, +0.026] | PASS |
| G-K | not registered for this read (UNVERIFIED on the toy, X37c) | — |

**Reported, never deciding:**
- AM-X47-5's per-step gap is not triggered (it applies to PER-STEP ONLY only). It is still computed: D (S_final −
  S_s_eq) = +0.47 pp [−2.15, +3.09], labelled BOUNDARY.
- AM-X47-7, per-member Δ at s_eq (97.5 % Welch):
  - toy_pstar −0.57 [−6.40, +5.27];
  - toy_280m +1.00 [−2.16, +4.16];
  - toy_220m −2.22 [−5.58, +1.14];
  - toy_150m −0.32 [−7.64, +7.01];
  - toy_080m −2.07 [−4.84, +0.71];
  - out of lineage, **pstar_x26 −1.13 [−4.76, +2.49]**.
  - No lineage label applies, because the primary is not a GAIN.
- AM-X47-10, the weight norms (`norms_s360x.json`): the final ratios of both new rounds to the base rounds are about
  0.10–0.12 on every seed, so both rounds read "engaged". This descriptor is weak (F-AMX-4).
- AM-X47b-3, the policy difference (`policy_diff_s360x.json`, every 10th node plus the final): the mean KL on
  `bank_x26` between each node and the same weights with the new rounds zeroed.
  - It rises steadily on every seed: 0.05 nats at 300M, and 0.155 / 0.194 / 0.188 at the finals.
  - The argmax moves on 25–29 % of the 7,028 decisions.
  - So the extra rounds change the policy materially, yet S does not rise.

## How it was run

- **G-C:**
  - `main.policy_spectrum read` on the six finals, on `bank_x26`;
  - then `python -m main.x35.gc pool --side toy --truth …/x35_toy_b2_truth_2026-10-20/rows_toyTstar290M_S32… --pair
    360x=X47_s360x:C0_s360x`, which wrote `gc_pooled_look1.json`.
- **Cells:**
  - [`play_cells.py`](play_cells.py) ran 63 cells, each its own `main.h2h play-many` invocation, under the four
    registered requests `x35_x47_look1_{a,gf,b,c}`;
  - registered settings: family `x35_x47` (registered with commit `4609c155`), purpose `ab`, 500 pairs, seed 0,
    label `x35_x47_look1`, greedy against greedy;
  - three CPU lanes (cores 5-7,13-15, `mem_cap.sh`, `GEN3AI_BLOCKING`), 06:47–07:36 PDT. A cell took about 1.5 min
    (toy against toy) or about 6 min (against P\*).
  - The cell rows are `ledger_x47_look1.tar.gz` (the ledger root, played OUTSIDE `models/`).
- **The read JSON:** [`assemble_read.py`](assemble_read.py) wrote `read_look1.json`:
  - `population_score` over the union of `_a` / `_b` (X47) and `_c` (C0);
  - `matrix_from_cells(min_pairs=500)` for G-F;
  - the G-C record and `seq`.

## FINDINGS (standing rule 7)

- **F-X47L1-1, the record's file name.** The launch sheet §0 / §4.5 writes the record as `reads/x47_look<k>.json`,
  but `batch.read_record` (and so `kit stage-check`) reads `reads/<kit arm>_look<k>.json` = `reads/toy_x47_look1.json`.
  Under the sheet's name, stage-check still said "look 1 is not read yet". The record is saved under the kit's name.
- **F-X47L1-2, one invocation per cell.** §4.5 says each engine's cells are ONE `play-many` call. Each cell was played
  as its own invocation under the same request, to keep within the 45-minute chunk bound. The schedule key is the
  order-independent digest of the two checkpoints and the batch is the default 500, so the rows are the ones a
  single call would write.
- **F-X47L1-3, the ledger root.** The read was played into `~/gen3ai_archive/x47_look1_2026-10-21/ledger`, not
  `models/_ledger`, because this unit was read-only on `models/`. A copy is committed here.
- **F-X47L1-4, the playing commit.** The cells were played at main `f4b0a74f`, not at the toy pin `4609c155` (the
  family's declared commit), as every h2h read is. The rows record the commit they were played at.
