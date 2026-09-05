# Teacher-content 2×2 — funded vs unfunded teachers, and two frozen replicate pairs

**Results are NOT in this directory yet.** The four arms finished 2026-09-04 19:46; untaught scoring
started automatically at 19:47 and lands in its own entry. This README records the batch's
*provenance* — what was run, what is clean, and the two bookkeeping defects a reader will otherwise
misread.

## The design

Four arms, forked from `ai_v9_59_R2ACTION_0827` at step 28,115,184 and trained to 32,567,760
(span 4,452,576 — identical to the reuse batch, so `p1M`/`mid`/`end` denote the same training depth
across both sets).

| arm | run | teachers | pin |
|---|---|---|---|
| TC_FUND_A | `ai_v9_160_TCFUNDA_0903` | the 8 **funded** R5FUND forks | `0c76e2ee` |
| TC_UNF_A | `ai_v9_162_TCUNFA_0903` | their 8 **unfunded** R5F parents | `52ab5914` |
| TC_FUND_B | `ai_v9_161_TCFUNDB_0903` | funded (replicate) | `52ab5914` |
| TC_UNF_B | `ai_v9_163_TCUNFB_0903` | unfunded (replicate) | `52ab5914` |

Both halves resolve to the **same 16 teams**. Within a half the two arms differ in exactly one argv
token (`--run-name`), so each half's spread is a **draw**, not an effect — which is what makes this
batch yield two independent frozen-regime replicate floors as well as the contrast. Every arm:
`coef 0.1761`, `K=3`, `fork_lr 2.8e-05`, `--fork-lr-freeze`, dose `4.5572917e-08` (2.12× v8), pool
14 snapshots / 90.12% / 90% — all four pool lines character-identical.

Arms ran **interleaved** (FUND_A, UNF_A, FUND_B, UNF_B) so that 24 hours of box drift loads on both
halves symmetrically rather than onto the contrast.

## What is clean and what is not

| comparison | pins | |
|---|---|---|
| FUND replicate pair | `0c76e2ee` / `52ab5914` | **pin-split** |
| UNF replicate pair | `52ab5914` / `52ab5914` | **clean** |
| contrast leg A (FUND_A−UNF_A) | split | **pin-split** |
| contrast leg B (FUND_B−UNF_B) | same | **clean** |

Main moved between arm 1 and arm 2 (operator error — see the ledger entry dated 2026-09-04); it was
frozen immediately afterwards, so arms 2–4 share a commit. The difference was verified inert four
ways, and the resulting structure leaves **one fully clean contrast leg and one fully clean frozen
replicate pair**, with the drift confined to the FUND_A cells. **Leg A vs leg B is therefore an
empirical check on the inert verdict** — agreement corroborates it independently of the diff
reading; disagreement makes the pin the first suspect.

## Two bookkeeping artefacts that mean less than they look like

**`tc_failed_arms.txt` holds THREE `PIN_DRIFT` lines but records ONE incident.** The chain's
`EXPECT_PIN` was set to arm 1's commit, so every arm correctly running on the frozen commit reads as
a mismatch against it. Do not count three incidents.

**The chain's closing line says `1/4 arms clean`. It is wrong as a health statement** — same cause,
it counts those pin mismatches. On the only gate that matters, `final_model.zip`, it is **4/4**.

**Arm 1's `metadata.json` has no `distill_anchor_monitor_source` field** (it predates the default-on
build `3f2e8a14`) while arms 2–4 record `cli`. A *schema* difference, not a behavioural one — the
instruments were carried by explicit flags on every arm, and all four DISTILL-ANCHOR / DISTILL-STOP
startup lines are character-identical. Anything diffing the four arms' metadata should expect it.

## Cost

Measured: `FUND_A 5.97 h · UNF_A 5.88 h · FUND_B 5.92 h · UNF_B 5.92 h` = **23.68 GPU-h** against
~18 approved (+32%). Spread across arms 0.09 h — the overrun was the initial per-arm estimate, not
variance.

## Reading the results

`tc_readout.py` (committed here). It labels every comparison PIN-SPLIT or pin-clean from each arm's
recorded `git_hash`, reports the frozen-only floor beside the 3-draw floor so the
controller-live/frozen regime assumption stays visible, and refuses to pool the controller-live
N1/N2 draws as a frozen floor.
