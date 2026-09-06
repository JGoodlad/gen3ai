# Teacher-content 2×2 — funded vs unfunded teachers, and two frozen replicate pairs

**The results ARE in this directory** (since 2026-09-06). The four arms finished 2026-09-04 19:46;
untaught scoring started automatically at 19:47 and its readings are banked in the ledger entries
dated 2026-09-04/05. This README records the batch's *provenance* — what was run, what is clean, and
the two bookkeeping defects a reader will otherwise misread.

**Where the numbers live, and the defect that is now closed.** The per-team artifacts every banked
funded / unfunded / K=6 untaught and taught number rests on existed for two days ONLY inside a
session-scoped job directory (`~/.claude/jobs/1046b1d6/tmp/probes/`) — one cleanup from gone, and
`tc_readout.py` could not run as committed because it reads them from its own directory, where they
were not. The teacher-distance probe rescued them into
`arch_transfer_2026-09-05/teacher_distance/inputs/` on 2026-09-05 (its hazard 1); on 2026-09-06 they
were moved HERE, to the batch that produced them, so there is exactly ONE copy in the tree:

| here | what |
|---|---|
| `untaught_{TCFUNDA,TCFUNDB,TCUNFA,TCUNFB,TCUNFK6A,TCUNFK6B}_{p1M,mid,end}.json` | the 8 untaught teams × 200 games, per arm per depth (the 2×2's 96 cells and the K=6 cell's 48) |
| `taught_{…}_end.json` + `taught_R2ACTION_end.json` | the 16 taught teams at the endpoint, and the PARENT baseline the +4.98 / +4.48 gains are measured against |
| `untaught_probe.py` · `taught_probe.py` · `untaught_teams.json` | the probes that produced them and the untaught team list |
| `tc_readout.py` · `taught_readout.py` · `recovery_readout.py` · `k6_readout.py` | the four readouts — each runs as committed from this directory, each takes `--check`, each is in the gate's registry (see *Reading the results*) |

The reuse-batch arms it also reads (`N1` / `N2` for the controller-live floor, `C1` / `B2`) stay in
`reuse_batch_2026-09-03/` and are resolved there by name, so no file is duplicated.
`arch_transfer_2026-09-05/teacher_distance/fold_table.py` reads the endpoint and p1M artifacts from
here as well. `src/measurements_readout_gate_test.py` fails if any of them goes missing again.

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

## Reading the results — FOUR readouts, one per family of banked number

Every banked number from this batch (and from the K=6 cell, whose artifacts live here too) is now
re-derivable by running a committed script from this directory. Each takes `--check`, which resolves
its declared inputs without computing anything, and each is registered in
`src/measurements_readout_gate_test.py`, which fails if an input ever goes missing again.

| script | what it proves | banked headline it reproduces |
|---|---|---|
| `tc_readout.py` | the UNTAUGHT funded−unfunded contrast at three depths, and the six frozen replicate draws that set the floor | frozen floor **1.66**, contrast p1M +0.16 WITHIN FLOOR · mid **−6.12** · end **−4.37** |
| `taught_readout.py` | the TAUGHT-16 side of all six arms vs the fold parent — did the fold teach anything, does teacher funding teach more, does halving the dose cost teaching | FUNDED **+5.11** · UNFUNDED **+4.86** · pooled **+4.98** · funded−unfunded **+0.25** WITHIN FLOOR · K=6 **+4.48** · K=6−K=3 −0.38 |
| `recovery_readout.py` | the UNTAUGHT arm-vs-parent LEVELS at p1M / mid / end, and the within-arm p1M→mid/end change (the parent cancels exactly) | halves p1M −3.12 / −3.28 · mid −4.88 / +1.25 · end −2.41 / +1.97; **UNFUNDED−FUNDED recovery +6.28 [+3.16, +9.81] and +4.53 [+1.94, +7.41]** |
| `k6_readout.py` | the K=6 (v8-dose) cell's pre-registered P1 / P2 / P3 | P1 end **−0.22** (parent-neutral), p1M −4.19 · P2 dose null at every depth · P3 the K=6 floor **2.46**, not small, not depth-stable |

**Closed by these files:** the 2026-09-06 bookkeeping entry's finding 1 (`tc_readout.py`'s bars) and
finding 3 (the three families with artifacts but no aggregating script).

**Point estimates reproduce EXACTLY; intervals reproduce to ≤0.12pp.** Every point estimate is
seed-free and matches the ledger digit for digit. The intervals are cluster bootstraps and the seed
the ad-hoc session used was never recorded — the four scripts declare their own (`SEED = 20260904`),
and across every seed tried the ends move by at most 0.12pp on the untaught statistics and 0.07pp on
the taught ones, which is 1–2 steps of each statistic's own resolution grid (0.0625pp on 8 teams ×
200 games, 0.031pp on 16). No interval's verdict depends on the difference.

### The bar each readout uses, and why they differ

This is the part that was wrong until 2026-09-06, so it is stated per script rather than assumed.

* **A comparison takes the floor of its OWN regime.** `tc_readout.py` computes two bars and never
  pools them: the **frozen 1.66pp** (six draws, both 2×2 replicate pairs × three depths) for the
  2×2 contrast, whose arms are all frozen at K=3, and the **controller-live 4.27pp [+1.23, +6.92]**
  (the three N1−N2 draws, the ruled pooled fold floor) for C1-vs-B2, whose arms both ran
  controller-live. The **2.53pp nine-draw pooled bar the script used until 2026-09-06 is
  RETRACTED** — it let a frozen comparison borrow a live floor's slack and a live comparison borrow
  a frozen floor's strictness.
* **C1 − B2 is STILL OWED and this batch does not discharge it.** At the 4.27 bar only the +1M leg
  (+10.12) clears robustly; mid (+5.87) and end (+5.25) clear the point estimate but sit INSIDE the
  floor's own interval, so they print `SIGNIFICANT? (bar-uncertain)`. The re-read needs more
  CONTROLLER-LIVE draws; frozen draws say nothing about a controller-live spread.
* **A cell with FEWER draws keeps its own, LARGER bar.** `k6_readout.py` reads P1 and P2 at K=6's
  own **2.46pp** (three draws), never at the 2×2's 1.66 or the nine-draw 1.92, and prints all three
  so the choice is visible.
* **The taught slice takes the taught floor, PROVISIONALLY.** `taught_readout.py` labels against the
  2×2's own two taught draws (**0.47pp**) — the right slice, but two draws at one depth, the exact
  structure that put the untaught floor at 0.12 before six draws made it 1.66. Every such label is
  stamped `(prov.)`, and the script then re-reads the same rows at the untaught six-draw 1.66 as a
  SENSITIVITY, naming the rows that move. Only one does (leg A, +0.72, NOT DETECTED → WITHIN FLOOR);
  the readout row and the dose row do not.

**None of these bars changes a verdict banked in the ledger.** They change which bar the output
*says* it used, and they stop `tc_readout.py` printing a discharged-looking SIGNIFICANT on a
comparison the ledger holds open.

### Where the K=6 cell's script lives

The K=6 cell (`ai_v9_170_TCUNFK6A_0904` / `ai_v9_171_TCUNFK6B_0904`) never had a measurements
directory of its own. Its arms are this batch's UNFUNDED argv minus two tokens
(`--grad-accum-steps 3→6`, `--run-name`) and its per-team rows were produced by this batch's own
probes at the same stamp, so they were homed here on 2026-09-06 as
`untaught_TCUNFK6{A,B}_{p1M,mid,end}.json` and `taught_TCUNFK6{A,B}_end.json`. `k6_readout.py` is
therefore committed **beside its inputs**, rather than a directory being invented to hold a script
whose data is elsewhere — which is the defect the readout gate exists to catch.

### The independent cross-check that predates all four

`arch_transfer_2026-09-05/teacher_distance/fold_table.py` recomputes each arm's ENDPOINT untaught
delta vs its parent from these same rows, under a different seed (20260905) and a different code
path. It reproduces TC_FUND_A −2.50, TC_FUND_B −2.31, TC_UNF_A +2.00, TC_UNF_B +1.94,
TC_UNF_K6_A +0.37, TC_UNF_K6_B −0.81 — the ledger's point estimates exactly, and `recovery_readout.py`
and `k6_readout.py` agree with it row for row.
