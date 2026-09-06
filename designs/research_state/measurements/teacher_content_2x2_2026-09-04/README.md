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
| `tc_readout.py` | the readout — runs as committed, from this directory |

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

## Reading the results

`tc_readout.py` (committed here; `--check` resolves its inputs without computing). It labels every
comparison PIN-SPLIT or pin-clean from each arm's recorded `git_hash`, reports the frozen-only floor
beside the 3-draw floor so the controller-live/frozen regime assumption stays visible, and refuses
to pool the controller-live N1/N2 draws as a frozen floor.

🚨 **THE COMMITTED SCRIPT IS ONE REVISION BEHIND THE LEDGER, and only its BARS are affected — every
delta and interval it prints matches what is banked, exactly.** It was committed on 2026-09-04
19:53 (`5af5ff88`), before the 2026-09-05 retraction; a newer revision exists but was never landed.
Two differences, both about which bar labels a comparison, neither about a number:

* It pools all nine draws — six frozen + the three controller-live N1/N2 — into a **2.53pp** bar and
  applies that to the funded−unfunded contrast. The ledger's ruling is that a frozen-regime
  comparison takes the **frozen 1.66pp** bar (which the script prints, correctly, on its own line).
  **No verdict moves**: p1M is WITHIN FLOOR and mid/end SIGNIFICANT under either bar.
* It then re-reads **C1 − B2** at that pooled bar and prints SIGNIFICANT at all three depths. The
  ledger explicitly refuses this — C1 and B2 ran controller-live, the owed re-read needs more
  controller-live draws, and this frozen batch does not discharge it. **Ignore that section**; the
  legs' own values (+10.12 / +5.87 / +5.25) are the banked ones and are read at the 4.27pp
  controller-live floor, where only the +1M leg clears.

The script's header still describes the pooled bar as intended behaviour, so the docstring and the
ruling disagree too. Landing the newer revision is a separate, ledger-visible change; nothing here
was adjusted to make the output agree with what is banked.

⚠️ **`tc_readout.py` covers the UNTAUGHT 2×2 only. Three families of banked number have their
artifacts here but NO committed script that aggregates them**, so they cannot be re-derived by
running anything: the **taught** side (+5.11 / +4.86 vs parent, funded−unfunded +0.25, pooled
**+4.98**, and K=6's **+4.48**), the **arm-vs-parent** untaught levels and the p1M→mid/end
**recovery** table, and the whole **K=6** cell (P1/P2/P3). Those were computed ad hoc in the session
that banked them. The nearest committed check is
`arch_transfer_2026-09-05/teacher_distance/fold_table.py`, which independently recomputes each arm's
ENDPOINT untaught delta vs its parent from these same rows — it reproduces TC_FUND_A −2.50,
TC_FUND_B −2.31, TC_UNF_A +2.00, TC_UNF_B +1.94, TC_UNF_K6_A +0.37, TC_UNF_K6_B −0.81, matching the
ledger's point estimates exactly (its intervals differ slightly: a separate bootstrap, seed
20260905). Writing the missing readouts is unclaimed work, not something this pass invented.
