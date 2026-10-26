# L1's registered replication on the plateau bank (read 2026-10-25)

**Status: READ — the registered cell REPLICATES (thin), and it is the 4-turn leaf's own signal and a 25M-policy signal,
not the anchor's and not P\*'s. ALL SELF-REFEREE.** Registration: [`PREREGISTRATION.md`](PREREGISTRATION.md) (committed
`1f980728a` before the read; it writes down the cell and the bar that L1's text left implicit, FINDING 1). The read is
[`l1_plateau.py`](l1_plateau.py); the result is [`result_l1_plateau.json`](result_l1_plateau.json). It loads the plateau
bank's banked rows (`~/gen3ai_archive/hidden_info_maturity/plateau/`, read only), applies L1's rules at the values
fitted on Tier 1 (`../search_l1_l5_2026-10-20/fit_l1.json`, sha256 `946c933f…db9995`, never refitted), and plays nothing
(scoring only, 1 CPU-minute).

## 1. The registered cell

Plateau bank (P\*'s own newest cycles; 25M and P\* anchors; 719 common states each, 1,438 rows, 464 battles; 2,000
battle-clustered draws, seed 20261011). IMP = the win probability gained per decision by the rule's pick over the
policy's top choice, judged by the SELF-REFEREE (the truth played out with the anchor on both sides).

| cell | IMP [95 %] | [98.33 % verdict] | clear? (lower > +0.0005) |
|---|---|---|---|
| **PIKL (λ 0.015) \| TRUNC-4-K8 \| plateau** | **+0.0097 [+0.0025, +0.0169]** | **[+0.0010, +0.0191]** | **YES: REPLICATES** |
| the same cell on E2 (L1's read) | +0.0123 | [+0.0010, +0.0230] | (the original) |

The anchor's own part: IMP(piKL) − IMP(ARGMAX) = **+0.0025 [−0.0039, +0.0089]** (E2: +0.002 [−0.005, +0.008]); the plain
ARGMAX at the 4-turn leaf already reads +0.0072 [−0.0025, +0.0170]. So, as on E2, **the cell is the leaf's own weak
positive, not the anchor's gain**; the anchor is not detected to add anything.

## 2. The other cells (DESCRIPTIVE; same table, same frozen fits)

| rule | LEAF-K8 | TRUNC-4-K8 | CRITIC |
|---|---|---|---|
| ARGMAX | +0.0006 | +0.0072 | +0.0034 |
| PIKL | +0.0045 [−0.0009, +0.0105] | **+0.0097** [+0.0025, +0.0169] (clear) | +0.0036 |
| GUMBEL | +0.0056 [−0.0009, +0.0126] | **+0.0095** [+0.0020, +0.0173] (clear at 98.33 %: lower +0.0006, a boundary) | +0.0037 |
| THRESH | +0.0014 [−0.0051, +0.0076] | +0.0090 [+0.0003, +0.0174] (98.33 % lower −0.0014: not clear) | +0.0058 |

Single-bank form of L1 §4.3: **PAYS-SCOPED** (clear cells: PIKL and GUMBEL at TRUNC-4-K8). No rule is clear at both
primary scorers, so no PAYS; the NULL bound fails. Anchored rollout minus anchored critic: +0.0009 / +0.0019 / −0.0044
at LEAF-K8 and +0.0061 / +0.0059 / +0.0033 at TRUNC-4-K8 (PIKL / GUMBEL / THRESH), none clear. Oracle world, plain
ARGMAX (context): +0.0600 [+0.0493, +0.0707].

## 3. Where the replicating cell comes from (DESCRIPTIVE, POST HOC, no multiplicity correction)

| split | PIKL TRUNC-4-K8 | ARGMAX TRUNC-4-K8 |
|---|---|---|
| anchor 25M (n 719) | **+0.0193 [+0.0089, +0.0303]** | +0.0156 [+0.0024, +0.0294] |
| anchor P\* (n 719) | **+0.0001 [−0.0097, +0.0101]** | −0.0012 [−0.0131, +0.0113] |
| snapshot stratum (726; SELF-REFEREE) | +0.0138 [+0.0043, +0.0234] | +0.0115 [−0.0009, +0.0242] |
| bot stratum (712; SELF-REFEREE) | +0.0056 [−0.0060, +0.0178] | +0.0027 [−0.0126, +0.0179] |

The whole gain sits on the **25M anchor**; at **P\*** (the plateau policy, the one that matters for today) the same rule
reads +0.0001. The E2 cell was also measured on immature 15M policies. Read together, the 4-turn leaf's small positive
is a property of the less mature policy, and it is gone at the plateau. The anchor adds nothing detectable at either
anchor (25M: +0.0037 over ARGMAX). The held-out grid curve (upper envelope, not a verdict) peaks at +0.0116 (λ 0.03),
so the fitted λ is not special.

## 4. GIGO

G1 PASS: the recomputed ARGMAX IMP equals the plateau table's per-state F IMP (LEAF-K8, TRUNC-4-K8, CRITIC) and the O
row's LEAF-K8 on all 719 × 2 states (0 mismatches); the table sha256 (`cd443030…bd9f0e`) and the common-ids sha256
(`ffb3e84a…551ed`) equal `result_mat_plateau.json`'s; G3 719 per anchor. Peak 1.0 GB (`mem_cap.sh --max-s 600`, 8 GB cap).

## 5. FINDINGS

1. **L1's registration named the bank and the trigger but not the cell, the bar or the statistic.** `PREREGISTRATION.md`
   fixed them from L1 §4.1–§4.3 before the read (the PAYS-SCOPED cell; lower bound at 98.33 % > +0.0005; the frozen
   Tier-1 fit). A different reading (say, all six cells with a Bonferroni correction, or a 95 % bar) would be a
   different registration. The verdict of the cell does not change with the level: lower bound +0.0025 at 95 %.
2. **The replication is thin and single-sourced.** The lower bound (+0.0010) is twice the bar, equals the E2 cell's, and
   rests on the 25M anchor alone (§3). Cells are not independent (one bank, two anchors on the same 719 states, one
   cluster per battle), so this is one replication, not two.
3. **SELF-REFEREE.** Every cell is scored with the anchor's own policy playing both sides (E1-0, `015f00e2`: the bot
   stratum is UNDECIDED against a real-bot referee, a gross misreading not ruled out). The bot-stratum cells above are
   labelled SELF-REFEREE; none is a real-opponent size. On the bot stratum alone the cell is not clear
   (+0.0056 [−0.0060, +0.0178]).
4. **The anchor is not shown to add anything.** The registered cell's anchor part is +0.0025 [−0.0039, +0.0089]. The
   ledger's "an anchor is a guard against the argmax's harm, not a source of gain" stands; on this bank the argmax
   does not even hurt (+0.0006 / +0.0072 / +0.0034), so the guard has nothing to remove.
5. **Nothing was adopted.** An anchored rule stays a possible guard only; the honest default is unchanged (a code change is
   a separate unit).
6. No hazard hit: CPU only, `CUDA_VISIBLE_DEVICES=""`, agent cores, `nice -n 19`, `mem_cap.sh --max-s 600` (short lane);
   nothing written under `models/`, `data/` or the archive.
