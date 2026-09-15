## 5. PART B — the paired refit. **The DATA is the fix; the LOSS FORM is not.**

Split by BATTLE (957 battles → 65 / 15 / 20 %): **9,591 train / 2,140 val / 3,049 test** branch
rows; **3,033 test pairs, 562 of them non-tied.** Both arms WARM-START from the checkpoint's own
head, so arm (i) at step 0 *is* the original head. Adam lr 1e-3, batch 1024, validation every 25
steps, patience 40 checks; every fit early-stopped at ~1,025–1,050 steps.

### 5.1 The headline table — held-out, read once

| head | **PAIRWISE ACCURACY** (562 non-tied pairs, bootstrap over FORKS) | separation ratio \|ΔV\| non-tied / tied | ECE (15 equal-mass bins) | Brier |
|---|---|---|---|---|
| **`original`** — the promoted win-prob critic @10M | **0.5169 [0.4800, 0.5524]** — **the CI straddles 0.50** | **1.020** | 0.1246 | 0.2154 |
| **(i) `control_bce`** — ordinary BCE on the fork outcomes | **0.6032 [0.5690, 0.6374]** | 1.711 | 0.0520 | 0.1417 |
| (ii) `rank@0.1` — + pairwise term *(chosen on validation)* | 0.5925 [0.5575, 0.6278] | 1.723 | 0.0421 | 0.1398 |
| (ii) `rank@0.3` | 0.5925 [0.5574, 0.6278] | 1.713 | 0.0398 | 0.1400 |
| (ii) `rank@1.0` | 0.5890 [0.5546, 0.6229] | 1.711 | 0.0398 | 0.1405 |

| paired delta (its OWN bootstrap CI over forks) | |
|---|---|
| **(i) control − original** | **+0.0863 [+0.0384, +0.1347] DETECTED** |
| **(ii) best rank − original** | **+0.0756 [+0.0276, +0.1233] DETECTED** |
| **(ii) best rank − (i) control** | **−0.0107 [−0.0249, +0.0028] NOT DETECTED** |

### 5.2 What those three rows say

**1 — THE BASELINE IS THE FINDING. The promoted win-prob critic ranks siblings AT CHANCE:
0.5169 [0.4800, 0.5524].** This is the first direct measurement of the property the one-ply search
test consumes, and it reads as if the head had no opinion at all about which of two states one move
apart is better. Every behavioural result on this leaf now has an arithmetic cause: unguarded
search loses three games in four because it re-ranks actions with a coin.

**2 — the TRUNK is not the problem.** With the trunk FROZEN and only four head tensors moving,
pairwise accuracy goes to **0.6032**, +0.086 with its CI clear of zero. `value_pooled` already
carries enough to rank siblings well above chance; the online head simply does not read it that
way. **Pairwise accuracy is a RANK statistic, so no monotone recalibration can move it** — the
refit did not re-scale the head, it re-ordered it.

**3 — the pairwise RANKING term buys NOTHING over ordinary BCE on the same data**
(−0.0107 [−0.0249, +0.0028], point estimate slightly negative, at every swept coefficient). The
fix is not the loss FORM. It is the **TARGET AND THE STATES**: the head has never been trained on
successor states of actions the policy did not take, with outcomes measured under common random
numbers. Give it those rows and plain BCE extracts the ordering; adding an explicit ranking term to
the same rows adds nothing.

**4 — ranking was not bought with calibration; it came WITH calibration.** ECE falls
0.1246 → 0.0520 (i) → 0.0398–0.0421 (ii) and Brier falls 0.2154 → 0.1417. The guard fires the
other way from the registered worry.

**5 — the CONDITIONING GUARD holds.** `cond.opp_class_auc.t4_10` (orientation-matched, label 1 =
scripted bot, 1,800 turns-4–10 states over all 12 opponents): **original 0.711 · (i) 0.706 ·
(ii) 0.699.** The original head's 0.711 reproduces the N-curve's published 0.723 / 0.700 for the
online head on a different draw, which is the check that this frame is the right one. The refit
costs at most ~0.012 of opponent-class information — nothing like a trade.
⚠️ This is a RAW monotone AUC of V, not `conditioning_meters`' out-of-fold decoder FIT; the tool's
own note ("a 1-D monotone decoder's AUC is nearly the AUC of V itself") is why the magnitudes are
comparable, and `main.ops.critic_read` **cannot take an external head** — it addresses an arm by
run NAME — so the fallback in the task's spec is the one that ran.
