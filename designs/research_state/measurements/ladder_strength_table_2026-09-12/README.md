# THE CRITIC LADDER, READ ON STRENGTH — fifteen 10M arms, matched snapshot count

**2026-09-12. Zero GPU. Nothing written under `models/`.** Pre-registration: `PREDICTION.md`
(committed at `937775b6`, before the matched-count refit produced a number — with an explicit
disclosure of what had already been seen; read §6 there before weighting P1/P2).

This file does **not** edit `ledger.md`, `UNDERSTANDING.md`, or
`winprob_critic_ladder_2026-09-08.md`. §8 carries a ready-to-append ledger paragraph.

---

## 1. THE VERDICT, IN FOUR LINES

1. **No arm is outside the run-to-run floor.** The floor, measured here, is **45.0 Elo** — the max
   pairwise |Δ| over the three controls, exactly reproducing the ledger's 2026-09-09 figure from an
   independent matched-count refit. Every one of the fifteen arms is **WITHIN FLOOR — NOT DETECTED**
   on the registered rule.
2. **Two arms are CANDIDATES, both DOWNWARD, and one of them survives the node cross-check.**
   `vf15_b` (−101.9 / −56.9 / −66.6 vs the three controls) and `strata_b` (−90.7 / −45.7 / −55.4)
   clear all three control points with a consistent sign, but **no delta CI clears the floor**
   (clause 3), so neither is a detection. On the 8M cross-check node `strata_b`'s lean **collapses**
   (−48.9 / +6.0 / −0.6 — the sign does not even hold) while `vf15_b`'s **grows** (−133.4 / −78.5 /
   −85.1, past the 8M floor of 54.9 on all three). `vf15_b` is the only node-robust lean in the family.
3. **The instrument cannot resolve its own floor.** `se(Δ)` is **21–24 Elo** at four nodes, i.e. a
   95% CI of **±43 to ±48** — *the same size as the 45.0 floor*. Clause 3 therefore can essentially
   never fire on this table. This was registered as prediction P3 before the read, precisely so it
   could not be presented afterwards as a discovery about the arms. **The dense ladder at four
   nodes is not an instrument for a 45-Elo question.**
4. **The family's spread is run-to-run variance, and the control triple UNDERSTATES it.** Two levers
   have a same-pin seed replicate whose pair difference has a CI clear of zero: `vf15` − `vf15_b` =
   **82.6 Elo [39.5, 125.7]** (1.84× the floor) and `strata` − `strata_b` = **58.1 [16.2, 100.0]**
   (1.29×). The same-configuration control pair `ctrl10M_b` − `ctrl10M_c` is **9.7 [−53.1, +33.7]**
   (0.22×). This is standing rule 22 reproduced on STRENGTH: *a control-replicate floor bounds the
   CONTROL's run-to-run variance, not the lever's.*

The strength read therefore agrees with the critic read: **after fifteen 10M arms and seven levers,
the campaign has zero detections on strength as well as zero confirmed detections on a decision row.**

## 2. THE INSTRUMENT, AND WHAT WAS NOT USED

- Headline = `<run>/snapshot_ladder/ladder.json`, the dense ±10 ladder, never `eval/elo`
  (`UNDERSTANDING.md` §3.2 rule 1).
- **`python -m main.elo` has NO matched-count cross-run mode** — one positional `run_dir`, and it
  writes `<run_dir>/elo/`. **Not used**, and it could not have been: it would have written under
  `models/`.
- **`python -m main.critic_gate --at-snapshots N` DOES implement the rule** (`ladder_section` →
  `_refit_first_n`), but only **pairwise** (`run` vs `--parent`), and it drags four other endpoints
  and the untaught meter along.
- **So this read calls the library function both of those sit on**:
  `agents.training.snapshot_ladder.fit_ladder(run_dir, first_n=4, steps=COMMON, write=False)` —
  whose own docstring is the definition of "matched SNAPSHOT COUNT", and in which `first_n` forces
  `write = False`. Script: `read_ladder_strength.py`; post-hoc addenda: `addenda.py`.
- **Every side is refit through THIS tree's `fit_ladder`**, never trusted from the committed file —
  the reason `critic_gate` refits both sides even when the node counts already match.

## 3. THE MATCHED COUNT — 4, over the COMMON step set, and why

**Matched count = 4. Common step set = `{4000032, 6000000, 8000016, 10000032}`. Headline node =
10000032. Cross-check node = 8000016.**

Twelve arms rate exactly those four nodes. Three — `lambda09`, `lambda095`, `rollout` — rate a
**fifth, EARLIER** node at `2000016`, because they promoted at 2M: the self-play crossing is a
per-arm lottery (standing rule 15's 2026-09-10 amendment; `SELF_PLAY_START = 0.55` sits inside the
2M bot-win-rate replicate floor).

🚨 **A naive `first_n=4` on those three is WRONG and would have gone unnoticed.** It keeps
`{2M, 4M, 6M, 8M}` and reads their **8M** node against everyone else's **10M** — matching the fit
size while un-matching the amount of training, which is the confound the matched-count rule exists
to remove. All fifteen are therefore refit on the **common four steps**, dropping the 2M node and
its edges from the three five-node arms. Every arm is then the newest node of a four-node fit over
the identical step set: matched count, matched fit size, matched step set. The script REFUSES if the
common step set and the matched count ever disagree in size.

Effect of the correction on those three arms (refit − committed at 10M): `lambda09` **−6.3**,
`lambda095` **+3.1**, `rollout` **−4.8**. The twelve four-node arms reproduce their committed
numbers **exactly** (every committed ladder already carries `eval_sentinel_edges_dropped` 6 or 10,
so none of them was fit with the biased sentinel edges).

**Newest-node inflation (§3.2 rule 2) is handled by symmetry and CHECKED.** All fifteen runs
FINISHED at exactly **10,027,008** steps, so no arm's rating is the transient value of a run still
growing, and every arm's 10M node occupies the identical structural position (newest of a four-node
fit) — the inflation is common-mode and cancels in a pairwise Δ to first order. The registered
cross-check is the **8M node of the same fit** (second-newest, refit once): §5. Spearman rank
correlation between the two rows across the fifteen arms is **0.775** (P5 predicted 0.4–0.8).

## 4. THE TABLE — headline node 10000032, matched four-node fit

`Δ` columns are arm − control. `se(Δ) = sqrt(se_arm² + se_ctrl²)` from the matched-count BT fit;
the CI95 half-width column is 1.96·se(Δ) and applies to every Δ in the row.

| arm | lever | Elo@10M | se | Δ ctrl10M | Δ ctrl10M_b | Δ ctrl10M_c | se(Δ) | CI95 ± | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rollout` | `--win-prob-rollout-target --win-prob-rollout-weight 64` | **2062.3** | 17.8 | +43.6 | +88.6 | +78.9 | 24.4 | ±48 | WITHIN FLOOR |
| `truevalue` | `--value-true-team` (privileged critic) | 2054.0 | 17.4 | +35.3 | +80.3 | +70.6 | 24.1 | ±47 | WITHIN FLOOR |
| `denseaux` | `--win-prob-dense-aux 1.0` | 2044.6 | 17.3 | +25.9 | +70.9 | +61.2 | 24.0 | ±47 | WITHIN FLOOR |
| `tdaux` | `--td-aux-coef 1.0` | 2034.7 | 17.1 | +16.0 | +61.0 | +51.3 | 23.9 | ±47 | WITHIN FLOOR |
| `lambda095` | `--win-prob-lambda 0.95` | 2032.9 | 16.8 | +14.2 | +59.2 | +49.5 | 23.7 | ±46 | WITHIN FLOOR |
| `lambda09_b` | λ-0.9 seed replicate | 2030.5 | 16.7 | +11.8 | +56.8 | +47.1 | 23.6 | ±46 | WITHIN FLOOR |
| `cflabels` | `--cf-records --cf-winprob-coef 0.5` | 2021.1 | 16.7 | +2.4 | +47.4 | +37.7 | 23.6 | ±46 | WITHIN FLOOR |
| **`ctrl10M`** | **CONTROL** (arm A @10M) | 2018.7 | 16.7 | — | +45.0 | +35.3 | 22.9 | ±45 | control |
| `vf15` | `--vf-coef 1.5` | 1999.4 | 16.6 | −19.3 | +25.7 | +16.0 | 23.5 | ±46 | WITHIN FLOOR |
| `lambda09` | `--win-prob-lambda 0.9` | 1997.2 | 16.0 | −21.5 | +23.5 | +13.8 | 23.1 | ±45 | WITHIN FLOOR |
| `strata` | `--win-prob-strata-weight 1.0` | 1986.1 | 15.8 | −32.6 | +12.4 | +2.7 | 23.0 | ±45 | WITHIN FLOOR |
| **`ctrl10M_c`** | **CONTROL** replicate | 1983.4 | 15.7 | −35.3 | +9.7 | — | 22.9 | ±45 | control |
| **`ctrl10M_b`** | **CONTROL** replicate | 1973.7 | 15.6 | −45.0 | — | −9.7 | 22.9 | ±45 | control |
| `strata_b` | `strata` seed replicate | 1928.0 | 14.4 | −90.7 | −45.7 | −55.4 | 22.1 | ±43 | **CANDIDATE** (no Δ CI clears) |
| `vf15_b` | `vf15` seed replicate | **1916.8** | 14.4 | −101.9 | −56.9 | −66.6 | 22.1 | ±43 | **CANDIDATE** (no Δ CI clears) |

**Elo range across the fifteen: 145.5.** (P1 predicted 100–140 — the spread came in just above the
registered band, which does not change the verdict but is recorded as a miss.)

## 5. THE FLOOR, AND THE DECISION RULE APPLIED

| control pair | Δ | se(Δ) | CI95 | pins | seeds |
|---|---|---|---|---|---|
| `ctrl10M` − `ctrl10M_b` | **+45.0** ← the floor | 22.9 | [+0.2, +89.8] | f3502568 → 377a5aa1 | 42 / 1001 |
| `ctrl10M` − `ctrl10M_c` | +35.3 | 22.9 | [−9.6, +80.2] | f3502568 → 377a5aa1 | 42 / 1002 |
| `ctrl10M_b` − `ctrl10M_c` | −9.7 | 22.1 | [−53.1, +33.7] | 377a5aa1 (same) | 1001 / 1002 |

**FLOOR (10M) = 45.0 Elo** — the MAX pairwise |Δ|, per standing rule 3 (never the mean, 30.0, and
never the smallest pair, 9.7). **FLOOR (8M) = 54.9 Elo.**

The registered rule (an arm is OUTSIDE only if **all three** |Δ| exceed the floor, **with the same
sign**, **and each Δ's own 95% CI excludes the floor in that direction**) fires on **no arm**.
`strata_b` and `vf15_b` satisfy clauses 1 and 2 and fail clause 3 ⇒ CANDIDATE, per rule 22.

**The 8M cross-check node (same four-node fit, floor 54.9):**

| arm | Elo@8M | Δ ctrl10M | Δ ctrl10M_b | Δ ctrl10M_c | verdict |
|---|---|---|---|---|---|
| `denseaux` | 2032.2 | +27.7 | +82.6 | +76.0 | WITHIN FLOOR |
| `lambda09_b` | 2014.4 | +9.9 | +64.8 | +58.2 | WITHIN FLOOR |
| `ctrl10M` | 2004.5 | — | +54.9 | +48.3 | control |
| `truevalue` | 1992.3 | −12.2 | +42.7 | +36.1 | WITHIN FLOOR |
| `tdaux` | 1987.7 | −16.8 | +38.1 | +31.5 | WITHIN FLOOR |
| `rollout` | 1977.9 | −26.6 | +28.3 | +21.7 | WITHIN FLOOR |
| `cflabels` | 1975.9 | −28.6 | +26.3 | +19.7 | WITHIN FLOOR |
| `lambda09` | 1972.1 | −32.4 | +22.5 | +15.9 | WITHIN FLOOR |
| `lambda095` | 1958.5 | −46.0 | +8.9 | +2.3 | WITHIN FLOOR |
| `strata` | 1956.4 | −48.1 | +6.8 | +0.2 | WITHIN FLOOR |
| `ctrl10M_c` | 1956.2 | −48.3 | +6.6 | — | control |
| `strata_b` | 1955.6 | −48.9 | +6.0 | **−0.6** | WITHIN FLOOR — *the 10M candidacy does not survive* |
| `ctrl10M_b` | 1949.6 | −54.9 | — | −6.6 | control |
| `vf15` | 1910.2 | −94.3 | −39.4 | −46.0 | WITHIN FLOOR |
| `vf15_b` | **1871.1** | −133.4 | −78.5 | −85.1 | **CANDIDATE** — *survives, and grows* |

Per the registration, an arm whose two rows disagree is reported **INCONCLUSIVE** rather than
resolved in favour of either row: **`strata_b`'s downward lean is INCONCLUSIVE** (a 10M-node
candidate that the 8M node of the same fit does not see at all — note also that `strata_b` is the
one arm whose rating FALLS from 8M to 10M, 1955.6 → 1928.0). **`vf15_b`'s downward lean is a
CANDIDATE on both nodes.**

`rollout` sits top of the 10M table at +43.6 vs `ctrl10M` — and **sixth** on the 8M row at −26.6.
It is the clearest illustration on this table of why a single node is not a reading.

## 6. DESCRIPTORS — not the read

🚨 **`win_rate_vs_bots` is NOT an independent descriptor of the headline; it is an INPUT to it.**
`fit_ladder`'s source (2) folds each snapshot's historical **bot edges** into the fit — they are
what pins the absolute scale. So the correlation below (Pearson **0.889**, Spearman **0.768** across
the fifteen) is **partly mechanical** and must never be read as two instruments agreeing.

All fifteen rows are the 10M eval cycle, **800 games over 8 bots**, `random` **excluded** (ledger
2026-09-12; `eval_callback.py:363` — "*Random leads (a cheap 'is the model broken' floor; eval-only,
excluded from `win_rate_vs_bots`)*"). Every arm's `random` row is 0.99–1.00, so including it would
shift every number by the same ~+0.012 and order nothing. Eval regime read from
`model_config.json`'s `eval_sentinel_greedy` — **never assumed** (§3.2 rule 5) — and corroborated by
each 10M row's own recorded `sentinel_regime`. Seed from the run's `original_command` where the flag
is present, otherwise from the resolved `cli_args` (a flagless launch takes the parser default, 42);
`seed_source` is recorded per arm in `ladder_strength.json`. Pin from `metadata.json`'s immutable
`pin_history` (every one of the fifteen is a single `pin_commit` span, first_step 0).

| arm | Elo@10M | `win_rate_vs_bots` @10M | 95% CI | `eval_sentinel_greedy` | recorded `sentinel_regime` | pin | seed | seed source |
|---|---|---|---|---|---|---|---|---|
| `rollout` | 2062.3 | **0.910** | [0.890, 0.930] | `true` | greedy + symmetric_teams | `40bf23d9` | 1003 | `--seed` |
| `truevalue` | 2054.0 | 0.901 | [0.881, 0.922] | `true` | greedy + symmetric_teams | `d11386dc` | 42 | default |
| `denseaux` | 2044.6 | 0.896 | [0.875, 0.917] | `true` | greedy + symmetric_teams | `46ca68ef` | 42 | default |
| `tdaux` | 2034.7 | 0.885 | [0.863, 0.907] | `true` | greedy + symmetric_teams | `f3502568` | 42 | default |
| `lambda095` | 2032.9 | 0.904 | [0.883, 0.924] | `true` | greedy + symmetric_teams | `28ece02a` | 42 | default |
| `lambda09_b` | 2030.5 | 0.904 | [0.883, 0.924] | `true` | greedy + symmetric_teams | `28ece02a` | 1002 | `--seed` |
| `cflabels` | 2021.1 | 0.899 | [0.878, 0.920] | `true` | greedy + symmetric_teams | `377a5aa1` | 42 | default |
| `ctrl10M` | 2018.7 | 0.899 | [0.878, 0.920] | `true` | greedy + symmetric_teams | `f3502568` | 42 | default |
| `vf15` | 1999.4 | 0.880 | [0.858, 0.903] | `true` | greedy + symmetric_teams | `f3502568` | 42 | default |
| `lambda09` | 1997.2 | 0.879 | [0.856, 0.901] | `true` | greedy + symmetric_teams | `28ece02a` | 42 | default |
| `strata` | 1986.1 | 0.893 | [0.871, 0.914] | `true` | greedy + symmetric_teams | `f871e79f` | 42 | default |
| `ctrl10M_c` | 1983.4 | 0.896 | [0.875, 0.917] | `true` | greedy + symmetric_teams | `377a5aa1` | 1002 | `--seed` |
| `ctrl10M_b` | 1973.7 | 0.880 | [0.858, 0.903] | `true` | greedy + symmetric_teams | `377a5aa1` | 1001 | `--seed` |
| `strata_b` | 1928.0 | **0.839** | [0.813, 0.864] | `true` | greedy + symmetric_teams | `f871e79f` | 1002 | `--seed` |
| `vf15_b` | 1916.8 | **0.839** | [0.813, 0.864] | `true` | greedy + symmetric_teams | `f3502568` | 1001 | `--seed` |

**The whole family is on the post-2026-09-07 regime** (`eval_sentinel_greedy: true`, symmetric
teams), so these bot rows are comparable **within** the family and **not** against anything before
that boundary. P4 predicted `true` for all fifteen (✓) and 0.93–1.00 for the win rates (**✗** — the
observed band is 0.839–0.910; the bots are compressed but not saturated at 10M, and the prediction
was wrong on level).

## 7. HAZARDS — reported as FINDINGS

1. 🚨 **The matched-count rule can un-match the steps.** Three of fifteen arms rate a fifth,
   *earlier* node, so `first_n=4` on them reads 8M against everyone else's 10M. Nothing in the tree
   warns about this: `critic_gate._refit_first_n` takes the count and refuses only if the prefix
   cannot be fit **on its own** — a prefix over a different step set fits fine and reads wrong.
   **A matched-count comparison across a family with heterogeneous promotion cadence must match the
   STEP SET explicitly.** This read does, and refuses if it cannot.
2. 🚨 **The dense ladder at four nodes cannot resolve its own run-to-run floor.** `se(Δ)` 21–24 ⇒
   CI95 ±43–48 against a 45.0 floor. Any future ladder-strength question on 10M arms needs more
   nodes (more promotions), not more arms. Registered as P3 before the read.
3. ⚠️ **`win_rate_vs_bots` is an input to the ladder rating, not an independent check on it** (§6).
4. ⚠️ **Non-transitivity is material on these pools.** `fit_quality.max_abs_err` — the worst
   |predicted − observed| over the six dense frozen pairs — runs to **0.194** (`lambda09`), 0.186
   (`denseaux`), 0.173 (`tdaux`), 0.171 (`rollout`); mean_abs_err spans 0.042–0.113. A scalar Elo
   misrepresents at least one pair in those pools by up to 19 pp. Four nodes is also the minimum at
   which the read is possible at all, and `n_frozen_pairs = 6` everywhere.
5. ⚠️ **The floor's widest pair CROSSES a pin boundary; the one same-pin control pair does not.**
   `ctrl10M` (pin `f3502568`) vs `ctrl10M_b`/`_c` (pin `377a5aa1`) read 45.0 and 35.3; `ctrl10M_b`
   vs `ctrl10M_c` (same pin, seeds 1001/1002) reads 9.7. With `se(Δ) = 22.9` on all three, **9.7 and
   45.0 are not distinguishable**, so this is an OBSERVATION with the seed lottery fully open — not
   a pin effect, and the 2026-09-09 pinned-input functional test found that span neutral. What it
   does mean: the 45.0 floor bounds run-to-run variance **including a code increment**, which makes
   it conservative for a same-pin comparison. Rule 3 wants exactly that bar.
6. ⚠️ **The control triple understates a lever arm's variance** (§1 point 4, rule 22). Two
   same-pin seed replicate pairs have CIs clear of zero at 1.84× and 1.29× the floor.
7. ⚠️ **`strata_b` and `vf15_b` have bit-identical `win_rate_vs_bots` (0.83875 = 671/800) on
   completely different per-bot rows** (`strata_b` heuristic 0.87 / heuristic2 0.78 …; `vf15_b`
   0.78 / 0.84 …). Coincidence, verified — not a copied row.
8. ⚠️ **`ai_v12_26_ladder_ctrl10M_shaped` (the matched-pin SHAPED-critic control) is EXCLUDED**: it
   was LIVE on the GPU at read time and has **no `snapshot_ladder/` directory at all**. The script
   asserts its absence and prints a 🚨 if a ladder has since appeared — rule 16 (*an instrument that
   reads an artifact another process is still writing asks first whether it is complete*). Nothing of
   that run was touched. It is the sixteenth arm of this family and this table must be re-read, not
   patched, when it lands.
9. ⚠️ **The registration was not blind** — `PREDICTION.md` §6 discloses that the committed
   `ladder.json` ratings had been seen during the roster survey. What is genuinely pre-registered
   is the decision rule and the floor definition. A future read wanting a blind registration must
   register before the roster survey.

## 8. READY-TO-APPEND LEDGER PARAGRAPH

> ### 2026-09-12 · MEASUREMENT · THE CRITIC LADDER READ ON STRENGTH — fifteen 10M arms at matched snapshot count: **no arm outside the 45.0-Elo floor**, the floor REPRODUCED from an independent refit, two DOWNWARD candidates (`vf15_b`, `strata_b`) neither of which is a detection, and the ladder at four nodes shown unable to resolve its own floor (se(Δ) 21–24 ⇒ CI95 ±43–48)
>
> `measurements/ladder_strength_table_2026-09-12/`, pre-registered at `937775b6` (with a disclosure
> that the committed ratings had been seen during the roster survey, so the registration's value is
> the DECISION RULE, not blindness). Zero GPU; nothing written under `models/`.
> **The instrument.** `main.elo` has no matched-count cross-run mode and writes under `models/`;
> `main.critic_gate --at-snapshots` implements the rule but only pairwise. This read calls the
> function both sit on — `snapshot_ladder.fit_ladder(first_n=4, steps=COMMON, write=False)` — on all
> fifteen arms. **Matched count = 4 over the COMMON step set `{4.0M, 6.0M, 8.0M, 10.0M}`**: three
> arms (`lambda09`, `lambda095`, `rollout`) rate a FIFTH, EARLIER node at 2M (the self-play crossing
> lottery, rule 15), so a naive `first_n=4` would have read their **8M** node against everyone
> else's 10M — matched count, un-matched steps, and nothing in the tree warns about it (hazard 1).
> The correction moves those three by −6.3 / +3.1 / −4.8; the twelve four-node arms reproduce their
> committed numbers exactly. All fifteen finished at 10,027,008 steps, so the newest-node inflation
> is common-mode; the 8M node of the same fit is the registered cross-check (Spearman 0.775).
> **The floor.** Max pairwise |Δ| over the three controls = **45.0 Elo** (pairs 45.0 / 35.3 / 9.7),
> reproducing the 2026-09-09 figure exactly from an independent refit. 8M floor 54.9.
> **The read.** Range 145.5 Elo, `rollout` 2062.3 top to `vf15_b` 1916.8 bottom. On the registered
> rule (all three |Δ| past the floor, same sign, each Δ's own CI excluding the floor) **NO ARM is
> outside the floor.** `vf15_b` (−101.9 / −56.9 / −66.6) and `strata_b` (−90.7 / −45.7 / −55.4)
> satisfy the first two clauses and fail the third ⇒ CANDIDATES, per rule 22. On the 8M node
> `strata_b`'s lean COLLAPSES (−48.9 / +6.0 / −0.6; it is also the one arm whose rating falls 8M →
> 10M) and is reported INCONCLUSIVE; `vf15_b`'s GROWS (−133.4 / −78.5 / −85.1, past the 54.9 floor
> on all three) and is the family's only node-robust lean. `rollout` is first at 10M and sixth at
> 8M — the cleanest demonstration on this table that one node is not a reading.
> **THE INSTRUMENT'S OWN LIMIT, registered as P3 before the read.** `se(Δ)` = 21–24 Elo at four
> nodes ⇒ CI95 ±43–48, the size of the floor itself, so the delta-CI clause can essentially never
> fire here. **A ladder-strength question on 10M arms needs more PROMOTIONS, not more arms.**
> **RULE 22 REPRODUCED ON STRENGTH.** Same-pin seed replicate pairs: `vf15` − `vf15_b` = **+82.6
> [+39.5, +125.7]**, 1.84× the floor, CI clear of zero; `strata` − `strata_b` = **+58.1 [+16.2,
> +100.0]**, 1.29×; against the same-configuration control pair's 9.7 [−53.1, +33.7], 0.22×. The
> control triple bounds the CONTROL's variance, not the lever's — and on strength the understatement
> is a factor of 6–8.
> **Descriptors (not the read).** All fifteen carry `eval_sentinel_greedy: true` with
> `sentinel_regime {greedy, symmetric_teams}` recorded on every 10M row, so the whole family is
> post-boundary and comparable within itself only. `win_rate_vs_bots` @10M spans **0.839–0.910**
> (800 games, 8 bots, `random` excluded per `eval_callback.py:363`; every `random` row 0.99–1.00),
> i.e. compressed but NOT saturated — P4's 0.93–1.00 was wrong on level. 🚨 **That row is an INPUT
> to the headline, not an independent descriptor of it**: `fit_ladder` folds each snapshot's bot
> edges in as the anchor, so the Pearson 0.889 / Spearman 0.768 against Elo is partly mechanical
> (hazard 3). Pins span seven commits (`f3502568`, `377a5aa1`, `f871e79f`, `28ece02a`, `46ca68ef`,
> `40bf23d9`, `d11386dc`), each a single `pin_commit` span from step 0; seeds 42 / 1001 / 1002 /
> 1003. **The floor's two widest control pairs CROSS the `f3502568` → `377a5aa1` boundary and the
> one same-pin pair does not** (45.0 / 35.3 vs 9.7) — an OBSERVATION only, since se(Δ) = 22.9 cannot
> separate 9.7 from 45.0, and it makes the 45.0 bar conservative for a same-pin comparison.
> **Other hazards as findings:** non-transitivity is material (`fit_quality.max_abs_err` to 0.194 on
> `lambda09`); `ai_v12_26_ladder_ctrl10M_shaped` was LIVE with no `snapshot_ladder/` at all and is
> EXCLUDED, not read (rule 16) — it is this family's sixteenth arm and the table must be re-read,
> not patched, when it lands; `strata_b` and `vf15_b` share a bit-identical 0.83875 bot rate on
> completely different per-bot rows (coincidence, verified).
> **Consequence.** The strength read agrees with the critic read: after fifteen 10M arms and seven
> levers the campaign has **zero detections on strength** as well as zero confirmed detections on a
> decision row. Strength adds no arm to the queue and removes none; what it adds is the floor's
> scale on the family's own yardstick and a measured bound on the yardstick's resolution.
> Tag: **MEASURED · NO ARM OUTSIDE THE FLOOR · rule 22 reproduced on strength · the four-node
> ladder cannot resolve a 45-Elo question**.

## 9. FILES

| file | what |
|---|---|
| `PREDICTION.md` | the pre-registration, committed separately at `937775b6` (incl. the disclosure) |
| `read_ladder_strength.py` | the read — roster, matched-count refit, floor, decision rule, cross-check |
| `addenda.py` | the three POST-HOC observations (bot-row mechanism, within-lever spread, pin cells) |
| `ladder_strength.json` | every number, plus each arm's committed ladder verbatim |
| `addenda.json` | the post-hoc rows |
| `ladder_strength_table.txt` | the script's own rendering of both tables |

Reproduce: `export PYTHONPATH=$PYTHONPATH:src && python3 read_ladder_strength.py <out-dir>` then
`python3 addenda.py <out-dir>/ladder_strength.json`. CPU only; no GPU; no write under `models/`.
