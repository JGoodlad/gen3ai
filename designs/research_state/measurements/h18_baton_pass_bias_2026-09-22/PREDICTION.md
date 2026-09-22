# H18 — HOW BIG IS THE BATON PASS BIAS? — the pre-registration

**Committed BEFORE the first measured game.** The backlog row it discharges is
[`metamon_obs_faithfulness_2026-09-22/README.md`](../metamon_obs_faithfulness_2026-09-22/README.md)
§8; the hazard it sizes is `EXTERNAL_ANCHORS_SOP.md` **H18**.

---

## 0. The finding this measures the SIZE of

Metamon runs **upstream poke-env 0.8.3.3**, which clears the entrant's boosts on a
`|switch|…|[from] Baton Pass`. The protocol emits nothing else about the pass
(`sim/pokemon.ts:1249` — `this.boosts = pokemon.boosts`), so Metamon **loses every passed stat
stage**, on its own side and on ours. Our fork fixed exactly this on 2026-08-23
(`src/poke_env/battle/baton_pass_carryover_test.py`).

Metamon therefore plays a Baton Pass line weaker than it otherwise would, so **every anchor win
rate this era is OPTIMISTIC for us by an amount that has never been measured.** The exposure is
known (13 of 871 decision points, 3 of 20 battles); the win-rate effect is not.

🚨 **A direction is not a size.** 13 mis-read decisions in 871 is not a win-rate delta: a mis-read
boost decides one turn and is irrelevant on another. This cell measures the delta.

---

## 1. THE CELL — fixed here, before any measured game

| | |
|---|---|
| opponent | `metamon:SmallRL` (ckpt 40, 13.9M), `--regime greedy` — greedy-vs-greedy per SOP RULE 2 |
| our side (arm W) | `models/ai_v13_02_flywheel_winprob/final_model.zip` @ **75,005,952** steps |
| transport | `--server rust` (the default; no Node), ports 9500–9599, stopped by PID |
| compute | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, `OMP_NUM_THREADS=1`, ≤ 2 cells at once |
| **n** | **400 games per arm per team set** — 1,600 games total |
| seeds | `--seed-base 20260922`, `--team-seed 20260914` (default) — **identical in both arms** |
| code pin | worktree `h18` @ `d5c465fd`, so a moving `main` cannot enter the campaign |

### The two ARMS

| arm | Metamon's `poke_env` is | how |
|---|---|---|
| **U** | **as installed** — upstream 0.8.3.3, unpatched | `$GEN3AI_METAMON_PYTHON` → `python_unpatched` |
| **P** | a **SHADOW COPY** carrying our fork's fix, ahead of site-packages on `PYTHONPATH` | `$GEN3AI_METAMON_PYTHON` → `python_patched` |

**The shared conda env is NEVER edited.** The patch is applied to a *copy* of the installed
package under the session scratch, and each arm's wrapper prints `poke_env.__file__` into the
peer log, so which copy loaded is a **recorded fact, not an assumption**.

🚨 **A missing arm selection REFUSES.** `opponents.metamon.python` in the H18 config is a path
that does not exist, so a cell launched without `$GEN3AI_METAMON_PYTHON` dies by name instead of
quietly running the installed (unpatched) interpreter. This is not hypothetical — that slip
happened once during setup here, and the provenance line is what caught it.

### The two TEAM SETS

| team set | what | Baton Pass density |
|---|---|---|
| **away** | Metamon's own 20-team `competitive` gen3ou set — the standing Tier-B set | **8 / 20 teams** carry Baton Pass |
| **enriched (home)** | the 20 pool teams drawn by `random.Random(20260922).sample(sorted(BP_teams), 20)` from the **157 of 719** pool teams that carry Baton Pass | **20 / 20** |

Both sides draw from the SAME set in a cell, as always; `team_source_asymmetry` must read false.
The enriched files are the pool's own EXPORTED pastes (complete Hidden Power IV lines,
nickname-free — hazards H2/H3), byte-identical for our side and the peer.

**The 20 enriched teams, stated here so the draw cannot be re-rolled:**

```
015e78dda7  13e0ce541b  1baeb6d658  1d4c4974a0  2439039083
355789e24b  42808b709b  5876ebb9eb  8f74a43699  a917696663
b61e07ab58  c885693dcf  cc8bc0e125  cc901d75a9  cd808fc11f
ce61c549f7  d3f703b88d  df3f7a916c  f577e001c2  f947eeb25e
```
(each `<name>.gen3ou_team`; the full list with its provenance is `bp_enriched_sample.json`.)

---

## 2. THE STATISTIC, and the BAR

For each team set, the contrast is **P − U** on OUR win rate, with a **Newcombe 95% CI**
(`main.anchors.results.newcombe`, method 10 — the difference interval consistent with the Wilson
cell intervals). **The bias on our reported anchor numbers is `−(P − U)`**: patching Metamon
*removes* the handicap, so our win rate should FALL, and the amount it falls is the amount every
unpatched anchor number has been flattering us by.

🚨 **An effect whose CI covers zero is NOT DETECTED — never "equal", never rounded to a pass
because the point estimate has the expected sign.** At n = 400 per arm the Newcombe half-width
near 0.5 is ≈ ±0.069, so this cell resolves a bias of about **7 pp** and no better.

### The branches, fixed in advance

| branch | condition | what gets written |
|---|---|---|
| **(a)** | detected on **enriched** and NOT on **away** | the bias is real and **concentrated where Baton Pass appears**; quote it per team set, and say plainly that the standing `away` numbers are not measurably affected |
| **(b)** | detected on **BOTH** | quote it per team set **and** re-state every era anchor number with the correction as a footnote — **a table in this README, never an edit to the ledger** |
| **(c)** | detected on **NEITHER** | the bias is **below the resolution of this exposure**; state the bound the CIs actually support and do not round it to "no bias" |

A fourth outcome is possible and must be reported as loudly: **detected on `away` but not on
`enriched`**. That would contradict the mechanism, and the honest response is to say so rather
than to pick the half that agrees.

### Recorded alongside, so the bias can be quoted PER EXPOSURE

From the captured protocol of every battle (`bp_exposure.py`):

* games containing **any** Baton Pass;
* games containing a Baton Pass that **carried a nonzero stat stage** — the only kind the defect
  can lose anything to.

**A bias quoted per game is diluted by every game the mechanism never appeared in.** Both the
per-game and the per-exposure figure are reported.

---

## 3. WHAT MAKES A CELL NOT A MEASUREMENT

Fixed here so it cannot be decided after seeing a number. A cell is discarded and re-run if:

1. `status != "OK"`;
2. `regime_verified_decisions != true`, or any `argmax_match_rate != 1.0000` (greedy);
3. `team_source_asymmetry != false`;
4. the arm's `[h18] ARM=…` provenance line is absent from a peer log, or names the wrong copy;
5. `hit_forfeit_limit` exceeds 25% of the cell (a timeout is never a semantic outcome).

`peer_clean == false` is **NOT** grounds for discarding a cell (hazard **H16/H17**) — it is a
reason to read a log. Metamon's post-game `RecursionError` happens after the last decision.

### The patch is proved to be a patch — a POSITIVE-CONTROL PAIR

`baton_pass_patch_test.py` is our fork's carry-over test ported to the upstream API. It is run
under **both** interpreters before the campaign, and both directions must hold:

* **arm U must FAIL** all four carry-over assertions (boosts, Substitute, the opponent's pass,
  negative stages) — otherwise the arms are not distinct and the whole cell is vacuous;
* **arm P must PASS** all four, **and both arms must pass both invariants** (a plain switch and a
  phaze drag still clear the boosts) — otherwise the patch is not the fix, it is a blanket.

Run 2026-09-22, before registration: **U fails 4/4 carry-overs and holds both invariants; P
passes 4/4 and holds both invariants.**

---

## 4. THE PREDICTION

**Registered belief: branch (a) or (c) — most likely (c) on `away` and (a)-or-(c) on
`enriched`; branch (b) is unlikely.**

The reasoning, so a wrong prediction is informative rather than embarrassing:

* the mechanism needs a pass that **carries stages**. In the faithfulness capture only **3 of 24**
  Baton Pass switch events carried a nonzero boost, and in this campaign's 10-game pilot **0 of 8**
  did. Most Baton Pass usage in gen3ou is a pivot, not a setup pass;
* the defect is **symmetric in whose setup is lost** — Metamon under-reads its OWN passed stages
  as well as ours. Fixing it makes Metamon value its own passes correctly, which helps it, *and*
  makes it respect ours, which also helps it; but the two do not obviously add to a large number;
* the enriched set is the strongest exposure obtainable (20/20 teams), so if the effect is not
  visible there at n = 400 it is not visible anywhere this instrument can reach.

**Point prediction: |P − U| < 0.05 on `away`; 0.00–0.08 against us on `enriched`.**
If `enriched` lands outside that, the prediction was wrong and the number stands.

---

## 5. DISCLOSURE — what had already been seen when this was written

**Two 10-game pilots on the ENRICHED set had already run** as pipeline validation, and their
numbers were visible before this file was committed. They are stated here rather than buried,
because a registration written after glancing at a result is only credible if the glance is
declared:

| pilot | arm | n | win rate |
|---|---|---:|---:|
| `pilot_default_python_UNPATCHED` | U (the slip that made the config refuse) | 10 | 0.800 |
| `pilot_P` | P | 10 | 0.700 |

At n = 10 a difference of 0.100 carries a Newcombe CI of roughly **[−0.30, +0.47]** and is worth
exactly nothing; it is not evidence for any branch and did not set the prediction above (which
follows from the 3-of-24 exposure rate, known since the faithfulness pass). **Neither pilot's
games enter the reported cells** — the campaign's four cells are run fresh at n = 400.

---

## 6. WHAT THIS CELL STILL CANNOT SAY

* **Nothing about `SyntheticRLV2`.** It is a different policy with a different sensitivity to
  the same mis-read; this measures `SmallRL`.
* **Nothing about the volatile half of the mechanism** beyond whatever fires by chance. Substitute
  and Leech Seed ride a pass too; the patch carries them, but the exposure is not controlled.
* **Nothing about whether the anchor SHOULD be patched.** It should not: the SOP's answer stands
  — a patched Metamon is a different opponent from the one its authors publish. This measures the
  size of the handicap so that the published number can be read with it.
