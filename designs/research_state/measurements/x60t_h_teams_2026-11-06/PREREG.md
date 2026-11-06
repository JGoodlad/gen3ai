# X60-T: human-built ladder teams (H-TEAMS) at the production plateau, against C0 and D1: PRE-REGISTRATION (DRAFT, 2026-11-06)

**Status: DRAFT. Not registered, not built, not launched.** No `rb_x60t_*` run exists and no X60 game has been
played. It becomes a registration only when every `<<TBD: …>>` below is filled by its stated rule, the two
adversarial reviews have run (data / bias; experimental / causal, design §6.9 item 6), and the coordinator commits
it. After the first ON arm launches, nothing changes except by a dated amendment, written before the read, that
says why.

**Written AFTER X59-S's outcome (REVIEW F F-M1, recorded honestly).** X59-S read on 2026-10-30
(SELF-PLAY-DATA-BETTER; Δ_labels OWN-LABELS-BETTER; the closure BLOCKED by Δ_restriction, `5bbba6096`). Its ≥ 1500
follow-up read on 2026-11-03 (rating NOT-RESOLVED). X61P's C0 cells, which this read imports, were read on 2026-10-30
(C0 F 3,023 / 6,000 = 0.5038). The route was fixed by design §6.1's table, row 3: the actions hold OWN-LABELS-BETTER
and no closure. That table names OWN-LABELS-BETTER explicitly ("anything else … ⇒ X60-T"). The routing was applied
by reading the X59-S record, not by a coded `x60_route()` (F-M1 fix 2 is unbuilt, FINDING 1). To keep the
post-outcome freedom closed, §2 freeze-diffs this file against the design's §6 and lists every departure. Each
departure is a placeholder filled by a stated rule, or a REVIEW F fix, named by its finding id.

**The analysis behind every number:** [`analysis.py`](analysis.py) → [`results/analysis.json`](results/analysis.json)
(CPU, seeded `20261106`, 40,000 draws a cell). It IMPORTS the design's simulator
[`x60_redesign_2026-10-26/power.py`](../x60_redesign_2026-10-26/power.py) (`finals`, `rule_x60`; sha256
`7c1330a5…8f25`) and never re-implements the registered rule.

**Abbreviations.**
- P\* = `rb_x26_s2601` @ 280,000,170.
- C0 = X35's plain continuation (`rb_x35_c0_s<seed>`, pin `f4b25238`).
- D1 = `rb_x35_d1wide_s3501`–`s3505` (forum teams `w_forum_v1` at share 0.5).
- ON = the H-TEAMS arm, `rb_x60t_s<seed>`.
- F / H / P72 = `forum72` / `heldout72_hpfix` / `p72`.
- S = X35's population score against the frozen reference population R (`reference_population_production.json`,
  content `cef0960e…`), 7 members × 500 mirrored pairs a final.
- I_F = ON − OFF in S on F.
- R72 = the positive control: 72 of the corpus's own teams.
- σ = the seed-to-seed SD of a final's F rate.
- pp = percentage points; LB / UB = lower / upper bound; lane-h = one CPU lane for an hour.

## 1. Question and mechanism

Does mixing **human-built ladder teams** (`hr_spec_v1.r3`, ≈ 2,000–3,000 teams, the exact count `<<TBD: N>>`) into
the production plateau's training, at D1's dose, raise how well it pilots **human-built forum teams it never saw**
(F) against Kakuna? And is the gain the LADDER part, or would forum-team breadth (D1) do as well?

D1's forum teams read F +4.0 pp at 140 games a final, but F is D1's own population. H-TEAMS trains on a different
human population (ladder), so F is unfamiliar to it in team AND population. That is the claim F carries.

**What it does not test:** a fresh run or a warm start (the plateau-plasticity confound, as N1-P); any other share
or corpus. A NO-USEFUL-GAIN rules out "H-TEAMS at the plateau, at share 0.5, for 20M steps" only.

## 2. The freeze (F-M1 fix 1) and every departure from design §6

- **Frozen source:** `designs/endstate/design_human_replay_lever.md` at `e1512c361`.
  - File sha256: `ad34aee94f629b873d538722e57abe352f6c43bfdc1f0e30cd880d07f942174f`.
  - §6 text ("## 6. X60" up to "## 7. Appendix") sha256: `f288b331245c9652aba69542c11dae7bd152029f4c85ef4110b08554ff2330fe`.
  - §6 is byte-identical at this worktree's HEAD `0b9e6f354`. The two later commits to the file (`0534e90a6`,
    `8e9448a77`) touch §2.2 and §9 only.
- **Everything not listed below is §6.3 / §6.4 / §6.6 / §6.7 verbatim:** the arms, the side pin, the 60-unit F cell,
  the interval (binomial + `2 · max(1.0 pp, σ̂)² / n` + the Welch guard), the futility-only look 1, the classes and
  their 1e-9 tie rule, Δ_P's rule, goal (a), the X35 guards, the VOID list, R72's construction and the T − D1 arena.

| # | departure | answers | why |
|---|---|---|---|
| X1 | **8 seeds an arm** (ON 3501–3508; C0 + 3506–3508), not 5. Look 1 stays 3 v 3; look 2 is 8 v 8 | F-M4 fix 3 | σ̂'s one-sided 95 % UB from X61P's 10 finals is **2.04 pp > 1.5** (§5.1), decided on control-only data before ON trains |
| X2 | power quoted as a range over σ ∈ [σ̂ point, σ̂ UB], headline at the 1.0 pp floor | F-M4 fix 1 | the old 0.80 was the σ = 0 cell |
| X3 | **KAKUNA-SPECIFIC iff I_F's one-sided 90 % LB ≤ 0** (was: the point ≤ 0) | F-M6 fix 3 | leak 0.10 against 0.50; at any exchange rate ≥ 0.5, its cost to a true +3 pp gain ≤ 0.02 (§7) |
| X4 | NO-TRANSFER also needs I_F not GAIN; new class **IN-HOUSE-ONLY** | F-M5 fix 2 | R72 and I_F are both in-house; an in-house F gain with a flat Δ_F is not "no transfer" |
| X5 | a look-1 futility stop reports **NO-USEFUL-GAIN undivided** (no NO-TRANSFER / NOT-DELIVERED) | F-M5 fix 3 | R72's 3 v 3 MDE (1.1–2.6 pp) is not shown to be reachable (§9) |
| X6 | **ONE eval harness for every X60 game:** the play path of `de15dfc7b` (engine E3). C0's cells import from X61P's states. D1's units 0–29 are re-played at E3 | F-M2 fix 1 | the regime C0's banked cells were played at; D1's banked units are E1 and `train` mode, so they cannot import (§4) |
| X7 | the C0 reuse check's remedy is split: a training-side miss ⇒ a fresh C0h; an eval-side miss ⇒ a CPU re-play | F-M2 fix 3 | an eval mismatch needs no GPU |
| X8 | the corpus passes the soak, `corpus-check` and a pack-identity check **at the side pin**. Differing teams drop. The pin carries the AM-D1-FIX label guard | F-M3 fixes 1–4 | at `f4b25238` the packer mis-packs Frustration and Hidden Power IV lines |
| X9 | T − D1 is always read **5 v 5** (ON's first five finals v D1's five), only after look 2 | m1 | §6.4's "3 v 3 if look 1 ended it" let the primary pick the contrast's seeds |
| X10 | LADDER-BETTER / FORUM-BETTER reworded as home-advantage sums | m3 | each arm owns one panel |
| X11 | **pause / resume:** a seed resumed under §13's conditions is read as that seed's final. §6.3's "re-launched with its seed" applies only when they fail | brief item 6 | §6.3 had no resume rule |
| X13 | the F floor: each arm counts ≥ 1,200 · n − 20 F games, else INCONCLUSIVE | none (X61P §4's floor, carried) | §6.3 names no floor; a short cell must not decide |
| X12 | placeholders: r3's id, digests and N; the side pin; the clause hash; R72's digest; the play pin; the specs | F-M1 fix 1's list | sibling builds in flight |

## 3. The arm

| item | value |
|---|---|
| start | P\* = `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…91ed`, R's frozen call), as C0 |
| the ONE lever | `--human-data teams:hr_spec_v1.r3@0.5`, team sides both: with probability 0.5 per episode and side the team is a corpus team, else exactly C0's draw. This is D1's dose (`--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both`), so ON − D1 differs in the corpus alone. Clause hash `sha256("human_data_v1:" + canonical)[:16]` = `<<TBD: clause hash>>` |
| everything else | C0's argv token for token: `--behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --seed <s> --run-name rb_x60t_s<s>`, the recipe inherited from X26's record (`<<TBD: kit arm name and its kit_test "C0 + the clause and nothing else">>`) |
| pin | **the side pin** `<<TBD: sha, tag>>` = `f4b25238` + the switch commits + the corpus files + the AM-D1-FIX label guard (X8). `kit pin-check` enumerates every backported path (F-M3 fix 4) |
| seeds | look 1: **3501–3503**. Look 2, on CONTINUE only: **3504–3508**, plus C0 **3506–3508** at `f4b25238` with C0's argv (X1). Fixed: no extension, no adaptive rule. The 8-seed count is decided here, on X61P's control-only data (§5.1) |
| control | C0 s3501–s3505 (each K9(b) PASS), already trained: 0 GPU-h. C0 s3506–s3508 are trained on CONTINUE (≈ 12.3 GPU-h; §12). Every C0 final is pinned by sha256 in the read specs |
| contrast arm | D1 s3501–s3505 (pin `440b61c32`), already trained: 0 GPU-h |
| corpus | `hr_spec_v1.r3` (chain `hr_spec_v1` `e7d491e7…` → r1 `e9e0df42…` → r2 `2a413c85…` → r3 `<<TBD: rows-manifest sha256>>`): r2's kept battles, actor ≥ 1500 full-six seats, C-GAME-completed, HP-repaired, validator-clean, ≤ 5 teams per account, both panel nets. Count `<<TBD: N>>`. Built once, as `n1_hrep_v1` too, behind ONE provenance record (F-M7) |

**Dose (ESTIMATE, D1 doc §4.1's rule: ≈ 570k games in 20M).** At share 0.5 a side draws ≈ 285k corpus teams, so each
corpus team is drawn **285,000 / N times a side**: 190 (N 1,500), 143 (2,000), 114 (2,500), 95 (3,000). That is
12–24 % of a pool team's ≈ 790 draws a side in C0, and 1.0–2.0 % of P\*'s ≈ 9,300 cumulative pilots per pool team.
The pool's exact exposure falls by half (as D1's and N1-P's), which is what Δ_P reads. D1's own doc quotes ≈ 118
draws a side at 1,981 teams, where this rule gives 144 (FINDING 9).

## 4. One eval harness (F-M2) and what the imports need

**Registered:** every X60 Kakuna unit and every S cell is played on the play path of **`de15dfc7b6d19d6f0e4a3724f269e26e7aa22957`**
(X61P look 2's commit, tag `x61p-n1p-look1-sched-de15dfc7`). Its regime, checked by `analysis.py` `fm2`:
- engine **E3** (`port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126`, from `main.reads.engine.engine_at`), equal to X61P's cells;
- Kakuna's team pack: legacy;
- peer code / policy mode: legacy `b817d3216c418f75` / `eval`;
- OUR side's pack: post-fix.

Both arms trained on E1 (`f4b25238`). Both are read on E3, so the train / eval engine gap is common to ON and C0,
not charged to ON alone. The record is written from a main checkout (AM-N1P-2's pattern).

**The play pin** is `<<TBD: sha>>` = `de15dfc7b` + ONE commit (X63's successor-pin pattern). The commit holds:
- the `x60` read rule (`main.reads.rules`; design §6.9 item 2), because the play commit's spec parser refuses a rule
  it does not hold;
- R72's panel registration (`heldout_panels.json`, `held_out_from: []`), which `main.h2h --panel` needs;
- optionally, the two-state import below.

**Required at the play pin, before any X60 game** (each a HOLD on failure, never a launch):
1. `playpath.compare` against `de15dfc7b` shows no game-path difference beyond the panel registry.
2. `engine_at(<play pin>) == E3`.
3. forum72's id stays `2118812c27885de3` (X61P's primary group `00c8c0114bdf0836`), the reported group's stays
   `3d2cbdc0b1413bcb`, and S home72's stays `d4de0fd7c16a2658`. Else C0's cells cannot import (an eval-side miss ⇒
   re-play, X7).
4. One ON final loads at the play pin and plays one unit. The side pin adds a provenance record to the checkpoint,
   and X59-S's Amendment 8 met a pin that could not load newer students (`<<TBD: load check>>`).

**What `main.reads` import needs, and where C0 falls short** (`main.reads.imports`, read for this draft):
- **Needs:** the earlier state's `spec.json` sha256 (look 1 `1266daad…9f03`; look 2 `f28654cc…2620`); the same cell
  (opponent `metamon:Kakuna`, team sets and unit seeds that include X60's, arm / seed / load options per label); and
  the same bytes PLAYED (C0 sha256s: s3501 `09a6a2df…`, s3502 `7e9a9890…`, s3503 `99096be8…`, s3504 `26835480…`,
  s3505 `bb10cd96…`).
- **On the same engine** (an import across engines refuses), the same Kakuna pack and the same sides.
- **One regime id per team set** after pooling. For S: the same R content, panel `home72`, 500 pairs, batch 100,
  regime `d4de0fd7c16a2658`.
- **The gap: C0's cells sit in TWO states.** s3501–s3503 were played in look 1's state; s3504–s3505 in look 2's,
  which only imported the first three. A group takes ONE `import` (`spec._import`), and an import reads only the
  earlier state's OWN played units (`imports.anchor_rows`), never a transitive import. So:
  - **Look 1 imports C0 s3501–s3503 from look 1's state: 0 C0 games.**
  - **At look 2,** C0 s3504–s3505's F, reported and P cells are either:
    - imported through a two-state import built into the play pin (`<<TBD: build commit, or "not built">>`); or
    - RE-PLAYED at the play pin: 2 × (1,200 F + 420 reported) games + 2 S sets.
  - Kakuna play is deterministic per model, unit seed and engine (0 of 2,100 replayed games differed on one engine),
    so a re-played F count equals look 2's. A differing count is an eval-side drift: HOLD (FINDING 3).
- **The determinism guard** (§6.3): one imported C0 cell (c0_s3501 forum72 unit 0) is re-played game for game,
  choices included, at the play pin. A difference is a HOLD; the cells are re-played (X7).
- **D1 cannot import.** D1's banked H / F units were played on E1, with peer code `f230353c40e43488` and policy mode
  `train`. An import across the engine or the sides refuses (`imports.check_engine`, `check_sides`). **D1's units
  0–29 on H and F are re-played at the play pin: 5 × 2 × 600 = 6,000 games.** D1's banked units 0–6 enter nothing.

## 5. PRIMARY Δ_F (`main.reads`, rule `x60`, design §6.3 verbatim except X1)

- **Cell:** Kakuna ckpt 34, greedy v greedy, `forum72`'s committed symmetric config, **60 role-balanced 20-game units
  per final (unit seeds 0–59, 1,200 games)**, both arms.
- **Interval:** Δ̂ = ON's pooled F rate − OFF's. V = Σ_arm p(1 − p) / (1,200 n) + 2 · max(1.0 pp, σ̂)² / n, with
  σ̂² = W − b (W the mean of the arms' sample variances of their per-final F rates, df 2 (n − 1); b the mean per-final
  binomial variance). **Guard:** if σ̂²'s one-sided 95 % LOWER bound W · df / χ²(df, 0.95) − b exceeds (1.0 pp)², a
  Welch t interval on the per-final rates decides instead. 95 % two-sided.
- **Looks:**
  - **Look 1 (3 v 3) is futility only:** Δ̂ ≤ 0 (within 1e-9 counts) ENDS the read with look 1's class, never GAIN.
  - **Otherwise look 2 (8 v 8) decides.**
- **Classes** (a bound within 1e-9 of 0 or +3.0 takes the undecided class):
  - **GAIN** iff LB > 0;
  - **LOSS** iff UB < 0;
  - **NO-USEFUL-GAIN** iff 0 ≤ UB < +3.0 pp;
  - else **NOT-RESOLVED**.
  - "Equivalent" (inside ±3 pp) is reported, deciding nothing.
- **Floor:** each arm counts ≥ 1,200 · n − 20 F games; else INCONCLUSIVE.

### 5.1 σ̂ from X61P's 10 finals (F-M4 fix 2), and the seed count (fix 3)

The estimator is §6.3's, on X61P look 2's `cells.json` (forum72, E3, 60 units a final, ties as non-wins):

| finals | per-final F rates (%) | W SD | binomial SD | φ̂ | df | σ̂ point | **σ̂ one-sided 95 % UB** |
|---|---|---:|---:|---:|---:|---:|---:|
| **C0 + N1-P (registered)** | C0 50.58 52.08 49.83 50.83 48.58 · N1-P 51.58 51.83 51.33 51.83 48.08 | 1.46 pp | 1.44 pp | 1.02 | 8 | **0.21 pp** | **2.04 pp** |
| C0 alone | as above | 1.29 | 1.44 | 0.80 | 4 | 0 | 2.70 |
| N1-P alone | as above | 1.61 | 1.44 | 1.24 | 4 | 0.71 | 3.53 |

**Decision (F-M4 fix 3, control-only data, no X60 data): UB 2.04 pp > 1.5 pp ⇒ 8 seeds an arm.** Every subset agrees.
The rule was written in REVIEW F before this σ̂ was computed. The new C0 seeds 3506–3508 cost ≈ 12.3 GPU-h
(3 × 4.10, the mean of the 20 X35 production forks' dose records, 3.84–4.43). The guard's lower bound is negative
(−0.99 pp²), so on this data the Welch guard would not fire.

### 5.2 Power of the registered rule at 8 seeds (headline: the 1.0 pp floor)

Rows from `power.py`'s `finals` + `rule_x60` with `seeds=8` (look 1 at 3). Base F 0.49 is power.py's; X61P's C0 read
0.504, so the binomial variance moves < 0.1 %.

| true σ | GAIN at +2 / **+3** / +4 pp | false GAIN at 0 | NO-USEFUL-GAIN at 0 | NO-USEFUL-GAIN at a true +3 (m2) | half-width, look 2 | Welch guard fires at 0 |
|---:|---|---:|---:|---:|---:|---:|
| 0.21 pp (σ̂ point) | 0.63 / 0.96 / 1.00 | 0.009 | 0.93 | 0.013 | 1.72 pp | 0.002 |
| **1.0 pp (the floor)** | **0.56 / 0.89 / 0.99** | **0.019** | **0.83** | 0.030 | 1.72 pp | 0.05 |
| 1.5 pp | 0.45 / 0.78 / 0.95 | 0.026 | 0.67 | 0.044 | 1.99 pp | 0.27 |
| 2.04 pp (σ̂ UB) | 0.33 / 0.61 / 0.84 | 0.029 | 0.48 | 0.054 | 2.64 pp | 0.63 |
| 2.75 pp (power.py's old UB) | 0.22 / 0.42 / 0.66 | 0.024 | 0.28 | 0.048 | 3.29 pp | 0.91 |

- **Headline: GAIN at a true +3 pp is 0.89 at the floor, ranging 0.61–0.96 over σ ∈ [0.21, 2.04] pp.** At 5 seeds
  the same cells read 0.70 (floor), 0.80–0.42 over the range.
- A look-1 futility stop ends 50 % of nulls, and 1.8 % of true +3 pp levers at the floor (3.9 % at 1.5 pp).

## 6. CO-PRIMARY Δ_P (D1's rule verbatim)

S on `home72` vs R, 7 × 500 pairs a final (regime `d4de0fd7c16a2658`). Δ_P = mean ON S − mean C0 S, Welch two-sided
97.5 %. **NON-INFERIOR iff LB > −2.0 pp.**
- **C0's P nodes:** s3501–s3503 import from X61P look 1's S group `p` (family `x61p_n1p_p_look1_2026-10-26`, ledger
  `70187d7b…`). s3504–s3505 come by the two-state import or a re-play (§4), and s3506–s3508 are played.
- **Known now** (m6): C0's five S values are 56.82, 57.39, 56.84, 57.37, 56.59 (SD 0.36 pp).

## 7. I_F, the clean meter, and the KAKUNA-SPECIFIC guard (F-M6)

I_F = ON − OFF in S on **forum72** (`main.h2h play-many --panel forum72`, 7 × 500 pairs a final, a new S family at
the play pin), Welch across finals.
- **I_F GAIN** iff its two-sided 95 % LB > 0.
- **The guard: KAKUNA-SPECIFIC iff I_F's one-sided 90 % LB ≤ 0.**

**The exchange rate between the meters (F-M6 fix 1) cannot be measured from banked data:**
- **No arm has I_F.** No S group on forum72 exists in any read state or spec in the archive or the measurement tree
  (`analysis.py` `fm6.inventory`).
- **What each arm has:**
  - D1, D3 and N1-P: Kakuna F (and H, P72) plus S on home72.
  - X26c: Kakuna vs P\* but S vs X26 @380M (different baselines).
  - The belief-prior arms: Kakuna only (a load option; no in-house read).
- **The only SAME-panel pair is home72, N1-P − C0.** Kakuna −1.86 ± 2.66 pp (379 / 700 vs 392 / 700) against S
  −3.28 ± 0.30 pp. The ratio is 1.77, but its 80 % range is [−2.6, +3.5]: uninformative.
- **Cross-panel, the meters move OPPOSITE ways under breadth levers** (Kakuna F vs S home72):
  - D1 +4.00 vs −1.61;
  - D3 +1.14 vs −0.45;
  - N1-P +0.55 vs −3.28.
  So P is a cost meter, never a proxy for I_F.
- **The rate registered for planning:** the structural one. If the lever shifts the trainee's play on F by δ logits
  against any opponent, each meter moves p(1 − p) δ, so the in-house gain per Kakuna pp is ≈ 0.98–1.00 (p_K ≈ 0.50,
  p_S ≈ 0.50–0.57). The error rates are registered across rates 0.25–1.8. The rate on F stays unmeasured until
  X60's own I_F (FINDING 4).

**The guard's two error rates (8 v 8, Welch, `fm6.guard_errors_8v8`):**

| per-final S SD, both arms | LEAK: pass at a true I_F of 0 (point / **LB90**) | COST: fail at Kakuna +3 pp, rate 0.25 (I_F +0.75) (point / **LB90**) | rate 0.5 (I_F +1.5) | rate ≥ 1.0 |
|---:|---|---|---|---|
| 0.36 pp (C0) | 0.50 / **0.10** | 0.00 / **0.003** | 0 / 0 | 0 / 0 |
| 0.56 pp (N1-P) | 0.50 / **0.10** | 0.004 / **0.10** | 0 / 0 | 0 / 0 |
| 0.86 pp (D1) | 0.50 / **0.10** | 0.04 / **0.34** | 0 / **0.017** | 0 / 0 |

**Choice, by the fix's own text ("if the rate allows, use I_F's one-sided 90 % LB > 0"):** the registered criterion
is "LB90's cost to a true +3 pp gain is ≤ 0.20 at the planning rate (1.0) and the largest measured per-final S SD
(0.86 pp)". Its cost there is 0.00, and it stays ≤ 0.02 down to a rate of 0.5. **So the LB90 rule is registered.** It
cuts the leak from 0.50 to 0.10. Its cost exceeds 0.20 only if the true rate is ≤ ≈ 0.25 AND both arms' SD is ≈ 0.86
pp. The I_F record reports its own per-arm SDs, so a reader can see when that corner applies.

### 7.1 End-to-end ADOPT-CANDIDATE power (F-M6 fix 2)

The simulation chains each meter (`fm6.end_to_end_adopt`):
- Δ_F through `power.py`'s rule at σ 1.0 pp;
- I_F through LB90;
- Δ_P through Welch 97.5 % > −2.0;
- goal (a) through ON's Kakuna home72 at 1,120 games, Wilson LB > 0.50, true rate C0's 0.560 + the lever's home72
  cost.

G-F / G-E / G-C are taken as PASS. The meters are drawn independently given the truth (FINDING 5). True Δ_F = +3 pp,
rate 1.0:

| true Δ_P, true home72 Kakuna cost | S SD ON / C0 | P(GAIN) | P(Δ_P NON-INF) | P(goal a) | **P(ADOPT-CANDIDATE)** | P(TRADE) |
|---|---|---:|---:|---:|---:|---:|
| 0, 0 | 0.56 / 0.36 | 0.89 | 1.00 | 0.98 | **0.88** | 0.02 |
| −1.0, −2.0 | 0.56 / 0.36 | 0.89 | 0.93 | 0.77 | **0.64** | 0.25 |
| −1.0, −2.0 | 0.86 / 0.36 | 0.89 | 0.65 | 0.77 | **0.45** | 0.45 |
| **−1.6 (D1's measured P cost), −2.0** | 0.56 / 0.36 | 0.89 | 0.23 | 0.77 | **0.16** | 0.73 |
| −1.6, −2.0 | 0.86 / 0.36 | 0.89 | 0.12 | 0.77 | **0.08** | 0.81 |
| −3.3 (N1-P's), −4.0 | either | 0.89 | 0.00 | 0.27 | **0.00** | 0.89 |

- **Under the null** (Δ_F = 0): false ADOPT is 0.002 with LB90 and 0.009 with the point rule.
- **Under a Kakuna-only +3 pp** (I_F 0), at Δ_P 0 or −1 pp: ADOPT is 0.04–0.09 with LB90, against 0.22–0.44 with
  the point rule.
- **What this says:** at D1's in-pool cost the likely positive outcome is **TRADE**, not ADOPT-CANDIDATE
  (FINDING 6). The rule is unchanged: Δ_P is not a REVIEW F finding.

## 8. Goal (a) guard

ON's Kakuna `home72`, 7 units a final (1,120 games pooled at 8 finals). Pass iff Wilson LB > 0.50. For reference:
- P\* reads 0.598;
- C0 reads 0.560 (392 / 700, X61P);
- N1-P reads 0.541.

## 9. Positive control R72 (F-M5, route T)

- **What it is:** 72 of r3's teams, ≤ 1 per account, seeded draw `<<TBD: seed, digest, ids>>`. They are in ON's pool,
  never in C0's or D1's.
- **Registered** in `heldout_panels.json` as a PRACTISED set (`held_out_from: []`).
- **The order (m5):** build r3 → draw R72 from it → register it, with a test that a rebuild does not exclude R72's
  teams.
- **The read:** ON − OFF in S on R72 (7 × 500 pairs a final), Welch 95 %. **DELIVERED iff LB > 0.**

**Dose per R72 team (§3):** 95–190 draws a side in 20M (N 3,000–1,500).

**MDE** (Welch two-sided 95 %, 80 % power, `fm5`):

| per-final S SD | 3 v 3 | 5 v 5 | **8 v 8** |
|---:|---:|---:|---:|
| 0.36 pp (C0, X61P) | 1.09 | 0.73 | **0.54** |
| 0.56 pp (N1-P) | 1.70 | 1.13 | **0.84** |
| 0.86 pp (D1) | 2.61 | 1.74 | **1.30** |

**The expected R72 effect is unmeasured** (F-M5's "not estimated" stands):
- Scaling P\*'s Kakuna home premium (+7.5 pp at ≈ 9,300 pilots a team, H-CTX) to this dose gives 0.08–0.15 pp if
  linear in exposure and 3.8–4.3 pp if logarithmic (GUESS bracket, Kakuna scale).
- The in-house scale multiplies that by the unmeasured rate (§7).

**Fix 3 applied:** the effect cannot be shown to reach the 3 v 3 MDE.
- **A look-1 stop reports NO-USEFUL-GAIN undivided:** R72 and I_F are REPORTED, and no NO-TRANSFER / NOT-DELIVERED
  split is made.
- **R72 decides only at look 2 (8 v 8).**

## 10. The contrast T − D1 (design §6.4, with m1 and m3)

- **Arms:** ON's first five finals (s3501–s3505) against D1's five, on **H + F at equal weight**. Units 0–29 of each
  (600 games per final and panel). D1's units are re-played at E3 (§4); ON's F units 0–29 are nested in the primary's.
- **Rule:** §6.3's interval, one look. **Read only after look 2, always 5 v 5** (m1). After a look-1 futility stop it
  is NOT READ, and the record says so. So its false rates are the "continued" path's: 0.009 FORUM-BETTER and 0.027
  LADDER-BETTER at σ 1 pp (REVIEW F m1). The record reports them beside the read.
- **Classes:**
  - **LADDER-BETTER** iff LB > 0;
  - **FORUM-BETTER** iff UB < 0;
  - **NO-DIFFERENCE-AT-3PP** iff the interval is inside ±3 pp;
  - else NOT-RESOLVED.
  - It is read whatever Δ_F's class; per-panel rows are reported.
- **Attribution (m3):** LADDER-BETTER means "H-TEAMS's gain on its own population (H) plus its transfer to F exceeds
  D1's gain on its own population (F) plus its transfer to H". It does NOT mean "ladder data beats forum-team breadth".
- **Power** (power.py, 600 a panel, 5 v 5): LADDER-BETTER at a true +3 pp 0.80 / 0.70 / 0.58 at σ 0 / 1 / 1.5 pp.

## 11. Verdicts and what each licenses

| verdict | rule | licenses |
|---|---|---|
| **ADOPT-CANDIDATE** | Δ_F GAIN (look 2), I_F LB90 > 0, Δ_P NON-INFERIOR, goal (a), G-F / G-E / G-C PASS | the re-read at the production checkpoint, then X54's sealed read. Reported with T − D1's class in m3's words |
| **TRADE** | Δ_F GAIN, I_F LB90 > 0, and Δ_P's inferiority not excluded, or goal (a) fails, or a guard FAILs | the exchange rate Δ̂_F / −Δ̂_P to the owner |
| **KAKUNA-SPECIFIC** | Δ_F GAIN, I_F LB90 ≤ 0 | nothing (a gain the in-house meter does not confirm; leak 0.10) |
| **LOSS** | Δ_F LOSS (either look) | drop H-TEAMS at the plateau |
| **NO-TRANSFER** | Δ_F NO-USEFUL-GAIN at look 2, DELIVERED, I_F not GAIN | stop this lever at the plateau: it moved what it trained on, not unfamiliar teams on either meter |
| **IN-HOUSE-ONLY** (new, F-M5 fix 2) | Δ_F NO-USEFUL-GAIN at look 2, DELIVERED, I_F GAIN | nothing for adoption, and the lever is NOT stopped: the meters disagree on F; reported to the owner |
| **NOT-DELIVERED** | Δ_F NO-USEFUL-GAIN at look 2, not DELIVERED | closes nothing: at this dose the plateau did not move even practised teams. A dose or fresh-run follow-up is a new registration |
| **NO-USEFUL-GAIN (look 1)** | Δ_F NO-USEFUL-GAIN at a look-1 stop | stop this lever at the plateau at this dose. Undivided (X5): R72 / I_F reported |
| **NOT-RESOLVED** | otherwise | nothing licensed; the next lever of §4; no extension |

**X35's guards** (G-F, G-E with a fresh `ge_x60t` reader, G-C) are read only on a Δ_F GAIN; otherwise
`guards_not_read` records why. **Shared control (m7):** C0 is X61P's and D3's control too, so a lucky-low C0 inflates
all three. Their GAINs are not independent replications, and wherever two are cited together it is said so.

## 12. REPORTED (never decided)

- **Δ_H, Δ_P72, Δ_home72** (Kakuna, units 0–6 both arms; ON plays H 0–29 for §10).
- **U** and the goal-(b) F + P72 statistic, each with its contamination named. P72 is fill-adjacent to the corpus;
  H is the corpus's own population.
- **σ̂ and each arm's own per-final SD,** for F and each S group (m8: a lever may add seed variance).
- I_F and R72's per-arm SDs; per-team proximity terciles; belief / intent calibration on the Kakuna cells.
- The share of corpus teams dropped by the pin's pack-identity check, and the corpus mons that touch any 10-26
  parity class (F-M3 fix 2).
- Every resume (§13).

## 13. Pause / resume of a seed (brief item 6)

**The resume.** A seed paused mid-run (a graceful stop, a crash, a box event) is resumed with:

```
python -m main.launcher --model models/rb_x60t_s<s>/checkpoints/<latest>.zip --steps 300000170 --device cuda \
  --human-data teams:hr_spec_v1.r3@0.5   # pinned to the side pin, from the checkpoint's own git_hash
```

The checkpoint lives in the run's own dir, so this is a SAME-RUN restart (`designs/launcher/restarts_and_resume.md`):
- `--fork-lr` is not re-applied;
- the self-play pool is not re-seeded;
- the recipe and run flags are restored from the run's own record;
- SB3 restores the optimizer and its LR.

The clause is typed: the switch's fork / resume refusal (§6.9 item 1) holds a resume that drops or changes it.

**The read takes a resumed seed's final iff ALL of these hold** (else it is VOID and re-launched from P\* with its
seed, §6.3):
1. `final_model.zip` at step ≥ 300,000,170 (the read's `final` ref, quiet 300 s).
2. `pin_history` is ONE span, the side pin (`python -m main.sidecar_audit models/rb_x60t_s<s>` not PIN-SPLIT).
3. `python -m main.dose models/rb_x60t_s<s>`: `lr_median`, `updates_per_env_step` and `dose_rate` each equal C0's
   to a relative 1e-6. All 20 X35 production forks read `dose_rate` 2.0440550334496103e-08 exactly, LR at its
   floor; a resume restores the optimizer, so equality is expected, and a miss means the resume changed the dose.
4. The `[HUMAN-DATA]` startup stamp, with the registered clause hash and rows sha, is in EVERY launch log of the
   seed, the resumed one included.
5. K9(b) PASS at arm end (it covers the whole run).
6. **One note in the record** per resume: the time, the checkpoint step resumed from, the steps re-trained, the cause.

**What it costs:**
- **Lost steps:** everything since the last resumable checkpoint, ≤ 1,000,000 steps (≈ 0.2 GPU-h at the measured
  1,371–1,482 steps/s of the 20 forks). A graceful stop saves at the next safe point and loses ≈ 0.
- **Startup:** each resume pays T2 parity and the compile again (≈ 4–8 min, ESTIMATE from the runbook's 256-env
  figures).
- **RNG:** the resumed run is NOT bit-identical to an uninterrupted one. The env, opponent-sampling and corpus-draw
  streams restart from the seed at the resume step, and the rollout in flight is discarded.
- **How that enters the seed term:**
  - A resumed final is a different draw from the same seed's distribution: same pin, data, recipe and dose.
  - The interval already treats each final's seed effect as random, with SD max(1.0 pp, σ̂). A resume adds no bias
    to that term and needs no correction.
  - The seed label no longer reproduces the trajectory, so "replicate = same pin + seed" is lost for that seed. The
    record says so.
  - C0's 8 finals run 0 restarts (dose records: `restarts 0`). If ≥ 3 ON seeds resume, the record reports Δ_F
    with and without the resumed seeds, descriptive only.

## 14. Pre-launch checks (a failure is a HOLD, never a launch)

`launch_x35.sh <arm> <seed> --pin <side pin> --lease-owner <owner>` runs `pin-check`, `corpus-check`,
`checkargs --pin`, the dry-run with the card visible and `stage-check`. Before the first GO, also:

1. **The smoke** (≈ 0.2 GPU-h, or the CPU `--debug` form): the pin's trainer, `--debug --steps 10000 --human-data
   teams:hr_spec_v1.r3@0.5`, `GEN3AI_MODELS_DIR` a scratch dir, under `mem_cap.sh`. It must finish with the
   `[HUMAN-DATA]` stamp, the realised human share within ±10 % of 0.5, and no FATAL. `<<TBD: smoke result>>`.
2. **Off-means-off at the pin:** `--human-data off` reproduces `f4b25238`'s `learner_golden` and the played-bits
   sentinel. `<<TBD>>`.
3. **Null dose at the pin (m8):** `teams:hr_spec_v1.r3@0` reproduces OFF's played bits. If the team-mix path draws
   from the shared RNG at share 0, this fails. Then the remedy is a fresh C0h at the side pin, 8 × 4.10 ≈ 32.8 GPU-h
   (a training-side miss, X7), unless the loader is given its own generator first. `<<TBD>>`.
4. **The corpus at the side pin (F-M3 fix 1):**
   - the pin's soak (`gen3_team_corpus_soak_v2`) PASS;
   - `kit corpus-check` PASS;
   - main's pack oracle against the pin's packer, team for team: teams that differ are dropped, and their share is
     reported;
   - the post-fill near-neighbour check at ≥ 3 shared sets against every panel (m4).

   `<<TBD: each result; the at-pin team count N>>`.
5. **F-M7 guard landed:** `--team-mix` refuses a `"human_data": true` corpus without the matching `--human-data`
   clause, with a test that fails on revert. `<<TBD: commit>>`. Both pins' soak results are in the corpus manifest.
6. **The C0 reuse check** (§6.3): `data/` content hashes, the Kakuna harness commit (the play pin), R's sha, torch /
   CUDA / driver. A training-side miss ⇒ C0h; an eval-side miss ⇒ re-play (X7).
7. **The play-pin checks of §4** (1–4).

## 15. VOID (a VOID seed is never read; it is re-run at the same pin and seed after the cause is fixed)

- The `[HUMAN-DATA] lever=teams rowset=hr_spec_v1.r3 rows_sha=…` stamp is missing, or not the registered one, in
  any launch of the seed.
- The realised human share is outside ±10 % of 0.5.
- A panel-contamination hit (`main.heldout_panels.contamination` on the completed teams).
- K9(b) not PASS at arm end.
- A FATAL naming a corpus team.
- The recorded pin is not the side pin, or a resume fails §13.

The T54 pool-seed check applies as for X35. **INCONCLUSIVE:** an arm below the floor, or any short cell.

## 16. Predictions (GUESS, before any X60 game)

| quantity | point | 80 % interval |
|---|---|---|
| **Δ_F (primary)** | **+1.5 pp** | [−1.5, +4.5] |
| I_F | +0.8 | [−0.5, +2.5] |
| **Δ_P** | **−1.5** | [−3.0, 0.0] |
| R72 (in-house) | +0.8 | [−0.3, +2.5] |
| T − D1 (H + F) | −0.5 | [−3.5, +2.5] |
| Δ_H (reported) | +2.5 | [−1.0, +6.0] |

The basis:
- D1's F +4.0 came on its own (forum) population, at a P cost of −1.6.
- H-TEAMS's corpus is the ladder population, so F is unfamiliar in population as well as team, and less transfer to
  F than D1's is expected.
- Its own population is H's.

Final classes: GAIN 0.25 · NO-USEFUL-GAIN 0.35 (look 1 0.25 of it) · NOT-RESOLVED 0.37 · LOSS 0.03. Given a GAIN:
ADOPT-CANDIDATE 0.3, TRADE 0.6, KAKUNA-SPECIFIC 0.1.

## 17. Cost (MEASURED rates, F-M8)

**GPU-h** (the 20 X35 production fork dose records, 3.84–4.43, mean **4.10** a fork):

| step | GPU-h |
|---|---:|
| smoke | 0.2 |
| look 1: ON s3501–s3503 | 12.3 |
| on CONTINUE: ON s3504–s3508 (5 forks) | 20.5 |
| on CONTINUE: C0 s3506–s3508 (3 forks, `f4b25238`) | 12.3 |
| **total** | **45.3** (12.5 if look 1 stops) |
| contingency: a C0h if the null dose fails (§14.3) | +32.8 |

**CPU, at the rates MEASURED on this box** (`fm8.rates`):
- **Kakuna:** X61P look 1 / look 2 held units at a mean 29.2 / 20.9 s a game a lane (median 23.1 / 18.9, p90 49.8 /
  25.3), lane occupancy including any heavy-slot wait. D3's looks held 30.1 / 14.8. **The cost uses X61P's 25.0 s.**
  The design's 10 s was 2–3× low.
- **Throughput:** X61P realised 234–344 Kakuna games per wall hour at concurrency 2.
- **In-house:** 2.31–2.58 lane-h per 7 × 500 set (X61P), 2.20–3.01 (D3). **The cost uses 2.45.**

**The reads, ranked so a trimmed set still decides** (8 seeds; `fm8.plan_8_seeds`):

| tier | what | Kakuna games | in-house sets | CPU lane-h | Kakuna wall (h, 2 lanes) |
|---|---|---:|---:|---:|---:|
| **T1** | the primary F + goal (a) + Δ_P (incl. C0 s3504–s3505 re-played, the new C0 seeds, the determinism guard) | 16,740 | 13 | 116.5 + 31.8 | 49–72 |
| **T2** | I_F + R72 (both arms) | 0 | 32 | 78.2 | (39 on 2 S lanes) |
| T3 | REPORTED H / P72 / home72 at 7 units | 3,220 | 0 | 22.4 | 9–14 |
| T4 (last; decides nothing for adoption) | T − D1: ON H 0–29 (5 finals) + D1 H + F 0–29 at E3 | 9,000 | 0 | 62.6 | 26–39 |
| **total** | | **28,960** | **45** | **201.5 Kakuna (242 at the slowest state's rate) + 110.0 in-house** | **84–124** |

- **Look 1 needs only ON s3501–s3503's F:** 3,600 games, ≈ 10.5–15 h on 2 lanes. C0 imports.
- **At 5 seeds** (had UB ≤ 1.5): 19,660 games, 27 sets, 137 + 66 lane-h.
- **Wall:** the Kakuna and S lanes run in parallel (as X61P's), so the read after the last final takes ≈ 4–5 days if
  it has the 2 heavy slots to itself. Queue position at registration: `<<TBD: heavy-slot queue and the GPU queue at
  registration>>`.

## 18. Dispositions of REVIEW F (route T)

| finding | here |
|---|---|
| F-M1 | fix 1 = §2 (freeze sha256 + departures). Fixes 2 and 3 (a coded `x60_route`, a sha-pinned `verdict()` spec) are NOT built; the route was read from the X59-S record (FINDING 1). Fix 4: X59-S has a verdict; the follow-up fired and bears on X60-B only |
| F-M2 | §4 (one harness; D1 re-played; C0's two-state gap). Fix 2 is landed in main (`20875e962`: the engine enters the regime id). Fix 3 = X7. Fix 4: I_F, Δ_P and R72 are S families at the play pin, stamped `env=393631b8…/126` |
| F-M3 | §14.4, X8 |
| F-M4 | §5.1, §5.2, X1, X2 |
| F-M5 | route-T parts: §9 (dose, MDE, fix 3), §11 (fix 2). Fix 1 is X60-B only: N/A |
| F-M6 | §7, §7.1, X3 |
| F-M7 | §14.5 (sibling build) |
| F-M8 | §17 |
| F-M9 | X60-B's teacher: N/A for route T |
| m1 | §10 · **m2** §5.2's column · **m3** §10 · **m4** §14.4 · **m5** §9 · **m6** §2 and §6 (C0's F, P and σ̂ are known before registration; the floor, units and classes are §6's unchanged; only the seed count moved, by fix 3's control-only rule) · **m7** §11 · **m8** §14.3, §12 · **m9** X60-B's coefficient smoke: N/A · **m10** X60-B's r2 dirty paths: N/A · **m11** F alone stands on X61P's precedent (its review's B1), not on "ruling `4f555e57`" |

## 19. FINDINGS at this draft (standing rule 7)

1. **The route was applied by reading, not by code** (F-M1 fixes 2–3 unbuilt). X59-S's classes are
   SELF-PLAY-DATA-BETTER, OWN-LABELS-BETTER and Δ_restriction RESTRICTION-COSTS. §6.1 row 3 names
   OWN-LABELS-BETTER, so the outcome is unambiguous, but no `x60_route()` was run.
2. **σ̂'s UB fires the 8-seed rule.** That is +32.8 GPU-h on CONTINUE over the 5-seed plan: 20.5 for ON s3504–s3508
   and 12.3 for three new C0 seeds. The kit's C0 stage must accept seeds 3506–3508 (`<<TBD: kit stage table>>`).
3. **C0's X61P cells span two read states, and `main.reads` takes one import per group with no transitive import.**
   Look 1 imports cleanly. Look 2 needs a two-state import in the play pin, or a re-play of C0 s3504–s3505
   (3,240 games + 2 sets).
4. **No arm has I_F, so the meters' exchange rate on F is unmeasured.** The registered planning rate is structural.
   The only same-panel datum (N1-P home72) is uninformative. I_F's per-final SD on forum72 is unmeasured too; the
   power uses home72's.
5. **The end-to-end simulation treats the meters as independent given the truth.** A seed effect shared across F, I_F
   and P would correlate them, and G-F / G-E / G-C were not modelled.
6. **At D1's measured in-pool cost (Δ_P −1.6 pp), Δ_P NON-INFERIOR passes 0.12–0.23, so P(ADOPT-CANDIDATE) for a true
   +3 pp lever is 0.08–0.16 and TRADE is the likely positive outcome.** Δ_P's rule is D1's and is not changed here.
   The coordinator may want the owner told before launch.
7. **The always-current docs still quote the old power and the point guard:** design §0 / §6.0 / §6.7 and Q-H11 in
   `human_data_architecture.md`. F-M4 fix 1 and F-M6 fix 3 ask for those edits. This draft did not edit them (brief);
   the coordinator lands them with the registration.
8. **The play pin `de15dfc7b` predates later harness guards.** G14's forced-end check reads `unavailable` on home72
   (no capture there), as in both X61P looks. The X35-frame shared play pin `86cfdf7ef` (E1) is a different regime,
   and choosing it would forfeit every C0 import. `de15dfc7b` holds no R72 panel and no `x60` rule, hence the
   one-commit play pin.
9. **D1's doc quotes ≈ 118 draws a corpus team a side at 1,981 teams; its own "570k games" rule gives 144.** The
   dose here uses the rule (N1-P's arithmetic). The 118 is unexplained.
10. **Not verified here:**
    - that an ON checkpoint from the side pin loads at the play pin (§4.4);
    - that adding R72 to the registry leaves forum72 / home72's ids unchanged (§4.3);
    - the corpus at-pin checks, the smoke, off-means-off and the null dose (all `<<TBD>>`).
11. **The Kakuna rate includes heavy-slot waits** (units are timed from start to end through `mem_cap.sh`), so
    25 s a game is lane occupancy on this box under its usual contention, not pure compute.
12. **Process.**
    - CPU only, at nice 19 on cores 5-7,13-15, `CUDA_VISIBLE_DEVICES=""`.
    - Read-only on `models/` (dose sidecars via `main.dose`) and `~/gen3ai_archive/`. No GPU, game or launch.
    - The design docs were not edited.

## `<<TBD>>` placeholders, each with its filling rule

| placeholder | filled by |
|---|---|
| N, the r3 rows-manifest sha256, rowset id confirmation | the corpus build (sibling): the next free id is r3 iff no other rowset registered first |
| the clause hash; the `[HUMAN-DATA]` stamp line | the switch build (sibling), computed from the canonical clause |
| side pin sha / tag; backported paths | `kit pin-check` output at the pin |
| at-pin soak, corpus-check, pack-identity drop share, post-fill ≥ 3 check, 10-26 touch list | §14.4, run at the side pin |
| F-M7 guard commit | the switch build |
| R72 seed, digest, team ids, registration commit | §9's order, after r3 exists |
| play pin sha; playpath digest; the two-state import (built or not) | §4 |
| the read specs and their sha256s; the kit arm name | written after the pins, `check-spec` exit 0 |
| smoke, off-means-off, null-dose results; the ON load check | §14.1–3, §4.4 |
| the kit's C0 stage table for s3506–s3508 | the kit build |
| queue position | at registration |
