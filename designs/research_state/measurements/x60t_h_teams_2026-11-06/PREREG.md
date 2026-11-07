# X60-T: human-built ladder teams (H-TEAMS) at the production plateau, against C0 and D1: PRE-REGISTRATION (2026-11-06)

**Status: REGISTERED 2026-11-06, before any launch.** No `rb_x60t_*` run exists and no X60 game has been played.
Every decision rule, seed, class, threshold and cell is fixed HERE. A few build facts are still open, and each has a named
amendment and a mechanical filling rule (§20). None is a choice that X60 data could inform. After the first ON arm
launches, nothing changes except by a dated amendment, written before the read, that says why.

**Amendments:** AM-X60T-1 (§21, 2026-11-06, before any read game): the play pin `f564c333f`, §4's checks, look 1's spec,
and G-C read over 8 pairs. AM-X60T-2 (§23, 2026-11-07, after look 1's CONTINUE, before any X60Tb / X60T-C0b launch or
look-2 game): the stage record, R72 at main, look 2's spec and guards stub, the `ge_x60t` reader, the validated launches.

**Two adversarial reviews.**
- **REVIEW F** reviewed design §6 ([`REVIEW_F_x60.md`](../human_replay_quality_2026-10-25/REVIEW_F_x60.md));
  dispositions in §18.
- **REVIEW I** reviewed this PREREG's draft (`2589a7fa7`) on two lenses, data / selection bias and experimental /
  causal validity ([`REVIEW_I.md`](REVIEW_I.md), READY-WITH-FIXES); dispositions in §18b.

**Written AFTER X59-S's outcome (REVIEW F F-M1, recorded honestly).**
- X59-S read on 2026-10-30: SELF-PLAY-DATA-BETTER; Δ_labels OWN-LABELS-BETTER; the closure BLOCKED by
  Δ_restriction (`5bbba6096`).
- Its ≥ 1500 follow-up read on 2026-11-03: rating NOT-RESOLVED.
- X61P's C0 cells, which this read imports, were read on 2026-10-30: C0 F 3,023 / 6,000 = 0.5038.
- **The route** was fixed by design §6.1's table, row 3, which names OWN-LABELS-BETTER explicitly ("anything else …
  ⇒ X60-T"). It was applied by reading the X59-S record, not by a coded `x60_route()` (FINDING 1).
- §2 freeze-diffs this file against design §6 and lists every departure. Each departure is a fact filled by a stated
  rule, or a review fix named by its finding id.

**The analysis behind every number:** [`analysis.py`](analysis.py) → [`results/analysis.json`](results/analysis.json)
(CPU, seeded `20261106`, 40,000 draws a cell). It IMPORTS the design's simulator
[`x60_redesign_2026-10-26/power.py`](../x60_redesign_2026-10-26/power.py) (`finals`, `rule_x60`; sha256
`7c1330a5…8f25`) and never re-implements the registered rule. REVIEW I re-ran §5's power independently: GAIN at +3 pp
0.888 at σ 1.0. It found 0 class mismatches between `power.py` and the shipped rule `x60` over 3,600 cells.

**Abbreviations.**
- P\* = `rb_x26_s2601` @ 280,000,170.
- C0 = X35's plain continuation (`rb_x35_c0_s<seed>`, pin `f4b25238`).
- D1 = `rb_x35_d1wide_s3501`–`s3505` (forum teams `w_forum_v1` at share 0.5, pin `440b61c32`).
- ON = the H-TEAMS arm, `rb_x60t_s<seed>`.
- F / H / P72 = `forum72` / `heldout72_hpfix` / `p72`.
- S = X35's population score against the frozen reference population R (`reference_population_production.json`,
  content `cef0960e…`), 7 members × 500 mirrored pairs a final.
- I_F = ON − OFF in S on F.
- R72 = the positive control: 72 of the corpus's own teams.
- σ = the seed-to-seed SD of a final's F rate.
- pp = percentage points; LB / UB = lower / upper bound; lane-h = one CPU lane for an hour.

## 1. Question and mechanism

Does mixing **human-built ladder teams** (`hr_spec_v1.r3`, 2,295 teams from 812 accounts) into the production
plateau's training, at D1's dose, raise how well it pilots **human-built forum teams it never saw** (F) against
Kakuna? And is the gain the LADDER part, or would forum-team breadth (D1) do as well?

D1's forum teams read F +4.0 pp at 140 games a final, but F is D1's own population. H-TEAMS trains on a different
human population (ladder), so F is unfamiliar to it in team AND population. That is the claim F carries.

**What the corpus is: a six-seen, loser-enriched, long-game slice of ≥ 1500 ladder teams** (REVIEW I M4). The
"all six mons seen" rule mostly keeps the side whose team was fully revealed, which in practice is the losing side
in the longer games:

| seats | won their battle | median turns |
|---|---:|---:|
| all ≥ 1500 candidate seats of r2's kept battles (9,193) | 54.3 % | — |
| six seen, kept (4,925) | 41.1 % | 31 |
| not six seen, dropped (4,268) | 69.6 % | 18 |
| **the final corpus (2,295)** | **39.0 %** | 31 |

This does not bias Δ_F as an A/B of THIS corpus. It bounds what a result can be attributed to: "ladder teams" means
this slice (§10, §11).

**What it does not test:** a fresh run or a warm start (the plateau-plasticity confound, as N1-P); any other share
or corpus. A NO-USEFUL-GAIN rules out "this slice of H-TEAMS at the plateau, at share 0.5, for 20M steps" only.

## 2. The freeze (F-M1 fix 1) and every departure from design §6

- **Frozen source:** `designs/endstate/design_human_replay_lever.md` at `e1512c361`.
  - File sha256: `ad34aee94f629b873d538722e57abe352f6c43bfdc1f0e30cd880d07f942174f`.
  - §6 text ("## 6. X60" up to "## 7. Appendix") sha256: `f288b331245c9652aba69542c11dae7bd152029f4c85ef4110b08554ff2330fe`.
  - Main's §6 has since moved (sha256 `8833dc8f…ca4c4` at `f7f42003f`). The switch (`fbd1afa38`) and builder
    (`636bdcc1e`) commits recorded build facts there, listed as X14. This PREREG registers against the FROZEN §6
    plus the departures below.
- **Everything not listed below is §6.3 / §6.4 / §6.6 / §6.7 verbatim:** the arms, the side pin's principle, the
  60-unit F cell, the interval (binomial + `2 · max(1.0 pp, σ̂)² / n` + the Welch guard), the futility-only look 1,
  the classes and their 1e-9 tie rule, Δ_P's rule, goal (a), the X35 guards, the VOID list, R72's construction and
  the T − D1 arena.

| # | departure | answers | why |
|---|---|---|---|
| X1 | **8 seeds an arm** (ON 3501–3508; C0 + 3506–3508), not 5. Look 1 stays 3 v 3; look 2 is 8 v 8 | F-M4 fix 3 | σ̂'s one-sided 95 % UB from X61P's 10 finals is **2.04 pp > 1.5** (§5.1), decided on control-only data before ON trains |
| X2 | power quoted as a range over σ ∈ [σ̂ point, σ̂ UB], headline at the 1.0 pp floor | F-M4 fix 1 | the old 0.80 was the σ = 0 cell |
| X3 | **KAKUNA-SPECIFIC iff I_F's one-sided 90 % LB ≤ 0** (was: the point ≤ 0) | F-M6 fix 3 | leak 0.10 against 0.50; at any exchange rate ≥ 0.5, its cost to a true +3 pp gain is ≤ 0.02 (§7) |
| X4 | NO-TRANSFER also needs I_F not GAIN; new class **IN-HOUSE-ONLY** | F-M5 fix 2 | R72 and I_F are both in-house; an in-house F gain with a flat Δ_F is not "no transfer" |
| X5 | a look-1 futility stop reports **NO-USEFUL-GAIN undivided** (no NO-TRANSFER / NOT-DELIVERED split) | F-M5 fix 3 | R72's 3 v 3 MDE (1.1–2.6 pp) is not shown to be reachable (§9) |
| X6 | **ONE eval harness for every X60 game:** `de15dfc7b`'s play path (engine E3). C0's cells import from X61P's two states as two groups; D1's units 0–29 are re-played at E3 | F-M2 fix 1; REVIEW I m6 | the regime of C0's banked cells; D1's banked units are E1 and `train` mode, so they cannot import (§4) |
| X7 | the C0 reuse check's remedy is split: a training-side miss ⇒ a fresh C0h; an eval-side miss ⇒ a CPU re-play | F-M2 fix 3 | an eval mismatch needs no GPU |
| X8 | the corpus is checked **at the side pin**: soak, pack identity (58 teams dropped), engine-playability (7 dropped). The side pin carries the AM-D1-FIX label guard and the corpus's sha256 STUB; **the corpus itself never enters git** | F-M3 fixes 1–4; REVIEW I M1 | at `f4b25238` the packer mis-packs Frustration and Hidden Power IV lines; human data stays local |
| X9 | T − D1 is always read **5 v 5** (ON s3501–s3505 v D1 s3501–s3505), only after look 2 | m1 | §6.4's "3 v 3 if look 1 ended it" let the primary pick the contrast's seeds |
| X10 | LADDER-BETTER / FORUM-BETTER reworded as home-advantage sums plus the outcome-selection confound | REVIEW F m3; REVIEW I M4 | each arm owns one panel; the corpus is loser-enriched, D1's forum teams are curated |
| X11 | **pause / resume:** a seed resumed under §13's conditions is read as that seed's final. §6.3's "re-launched with its seed" applies only when they fail | the brief | §6.3 had no resume rule |
| X12 | build facts: r3's digests and N, the side pin, the clause hash, R72's digest, the play pin, the specs | F-M1 fix 1's list | filled at registration (§3, §14), or by the named amendments (§20) |
| X13 | the F floor: each arm counts ≥ 1,200 · n − 20 F games, every final > 0, else INCONCLUSIVE | none (X61P §4's floor, carried) | §6.3 names no floor; a short cell must not decide |
| X14 | the switch keeps provenance in `metadata.json` only (not `model_config.json`), has NO consumer snapshot check, and refuses only an UNTYPED resume (a typed `off` or another clause is warned) | REVIEW I m1 | departures from design §6.9 item 1, recorded on main at `fbd1afa38`; §13 condition 4 is the real resume guard |
| X15 | the realised-share VOID and smoke check become the stamp's share and sides plus a measured draw count at the pin | REVIEW I B1 (option b) | nothing counts corpus draws in a run; over ≈ 10⁶ draws a Bernoulli(0.5) cannot miss ±10 %, so the check's only job is a mis-wired share, which the stamp shows |
| X16 | the read code fixes every seed set, the Welch guard's tie band, and requires the guards and the VOID gate at look 2, with a pin per arm | REVIEW I M3, m3, m4 | the text fixed them; the code did not |

## 3. The arm

| item | value |
|---|---|
| start | P\* = `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…91ed`, R's frozen call), as C0 |
| the ONE lever | `--human-data teams:hr_spec_v1.r3@0.5`, team sides both: with probability 0.5 per episode and side the team is a corpus team, else exactly C0's draw. It builds the same `TeamMix` → `Gen3Teambuilder._install_mix` path `--team-mix` does, so ON − D1 differs in the corpus alone (REVIEW I, checked in code). **Clause hash `544c9bc00ea95030`** |
| everything else | C0's argv token for token: `--behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --seed <s> --run-name rb_x60t_s<s>`, the recipe inherited from X26's record. Kit arm **`x60t`** (`batch.X60T_PIN`, `X60T_PIN_PATHS`, `x60t_pin_check`; kit commit `7e94d1de8` on main, "AM-X60T"; `7f5c055e8` adds `rb_x60t_s*` to `canary_decide`'s C0 family). `checkargs --pin` rc 0 on the ON and C0 argvs |
| **side pin** | **`7fe15726c77cfa823aee78110444bf25c3c7370e`** (annotated tag `x60t-pin-7fe15726`, on origin) = D1_PIN `440b61c32` (= `f4b25238d` + `src/utils/team_corpora/` only: the AM-D1-FIX label guard) + ONE commit of 19 paths. Those paths are the switch backport, including `utils.paths`' archive-dir helpers, and the corpus's sha256 stub `src/utils/team_corpora/human_corpora.json`. **The corpus never enters git:** `kit corpus-check` reads the stub from git and the LOCAL corpus by sha256 (`kit.human_corpus_check`). `kit pin-check` PASS |
| seeds (stages) | **X60T** = `x60t` 3501–3503 (look 1). **X60Tb** = `x60t` 3504–3508 and **X60T-C0b** = `c0` 3506–3508 at `f4b25238d` with C0's argv; both run only on `x60t`'s look-1 CONTINUE. Fixed: no extension, no adaptive rule. The 8-seed count is decided here, on X61P's control-only data (§5.1) |
| control | C0 s3501–s3505 (each K9(b) PASS), already trained: 0 GPU-h. C0 s3506–s3508 on CONTINUE (≈ 12.3 GPU-h). Every C0 final is pinned by sha256 in the read specs |
| contrast arm | D1 s3501–s3505 (pin `440b61c32`), already trained: 0 GPU-h |

### 3.1 The corpus `hr_spec_v1.r3` (local only: `~/gen3ai_archive/human_teams/hr_spec_v1.r3/`)

- **Built by:** `main.human_data.hteams` (landed on main at `636bdcc1e`).
- **Provenance (REVIEW I m9):**
  - The manifest's `commit` field is `0b9e6f354`, which does NOT contain `hteams.py`. Its `source_sha256` ties it
    to the builder file at `eb7e67407`. Main's landed builder differs from that file only in the `human_data_ok`
    arguments.
  - The stub's `"snapshot"` is null. The chain `hr_spec_v1` `e7d491e7…` → r1 `e9e0df42…` → r2 `2a413c85…` → r3 is
    recorded in the manifest.
- **Digests:**
  - manifest `71783ac88917a96a7ec0d61e3e29f17d1cdc2b3633edfd5cd9b00b55701d6bfb`;
  - data file `2e86457282f719eafaa75c6477c08420a03fafeb0e21f9ee5c24e1d2703afe88`;
  - tier `31ca149cc4c0…`;
  - soak file `30b5cec4d843…`.
- **The funnel:**

  | stage | teams |
  |---|---:|
  | r2's kept battles, actor ≥ 1500, all six seen | 4,925 |
  | C-GAME-completed | 4,920 |
  | validator / level / Hidden Power repair | 4,920 |
  | both panel nets: §3.1's revealed rule 0 dropped, exact-set 0, the post-fill ≥ 3 net (m4) 92 | 4,828 |
  | pack identity at the side pin: 58 happiness-0 Frustration teams dropped, 0 remain (verified) | 4,770 |
  | engine-playable at the pin: 7 dropped (Sky Attack, Snore, Macho Brace) | 4,763 |
  | duplicates (same six completed sets): 0 dropped | 4,763 |
  | ≤ 5 teams per account | **2,295** |

  The post-fill net's 92 drops split forum72 9, H 4 (+1 on the retired heldout72), home72 57, away 15.
- **Soaks:** `gen3_team_corpus_soak_v2` PASS at the side pin's env and at main, 22,950 battles each, 0 core errors.
- **10-26 parity exposure (F-M3 fix 2):** 709 teams (30.9 %) touch a 10-26 parity class, below the pool's own
  33–39 % (REVIEW I). ON meets no more mis-simulated mechanics at E1 than C0 does.
- **Effective N (REVIEW I m8):** 2,295 teams, but 791 repeat their own account's species-six, leaving ≈ 1,504
  distinct (account, species-six). The duplicate rule dropped 0 because C-GAME's fill is seeded per battle, seat
  and attempt, so a repeated team never matches exactly.
- **Not `n1_hrep_v1`** (REVIEW I m10): no `n1_hrep_v1` exists. `--team-mix` refuses a `"human_data": true` corpus
  without the matching clause (the F-M7 guard, `fbd1afa38`). F-M7 fix 2, reconciling X61 §10 with §6.4, stays open
  and bears on N1H only.
- **Smogon-only priors:** no prior, usage or set-inference consumer reads team corpora (REVIEW I, checked).

**Dose (ESTIMATE, D1 doc §4.1's rule: ≈ 570k games in 20M).** At share 0.5 a side draws ≈ 285k corpus teams, so:
- **≈ 124 draws a side per team** (285,000 / 2,295);
- **≈ 190 per distinct (account, species-six)** (285,000 / 1,504).

That is 16–24 % of a pool team's ≈ 790 draws a side in C0, and 1.3–2.0 % of P\*'s ≈ 9,300 cumulative pilots per pool
team. The pool's exact exposure halves (as D1's and N1-P's), which Δ_P reads. D1's own doc quotes ≈ 118 draws a
side at 1,981 teams, where this rule gives 144 (FINDING 9).

## 4. One eval harness (F-M2) and the imports

**Registered:** every X60 Kakuna unit and every S cell is played on the play path of **`de15dfc7b6d19d6f0e4a3724f269e26e7aa22957`**
(X61P look 2's commit, tag `x61p-n1p-look1-sched-de15dfc7`). Its regime, checked by `analysis.py` `fm2`:
- engine **E3** (`port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126`), equal to X61P's cells;
- Kakuna's team pack: legacy;
- peer code / policy mode: legacy `b817d3216c418f75` / `eval`;
- OUR side's pack: post-fix.

Both arms trained on E1. Both are read on E3, so the train / eval engine gap is common to ON and C0. **`plan` and
play run at the play pin; `assemble`, `decide` and `record` run from a main checkout** (AM-N1P-2's pattern), so
main's import checks (engine, sides, Kakuna pack) and the re-derived guards apply (REVIEW I m7).

**The play pin = `de15dfc7b` + ONE commit,** built and registered by AMENDMENT **AM-X60T-1** before any read game
(X61P AM-N1P-1/2 pattern). That commit holds:
- the `x60` rule's port (the rule shipped on main at `f7f42003f`; the play commit's spec parser refuses a rule it
  does not hold);
- R72's registration in `heldout_panels.json` (§9);
- the VOID-gate support the rule's `void.pin` / `void.dose` need, since `de15dfc7b`'s `rules._void_check` accepts
  only `{k9b_dir, runs}` and fails closed on them (REVIEW I m7).

The training launch does not wait on it. **The rule, the seeds, the classes and every threshold are fixed HERE;** the
amendment may add only the commit, its checks and the spec sha256s.

**Required at the play pin, before any X60 game** (each a HOLD on failure, never a launch):
1. `playpath.compare` against `de15dfc7b` shows no game-path difference beyond the panel registry.
2. `engine_at(<play pin>) == E3`.
3. forum72's id stays `2118812c27885de3` (group `00c8c0114bdf0836`), the reported group's stays `3d2cbdc0b1413bcb`,
   and S home72's stays `d4de0fd7c16a2658`. Else C0 cannot import: an eval-side miss ⇒ re-play (X7).
4. An ON checkpoint plays at `de15dfc7b`'s `main.h2h` path. Checked on the smoke checkpoint: 2 games, rc 0 (§14). It
   is re-checked on s3501's first 1M checkpoint before look 1 plays.

**C0's imports (REVIEW I m6 supersedes the draft's two-state gap).** A group takes ONE `import`, and an import reads
only the earlier state's own played units. C0's cells sit in two states, so they come in as **two groups**, each with
its one import. Rule `x60`'s `pool_anchor` / `pool_s` pool groups only when they are ONE cell (one engine, one regime
id per team set, one peer code and policy mode, one Kakuna pack; each label in one group); else the meter is
INCONCLUSIVE.

| group | earlier state | spec sha256 | labels | checked |
|---|---|---|---|---|
| C0 look-1 group (F, reported, S `p`) | `~/gen3ai_archive/x61p_n1p_read/look1` | `1266daad…9f03` | c0_s3501–s3503 | `imports.check` returns 180 rows (607 / 625 / 598 of 1,200), E3, `2118812c27885de3` (REVIEW I) |
| C0 look-2 group (F, reported, S `p`) | `~/gen3ai_archive/x61p_n1p_read/look2` | `f28654cc…2620` | c0_s3504–s3505 | F: 120 rows (610 / 583 of 1,200), E3, same regime (REVIEW I m6). S `p`: 210 + 140 ledger rows, regime `d4de0fd7c16a2658`, pooled S equal to look 2's exactly (the read-rule build, `4d7367228`) |

- **Same bytes played:** the C0 sha256s are s3501 `09a6a2df…`, s3502 `7e9a9890…`, s3503 `99096be8…`, s3504
  `26835480…`, s3505 `bb10cd96…`.
- **The determinism guard** (§6.3): one imported C0 cell (c0_s3501 forum72 unit 0) is re-played game for game,
  choices included, at the play pin. A difference is INCONCLUSIVE and the imported cells are re-played (X7).
- **D1 cannot import.** D1's banked H / F units were played on E1, with peer code `f230353c40e43488` and policy mode
  `train`, and an import across the engine or the sides refuses. **D1's units 0–29 on H and F are re-played at the
  play pin: 5 × 2 × 600 = 6,000 games.**

## 5. PRIMARY Δ_F (`main.reads`, rule `x60`, design §6.3 verbatim except X1, X16)

- **Cell:** Kakuna ckpt 34, greedy v greedy, `forum72`'s committed symmetric config, **60 role-balanced 20-game units
  per final (unit seeds 0–59, 1,200 games)**, both arms.
- **Seeds, fixed in code (REVIEW I M3).** The rule refuses any other set, with a refusal test for each:

  | set | seeds |
  |---|---|
  | `LOOK1` | ON and C0 3501, 3502, 3503 |
  | `LOOK2` | ON and C0 3501–3508 |
  | `CONTRAST` | ON and D1 3501–3505 |

  `look1_seeds` is removed from the spec. Each run dir must end in `_s<seed>`, and a recorded seed that differs
  makes the read INCONCLUSIVE. Fix commit **`4d7367228`** (main), before look 1 is read.
- **Interval:** Δ̂ = ON's pooled F rate − OFF's. V = Σ_arm p(1 − p) / (1,200 n) + 2 · max(1.0 pp, σ̂)² / n, with
  σ̂² = W − b (W the mean of the arms' sample variances of their per-final F rates, df 2 (n − 1); b the mean per-final
  binomial variance).
- **The Welch guard:** if σ̂²'s one-sided 95 % LOWER bound W · df / χ²(df, 0.95) − b exceeds (1.0 pp)², a Welch t
  interval on the per-final rates decides instead. **Tie band (REVIEW I m3, rule 8):** a lower bound within 1e-12 of
  (1.0 pp)² takes the WIDER of the two intervals. 95 % two-sided throughout.
- **Looks:**
  - **Look 1 (3 v 3) is futility only:** Δ̂ ≤ 0 (within 1e-9 counts) ENDS the read with look 1's class, never GAIN.
  - **Otherwise look 2 (8 v 8) decides.** Look 2 recomputes look 1 on `LOOK1`, so a look-2 decision IS `rule_x60`
    on the same counts.
- **Classes, as the code computes them** (REVIEW I m5; TIE = 1e-9):
  - **GAIN** iff LB > TIE;
  - **LOSS** iff UB < −TIE;
  - **NO-USEFUL-GAIN** iff UB < +3.0 pp − TIE. This includes a UB in [−TIE, 0), where the 1e-9 tie rule declines
    LOSS.
  - else **NOT-RESOLVED**.
  - "Equivalent" (inside ±3 pp by TIE) is reported, deciding nothing.
- **Floor (X13):** each arm counts ≥ 1,200 · n − 20 F games, with every final > 0; else INCONCLUSIVE.

### 5.1 σ̂ from X61P's 10 finals (F-M4 fix 2), and the seed count (fix 3)

The estimator is §6.3's, on X61P look 2's `cells.json` (forum72, E3, 60 units a final, ties as non-wins):

| finals | per-final F rates (%) | W SD | binomial SD | φ̂ | df | σ̂ point | **σ̂ one-sided 95 % UB** |
|---|---|---:|---:|---:|---:|---:|---:|
| **C0 + N1-P (registered)** | C0 50.58 52.08 49.83 50.83 48.58 · N1-P 51.58 51.83 51.33 51.83 48.08 | 1.46 pp | 1.44 pp | 1.02 | 8 | **0.21 pp** | **2.04 pp** |
| C0 alone | as above | 1.29 | 1.44 | 0.80 | 4 | 0 | 2.70 |
| N1-P alone | as above | 1.61 | 1.44 | 1.24 | 4 | 0.71 | 3.53 |

**Decision (F-M4 fix 3, control-only data, no X60 data): UB 2.04 pp > 1.5 pp ⇒ 8 seeds an arm.**
- Every subset agrees; REVIEW I adds the 10 finals as one sample, UB 1.81.
- REVIEW F's 8-seed rule (`e41beafe1`, 10-26) predates X61P look 2's read (`62e46874a`, 10-30).
- The new C0 seeds cost ≈ 12.3 GPU-h: 3 × 4.10, the mean of the 20 X35 production forks' dose records (3.84–4.43).

### 5.2 Power of the registered rule at 8 seeds (headline: the 1.0 pp floor)

`power.py`'s `finals` + `rule_x60` with `seeds=8` (look 1 at 3). Base F 0.49 is power.py's; X61P's C0 read 0.504,
so the binomial variance moves < 0.1 %.

| true σ | GAIN at +2 / **+3** / +4 pp | false GAIN at 0 | false LOSS at 0 | NO-USEFUL-GAIN at 0 | NO-USEFUL-GAIN at a true +3 (m2) | half-width, look 2 | Welch guard fires at 0 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.21 pp (σ̂ point) | 0.63 / 0.96 / 1.00 | 0.009 | 0.007 | 0.93 | 0.013 | 1.72 pp | 0.002 |
| **1.0 pp (the floor)** | **0.56 / 0.89 / 0.99** | **0.019** | 0.019 | **0.83** | 0.030 | 1.72 pp | 0.05 |
| 1.5 pp | 0.45 / 0.78 / 0.95 | 0.026 | 0.030 | 0.67 | 0.044 | 1.99 pp | 0.27 |
| 2.04 pp (σ̂ UB) | 0.33 / 0.61 / 0.84 | 0.029 | 0.041 | 0.48 | 0.054 | 2.64 pp | 0.63 |
| 2.75 pp (power.py's old UB) | 0.22 / 0.42 / 0.66 | 0.024 | 0.043 | 0.28 | 0.048 | 3.29 pp | 0.91 |

- **Headline: GAIN at a true +3 pp is 0.89 at the floor, ranging 0.61–0.96 over σ ∈ [0.21, 2.04] pp.** At 5 seeds
  the same cells read 0.70 at the floor and 0.80–0.42 over the range.
- **A look-1 futility stop** ends 50 % of nulls, and 1.8 % of true +3 pp levers at the floor (3.9 % at 1.5 pp).
- **A stop does not always mean NO-USEFUL-GAIN (REVIEW I m2).** Under the null, a stop ends NOT-RESOLVED in
  ≈ 0.065 of nulls at σ 1.0 (0.21 at 2.04) and LOSS in 0.019 (0.040).
- **The overall false-LOSS rate is 0.019–0.041,** above the nominal 0.025 at σ ≥ 1.5 pp. LOSS licenses dropping
  H-TEAMS at the plateau, so that rate is quoted beside any LOSS.

## 6. CO-PRIMARY Δ_P (D1's rule verbatim)

S on `home72` vs R, 7 × 500 pairs a final (regime `d4de0fd7c16a2658`). Δ_P = mean ON S − mean C0 S, Welch two-sided
97.5 %. **NON-INFERIOR iff LB > −2.0 pp.**
- **C0's P nodes:** s3501–s3503 import from X61P look 1's S group `p`; s3504–s3505 from look 2's (§4); s3506–s3508
  are played.
- **Known now** (m6): C0's five S values are 56.82, 57.39, 56.84, 57.37, 56.59 (SD 0.36 pp).

## 7. I_F, the clean meter, and the KAKUNA-SPECIFIC guard (F-M6)

I_F = ON − OFF in S on **forum72** (`main.h2h play-many --panel forum72`, 7 × 500 pairs a final, a new S family at
the play pin), Welch across finals.
- **I_F GAIN** iff its two-sided 95 % LB > 0.
- **The guard: KAKUNA-SPECIFIC iff I_F's one-sided 90 % LB ≤ 0.**

**The exchange rate between the meters (F-M6 fix 1) cannot be measured from banked data:**
- **No arm has I_F.** No S group on forum72 exists in any read state or spec in the archive or the measurement tree
  (`fm6.inventory`).
- **What each arm has:**
  - D1, D3 and N1-P: Kakuna F (and H, P72) plus S on home72.
  - X26c: Kakuna vs P\* but S vs X26 @380M (different baselines).
  - The belief-prior arms: Kakuna only.
- **The only SAME-panel pair is home72, N1-P − C0:** Kakuna −1.86 ± 2.66 pp against S −3.28 ± 0.30 pp. The ratio is
  1.77, but its 80 % range is [−2.6, +3.5]: uninformative.
- **Cross-panel, the meters move OPPOSITE ways under breadth levers** (Kakuna F vs S home72):
  - D1 +4.00 vs −1.61;
  - D3 +1.14 vs −0.45;
  - N1-P +0.55 vs −3.28.
  So P is a cost meter, never a proxy for I_F.
- **The rate registered for planning:** the structural one. If the lever shifts the trainee's play on F by δ logits
  against any opponent, each meter moves p(1 − p) δ, so the in-house gain per Kakuna pp is ≈ 0.98–1.00. The error
  rates below are registered across rates 0.25–1.8.

**The guard's two error rates (8 v 8, Welch, `fm6.guard_errors_8v8`):**

| per-final S SD, both arms | LEAK: pass at a true I_F of 0 (point / **LB90**) | COST: fail at Kakuna +3 pp, rate 0.25 (point / **LB90**) | rate 0.5 | rate ≥ 1.0 |
|---:|---|---|---|---|
| 0.36 pp (C0) | 0.50 / **0.10** | 0.00 / **0.003** | 0 / 0 | 0 / 0 |
| 0.56 pp (N1-P) | 0.50 / **0.10** | 0.004 / **0.10** | 0 / 0 | 0 / 0 |
| 0.86 pp (D1) | 0.50 / **0.10** | 0.04 / **0.34** | 0 / **0.017** | 0 / 0 |

**Choice, by the fix's own text:**
- The registered criterion: "LB90's cost to a true +3 pp gain is ≤ 0.20 at the planning rate (1.0) and the largest
  measured per-final S SD (0.86 pp)".
- Its cost there is 0.00, and it stays ≤ 0.02 down to a rate of 0.5. **So LB90 is registered.**
- Its cost exceeds 0.20 only at a true rate ≤ ≈ 0.25 with both arms' SD ≈ 0.86 pp. The I_F record reports its own
  per-arm SDs, so a reader can see when that corner applies.

### 7.1 End-to-end ADOPT-CANDIDATE power (F-M6 fix 2)

The simulation chains each meter (`fm6.end_to_end_adopt`):
- Δ_F through `power.py`'s rule at σ 1.0 pp;
- I_F through LB90;
- Δ_P through Welch 97.5 % > −2.0;
- goal (a) through ON's Kakuna home72 at 1,120 games, Wilson LB > 0.50, true rate 0.560 + the lever's home72 cost.

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

- **Under the null:** false ADOPT is 0.002 with LB90, against 0.009 with the point rule.
- **Under a Kakuna-only +3 pp** (I_F 0), at Δ_P 0 or −1 pp: ADOPT is 0.04–0.09 with LB90, against 0.22–0.44 with
  the point rule.
- **At D1's in-pool cost, the likely positive outcome is TRADE** (FINDING 6).

## 8. Goal (a) guard

ON's Kakuna `home72`, 7 units a final (1,120 games pooled at 8 finals). Pass iff Wilson LB > 0.50. For reference:
P\* reads 0.598, C0 0.560 (392 / 700, X61P), N1-P 0.541.

## 9. Positive control R72 (F-M5, route T)

- **What it is:** 72 of r3's teams, ≤ 1 per account. They are in ON's pool, never in C0's or D1's.
- **Content digest** (`gen3_team_set_digest_v1`, irreversible):
  `10364442c1c3fe5216f46ab445455497dc350a9f9601fc7af9af1eb39a335501`.
- **The draw:** seeded; the seed and the rule are in the LOCAL `hr_spec_v1.r3/r72/manifest.json`.
- **Team ids are NEVER committed** (REVIEW I M5). They are unsalted hashes of public battle pointers and so
  reversible to battles and accounts. Git carries the digest, the seed and the rule only.
- **Registration:** in `heldout_panels.json` as a PRACTISED set (`held_out_from: []`, a local absolute `team_dir`).
  It lands in the play-pin commit by AM-X60T-1, before any read game, and is not portable off this box.
- **The order (m5):** r3 built → R72 drawn from it → registered. A rebuild cannot silently drop R72's teams: R72 is
  digest-pinned, and `human.load` refuses any corpus other than the stub's.
- **The read:** ON − OFF in S on R72 (7 × 500 pairs a final), Welch 95 %. **DELIVERED iff LB > 0.**

**Dose per R72 team:** ≈ 124 draws a side in 20M (≈ 190 per distinct account-six).

**MDE** (Welch two-sided 95 %, 80 % power, `fm5`):

| per-final S SD | 3 v 3 | 5 v 5 | **8 v 8** |
|---:|---:|---:|---:|
| 0.36 pp (C0, X61P) | 1.09 | 0.73 | **0.54** |
| 0.56 pp (N1-P) | 1.70 | 1.13 | **0.84** |
| 0.86 pp (D1) | 2.61 | 1.74 | **1.30** |

**The expected R72 effect is unmeasured.** P\*'s Kakuna home premium (+7.5 pp at ≈ 9,300 pilots a team, H-CTX),
scaled to this dose, gives 0.08–0.15 pp if linear in exposure and 3.8–4.3 pp if logarithmic (GUESS bracket, Kakuna
scale). The in-house scale multiplies that by the unmeasured rate (§7).

**Fix 3 applied:** the effect cannot be shown to reach the 3 v 3 MDE.
- **A look-1 stop reports NO-USEFUL-GAIN undivided:** R72 and I_F are REPORTED, and no split is made.
- **R72 decides only at look 2 (8 v 8).**

## 10. The contrast T − D1 (design §6.4, with REVIEW F m1 / m3 and REVIEW I M3 / M4)

- **Arms:** ON s3501–s3505 against D1 s3501–s3505 (`CONTRAST`, fixed in code), on **H + F at equal weight**. Units
  0–29 of each (600 games per final and panel). D1's units are re-played at E3 (§4); ON's F units 0–29 are nested in
  the primary's.
- **Rule:** §6.3's interval, one look. **Read only after look 2, always 5 v 5.** After a look-1 futility stop it is
  NOT READ, and the record says so. So its false rates are the continued path's: 0.009 FORUM-BETTER and 0.027
  LADDER-BETTER at σ 1 pp (REVIEW F m1), reported beside the read. A short contrast is NOT READ, never a blocked
  record.
- **Classes:**
  - **LADDER-BETTER** iff LB > 0;
  - **FORUM-BETTER** iff UB < 0;
  - **NO-DIFFERENCE-AT-3PP** iff the interval is inside ±3 pp;
  - else NOT-RESOLVED.
  - It is read whatever Δ_F's class; per-panel rows are reported.
- **Attribution (REVIEW F m3 + REVIEW I M4).**
  - LADDER-BETTER means "H-TEAMS's gain on its own population (H) plus its transfer to F exceeds D1's gain on its
    own population (F) plus its transfer to H". It does NOT mean "ladder data beats forum-team breadth".
  - **Both classes also confound SOURCE with OUTCOME SELECTION:** 39 % winners in this corpus, against D1's curated
    forum teams.
- **Power** (power.py, 600 a panel, 5 v 5): LADDER-BETTER at a true +3 pp 0.80 / 0.70 / 0.58 at σ 0 / 1 / 1.5 pp.

## 11. Verdicts and what each licenses

| verdict | rule | licenses |
|---|---|---|
| **ADOPT-CANDIDATE** | Δ_F GAIN (look 2), I_F LB90 > 0, Δ_P NON-INFERIOR, goal (a), G-F / G-E / G-C PASS | the re-read at the production checkpoint, then X54's sealed read. Reported with T − D1's class in §10's words |
| **TRADE** | Δ_F GAIN, I_F LB90 > 0, and Δ_P's inferiority not excluded, or goal (a) fails, or a guard FAILs | the exchange rate Δ̂_F / −Δ̂_P to the owner |
| **KAKUNA-SPECIFIC** | Δ_F GAIN, I_F LB90 ≤ 0 | nothing (a gain the in-house meter does not confirm; leak 0.10) |
| **LOSS** | Δ_F LOSS (either look) | drop H-TEAMS at the plateau; quoted with §5.2's false-LOSS rate |
| **NO-TRANSFER** | Δ_F NO-USEFUL-GAIN at look 2, DELIVERED, I_F not GAIN | stop this lever **for this six-seen, loser-enriched slice** at the plateau: it moved what it trained on, not unfamiliar teams on either meter |
| **IN-HOUSE-ONLY** (F-M5 fix 2) | Δ_F NO-USEFUL-GAIN at look 2, DELIVERED, I_F GAIN | nothing for adoption, and the lever is NOT stopped: the meters disagree on F; reported to the owner |
| **NOT-DELIVERED** | Δ_F NO-USEFUL-GAIN at look 2, not DELIVERED | closes nothing: at this dose the plateau did not move even practised teams. A dose or fresh-run follow-up is a new registration |
| **NO-USEFUL-GAIN (look 1)** | Δ_F NO-USEFUL-GAIN at a look-1 stop | stop this lever **for this slice** at the plateau at this dose. Undivided (X5): R72 / I_F reported |
| **NOT-RESOLVED** | otherwise | nothing licensed; the next lever of §4; no extension |
| **INCONCLUSIVE** (REVIEW I m5) | a meter the verdict needs is missing or short; an arm below the floor; a failed determinism guard; a C0 import group that is not one cell | no class; the missing piece is played or re-played and the read re-decided, with nothing else changed |

- **X35's guards** (G-F, G-E with a fresh `ge_x60t` reader, G-C) are read only on a Δ_F GAIN; otherwise
  `guards_not_read` records why.
- **At look 2 the spec MUST carry the `guards` artefact and the VOID gate** (`LOOK2_REQUIRED`, REVIEW I m4), with a
  pin per arm (`void.pin`): ON at `7fe15726c`, C0 at `f4b25238d`, D1 at `440b61c32`; every final is gated. Without
  them the decision refuses (`4d7367228`).
- **Shared control (m7):** C0 is X61P's and D3's control too, so a lucky-low C0 inflates all three. Their GAINs are
  not independent replications, and wherever two are cited together it is said so.

## 12. REPORTED (never decided)

- **Δ_H, Δ_P72, Δ_home72** (Kakuna, units 0–6 both arms; ON plays H 0–29 for §10).
- **U** and the goal-(b) F + P72 statistic, each with its contamination named. P72 is fill-adjacent to the corpus;
  H is the corpus's own population.
- **σ̂ and each arm's own per-final SD,** for F and each S group (m8: a lever may add seed variance).
- I_F and R72's per-arm SDs; per-team proximity terciles; belief / intent calibration on the Kakuna cells.
- **The corpus's selection** (REVIEW I M4): 39.0 % winners, median 31 turns, against 54.3 % of all ≥ 1500 seats.
- The pin's pack-identity drops (58) and engine drops (7), and the 709 teams touching a 10-26 class.
- Every resume (§13).

## 13. Pause / resume of a seed

**The resume** (REVIEW I M2):

```
python -m main.launcher --pin-commit 7fe15726c77cfa823aee78110444bf25c3c7370e \
  --model models/rb_x60t_s<s>/checkpoints/<latest>.zip --steps 300000170 --device cuda \
  --seed <s> --behaviour-check warn --checkpoint-every-steps 1000000 \
  --human-data teams:hr_spec_v1.r3@0.5
```

**Every pause or resume goes through MAIN's launcher with `--pin-commit` (`launch_x35.sh`'s path), typing all three
flags and the clause; never the pin's trainer directly.** Measured at the pin (`~/gen3ai_archive/x60t_prelaunch/review_i/`):
- **Untyped, through the pin's trainer:** a resume resolved seed 42, `behaviour_check` fatal and
  `checkpoint_every_steps` None, and it OVERWROTE `metadata.json`'s `cli_args`. The pin (`f4b25238`'s trainer)
  predates the 10-24 run-flag restore.
- **Main's launcher then restores whatever the record holds,** so a corrupted record propagates to every later
  restart.
- **Typed:** the dry-run, `checkargs` and the real resumed child all resolved seed 3501, `warn`, 1,000,000, the
  clause and hash `544c9bc0`.
- **The launcher's own crash / interval restarts are safe:** the kit argv types all three flags, and a restart
  re-launches the session's typed child argv.

**`--run-name` is NOT typed:** with it, a `--model` launch becomes a FORK into a fresh dir
(`checkpoint.resolve_launch_run_dir`). Without it, the checkpoint in the run's own dir makes this a SAME-RUN restart:
- `--fork-lr` is not re-applied;
- the self-play pool is not re-seeded;
- SB3 restores the optimizer and its LR.

The corpus, env and team streams key on `segment_seed(seed, num_timesteps)` at `f4b25238`, so a resume does not
replay step 0's draws (REVIEW I).

**The switch's own refusal is narrow (X14):** it refuses an UNTYPED resume only. A typed `off`, or another clause,
passes with a warning. Condition 4 below is the real guard.

**The read takes a resumed seed's final iff ALL of these hold** (else it is VOID and re-launched from P\* with its
seed, §6.3):
1. `final_model.zip` at step ≥ 300,000,170 (the read's `final` ref, quiet 300 s).
2. `pin_history` is ONE span, the side pin (`python -m main.sidecar_audit models/rb_x60t_s<s>` not PIN-SPLIT; the
   rule's `void.pin`).
3. `python -m main.dose models/rb_x60t_s<s>`: `lr_median`, `updates_per_env_step` and `dose_rate` each equal C0's to
   a relative 1e-6 (the rule's `void.dose`). All 20 X35 production forks read `dose_rate` 2.0440550334496103e-08
   exactly.
4. §15's two lines are in EVERY launch log of the seed, the resumed one included: the `[HUMAN-DATA]` stamp (clause
   hash `544c9bc00ea95030`, rowset `hr_spec_v1.r3`) and the `[TEAM-MIX]` banner (share 0.50 on both sides).
5. K9(b) PASS at arm end.
6. **One note in the record** per resume: the time, the checkpoint step resumed from, the steps re-trained, the
   cause.
7. **The resumed launch's resolved seed, behaviour check and checkpoint cadence equal the run's recorded ones**
   (seed `<s>`, `warn`, 1,000,000; REVIEW I M2). They are read from the `--dry-run` output for the exact resume
   argv, kept in the record. A resume that ran the pin's trainer directly is VOID: it can rewrite `cli_args`.

**What it costs:**
- **Lost steps:** everything since the last resumable checkpoint, ≤ 1,000,000 steps (≈ 0.2 GPU-h at the measured
  1,371–1,482 steps/s of the 20 forks). A graceful stop saves at a safe point and loses ≈ 0.
- **Startup:** each resume pays T2 parity and the compile again (≈ 4–8 min, ESTIMATE).
- **RNG:** the resumed run is NOT bit-identical to an uninterrupted one. The streams restart from
  `segment_seed(seed, step)`, and the rollout in flight is discarded.
- **How that enters the seed term:**
  - A resumed final is a different draw from the same seed's distribution: same pin, data, recipe and dose.
  - The interval already treats each final's seed effect as random, with SD max(1.0 pp, σ̂), so a resume needs no
    correction.
  - "Replicate = same pin + seed" is lost for that seed, and the record says so.
  - C0's finals ran 0 restarts. If ≥ 3 ON seeds resume, Δ_F is also reported without them, descriptive only.
  - `main.dose` on an actually resumed run is unverified (REVIEW I FINDING 1).

## 14. Pre-launch checks, at the side pin (a failure is a HOLD, never a launch)

`launch_x35.sh x60t <seed> --pin 7fe15726… --lease-owner <owner>` runs `pin-check`, `corpus-check`, `checkargs
--pin`, the dry-run with the card visible and `stage-check`. Logs: `~/gen3ai_archive/x60t_prelaunch/`.

| check | result |
|---|---|
| 1. **The smoke**: the pin's trainer, `--debug --steps 10000` with the clause, a scratch `GEN3AI_MODELS_DIR` | **rc 0**. The `[HUMAN-DATA]` stamp and `[TEAM-MIX] hr_spec_v1.r3@ALL 2295 teams share 0.50 both sides` appear; no FATAL. Measured corpus-draw count at the pin (B1 option b): **50,176 / 10⁵ trainee-side (share 0.5018, 95 % CI [0.4987, 0.5049]) and 50,023 / 10⁵ opponent-side (0.5002, [0.4971, 0.5033])**, drawn through the pin's own path (parser → `MatchupSpec` → `TeamSource` with the mix installed → `yield_team` on a seeded builder copy, as `TeamStager` does); 2,295 corpus packs, 0 shared with the pool (`~/gen3ai_archive/x60t_prelaunch/review_i/`) |
| 2. **Off-means-off**: `learner_golden` + the played-bits sentinel | **8 / 8**; fixtures byte-identical at `440b61c3` and `f4b25238` |
| 3. **Null dose (m8)**: `@0` against `off` over 12,800 staged episodes | team table, indices, battle seeds and builders **identical**; `@0.5` differs. No C0h needed |
| 4. **The corpus at the pin (F-M3 fix 1)** | `kit corpus-check` PASS (stub from git, local corpus by sha256). Soak PASS at the pin and at main. Pack identity: 58 dropped. Engine: 7 dropped. Post-fill ≥ 3 net: 92 dropped |
| 5. **F-M7 guard** | landed `fbd1afa38` |
| 6. **The C0 reuse check** | the play pin as harness; R's sha; `data/` at the pin. A training-side miss ⇒ C0h; an eval-side miss ⇒ re-play (X7) |
| 7. **The ON checkpoint on the read path** | the smoke checkpoint plays 2 games on `de15dfc7b`'s `main.h2h` path, rc 0 |
| 8. **Resume refusal** | without the clause: FATAL rc 3; with it: rc 0 |
| 9. `checkargs --pin` | rc 0 for the ON and C0 argvs |
| 10. `launcher --dry-run` | refused only by the CUDA-visibility preflight (no GPU in that shell). **The lease holder re-runs it with the card visible before GO** |
| 11. **The play-pin checks of §4** (1–4) | at AM-X60T-1, before any read game |

## 15. VOID (a VOID seed is never read; it is re-run at the same pin and seed after the cause is fixed)

- **The two startup lines (REVIEW I B1, m12; X15).** VOID unless EVERY launch log of the seed holds BOTH:
  - the stamp `[HUMAN-DATA] lever=teams rowset=hr_spec_v1.r3 rows_sha=2e86457282f719eafaa75c6477c08420a03fafeb0e21f9ee5c24e1d2703afe88 …`
    with clause hash `544c9bc00ea95030` (the share appears only inside its clause text,
    `clause=teams:hr_spec_v1.r3@0.5`);
  - the banner `[TEAM-MIX] hr_spec_v1.r3@ALL (2295 teams …) share 0.50 on both sides`.

  `rows_sha` is **the corpus DATA file's sha256**, not the manifest's or the tier's. Nothing counts corpus draws
  inside a run, so the ±10 % share rule is checked ONCE, at the pin (§14.1: 0.5018 / 0.5002 over 10⁵ draws a side).
  Over ≈ 10⁶ draws a run cannot miss it unless something is mis-wired, and the two lines catch that.
- A panel-contamination hit (`main.heldout_panels.contamination` on the completed teams). The side pin's own registry
  holds forum72, H and p72, so this check is not vacuous there.
- K9(b) not PASS at arm end (`kit k9b --stage`; the ON records go to
  `x35_breakers_stage_2026-10-17/k9b/`, beside C0's and D1's).
- A FATAL naming a corpus team.
- The recorded pin is not the side pin, or a resume fails §13.

The T54 pool-seed check applies as for X35. **INCONCLUSIVE, not VOID:** an arm below §5's floor (≥ 1,200 · n − 20
counted F games; one short unit per arm is inside it), or an S cell short of its 500 pairs.

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
- H-TEAMS's corpus is the ladder population, and loser-enriched, so less transfer to F than D1's is expected.

Final classes: GAIN 0.25 · NO-USEFUL-GAIN 0.33 (look 1 0.22 of it) · NOT-RESOLVED 0.37 · LOSS 0.05. Given a GAIN:
ADOPT-CANDIDATE 0.3, TRADE 0.6, KAKUNA-SPECIFIC 0.1.

## 17. Cost (MEASURED rates, F-M8)

**GPU-h** (the 20 X35 production fork dose records, 3.84–4.43, mean **4.10** a fork):

| step | GPU-h |
|---|---:|
| smoke | done (CPU `--debug`) |
| look 1: ON s3501–s3503 (stage X60T) | 12.3 |
| on CONTINUE: ON s3504–s3508 (X60Tb) | 20.5 |
| on CONTINUE: C0 s3506–s3508 (X60T-C0b, `f4b25238d`) | 12.3 |
| **total** | **45.1** (12.3 if look 1 stops) |

**CPU, at the rates MEASURED on this box** (`fm8.rates`):
- **Kakuna:** X61P look 1 / look 2 held units at a mean 29.2 / 20.9 s a game a lane, lane occupancy including any
  heavy-slot wait. **The cost uses 25.0 s.** The design's 10 s was 2–3× low.
- **Throughput:** X61P realised 234–344 Kakuna games per wall hour at concurrency 2.
- **In-house:** 2.31–2.58 lane-h per 7 × 500 set. **The cost uses 2.45.**

**The reads, ranked so a trimmed set still decides** (8 seeds; `fm8.plan_8_seeds`; C0's X61P cells all import, m6):

| tier | what | Kakuna games | in-house sets | CPU lane-h | Kakuna wall (h, 2 lanes) |
|---|---|---:|---:|---:|---:|
| **T1** | the primary F + goal (a) + Δ_P (incl. the new C0 seeds and the determinism guard) | 14,340 | 11 | 99.8 + 26.9 | 42–61 |
| **T2** | I_F + R72 (both arms) | 0 | 32 | 78.2 | (39 on 2 S lanes) |
| T3 | REPORTED H / P72 / home72 at 7 units | 2,380 | 0 | 16.6 | 7–10 |
| T4 (last; decides nothing for adoption) | T − D1: ON H 0–29 (5 finals) + D1 H + F 0–29 at E3 | 9,000 | 0 | 62.6 | 26–39 |
| **total** | | **25,720** | **43** | **179.0 Kakuna (215 at the slowest state's rate) + 105.1 in-house** | **75–110** |

- **Look 1 needs only ON s3501–s3503's F:** 3,600 games, ≈ 10.5–15 h on 2 lanes. C0 imports.
- **Wall:** the Kakuna and S lanes run in parallel, so the full read after the last final takes ≈ 4–5 days with the 2
  heavy slots to itself. Queue position at launch: `<<AM-X60T-1: the queue at the read's start>>`.

## 18. Dispositions of REVIEW F (route T)

| finding | here |
|---|---|
| F-M1 | fix 1 = §2. Fixes 2 and 3 (a coded `x60_route`, a sha-pinned `verdict()` spec) NOT built; the route was read from the X59-S record (FINDING 1). Fix 4: X59-S has a verdict; the follow-up bears on X60-B only |
| F-M2 | §4. Fix 2 landed (`20875e962`). Fix 3 = X7. Fix 4: I_F, Δ_P and R72 are S families at the play pin, stamped `env=393631b8…/126` |
| F-M3 | §3.1, §14.4, X8 (DONE at the pin) |
| F-M4 | §5.1, §5.2, X1, X2 |
| F-M5 | route T: §9 (dose, MDE, fix 3), §11 (fix 2). Fix 1 is X60-B only: N/A |
| F-M6 | §7, §7.1, X3 |
| F-M7 | fix 1 landed (`fbd1afa38`). Fix 2 (reconcile X61 §10 with §6.4) OPEN, bearing on N1H only (§3.1). Fix 3: the manifest records both soaks |
| F-M8 | §17 |
| F-M9 | X60-B's teacher: N/A for route T |
| m1 | §10 |
| m2 | §5.2's column |
| m3 | §10 |
| m4 | §3.1 (92 dropped by the post-fill ≥ 3 net) |
| m5 | §9 |
| m6 | §2, §6: C0's F, P and σ̂ are known before registration. The floor, units and classes are §6's unchanged; only the seed count moved, by fix 3's control-only rule |
| m7 | §11 |
| m8 | §14.3 (the null dose passes), §12 |
| m9, m10 | X60-B: N/A |
| m11 | F alone stands on X61P's precedent (its review's B1), not on "ruling `4f555e57`" |

## 18b. Dispositions of REVIEW I

| finding | sev | disposition |
|---|---|---|
| B1 realised-share VOID unevaluable | BLOCKER | **Fixed, option (b)** (X15): §14.1 and §15 now check the stamp's share 0.5 and sides `both` in every launch log, plus the corpus-draw count measured at the pin (§14.1: share 0.5018 / 0.5002 over 10⁵ draws a side). No pin change |
| M1 corpus-check needs the corpus in git | MAJOR | **Fixed upstream:** `kit.human_corpus_check` reads the stub from git and the local corpus by sha256. §3 / X8 reworded: the corpus never enters git. The no-git-read test is in the kit commit `7e94d1de8` (`src/main/x35/x60t_test.py`) |
| M2 resume changes the seed | MAJOR | **Fixed, and the hazard was MEASURED at the pin:** an untyped resume through the pin's trainer ran seed 42, K9(b) fatal and no cadence, and rewrote `cli_args`. §13: every resume goes through main's launcher with `--pin-commit`, typing `--seed`, `--behaviour-check warn`, `--checkpoint-every-steps 1000000` and the clause; `--run-name` stays out (it would fork); condition 7 checks the resolved seed, mode and cadence from the dry-run; a direct pin-trainer resume is VOID |
| M3 the read code does not fix seeds | MAJOR | **Fixed:** `LOOK1`, `LOOK2` and `CONTRAST` hard-coded and tied to each run's recorded seed, with refusal tests (§5, §10, X16). Fix commit **`4d7367228`**, before look 1 is read |
| M4 the corpus is loser- and long-game-enriched | MAJOR | **Fixed (text):** §1's table, §3, §10's attribution, §11's licenses ("this six-seen, loser-enriched slice"), §12's report. A win-balanced rebuild would be a separate registration |
| M5 R72 ids are reversible | MAJOR | **Fixed:** only R72's content digest, seed and rule are committed (§9); the ids stay local. A push-guard pattern for `hr3_[0-9a-f]{16}` is recommended to the coordinator (FINDING 12) |
| m1 freeze incomplete; the switch's resume refusal overstated | MINOR | X14; §2 records main's §6 sha; §13 reworded |
| m2 the look-1 stop's other outcomes | MINOR | §5.2: the NOT-RESOLVED and LOSS stop rates and the overall false-LOSS rate; §11's LOSS row quotes it |
| m3 no tie band on the Welch guard | MINOR | §5: within 1e-12 ⇒ the wider interval; in the rule fix `4d7367228` |
| m4 guards / void optional at look 2; one pin for all runs | MINOR | §11: both required at look 2, with a pin per arm (C0 `f4b25238d`, D1 `440b61c32`, ON `7fe15726c`), every final gated: `4d7367228` |
| m5 text vs code | MINOR | §5's classes as coded; §15's short cell aligned with the floor; INCONCLUSIVE added to §11 |
| m6 C0 s3504–s3505 import as a second group | MINOR | §4, §17 and FINDING 3 corrected: no re-play of F or P; both confirmed in the rule build `4d7367228` (P: 210 + 140 ledger rows, pooled S equal to look 2's) |
| m7 the play-pin commit needs void-check support | MINOR | §4: the play-pin commit carries it; `assemble` / `decide` / `record` run from main |
| m8 effective N | MINOR | §3.1 and §9: 2,295 teams and ≈ 1,504 distinct account-sixes; dose ≈ 124 and ≈ 190 a side |
| m9 provenance nits | MINOR | §3.1 records the manifest `commit` / `source_sha256` gap and the null `snapshot` |
| m10 `n1_hrep_v1` does not exist | MINOR | §3.1 reworded; F-M7 fix 2 listed OPEN in §18 |
| m11 backport hazards | MINOR | the side pin backports `utils.paths`' archive-dir helpers. `_install_mix` has no run-time pack check, so it is safe only because the pin-check dropped the 58. Main's `TeamLoader` is NOT backported. Confirmed at the pin: `gen3ai_archive_dir` / `human_teams_dir` present, `team_loader` and `teambuilder` byte-identical to `440b61c3`. `kit pin-check`'s path list enumerates the backport (`X60T_PIN_PATHS`; FINDING 13) |
| m12 the stamp's `rows_sha` | MINOR | §15 names it the corpus DATA file's sha256 |
| FINDING 1 not verified | — | carried into §19.10 |
| FINDING 2 the slow-tier recorder wrote main's tracked file | — | §19.15; this unit ran gates with `-m "not slow"` |
| FINDING 3 TBD rules | — | §20 |

## 19. FINDINGS (standing rule 7)

1. **The route was applied by reading, not by code** (F-M1 fixes 2–3 unbuilt). §6.1 row 3 names OWN-LABELS-BETTER,
   so the outcome is unambiguous, but no `x60_route()` was run.
2. **σ̂'s UB fires the 8-seed rule:** +32.8 GPU-h on CONTINUE over the 5-seed plan.
3. **C0's X61P cells span two read states;** they import as two one-cell groups (REVIEW I m6), so neither F nor P is
   re-played. Both imports were confirmed in the rule build `4d7367228`.
4. **No arm has I_F,** so the meters' exchange rate on F is unmeasured; the planning rate is structural. I_F's
   per-final SD on forum72 is unmeasured too.
5. **The end-to-end simulation treats the meters as independent given the truth,** and G-F / G-E / G-C are not
   modelled.
6. **At D1's measured in-pool cost (Δ_P −1.6 pp), P(ADOPT-CANDIDATE) for a true +3 pp lever is 0.08–0.16;** TRADE is
   the likely positive outcome. Δ_P's rule is D1's and is unchanged. The owner may want to know before launch.
7. **The always-current docs still quote the old power and the point guard:** design §0 / §6.0 / §6.7 and Q-H11.
   F-M4 fix 1 and F-M6 fix 3 ask for those edits; this unit edits only the measurement dir.
8. **The play pin `de15dfc7b` predates later harness guards.** G14's forced-end check reads `unavailable` on home72,
   as in X61P. The X35-frame shared pin `86cfdf7ef` (E1) would forfeit every C0 import.
9. **D1's doc quotes ≈ 118 draws a team a side at 1,981 teams;** its "570k games" rule gives 144. This PREREG uses
   the rule.
10. **Not verified** (with REVIEW I FINDING 1):
    - that registering R72 leaves forum72 / home72's ids unchanged (AM-X60T-1);
    - `main.dose` on an actually resumed run;
    - whether H carries the corpus's six-seen skew;
    - that no training-time writer puts corpus team text under `models/`.
11. **The Kakuna rate includes heavy-slot waits,** so 25 s a game is lane occupancy under this box's usual contention.
12. **R72 ids are reversible battle pointers** (REVIEW I M5). They stay local; a push-guard pattern for
    `hr3_[0-9a-f]{16}` is not built.
13. **`_install_mix` at the side pin has no run-time pack-identity check.** A corpus that has not passed the offline
    pin-check would load. `human.load` refusing any corpus other than the stub's is what keeps it safe.
14. **A resume through the pin's trainer silently corrupts the run's record** (measured, REVIEW I M2): seed 42, K9(b)
    fatal, no cadence, `cli_args` overwritten, and main's launcher then restores the corrupted record. §13 forbids it;
    nothing in code does.
15. **The slow-tier recorder writes into whichever checkout runs it** (REVIEW I FINDING 2), so this unit's gates ran
    with `-m "not slow"`.
16. **Process:**
    - CPU only, `CUDA_VISIBLE_DEVICES=""`. The draft ran at nice 19; the registration as `GEN3AI_BLOCKING` at nice 0
      on 5-7,13-15, under `mem_cap.sh`.
    - Read-only on `models/` and the archive. No GPU, game or launch.
    - REVIEW I was checked for team text and ids before it was copied: only agent ids, a regime id and the clause
      hash are present.

## 20. Remaining placeholders, each tied to its amendment

| placeholder | amendment | filling rule |
|---|---|---|
| the play pin SHA (`de15dfc7b` + the port of rule `x60` at `4d7367228`, R72's registration, the void-check support); `playpath.compare`; the ids of §4.3; the read specs and their sha256s | **AM-X60T-1**, before any read game | §4's four checks; `check-spec` exit 0; nothing in §5–§11 may change |
| the queue at the read's start | AM-X60T-1 | `heavy_slots` status at the first read game |

## 21. AMENDMENT AM-X60T-1 (2026-11-06, before any read game): the play pin, its checks, look 1's spec, G-C at 8 pairs

**Why, and when.** Written while look 1's training (ON s3501–s3503) runs, before any X60 read game: no `rb_x60t_*`
final exists, no `main.reads plan` has run on an X60 spec at any real state path, no G-C record has been pooled for
any X60 seed, and C0 s3506–s3508 are not trained. It fills §20's placeholders and makes ONE fix the registered intent
needs (§21.5). **Nothing in §5–§11 changes:** the arms, seeds, cells, classes, thresholds, floor and the verdict table
stand as registered.

### 21.1 The play pin

**`f564c333f9437f6a636e9daf890535c2c300aeb2`** (annotated tag `x60t-play-f564c333`, on origin) = `de15dfc7b` + ONE
commit:

| path | what |
|---|---|
| `src/main/reads/rules_x60.py` | rule `x60`, byte-identical to main's (this amendment's main commit) |
| `src/main/reads/rules.py` | the registration; `_void_check` takes main's `void.pin` (a sha or `{arm: sha}`) and `void.dose`; `void_gate` REFUSES to read them there (fail closed: they are read only from main, §4) |
| `designs/research_state/heldout_panels.json` | **R72** as `r72`: PRACTISED, `held_out_from: []`, `n` 72, content digest `10364442…5501` (§9), seed 6072, the local manifest by sha256 `7aa2342e…e66f`; `team_dir` and manifest are LOCAL ABSOLUTE paths under `~/gen3ai_archive/human_teams/hr_spec_v1.r3/r72/`. Team ids and texts never enter git (REVIEW I M5). Not portable off this box |
| `src/main/reads/rules_x60_test.py`, `rules_test.py`, `src/main/heldout_panels_test.py` | main's x60 test less its `inputs` / `decide` / `power.py` tests (main holds them; the pin predates their modules), plus: the look-1 spec parses there and the run-dir guards are refused there; the rule and panel sets name `x60` / `r72` |

**The playing checkout:** `~/dev/gen3ai-reads/x60t_play_de15dfc7` at the tag, clean. It was built by its own
`scripts/bootstrap.sh --skip-env --with-rust` on 2026-11-06, and §21.2 check 5 verifies the build.

**Main learns the pin** (this amendment's main commit): `engine.LEGACY_ENGINE` maps it to E3,
`side_identity.LEGACY_COMMIT_MODE` to `eval`, and `LEGACY_PEER_CODE`'s `b817d3216c418f75` row names it. The pin predates
the engine and side stamps, so main reconstructs its units' engine and sides from the playing commit. A commit outside
those tables is UNKNOWN, and an import never accepts it: without the rows every X60 group would read INCONCLUSIVE.
`engine_test` / `side_identity_test` re-derive all three from git.

### 21.2 §4's four checks at the play pin

| # | check | result |
|---|---|---|
| 1 | `playpath.compare('de15dfc7b', play pin)` | **one difference: `changed designs/research_state/heldout_panels.json`** (1,537 play-path files; digest `5f0b3151…f6cf`). PASS |
| 2 | `engine_at(play pin) == E3` | **E3** (`port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126`), as at `de15dfc7b`. PASS |
| 3 | the ids of §4.3 | **No registered id reads the registry outside its own panel's entry.** The forum72 / home72 / heldout72_hpfix / p72 / away entries are byte-identical, and so are their `panel_team_set_id`s at both commits (`ts:61b475d4…` forum72, `ts:08142ee8…` home72; the S regime's only registry input). The Kakuna id's inputs (`assemble.ANCHOR_REGIME_KEYS`, team digests, weights, engine, sides) read the cell config and team dirs, never the registry. PASS by construction. **Machine-checked at read time:** main's `assemble` gives group `primary` ONE regime id per team set, over the imported C0 units and the ON units played at the pin together. Two ids make the group, and so Δ_F, INCONCLUSIVE. Main's assemble of the dry-run state already reads the imported side as `2118812c27885de3` (group `00c8c0114bdf0836`), E3, `their=legacy` / `eval` |
| 4 | an ON checkpoint plays on the pin's `main.h2h` path | **PASS.** s3501's first checkpoint `checkpoint_281000212_steps.zip` (sha256 `c5b44c42…4952`, a copy beside the run's `model_config.json`) played 1 mirrored pair against itself at the play pin: 2 games, rc 0, W/L/D 1/1/0, CPU, at the coordinator's request (`~/gen3ai_archive/x60t_prelaunch/am1_h2h_check/`). The first try refused before any game because a bare zip copy has no `model_config.json` beside it; the loader requires one |
| 5 | the build is E3 (added by REVIEW AM-1 B1, AM-N1P-2's check) | **both `Cargo.lock` blobs equal `LEGACY_LOCK_BLOBS`** (`676e633e…` sim, `3d00802e…` env). The built `sim_bridge` stamps `commit=f564c333f…;src=0d3d7629f16a0eaf;nfiles=91`; the env tree hashes `393631b886965b0c` / 126. PASS. The env core's release build happens on demand at the first unit, from the same tree and lock |

### 21.3 Look 1's spec (registered)

[`read_spec_look1.json`](read_spec_look1.json), sha256 **`0f5d9a5fe18aa0b9545f66cc40759024d407618c0d7b389a13e27c7a94d0c57a`**.
`rules_x60_test` holds it equal to the builder's draft plus the `pin` and note.
- `pin` = the play pin. `plan` refuses any other playing commit, or a dirty tree.
- **Models.** ON `x60t_s3501`–`s3503` by the settled `final` rule (`models/rb_x60t_s<s>`, step ≥ 300,000,170, quiet
  300 s; §13.1). C0 `c0_s3501`–`s3503` pinned by sha256 (§4).
- **Anchor `primary`:** Kakuna, forum72, unit seeds 0–59, all six. C0's three are IMPORTED from X61P look 1's group
  `primary` (state `~/gen3ai_archive/x61p_n1p_read/look1`, spec sha256 `1266daad…9f03`). ON's are played.
- **Anchor `determinism`:** c0_s3501, forum72, unit seed 0, played. §4's guard against the imported unit, game for game
  and choice for choice. X61P's unit holds all 20 choice captures (`assemble.cell_choices`: 20 games, 20 choice
  records), so main's default `choices_guard: require` compares choices.
- **Decision:** rule `x60`, `look` 1, `on` ON, `off` C0, `primary` forum72 at 60 units, the determinism guard. The VOID
  gate: K9(b) from `x35_breakers_stage_2026-10-17/k9b`, every ON and C0 run gated, `pin`
  `{x60t: 7fe15726c…, c0: f4b25238d}`, `dose` against `models/rb_x35_c0_s3501` at a relative 1e-6.
- **Look 1 plays Δ_F only: a REPORTING-ONLY reconciliation of §9 / §11 with §17** (REVIEW AM-1 M3).
  - §17 priced look 1 as "only ON s3501–s3503's F". But §9 and §11's look-1 row say I_F and R72 are REPORTED at a stop,
    and §12 lists REPORTED H / P72 / home72. The registered text contradicted itself.
  - No verdict depends on the choice: X5 keeps I_F and R72 out of every look-1 decision, and `verdict()` never reads
    them there.
  - Playing R72 needs it resolvable by MAIN's assemble (FINDING 18). Every S or REPORTED set would also hold the two
    heavy slots look 1's Kakuna lanes need.
  - **So a look-1 stop record carries none of I_F, R72, H, P72 or home72.** Its decision shows `i_f` / `r72` as
    `{"read": false, "reasons": ["…: absent"]}`.
  - If the orchestrator wants them after a stop, it is a separate REPORTED read at this play pin. I_F can be read at
    once; R72 only after FINDING 18 is resolved.
  - On CONTINUE, all 16 finals play I_F and R72 at look 2 (§17's 32 T2 sets).

**Validated by executing (CPU, no game):**
- `python -m main.reads check-spec`: **exit 0 at main and at the play pin.** Main also reports `rederived_guards:
  count` and `choices_guard: require`.
- **`plan` at the play pin into a scratch state** (not the registered path): **exit 0**, spec sha256 `0f5d9a5f…`.
  - C0 s3501–s3503 were copied at steps 300,061,940 / 300,061,805 / 300,061,491, each matching its sha256.
  - The X61P import passed `imports.check`, and the determinism group planned 1 unit.
  - ON s3501–s3503 are `waiting` ("no final yet").
- **MAIN's `assemble` of that scratch state:** exit 0. Group `primary` imports 180 C0 unit rows, all counted, on E3
  with regime `2118812c27885de3`, sides `their=legacy` / `eval`, and 180 ON units unplayed. So main's engine, sides and
  regime checks accept the import before any game.
- **Main's VOID gate on the spec now:** c0_s3501–s3503 **PASS** on K9(b), pin `f4b25238d` and dose. ON reads MISSING,
  since no final, record or `metadata.json` exists yet.

### 21.4 How look 1 is played and read (the operator: Training Oct)

The registered state is **`~/gen3ai_archive/x60t_read/look1`**. Steps 1–3 run in the playing checkout with
`export PYTHONPATH=$PWD/src`; step 4 runs in a main checkout.
1. **§4.4 is DONE (§21.2 check 4, PASS).**
   - **It checks the S loader, not the Kakuna loader look 1 uses** (REVIEW AM-1 m5).
   - A Kakuna-side load refusal shows only as uncounted units. So the operator reads the anchor lane's first ON unit
     (`status`: counted, not attempt-exhausted) before leaving the lane unattended.
2. `python -m main.reads plan --spec <main>/designs/research_state/measurements/x60t_h_teams_2026-11-06/read_spec_look1.json --state ~/gen3ai_archive/x60t_read/look1`.
   Plan may run before the finals settle: an unsettled ON model waits, and the C0 imports are checked at plan.
3. `python -m main.reads run --state ~/gen3ai_archive/x60t_read/look1 --lane anchor --hours 100 --concurrency 2`,
   detached, as X61P look 2 ran. Record the queue at the first read game (`scripts/ops/mem_cap.sh --status`) in the
   look-1 record: §17's placeholder.
4. From main, after every ON unit is played and the three AM-6 K9(b) records `k9b/rb_x60t_s350{1,2,3}.json` exist:
   `python -m main.reads assemble|decide|record --state ~/gen3ai_archive/x60t_read/look1`. Then the operator checks
   §15's launch-log lines and the panel-contamination check, which no module reads.

### 21.5 G-C at 8 pairs (the fix; FINDING 2 of the rule's build)

**The problem, checked in code.** §11 reads G-C over the read's seed pairs. Look 2 is 8 v 8, so its record must pool
s3501–s3508.
- `main.x35.gc.SIDE_SEEDS["production"]` was s3501–s3505.
- `gc.pool_arrays` refused any other production seed. With 3506–3508 a seed set has no side, so `check_record`
  refused as well.
- So G-C read INCONCLUSIVE at every 8 v 8 read, and §11's ADOPT-CANDIDATE needs G-C PASS. An otherwise passing GAIN
  could only read INCONCLUSIVE, never ADOPT-CANDIDATE.
- The rule's own test had hidden this by monkeypatching `guard_c_record`.
- G-F (`cross_stat`, any n × n) and G-E (`gap_diff`, the s3501 finals) carry no seed set, so they were never affected.

**The fix:** a side's seeds are its C0's (`kit.ARMS["c0"].seeds` = 3501–3508, AM-X60T's top-up). A pair is an arm's
final against its same-seed C0 final, so that is the side.
- `check_record` still requires the record's ON and OFF seeds to EQUAL the read's. So an X35 five-seed read accepts only
  its own five-pair record, and an 8 v 8 read refuses a five-pair one (`gc_test`, `rules_x60_test`, both on real
  constructed records).
- Unchanged: G-C's arithmetic, its bars, its truth (X26@280M, the production side's), and its place in §11.

**Why this is legitimate now:** no X60 read data exists. No X60 game has been played, no G-C record has been pooled for
any X60 seed, and the seeds it admits (3506–3508) have no trained final in any arm. The fix changes no class or
threshold. It only lets §11's registered "G-C PASS" be evaluated at the seed count §5.1 registered.

### 21.6 Not registered here (each an amendment before its first game)

- **Look 2's spec** (AM-X60T-2). It needs:
  - C0 s3506–s3508's final sha256s;
  - look 1's `spec.json` sha256 for its imports: the sha above, if look 1 is planned from this file;
  - the guards artefact `{path, sha256}`, written only once look 2's finals exist;
  - R72 resolvable by main's assemble (FINDING 18).
- **`ge_x60t`**, §11's fresh G-E reader: not built (FINDING 19). It is needed only on a look-2 GAIN.

### 21.7 Its review (REVIEW AM-1, an independent opus-high reviewer, 2026-11-06): READY-WITH-FIXES, all fixed

| finding | fix |
|---|---|
| B1: the play checkout was not built (no `Cargo.lock`, no `target/`) when the reviewer read it | it was mid-build; built, and the locks and stamps verified (§21.2 check 5) |
| M1: the tag was not on origin | tagged and pushed (the push guard ran against the pin's base `de15dfc7b`: OK) |
| M2: the plan result was a placeholder | the plan and main-assemble dry-runs above |
| M3: §21.3 changed §9 / §11's look-1 reporting | restated as a reporting-only reconciliation with §17, including §12's panels (§21.3); FINDING 23 |
| m1: check 3 named the wrong mechanism | reworded: `assemble`'s one-regime-per-team-set rule over `primary` |
| m2: the record's wording | aligned to `decide`'s `read: false` |
| m3: `LEGACY_ENGINE`'s "never edited" | the docstring says when a row is added |
| m4: reverting the new table rows failed no test | `rules_x60_test.test_main_reconstructs_the_play_pins_engine_and_sides` (fails on revert, checked) |
| m5: §4.4's check tests the S loader | §21.4 step 1 names it and adds a first-unit read |

### 21.8 FINDINGS (this amendment)

17. **The play pin is not portable off this box:** R72's `team_dir` and manifest are local absolute paths, by design
    (§9).
18. **R72 cannot be ASSEMBLED from main as main stands.**
    - `assemble.s_group` resolves an S panel's team set through main's `heldout_panels.json`, and `r72` is registered
      only at the play pin.
    - Registering it on main changes main-wide behaviour:
      - every corpus builder's exclusion list (`build_w_forum`, `build_n1_proc`) would read it;
      - the registry tests would need the local archive.
      That is outside this amendment.
    - Options for AM-X60T-2:
      - (a) register `r72` on main, with the builders made to skip PRACTISED panels;
      - (b) let an S group name its team-set id in the spec when its panel is pin-only.
    - Until then, look 2's R72 meter would read INCONCLUSIVE, and so would a look-2 NO-USEFUL-GAIN verdict.
19. **`ge_x60t` (§11's G-E reader) is not built;** `main.x35.ge` reads only `TARGET_SEEDS = (3501,)` and named arms.
20. **§4.4 played 2 h2h games on a TRAINING checkpoint** (s3501 @ 281,000,212; no final), at the coordinator's
    request: the S loader only, not the Kakuna loader.
21. **§17's queue placeholder cannot be filled before the first read game.** It is §21.4 step 3's record line.
22. **The pin's own test file is a subset of main's.** The interval-versus-`power.py` and decide-side tests run only
    on main, where `decide` runs.
23. **A look-1 stop reports no I_F, R72, H, P72 or home72** (§21.3). That is the reporting-only reconciliation of §9 /
    §11 / §12 with §17. If those numbers are wanted after a stop, it is the orchestrator's call: a separate REPORTED
    read at the play pin.

## 22. Look 1's run record (Training Oct, written 2026-11-07 BEFORE `decide`)

Every fact below was fixed before the verdict was computed. Evidence: `~/.claude/jobs/x35/status.txt` (the `x60t s350X` and
`X60TD` lines), each seed's kit launcher log `~/.claude/jobs/x35/rb_x60t_s350X.launcher.log*`, and the archives named.

**1. The aborted s3501 (never read).**
- Launched 2026-11-06 08:51:41 by `launch_x35.sh x60t 3501` GO.
- The sequencer had wrapped the GO in `timeout 2400`. At 09:31:32 GNU timeout's process-group SIGTERM reached the
  launcher, which stopped gracefully at 281,776,556.
- The orchestrator chose a fresh relaunch over a resume (option A). The dir was moved WHOLE to
  `models/rb_x60t_s3501.aborted-20261106`, and its launcher log to `rb_x60t_s3501.launcher.log.aborted-20261106`.
  Nothing was deleted. The read's `final` rule names only `models/rb_x60t_s3501`.
- The class fix is `2e63e5f0e`: the kit's spawn uses `exec setsid`, plus a header warning and two tests.
- s3501 then ran fresh at 09:34:01, at the same pin, seed and argv.

**2. Resume #1 of s3501 (§13).**

| field | value |
|---|---|
| cause | FATAL T2 `ParityFailure` at 10:00:04, slot 24, bucket 256, legal-logp 0.1432 over the 1e-3 bar; launcher_rc 3 |
| root cause | A selection near-tie in `hypothesis_set.stable_order`: fixture rows 11 / 12, HP Grass vs HP Ice at the K=6 cut, relative margin 1.6e-6. The b256 compiled graph's autotuned launch configs resolve it the other way. Reproduced in the training geometry and confirmed in a GPU window (`RESULT C1 PASS`, `RESULT B_POISON PASS`; `~/gen3ai_archive/x60t_parity_2026-11-06/`). |
| fix | The fix is for main and future runs only (`gen3_t2_parity_tie_judge_v1`). The arm stayed on `7fe15726` with no amendment, by the orchestrator's ruling. |
| resumed | 10:32:41, from `checkpoint_281000181_steps.zip` (281,000,181) |
| steps re-trained | 875,048 |
| argv | §13's, through main's launcher with `--pin-commit 7fe15726…`, no `--run-name` |
| condition 7 | The dry-run (`~/gen3ai_archive/x60t_resumes/s3501_resume1_dryrun.txt`, sha256 `b5eab43139118f15…`) resolved `--seed 3501` (TYPED), K9(b) `warn` (typed) and `--checkpoint-every-steps 1000000` (TYPED). The clause was `teams:hr_spec_v1.r3@0.5`, hash `544c9bc00ea95030`. |
| outcome | Ended clean at 14:19, at 300,078,185 |
| live-row guard | The orchestrator's resume condition: any K9(b) logp max > 1e-3 ⇒ HOLD. The max over the run was 4.83e-4. |
| replicate | "Replicate = same pin + seed" is lost for s3501 (§13). |

1 of 3 ON seeds resumed, which is under §13's ≥ 3 bar, so there is no without-resumed report. s3502 and s3503 ran 0
restarts.

**3. §13's conditions and §15's VOID checks, per seed.**

| seed | final (step) | launch logs read | `[HUMAN-DATA]` stamp (rows_sha `2e864572…`, hash `544c9bc00ea95030`) | `[TEAM-MIX]` 2295 teams, 0.50 both sides | held-out panels | `corpus_<n>` / FATAL naming a corpus team | pin_history | dose = C0 (display) | K9(b) at arm end |
|---|---|---|---|---|---|---|---|---|---|
| 3501 | 300,078,185 | 2: the fresh launch (`.20261106T103241.prev`) and the resume | in both | in both | 0 hits over 5 panels (registry `ea3917c2f1e6`); 0 corpus teams pack to a pool team, in both | none | 1 span, `7fe15726` | lr_median 1.005e-4, upd/step 2.035e-4, dose_rate 2.044e-08 | PASS 4.83e-4 / p99 3.59e-5 |
| 3502 | 300,061,763 | 1 | yes | yes | as s3501 | none | 1 span | as s3501 | PASS 2.18e-4 / p99 3.14e-5 |
| 3503 | 300,061,279 | 1 | yes | yes | as s3501 | none | 1 span | as s3501 | PASS 2.06e-4 / p99 3.62e-5 |

- The K9(b) records are `x35_breakers_stage_2026-10-17/k9b/rb_x60t_s350{1,2,3}.json` (`7880cd902`).
- Every T54 post-spawn pool-seed check PASSED.
- The exact relative-1e-6 dose comparison is the read's `void.dose`.
- Each log's `FATAL` matches are the three standing banners: compile is "FATAL by design", the `FATAL_SUPPLY (5)` notice and
  the K9 learner-golden notice. s3501's fresh-launch log also holds the T2 `ParityFailure` above.

**4. §17's queue placeholder: the queue at the read's start** (`mem_cap.sh --status`, 2026-11-06 09:39:10,
`~/gen3ai_archive/x60t_read/look1_queue_at_start_20261106T093910.txt`, sha256 `75f3e7f5dc8a132e…`):
- Both heavy slots were held by the S′-HONEST stage-1b jobs.
- 3 were queued:
  - the read's first Kakuna unit (the determinism unit c0_s3501, SHORT ≤ 600 s);
  - an `honest_inc_test` pytest;
  - an S′-HONEST PREREG job.
- No BLOCKING bypassers.

**5. How it was played.**
- Playing code `f564c333f` (`code.json`: clean, `diff_sha` empty), checkout `~/dev/gen3ai-reads/x60t_play_de15dfc7`.
- Lane `anchor`, concurrency 2, under the user unit `x60t-look1-anchor.service` (CPUAffinity 5-7,13-15, Nice 19).
- The lane ran 2026-11-06 09:38 → 2026-11-07 00:40. It ended by itself, `done`, with `not_counted []`.
- §21.4 step 1: the operator read s3503's first unit as counted (it settled at 23:07; counted 23:11). s3501's and
  s3502's units were all counted (60 / 60 each) when that read was made; their first units were not read one by one.
- 180 / 180 ON units counted (3,600 games); the determinism unit 1 / 1; C0's 180 units imported (§21.3).

## 23. AMENDMENT AM-X60T-2 (2026-11-07, after look 1's CONTINUE, before any X60Tb / X60T-C0b launch or look-2 game)

**Why, and when.** Look 1 read CONTINUE (`af57d4fd3`, §22). No `rb_x60t_s3504`–`s3508` or `rb_x35_c0_s3506`–`s3508`
exists and no look-2 game has been played. This fills §21.6, closes FINDINGS 18 and 19, and fixes the stage-record
defect look 1's read found. **Nothing in §5–§11 changes:** the rule `x60`'s check, inputs, decide and verdict are
byte-for-byte as registered (`rules_x60.py`'s registered functions untouched; one new function, `stage_keys`, that no
verdict reads). Look 2 reads the secondaries as registered (I_F, R72, Δ_P, goal (a), H / P72 / home72, T − D1).

### 23.1 The stage record (the defect, the fix, and the re-serialisation proof)

- **The defect.** Rule `x60`'s decision keeps its own shape (`decision.decision.verdict`), so its record had no top-level
  `arm` / `look`. `main.x35.batch.read_record` refused it ("arm None look None, not X60T look 1"), so
  `kit stage-check --arm x60t --seed 3504` read look 1 as not yet read (PENDING, fail closed), and the record could not
  sit in `reads/` (`batch_test::test_every_committed_read_record_carries_its_arms_registered_label`).
- **The fix, at the record's interface** (one contract for every stage reader, not an x60 special case in `batch`):
  `Rule.stage_keys`. A rule whose decision keeps its own shape names its stage identity; `record.build` writes it at the
  top level and `record.check` re-derives it from the record's own `models` and `decision`, so a hand-edited key fails.
  For `x60`: `arm` = `READ_LABEL` of the ON arm (`X60T`), `look`, and `delta_f` = `{final, class}` (`batch.PRIMARY_KEY`):
  final at a look-1 futility stop or at a classed look 2; not final at a look-1 CONTINUE or an INCONCLUSIVE Δ_F. The
  decision is never reshaped, and `read_record` is unchanged.
- **Re-recorded, not re-decided in substance.** `python -m main.reads record --state ~/gen3ai_archive/x60t_read/look1
  --out <stage>/reads/x60t_look1.json`, run from the MAIN checkout once the fix had shipped (`80112244c`; the `void` block names the
  K9(b) files by the recorder's checkout, so a worktree run would name worktree paths; the `--out` went to this
  amendment's worktree, never into main's tree). Against `read/record_x60t_look1.json` (sha256 `538700ee…`): every key
  but `created` is byte-identical in canonical serialisation (`decision`, `inputs`, `verdict`, `models`, `regimes`,
  `imports`, `void`, `code`, …); the only additions are `arm` `X60T`, `look` 1, `delta_f` `{final: false, class:
  CONTINUE}`. `x60t_test` holds exactly that.
- **What it wrote in the registered state** (`record` re-assembles, re-decides and writes the state): `cells.json`,
  `inputs.json`, `decision.json` and `void.json` byte-identical to before; `record.json` is now the stage copy (the
  original `538700ee…` is the committed `read/record_x60t_look1.json`). All eight state files as they were before
  are kept in `~/gen3ai_archive/x60t_read/look1_before_am2_rerecord/`. A trial run from the worktree before the fix
  shipped had rewritten the same files; its `void.json` (worktree k9b paths) and `record.json` were restored from that
  backup at once (timestamps preserved, `cmp`-equal; REVIEW AM-2 m2).
- **`batch.read_record` now runs `record.check` on any `main.reads` record** (REVIEW AM-2 m5), so a hand-edited or
  stale stage key refuses instead of licensing a launch.
- **`kit stage-check`** (was PENDING): `x60t` s3504–s3508 and `c0` s3506–s3508 **ALLOWED** ("x60t look 1 is CONTINUE").

### 23.2 R72 at main (FINDING 18, option (b), with a field both parsers already accept)

- **Option (a) was not taken:** registering `r72` on main changes main-wide behaviour (every corpus builder, the
  contamination checks and the registry's sha in every launch stamp would read a PRACTISED, local-only, reversible
  panel), and the registry tests would need the local archive.
- **Option (b), by the S group's existing `regime_id`.** The play pin's spec parser already accepts it, so the spec
  plans there unchanged. `assemble.panel_team_set`: a panel main does not register is read ONLY when its group
  registers a `regime_id`. That id digests the team set (`eval_ledger.schema.REGIME_IDENTITY_V2`), so the
  registered-regime check pins R72's teams, and the rows must be on one team set. Without one the group is INCONCLUSIVE
  (fail closed). Every panel main registers is checked exactly as before.
- **The three registered S ids:** home72 `d4de0fd7c16a2658` (X61P's), forum72 `0c972bc06eab468f`, r72
  `1bfe0b8a101ba8fb` (team set `ts:a81d977d7a44b6f4`). They were computed at the play pin by its own
  `plan_regime(250, "off", panel)`. **Main re-derives all three from its own code**, R72's from §9's committed digest
  `10364442…5501` and n 72 alone (`rules_x60_look2_test`).

### 23.3 Look 2's spec (registered)

[`read_spec_look2.json`](read_spec_look2.json), sha256 **`a5ad74116e9f140c878cd3698d06d81f4f0ba997806e91aa92111d8205cb8ed7`**,
`pin` = the play pin `f564c333f`. `rules_x60_look2_test` holds the file equal to its builder byte for byte.

**Models (21).** ON `x60t_s3501`–`s3503` pinned by sha256 (look 1's finals: `d576e096…`, `13501785…`, `5979dbb1…`);
ON `x60t_s3504`–`s3508` and C0 `c0_s3506`–`s3508` by the settled `final` rule (`min_step` 300,000,170, quiet 300 s,
§13.1), as look 1's ON were; C0 `c0_s3501`–`s3505` pinned (§4's sha256s); D1 `d1_s3501`–`s3505` pinned (`3d5c9d45…`,
`dc3d5cd7…`, `30e4e79a…`, `fdaf6daa…`, `2e36697e…`, each equal to D1's own read inputs).

**Anchor groups (Kakuna ckpt 34).**

| group | team sets, units | models | import |
|---|---|---|---|
| `primary` | forum72, 0–59 | ON × 8, C0 s3506–s3508 | ON s3501–s3503 from X60-T look 1 (`0f5d9a5f…`) |
| `primary_c0a` | forum72, 0–59 | C0 s3501–s3503 | X61P look 1 `primary` (`1266daad…`) |
| `primary_c0b` | forum72, 0–59 | C0 s3504–s3505 | X61P look 2 `primary` (`f28654cc…`) |
| `determinism` | forum72, 0 | c0_s3501 | played; the guard against `primary_c0a` (§4's unit) |
| `home72` | home72, 0–6 | ON × 8, C0 s3506–s3508 | goal (a) (ON) and REPORTED Δ_home72 |
| `p72` | p72, 0–6 | ON × 8, C0 s3506–s3508 | REPORTED Δ_P72 |
| `h` | heldout72_hpfix, 0–6 | ON s3506–s3508, C0 s3506–s3508 | REPORTED Δ_H |
| `h_contrast` | heldout72_hpfix, 0–29 | ON s3501–s3505 | T − D1's ON H; ON's Δ_H units 0–6 are nested in it (§12: "ON plays H 0–29 for §10") |
| `d1` | heldout72_hpfix + forum72, 0–29 | D1 × 5 | T − D1, re-played at E3 (§4) |
| `reported_c0a` / `reported_c0b` | H + p72 + home72, 0–6 | C0 s3501–s3503 / s3504–s3505 | X61P look 1 / look 2 `reported` |

**S groups** (R = `reference_population_production.json`, content `cef0960e…`; 7 × 500 pairs; in this order, which is
the S lane's play order: T1 before T2): `p` (home72; ON × 8 + C0 s3506–s3508), `p_c0a` / `p_c0b` (home72; C0
s3501–s3503 / s3504–s3505 imported from X61P look 1 / look 2's `p`), `i_f` (forum72; ON × 8 + C0 × 8), `r72` (r72;
ON × 8 + C0 × 8), each with its registered `regime_id` (§23.2).

**Decision:** rule `x60`, `look` 2; `primary` = the three forum72 groups at 60 units; `determinism_guard`
`{determinism, primary_c0a}`; `goal_a` `home72` at 7 units; `p` = `p` + `p_c0a` + `p_c0b`; `i_f`; `r72`; `contrast`
ON s3501–s3505 v D1 on H + F at 30 units over `primary` + `h_contrast` + `d1`; `reported` = home72, p72, h,
h_contrast, reported_c0a, reported_c0b; `guards` = the stub of §23.4; `void` = K9(b) from `x35_breakers_stage_2026-10-17/k9b`
on all 21 runs, pin `{x60t: 7fe15726c…, c0: f4b25238d, d1: 440b61c32}`, dose against `models/rb_x35_c0_s3501` at
relative 1e-6.

**Every choice here is mechanical, and each is stated:**
1. **ON s3501–s3503's F units are imported from look 1, not re-played:** the same play pin, the same bytes (pinned by
   sha256). §17's plan counted them again (3,600 games).
2. **C0 s3506–s3508 are read by the `final` rule, not pinned by sha256** (§21.6 asked for their sha256s): they do not
   exist yet, and the spec must be registered before they launch. The rule resolves exactly one settled file; its
   sha256 is fixed in `models.json` at settle and in the record's `models`. §3's "every C0 final is pinned by sha256"
   holds for every C0 final that existed at registration.
3. **The determinism guard re-plays §4's unit** (c0_s3501 forum72 unit 0) against its import. The rule takes one guard.
   X61P look 2's units were played at `de15dfc7b`, on the same play path (§21.2 check 1).
4. **Δ_H / Δ_P72 / Δ_home72 are reported per group** (rule `x60`'s `reported`), so C0's three groups and ON's are
   separate rows, and `h_contrast` is reported over its 30 units. The pooled Δs at units 0–6 are summed from the
   state's `cells.json` unit rows in the look-2 README. They are REPORTED only; no verdict reads them.
5. **The S lane plays `p` first** (T1), then `i_f` and `r72` (T2), §17's ranking.

### 23.4 The guards artefact, registered before any look-2 game

[`read/guards_look2_not_played.json`](read/guards_look2_not_played.json), sha256
`282f843c664ff16bf172d60b07ae3520e67e894d920a5543edefa68c91b8bc70`, holds `g_f`, `r2` and `g_c` all null.
- **Why a stub:** §11 reads G-F / G-E / G-C only on a Δ_F GAIN, and their inputs can only be produced after one:
  - the 8 × 8 ON × OFF head-to-head matrix;
  - the `ge_x60t` / `ge_c0` readers' `r2` (§23.5, GPU, triggered by the GAIN);
  - the 8-pair G-C record (§21.5).
  But `LOOK2_REQUIRED` makes the spec carry an artefact, and a state is one spec, so its sha256 is fixed now.
- **What it does:** rule `x60` reads each guard INCONCLUSIVE ("no head-to-head matrix", "no R2 read", "no G-C read").
  - Off a GAIN, nothing reads it.
  - On a GAIN, §11 resolves as registered: KAKUNA-SPECIFIC and TRADE are decided without the guards (TRADE as soon as
    one of its conditions is known), and otherwise the verdict is INCONCLUSIVE on the guards alone. That is §11's
    INCONCLUSIVE row: "the missing piece is played … and the read re-decided, with nothing else changed".
- **The re-decision, on a GAIN only (AM-X60T-3, before any guard game):**
  - the guards are produced and written to a filled artefact;
  - the read is re-decided in a new state, whose spec imports every played cell from the look-1, look-2 and X61P
    states and plays one unit: the determinism guard's (§4's c0_s3501 forum72 unit 0, 20 games), since rule `x60`'s
    guard compares a unit played in its own state (`determinism_guard.group` takes no import; REVIEW AM-2 m3).
  - Its producing commands are AM-X60T-3's.

### 23.5 `ge_x60t`, §11's fresh G-E reader (FINDING 19)

- **The arm:** `kit.ARMS["ge_x60t"]`. It reads `rb_x60t_s3501`'s final, as `ge_c0` reads `rb_x35_c0_s3501`'s. The run is
  `rb_x35_ge_x60t_s3501`; the reader run name's tag rule was fixed for a target named outside `rb_x35_`.
- **Its argv is `ge_c0`'s plus `--human-data off`,** with the target and the run name changed (`ge_test`).
- **Its pin is X60-T's own, `7fe15726c`** (`batch.ARM_PINS`, amendment AM-X60T-2), not the side pin:
  - Main's launcher refuses a fork of a human-data run that types no clause.
  - The side pin's parser has no `--human-data` flag. `checkargs --pin f4b25238d` refuses the typed argv:
    "NOT IN PINNED TREE".
  - At `7fe15726c` with `off`, the training path is the side pin's byte for byte: §14 check 2 (learner golden and
    played-bits fixtures identical at `440b61c3` and `f4b25238`) and check 3 (`@0` ≡ `off` team streams). So it
    best-responds in `ge_c0`'s regime, at the same budget (+5M), dose (`--fork-lr 5.6e-05 --fork-lr-freeze`) and seed
    (3531). `ge.r2_input` checks budget, dose and regime matched, as for every reader.
- **Its trigger** (`ge.trigger`, `kit stage-check`): x60t's Δ_F FINAL class GAIN, from the stage record's `delta_f`
  (§23.1). `ge_c0` is also ALLOWED by an x60t GAIN. Now both read PENDING (look 1 CONTINUE is not final).
- **The launch:** `launch_x35.sh ge_x60t 3501 --pin 7fe15726c77cfa823aee78110444bf25c3c7370e --lease-owner <owner> GO`,
  and `launch_x35.sh ge_c0 3501 --pin f4b25238def57900c0305bb2a3a5a8f9edaef285 --lease-owner <owner> GO` for its
  b-OFF reader, each only once `kit stage-check` reads ALLOWED. A side-pin `ge_x60t` attempt HOLDs at pin-check.
- **Validated on the REAL target** (CPU; FINDING 27):
  - `checkargs --pin 7fe15726c` rc 0, human data "off (no human data)".
  - `launcher --dry-run` refused ONLY by the CUDA-visibility preflight (this shell has no GPU). It reads FORK of
    `rb_x60t_s3501`, +5,000,000 steps, `self_play False (TYPED)`.
  - `kit pin-check --arm ge_x60t` and `corpus-check` PASS.
- **Cost on a GAIN only:** ≈ 2.1 GPU-h for the two readers (5M steps each, ≈ 1.03 GPU-h at §17's 4.10 per 20M).

### 23.6 The launches (validated now; each GO is the orchestrator's)

Through `launch_x35.sh` (`x35_breakers_stage_2026-10-17/`), from the Training session that holds the lease.

| stage | command | argv (kit `describe`) |
|---|---|---|
| X60Tb | `launch_x35.sh x60t <s> --pin 7fe15726c77cfa823aee78110444bf25c3c7370e --lease-owner <owner> GO`, s = 3504 … 3508 | `--model models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --human-data teams:hr_spec_v1.r3@0.5 --seed <s> --run-name rb_x60t_s<s>` |
| X60T-C0b | `launch_x35.sh c0 <s> --pin f4b25238def57900c0305bb2a3a5a8f9edaef285 --lease-owner <owner> GO`, s = 3506 … 3508 | the same less the clause, `--run-name rb_x35_c0_s<s>` |

**Checks run here (CPU, `CUDA_VISIBLE_DEVICES=""`), all eight argvs:**
- `checkargs --pin` rc 0.
- `launcher --dry-run` refused ONLY by the CUDA-visibility preflight. Each reads FORK of `rb_x26_s2601`, +20,000,000
  steps, 12.95 GiB REQUIRED.
- `kit pin-check` PASS (x60t at `7fe15726c` with `corpus-check` PASS; c0 at the side pin); `kit stage-check` ALLOWED.

The script's own pre-checks 1, 1b and 6 (no live run, the lease, the dry-run with the card visible) run at GO.

### 23.7 How look 2 is played and read (the operator: Training Oct)

The registered state is **`~/gen3ai_archive/x60t_read/look2`**.
1. **Plan and run in the playing checkout** (`~/dev/gen3ai-reads/x60t_play_de15dfc7`, `export PYTHONPATH=$PWD/src`):
   - `python -m main.reads plan --spec <main>/designs/research_state/measurements/x60t_h_teams_2026-11-06/read_spec_look2.json --state ~/gen3ai_archive/x60t_read/look2`
   - the anchor lane: `python -m main.reads run --state ~/gen3ai_archive/x60t_read/look2 --lane anchor --hours 150 --concurrency 2`, detached;
   - the S lane: `python -m main.reads run --state ~/gen3ai_archive/x60t_read/look2 --lane s --hours 150`, detached.
2. **Plan may run before the finals exist.** The 13 existing finals' cells (ON s3501–s3503's home72 / p72 / H,
   the determinism unit, D1, and their S nodes) can play at once, and the eight new finals wait for their settle.
3. **Read from a main checkout,** after every unit and S node is played and the eight new runs' AM-6 K9(b) records
   exist: `python -m main.reads assemble|decide|record --state ~/gen3ai_archive/x60t_read/look2`, then `record --out
   <stage>/reads/x60t_look2.json`. Then §15's launch-log lines and the contamination check, by hand.

### 23.8 Validated by executing (CPU, no game)

- **`check-spec`** exit 0 at main (`rederived_guards: count`, `choices_guard: require`) and at the play pin.
- **`plan` at the play pin into a scratch state** (`~/gen3ai_archive/x60t_read/scratch_am2_look2_plan_dryrun`, not the
  registered path; removed after the review, 2.5 GB of copied finals and R members): exit 0, spec sha256 `a5ad7411…`.
  - The 13 existing finals were copied, each matching its sha256.
  - Every import passed `imports.check`.
  - It planned 1 determinism unit, 7 units each for ON s3501–s3503's home72 and p72, 30 for their H, and 60 for each
    D1 final.
  - The eight new finals read `waiting` ("no final yet").
- **Main's `assemble` of that state:** exit 0. All forum72 groups are on E3 (`port=0d3d7629f16a0eaf/91;env=393631b886965b0c/126`),
  sides `their=legacy` / `eval`, group regime `00c8c0114bdf0836`.
  - `primary` imports ON's 180 look-1 units; `primary_c0a` 180, `primary_c0b` 120; `reported_c0a` 63 and
    `reported_c0b` 42 at `3d2cbdc0b1413bcb` (§4.3's reported id).
  - S `p_c0a` / `p_c0b` hold 210 / 140 rows at `d4de0fd7c16a2658`.
  - So every import pools as ONE cell before any game.

### 23.9 Cost

| | GPU-h | Kakuna games (lane-h at 25 s) | in-house sets (lane-h at 2.45) |
|---|---:|---:|---:|
| X60Tb (5 × ON) + X60T-C0b (3 × C0), §17 | 20.5 + 12.3 = **32.8** | | |
| T1: F 9,600 (ON s3504–s3508, C0 s3506–s3508) + determinism 20 + goal (a) home72 1,120 (ON × 8) | | 10,740 (74.6) | P 11 (27.0) |
| T2: I_F 16 + R72 16 | | | 32 (78.4) |
| T3: P72 1,540 + H 840 + C0 s3506–s3508 home72 420 | | 2,800 (19.4) | |
| T4: ON H 0–29 3,000 + D1 H + F 6,000 | | 9,000 (62.5) | |
| **look 2** | **32.8** (+ 2.1 on a GAIN) | **22,540 (156.5)** | **43 (105.4)** |

- Against §17: −3,600 Kakuna games (ON s3501–s3503's F imported) and +420 (ON H 0–6 at s3506–s3508, which §17's
  plan omitted; FINDING 25).
- **Wall:** the anchor lane runs 66–96 h at X61P's 234–344 games per wall hour. The S lane is ONE lane per state, so
  ≈ 105 h at 2.45 lane-h a set, and it is the long pole. §17's "39 on 2 S lanes" assumed two (FINDING 26). Both
  lanes take heavy slots, so the S′-HONEST read sharing the box stretches both.

### 23.10 Review (REVIEW AM-2)

An independent opus-high reviewer read the branch (code, spec, docs, the validation transcripts and the scratch state)
and re-ran the tests, `check-spec` at main and the pin, the three regime ids at the pin, main's VOID gate on the 13
existing finals (all PASS), `kit pin-check` (x60t, c0, ge_x60t, ge_c0, d3) and `stage-check`. **Verdict:
READY-WITH-FIXES**; no verdict, class, threshold or seed set changes.

| finding | fix |
|---|---|
| M1: the stage record, made from the worktree, names worktree k9b paths, so its own test failed and §23.1's identity claim was false | re-recorded from the main checkout after the code shipped (two ships), as the brief's `record --out` from the same state; §23.1 states what the re-record writes in the state. The reviewer's alternative (a pure transform of the shipped record) was not taken: the brief asks for `record --out` |
| m2: the trial re-record's write to the look-1 state was undisclosed | §23.1 records it and the restore |
| m3: a "plays nothing" re-decision cannot carry the determinism guard | §23.4: it re-plays the one guard unit |
| m4: the record holds `h_contrast` over 30 units only | §23.3 choice 4: pooled from `cells.json` unit rows |
| m5: nothing at launch time re-derives the stage keys | `batch.read_record` runs `record.check` on a `main.reads` record; a test edits `delta_f` and is refused |
| m6: no `ge_x60t` launch line; its pin-check line named AM-X60T | the line in §23.5 and `launch_x35.sh`'s header; the check's lines name AM-X60T-2 |
| F7: `ge_x60t`'s exploiter-mode equivalence across the two pins is unverified | FINDING 27 says so |
| m8: the scratch state was not removed | removed |
| m9: `design_beat_metamon_path.md` and `human_data_architecture.md` still said X60-T was not ready / not launched | updated |

### 23.11 FINDINGS (this amendment)

24. **C0 s3506–s3508 are read by the `final` rule, not pinned by sha256** (§23.3 choice 2): a departure from §21.6's
    list, forced by registering before they train. The settled file's sha256 is fixed at settle.
25. **§17's plan omitted ON's H 0–6 at s3506–s3508** (420 games; §12 needs them for Δ_H).
26. **One S lane per state,** so the 43 in-house sets take ≈ 105 h of wall, not §17's 39 h on two lanes.
27. **`ge_x60t` runs at X60-T's pin, not the side pin** (§23.5), with its off-path equivalence resting on §14's checks
    2–3 at that pin. **Not verified:** those checks ran in self-play mode, not in the readers' exploiter mode
    (`--exploiter --no-self-play --trainee-teams`), and `ge.r2_input`'s `check_matched` compares budget, dose and regime,
    never the two pins (REVIEW AM-2 F7). Its dry-run's recipe surface matches `ge_c0`'s validated transcript
    (`n_epochs` 10, `self_play` False). A CPU `--debug` exploiter comparison at both pins would close it; it is training,
    so it was not run here. No reader transcript is committed in `ge_readers_validation_*` (as for `ge_d3` / `ge_n1p`);
    the launcher dry-run with the card visible is the lease holder's, at the reader's GO.
28. **Δ_H / Δ_P72 / Δ_home72 are per-group rows,** so the pooled reported Δs are summed by hand (§23.3 choice 4).
29. **The guards stub makes a passing GAIN read INCONCLUSIVE at look 2** until the AM-X60T-3 re-decision (§23.4). That
    is §11's registered INCONCLUSIVE path, not a new class.
