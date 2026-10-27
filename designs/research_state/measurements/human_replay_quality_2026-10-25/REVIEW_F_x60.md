# REVIEW F: adversarial review of X60's redesign (e1512c36, 2026-10-26)

**Under review (commit `e1512c36`, "X60 REDESIGNED, pending review"):**
- [`design_human_replay_lever.md`](../../../endstate/design_human_replay_lever.md) §6, "the design";
- [`x60_redesign_2026-10-26/power.py`](../x60_redesign_2026-10-26/power.py) and its `results/power.json`;
- [`human_data_architecture.md`](../../../endstate/human_data_architecture.md): Q-H11, Q-H13 and §9 item 6.

**Context read:** REVIEWS A–E and [`RESPONSE.md`](RESPONSE.md); the X59-S PREREG with Amendments 1, 2, 2b and 3, and
`src/main/human_data/x59s_read.py`; D3's look 1 (`58287c52`, `x41_d3_setmut_2026-10-25/read/`); the X61P / N1-P
PREREG and its read spec; the belief-prior-mix README; X53-N1's design and X61's PREREG §10; `heldout_panels.json` and
`src/main/heldout_panels.py`; `src/main/reads/assemble.py`; the r1 / r2 rows manifests in `~/gen3ai_archive/x59s/`.

**Method.** CPU only, at nice 19 on cores 5-7,13-15. No GPU, game, training or launch. Read-only on `models/`, the
archive and the panels; the sealed `forum216b` was not read. One script sits beside this file:
[`review_f_checks.py`](review_f_checks.py), with its output in `results/review_f_checks.json`. It re-implements §6.3's
rule from the design's TEXT, not from `power.py`, so the two would not share a bug. The design was not edited.

**Abbreviations.** P\* = X26's plateau; C0 = X35's plain-continuation control (`rb_x35_c0_s3501`–`s3505`, pin
`f4b25238`); F / H / P72 = `forum72` / `heldout72_hpfix` / `p72`; I_F = the in-house meter on F; R72 = the positive
control (72 of the lever's own teams); S = X35's population score against the frozen reference population R; σ = the
seed-to-seed SD of a final's F rate; pp = percentage points; lane-h = one CPU lane for an hour.

## Verdict: READY-WITH-FIXES

The estimator is now one rule, simulated as registered. I reproduced every key cell of `power.py` independently,
within ±0.006 (§3). The futility-only look 1 is sound: a binding futility stop can only lower the false-GAIN rate.
The side pin is the right choice (§2, F-M3), and so are F alone (X61P's precedent) and P-BC on fresh games. The
encoder identity of r2's rows matches the pin.

There is no BLOCKER, but nine MAJORs. Four of them must be fixed before any X60 build starts:
- **F-M1** (the routing can be gamed after the outcome);
- **F-M2** (the eval engine drifts under the imported control);
- **F-M3** (the corpus is not checked at the side pin);
- **F-M7** (one corpus, two entry points).

The other five must be fixed before the firing route's PREREG registers:
- **F-M4** (the power headline);
- **F-M5** (X60-B's delivery logic);
- **F-M6** (the KAKUNA-SPECIFIC guard);
- **F-M8** (cost);
- **F-M9** (X60-B's teacher).

None of them needs a redesign.

## 1. Findings at a glance

| id | sev | one line |
|---|---|---|
| F-M1 | MAJOR | The routing is mechanical in code, but the firing PREREG is written AFTER X59-S's outcome with open placeholders. It routes on prose strings, `verdict()` has no registered entry point, and two outcomes have no route: no verdict at all, and the ≥ 1500 follow-up. |
| F-M2 | MAJOR | Kakuna games run on the in-process Rust sim. The regime id omits the engine binary, so C0's imported F cells and ON's games can sit on different engines; 10-26 alone landed OU-visible parity changes. |
| F-M3 | MAJOR | The side pin is right, but the H-TEAMS corpus is not required to pass the side pin's soak and pack-identity check. At `f4b25238` the packer mis-packs Frustration and HP IV lines, and the AM-D1-FIX label guard is absent. |
| F-M4 | MAJOR | "GAIN 0.80 at +3 pp at the measured noise" is the σ = 0 cell. It is 0.70 at the design's own 1.0 pp floor and 0.27 at the measured upper bound of 2.75 pp. A free re-measurement is about to exist. |
| F-M5 | MAJOR | R72 is not a positive control for H-BC. X60-B's most damning null (labels learned, play unchanged) lands in NOT-DELIVERED, "closes nothing". NO-TRANSFER pairs an in-house read with a Kakuna read. |
| F-M6 | MAJOR | The KAKUNA-SPECIFIC guard (I_F point > 0) passes a pure Kakuna-only gain 50 % of the time. Its cost to true gains depends on an unmeasured exchange rate between the meters. End-to-end ADOPT power was never computed. |
| F-M7 | MAJOR | `n1_hrep_v1` has two entry points. A `--team-mix` run trains on human data while its provenance reads "off", and it can launch before the switch exists. The two corpus specs also differ. |
| F-M8 | MAJOR | The lane-h figures assume 10 s a Kakuna game. D3 measured 27–93 s on the contended box, so X60-T's Kakuna games alone are ≈ 70–95 h of wall time in an already full queue. |
| F-M9 | MAJOR | X60-B trains on r2: 77.5 % of its rows come from 1300–1499 players, a population whose best models all lose to P\*. The registered ≥ 1500 follow-up is ignored, and X60-T's ≥ 1500 rationale is not applied to X60-B. |
| m1–m11 | MINOR | §5 |

## 2. MAJORs

### F-M1 (MAJOR): the routing is mechanical in code, but not end to end

**Evidence.**
- **What holds.** `x59s_read.verdict` (lines 145–182) cannot emit the closure together with "H-BC first": the closure
  requires `lab != "HUMAN-LABELS-BETTER"`. KAKUNA-SPECIFIC and UNINFORMATIVE fall to the "anything else" row as §6.1
  says, so the table matches the code.
- **(a) Post-outcome freedom.** §6 says that after this review "the route X59-S picks is committed as a PREREG … before
  its arm launches". That PREREG is written after X59-S's numbers are known, and §6 leaves these open:
  - the coefficient grid and the smoke's batch count (§6.5);
  - r<n>'s exact filter (§6.4: "≈ 2,000–3,000 teams, ESTIMATE");
  - the BC "human rows per update" ("fixed in the PREREG");
  - the read specs.
  Each of them can be tuned after the outcome.
- **(b) Routing on prose.** §6.1 keys the route on substrings of human-readable action strings ("closes 1300+ ladder
  data at this mix and volume", "H-BC first (against P\*-restricted-to-human-support)"). Rewording a string, or fixing
  a typo in one, silently re-routes X60.
- **(c) No registered entry point.** `python -m main.human_data.x59s_read` prints only `read(arms)`. Nothing that
  ships calls `verdict()` except its tests. Its six inputs are passed by hand: `fr2` (a string), `sp_yard` / `sh_yard`
  (NLL arrays), `inhouse_`, `floor_rates`, and `read_["floor"]`.
- **(d) Two outcomes with no route.**
  - X59-S can end with no verdict: a VOID student, FR-2 never read (`verdict` raises `ValueError`), or the window
    abandoned. §6.1 has no row for that.
  - A2.7's ≥ 1500 follow-up fires whenever Δ_data is neither HUMAN-DATA-BETTER nor UNINFORMATIVE, including
    alongside "H-BC first". X60 neither waits for it nor reads it (see F-M9).

**Fix.**
1. Freeze §6 by sha256 now. The route's PREREG may differ from §6 only in a listed set of placeholders, each with its
   rule: r<n>'s id and digests, c from the smoke, the shas.
2. Route on structured classes. Write a pure `x60_route(verdict) -> "B" | "T" | None`, with a test over every class
   combination (including the closure + UNCLEAR co-occurrence).
3. Register a read spec that computes `verdict()` from sha-pinned artefacts: FR-2's record, the yardstick files, and
   the in-house reads.
4. Add the rows "no verdict ⇒ no X60" and "the follow-up fired ⇒ X60-B's rowset per F-M9".

### F-M2 (MAJOR): the eval engine drifts under the imported C0 cells

**Evidence.**
- **Kakuna games run on our engine.** D3's cells stamp
  `server_version: "ws_frontend[in-process]@2cc864f2f+rust:cargo:src/rust_sim"` on all 70 records: the in-process
  Rust sim at the harness commit, not a pinned Showdown.
- **The regime id cannot see the engine.** `main.reads.assemble.ANCHOR_REGIME_KEYS` carries `server_impl` and
  `showdown_pin`, but not `server_version` or the binary. So "regime id matches" (§6.3) does not imply the same
  engine.
- **The engine has moved.** Since D1's banked rows, main has landed BF (`706faa0e1`: sixteen divergence classes,
  "OU-visible"), B1–B10 (moves 325 → 364 of 369; the recharge family now recharges), and the King's Rock / Beat Up and
  forced-Struggle fixes (`8dd36e9db`, `995c04a8a`), all on 10-26.
- **The guard is weak.** X60's ON games will be played weeks after X61P's C0 cells. D3's determinism guard re-plays
  ONE cell (20 games), and most of these classes carry ≤ 0.01 % of fielded mons, so the guard passes whether or not
  the engine changed.
- **The bias falls on ON alone.** Both arms trained on `f4b25238`'s engine. Evaluating ON on a newer engine than C0
  charges a train/eval mismatch to ON alone.
- **The design's check sits in the wrong place.** The sim binary hash is in the C0 reuse check, and its remedy is "a
  fresh C0h (5 × ≈ 4.1 GPU-h)". An eval-engine mismatch needs a CPU re-play, not new GPU training.

**Fix.**
1. Play every X60 Kakuna and in-house game, ON and any C0 top-up, at ONE eval harness commit: the commit X61P's C0
   cells were played at, held in a pinned worktree. Otherwise re-play C0's F at X60's harness (≈ 5,300 games).
2. Add the transport identity (`server_version`, or the sim binary's sha) to the import check in `main.reads.imports`.
3. Split the reuse check's remedy: a training-side miss ⇒ C0h; an eval-side miss ⇒ re-play.
4. Do the same for the `main.h2h` regime behind I_F, Δ_P and R72.

### F-M3 (MAJOR): the side pin is right; the corpus is not checked at it

**Evidence.**
- **The premise to correct.** The design does NOT launch ON at a later pin. §6.3 registers `f4b25238` + the switch
  commits (N1-P's pattern). Moving to main's later pin would add the `d8d674b9` GIGO fixes as a second lever:
  - an obs change (golden 36 / 991);
  - op config v162;
  - an intent-label change.
  X61P's PREREG §2 says the same. Reusing C0 at the side pin is sound **if** the arm differs only by the lever.
- **r2's rows are consistent with the pin (verified).** The parts were extracted at `da36173c`, whose
  `golden_obs_fixture.json` is `9a6f947b…`, the same bytes as `f4b25238`'s (main's is now `95e24c32…`). There is no
  diff in `src/rust_env`, `src/rust_sim/src`, `src/agents/battle` or `src/agents/observation` between the two, and
  `ARCH_SIGNATURE` is equal. So `bc:hr_spec_v1.r2` passes VOID check 2 at the side pin and means the same thing there.
- **What breaks: the H-TEAMS corpus.**
  - It is to be built once (as `n1_hrep_v1`) with main's builder: C-GAME, HP repair, and the post-`d8d674b9` pack
    oracle.
  - At `f4b25238` the packer mis-packs happiness-0 Frustration (BP 1; 13 pool mons) and HP type-line IVs (45 pool
    mons). That is `d8d674b9`'s C-2 / C-3, the HP class behind the retired `heldout72`.
  - At `f4b25238` the sim also lacks every 10-26 parity fix and moves 325 → 364.
  - Ladder teams carry these mechanics at their own rates, not the pool's. ON would meet mis-simulated or mis-packed
    teams that C0 rarely meets: an asymmetric input defect that is not a common cause of both arms.
- **N1-P handled exactly this; §6.4 does not.** N1-P soaked at the pin's env build, ran the pack oracle over its
  corpus, and ran `kit corpus-check` at the pin. §6.4 says only "label-path soaked".
- **The label guard is missing at the pin.** X61P FINDING 3: a corpus at `f4b25238` "would load unguarded" by the
  AM-D1-FIX label guard. Human teams are the riskiest corpus for it.
- **"Switch commits only" is a large backport.** It is `src/main/human_data/`, `agents.info_levers.human_data`, the
  learner fold and the corpus files, all authored on main after `d8d674b9`.

**Fix.**
1. The corpus passes, AT THE SIDE PIN: the pin's soak, `corpus-check`, and a pack-identity check (main's pack oracle
   against the pin's packer, team for team). Teams that differ are dropped, and their share is reported. The same
   applies at N1H's pin (F-M7).
2. Report the corpus mons that touch any 10-26 class.
3. The side pin carries the AM-D1-FIX guard.
4. `pin-check` enumerates the backported paths. The switch's off-means-off and null-dose gates run AT the pin.

### F-M4 (MAJOR): the power headline is the σ = 0 cell

**Evidence (`review_f_checks.json`, 60,000 draws a cell; matches `power.json`):**

| σ | GAIN at +3 pp | false GAIN | NO-USEFUL-GAIN at a true +3 pp |
|---:|---:|---:|---:|
| 0 | 0.80 | 0.007 | 0.011 |
| 1.0 pp (the registered floor) | **0.70** | 0.020 | 0.028 |
| 2.0 pp | 0.43 | 0.034 | 0.050 |
| 2.75 pp (the measured one-sided upper bound) | **0.27** | 0.029 | 0.048 |

- **Where the 0.80 appears.** §0, §6.0, the always-current Q-H11 and the commit subject all quote "0.80 at the
  measured noise".
- **Why "0" is weak.** The point estimate is 0 because φ̂ = 0.56 (under-dispersed). At 140 games a final, the
  binomial SD (4.22 pp) swamps any seed SD below ≈ 3 pp.
- **The 8-seed rejection.** It is argued at "+0.16 power". That is the σ = 0 gain; at σ 1.5 pp it is +0.21 (0.57 → 0.78).
- **A better measurement is free and imminent.** X61P's read plays C0 (and N1-P) at 60 F units a final, which gives
  a binomial SD of 1.44 pp. With 6–10 such finals, σ ≥ 1.5 pp becomes resolvable, and the CPU is already being spent.

**Fix.**
1. Quote the power as a range over [0, 2.75] pp, with the floor's cell (0.70) as the headline, in §0, §6.0 and Q-H11.
2. Re-estimate σ from X61P's 1,200-game C0 / N1-P F cells before the PREREG.
3. Pre-register, before ON trains: if that σ̂'s one-sided upper bound exceeds 1.5 pp, the arm runs 8 seeds. This is
   decided on control-only data, so it is not an adaptive look.

### F-M5 (MAJOR): R72 is not a positive control for H-BC, and the verdict table inverts X60-B's null

**Evidence.**
- **What R72 is in each route.**
  - In X60-T, R72's teams are in ON's pool, so ON pilots them: a delivery check.
  - In X60-B, ON never pilots R72 in self-play. Each R72 side is ≤ 1 battle (≈ 28 decisions) of r2's 1,201,538
    TRAIN rows. R72 then asks "did BC move in-house play on LADDER-population teams?", which is a transfer question.
- **The inversion.** DELIVERED needs R72 AND the label check (§6.5). So the most damning null for H-BC lands in
  NOT-DELIVERED, "closes nothing" (§6.6): the label check passes (BC learned human choices), while R72 and Δ_F are
  flat (play unchanged anywhere). Under a true null, NO-USEFUL-GAIN is 0.67–0.79 of outcomes, so X60-B's main null
  cannot stop H-BC.
- **Mismatched meters.** NO-TRANSFER pairs R72 (in-house, practised) with Δ_F (Kakuna, unfamiliar). The like-for-like
  contrast is R72 against I_F, both in-house. An I_F gain with a flat Δ_F is not "no transfer".
- **The futility path is underpowered.** Look 1's futility stop (≈ 50 % of nulls) reads R72 at 3 v 3. Its Welch MDE
  is 1.12 / 1.82 / 2.61 pp at the per-final S SD of 0.37 / 0.60 / 0.86 pp, against 0.75 / 1.21 / 1.74 at 5 v 5. The
  expected R72 effect (the draws per R72 team in 20M steps) is not estimated, so P(DELIVERED | a delivered lever) is
  unknown.

**Fix.**
1. X60-B: DELIVERED = the label check with a registered minimum (for example, ON closes ≥ 10 % of P\*'s NLL gap to a
   `hom_<r2>` on the account-disjoint validation rows). R72 is reported only.
2. NO-TRANSFER also requires I_F not GAIN.
3. X60-T: state R72's per-team draw dose and its MDE on the path it is read (3 v 3 or 5 v 5). If R72 cannot reach its
   MDE at 3 v 3, a look-1 stop reports NO-USEFUL-GAIN without the NO-TRANSFER / NOT-DELIVERED split.

### F-M6 (MAJOR): the KAKUNA-SPECIFIC guard is a coin flip against what it guards

**Evidence.**
- **It leaks half of what it guards against.** With the I_F rule "point ≤ 0 ⇒ KAKUNA-SPECIFIC", a gain that only
  Kakuna sees (true in-house 0) passes 50 % of the time.
- **It also costs true gains, by an unknown amount.** A true gain is mislabelled KAKUNA-SPECIFIC with probability
  P(I_F point ≤ 0) (`review_f_checks.json`, 5 v 5):

  | true in-house gain | per-final S SD 0.37 pp | 0.60 pp | 0.86 pp |
  |---:|---:|---:|---:|
  | +0.25 pp | 0.14 | 0.26 | 0.32 |
  | +0.5 pp | 0.02 | 0.09 | 0.18 |

  The in-house gain that a true +3 pp on Kakuna F implies (the exchange rate between the meters) is unmeasured, so
  neither error rate is known.
- **The 0.80 is not end to end.** It is Δ_F GAIN alone. P(ADOPT-CANDIDATE), which needs GAIN, I_F > 0, Δ_P
  non-inferior, goal (a) and three guards, is not simulated.
- **The doc overstates the guard.** REVIEW D M6 proposed the point rule. But the always-current Q-H11 now says a gain
  "needs the in-house meter (never trained on human data) to agree", and a point estimate does not deliver that.

**Fix.**
1. Estimate the exchange rate from arms that already have both meters: D1, D3, X26c, X61P once read, and the
   belief-prior arms.
2. Register the guard's two error rates at that rate, and simulate end-to-end ADOPT power.
3. If the rate allows, use I_F's one-sided 90 % LB > 0. Otherwise state the 50 % leak in Q-H11 and §6.10.

### F-M7 (MAJOR): one corpus, two entry points, two specs

**Evidence.**
- **Two entry points.** `n1_hrep_v1` is reached by `--team-mix n1_hrep_v1` (X53-N1's N1H; X61 PREREG §10) and by
  `--human-data teams:…` (X60-T). The arch doc leaves this OPEN ("before either arm launches"), with no mechanism.
- **Provenance breaks.** A `--team-mix` run trains on human data while the switch's `record()` reads `off`. That
  breaks Q-H14 (provenance) and the audit trail of the owner's "toggleable, OFF by default" grant.
- **N1H can launch first.**
  - N1H runs on N1-P's GO when X59-S does not close human data.
  - The switch is built "only once a first read routes a lever to a build" (arch §9 item 2).
  - Nothing orders the two.
- **The specs differ.**
  - X60-T (§6.4): r2's kept battles (the Kakuna-family audit's drop, the ≥ 11-turn floor), ≤ 5 teams per account,
    the ≥ 1500 full-six seats.
  - X61 §10: "ONE registered snapshot (`hr_spec_v1` or `hr_arch_2026-10-24`)", with no account cap and no audit drop.
  - "Built once under one id" is therefore not yet one spec.
- **Different pins.** X53-N1 pins at main, X60-T at `f4b25238` (F-M3), so the shared bytes are packed and simulated
  differently by the two arms.

**Fix.**
1. Add a throwing guard now, before either build: `--team-mix` refuses any corpus whose manifest says
   `"human_data": true` unless the argv carries the matching `--human-data teams:` clause, or the switch is the only
   way in. Add a test that fails on revert.
2. Reconcile X61 §10 with §6.4 into one spec (snapshot, account cap, audit drop) before the build.
3. Record both pins' soak results in the corpus manifest.

### F-M8 (MAJOR): the CPU cost is quoted at 3–9× the measured rate

**Evidence.**
- **The assumed rate.** §6.9 prices Kakuna at ≈ 10 s a game a lane.
- **The measured rate.** D3's look 1 held 20-game units at 9–31 min each (27–93 s a game), ≈ 11.7 h for its 2,520
  games (D3 read README, FINDING 5).
- **The volume.** X60-T plays ≈ 15,000–20,300 Kakuna games:
  - ON: 10,400;
  - D1's units 7–29: 4,600;
  - C0's F, if not imported: ≤ 5,300.
- **The wall time.** At D3's realised ≈ 215 games an hour, that is ≈ 70–95 h of wall for Kakuna alone, plus 25
  in-house sets.
- **The queue it joins.** X61P's look 1 (9,720 games) and look 2, X59-S (≈ 80 lane-h, at the same 10 s assumption),
  and D3's look 2. "Wall time is set by the CPU queue" is stated, but never quantified.

**Fix.**
1. Cost at the measured per-game rate.
2. Give the wall estimate and the queue position.
3. Rank the reads so a trimmed set still decides:
   - the primary and Δ_P first;
   - then I_F and R72;
   - the T − D1 contrast's 7,600 games last; it decides nothing for adoption.

### F-M9 (MAJOR): X60-B imitates a teacher weaker than the student

**Evidence.**
- **The teacher population.** r2 is 77.5 % 1300–1499 actors (PREREG Amendment 2b), and every human-only model
  upstream loses to P\*: U 0.63–0.79 for P\* (Q-H15).
- **What X59-S can and cannot show.** Its HUMAN-LABELS-BETTER compares from-scratch students. It shows that human
  labels teach a fresh network more than P\*-restricted labels do, not that they improve P\*.
- **The guards are in-pool.** Δ_P (non-inferiority at −2 pp) and goal (a) watch in-pool cost. On the primary, a pull
  toward 1300-level play shows only as a smaller Δ_F.
- **The design argues ≥ 1500 for one route only.** §6.4 restricts X60-T to ≥ 1500 seats ("fewer invented species");
  X60-B gets no rating argument at all.
- **The registered rating read is ignored.** A2.7's follow-up (S-H≥1500 vs S-H-mix at matched rows; 270,260 ≥ 1500
  TRAIN rows) is registered to answer exactly this, and X60 ignores it (F-M1 d).

**Fix.** Pre-write both X60-B rowsets:
- r2;
- r2's ≥ 1500-actor rows.
Route between them mechanically on A2.7's follow-up when it fires: ≥ 1500 iff S-H≥1500 − S-H-mix is not
OWN-worse. When the follow-up does not fire, use r2 and state the teacher-strength limit in §6.10. The
coefficient smoke runs on the chosen rowset.

## 3. Independent recomputation (`review_f_checks.py`)

- **The primary.** Every cell in F-M4 agrees with `power.json` within ±0.006: false GAIN 0.008 / 0.020 / 0.033 at
  σ 0 / 1 / 2; GAIN at +3 pp 0.80 / 0.71 / 0.43.
- **The half-widths, analytically.**
  - Look 2: √(2 · 0.49 · 0.51 / 6,000 + 2 · 0.01² / 5) = 1.11 pp, so ± 2.18 pp, as quoted.
  - Look 1: √(2 · 0.2499 / 3,600 + 2 · 0.01² / 3) = 1.43 pp, so ± 2.81 pp.
- **The futility look.** It stops 50 % of nulls and 0.5 % / 1.8 % / 6.8 % of true +3 pp levers (σ 0 / 1 / 2).
- **The Welch guard.** `W · df / χ²₀.₉₅(df) − b` is a correct one-sided lower bound with b treated as known. The
  pooled σ̂ assumes equal seed SDs in both arms; the guard covers gross inequality only (m-level; see m8).
- **The T − D1 contrast's path.** See m1.

## 4. What holds (checked, not just read)

- **The routing table matches `verdict()`.** The closure and "H-BC first" are mutually exclusive in code.
- **The side pin.** Its rationale and r2's encoder identity match the pin (F-M3).
- **The contamination VOID is safe from R72.** `contamination()` checks `held_out_only` panels, so a PRACTISED R72
  registered with `held_out_from: []` does not trip it (m5 covers the builder side).
- **X61P really plays C0 at 60 F units** (`read_spec_look1.json`, unit seeds 0–59). Its Δ_P re-plays C0 at regime
  `d4de0fd7c16a2658`, so X60 can import C0's P rows from it. Note that D1's banked P rows carry no regime id and
  cannot be imported.
- **"Pairing credited nothing" holds.** It is supported by the belief-prior-mix FINDING 1: the paired half-width,
  4.24 pp, is wider than the unpaired one, 3.99 pp.

## 5. MINORs

- **m1. The T − D1 contrast's seed count is chosen by the primary.** §6.4 says "3 v 3 if look 1 ended it", and T's F
  units 0–29 are nested in the primary's units 0–59. Under the global null the false rates by path are:

  | σ | FORUM-BETTER, stopped at look 1 | FORUM-BETTER, continued | LADDER-BETTER, stopped | LADDER-BETTER, continued |
  |---:|---:|---:|---:|---:|
  | 1 pp | 0.030 | 0.009 | 0.008 | 0.027 |
  | 2 pp | 0.066 | 0.016 | 0.016 | 0.049 |

  The overall rates are near nominal, and the 3 v 3 path is not powered. **Fix:** always read D1 at 5 seeds (they
  exist), and report the conditional rates.
- **m2. NO-USEFUL-GAIN at a true +3 pp is 2.8–5.0 % at σ 1–2 pp**, above the nominal 2.5 %. So a NO-TRANSFER stop
  can fall on a useful lever about 3–5 % of the time. Report it in §6.7.
- **m3. "Each arm owns one panel" makes T − D1 a comparison of home advantages.** LADDER-BETTER means "H-TEAMS's gain
  on its own population plus its F transfer exceeds D1's on F plus its H transfer". It does not mean "ladder data
  beats forum-team breadth". Reword §6.6's attribution accordingly.
- **m4. Q-H13 (always-current) says "Every panel is TEAM-clean once the ≥ 3-fit exclusion runs".**
  - The exclusion counts revealed facts before the fill.
  - The post-fill net is the registry's ≥ 4 shared sets (`near_neighbour.min_shared_mon_sets = 4`).
  - C-GAME fills popular Smogon sets into mons with no revealed move, so they can recreate a 3-set core.
  - REVIEW D FINDING 1 checked move sets only.
  **Fix:** run the post-fill check at ≥ 3 on the H-TEAMS corpus, or qualify the sentence.
- **m5. R72's registration order.**
  - Builders differ: `build_w_forum` excludes every panel, in-pool panels included (`panel_names()`), while
    `build_n1_proc` / `build_d3_setmut` exclude held-out panels only.
  - If H-TEAMS's builder takes "D1's filters" and R72 is registered before a (re)build, R72 and its near neighbours
    leave the corpus.
  - **Fix:** specify the build → draw → register order, `held_out_from: []`, and a test.
- **m6. C0's F cells are known before X60 registers** (X61P's read). §6 is committed now (`e1512c36`). The PREREG
  must forbid changing the floor, the units or the classes after X61P's C0 cells are read (F-M1's freeze covers this).
- **m7. One control is shared by D3, X61P and X60.** A lucky-low C0 inflates all three, so their GAINs are not
  independent replications. Say so wherever two are cited together.
- **m8. The null-dose bit check for H-BC (c = 0) will fail if the BC loader draws from the shared RNG.** The
  contingency, a fresh C0h (+≈ 20.5 GPU-h), is not in §6.9's table. Give the loader its own generator and test the
  null dose at the pin. The pooled σ̂ also assumes equal ON and OFF seed SDs, while a lever may add seed variance.
  Report each arm's SD.
- **m9. The coefficient smoke picks c by a median gradient-norm ratio**, which is noisy at a plateau. Register the
  batch count and a tie rule: within the CI ⇒ the smaller c.
- **m10. r2's parts were extracted with `sources_dirty_vs_commit: true`.**
  - The golden matches the pin, but the layout hash keys only the golden file's bytes, `obs_dim` and the arch
    signature.
  - A dirty-tree encoding change would pass VOID check 2.
  - **Fix:** name the dirty paths in X60-B's PREREG.
- **m11. "Ruling `4f555e57`" is a measurement commit** (the panel-validity read), not an owner ruling. F alone is
  defensible on X61P's precedent; cite it as such, and keep F + P72 reported.

## 6. FINDINGS (standing rule 7)

1. **Not verified:**
   - whether f4b25238's sim refuses, or silently mis-simulates, moves added on 10-26 (F-M3 assumes the latter is
     possible);
   - the size of the engine drift between X61P's C0 cells and a future ON read (F-M2 is about the check, not a
     measured bias).
2. **The exchange rate between the meters (F-M6) and R72's expected effect (F-M5) were not estimated.** Each is the
   fix's first step.
3. **HEAD did not move under the reviewed files during this review.**
4. **Process.**
   - CPU only, at nice 19 on 5-7,13-15, ≈ 2 min of simulation.
   - Read-only on `models/`, `~/gen3ai_archive/x59s/` (manifests only) and the panels.
   - `forum216b` was never read, and the design was not edited.

## 7. Reproduce

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
  designs/research_state/measurements/human_replay_quality_2026-10-25/review_f_checks.py
```
