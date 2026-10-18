# X37a — the evaluation matrix's first read (2026-10-18)

**What this is.** The first read of the evaluation MATRIX of
[`design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md) §5: X26
(`rb_x26_s2601` @161,000,169) against `metamon:Kakuna` (Metamon's best public policy), greedy against greedy, on the
CPU. It splits the held-out drop into PILOTING (we play a team we never trained on) and FACING (the opponent plays one),
calibrates the team-strength term with a Kakuna-vs-Kakuna cell, and produces THE external headline **L**, a
ladder-weighted deployment estimate. **The preregistration ([`PREREG.md`](PREREG.md), commit `8a2e80de`) was committed
before any game**; nothing in it changed afterwards. Abbreviations: pp = percentage points; P = home72 (our
`data/teams/sample`, 72 teams); H = `heldout72_hpfix` (72 replay-derived teams none of our models trained on); S(X, Y) =
X26's win rate when X26 pilots set X and Kakuna pilots set Y; K(X, Y) = Kakuna piloting X against Kakuna piloting Y.

## STAGE 1 — the matrix's core

**Answer.**

- **THE headline: L (ladder_weights_v1) = 0.630 [0.573, 0.686]** (stratified team-clustered bootstrap; game-only
  Wald [0.592, 0.669]). When X26 brings its own sample teams and Kakuna brings ladder-like teams, X26 wins about 63 %.
  **L_c1 = 0.624 [0.570, 0.676]** (the ladder's CORE stratum read as literal sample teams instead).
- **The held-out drop is about PILOTING, not facing.** X26 FACING held-out teams does not lose anything measurable:
  it scores 0.631 when Kakuna pilots H against our P, against 0.575 when Kakuna pilots P. X26 PILOTING held-out teams
  against Kakuna's P scores **0.380**. Corrected for team strength: PILOT_c = **+0.123 [+0.039, +0.206]** (detected),
  FACE_c = +0.017 [−0.062, +0.095] (not detected).
- **H-PILOT ("we do not pilot unfamiliar teams well") is UNDECIDED by its registered rule**, by a hair. PILOT_c is
  detected above 0, but PILOT_c − FACE_c = +0.106 [**−0.009**, +0.221] misses SUPPORTED's bar (lower bound > 0) by
  0.9 pp. KILLED needs PILOT_c's upper bound below +3 pp (it is +21 pp). Every point estimate leans to SUPPORTED.
- **H-PILOT-OFFMETA is SUPPORTED.** In the reverse cell X26 pilots held-out FAR (off-meta) teams at 0.263 and CORE
  (sample-like) teams at 0.463. FAR − CORE = **−0.199 [−0.385, −0.017]** (team-clustered bootstrap).
- **The held-out teams are weaker in anyone's hands:** K(P, H) = 0.5725 [0.524, 0.620], so Δ̂ = +0.073. That is part
  of why L is high. L is "X26 vs Kakuna when Kakuna brings ladder-reconstructed teams", never a ladder win rate.
- PROVISIONAL: one checkpoint (161M) of a live, unplateaued run, one seed.

### The matrix (stage 1)

Wilson 95 %. Ties are in the denominator, not the numerator. Bias = the direction a cell leans (§5.1).

| cell | X26 / side A pilots | Kakuna / side B pilots | n | W–L–T | win rate [95 %] | bias |
|---|---|---|---:|---|---|---|
| S(P, P) (banked, `k2_x26_kakuna_both_teamsets_2026-10-17`) | P | P | 400 | 230–168–2 | 0.575 [0.526, 0.623] | toward us |
| **D_CORE** = S(P, H_CORE) | P | H's 16 CORE teams | 100 | 61–39–0 | **0.610 [0.512, 0.700]** | deployment shape |
| **D_NEAR** = S(P, H_NEAR) | P | H's 44 NEAR teams | 400 | 249–150–1 | **0.623 [0.574, 0.669]** | deployment shape |
| **D_FAR** = S(P, H_FAR) | P | H's 12 FAR teams | 100 | 69–30–1 | **0.690 [0.594, 0.772]** | deployment shape |
| S(P, H) (D_s reweighted 16 / 44 / 12 of 72, Wald) | P | H | 600 | — | 0.631 [0.592, 0.670] | |
| **S(H, P)** (REVERSE) | H | P | 400 | 152–248–0 | **0.380 [0.334, 0.428]** | toward Kakuna (we pilot unfamiliar teams) |
| S(H, H) (banked, `heldout72_hpfix_kakuna_2026-10-18`) | H | H | 400 | 179–220–1 | 0.448 [0.400, 0.496] | toward Kakuna |
| **K(P, H)** (Kakuna vs Kakuna) | Kakuna on P | Kakuna on H | 400 | 229–171–0 | **0.573 [0.524, 0.620]** | the strength calibration |

`ladder_weights_v1` (frozen by the registration): CORE 4025 / NEAR 15424 / FAR 3413 of 22,862 (0.1761 / 0.6747 /
0.1493).

| quantity (PREREG §3) | estimate | 95 % |
|---|---:|---|
| **L** = Σ w_s D_s | **0.630** | bootstrap [0.573, 0.686]; Wald [0.592, 0.669] |
| **L_c1** = w_CORE S(P, P) + w_NEAR D_NEAR + w_FAR D_FAR | 0.624 | bootstrap [0.570, 0.676]; Wald [0.588, 0.660] |
| PILOT = S(P, P) − S(H, P) | +0.195 | |
| FACE = S(P, P) − S(P, H) | −0.056 | |
| Δ̂ = K(P, H) − 0.5 (the strength term, under A-K) | +0.073 | |
| **PILOT_c** = PILOT − Δ̂ | **+0.123** | [+0.039, +0.206] |
| FACE_c = FACE + Δ̂ | +0.017 | [−0.062, +0.095] |
| PILOT_c − FACE_c | +0.106 | [−0.009, +0.221] |
| I (the interaction; additivity check) | −0.012 | [−0.104, +0.081], not detected |
| **H-PILOT** | **UNDECIDED** | SUPPORTED needs the line above's lower bound > 0 |
| reverse cell by OUR team's stratum: CORE / NEAR / FAR | 0.463 (80 games, 16 teams) / 0.389 (244, 41) / 0.263 (76, 12) | |
| **H-PILOT-OFFMETA**: FAR − CORE | **−0.199** | [−0.385, −0.017] ⇒ **SUPPORTED** |

### Reading

- **Deployment looks good; transfer does not.** Bringing the teams it knows, X26 beats Kakuna whatever kind of team
  Kakuna brings. D_s ≥ 0.61 in every stratum, and L's interval lies above 0.50. The deficit is X26 playing a team it
  never trained on: −0.195 against home72 (raw), −0.123 after the strength correction.
- **Facing unfamiliar teams costs nothing measurable** (FACE_c +0.017, not detected; its interval allows up to about
  +9.5 pp). If anything, the held-out teams are easier to face, because they are weaker teams (K(P, H) = 0.573).
  Implicit team identification (H2) would predict a facing cost, and none shows. That is not a kill under §3, whose
  test is D0e.
- **What this means for the levers (a proposal, not a decision).** A piloting deficit points at OUR side's team
  diversity: D1 widens both sides, and its own-side half is the half this read supports. D2 (opponent-only widening)
  loses most of its motivation. The off-meta result says the widening corpus should cover FAR cores, not only new
  sets on sample cores.
- **L leans toward us in three ways.**
  - Kakuna's teams in the deployment cells are usage-filled ladder RECONSTRUCTIONS, weaker than the sample teams
    (Δ̂ = +0.073). A real ladder opponent brings the team they built.
  - Our side pilots exactly the teams it trained on most (about 2.1× a pool team).
  - The opponent is Kakuna, not a human.

  L is the deployment estimate the matrix defines (§5.3), not a ladder win rate.
- **A-K is UNVERIFIED.** The strength correction assumes Kakuna's piloting and facing familiarity are equal in size.
  If Kakuna pilots its own replay population unusually well, Δ̂ is understated, and PILOT_c is OVERstated by the same
  amount.

### Integrity

- **All 70 units counted on their first try** (`skipped_units` empty, the queue's `fails: {}`). Each unit passed:
  - `unit_counts` (B-D6): `status OK`, n = 20, `regime_verified_decisions`, `peer_clean`;
  - the declared per-side content digests equal the registration's. They were recomputed from the sha-checked files
    for the `63f97a57` units. The `kcal` summaries carry them natively and match.
- **The CAPTURE check:** every game's front-end START record carries both packed teams. For every X26 game, our team
  is the row's `our_team`, and Kakuna's team matches exactly one team of its declared set. That gives Kakuna's team per
  game (`rows_stage1.jsonl` `their_team`) and resolves F-DV-6 for these cells. Kakuna drew 15 / 16 CORE, 44 / 44 NEAR
  and 12 / 12 FAR teams. In `kcal`, every game had one team from each declared set.
- **Integrity counters:**
  - the 250-turn forfeit fired 1 / 2 / 0 / 0 times in CORE / NEAR / FAR / reverse; `kcal` is a pair cell with no
    forfeit rule (§2 of the SOP) and had 0 ties;
  - median length 34–35 turns in the X26 cells, 42 in `kcal`;
  - by role (we challenge / they challenge): CORE 0.54 / 0.68, NEAR 0.630 / 0.615, FAR 0.68 / 0.70,
    reverse 0.365 / 0.395, `kcal` 0.55 / 0.595.
- **Overlap** (`gen3_team_overlap_v1`, `cells_stage1.json` `team_overlap`, computed here because `63f97a57` predates
  the module):
  - every H sub-set: 0 exact, 0 sharing ≥ 4 mon sets with our pool or `competitive`;
  - P: 72 / 72 pool;
  - H against `hl_05_26`, its own source: 5 exact, 42 sharing ≥ 4.

**Regime stamp.** The same stamp as the 10-17 / 10-18 reads, except the team sets:

- **X26 cells:** Kakuna ckpt 34, Metamon `0a00a759`, CPU, `OMP_NUM_THREADS=1`, greedy / greedy, Showdown pin
  `e0551883f`, 250-turn forfeit, 20-game role-balanced units with `--team-seed = --seed-base` = the unit seed.
  X26: `checkpoint_161000169_steps.zip` sha256 `bc6050d1…2992` (re-hashed before every unit), commit `63f97a57` in
  a fresh detached checkout, `bare` load, `rust_core_slot`, `ws_frontend[in-process]@63f97a57`.
- **`kcal`:** commit `cb0f1769` (B-D9's `--our-teamset`), `ws_frontend@cb0f1769c`, two Kakuna peers, reported from
  side A (Kakuna on P), `--declare-asymmetric-teams`.
- **Seeds:** CORE 20–24, NEAR 0–19, FAR 25–29, reverse 0–19, `kcal` 0–19.

**Cost.** One lane on rule 5b's cores (`nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh 12`), beside the live X26 trainer
and the maturity battery. The X26 cells took about 200 s per 20-game unit (≈ 10 s / game, 1,000 games in 2 h 20 min);
`kcal` took about 2.8 min per unit (400 games in 56 min). 0 GPU.

### Stage 1 FINDINGS

1. **Two spellings of one commit in the stamp.** `server_version` reads `…@63f97a57…` on some units and
   `…@63f97a575…` on others. Git's abbreviated hash grew one character as the repo gained objects mid-read. It is the
   same commit, but each X26 cell carries two stamp strings. A reader matching stamps by string must normalise the hash.
2. **`kcal` crosses a harness boundary.** It runs at `cb0f1769`, not `63f97a57`. It is a pair cell: Metamon's CSV is
   the record, there is no 250-turn forfeit, and a tie is booked as a loss (0 ties occurred). Its number is used only
   as Δ̂.
3. **A-K, the assumption behind Δ̂, is UNVERIFIED** (above). The H-PILOT verdict and PILOT_c carry it.
4. **One game result was seen early.** While confirming the first unit was alive (13:13), a log tail showed the
   results of two `deploy_near_00` games. PREREG §2 says no win rate is looked at before a cell ends. No stop rule
   exists and nothing was decided on it; it is reported for completeness.
5. **L_c1's S(P, P) term is clustered by OUR team**, because the banked rows carry no peer team (F-DV-6). L's own
   strata are clustered by Kakuna's team, read from the capture.
6. **The `not slow` anchors tier plays a real local game.** `test_one_capped_real_game_against_metamon_smallrl`
   starts a local Metamon peer (60 s CPU beside the live run; not a public server, not 8000 / 8001). That is the build
   worker's finding; the test is not re-tiered here.
7. **B-D6's digest reads the directory.** Metamon instead draws from a set's `index.csv` when one exists. Only
   `competitive` has one, and no X37a cell uses it. A set that grows an `index.csv` listing a subset would make the
   digest describe more than the draw set.
8. No other hazard hit:
   - the T28 halt marker was clear before every unit;
   - the tool's own server ran on 9537;
   - nothing was written under `models/` or `data/`;
   - no GPU (`CUDA_VISIBLE_DEVICES=""`).

## STAGE 2 — the third-source rows

PENDING (registered: PREREG §4; panels frozen in `cb0f1769`).

## Files

- `PREREG.md`, `x37a_stage1_registration.json`, `x37a_stage2_registration.json`, `configs/`: the registration.
- `forum72_manifest.json`, `teams_forum72/`, `p72_manifest.json`, `teams_p72/`: the stage-2 panels.
- `cells_stage1.json`: every cell, every unit's counting verdict, the overlap blocks and the stage-1 analysis.
  `rows_stage1.jsonl` holds 1,400 game rows, with `read_cell`, `unit`, `unit_seed` and, from the capture,
  `their_team`.
- `scripts/`:
  - `setup_stage1.py`: strata, copies, Metamon sub-sets, configs, registration;
  - `make_stage2_panels.py`: the forum72 draw and the P72 build;
  - `xq.py`: the one-lane queue;
  - `aggregate.py`: counting, the capture check and the estimators.
- **Not committed:** the raw unit dirs, the captures, the logs and `units.jsonl`, all in
  `/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/`.
