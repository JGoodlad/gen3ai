# X37a — the evaluation matrix's first read: PREREGISTRATION (2026-10-18)

**Written and committed BEFORE any X37a game.** Spec: `designs/research_state/EXPERIMENT_BACKLOG.md` row DV-1 / X37a and
[`design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md) §3.1, §5.2, §5.3, §6.3.
Nothing below is changed after a game is played; a deviation is reported as a FINDING in the read, never edited here.
Abbreviations: pp = percentage points; CI = confidence interval; P = home72 (our `data/teams/sample`, 72 teams);
H = `heldout72_hpfix` (72 replay-derived teams, Hidden Power repaired); S(X, Y) = X26's win rate when X26 pilots set X
and Kakuna pilots set Y; K(X, Y) = Kakuna's win rate piloting X against Kakuna piloting Y.

## 0. The regime (every cell)

- **Opponent:** `metamon:Kakuna` ckpt 34 (Metamon `0a00a759`), **greedy**; our side greedy. CPU only
  (`CUDA_VISIBLE_DEVICES=""`), `OMP_NUM_THREADS=1`, Showdown pin `e0551883f`, the Rust websocket front end, 250-turn
  forfeit on our side, 20-game role-balanced units (10 we challenge, 10 they challenge), `--team-seed = --seed-base =`
  the unit seed.
- **X26** = `rb_x26_s2601` `checkpoint_161000169_steps.zip`, sha256 `bc6050d1ca254abd9866945f3dde74a2189514a551e83c29abb79b2a20e52992`,
  the read-only copy at `~/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17/model/` (re-hashed before every unit), played
  by commit **`63f97a57`** in a fresh detached checkout through the core slot (`rust_core_slot`,
  `ws_frontend[in-process]@63f97a57`), `gen3ai_torch28`, `bare` load: the stamp of every earlier Kakuna read.
- **K(P, H)** (no X26) runs from a detached checkout of the commit that lands B-D9 / B-D6 (named in the read).
- **Rule 5b:** ONE lane, `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 12`, the queue `scripts/xq.py`
  (one unit at a time; the T28 halt marker, the model sha256, the config sha256 and EVERY team file on BOTH sides
  sha256-checked against the committed manifests before every unit). Every unit also writes `--capture-dir` (each
  battle's repro record, whose START command carries both packed teams), which changes no stamp and no play.

## 1. Frozen inputs (`x37a_stage1_registration.json`, written by `scripts/setup_stage1.py`)

- **`ladder_weights_v1` is FROZEN by this commit** (its `status` field says so): CORE **4025 / 22862**, NEAR
  **15424 / 22862**, FAR **3413 / 22862** (≈ 0.1761 / 0.6747 / 0.1493; the EXACT fractions are used), from the
  ladder proxy `hl_05_26/gen3ou` (digest `c8160dfd…`) against the sample reference home72 ∪ `away` (digest
  `503572fc…`), matcher `gen3_team_overlap_v1`. Strata by a team's best SPECIES overlap with one sample team:
  CORE = 6, NEAR = 4–5, FAR ≤ 3.
- **The stratum assignment of H** is the frozen `ladder_weights_v1` membership (16 CORE / 44 NEAR / 12 FAR),
  re-derived by `setup_stage1.py` and refused unless identical; the team lists are in the registration JSON.
- **Metamon sub-sets** `gen3ai_x37a_hl72hp_{core,near,far}`: byte copies of `gen3ai_heldout72_hpfix`'s members.
- **Configs** `configs/anchors_x37a_{deploy_core,deploy_near,deploy_far,reverse,kcal}.json`, sha256 in the registration.
- **Declared asymmetric cells and their per-side content digests** (`gen3_team_set_digest_v1`: sha256 over the sorted
  sha256 of each team file's stripped text): `declared_digests` in the registration JSON.

## 2. STAGE 1 cells

| cell | X26 / side A pilots | Kakuna / side B pilots | seeds (units) | n | tree |
|---|---|---|---|---|---|
| `deploy_core` = D_CORE | P | H's 16 CORE teams | 20–24 (5) | 100 | `63f97a57` |
| `deploy_near` = D_NEAR | P | H's 44 NEAR teams | 0–19 (20) | 400 | `63f97a57` |
| `deploy_far` = D_FAR | P | H's 12 FAR teams | 25–29 (5) | 100 | `63f97a57` |
| `reverse` = S(H, P) | H | P | 0–19 (20) | 400 | `63f97a57` |
| `kcal` = K(P, H) | Kakuna on P (`--our-teamset`, the REPORTED side) | Kakuna on H | 0–19 (20) | 400 | the B-D9 commit |

Banked cells used as they stand (same regime stamp): **S(P, P)** = X26 home72, `k2_x26_kakuna_both_teamsets_2026-10-17`
cell `X26_greedy_h72` (n = 400, 0.575); **S(H, H)** = X26 heldout72_hpfix, `heldout72_hpfix_kakuna_2026-10-18` cell
`X26_greedy_hl72hp` (n = 400, 0.448).

The queue runs the four X26 cells interleaved by progress fraction, then `kcal`. No win rate of a cell is looked at
before its last unit ends (only `status` / `n` / the integrity fields); there is no early stop on a result. If the
10 h time box ends a cell early, the cell is read at its counted whole units and the shortfall is a FINDING.

## 3. Counting, scoring and the decision rules (stage 1)

**A unit COUNTS** iff `main.anchors.results.unit_counts` (B-D6) says so: `status OK`, `n == 20`,
`regime_verified_decisions`, `peer_clean`, and its team sets:
- **declared cells** (all five above): the per-side content digests equal the registration's `declared_digests`.
  For a summary from `63f97a57` (which predates the digest), the digests are the queue-verified ones recomputed by the
  aggregate from the same sha-checked files (`verified_digests`);
- **AND the capture check:** for every game, the team our side piloted is in its declared set, and Kakuna's packed team
  in the capture's START command matches (species + moves, `gen3_team_overlap_v1`) exactly one team of ITS declared set.
  A game that fails voids its unit.

A unit that does not count is reported with its reasons and not replaced beyond the queue's 2 tries.

**Score.** Win rate = wins / n; a tie (or a 250-turn forfeit draw) is in the denominator and not the numerator (every
earlier read's convention). In `kcal`, Metamon's CSV books a tie as a loss of the reported side, the same convention.

**Estimators.** z = 1.959964; Wald variance p(1 − p) / n per cell; cells independent.
- **L** (THE headline) = Σ_s w_s · D_s with the exact `ladder_weights_v1` fractions. **L_c1** = w_CORE · S(P, P) +
  w_NEAR · D_NEAR + w_FAR · D_FAR.
- **L's interval:** a stratified two-stage bootstrap, B = 20,000, `numpy.random.default_rng(3700)`: within each stratum
  resample Kakuna's teams that were played (with replacement, as many as were played), then each resampled team's
  games (with replacement); the percentile 95 % interval. Kakuna's team per game comes from the capture. L_c1's
  S(P, P) term resamples its games within our piloted team (the banked rows carry no peer team, F-DV-6). The
  game-only Wald interval is reported beside it.
- **S(P, H)** = Σ_s (n_s / 72) · D_s with n_s = 16 / 44 / 12 (H's own strata), Wald variance Σ (n_s / 72)² var(D_s).
- **PILOT** = S(P, P) − S(H, P); **FACE** = S(P, P) − S(P, H); **Δ̂** = K(P, H) − 0.5; **PILOT_c** = PILOT − Δ̂;
  **FACE_c** = FACE + Δ̂; **I** = S(P, H) + S(H, P) − S(P, P) − S(H, H). Each with a 95 % Wald interval from the
  independent cells' variances (S(P, P) cancels in PILOT_c − FACE_c).

**H-PILOT** ("we do not PILOT unfamiliar teams well", §5.2), applied in this order:
1. UNDECIDED if I's 95 % CI excludes 0 (I detected), or any bound used below lies within 1e-9 of its boundary (rule 8);
2. **SUPPORTED** iff PILOT_c's lower bound > 0 AND (PILOT_c − FACE_c)'s lower bound > 0;
3. **KILLED** iff PILOT_c's upper bound < +0.030, OR (FACE_c − PILOT_c)'s lower bound > 0;
4. otherwise UNDECIDED.
Its assumption **A-K** (Kakuna's piloting and facing familiarity are equal) is UNVERIFIED and stated with the verdict.

**H-PILOT-OFFMETA** (the owner's "off-meta" reading): in `reverse`, our score piloting H's FAR teams minus piloting
H's CORE teams (by OUR team's frozen stratum), a team-clustered bootstrap (resample our piloted teams within each
stratum, then games), B = 20,000, `default_rng(3701)`. **SUPPORTED** iff the 95 % upper bound of FAR − CORE < 0;
**NOT SUPPORTED** otherwise; UNDECIDED if that bound is within 1e-9 of 0.

**Descriptive only (no verdict):** D_s by stratum against S(P, P); the FACE drop by stratum; per-role splits; the
forfeit and tie counts; game lengths; the `team_overlap` block of every set (computed by the aggregate with
`main.anchors.team_overlap` at the read commit, because `63f97a57` predates it).

## 4. STAGE 2 (registered now; its panels are drawn and frozen by the rules below BEFORE any stage-2 game)

**forum72** — from `smogon_forum_teams_2026-10-18/` **v1** (its committed manifest; the v2 being collected is NOT
used): the 166-team `independent_core`.
- **Curator** := the team's `credited` builder, else (uncredited) the posting archivist `[post_author]`.
- **Per-curator cap: 14.** Stated before drawing, from the curator COUNTS only (no team was drawn or played):
  cap 12 leaves 67 eligible teams (< 72); cap 14 leaves 73. thelinearcurve (60), Ibidem (56) and the uncredited
  Teams-Through-the-Ages archive `[M Dragon]` (19) are capped at 14 each; every other curator has ≤ 6.
- **Draw:** core ids sorted, shuffled ONCE with `random.Random(3720)`, taken in that order, skipping a team whose
  curator already has 14, until 72 are accepted. A candidate is rejected (and the next taken) if the pinned
  validator (`validate_teams_locally`, gen3ou) rejects it, or if it is an exact team or shares ≥ 4 mon sets with any
  ONE team of: our 719-team pool, `competitive`, home72, H, or a team already drawn (`gen3_team_overlap_v1`).
- Our side reads a byte copy; Metamon reads `gen3ai_forum72` (byte copies); every file sha256-checked by the queue
  against `forum72_manifest.json`.

**P72** — procedural, Smogon-only (B-D4): `node src/rust_sim/harness/ou_random_teams.js --emit 200 --seed 3721`
(every input a Smogon-derived `data/pokemon/` file; NOT the pool's co-occurrence), each packed team exported to
paste text by Showdown's own `Teams.export`, kept in emit order, the first 72 accepted under the same rejection rules
(validator; exact or ≥ 4 shared with pool, `competitive`, home72, H, forum72, or an earlier P72 team) plus every
Hidden Power set playing the type its line names at power 70 (`utils.bridge.hp_set_check`). Installed as
`gen3ai_p72`, `p72_manifest.json`, the same checks.

**Cells:** `forum72` = S(forum72, forum72) and `p72` = S(P72, P72), X26 vs Kakuna, symmetric, seeds 0–19, n = 400
each, the stage-1 regime and counting rule (undeclared, symmetric: content digests must be equal), from `63f97a57`.
Each panel's strata are computed by the frozen `ladder_weights_v1` rule (descriptive).

**The third-source rule (§3.1).** For T ∈ {P72, forum72}: d1 = X26(T) − X26(H) and d2 = X26(T) − X26(home72),
Newcombe 95 % (independent samples; X26(H) = 0.448 / 400, X26(home72) = 0.575 / 400, the banked cells):

| class | condition |
|---|---|
| **UNSEEN-DRIVEN** | d1 NOT detected AND d2's upper bound < 0 |
| **POPULATION / KAKUNA-DRIVEN** | d1's lower bound > 0 AND d2 NOT detected |
| **MIXED** | d1's lower bound > 0 AND d2's upper bound < 0 |
| **UNDECIDED** | otherwise, or any bound within 1e-9 of 0 |

"Detected" = the 95 % interval excludes 0. **When P72 and forum72 disagree, forum72 governs**; P72's class is reported
beside it. H4's kill test (§3) is the P72 row's UNSEEN-DRIVEN class.

## 5. What is reported

The full matrix (every cell with n, W–L–T, Wilson 95 %, its bias row from §5.1 and its `team_overlap` block), L and
L_c1 quoted as "L (ladder_weights_v1) = p [lo, hi]", the H-PILOT and H-PILOT-OFFMETA verdicts with their numbers,
the third-source classes, every unit that did not count, and every hazard as a FINDING. PROVISIONAL throughout: X26
is one checkpoint (161M) of a live, unplateaued run, one seed.
