# N2 exact-set identity test: result (2026-10-26)

**Verdict: COMPOSITIONAL. N2 (training-only noise on the network's view of its own team's sets) is dropped.**
Registered in [`PREREG.md`](PREREG.md) (`98bcb1146`, on main before any policy forward). CPU only, read-only on
`models/`.

Abbreviations: P\* = `rb_x26_s2601` @ 280,000,170 steps (the X26 plateau); pool = the 719-team training pool;
home72 = 72 pool teams; F / H / P72 = the forum, held-out and procedural panels (the three UNFAMILIAR panels);
KL = Kullback–Leibler divergence KL(p_base ‖ p_perturbed) of the policy over the legal actions, in nats;
Δv = the change of the critic's P(win); R = the POOL / UNFAMILIAR ratio; CI = 95 % bootstrap interval over games.

## The question and the answer

Does P\* choose by the EXACT identity of our own sets (a memorised team → plan), beyond what those sets do
mechanically? **Barely.** Re-drawing one of our bench mons' sets from Smogon usage moves the policy **1.19× as much on
a pool team as on an unfamiliar team** (CI 1.07 to 1.33). That is a small, real familiarity premium, but well under the
1.5× the memorisation mechanism behind N2 needs. Two other reads point the same way:

- Erasing the identity embeddings of all six of our mons hurts familiar and unfamiliar teams alike (R 1.07, CI 0.91 to 1.25).
- Perturbations that keep a set's mechanics (move order, a nickname, an EV change the game rounds away) are
  invisible or nearly so.

## Numbers

Rows: the architecture-generalisation review's archived P\* vs P\* decision rows, side 0 only, so the opponent is a
home72 team in both groups. POOL = home72, 48 games, 1,072 rows, 7,260 units. UNFAMILIAR = F + H + P72, 143 games,
3,594 rows, 24,384 units. A unit is (row, target bench mon, draw). There were 3 Smogon usage draws per target, from
D3's generator, and all of them passed the validator. The splice guard failed 0 times in 10,548 targets.

**PRIMARY** (the pattern-standardised KL ratio; the eight cells are moves differing 1–4 × item differs, weights pooled):

| | POOL | UNFAMILIAR | R | CI |
|---|---:|---:|---:|---|
| **mean KL, standardised (the registered statistic)** | 0.102 | 0.086 | **1.19** | **[1.07, 1.33]** |

**COMPOSITIONAL, by the declared rule** (CI upper 1.33 < 1.5). **Read this with care:** the interval EXCLUDES 1.0. So
the strategy doc's literal KILL clause ("covers 1.0 and excludes 1.5") is NOT met. The verdict rests on the
extension the PREREG declared in §3: an upper bound below 1.5 excludes a premium of the registered size whether or
not 1.0 is covered. The team-level guard did not fire (S4's identity ratio CI lower bound is 0.91, not > 1.5).

**Secondaries:**

| read | POOL | UNFAMILIAR | R | CI |
|---|---:|---:|---:|---|
| S1 raw KL ratio (unstandardised) | 0.098 | 0.086 | 1.14 | [1.02, 1.28] |
| S1 argmax flip rate | 0.161 | 0.155 | 1.04 | [0.93, 1.16] |
| S1 \|Δv\| | 0.024 | 0.022 | 1.11 | [0.98, 1.26] |
| S1 the 29 species present in both groups | 0.102 | 0.085 | 1.20 | [1.06, 1.34] |
| S1 POOL against F alone | | 0.081 | 1.28 | [1.11, 1.47] |
| S1 POOL against H alone | | 0.090 | 1.10 | [0.96, 1.25] |
| S1 POOL against P72 alone | | 0.086 | 1.21 | [1.06, 1.39] |
| S2 \|ΔP(switch to the re-drawn mon)\| | 0.044 | 0.043 | 1.03 | [0.92, 1.16] |
| S4 IDENTITY ablation (our six mons' species / move / item embeddings → mean), KL | 0.572 | 0.537 | 1.07 | [0.91, 1.25] |
| S4 IDENTITY ablation, flip | 0.441 | 0.430 | 1.03 | [0.94, 1.12] |
| S4 MECHANICS ablation (our stats, move mechanics columns, `MOVE_ATTR` → mean), KL | 0.183 | 0.188 | 0.97 | [0.84, 1.12] |
| S4 MECHANICS ablation, flip | 0.273 | 0.260 | 1.05 | [0.92, 1.20] |

By cell (KL, POOL / UNFAMILIAR, units): 1 move 0.063 / 0.058 (1,942 / 5,421); 1 move + item 0.111 / 0.102;
2 moves 0.090 / 0.080; 2 + item 0.136 / 0.089; 3 moves 0.121 / 0.105; 3 + item 0.131 / 0.121; 4 moves 0.220 / 0.112
(182 / 1,131); 4 + item 0.114 / 0.146 (84 / 274). The premium is not monotone in the size of the change. The one
large cell ratio (all four moves re-drawn, item kept, 2.0×) rests on 182 POOL units and reverses in its item-changed
neighbour. DESCRIPTIVE.

**S3, the mechanics-preserving perturbations** (every 4th eligible row, its first target; `s3.json`):

| perturbation | n (POOL / UNF) | observation changed | max \|Δp\| | mean KL (POOL / UNF) | flip |
|---|---|---|---:|---|---:|
| move lines reversed | 268 / 899 | never (moves are sorted by id) | 9e-6 | ~1e-9 / ~1e-9 | 0 |
| a nickname | 268 / 899 | never (not observed) | 5e-6 | ~0 / ~0 | 0 |
| a stat-identical EV change (4 → 6 or 252 → 255) | 73 / 255 | always (an EV column) | 0.035 | 2e-6 / 2e-5 | 0.014 / 0 |

The first two are invisible to the network by construction: a nickname is not observed, and the move slots are
sorted by id. The max |Δp| of up to 1e-5 is float noise from batch composition. The third moves the network's SMOOTH
stat input (`2·base + IV + EV/4`, no floor) by half a stat point the game does not see. Its KL is 3–4 orders of
magnitude below a usage re-draw's, so it is a noise floor, not a channel.

**S4's patched encoder** reproduced the live forward exactly (max |Δp| 0.0) with nothing ablated.

## What it means

- **There is a small familiarity premium, and it is about the same size as the review's.** The review's P8 found a
  ≈ +19 % premium the other way round: on a pool team, swapping in an UNFAMILIAR set moved the policy more than a
  familiar one. Here, re-drawing a FAMILIAR set moves it ≈ +19 % more than re-drawing an unfamiliar one. Both reads
  say P\* is slightly more "attached" to sets it trained on. Neither shows the 1.5× that would make the set's exact
  identity a key worth masking.
- **Identity embeddings are a vocabulary, not a team key.** P\* leans on them heavily (S4: erasing them gives KL 0.55
  against 0.19 for erasing the mechanics columns, the review's P2 / P7 again), but equally on teams it has and has not
  trained on. A network that looked a team up by its sets would lose more on the teams it can look up.
- **Consequence for N2.** Masking our own set fields in training attacks a channel whose familiar-team excess is
  ≈ 0.2× of a one-set change. That premium is at most what N2 could remove, and the mask would cost the real
  information our own sets carry. The registered rule drops it. The breadth levers (D3, N1, the stack, N5) are
  unaffected by this read.
- **What this does not rule out.** It does not rule out plan memorisation that a one-mon re-draw or an all-six
  identity erasure cannot reach: one keyed on the revealed course of the battle, or on a mon only once it is active
  or revealed (the probe sees only never-seen bench mons, so early-game rows dominate). The review's F-AG-5
  structural point (the flat policy tower as the cheapest store of per-team plans) stands untested.
- **It agrees with the piloting-gap forensics** (`piloting_gap_forensics_2026-10-26/`, same day, descriptive): P\*'s
  per-mon edge was read as dropping with the TEAM, not the set ("team-context play, not set knowledge"); that is
  now OPEN (2026-10-27 note below). This
  probe is the policy-side check of the "not set knowledge" half. The team-context half is H-CTX's to read (the
  recombined-pool panel).

## FINDINGs and deviations (standing rule 7)

- **F-N2-H1 (environment).** The probe ran with MAIN's `src/` on `PYTHONPATH`, not the worktree's. The worktree has
  no Rust build, and main's release build refuses to load from a worktree (its stamp names main's data dir). Main moved
  during the run (from `98bcb1146`, the PREREG commit); none of its commits since touch the model, observation, Rust,
  team-corpus, data-facade or teambuilder code these scripts import, and this unit changes no `src/`.
- **F-N2-H2 (a wrong description in the review).** The review says its rows use "the panel's first 48 teams, in
  order". They use 48 distinct teams with indices up to 68 per panel. `collect.py` stages episodes ahead, so the
  first 48 games to FINISH are kept, and the game id equals the team index. The first probe run failed on this
  (an index error, before any output). The draws were rebuilt for all 72 teams. The seeds are per (panel, team,
  slot, attempt), so every draw already built was unchanged. The design was not altered.
- **F-N2-H3 (unregistered handling).** 27 of the 2,000 bootstrap resamples drew no POOL unit in a sparse cell
  (4 moves + item, 84 units) and were discarded. The PREREG did not say how to treat this. The unstandardised ratio
  (no cells, no discards) gives the same verdict class: 1.14 [1.02, 1.28].
- **F-N2-H4 (gate shape).** `decide.py` (seconds of numpy) ran outside `mem_cap.sh`, at nice 19 on the agent cores,
  after it had queued for minutes behind heavy jobs. Every forward ran under `mem_cap.sh` in the short lane.
- **F-N2-H5.** One checkpoint, one seed, P\* vs P\* rows (not Kakuna, not humans). There are 48 POOL games, so the
  CI is mostly POOL's. Byte identity of a re-run is UNVERIFIED (a multi-threaded CPU forward).
- **F-N2-H6.** The stat-identical EV change applied to only 328 of 1,167 S3 targets. The rest have no 4-EV stat and
  no room for 252 → 255 under the 510 total.
- **F-N2-H7 (observation, out of scope).** The network's stat input is the smooth `EV / 4` (`static_tokens.py`
  `_actual_stats`), so it distinguishes spreads the game treats as identical. It is harmless at this size (S3), and
  it is noted for anyone building a set-equivalence test on the observation.

## Files and reproduce

- `PREREG.md`
- `result.json` (every number above)
- `s3.json`, `probe_meta.json`, `s4_meta.json`
- `scripts/`: `build_draws.py` (draws + Rust encodes), `probe.py` (primary, S3), `ablate.py` (S4), `decide.py`
  (statistics, verdict), `run_all.sh` (the chain, rule 5b's shape), `count_eligible.py` (the PREREG's design
  counts), `n2common.py`.
- The regenerable binaries (`draws.npz`, `units.npz`, `s4.npz`) and `draws_meta.json` (every drawn set, 1.2 MB) are
  not committed. They are sha256-pinned in `artifacts_sha256.txt` and copied to
  `/home/goodlad/gen3ai_archive/n2_identity_test_2026-10-26/`.
- The input rows are the review's (`arch_generalisation_2026-10-26/rows_sha256.txt`; the load refuses on a mismatch).
- `scripts/run_all.sh` regenerates everything in about 10 CPU-minutes of compute, plus queueing for the heavy slots.

2026-10-27 note: "the edge drops with the TEAM, not the set" is not established; team vs set is OPEN until rc72b (H-CTX), per review `90708bff` B1. The result above (COMPOSITIONAL, N2 dropped) is unchanged.
