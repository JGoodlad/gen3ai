# N2 exact-set identity test: PREREG (2026-10-26)

Registered BEFORE any policy forward of this unit. The only data looked at so far are eligibility counts (no policy
output): `scripts/count_eligible.py`.

**The question.** Does P\*'s choice depend on the EXACT identity of our own sets, a memorised team → plan, beyond
their mechanical content? The answer decides lever N2 (training-only noise on the network's view of its own team's
sets; `designs/endstate/strategy_if_levers_null_2026-10-26.md` §3 rank 2, `designs/research_state/literature/
sota_survey_2026-10-26.md` N2). The registered zero-GPU gate is the strategy doc's "exact-set sensitivity probe":
pool-team KL ≥ 1.5 × unfamiliar-team KL under a same-species usage draw; KILL N2 if the ratio's interval covers 1.0
and excludes 1.5.

Abbreviations: P\* = `rb_x26_s2601` @ 280,000,170 steps; pool = the 719-team training pool; home72 = 72 pool teams;
F / H / P72 = the forum, held-out and procedural panels (the three unfamiliar panels); KL = KL(p_base ‖ p_perturbed)
over the legal actions, in nats; Δv = the change of the critic's P(win); CI = 95 % interval.

## 1. Data (fixed, not collected by this unit)

The architecture-generalisation review's archived decision rows (`arch_generalisation_2026-10-26/rows_sha256.txt`
pins them; the load refuses on a hash mismatch): P\* greedy vs P\* greedy, player 1 pilots a panel team (the panel's
first 48 teams), player 2 a seeded home72 draw. **Only side-0 rows are used**, so in both groups the opponent is a
familiar home72 team piloted by P\*.

- **POOL group:** `rows_home72` side 0 (48 games). A target mon's set must be an EXACT pool set (species, sorted moves,
  item, ability, spread).
- **UNFAMILIAR group:** `rows_F`, `rows_H`, `rows_P72` side 0 (143 games). A target mon's set must NOT be a pool set.
- **Eligible target mon (both groups):** on the bench (not active), HP fraction 1.0, no status, its OBS-FACTS `seen`
  row all zero (never on the field), and its species D3-eligible (a Smogon prior with ≥ 4 engine-playable moves, an
  item prior and a spread prior). Counted: pool 1,072 rows / 2,420 target mons; unfamiliar 3,594 rows / 8,128 mons.
  Every eligible (row, mon) is used.

## 2. The perturbations

**PRIMARY: a Smogon usage draw of the same species** (D3's own generator, `utils.team_corpora.build_d3_setmut`:
`species_prior` with its engine prefilter, `mutate_block`, the HP repair). K = 3 draws per (team, slot), the first 3
that pass, in attempt order, of: rule 4 (a NEW mon set: species + sorted moves, HP typed, differs from the source's)
and the pinned `gen3ou` validator on the whole team. Seeds: sha256 of (`n2_identity_v1`, panel, team index, slot,
attempt). Ability, nickname and gender are the source's (D3's rule). The mutated team is encoded by the Rust core's
turn-1 observation; the target slot's SET columns (item, abilities, the four move slots, spread, Hidden Power block)
are spliced into the row; nothing else changes.

- **Splice guard (GIGO):** for every target, the source team's OWN turn-1 set columns must equal the row's set columns
  bit for bit. A target failing it is dropped and counted.

**Per unit (row, target mon, draw):** KL, argmax flip, |Δv|, ΔP(switch to the target mon), and the difference
PATTERN (number of moves differing 1–4, item differs 0 / 1).

## 3. The primary statistic and the decision rule

**R = mean KL (POOL) / mean KL (UNFAMILIAR), pattern-standardised:** the cells are the patterns; a cell is kept if each
group has ≥ 30 units in it; the cell weights are the kept cells' pooled unit shares (both groups); each group's mean
is Σ_c w_c · mean_c. **CI:** 2,000 bootstrap resamples of GAMES, each group resampled independently (numpy seed 0),
the ratio recomputed with the weights fixed; percentile 2.5 / 97.5.

| verdict | rule | consequence |
|---|---|---|
| **KEYS-ON-SETS** | R ≥ 1.5 **and** CI lower bound > 1.0 | N2 is funded (the flag, then its fork) |
| **COMPOSITIONAL** | CI upper bound < 1.5 | N2 is dropped (the registered effect size is excluded) |
| **INCONCLUSIVE** | neither | N2 waits; the report says which bound blocked it |

- COMPOSITIONAL contains the strategy doc's KILL ("covers 1.0 and excludes 1.5") and extends it, declared here, to an
  interval that excludes 1.5 from above without covering 1.0: either way a premium of the registered size is excluded.
- **Boundary rule (deterministic checks):** if R or a CI bound a rule reads lies within 0.005 of its threshold
  (1.0 or 1.5), that rule is treated as NOT met.
- **Team-level guard.** A one-mon swap cannot see a plan keyed on the whole team (the review's own caveat). So: if
  the primary reads COMPOSITIONAL but S4's IDENTITY-ablation ratio (POOL / UNFAMILIAR) has its CI lower bound > 1.5,
  the verdict is INCONCLUSIVE instead.
- **VOID** if: > 5 % of targets fail the splice guard; either group has < 300 units after it; or S4's patched encoder
  without an ablation does not reproduce the base forward (max |Δp| > 1e-5).

## 4. Secondaries (reported; only the guard above can change the verdict)

- **S1.** The raw (unstandardised) KL ratio; the flip-rate and |Δv| ratios (standardised); the standardised ratio on
  the target species present in BOTH groups; the unfamiliar group split by panel (descriptive).
- **S2 ("decisions involving that mon").** The standardised ratio of |ΔP(switch to the target mon)|.
- **S3 (mechanics-preserving perturbations, the brief's (a)).** On a deterministic subsample (every 4th eligible row
  per group, its first eligible target): (i) the move lines reversed in the paste; (ii) a nickname added;
  (iii) a STAT-IDENTICAL EV change (an EV of 4 → 6 if present and the total permits, else 252 → 255; the game's
  `floor(EV / 4)` is unchanged). Each is encoded by the Rust core and spliced like the primary. Reported: the
  fraction of targets whose slot columns change at all, max |Δp| and mean KL per group. Expected (structural, from
  the code): (i) and (ii) leave the observation identical; (iii) changes an EV column, and the network's stat input is
  the SMOOTH formula `2·base + IV + EV / 4` (no floor), so P\* sees +0.5 of a stat point the game does not: a
  measured mechanics-null noise floor.
- **S4 (identity vs mechanics ablation, the brief's (c)).** On every eligible row of both groups (one forward per row),
  OUR six mons only (the opponent's tokens and the hypothesis rows untouched): IDENTITY = the species, move and item
  id embeddings replaced by a constant mean; MECHANICS = the 13 stat columns, the move mechanics columns and
  `MOVE_ATTR` replaced by a constant mean. The means are over our real mons of all side-0 rows of the four files
  (pooled). Reported: KL and flip per group, the POOL / UNFAMILIAR KL ratio of each ablation with the game bootstrap.
  Mean ablation is off-distribution: a reliance contrast between groups, not an effect size.

## 5. Predictions (registered)

From the review (F-AG-2: our team is handled compositionally; an unfamiliar SET moved the policy only ≈ 1.2× a familiar
one): **I predict COMPOSITIONAL, R ≈ 0.8–1.2.** Under the memorisation hypothesis R ≥ 1.5, and S4's identity
ablation would also hit POOL rows harder than UNFAMILIAR ones.

## 6. Hazards declared up front

- One checkpoint, one seed, P\* vs P\* rows (not Kakuna, not humans); 48 POOL games limit the CI.
- The groups differ in species mix and in the panels' own set style (P72's sets ARE usage draws, so a draw is closer
  to them); pattern standardisation and the shared-species read (S1) address the first two partly.
- A bench, never-seen mon only (early-game rows dominate): an identity effect that shows only once a mon is revealed
  or active is outside this probe.
- The probe measures SENSITIVITY of the policy, not quality: a large KL on POOL rows could also be a correct
  re-plan. The ratio against UNFAMILIAR rows is what isolates memorisation.

## 7. Reproduce

`scripts/run_all.sh` (rule 5b's shape: `nice -n 19 taskset -c 5-7,13-15` under `scripts/ops/mem_cap.sh`; CPU only;
read-only on `models/`). `build_draws.py` (the draws and their encodings) → `probe.py` (primary, S1–S3) →
`ablate.py` (S4) → `decide.py` (the verdict).
