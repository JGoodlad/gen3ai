# H-MID: does P\*'s critic see its unfamiliar-team mid-game deficit? PRE-REGISTRATION (2026-10-26)

**Status: REGISTERED before any value (V) is read on the analysis games.** Before this file landed, the only model
outputs computed on these games are the reconstruction checks of §3 on TWO games (`pstar_hpfix_03`, battles 1-2, run with
`--no-values`, so V was never written): protocol identical, 64 / 64 frames, 64 / 64 argmax agreement. Nothing below changes after the first V is read except by a dated amendment,
written before the read, that says why.

Abbreviations: pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; P\* = `rb_x26_s2601`
@280,000,170 (the X51 zip, sha256 `814823c5…91ed`); U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal panel
weight (X51's definition); V = the critic's win probability, `sigmoid(win_head(value_pooled))` (`critic: winprob`,
`ARCHITECTURE.md` §6; `values` equals `win_probs`, the realised return is the terminal win indicator at gamma 1);
D(t) = the material lead at the start of turn t (the forensics' definition: our summed HP fractions over six mons
minus theirs, a mon that never entered counts 1.0); **level = −1.0 < D(t) < 1.0** (within one mon: the forensics'
`conditional` "level", the population of its 0.609 / 0.501; not its ±0.4 decomposition bin).

## 1. Question

The piloting-gap forensics (`c758fcb8`, [`../piloting_gap_forensics_2026-10-26/README.md`](../piloting_gap_forensics_2026-10-26/README.md))
found that P\*'s extra unfamiliar-team losses against Kakuna come from materially LEVEL positions in the mid and late
game: from level material at turn 10 P\* wins 0.609 [0.559, 0.664] on home72 and 0.501 [0.464, 0.541] on U (+10.8 pp
[+4.8, +17.0]). **H-MID** asks which of two things carries that: in those positions on unfamiliar teams, P\* either
**mis-values the position** (its critic is wrong there) or **values it right and its policy cannot act on it**.

## 2. Data (banked games only; no new play)

| | games | cells | read pin (code that played them) |
|---|---|---|---|
| **primary**: X51 P\* vs Kakuna ckpt 34, greedy | 1,600 = home72 400 + H / F / P72 400 each | `gen3ai-reads/x51_2026-10-20/cells/pstar_{home72,hpfix,forum72,p72}_{00..19}` | `63f97a575` |
| **replication**: P\* vs PastaMittens, greedy (the forensics' twins) | 1,600, same split | `gen3ai-reads/mm_strongest_2026-10-25/cells/pm_{home72,hpfix,forum72,p72}_{00..19}` | `d32ffcee4` |

Each capture holds the START command (both packed teams, the seed, `core_obs` naming our side), every CHOOSE of both
sides in submission order, and both sides' protocol. D(t) and the result come from the forensics' per-game rows
(`~/gen3ai_archive/piloting_gap_forensics_2026-10-26/games.jsonl.gz`, sha256 `a35d2ebf…7ad2`), joined on (unit, battle
tag); a join whose result differs from the replay's protocol result is a GIGO failure (stop and report).

## 3. Reconstruction (forward passes only, CPU)

Each capture's command stream is replayed through the **sim_bridge built at the read's own pin** (a detached checkout,
its own `target/`), which emits P\*'s `__OBS__` frame (row, mask, choice tokens) at every decision exactly as the
in-process anchor slot received it. P\* is loaded with that pin's code (`load_checkpoint_strict` +
`historical_load_kwargs`, as `main.anchors.core_side` does) and run on CPU on every frame: V, and the masked logits.

**Fidelity gate (precondition; every check is reported):**
1. **Protocol identity.** The replay's per-side protocol text equals the capture's under
   `utils.bridge.ws_frontend_replay.side_text` (its two normalisations: `|t:|`, the front end's `rqid`).
2. **Frame count.** Our side's frames equal our CHOOSE commands, plus one in a game stopped by the 250-turn forfeit.
3. **Argmax agreement.** At each decision P\*'s masked argmax on the reconstructed row picks the recorded choice token
   (both reads were greedy).

A game failing 1 or 2, or with an argmax disagreement whose top-two logit margin exceeds 1e-3, is EXCLUDED and counted.
**If more than 1 % of a read's games are excluded, or pooled argmax agreement is below 99.5 %, that read is VOID** (the
reconstruction is not the played game): no V statistic from it is reported as a finding, and the verdict is NOT READ
if the void read is the primary.

## 4. States and outcomes

- **The state at turn t** is P\*'s first decision frame whose `turn` is ≥ t, in a game whose final turn is ≥ t (a game
  over before t has no state at t). t ∈ {1, 5, 10, 15, 20, 30}; **t = 10 is the primary**.
- **The outcome** y = 1 win, 0 loss (a forfeit at the turn cap is a loss, as played), 0.5 tie (counted and reported).
- **Weights and intervals:** U's three panels at equal weight, games equal within a cell. Every interval is a 95 %
  percentile bootstrap over 20-game units resampled within each cell, B = 2,000 (the all-decision AUC: 500, for its
  cost), RNG seed 20261026; the AUC is computed per panel and U's is the mean of its three. The scripts as registered:
  `scripts/reconstruct.py` (replay + forward), `scripts/analyze.py` (every statistic and the verdict).

## 5. Measures

For a state set S (home72 or U) at turn t, level states unless "all" is said:

- **(a) Calibration:** W_S (win rate), V̄_S (mean V), **bias_S = V̄_S − W_S**, Brier_S = mean (V − y)², log loss
  (V clipped to [1e-6, 1 − 1e-6]); reliability by V bin, edges 0.2 / 0.4 / 0.6 / 0.8. Contrasts home72 − U:
  G_y = W_h − W_U (the outcome gap), G_V = V̄_h − V̄_U (the gap the critic sees), and
  **Δbias = bias_U − bias_h = G_y − G_V** (the part of the outcome gap the critic does not see), ΔBrier = Brier_U −
  Brier_h, Δlog loss likewise.
- **(b) The policy given the value:** conversion by V bin, W per bin by set, and at V ≥ 0.6 ("V says winning");
  the **matched-V outcome gap**, Σ_b w_h(b)·(W_h(b) − W_U(b)) over the five V bins with home72's state shares as weights
  (a bin empty in either set drops out and the weights renormalise; the dropped share is reported). If V accounts for
  the gap, the matched-V gap is ≈ 0; if V does not see it, ≈ G_y.
- **(c) Discrimination:** AUC of V for y, by set, at level-10 states, all turn-10 states, and over all decisions (each
  game weighted 1 / its decision count).
- **Secondary (descriptive):** every measure at t ∈ {1, 5, 15, 20, 30}, level and all states; the V drift
  mean(V(min(20, T)) − V(10)) with V(T) = y once the game is over (a calibrated critic is a martingale: drift ≈ 0, with
  no outcome noise in the read).

## 6. The decision rule (primary: level states at turn 10, the X51 Kakuna read)

| verdict | condition | the lever (the brief's) |
|---|---|---|
| **CRITIC-MISCALIBRATED-ON-U** | Δbias ≥ +0.05 **and** Δbias LB > 0 (relative to home72, V is optimistic on U: it misses at least ≈ half of the 10.8 pp gap) | value training breadth, or search on the value |
| **POLICY-CANNOT-ACT** | Δbias's interval covers 0 **and** G_V ≥ +0.05 with G_V LB > 0 (the critic itself rates the level U positions at least 5 pp below the home72 ones, and what it misses is not detected) | policy breadth |
| **INCONCLUSIVE** | anything else | none from this read |

The two named verdicts are exclusive (Δbias LB > 0 against an interval covering 0). **Boundary rule** (root rule 8):
a statistic within 0.001 of its threshold, or a bound within 0.001 of 0, does not satisfy the condition it is tested
against. The rule is evaluated once, by `scripts/analyze.py`, and printed with every input.

**Why the rule reads only the difference.** V's ABSOLUTE bias on either set mixes the opponent mismatch (V was never
trained against Kakuna) with any set effect; Δbias holds the opponent fixed. bias_h and bias_U are reported, not
decided on. (A first draft also required bias_U LB > 0 for the first verdict and bias_U covering 0 for the second; a
synthetic check of this script, with a stand-in V that is a function of material alone and so cannot see the gap,
read INCONCLUSIVE under that draft because its overall level sat between the sets. Dropped before registration.)
The registered script reads that stand-in as CRITIC-MISCALIBRATED-ON-U (Δbias +0.126 [+0.057, +0.195], G_V −0.018),
and the same stand-in lowered by 0.12 on U as POLICY-CANNOT-ACT (Δbias +0.006 [−0.063, +0.075], G_V +0.102): both
branches are reachable, and the synthetic level-10 population reproduces the forensics' 0.609 / 0.501 (n 298 / 930).

**Not in the verdict:** the PastaMittens read (it replicates the opponent, not the team sample: the twins share
teams, seeds and our deterministic policy), every secondary measure, and the forensics README's own prediction, which
is scored separately: **P1 (forensics README, H-MID): V is less calibrated on U, ΔBrier ≥ +0.02** at level-10 states.

## 7. Predictions (registered; scored in README.md)

- **P1** (above): ΔBrier ≥ +0.02. HIT iff the point ≥ 0.02 and LB > 0.
- **P2 (this author, GUESS): CRITIC-MISCALIBRATED-ON-U.** The critic was trained on the same 719 pool teams as the
  policy, so it should share the memorisation: V on U level states sits near home72's (G_V ≤ +0.04) while outcomes
  differ by 10.8 pp. Confidence ≈ 0.55.
- **P3 (GUESS):** AUC on U is lower than on home72 at turn 10 (all states) by ≥ 0.03.

## 8. Reading notes and known limits (declared now)

- **What a calibrated V means here.** V is the critic's estimate of P\*'s win probability under P\*'s OWN future play,
  trained on self-play and league games, never against Kakuna. Its bias on home72 measures that opponent mismatch; the
  verdict reads the DIFFERENCE (Δbias), which holds the opponent fixed. A V that is unbiased on U and lower there
  ALSO says the turn-10 state is worse on U in ways material does not show; the turn-1 and turn-5 profile is reported
  so that reading can be checked.
- **The brief's lever map is not this read's to adjudicate.** The brief maps CRITIC-MISCALIBRATED-ON-U to "value
  training breadth or search on the value"; the forensics README reads the same branch as "search on V cannot help"
  (a search that reads V inherits V's error on U). This registration decides the BRANCH; which lever follows is
  reported as written and the tension is a FINDING.
- **Power (ESTIMATE).** ≈ 300 home72 and ≈ 900 U level-10 states; outcome noise dominates, so Δbias's interval is
  ≈ ±0.06 wide. A critic that sees none of the gap (Δbias ≈ +0.11) reads CRITIC-MISCALIBRATED; one that sees all of it
  (Δbias ≈ 0, G_V ≈ +0.11) reads POLICY-CANNOT-ACT; a split near half reads INCONCLUSIVE by design.
- **Rows are not independent across turns**; the secondary turn profile is descriptive and gets no multiplicity
  correction.

## 9. Cost

CPU only, no GPU, no training. Two sim_bridge builds (one heavy slot each), then replay + forward passes in ≤ 45-min
chunks on the agent cores (`nice -n 19`, `taskset -c 5-7,13-15`, `scripts/ops/mem_cap.sh`). READ-ONLY on `models/`,
the read dirs and the archive.
