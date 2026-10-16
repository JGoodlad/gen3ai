# The hidden-info study's never-seen arms, ORIGINAL against the BLIND U-arm build: the third-party comparison (2026-10-16)

**Status: a THIRD-PARTY comparison under this folder's [`PREREGISTRATION.md`](PREREGISTRATION.md)** (`8dd25833`,
committed before any arm output of the blind build existed). The comparing agent built neither implementation and changes
neither's verdicts. Every number here comes from `third_party_comparison.json`, written by `third_party_compare.py` (this
folder). The script mirrors the earlier third-party comparison (`../hidden_info_tier1_2026-10-13/replica_compare.py`,
`bc1e203b`). It reads:
- the original's frozen blinded table (sha `fc3aced3…`, checked on load), its arm key and its world files;
- the pilot's Phase 1 rows and bank;
- this folder's `imp_per_state.json` and the blind build's rows (`~/gen3ai_archive/hidden_info_u_replica/rows/`).

It is not blind: both builds' headline numbers, the red team and the earlier comparison had been read before the criteria were
written (they are stated in the script's `CRITERIA`, before its first run).

## In plain language

Two agents built the never-seen-Pokémon arms separately: F+U (honest worlds with the never-seen mons told TRUE) and O−U (the
true battle with only the never-seen mons re-drawn). The second agent saw neither the first one's code nor its results.
**On the 978 states where both built all six arms, they agree on everything the pre-registration asked about.**
- **The never-seen mons' shares agree.** Revealing only them closes 0.32 vs 0.31 of the gap. Hiding only them opens 0.83 vs
  0.92.
- **Rule R3 reads the same in both.** R3 says the never-seen mons beat the seen mons' hidden moves on both sides. Both
  numerators are positive and clear of zero in both builds.
- **The two absolute contrasts agree within 0.005.** The reveal-only gain is +0.022 vs +0.020, and the hide-only cost is
  +0.057 vs +0.058.

**So the never-seen half of "the moves are not the binding constraint, the never-seen Pokémon carry more" is now REPLICATED,
at the registered scorer, on the registered mix of states.** Three limits stay. First, the agreement holds at the
registered 8-rollout leaf only: at 4 or 1 rollouts the R3 conditions flip, in opposite directions in the two builds. Second,
under a stricter reading of the pre-registration's rule-8 clause it fails. The failing side there is the ORIGINAL's own
reveal-only lower bound (+0.0020), which that reading would call not clear. Third, the red team's strongest attack also
replicates: in BOTH builds the never-seen dominance lives in the games against scripted bots and is absent in the games
against our own snapshots.

## 1. What each build ran: like with like

| dimension | ORIGINAL (Tier 1, `fc3aced3…`) | BLIND U-arm build (`hiu/`, `5fafe276`) | consequence |
|---|---|---|---|
| checkpoints / states | `end2_s2001`, `prod_s2001`; the registered 600 each + the supplement | the same two; the registered 600 each | compared on the registered states only |
| arms | O, F, F+M, F+U, O−M, O−U (+ eleven more) | O, F, F+M, F+U, O−M, O−U | the six compared |
| move prior | Smogon usage, no learnset filter | Smogon usage filtered by the CURRENT learnset table (`hiu.fix`; sha `c90202a1…`) | the earlier replica's learnset HANDICAP is gone (§8: missing-pair hit 0.313 vs 0.321) |
| never-seen placement (**U2**) | slot-ordered draw, then gender-shuffled | drawn as the replica draws, then permuted UNIFORMLY onto the never-seen slots | the first never-seen slot's species distribution differs (§7); the IMP contrasts agree anyway |
| the opponent's pending root choice (**U1**) | kept by slot in every world | DROPPED in F / F+M / F+U (as the replica), KEPT in O−M / O−U | 6 states in C; dropping them changes nothing (§6) |
| consistency guard | the world screen + `world_guard` (later) | `hiu.guard`: contract + engine view, raises | the guard's one false positive (§3) |
| scorers | LEAF-K1 / K4 / K8 (+ TRUNC, CRITIC) | LEAF-K1 / K4 / K8 | LEAF-K8 is registered (§12.3); K4 / K1 secondary |
| the comparison set | | | **C = the registered states where BOTH hold all six arms: 978** (488 end2 / 490 prod; 716 battles; 574 bot-game / 404 snapshot-game). The original alone holds 1,057, the blind build 1,089 |
| bootstrap | | | ONE `t1_stats.Boot` for both (2,000 draws, seed 20261011, battles resampled within each checkpoint), every cross-build difference paired |

## 2. The criteria (pre-registered; this party's reading stated before the first run)

- **REGISTERED (the PREREGISTRATION):** AGREE iff (a) the two builds' 95 % intervals overlap on s_U, n_U and R3's two paired
  numerators, AND (b) the R3 conditions read the same, AND (c) |Δ| < 0.005 on σ_U = IMP(F+U) − IMP(F) and ν_U = IMP(O) −
  IMP(O−U).
- **R3's conditions** (as `unblind_t1.r_moves` registered them):
  - the reveal-only numerator IMP(F+U) − IMP(F+M) has a lower bound > 0;
  - the hide-only numerator IMP(O−M) − IMP(O−U) has a lower bound > 0;
  - the registered RATIO forms (each numerator / G) read the same.
- **Rule 8, this party's reading (PRIMARY).** The study's rule 8 (`t1_stats.EDGE` 0.0005): a lower bound must exceed
  +0.0005. The |Δ| rule is AGREE below 0.0045, DISAGREE above 0.0055 and BOUNDARY between (BOUNDARY is not clear, so it
  does not agree). This is how the earlier comparison applied §12.2's 0.005 point rule.
- **Rule 8, SENSITIVITY (the blind builder's reading).** The pre-registration's "a bound within ±0.005 is not clear" read as
  a ±0.005 band around 0 for the R3 numerators.
  - Why it is not primary: the original's own registered R3 used ±0.0005. That band would reverse the ORIGINAL's s-side on
    its own registered states (lower +0.0020), and a comparison's registration does not silently re-write the rule it
    compares.
  - Both readings are reported.
- **SECONDARY** (pre-registered, no verdict): R3 per opponent population (scripted bots / own snapshots) and per checkpoint.
  Also reported: the earlier comparison's broader criterion (every R-rule condition computed from a quantity, e.g. R-RIVALS
  U's B8), the M side, and the other leaf scorers.
- **TV:** per class and arm, the TV of the pooled draws against a PERMUTATION NULL that swaps each state's two 8-world sets
  with probability ½ (2,000 permutations, seed 20261016). AGREE iff p ≥ 0.05 (BOUNDARY within ±0.005); Holm-adjusted p is
  reported. Teeth: every class an arm keeps TRUE must read TV 0 exactly.
- **IDENTITIES:** at k = 6, F+U ≡ F and O−U ≡ O outcome for outcome; ruled on in §3.

## 3. The instrument checks, and the IDENTITY RULING

**Instrument (no outcome involved): PASS.**
- IMP_O is equal state by state on all 1,200 registered states at LEAF-K1, K4 and K8.
- The blind build's O rows equal the Phase 1 outcome strings (seeds 0–7) on 1,200 / 1,200.
- One scorer (`score_blind.imp_of`) reproduces every IMP of both builds from their per-world outcomes: 20,682 original and
  20,880 blind state-arm-scorer triples, 0 differences.
- The searched side's view (the seen mons and their revealed moves, Hidden Power normalised) is equal on 978 / 978 states of
  C.

**The identities, re-checked from the per-world outcomes of BOTH builds** (k = 6 registered states: end2 129, prod 103):

| build | checkpoint | F+U = F | F+U or F missing | O−U = O | O−U missing, O present | an outcome DIFFERS |
|---|---|---|---|---|---|---|
| original | end2 | 117 | 12 | 129 | 0 | **0** |
| original | prod | 84 | 19 | 103 | 0 | **0** |
| blind | end2 | 114 | 15 | 129 | 0 | **0** |
| blind | prod | 94 | 9 | 101 | **2** | **0** |

**THE RULING: a guard-caused EXCLUSION (a missing row), NOT an identity failure. The blind build is VALID.** Reasons:
1. **No outcome differs anywhere.** On every k = 6 state where both rows exist, F+U = F and O−U = O hold outcome for
   outcome (438 pairs in the blind build, 433 in the original). An identity is a statement about outcomes. A row that
   was never played cannot contradict it.
2. **The guard raised BEFORE any playout.** The two O−U rows (`prod …/random/loss_s2_003#p1#22`,
   `…/sentinel_1/loss_s1_005#p1#25`) carry no outcome.
3. **Its cause is independent of the never-seen construction.**
   - In all 1,200 registered states' banks, exactly ONE true set spells Hidden Power UNTYPED:
     `Charizard||leftovers|blaze|substitute,flamethrower,hiddenpower,focuspunch|Mild|4,,,252,,252`. Its IVs are blank,
     i.e. all 31, which in gen 3 is Hidden Power Dark.
   - The engine reports it typed (`hiddenpowerdark`); the packed truth says `hiddenpower`. The guard normalises Return /
     Frustration but not Hidden Power.
   - The guard raised on exactly the 4 states that hold this set, and on no other:
     - in F, its identity check of the TRUE team raised on all 4;
     - in F+M, O−M and O−U it raised on the 2 states at k = 6.
   - At k = 6 an O−U world IS the true team, so the guard rejected the truth itself. That is a property of the guard, not
     of any draw.
   - (The blind README's prose swaps the two spellings: the guard's message prints the ENGINE's list first. The README's
     table and reading are otherwise right.)
4. **The ORIGINAL reproduces the truth on these very states.** Its O−U on both states holds the same true team with Hidden
   Power spelled typed, and it equals O outcome for outcome. The two spellings are the same move in the engine.
5. **Nothing compared depends on the ruling.**
   - The blind F failed on all 4 states by the same guard, so they lie outside C, and every number in §4–§7 is computed
     without them.
   - Dropping all 4 states from C leaves it at 978. So does dropping every state of the two battles.
   - The registered verdict is AGREE in each case (§6).

**Stated openly:** the check as `hiu.score` scores it reads `HOLD = false` on prod, and the registration says "else the build
is invalid". This ruling reads the registration's intent ("outcome for outcome") and the cause of the missing rows. A
reader who takes the literal reading would void the build on two states that no compared number contains.

## 4. The agreement table (pooled end2 + prod, LEAF-K8, on C: 978 states / 716 battles)

| quantity | ORIGINAL | BLIND | paired orig − blind | registered criterion |
|---|---|---|---|---|
| **s_U = σ_U / G** | **0.320** [0.169, 0.452] | **0.309** [0.156, 0.446] | +0.011 [−0.161, +0.189] | **AGREE** |
| **n_U = ν_U / G** | **0.832** [0.703, 0.985] | **0.920** [0.774, 1.091] | −0.087 [−0.294, +0.115] | **AGREE** |
| **R3 reveal-only: IMP(F+U) − IMP(F+M)** | +0.0132 [+0.0020, +0.0239] | +0.0211 [+0.0099, +0.0334] | −0.0079 [−0.0218, +0.0057] | **AGREE** (R3 s-side holds in both) |
| **R3 hide-only: IMP(O−M) − IMP(O−U)** | +0.0305 [+0.0203, +0.0412] | +0.0240 [+0.0126, +0.0361] | +0.0064 [−0.0065, +0.0193] | **AGREE** (R3 n-side holds in both) |
| **σ_U = IMP(F+U) − IMP(F)** | +0.0217 [+0.0108, +0.0331] | +0.0195 [+0.0090, +0.0298] | +0.0022 [−0.0109, +0.0159] | **AGREE** (\|Δ\| 0.0022 < 0.005) |
| **ν_U = IMP(O) − IMP(O−U)** | +0.0566 [+0.0467, +0.0673] | +0.0580 [+0.0479, +0.0693] | −0.0014 [−0.0109, +0.0083] | **AGREE** (\|Δ\| 0.0014 < 0.005) |
| R3 ratio forms (s-side / n-side) | 0.19 [0.03, 0.35] / 0.45 [0.31, 0.61] | 0.33 [0.16, 0.54] / 0.38 [0.20, 0.58] | −0.14 [−0.37, +0.07] / +0.07 [−0.15, +0.28] | both clear of 0 in both builds |
| G = IMP(O) − IMP(F) | +0.0680 [+0.0578, +0.0784] | +0.0631 [+0.0525, +0.0741] | +0.0049 [−0.0050, +0.0146] | (B2 both; §12.2 point rule BOUNDARY) |
| IMP per arm O / F / F+M / F+U / O−M / O−U | +0.073 / +0.005 / +0.014 / +0.027 / +0.047 / +0.017 | +0.073 / +0.010 / +0.009 / +0.030 / +0.039 / +0.015 | O identical; O−M +0.0078 [−0.0015, +0.0170] | (O−M fails the point rule; every other arm within it or BOUNDARY) |

**THE REGISTERED VERDICT: AGREE.** All six intervals overlap. The R3 conditions are identical in both builds (s-side, n-side,
R3, and the ratio forms: all hold). σ_U and ν_U differ by 0.0022 and 0.0014.

**Sensitivity (the ±0.005-band reading): DISAGREE.** The ORIGINAL's reveal-only lower bound, +0.0020, falls inside the band,
so its R3 s-side reads "not clear" while the blind build's (+0.0099) is clear. The disagreement under this reading is the
original's own knife edge, not a difference between the builds: the paired difference contains 0. The same reading would
reverse the original's REGISTERED s-side on its own 1,057 / 1,097-state sets, which have the same lower bound (+0.0020–0.0021).

**What the registered criterion does not cover, reported (none of these is part of the verdict).**
- **R3 numerators by point.** Their point estimates differ by 0.0079 and 0.0064. That is more than 0.005, though the paired
  intervals contain 0. The pre-registration put the 0.005 point rule on σ_U and ν_U only.
- **R-RIVALS U's B8.** The original's end2 σ_U lower bound is −0.0012, against the blind build's +0.0017, so B8 differs.
  The R-RIVALS U verdict is **U-SHARED in both** (B3 fails: s_U < 0.5).
- **The M side** (handicap-free now) differs in places; see §8.

## 5. Per opponent population and per checkpoint (SECONDARY; pre-registered, no verdict)

R3 conditions are shown as `s / n`, where Y means the lower bound is > 0.0005.

| frame | n | build | G | s_U | n_U | s_M | n_M | R3 reveal-only | R3 hide-only | R3 |
|---|---|---|---|---|---|---|---|---|---|---|
| **scripted bots** | 574 | orig | +0.072 | 0.43 [0.26, 0.58] | 0.92 [0.76, 1.10] | 0.09 | 0.29 [0.17, 0.42] | +0.0251 [+0.0104, +0.0389] | +0.0453 [+0.0314, +0.0603] | Y / Y |
| | | blind | +0.071 | 0.49 [0.34, 0.63] | 0.88 [0.71, 1.06] | 0.01 | 0.38 [0.24, 0.52] | +0.0340 [+0.0198, +0.0482] | +0.0355 [+0.0197, +0.0510] | Y / Y |
| **own snapshots** | 404 | orig | +0.062 | 0.13 [−0.14, 0.36] | 0.69 [0.48, 0.95] | 0.19 | 0.54 [0.31, 0.80] | −0.0038 [−0.0217, +0.0143] | +0.0094 [−0.0064, +0.0257] | n / n |
| | | blind | +0.052 | −0.05 [−0.43, 0.25] | 1.00 [0.73, 1.41] | −0.10 | 0.85 [0.56, 1.23] | +0.0027 [−0.0166, +0.0215] | +0.0078 [−0.0105, +0.0255] | n / n |
| **end2** | 488 | orig | +0.059 | 0.24 [−0.02, 0.46] | 0.73 [0.52, 0.98] | 0.14 | 0.39 [0.19, 0.60] | +0.0061 [−0.0102, +0.0216] | +0.0201 [+0.0049, +0.0361] | n / **Y** |
| | | blind | +0.062 | 0.28 [0.03, 0.48] | 0.80 [0.61, 1.03] | 0.07 | 0.56 [0.35, 0.80] | +0.0128 [−0.0032, +0.0287] | +0.0147 [−0.0021, +0.0319] | n / **n** |
| **prod** | 490 | orig | +0.077 | 0.38 [0.19, 0.54] | 0.91 [0.75, 1.09] | 0.11 | 0.38 [0.24, 0.53] | +0.0202 [+0.0043, +0.0358] | +0.0408 [+0.0268, +0.0556] | Y / Y |
| | | blind | +0.064 | 0.34 [0.13, 0.54] | 1.03 [0.81, 1.30] | −0.12 | 0.51 [0.35, 0.71] | +0.0293 [+0.0121, +0.0467] | +0.0334 [+0.0168, +0.0500] | Y / Y |
| end2, bots | 281 | orig / blind | +0.055 / +0.064 | 0.20 / 0.44 | 0.97 / 0.78 | | 0.34 / 0.30 | +0.0107 / +0.0208 | +0.0348 / +0.0311 | n / Y both |
| end2, snapshots | 207 | orig / blind | +0.064 / +0.058 | 0.29 / 0.04 | 0.46 / 0.83 | | 0.45 / 0.96 | −0.0002 / +0.0019 | +0.0001 / −0.0076 | n / n both |
| prod, bots | 293 | orig / blind | +0.089 / +0.078 | 0.57 / 0.53 | 0.88 / 0.95 | | 0.26 / 0.44 | +0.0389 / +0.0467 | +0.0553 / +0.0398 | Y / Y both |
| prod, snapshots | 197 | orig / blind | +0.059 / +0.045 | −0.05 / −0.16 | 0.96 / 1.25 | | 0.64 / 0.71 | −0.0075 / +0.0035 | +0.0192 / +0.0239 | n / n both |

**The population read: the red team's strongest attack REPLICATES.**
- R3 holds on the bot-game states and fails on the snapshot-game states, in BOTH builds.
- The bot − snapshot difference of the absolute numerators is clear on both sides in both builds:
  - original: reveal-only +0.029 [+0.006, +0.051], hide-only +0.036 [+0.014, +0.057];
  - blind: +0.031 [+0.007, +0.055], +0.028 [+0.0045, +0.052].
- On snapshot states the blind build's n_M is 0.85 [0.56, 1.23]: hiding the moves alone opens most of the gap, as the red
  team found (0.58 on the original's registered 600).
- All four population × checkpoint cells read the same R3 conditions in both builds.

**The checkpoint read: R3 fails on end2 in both builds, and holds on prod in both.** One condition differs: end2's n-side.
The original's lower bound is +0.0049 (clear); the blind build's is −0.0021 (not). The paired difference there is +0.0054
[−0.0119, +0.0232]. The end-state checkpoint, today's production, is where U's dominance is weakest in either build.

## 6. Robustness

| variant | n | registered verdict | R3 (orig / blind) |
|---|---|---|---|
| C as compared | 978 | AGREE | holds / holds |
| without the 4 untyped-Hidden-Power states (the identity issue) | 978 | AGREE | holds / holds |
| without every state of the 2 battles holding that set | 978 | AGREE | holds / holds |
| without the 6 pending-choice states (spec choice U1) | 972 | AGREE | holds / holds |
| each build on its OWN all-six set (orig 1,057; blind 1,089; descriptive, different selections) | | — | holds / holds; s_U 0.32 / 0.31, n_U 0.82 / 0.92 |
| **LEAF-K4** (not registered) | 978 | **DISAGREE** | s-side fails / holds; n-side holds / holds. Hide-only paired difference **+0.0175 [+0.0034, +0.0313]** (clear) |
| **LEAF-K1** (not registered) | 978 | **DISAGREE** | holds / fails on both sides (blind numerators +0.0082 [−0.0011, +0.0172], +0.0044 [−0.0055, +0.0150]) |

**The scorer caveat (a FINDING).** At the registered LEAF-K8 the two builds agree. At LEAF-K4 the ORIGINAL loses R3's s-side
while the blind build holds it. At LEAF-K1 the reverse happens: the BLIND build loses both sides. At LEAF-K4 the hide-only
numerators also differ clearly between the builds.

The flips run in opposite directions, so neither build is systematically more U-favourable. R3 at a coarser leaf is a
knife edge in both. The registered claim is a LEAF-K8 claim and should be quoted as one.

## 7. TV: the drawn worlds' marginals against a permutation null (on C)

| arm | class | TV | null median / 95th pct | p (Holm) | read |
|---|---|---|---|---|---|
| O−U | **U.species** | 0.033 | 0.031 / 0.036 | 0.24 (0.72) | AGREE |
| O−U | U.item / U.ability / U.nature | 0.008 / 0.025 / 0.014 | 0.008 / 0.021 / 0.014 | 0.45 / 0.09 / 0.59 | AGREE |
| O−U | **U.move** (species:move) | 0.058 | 0.051 / 0.055 | 0.002 (0.010) | DIFFER, slightly |
| O−U | **U.first_slot_species** (the lowest never-seen slot) | 0.098 | 0.055 / 0.064 | 0.0005 (0.003) | DIFFER: **U2 by construction** |
| F, F+M | U.species, U.ability, U.nature | 0.012–0.033 | at their nulls | 0.14–0.81 | AGREE |
| F | U.item | 0.013 | 0.008 / 0.013 | 0.039 (0.23) | DIFFER raw, AGREE after Holm |
| F, F+M | U.move; U.first_slot_species | 0.060–0.061; 0.085–0.090 | 0.051–0.052; 0.055 | ≤ 0.001 | DIFFER |
| F, F+U | **M.move** (seen mons' hidden moves) | 0.078 | 0.034 / 0.037 | 0.0005 | DIFFER (0.066 without Hidden Power) |
| O−M | **M.move** | 0.097 | 0.034 / 0.036 | 0.0005 | DIFFER (0.071 without Hidden Power) |
| F, F+M, F+U | **S.nature** (seen mons' natures) | 0.133–0.140 | 0.073–0.078 | 0.0005 | DIFFER |
| F, F+M, F+U | I.item, A.ability | 0.003–0.044 | at their nulls | 0.03–0.82 | AGREE (F+M I.item p 0.029 raw, 0.17 Holm) |
| **teeth** | every class an arm keeps TRUE (F+U: U; F+M: M; O−M: U, I, A, S; O−U: M, I, A, S) | **0.0000** | 0 | 1 | exact |

What this says:
- **The never-seen draw's CONTENT agrees.** Which species, which item, ability and nature are all at the permutation null.
- **Its PLACEMENT differs, as U2 predicts.** The original's slot-ordered draw puts the popular species (Tyranitar 7.6 % vs
  6.5 %, Snorlax, Aerodactyl) on the first never-seen slot more often.
- **Its MOVES differ by a small margin.** TV 0.058 against a null of 0.051; the earlier replica's U.move TV was 0.111, with
  the learnset handicap.
- **Slot order inside U is bundled into U by the partition** (the red team's E7). The IMP contrasts agree despite the
  different placement, so its effect on σ_U / ν_U is below this comparison's resolution: |Δ| ≤ 0.0022, with paired
  intervals of about ±0.012.
- **The SEEN mons' draws (the M and S classes) still differ beyond noise** with the learnset fixed (§8).
  - Part of the M.move difference is Hidden Power's PRESENCE: the original draws it as a hidden move markedly more often
    (O−M: Metagross 1.3 % vs 0.1 %, Tyranitar 1.3 % vs 0.4 % of hidden-move draws). That is the spec's open question of
    whether Hidden Power's type belongs to M or to S.
  - S.nature differs on common sets (Modest Zapdos / Celebi, Adamant Tyranitar more often in the blind build).
  - Neither class is the U class this comparison was registered on.

**§12.2's "TV < 0.02" read as the pre-registration replaces it:** the U class's content passes; U.move and the first-slot
placement do not, and neither do the M and S classes. R-REPLICATE §12.2 as a whole therefore stays NOT MET (§9).

## 8. The learnset handicap is FIXED, and what remains on the M side

**The fix worked.** The F worlds' hit rate on the seen mons' hidden TRUE moves (7,936 missing-pair slots, 67,904 others,
on C):
- missing pairs: original **0.321**, blind **0.313** (the earlier replica: 0.000);
- other moves: 0.531 vs 0.508.

**The M side still differs** (secondary; the earlier comparison attributed most of it to the handicap):

| quantity | original | blind | paired | |
|---|---|---|---|---|
| ν_M = O − O−M | +0.0261 [+0.0180, +0.0345] | +0.0340 [+0.0248, +0.0432] | −0.0078 [−0.0170, +0.0015] | point rule FAILS (as in §16.9: −0.0067) |
| σ_M = F+M − F | +0.0086 | −0.0016 | +0.0101 [−0.0036, +0.0251] | point rule FAILS |
| n_M | 0.384 [0.273, 0.501] | **0.539** [0.399, 0.683] | **−0.154 [−0.316, −0.005]** | B4 FAILS / HOLDS: differs |
| s_M | 0.126 [−0.021, 0.259] | −0.025 [−0.216, 0.125] | +0.151 [−0.055, +0.381] | the 0.25 knife edge differs; B3 fails on both |

- **The ν_M gap is the same size and sign it had WITH the handicap** (−0.0078 now, −0.0067 then), so the handicap did not
  cause it. The earlier comparison's attribution is not supported. The candidate cause is UNVERIFIED: how each build ties
  a drawn Hidden Power's type to the IVs (§7).
- **It does not touch R3's verdict.** Both builds put n_U clearly above n_M on C.
- **It does touch the M-only cascade.** With the handicap fixed, the blind build HOLDS B4 (n_M ≥ 0.5) on C. So whether the
  moves' hide-only share is a majority is implementation-dependent at this power. MOVES-BINDING stays out of reach in both
  builds (B3 fails, s_M ≤ 0.13).

## 9. Rule 1's replication condition: what changes

| verdict (original, §16.8) | before (§16.9) | NOW | why |
|---|---|---|---|
| **R-MOVES = MOVES-REFUTED (R3): its never-seen (U) half** | NOT replicated (no U arms) | **REPLICATED (MET), scoped** | the pre-registered criterion AGREES on C at LEAF-K8: shares, both R3 numerators and σ_U / ν_U; the identity issue is an exclusion that no compared number contains. **Scope:** LEAF-K8; the registered bot-heavy mix; not under the ±0.005-band reading (the original's own knife edge); not at LEAF-K4 / K1; R3 fails in BOTH builds on end2 alone and on the snapshot-game states |
| R-MOVES: its M half | PARTLY MET | PARTLY MET (changed detail) | B3 fails in both (moves not THE constraint: SURVIVES); n_M now DIFFERS (B4 holds in the blind build), so "a minority by necessity" is not replicated |
| R-RIVALS U = U-SHARED | NOT MET | **PARTLY MET** | the same verdict on C; B8 differs (the original's end2 σ_U lower −0.0012) |
| the population split (red team §2.6 (b), post hoc) | single-implementation | **REPLICATED** | R3 holds on bots, fails on snapshots, in both; the bot − snapshot difference is clear on both sides in both |
| R-REPLICATE (§12.2) as a whole | NOT MET | NOT MET | the M side's point rule (ν_M, σ_M, IMP_O−M); the permutation-null TV DIFFERS on M.move, S.nature, U.move and the first-slot placement |
| Amendment 3 (U curves), R-STAGE-CLASS, R-DOSE's U parts | NOT MET | NOT COMPARED | outside the pre-registration (the per-world arrays exist; a later read could) |

So §13.2's gate opens for **one** claim: "on the registered mix, at the 8-rollout leaf, the never-seen Pokémon carry more of
the honest gap than the seen mons' hidden moves, on both the reveal-only and the hide-only side". It does NOT open for
"the never-seen Pokémon are the largest class wherever play is competent". There, both builds fail to detect it.

## 10. Confidence (the third party's; the author's §16.8, the earlier comparison's §16.9 and the red team's §16.10 figures stay as they wrote them)

| statement | third party, now | was |
|---|---|---|
| hidden TEAM information binds honest search (G ≈ +0.07) | ≈ 95 % | ≈ 95 % (§16.9) |
| the moves are THE binding constraint | ≈ 5 % | ≈ 5 % |
| the moves are a material share (n_M ≥ 0.25) | ≈ 90 % | ≈ 85 % (red team) |
| **the never-seen Pokémon are the largest single class, on the registered mix, at LEAF-K8** | **≈ 85 %** | ≈ 80 % (§16.9) / ≈ 70 % (red team) |
| … on competent-opponent (snapshot-game) states | ≈ 30 % | ≈ 35 % (red team) |
| U's dominance is a property of the bot-game states | ≈ 80 % | — |

## 11. What remains open

1. **E2, the competent-opponent bank** (40–65 CPU-h). It is now the decisive read. Both builds agree that U's dominance is
   a bot-game property, and the snapshot-game states (n = 404 on C) are too few to say what replaces it.
2. **E3, the second referee.** This comparison cannot see what both builds share: the truth rows, referee seeds 16–39, the
   true dice seed (the dice-silent leak) and the same 1,200 states.
3. **The M side's residual difference** (ν_M, M.move, S.nature) after the learnset fix. The Hidden Power type-in-M-or-S
   question is the first suspect, UNVERIFIED.
4. **Scorer dependence.** R3's conditions flip at LEAF-K4 / K1 in opposite directions. A tie-free or 16-world leaf (E5)
   would say whether LEAF-K8's agreement is robust or lucky.
5. **E7, the never-seen slot order.** The two builds place U differently (§7) and agree on the contrasts. A direct read
   would bound it.
6. **E6, the plateau replication.** Every read here is at 15M, PROVISIONAL.

## 12. FINDINGS (standing rule 7)

1. **The registered U-arm comparison AGREES** (LEAF-K8, C = 978): s_U 0.32 / 0.31, n_U 0.83 / 0.92, R3's numerators
   +0.013 / +0.021 and +0.031 / +0.024, σ_U |Δ| 0.0022, ν_U |Δ| 0.0014. R3 holds in both.
2. **The identity "failure" is a guard false positive on ONE true set's Hidden Power spelling** (prod Charizard, untyped HP
   with all-31 IVs). It is ruled an EXCLUSION. The original's O−U on the same 2 states equals O outcome for outcome, and the
   states lie outside C. The blind README's prose swaps the engine and packed spellings (cosmetic).
3. **The pre-registration's rule-8 clause is ambiguous.** "A bound within ±0.005 is not clear" can be read as a band on R3's
   numerators (the blind builder's reading). Under it the verdict is DISAGREE, and the cause is the ORIGINAL's own s-side
   lower bound +0.0020. The same band would reverse the original's registered R3. The primary reading here is the study's
   ±0.0005.
4. **The agreement is scorer-specific.** At LEAF-K4 the original fails R3's s-side and the hide-only numerators differ
   clearly (+0.0175 [+0.0034, +0.0313]); at LEAF-K1 the blind build fails R3 on both sides.
5. **The red team's population split replicates** in an independent build: R3 holds on bot-game states, fails on
   snapshot-game states, and the bot − snapshot difference is clear on both sides in both builds. The end-state checkpoint
   fails R3 in both builds.
6. **The learnset handicap is gone but the M-side disagreement is not.** ν_M differs by −0.0078 (it was −0.0067 with the
   handicap), n_M by −0.154 [−0.316, −0.005], and B4 now differs. The earlier comparison's attribution of the ν_M difference
   to the handicap is NOT supported.
7. **The drawn worlds' content agrees for U** (species, item, ability, nature at the permutation null). It DIFFERS on U.move
   (slightly), on the first never-seen slot's species (U2, by construction), on M.move (Hidden Power presence is part of it)
   and on the seen mons' natures (S).
8. **Not verified here.** No playout was run: the two guard-excluded rows were not re-played. The ruling rests on the
   original's equal outcomes and the guard's message. The blind build's code was read for the guard and the scorer only.
   Its world construction was checked through its outputs: teeth, view, marginals and identities. Rule 5b: nice 19, cores
   5–7 / 13–15, `mem_cap` 8 GB, peak 1.9 GB, no GPU.

## Reproduce

```bash
PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
  designs/research_state/measurements/hidden_info_u_replica_2026-10-15/third_party_compare.py   # → third_party_comparison.json (~3 min)
```
