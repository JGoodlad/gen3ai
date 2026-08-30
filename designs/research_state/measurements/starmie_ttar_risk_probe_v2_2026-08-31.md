# Starmie/Tyranitar risk probe v2 — revealed threat, and the bias as numbers

**2026-08-31 · owner-ordered rerun of
[`starmie_ttar_risk_probe_2026-08-30.md`](starmie_ttar_risk_probe_2026-08-30.md) with two
extensions.** v1's verdict (mask reads the KO boundary at ~⅓ amplitude, ~0.58 Surf floor,
argmax never flips) gets (1) a REVEALED-THREAT condition — Tyranitar shows all four of its
moves before the engineered decision — and (2) a fine two-sided sweep around the 80% payoff
crossover, in both conditions, delivering the bias as numbers. Cross-references: the risk
capstone (`designs/ai_v12/probe_risk_modulation_capstone.md`), SI-2's "credit binds" verdict
([`bot_gap_oracle_voi_2026-08-31.md`](bot_gap_oracle_voi_2026-08-31.md)), theory
`designs/learning/temperature_mixing_and_risk.md` §3.

Artifacts: `starmie_ttar_risk_probe_v2_2026-08-31.json` (every number, full captures) ·
`starmie_ttar_risk_probe_v2.py` (imports v1's construction; v1's artifacts untouched).
Checkpoint: `models/ai_v9_70_R3ACTION_0828/final_model.zip` (lineage:
`models/ai_v9_29_rev1_0823/final_model.zip`, base cell both conditions). Bridge: node,
`gen3customgame`, capture seed `[7,11,13,17]`.

## 1. The headline — the bias numbers, per condition

Truth per action is v1's one-ply CRN value: `E[win|Surf] = 1/16 + (15/16)·k/16` (k = Surf KO
rolls at the engineered H, measured crit fraction exactly 1/16), `E[win|Pump] = 0.80`
(measured: every recorded hit KOs; miss 0.2031). Truth-delta = E[win|Surf] − E[win|Pump].
True indifference sits between k=12 (0.7656) and k=13 (0.8242) — gen3's discrete rolls allow
nothing closer; the equality point is interpolated across that 5.9pp step (H=295 → H=296).

| bias number | HIDDEN (v1's condition) | REVEALED (all 4 moves shown) | Δ (revealed − hidden) |
|---|---|---|---|
| **1. P(Surf) at true equality** (0.5 = unbiased) | **0.787** | **0.803** | **+0.016** |
| **2. argmax flip threshold** | **no flip in range** — censored: > **0.7375 wp** deficit | no flip — censored: > **0.7375 wp** | none flips |
| **3a. amplitude dP(Surf)/d(truth-delta)**, OLS over all 22 points | **0.403** | **0.187** | **−0.216** |
| 3b. amplitude, local (\|delta\| ≤ 0.16, 11 pts) | 0.410 | 0.255 | −0.155 |
| 3c. span ratio (v1's "~⅓" made exact) | 0.347 | 0.172 | −0.175 |

Value-stack deltas, paired over all 22 sweep points (revealed − hidden): **ΔP(Surf) +0.064 ·
ΔV −1.320 · Δwin-prob −0.0142**.

**Reading.** The mask's bias toward the safe move at true equality is ~+0.29 of probability
mass (0.787 vs the unbiased 0.5), and revealing the full threat makes it marginally *worse*
(+0.016). The argmax never flips in either condition even at a 0.74-wp true deficit (Surf
0.0625 vs Pump 0.80). And the sharpest v2 result: **revealing the threat roughly HALVES the
amplitude at which the mask tracks the truth** (0.403 → 0.187 global; the low-truth Surf floor
rises from ~0.53 to ~0.69). Threat visibility is not the missing input — the value stack
registers the reveal while the action split gets *less* truth-sensitive.

## 2. Extension 1 — the revealed-threat condition

### Construction (deltas off v1; everything else is v1's imported machinery)

Both conditions use the SAME four-move CB Tyranitar (Earthquake / Rock Slide / Focus Punch /
Double-Edge — every move Protect-blockable, none self-damaging when blocked) and the SAME
choreography: Marshtomp (now Endeavor/Growl/Protect) fronts a deterministic 9-turn reveal
phase — three pivot stints, each "Tyranitar clicks a move into Marshtomp's fresh Protect →
pivots out to Koffing → Koffing pivots back" — then Endeavors the still-untouched Tyranitar
on the final pivot-in turn (switches resolve first, so Endeavor executes against the incoming
Tyranitar at full HP). The v1 explosion-into-Protect parity chain follows unchanged. The ONLY
difference between conditions is which move the stints click: **hidden = Earthquake ×3** (only
EQ ever shown, as in v1) vs **revealed = Rock Slide / Focus Punch / Double-Edge** (EQ revealed
killing Marshtomp/Grimer). Decision-turn state is identical either way — 5/5 faints, Starmie
full, Tyranitar at the engineered H, choice-locked into EQ — and asserted per battle, plus a
new reveal assert (`{'earthquake'}` vs all four). Both conditions land on turn 22.

### The base cell (H=295, k=13, truth Surf 0.8242 / Pump 0.8000)

| readout | HIDDEN | REVEALED | Δ |
|---|---|---|---|
| P(Surf) | 0.7901 | 0.8049 | +0.015 |
| P(Hydro Pump) | 0.2098 | 0.1951 | −0.015 |
| argmax | Surf | Surf | — |
| V(s) | 20.82 | 19.49 | **−1.33** |
| win-prob head | 0.8977 | 0.8822 | **−1.55pp** |
| α top-5 | RS .309 / DD .301 / **EQ .182** / DE .090 / Leer .087 | RS .343 / DD .187 / **Tbolt .180** / **EQ .146** / DE .102 | α(EQ) −0.036 |

**The question v2 was ordered to answer — does making the danger obvious move the action
split, the value stack, or neither?** Answer: **the value stack, coherently; the action split,
barely — and in the unhelpful direction.** R3ACTION's V drops 1.33 and win-prob drops 1.55pp
when four attack moves are revealed (the board got more dangerous and the value stack says
so), while P(Surf) moves +1.5pp — no re-weighting toward caution, and across the sweep the
truth-tracking amplitude *halves*. v1's knows≠uses split, now with the information supplied
explicitly: giving the policy the threat does not repair the action split. This is the same
shape as SI-2's verdict on the bot gap — supplying the missing information (there: the oracle
opponent; here: the revealed moveset) moves the value stack, but what binds is credit/valuation
inside the policy head, not information.

**α incidental, banked honestly:** at 4/4 moves revealed, the α read still places 0.37 of its
mass on moves NOT in the revealed set (Dragon Dance .187 + Thunderbolt .180), only 0.146 on
Earthquake — the move Tyranitar is choice-locked into (the lock is unknowable from the obs;
the reveal-completeness is not) — and the five believed seats OMIT the revealed Focus Punch
outright while carrying the two unrevealed candidates. The move-belief/α pipeline carries no
"they have exactly four and I have seen all four" constraint, so reveal-completeness never
collapses the posterior. Not a defect of this probe — a measured property of the belief stack
worth knowing.

### Lineage (rev-1, `ai_v9_29_rev1_0823`, base cell)

| readout | HIDDEN | REVEALED |
|---|---|---|
| P(Surf) / P(Pump) | 0.9810 / 0.0190 | 0.9813 / 0.0187 |
| V(s) | 15.76 | **20.27** |
| win-prob | 0.8698 | **0.9281** |

rev-1's action split is byte-flat across conditions (and replicates its v1 sharpness, 0.981,
on the new choreography), while its value stack moves the OPPOSITE direction from R3ACTION's:
+4.5 V, +5.8pp win-prob *up* on reveal. Two checkpoints, one state, opposite-signed value
responses to the same information, neither moving the action split — a direction to check
against population instruments, not a conclusion.

## 3. Extension 2 — the fine two-sided sweep (2 × 22 points)

v1's 17 points (k = 16/16 → 0/16) plus micro-steps H ∈ {294, 295, 297, 298, 299} bracketing
the crossover. One battle per point per condition, same capture seed. Full tables in the JSON;
the crossover-adjacent region:

| H | k | E[KO\|Surf] | truth-delta | P(Surf) hidden | P(Surf) revealed |
|---|---|---|---|---|---|
| 293 | 13 | 0.8242 | +0.0242 | 0.8037 | 0.8134 |
| 294 | 13 | 0.8242 | +0.0242 | 0.7970 | 0.8092 |
| 295 | 13 | 0.8242 | +0.0242 | 0.7901 | 0.8049 |
| **← true indifference (interpolated)** | | 0.8000 | 0 | **0.787** | **0.803** |
| 296 | 12 | 0.7656 | −0.0344 | 0.7831 | 0.8005 |
| 297 | 12 | 0.7656 | −0.0344 | 0.7757 | 0.7960 |
| 298 | 12 | 0.7656 | −0.0344 | 0.7683 | 0.7914 |
| 299 | 12 | 0.7656 | −0.0344 | 0.7605 | 0.7866 |

Range endpoints: hidden P(Surf) 0.8535 (k=16) → 0.5282 (k=0); revealed 0.8456 → 0.6937.
Argmax = Surf at all 44 cells. Win-prob spans 0.9027 → 0.8783 (hidden) / 0.8867 → 0.8691
(revealed) — the ~40× compression of v1, again.

**The micro-step finding — the mask prices the HP bar, not the roll table.** Per-HP-point
steps in P(Surf) across H = 293…299: hidden −0.67, −0.69, −0.71, −0.73, −0.75, −0.77 pp;
revealed −0.42, −0.43, −0.44, −0.45, −0.47, −0.48 pp. The 295→296 step — where the TRUE Surf
value drops 5.9pp — is indistinguishable from its neighbours, where truth does not move at
all. Locally there is **zero excess response at the KO boundary**: the policy's instrument is
a smooth function of displayed HP, and its correlation with the truth across the full sweep
exists only because the KO fraction is itself monotone in HP. v1's "genuinely reading the KO
boundary" is therefore sharpened: the model reads the *HP bar* everywhere at ~0.7pp per HP
point (hidden), and the discrete roll structure that actually decides the gamble is invisible
to it at this resolution. A boundary-aware evaluator would show a kink at 295/296; there is
none, in either condition.

## 4. Monte-Carlo verification (crossover-adjacent cells, v2 choreography)

Unseeded battles, forced first move at the decision, always-Surf continuation, per-battle
decision-state asserts (n = 250 per arm):

| arm | analytic | MC p(win) | Wilson 95% |
|---|---|---|---|
| H=295 (k=13), Surf | **0.8242** | 0.8240 | [0.772, 0.866] |
| H=295, Pump | **0.8000** | 0.8160 | [0.763, 0.859] |
| H=299 (k=12), Surf | **0.7656** | 0.7480 | [0.691, 0.798] |
| H=299, Pump | **0.8000** | 0.7760 | [0.720, 0.823] |

Every analytic sits inside its arm's Wilson 95% — the v2 choreography (reveal stints, the
delayed switch-in Endeavor) still lands the engineered probabilities exactly. At k=12 the MC
point estimates order Pump above Surf (0.776 vs 0.748), consistent with the true preference
flip across the crossover, though the two intervals overlap at n=250.

## 5. Honest caveats

1. **One scenario, one checkpoint family.** Everything here is an existence demonstration on
   a constructed, off-distribution state (gen3customgame, explosion parade, 4-attack Starmie
   vs 0-Spe CB Tyranitar). The obs pipeline is genuine; the state distribution is not.
2. **Discrete rolls limit the crossover resolution.** The achieved fractions straddling
   equality are 12/16 (0.7656) and 13/16 (0.8242); bias #1 interpolates linearly across that
   5.9pp truth step. Given §3's micro-step finding — the mask responds only to HP inside that
   window — the interpolated "P(Surf) at true equality" is operationally "P(Surf) at ~86.3%
   opponent HP", and the achieved fractions are reported wherever the number is used.
3. **The slope numbers are HP-mediated.** dP(Surf)/d(truth-delta) is a real description of
   the sweep, but §3 shows the mechanism is smooth HP pricing, not boundary evaluation — the
   amplitude ratio would not survive a re-parameterization that decouples HP from KO fraction
   (e.g. a different move's roll table). Report and use it as a descriptive gain, not as
   evidence the policy computes KO fractions at ⅓ gain.
4. **Belief caveat (v1's, still binding):** the truth columns use the engineered hidden
   HP/item; the model's subjective KO fractions run through its spread/item beliefs and may
   place the crossover elsewhere. The monotone shape, the no-kink finding, and the
   revealed-vs-hidden deltas are robust to this; the exact "should have flipped at H=X" is not.
5. **The two conditions differ in deep history as well as in revealed moves** (the reveal
   stints' |move| identities, turns 1–8). Turn structure, damage stream, and decision-turn
   board are matched; the event window at the decision is 13+ turns past the stints.
6. **v2-hidden is not v1-base byte-for-byte** (Marshtomp carries Protect, Tyranitar four
   moves, nine extra turns): base P(Surf) reads 0.790 here vs v1's 0.823. The v1 shape
   replicates fully (monotone, no flip, ~0.53 floor vs 0.58); the ~3pp level difference is
   itself a mild state-sensitivity reading — same decision geometry, different prelude, ~3pp
   of mask movement.
7. **Why the revealed curve is flatter is a hypothesis, not a measurement.** The α read
   shifts its believed-intent mass away from the lethal EQ (0.182 → 0.146) toward the
   revealed non-lethal moves when all four are shown; if the policy prices Surf-failure
   through that mix, obviousness *dilutes* the perceived lethality. Plausible, consistent
   with the numbers, unproven here.

## 6. Rerun

```bash
export PYTHONPATH=$PYTHONPATH:src
python designs/research_state/measurements/starmie_ttar_risk_probe_v2.py --phase all
```

Phases are resumable (`capture`/`sweep`/`mc`/`analyze`, `--phase smoke` for a 2-battle
sanity pass); the JSON is incrementally saved. v1's script and JSON are required (the roll
tables are read from `starmie_ttar_risk_probe_2026-08-30.json`) and are not modified.
