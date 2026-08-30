# Starmie/Tyranitar constructed risk probe — the policy's mask at an engineered equal-EV gamble

**2026-08-30 · owner-ordered.** The constructed-scenario complement to
[`designs/ai_v12/probe_risk_modulation_capstone.md`](../../ai_v12/probe_risk_modulation_capstone.md)
(theory: [`designs/learning/temperature_mixing_and_risk.md`](../../learning/temperature_mixing_and_risk.md) §3).
Where the capstone's accuracy-tradeoff curve is the *population* instrument over eval traces,
this is one hand-built decision, verified to the damage roll: a real bridge battle steered to a
5-fainted-each 1v1 endgame — our Starmie vs their Choice Band Tyranitar at an Endeavor-set exact
HP — where **Surf (100% acc, 13/16 rolls KO) and Hydro Pump (80% acc, every roll KOs) have as
near-equal an expected KO probability as gen3's discrete rolls allow**, and the policy's masked
probability distribution is read at that decision.

Artifacts: `starmie_ttar_risk_probe_2026-08-30.json` (every number below, plus full tables) ·
`starmie_ttar_risk_probe.py` (the construction script, reruns end-to-end) — both beside this file.
Checkpoint: `models/ai_v9_70_R3ACTION_0828/final_model.zip` (loadable at HEAD; lineage
comparison: `models/ai_v9_29_rev1_0823/final_model.zip`). Bridge: node, `gen3customgame`
(no species clause for the filler chain; the protocol stream is structurally identical to
gen3ou for the parser/encoder). Capture seed `[7,11,13,17]`.

## 1. The engineered numbers (damage-probe-verified, not calculated)

All damage figures below are the sim's own realized damage from `utils/bridge/damage_probe.js`
(omniscient stream, exact both-side HP, the sim's own stats — zero measurement confounds),
320 seeds per attacking move; crits and misses separated by observation.

| quantity | value |
|---|---|
| Starmie (base) | Modest, 4 HP / 252 SpA / 252 Spe, **Def IV 0**, Leftovers — 262 HP |
| Tyranitar | Adamant, 4 HP / 252 Atk, **0 Spe**, Choice Band (hidden) — 342 max HP |
| Surf roll table (16 rolls) | 285 288 292 295 299 302 305 309 312 315 319 322 325 329 332 336 |
| **Engineered H (Ttar current HP)** | **295** = 86.3% — set by Endeavor from a 295-max-HP Marshtomp |
| Surf KO rolls at H=295 | **13/16 = 81.25%** (crit always KOs; measured crit fraction 0.0625 = 1/16 exactly) |
| Hydro Pump at H=295 | **every hit KOs** (all 255 recorded hits were full-342 KOs; measured miss 65/320 = 0.2031) |
| **E[KO \| Surf]** | **0.8242** = 1/16 + (15/16)(13/16) |
| **E[KO \| Pump]** | **0.8000** = accuracy |
| Retaliation | CB Earthquake KOs base Starmie on **96/96** probe rolls (symmetric fatality) |

The 2.4pp gap is the honest floor: gen3 has 16 discrete rolls plus a 1/16 crit, so the closest
achievable pair straddling equality is k=13 (0.8242) vs k=12 (0.7656) against Pump's fixed
0.8000. We take k=13 and report it — Surf is *slightly strictly better* in the base cell, not
exactly equal. (Do not read "indifference" as the correct target; the correct preference is a
mild Surf lean.)

**Three gen3 mechanics findings the engineering flushed out** (each cost one design iteration,
each verified by the probe):

1. **Crunch is SPECIAL in gen3** (Dark type; phys/spec split is by type until gen4). A Choice
   Band boosts it not at all — CB Crunch = plain Crunch = max non-crit roll 220 vs Starmie's
   262 HP. The guaranteed retaliation had to be CB **Earthquake** into a **Def-IV-0** Starmie
   (96/96 probe rolls KO).
2. **poke-env cannot track duplicate species on one side** (`get_pokemon`'s ident re-keying
   matches by species; four nicknamed Shedinja produced `KeyError: 'p1: Shed1'` at the first
   request). The elegant 5×-Shedinja-dies-to-sandstorm prelude is unimplementable against our
   own battle layer; the replacement is the explosion chain below, all species distinct.
3. **A "maximal miss-is-fatal" cell is unconstructible with a faster attacker.** Moves are
   chosen simultaneously; the defender's retaliation for the decision turn is committed before
   the attack resolves, and in every later turn a surviving Starmie kills the sliver Tyranitar
   before the retaliation resolves (it is faster). So no opponent policy can punish a Pump miss
   *more* than a Surf low-roll on the decision turn itself — the failure-mode asymmetry is
   exactly the RECOVERABLE shape (§3), never the maximal 1.0-vs-0.8 shape. This is a real
   fact about simultaneous-move games, not a scripting limitation.

## 2. The construction (how a real battle reaches the position deterministically)

State-driven scripts keyed to **faint-count parity, never turn numbers** — a crit-shifted
timeline self-heals:

- **T1**: our Marshtomp (Jolly, max HP = H = 295, faster than the 0-Spe Ttar) **Endeavors**
  Tyranitar → Ttar HP := 295 exactly. Ttar's CB EQ then kills Marshtomp (T1–T3).
- **The filler chain**: our Golem/Forretress/Camerupt/Grimer vs their
  Koffing/Weezing/Graveler/Pineco/Exeggutor. Each turn one side's filler **Explodes into the
  other side's first-use Protect** (100%; the exploder faints even when blocked — probe-verified,
  protector takes zero). Parity rules: ours explode iff `our_faints < their_faints`, theirs iff
  `their_faints <= our_faints`. Tyranitar sits on the bench from T3 (voluntary switch, which
  also clears the choice lock and the Endeavor-turn Growl) until their 5th faint forces it back.
- **The last our-filler is Grimer** (Poison, slower than Ttar): it dies to a guaranteed-overkill
  super-effective CB EQ *before its own move executes*, so Tyranitar is never touched after the
  Endeavor. Starmie (slot 6) enters **on the decision turn itself** — zero sandstorm ticks,
  exactly full HP.
- Decision-state asserts (fire loudly, every battle): faints 5/5, Starmie full, opponent
  Tyranitar at exactly 295/342, decision reached exactly once.

At the decision the model has seen: Ttar's EQ (nothing else — no CB reveal), both full rosters,
13 turns of history. The obs is a genuine `RLPlayer.embed_battle` product — the tracker
(event window, pair history, recency, progress clock) is fed on every scripted decision of the
prelude exactly as live play would.

## 3. Results — the mask, the heads, the truth

### BASE — symmetric fatality (any failure = certain loss)

Full 11-dim masked probability distribution, verbatim (actions 0–5 switches, 6–9 = request-order
moves [surf, hydropump, icebeam, recover], 10 struggle):

```
mask : [0, 0, 0, 0, 0, 0, 1,        1,        1,       1,       0]
probs: [0, 0, 0, 0, 0, 0, 0.822515, 0.177456, 0.000028, 0.000001, 0]
                          SURF      HYDRO PUMP ICE BEAM  RECOVER
```

| readout | value | truth |
|---|---|---|
| P(Surf) | **0.8225** | E[win\|Surf] = **0.8242** |
| P(Hydro Pump) | **0.1775** | E[win\|Pump] = **0.8000** |
| argmax | Surf ✓ (the strictly-better action) | |
| V(s) (shaped critic) | 20.68 | — |
| **win-prob head** | **0.8966** | 0.8242 under its own chosen action (+7.2pp optimistic) |
| α (opp-intent) top | Rock Slide 0.31, DD 0.30, EQ 0.18 | opponent is CB-locked EQ (unknowable from the obs) |

The model does not treat this as a coin flip: it puts 4.6:1 on the accurate move, with the
nuke-but-miss option carrying the entire remainder and the non-solutions (Ice Beam, Recover) at
1e-5 — the action geometry is understood. Under the truth the preference *direction* is correct
(Surf is +2.4pp better), but nothing in a 0.82/0.18 mask can be read as "calibrated to the
+2.4pp edge" — a policy trained by PPO has no obligation to match action-probabilities to
action-values, only to order them. Report is the mask, not a calibration claim.

### RECOVERABLE — bulky Starmie (252 HP/252 SpA, 324 HP) survives exactly one non-crit EQ

Same H, same surf table (SpA unchanged — probe-verified). Now Surf-fail is convertible (sliver
Ttar dies to the follow-up Surf) while Pump-miss must re-roll the gamble under a
one-more-EQ clock:

```
probs: [0, 0, 0, 0, 0, 0, 0.822092, 0.177863, 0.000043, 0.000002, 0]
```

| readout | value | truth (analytic exact, always-Surf continuation) | truth (MC) |
|---|---|---|---|
| P(Surf) | **0.8221** | E[win\|Surf] = **0.9890** | 0.9825 [0.964, 0.992] |
| P(Hydro Pump) | **0.1779** | E[win\|Pump] = **0.9545** | 0.9625 [0.939, 0.977] |
| V(s) | 22.15 (> base's 20.68 ✓) | | |
| win-prob head | **0.9224** | 0.9890 under Surf (−6.7pp pessimistic) | |

The mask is **byte-identical to base at the 3rd decimal** (Δ < 0.0005) although our own spread
and HP bar — the only obs difference — changed substantially, and although the true stakes
changed from "13/16-or-die" to "certain-recovery-vs-clocked-regamble". The preference again
points at the truth-better action (Surf, +3.5pp). The win-prob head *did* move (+2.6pp for the
bulk) and V moved (+1.5), so the value stack sees the bulk; the policy head's action split
simply does not re-weight on it.

### MUST-KO-NOW / MISS-IS-FATAL (variant 3) — resolved analytically, not run

The "only an immediate KO wins, failure = loss both ways" cell **is the base cell** — CB EQ
makes any failure lethal on the decision turn itself, so base already carries that framing and
its mask is the variant-3 mask. The stronger asymmetric cell (Pump-miss loses, Surf-fail
survives, truths 1.0 vs 0.8) is **unconstructible** in this geometry (§1 finding 3:
simultaneous choice + the faster attacker auto-converting every sliver). A conditional
opponent policy ("EQ at high HP, pass at sliver") was designed and discarded after working the
turn order through: its branch only fires in lines Starmie has already won, collapsing it to
the RECOVERABLE cell exactly. An "our-HP behind" framing (Starmie at ~50%) was also cut: every
deterministic self-chip route (Substitute) leaves a confounding volatile up at the decision,
and every clean route is roll-dependent. Both cuts are documented rather than approximated.

### Lineage: rev1 (ai_v9_29, 2026-08-23) on the identical base obs (same obs sha `76e58fd0`)

```
probs: [0, 0, 0, 0, 0, 0, 0.981182, 0.018727, 0.000057, 0.000034, 0]
```

P(Surf) 0.9812 / P(Pump) 0.0187, win-prob 0.8699, V 15.70. Same argmax, **much sharper**: the
week's training between rev1 and R3ACTION (which included the R-ladder's counterfactual work)
moved the same decision from 52:1 toward 4.6:1 — the newer policy holds a fat, non-vestigial
probability on the high-power/low-accuracy option where the older one had effectively
extinguished it. Two checkpoints, one state: a direction to check against the capstone's
population curves, not a conclusion.

### The HP sweep — mask vs true KO fraction (the mask-vs-probability curve)

One battle per point, H swept across the entire Surf roll table (16/16 KO rolls down to 0/16;
Pump KOs on every roll at every point; opp HP display moves 83%→99.7%):

| H | opp HP% | surf KO rolls | E[KO\|surf] | E[KO\|pump] | **P(surf)** | **P(pump)** | argmax | V | win-prob |
|---|---|---|---|---|---|---|---|---|---|
| 284 | 83.0 | 16/16 | 1.000 | 0.8 | 0.880 | 0.120 | surf | 20.91 | 0.902 |
| 286 | 83.6 | 15/16 | 0.941 | 0.8 | 0.871 | 0.129 | surf | 20.87 | 0.901 |
| 289 | 84.5 | 14/16 | 0.883 | 0.8 | 0.856 | 0.144 | surf | 20.80 | 0.899 |
| 293 | 85.7 | 13/16 | 0.824 | 0.8 | 0.834 | 0.166 | surf | 20.72 | 0.898 |
| 296 | 86.5 | 12/16 | 0.766 | 0.8 | 0.816 | 0.184 | surf | 20.66 | 0.896 |
| 300 | 87.7 | 11/16 | 0.707 | 0.8 | 0.787 | 0.213 | surf | 20.58 | 0.894 |
| 303 | 88.6 | 10/16 | 0.648 | 0.8 | 0.762 | 0.238 | surf | 20.52 | 0.893 |
| 306 | 89.5 | 9/16 | 0.590 | 0.8 | 0.735 | 0.265 | surf | 20.45 | 0.892 |
| 310 | 90.6 | 8/16 | 0.531 | 0.8 | 0.692 | 0.308 | surf | 20.35 | 0.890 |
| 313 | 91.5 | 7/16 | 0.473 | 0.8 | 0.656 | 0.344 | surf | 20.28 | 0.889 |
| 316 | 92.4 | 6/16 | 0.414 | 0.8 | 0.623 | 0.377 | surf | 20.21 | 0.887 |
| 320 | 93.6 | 5/16 | 0.356 | 0.8 | 0.615 | 0.385 | surf | 20.15 | 0.886 |
| 323 | 94.4 | 4/16 | 0.297 | 0.8 | 0.608 | 0.392 | surf | 20.10 | 0.885 |
| 326 | 95.3 | 3/16 | 0.238 | 0.8 | 0.602 | 0.398 | surf | 20.05 | 0.884 |
| 330 | 96.5 | 2/16 | 0.180 | 0.8 | 0.593 | 0.407 | surf | 19.98 | 0.882 |
| 333 | 97.4 | 1/16 | 0.121 | 0.8 | 0.587 | 0.413 | surf | 19.95 | 0.881 |
| 337 | 98.5 | 0/16 | 0.063 | 0.8 | **0.579** | **0.421** | **surf** | 19.90 | 0.880 |

**Reading — direction-sensitive, magnitude-blind, and the argmax never flips.**

1. **The slope has the correct sign at every one of the 17 points** — P(Surf) falls
   monotonically (0.880 → 0.579) as the true Surf KO fraction falls (1.00 → 0.06). The policy
   is genuinely reading the KO boundary off the HP bar: a ~1% opp-HP step it can only price
   through the damage machinery moves the action split every single time. This is the
   constructed-scenario analogue of a *falling* capstone accuracy-tradeoff curve — the
   risk-relevant quantity is in there.
2. **The magnitude is a fraction of what the truth demands, and it saturates.** At the true
   crossover (KO fraction ≈ 0.786, between H=293 and 296) the split should pass through
   ~50/50; the model is at ~83/17. Below ~5/16 the curve flattens toward a Surf floor of
   ~0.58 — the last five points (true E[KO|surf] 0.36 → 0.06, a catastrophic range) move
   P(Surf) by only 3.6pp.
3. **Greedy play never switches moves.** Argmax = Surf at all 17 points, including H=337 where
   Surf's true value is 0.063 against Pump's 0.800 — greedy play donates 74pp of win
   probability, and even temperature-1 sampling recovers only 0.579·0.063 + 0.421·0.800 =
   **0.37 of the 0.80 available**.
4. **The win-prob head compresses the same structure ~40×.** Truth under optimal play falls
   1.00 → 0.824 over the first four points (then flat at 0.80); the head moves
   0.902 → 0.898 over the same points and 0.902 → 0.880 over the whole sweep — right level,
   almost no resolution on the boundary. Against the policy's own greedy behavior (true value
   0.063 at the top) it is not calibrated at all: it prices the state as if the good action
   will be taken while the policy head takes the bad one.

**Belief caveat, stated before anyone over-reads point 3:** the truth columns use our
engineered (hidden) Tyranitar HP/item; the model's *subjective* KO fractions run through its
spread/item beliefs and may place the crossover elsewhere. Points 1–2 are robust to that (a
monotone belief still crosses somewhere in-range); the precise "should have flipped at H=X"
claim is not. What is belief-independent is the outcome: at the top of the sweep the greedy
policy actually loses these games.

## 4. Monte-Carlo verification

Unseeded battles (fresh dice each; the prelude is roll-insensitive by construction), forced
first move at the decision, always-Surf continuation, per-battle decision-state asserts:

| arm | n | wins | MC p(win) | Wilson 95% | analytic |
|---|---|---|---|---|---|
| base, Surf | 250 | 206 | 0.8240 | [0.772, 0.866] | **0.8242** ✓ |
| base, Pump | 250 | 197 | 0.7880 | [0.733, 0.834] | **0.8000** ✓ |
| recoverable, Surf | 400 | 393 | 0.9825 | [0.964, 0.992] | **0.9890** ✓ |
| recoverable, Pump | 400 | 385 | 0.9625 | [0.939, 0.977] | **0.9545** ✓ |

Every analytic truth sits inside its arm's Wilson 95% interval — the engineered probabilities
are real, not calculated hopes. (The recoverable-Pump analytic uses the 2×-roll crit
approximation noted above; the MC needs no such assumption.) One process note, kept because it
is the fail-loud doctrine earning its keep: the first Pump MC arms silently played *Surf* — the
forced-move lookup used the label `"pump"` where the move id is `"hydropump"`, and the miss
fell through to poke-env's default order (= slot 1 = Surf). The tell was a "Pump" arm matching
Surf's win rate. The script now asserts the scripted move exists rather than defaulting; both
arms were rerun.

## 5. Honest interpretation

1. **This is an existence demonstration, not a population statistic.** One constructed state
   (plus a 17-point sweep along one axis of it), two checkpoints. The capstone's trace-derived
   accuracy-tradeoff curve is the population version; this probe pins one point of it to
   ground truth with the damage-probe's exactness.
2. **What the base cell can and cannot say.** At near-equal EV any mask is "legal"; the
   informative readouts are (a) the direction (Surf lean = correct, both checkpoints), (b) the
   sharpness (0.82 vs rev1's 0.98 on identical bytes), and (c) the sweep — where the true KO
   fraction moves 1.00 → 0.06 and the mask follows with the right sign at all 17 points but
   at ~⅓ of the required amplitude, saturating at a ~0.58 Surf floor with the argmax never
   flipping. **The single most interesting finding is that pair: the boundary is *read* but
   not *acted on* — a smooth, monotone, correctly-signed preference curve that never crosses
   0.5 even where the preferred action's true value is 0.06 vs 0.80.**
3. **Knows-vs-uses, localized.** The win-prob head and V respond to the bulk change (+2.6pp,
   +1.5) while the action split does not move at all (Δ < 0.0005). At this single state the
   value stack carries more scenario information than the policy head re-weights on — the same
   knowing≠using split the bait verdict measured, here visible inside one decision.
4. **Win-prob head calibration at this state**: +7.2pp optimistic in the fatal cell, −6.7pp
   pessimistic in the recoverable cell. Two states; the `calibration` CLI owns the population
   claim.
5. **The α readout is honest about its inputs**: it distributes over the Tyranitar movepool
   prior (Rock Slide/DD/EQ) — the CB lock on EQ is genuinely unknowable from the obs (only EQ
   revealed, item hidden). Not a defect; noted so nobody reads the α row as an error.
6. **Scenario realism caveats**: gen3customgame (no clauses), an explosion-parade prelude, and
   a 4-attack Starmie facing a 0-Spe CB Ttar are all off-distribution for gen3ou eval traces.
   The obs pipeline is genuine; the *state distribution* is not. A mask read here transfers to
   ladder play only as far as the policy generalizes.

## 6. Rerun

```bash
export PYTHONPATH=$PYTHONPATH:src
python designs/research_state/measurements/starmie_ttar_risk_probe.py --phase all
```

Phases are resumable (`tables`/`capture`/`mc`/`sweep`); the JSON is incrementally saved.
