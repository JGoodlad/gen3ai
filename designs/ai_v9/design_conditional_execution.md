# design — CONDITIONAL EXECUTION: the moves whose value is a function of what they do

> **[STATE 2026-08-14]** The α/β belief this doc consumes is BUILT (v68) and consumed on the
> CRITIC side (v74 `intent_value_reduce`, live in gen-9). This doc's move-cell route — the
> POLICY-side consumer — is still the open piece, unchanged.

> **What this document is.** A per-mechanic specification of every gen3 move whose value cannot be
> computed without a belief over the opponent's action, the exact conditional each one needs, and
> what exists today. It is the **outgoing** consumer of the `α`/`β` intent belief that
> `design_opponent_intent.md` builds; that doc supplies the distribution, this one lists what to do
> with it.
>
> **The one-line claim.** `E[my move m] = Σ_k α_k · f(m, their move k)` — the same Contract-W
> contraction as the incoming side, applied to the outgoing axis and delivered to the **pointer
> move cell**, which is a channel measured to work.

---

## 0. Why this is not a new idea — it is the missing input to five dead families

The consequence edge families (`c1`–`c5`) and the entry/exit family (`x`) already compute
"what would happen if I clicked X". **Every one of them reads at the noise floor** (gen-4 end-of-run,
stratified, `gen4_edge_family_audit_25M.json`): `c2` 1.20%, `x` 1.05%, `c1` 0.65%, `c3` 0.50%,
`c5` 0.28%, `c4` 0.15%.

Reading the code, the conditioning is wrong in a specific and repeated way:

| family | what it conditions on | what the decision needs |
|---|---|---|
| **`c4` Protect** (`damage_op.py:2259`) | `p_success` = the **mechanical** consecutive-use decay odds, plus the end-of-turn ledger | **P(they attack at all)** — Protect against a switch or a setup move is a wasted turn |
| **`x` Pursuit** (`damage_op.py:2288`) | `pursuit_p` = **P(the other side CARRIES Pursuit)** | **P(they click it)**, and on our side **P(we switch)** — the condition under which it doubles |

That is `α ≠ w` — **presence versus usage** — appearing in a second, entirely independent place from
the incoming reduction. These families were asked to compose `p · net` where `p` is the wrong `p`.

**So the hypothesis this document rests on:** the consequence line was not a bad idea badly measured;
it was a correct idea missing its conditioning distribution *and* routed through an attention bias
(a ratio) instead of a per-action cell (an absolute). `α` fixes the first. The move cell fixes the
second. **If both fixes land and the numbers stay at zero, the consequence line is dead for real**,
and that is worth knowing.

---

## 1. The contract — and why `α` needs no change

```
E[value of my move m] = Σ_k α_k · f(m, k)
```

`f(m, k)` is a **deterministic rule lookup** — "does Focus Punch execute given that they clicked
move `k`". It is not learned and not a belief. Therefore:

> **v65 (`7bbcbfe`) strengthened the foundation this rests on.** The move-belief prior previously
> placed non-zero mass on **every** `(species, move)` pair, so a seat could be a move the species
> cannot physically learn — the phantom *"this special attacker might be holding Explosion"*. The
> learnset gate is now **unconditional**. Two consequences here: every seat `f(m,k)` is evaluated
> against is now a **legal** move, and `α`'s discrete support is correspondingly cleaner. Note that
> `design_opponent_intent.md`'s **G2a seat-coverage figure (89.3%) was measured on gen-5, i.e.
> PRE-v65** — it should be re-measured, and the expected direction is up.

- **`α` is unchanged.** It remains one distribution over their believed move seats + `SWITCH`,
  computed from the board alone, with no dependence on our candidate action. Contract W holds
  verbatim (`design_opponent_intent.md` §2).
- **The per-action structure comes from the action space**, exactly as it does for the incoming
  switch cell. Move cell `k` feeds move logit `6+k` (the `gen3_op_move_align_v1` guarantee), so
  each of our moves gets its own conditional without any new seats.
- **Channel:** the pointer **move cell** (13 dims today: `[low, high, crit, pko, p_land, known,
  sec×7]`). A per-action absolute at the logit — the channel `d1` (12.17%) and `d2` (19.25%)
  demonstrate works. **This design does not fight the delivery problem the defensive side has.**

**Add beside, never replace.** New channels ride alongside the unconditional ones; the head composes
them. Decorrelated per the provide-facts convention — emit `p_executes` and
`value_given_executes` separately rather than pre-multiplying.

### 1.1 The level-2 loop, and why it does not need special handling

If I am likely to click Focus Punch, they are likelier to attack me to break it — so `α` should
depend on my incentives, which depend on `α`. This is the same fixed point
`design_opponent_intent.md` §3 resolves: `α` sees the board (including that my active is a
Focus-Punch carrier and what my bench threatens), and **self-play finds the fixed point during
training**, never at inference. No solver, no recursion, no new contract.

---

## 2. The taxonomy — three different conditionals, and they are not interchangeable

| class | conditional on | horizon | operator |
|---|---|---|---|
| **A. THIS-TURN EXECUTION** | what they click *this turn* | 1 ply | `Σ_k α_k · f(m,k)` |
| **B. SWITCH-CONTINGENT** | whether they *pivot* | 1 ply | `α`'s `SWITCH` mass × `β` over their slots |
| **C. LONG-HORIZON** | how often they switch *over the rest of the game* | many plies | a rate, **not** a one-ply expectation |

**Class C is a trap and is deliberately scoped OUT of v1.** Spikes' value is "chip on every future
switch-in", which is not `Σ_k α_k · f(m,k)` for any `f` — a one-ply α cannot express a rate. Modelling
it as a one-turn conditional would be wrong in a way that looks right. Recorded in §5.

---

## 3. The mechanics

Exposure is measured over the 773 committed team files in `data/teams/` (2026-08-12).

> **⚠️ Exposure is a POOR prioritiser on its own, and an earlier draft of this document used it as
> the primary one. That was wrong** (owner, 2026-08-12), and wrong for a reason this session had
> already written down: **aggregate frequency systematically under-weights decisive-but-rare
> mechanics.** Destiny Bond appears on 6 teams and *decides the games it appears in*. The same
> blind spot voided the reading of `OUR_MOVE_OUTCOME` (5.35% aggregate, but the Focus-Punch
> signature is 0.70% of decisions). **Carriage frequency measures how often a mechanic is
> available, not what it is worth when it fires.** Both numbers are reported below; neither alone
> orders the work.

### 3.0 The shared primitive — one operator, five mechanics

Five of the mechanics below are the *same computation* with a different threshold and comparison
direction. This is the single most important structural fact in the document:

```
p_thresh(τ, ⋛) = Σ_k α_k · 1[ damage(k, me) ⋛ τ ]
```

| mechanic | τ | direction | reading |
|---|---|---|---|
| **Focus Punch** | 0 | `>` | any damage breaks it |
| **Substitute** | 25% maxhp | `<` | the sub survives |
| **Endure** | current HP | `≥` | I would otherwise die |
| **Destiny Bond** | current HP | `≥` | **same threshold, opposite valence** |
| **Endeavor** | current HP | `<` | I survive to act |

So it is **one operator with two parameters**, not five features. And the `τ = current HP` case has a
name worth giving it explicitly:

```
p_KO = Σ_k α_k · 1[ damage(k, me) ≥ my current HP ]        # α-weighted P(I die this turn)
```

**`p_KO` is nearly free once `α` exists.** The op already computes a per-believed-move `pko`
channel — today collapsed by `_chan_max` into a hard max over their moves. `α` turns that max into
a **calibrated probability**. Same tensor, correct functional.

Two consequences beyond this document. First, `p_KO` is exactly the quantity ledger **H1**'s
self-KO defect mis-values — *"am I about to die"* is what makes a trade look good or bad, and the
critic is currently inferring it from a max. Second, the threshold family is the concrete payoff of
`design_pair_reduction.md` §4.1: **a threshold on the branch distribution is not a function of its
mean**, so `max` cannot produce any row of the table above.

### 3.1 Explosion / Self-Destruct — 69.3% of teams — class B

**Rule (gen3):** user faints; halves the target's Defense for the damage calculation. `bp` 250 / 200,
priority 0.

**The conditional.** Value depends almost entirely on what they do:
- they **stay** → the trade lands, and its worth is `their mon's remaining value − ours`
- they **switch** → we detonate on a switch-in we did not choose (`β` decides which)
- they **Protect** → we lose the mon for nothing (the worst branch, and it is `α`-visible)

```
E[explosion] = Σ_k α_k · [ executes(k) · trade_value(target | k) ]
             where executes = 0 on their Protect, and the target is β-weighted on their SWITCH mass
```

**Why this is the highest-value entry in the document.** Ledger **H1** is a ✅ CONFIRMED behavioural
defect: the policy explodes **healthy** mons — ~38% of Explosions at ≥80% HP, confidence ~0.5 (a
*learned* preference, not exploration), the reward is **correct** (−2.7), and the **critic
over-values the trade** (dV +2.9 → PPO advantage +1.5). The built-but-unshipped fix is
`--self-ko-hp-penalty`, a **reward** term.

But the ledger's own diagnosis says the reward is right and the *valuation* is wrong. A reward
penalty papers over a valuation defect; **conditioning Explosion's value on `α`/`β` is the
representation fix for the same defect** — it gives the critic the quantity it is currently
guessing. Worth running as an arm against the reward penalty rather than instead of measuring it.

**Cell channels:** `[is_boom, p_executes, e_trade_value]`.

### 3.2 Spikes — 45.7% — **class C, DEFERRED**

**Rule (gen3):** up to 3 layers; 1/8, 1/6, 1/4 max HP on a grounded switch-in. Flying/Levitate immune.

**Why deferred:** its value is a **rate over the rest of the game**, not a one-ply conditional. The
right conditioning variable is "how often will they be forced to switch", which needs a
phazing/pressure model, not `α`. `x`'s `entry_chip` already delivers the per-mon chip magnitude
unconditionally; that is the correct division. **Do not model Spikes as a one-turn conditional.**

### 3.3 Protect / Detect — 38.2% — class A

**Rule (gen3):** priority +3. Consecutive use decays success (the floored-doubling counter,
`gen3_protect_odds_v1`, already in the obs).

**The conditional — and `c4` has the wrong one today:**

```
E[protect] = Σ_k α_k · [ damage_avoided(k) ]  −  tempo_cost
           where damage_avoided = full incoming damage on an attack, ~0 on their switch/setup
```

`p_success` (mechanical) is a **multiplier** on this, not a substitute for it. Today `c4` carries
the multiplier and omits the quantity.

**Cell channels:** `[is_protect, p_success, e_damage_avoided, e_status_avoided]` — keeping
`p_success` and the α-weighted avoided damage **decorrelated** so the head forms the product.

### 3.4 Substitute — 29.6% — class A

**Rule (gen3):** costs 25% max HP; the sub absorbs damage and status until broken.

**The conditional:** a Substitute is excellent if their expected hit is **below** 25% max HP and
worthless if above.

```
p_sub_survives = Σ_k α_k · 1[ damage(k, me) < 0.25 · maxhp ]
```

**This is a threshold on the branch distribution, not on its mean** — which is precisely the
second-moment capability `design_pair_reduction.md` §4.1 argues `max` cannot produce and a
weighted sum can. Substitute is the cleanest worked example of why the reduction needs `Σ α_k [o, o²]`
or an explicit threshold channel.

**Cell channels:** `[is_sub, p_sub_survives, e_damage_absorbed]`.

### 3.5 Focus Punch — 26.1% — class A

**Rule (gen3):** priority **−3**, `bp` 150. **Fails if the user takes damage from an attack that
turn before it executes.** Because of the negative priority it resolves last, so essentially any
damaging move from them breaks it.

```
p_executes = Σ_k α_k · 1[ k does no damage to me ]
           = α(status moves) + α(setup) + α(SWITCH) + α(moves I am immune to)
```

**The immunity term matters and is easy to miss:** a Ghost-type user is not broken by a Normal
Fighting move, a Levitate mon is not broken by Earthquake. `f(m,k)` must use the real type/ability
interaction, not "is `k` a damaging move".

**This is the owner's motivating example** (2026-08-12) and it is what makes the mechanic a *forward*
computation rather than a memory: today the model can only learn "Focus Punch failed" associatively
from the `t−1` history frame, which decays out of a 7-turn window and does not generalise. With
`p_executes` it never needs to remember.

**On the G7 team.** `data/teams/specialist/tss_starmie.txt`'s Tyranitar runs Focus Punch / Rock
Slide / HP Bug / EQ — so the capability gate gets a much crisper readout than the hedging one:
**does the model click Focus Punch when `α` says status/switch, and Rock Slide when `α` says
physical attack?** A directional bifurcation on one mon.

**Cell channels:** `[is_focuspunch, p_executes]` — the damage channels already exist and stay
unconditional beside it.

### 3.6 Pursuit — 21.0% — class B

**Rule (gen3):** `bp` 40, Dark. **Doubles power and strikes before the switch** if the target
switches out that turn.

```
E[pursuit] = α(SWITCH) · 2·dmg(target = β-weighted switch-in)  +  (1 − α(SWITCH)) · dmg(current active)
```

**The purest `β` consumer in the game** — and note the target itself differs between branches, which
is why `β` (which mon comes in) is needed and not just `α`'s switch mass.

Today `x` carries `pursuit_p` = P(they *carry* Pursuit), a presence belief on the wrong side of the
question.

**Cell channels:** `[is_pursuit, p_target_switches, e_damage_on_switch, e_damage_on_stay]`.

### 3.7 Counter / Mirror Coat — 8.9% / ~0% — class A

**Rule (gen3):** priority **−5**. Counter returns **2× the physical** damage taken this turn;
Mirror Coat returns **2× the special**. Each fails entirely against the other category, against
status, and against a switch.

```
E[counter]    = Σ_k α_k · 1[ k is PHYSICAL ] · 2 · damage(k, me)
E[mirrorcoat] = Σ_k α_k · 1[ k is SPECIAL  ] · 2 · damage(k, me)
```

**These are the purest read-the-opponent moves in gen3 — literally unplayable without an intent
model.** There is no safe Counter; its entire value is `P(they go physical)`. It is unsurprising if
the current policy simply never clicks them, and that is worth measuring as a **pre-build
behavioural baseline** (§6 G3).

*Mirror Coat has **zero** carriers in the committed pool.* Model it only because it is the exact
mirror of Counter and costs one line once Counter exists — not for its own sake.

**Cell channels:** `[is_counter, p_category_match, e_return_damage]`.

### 3.8 Flinch (Rock Slide 30%, Fake Out 100%) — class A

**Rule (gen3):** a flinch only lands if the flincher **moves first** and the target **has not yet
acted**. Our data already carries the chance: `rockslide.secondaryEffects = {'flinch': 30}`,
`fakeout = {'flinch': 100}`, and the move cell already delivers 7 live secondary columns.

```
p_flinch_useful = p_outspeed · p_secondary_fires · (1 − α(SWITCH))
```

**The `α(SWITCH)` term is the whole point and is missing today.** A flinch against a switching
opponent is worth exactly nothing — they were not going to attack. The raw flinch chance is already
delivered; what is absent is the conditioning that makes it *meaningful*.

**Cell channels:** one extra — `p_flinch_useful` — beside the existing secondary column.

### 3.9 Destiny Bond — 0.8% of teams — class A — **the purest α move in gen3**

**Rule (gen3):** priority 0, targets self. If the user **faints from a direct attack** before its
next turn, the attacker faints too.

```
E[destiny_bond] = p_KO · value(their active)        # p_KO from §3.0
```

**Its value is *monotonically increasing* in the probability that they kill you.** Every other move
in this document is worth more when you are safe; Destiny Bond is worth more when you are dead.
That inversion is why it cannot be learned as a correlate of "danger" — the model needs the actual
conditional, and a max-based `pko` gives it a worst case rather than a probability.

It is also the exact **mirror of Endure** — same threshold, opposite response: Endure says *survive
it*, Destiny Bond says *don't, and take them with you*. A model that has `p_KO` can choose between
them; a model without it can do neither.

**Low carriage, decisive when it fires** — the canonical case for §3's warning about exposure.

**Cell channels:** `[is_dbond, p_KO, e_target_value]`.

### 3.10 Endure — 1.6% — class A

**Rule (gen3):** priority **+4**, `isProtect: true` — so it **shares Protect's consecutive-use decay
counter** (it is already in `_PROTECT_NUMS` and gated by `c4`, with the same wrong conditioning as
§3.3).

```
E[endure] = p_KO · (survival value at 1 HP)  −  tempo_cost
```

**Endure against a non-lethal hit is a wasted turn; against a lethal one it is a saved mon.** So it
is `p_KO`-gated in the strictest sense — the *only* branch where it pays is the branch where you
would have died.

**Cell channels:** `[is_endure, p_success, p_KO]` — `p_success` (mechanical) and `p_KO` (α-weighted)
stay decorrelated; the head forms the product.

### 3.11 Endeavor — 4.1% — class A — the low-HP inversion

**Rule (gen3):** priority 0, `bp` 0. Sets the **target's HP equal to the user's HP**. Fails if the
user's HP ≥ the target's HP.

```
E[endeavor] = p_survive_to_act · max(0, their_HP − my_HP)
            where p_survive_to_act = Σ_k α_k · 1[ damage(k, me) < my HP ]   # = 1 − p_KO
```

Two conditions, and both need `α`:
- the **damage** term is deterministic and known (both HPs are observed) — this part is free;
- but Endeavor is **priority 0**, so a faster opponent who KOs you first makes it worth nothing.

**This is the mechanic that most inverts normal evaluation.** Endeavor's value *rises* as your HP
falls, which is the opposite of every damage-based feature in the op. A model reading "I am at 8% HP,
this mon is nearly worthless" has exactly backwards the state in which Endeavor is strongest.

**Pairs with Endure** (Endure at 1 HP → Endeavor brings them to 1 HP). That *combination* is a
two-turn plan and stays out of scope per §5 — but **each leg is a one-ply conditional and is in
scope**, which is the right decomposition.

**Cell channels:** `[is_endeavor, p_survive_to_act, e_hp_swing]`.

### 3.12 Magic Coat — 0 pool carriers — class A — a **status**-conditional, the fourth shape

**Rule (gen3):** priority **+4**, targets self. Bounces the next *status* move aimed at the user
back at the opponent.

```
p_magiccoat_useful = Σ_k α_k · 1[ k is a STATUS move that targets me and is reflectable ]
```

**This is a fourth conditional shape** — not a damage threshold (§3.0) and not a category test
(§3.7), but a *reflectability* predicate. It needs its own lookup and its own oracle scenarios.

⚠️ **The reflectable set must be verified against the sim, not assumed.** Whether gen3 Magic Coat
bounces side-targeting moves (Spikes) versus only user-targeting ones (Toxic, Thunder Wave,
Will-O-Wisp, Leech Seed) is exactly the kind of rule that differs by generation, and getting it
wrong puts a false conditional in a per-action cell. **`f(m,k)` here is `UNVERIFIED` until G0
covers it.**

Its natural mirror is **Snatch** (0.5% carriage, priority +4, steals a status move rather than
bouncing it) — the same predicate with a different effect, and one line once the reflectable
lookup exists. Snatch is the only mechanic still deliberately out of scope, and only because it is
strictly dominated in build order by Magic Coat.

**Cell channels:** `[is_magiccoat, p_status_incoming, e_bounced_value]`.

---

## 4. Summary table

| # | mechanic | pool | class | conditional | today |
|---|---|---|---|---|---|
| 1 | **Explosion / Self-Destruct** | **69.3%** | B | trade value; 0 on their Protect; target is `β`-weighted on switch | nothing (and **H1** is a confirmed defect here) |
| 2 | Spikes | 45.7% | **C** | switch RATE over the game | `x` chip, unconditioned — **DEFERRED** |
| 3 | Protect / Detect | 38.2% | A | `Σ α_k · damage_avoided(k)` | `c4`, **mechanical** odds only |
| 4 | Substitute | 29.6% | A | `Σ α_k · 1[dmg < 25% maxhp]` — a **threshold**, not a mean | nothing |
| 5 | **Focus Punch** | 26.1% | A | `Σ α_k · 1[k does no damage to me]`, incl. immunities | nothing |
| 6 | Pursuit | 21.0% | B | `α(SWITCH)` × 2× vs `β`-weighted target | `x`, **presence** belief |
| 7 | Counter / Mirror Coat | 8.9% / 0% | A | `Σ α_k · 1[category match] · 2·dmg` | nothing |
| 8 | Flinch (Rock Slide/Fake Out) | high | A | `p_outspeed · p_sec · (1 − α(SWITCH))` | chance delivered, **unconditioned** |
| 9 | **Destiny Bond** | 0.8% | A | **`p_KO`** × their value — *rises* with danger | nothing |
| 10 | **Endure** | 1.6% | A | **`p_KO`** × survival, × mechanical `p_success` | `c4` gates it, wrong `p` |
| 11 | **Endeavor** | 4.1% | A | **`1 − p_KO`** × `(their_HP − my_HP)` — *rises* as my HP falls | nothing |
| 12 | **Magic Coat** | 0% | A | `Σ α_k · 1[k is a reflectable STATUS move]` — a 4th shape | nothing |
| — | Snatch | 0.5% | A | Magic Coat's predicate, different effect | **out of scope** — one line once #12 lands |

**Note that #9–#11 are all `p_KO`** (§3.0), so the three of them cost one scalar plus their
valuations. That is why scoping them in is cheap even at low carriage — and #9 and #11 are the two
mechanics in the whole document whose value moves **opposite** to every damage feature the op
currently computes.

---

## 5. What this design does NOT do

- **No long-horizon rates (class C).** Spikes and hazard pressure need a switch-*rate*, not a one-ply
  expectation. Modelling them here would be confidently wrong.
- **No conditioning of `α` on our action.** `f(m,k)` is a rules lookup; `α` stays defender- and
  move-independent (§1). Any proposal that makes `α` per-move is the D3 decorrelation defect in a
  new costume.
- **No multi-turn plans** — "they are saving Explosion for my Celebi" stays out of reach, per
  `design_pair_reduction.md` §4.2. **This is the Endure→Endeavor caveat**: the *combination* is a
  two-turn plan and is out of scope; **each leg is a one-ply conditional and is in scope** (§3.10,
  §3.11). The decomposition is what makes the pair tractable at all — a model that prices both legs
  correctly can discover the sequence through play without ever representing it as a plan.
- **No new seats.** Everything lands in the existing per-action move cells.
- **Snatch only** (0.5%) stays out, and only because it is strictly dominated in build order by
  Magic Coat's reflectable predicate (§3.12).

---

## 6. Gates

| # | gate | needs a run? |
|---|---|---|
| **G0** | **Physics oracle per mechanic.** Each `f(m,k)` is a *rule*, so it gets constructed single-turn scenarios verified against the sim — the `damage_op_probe_fuzz_test` pattern. **Non-negotiable:** a wrong `p_executes` is a GIGO defect sitting in a per-action cell at a logit. Focus Punch's immunity term and Counter's category test are the two most likely to be got wrong. | no |
| **G1** | **Byte-identity when off**, and zero-init so ON starts identical — asserted on a **real `MaskablePPO`-built policy** (ledger **M1**: SB3's ortho pass clobbers extractor zero-inits). | no |
| **G2** | **Pre-build behavioural baseline.** Measure current usage rates of each mechanic — how often does the policy click Counter, Focus Punch, Protect, and *with what confidence*? If it already never clicks Counter, that is the number the retrain has to move. | no |
| **G3** | **The discriminating single-family arm.** Before building all eight: re-deliver **ONE** existing consequence family (`c2`, the least-dead at 1.20%) through the **move cell** with `α` conditioning. If it comes alive, both halves of §0's hypothesis are confirmed at once. If it stays at zero, **stop** — the consequence line is dead and the remaining seven are not worth building. | no |
| **G4** | **Behavioural bifurcation** (the capability gate). On the G7 team: does Focus-Punch-vs-Rock-Slide selection become conditional on `α`'s physical/status split? Directional, not a win-rate delta. | 2 short forks |
| **G5** | **Cost** — no measurable regression on the compiled B=1 CPU path. | no |

**G3 is the gate that matters.** It is one family, offline, and it prices the entire document.

---

## 7. Build order

| # | step | gate |
|---|---|---|
| 0 | `α` exists and clears its own gates (`design_opponent_intent.md` G2a/G2b) | — |
| 1 | **G2** behavioural baseline — free, and it defines "did it work" | G2 |
| 2 | **G3** the single-family discriminator (`c2` → move cell + `α`) | **G3** |
| 3 | **`p_thresh(τ, ⋛)` — the §3.0 shared primitive.** One operator; it lands **Focus Punch, Substitute, Endure, Destiny Bond and Endeavor** at once, and `p_KO` falls out of it | G0, G1 |
| 4 | Counter / Mirror Coat + flinch — the category test and the `(1 − α(SWITCH))` gate | G0, G1 |
| 5 | Protect — reconcile with `c4`'s existing mechanical `p_success` | G0, G1 |
| 6 | Magic Coat — the reflectable predicate (**+ its own oracle**, §3.12 is `UNVERIFIED`) | **G0** |
| 7 | Explosion + Pursuit — class B, needs `β` | G0, G1 |
| 8 | Ship at a generation boundary | G4, G5 |

**Steps 1–7 need no training run.** Retrain-class at step 8 (new cell channels ⇒
`MODEL_CONFIG_VERSION` bump).

**Step 3 is the efficient one and should go first among the builds:** five mechanics, one operator,
and it produces `p_KO` — which is independently useful to the critic (§3.0) whatever happens to the
rest of this document.

---

## 8. Provenance

| claim | source |
|---|---|
| move numbers, base power, **priority**, accuracy for all 25 candidate mechanics | `agents.gen3_data.moves` (read 2026-08-12): focuspunch −3/150, counter −5, mirrorcoat −5, protect +3, endure +4, pursuit 0/40, snatch +4, magiccoat +4, explosion 0/250, selfdestruct 0/200, fakeout +1/40 |
| flinch chances are already in the data | `moves.raw()`: `rockslide.secondaryEffects={'flinch':30}`, `fakeout={'flinch':100}`; 7 live secondary columns already ride the move cell (`_N_OUT_SECONDARY`) |
| **pool exposure** over 773 committed team files | `data/teams/**/*.txt`, counted 2026-08-12 — explosion 536 (69.3%), spikes 353 (45.7%), protect 295 (38.2%), substitute 229 (29.6%), focuspunch 202 (26.1%), pursuit 162 (21.0%), counter 69 (8.9%), endeavor 32, endure 12, destinybond 6, snatch 4; mirrorcoat/magiccoat/detect/selfdestruct/fakeout **0** |
| `c4` conditions on MECHANICAL protect odds + the end-of-turn ledger, never on "will they attack" | `damage_op.py:2259-2286` (`pairwise_protect`) |
| `x`'s `pursuit_p` = P(the other side **carries** Pursuit) | `damage_op.py:2288-2301` (`pairwise_entry`) |
| every consequence family is at the noise floor after a full run | `gen4_edge_family_audit_25M.json`: c2 1.20%, x 1.05%, c1 0.65%, c3 0.50%, c5 0.28%, c4 0.15% |
| the move cell is a per-action ABSOLUTE and its channel works | `ARCHITECTURE.md` §3.3, §5.3; `d1` 12.17% / `d2` 19.25% same audit |
| **H1** — policy explodes healthy mons, reward correct, **critic** over-values the trade | `research_state/ledger.md` H1 (✅ CONFIRMED): ~38% of Explosions at ≥80% HP, conf ~0.5, reward −2.7, dV +2.9 → advantage +1.5; `--self-ko-hp-penalty` built, not shipped |
| the Focus Punch motivating example, and the joint-fact constraint behind it | owner, 2026-08-12 |
| **Endeavor / Endure / Destiny Bond / Magic Coat scoped IN despite low carriage** — and the correction that exposure alone is a poor prioritiser (it under-weights decisive-but-rare mechanics, the same blind spot that voided the `OUR_MOVE_OUTCOME` reading) | **owner, 2026-08-12** |
| Endure is `isProtect: true` ⇒ it shares Protect's consecutive-use decay counter and is already in `_PROTECT_NUMS` | `moves.raw()['endure']`; `damage_op.py:2259` `pairwise_protect` docstring ("Protect/Detect/Endure") |
| priorities for the newly-scoped set: endure **+4**, magiccoat **+4**, endeavor **0**, destinybond **0** | `agents.gen3_data.moves` (read 2026-08-12) |
| the move-belief learnset gate is now UNCONDITIONAL, so every seat is a LEGAL move (and G2a's 89.3% predates it) | `7bbcbfe` (v65) — the `learnset_gate` parameter DELETED rather than defaulted; `_ILLEGAL_PROB` 1e-6 for out-of-learnset, floor 0.02 for legal-unobserved, and a floor everywhere for an unknown species |
| ⚠️ gen3 Magic Coat's **reflectable set** (does it bounce side-targeting Spikes, or only user-targeting status?) | **UNVERIFIED** — must be established by G0 constructed scenarios before `f(m,k)` is written |
| `α` is one board-conditioned distribution; the fixed point is found by training | `design_opponent_intent.md` §2, §3 |
| Substitute's threshold needs a second moment, which `max` cannot produce | `design_pair_reduction.md` §4.1 |
| move cell `k` feeds move logit `6+k` | `gen3_op_move_align_v1`; `damage_op.py::pointer_cells` |

## See also

- `design_opponent_intent.md` — where `α` and `β` come from; this doc is their outgoing consumer
- `design_pair_reduction.md` — Contract W, the second-moment argument, add-beside-never-replace
- `design_conditional_opponent_cells.md` — OA1/OA2, the per-action query-conditioned route
- `research_state/ledger.md` — H1 (Explosion), and the consequence-family measurements
