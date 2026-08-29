# PROBE N — `no_progress_tax`: INTENT vs IMPLEMENTATION, and the CLEAN-WORLD config surface

*Code archaeology 2026-08-29 · read-only, no model, no battles · every claim carries a sha, a
`file:line`, or is marked **UNVERIFIED**. Companion measurement: probe M
(`bias_tax_head_alignment_2026-08-29.md`).*

---

## THE HEADLINE, before the detail

The owner's recollection — "I wrote it to punish only obviously irrefutably poor choices" — is
**half right, and the half that is wrong is the more interesting half**.

* Charging a **voluntary switch that lands nothing** was **DESIGNED, in writing, deliberately**
  (`design_markovian_reward_and_features.md:714-718`). It is not drift.
* But it was designed **inside a reward where a voluntary switch also earned `switch_base = +0.5`**,
  plus `se_switch +0.2` / `escape_threat_switch +0.25` / `pivot_* +0.10..0.15`. The net incentive on
  a tempo pivot was **positive**. `928a00b` (2026-06-12) zeroed **every one of those credits and kept
  the tax**; `43673ed` (2026-08-18) made that the default. **The sign of the switch incentive flipped
  and nobody re-derived the term's meaning.** That is the drift, and it is a COMPOSITION drift, not a
  code drift — which is exactly why no test caught it.
* Charging a **post-faint replacement** was designed to be **impossible** — the design says so in one
  sentence — and the implementation has charged it since the term's first commit. That is a bug,
  present at birth, caused by reusing an attribute built for a different question.

---

## 1. THE INTENT RECORD

### 1.1 Provenance chain

| # | sha | date | what |
|---|---|---|---|
| 1 | `045e8b8` | 2026-05-26 | `phase_is_forced_switch` **introduced — for the OBS history slot**, not the clock ("Per-turn history slot grows 39 → 88 dims: … **phase_is_forced_switch flag** …") |
| 2 | `6a2cab4` | 2026-06-07 | the DESIGN doc `designs/ai_v5/design_markovian_reward_and_features.md` |
| 3 | `adc0fe4` | 2026-06-07 | **ORIGINATING commit** — `ProgressClock` + `no_progress_tax` (`gen3_markovian_progress_v1`) |
| 4 | `d7aa983` | 2026-06-08 | heal-war grace + winning-residual exemption + `--draw-penalty` |
| 5 | `928a00b` | 2026-06-12 | `--all-shaping-pbrs`: "**zero every BIAS term except the no_progress_tax tilt**" ← the composition drift |
| 6 | `43673ed` | 2026-08-18 | that composition becomes the DEFAULT |

### 1.2 What "progress" was defined to mean

`design_markovian_reward_and_features.md:528-537`, verbatim:

```
PROGRESS(delta, live, prev_spikes) :=
      (delta.our_damaging_event is not None  AND  our move dealt ≥ PROGRESS_DMG_EPS to a non-fainted opp)   # (i)
   OR  delta.opp_status_applied is not None                       # (ii) a status LANDED on opp (the event)
   OR (opp_spikes_now − prev_spikes) > 0                          # (iii) a hazard LAYER was added
   OR  delta.opp_switch_to is not None                            # (iv) forced an opp commit (phaze / forced switch)
```

Four clauses, all **offensive**, and the doc says so in as many words (`:660`): *"The `PROGRESS`
predicate (§4.1) is **offense-centric** (damage / status-on-opp / hazard / commit)."* Note clause
(iv) is **"we forced an opp commit"** — the opponent being moved by *us*, not us moving.

The classification table (`:668-673`) names what is charged:

> | **offense/setup/hazard** achieves nothing & moves no Φ (immune attack, <3% chip, capped setup,
> redundant status, Spikes-at-3) | futile active | **INCREMENT + futile** | a deliberate,
> **obs-knowable wheel-spin** |

"A deliberate, obs-knowable wheel-spin" is the closest thing the record has to the owner's
"obviously irrefutably poor choices", and it is a good match — **for moves**.

### 1.3 Was switching considered? YES — explicitly, and charged on purpose

`design_markovian_reward_and_features.md:714-718`, verbatim, immediately after the two penalty gates:

> **Productive repeats/bounces escape entirely.** A damage-dealing repeated attack
> (`our_damaging_event ≥ 3%` each window) pins the clock at 0 → zero penalty (fixing the 14%
> over-reach). A matchup-improving A→B→A pivot that lands damage/status/commit resets; **a pure
> tempo-pivot that lands nothing pays only the front-loaded toll once (correctly — it *was* a
> no-progress window)** without a separate bouncing tax.

And in the anti-spam collapse table (`:499`):

> | `switch_bouncing_tax` | **subsumed** | an A↔B bounce burns tempo with no board change = no progress |

So: **the clock was explicitly given the job of taxing pivots that achieve nothing.** The switch
half is not an oversight. It is the term's stated purpose, inherited from `switch_bouncing_tax`.

**What makes it wrong today is what was around it.** In the design's composition a voluntary switch
also collected `SWITCH_BASE_BONUS = 0.5` (`reward_weights.py:52`), `SE_SWITCH_BONUS = 0.2` (`:76`),
`ESCAPE_THREAT_BONUS = 0.25` (`:86`), and `pivot_damage` +0.10/+0.15. The reward manager still says
so, in a comment that has outlived the world it describes (`reward_manager.py:1572-1575`):

> `switch_bouncing_tax` is **deliberately NOT suppressed** (ai_v5_6 regression): **the flat
> no-progress charge (−0.15) does not out-weigh the per-switch reframes**, so the escalating 2-cycle
> bounce tax is kept as a real negative brake …

That sentence is the intent, stated as a quantity: *the switch charge was never meant to be the net
signal on switching*. Under `--all-shaping-pbrs` (`_apply_pbrs_suppression`, `reward_manager.py:1610-1614`)
every one of those credits is set to `0.0` and `no_progress_tax` is the sole survivor. The −0.15 no
longer competes with +0.5; it IS the signal. Probe M measured the consequence: **−0.101 expected
charge per voluntary switch against −0.010 per move, a 10:1 differential.**

### 1.4 The two penalty GATES the design specified

`design_markovian_reward_and_features.md:704-712`, verbatim:

> **Two gates on the penalty [RT-2-major×2]:**
>
> - **Forced-switch windows are no-ops.** A post-faint / phaze replacement allows only switches, so
>   it fails the predicate spuriously. Gate on `delta.phase_is_forced_switch`: do **not** increment
>   and do **not** charge on a forced window.
> - **Trapped-vs-wall is not charged.** A trapped mon with no progress move has switches masked
>   illegal, so a penalty would be unavoidable by *any* policy (punishing a state, not a choice).
>   Gate the **charge** on a switch being legal **this decision** (`legal.switches` non-empty).

Both gates name the same thing the implementation gets wrong: **"a forced window"** and **"this
decision"** both mean *the decision whose window is being folded*. The implementation reads the
**next** decision for both. See §3.

### 1.5 The FLAT shape was chosen knowing it charges isolated turns

`:731-748` — three shapes were costed; **(b) FLAT at c = 0.15** was recommended over (a) LOG
specifically because *"FLAT carries one fully-legible −0.15 per useless turn with **zero variance**
for the critic to learn"*, with the acknowledgement *"satisfies the goal, just not strictly
front-loaded"* and an explicit fallback trigger:

> **Switch to (a)** if A/B shows the policy "nibbles" (repeated isolated single useless turns
> because one −0.15 is cheap).

Probe M's finding that **70.9% of charges land at `n_t = 0`** is therefore not a defect against
intent — a flat schedule charges the first no-progress turn at full price *by design*. It is,
however, the empirical answer to the design's own open question, and it points the opposite way from
the fallback: the problem is not that isolated turns are too cheap, it is that they are ~all of the
tax.

---

## 2. LINE-BY-LINE — every charging path, classified against intent

`ProgressClock.update` (`progress_clock.py:135-242`). Three code paths set a non-zero
`last_penalty`; all three run the identical two lines (`n += 1`; charge iff `legal.switches`).

| # | path | `file:line` | verdict vs intent |
|---|---|---|---|
| A | **capped-Spikes short-circuit** — `our_move_id == "spikes"` at 3 layers | `progress_clock.py:174-181` | **INTENDED.** Design `:671` lists "Spikes-at-3" as a futile active. Exactly "obviously irrefutably poor". |
| B | **wasted self-cure** — Refresh with nothing to cure | `:191-196` | **INTENDED** (later addition, `8c35e70`; same "wasted support" row of the design table `:673`). |
| C | **filler Rapid Spin** — no hazards on our side, no KO | `:204-206` | **INTENDED** in spirit (a 20-BP pseudo-attack that "moves no Φ"); NEVER-CONSIDERED in the 2026-06 text, added later. Benign. |
| D | **general NO_OP: a MOVE that failed the predicate** | `:240-242` | **INTENDED.** The core case. Probe M: 6.7% of moves charged. |
| E | **general NO_OP: a VOLUNTARY SWITCH** | `:240-242` | **DESIGNED** (`design:714-718`) — but the *net* signal is DRIFT. The counterweights (`switch_base` +0.5 et al.) were zeroed by **`928a00b`** while the tax was explicitly kept ("zero every BIAS term except the no_progress_tax tilt"), and defaulted by **`43673ed`**. 42.7% of all charges. |
| F | **general NO_OP: a FORCED SWITCH (post-faint replacement)** | `:240-242`, reached because the guard at `:161` missed | **DRIFT — a bug present at the ORIGINATING commit `adc0fe4`.** The design forbids it in one sentence (`:706-708`). 36.3% of all charges. |
| G | **sustained heal past `HEAL_FREEZE_GRACE`** | `:231-236` → `:240` | **INTENDED** (`d7aa983`: "a SUSTAINED no-progress heal-war charges … the 250-turn mirror-stall hole"). |
| H | **rest-loop (grace denied)** | `:232` | **INTENDED**, later addition, narrowly scoped with a Sleep-Talk exemption. |
| I | **our own FAILED Protect/Detect** | `_denial_kind` `:411-414` returns `None` → `:240` | **INTENDED**, argued in the comment at `:405-410`: *"a failed Protect IS a no-progress turn"*. Probe M nonetheless measures taxed Protect's cost CI straddling zero — an intent/reality gap, not an intent/implementation gap. |

**Why E and F together are 79% of the tax: `_is_progress` has no clause a switch can satisfy.**
Walking `progress_clock.py:310-358` for a window whose action was a switch:

| clause | can a switch trigger it? |
|---|---|
| (i) our damaging event | **No** — requires our move; `our_move_id is None` on a switch |
| (ii) opp status applied | **Yes, but never *because* we switched** — a contact ability on the switch-in (Static/Poison Point/Effect Spore/Flame Body) paralysing the attacker, or the opponent self-statusing (Rest). Opponent- or ability-determined. |
| (iii) hazard layer added | **No** — requires our Spikes |
| (iv) opp forced to commit | **Yes, but opponent-determined** — `opp_switch_to` is set when *they* also switched |
| (v) winning residual | **Yes, but pre-existing** — our Toxic/Leech already ticking |
| (vi) our boost sum rose | **No** — "a switch-in is boostless (boosts reset on switch)" (comment `:339-342`) |
| (vii) new Substitute | **No** — a Sub does not survive a switch |
| (viii) Wish cast | **No** — requires our move |

So the escapes exist (probe M measures 27% of voluntary switches escaping) but **not one of them is
caused by the switch**. The design's own framing of clause (iv) — *"we forced an opp commit"* —
makes this explicit: the predicate asks "did OUR ACTION advance the board", and a switch's whole
value (position, tempo, avoided damage) is a *state* fact the predicate never looks at.

---

## 3. PROBE M's TWO DEFECTS — verified from source

### 3.1 "`_is_progress` is unsatisfiable by any switch" — **CONFIRMED IN SUBSTANCE, over-stated literally**

Probe M §2 states it correctly: *"none of which any switch action can produce … A switch's only
escape routes are opponent-determined (they also switched) or a pre-existing residual."* That is
exactly right (table above), with one route probe M does not name: clause (ii) via a **contact
ability on the switch-in** (Static/Poison Point/Effect Spore/Flame Body) or an **opponent
self-status** (Rest, Sleep Talk-less), both of which set `opp_status_applied` on a turn we merely
switched. Rare; still opponent/ability-determined, so the conclusion is untouched.

Probe M §7's compressed restatement — *"**A switch can never satisfy `_is_progress`**"* — is
**literally false** (clauses ii/iv/v can fire) and should be read as the §2 wording. **The operative
claim — the tax prices an action KIND rather than discriminating within it — stands.**

### 3.2 "The SITOUT exemption is off by one window" — **CONFIRMED, exactly as described**

The chain, verifiable in four lines:

1. `progress_clock.py:161` — `if getattr(delta, "phase_is_forced_switch", False): last_penalty = 0.0; return`
2. `turn_delta.py:405` **and** `:531` — `phase_is_forced_switch=(curr_ctx.phase == "forced_switch")`
3. `battle_snapshot.py:267` — `phase="forced_switch" if battle.force_switch else "move_selection"`, i.e. read off the **live request**
4. `episode_tracker.py:732` then `:908-921` — `record(battle, …)` appends `curr_ctx` for the **upcoming** request, then `update_progress_clock` builds `delta = (prev_ctx, curr_ctx)` — the window of action `a_t` — and folds it

So the flag on the window of `a_t` carries the phase of **decision `t+1`**. Consequence, both halves:

* decision `t` = a normal turn in which our mon is KO'd ⇒ `t+1` is a forced switch ⇒ **the clock
  sits out on a full-agency window** (probe M's SITOUT class: 19,503 decisions, the costliest in the
  corpus at −5.1pp);
* decision `t` = the post-faint replacement ⇒ `t+1` is normal ⇒ **the clock runs on a zero-agency
  window**, which no action can rescue ⇒ charged 63.9% of the time (12,432 charges).

Probe M reached this by measurement (0/10,442 violations under the `t+1` alignment against
8,710/10,424 under `t`); it is reproduced here from source alone. **Confirmed.**

**Root cause, and it is instructive.** `phase_is_forced_switch` was minted eleven days earlier
(`045e8b8`, 2026-05-26) for the **TurnDelta obs history slot**, where "what phase did this window END
in" is the natural and correct read of `curr_ctx`. `adc0fe4` reused the attribute for a question it
does not answer — "was the decision that OPENED this window forced". Same name, different tense. No
test could catch it because both readings are true statements about the same delta.

### 3.3 A THIRD defect, in the same three lines, that probe M did not name

`episode_tracker.py:921` passes the **same upcoming-request `legal`** to `ProgressClock.update`, so
the trapped-helplessness gate (`progress_clock.py:178/193/241`, `len(legal.switches) > 0`) also
asks about decision `t+1`. The design specified *"a switch being legal **this decision**"*
(`design:711`).

Effect: a mon that was genuinely trapped at `t` (Mean Look / Wrap family / an empty bench) but whose
successor state permits switching is **charged for helplessness**; and the mirror case exempts a
free choice. Magnitude is small in gen3 OU (probe M's TRAPPED class is 2.9% of decisions) and probe
M's finding that TRAPPED is a clean null is unaffected — but it is the **same off-by-one, in the
same call, from the same commit**, and any fix must move both or the fix is half-done.

---

## 4. THE MINIMAL INTENT-RESTORING FIX — SPEC ONLY (retrain-class; do not implement mid-campaign)

Three changes, independent, in ascending order of blast radius. **F1 alone restores ~36% of the
charges to the intent; F1+F2 restore ~79%.**

### F1 — point both gates at the DECISION being charged (fixes §3.2 + §3.3)

*The design's sentence, implemented.*

**Change.** `ProgressClock.update` must receive the phase and the legality **of the decision that
opened the window**, not of the one that follows it. Two options, both small:

* **(a) preferred — carry it on the delta.** Add `decision_was_forced_switch: bool` to `TurnDelta`
  (`turn_delta.py:195` area), set from `prev_ctx.phase == "forced_switch"` at the three construction
  sites (`:405`, `:531`, `:573`). Leave `phase_is_forced_switch` **untouched** — the obs encoder
  (`turn_delta_encoder.py:649`), `opp_intent_labels.py:81` and `reward_manager.py:1068` all want the
  existing tense, and re-pointing it would silently change the obs vector. Switch
  `progress_clock.py:161` to the new field.
* **(b) — pass `prev_ctx.legal` alongside.** `EpisodeTracker.update_progress_clock` already holds
  `self._history[-2]`; thread `prev_ctx.legal` into `_progress_clock.update(delta, live, legal_prev)`
  (`episode_tracker.py:921`) and drop the `legal` parameter from the env call site
  (`gen3_env.py:365`) or keep it for the obs. `BattleContext` already stores `legal`
  (`battle_snapshot.py` — `mask`/`legal`/`active_move_ids` are recorded per decision), so no new
  capture is needed.

**Blast radius.**
- *Recorded behaviour that changes:* every forced-switch replacement stops being charged
  (−36.3% of all charges, ≈ −2.5 charges/battle ≈ +0.37 reward units/battle); the KO window stops
  being exempted and becomes a normal window (mostly it will be PROGRESS or a legitimate NO_OP — the
  KO turn usually dealt damage, so expect a small net increase there, nowhere near the 19,503
  exemptions it currently spends).
- *The obs scalar `turns_since_progress` changes* (the counter no longer increments on
  replacements) ⇒ **the observation stream changes** ⇒ this is retrain-class even though no dim
  moves.
- *Tests to update:* `progress_clock_fuzz_test.py`, `reward_manager_test.py`,
  `reward_redesign_test.py`, `reward_skip_parity_test.py` / `_fuzz_test.py`,
  `reward_invariants_e2e_test.py`, `reward_value_regression_fuzz_test.py`, `reward_tracker_test.py`,
  plus any golden-obs capture that pins column 1602 (`bias_tax_head_alignment_census.py`'s decode
  and its four gates).
- *Must NOT change:* `ARCH_SIGNATURE`, `MODEL_CONFIG_VERSION` (no weight shape), the obs LAYOUT.
- **Add a named regression test** (per the standing rule): a deterministic two-decision fixture
  where decision `t` is a move that KOs our own mon and `t+1` is the replacement — assert `n`
  unchanged and `last_penalty == 0.0` on the replacement window, and that it FAILS on revert.

### F2 — give `_is_progress` a switch clause, or stop charging switches (fixes §3.1 / path E)

Two spellings; **pick one, do not build both.**

* **F2a — a switch-progress predicate.** Add a clause: a voluntary switch is PROGRESS iff it
  *improved the position by an obs-knowable margin* — the natural candidates already computed
  elsewhere are (1) the incoming-KO belief fell (`_prev_active_ko_risk` → the new active's
  `active_risk`, the same quantity `pbrs_belief` and `stay_risk_tax` read), or (2) the switch-in's
  best damaging multiplier against the opp active strictly exceeds the outgoing mon's (the
  `se_switch` / `pivot_damage` machinery, currently dead code under `--all-shaping-pbrs`).
  *Blast radius:* this reintroduces a hand-tuned switch heuristic — exactly the class `928a00b`
  deleted on the argument that switching value is **learnable** from Φ_mat + `pbrs_belief` + the
  terminal. **Recommend against**, on the record of that commit.
* **F2b — exempt voluntary switches from the charge (preferred).** Where `delta.our_switch_to is not
  None` and the window was not forced, **freeze** (increment nothing, charge nothing) rather than
  NO_OP. Rationale that matches BOTH the design and probe M: the design's "pure tempo-pivot pays the
  toll" was written for a reward that also paid `+0.5` for the pivot; with the counterweights gone,
  a freeze is the composition-corrected reading of the same intent. Probe M supplies the empirical
  half — within the switch branch the tax's discrimination is **inverted** (Δ mean `d_out` **+0.0103**
  [+0.0076, +0.0131], the charged switches are worth *more* win probability than the exempt ones).
  *Blast radius:* removes 42.7% of charges; the anti-stall job is not lost, because a pivot-loop
  that makes no progress still gets charged on every **move** turn between the pivots, and
  `--draw-penalty` + the 250-turn forfeit remain the hard backstop. **A pure A↔B switch-loop becomes
  free** — that is the honest cost, and the stall-rate canary is the endpoint that would catch it.
  *Tests:* same list as F1, plus the anti-spam-collapse tests that assert a bounce is charged.

### F3 — the ZERO-AGENCY exemption, stated generally (subsumes F1's second half)

If a decision's legal action set contains **no action that could satisfy `_is_progress`**, do not
charge it. Today this is exactly the forced-switch case (F1) and the trapped case (already gated),
so F3 buys nothing over F1+F2 **unless** F2a is chosen — in which case F3 is the principled
statement F2a's clause is an instance of. *Listed for completeness; not recommended as a separate
build.*

### The control arm, for free

For the causal test the ledger already registered ("a `no_progress_tax`-OFF arm with switch-rate as
an endpoint"), **no code is needed**: `--no-progress-penalty 0.0` sets
`ProgressClock.no_progress_penalty = 0.0` (`gen3_env.py:326`) and every charge becomes
`-abs(0.0) = 0.0`, with the obs counter still ticking — a one-field, resume-immutable, value-checked
change. Caveat: `_bias_term_active` does not read the magnitude, so
`reward_class_composition` will still announce `1 BIAS (no_progress_tax)`. **Cosmetic, but it will
mislead the next reader of that run's `metadata.json`** — worth a one-line magnitude check in
`_bias_term_active` whenever this arm is run.

---

## 5. SCOPE 2 — THE CLEAN-WORLD CONFIG ENUMERATION

**Target (ledger `579279d`):** terminal ∈ {+1, −1}, draw 0; FULL PBRS from a frozen mature win-prob
head as the ONLY dense signal; every hand-tuned PBRS term and the tax OFF.

### 5.1 What today's flags CAN produce

The reward composition is decided by two predicates:
`_pbrs_term_active` (`reward_manager.py:389-398`) and `_bias_term_active` (`:401-422`).

| flags | TERMINAL | PBRS | BIAS |
|---|---|---|---|
| **production today** (all defaults) | 1 | **7** (material, belief, status, hazard, boost, opp_boosts, roar) | **1** (`no_progress_tax`) |
| `--stall-pbrs` added | 1 | **8** (+ progress) | **0** |
| `--no-all-shaping-pbrs` | 1 | **2** (material, belief) | **26** ← the additive objective the v9 era drifted into |
| `--no-all-shaping-pbrs --drop-redundant-bias --drop-switch-bias` | 1 | 2 | **~15** |

**The closest reachable point is `--all-shaping-pbrs --stall-pbrs` ⇒ 1 TERMINAL + 8 PBRS + 0 BIAS.**
The BIAS class — the only class that biases the converged optimum — can be driven to zero **today,
by flag**. The 8 PBRS terms cannot.

**`pbrs_material` and `pbrs_belief` are UNCONDITIONAL** — `_pbrs_term_active` returns `True` for them
with no gate (`:398`), and `_fold_material_pbrs` / `_fold_belief_pbrs` have no early return. **There
is no flag that turns them off.** That is the first build gap.

Turning off the other six is *anti-correlated with turning off the BIAS class*: `--no-all-shaping-pbrs`
kills five of them (status/hazard/boost/opp_boosts/roar) and simultaneously **switches 26 BIAS terms
back on**, because `_bias_term_active` uses `asp` as its master gate (`:410-411`). So the flag
surface cannot express "no hand PBRS **and** no BIAS".

### 5.2 What the clean arm NEEDS BUILT (four items)

**B1 — a way to zero the 8 hand PBRS terms without re-enabling BIAS.** Minimal shape: one new
resume-immutable `RewardConfig` field, e.g. `hand_shaping: bool = True`, whose `False` value
early-returns from all eight `_fold_*_pbrs` (including material and belief) and is ORed into
`_bias_term_active`'s zeroing. Blast radius: `_REWARD_IMMUTABLE_FIELDS` (`model_version/constants.py:172`),
`_REWARD_FIELD_FLAGS`, `check_reward_config`, `reward_class_composition`, `reward_defaults_test.py`,
`reward_skip_parity_test.py`, `MODEL_CONFIG_VERSION` bump (value-meaning, not weight-shape ⇒ **no**
`ARCH_SIGNATURE` bump).

> ⚠️ **State this honestly in the experiment's write-up:** every PBRS term is
> **policy-invariant by construction** (`Φ(terminal)=0`, telescoping — the manager's own docstrings
> and `928a00b`'s verification say so). Removing them therefore **cannot change the optimal policy**;
> it changes *learning dynamics and conceptual complexity*. The clean-world claim's real content is
> "the hand terms cost more in interference and tuning than they buy in credit assignment", **not**
> "the hand terms bias the objective". The only term that biases the objective is the BIAS class, and
> that is already flag-zeroable today.

**B2 — a ±1 terminal.** `VICTORY_VALUE = 30.0` is a **module constant** (`reward_weights.py:20`),
read at `reward_manager.py:1650` (`+VICTORY_VALUE`) and `:1658` (`-VICTORY_VALUE`). It is **not** a
config field and **not** reachable by any flag. Minimal shape: promote it to
`RewardConfig.victory_value: float = 30.0` (same immutable/value-checked treatment as `draw_penalty`),
substitute at both sites. Blast radius: the same list as B1, plus every test that asserts a `±30`
literal (`reward_redesign_test.py`, `reward_invariants_e2e_test.py`,
`reward_value_regression_fuzz_test.py`), plus `MAT_HP_WEIGHT`/`MAT_ALIVE_WEIGHT`, which are
calibrated *against* the 30 scale (moot if B1 is on, since Φ_mat is then off).

**B3 — draw = 0.** `--draw-penalty 0.0` handles the **timeout** branch (`:1657-1658`). A **pre-cap
tie** — `finished and not won and not lost and turn < 250` — is hardcoded `-VICTORY_VALUE` (`:1658`,
the `else` arm) and would read `−1` after B2, not 0. Rare, but a spec deviation to decide
deliberately.

> 🚨 **A `draw = 0` with a `±1` terminal INVERTS the outcome ordering the current numbers were built
> to enforce.** `draw_penalty = −35 < −VICTORY_VALUE = −30` exists precisely so that *"stalling to
> the turn cap is strictly worse than losing cleanly"* (`RewardConfig` comment `:89-93`; design
> `:749-753`). At `{+1, −1, draw 0}` the 250-turn stall becomes the **best non-winning outcome** and
> a losing agent's optimal play is to run out the clock. With every anti-stall term also removed
> (`no_progress_tax` off, `stall_tax` off, `Φ_progress` off), **nothing in the clean arm opposes
> that.** Recommend `draw = −1` (equal to a loss) or `−1.2`, and register stall-rate / mean-game-length
> as a **primary safety endpoint**, not a secondary one.

**B4 — `--win-prob-pbrs-source <ckpt>`** — see §6.

### 5.3 The full clean-arm command, as it would read once B1-B4 exist

```
--hand-shaping false            # B1: zero all 8 hand PBRS + the whole BIAS class
--victory-value 1.0             # B2
--draw-penalty -1.0             # B3 (NOT 0.0 — see the ordering hazard above)
--win-prob-mode read_only       # the head must exist to be read
--win-prob-pbrs-coef <c>        # the ONLY dense signal
--win-prob-pbrs-source models/<rev1>/checkpoints/<ckpt>.zip   # B4
```

Control arm = today's defaults, same seeds/teams/steps.

---

## 6. `--win-prob-pbrs-source <ckpt>` — SPEC (not built)

### 6.1 What exists

`winprob_pbrs.py` (landed `e5240e8`) is already the right shape: **pure shaping arithmetic**
(`successor_potential` / `pbrs_shaping`) separated from **the φ read** (`buffer_potentials` →
`_forward_phi`). Only `_forward_phi` (`:185-195`) is model-bound:

```python
model.policy.predict_values(obs_t)          # runs the LIVE extractor
fe = getattr(model.policy, "features_extractor", None)
return _phi_from_logits(getattr(fe, "last_win_prob_logits", None), "buffer forward")
```

So the change is **one function and one loading site**. Everything downstream (terminal/truncation
conventions, in-place reward mutation, GAE re-run, the `train/pbrs_reward_share` metric) is unchanged.

### 6.2 Where the frozen head loads

Reuse the **`--distill-teacher` precedent verbatim** (`main/train/model_build.py:204-226`):
`agents.model.snapshot.load_foreign_opponent(zip, current_version=…, device=str(model.device),
config_path=cfg)` via `fixed_opponent_pool._resolve_zip_and_config`, then
`.policy.set_training_mode(False)`, stashed on the model as `model._winprob_phi_source`. A bad path
must `os._exit(FATAL_CONFIG)` like the teacher path does, never crash-restart. Add
`"_winprob_phi_source"` to `_excluded_save_params` (`instrumented_ppo/hparams.py:374`) — a frozen
foreign model must never be pickled into a checkpoint.

`load_foreign_opponent` already tolerates a foreign obs space (ppo.py:845: *"Each frozen teacher has
its OWN (older) obs space — pass only the keys it knows"*), which is what makes a *prior-generation*
φ source viable at all.

### 6.3 Is a full frozen-extractor forward required? — **YES. There is no head-only shortcut.**

`WinProbHead.forward(value_pooled)` (`aux_value_heads.py:45-47`) consumes `value_pooled [B, D_MODEL]`
— the **whole-board value pool** produced by the frozen network's own trunk with its own weights
(delivery graph: `CLSPool.value_cls -> WinProbHead`, `delivery_graph.py:773`). Running the frozen
*head* over the *live* model's stashed `value_pooled` computes a function of a representation the
head was never trained on: not the calibrated φ, not any φ, and it would drift with the live trunk —
destroying the exact property (a genuinely FIXED potential) the frozen source exists to buy.

**So: a second, full `no_grad` extractor forward over the buffer, with the frozen model.**

### 6.4 Cost

It **replaces** the live forward rather than adding to it: `buffer_potentials` would call
`_forward_phi(model._winprob_phi_source or model, …)`. Per the module's own note (`:51-52`) that is
*"roughly 1/`n_epochs` of one training epoch"* — with `--n-epochs 10`, **~10% of one epoch ≈ ~1% of
`train()`**, unchanged from today's live-φ path. The genuinely new costs:

* **GPU memory** — one extra full extractor (frozen, eval mode, no optimizer state, no grads). Same
  order as one `--distill-teacher`, which the tree already runs at N ≥ 3.
* **Startup** — one `load_foreign_opponent` (seconds).
* **`--compile-trainer` interaction, UNVERIFIED and worth checking before building:** the frozen
  model is a *separate* policy object; whether it should be compiled (a second Inductor graph, its
  own warm-up) or left eager has not been measured. Eager is the safe default — it runs once per
  rollout, not per minibatch.

### 6.5 The affine detail: φ' = 2p − 1 vs coef 2 on p

**They are exactly equivalent, and the code should prefer the coefficient.**

PBRS is affine-invariant in φ. For `φ' = aφ + b` with constant `a, b`:

```
F'(s,s') = coef·(γφ'(s') − φ'(s))
        = coef·a·(γφ(s') − φ(s))  +  coef·b·(γ − 1)
```

The first term is the original shaping with `coef → coef·a`. The second is a **state-independent
constant** `coef·b·(γ−1)` added to every transition — at `γ = 0.9999` and `b = −1` that is
`+1e-4·coef` per step, which is a per-step constant, not a per-policy quantity, and telescopes into
a length-dependent additive term. So:

* **`φ' = 2p − 1` ≡ `coef ← 2·coef` on `φ = p`, plus a per-step constant of `coef·(γ−1)·1 ≈ −1e-4·coef`.**
* That residual constant is *not* strictly zero at `γ < 1`, and it **rewards longer episodes** when
  `b < 0` (each extra step adds `+1e-4·coef`) — a tiny stall incentive, in an arm that has already
  deleted every anti-stall term. Small (a 250-turn game accrues `≈ 0.025·coef`), but it is exactly
  the wrong sign.

**Recommendation: keep the code's existing `φ = sigmoid(logit) ∈ [0,1]` and express the [−1,+1]
mapping as a coefficient.** i.e. write the clean arm as `--win-prob-pbrs-coef 2c` rather than
introducing a `2p−1` spelling. It is one fewer moving part, it keeps `φ(terminal)`'s convention
(`successor_potential` sets `φ(s′) := 0` at terminals — a convention that is *correct for a [0,1]
potential* and would be **wrong** for a [−1,+1] one, where a loss "should" be −1 and 0 is the middle
of the range), and it avoids the length term above. **This is the one place where the ledger's
"φ' = 2·P(win)−1" phrasing would, implemented literally, introduce a real (small) defect** — the
terminal convention, not the affine algebra, is what breaks.

---

## 7. WHAT THE CLEAN ARM SILENTLY NEEDS — sizing flags (raised, not solved)

1. **The stall ordering** — §5.2/B3. The single largest risk. **Primary safety endpoint: stall rate
   and mean game length.**
2. **PopArt.** `--use-popart` normalizes value TARGETS by a running (μ,σ) over `rollout_buffer.returns`
   (`instrumented_ppo/ppo.py:356-358`), so it is **scale-free by construction** — a ±1 stream is fine
   and needs no retuning. The real question is *variance*: with every dense term removed, returns
   become near-sparse and σ collapses toward the terminal magnitude, which **amplifies** normalized
   targets and can make the value loss twitchy early. `--use-popart` also **requires an explicit
   `--clip-range-vf none`** (parser `:331-336`) — do not forget it on the clean arm.
3. **`--ent-coef 0.02` was tuned against a ±30 dense stream — but it is *mostly* scale-free.** SB3
   normalizes advantages per minibatch (`ppo.py:413`, `if self.normalize_advantage`), which forces
   the policy-gradient term's scale independent of the reward scale, so `ent-coef` competes against a
   normalized surrogate either way. **Do not rescale it by 1/30.** What *does* change is the
   advantage *signal-to-noise* before normalization (sparser rewards ⇒ noisier advantages ⇒ the
   normalized gradient is relatively noisier), so if anything the clean arm may want **more**
   exploration, not less. Treat as an open sizing question, not a mechanical rescale.
4. **`--vf-coef`** — value loss runs in PopArt-normalized space (`ppo.py:429-435`), so it is likewise
   scale-free. But it is **resume-immutable and FATAL on mismatch** (`check_vf_coef`), so it must be
   chosen at launch.
5. **`coef` sizing for the φ-PBRS.** The only meter is `train/pbrs_reward_share` (the shaping's mean
   |·| over the unshaped stream's mean |·|). In the clean arm the unshaped stream is **terminal-only**,
   so that ratio's denominator is near zero on most steps and the metric will read enormous and
   uninformative. **The metric needs a companion for this arm** (e.g. shaping absmean against the
   terminal magnitude, or per-episode discounted shaping sum against ±1) — otherwise the arm ships
   with no way to tell whether the coefficient is sane. Flagging as a build item, not solving it.
6. **`--win-prob-mode` must be `read_only` or `shaping`** for the head to exist at all — but note
   that on the *clean* arm the head being read is the **frozen** one, so the LIVE run's own
   `--win-prob-mode` governs only whether the live head trains as a diagnostic. `read_only` is the
   right choice (risk-free, and it keeps a *live* φ trajectory to compare against the frozen one — a
   free measurement of how far the frozen potential has drifted from the run's own beliefs).
7. **γ.** `RewardConfig.gamma` is asserted `== 0.9999` (`from_args`, `:152`) and
   `pbrs_shaping` deliberately has no separate discount (`:139`) — nothing to do, but it is why the
   `2p−1` residual term above is ~1e-4 rather than ~0.
8. **Provenance.** `--win-prob-pbrs-source` must be recorded in `metadata.json` (the `cli_args`
   path already does this) **and its resolved git hash / `arch_signature` recorded too** — a clean-world
   run is uninterpretable if the identity of its frozen potential is not pinned.

---

## 8. CAVEATS

1. **No code was executed.** Every claim is source reading plus git history; probe M's numbers are
   quoted, not re-measured.
2. **§3.3 (the `legal` off-by-one) is source-derived and NOT measured.** Its magnitude is bounded
   above by probe M's TRAPPED class (2.9% of decisions) but the actual mis-attribution rate is
   **UNVERIFIED**.
3. **The clause-(ii) switch escapes** (contact abilities, opponent Rest) are derived from the
   predicate and gen3 mechanics, **not** counted in the corpus.
4. **The §6.4 cost figures are the module's own claim** (`winprob_pbrs.py:51-52`) carried over, not a
   fresh benchmark; the `--compile-trainer` interaction is explicitly **UNVERIFIED**.
5. **Blast-radius test lists are derived by filename/grep**, not by running the suite — treat them as
   a starting set, not a complete one.
6. This document is a **spec and an intent record. Nothing here is implemented**, per the standing
   rule that reward changes are retrain-class and frozen mid-campaign.
