# Frame deletion — what the dV licence could not see

**Status:** 🔬 open — written 2026-08-17 alongside `gen3_frame_deletion_v1`, which SHIPS WITH THESE
GAPS OPEN by owner decision. This document exists so they can be reconciled out of band rather than
blocking gen-14. The §3.1-3.3 gaps are not defects in the deletion — they are the inventory of what
it costs, which the measurement that licensed it was structurally unable to report. **§3.5 and §3.7,
added by the 2026-08-17 follow-up audit, ARE defects** and are marked as such.

---

## 1. The methodological finding (the durable part)

The frame deletion was licensed by gen-13.5 §4: `event_seats` dV **2.7714** vs `frames` **1.3015**,
ratio 0.47, on a falsified instrument (positive control + exact-zero null arm). That reading is
sound and is not in question.

**But dV answers one question and the deletion needed two.**

| question | answered by |
|---|---|
| Does the trained model LEAN on this block? | dV — yes, this is what §4 measured |
| Does every FACT in this block have a home elsewhere? | **nothing measured this** |

These come apart in a specific and predictable way: **a fact with no substitute reads LOW on dV
whenever the model never learned to use it.** Low dependence is equally consistent with "redundant"
and with "delivered so badly it was never learned" — and the second reading argues for fixing
delivery, not for deleting the fact. An ablation cannot tell them apart, because both produce the
same number.

The gaps below were found by a different method: enumerating the fields the `feature_coverage`
probes vary, and checking each against the 19 (now 20) event-window columns. That is a **coverage
audit**, not a dependence measurement, and it is the check this class of deletion needs.

> **Proposed standing rule:** an irreversible block deletion needs BOTH a dependence reading (does
> the model lean on it) AND a per-fact coverage audit against the substitute (does each fact have a
> home). §4's falsification set — positive control, exact-zero null, independent route — is about
> trusting the *instrument*; this is about the *scope* of what the instrument can say.

---

## 2. What was closed before shipping

**`cant_reason` — CLOSED (`EVENT_T_CANT` + column 19 `cant_id`).**

"This mon could not move, and why": full paralysis, sleep, flinch, recharge. `EventKind.CANT` was
already in the battle event log *with its reason*, and the `TurnDelta` fold already read it into
`our_cant_reason` / `opp_cant_reason` — but `EventWindowTracker` emitted nine event types and CANT
was not among them, so the fact reached the model through the lag frames and nothing else.

Closed because it was cheap, because gen3 OU is status-dense (Thunder Wave, Spore, Body Slam para),
and because it is the clearest instance of the pattern above — the data was present at every layer
except the one that delivers it.

Verified end to end: writing `cant_id` into an event row moves BOTH the policy and value heads.

---

## 3. What ships OPEN — the gaps

**§3.1-3.3** are the original three, found by repointing the `feature_coverage` probes (what the
deletion COST). **§3.4-3.7** were added by the 2026-08-17 follow-up audit, which asked the inverse
question (what the window never carried) and additionally turned up **two live defects** — §3.5 (the
move-magnitude column is GIGO) and §3.7 (`ability: Damp` crashes the obs encoder). Those two are
NOT accepted losses; they are flagged for the owner.

### 3.1 `our_attempted_switch_spec` — which bench mon a refused switch aimed at

**Status:** ACCEPTED on VALUE grounds. **⚠️ This section originally said "structurally unreachable"
and that was WRONG — corrected on review (ledger, 2026-08-17), and the error is kept visible here
because a false impossibility is the kind of claim that outlives its context and forecloses work
nobody re-examines.**

The mistaken reasoning: `Gen3Battle.record_choice_rejected` documents that the attempted target
*"is not on the wire and is recovered at fold time from the action index"*, from which I concluded
an events-only fold could never carry it. **"Not on the protocol wire" is not "not available at
emission."** `record_choice_rejected` (`gen3_battle.py:202`) is called from the PLAYER layer, which
knows the action it just attempted — the fact is right there when the event is created.

So the clean path exists and does not strain anything: **event-payload enrichment at emission** —
the attempted target enters the event's `value` dict, the LOG gains the fact, and the fold stays a
pure function of the log (the invariant that rules out recurrence in the first place). That is
strictly NOT option D's fold-time action-index threading, which would have broken fold purity; the
option-D framing was an artifact of the same error.

**What survives today:** the rejection FACT (`EVENT_T_SWITCH_REJECTED`) and trappedness itself, on
the per-mon slots (`gen3_entity_rehome_v1`). **What is lost:** the identity of the refused target —
"I am trapped" vs "I tried to bring Skarmory and was refused".

**Ruling: ACCEPT the loss, on value grounds only** — it is the narrowest of the three and the
trap-reveal signal is intact. If its value ever materialises, enrich the payload.

### 3.2 `our_faint_causes` — why a mon fainted

**Status:** partially inferable, not represented.

The eight causes (attack / weather / status / hazard / recoil / selfko / leechseed / other) had a
dedicated multi-hot in the lag frames. The event window's `EVENT_T_FAINT` row has **no cause
column**.

The honest argument that this is survivable: the window is a SEQUENCE, so a faint preceded by their
MOVE row with damage magnitude *is* the attribution — adjacency replaces an explicit field, which is
what "the sequential residue" means. The honest counter: **weather, status, hazard and Leech Seed
deaths emit no preceding event to infer from.** Residual damage is not an event. So the inferable
subset is roughly {attack, recoil, selfko} and the non-inferable subset is roughly {weather, status,
hazard, leechseed, other}.

**Judgement: the highest-value of the three.** "Did hazards kill me or did they" is a real strategic
distinction, and the non-inferable half is exactly the slow-attrition half that matters in the stall
regime §7 keeps flagging. **The obvious fix is a `cause_id` column on the FAINT row, the same shape
as the `cant_id` fix** — ~32 obs dims, one tracker branch, one embedding.

### 3.3 `our_item_lost` / `opp_item_lost` — an item was CONSUMED

**Status:** conflated, not absent.

`EVENT_T_ITEM_REVEAL` exists, but it does not distinguish "an item was revealed" from "an item was
consumed and is now gone". Sitrus Berry eaten and Leftovers revealed produce the same row.

**Judgement:** middling. Item identity is separately modelled (`--item-belief`, v83), so the belief
stack knows what they hold; what is lost is the transition to holding-nothing.

**Fix — and it must cover the full item-GONE FAMILY, not just consumption** (refined on review):
gen3 has THREE ways an item stops being held — **consumed** (berries, herbs), **removed** (Knock
Off, permanent in ADV), and **swapped/stolen** (Trick / Thief / Covet). A bare `consumed` flag
leaves the conflation half-alive, which is the failure this fix exists to end. Use a TRANSITION
ENUM on the ITEM row: `revealed / consumed / removed / swapped`.

---

### ⟨FOLLOW-UP AUDIT, 2026-08-17⟩ — §3.4 … §3.7

The three gaps above were found by repointing the `feature_coverage` probes: they are facts the LAG
FRAMES carried that the event window does not. A separate follow-up audit asked the inverse question
— **not "what did the deletion cost" but "what did the window never carry in the first place"** — for
three battle facts nobody had traced end to end. It found one more absence, one MISATTRIBUTION class
(worse than an absence: the row reaches the network carrying a false number), and one **live crash in
the just-shipped `cant_id` feature**.

Method note, because it is why these were found at all: each verdict rests on **observed protocol from
real bridge battles** (`gen3ou` via the node bridge, plus constructed single-turn scenarios through
the omniscient `utils/bridge/damage_probe.js`), not on reading the fold's intent. Two of the four
contradict what the code reads like it does. Probes:
`src/agents/model/feature_coverage/substitute_confusion_feature_test.py` (19 passing + 8 strict-xfail).

### 3.4 Substitute — a sub absorbing damage, and a sub BREAKING

**Status:** absent. The ATTACKER's row is fine; the DEFENDER's sub is invisible.

Measured protocol: a hit a Substitute absorbs emits **no `|-damage|` at all**. Sub survives →
`|-activate|p1a: Blissey|Substitute|[damage]`; sub breaks → `|-end|p1a: Blissey|Substitute`. The
effectiveness/crit trio DOES still fire either way (Showdown raises it inside `getDamage`, which
`substitute.onTryPrimaryHit` calls) — live: `|-supereffective|p1a: Blissey` immediately before
`|-end|`.

**What survives:** the attacker's `EVENT_T_MOVE` row reads outcome **HIT** with **magnitude 0** —
"it connected and did nothing" — plus the correct effectiveness one-hot. And `substitute` is a binary
volatile slot in the active-context block, so *"a sub is up right now"* is explicit.

**What is lost:** the TRANSITION. `|-activate|` folds to `EventKind.ACTIVATE` and `|-end|` to
`EventKind.VOLATILE_END`; `EventWindowTracker.update` (`episode_tracker.py:365-496`) has a branch for
neither, so **no row records a sub going up, chipping, or breaking**. "My sub just broke" is the most
decision-relevant moment of a sub turn and the window is silent on it. Only weakly inferable:
HIT-with-zero-magnitude is also what a hit on a **Protect**ed mon looks like once the immune one-hot
is not set.

**Judgement: middling-to-high.** Substitute is a top-tier gen3ou mechanic (Suicune / Snorlax /
Jirachi sub-stalling is a whole archetype). **Fix sketch:** the same shape as the `cant_id` fix — an
`EVENT_T_VOLATILE` row type with a small volatile-id column and a start/end/activate transition flag,
which would also subsume Leech Seed, Taunt, Encore, Disable and confusion's own start/end.

### 3.5 🚨 Residual damage is CREDITED TO THE ATTACKER'S MOVE — the magnitude column is GIGO

**Status:** NOT a gap — a wrong number, shipped. **Flagged for the owner as fix-now-or-not; not
touched here.**

`EventWindowTracker` attaches a `DAMAGE` event to the other side's open MOVE record only when the
damage has no `[from]` clause and lands on that move's recorded target. The clause test is
`not e.value.get("from")` (`episode_tracker.py:400`) — but the parser writes the `[from]` clause to
**`value["reason"]`** (`gen3_battle.py:502-503`, and 516-517 / 537-538). `value["from"]` is set only
for MISS/FAIL move-suffixes (`gen3_battle.py:346,349`); a DAMAGE event never carries it. **The guard
is dead code and always passes.**

Consequence: every clause-carrying residual on the move's target that turn is SUMMED into the
attacker's move magnitude — sandstorm, hail, burn, poison/toxic, Leech Seed, confusion self-hit,
recoil. On a Tyranitar-sand board that is essentially every move row in the game.

Measured live, not derived: a **0-BP `confuseray`** row read `hp_delta = -0.12`, and a **FAILED**
`confuseray` read `-0.13` — those numbers are the confused Snorlax's own self-hits. Unit reproduction
(`test_end_of_turn_residuals_are_not_credited_to_the_attackers_move`): a real 0.10 hit plus 0.0625
sandstorm plus 0.0625 burn folds to `-0.225`.

A second, independent path produces the same class **without any `[from]` clause**: Substitute's own
25% HP cost is a bare `|-damage|` on the mon that is also the opponent's recorded target, so it is
credited to the opponent's move. Live: `machamp seismictoss hp_delta = -0.2493` on a turn its Seismic
Toss dealt **zero** (the sub ate it) — 0.2493 is exactly Blissey's 178/714 sub cost. Order-dependent:
it only lands when the opponent moved first that turn, so the same board yields different magnitudes
depending on speed order.

**Why it was invisible:** `event_window_fuzz_test.py`'s independent oracle (`_oracle_rows`) reads the
same wrong key, so the fuzz agrees with the bug. That is the mirror-oracle trap — an oracle derived
from the implementation validates the implementation's mistakes.

**Judgement: the highest-value item in this document, and the only one that is a defect rather than
an accepted loss.** Fix sketch: read `value["reason"]`; add a typed `from_clause` accessor on
`BattleEvent` so no third consumer can guess the key again; rewrite the fuzz oracle to read the
accessor; and add an explicit "self-inflicted damage (Substitute cost, recoil, Struggle, Belly Drum,
confusion) never attaches to the other side's move" rule, since target-identity alone cannot separate
it.

### 3.6 Confusion self-hit — a lost turn with no row

**Status:** absent as a fact; actively MISREPORTED via §3.5.

gen3 emits `|-activate|p2a: Snorlax|confusion` then `|-damage|p2a: Snorlax|87/100|[from] confusion`.
There is **no `|move|` line and no `|cant|` line** — so confusion's absence from `CANT_REASONS` is the
CORRECT modelling of the protocol, not an oversight.

**What survives:** `confusion` is a binary volatile slot in the active-context block, so *"this mon is
confused right now"* — and hence the standing 33% risk — is explicit.

**What is lost:** the RESOLUTION. A confusion self-hit is strategically the twin of a full-paralysis
`EVENT_T_CANT` row (a turn thrown away, plus self-damage), and it produces **zero rows**: the
`-activate` is an `ACTIVATE` with no tracker branch, and the damage is filtered out of its own side's
accounting. Worse, per §3.5 it is not merely dropped — it is re-credited to the OPPONENT's move row.
So the window does not say "they lost a turn to confusion"; it says "our status move dealt 12%".

**Judgement: middling on its own, higher once §3.5 is fixed** — with the residual guard repaired the
damage would be correctly dropped and this becomes a clean absence rather than a lie. **Fix sketch:**
the §3.4 `EVENT_T_VOLATILE` row covers confusion start/end; the self-hit itself wants either a
dedicated row type or, cheaper, a `self_inflicted` flag on a damage-carrying row.

### 3.7 🚨 `ability: Damp` is a `|cant|` reason the vocabulary lacks — it CRASHES the run

**Status:** LIVE DEFECT in the just-shipped `cant_id` feature. **Flagged for the owner; not fixed
here.**

**Freeze is NOT the problem.** `constants.py`'s comment abbreviates the reason list as *"(full
paralysis / sleep / flinch / recharge)"*; that is prose, not coverage. `frz` is `CANT_REASONS[1]` and
always was. A census of every `add('cant', …)` over the gen3-reachable Showdown sources gives 13
reasons; 12 are in the vocabulary.

The thirteenth is **`ability: Damp`** (`data/abilities.ts:805`, `onAnyTryMove`, blocking Self-Destruct
and Explosion; the rust port emits it too — `src/rust_sim/src/protocol.rs:690-692`). It is not modded
out of gen3, Damp is gen3-legal on Psyduck/Golduck, the Poliwag line, Wooper/Quagsire, Politoed, the
Horsea line and the Paras line, and Explosion is ubiquitous in gen3ou. So the reason is reachable in
ordinary play.

`normalize_cant_reason` is crash-don't-drop: it raises `UnknownCantReasonError`, which propagates out
of `state_encoder.encode` (`state_encoder.py:355`) and kills the episode — and, in training, the run.
**Reproduced on battle #1** of a scripted Quagsire-vs-Snorlax bridge battle.

Note this is **not new with `cant_id`** — `TurnDeltaEncoder` normalised the same reasons — but the
frame deletion made the event window the fact's only route, so the exposed surface did not shrink.

**A second defect rides the same row, and adding `"damp"` alone would ship it.** Showdown puts this
`|cant|` on the ABILITY HOLDER with the blocked move as its argument:
`|cant|p1a: Quagsire|ability: Damp|Self-Destruct|[of] p2a: Snorlax`. Our fold therefore emits
`actor=quagsire, side=ours, move_id=selfdestruct` — naming a mon that never had Self-Destruct as the
one that could not move, while the side that actually lost its turn is the opponent. Live record:
`{'t': 10, 'actor': 'quagsire', 'side': 'ours', 'move_id': 'selfdestruct', 'cant': 'ability: Damp'}`.

**Judgement: fix-now candidate, owner's call.** Fix sketch: add `"damp"` to `CANT_REASONS`, AND
re-attribute the row using the `[of]` clause (the blocked mover) so `side`/`actor` name the mon that
lost the turn. The durable lesson matches §3.5's: the vocabulary was derived by reading
`conditions.ts` (where 5 of the 13 live) rather than by censusing every `add('cant', …)` the format
can reach, so the ability- and move-sourced reasons were assembled from recall.

---

## 4. The translation insight (why the probe repoint is incomplete)

Repointing the `feature_coverage` probes surfaced a structural mismatch worth recording:

> **A `TurnDelta` is a per-turn AGGREGATE. The event window is a SEQUENCE.**

One turn produces our move AND their move AND the resulting status AND the faint. The lag frame held
all of that in one 159-dim slot. The faithful event-window translation is therefore **several rows**,
not one — `_support.delta_to_event_cols` returns a single row and is lossy by construction for any
delta that carries both sides.

That is why ~22 probes remain red: not because the facts are missing, but because the translator
cannot express a two-sided turn. **The fix is `delta_to_event_rows` returning a list**, with
`obs_with_event` writing several rows. It is a design correction rather than a bug fix, and it is
deliberately left for this reconciliation rather than rushed before a launch.

The translator's `b != v` guard — which refuses to run a probe whose two deltas translate
identically — is what surfaced this at all, and caught two real bugs in the mapping itself (a Status
enum's `.value` is an int, so every status collapsed to one id; and `anchor_delta`'s presence anchor
shadowed the field actually under test in 35 probes). **A translation layer without that guard would
have reported those as passes.**

---

## 5. Reconciliation options (for the owner, out of band)

| option | scope | cost |
|---|---|---|
| **A. Accept all three** | Record in ARCHITECTURE.md §1.6; revisit only if gen-14 underperforms | 0 |
| **B. Close faint-cause only** | `cause_id` column on the FAINT row; +32 obs dims | ~1 h, signature-neutral within gen-14's own bump |
| **C. Close faint-cause + item-consumed** | + a `consumed` flag or a distinct event type | ~1.5 h |
| **D. Close all three** | + thread the action index into the event fold (contract change) | ~3 h, and the last is the invasive one |

**RULED (owner, 2026-08-17, refined on review) — option C, at the next pre-launch signature
window:** close faint-cause AND the item-GONE family; ACCEPT the refused-switch target on value
grounds. My original recommendation of B is superseded on two counts: the non-inferable faint
causes are exactly the stall-attrition class C6 flags, and **a CONFLATED signal is worse than an
absent one** — which upgrades the item gap from "middling" to worth closing, since a single
`ITEM_REVEAL` row that silently means four different things actively misleads.

Gen-14 launched on the shipped state (the deletion + the cant fix), so these land at gen-15's
signature window rather than being retrofitted mid-generation. If gen-14 comes back INFERIOR, §2 of
the gen-14 runbook already asks whether `event_seats` rose — and this document is the list of
candidate explanations to check first.

---

## 6. What would falsify the concern entirely

If gen-14 reads NON_INFERIOR on the tail-4 dense ladder AND its `event_seats` dV rises above
gen-13's 2.7714, then the seats absorbed the frames' role including these facts' contribution, and
all three gaps are academic. That is the outcome to expect if the facts were genuinely unused —
which, per §1, is exactly what a low dV is consistent with. **This document is the hedge against the
other reading, not a prediction that it is right.**
