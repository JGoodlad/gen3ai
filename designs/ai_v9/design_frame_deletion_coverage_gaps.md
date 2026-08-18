# Frame deletion — what the dV licence could not see

**Status:** 🔬 open — written 2026-08-17 alongside `gen3_frame_deletion_v1`, which SHIPS WITH THESE
GAPS OPEN by owner decision. This document exists so they can be reconciled out of band rather than
blocking gen-14. **Nothing here is a defect in the deletion.** It is the inventory of what the
deletion costs, which the measurement that licensed it was structurally unable to report.

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

## 3. What ships OPEN — the three gaps

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
