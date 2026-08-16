# design — HISTORY IN THE ENTITY WORLD: E9 steps 2–3, the compiled tier, and the last-turn question

**Status:** Tier **H-A BUILT and SHIPPED** (v79 `gen3_pair_history_v1`, 2026-08-15): the
last-action fields (H-A1, `POKEMON_FULL_DIM` 116→122) and the pair-history block + `h` edge
family (H-A2, obs 2669→2921; the family is opt-in, NOT in the production string — gen-12 is the
intended first enable). The independent event-log fuzz caught two real bugs before ship (a
fainted-active resurrection on forced-switch resync; a pre-existing recency cross-episode reset
leak). **Tier H-B is BUILT too** (v81 `gen3_event_window_v1`, 2026-08-16: the fold + the 32×19 obs
window + the opt-in `--history-events` event seats; the §6 row-2 DELETION of the 7×159 frames
deliberately NOT taken — that lands with the generation that enables the seats, per the
one-behavioral-change discipline; pre-enable gates: the event-fold fuzz + the obs benchmark).
**H-C is BUILT too** (`gen3_event_ref_edges_v1`, same day: the `r` edge family — structural
`[is_actor, is_target]` reference cells from each event seat to the 12 live tokens, side-gated
against mirror-species false links, zero-init, riding the families string; requires
`--history-events`). All three tiers now exist opt-in; §6's sequencing governs ENABLEMENT
(H-A's verdict → enable H-B seats; H-B's verdict → enable `r` + delete the 7×159 frames). Elaborates `design_generation_roadmap.md` §4's decided
direction (recency features → turn/event tokens → entity-linked event tokens; recurrence RULED
OUT) with the full mapping from today's history surfaces. E9 step 1 (the per-mon recency
triplet) is SHIPPED; this doc owns the rest. The concept frame is `designs/learning/entity_tokens_biases_pointers.md`
(history's job in a POMDP; §6.9 on what stays positional).

---

## 0. Constraints (all standing, none negotiable here)

1. **Event-log purity.** Every history representation must be a pure function of the replayed
   event log per decision window — reconstruction, reroll-parity, clone-search and the
   obs-roundtrip fuzz all depend on it. This is the constraint that ruled out recurrence, and it
   is why every tier below is a *re-encoding of trusted data*, not new state-tracking.
2. **Public events only.** Everything folds from observed protocol lines (the recency-triplet
   precedent) — no privileged key can feed a history feature.
3. **Within-battle only.** Self-play opponents are fresh per episode; cross-battle opponent
   modeling is a separate slow-clock context (FiLM-class), not history.
4. **Time may stay positional — as RECENCY, not as slot index.** §6.9: "before" is a real
   asymmetry and must survive; what must die is the v51-for-time defect, where the same event at
   a different lag occupies different weights. One smooth log-scaled turns-ago channel carries
   the asymmetry; no weight is ever indexed by lag.
5. **Budgets.** The obs-build benchmark gates every tracker addition; the B=1 CPU forward gates
   every new seat (the entity spike measured n=64 seats at +0.27 ms — sub-quadratic, but measure
   again at the real window size).

---

## 1. What history is TODAY — the complete inventory

Four surfaces, one of which is the problem:

| Surface | Size | Nature |
|---|---|---|
| **TurnDelta frames** — 7 × 159, positional-in-time, + a per-slot positional embedding on the 7 trunk history tokens | 1113 obs dims (42% of the vector) | the problem: lag-indexed weights, entity-blind embedded ids, 7-turn horizon |
| Prev-turn action mask | 11 obs dims | our own last-turn legality, positional |
| Per-mon recency triplet (`since_seen/acted/hit`, E9 step 1) | 36 obs dims | already entity-native — the pattern to extend |
| **Compiled cross-turn state** (the invisible tier): revealed moves/items/abilities `known` flags, sleep & toxic counters, sleep-wake belief, choice-lock, protect-success counter, pending Wish, `turns_since_progress`, the clock, HP-type tracker | scattered through the per-mon slots and global block | history already converted into sufficient statistics — a year of piecemeal E9, and the model's largest actual history consumer |

The per-slot TurnDelta layout (`turn_delta_encoder.py`, the "today" column of §5's mapping): per
side, the move block `[move_id, power, has_secondary, has_recoil, type_id]`; switch/failed-to-move
bits; `cant` reason one-hots (12); HP delta sums and per-slot HP levels (6 per side); fainted bits
+ faint-cause multi-hots (8); effectiveness one-hots (4); move-outcome one-hots (hit/miss/fail);
crit bits; move order; boost deltas (7 per side); target status at fire time; status
applied/cured one-hots; actor/target/switch-to species ids; attempted-move and
attempted-switch ids + the switch-rejected bit; `phase_is_forced_switch`.

Two structural defects, one hidden strength. Defects: **lag-indexed weights** (turn t−3 is a
fixed weight range) and **entity-blindness** (actor/target are embedded ids the model must match
against live slots by content, per lag position, from scratch). Strength: the field list itself
is *good* — v2 through trapping-signals iterated it against real losses, and §5 shows almost
every field survives into the new form. This is a re-homing, not a redesign of what history says.

---

## 2. What history is FOR — and the last-turn-outcome question

In a POMDP, history has exactly three jobs. Everything below is admitted or rejected by them:

* **J1 — belief completion (inversion evidence).** Past observations that reveal hidden state:
  the move they clicked (moveset), the damage a hit dealt (spread/item, via the op's inversion),
  the `cant` reason (sleep-turn count), the faint cause.
* **J2 — tendency estimation (policy evidence).** Their past choices are samples of their
  policy: what they click into which of our mons, whom they switch in on what, Protect timing.
  This is exactly the input the α/β intent heads need and currently lack.
* **J3 — mechanical carryover.** Constraints the past places on the next turn: lock-in,
  recharge, Encore. Note: **J3 is essentially fully compiled already** (choice-lock state,
  volatile timers, the request itself) — the compiled tier owns it, and no new history mechanism
  should re-carry it.

**Do we need the outcome of the last turn?** The Markov answer is "no": V(s) does not care *why*
you are at s, and a crit that already happened predicts nothing forward. But we are not Markov —
we are belief-completing — and that changes the answer field by field. The admission rule:

> A transition fact earns a place iff it is **not derivable from current state** AND it serves
> J1 or J2 (J3 being compiled). Pure dice outcomes are admitted **only as deflators attached to
> the evidence they explain**.

Worked through the contentious fields:

* **Damage dealt, attributed to the move that dealt it** — YES, J1 core. Damage magnitude is the
  single strongest spread/item inversion signal and it is *gone from current state* the moment
  the HP bar is read (you see 43%; you cannot recover "Blissey's Ice Beam did 38% last turn").
* **The crit flag** — YES, but *only* as the deflator on that damage evidence. A 2× crit read as
  a clean hit doubles the inferred attack stat; without the flag, the damage number actively
  corrupts the spread belief. It carries no forward information of its own.
* **Miss/fail** — marginal J1 (accuracy-modifying items are fringe in gen3); kept as a cheap
  outcome enum mostly so that "no damage happened" is distinguishable from "no move happened".
* **What they clicked** — YES, doubly: J1 (moveset reveal — already compiled into `known`
  flags) and J2 (the tendency sample, NOT compiled anywhere). The J2 half is the reason the
  event survives after the reveal is compiled.
* **Effectiveness one-hot** — NO as evidence (derivable: both types and the chart are known once
  the move is revealed; the op computes it exactly). Kept only as a cheap label inside the event
  token because deriving it costs the model capacity that a 4-dim one-hot costs nothing.
  Honest note: this is a convenience admission, flagged for the usage audit.
* **Boost deltas** — current boosts are state (E2); the *delta event* is J2 only ("they set up
  on my switches"). Admitted as an event type, not as fields to sum.
* **HP levels per slot** — NO. Pure current-state duplication (the per-mon slots carry HP);
  today's frames carry them only because the frames couldn't reference the entities. Deleted.
* **Move order** — YES, J1: realized order is speed-spread evidence the V-edge inversion wants,
  and it is not in current state.

**The most recent transition is special, and gets a second, compiled delivery.** Their last
action autocorrelates hard with their next (lock-in, Protect alternation, momentum), and it is
α's realized previous sample. So beyond its event token, "their last action" is compiled as
token content on their active (H-A1 below) — the cheap, always-visible form — while the event
stream carries the same fact with full context for deeper queries. This duplication is
deliberate and mirrors the E2-injection precedent (deliver on the entity AND leave the richer
route intact).

---

## 3. The design — three tiers, one admission rule

### H-A — the compiled tier (sufficient statistics; extends what already works)

**H-A1: last-action fields on the active entities.** On each side's ACTIVE mon slot:
`[last_move_id (embedded), last_was_switch, last_outcome (3), last_was_crit]` for that side's
most recent action. ~7 dims/side, retrain-class obs addition, EpisodeTracker-fold, fuzz-gated.

**H-A2: the pair-history edge family `h`.** Per (their mon *i*, our mon *j*), a cell folded from
the whole battle's event log — the two questions this program keeps asking, delivered as the
pair facts they are:

```
h[i, j] = [ switch_ins_i_while_j_active,   # "whom do they bring in on j"
            attacks_i_on_j,                # "what do they do into j" (damaging clicks)
            status_clicks_i_on_j,          # ... (status clicks)
            shared_field_turns,            # exposure normalizer
            recency_of_last_pairing ]      # log-saturated, like since_seen
```

Counts log-saturated (`log(1+min(n,10))/log(11)`). Delivered through the shipped `EdgeBias`
mechanism at the mon×mon block (a 16th family) — a RATIO delivery is *correct* here, because
tendencies are relative ("Blissey more than Skarmory"), unlike damage magnitudes. The α/β heads
are the primary intended consumers. Zero-init map ⇒ byte-identical at init. CPU cost is a
6×6×5 counter fold per decision — obs-benchmark-gated.

H-A ships **without waiting for H-B/H-C** — it is small, answers the two critical queries
("what did they click into this mon", "whom did they switch into") directly, and rides existing
machinery end to end.

### H-B — event tokens (the sequential residue becomes queryable)

Replace the 7 turn-frames with a window of the last **N events** (working number: 48 ≈ 12–16
turns; sized by the B=1 benchmark and the usage audit, not guessed), mask-padded, one token per
decision-relevant event. Schema per token, all through per-type input projections onto d_model
(the E3/E4 pattern):

| Group | Content | Justification |
|---|---|---|
| type | event-type embedding: `move / switch_in / faint / status_applied / status_cured / boost / item_reveal / hazard / forced_switch_window / switch_rejected` | the vocabulary of §1's field list, factored |
| actor / target | species embedding + side bit (+ the H-C reference edge) | entity linking, J1+J2 |
| move | the `MoveLatentEncoder` latent (32) — the SAME space the live E3/E4 seats use | one move representation everywhere |
| outcome | `[hp_frac_delta_on_target, outcome(3), crit, eff(4), we_first]` | §2's admissions: damage+deflator, order |
| status | applied/cured one-hot where applicable | J1 |
| faint | cause multi-hot (8) where applicable | J1 |
| time | **relative recency embedding** — log-scaled turns-ago + same-turn phase tag | constraint 4; no lag-indexed weights anywhere |

Notes. (a) **The "turn" framing dissolves**: forced-switch windows, multi-KO turns and
Pursuit-on-the-switch — which the frame representation squeezes through `phase_is_forced_switch`
— are just events with the same recency value and a phase tag. (b) Events enter the trunk as
extra seats after E5 (the position-stable append convention); no bias families target them in
v1 — token content + H-C references only. (c) The window is events-not-turns on purpose: quiet
turns stop costing seats, violent turns stop being truncated.

### H-C — entity reference edges (the "linked" in entity-linked)

For each event token, an explicit **reference edge** (the shipped per-pair bias mechanism) to
the live token of its actor and of its target — legal because slot identity is stable within a
battle. This is what turns the two critical queries into single attention hops:

* *"What did they click into this Pokémon?"* — my mon j's token attends over event seats whose
  target-edge points at j.
* *"Whom did they switch into?"* — switch-in events carry an actor-edge to the arriving mon and
  a recency-adjacent context (the co-timed events link to my then-active mon).

Content linking (the species embedding inside the token) rides along as the soft form, which
also covers the one nuance reference edges miss: **events referencing fainted mons**. Fainted
seats are key-masked in the trunk, so a reference edge into them is inert — but the event token
itself still carries the species content, so "how their Blissey died" remains readable. Accepted
as-is in v1; flagged for the audit.

---

## 4. What gets DELETED (with its gate)

| Deleted | Gate |
|---|---|
| The 7×159 TurnDelta obs block + `embed_delta_slot` + the history positional embedding + the 7 history trunk seats | §5's nothing-lost mapping; the usage audit on H-B seats; non-inferiority ladder |
| Per-slot HP levels, HP delta sums inside the frames | pure state duplication (§2) — no replacement needed |
| Prev-turn action mask (11 obs dims) | audit-then-delete: its live content ("we were locked/disabled last turn") is compiled state (choice-lock, volatiles); zero the block on a trained checkpoint first — if flips ≈ 0, delete |
| `TURN_DELTA_EMBEDDED_IDS` manifest + the TD offset constants | die with the block |

This is the obs vector's largest single deletion (−1124 dims of 2669) and, with the earlier
re-homes, leaves the flat vector as per-mon slots + ctx + global/board — the Stage-3 endgame.

---

## 5. The mapping — every field of today's frames → its new home

| Today (per TurnDelta frame) | New home | Tier |
|---|---|---|
| our/opp move block `[id, power, sec, recoil, type]` | the event token's move LATENT (richer: full `MoveLatentEncoder` content) | H-B |
| our/opp `switched` bits + switch-to species | `switch_in` event (actor = arriving mon) + H-A2 counts | H-B, H-A2 |
| `failed_to_move` + `cant` one-hot (12) | `move` event with outcome=fail + a cant-reason field on it; sleep-turn evidence also already compiled (sleep-wake belief) | H-B (+ compiled) |
| HP delta sums / per-slot HP levels (6+6) | **deleted** — current state carries HP; per-hit attribution survives as `hp_frac_delta_on_target` on each move event | H-B / deleted |
| fainted bits + faint-cause multi-hot | `faint` event, cause multi-hot, target-linked | H-B |
| `opp_move_known` | already compiled (the `known` flags the reveal updates) | compiled |
| effectiveness one-hots | event outcome group (convenience admission — audit) | H-B |
| move-outcome one-hots (hit/miss/fail) | event outcome group | H-B |
| crit bits | event outcome group — the damage-evidence DEFLATOR (§2) | H-B |
| move order (2) | `we_first` on the event pair (speed-inversion evidence) | H-B |
| boost deltas (7×2) | `boost` event (type + actor); current boosts are E2 state | H-B |
| target status at fire time | derivable from co-timed events + state; **dropped**, audit flag | deleted |
| status applied/cured one-hots | `status_applied` / `status_cured` events, target-linked | H-B |
| actor/target/switch-to species ids | the event's actor/target refs — species content + H-C reference edges (the entity-blindness fix) | H-B/H-C |
| attempted-move id (pressed but prevented) | `move` event outcome=fail keeps the ATTEMPTED move's latent | H-B |
| attempted-switch id + `switch_rejected` bit | `switch_rejected` event (actor-edge to the mon we tried to bring) — the trapping signal survives verbatim | H-B |
| `phase_is_forced_switch` | the per-event phase tag (the turn framing dissolves) | H-B |
| the 7-slot lag structure itself | the recency embedding — time as content, not position | H-B |
| prev-turn action mask | compiled state (choice-lock/volatiles); audit-then-delete | deleted |
| *(nothing today)* | "their last action" on the active token | **H-A1 (new)** |
| *(nothing today)* | pair-history counts h[i,j] — the two critical queries | **H-A2 (new)** |

Nothing-lost check: every frame field lands in an event field, a compiled statistic, or a
justified deletion (state-duplication only). The two *new* rows are the capabilities today's
representation cannot express at any horizon.

---

## 6. Sequencing and gates

| Step | What | Class | Gates |
|---|---|---|---|
| 1 | **H-A1 + H-A2** (last-action fields, pair-history edge family) — ✅ **BUILT 2026-08-15 (v78 `gen3_pair_history_v1`)**; the fuzz caught a real fainted-active-resurrection bug in the resync before it shipped, and the leads-don't-count semantics were pinned by the same oracle | retrain-class obs/edge addition; can ride the next generation alongside other work | poke_env_gaps fuzz (encoded counters == protocol reconstruction); obs-build benchmark; zero-init byte-identity; per-family ablation joins the edge audit |
| 2 | **H-B + the deletion** (event tokens in, 7×159 out) — one generation, nothing else behavioral beside it (attribution discipline) | retrain-class, the Stage-3-scale change | event-fold fuzz; obs-roundtrip fuzz ported to the event window; B=1 CPU benchmark at the real N; usage audit on the event seats; non-inferiority ladder with pre-fixed margin |
| 3 | **H-C** reference edges | in-generation with 2 if cheap, else next | the audit decides whether references beat content-linking alone |
| 4 | Prev-mask deletion | with 2 | the zero-arm audit above |

Interplay to exploit: α/β gain their tendency inputs at step 1 — measure `alpha_acc` before and
after H-A2 lands; a move there is the cheapest possible validation that history-as-evidence is
reaching the consumer built for it.

## 7. Open questions / honest risks

1. **N (the event window).** 48 is a guess; size it by measuring how much J2 signal survives at
   the horizon (how far back do switch-in tendencies still predict?) — a probe on eval traces,
   no training needed.
2. **The convenience admissions** (effectiveness one-hot; fainted-mon reference inertness) are
   flagged for the usage audit rather than argued from principle.
3. **Compiled-vs-token double delivery** (H-A1 duplicates the newest event) is deliberate but
   unmeasured; the audit can zero either route.
4. **Risk: the frames' 7-turn horizon may be doing quiet regularization** — an event window of
   48 admits longer-range spurious correlations in self-play. The non-inferiority ladder is the
   backstop; if step 2 regresses, shrink N before abandoning the form.
5. **Cost honesty:** +N seats is the largest token-count change since Stage 1. The spike says
   sub-quadratic; it has not been measured at n≈84 total seats with 16 bias families live.
