# design — THE TIERED PIPELINE: resolve → reason → decide → deliver

**Owner-constrained, 2026-08-13. Goal: REDUCE architectural complexity while improving robustness.
Acceptance is NON-INFERIORITY, not gain** — "okay if it doesn't make it meaningfully better so long
as it doesn't make it meaningfully worse, as long as we're trending towards a more universal
approximation of how the game actually occurs."

**Two information sources only** (owner constraint): (1) the **Smogon prior**, injected directly;
(2) **weightings the model learns on top**. No set vocabulary, no pool co-occurrence table, no
third data artifact. This resolves the set-prior problem of `design_unified_belief.md` §1.2 —
we do not need a joint prior, because the couplings live in the LEARNED half.

---

## 1. The finding that makes this cheap

**Production already runs the tiered order.** With `--damage-op-prefuse` (production), the forward is:

```
pokemon_encoder → belief_slots → move_belief → spread/HP → damage_op → transformer
                  └──────────── T0 RESOLVE ──────────┘   └ T1 ┘
```

The complexity is not that the tiers are wrong. It is that **the untiered order also exists**, in
parallel, selected by flag:

| duplication | why it exists | status |
|---|---|---|
| `move_belief` runs PRE **or** POST transformer | `--move-belief-prefuse` | both paths live |
| `damage_op` runs PRE **or** POST transformer | `--damage-op-prefuse` | both paths live |
| `belief_slots` (pre) vs `belief_head` (post) | the species belief is **split across the trunk** | both live, different objects |
| the refine loop | `--damage-refine-rounds` | **0 in production; 3 dependent flags UNREACHABLE** |
| `damage_reattend` | re-attends physics onto tokens AFTER the pools | a compensation for post-ordering |
| `move_belief_single_compute` | make the refine callback reuse the posterior | **INERT** — no callback is built |

So the work is mostly **deletion**: collapse to the one order that production already uses, and
close the one loop that is open (α has no consumer).

**Scale of the surface being reduced: 183 CLI flags, 52 version-checked arch toggles.**

---

## 2. The four tiers

### T0 — RESOLVE: what is on the board?

One belief block, entirely pre-transformer, producing a **resolved opponent state**. Every leg has
the same two-source shape, which is the pattern `move_prior_fusion` already uses:

```
posterior_logit = smogon_prior_logit  ⊕  learned_delta(evidence)
```

* **prior** — injected, frozen, from `data/pokemon/`. Legality is a HARD constraint (v65:
  illegal → 1e-6, legal-unobserved → floor, legal-with-usage → its true rate).
* **learned delta** — the model's own weighting, conditioned on everything revealed so far.

**This is where the couplings live, and why no set prior is needed.** We do not inject
`P(FocusPunch | Substitute)`. The delta sees the revealed moves and learns it — the same way the
HP-type head already learns a per-species type posterior on top of a Smogon prior. Marginals in,
couplings learned. That is strictly more expressive than an injected joint and needs no new data.

Order within T0 is the evidence order the game has: **species first, then everything conditioned on
it** — moves, spread/nature, item, HP type. Species must be first because every other prior is
`P(x | species)`.

### T1 — REASON: what follows from that board?

The `DamageOperator` over the resolved state → the pair tensor `pair[a, k, :]`. Then the transformer:
attention over entity tokens that **already carry** belief and physics.

This is unchanged from production. The only edit is that it becomes unconditional — it is no longer
one of two possible placements.

### T2 — DECIDE: what will they do, and what does that make my moves worth?

Two steps, in this order:

1. **α / β** — read the post-attention threat seats + pooled board (this is *already* what v67 does:
   `_seat_out[:, 4:4+K]`, whose header carries the belief weight and the physics). α is a
   distribution over their K believed move seats + SWITCH; β over their bench.
2. **THE RECALIBRATION** — α becomes the reduction weight over their axis:

   ```
   my_move_value[a] = Σ_k α_k · f(pair[a, k, :])
   ```

   This is the **one open loop in the whole system**. `damage_op.py::_chan_max` is already written as
   THE single arity-2→arity-1 reduction site, with `conditional(λ)` and `learned_attention(k)` named
   as settings of one knob. Today it runs `hard_max`, which needs no distribution — which is exactly
   why α currently has no consumer. **Closing this is one argument at one call site, not a new path.**

### T3 — DELIVER: one contract, two pools

The reduction emits per-`(entity, action)` rows. **Policy** pools over entities per action (the
pointer head — unchanged). **Critic** pools over the board (attention — permutation-invariant).

Same object, two pools. This replaces four delivery routes, two of which exist only because the
other two could not reach the head that needed them (pointer cells are policy-only; the seed readout
and `--value-threat-inject` are critic-only).

---

## 3. What gets DELETED

| # | delete | why it is safe |
|---|---|---|
| 1 | the POST-transformer `move_belief` + `damage_op` call sites | production never takes them |
| 2 | `--move-belief-prefuse`, `--damage-op-prefuse`, `--move-belief-single-compute` | become unconditional; the last is already INERT |
| 3 | `--damage-refine-rounds` + the refine callback | 0 in production; it is what makes 3 flags UNREACHABLE |
| 4 | `--threat-refine-outgoing`, `--threat-unrevealed-outgoing`, `--threat-status-refine` | **UNREACHABLE today** — they raise at build |
| 5 | `--damage-reattend` | a compensation for post-ordering that T0/T1 removes the need for |

**Six-plus flags and one whole duplicated forward path.** Nothing in production loses a capability;
every deletion is either the unused half of a fork or already unreachable.

---

## 4. What gets ADDED

Deliberately small — three things:

1. **α as the reduction's `how=`** at the single `_chan_max` call site. This is the only behavioural
   change in the whole design.
2. **A tier-ordering CONTRACT with a test.** Today the tiering is satisfied *by construction*;
   nothing prevents a future head from reading α at T0 or computing intent off raw tokens. Make it
   an asserted invariant, the way leak-safety is: *a T_n module may read only T_{<n} outputs.*
3. **One masking convention.** Each belief leg currently masks differently and each mask rate means
   something different. One question, asked identically at every leg: how much mass is this
   posterior spread over?

---

## 5. Migration — smallest safe steps, each independently shippable

| # | step | risk | gate |
|---|---|---|---|
| 1 | Delete the 3 UNREACHABLE flags (#4) | **zero** — they raise today | suite |
| 2 | Delete the refine loop + `single_compute` (#2,#3) | zero in production (0 rounds) | suite + byte-identical forward |
| 3 | Make prefuse unconditional; delete the POST call sites (#1) | zero in production; **breaks non-prefuse configs by design** | forward bit-identity vs production config |
| 4 | Delete `--damage-reattend` (#5) | off in production | suite |
| 5 | Land the tier-ordering contract + test | zero | new test fails on a violation |
| 6 | **α → the reduction** (`how=` at `_chan_max`) | **the real change** | A/B at the one call site; retrain-class |
| 7 | Unify delivery (T3) | largest | non-inferiority on the ladder |

**Steps 1–5 are pure deletion and are byte-identical on the production config.** They can ship
before gen-9 finishes and be validated by the existing forward-identity gates. Step 6 is the first
behavioural change and the first that needs a generation.

---

## 6. Acceptance — non-inferiority, stated in advance

Per the owner's constraint, this is an ARCHITECTURE project, not a strength project. Gen-8 is the
cautionary precedent: the belief stack learned well (`species_acc_above_chance` 0.67, `move_recall`
0.19→0.58, `spread_largest_bias` −26→−13) and anchored ELO **fell** (tail-4: gen-4 2057.8 / gen-5
2038.4 / gen-8 2016.5, matched steps).

* **Steps 1–5:** forward **bit-identity** on the production config. Not "no regression" — identity.
  A deletion that changes a number is a bug in the deletion.
* **Step 6:** anchored ladder **non-inferiority** vs the preceding generation, margin fixed BEFORE
  the run (suggest: within −15, CI excluding −40 — the `design_conditional_opponent_cells.md` §5
  convention), plus α's own gate (`alpha_acc_move` vs its `argmax(w)` baseline, both logged in gen-9).
* **Step 7:** the same ladder bar, plus the delivery claim measured separately from the behaviour
  claim (an oracle move-belief once flipped 19.3% of actions while moving switch mass by +0.019 —
  delivery ≠ behaviour).

**The trend that justifies it even at parity:** one resolve path, one reason path, one decision
path, one delivery contract, one masking convention — and a forward pass whose order matches the
order the game actually resolves in.

---

## 7. Honest risks

* **Step 3 is a real breaking change** for any non-prefuse config. Production is prefuse, so this is
  a deletion of an unused branch — but any archived experiment that used the post path can no longer
  be reproduced from HEAD. Acceptable under the project's stated no-back-compat stance; worth saying
  once, out loud.
* **Step 6 could be a null.** `design_pair_reduction.md`'s G1 measured no rung beating `hard_max`
  beyond seed spread — but it tested the reduction *without a distribution to reduce by*, on
  damage-only cells. This design supplies the distribution. That makes the null less likely, not
  impossible.
* **T3 is the one step that could genuinely regress**, because it changes what both heads read. It
  is deliberately last, and it is the only step gated on strength rather than identity.
* **No long-horizon reasoning.** The tiers are one-ply. "They are saving Explosion for my Celebi"
  remains out of scope at every tier.

---

## See also

* `design_unified_belief.md` — the reframe this operationalises (its §1.2 set-prior problem is
  resolved here by the two-source constraint)
* `design_opponent_intent.md` — where α/β come from
* `design_pair_reduction.md` — Contract W and the one reduction site
* `design_conditional_execution.md` — the twelve readouts T2 makes computable
