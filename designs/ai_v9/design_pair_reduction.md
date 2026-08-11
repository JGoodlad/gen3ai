# design — the PAIR REDUCTION: replacing the hard max with a typed, learned, swappable operator

> **Window this occupies — UPDATED 2026-08-09, and the update is not cosmetic.** This document was
> written to occupy the gap *after* the op head-concat is removed and *before* the next generation
> launches. **Both events have now happened**: the concat died in `6aac795`
> (`gen3_no_concat_v1`, v61) and gen-5 `ai_v9_06_gen5_no_concat_0809` launched the same day.
> So the clean pre-generation window this doc assumed **does not exist** — see §1, which is
> rewritten rather than patched. The design is unaffected; its *scheduling* is, and §8 now has a
> gen-5-is-running build order.
>
> **What this cost, honestly: nothing structural, and the deletion evidence made the case stronger
> rather than weaker.** §1(a) predicted the reduction was measured worst exactly where it was about
> to matter most, and the gen-4 stratified split then measured `FULL_CONCAT` net policy dependence
> at **+0.00%** — the route the reduction was competing against was already carrying no net policy
> signal at deletion time.
>
> **Relationship to existing docs.** This is the deep specification of the one line
> `design_op_tensors.md` §3.2 sketches as `REDUCE(pair_in, over=MOVE_AXIS, how=…)`. It does not
> replace that doc; it makes `how` buildable and pre-registers how to choose it.
> `design_conditional_opponent_cells.md` OA1 appears here as one rung of the ladder (R4), not as a
> separate feature.
>
> **ADOPTED 2026-08-10 (owner).** The §8 build order is the plan of record; **§8.1** records the
> adopted schedule, the step-0 downscope rule, the **seed-VICReg prerequisite** (the §3.2 critic-route
> collapse caveat is now a gen-5 *measurement*, not a hypothetical), and G7's post-gen-5 slot.

---

## 0. Goal

**Stop reducing their move axis with a hand-chosen risk functional applied independently per
channel and per defender, and make the reduction a declared message-passing operator whose
signature enforces coherence, whose expressiveness is a flag, and whose choice is decided by cheap
gates before a generation commits to it.**

Two words carry the design: **coherent** (one reduction speaks for one opponent) and **swappable**
(choosing wrong costs a flag, not a generation).

The capability this unlocks, stated in the terms that motivate it: **hedging is not a depth
phenomenon.** Taking the middle ground under uncertainty is a property of *the reduction* — of
declining to collapse the branch distribution before choosing — not of search. §4.1 gives the
one-ply construction; §7 G7 gives the experiment that decides whether it pays.

---

## 1. Why this matters now — restated after the concat deletion

The original version of this section argued for a window between the concat's removal and the next
generation's launch. **That window closed on the day this doc was written** (`6aac795` deleted the
concat; gen-5 launched hours later). The three facts below are the same three facts, restated
against the tree as it actually is.

**(a) The reduction is now the ONLY path from pair physics to the policy head — and the measurement
that used to hide that is gone.** While the concat existed, any decision needing the full
`(move × defender)` grid could take the flat route, so the reduction read low and the number could
not be trusted as a price. It is worth recording exactly how low, on **clean** sampling:

| block (gen-4 @25M, stratified, n=6000) | width | shuffle flips | zero flips | net |
|---|---|---|---|---|
| `FULL_CONCAT` | 660 | 22.70% | 22.70% | **+0.00%** |
| `imx_CELLS` (the reduced pair cells) | 216 | **6.53%** | 6.05% | −0.48% |
| `imx_HEADERS` | 306 | 18.30% | 21.12% | +2.82% |

*(`gen4_op_block_split_25M.json`. **Use these, not the gen-3 figures this section originally
quoted** — `gen3_op_block_split_40M.json` and `gen3_op_block_dependence_40M_6k.json` were produced
by the pre-`754ca78` sampler, which drew its whole sample from one mid-run trace dir because step
dirs sort lexically. The committed old numbers keep the defect; only new measurements are clean.)*

The prediction in the original §1(a) — *the reduction is measured worst exactly where it is about
to matter most* — is now **confirmed rather than asserted**: `FULL_CONCAT`'s net policy dependence
was **+0.00%** at deletion time, so the alternative route the cells were losing to was, by then,
carrying no net policy signal at all. The cells' 6.53% was a measurement of a suppressed quantity.

**What replaces the old argument:** the price of the reduction can now be measured honestly for the
first time, because there is no longer a competing path. **Re-run the split audit on gen-5** and the
`imx_CELLS` figure means what it says. That is a new, cheap, pre-registered observation this design
did not previously have access to — and it should be taken before any rung is built (§8 step 0).

**(b) The reduction is load-bearing and expensive to revise — and gen-5 is already training against
it.** It feeds the pointer switch cell (`_PTR_SWITCH_CELL_IN` = 15 = the per-defender row 12 + the
CB tail 3) and `prefuse_proj`'s token injection. A generation fits its whole policy to that summary,
so swapping it mid-run is not an A/B. **Gen-5 is committed to R0 `hard_max` for its lifetime**; every
rung below is therefore gen-6 work, and the honest consequence is stated in §8.

**(c) Almost all of the work here is provable or offline — and that is what rescues the schedule.**
Steps that are byte-identical refactors, plus a probe on a frozen checkpoint, plus one
constructed-scenario test: **none of them need a training run, so none of them are blocked by gen-5
occupying the GPU.** The original framing ("do it before the generation or pay a generation") turns
out to overstate the loss. What is actually lost by having missed the window is one thing only:
**gen-5 will not itself carry a better reducer.** Everything else — the contract, the ladder, the
offline kills, G7's harness — proceeds in parallel with gen-5 and lands ready for gen-6.

**And one thing was gained.** §7 G5's acceptance measurement *could not be taken* while the concat
existed; now it can. The sequencing is therefore better than the one this doc originally proposed:
measure the reduction's true price on gen-5 (§8 step 0) → build the contract and kill rungs offline
during gen-5 → commit the winning rung at the gen-6 boundary, with a real number in hand instead of
an argument.

---

## 2. The defect, from the live code

`damage_op.py:2777-2780` produces the eight damage numbers of the per-mon block by **eight
independent maxima**:

```
phys_high[j] = max_c ( w_c · high_frac[j,c] · phys_mask[c] )     # and 7 more, each its own amax
```

with a **ninth and separate** maximum for status (`:2565`,
`p_major = (w_b · land).amax(dim=-1)`), and `acc` / `provenance` gathered at the *damage* argmax
(`:2796-2805`).

Three separable defects follow, and they need different fixes:

| # | defect | fix class |
|---|---|---|
| **D1** | **The objective is `w·d`** — a belief times an HP fraction. Neither `max_c d_c` (worst case) nor `Σ_c w_c d_c` (expectation). A 0.40-belief 90% roll (0.36) outranks a 0.95-belief 37% roll (0.35). A hand-chosen risk functional corresponding to no decision-theoretic quantity. | choose the functional deliberately — §4 ladder |
| **D2** | **Incoherence across channels.** phys / spec / low / high / crit / pko / status each maximize separately, so up to nine *different* moves describe one defender. The block's "profile" of mon *j* is an opponent who clicks several moves at once. | **a contract**, not a knob — §3.1 |
| **D3** | **Incoherence across defenders.** The max is independent per *j*, so the winning move differs down the column. Same fiction, other axis. The defect `design_conditional_opponent_cells.md` §1.1 names. | **the same contract** — `α` is defender-independent (§3.1) |

**D2 and D3 fall to ONE restriction** — `α` computed from the board alone, shared across every
channel and every defender (§3.1). That is the highest-leverage line in this document.
Note that `_chan_acc`
(`:2796`) already fixes D2 *within* a channel — accuracy is gathered coherently at the channel's
dominant-damage move so `{pko, acc}` describe one threat. The idea is present in the code; it
simply stops at the channel boundary.

**The consequence that makes this strategic rather than cosmetic.** Status moves have zero damage.
The argmax that selects the "dominant" move is over damage, so a Will-O-Wisp or a Toxic can never
win it; status gets its own separate max on a different objective. Nothing in the block can
therefore represent *"the single move they will click is a status move, so the correct switch is
the Natural Cure mon rather than the physically bulky one."* The two numbers that decision needs
were maximized independently and describe different worlds. **This is a reduction failure, not a
belief failure** — a perfect move belief does not fix it.

### 2.1 CORRECTION (2026-08-10) — on the PATH THAT DECIDES, status is not mis-reduced, it is ABSENT

Everything above describes the **flat/trunk** block. Traced through to the **pointer switch logit**
— the sink that actually chooses a switch — the defect is **strictly worse than "a ninth max"**, and
the doc understated its own case.

Dissect the switch cell the pointer head receives (`pointer_cells`, `_PTR_SWITCH_CELL_IN` = 15,
verified against production where `damage_matrices_outgoing_all = False` so there is no OAX tail):

| | contents |
|---|---|
| physical | `low_roll, high_roll, crit, pko, accuracy` |
| special | `low_roll, high_roll, crit, pko, accuracy` |
| plus | `p_outspeed`, `provenance` |
| Choice-Band tail | `phys_high_cb`, `pko_cb`, `p_cb` |

**Ten damage numbers, one speed number, two belief-mass numbers, and NO status coordinate in any
currency.** `p_major` / `p_immob` are never sliced into the switch cell at all.

Where incoming status actually goes in production (gen-5, v61):

| route | status? | what it carries |
|---|---|---|
| pointer switch cell *j* | **absent** | — |
| trunk residual `status_in_proj` | **OFF** — gated by `threat_status_refine`, which is `False` | (would be a token residual) |
| **`s3` edge family** | **the only live route** (`edge_bias_families` includes `s3`) | an attention **bias** — a softmax-normalised RATIO, not an absolute |

So the honest statement of the defect, on the path that decides:

> The decision *"they will click Will-O-Wisp, so bring the Natural Cure mon"* is made at the switch
> logit. The switch logit's per-action cell contains **no status information at all**. Status reaches
> the policy only after conversion into a **ratio**, on a different route.

**This is a CURRENCY failure sitting one level below the reduction failure.** You cannot trade
"35% of my HP" against "80% chance of burn" if the two never appear in the same vector in the same
units — no reducer, however expressive, fixes that. It also sharpens the ordering in §4.1 and §8:
**what the message CARRIES binds before how it is aggregated**, and a status coordinate in a damage
currency is a prerequisite for the hedging rungs, not an optional enrichment.

**Two further coordinates are missing for the same reason** — named here so §3.2's `φ` has a target
list, and both pass the admission test in §9a (name two actions whose ordering they flip):

| coordinate | currency | flips |
|---|---|---|
| **neutralization** | fraction of this mon's future contribution destroyed *without* a KO | Swampert → Celebi against Gengar's burn/Thunderbolt branch. On damage alone Swampert reads **0.0 in both branches** (immune to one; burn deals none) and wins forever |
| **tempo** | turns of my clock spent undoing it | Milotic **Refresh**es the burn away, so *neutralization* correctly reads ≈0 — but it costs a **turn**. Nothing in ten damage numbers plus speed encodes that, so "absorbs it" and "absorbs it and falls a turn behind a setup sweeper" are the same state |

Milotic is the sharper example precisely because it is **ours** — the receiver `j` indexes our six
mons, so its ability, moveset and stats are **fully observed**. All the uncertainty in `φ` lives on
the *sender* axis (their believed moves), and the belief head has already run by then. There is no
marginalization question on the receiver side at all.

*(Deliberately out of scope: **physics mutation.** Marvel Scale means burning Milotic multiplies its
Def by 1.5 and moves **every subsequent number in the matrix**. That is a statement about the
successor state, not an outcome coordinate. A one-ply reduction can learn "burn into Milotic is
fine"; it cannot represent "and here is the new matrix." Naming the limit beats pretending a richer
`φ` closes it.)*

**The model answer is already in the code, one field over.** The Choice-Band tail delivers
`phys_high_cb` (the magnitude **conditional** on them being Banded) beside `p_cb` (**how likely
that is**) — both present, **un-collapsed**, left for the reader to combine. That is Contract W's
spirit already shipped, in exactly one place. Status should look like that.

*(Verified 2026-08-10 against `damage_op.py::pointer_cells` / `_DMG_PER_MON`,
`features_extractor.py:2707` and `:3598`, and `designs/production_config.json`.)*

**Read this against the G1 null — the two fit together, and the fit is the point.** G1 FINAL
(n=299, 5 seeds) found no rung beating R0 beyond seed spread, **and the 2800-dim SKYLINE over the
un-collapsed pair grid showed no linear headroom either** (R0 0.403±0.034 · SKYLINE 0.413±0.037).
A skyline that cannot beat the collapsed summary is not a statement about aggregation at all — it
says **the quantity the decision needs was never in the grid.** That is exactly what a currency
failure predicts, and §2.1 names the missing quantities independently of the bake-off.

**Stated with the caveat it deserves:** this is a *post-hoc* reading, offered because it is the most
economical one, not because G1 was designed to test it. The alternative readings stay live — the
target is the beam (our own critic's preference, not ground truth, §10.3), and a linear probe cannot
rule out non-linear headroom. **What it does justify is a re-ordering, not a new rung:** the next
experiment on this line should add a coordinate (§9a) and re-run G1, rather than add expressiveness
to a reducer over a vector that G1 has now shown to be linearly exhausted.

---

## 3. Contract vs knob — the structural split

The core move of this design: **separate what must be true of any reduction (a contract, enforced
by the type signature) from how expressive it is (a knob, chosen by measurement).** Conflating
them is why "which reducer?" has felt like an open research question rather than a flag.

### 3.1 The coherence contract — a reducer emits WEIGHTS, not numbers

> **Contract W.** A weighted reducer emits `α ∈ Δ^K` — **one distribution over the move axis,
> computed from the board and their moves ONLY** — and the reduced cell is
> `Σ_k α_k · pair_in[k, j, :]`, applied with the same `α` to **every channel and every defender**.

**`α` must NOT be a function of the defender, and that restriction is the load-bearing part.**
The justification is game-theoretic, not aesthetic: gen3 is simultaneous-move with no team
preview, so **they choose without seeing which mon you bring.** Their intention cannot legitimately
depend on your *j*. What legitimately depends on *j* is how much their move *matters* to you —
and that belongs in the message (§3.2), where a type immunity is already a zero.

Allowing `α_j` would let Skarmory's row assume "they click Rock Slide" while Blissey's assumes
"they click Thunderbolt" — the decorrelated-opponent defect **D3**, in soft form rather than hard.
One restriction therefore kills **D2 and D3 together**.

`α` is computed at the **ACTIVE** defender ("what will they click into *X*"), which is the one
distinguished, content-addressed slot — so no positional axis is introduced anywhere.

**`α ≠ w`.** The move-belief posterior `w` is a *presence* belief (does this move exist in their
set). `α` is a *usage* belief (will they click it this turn, into this active). Today the op feeds
`w` straight in as though they were the same quantity; that substitution is the substantive
modelling error underneath the whole block.

Everything follows from that signature:

- **D2 dies structurally.** One α per defender ⇒ all six channels describe the same believed
  move mixture. Incoherence across channels is no longer *tested against*; it is unrepresentable.
  (Same class of move as the pointer-native head and the arity typing: an alignment defect becomes
  a type error.)
- **Magnitudes survive.** `Σα = 1`, `α ≥ 0` ⇒ the output is a **convex combination of HP
  fractions, which is an HP fraction.** Units, range and scale are preserved
  ([[marginalization_and_uncertainty]] § convex combinations), so the result is still legal input
  to a pointer switch cell. **Range** is the operative word: the pointer head reads the cell through
  a `tanh`, and a convex combination cannot leave `[0,1]` — so it lands in the responsive region
  rather than the saturated tails. An unbounded latent carries no such guarantee.
- **Today's behaviour is inside the contract**, so migration can start byte-identical:
  `α = onehot(argmax_k w_k · high_frac[k,j])`.
- **`acc` and `provenance` stop being special.** They are currently gathered at an argmax, which
  has no meaning under a soft α. Fold accuracy into the cell (`pair_in` already can) and let the
  same α reduce it; `provenance` becomes `Σ_k α_{jk} · w_k`, the belief mass the summary rests on
  — which is what it was always trying to say.

### 3.2 The second family — LATENT reducers, and why both exist

Contract W buys coherence and units, and pays expressiveness: the output is **linear in the
cells** given α, so it cannot express `max`, `count above threshold`, or `second-worst` (the
rank-*h* limit, [[entity_tokens_biases_pointers]] §6.9).

> **Contract L.** A latent reducer emits
> `ρ( POOL_k φ(pair_in[k,j,:], opp_move[k], our_mon[j]) )` — shared MLP `φ`,
> permutation-invariant `POOL` (sum and/or max), shared MLP `ρ`. Output is a **latent**, not an
> HP fraction.

**This is message passing, and naming it that is clarifying.** In Gilmer et al.'s formulation
(*Neural Message Passing*, 2017) the three roles are message → aggregate → update; ours are `φ`,
`POOL`, `ρ`, with sender = one of their believed moves, receiver = one of my mons, edge data =
`pair_in[k,j,:]`. **Today's code is the fully degenerate instance**: `φ` = identity, `α` = one-hot,
`POOL` = max, `ρ` = identity — every knob at its weakest available setting, instantiated nine
times.

**`φ` takes the RECEIVER.** This is not decoration: it is what makes the Natural Cure / immunity
discrimination *learnable* rather than hand-coded. The same Will-O-Wisp must produce a different
message to Swampert (a physical attacker, permanently crippled) than to Celebi (Natural Cure,
barely inconvenienced). We provide the facts — the ability sits in the mon's features, the
status-landing probability sits in the cell — and the interaction is learned. That is the
provide-vs-learn line drawn in the right place.

**⚠️ Scope decision, to be made deliberately.** What `our_mon[j]` *contains* depends on where the
op sits in the phase chain. **Pre-attention**, `φ` sees raw per-mon features; **post-attention**,
it sees board-aware refined tokens. That bounds what a message can express — notably, set-level
facts like "am I the last answer to their sweeper" are **not in scope for `φ` at all** under
either placement, and need the separate treatment in §12. Record the choice; do not inherit it.

Contract L is universal over set functions (Deep Sets, Zaheer et al. 2017) and, because `φ` runs
**before** the pool, it can carry "worst damage" and "worst status risk" in different latent
coordinates simultaneously — which is precisely what D2's separate maxima were failing to do.
That is PointNet's actual lesson: **max over *learned features*, never over the raw physical
quantity.**

**Neither family dominates. Route by what the consumer can physically accept:**

| consumer | reads | why |
|---|---|---|
| pointer switch cell *j* | **W** (physical units) — **but see the correction below; W ⊕ L is permitted, W-replaced-by-L is not** | a **shared, thin** reader with a bounded nonlinearity: `Linear([token ‖ cell]) + ctx → tanh → Linear → scalar`, the SAME weights for all six slots |
| `prefuse_proj` token injection | **W + L** | token content can carry a latent; more is strictly better here |
| `MultiSeedValueReadout` — the critic's k=4 seed queries over the op's per-our-mon rows | **W + L** | same |
| d3 / s3 edge biases | neither — they read `pair_in` directly | an edge carries a ratio; it never needed the reduction |

This routing table is the same "what can each channel physically carry" rule as
`design_conditional_opponent_cells.md` §0.1, applied one level down.

> **CORRECTION (2026-08-10) — an earlier draft of this row said the switch cell is "read affinely
> at a logit." That is FALSE about the code, and the overclaim mattered.** `PointerHead` computes
> per switch slot `Linear([team_token ‖ switch_cell]) + ctx → tanh → Linear → scalar`
> (`features_extractor.py:2069-2076`) — a one-hidden-layer MLP with the policy context added as a
> bias and a zero-init scorer. **An MLP over the cell already exists.** So "why not just put an MLP
> over a latent" is not a hypothetical to argue against; it is the status quo, and the cell is
> physical anyway. Four reasons survive, and they are narrower than "affine":
>
> 1. **The reader is SHARED across all six slots** — one `switch_proj`, one `switch_score`. Weight
>    sharing is only *legitimate* if the input means the same thing in every slot. An HP fraction
>    does, by construction, in every slot and against every opponent. **Units are what license the
>    sharing** — this is the strongest of the four.
> 2. **The reader is THIN and the nonlinearity is BOUNDED.** One shared hidden layer is little
>    machinery for un-scrambling an arbitrary latent, and `tanh` saturates: a quantity in `[0,1]`
>    sits in the responsive region, an unbounded latent can reach the flat tails where the gradient
>    dies.
> 3. **Auditability — unaffected by the reader, and the biggest loss.** A cell carrying an HP
>    fraction is checkable against the sim; that is what the physics oracle (22/22) and the
>    constructed-scenario probes do. Replace it with a latent and **we delete our own oracle.**
> 4. **Extrapolation.** A monotone calibrated input means learning a slope, not a manifold — and
>    slopes extend past the observed range. Weakened by the `tanh`, not eliminated.
>
> **And regularization is not a substitute, for the same reason it is not a substitute for grounded
> seed queries (§3.2 note): regularizers REMOVE degenerate solutions; they do not CREATE meaning.**
> No weight-decay / spectral-norm / Lipschitz penalty makes coordinate 3 of a latent mean "fraction
> of max HP." Calibration, monotonicity and cross-slot comparability are semantic properties, and
> semantics come from supervision or construction. Supervise a latent to predict real damage and you
> have rebuilt Contract W with extra steps and a worse audit trail.
>
> **So this row is NOT a prohibition.** `W ⊕ L` at the same sink is fine and is probably the right
> end state — the physical cell as a calibrated, auditable backbone, plus a latent for what units
> cannot express (the neutralization and tempo coordinates §2.1 shows are missing). The rule to keep
> is narrower: **do not REPLACE the physical cell with a latent.** Adding one beside it is fine.
>
> Independent empirical support for resisting "just make the reader bigger": capacity has returned
> null in this project repeatedly — ledger **P3**, both LUT conditioning arms,
> `project_arch_compute_decision`. What has moved this model is changing *what gets delivered*.

> **The critic row is no longer hypothetical (2026-08-09, `6aac795`).** The concat deletion shipped
> `MultiSeedValueReadout` — **k=4 × 64 learned seed queries cross-attending the op's per-our-mon
> rows, vf-only** — as the critic route that replaced the flat window. Two consequences for this
> design, and they pull in opposite directions:
>
> **It is a consumer that can accept anything.** Attention over rows reads token content, not an
> affine cell, so Contract **L** output has a real home the day it exists. The reduction's richest
> form is not blocked on a new sink.
>
> **But it is also, itself, a learned reduction over a set — with the failure mode.** k seeds with
> no grounding can collapse to one query and the loss will not say so. That risk was pre-registered
> rather than discovered: v61 shipped a TensorBoard contract logged every `train()` (query/output
> cosine, uncentered effective rank, a VICReg variance target, and a pre-registered trigger) — the
> `z_arch` collapse lesson mechanized. **Read those diagnostics before designing anything that feeds
> this sink**; if the seeds have collapsed, a richer message is being delivered to an effectively
> rank-1 reader and any bake-off against it is measuring the wrong thing.
>
> *Naming, current as of v62: the TB family is **`value_seeds/*`** (renamed from `seeds/*`), and the
> VICReg term is **no longer diagnostic-only** — `--value-seed-vicreg-coef` adds a variance+covariance
> floor to the loss (`instrumented_ppo.py:1590-1596`). Under v61 it was `.detach()`-and-log only; the
> trigger fired on gen-5 and v62 wired it. **Consequence for honest monitoring:** `value_seeds/out_var`
> is now a trained quantity and has stopped being a diagnostic — `query_cos` and `out_effective_rank`
> stay outside the loss and remain the honest reads.*
>
> **CONFIRMED on gen-5 (2026-08-10): the seeds ARE collapsed.** `seeds/out_cos` = 1.000 and
> `seeds/out_effective_rank` = 1.0 at every measurement from 196k through 15.7M steps
> (`seeds/out_var` ≈ 5e-6; `seeds/query_cos` only 0.33 — distinct queries, identical outputs: the
> attention distributions are indistinguishable). The pre-registered trigger (eff-rank < k/2
> sustained past ~2M) has **FIRED**, and the VICReg variance+**covariance** wiring
> (`--seed-vicreg-coef`, resume-immutable, off for gen-5) is being built 2026-08-10 for enablement
> at the gen-6 launch. Consequence for this design: **until the regularizer is trained in, the
> critic route is effectively rank-1** — a G1/bake-off reading against the `MultiSeedValueReadout`
> sink is not interpretable before gen-6.
>
> **The design preference this reinforces:** where a *grounded* query exists, prefer it to a learned
> seed. R4/OA1 gets its queries from the action space — one per candidate action, each tied to a
> real entity — so it cannot collapse and needs no auxiliary loss to stay honest. Learned seeds are
> the right tool on the critic side precisely because there is no action to key on there.

### 3.3 What stays fixed forever, and what is a flag

| | |
|---|---|
| **Contract (never a flag)** | `pair_in` has exactly one home · a reduction is *declared*, never buried in a forward · **`α` is computed from the board and their moves ALONE — one distribution, shared across every channel and every defender** · `φ` takes the receiver · the routing table §3.2 |
| **Knob (chosen by measurement)** | which rung(s) of the §4 ladder · how many aggregators · pool = sum or max or both · whether `φ` emits second-order terms · latent width |

---

## 4. The ladder

Every rung is an instance of Contract W (except R2L, which is Contract L). Each is a `how` value
at one call site.

| rung | α (or pool) | expressiveness | cost | notes |
|---|---|---|---|---|
| **R0** `hard_max` | `onehot(argmax_k w_k·high)` | today | 0 | the byte-identity anchor. **Keep it shipped**; the regret is in *replacing*, not in *adding beside* |
| **R1** `belief_mean` | `α_k = w_k / Σw` | marginal expectation | ~0 | one line, no params. **Test this first** — it is the honest marginal D1 says we never computed |
| **R2W** `learned` | `α = softmax_k g(pair_in[k,j], opp_move[k])` | any query-independent soft selection, incl. a sharpened max | one small MLP | `g` shared over (k,j) ⇒ equivariant both axes |
| **R2L** `deepsets` | `ρ(Σ_k φ(·))` and/or `ρ(max_k φ(·))` | **universal** set function | one `φ`, one `ρ` | latent width ≥ 6 for exact universality (Wagstaff et al. 2019 — set size is 6, so 16 is free) |
| **R3** `multi` | emit **several** α's / pools and concatenate | strictly ⊇ any single rung | linear in count | see §5 |
| **R4** `conditional` | α depends on the **candidate action** — "which of their moves matters *given I am considering Blissey*" | a readout, not a statistic | per-action | **this is OA1.** The action space supplies the output slots, so no collapse at all |

**R4 is the qualitative jump, not R2.** R0–R3 are all *statistics*: one summary per defender,
computed before anyone asks a question. R4 makes the reduction depend on the query, which is the
difference between "how scary is this mon on average" and "if I bring this mon in, what actually
happens." Pivot decisions live in R4; R0–R3 decide how much the trunk knows before it gets there.

Pleasant coincidence for switches: **the action IS the receiver.** `m_kj` is already indexed by
the thing switch logit *j* cares about, so per-action conditioning for switches costs essentially
nothing beyond receiver-conditioned messages. The expensive version — conditioning **`φ` itself**
on the action, so message *content* changes per action rather than only its weighting — is
strictly more expressive at `36 × n_actions` message evaluations. **Named here as a rung we are
deliberately NOT taking**, with its price attached.

### 4.1 The rung that buys HEDGING — second moments

High-level play is largely about **taking the middle ground unless you are convinced**. That
capability has a precise architectural requirement, and it is *not* depth.

Worked case: they may click Thunderbolt or Will-O-Wisp, ~50/50. Swampert is **immune** to
Thunderbolt but is a physical attacker permanently crippled by burn. Celebi **resists**
Thunderbolt (0.5×) and has **Natural Cure**, so burn is nearly free. Swampert is the argmax
against one branch; **Celebi is the hedge** — never best, never bad.

Two observations, both structural:

1. **On a damage-only outcome vector Celebi cannot win.** Swampert's damage is 0.0 in *both*
   branches (immune to one, and burn deals none). Celebi's is 0.06. Damage-only scoring picks
   Swampert. The hedge is invisible not because of depth but because **the cost of the burn branch
   is not damage** — it is *"this mon is now functionally half a Pokémon."* Give `φ` a coordinate
   for `neutralized-without-KO` and the same one-ply expectation flips to Celebi.
2. **Hedging is risk aversion over the branch distribution — it means caring about SPREAD, not
   just the mean.** Swampert is high-variance (perfect / crippled); Celebi is low-variance
   (fine / fine).

So the requirement is that the reduction can compute a **second moment**:

| what the reduction emits | what it can express |
|---|---|
| `max_k` | no distributional information at all |
| `Σ α_k o` | the mean — risk-neutral only |
| **`Σ α_k [o, o²]`** | **mean + variance ⇒ hedging becomes expressible** |
| `Σ α_k [o, o², σ(o−τ₁), σ(o−τ₂), …]` | a quantized read of the branch distribution: thresholds, CVaR, catastrophic mass |

Have `φ` emit second-order terms; the α-weighted **sum** then yields `E_α[o]` and `E_α[o²]`, and
`ρ` forms `Var = E[o²] − E[o]²` for free. A learned combination of mean and spread **is** a
learned risk attitude. **`max` cannot produce a second moment — not "does it badly", cannot.**

*Relation to the distributional-critic null:* `ValueDistHead` was a quantified NO because return
residuals were sub-Gaussian with no tail to re-weight. The **branch** distribution is a different
object — discrete, K=6, and genuinely multimodal (Thunderbolt and burn are different worlds, not
two samples from a smooth density). The null does not obviously transfer; neither is it evidence
in favour.

### 4.2 "They expect me to switch" — level-2, and what one ply can and cannot do

**Expressible one ply**, because it is a property of how `α` is computed, not of depth: if `g` can
see that my active is dead weight and my bench holds an obvious answer, it can learn *"in boards
like this, opponents click the switch-punishing move."* That is a feedforward function of the
board.

**Not computable one ply.** True level-2 is a fixed point — their action depends on their belief
about mine, which depends on mine about theirs. The object is an **equilibrium strategy, not a
best response** (the same simultaneous-move correction the search notes already make).

**The resolution is that self-play is the fixed-point finder.** The opponent literally is a copy
of us, and `α` is supervisable against what they actually did (the event log has it). The
equilibrium is approached by **training**, not computed at decision time. We need `α` to be a
learned function that self-play drives toward it — not a search.

**What one ply genuinely cannot do**, stated so the boundary is not fudged:
*"they're saving Explosion for my Celebi"* (a multi-turn plan) · *"I hedge now so I can pivot for
free next turn"* (a sequence) · off-distribution opponents, where `α` is amortized from training
data and simply wrong until retrained (the amortization gap, again).

---

## 5. Emit several statistics, not one (the PNA rule)

The literature's verdict on "which aggregator" is unusually decisive, and it is not "pick well":

- **GIN** (Xu et al. 2019, *How Powerful are GNNs?*) proves **sum ≻ mean ≻ max** in ability to
  distinguish multisets. Max is provably the weakest of the three common choices. We use max.
- **PNA** (Corso et al. 2020, *Principal Neighbourhood Aggregation*) shows **no single aggregator
  is sufficient** — every one has multisets it cannot separate — and that running several in
  parallel beats any one of them.

So the default is not a chosen rung but a **bundle**: `[R0, R1, R2L-sum, R2L-max]`, four
statistics where there is one today, at O(K) cost with K = 6. This converts "which reducer is
right?" from an architectural commitment into a measurement, and it is the single highest-value
regret-reduction in the document.

**Why SUM specifically matters here — the concrete consequence the generic citation hides.**
`max` cannot count. *"I have one Blissey"* and *"I have three Blisseys"* have **identical maxima**
and different sums. So every scarcity notion in this game — **is this my last answer to their
sweeper**, my only spinner, my sole Ground-immunity — is invisible to a max-pooled architecture
**no matter how good the belief is**. Second moments (§4.1) are the same story: a mean and a
variance are sums; a max is neither.

We have **no sum-aggregation anywhere in the op today.** That single absence is what makes
hedging, scarcity and risk attitude simultaneously unreachable, and it is why R2L is the gate for
everything interesting rather than merely a nicer reducer.

---

## 6. What this makes unrepresentable

Consistent with the generation's discipline of turning alignment defects into type errors:

- A reduction that describes **two different opponent moves in one defender's row** (D2) — because
  there is one α.
- A reduction **buried in the forward** — `REDUCE` is a declared call site with a named `how`.
- A pair fact **placed on a token without a named reduction** — inherited from `OpTensors` arity
  typing.
- A reduced output **used as an absolute at a logit when it is a latent** — the W/L type split
  makes the routing table §3.2 checkable rather than conventional.

---

## 7. Pre-registered gates

Written before any number is taken, per the project's standing rule.

**G0 — byte-identity (provable).** `how=hard_max` reproduces the current op block bit-for-bit:
pi/vf outputs, the raw 660-dim block, unchanged `state_dict` keys, plus
`damage_op_probe_fuzz_test.py` (the constructed-scenario physics oracle) and the full suite. Same
four-part gate the `gen3_damage_op_split_v1` relocation used. **If G0 fails, nothing else runs.**

**G1 — the offline reducer bake-off (free, no training, THE decisive early gate).** On a frozen
checkpoint, take ground-truth `pair_in` cells and ask each rung to predict a target we already
have tooling for: the switch chosen by the `better-line` CRN-anchored beam
(`main.prober.query better-line`, `utils/bridge/search_session.py`), over eval-trace decisions
where a switch is legal and the beam's ΔV is material.
*Pre-registered reading:* a rung that cannot predict the right switch **from ground-truth cells
with a linear probe** will not learn to once it is in the loop. Rank the ladder; kill everything
that does not beat R0 by a margin exceeding the seed spread. Report the full ranking, not the
winner.

**G2 — the coherence acceptance test (constructed, deterministic, named).** Build a board via the
omniscient `utils/bridge/damage_probe.js` path where (i) the opponent's most likely move is a
status move, (ii) the damage-argmax move is physical, and (iii) one bench mon has Natural Cure and
another is physically bulkier. **Assert the reduced row for the Natural Cure mon reflects the
status threat.** Under R0 this must FAIL (D2 says it cannot pass); under the chosen rung it must
pass. A fixed-seed regression test that fails if the fix reverts — the project's edge-case rule.

**G3 — identity at init, on a REAL policy.** Every new zero-init Linear (`ρ`'s output projection,
`g`'s logit head) must still be zero **after** `MaskablePPO` builds the policy — SB3's ortho-init
pass clobbers extractor zero-inits (`gen3_identity_init_guard_v1`, ledger **M1**). Assert on a
real `MaskablePPO`-built policy, never on a bare extractor; the protected set is captured by
observation, so a new zero-init module is covered automatically once it is zero at construction.

**G4 — equivariance.** Permute the K believed moves → every Contract-W output is **invariant**;
permute our team → every per-defender row **permutes with it**. Executable form of
[[entity_tokens_biases_pointers]] §6.9.

**G5 — the acceptance measurement (POST-concat; now takeable).** Permutation importance of the
reduced `our_mon` block must **exceed the baseline established by step 0**, and the constructed G2
scenario must move the switch logits.

> **The threshold was re-pinned 2026-08-10.** An earlier draft set it against "the 6.60% `in_permon`
> reads today" — a figure from the **defective** pre-`754ca78` sampler, and measured while the concat
> was still carrying the traffic. **Both defects point the same way: it is too low.** Comparing a
> post-concat reducer against a suppressed, mis-sampled baseline would pass this gate on
> arithmetic rather than on merit.
>
> **The baseline is whatever §8 step 0 measures on gen-5** — same probe, stratified sampler, no
> concat, `how=hard_max`. That is a like-for-like R0 control and the only honest comparator. Record
> it here when taken; do not forecast it.
>
> **TAKEN (2026-08-10) — and it vindicates the re-pin emphatically.** `gen5_op_block_split_24M_site_op`:
> the reduced route reads **65.07% shuffle flips** (kl 1.82, |dV| 5.54 zero / 3.30 shuf). The old
> threshold was **6.60%**. A rung clearing 6.6% would have "passed" G5 while sitting an order of
> magnitude below the R0 control it was supposed to beat — the failure mode this correction was
> written to prevent, confirmed with a real number rather than an argument.

**G6 — cost.** Contract L is ~36 small MLP evals per side per forward. Measure the B=1 CPU
opponent forward (the compiled path, `--compile-extractor`, currently 0.976 ms) and the training
forward before/after. **Budget: no measurable regression on the compiled B=1 path** — that path is
a shipped 6.53× lever and an invisible regression there is expensive. Not asserted; measured.

**G7 — THE CAPABILITY GATE: a single-team exploiter A/B (owner design, 2026-08-09).** This, not
an information probe, is what decides whether the capability pays. Two warm-forked exploiters on
**one fixed team**, differing only in the reduction.

*Why this design and not a value-of-information probe:* fixing the team collapses the dominant
noise source (team sampling; the LUT arms returned CI [−0.016, +0.064], wide enough to hide the
effect), warm-forked exploiters converge around 2M steps rather than 40M, and — the decisive
property — **it isolates the variable**. Against a *known* team the presence belief `w` is
near-perfect, so any remaining gap is purely the *usage/intention* component, which is exactly
what is under test. A probe measures whether information helps; this measures whether the
architecture plays better.

- **Team spec — "BIG FIVE + STARMIE" (owner-specified, 2026-08-09).** The fixed team is
  **Tyranitar · Blissey · Gengar · Swampert · Starmie · Skarmory**.

  **This is an existing SAMPLE TEAM — select it from `data/teams/`, do not construct it.**

  | | |
  |---|---|
  | File | `data/teams/specialist/tss_starmie.txt` (byte-identical to `data/teams/sample/f6229d2c867e21d6.txt`) |
  | `pin_sha` | **`bcd4d09ee9`** (resolved via `agents.training.team_archetypes.load_team_archetypes`, **not** a raw-file hash) |
  | Archetype | **`semi_stall`**, pace 0.4 |
  | Tags | `sand` · `spikes` · `spin` · `spinblock` · `phaze` · **`status_heavy`** |
  | Features | `n_status` **3** · `n_attackers` 4 · `n_recovery` 1 · `n_setup` 0 · `n_boom` 1 |

  Two consequences. First, the archetype label joins every provenance record for free and "same
  team across both arms" is provable rather than asserted. Second — and this is a *data-backed*
  argument for the choice rather than my type-chart reasoning — the pool's own labeller calls this
  team **`status_heavy` (`n_status` = 3)**. Status is precisely the axis a damage-argmax reduction
  is structurally blind to, so in mirrors and near-mirrors the opponent will be clicking exactly
  the moves the current block cannot see.

  Prior lineage worth knowing: **TSS = Tyranitar/Skarmory/Starmie, and this project already ran a
  TSS specialist arc** (`project_tss_specialist_poc`, MatchupSpec + fold-back shipped). Tooling and
  possibly baseline numbers already exist for this exact team, which materially de-risks G7.

  **The actual sets make the test sharper than a generic version of this team would:**

  | slot | set (as committed) | why it gives the experiment power |
  |---|---|---|
  | **Gengar** | Timid, **Will-O-Wisp / Thunderbolt** / Ice Punch / Explosion | **§4.1's exact branch ambiguity is ON THE TEAM.** In a mirror the opposing Gengar presents the burn-or-Thunderbolt decision directly — plus Explosion as a qualitatively third branch |
  | **Blissey** | Modest, Natural Cure, Soft-Boiled / Ice Beam / Toxic / Fire Blast | walls the Thunderbolt **and** Natural-Cures the burn ⇒ **the hedge** |
  | **Starmie** | Timid, Natural Cure, Hydro Pump / Ice Beam / Thunderbolt / **Rapid Spin** | a **second** Natural Cure mon — but Water/Psychic takes Electric at 2×, so it is **NOT** the right hedge here. Also the only spinner ⇒ scarcity is live |
  | **Swampert** | Relaxed, defensive (240 HP / 136 Def), EQ / Ice Beam / Hydro Pump / Protect | **Electric-immune** ⇒ the argmax against the Thunderbolt branch; burn costs it its only physical move. Perfect on one branch, poor on the other |
  | **Tyranitar** | Adamant **196 Atk**, Focus Punch / Rock Slide / HP Bug / EQ | the team's **physical breaker**, so burn is devastating. Psychic-immune and resists Fire/Ghost/Dark, but **4× Fighting** — and an opposing TTar's Focus Punch is exactly that. *(No weather help: **gen3 Sandstorm gives Rock-types NO SpD boost** — that ×1.5 is gen4+, pinned by `incoming_damage_test.py::test_gen3_sandstorm_has_no_spd_boost`.)* |
  | **Skarmory** | Careful **248 SpD**, Spikes / Protect / Roar / **Toxic** | a *specially*-defensive Skarmory and a second Toxic user — it supplies status pressure to the opponent side in mirrors |

  **The decision this manufactures is unusually discriminating.** Facing a Gengar that may click
  Will-O-Wisp or Thunderbolt: Swampert is the argmax against Thunderbolt (immune) but loses its EQ
  to a burn; Tyranitar is ruined by the burn; Starmie has Natural Cure but takes Electric at 2×.
  **Only Blissey is good on both branches.** So a model that has merely learned *"Natural Cure
  absorbs status"* picks Starmie and is wrong — it must weigh the Electric branch **at the same
  time**, under one coherent `α`. That is the capability under test, and the team produces the
  decision naturally rather than by construction.
- **Opponent spec.** The ambiguity must be present or the experiment has no power: opponents
  carrying a strong special attack **and** a status move on the same set — Gengar
  (Thunderbolt + Will-O-Wisp), Zapdos (Thunderbolt + Toxic/Roar). Chosen, not sampled.
- **Arms.** A0 = current reduction; A1 = coherent `α` + rich message. Same fork point, same
  opponents, same seed schedule, several seeds (single-seed deltas of 0.02–0.03 are noise here).
- **Fork identity is the control.** The new modules are zero-init, so **A1 at step 0 is
  byte-identical to A0 at step 0** — which holds only if `restore_identity_init()` covers them
  (G3). That is what makes the fork point a genuine control rather than a starting-point
  difference, and it is why G3 is an experimental-validity gate, not just a stability one.
- **No capacity-matched arm** (owner decision, 2026-08-09). Repeated measured nulls — P3's
  `value_pooled` widening (effective rank 3–4; AUC 0.833 vs 0.835 over 384 dims), both LUT
  conditioning arms, the arch/compute decision — establish that added capacity does not move this
  model. It is additionally unnecessary *here*: the primary readout is a **directional** usage
  change, and parameters cannot manufacture "brings Blissey in against status users specifically."

**Readouts, most to least diagnostic:**

1. **Behavioural bifurcation (primary).** Condition Blissey switch-ins on the opponent's revealed
   threat type. Does A1 bring her in against *status* users at a higher rate, while both arms
   bring her in against special attackers? Measurable per switch from the event log, and it can
   fire **even if win rate does not move**.
2. **`α` accuracy.** Does `α`'s argmax predict the move they actually clicked better than `w`
   does? Free, offline, on the exploiter's own traces.
3. **Hedge preference.** Use `reroll_many` / `replay-counterfactual` to score both candidate
   switches against *each* opponent branch, identify turns where a hedge exists, and measure how
   often each arm takes it.
4. **Win rate** — what matters, and the noisiest per step.

**Pre-registered falsification:**
- no bifurcation **and** no win-rate delta → the capability does not pay on this team. Clean kill.
- bifurcation **without** a win-rate delta → it works and does not matter, i.e. the decision was
  already being made correctly by another route. **The most informative of the three outcomes.**
- bifurcation **and** a delta → build it into the generation.

---

## 8. Build order

**Rewritten 2026-08-09 for "gen-5 is already running."** Two changes from the original list: a new
**step 0** that only became possible once the concat died, and **step 1 is already done on main.**

| # | step | provable? | needs a run? |
|---|---|---|---|
| **0** | **Measure the reduction's TRUE price** — re-run `op_block_split_audit` on gen-5 (stratified sampler, `audit_states.py`). With no concat, `imx_CELLS` is no longer suppressed by a competing route, so this is the first honest number the design has ever had. **Do this before building any rung** — it sets the size of the prize and could itself kill the whole line. | — | no (frozen ckpt) |
| 1 | ~~`REDUCE` call site + Contract W typing, `how=hard_max` only~~ **DONE on main** — `damage_op.py:534` `_chan_max(value, channel_mask, how="hard_max")` is the named single site, and non-`hard_max` raises `NotImplementedError` so a rung cannot be added anywhere else. Landed with the concat deletion (`6aac795`). What remains of this step is the **Contract W typing** (α as the returned object rather than a reduced value) — still byte-identical. | **byte-identical (G0)** | no |
| 2 | Fold `acc` into `pair_in`; `provenance` := `Σ α·w` | byte-identical under R0 | no |
| 3 | Implement R1, R2W, R2L behind `how` (incl. second-order terms in `φ`, §4.1) | — | no |
| 4 | `α` as a learned board-conditioned usage model, zero-init to `normalize(w)` | identity at init | no |
| 5 | **G1 bake-off** — rank the ladder offline | — | **no** |
| 6 | **G2** constructed coherence test | — | no |
| 7 | G3 / G4 / G6 | — | no |
| 8 | **G7 — the single-team exploiter A/B.** The capability gate | — | **2 short forks** |
| 9 | Ship the R3 bundle as the default | — | — |
| 10 | R4 / OA1 — the query-conditioned rung | — | with the generation |
| 11 | **G5** — now takeable, on gen-5 (the first post-concat run) | — | yes |

**Steps 0–7 need no training run at all**, and step 8 needs two ~2M-step forks rather than a
generation. **This is what makes missing the pre-generation window survivable**: the expensive part
was always the *decision*, and the decision is almost entirely purchasable offline. Gen-5 occupies
the GPU for ~40M steps; steps 0–7 run beside it on frozen checkpoints and land the rung choice at
the gen-6 boundary — with step 0's number in hand, which the original plan never had.

**The one real cost of the missed window, stated plainly:** gen-5 trains against R0 `hard_max` and
cannot be changed mid-run, so the earliest a better reducer can be *in* a generation is gen-6.

### 8.1 ADOPTED PLAN (owner, 2026-08-10)

The §8 order is adopted as the plan of record, with the following amendments and schedule
commitments from the adoption review:

| when | work | gate / decision rule |
|---|---|---|
| **now, beside gen-5** (offline) | **Step 0 first** — `op_block_split_audit` on the gen-5 **final** checkpoint (~25M), stratified sampler (`audit_states.py`) | **Downscope rule, pre-registered:** if the unsuppressed `imx_CELLS` shuffle-flips reading stays ≲7% — no larger than the suppressed gen-4 6.53% — only the cheap rungs proceed (Contract-W typing + R1); R2L and G7 are dropped and the design closes with a measured null |
| now, beside gen-5 | Step-1 remainder + step 2 — Contract-W typing (`α` as the returned object) + the acc/provenance fold. Byte-identical, G0-gated | lands regardless of step 0's outcome |
| now, beside gen-5 | Steps 3–4 — R1 / R2W / R2L behind `how` (second-order `φ` terms included, §4.1), `α` zero-init to `normalize(w)` | G3 identity-at-init **on a real `MaskablePPO`-built policy** |
| now, beside gen-5 | **Seed VICReg** (`--seed-vicreg-coef`) — a named **prerequisite** of this design's critic route, per the §3.2 gen-5 collapse confirmation | enable at gen-6 launch; `seeds/*` must show un-collapse before any Contract-L bake-off against the critic sink is read |
| after steps 3–4 | **G1** offline bake-off (rank the ladder vs the beam) + **G2** constructed coherence test + G4 / G6 | kill every rung that does not beat R0 beyond seed spread; G6 budget = **no measurable regression on the compiled B=1 path** |
| **immediately post-gen-5** (first free GPU window) | **G7** — the single-team exploiter A/B on `tss_starmie` (2 × ~2M warm forks) | the §7 three-outcome falsification; decides whether R4/OA1 rides gen-6 or waits |
| gen-6 launch | ship the surviving R3 bundle **beside** `hard_max`, VICReg on; R4/OA1 per G7's verdict | — |

Two review cautions recorded with the adoption: **let R1 lose honestly** (D1 proves the current
functional is *arbitrary*, not that the marginal is *better* for a risk-sensitive game — §10.2);
and the R3 bundle roughly quadruples the reduced per-mon statistics feeding the switch cell and
prefuse rows, so the G6 compiled-path check is mandatory **before** the gen-6 launch, not after.

---

## 9. Anti-patterns (do not relitigate)

- **Do NOT widen `prefuse_proj`'s per-mon row as the re-home for `in_matrix`.** `in_permon` *is*
  the collapse; widening delivers more of the collapsed form and cannot carry the un-collapsed
  axis without re-collapsing it. (`design_conditional_opponent_cells.md` §0.4; the structural half
  of that argument never depended on the ratio.)
- **Do NOT replace the hard max — add beside it.** R0 stays shipped. The measured regret is
  asymmetric: adding a statistic costs dims, removing one can cost a generation.
- **Do NOT pre-blend probabilistic branches into one column** (ibid. §2.3).
- **Do NOT revealed-gate a marginal over the opponent's bench** (ibid. §4.1).
- **Do NOT decide `how` from the superseded gen-3 cell reading.** §1(a) explains why that number was
  structurally unable to price the post-concat world — and the sampler that produced it was defective
  besides. Take step 0's gen-5 measurement instead.

### 9a. The admission test for a new message coordinate

`φ`'s input width is not free: the pointer reader is thin and shared (§3.2 correction), so every
coordinate dilutes the others. One cheap, strict rule before adding one:

> **Name two specific actions whose ordering it flips.** If you cannot produce the pair, it is
> decoration.

- neutralization → flips Swampert → Celebi against the burn/Thunderbolt branch ✅
- tempo → flips "Milotic absorbs it" → "absorbs it but falls a turn behind" ✅
- "average threat level" → flips nothing; a smooth function of what is already delivered ❌

**And the derivability rule, with its counter-rule — both halves are needed.**

*Do not deliver what is derivable.* If a quantity is a simple function of coordinates already
present, the model can compute it; adding it is redundancy. This is the provide-facts line.

*But "derivable in principle" is not "derivable in practice."* Gradient starvation (Pezeshki et al.
2021) says a feature that could be computed but sits behind a collapse never develops — whatever
already explains the loss starves it. So:

> **Derivable from what IS delivered → do not add it. Derivable only from something the reduction
> already destroyed → you MUST add it, because "in principle" died at the `amax`.**

Status is the clean case: it is not derivable from ten damage numbers by *any* function, because it
was never delivered in a compatible currency (§2.1). That is absence, not laziness.

**Validate with a representation probe, not the loss** — freeze, linearly decode the quantity from
the cell, report r². The recorded warning is that damage **spread** decodes at **r² 0.06**
(`project_representation_probe`): the model carries magnitude and barely carries uncertainty — the
same second-moment hole as §4.1, one level up.

---

## 10. Open questions and honest caveats

1. **Sum-pool conditioning.** Deep Sets is universal but a shared `φ` + sum trains worse than a
   `Linear` early on. Mitigation is zero-init `ρ` (+ G3), but "universal" is about *representable*,
   not *learnable*, and this design does not pretend otherwise.
2. **`w·d` vs `Σ w d` is not settled by theory.** D1 says the current functional is arbitrary; it
   does **not** say the marginal is better for a risk-sensitive decision. That is what G1 R0-vs-R1
   is for, and R1 may lose.
3. **The G1 target is the beam, not ground truth.** The `better-line` beam is a shallow CRN-anchored
   search over our own critic — a policy-improvement operator, not an oracle. A rung that predicts
   it well predicts *our search's* preference. Stated so nobody later reads G1 as a correctness
   result.
4. **Generalist only.** No gen-3 exploiter checkpoint exists; a reduction adequate for a generalist
   may not be for an archetype specialist. Same standing caveat as `design_op_tensors.md` §2.5.3.
5. **G2 is one scenario.** It proves the coherence defect is real and fixed at that point; it is
   not a distributional claim about how often it costs games. The field-live stratification that
   would answer that has still not been run.
6. **Does the pointer switch cell even want a summary?** With six switch logits the head has output
   slots for the *defender* axis, so R4 could deliver a per-action conditional and the per-mon
   statistic could shrink rather than grow. Not decided here; it is the natural follow-on once R4
   exists.
7. **A phys/spec REFUND, hypothesised not measured.** The `phys_*` / `spec_*` split in the 12-float
   per-mon row exists largely *because* the reduction could not say which threat was live. Under
   one coherent `α` you emit `is_phys` as a channel reduced under that same `α`, and the row
   plausibly collapses from 12 floats to ~8 — partly paying for the new machinery. Choice-Band
   conditioning still needs the physical channel separately, so this is a hypothesis with a check,
   not a plan.
8. **⚠️ A RETRACTION — the value-of-information ceiling argument was wrong.** An earlier draft of
   this reasoning claimed: optimal hedging ≤ play with perfect knowledge, therefore the gain from
   this whole line is bounded by the measured L3 oracle VoI ≈ 0.03. **That inequality does not
   hold.** It assumed the oracle arm plays the *best response* to the known action; it does not —
   it plays whatever *that architecture* can compute given one more input. What was measured is
   `E[collapsed reader + oracle] − E[collapsed reader]`, not
   `E[optimal play with perfect info] − E[collapsed reader]`. A reader that collapses the grid into
   an incoherent per-mon max cannot act on the oracle even when handed it, so the oracle has
   nowhere to land and the 0.03 bounds nothing about capability. (Owner pushback, 2026-08-09;
   correct.)
   **The correct reading inverts it:** the model *already holds* every number this decision needs
   (`pair_in` contains all of it), so "extra information does not help" localises the bottleneck
   **downstream of information delivery** — which is evidence *for* the coherence hypothesis, in
   the same shape as the project's amortization-vs-bottleneck finding. G7, not a VoI re-run, is the
   gate.

---

## 11. NOT this document — the scarcity feature ("is this my last answer?")

*"Is this my last mon that's good against their sweeper?"* is **not** a setting of
`REDUCE(pair_in, …)`, and filing it here would be a category error. Recorded so it is not lost,
and so the boundary is explicit:

- `pair_in` is **[their MOVE × my mon]**. The sweeper question needs **[their MON × my mon]** — a
  different tensor, currently living in the `d2` / `d4` edge families.
- It reduces over **my** axis, not theirs.
- It produces a **cardinality**, then a per-action counterfactual delta:
  `answers_b = Σ_j alive_j · good(j,b)` and `remaining_b = answers_b − good(j,b)` — the C-family
  hypothetical-world pattern, one subtraction. `remaining_b ≈ 0` is literally *"this is my last
  answer and I am about to spend it."*

**The `OpTensors` arity typing catches this automatically** — different shape, different tensor,
so the type system refuses to let it be called the same reduction. That is the typing earning its
keep on the first new feature we tried to attach to it.

It is *enabled by* this document's machinery (it needs sum-pooling, which exists nowhere in the op
today) and **gated on R2L landing**, but it deserves its own short design.

## 12. Provenance

| claim | source |
|---|---|
| the eight independent maxima, `w·d` objective | `src/agents/model/damage_op.py:2777-2780` (read 2026-08-09) |
| the separate status max | ibid. `:2565` |
| `acc` / `provenance` gathered at the damage argmax | ibid. `:2796-2805` |
| per-mon block = 12 features; cell = 6 channels; switch cell = 15 | `_DMG_PER_MON`, `_DMG_IMX_CELL`, `_PTR_SWITCH_CELL_IN` |
| **`FULL_CONCAT` net +0.00%, `imx_CELLS` 6.53% shuf, `imx_HEADERS` 18.30% shuf (gen-4 @25M, n=6000, STRATIFIED)** | `designs/research_state/measurements/gen4_op_block_split_25M.json` — **the citable figures** |
| ⚠️ the gen-3 figures this doc originally quoted (`imx_CELLS` 5.77%, `in_permon` 4.52→6.60%) are **defective** | `gen3_op_block_split_40M.json`, `gen3_op_block_dependence*.json` — produced by the pre-`754ca78` sampler (lexical step-dir sort ⇒ whole sample from one mid-run dir). Superseded; do not re-quote |
| the concat is deleted; gen-5 is the first run without it | `6aac795` (`gen3_no_concat_v1`, `MODEL_CONFIG_VERSION` 61 — verified live via `agents.model.model_version`); run `ai_v9_06_gen5_no_concat_0809` |
| the reduction is already ONE named call site with a `how` knob | `src/agents/model/damage_op.py:534` `_chan_max(..., how="hard_max")`; non-`hard_max` raises `NotImplementedError` (read 2026-08-09) |
| `MultiSeedValueReadout` k=4 × 64 vf-only seed queries + the collapse contract (TB family `value_seeds/*`, renamed from `seeds/*` at v62) | `6aac795`; `src/agents/training/instrumented_ppo.py` (`seed_diagnostics` every `train()`) |
| ⚠️ **SUPERSEDED** — an earlier row here said the seed diagnostics are "logged only, NOT in the loss." True of **v61 only**. The trigger (`query_cos` > 0.6 **or** `out_effective_rank` < k/2 past ~2M) **FIRED on gen-5**, and **v62 wired VICReg into the loss**: `loss += value_seed_vicreg_coef * seed_vicreg_loss(...)` | `instrumented_ppo.py:1590-1596`, `:357`; `--value-seed-vicreg-coef` (`train_rl_agent.py:859`), resume-immutable; gen-6 runs it at 0.1 (read 2026-08-10) |
| **the pointer switch cell contains NO status coordinate** (15 = 12 incoming + `[phys_high_cb, pko_cb, p_cb]`) | `damage_op.py::pointer_cells`; `_DMG_PER_MON` = 2×5 + 2 = 12 (read 2026-08-10) |
| in production, incoming status reaches the model ONLY as the `s3` edge family (a ratio); the trunk route is OFF | `production_config.json`: `threat_status_refine` **False** (⇒ `status_in_proj` is `None`, `features_extractor.py:2707`), `edge_bias_families` includes `s3` (`:3598`); `damage_matrices_outgoing_all` **False** ⇒ no OAX tail, so the cell is exactly 15 |
| the pointer head reads the cell through a shared thin MLP, **not** affinely | `features_extractor.py:2049`, `:2069-2076` — `Linear([token ‖ cell]) + ctx → tanh → Linear`, same weights for all 6 slots (corrected 2026-08-10) |
| gradient starvation — a computable-but-buried feature never develops | Pezeshki et al. 2021, *Gradient Starvation: A Learning Proclivity in Neural Networks* |
| simplicity bias — the simplest predictive feature wins even when richer ones exist | Shah et al. 2020, *The Pitfalls of Simplicity Bias*; Geirhos et al. 2020, *Shortcut Learning* |
| damage **spread** decodes at r² 0.06 | ledger; `project_representation_probe` |
| sum ≻ mean ≻ max for multisets | Xu et al. 2019, *How Powerful are Graph Neural Networks?* |
| no single aggregator suffices | Corso et al. 2020, *Principal Neighbourhood Aggregation* |
| max over learned features | Qi et al. 2017, *PointNet* |
| `ρ(Σφ(x))` universality | Zaheer et al. 2017, *Deep Sets* |
| latent width ≥ set size | Wagstaff et al. 2019, *On the Limitations of Representing Functions on Sets* |
| message → aggregate → update | Gilmer et al. 2017, *Neural Message Passing for Quantum Chemistry* |
| SB3 clobbers zero-init | `src/agents/model/CLAUDE.md` → identity-init guard; ledger **M1** |
| capacity does not move this model (⇒ no capacity-matched arm in G7) | ledger **P3**; both LUT conditioning arms; `project_arch_compute_decision` |
| the L3 opp-action oracle, VoI ≈ 0.03 | `project_l3_oracle_grind_l4`, `project_opp_action_head_falsified` — see §10.8 for why it does NOT bound this work |
| G7 design; the Big Five + Starmie team; rejection of the capacity-matched arm | owner, 2026-08-09 |
| the adopted plan §8.1 — schedule, the step-0 downscope rule, G7's post-gen-5 slot | owner + adoption review, 2026-08-10 |
| gen-5 seed collapse: `out_cos` 1.000, `out_effective_rank` 1.0 sustained 196k→15.7M steps | gen-5 `ai_v9_06_gen5_no_concat_0809` TB `seeds/*`, read 2026-08-10 |
| the team's file, `pin_sha` `bcd4d09ee9`, archetype `semi_stall` / `status_heavy` | `data/teams/specialist/tss_starmie.txt`; `agents.training.team_archetypes.load_team_archetypes` (resolved 2026-08-09) |
| gen3 Sandstorm gives Rock-types no SpD boost | `src/agents/observation/incoming_damage.py:95`; `incoming_damage_test.py::test_gen3_sandstorm_has_no_spd_boost` (the gen3 mod leaves `onModifySpD` undefined) |

## See also

- `design_op_tensors.md` §3.2 — the one-line sketch this document specifies
- `design_conditional_opponent_cells.md` — OA1 (= R4), the magnitude rules, the anti-patterns
- [[entity_tokens_biases_pointers]] §6.9 — invariance vs equivariance vs position; the rank-*h* trade
- [[marginalization_and_uncertainty]] — convex combinations; why Contract W preserves units
- [[shortcut_learning_and_feature_delivery]] — the axis rule ("never collapse an axis you must choose along")
