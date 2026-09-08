# ai_v9 — the concat deletion and the six forward designs, as of 2026-09-07

> **This directory is HISTORY — additive only.** A file here is written once, when narrative is
> lifted out of a `CLAUDE.md`, and is never updated afterwards; when the world changes, the
> `CLAUDE.md` changes and this file stays as the record of what was believed then.

Lifted verbatim from the **ai_v9** cell of `designs/CLAUDE.md`'s state table on **2026-09-07**,
when that cell (20.8 KB in one table cell) was cut to a pointer. It is the reasoning record behind
the op head-concat deletion and the ai_v9 forward designs — `design_conditional_opponent_cells.md`,
`design_conditional_execution.md`, `design_opponent_intent.md`, `design_pair_reduction.md`,
`design_history_entity.md`, `design_cleanup_journey.md` — including the gates, the owner
reconciliations of 2026-08-11 and the §7a review notes.

🚨 **Nothing here is current.** Several of these designs shipped (Stage 0/1/2, the H-A/H-B/H-C
history tiers, the v93–v95 substrate) and several were closed; `designs/ai_v9/` holds the docs
themselves and `designs/ARCHITECTURE.md` states the architecture as it is now.

---

Roadmap: `design_generation_roadmap.md` (the operative staged plan, slice statuses current). **The
op head-concat deletion is DONE (2026-08-09, `6aac795`) — it was the last of the stated goals and it
landed on evidence, not on schedule:** gen-4's stratified end-of-run audits showed `FULL_CONCAT` net
policy dependence **+0.00%**, all-edges-off flips (29.22%) **exceeding** the concat arm (22.70%) for
the first time in the lineage, and `act_threat` still decodable from `vf` with the concat zeroed (r²
0.400 → 0.418), so the remaining `|dV|` 4.75 was trained reliance on an open window rather than
structural necessity. Still open: C1b/C2/C3/C5 consequence edges, E9 history, and OpTensors steps
1–2 (typed views, recompute dedup) — deferred honestly to background work during gen-5, since the
§9.1 evidence showed the removal was not waiting on them. **NEW forward design (not built):
`design_conditional_opponent_cells.md`** — the magnitude rule for the entity world + the OA1
conditional threat cell (defensive pivot) and OA2 switch-branch move cell (punish the switch), plus
PV pair-value attention (a critic route), the unrevealed-marginalisation prerequisite and
pre-registered gates. **RESOLVED 2026-08-09 — read the two-route precondition below as history.**
The 2026-08-08 amendment required OA1 (policy) **and** a critic route (PV *or* generalized
token-content injection) to land before the concat could die, accepted only on **flips AND `|dV|`**.
What actually happened: the **flips** half was met by training alone (all-edges-off 29.22% > concat
22.70% on gen-4), the policy side needed no OA1 at all (net concat dependence +0.00%), and the
critic route that shipped was **neither** of the two candidates — it is `MultiSeedValueReadout`,
readout **multiplicity** rather than width (P3 refuted width, never multiplicity). `|dV|` remained
concat-led (4.62 vs 2.44) and was overridden on the conditional-coverage evidence above rather than
waited out. **OA1/OA2 and PV therefore survive as forward designs on their own merits, no longer as
preconditions for anything.** **OA1/OA2 are pointer CELLS, not edge families** — do not confuse them
with the C1-C5 consequence edges. **NEW forward design (not built), 2026-08-12:
`design_conditional_execution.md`** — the **OUTGOING** consumer of the `α`/`β` intent belief: a
per-mechanic spec of every gen3 move whose value cannot be computed without knowing what the
opponent will do. Same Contract-W contraction as the incoming side (`E[my move m] = Σ_k α_k ·
f(m,k)`, where `f` is a deterministic RULES lookup so `α` needs no change), but delivered to the
pointer **MOVE cell** — a per-action absolute in a channel measured to WORK (`d1` 12.17% / `d2`
19.25%), unlike the defensive edges. **Its motivating evidence: the consequence families are ALREADY
this idea, built with the wrong conditioning** — `c4` Protect carries `p_success` = the MECHANICAL
consecutive-use decay odds and never asks "will they attack"; `x` Pursuit carries `pursuit_p` =
P(they *carry* Pursuit), not P(they click it) × P(I switch). That is `α ≠ w` (presence vs usage) in
a second, independent place, and it plausibly explains why every consequence family sits at the
noise floor (c2 1.20% … c4 0.15%). **The structural find: FIVE mechanics are ONE operator** —
`p_thresh(τ,⋛) = Σ_k α_k · 1[damage(k,me) ⋛ τ]` covers Focus Punch (τ=0,>), Substitute (25%
maxhp,<), Endure (HP,≥), Destiny Bond (HP,≥ — same threshold, OPPOSITE valence) and Endeavor (HP,<);
the `τ=HP` case is **`p_KO`**, the α-weighted P(I die this turn), which is nearly FREE (the op
already computes per-move `pko` and `_chan_max` collapses it to a max — α turns that max into a
calibrated probability) and is exactly the quantity ledger **H1**'s self-KO defect mis-values.
Destiny Bond and Endeavor are the two mechanics whose value moves **OPPOSITE** to every damage
feature the op computes (DB rises with P(they kill you); Endeavor rises as your HP falls). Taxonomy:
class A this-turn / class B switch-contingent (`β`) / **class C long-horizon — Spikes is a switch
RATE, not a one-ply conditional, and is DEFERRED because modelling it here would be confidently
wrong**. Pool exposure over 773 team files: explosion 69.3%, spikes 45.7%, protect 38.2%, substitute
29.6%, focuspunch 26.1%, pursuit 21.0%, counter 8.9% — but §3 carries an explicit warning that
**carriage frequency is a POOR prioritiser alone** (it under-weights decisive-but-rare mechanics;
Destiny Bond is 0.8% and decides the games it appears in — the same blind spot that voided the
`OUR_MOVE_OUTCOME` reading). Endure→Endeavor is a two-turn PLAN (out of scope) but **each leg is a
one-ply conditional (in scope)** — that decomposition is what makes the pair tractable. Magic Coat's
reflectable set is marked **UNVERIFIED** pending constructed-scenario oracle work. **G3 is the gate
that prices the whole document: re-deliver ONE existing consequence family (`c2`, least-dead)
through the move cell with α — if it stays at zero, the consequence line is dead and the other seven
are not worth building.** Steps 1–7 need no training run. **NEW forward design (not built),
2026-08-11: `design_opponent_intent.md`** — the build for one sentence the model cannot express:
*"they are likely to click **this**, so **this** is my answer."* Supplies the two things the
pair-reduction operator was missing: a **distribution worth weighting by** (`α`, a SUPERVISED usage
belief over their K believed move seats + `SWITCH`; `β` over their team slots, conditional on
`SWITCH`) and an **outcome vector worth weighting** (one unified `pair_in` carrying damage AND
status AND `neutralization` AND `tempo_cost` — today damage and status are computed in two functions
with two reductions, and one `α` cannot weight two tensors). **The three-part framing is the doc's
core claim:** a distribution + a rich outcome vector + the weighting done per-action before the
logit — missing any one makes the other two useless, which is why **G1-FINAL's null was
near-guaranteed** (it tested part 3 alone, on damage-only cells, with `w` substituted for `α`).
Grounded in the gen-4 end-of-run edge audit: `d2` 19.25% / `d1` 12.17% (our offense) vs **`d3`
0.63%** (their believed threat, DOWN from 1.9% at gen-3 9.6M) — the entity system is overwhelmingly
offensive because the anticipatory half is routed through edges, which carry a softmax-normalised
RATIO and cannot deliver a per-action absolute. **Two owner reconciliations settled 2026-08-11:**
(1) *both sides anticipate* — `α` may not depend on our REALIZED action but MAY depend on our
POLICY, and since the policy is a function of the board, `α = f(board)` is already the right form;
the forced change is that **`α`'s INPUT must include OUR outgoing physics** (`d1`/`d2` grids), and
the fixed point is found by TRAINING (self-play), never solved at inference — reading our own policy
logits would be level-3 but creates a forward-pass cycle, so it is deliberately not taken; (2)
*belief-derived seats*, now governed by a **HARD OWNER CONSTRAINT (2026-08-11): the model must
always pick among the belief's DISCRETE states and may never invent a move — interpretability is the
reason.** So `α ∈ Δ^(K+1)` over named seats + `SWITCH`, no `UNKNOWN` slot and no learned property
head (both proposed in earlier drafts and CUT — §4.6 keeps their causes of death). **ONE rule on
both axes: "if we can't name it, we don't train on it"** — hard target when the belief holds it,
**masked** otherwise, mask rate logged as a first-class diagnostic. (A property-similar soft-target
scheme was drafted for the move axis and CUT: it yields a smeared object rather than the clean
`P(seat

modeled)`, it injects the belief's non-random blind spots as a bias invisible in `α`'s accuracy, its
similarity metric has no principled setting, and it was an unjustified asymmetry against `β`, which
masks. Its motivating example also dissolved — under canonical-id matching a bare-`hiddenpower` seat
MATCHES a used HP Ice.) Matching is by canonical id, never by index (seats permute per turn — the
Hungarian precedent). **The division of labour that buys:** `α`/`β` own *which of the things we
believe*, the belief head owns *whether we believe the right things* — two failure modes, two
measurements, instead of one head absorbing the other's errors. **`β` answers "switch to WHOM"** —
discrete over alive/non-active/revealed slots, masked in v1 when they bring an unrevealed mon (rate
logged; **B1**, BUILT-but-never-run, is the named upgrade that turns that mask into a posterior
soft-target). **`β` is also what makes the (bench × bench) offense grid actionable** — that grid
alone is an unweighted outer product; with `β` it answers *"if I bring Skarmory and they pivot to
Blissey, is Skarmory still doing anything?"*, so `d2`/`d5` are not independent cheap wins but the
grid `β` needs. The RL loss is **stop-gradiented** out of both heads so a null is interpretable.
**The constraint's real cost is a ceiling at belief quality, so §4.5 audits the WHOLE BELIEF STACK
against the live config — seven legs, and EXACTLY ONE is supervised.** B-move (which moves they
hold) **ON but UNSUPERVISED** (`move_belief_coef` `0.0`; `known_moves` already emitted + plumbed,
BCE unconsumed — shaped only by the Smogon prior + RL gradient); B-hptype ✅ supervised 0.05, acc
≈0.91; B-spread ❌ OFF; B-team (B1) ❌ OFF (BUILT, never run); B-latent ❌ OFF; **B-item and B-ability
are STATIC lookups that cannot improve with training** (`p_cb` = a species usage prior collapsing to
0/1 on reveal; abilities = Smogon per-species priors). **B-spread is a PHYSICS DEFECT, not a missing
signal:** with it off the op prices every opponent's offense as `(2·base_atk + const) × 1.1` — 252
EV + boosting nature, uniformly, at **nine sites** (`damage_op.py:1727` et al) — so `pair_in` is
computed against a fictional maximally-invested opponent, and the over-estimate scales with base
stats so it distorts the RELATIVE threat ordering, not just the level. A better `α` over de-timid
physics inherits the distortion ⇒ **B-spread is a correctness fix to component 1, not a third belief
leg to stack.** Operationally they differ: `--move-belief-coef` is **training-only /
resume-mutable**, `--spread-belief` is **STRUCTURAL / version-checked / FRESH-ONLY** (cannot join a
running generation). **G2a runs FIRST and needs no head at all** (how often does the top-K hold what
they clicked?); then G2b (does `α` beat `w`, `β` beat the alive-bench base rate?); G3b asserts the
discrete constraint as a TEST. **G0–G7 need no training run.** Not this doc: physics mutation
(Marvel Scale changing the whole matrix) is explicitly out of scope for a one-ply reduction. §9.1
records that the G1-FINAL SKYLINE is likely underpowered (2800 params on ~239 rows at L2 1e-3) and
should not be read as "the grid is exhausted" until re-conditioned. **§7a (REVIEW, 2026-08-11) adds
four notes:** (1) ⚠️ this is a **POLICY-side** design and the critic deficit is **separate** — the
dense frozen-vs-frozen ladder reads gen-4 @24M **2080.6 (se 10.70)** vs gen-5 **2037.4 (se 10.10)**,
and the sparse `eval/elo` at ±30 that reported "parity" could not resolve that. **§7a.1's amendment
(2026-08-11) downgrades "cost ~44 Elo" to SUGGESTIVE, not established**, on three grounds read back
from both runs' `snapshot_ladder/ladder.json`: the original "±11" was the **standard error, not the
CI** (at 1.96·se the intervals are [2059.6, 2101.6] vs [2017.6, 2057.2] — disjoint by **2.4 Elo**,
marginal); the **trajectories INTERLEAVE** (gen-5 is *ahead* at 14M and 16M, dead level at 20M); and
**the entire 43-point gap is gen-4's final checkpoint jumping +30** while gen-5's stayed flat — a
single-endpoint comparison across crossing trajectories cannot separate a real cost from a lucky
last snapshot. Disambiguate by fitting the last 3–4 checkpoints, or laddering gen-4 @22M against
gen-5 @24M. **The conclusion does NOT rest on this number**: the seed readout measuring ~1 effective
direction under BOTH VICReg and quantile pressure is an independent, self-standing critic-side
finding, so the scope gap is real on structural grounds and (2) remains the right response either
way; (2) the **critic route that falls out of this design** — send the same `Σ_k α_k·pair_in[k,j,:]`
row to the critic as **token content on our mon j's token**, pooled by `value_cls`: equivariant in
BOTH axes (`α` invariant under permuting their moves since `g` is shared over `k`; the row rides mon
`j`'s token; attention pooling is permutation-invariant), **no seeds** (which matters given seed
collapse measured at ~1 effective direction under BOTH VICReg and quantile pressure), and testable
BEFORE `α` exists by substituting `α := normalize(w)` (the shipped R1 rung) — separating the
DELIVERY claim from the DISTRIBUTION claim; (3) an **alternative hypothesis for `d3` = 0.63%** the
doc does not raise — a channel carrying DISTORTED content also reads low, and the de-timid defect
corrupts exactly the relative threat ordering `d3` conveys, so **re-measure `d3` after the B-spread
fix before concluding the channel was the problem**. **Refined 2026-08-11: the two are NOT mutually
exclusive and the build order does NOT fork on them.** The channel argument is STRUCTURAL — an edge
writes a ratio and cannot deliver a per-action absolute *however accurate the numbers being ratio-ed
are*, and fixing de-timid does not move the `argmax_a` off the pointer logits. Both are almost
certainly true at once, so the post-B-spread re-measurement buys a **magnitude, not a verdict**: it
says how much of the 0.63% was content. Neither outcome removes the need for the per-action route;
(4) the **coverage-risk fallback pre-registered**: `α ∈ Δ²` over {ATTACK, SWITCH} only —
belief-free, zero mask rate, carries the largest single effect — ship it if G2a returns poor
coverage. Plus schedule honesty: step 0b is fresh-only, so this is **gen-8 (foundation) + gen-9
(`α`)**, not one generation. **NEW forward design (not built), now scoped to RUN BESIDE GEN-5 and
land at the gen-6 boundary: `design_pair_reduction.md`** — the deep spec of the one line
`design_op_tensors.md` §3.2 sketches as `REDUCE(pair_in, over=MOVE_AXIS, how=…)`. Splits
**contract** from **knob**: a weighted reducer must emit ONE distribution over the move axis per
defender, shared across every channel, which kills the incoherence defect structurally (the
flat/trunk block takes NINE independent maxima, so up to nine different opponent moves describe one
defender). **§2.1 (added 2026-08-10, verified against the live tree) makes that understatement
precise and WORSE on the path that decides:** the pointer **switch cell is 15 numbers — ten damage,
`p_outspeed`, `provenance`, and the Choice-Band tail — with NO status coordinate in any currency at
all.** In production `threat_status_refine` is `False`, so incoming status reaches the policy *only*
as the `s3` edge family — an attention **bias**, i.e. a softmax-normalised RATIO. So "they'll click
a status move, so bring the Natural Cure mon" is unrepresentable not because status was mis-reduced
but because **the two quantities never meet in the same vector in the same units** — a CURRENCY
failure one level below the reduction failure, which no reducer fixes. **This is the most likely
reading of the G1 n=299 null** (no rung beats R0 beyond seed spread): the ladder was asked to
improve an aggregation over a vector that never carried the quantity the decision turns on. §9a adds
the admission test for new message coordinates (name two actions it flips) and the derivability rule
+ its gradient-starvation counter-rule; §2.1 names `neutralization` and `tempo` as the two missing
ones, with `physics mutation` (Marvel Scale) explicitly out of scope for a one-ply reduction. Ladder
R0 hard_max (byte-identity anchor, stays shipped) → R1 belief mean → R2 learned / Deep-Sets → R3
multi-aggregator default (GIN/PNA: no single aggregator suffices) → R4 = OA1. **Two claims carry
it:** `α` must be computed from the board ALONE — not per defender, because they choose without
seeing your switch — which kills the channel AND defender incoherence with one restriction; and
**hedging is not a depth phenomenon** — taking the middle ground under uncertainty needs a *second
moment* (`Σ α[o, o²]` ⇒ variance ⇒ a learned risk attitude), which `max` structurally cannot
produce, so it is reachable one-ply. Also names what the doc is NOT: the "is this my last answer"
scarcity feature is a different arity ([their MON × my mon], reduced over OUR axis) that the
OpTensors typing rejects as a shape error (§11). Gates G0–G7; **steps 0–7 need no training run**, so
the whole ladder is decidable *beside* gen-5 on frozen checkpoints — the one real cost of the concat
having died the same day is that gen-5 trains against R0 `hard_max` and cannot change mid-run, so
the earliest a better reducer ships inside a generation is **gen-6**. **ADOPTED 2026-08-10 — §8.1 is
the plan of record**: step-0 carries a pre-registered downscope rule (unsuppressed `imx_CELLS` ≲7% ⇒
cheap rungs only), **seed VICReg is a named prerequisite of the critic route** (gen-5's `seeds/*`
measured the k=4 readout COLLAPSED — `out_effective_rank` 1.0 sustained — so the trigger fired and
`--seed-vicreg-coef` is being wired for gen-6), and G7 runs in the first post-gen-5 GPU window.
**Step 0 is new and comes first: re-run the split audit on gen-5** — with no concat there is no
competing route, so `imx_CELLS` finally means what it says (on gen-4 it read 6.53% shuffle *while*
`FULL_CONCAT` carried the traffic). **Step 1 is already DONE on main** — `damage_op.py:534`
`_chan_max(..., how="hard_max")` is the single named call site and any other `how` raises. **G7 — a
single-team exploiter A/B with a behavioural-bifurcation readout — is the capability gate** (owner
design 2026-08-09; fixed team = **Big Five + Starmie**:
Tyranitar/Blissey/Gengar/Swampert/Starmie/Skarmory, chosen because every slot's value is
branch-dependent; deliberately no capacity-matched arm, per P3 + the LUT nulls). §10.8 retracts an
earlier VoI-ceiling argument of mine that was simply wrong. **NEW forward design (not built),
2026-08-14: `design_history_entity.md`** — E9 steps 2–3 made buildable: the three-tier history
design (H-A compiled last-action fields + the pair-history edge family `h[i,j]` answering "what do
they click into this mon / whom do they switch into"; H-B event tokens with RELATIVE recency
embeddings replacing the 7×159 lag-indexed frames; H-C entity reference edges), the
**last-turn-outcome admission rule** (a transition fact survives iff not derivable from current
state AND it serves belief-INVERSION or TENDENCY estimation — crit flags live only as deflators on
the damage evidence they explain; per-slot HP levels die as state duplication), and the
field-by-field nothing-lost mapping from today's TurnDelta layout. Deletes 1124 obs dims when H-B
lands; H-A ships independently first and feeds α/β their tendency inputs (measure `alpha_acc` across
it). **NEW operational plan, 2026-08-14: `design_cleanup_journey.md`** — the flag/delivery cleanup
phases as a decision record: the three flag roles (select/record/gate) and three explicitness tiers
(CLI / config-only / constructor-only); Phase 1 issued (registry + consistency test, zarch +
seed-pressure deletions, first demotions, the pubval decision package); the owner amendments —
**training transport defaults to rust** (node kept as the parity arm; offline seams stay node; soak
gate before the flip since the lock-in fix is a day old) and **diagnostics DEFAULT ON** (win_prob +
the distributional readout in `read_only` on fresh runs — instruments, not levers; per-decision
quantile/tail-mass trace recording feeding the prober's `knew_by_turn`/`lead_time`/`blind_loss`
verdicts); Phase 2 launch-by-manifest; Phase 3 post-gen-9 audit deletions; and the **SPARED
register** (pairwise kernels, pair_reduce rungs incl. R2/R3 awaiting a fair α retest, the diagnostic
heads, node seams, hidden-opp pool) so no stale deletion list outlives its evidence. **Reading aid:
the delivery digraph is browsable** — [`architecture_viewer.html`](../../architecture_viewer.html) via
`file://`, or served live at `model.g5d.io` (`python -m agents.model.build_arch_viewer --serve`,
which re-renders from the checkout on every request). Edge hue = what the channel physically
carries, thickness = measured dependence at a selectable checkpoint, plus a path filter for "what
does the critic read" — the fastest way to see the concat's critic-side residual above. It is
**generated** from the committed graph snapshot + `research_state/measurements/`, so rebuild it with
`python -m agents.model.build_arch_viewer` rather than editing the HTML; `--check` fails on drift.
