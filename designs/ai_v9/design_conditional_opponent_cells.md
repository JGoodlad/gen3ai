# design — magnitude in the entity world + the CONDITIONAL OPPONENT-ACTION cells (OA1 / OA2)

**Status:** forward design, not built. Written 2026-08-07 off the gen-3 @9.6M measurements below.
**Owner decision needed:** none to start Part 0/1; Part 2 has a hard prerequisite (§4.1).

> **⚠️ NAMING — these are NOT edge families.** `OA1`/`OA2` are per-action **pointer cells**
> (flag `--opp-action-cells`), delivered through `DamageOperator.pointer_cells` →
> `PointerNativeActionHead`. They are unrelated to the shipped **C1–C5 consequence EDGE
> families** (`--edge-bias-families …,c1,c2,c3,c4,c5`), which are additive attention biases.
> The distinction is the entire point: **an edge bias cannot carry a magnitude** (§0.1), so
> these must go through the pointer path. Implementing them as edge families would silently
> reproduce the null this design exists to avoid. (An earlier draft called them C6/C7 — that
> numbering was misleading and is retired.)

Three deliverables, in order:

* **Part 0** — the *rule* for representing an attack's MAGNITUDE in an entity model. Read first;
  it is why Parts 1–2 have the shape they do.
* **Parts 1–2** — two new per-action cell families that make the two core gen3ou competencies a
  direct linear read at the logits instead of a physics rediscovery:
  * **OA1 (defensive)** — "they'll Ice Beam my Salamence; switch to the mon that eats Ice Beam."
  * **OA2 (offensive)** — "they'll switch out of my threat; click the move that beats the
    switch-in."
* **Part 2b — PV** — pair-VALUE attention: the same magnitudes delivered into the TRUNK, which
  is the only route that reaches the **critic** (which has no pointer head). OA1/OA2 solve the
  policy; PV solves the critic and cross-pair reasoning.

---

## 0. Part 0 — the rule for MAGNITUDE in an entity model

### 0.1 What each channel can physically carry

| Channel | Carries | Why |
|---|---|---|
| **Edge bias** (v56 families) | a **RATIO** only | it enters attention through the logits, and softmax normalises the row; what survives is a ranking within the row, not a scale |
| **Token content** (`prefuse_proj`) | an **ABSOLUTE** | written into the value vector; any query that attends the token receives it linearly, and an MLP can THRESHOLD it |
| **Pointer cell** (`pointer_cells`) | an **ABSOLUTE at the logit** | affine read, per-action, zero interference with routing |
| **Attention with PAIR VALUES** (Shaw 2018) | an **ABSOLUTE**, equivariantly | `out_j = Σ_k α_jk·(W_v·seat_k + W_p·cell_kj)`; `Σα=1` ⇒ a convex combination of cells is still an HP fraction |

**Rule 1 — an edge bias can never deliver "53% of max HP." Absolutes need token content, a
pointer cell, or pair VALUES.** An edge routes; it does not transmit.

**Rule 2 — pick the head by consumer.** The pointer head serves ONLY the policy; the value head
reads pooled tokens. **Token content is the only entity-native channel that reaches the critic**
— and the critic is the concat's heaviest user (gen-3 @9.6M: |dV| **5.67** concat vs **1.86** all
edges).

### 0.2 What to deliver (transform beats plumbing)

1. **Share a denominator.** Damage is already a fraction of the defender's max HP (`_rolls` →
   `high_frac`). Keep it.
2. **Precompute every nonlinearity of two numbers in the op.** `pko = acc·P(KO|hit)` compares
   damage to CURRENT hp inside the operator. Never make the net locate two scalars and threshold.
3. **Probabilities SATURATE; ship the MARGIN too.** `pko` is flat across "barely survives" and
   "survives comfortably" — the difference that decides a pivot. Add
   **`margin = high_frac − hp_frac`** (>0 ⇒ dead) wherever `pko` is shipped: constant threshold at
   0, linear both sides, ~1 float, reuses tensors already in hand.
4. **Deliver a distribution, not a point** (`[low, high, crit, pko]` — keep).

### 0.3 Order-freeness — four ways to handle an axis you cannot concatenate

| Way | Equivariant | Keeps identity | Expressive | Cost |
|---|---|---|---|---|
| concatenate in slot order | ✗ | ✓ | full | free — **do not** |
| **canonicalize**: index the ONE distinguished element (the ACTIVE — content-addressed) | ✓ | ✓ | full | free |
| **contract**: `Σ_b q_b · X(·,b)` with `q` from CONTENT | ✓ | via `q` | soft | ~free |
| attend with pair values / promote to pair tokens | ✓ | ✓ | full / rank-h | a module / O(n²) seats |

**Rule 3 — prefer content-addressed selection + a convex combination.** A softmax over a computed
quantity is permutation-invariant in the contracted axis AND magnitude-preserving. Both OA1 and OA2
are built entirely out of rows 2–3, so **neither introduces a positional axis anywhere.**

### 0.4 The anti-patterns this replaces

* **Do NOT** widen `prefuse_proj`'s per-mon row as the re-home for `in_matrix`. `in_permon` **is**
  the collapse of `in_matrix` (independent `max_k` per defender). Measured gen-3 @9.6M,
  shuffle-controlled flips: `in_permon` **4.52%** vs `in_matrix` **16.27%** — widening delivers
  more of the block the policy leans on LEAST, and cannot carry the other without re-collapsing
  the axis that makes it useful.
* **Do NOT** pre-blend probabilistic branches into one column (see §2.3).
* **Do NOT** revealed-gate a marginal over the opponent's bench (see §4.1).

---

## 1. Part 1 — OA1, the CONDITIONAL THREAT CELL (defensive pivot)

### 1.1 The defect

`_incoming_matrix` holds cell `(k, j)` = *their believed move k's damage to our mon j* — exactly
what a pivot decision needs. It reaches the policy only via the flat concat → `latent_pi`, i.e. as
**shared context, identical for every action**. What reaches the switch LOGIT per-action is
`pointer_cells`' switch cell = the **collapsed** incoming row (`max_k` per defender) + CB tail —
an INDEPENDENT max per defender, decorrelated from the move actually incoming.

The pointer scorer is Bahdanau-style (`tanh` AFTER the sum), so the context *can* re-rank
candidates — the competency is expressible. It is expressible only by RE-DERIVING type
effectiveness and bulk from `latent_pi × token_j` at rank ≤ `POINTER_HIDDEN`(64), from a
~1-bit-per-game signal, while the exact answer sits one tensor away.

### 1.2 The cell

Per switch candidate `j` (our 6 mons):

```
threat_k = damage(k → our ACTIVE)                       # from _incoming_matrix, active column
w_k      = softmax_k( λ · threat_k + log belief_k )     # who they will aim at us
cell_j   = Σ_k w_k · [ high(k,j), pko(k,j), type_mult(k,j), status_lands(k,j) ]
           ⊕ margin_j = Σ_k w_k·high(k,j) − hp_frac_j   # §0.2(3)
```

* Shape `[B, TEAM_SIZE, 5]`; appended to `pointer_cells`' **switch** cell (widens
  `switch_cell_dim`).
* `λ` is a **learned scalar** (`nn.Parameter`, init 0). λ→0 = paranoid (pure belief weighting);
  λ→∞ = greedy opponent. The model discovers how rational to assume the opponent is.
* Every term already exists in `_incoming_matrix`. **No new physics.**
* Order-free: `k` selected by content, `j` is the action's own entity, the contraction is convex.

### 1.3 Also do

Turn on **`--damage-matrices-outgoing-all`** (existing flag, currently `False` in gen-3). Without
it the switch cell carries the OAX attacker row nowhere, so the switch logit has **no offensive
information about the switch-in at all** — the second half of "defensive pivot → offensive pivot"
has no substrate at the per-action path.

---

## 2. Part 2 — OA2, the SWITCH-BRANCH MOVE CELL (punish the switch)

### 2.1 The mechanical constraint

Gen-3 is **simultaneous-move**: they commit without seeing our move, so `P(they switch)` is **ONE
scalar for the turn**, never per-move. Per-move is the CONSEQUENCE (a KO is wasted into a switch;
hazards/status/setup are better; Pursuit punishes). Switches resolve first, so our move lands on
the incoming mon.

### 2.2 The three factors — all from existing kernels

```
p_switch = σ( α·danger + β·bench_answer − γ ) · (1 − p_trapped)
           danger     = max_k' pko(our move k' → their active)      # outgoing block
           p_trapped  = the T edge (pairwise_trap), Smogon prior for unrevealed abilities
q_b      ∝ exp( −μ·E[dmg(our threat move → b)] + ν·threat(b → our active) ) · alive_b
X_sw(k)  = Σ_b q_b · X(k → b)                                       # pairwise_outgoing [B,4,6,6]
```

`α, β, γ, μ, ν` learned scalars. `pairwise_outgoing` and `pairwise_trap` are already computed each
forward for the D1 / T edge families — **no new physics.**

### 2.3 The cell — keep the branches DECORRELATED

Per move `k` (4 request slots):

```
[ high, pko, type_mult ]_stay          # exists today (vs their ACTIVE)
[ high, pko, type_mult ]_switch        # NEW  = X_sw(k)
wasted_ko = pko_stay(k) · p_switch     # NEW  "don't click the KO into the obvious switch"
```
plus **one shared `p_switch` scalar** on the move cell (broadcast) — **never**
`(1−p)·stay + p·switch` collapsed into one column.

Rationale: the head learns its own effective `p_switch` instead of inheriting the op's heuristic;
the *difference* between branches is the strategic quantity ("only good if they stay"); and it is
the convention C2 already follows (raw deltas decorrelated from `land`).

Shape `[B, _DMG_OUT_N_MOVES, 7]`; appended to `pointer_cells`' **move** cell.

### 2.4 Side effect worth having

The **X (Pursuit)** and **T (trapping)** edges currently read decorative (0.33% / 0.65% flips).
They are switch-punish and switch-denial mechanics — meaningless unless something reasons about a
switch that might happen. OA2 gives both a consumer. *Decorative-in-ablation may mean UNCONSUMED,
not unimportant.*

---

## 2b. Part 2b — PV, PAIR-VALUE attention (the critic's route to magnitude)

### 2b.1 The mechanism

Our fifteen edge families compute a rich per-pair cell and then discard all but `n_heads`
softmax-normalised scalars of it at the logits. Shaw et al. 2018 (*relative position
representations*) adds the pair term to the **values** as well:

```
out_i = Σ_j α_ij · ( W_v·x_j  +  W_p·cell_ij )
```

Since `Σ_j α_ij = 1` and each `cell_ij` is in HP-fraction units, the second term is a **weighted
average of real HP fractions — an absolute**, written straight into the residual stream. The two
halves are complementary: **the BIAS chooses the weighting, the VALUE carries the number.** Set
the weighting from a *different* channel than the one being averaged and you get a CONDITIONAL
expectation — i.e. **OA1 is the closed-form special case of PV with the right cell design**
(include `damage(k → our ACTIVE)` as a channel of every pair's cell, and the bias can drive
`α_jk` from threat-to-active regardless of which mon is querying; then `Σ_k α_jk·damage(k,j)`
IS OA1's `Σ_k w_k·damage(k,j)`).

### 2b.2 The two orthogonal knobs

* **Per-pair width** — `W_p·cell_ij` as a full `d_head` vector vs a scalar × learned direction.
  Full vectors are affordable at sub-block scale (6×6×32×4 heads ≈ 75 MB at B=4096); the rank-1
  scalar form drops memory from `n²·d_head` to `n²` and is enough to carry a magnitude.
* **Number of output slots = how much collapses.** *You can only preserve an axis you have
  output slots for.* One query per our-mon = 6 slots for K×6 cells ⇒ a reduction is arithmetic,
  not a flaw in attention. The dial is a continuum:

| slots | mechanism | preserves |
|---|---|---|
| 1 / mon | single pair-value read | a weighted mean |
| k / mon | **PMA — k learned seed queries** (Set Transformer) | a rank-k sketch |
| K×6 | pair-token promotion (n 29→65, +0.25 ms B=1 ≈ +5%) | everything |

**Seeds are the underused middle**: cost linear in k, **no new seats** (so no n² growth and no
attention dilution), fully equivariant (one shared `W_p`), and each seed learns a different
question ("biggest hit", "most likely hit", "what status lands").

### 2b.3 Placement + cost

Two blocks, mirroring OA1/OA2: our-mon tokens × E4 threat seats (cells from `pairwise_incoming`,
D3's) and E3 move seats × opp-mon tokens (cells from `pairwise_outgoing`, D1's).

**Build it as a SEPARATE small cross-attention module, not inside `BiasedEncoderLayer`.**
Pair-values require `α` explicitly, which breaks fused SDPA — and the compiled trunk is a
measured 6.5× lever. Keep the main layers fused. Zero-init `W_p` ⇒ identity at init; register it
in `restore_identity_init()` (M1).

### 2b.4 Gate it BEFORE building — the coverage probe

PV and promotion buy **cross-pair reasoning** (joint properties of the matrix: *"their Ice Beam
threatens three of my mons"*, *"every switch-in I have loses to something"*, *"they have no
answer to X"*). OA1/OA2 already solve per-action magnitude, because **the action space supplies
the output slots** (6 switch logits = 6 exact cells, zero collapse).

So probe first, with the existing representation-probe harness (`python -m main.prober.query
probe`, the one that found `damage_taken` r² 0.06 / `is_faster` AUC 0.94): linear-probe the
trunk for the joint quantities above. **Decodable at good r² ⇒ cross-pair reasoning already
happens and PV/promotion buys little. At chance ⇒ that is the gap, and PV (k=2–4 seeds) is the
cheap way to close it before considering 36 seats.**

Honest prior: physics-into-the-trunk measured NULL 3-for-3 (K9/K10). PV is a genuinely different
intervention — those varied *content* through a channel that could not carry magnitude, this
changes *what the channel can carry* — but that is an argument, not evidence, and M1 weakens
those nulls rather than strengthening this. Treat the critic-side gate as the real test.

---

## 3. Integration spec

| Item | Value |
|---|---|
| Flag | `--opp-action-cells {off,defensive,offensive,both}` (one knob; `defensive`=OA1, `offensive`=OA2) |
| Producer | `DamageOperator.conditional_threat_cells()` / `.switch_branch_cells()`; the op OWNS the layout, offsets pinned against `decode_damage_block` (the `pointer_cells` convention) |
| Consumer | `pointer_cells` → `PointerNativeActionHead` (`switch_cell_dim` / `move_cell_dim` widen; a missing source NARROWS the Linear, never zero-pads) |
| Learned params | `λ` (OA1); `α, β, γ, μ, ν` (OA2) — `nn.Parameter` on the op beside `out_gain`, all init 0 |
| Version | STRUCTURAL string in `check_compatible` (the `win_prob_mode` pattern). **OFF byte-identical ⇒ NO `ARCH_SIGNATURE` bump.** `MODEL_CONFIG_VERSION` +1 |
| Identity-at-init | **zero-init the NEW input columns** of `switch_proj` / `move_proj` so ON == OFF bitwise at step 0. The pointer head is built AFTER SB3's ortho-init, so M1 does not clobber it — **pin that with a test** |
| Threading | `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + BOTH `extractor_kwargs` sites |
| Requires | `--damage-op`; OA1 also `--damage-matrices incoming`; OA2 also `--damage-outgoing` + the T edge |
| Obs | **unchanged** — no obs-dim change, no obs benchmark needed |

---

## 4. Prerequisites and risks

### 4.1 ⚠️ HARD PREREQUISITE for OA2 — the unrevealed marginalisation

v34's outgoing matrix is **REVEALED-gated** (unrevealed opp slots zeroed). A revealed-gated `q_b`
reads ≈0 early — exactly when switching is most frequent — so the model concludes *"my move always
lands on their active."* Same GIGO class as the typeless-HP "immune" bug: **misleading, not merely
incomplete, in the phase where the feature matters most.**

The machinery exists — v36's expected-latent defender (`SPECIES_EXP_MULT` ⊕
`SPECIES_SPREAD_PRIOR` marginalised through the species belief) — but it rides
`--threat-refine-outgoing`, hence `--damage-refine-rounds > 0`, which the prefuse config sets to
**0**, so it is **inert in gen-3**. **Re-home the expected-latent read onto the prefuse path
before OA2.** Ship OA1 first; it has no such dependency.

### 4.2 Standing risks

* **Prior null:** an opponent-action AUX head was falsified on value-of-information (~0.03). This
  is a different use (a weighting inside the physics, gradient riding the damage, not a prediction
  target) — but state the null rather than ignore it.
* **One-ply only.** OA1/OA2 buy "answer the threat" and "punish the switch." They do NOT buy the
  two-ply plan ("pivot now to threaten next turn") — that stays a credit-assignment / teacher
  problem.
* **Delivery ≠ behaviour.** Measured: an oracle move-belief flipped **19.3%** of actions but moved
  switch mass by **+0.019**. Better incoming delivery may re-home the concat (an architecture win)
  without moving strength. Gate the two claims SEPARATELY (§5).

---

## 5. Tests and pre-registered gates

**Tests (all required):**
1. `damage_op_probe_fuzz_test` extension — constructed scenarios where the conditional cell equals
   a hand-computed marginal (the op's authoritative-physics convention).
2. ON == OFF **bitwise at init** (zero-init columns), on a REAL policy (post-SB3-ortho — M1).
3. `p_switch → 0` when the T edge says trapped.
4. `q_b` carries **non-zero mass on unrevealed slots** (the §4.1 guard, throwing).
5. **Equivariance**: permute our bench and their bench → OA1 cells permute with our team; the
   contracted OA2 column is **invariant** to their bench permutation.
6. B=1 CPU forward delta measured and reported (expect small — no new physics).

**Gates — pre-register BOTH, do not let one stand for the other:**
* **Re-home success:** `in_permon` / `in_matrix` sub-block dependence FALLS on the trained
  checkpoint (`tmp/incoming_conditional_probe.py`, shuffle arm, vs gen-3's numbers below).
* **Strength:** anchored ELO **non-inferiority** vs gen-3 at matched tranches — fix the margin
  before the run (e.g. within −15, CI excluding −40).
* **Behavioural (new, cheap):** a **wasted-KO rate** — fraction of decisions where the policy
  picks a move with `pko_stay ≥ 0.8` on a turn the opponent switches. OA2 should reduce it.

**Baselines to beat (gen-3 @9.6M, 6000 states, shuffle-controlled flips):** `in_matrix` 16.27% ·
`out_active` 6.25% · `in_permon` 4.52% · `in_cb` 1.28% · `out_status` 1.00% · whole concat 18.58%
· all 15 edges 13.9% · concat |dV| 5.67 vs edges 1.86. Reports:
`tmp/incoming_cond_gen3_6k.json`, `tmp/edge_audit_gen3_9p6M.json`, `tmp/oracle_voi_gen3.json`.
**Re-read at gen-3's 40M audit before acting** — at 9.6M the edges are still growing faster than
the concat and every level is mid-curve.

---

## 6. Build order

1. **margin channel** (§0.2(3)) — trivial, independent, ships anywhere `pko` ships.
2. **`--damage-matrices-outgoing-all` ON** — existing flag, gives the switch cell offense.
3. **OA1** — no prerequisites.
4. **unrevealed expected-latent onto the prefuse path** — §4.1.
5. **OA2** — after 4.
6. **coverage probe** (§2b.4) — offline, no training, decides whether 7 is worth building.
7. **PV, k=2–4 seeds** — only if 6 says cross-pair reasoning is missing. Critic-gated.
8. **pair-token promotion** — only if PV is insufficient. Promote the INCOMING block
   (`in_matrix`, the measured-dominant head block); note the head funnel pools it anyway, so
   the buy is trunk-side joint reasoning, NOT magnitude delivery.

## See also
* `designs/learning/entity_tokens_biases_pointers.md` — the worked example, the four-ways table,
  the MIRROR section (the reasoning behind this spec)
* `designs/learning/shortcut_learning_and_feature_delivery.md` Part 6/7 — the channel argument,
  the measurements, the concat end-state decision rule
* `designs/ai_v9/design_generation_roadmap.md` §3 — Stage 2/3 and the concat decision rule
