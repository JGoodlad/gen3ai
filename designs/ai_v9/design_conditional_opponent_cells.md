# design — magnitude in the entity world + the CONDITIONAL OPPONENT-ACTION cells (C6 / C7)

**Status:** forward design, not built. Written 2026-08-07 off the gen-3 @9.6M measurements below.
**Owner decision needed:** none to start Part 0/1; Part 2 has a hard prerequisite (§4.1).

Two deliverables, in order:

* **Part 0** — the *rule* for representing an attack's MAGNITUDE in an entity model. Read first;
  it is why Parts 1–2 have the shape they do.
* **Parts 1–2** — two new per-action cell families that make the two core gen3ou competencies a
  direct linear read at the logits instead of a physics rediscovery:
  * **C6 (defensive)** — "they'll Ice Beam my Salamence; switch to the mon that eats Ice Beam."
  * **C7 (offensive)** — "they'll switch out of my threat; click the move that beats the
    switch-in."

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
quantity is permutation-invariant in the contracted axis AND magnitude-preserving. Both C6 and C7
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

## 1. Part 1 — C6, the CONDITIONAL THREAT CELL (defensive pivot)

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

## 2. Part 2 — C7, the SWITCH-BRANCH MOVE CELL (punish the switch)

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
switch that might happen. C7 gives both a consumer. *Decorative-in-ablation may mean UNCONSUMED,
not unimportant.*

---

## 3. Integration spec

| Item | Value |
|---|---|
| Flag | `--opp-action-cells {off,defensive,offensive,both}` (one knob; `defensive`=C6, `offensive`=C7) |
| Producer | `DamageOperator.conditional_threat_cells()` / `.switch_branch_cells()`; the op OWNS the layout, offsets pinned against `decode_damage_block` (the `pointer_cells` convention) |
| Consumer | `pointer_cells` → `PointerNativeActionHead` (`switch_cell_dim` / `move_cell_dim` widen; a missing source NARROWS the Linear, never zero-pads) |
| Learned params | `λ` (C6); `α, β, γ, μ, ν` (C7) — `nn.Parameter` on the op beside `out_gain`, all init 0 |
| Version | STRUCTURAL string in `check_compatible` (the `win_prob_mode` pattern). **OFF byte-identical ⇒ NO `ARCH_SIGNATURE` bump.** `MODEL_CONFIG_VERSION` +1 |
| Identity-at-init | **zero-init the NEW input columns** of `switch_proj` / `move_proj` so ON == OFF bitwise at step 0. The pointer head is built AFTER SB3's ortho-init, so M1 does not clobber it — **pin that with a test** |
| Threading | `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + BOTH `extractor_kwargs` sites |
| Requires | `--damage-op`; C6 also `--damage-matrices incoming`; C7 also `--damage-outgoing` + the T edge |
| Obs | **unchanged** — no obs-dim change, no obs benchmark needed |

---

## 4. Prerequisites and risks

### 4.1 ⚠️ HARD PREREQUISITE for C7 — the unrevealed marginalisation

v34's outgoing matrix is **REVEALED-gated** (unrevealed opp slots zeroed). A revealed-gated `q_b`
reads ≈0 early — exactly when switching is most frequent — so the model concludes *"my move always
lands on their active."* Same GIGO class as the typeless-HP "immune" bug: **misleading, not merely
incomplete, in the phase where the feature matters most.**

The machinery exists — v36's expected-latent defender (`SPECIES_EXP_MULT` ⊕
`SPECIES_SPREAD_PRIOR` marginalised through the species belief) — but it rides
`--threat-refine-outgoing`, hence `--damage-refine-rounds > 0`, which the prefuse config sets to
**0**, so it is **inert in gen-3**. **Re-home the expected-latent read onto the prefuse path
before C7.** Ship C6 first; it has no such dependency.

### 4.2 Standing risks

* **Prior null:** an opponent-action AUX head was falsified on value-of-information (~0.03). This
  is a different use (a weighting inside the physics, gradient riding the damage, not a prediction
  target) — but state the null rather than ignore it.
* **One-ply only.** C6/C7 buy "answer the threat" and "punish the switch." They do NOT buy the
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
5. **Equivariance**: permute our bench and their bench → C6 cells permute with our team; the
   contracted C7 column is **invariant** to their bench permutation.
6. B=1 CPU forward delta measured and reported (expect small — no new physics).

**Gates — pre-register BOTH, do not let one stand for the other:**
* **Re-home success:** `in_permon` / `in_matrix` sub-block dependence FALLS on the trained
  checkpoint (`tmp/incoming_conditional_probe.py`, shuffle arm, vs gen-3's numbers below).
* **Strength:** anchored ELO **non-inferiority** vs gen-3 at matched tranches — fix the margin
  before the run (e.g. within −15, CI excluding −40).
* **Behavioural (new, cheap):** a **wasted-KO rate** — fraction of decisions where the policy
  picks a move with `pko_stay ≥ 0.8` on a turn the opponent switches. C7 should reduce it.

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
3. **C6** — no prerequisites.
4. **unrevealed expected-latent onto the prefuse path** — §4.1.
5. **C7** — after 4.

## See also
* `designs/learning/entity_tokens_biases_pointers.md` — the worked example, the four-ways table,
  the MIRROR section (the reasoning behind this spec)
* `designs/learning/shortcut_learning_and_feature_delivery.md` Part 6/7 — the channel argument,
  the measurements, the concat end-state decision rule
* `designs/ai_v9/design_generation_roadmap.md` §3 — Stage 2/3 and the concat decision rule
