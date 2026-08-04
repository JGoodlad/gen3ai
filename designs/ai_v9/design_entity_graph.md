# ai_v9 — The Entity Graph: every entity and edge (research design, no code)

**Status:** the INVENTORY for the active fresh generation (2026-07-21, owner + assistant
sessions; re-scoped 2026-08-03). **The operative staged plan is
`design_generation_roadmap.md`** (it supersedes the ai_v8 `next_run_plan.md` on-ramp for
generation-crossing items and records the E9 history decision); Stage 0 — the pointer-native
action head — is SHIPPED (v51 `gen3_pointer_native_v1`). The concept vocabulary is in
`designs/learning/entity_tokens_biases_pointers.md`. This doc is the inventory: what the
entities are, what the edges are, what each carries, and where every current obs signal
re-homes so nothing is lost.

**The sorting rule (governs every decision below):**
- pair-varying fact → **edge** (an activation, recomputed every forward from the live board)
- entity-invariant fact → **token** (feature on the entity)
- probabilistic fact → **distribution summary** (e.g. `[low, high, crit, pko]`)
- future-facing value (tempo, information, plans) → **attention** (learned, never tabled)

---

## 1. Entity catalog (token types)

All tokens share one width (d_model) via per-type input projections + a token-type embedding.
Counts are per battle state; "(set)" = unordered, mask-padded; "(ordered)" = order is
load-bearing.

### E1. Mon-identity (×12 — our 6 + their 6)
| Field group | Ours | Theirs |
|---|---|---|
| Species (+types) | known | revealed, else BeliefSlots unknown-mon token / species belief |
| Item | known | revealed → exact; else usage prior (CB probability feeds the CB edges) |
| Ability | known | revealed → exact; else prior (immunity/Thick-Fat expectations fold into edges) |
| Stats (5 derived) | exact | believed: nature⊕EV generative spread belief (v40) |
| HP fraction, level | exact | HP fraction public; level 100 |
| Alive / revealed / active flags | exact | exact |
| Provenance scalars | — | how-much-is-guessed (per field group) |

### E2. Mon-condition (×12; DESIGN CHOICE: separate token vs folded into E1 — see §7)
Status one-hot + counters (toxic stage, sleep-wake belief 3-dim incl. Rest determinism),
boosts (7 stats), volatiles (Substitute w/ HP, Leech Seed, confusion, Encore/Disable/Taunt/
Torment + their timers where public, Focus Energy, Perish count, Ingrain, Yawn pending,
partial-trap), protect/endure consecutive counter, choice-lock state (CB-locked move id),
trapped-by state, pending Wish (slot-keyed, the wish-wired signal).

### E3. Our-move (×4 per mon; the ACTIVE's 4 are (ordered) — request order = action slots)
Move latent (BP, type, category, accuracy, priority, effect flags, secondary chances — the
v24 `MoveLatentEncoder` content, typed-HP-aware), current/max PP, legal-now (disabled /
choice-locked / taunted / imprisoned), request-slot tag (active only). Bench mons' moves may
enter only their local bag (hierarchy) or be summarized — §7.

### E4. Their-threat-move (top-K believed per their mon; (set); K=16 active — probe-sized:
94% channel ownership; bench K smaller, §7)
Move latent, belief weight w (revealed→~1, else learnset-gated prior ⊕ learned posterior;
typed-HP scattered onto the 16 real typed ids), provenance (revealed / prior / learned),
accuracy, is-phys.

### E5. Tail-threat (×1 per their mon; the truncation insurance)
`[P(tail mass), tail worst-case phys, tail worst-case spec]` from the precomputed
per-(type,category) max-BP bound — covers the ~2-6% of max-channels owned below rank K
(bimodal-miss finding: truncation loses candidates entirely, never shaves them).

### E6. Side (×2)
Spikes layers (0-3), Reflect / Light Screen (+ remaining turns where public), Safeguard,
Mist, Sleep Clause consumed (side-level!), pending Wish summary, hazard-relevant flags
(has-spinner / has-spinblocker are derivable — attention's job, not features).

### E7. Global board (×1)
Weather (4 + none, + remaining duration where public), turn number, turns-since-progress
clock (log-saturated), prev-turn action-mask summary.

### E8. Readouts
pi-CLS and vf-CLS note-takers (attention-derived board summaries feeding the projection
heads); a decision-context vector for the pointer head (§5). FiLM conditioning (z_arch, and
the planned z_opp) stays a head-level modulation — team-level latents are side-token features
if ever needed in-graph.

### E9. History (DECIDED direction — `design_generation_roadmap.md` §4)
Options were: (a) keep the 7 TurnDelta slots as 7 global history tokens; (b) per-mon recency
features on E1/E2 (last-seen move, damage taken last turn); (c) both, then ablate.
Decision (2026-08-03): land (b) first (the sufficient-statistic view — most history belongs
compiled into entity state), then a short window of (a)-style turn tokens for the sequential
residue, with entity-LINKED event tokens as the audited end-state; recurrence RULED OUT (the
obs must stay a pure function of the event log — the forensic-stack invariant).

---

## 2. Edge catalog (computed biases + edge features)

Every edge is recomputed per forward by the already-fuzz-validated kernels (v26 physics, v27
landing, protect/wish/toxic schedules). Delivery: a small learned map cell→per-head bias
scalars, full cell available as edge features where marginals need it.

### D. Damage edges — (attacking move) → (defender mon)
Cell: `[low, high, crit, pko, type_mult]` (+ CB-conditional tail + p_cb on their side;
accuracy folded into pko; fixed-damage moves routed per the v26 rules).
Quadrants (completing the all-way spec):
- **D1** our active's 4 (ordered) × their 6 — exists (v34).
- **D2** our bench 5 × their active (neutral-boost convention, gen3 resets on switch) —
  exists (v39, the forced-switch fix).
- **D3** their active's top-K × our 6 — exists (v35).
- **D4** their bench's top-K(bench) × our 6 — **the missing quadrant** ("after I KO, what
  comes in and what does it threaten"); affordable under truncation.
- (Full our-bench × their-bench is D2/D4's closure; defer, cost/value unproven.)
Modifiers internal to the kernel (not separate edges): STAB, chart incl. typed-HP 16,
boosts, burn, screens, weather, Thick Fat / Levitate / absorb abilities (revealed exact,
else prior-expected), Explosion's halved defense.

### S. Status-landing edges — (status move) → (defender)
Cell: `[P(lands), P(major), P(immobilize)]` — v27/v37 physics: accuracy × type immunity
(incl. Leech-Seed→Grass, T-Wave→Ground, Toxic→Steel/Poison) × ability (exact/prior) ×
already-statused × Sleep Clause (side token) × Substitute × Safeguard.

### C. Consequence-delta edges (hypothetical worlds — the calculator prices the change)
- **C1** stat-move self-loop → per-opposing-mon deltas: damage table re-run at post-boost
  stats ("SD flips EQ vs Swampert 3HKO→2HKO"); Agility additionally → speed-flip list.
- **C2** status consequences: paralysis → speed-order flips per pair (+25% full-para as
  immobilize); burn → delta on their outgoing table; Toxic → the ramping HP schedule.
- **C3** recovery self-loop: which of their pko cells flip back to 0 at +50% HP.
- **C4** Protect self-loop: success odds (floored-doubling counter) + the TURN LEDGER (net
  scheduled HP: Toxic/sand/Leftovers/Leech ticks, Wish capture, blocked priority).
- **C5** Baton Pass: transfers current boosts/volatiles to a recipient — edge from the move
  to each bench candidate carrying the recipient's C1-style delta under the passed state.

### V. Speed edges — (mon) ↔ (mon), both sides
`P(outspeeds)` from believed spread/nature with uncertainty-aware std (v36), adjusted for
paralysis and speed boosts; priority brackets are a move-token feature that overrides the
pair edge at decision time (attention composes "Extreme Speed beats the speed check").

### T. Trapping edges — (trapper) → (candidate victim)
gen3-critical: Shadow Tag (Wobbuffet), Arena Trap (vs grounded), Magnet Pull (vs Steel),
partial-trap moves. Cell: `P(cannot switch)` (theirs believed via ability prior; ours exact).
Both directions — "my Dugtrio traps their weakened Blissey" is a plan-defining edge (the
`trap_core` archetype tag exists for this).

### X. Entry/exit edges — (bench mon) → (side/board)
Entry chip: Spikes × grounded (item/ability exact ours, believed theirs); Pursuit exposure
(doubled BP on the switch-out — an edge from their Pursuit-carrier to each of our mons,
priced from D-cells at doubled BP); phaze exposure (Roar/Whirlwind → expected random-drag
chip through hazards + boost reset — priced over the bench set).

### G. Global-schedule features (per-mon, end-of-turn ledger — attribute not edge)
Sand/hail chip (typing/ability-gated), Leftovers, Leech Seed transfer, burn/poison/Toxic
tick, Perish countdown, partial-trap tick, weather expiry. (These are per-entity schedules —
they live as E2/E7 features; C4 composes them for the Protect ledger.)

---

## 3. Readout heads
- **Pointer action head**: move logits read from the active's 4 E3 tokens (ordered slots =
  logits 6..9... mapped to today's action space), switch logits from our 5 bench E1 summary
  tokens, plus the non-pointer specials (struggle/default) from the context vector. Legality
  masks ride the same tokens' legal-now features.
- **pi/vf CLS** feed the (FiLM-modulated) projection heads; the privileged critic (next-run
  plan item 1) FiLMs the true-team labels onto the vf side only, scouting-safe.

## 4. Certainty & provenance conventions
Ours: count 4, w=1, ordered (action alignment is load-bearing — guard it). Theirs: top-K
set, w = posterior, unordered, rank-free; revealed moves naturally occupy top ranks (w≈1);
below-threshold candidates masked (generalizing the v30 "5th slot zeroes when all revealed").
Every believed field carries provenance so the policy can price how much is guessed.

## 5. Dims consistency
One d_model; per-type input projections; token-type embeddings; K is a version-gated
structural int per site (the `damage_topk_k` pattern); cells share one schema per edge
family; the whole layout derives from ONE declarative schema module (env packer, model
unpacker, spaces, and dimension tests all generated from it — the anti-drift decision from
the serialization discussion: no IDL/protobuf in the hot path; fixed-layout arrays + parity
harness at any future Rust boundary).

## 6. Nothing-lost audit (current 2992-dim obs → new homes)
| Current block | New home |
|---|---|
| per-mon 110-dim slots (incl. sleep-wake 3-dim) | E1 + E2 |
| per-mon move slots (11-dim × 4) | E3 (+ E4 for theirs) |
| active context boosts + volatiles (116) | E2 of the two actives |
| global env (18) | E6 + E7 |
| reactive scalars: protect odds | C4 (Protect token feature) |
| reactive: wish_floating ×2 | E2/E6 pending-Wish + C4 ledger |
| reactive: turns_since_progress | E7 |
| incoming-damage / crit-split / OHKO block (51) | D3 edges + E1 marginals |
| move-effect action-aligned block (44) | E3 features + S/C edges |
| matchups (288) | D/V edges |
| active-req-moves (12) | E3 ordered tokens (alignment by construction) |
| prev-turn action mask (11) | E7 summary |
| turn history (7×159) | E9 (open) |
| op head-concat blocks (all) | D/S/C/V/T/X edges + marginals; deleted per the
deprecation playbook (build home → mask → A/B → delete) |

## 7. Open questions
1. E2 separate vs folded into E1 (addressability vs token count).
2. Bench move tokens: full local bags (hierarchy) vs summarized (flat + biases first).
3. Bench K for E4/D4 (probe suggests smaller than 16 suffices; measure).
4. ~~E9 history representation (tokens vs recency features vs both)~~ — DECIDED, see E9 +
   `design_generation_roadmap.md` §4 (recency features → turn tokens → entity-linked event
   tokens; no recurrence).
5. Hierarchy timing: flat 12-token body + move tokens + biases may capture most value before
   local/global two-level attention is needed — sequence Form A → flat+biases → hierarchy.
6. Compute budget: token count ~12 E1 (+12 E2?) + 4 E3 + K E4 + 2 E5-ish + 2 E6 + 1 E7 —
   ~35-50 seats vs today's 14; attention cost grows ~(n/14)², FFN ~n/14 — bounded by
   dropping the op's flat blocks and the sweep truncation (25×). Needs a real estimate
   before commitment.
7. What stays attention-only (never tabled): tempo, scouting/information value, PP-war
   strategy, win-condition identification, multi-turn lines, Sleep Clause *strategy* (the
   mechanic is an S-edge input; when to spend the sleep is learned).

## 8. Validation plan (research-stage)
Reuse the probe pattern: the top-K probe template generalizes to sizing bench-K (E4/D4) and
to measuring which edge families' biases the trained attention actually uses (bias-ablation
per family = the value audit that decides D4/T/X inclusion). Physics kernels are already
validated (damage_op_probe 19/19, status fuzz, wish fuzz, protect odds); the new surface is
delivery, guarded by the schema single-source + alignment throws + the deprecation playbook.
