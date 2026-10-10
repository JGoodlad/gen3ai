# Stash contracts, and the compile spelling

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The op's and the extractor's SIDE VALUES (`gen3_op_stashes_v1` / `gen3_extractor_stashes_v1`)

## The op's SIDE VALUES have ONE container (`gen3_op_stashes_v1`)

Every per-forward stash the op exposes (`last_topk_idx`, `last_pair_cells`, `last_w_all`,
`last_out_pko`, `last_raw_block`, `last_raw_tensors`, `last_tensors`, `last_worst_rows` (`--op-reduction
principled` only), …) lives in ONE `OpStashes` dataclass that the
forward replaces at ENTRY — so no stash can carry a previous batch, uniformly (three different
clearing conventions used to coexist, and the top-K trio had none). **Reads** use the `last_*`
properties (the documented surface); **writes** go through `op.stash.<field>` — writing a
`last_*` name raises. When you add a stash: add the dataclass field with its shape comment, write
via `self.stash`, and never add a bare `self.last_x = …` attribute. The extractor's tuple
stashes are typed the same way (`PointerInputs`, `ThresholdProbs` — NamedTuples, so positional
unpacks keep working).

**PRE-gain vs POST-gain: a consumer that reads an op value AS physics reads it PRE-gain** (the X5 version break's
part 5). The op's learned `out_gain` is a projection ADAPTER — one scalar per (block region, channel) with NO
position in its key: tied across request slots and move seats (part 4) and across our team slots and their mons
(`gen3_mon_tied_gain_v1`, config v145; no lead-mon special case); `damage_op_layout.out_gain_channel_keys` is the key map,
`expanded_out_gain()` / `apply_out_gain()` the one application) — so only a PROJECTION reads the post-gain block
(`last_tensors`: the pointer cells, the `prefuse_proj` injection, `value_entity_pool`'s op rows). Any consumer
that multiplies, compares or otherwise uses an op value as a probability or a damage fraction (P(we act first), a
roll, a secondary chance) reads a pre-gain stash (`pair_cells`, `pair_in`, `out_cells`, `out_pko`) or the block's
channels through **`last_raw_tensors`** — the `OpTensors` views over the PRE-gain block, LIVE (they carry the
op's upstream gradient, like every other pre-gain stash). `last_raw_block` is the same block DETACHED, for the
prober / forensic decode only; never feed it to a training-path consumer (it would cut that route's gradient).
Readers today: `intent_conditional` (our moves' high roll, P(first), the flinch chance) and the move-resolution
family (P(first)), in both `--speed-physics` modes. Tests: `x5_version_break_part45_test.py`.

**The EXTRACTOR's side values follow the same contract (`gen3_extractor_stashes_v1`).** Every
per-forward stash `Gen3FeaturesExtractor` exposes — `last_pointer_inputs`, the α/β intent trio,
every belief publication (`last_move_belief_logits`, `last_spread_*`, `last_item_logits`,
`last_hp_type_logits`, `last_belief_logits`, `last_opp_believed_mask`, `last_opp_active_local`,
`last_move_latent_table`), `last_damage_block`, `last_value_pooled`, `last_win_prob_logits`,
the internal T0→T1/T2 hand-offs (`t0_species_probs`,
`entity_latent_table`, `thresh_probs`) and the LIVE `belief_supervision` dict — lives in ONE
`ExtractorStashes` dataclass that `forward_internal` replaces at ENTRY. Same rules: **reads** on
the `last_*` properties (every cross-module consumer — the policy's pointer head,
`instrumented_ppo`, the prober, inference — uses the typed properties, never `getattr(..., None)`);
**writes** through `fe.stash.<field>`; a stray write to a `last_*` name raises. When you add a
stash: add the dataclass field with its shape comment, write via `self.stash`, add the read-only
property if it has an external consumer — never a bare `self.last_x = …` attribute. The
publication/stop-grad semantics (`_publish_belief` / `belief_supervision()`) are unchanged — the
supervision dict rides the container, so its per-forward clear IS the entry replacement.
Boundary rule: each producer module owns its own stash surface — the op keeps `OpStashes`,
`PokemonEncoder` keeps `last_move_tokens` (written unconditionally every encoder forward, read in
the same extractor forward) — a submodule never writes into its parent's container. Related
fail-loud: `Gen3DualHeadMaskablePolicy._critic_value` under the win-prob critic (the only critic) RAISES when the
win head/logits are missing or batch-stale — there is no scalar `value_net` to fall back to (deleted with
the SB3 value tower, audit F1; falling back was the silently-wrong-critic shape v89 exposed). Gate:
`extractor_stashes_test.py`.

## The op's MAXIMA: one spelling, one tie convention (audit F6a)

Every gradient-path value-reduction max in the forward is `max_by_index` (`agents/model/index_max.py`, a
LEAF module so any kernel can import it without a cycle; `damage_op` re-exports it): the value is `amax`'s bit
for bit, gathered at `x.detach().argmax(dim)`, and on an EXACT tie the whole gradient goes to the FIRST
maximum (a declared convention; `amax` splits it, and an Inductor-recomputed `x == amax` operand once matched
nothing and produced 0/0 — the 2026-10-05 compiled-gradient NaN). It covers the op's ten incoming channel
maxima in every configuration (belief on or off — there is no `amax` branch), the pairwise kernels, the
status-landing maxima, the E5 tail's worst-phys/spec and `pair_reduce`'s inert deepsets pool. A NEW max over
candidates on a gradient path uses it. `amax` / `amin` are legal only OFF any gradient path — the provenance
gate operand (a comparison), a TABLE lookup (cure / cleric), a constant (the cheapest-undo minimum), an
observation indicator (`we_have_pur`), a `no_grad` bracket (`fixed_size_tau`) or a diagnostic (the move-tie
gaps). `selection_sites` declares the one `MAX_VALUE` EXACT site at `index_max`.

## The op's reductions over THEIR believed moves have ONE switch (`--op-reduction`, audit F6b)

`gen3_op_reduction_principled_v1` (config v146; ARCHITECTURE §4 states the facts). `agents/model/op_reduction.py` is
the one home of the principled forms — `presence_alpha` (α = presence / the attacker's TOTAL presence over the whole
move axis), `expectation`, `noisy_or` (`1 − Π(1 − p)`, a product, exact and finite at p = 1),
`believed_reduce(wv, w_total)` (the per-attacker kernels' ONE reduction: `w_total` None → `max_by_index(wv)`, the
legacy expression bit for bit; given → `Σ wv / w_total`) and `incoming_principled` (the incoming per-mon row). Four
rules:

* **A NEW reduction over an opponent's believed candidates goes through `believed_reduce` (or `incoming_principled`'s
  pattern), never a bare `max_by_index`** — so both modes reach it. The attacker's total comes from
  `DamageOperator._attacker_total` (None under `max`).
* **A reduction over OUR moves stays a max** (we choose: C1's `d_best_*`, D2's `best_*`, the outgoing status
  landing). `op_reduction_extractor_test.py` fails if one of those moves with the mode.
* **`max` must stay byte-identical, and dynamo names graph nodes after LOCAL VARIABLES.** A new local on the `max`
  path (even `tw = torch.where(...)` then `tw * x` for the same ops) renames nodes and changes the production graph's
  hash; keep the `max` expression verbatim and put the `principled` arithmetic in its own branch or helper.
* **The noisy-OR row never enters the flat block** (`op.stash.worst_rows`, read by the extractor's zero-init
  `op_worst_proj`): the block's layout, `out_gain` and every slicer are mode-independent. Its one discrete op (`high
  > 0`, "this candidate damages this mon") is a declared MARGIN threshold (`selection_sites`, `zero_exact`).

## The op's READ contracts: an opponent's ability, and the status rules (`gen3_op_ability_status_gigo_v1`)

Two input contracts every op kernel follows (2026-10-07; ARCHITECTURE §4 states the rules themselves):

* **An opponent's ability is read through ONE view.** `extractor_ctx.ability_known` / `revealed_ability1_ids`
  (the op's `opp_ability_view`, `opp_ability_damage_mult`, `opp_status_block`, `opp_secondary_block`,
  `opp_secondary_mult`, and `_known_or_prior` for a table of its own) — never `ctx.ability1_ids[...]` on the
  opponent's side and never `id > 0` as "revealed": an unrevealed slot's id1 is its species' top-1 PRIOR with
  `known = 0`. The unrevealed branch is a Smogon species marginal (`damage_tables.build_species_ability_marginal`),
  never the pool. Gate: `ability_known_gate_test.py` (AST; fails any forward-module `ability1_ids` subscript that
  is not our side; allowlist EMPTY).
* **Status landing has ONE rule per direction.** Outgoing: `_outgoing_status_land` (`_status_landing` reads its
  active slot, `discrete_outgoing_status` all six). Incoming: `_incoming_dedicated_land` + `_incoming_status_mask`
  (`_incoming_status_lands`, `discrete_incoming_status`, and `pair_outcome_coords`' identity split all read them),
  the side / clause mask itself being `status_rules.incoming_status_mask`, which the move-resolution family shares.
  A new status rule goes there, not into a consumer. Tests: `op_status_rules_test.py`.
* **Move ORDER has ONE rule** (`gen3_speed_physics_v1`, v143, audit F7b): `move_order.py` — the priority bracket
  (`p_seat_first`, read by the move-resolution family and `intent_conditional`) and, under `--speed-physics on`,
  the within-bracket rule (`p_first_same_priority`: their speed as the species' DISCRETE Smogon spreads mixture
  over the Speed stat — `SPEED_MIX`, gen3_speed_mixture_v1 — each support point through the same exact stage /
  paralysis arithmetic as our exact speed, the coin-flip tie, Quick Claw format-gated by `quick_claw_live`). Every
  op site that prices who moves first keeps its `off` code verbatim and, under `on`, builds its inputs in
  `damage_op_speed.py` (`_our_speeds_exact`, `_opp_speed_mix`, `_our_quick_claw` / `_opp_quick_claw`, `_p_first*`;
  an X5 OTHER slot reads the tail's mixture, `P_tail @ SPEED_MIX`) — a site's
  hypothetical (C1's post-setup stage, C5's inherited stages, a paralysis it prices) is an override ARGUMENT,
  never a second formula. Two `on`-only stashes: `item_qc_prob` (the item belief's P(Quick Claw); never set while
  the format bans it) and `speed_fast_pair` (the forward's P(first) and P(first | this mon paralysed), which
  `pair_outcome_coords` reads instead of re-deriving). The order rule reads the PRE-gain P(first) in both modes
  (`last_raw_tensors`, below).
  A Python closure inside an op method changes `off`'s dynamo graph (its captured locals become cell variables
  and are renamed) — put an `on`-only helper on `damage_op_speed`, never nested in a site. Tests:
  `move_order_test.py`, `speed_physics_extractor_test.py`, `speed_physics_bridge_integration_test.py`.

## Why the species posterior is spelled `log_softmax(...).exp()` — the whole diagnosis

`torch.softmax` over the last dim of the `[B,6,n_species]` logits lowers to a numerator buffer plus a
`[B,6,1]` denominator, and the Inductor **CPU** scheduler then trips `AssertionError: buf<N>` trying to
fuse the division. That single op was the reason the opponent compile (the trainer-side flag, deleted in U3) used to set
`torch._dynamo.config.suppress_errors = True`, which in turn meant the production config compiled only
partially (3.6× instead of 6.53×) and every other backend failure in the process went silent.

`tmp/softmax_variant_probe.py` measured the alternatives: `.contiguous()`, `.clone()`, a 2-D
reshape and a hand-rolled `exp / sum` **all still fail**; only the `log_softmax().exp()` factoring
lowers cleanly. It is mathematically identical and keeps the same max-subtraction stability (measured
max|Δ| vs eager 5.07e-07). Guarded by `extractor_compiles_test.py`, which owns the whole compile
matrix: the fast tests pin the math, and the compile cells run a real compile of the literal
production arch with suppression OFF (verified to fail if the old spelling returns) across
CPU/CUDA x forward/backward — `GEN3AI_SKIP_COMPILE_TESTS=1` opts out, `GEN3AI_TEST_ALLOW_GPU=1` is
needed for the CUDA cells (the root conftest hides the GPU from the suite). Repro:
`tmp/inductor_crash_repro.py`. Note the CPU **backward** does NOT lower — an `atomic_add` scatter
the C++ backend refuses — which is why the compiled-opponent artifact is inference-only.

**It is NOT "CPU cannot accumulate", and the distinction is the actionable part.** Inductor's C++
backend has THREE store kernels and two of them implement the mode: `CppKernel.store` emits
`atomic_add(&buf[i], v)`, `CppVecKernel.store` emits `atomic_add_vec<...>(...)`, and only
**`CppTile2DKernel.store`** carries the bare `assert mode is None`. `CppTile2DKernel` is the
2D-tiled/TRANSPOSED variant, chosen when the store's index pattern needs a transpose. So the
refusal is one missing case in one kernel variant, selected by memory LAYOUT — if a CPU training
compile ever mattered, the lever is to reshape the scatter so Inductor picks `CppVecKernel`, not to
wait upstream. Triton's `store()` has the case unconditionally (`tl.atomic_add(..., sem='relaxed')`),
which is why CUDA — the only compiled backward we actually run — is unaffected.

The op is an accumulate-scatter because it IS a backward: the gradient of a gather/index-select is
a scatter-ADD (indices may repeat, so writes must accumulate). Cut the gradient and it disappears.
**So the refusal is CONFIG-CONDITIONAL, and the pin now says so**: `--belief-grad-mode label_only`
publishes every belief output stop-grad, which deletes those backwards and lets the CPU backward
compile cleanly. Bisected 2026-08-15 — `shaping` REFUSED, `label_only` COMPILED, `win_prob_mode`
irrelevant. ⚠️ **Which gather is NOT pinned**: detaching the obvious candidate
(`damage_op.py`'s `w_all.gather(-1, topk_idx)`, whose shape matches the reported buffer exactly)
left the refusal in place, so there are several sites and the shape match was a coincidence. The
limitation test therefore builds at `belief_grad_mode="shaping"` explicitly, because production
moved to `label_only` at gen-11 and the pin would otherwise have gone green while testing nothing;
that is
pinned as a limitation test that fails if it ever lifts.

---

## Moved from the leaf (2026-10-10)

The full text of the model leaf's sections on this topic, moved here when `src/agents/model/CLAUDE.md` was cut to rules, commands, map and hazards (the leaf keeps a one-line pointer to each). Always-current like the rest of this doc; where it overlaps an earlier section, the earlier section is the fuller statement.

### The op's flat layout has ONE slicer (`gen3_op_tensors_views_v1`)

`DamageOperator.tensors_from_block()` is the only place the flat block's offsets are walked; it
returns **`OpTensors`** — named zero-copy views (`incoming_rows`, the CB tail, the outgoing/status
groups, the opaque matrix renders). Every same-forward consumer reads a field off
`damage_op.last_tensors` (prefuse injection, the assembler's `seed_rows`) or goes through
`pointer_cells` (which itself now assembles from the views); **never re-derive an offset at a
consumer** — the layout walk raises if a region is added to the block without a view. The flat
block remains the serialization: `decode_damage_block` (the prober's human-readable mirror) and
`last_raw_block` still read it, and dropping it from the forward is `design_op_tensors.md` step 3
(retrain-class — it shrinks `out_gain`). Landed as a byte-identical refactor under the proof
bundle recorded in `designs/research_state/claude_md_archive/model_leaf_history.md`, on 64 real
gen-9 eval states across three config arms.

🚨 **`last_tensors` is POST-gain and is for PROJECTIONS only.** The op's learned `out_gain` (one scalar per
(region, channel) with NO position in its key — tied across request slots / move seats, the X5 version break's part 4,
and across our team slots / their mons, `gen3_mon_tied_gain_v1`; `damage_op_layout.out_gain_channel_keys`) is a
projection adapter, not physics. A consumer that uses an op value AS a probability or a
damage fraction (P(first), a roll, a secondary chance) reads it PRE-gain: a typed pre-gain stash, or the block's
channels through **`last_raw_tensors`** (live views; `last_raw_block` is the detached prober copy — never a
training-path input). `intent_conditional` and the move-resolution family do (part 5;
`x5_version_break_part45_test.py`). A new region of the block needs a key in `out_gain_channel_keys` (the op
raises at build if the key walk and `out_dim` disagree), and its key names the channel only — a request slot, seat,
team slot or listing position is arbitrary and is never part of it (`mon_tied_gain_test.py` permutes team slots).

**The op's SIDE VALUES and the EXTRACTOR's follow one contract** (`gen3_op_stashes_v1` /
`gen3_extractor_stashes_v1`): every per-forward stash lives in ONE dataclass the forward replaces at
ENTRY, **reads** go through the `last_*` properties and **writes** through `op.stash.<field>` /
`fe.stash.<field>` — writing a `last_*` name raises. Add the dataclass field with its shape comment,
never a bare `self.last_x = …`; each producer owns its own stash surface and a submodule never writes
into its parent's. Gate: `extractor_stashes_test.py`. Detail:
[`designs/model/op_contracts.md`](op_contracts.md).

🚨 **THAT CONTRACT IS SINGLE-THREADED, AND `forward` IS NOT RE-ENTRANT.**
"The forward replaces the stash at ENTRY" makes a stale read unrepresentable only while ONE forward
is in flight per extractor — true of training (each env worker is its own process) and false the
moment two threads share a model object, which `main.search_dividend`'s mirror did by design (deleted
in P6 slice 6d-1, 2026-10-08). Measured 2026-09-22: two threads, one real extractor, 2,400 interleaved
forwards ⇒ **1,063 failures in seven classes**, including `ValueThreatInject shape mismatch: tokens (1, 6)
vs rows (9, 6)`. ⚠️ **The crash is the lucky case** — same-batch-size forwards corrupt each other
silently. **There is NO guard any more** (`forward_guard.py`, the opt-in lock, had no caller after that
deletion and was removed in the 2026-10-08 cleanup bundle): a thread-sharing caller gives each thread
its OWN extractor (a deepcopy of the policy) or serializes the whole forward + its `last_*` reads itself.

### 🚨 Our active's moves live in TWO orders — cross them by IDENTITY only (`gen3_move_legality_by_id_v1`)

The per-mon move slots (`ctx.all_move_ids`, every encoder) are SORTED BY `Move.id` STRING; the request
block (`ctx.our_active_req_move_*`) and actions 6–9 are in REQUEST order. **Never apply a request-order
tensor to the per-mon slots by position, or a per-mon tensor to the request slots** — go through
`extractor_ctx.active_request_sorted_match` (or `active_move_legality_sorted`), the ONE rule the pointer
head, `PokemonEncoder` and the static encoder share. The positional form fired twice (the old prev-turn
mask; then `PokemonEncoder`'s legality from `bcdd868b` to this fix, wrong on 6.8 % of real move-bearing
decisions) and is silent whenever every move is legal. `move_legality_alignment_test.py` fails on it;
`agents/action/ordering_integrity.check_obs_move_order` RAISES on a served row that breaks the rule's
preconditions.

### 🚨 An OPPONENT's ability is revealed only by its `known` flag (`gen3_op_ability_known_v1`)

An unrevealed opponent's ability slot carries its species' TOP-1 Smogon-prior ability in `id1` with `known = 0`.
The op read `ability1_ids > 0` as "revealed" until 2026-10-07 and so asserted the guess as CERTAIN in every
damage / status / secondary kernel (Toxic "never landed" on an unrevealed Snorlax). **Read an opponent's ability
through `extractor_ctx.ability_known` / `revealed_ability1_ids` (the op's `opp_ability_view` + `_known_or_prior`)
and take the Smogon species marginal when it is unknown.** Status landing has ONE rule per direction
(`_outgoing_status_land`; `_incoming_dedicated_land` + `status_rules.incoming_status_mask`). Gates:
`ability_known_gate_test.py` (AST, allowlist EMPTY), `op_status_rules_test.py`; detail:
[`designs/model/op_contracts.md`](op_contracts.md).

### 🚨 Every DISCRETE op in the forward is DECLARED (`selection_sites.py`, `gen3_behaviour_tie_exclusion_v1`)

The forward is piecewise-discontinuous: a `topk` / `argmax` / comparison whose operands sit within a
rounding error of the cutoff resolves differently in T2's compiled forward and the learner's eager one,
and log π jumps on that row. K9(b) EXCLUDES those rows by their TIE MARGIN and judges the rest
deterministically (`designs/training/learner_gates.md`), so it must know every such op. **A new
selection call, value-position comparison or float → int cast in a `FORWARD_MODULES` module must be
declared in `selection_sites.py`** — a `MARGIN` rule when its operand is a score (it moves with the
weights, or is float arithmetic a rounding error can push across the cutoff), an `EXACT` reason
otherwise (observation / table / integer / gathered at a declared selection). Three EXACT reasons cover a SCORE
operand whose flip cannot reach log π because the value is continuous across it — `MAX_VALUE` (an argmax read only
through a gather of its own operand), `BISECT` (a fixed-step bisection's direction test) and `COUNT_CONTINUOUS`
(`--ko-ramp exact_closed`'s two run counts: the closed-form sum does not jump at a count's step) — and each is exempt
from the jitter check below, its own test pinning the continuity. `selection_sites_test.py`
fails on an undeclared or stale entry, on a line mixing the two classes (the recorder resolves ops by
LINE — split it), and when a site declared EXACT moves under a few-ulp weight jitter; at run time an
undeclared op on a float operand is a typed FATAL at the first update. The keys are source text, so
editing a declared line means re-declaring it. 🚨 **An `argmax` MARGIN rule may declare its `payload`**
(`gen3_behaviour_tie_identity_v1`: the frame-local tensors its index gathers — a tie between candidates
whose payloads are bit-identical is then no tie). The payload must be EVERYTHING the index reaches:
`selection_sites_test` fails when the index is read anywhere but a `gather` of a declared payload, so a
new consumer of a payload site's index means extending the declaration (or dropping it). 🚨 **Every
`hypothesis_set.stable_order` caller declares how it READS the order** (`consumed`,
`gen3_behaviour_tie_consumed_v1`): a `SetCuts` (a set before each cut: the op's per-mon move orders), a long
count (in order up to it: the species order), or None (every pair: the move group). A new reader of an order
declared `SetCuts` that reads its positions IN ORDER makes the declaration unsound —
`tie_identity_integration_test` permutes every set prefix and requires log π bit-identical.
### ⚠️ One op's SPELLING is load-bearing for `torch.compile` (`gen3_species_posterior_spelling_v1`)

`BeliefHead.species_posterior` computes `P(species)` for the expected-latent defender. It is written
as **`log_softmax(...).exp()`, not `torch.softmax(...)`, and that is deliberate** — do not
"simplify" it.
`extractor_compiles_test.py` owns the compile matrix and pins the spelling (a real compile of the
production arch with suppression OFF; `GEN3AI_SKIP_COMPILE_TESTS=1` opts out, `GEN3AI_TEST_ALLOW_GPU=1`
for the CUDA cells, only under a GPU lease). The Inductor diagnosis:
[`designs/model/op_contracts.md`](op_contracts.md).

🚨 **A second load-bearing spelling: EVERY gradient-path value-reduction max is `max_by_index`, never `amax`**
(`index_max.max_by_index`, a LEAF module; `damage_op` re-exports it; architecture audit F6a, after
`gen3_fm_index_max_v1` / F-XC-4). `amax`'s backward divides by `Σ(x == amax)`; when Inductor RECOMPUTES
`x` in the backward kernel and Triton's FMA contraction rounds it differently from the forward kernel,
no element equals the saved max and the compiled gradient is 0/0 = NaN (CUDA, 41 parameters, 2026-10-05).
`max_by_index(x, dim, keepdim)` gathers at the detached argmax: the same value bit for bit, a scatter
backward, and on an exact tie the WHOLE gradient goes to the FIRST maximum (the declared convention). It covers
the op's ten incoming channel maxima in EVERY configuration (belief on or off), the pairwise kernels, the
status-landing maxima, the E5 tail's worst-phys/spec and `pair_reduce`'s inert pool; a NEW max over candidates on
a gradient path uses it. `amax` / `amin` are legal only OFF any gradient path (a comparison operand, a table
lookup, a constant, an observation indicator, a `no_grad` bracket, a diagnostic) — the list:
[`designs/model/op_contracts.md`](op_contracts.md) "The op's MAXIMA". Its argmax is
K9(b)'s one `MAX_VALUE` EXACT site (`selection_sites`; the index may only gather its own operand —
`selection_sites_test` pins it). Detail: `designs/training/compile_flags.md`.
**A max over THEIR believed candidates is a `--op-reduction` site** (`gen3_op_reduction_principled_v1`, audit F6b):
spell it `op_reduction.believed_reduce(w · v, w_total)` (`max` → this same `max_by_index`, bit for bit;
`principled` → the α-weighted expectation), never a bare `max_by_index`, or the `principled` arm silently keeps
your max. A max over OUR moves stays a max. Keep the `max` expression VERBATIM — dynamo names graph nodes after
local variables, so a renamed temporary changes the production graph hash
([`designs/model/op_contracts.md`](op_contracts.md) "ONE switch").

🚨 **A third: every FLOAT attention bias reaches SDPA through `dense_attn_bias`** (`gen3_dense_attn_bias_v1`,
F-ST-8). Under Inductor, a bias built by in-place slice writes (`EdgeBias._write_block`'s head-innermost
`m.permute(0, 3, 1, 2)`) is a FLEXIBLE buffer, and Inductor may lay it out head-innermost. Inductor's own SDPA
stride constraint passes it unfrozen whenever the key count is a multiple of 8. CUDA's efficient kernel then
raises "(*bias): last dimension must be contiguous". `--token-encoding static` (64 keys) died on this in the T2
service's first graph build; legacy (62 keys) took the padded-copy branch and never did. `dense_attn_bias` pins the
bias row-major (Inductor's `inductor_force_stride_order`; its identity backward is registered at import) and is
`.contiguous()` in eager. A NEW SDPA call with a float mask goes through it. `dense_attn_bias_test.py` fails when a
call site drops it. Its CUDA test reproduces the failure without the pin. A CPU compile cannot reproduce it,
because the constraint's CPU branch always requires the stride order.

🚨 **A forward reads PLAIN INTS precomputed in `__init__`, never a module attribute holding a sub-dict of `layout`**
(`gen3_static_layout_ints_v1`, F-ST-9). `ObsUnpack.layout` reaches the same object, so dynamo installs an
object-aliasing guard. On the `static` arm's CUDA launch that guard recompiled the learner's region after the
compile lock (`[CompileSentinel] FATAL` at update 1), although the two objects stayed identical.
`static_tokens_test.py` fails when the static encoder holds a layout container.

**The general lesson:** a backend that "can't compile our model" was one op, not a property of the
architecture. Before reaching for a global suppression flag, bisect to the op — see
`designs/training/compile_flags.md` → Compiled CPU opponents.
