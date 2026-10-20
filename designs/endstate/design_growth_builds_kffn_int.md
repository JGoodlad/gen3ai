# B-KFFN and B-INT: build specs for the two remaining growth levers

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A build or decision that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-20): B-KFFN BUILT (T66: `agents/model/trunk_kffn.py`, config v161) together with the §3 field-generic
refactor; B-INT is still a SPEC (T67).** These are the build specs for **B-KFFN** (per-token-kind feed-forward
experts, the lever of X48) and **B-INT** (the opponent-intent objective shaping the trunk, the lever of X49). Where the
build differs from the text below, the text was updated in place and the Decision record says so (F-GB-3 was wrong;
Q-K3's ruling is built). The
parent design, with the hypotheses, A/Bs, costs in GPU-h and ranking, is
[`design_bot_ideas_screen.md`](design_bot_ideas_screen.md) §2.2, §2.3 and §4. Both builds are filed in
[`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md): **T66 (B-KFFN)** and **T67 (B-INT)**. Each spec is written so that
one agent can build it in one pass. Each one extends B-GROW as built (T64, `agents/model/growth.py`,
`main/train/grow_fork.py`, [`../model/versioning.md`](../model/versioning.md) "Declared growth at a fork").

**Measurement basis.** Every figure here was MEASURED on 2026-10-19 on the CPU, fp32, eager, with one torch thread
unless a row says otherwise. The rows are the 64 real rows of the compile-parity fixture (`compile_parity_obs.npz`).
Two builds were used: a FRESH production build (`main.fresh_checkpoint.build_fresh_model(0)`, HEAD `b79b66bb`) and the
TRAINED production baseline `rb_es_end2_s2005/final_model.zip`, read-only. The box was shared with other agents' capped
jobs, so the wall-clock rows are medians of interleaved repetitions and carry that noise. The scripts were throwaway
and are not committed (FINDING F-GB-9).

---

## 0. One screen

- **B-KFFN** (opus-high, **≈ 1.5 agent-days**). Every trunk round gets FIVE feed-forward experts in place of its one
  shared FFN:
  - **Pokémon:** the 12 team tokens and OTHER_species;
  - **board:** 3 tokens;
  - **our moves:** 4 tokens (E3);
  - **threats:** 12 tokens (E4 + the E5 tail);
  - **events:** 32 tokens.

  The round's existing `linear1` / `linear2` STAY and become the Pokémon expert, so the parent's state dict loads
  unchanged. The four new experts are what is added. Tokens are routed by fixed position SEGMENTS computed at build
  time from the seat layout, never by token-type id (§1.2).
  - A growth fork copies the shared FFN into each new expert. **Splitting the FFN is bit-identical to the shared FFN at
    the gate's 64-row batch** (MEASURED: all three rounds, on the fresh and the trained weights, at one thread; a
    synthetic FFN at 1, 2, 3 and 6 threads).
  - Cost: +791,040 parameters (policy 2,609,183 → 3,400,223, +30.3 %) and zero extra FLOPs.
  - It adds a field and a config version, and no `ARCH_SIGNATURE` bump.
- **B-INT** (opus-medium, **≈ 0.5 agent-day**). `--grow opp_intent_grad_mode=shaping` removes the two `.detach()`
  calls on the flat intent head's INPUTS. The intent cross-entropy, at its recorded `opp_intent_coef` (production
  0.05), then reaches the trunk.
  - **That is the ONLY new trunk gradient.** MEASURED on the trained baseline: the policy loss's trunk gradient is
    bit-identical under the two modes (difference norm 0.0), because move resolution reads α / β as stop-grad
    publications. So X49 is one lever.
  - The forward is bit-identical (measured). The fork-time gate adds a GRADIENT check: the parent's intent loss reaches
    no trunk parameter, and the child's reaches them.
  - The flag must go back to the CLI tier. The "≈ 0.25 day" in the screen doc assumed that was optional, and it is
    REQUIRED (F-GB-1).
- **A shared prerequisite**, ≈ 0.25 agent-day, done by whichever build lands first. `growth.apply_growth` and
  `grow_fork.grow_seeded_pool` are hard-coded to `trunk_layers`, and a second field would raise `KeyError`
  (F-GB-2). Each `GROWABLE` row gets its own reader, grower and gate hooks (§3).
- **Order.** Build B-KFFN first: X48 ranks above X49, and B-KFFN does the refactor. B-INT follows. The two touch the
  same files, so build them in SERIES, never in parallel.

---

## 1. B-KFFN: per-token-kind feed-forward experts

### 1.1 What the network becomes

The trunk has three rounds: two post-LN `BiasedEncoderLayer`s, then production's one `IdentityInitRound`
(`trunk_layers` 3). Each round's FFN is today

    ffn(h) = linear2(relu(linear1(h)))        linear1: 128 → 256, linear2: 256 → 128   (65,920 parameters)

and it is applied to all 64 tokens. With `trunk_ffn_by_kind = "on"`, the same position in the round applies
`ffn_k(h[:, segment])` for each segment, where `k` is the segment's kind. The outputs are concatenated back in
position order. Nothing else in the round changes: the attention, the norms, the residuals, the shared edge bias and
the pre-LN versus post-LN form.

| kind | expert | tokens (production layout, MEASURED from the forward) |
|---|---|---|
| `mon` | the round's existing `linear1` / `linear2` (state-dict keys unchanged) | positions 0–11 (our 6 and their 6), and **position 31** (OTHER_species) |
| `board` | `kind_ffn.board.{linear1,linear2}` | 12–14 (OUR SIDE, THEIR SIDE, FIELD) |
| `move` | `kind_ffn.move.*` | 15–18 (E3, our active's 4 move seats) |
| `threat` | `kind_ffn.threat.*` | 19–30 (E4's K = 6 believed-move seats, then the E5 tail's 6 OTHER_move seats) |
| `event` | `kind_ffn.event.*` | 32–63 (the 32 event seats) |

So the **segments** are six contiguous runs: `[(0,12,mon), (12,15,board), (15,19,move), (19,31,threat), (31,32,mon),
(32,64,event)]`. OTHER_species is not adjacent to the team tokens: it sits after the entity seats and before the event
seats (`extractor_forward.py`, "OTHER_species joins the trunk as ONE extra seat") (F-GB-4).

**Why five kinds, and why the Pokémon expert is the shared one.** This is the screen doc's partition (§2.2):
- our and their Pokémon share an expert, which keeps the side-symmetric design;
- OTHER_species is a Pokémon (it is typed THEIR_TEAM);
- the E5 tail holds MOVE tokens (the opponent active's moves beyond the seats), so it goes with the threats.

Keeping `linear1` / `linear2` as the Pokémon expert means the parent's keys load STRICTLY with no renaming. Only the
`kind_ffn.*` keys are new. Whether the 3-token board kind should have its own expert is open question Q-K1.

### 1.2 Routing: by seat role, computed at build time, never by token type

- **The segment table** is a tuple of `(start, end, kind_index)` plain ints. It is computed ONCE, in
  `extractor_build`, after every seat module exists, by one function, `ffn_kind_segments(fe)`. That function reads
  `team_transformer._total_tokens`, `entity_seats.n_seats` (4 + K + the E5 tail), whether the OTHER_species seat is
  built (the X5 hypothesis set), and the event seat count. **No literal.**
- The table is then set on every round, `round.ffn_segments`. The round's forward reads plain ints that were
  precomputed in `__init__` or at build, which is the compile spelling `src/agents/model/CLAUDE.md` requires.
- **Never route by token-type id.** The table is read off the seat MODULES that write each position, so it cannot
  drift from them. (CORRECTED 2026-10-20, F-GB-3: the E5 tail seats are typed `TOKEN_TYPE_THEIR_THREAT`, like E4 —
  `EntityMoveSeats.seat_types`; the event seats' "E5 precedent" means only "no token-type table growth" — so on today's
  layout a type-id router would agree with the seat-role table. The rule stands because a type id is not a role: the
  OTHER_species seat shares THEIR_TEAM with the opponent's six, and a future seat may reuse an id. K-2 checks the table
  against the ids the real forward hands the trunk AND against the seat counts.)
- **The round asserts** that its input's token count equals the table's last `end`. A layout change that forgets the
  table then raises at the first forward instead of mis-routing. This is a Python int compare on a static shape, so it
  is free under compile.
- **REQUIRES all five kinds present** (`requires=` in the registry, enforced by `flag_requires_test.py`):
  - `token_encoding == "static"` (the 3 board tokens);
  - `entity_topk_seats > 0` and `entity_tail_seats` (move and threat seats);
  - `history_events` (event seats);
  - the X5 OTHER_species seat (`opp_intent`, the hypothesis set).

  Production satisfies all of them. The toy and the legacy encoding are REFUSED with a typed error that names the
  missing kind. The screen doc rejects experts on the toy as a different lever (§2.2), so the code does not offer it.

### 1.3 Modules and init

- **Where the experts live.** INSIDE the round classes (`BiasedEncoderLayer`, `IdentityInitRound`), as
  `self.kind_ffn: Optional[nn.ModuleDict]` (`None` when off), plus one method `round.ffn(h)` that both classes call.
  `ffn` is the shared path when off and the segment path when on. Keeping the experts inside the round is what makes
  bf16 automatic (§1.6) and keeps the forward signature `forward(x, bias)` that the bf16 wrapper calls.
- **The off path builds nothing.** `kind_ffn = None`, `ffn_segments = None`, and `ffn(h)` is the literal current
  expression. State-dict keys, bytes and every initial parameter are byte-identical to today (§1.8 test K-1).
- **A FRESH on build** constructs the experts as `IsolatedLinear`s (SB3's orthogonal re-init skips them) inside
  `torch.random.fork_rng` from a new private seed `KFFN_INIT_SEED` (`arch_constants.py`). So **every other parameter's
  initial bytes equal the off build's**, the `TRUNK_EXTRA_INIT_SEED` precedent. Inits:
  - for a `BiasedEncoderLayer`, the init SB3 gives the shared Linears (orthogonal, gain √2, bias 0);
  - for an `IdentityInitRound`, `linear1` as its shared one and `linear2` ZERO (weight and bias), so a fresh identity
    round is still the identity.
- **Extra rounds on a kffn network.** `trunk_depth.build_extra_rounds(trunk_layers, ffn_by_kind=…)` builds the experts
  too, so a later `--grow trunk_layers=4` on a kffn parent appends identity rounds that carry zero experts.
- **Experts are not shared across rounds.** Each round's experts are its own: 4 kinds × 65,920 × 3 rounds.

### 1.4 The grow path: `--grow trunk_ffn_by_kind=on`

The `GROWABLE` row becomes `built=True`. `fork_arch.DECLARED_GROWTH` then extends by itself, since it reads the built
rows. The row's hooks (§3):

- **parse / validate:** the child value is `"on"`. `off → on` is the only legal step. `on → off` is refused, because
  removing trained experts is not function-preserving.
- **current value:** `policy.features_extractor.trunk_ffn_by_kind`.
- **grow(policy):**
  1. compute `ffn_kind_segments(fe)`;
  2. for each round, build its `kind_ffn` with the SAME constructor a fresh child uses, so the keys and order match a
     fresh build;
  3. **copy** the round's `linear1` / `linear2` weight and bias into each new expert (`copy_`, under `no_grad`);
  4. set `ffn_segments`, and set the field on the extractor, `features_extractor_kwargs` and the algorithm's
     `policy_kwargs`, as `grow_trunk_layers` does.

  It returns the new `(name, parameter)` pairs in module order.
- **The optimizer (AS BUILT, ruling Q-K3).** Every parent parameter's AdamW state is carried (B-GROW's rule), and each
  new expert parameter gets an independent CLONE of its shared source's state (`exp_avg`, `exp_avg_sq`, `step`): the
  grower returns the copy map, `regrow_optimizer` clones, and the report counts them as `copied`. So all five experts
  start as ONE optimizer trajectory and diverge only through their gradients (no step-1 jolt on 30 % of the
  parameters). The stamp's `optimizer` string carries the row's addendum. B-GROW's identity rounds copy nothing and still
  start fresh. (The spec's first text — new experts with NO state — was Q-K3's rejected alternative.)
- **The fork-time gate.** It is B-GROW's: fresh CPU builds, fp32, eager, the 64 fixture rows, **bit-identical logits
  and values**, otherwise `GrowthError` → `FATAL_CONFIG`.
  - The bar is achievable, as MEASURED: the split FFN equals the shared FFN bit for bit in every round, on the fresh
    and trained weights at one thread, and for a synthetic FFN at 1, 2, 3 and 6 threads, at batch 64.
  - **It is NOT bit-identical at batch 1–2 (and at 7 under 2 or 6 threads).** The CPU GEMM picks a different kernel
    for small M, so the gate MUST run the 64 rows as ONE batch (F-GB-5).
  - The bar holds on this box's MKL. Another CPU is **UNVERIFIED**, and a failure there is a typed refusal, not a
    tolerance (Q-K2).
- **The seeded pool** grows the same way, through `grow_seeded_pool`, after the §3 refactor makes it field-generic.
  Every opponent snapshot gets copied experts and passes the same gate, and the pool's arch record becomes the
  child's.
- **Several fields at one fork** (for example `trunk_layers=4` together with `trunk_ffn_by_kind=on`) are applied in
  `GROWABLE`'s declaration order: depth first, so the new rounds' zero FFNs are what get copied. X48 uses one field.
  The order is tested.

### 1.5 Versioning and the strict-loader record

- **`ModelVersion.trunk_ffn_by_kind: str = "off"`** (values `off` / `on`). `MODEL_CONFIG_VERSION` 161 (as built: v160 was
  FP32_ALT's `tau_solve`). Add one `_migrate_config` block, `data.setdefault("trunk_ffn_by_kind",
  "off")`. Add a `check_compatible` clause whose mismatch text names `--grow trunk_ffn_by_kind=on`, like the
  `trunk_layers` clause.
- **No `ARCH_SIGNATURE` bump.** The off path builds nothing, so every existing state dict and forward is unchanged, and
  the on path is gated by the field. This is the `trunk_layers` precedent (v150, three structural fields, no
  signature bump). The CHANGELOG entry states this. No `OBS_SEMANTICS_VERSION` change and no `MIGRATION_FLOOR`
  change.
- **The registry row:** `ModelFlag("trunk_ffn_by_kind", "off", Tier.CLI, Klass.STRUCTURAL, <version>, requires=…)`
  and the five surfaces of [`../model/flag_registry_rules.md`](../model/flag_registry_rules.md):
  - argparse `--trunk-ffn-by-kind {off,on}` with default `None`;
  - the `_resolve` line;
  - `ARCH_ARG_KEYS` (generated);
  - `current_model_version` (generated);
  - the field.

  **It must be the CLI tier, not CONFIG_ONLY.** A CONFIG_ONLY field is FROZEN at its registry default through
  `extractor_arch.FROZEN_ARCH_KWARGS`, so a grown child would BUILD `off` whatever `--grow` set on the namespace, and
  its restart would mismatch (F-GB-1). On a fork, a typed `--trunk-ffn-by-kind` beside `--grow` is refused by
  `grow_fork.apply_to_namespace` ("name the field once"). Without `--grow` it is refused by `check_compatible`.
- **The strict loader.** `StrictCheckpointLoad` gains no exception. The parent loads at its own architecture. The
  growth adds the `kind_ffn.*` keys in place. `cpu_probe`'s strict load of the grown state dict into a fresh child
  build proves that the key set and shapes are exactly a fresh child's.
- **The records:** the lineage `grow` stamp and `model_config.json` `growth` (schema `gen3_grow_fork_v1`), with
  `fields: {"trunk_ffn_by_kind": {"parent": "off", "child": "on"}}`. No schema change.
- **Generated docs in the same commit:** `designs/flag_registry.md`, `ARCHITECTURE.md` §6's tables (`arch_tables`),
  the delivery graph only if its snapshot reads the field (production stays off, so it should not move), and the
  CHANGELOG entry. `ARCHITECTURE.md`'s trunk prose gets one sentence, because the lever exists but is not in
  production.

### 1.6 Compile and bf16

- **bf16.** `trunk_precision.set_trunk_precision` swaps the CLASS of every round to its autocast subclass. Since the
  experts live inside the round, they run in bf16 with fp32 accumulation like the shared FFN, with no change to
  `trunk_precision.py`. **The hazard is building them as SIBLING modules called from `TeamTransformer.forward`**: they
  would run in fp32 outside the region, silently. Test K-9 pins this.
- **Compile.** The routing is six static slices, six `Linear → relu → Linear` chains and one `cat` per round. Every
  index is a build-time int, there is no data-dependent op, and so `selection_sites` gains only a `PYTHON` entry for
  the `self.kind_ffn is not None` branch. MEASURED (K-11, 2026-10-20, CPU Inductor): the kffn
  production extractor compiles with `fullgraph=True` (no graph break) and matches eager at 1.30e-5 on the 64 rows,
  against 1.08e-5 for the OFF extractor under the identical procedure (a matched control: the shared FFN's own compile
  noise, ~1.2x); the test's bar is 1e-4, `cpu_compile_sites_slow_test`'s. The CUDA graph is not measured here.
  - The compile-cache key covers the field automatically: it hashes `extractor_arch.arch_toggles_from_args`, which is
    generated from the registry.
  - `--grad-checkpointing` wraps the whole round and is unaffected.
- **The R1 startup gate and the canary** compare compiled against eager at the SAME architecture, so their bars are
  unchanged. Inductor may fuse the six small GEMMs differently from one large one. The growth GATE is eager-CPU-only
  for that reason (B-GROW's design), and K-11 reads compiled kffn against eager kffn at the existing R1 bars.

### 1.7 Cost (MEASURED on the CPU; the GPU is ESTIMATED)

| quantity | value | source |
|---|---|---|
| policy parameters, production fresh build | 2,609,183 → **3,400,223** (+791,040, +30.3 %) | counted; 4 kinds × 65,920 × 3 rounds |
| trunk parameters | 401,536 → 1,192,576 | counted |
| optimizer state | +2 × 791,040 fp32 ≈ +6.3 MB (+9.5 MB with the weights) | arithmetic |
| extractor forward FLOPs per row | 94.3 MFLOP, of which the three rounds' FFNs are 25.2 MFLOP (26.7 %); **unchanged** by kffn (the same work per token) | `FlopCounterMode`, grad enabled, the 64 fixture rows |
| one round's FFN, split versus shared, 64 rows × 64 tokens | forward 6.96 versus 6.08 ms (+14.6 %); forward + backward 21.4 versus 19.7 ms (+8.6 %) | median of 101 / 41 interleaved repetitions |
| the same at 256 rows | forward 29.5 versus 28.4 ms (+3.8 %); forward + backward 83.6 versus 80.4 ms (+4.0 %) | same |
| whole eager forward, 64 rows | 308–677 ms (box-load noise), so the split's ≈ 3 × 0.9 ms is **≈ 1 % of the eager CPU forward** | same script |
| GPU ρ (wall per step against C0) | ≈ 1.00–1.05, **ESTIMATED** (the screen doc's figure, unchanged). Five more small GEMMs per FFN per round, about +30 kernels forward and +60 backward, mostly hidden under compile and CUDA graphs | the arm's first 1M measures it |

A grouped GEMM (padding each kind to the longest, 32 tokens) would cut the kernels but costs ≈ 2.5× the FFN FLOPs.
It is rejected.

### 1.8 Tests (each fails on revert of what it names)

| id | test | it fails when… |
|---|---|---|
| K-1 | off builds nothing: the state-dict keys and every initial byte equal a build without the field | an expert is built, or the RNG stream moves, when off |
| K-2 | the segment table matches the SEAT ROLES of a real production forward. Each position's kind is checked against the seat that wrote it (the type ids the forward hands the trunk): the E5 tail is `threat` (typed THEIR_THREAT, like E4; F-GB-3 corrected), position 31 is `mon`; and the same function on other seat counts moves every boundary, a missing kind a typed refusal | a literal layout, a wrong seat count |
| K-3 | tied experts (each a copy of the shared FFN) give a full policy whose logits and values are BIT-IDENTICAL to the shared FFN's (64 fixture rows, one batch, CPU fp32 eager) | a segment drops, duplicates or reorders tokens |
| K-4 | perturbing expert k changes, at that round's output, exactly the tokens of kind k (a forward hook) | mis-routing between kinds |
| K-5 | a fresh `on` build: every non-expert parameter's initial bytes equal the off build's | the experts are not `IsolatedLinear` or are not built in `fork_rng` |
| K-6 | an `IdentityInitRound` built with experts is the identity (its experts' `linear2` zero); `--grow trunk_layers` on a kffn parent stays bit-identical | the experts' zero init in extra rounds |
| K-7 | **the grow fork:** a production-shaped parent after one AdamW step (`growth_test`'s pattern); `apply_growth` with `trunk_ffn_by_kind off → on`; the child's logits and values bit-identical; the grown state dict loads strictly into a fresh `on` build; the optimizer carries every parent state and every expert holds a CLONE of its source's (Q-K3); with depth at the same fork, depth applies first | the copy, the key layout or the optimizer rule |
| K-8 | a non-copy expert init (monkeypatched) is REFUSED by the fork-time gate | a gate that does not compare |
| K-9 | under `set_trunk_precision(policy, "bf16")` the expert matmuls run in bf16 (a dtype hook) and the round returns fp32 | experts outside the round class |
| K-10 | refusals: `on` without one of the five kinds (the toy argv) is typed; `on → off` growth is refused; `--trunk-ffn-by-kind` beside `--grow` is refused; a fork typing it without `--grow` is refused by `checkargs` and the dry run (the F-BI-2 path) | a missing guard |
| K-11 (slow) | the compiled kffn extractor builds fullgraph (no graph break) and matches eager at the R1 bars; `slow_tier_status.json` records it | a compile regression |
| K-12 | versioning: a pre-version config migrates to `off`; a mismatched resume names `--grow`; a restart of a grown child keeps `on` (the `derived_toggle_resume_test` round trip); `flag_registry_test`'s five surfaces | the migration, the clause or the tier |
| K-13 | `grow_seeded_pool` grows a kffn pool; a relaunch that died mid-pool is idempotent | the field-generic pool path |
| K-14 | `main/probe_battery/pin_worker.py` on a kffn policy either routes its re-implemented round through `round.ffn` (with its truncation and spectra looping over the experts) or refuses, typed | F-GB-6: a silently wrong probe |

### 1.9 Effort

**≈ 1.5 agent-days, opus-high.** The pieces:
- the module, segments and init (0.5);
- the registry, versioning and migration (0.25);
- the grow row and pool (0.25, plus the §3 refactor at 0.25 if B-KFFN lands first);
- `pin_worker` (0.1);
- tests K-1…K-14 and the generated docs (0.4).

The slow compile test (K-11) runs on the CPU. A GPU confirmation of ρ comes from X48's first 1M, not from the build.

---

## 2. B-INT: the opponent-intent objective shapes the trunk

### 2.1 How the intent loss reaches the trunk today, and after

- **The head.** The flat opponent pointer (`flat_intent.py`) scores, with one softmax:
  - the opponent active's K move seats;
  - OTHER_move;
  - a switch to each slot;
  - OTHER_species.

  It reads candidate TOKENS (`flat_candidates`: the refined E4 seats, the active's E5 seat, their 6 team tokens and
  OTHER_species' token) and a context (`cat(our_team_pooled, their_team_pooled)`), all of them trunk outputs.
- **The loss.** `intent_fold` → `flat_intent_fold` reads the LIVE logits through `fe.belief_supervision(
  "flat_intent_logits")` and adds `opp_intent_coef × CE` to the PPO loss. The coefficient is a recorded training
  hyperparameter: production's `recipe.fresh.opp_intent_coef` is 0.05, and a fork inherits its parent's value.
- **The stop-gradient.** At `extractor_forward.py` (the `_keep = self.opp_intent_grad_mode == "shaping"` branch):

      if not _keep:
          _ftok, _ictx = _ftok.detach(), _ictx.detach()

  Under `detached` the CE trains only the head's own 24,832 parameters. Under `shaping` the two detaches are skipped,
  and the CE's gradient flows through the head into every trunk output it reads, then back through the trunk.
- **What does NOT change (MEASURED).** On the trained baseline, the policy loss's gradient on the trunk parameters is
  BIT-IDENTICAL under the two modes: its norm is 9.909 under both and the difference norm is 0.0. The reason is that
  the head's only forward consumer, move resolution, reads α / β as stop-grad publications (`move_resolution.py`:
  "the opponent model (stop-grad publications)"; β is `beta_logits.detach()`). The intent CE's trunk gradient is 0.0
  under `detached` and non-zero under `shaping` (13.4 for a log-sum-exp proxy objective, on the same rows).
  **So `shaping` opens exactly ONE new trunk route, the intent CE at weight `opp_intent_coef`, and X49 is one lever.**
  Test I-2 pins this, so a future consumer that reads the live flat logits fails it instead of silently adding a
  second lever.
- **The weight.** X49 keeps the parent's recorded `opp_intent_coef` (0.05): one lever. The CE's share of the trunk
  gradient is read by the existing grad-balance probe (`grad/opp_intent_share`) and the fight detector
  (`grad/opp_intent_policy_cosine`). Under `detached` both read ≈ 0 by construction. A dose arm (0.1 or 0.2) is a
  later, separate lever (Q-I1).
- **`belief_grad_mode`** does not interact. The CE reads `belief_supervision`, which is live under every belief mode,
  and `label_only` only cuts the PUBLICATION.

### 2.2 The grow path: `--grow opp_intent_grad_mode=shaping`

- **The registry tier must change: CONFIG_ONLY → CLI** (F-GB-1). A CONFIG_ONLY field reaches the extractor only
  through `FROZEN_ARCH_KWARGS`, at its registry default `detached`. So a `--grow` that set the namespace would still
  BUILD `detached`, record `detached`, and pass a bit-identity gate that cannot see gradients, giving a silent no-op
  arm. The CLI tier needs:
  - argparse `--opp-intent-grad-mode {detached,shaping}` with default `None`;
  - the `_resolve` line;
  - `ARCH_ARG_KEYS` (generated);
  - `current_model_version` (generated);
  - the registry note rewritten: "re-promoted 2026-10-xx for B-INT".

  The `ModelVersion` field, its v73 migration and its `check_compatible` clause already exist. The clause's text gains
  `--grow opp_intent_grad_mode=shaping`.
- **The `GROWABLE` row** becomes `built=True`. Its hooks:
  - **parse:** `detached` or `shaping`;
  - **validate:** `detached → shaping` only. The reverse is also function-preserving, but no experiment needs it, so
    it stays refused until one does;
  - **current value:** `fe.opp_intent_grad_mode`;
  - **grow:** set the extractor attribute, `features_extractor_kwargs` and `policy_kwargs`. It adds NO parameters and
    leaves the optimizer untouched (`regrow_optimizer` with no new parameters is a pass-through).
- **The fork-time gate, two parts:**
  1. **Forward:** B-GROW's bit-identity on logits and values. It is trivially true (MEASURED `torch.equal` on the 64
     rows) but stays, as the shared gate.
  2. **Gradient** (`gate_gradient` on the row): on the same CPU builds and rows, it takes the intent CE (the
     production fold's loss, with the fixture's labels, or a log-sum-exp proxy where the fixture has no labels;
     whether the fixture carries intent labels is **UNVERIFIED**) and `autograd.grad` with respect to the trunk parameters (`team_transformer.*`).
     - The PARENT must give exactly zero: every gradient `None` or 0.0. The graph is disconnected, so the boundary is
       exact.
     - The CHILD must give a finite, non-zero total norm.
     - Otherwise `GrowthError`. This is what catches a silent no-op such as the FROZEN-kwarg trap above. The check is
       deterministic: zero is structural, and a connected graph with an exactly-zero norm on 64 real rows is not a
       near-boundary case (rule 8).
- **The seeded pool** goes through the same field-generic path: each snapshot is loaded, its field set, the gate
  passed, and it is re-saved, so every zip's pickled kwargs and the arch record equal the child's. An opponent never
  trains, so this is bookkeeping, but it keeps the pool single-architecture with no new exception. The alternative is
  declaring the field opponent-irrelevant in `check_opponent_snapshot_compatible`, like the ride-along heads (Q-I2).

### 2.3 Versioning

- **No new field, and no `ARCH_SIGNATURE` or `OBS_SEMANTICS` change.** `MODEL_CONFIG_VERSION` is bumped only if the
  tier change alters what a fresh `model_config.json` records. It does not, because the field is already recorded, so
  **no version bump is expected**. The build confirms this with `config_only_pattern_test` and
  `derived_toggle_resume_test`, and bumps if either disagrees.
- **The records:** the `grow` stamp with `fields: {"opp_intent_grad_mode": {"parent": "detached", "child":
  "shaping"}}`, plus the gradient check's report (`gradient_check: {parent_trunk_norm, child_trunk_norm}`), an
  ADDITIVE key in `function_check`'s sibling, with the schema unchanged.
- **Compile:** the compile-cache key already contains the field (it is a version-checked arch toggle), and the
  compiled R1 graph differs (the detach is gone), so a cached artifact cannot be reused across modes.
  `selection_sites` already declares the `shaping` branch.
- **bf16:** none. The head runs in fp32 outside the trunk region. The CE's gradient enters the bf16 rounds through
  their fp32 outputs, exactly like the policy's.

### 2.4 Cost

- **Parameters:** 0. **Forward FLOPs:** 0.
- **Backward:** one extra input-gradient through the 24,832-parameter head. The trunk's own backward is shared by the
  summed loss.
- ρ ≈ 1.00, **ESTIMATED**. The screen doc's 1.00–1.03 stays the stated range, and the arm's first 1M measures it.

### 2.5 Tests (each fails on revert of what it names)

| id | test | it fails when… |
|---|---|---|
| I-1 | **the grow fork:** a production-shaped parent; `apply_growth` with `opp_intent_grad_mode detached → shaping`; logits and values bit-identical; the parent's intent-CE trunk gradient is exactly zero and the child's is non-zero | the switch does not take (the FROZEN_ARCH_KWARGS trap), or the gate skips the gradient check |
| I-2 | **one lever:** the policy loss's trunk gradient is bit-identical under `detached` and `shaping` (built from the same state dict) | a consumer starts reading the LIVE flat logits (X49 would become two levers) |
| I-3 | a gradient-check failure (a monkeypatched child that still detaches) is REFUSED, typed | a gate that does not compare |
| I-4 | the tier: `--opp-intent-grad-mode` parses with default `None`; a fresh `shaping` run builds and records `shaping`; a restart of a grown child keeps `shaping` (the round trip) | the CONFIG_ONLY freeze |
| I-5 | refusals: `shaping → detached` growth; the typed flag beside `--grow`; a fork typing `--opp-intent-grad-mode shaping` without `--grow`, in `checkargs` and the dry run; the mismatch text names `--grow` | a missing guard |
| I-6 | `grow_seeded_pool` rewrites the pool, every opponent's logits are bit-identical to its parent snapshot's, and the record equals the child's | the pool path |
| I-7 (slow) | one CPU PPO update under `shaping` logs `grad/opp_intent_share` > 0, and it is ≈ 0 under `detached` | the probe wiring, or the CE not folded |

### 2.6 Effort

**≈ 0.5 agent-day, opus-medium**, if B-KFFN has landed the §3 refactor (+0.25 if B-INT lands first). The pieces:
- the tier change and the five surfaces (0.15);
- the grow row and gradient gate (0.15);
- the pool (0.05);
- tests I-1…I-7 and the docs (0.15).

---

## 3. The shared prerequisite: field-generic growth (≈ 0.25 agent-day; whichever build lands first)

B-GROW's code names `trunk_layers` in three places that a second field would break:
- `apply_growth`'s `child = int(growth["trunk_layers"]["child"])`, a `KeyError` for any other field;
- its `have = trunk_layers_of(policy) if field == "trunk_layers" else None`;
- `grow_seeded_pool`'s `trunk_layers_of(model.policy) == fields["trunk_layers"]["child"]` skip.

Each `Growable` row gains:
- `parse(text)`;
- `validate(parent, child)`;
- `current(policy)`;
- `grow(policy, child) -> new named parameters`;
- an optional `gate(parent_probe, child_probe, rows) -> report`.

`apply_growth` loops over the declared fields in `GROWABLE` order: one probe of the parent, every grower, ONE
`regrow_optimizer` over the union of the new parameters, then one probe of the child. The forward bit-identity check
and every row's extra gate run on the same probes. `grow_seeded_pool`'s skip test becomes "every declared field's
`current` equals its child". `trunk_layers` moves into the same shape, with its behaviour and tests unchanged
(`growth_test.py`, `grow_fork_test.py` stay green untouched, which is the refactor's acceptance check).

---

## 4. Open questions for the orchestrator

- **Q-K1 (B-KFFN partition).** Five kinds (board its own expert over 3 tokens), or four (board shares the Pokémon
  expert)? **Recommendation: five**, the screen doc's partition and the bots' "per kind". The registered descriptor
  (pairwise expert weight distance over training) shows whether the board expert diverges or idles.
- **Q-K2 (the gate's bar on another CPU).** Bit-identity holds here at the 64-row batch (1–6 threads) and fails at
  batch 1–2. If a cloud box's BLAS breaks it at 64 rows, should the gate refuse (recommended: rule 8, no tolerance
  without a declared margin), or take a declared tolerance such as the screen doc's original ≤ 1e-6, with a rule-8
  exclusion band?
- **Q-K3 (the new experts' optimizer state).** B-GROW's rule gives new parameters NO AdamW state, so at the first
  update the four copies take bias-corrected fresh steps while the Pokémon expert keeps its moments, and they diverge
  at once. Alternative: give each copy the shared FFN's moments (`exp_avg`, `exp_avg_sq`, `step`), so all five start
  as one optimizer trajectory and diverge only through their gradients. **Recommendation: copy the moments for
  B-KFFN** (it is the natural "upcycling" analogue, and it avoids a step-1 jolt on 30 % of the parameters), recorded
  as a per-row optimizer rule in the stamp. Depth growth keeps "no state", since its new parameters copy nothing. This
  changes F-BI-3's single rule, so it needs your ruling.
- **Q-K4 (the attribution arm).** The screen doc's follow-up to an ADOPT-CANDIDATE is "a wider shared FFN with equal
  parameters". That too is function-preserving growth: append `linear1` rows and the matching ZERO `linear2` columns,
  FFN 256 → 1,024 for +4 × 65,920 per round. Build it now as a third `GROWABLE` row (≈ 0.5 agent-day), or only on an
  X48 ADOPT-CANDIDATE? **Recommendation: only on ADOPT-CANDIDATE.**
- **Q-I1 (the intent weight).** X49 at the inherited 0.05 (one lever, recommended). A dose arm (0.1 or 0.2) only after
  X49 reads something.
- **Q-I2 (the pool for a gradient-only field).** Re-save each snapshot through the uniform growth path (recommended:
  no new exception class), or declare `opp_intent_grad_mode` opponent-irrelevant in
  `check_opponent_snapshot_compatible`, as the ride-along heads are?

---

## FINDINGS (standing rule 7)

- **F-GB-1 (HAZARD, B-INT's scope).** `opp_intent_grad_mode` is CONFIG_ONLY, and a CONFIG_ONLY field is FROZEN at its
  registry default through `extractor_arch.FROZEN_ARCH_KWARGS`. Simply marking its `GROWABLE` row built would produce
  a child that builds, records and trains `detached`, and it would PASS a forward-only bit-identity gate: a silent
  no-op X49. The tier change and the gradient gate (§2.2) close it. B-KFFN's new field must be CLI for the same
  reason.
- **F-GB-2 (HAZARD, B-GROW as built).** `growth.apply_growth` and `grow_fork.grow_seeded_pool` are hard-coded to
  `trunk_layers` (a `KeyError` for any other field). Today the declared-but-refused rows hide this. §3 is the fix.
- **F-GB-3 (CORRECTED 2026-10-20 by the build).** The spec said the E5 tail seats carry `TOKEN_TYPE_HISTORY`. They do
  not: `EntityMoveSeats.seat_types` gives E4 and the E5 tail `TOKEN_TYPE_THEIR_THREAT` (the event seats' "E5 precedent"
  is only "no token-type table growth"); the real forward confirms it (K-2). So no hazard exists on today's layout, and a
  type-id router would agree with the table. Seat-role routing stays (§1.2: a type id is not a role).
- **F-GB-4 (layout).** OTHER_species sits at position 31, after the entity seats and before the event seats, not
  beside the 12 team tokens. The Pokémon kind is two segments, and the routing is six segments, not five.
- **F-GB-5 (MEASURED, a determinism boundary).** CPU split-versus-shared FFN equality is bit-exact at batch ≥ 8 here,
  but NOT at batch 1–2 (and at 7 under 2 or 6 threads): small-M GEMM kernels reassociate. A gate or test that splits
  its rows into small batches would fail at random across boxes. The gate runs the 64 rows as ONE batch. Another CPU
  is **UNVERIFIED** (Q-K2).
- **F-GB-6 (HAZARD, a re-implemented forward).** `main/probe_battery/pin_worker.py` re-implements the round forward:
  `linear1` / `linear2` directly, a low-rank truncation and the weight spectra. On a kffn network it would be silently
  wrong. B-KFFN routes it through `round.ffn`, or refuses it (K-14). `main/prober/model_capture.py` re-runs only the
  attention and is unaffected.
- **F-GB-7 (MEASURED, B-INT is one lever).** Under `shaping` the policy's trunk gradient is bit-identical to
  `detached` on the trained baseline (move resolution reads α / β stop-grad). The only new trunk route is the intent
  CE at `opp_intent_coef`. I-2 pins it.
- **F-GB-8 (CONFIRMED figures).** The screen doc's "2.61M → 3.40M" is exact: 2,609,183 → 3,400,223 for the fresh
  production policy. The trained baseline counts 5,624,179 because it carries its ride-along heads (declared
  overrides). FFN parameters are 65,920 per round. The intent head has 24,832 parameters.
- **F-GB-10 (B-KFFN as built, a capability gap).** `pin_worker depth` REFUSES a kffn checkpoint (`UnsupportedTrunk`,
  K-14) rather than routing its replica through `round.ffn`: the probe battery's depth / FFN-unit / truncation / spectra
  diagnostics are unavailable on X48's arms until the replica learns the segment table. Its `capture` command (forward
  hooks on the real rounds) is unaffected. X48's registered descriptor (the experts' pairwise weight distance) reads
  weights and needs neither.
- **F-GB-11 (Q-K2, MEASURED on the build).** The gate's bar held on this box on a TRAINED-shape production parent (one
  AdamW step): bit-identical at the 64-row batch, every round. Another CPU stays **UNVERIFIED**; a failure there is the
  typed refusal (ruling Q-K2), never a tolerance.
- **F-GB-9 (process).** The measurement scripts were throwaway and are not committed. They ran with nice 19 on cores
  5–7 / 13–15 under `ulimit -v` 12 GB instead of `scripts/ops/mem_cap.sh`: the capped scope sat queued behind other
  agents' capped jobs, and the jobs were light (≈ 1 min each). Their wall-clock figures carry the shared box's noise.
  The FLOP and parameter counts and the bit-identity results do not.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-19 | **This doc CREATED** (design worker, orchestrator brief): build specs for B-KFFN and B-INT | **B-KFFN:** five per-kind experts per round, the shared FFN kept as the Pokémon expert (parent keys unchanged), routed by build-time seat-role SEGMENTS, experts inside the round classes (bf16 for free), a private-seed `IsolatedLinear` init when fresh and COPIES at a fork, a CLI-tier `trunk_ffn_by_kind` field with a config bump and no `ARCH_SIGNATURE` bump, the gate bit-identical at one 64-row batch. **B-INT:** `--grow opp_intent_grad_mode=shaping` with the flag RE-PROMOTED to the CLI tier, a forward + GRADIENT gate, `opp_intent_coef` inherited. A field-generic growth refactor first. B-KFFN before B-INT, in series | routing by token-type id (F-GB-3); experts as sibling modules (they would escape the bf16 region); a padded grouped GEMM (≈ 2.5× FFN FLOPs); a tolerance gate for the experts (bit-identity MEASURED achievable at 64 rows); leaving `opp_intent_grad_mode` CONFIG_ONLY (F-GB-1: a silent no-op arm); a two-pass flat head to cut a consumer route (MEASURED unnecessary: F-GB-7); building experts on the toy (no move, threat or event tokens) | the CPU measurements above (fresh production build at `b79b66bb`; trained `rb_es_end2_s2005`); `growth.py`, `grow_fork.py`, `team_transformer.py`, `trunk_depth.py`, `trunk_precision.py`, `extractor_forward.py`, `move_resolution.py`, `extractor_arch.py`, `flag_registry.py`, `pin_worker.py` |
| 2026-10-19 | **Open questions RULED** (orchestrator, owner's full-discretion grant) | **Q-K1:** five kinds (board has its own expert). **Q-K2:** refuse when bit-identity fails at 64 rows (rule 8, no tolerance). **Q-K3:** a COPIED expert inherits the shared FFN's AdamW moments and step at the fork (a copy is the same function, so it keeps the same optimizer state); identity-init NEW layers (B-GROW depth) still start fresh, so F-BI-3's rule holds for them. **Q-K4:** the equal-parameter attribution arm is built only if X48 reads adopt-candidate. **Q-I1:** X49 runs at the inherited `opp_intent_coef` 0.05. **Q-I2:** re-save each pool snapshot through the uniform growth path | four kinds; a declared tolerance; fresh moments for copies; building the attribution arm now; a dose arm first; declaring the field opponent-irrelevant | the recommendations in "Open questions" above; F-BI-3 (B-GROW's optimizer rule) |
| 2026-10-20 | **B-KFFN BUILT (T66)** with the §3 refactor (B-KFFN worker, orchestrator brief; config v161) | As specified, with Q-K1…Q-K4 as ruled: five kinds, six seat-role segments, experts inside `TrunkRound`, private-seed fresh init, `--grow trunk_ffn_by_kind=on` copying weights AND AdamW state (`copied`), bit-identity refusal at one 64-row batch, CLI-tier field, no `ARCH_SIGNATURE` bump; `pin_worker depth` REFUSES (F-GB-10). What changed from the text: F-GB-3 CORRECTED (the E5 tail is typed THEIR_THREAT); §1.4's optimizer paragraph rewritten to the ruling; K-2 / K-7 rows updated; config v161 (v160 was FP32_ALT) | routing `pin_worker`'s replica through the experts now (refused instead: F-GB-10) | fresh production 2,609,183 → 3,400,223 params; the grow gate on a one-step production parent max \|Δ\| 0.0, 48 tensors all copied; OFF: learner golden, played-bits sentinel, obs golden unchanged; tests K-1…K-14 (`trunk_kffn_test.py`, `grow_fork_test.py`, `probe_battery_test.py`) |
