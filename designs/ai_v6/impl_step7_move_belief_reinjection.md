# Implementation: Step 7 — In-place opponent belief, move reinjection

**As-built record** (shipped to `main`, NOT yet run in a fresh A/B). The in-trunk realization of the
"predict the opponent's hidden party" aux head the v6 `todo.md` (the line after Step 6) anticipated as
the natural second aux head on the shared trunk. Live architecture detail: `src/agents/model/CLAUDE.md`
(MoveBelief, v17) + `src/agents/training/CLAUDE.md` (loss + labels + metrics). Predecessor (species
half): `designs/ai_v5/belief_aux_as_built.md`.

**Step status at entry.** The species half of the in-place belief shipped first (`opp_belief_slots` /
`--opp-belief-aux-coef`, v16): `BeliefSlots` fills the un-revealed opp team slots with distinct learned
unknown-mon tokens BEFORE the transformer (refined in-lineup, attended by both heads), and a `BeliefHead`
aux-supervises the refined opp tokens on **species (CE) + moves (multi-label BCE)** with order-invariant
(Hungarian) matching. Privileged labels (`battle2.team`) ride training-only Dict-obs keys; they never
enter the forward. That head is a **readout** — it predicts but does not feed its prediction back into
the policy/value representation.

---

## Motivation

The species `BeliefHead` answers "*which* mons are hidden" but its output is a dead-end aux readout. Two
gaps remain:

1. **The belief doesn't reach the decision.** A readout shapes the trunk only weakly (through the shared
   gradient); the policy never *attends over* the predicted moveset. To make the belief actionable, the
   prediction has to flow **into** the representation the heads read — not sit beside it.
2. **The highest-value move signal is on REVEALED mons, not hidden ones.** The loss-triage corpus's
   `surprise_ohko` bucket is dominated by a *seen* mon landing a move we hadn't scouted (we know the
   species; we hadn't seen that slot of its moveset). Predicting an *unseen* mon's exact moveset is
   intrinsically noisy; predicting a *seen* mon's unrevealed moves is the defensible, learnable lever.

Step 7 adds a **reinjection** head (the prediction flows back into the token) and makes its scope a flag
so the defensible (revealed) and omniscient (unrevealed) variants are a clean A/B.

---

## Architecture — `MoveBelief` (`features_extractor.py`)

Runs AFTER `BeliefHead` (so the species readout sees the pre-reinjection tokens — uncontaminated) and
BEFORE the CLS pools (so the enrichment DOES reach the heads):

```
ObsUnpack → PokemonEncoder → [BeliefSlots?] → TeamTransformer
          → [BeliefHead?] (readout) → [MoveBelief?] (reinject) → CLSPool → ProjectionAssembler → heads
```

Per opp slot:
1. **Predict** the moveset — `move_head: Linear(D_MODEL → n_moves)` → `move_logits [B, 6, M]`.
2. **Soft-embed** it — `sigmoid(move_logits) @ move_embedding.weight` → the *expected-moveset* embedding
   `[B, 6, move_emb_dim]` (the shared move-embedding table; per-move presence-weighted).
3. **Reinject** — `reinject: Linear(move_emb_dim → D_MODEL)` (small-init, so the enrichment starts ≈0
   and does no harm before the head sharpens), added as a **residual gated to the mode-selected slots**,
   then `LayerNorm`.

The enriched `their_team_out` feeds the CLS pools, so both heads reason about the believed moves. The
move logits are stashed at `features_extractor.last_move_belief_logits` (None when off) for the loss.

### Mode (`move_belief_mode`) — the slot population

| mode | mask | scores / enriches | character |
|---|---|---|---|
| `off` | — | nothing (module not built) | baseline byte-for-byte |
| `revealed` | `~opp_believed_mask` | seen mons → their UNREVEALED moves | **defensible** (surprise-OHKO lever) |
| `unrevealed` | `opp_believed_mask` | hidden mons (anonymous slots) | omniscient |
| `both` | all-ones | both populations | — |

The `revealed`/`unrevealed` names refer to the **mon's reveal status** (deliberately not `known`/
`unknown`, which collides with "known moves" in a move-prediction feature). The `revealed`-vs-`unrevealed`
axis is the **defensible-vs-omniscient A/B**.

---

## Loss — `_move_belief_loss` (`instrumented_ppo.py`)

Folded into `train()` at `move_belief_coef`, over two DISJOINT slot populations (the two label tensors
PAD each other's slots, so `both` simply scores each with its own rule):

- **REVEALED slots** (mode `revealed|both`) → **direct** multi-label BCE on `known_moves` (slot identity
  IS the revealed species; no matching). A slot whose moveset is all-PAD is not supervised (would push
  "predict nothing").
- **UNREVEALED slots** (mode `unrevealed|both`) → **order-invariant (Hungarian)** multi-label BCE on
  `belief_moves`. The believed slots are anonymous, so the k predictions are min-cost matched to the k
  hidden movesets — same defect the species aux fixed (a fixed slot↔mon target chases a reveal-shifting
  assignment). The matching cost is the **assignment-relevant part of BCE**, `-(pred · target)` (the
  per-slot constant BCE terms drop out of the argmin), so it is a cheap einsum, not a full pairwise BCE.

The move-loss gradient also reaches the shared trunk via the reinjection, so it joins the species aux in
the combined `grad/belief_share` probe. **FAIL-LOUD** on an out-of-vocab move id.

---

## Labels (`belief_labels.py` + `gen3_env.py`)

Privileged, training-only, sourced from `battle2.team` (agent2's own full team); read ONLY by the loss.

- `known_moves[6, 4]` — **new** (`build_known_move_labels`): each REVEALED slot's FULL privileged moveset
  (so the head learns the as-yet-unrevealed moves). The believed slots stay PAD. Emitted only when
  `move_belief_mode ∈ {revealed, both}`. (Name kept: it holds the privileged-*known* moveset of a
  revealed mon — orthogonal to the `revealed`/`unrevealed` mode names, which refer to the mon.)
- `belief_moves[6, 4]` — **reused** from the species aux: the hidden movesets at the believed slots
  (species-sorted, Hungarian-matched in the loss).

`Gen3Env` emits the label keys when species-belief OR move-belief is on; the believed-slot mask is read
DIRECTLY from the obs `species_known` (single source — labels can't diverge from where the model injects).

---

## Versioning (v17) + config storage

| field | kind | gate | resume | model_config.json | metadata.json |
|---|---|---|---|---|---|
| `move_belief_mode` | structural (str) | `check_compatible` (string compare) — fresh-only | inherited (read-back) | ✅ authoritative | — |
| `move_belief_coef` | training-only | none | inherited; freely mutable | ✅ authoritative | ✅ (display) |

`off` reproduces the baseline arch byte-for-byte → **NO `ARCH_SIGNATURE` bump**. `MODEL_CONFIG_VERSION`
bumped 16 → 17 (migration sets `move_belief_mode="off"` / `move_belief_coef=0.0`). The mode is threaded
into `current_model_version` / `arch_toggles_from_model` so a move-belief-ON self-play run doesn't FATAL
on its own sentinels — covering **all 4 opponent-load sites** (in-process pool, stable, `eval_worker`,
distill). `model_config.json` is the functional store (it drives the gate AND the flagless-resume
read-back); `metadata.json` mirrors the coef for provenance/TUI only.

---

## Guards (fail loud on nonsensical configs)

- `--move-belief-mode != off` **auto-forces** `--attend-unrevealed-opponents` (the slots must be
  attendable to be refined; the model side hard-gates on it).
- `--move-belief-mode {unrevealed, both}` **REQUIRES** `--opp-belief-aux-coef > 0` → `parser.error` if
  absent. Without the species head, the hidden slots are never filled with learned unknown-mon tokens
  (they stay encoder placeholders ≈ zeros), so predicting a hidden mon's moveset from an empty token is
  meaningless. `revealed` mode is **exempt** — it scores REVEALED slots, which carry real role-tokens
  regardless of the belief head.

---

## CLI

```
--move-belief-mode {off,revealed,unrevealed,both}   # structural, fresh-only (version-checked)
--move-belief-coef <float>                          # training-only loss weight (0 = reinject, no supervision)
```

Recommended A/B:
```bash
# Defensible arm — predict revealed mons' hidden moves; no species head needed
--move-belief-mode revealed   --move-belief-coef 0.1
# Omniscient arm — predict the hidden party's moves (species head required)
--move-belief-mode unrevealed --move-belief-coef 0.1 --opp-belief-aux-coef 0.05
```

---

## Metrics (`train/belief_move_*`)

`bce`, `precision`, `recall`, `revealed_slots`, `unrevealed_slots`, `loss`. Plus the shared-trunk
`grad/belief_share` (now species + move aux pull) + `grad/belief_policy_cosine` — the "is the aux
dominating / fighting the policy" signal. Tuning is empirical: keep `belief_share` modest (start
`--move-belief-coef` small); confirm `precision`/`recall` climb in warmup.

---

## Files Created / Modified

| File | Change |
|------|--------|
| `src/agents/model/features_extractor.py` | **New** `MoveBelief` module; `move_belief_mode` ctor arg + forward reinjection (mode mask) |
| `src/agents/model/model_version.py` | v17: `move_belief_mode` (version-checked) + `move_belief_coef`; migration; `check_compatible` |
| `src/agents/model/snapshot.py` | `current_model_version` / `arch_toggles_from_model` thread the mode |
| `src/agents/observation/belief_labels.py` | **New** `build_known_move_labels` / `zero_known_moves` |
| `src/agents/training/gen3_env.py` | emit `known_moves`; `move_belief_mode` plumbing + `_emit_known_moves` |
| `src/agents/training/instrumented_ppo.py` | **New** `_move_belief_loss`; fold + combined grad-balance probe |
| `src/main/train_rl_agent.py` | flags; resume read-back; the guards; threading at every build/env/version site; metadata hparam |
| `*_test.py` (model/training/observation) | `move_belief_loss_test`, `move_belief_test`, `known_moves` labels, version-gate/threading |

---

## Verification (done)

- **Unit**: full suite **2392 passed** (CPU). `move_belief_loss_test.py` (direct-BCE; Hungarian
  order-invariance + min-cost match; mode gating — a mode ignores the other population; grad; fail-loud).
  `move_belief_test.py` (module mask-gating; reinject grad; per-mode wiring; `off` projection-dim
  byte-identical; attend-unrevealed dependency guard). `snapshot_test.py` (version gate on/off + value
  mismatch + threading). `belief_labels_test.py` (`build_known_move_labels`).
- **Smoke** (`--debug`, bridge, CPU): `--move-belief-mode both` → roundtrip PASSED, `train()` folds the
  loss, `revealed_slots` + `unrevealed_slots` both > 0; `--move-belief-mode revealed` ran to completion
  with `unrevealed_slots == 0` (mode gating correct).
- **Fuzz**: `belief_labels_fuzz_test.py` (741 real-battle decisions) — species-label invariants held, no
  regression from the `_belief_labels` change.
- **Guards**: `--move-belief-mode unrevealed` without species belief → `parser.error` (exit 2); `revealed`
  runs standalone; the renamed values reject the old `known`/`unknown` as `invalid choice`.

---

## Open gate (honest)

It LEARNS (move `recall` climbs immediately at the tiny CPU smoke) but it is **UNMEASURED whether it
HELPS the policy**. Falsify-after-build = a fresh-run A/B (`revealed` vs `unrevealed` at matched
`--move-belief-coef`, plus a coef=0 reinject-only control) where the move-belief metrics climb AND a
NAMED behavioural metric moves — the **surprise-OHKO read-rate / crater share falls** AND win-rate is
non-regressing. This is the same honesty discipline as the species half and Step 6's Stage-3 gate: the
belief is a means; reaching the heads is necessary but not sufficient. **Risk**: "learnable but
inconsequential" — the prediction is right but the policy doesn't act on it (the incoming-belief
precedent shaped the trunk yet the policy under-switched; a credit-assignment gap, not representation).
The `revealed` arm is the bet I'd expect to pay off; the `unrevealed` arm is higher-variance and the one
to treat skeptically.

**Relationship to Step 6 (latent predictive representation).** Step 6 shapes the trunk to anticipate the
*next-ply outcome*; Step 7 shapes it to anticipate the *opponent's hidden moveset*. Both are
search-free, feedforward, aux-supervised trunk-shaping levers; both reuse privileged/oracle supervision
that never enters the forward. They compose — Step 7's reinjection (a prediction *fed back into* the
representation) is the same "make the belief actionable" pattern Step 6 Stage 4 pursues with per-action
outcome tokens.
