# Design — Discrete top-K incoming move-space for the DamageOperator (ai_v6)

**Status:** design (NOT built, NOT run). Retrain-class, flag-gated, **OFF byte-identical**.
`MODEL_CONFIG_VERSION` 29 → 30; `ARCH_SIGNATURE` stays `gen3_wish_wired_v1` (OFF reproduces the
baseline byte-for-byte, like every op toggle v19–v29). Proposed tag: `gen3_unified_topk_incoming_v1`.

> **Surface-before-build.** This doc is the design the task asks to review first. No code is
> written yet. The open decisions flagged **[DECISION]** below are the ones worth a look before
> implementation.

---

## 1. Why — the op destroys the move-identity work we paid for

We built two systems so the model could reason about *individual* opponent moves:

1. **`MoveLatentEncoder`** (`--move-latent`, v24): a context-free, mechanics-grounded per-move
   latent (`MLP(move_emb ⊕ type_emb ⊕ MOVE_ATTR[id]) → MOVE_LATENT_DIM=32`), a learned identity
   where Rock Slide ≈ Hidden Power Rock. `latent_table()` is `[n_moves, 32]` on the move-num axis.
2. **Status / secondary signals** (v24/v27): per-status secondary probabilities (para/flinch/freeze,
   accuracy-folded × Serene Grace / Shield Dust) + the status-landing block.

Then the `DamageOperator`'s **incoming** block throws that richness away. Its per-defender output
is a belief-weighted **hard max** (`_chan_max`, `features_extractor.py:1464`) — the single worst
PHYSICAL and worst SPECIAL hit per defender:

```python
phys_high = self._chan_max(wb * high_frac, phys_mask)   # [B,6]  — one number, the worst phys hit
spec_high = self._chan_max(wb * high_frac, spec_mask)   # [B,6]  — one number, the worst spec hit
```

With only a phys/spec magnitude per defender, the model **cannot** do the two things that decide
games:

- **Anticipate which discrete move** the opp will use (the worst-phys scalar hides whether it's
  Earthquake or Rock Slide — different identities, different follow-ups), and
- **Pick a safe pivot**: the collapse is *per defender independently*, so "the likely move OHKOs my
  active but does 0 to my Steel-type" is unrecoverable — the per-defender maxes don't share a move
  identity, so the policy can't line up *one* threat against *all six* of our mons.

We want the model to reason in the **discrete move space** — over the opponent active's actual
likely moves, each carrying its latent identity — *alongside* the existing worst-case summary.

This is exactly the differentiable-op design's own prescription (`design_differentiable_damage_op.md`
§4): **don't pre-collapse — feed per-candidate tokens and let attention aggregate**, because the
trunk is good at aggregation but bad at physics. §4.3 names the **hybrid** (top-k raw tokens **+** a
cheap pre-computed summary anchor) as the sweet spot. The current op shipped only the summary
(`_chan_max`); this adds the top-k half it always intended.

---

## 2. The feature — a discrete top-K incoming block

For the opponent active's **K most-believed candidate moves** (`K = damage_topk_k`, a configurable
int, **default 5** — see §8; the constant `_DMG_TOPK_DEFAULT_K = 5`), surface — per move — enough
to identify it AND evaluate every pivot, **including immunity** (§2.1, the headline safe-switch
case):

**(a) Opp-property block** (per top-K move, shared across our 6 defenders — a move is an opp
property, selected once):

| field | dim | source | grad |
|---|---|---|---|
| `latent` | `MOVE_LATENT_DIM` = 32 | gathered from the candidate latent table (§4) | **yes → MoveLatentEncoder** |
| `belief_w` | 1 | `w_all` at the selected candidate | **yes → move-belief logits** |
| `accuracy` | 1 | `acc_all` buffer | no (fact) |
| `is_phys` | 1 | `phys_all` buffer (1=physical, 0=special) | no (fact) |

`_DMG_TOPK_MOVE = 32 + 1 + 1 + 1 = 35` per move. (The per-status secondary identity rides *in the
latent* — `MOVE_ATTR` carries the 10 secondary chances — so we don't duplicate a 10-col block here;
the existing op-level belief-weighted `incoming_secondary[10]` aggregate stays, and the realized,
immunity-folded per-status threat is the new per-pivot `status_lands` below.)

**(b) Per-(defender, top-K move)** — the pivot-decidability block (the literal "switch in safe"
read), over `TEAM_SIZE=6` defenders × K moves:

| field | dim | source | grad |
|---|---|---|---|
| `high` | 1 | gathered from the raw `[B,6,C]` max-roll tensor (§3) — `0` for a damage-immune pivot | no (w-independent physics) |
| `pko` | 1 | gathered from the raw `[B,6,C]` accuracy-folded P(KO) tensor | no |
| `status_lands` | 1 | P(the move's status lands on THIS pivot), **immunity-folded** (§2.1) — `0` for a status-immune pivot | no |

`_DMG_TOPK_DMG_PER = 3`. (Lean by design: `high` + `pko` answer "is this pivot safe from the hit,"
`status_lands` answers "safe from the status." The roll band `low`/`crit` lives in the retained
worst-case block; they're a tail detail for the *pivot* decision. Richness lever in §6.)

**Total block:** `_DMG_TOPK(K) = K·_DMG_TOPK_MOVE + TEAM_SIZE·K·_DMG_TOPK_DMG_PER = K·(35 + 18) =
K·53`. **K=5 → 265**; K=4 → 212. Appended **at the very end** of the op output (after incoming + CB
+ outgoing + status), so existing offsets are untouched:
`out_dim = incoming_dim + (outgoing+status if outgoing) + (_DMG_TOPK if topk)`.
Appended to **both** projection heads (policy needs "which move → which pivot"; value needs the
incoming-KO tail per pivot). It does **not** enter the token stream (same seam as the rest of the op).

So the model can read a row like: *move 0 = ⟨latent: a special Ice-type, acc 1.0⟩, belief 0.62;
→ ▶active high 0.83 / pko 0.9 / status 0, Steel-bench high 0.0 / pko 0.0 / status 0* → **pivot to the
Steel-type.** And for *Thunder Wave, belief 0.4 → ▶active high 0 / status 1.0, Ground-bench high 0 /
status 0* → **pivot to the Ground-type** (immune to the paralysis).

### 2.1 Immunity is the most valuable switch — both kinds are preserved [DECISION, owner-driven]

The owner's priority: the best switches are *immunities*, and the signal must show them. Two kinds:

- **Damage immunity** — Ground→Flying / Ground→Gengar(Levitate) / Normal↔Ghost / Electric→Ground,
  etc. The per-pivot `high`/`pko` are gathered from the raw `_damage_rolls` `[B,6,C]` tensors, which
  already apply the **type-chart 0-entries** (the 8 immunities fall out of the `[19,19]` product)
  **and ability immunity** (`ABILITY_DAMAGE_MULT` — Levitate/Flash Fire/Water-Volt-Absorb). So a
  damage-immune pivot reads **exactly 0** — already proven by `test_immune_defender_reads_zero` /
  `test_buffers_axis_and_immunity`. This is preserved regardless of any dim lever.
- **Status-move immunity** — **Thunder Wave→Ground** (the owner's example), Toxic→Steel/Poison,
  Will-O-Wisp→Fire, sleep/para powders→Grass, Leech Seed→Grass. These are **non-damaging**, so the
  damage scalars can't show them. The new per-pivot **`status_lands`** scalar fixes exactly this —
  see §3.1.

So the per-pivot block makes *both* immunity kinds first-class: a Ground pivot vs a Thunder-Wave
user reads `high=0` (it's a status move, no damage) **and** `status_lands=0` (immune to the
paralysis) — the expert "this is the safe switch" read, in the discrete move space.

---

## 3. Top-K selection & differentiability (the load-bearing part)

The per-candidate roll tensors already exist before the collapse
(`features_extractor.py:1866`): `_damage_rolls(...)` returns `high_frac, low_frac, crit_frac,
ko_ramp, high_cb, ko_cb`, each **`[B, 6, C]`** of **raw** physics (the belief weighting `wb·roll`
happens *inside* `_chan_max`, not in these tensors). `C = n_moves + 16` (real move-nums + 16 typed
Hidden Powers), and `w_all [B, C]` is the belief over that same candidate axis (`:1860`).

Selection (mirrors the existing detached-argmax provenance gather at `:1903`), `K = self.topk_k`:

```python
topk_idx = w_all.detach().topk(self.topk_k, dim=-1).indices              # [B,K] — selection DETACHED
w_topk   = w_all.gather(-1, topk_idx)                                     # [B,K] — DIFFERENTIABLE in belief
```

Gathered values:

```python
latent_topk = move_latent_all[topk_idx]                                  # [B,K,32] — DIFFERENTIABLE in latent table
acc_topk    = acc_all[topk_idx]                                          # [B,K]    — buffer (no grad)
phys_topk   = phys_all[topk_idx]                                         # [B,K]    — buffer (no grad)
idxd        = topk_idx[:,None,:].expand(B, TEAM_SIZE, K)                 # [B,6,K]
high_topk   = high_frac.gather(-1, idxd)                                 # [B,6,K] — w-INDEPENDENT physics, 0 if damage-immune
pko_topk    = ko_ramp.gather(-1, idxd)                                   # [B,6,K]
status_topk = self._incoming_status_lands(ctx, topk_idx)                 # [B,6,K] — immunity-folded (§3.1)
```

### 3.1 Per-pivot incoming status-landing (the Thunder-Wave→Ground signal)

`status_lands[b, d, k]` = P(top-K move k's status lands on OUR defender d), the **incoming** mirror
of the existing OUTGOING `_status_landing` (`:1713`). Computed **post-selection** (on the K selected
candidates, not all C — cheap), folding the facts the status-landing machinery already holds, gathered
at **our** defenders' types/abilities/conditions (all known — no belief, no leak):

```
status_lands = chance · acc · (1 − type_immune@our_def_types) · (1 − ability_block@our_def_ability)
                              · (1 − already_statused@our_def) · damage_lands_gate
```
where, per selected move:
- **dedicated status moves** (Toxic/Thunder Wave/Will-O-Wisp/Spore/Leech Seed, BP=0): `chance = 1`,
  `acc` = the move's accuracy, `damage_lands_gate = 1`. `type_immune` from `MOVE_STATUS_TYPE_IMMUNE`
  gathered at our defender's `type1/type2` — **Thunder Wave→Ground = 1 → status_lands = 0** (the
  owner's case); Toxic→Steel/Poison, Will-O-Wisp→Fire, Leech Seed→Grass likewise.
- **damaging moves with a secondary status** (Body Slam 30% para, Ice Beam 10% frz): `chance` = the
  move's max secondary chance × Serene-Grace(opp); `damage_lands_gate = 1[high_topk > 0]` — a
  secondary can't fire if the move is **damage-immune** (Thunder's para can't hit a Ground type
  because the Electric move does 0). gen3 has no type-based para/freeze immunity beyond that, so this
  gate + the type-chart is the whole story.
- **abilities** (`ABILITY_STATUS_BLOCK` — Limber/para, Immunity/psn, Water Veil/brn, Insomnia·Vital
  Spirit/slp, etc.) read at our defender's KNOWN ability (exact, not a prior — our side).
- **already-statused** — a major status can't double-apply (our defender's condition one-hot).

All inputs are buffers + `ctx` (our-side, public) → **w-independent** (like damage); the belief
gradient rides `w_topk`. Reuses `gen3_mechanics` rules via the existing `damage_tables`
status-landing buffers (`MOVE_STATUS_TYPE_IMMUNE`, `MOVE_INFLICTS_STATUS`, `MOVE_IS_SLEEP`,
`ABILITY_STATUS_BLOCK`, `MOVE_SECONDARY`) — no new physics, just the incoming direction + the
defender gather. Sleep-Clause/Substitute (opp-side gates in the outgoing version) are N/A incoming
(the status lands on US — our Substitute would block, a v2 refinement; not the owner's named case).

**Gradient story (and the [DECISION] on "carry gradient via w"):**

- **Belief is sharpened** — `w_topk` is a live feature; the policy/value loss backprops through it to
  the move-belief logits of the K selected candidates. This is genuine, *additional* belief-sharpening
  on top of the retained `_chan_max` (which keeps the dense argmax gradient). Same surface as the
  existing op (`test_grad_flows_to_move_belief_head`).
- **Latent is sharpened** — `latent_topk` gathers (with a detached index) from a *differentiable*
  latent table, so gradient flows into `MoveLatentEncoder` for the selected moves. This is the
  "sharpen the move LATENT" the task names. (It composes with the v24 grading aux, whose stop-grad
  target prevents collapse; the op gradient is an extra RL shaping signal on the same table.)
- **Damage stays w-INDEPENDENT (deliberate).** `dmg_topk` is raw physics (a function of looked-up
  BP/type and obs stats), **not** `w·damage`. The task says "the gathered damage MUST still carry
  gradient via w" — I read that as *"the block must keep sharpening the belief,"* and satisfy it
  through `w_topk` + the retained `_chan_max`, **not** by re-coupling damage to `w`. Coupling them
  (`w·damage`) would re-introduce the exact correlation the project's "provide facts, let the head
  weight" / Jensen-gap principle forbids (`design_differentiable_damage_op.md` §2, §4.1; the v28
  Choice-Band decorrelation precedent). Keeping damage decorrelated lets the head weight identity ×
  belief × per-pivot damage itself. **Flagging this as the one interpretation worth confirming.**

### 3.2 Meaningful-K gate — drop the 5th slot once all 4 moves are known [owner-driven]

A gen3 mon has exactly **4** moves, so once all 4 are revealed the moveset is *closed* — a 5th
candidate is a move the mon definitionally cannot have, "nothing else to reason about." The K=5
tensor is fixed-dim (structural), so we **zero the 5th slot's values** rather than resize:

```python
n_revealed = (ctx.all_move_ids[ar, opp_act] > 0).sum(-1)        # [B] public revealed-move count, no leak
slot_live  = (torch.arange(K) < 4) | (n_revealed[:,None] < 4)   # [B,K] — slot k live unless (k>=4 and set closed)
```
`slot_live` multiplies BOTH the opp-property block and the per-defender block (w-independent gate,
like `has_opp`). So at K=5: `n_revealed==4` → slots 0–3 (the 4 known moves) live, slot 4 zeroed;
`n_revealed<4` → all 5 live (still guessing the unknown slots, so surfacing extra hypotheses is the
point). At K≤4 the gate never fires (`k<4` always). The model reads "5th slot all-zero ⇒ the moveset
is pinned." (HP edge: a revealed HP counts as one revealed slot via its bare num; its typed
candidates still populate the live slots.)

**Other selection edge cases:**
- **Fewer than K believed candidates / ties:** `topk` returns the K highest `w_all`; padding slots
  are the next-highest with `w ≈ floor` (prior floor / sigmoid(−10)), so `w_topk ≈ 0` signals "not a
  real threat" and the head down-weights. `torch.topk` is deterministic; the belief floor breaks all
  but exact ties.
- **No opp active:** multiply the whole block by `has_opp` (zeros incl. gradient).
- **Fainted defender:** the per-defender part is multiplied by `defender_alive` (the opp-property
  part is per-move, gated only by `has_opp` × `slot_live`).

---

## 4. Identity representation & the latent table [DECISION]

**Emit the full 32-d latent, not a projection.** The model already learned this space (the move
network reads it; the v24 grading aux targets it), so feeding the raw latent means the heads read
move identity in coordinates they already understand — maximal transfer, zero new params, and the
RL gradient sharpens the *same* table. A learned projection would add a bottleneck + params to
compress a signal we don't need to compress (the GPU is ~86% idle). **Decision: full latent.**
(Alternative considered & rejected: project 32→16, halving 128→64 latent dims.)

**Building the candidate latent table** `move_latent_all [C, 32]`:

- `last_move_latent_table` is **`is_grad_enabled`-gated** (`:2448`) → `None` in rollout. The op runs
  in *both* rollout and training and its output feeds both heads, so it **cannot** depend on a
  training-only stash. → In `forward_internal`, when `damage_topk` is on, compute the table
  **unconditionally**: `self.pokemon_encoder.move_latent_encoder.latent_table(self.embeddings)`
  (one MLP over ~400 moves — negligible on an idle GPU) and pass it to the op as a new forward arg
  `move_latent_all`. Reuse the same tensor for the (still `is_grad_enabled`-gated) grading-aux stash
  so we compute it once.
- **Typed Hidden Power** (the [DECISION] HP point): the belief candidate axis appends 16 typed HP
  slots (`C = n_moves + 16`), but `latent_table()[237]` is a single collapsed HP latent (MOVE_ATTR[237]
  is all-zero; the move net distinguishes HP only via the per-slot *resolved type embedding*). So a
  selected HP slot must carry its **typed** latent. Build a `[16, 32]` typed-HP latent block the
  **same way the move network builds its per-slot HP latent** —
  `move_latent_encoder.forward(move_emb=move_embedding(237)·broadcast, type_emb=type_embedding(HP_TYPE_IDX),
  move_ids=237)` — so HP-Rock's latent carries Rock, HP-Ice's carries Ice, and the construction is
  identical to (consistent with) the move-net path. Then
  `move_latent_all = cat([latent_table[n_moves,32], typed_hp_latent[16,32]], 0)` aligns row-for-row
  with `w_all`'s candidate axis. The typed-HP damage already comes for free from the existing
  `_damage_rolls` over `mty_all` (which already carries the 16 HP type indices). The known gap — HP's
  BP (always 70) is absent from MOVE_ATTR[237] so the typed-HP latent lacks the BP axis — is the
  *same* gap the move net already has (intentional v24 limitation), so we stay consistent rather than
  introduce a second HP encoding.

---

## 5. Keep the channel-max block? — YES, add alongside [DECISION]

**Add the top-K block; keep the existing 12-scalar `_chan_max` worst-case (+ effect + secondary +
CB) blocks unchanged.** The worst-case max is the cleanest **switch-SAFETY** summary — one
"worst incoming" number per defender — and a guaranteed-correct strong-prior anchor that gives the
value head a direct KO-tail read without re-deriving it (`design_differentiable_damage_op.md` §4.3).
The top-K block adds the **discrete identity + per-pivot detail** the summary can't carry. They are
complementary, and the retained `_chan_max` preserves the dense belief gradient. This is the §4.3
hybrid, finally complete. (Rejected: replacing `_chan_max` — it would lose the fast safety read and
the dense gradient for no benefit.)

---

## 6. Dim budget — `K·53` (K=5 → 265) [DECISION resolved with the owner]

The owner's question — "does the dim count really matter for performance?" — and concern — "we
can't lose the immunity nuance." Answer: **the immunity capability is independent of the dim
budget** (damage immunity is in the per-pivot `high`/`pko`=0; status immunity is the new
`status_lands`=0 — §2.1), so I optimized for *capability per dim* rather than max width:

- **Dropped** the opp-level per-status secondary block (10/move) — the latent already carries the
  per-status identity (`MOVE_ATTR`), and the existing op-level `incoming_secondary[10]` aggregate
  stays. The realized, immunity-folded per-status threat is the *more useful* per-pivot
  `status_lands` (1/pivot).
- **Per-pivot block is lean** `{high, pko, status_lands}` (3) — the three scalars that decide a
  pivot. The roll band `low`/`crit` stays in the retained worst-case block.

`_DMG_TOPK(K) = K·53`. **K=5 → 265** dims to both heads (op grows 166 → 431 with outgoing+topk);
added projection params ~`265·512·2 ≈ 0.27M`, negligible. The 32-d **latent stays full width** (the
point of the feature; §4). One richness lever remains if an A/B wants the per-pivot roll band:
per-pivot `{low,high,crit,pko,status_lands}` (5) → `K·83` (K=5 → 415). **Recommendation: K=5,
`{high,pko,status_lands}`** — capability-complete (both immunity kinds) and lean.

---

## 7. Output layout & decode (the SoT mirror)

New module-level constants (`features_extractor.py`, with the other `_DMG_*`). K is now a
**per-model int** (`damage_topk_k`), so `_DMG_TOPK` is computed from K, not a constant:

```python
_DMG_TOPK_DEFAULT_K  = 5                                   # default K when enabled (owner: reason about the 4th/5th move)
_DMG_TOPK_ID_DIM     = MOVE_LATENT_DIM                     # 32 — the identity latent
_DMG_TOPK_META       = 3                                   # [belief_w, accuracy, is_phys]
_DMG_TOPK_MOVE       = _DMG_TOPK_ID_DIM + _DMG_TOPK_META   # 35  (opp-property, per move)
_DMG_TOPK_DMG_PER    = 3                                   # [high, pko, status_lands] per (defender, move)
def _dmg_topk_dim(k): return k*_DMG_TOPK_MOVE + TEAM_SIZE*k*_DMG_TOPK_DMG_PER   # k·53   (K=5 → 265)
# intra-move offsets (opp-property)
_DMG_TOPK_IDX_LATENT = 0
_DMG_TOPK_IDX_W      = _DMG_TOPK_ID_DIM                    # 32
_DMG_TOPK_IDX_ACC    = _DMG_TOPK_ID_DIM + 1               # 33
_DMG_TOPK_IDX_PHYS   = _DMG_TOPK_ID_DIM + 2               # 34
# per-(defender,move) offsets
_DMG_TOPK_IDX_HIGH   = 0
_DMG_TOPK_IDX_PKO    = 1
_DMG_TOPK_IDX_STATUS = 2
```

`DamageOperator.__init__(layout, outgoing=False, topk_k=0)`; `self.topk_k = topk_k`;
`self.out_dim = incoming_dim + (outgoing block if outgoing) + (_dmg_topk_dim(topk_k) if topk_k>0)`.

`decode_damage_block(row, *, outgoing, topk_k=0, team_size=TEAM_SIZE)` gains a `topk_k` kwarg; when
`>0` it decodes the trailing block from `base = incoming_dim + (_DMG_OUTGOING+_DMG_STATUS if
outgoing else 0)` into:

```python
out["incoming_topk"] = {
    "moves": [ {"latent": [...32...], "belief": w, "accuracy": a, "is_phys": p}
               for k in range(topk_k) ],                                   # opp-property
    "per_defender": [ [ {"high","pko","status_lands"} for k in range(topk_k) ]
                      for i in range(team_size) ],                         # pivot read
}
```

The op also **stashes detached side tensors for the prober** (never fed forward → off-path stays
byte-identical, same pattern as `last_opp_believed_mask`):
`last_topk_idx [B,K]` (candidate indices) and `last_topk_w [B,K]` (beliefs). These let the prober
print **exact** move names (idx < n_moves → `_move_maps()`; idx ≥ n_moves → `hiddenpower(<HP type>)`),
which beats a nearest-latent decode.

`out_gain` (the learnable per-channel adapter, `:~1452`) is extended to cover the `K·53` new
channels: the latent/belief/accuracy/is_phys/status_lands dims init to `1.0` (already in `[0,1]` or
LayerNorm-normalized); the `high`/`pko` damage dims mirror the per-channel scale init the existing
incoming block uses (`chip ≤ 1.5`, etc.). `×`-only (no bias) preserves the `has_opp`/
`defender_alive` zeros.

---

## 8. Flag, dependencies, versioning

**New flag** `--damage-topk K` (`int`, default `None`), extractor kwarg `damage_topk_k: int`
(**0 = off**). It is a **STRUCTURAL int** toggle — `out_dim` (hence both projection `in_features`)
scales with K, so *every distinct value, incl. 0↔N, is a weight-shape change* — gated in
`check_compatible` exactly like `opp_belief_cls_k` / `value_dist_bins` (one unconditional int
compare). `0` reproduces baseline byte-for-byte → **no `ARCH_SIGNATURE` bump**. Default K when
enabled = `_DMG_TOPK_DEFAULT_K = 5` (owner: "reasoning about the 4th and 5th move is human-expert
level"). K=5 surfaces the 5 most-believed candidates (a mon runs 4 moves, so the 5th is the
"surprise/uncertain-set" slot).

**[DECISION, owner-driven] Auto-enabled by `--unified-moves`, explicitly overridable.** In the
desugar block: `if unified_moves != "off" and args.damage_topk_k is None: args.damage_topk_k =
_DMG_TOPK_DEFAULT_K`. An explicit `--damage-topk K` always wins (including `--damage-topk 0` to
**disable** it under `--unified-moves` — that's the clean A/B knob: `--unified-moves both` vs
`--unified-moves both --damage-topk 0`). Outside the umbrella, `--damage-topk K` works directly. The
default pure-unified+topk run is just `--unified-moves both --spread-belief --unified-obs` (topk on
at K=5 automatically).

**Dependencies** (fail-loud at the CLI + `ValueError` at extractor build), enforced when
`damage_topk_k > 0`:
- requires `damage_op` (it extends the op), and
- requires `move_latent` (it gathers the move latent for identity; without it `move_latent_all`
  doesn't exist).
- (transitively, `damage_op` already requires `move_belief_mode revealed|both`.)
Under `--unified-moves` these are already satisfied (it sets `damage_op` + `move_latent`).

**Versioning recipe** (`MODEL_CONFIG_VERSION` 29 → 30; the touch-points the codebase uses for every
op toggle — mirror `opp_belief_cls_k`, the int precedent):
1. `model_version.py`: `MODEL_CONFIG_VERSION = 30`; add `damage_topk_k: int = 0` field; migration
   `if version < 30: data.setdefault("damage_topk_k", 0)`; `check_compatible` int gate (dedicated
   message); `from_layout_and_policy_kwargs` int read; v30 migration-note comment.
2. `snapshot.py`: `current_model_version(..., damage_topk_k=0)` signature + `ext_kwargs` splice;
   `arch_toggles_from_model` extracts `"damage_topk_k": int(getattr(fe, "damage_topk_k", 0))`.
3. `train_rl_agent.py`: argparse `--damage-topk` (int); desugar (auto-5 under `--unified-moves`);
   `_resolve("damage_topk_k", 0)`; dependency validation; `extractor_kwargs["damage_topk_k"]`;
   `_run_arch_toggles(...)` entry.
4. `features_extractor.py`: `__init__(..., damage_topk_k=0)`; store `self.damage_topk_k`; pass to
   `DamageOperator(layout, outgoing=..., topk_k=damage_topk_k)`; build/pass `move_latent_all` in
   `forward_internal` (unconditionally when `damage_topk_k>0`, NOT `is_grad_enabled`-gated);
   the op forward gains `move_latent_all=None`.
5. `snapshot_test.py`: extend `test_arch_toggles_from_model_extracts_flags` with the new key.

This threads `damage_topk_k` through all **4 opp-load sites** (self-play pool/sentinels via
`_run_arch_toggles`, eval workers + distill workers via `arch_toggles_from_model`, stable opponents)
so a topk-ON self-play run doesn't FATAL on its own sentinels. A current `--damage-op` checkpoint
won't load into a topk-ON op (projection `in_features` mismatch) — acceptable, no shims (the rapid-
iteration policy).

---

## 9. Prober (decode + display + names)

- `model.py::damage_op_view` — pass `topk_k=int(getattr(op, "topk_k", 0))` to `decode_damage_block`;
  read `last_topk_idx`/`last_topk_w`, resolve **exact names** (`_move_maps()` + typed-HP), and attach
  a `top_moves` list (name, belief, per-pivot {high, pko, status_lands}) to the returned dict.
- `engine.py` — already passes `damage_op` through `InvocationAnalysis`; the new keys ride
  `asdict(...)` for the `analyze` CLI for free.
- `app.py::_render_matchups` — extend the "incoming (op)" section with an **"opp likely (op):"**
  block: per top-K move, `name belief% → ▶active high%→KOpko% · <safest-pivot mon> high%/status` so
  a human reads *"opp likely: ice beam 62% → ▶Tyranitar 83%→KO90% · Skarmory 0%"* and *"thunder wave
  40% → ▶Tyranitar par · Flygon immune."* The safest pivot = the bench mon with min (pko, status) —
  the literal "which mon switches in safe" read (damage **and** status immunity).
- Tests: mirror `engine_test.py::test_damage_op_view_attached_when_model_exposes_it` for the new
  `incoming_topk`/`top_moves` keys (round-trips, JSON-serializable, names resolved, typed-HP names).

---

## 10. Test plan

**`damage_op_test.py`** (constructed scenarios via `_fake_ctx` + `_logits_*`, asserted against the
numpy oracle `incoming_damage.gen3_damage_max`/`p_ko` which is itself bridge-fuzz-validated):
- `test_topk_off_path_projection_dims_unchanged` — `damage_op` on, `damage_topk_k=0` == today;
  `k>0` adds exactly `_dmg_topk_dim(k)` to **both** projection dims (the byte-identity gate).
- `test_topk_surfaces_distinct_special_threats` — believe two distinct special moves (Ice Beam +
  Thunderbolt) with high mass; assert the top-K block surfaces **both** as distinct slots (distinct
  `last_topk_idx`, distinct latents, distinct per-defender damage), **not** a single max.
- `test_topk_pivot_safety_damage_immunity` — a move (Ice Beam / a Ground move) that OHKOs the active
  but does ~0 to a resistant/**immune** bench mon (Levitate / type-immune); assert per-defender
  `high`/`pko` show the OHKO on the active and **exactly 0** on the immune bench in the **same** move
  slot (the headline pivot read; the owner's Gengar/Flying/Normal cases).
- `test_topk_pivot_safety_status_immunity` — believe **Thunder Wave**; assert `status_lands` is high
  on a non-immune pivot and **exactly 0** on a **Ground** pivot (the owner's named case), while
  `high`/`pko` are 0 everywhere (it's a status move). Also Toxic→Steel, Will-O-Wisp→Fire,
  Leech-Seed→Grass read 0.
- `test_topk_typed_hp` — a believed typed HP (HP Ice) selected → its latent differs from HP Fire's
  and its per-defender damage reflects the typed effectiveness.
- `test_topk_grad_flows_to_belief_and_latent` — gradient from the top-K block reaches both
  `move_belief.move_head` and `pokemon_encoder.move_latent_encoder`.
- `test_topk_decode_for_prober` — `decode_damage_block(..., topk_k=K)` exposes `incoming_topk` with
  K moves + 6×K per-defender {high,pko,status_lands}; `last_topk_idx`/`last_topk_w` present & detached.
- `test_topk_meaningful_k_gate` — with K=5: all 4 opp-active moves revealed (`all_move_ids` 4
  nonzero) → the 5th slot (opp-property + per-defender) is **exactly 0**; with ≤3 revealed → the 5th
  slot is live (nonzero belief/damage). (The owner's "drop the 5th when all 4 known" rule.)
- `test_topk_dependency_guard` — building with `damage_topk_k>0` but no `move_latent` (or no
  `damage_op`) raises.
- `test_topk_leak_free` — output bit-identical with/without privileged label keys (mirror
  `test_op_is_leak_free_of_privileged_keys`).

**`damage_op_probe_fuzz_test.py`** (the authoritative omniscient gate; the existing 19 stay green):
- Add a constructed **pivot** scenario (a move super-effective on one of our mons, resisted/immune on
  another) and assert the op's top-K per-(defender) damage band for the measured move matches the
  omniscient oracle (`_TOL = 0.06`) for **both** defenders — confirming the top-K gather indexes the
  right `[B,6,C]` rows against a real sim. (The top-K damage reuses the already-validated
  `_damage_rolls` physics, so this checks the *gather/selection*, not new physics.)

**End-to-end:** `_run_roundtrip_test` (`[ModelVersion] Round-trip smoke test PASSED`) +
`python src/main/train_rl_agent.py --debug --use-showdown-bridge --steps 8000 --unified-moves both
--damage-topk` (serverless). Full unit suite green
(`pytest src/ -m "not integration and not e2e" -q`).

---

## 11. Principles honored / honest cons

- **Discrete-move reasoning is the point** — the output distinguishes individual moves *and* their
  per-pivot consequences, not just a magnitude.
- **Differentiable** — gradient reaches the move-belief logits (via `w_topk`) and the move-latent
  table (via `latent_topk`).
- **Marginalize, don't mean-field** — top-K keeps the distribution's shape (multiple distinct
  threats, bimodal immunity) instead of one blended scalar; the head aggregates. Damage stays
  decorrelated from belief.
- **No leak** — reads only the predicted belief + public `ctx`; the prober side stashes never feed
  forward (off-path byte-identical).

**Cons (the honesty gate, same as every op feature):**
1. **Learnable ≠ helps.** The block is wired + differentiable, but whether the policy *uses* discrete
   identity to switch better is unmeasured. Gate: a fresh-run A/B (`--unified-moves both` vs
   `+ --damage-topk`) where **under-switching falls** (prober `human_agreement` switch-rate;
   `falsify-scan` surprise-OHKO / wrong-pivot crater share ↓) **AND win-rate non-regresses**. The
   `project_incoming_damage_outcome` precedent (the OHKO belief was wired + critic-read + calibrated
   but the *policy* under-switched) is the named risk — this feature attacks exactly that
   ("which pivot is safe"), so it's the most direct test of whether the obs/representation was the
   bottleneck or the policy/reward was.
2. **Redundancy with `_chan_max` + the per-slot move latent.** The worst-case block + the move-net's
   own per-slot latent already carry pieces of this; the top-K's value is the *cross-defender, single-
   identity* view neither has. Marginal, unmeasured (role-probe caution: "decodable ≠ helps").
3. **Dim cost** (276, both heads) for an unproven lever — mitigated by the OFF byte-identity and the
   §6 levers.

---

## 12. References

- The op: `src/agents/model/features_extractor.py` `DamageOperator` (`_damage_rolls` `:1516`,
  `_chan_max` `:1464`, candidate axis `:1842–1860`, provenance gather `:1887–1905`,
  `decode_damage_block` `:1978`, `out_gain` `:~1452`); `MoveLatentEncoder` `:439–476`
  (`latent_table` `:471`).
- Tables: `damage_tables.py` (`MOVE_SECONDARY`, `ABILITY_SECONDARY_MULT`, `HP_TYPE_IDX`,
  `MOVE_ATTR`/`MOVE_ATTR_COLS`).
- Versioning: `model_version.py` (`MODEL_CONFIG_VERSION=29`, `ARCH_SIGNATURE`),
  `snapshot.py` (`current_model_version` / `arch_toggles_from_model`),
  `train_rl_agent.py` (`_run_arch_toggles`, the op flag flow + validation `:1172–1276`).
- Prober: `src/main/prober/{model.py::damage_op_view, engine.py, app.py::_render_matchups,
  engine_test.py}`; `_move_maps()` / `_norm_move()`.
- Tests: `damage_op_test.py`, `poke_env_gaps/damage_op_probe_fuzz_test.py` (+ `damage_probe.js`),
  `incoming_damage.py` (numpy oracle).
- Design lineage: `design_differentiable_damage_op.md` (§4 don't-collapse, §4.3 hybrid top-k +
  anchor), `design_unified_move_system.md` (the move latent + secondary machinery this builds on),
  `src/agents/model/CLAUDE.md` (v19–v29 op notes). Memory: `project_gpu_damage_op`,
  `project_unified_move_system`, `project_incoming_damage_outcome` (the under-switching precedent).
```
