# Stash contracts, and the compile spelling

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

## The op's and the extractor's SIDE VALUES (`gen3_op_stashes_v1` / `gen3_extractor_stashes_v1`)

## The op's SIDE VALUES have ONE container (`gen3_op_stashes_v1`)

Every per-forward stash the op exposes (`last_topk_idx`, `last_pair_cells`, `last_w_all`,
`last_out_pko`, `last_raw_block`, `last_tensors`, …) lives in ONE `OpStashes` dataclass that the
forward replaces at ENTRY — so no stash can carry a previous batch, uniformly (three different
clearing conventions used to coexist, and the top-K trio had none). **Reads** use the `last_*`
properties (the documented surface); **writes** go through `op.stash.<field>` — writing a
`last_*` name raises. When you add a stash: add the dataclass field with its shape comment, write
via `self.stash`, and never add a bare `self.last_x = …` attribute. The extractor's tuple
stashes are typed the same way (`PointerInputs`, `ThresholdProbs` — NamedTuples, so positional
unpacks keep working).

**The EXTRACTOR's side values follow the same contract (`gen3_extractor_stashes_v1`).** Every
per-forward stash `Gen3FeaturesExtractor` exposes — `last_pointer_inputs`, the α/β intent trio,
every belief publication (`last_move_belief_logits`, `last_spread_*`, `last_item_logits`,
`last_hp_type_logits`, `last_belief_logits`, `last_opp_believed_mask`, `last_opp_active_local`,
`last_move_latent_table`), `last_damage_block`, `last_value_pooled`, `last_win_prob_logits`,
`last_value_dist_logits`, the internal T0→T1/T2 hand-offs (`t0_species_probs`,
`entity_latent_table`, `thresh_probs`) and the LIVE `belief_supervision` dict — lives in ONE
`ExtractorStashes` dataclass that `forward_internal` replaces at ENTRY. Same rules: **reads** on
the `last_*` properties (every cross-module consumer — the policy's pointer head + dist critic,
`instrumented_ppo`, the prober, inference — uses the typed properties, never `getattr(..., None)`);
**writes** through `fe.stash.<field>`; a stray write to a `last_*` name raises. When you add a
stash: add the dataclass field with its shape comment, write via `self.stash`, add the read-only
property if it has an external consumer — never a bare `self.last_x = …` attribute. The
publication/stop-grad semantics (`_publish_belief` / `belief_supervision()`) are unchanged — the
supervision dict rides the container, so its per-forward clear IS the entry replacement.
Boundary rule: each producer module owns its own stash surface — the op keeps `OpStashes`,
`PokemonEncoder` keeps `last_move_tokens` (written unconditionally every encoder forward, read in
the same extractor forward) — a submodule never writes into its parent's container. Related
fail-loud: `Gen3DualHeadMaskablePolicy._critic_value` under `--value-from-dist` RAISES when the
dist head/logits are missing or batch-stale instead of falling back to the FROZEN scalar
`value_net` (the silently-wrong-critic shape v89 exposed). Gate: `extractor_stashes_test.py`.

## Why the species posterior is spelled `log_softmax(...).exp()` — the whole diagnosis

`torch.softmax` over the last dim of the `[B,6,n_species]` logits lowers to a numerator buffer plus a
`[B,6,1]` denominator, and the Inductor **CPU** scheduler then trips `AssertionError: buf<N>` trying to
fuse the division. That single op was the reason `--compile-opponents` used to set
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
