# The static-port weight-mapping identity proof (2026-10-09)

**Question.** The static-token screen (`designs/endstate/design_static_tokens.md` §8.2) runs PINNED at P_st
`6c6d2e09` (config v143, pre-break). `--token-encoding static` is being ported onto main (`2e357971`, config
v146). Between the two commits landed the X5 version break v144 (parts 1–5), v145 (the mon-tied gain), v146
(`--op-reduction`), the poke-env deletions and `957d4dbd` (the update's host-sync batching). The question: does
static at HEAD, with every post-pin lever OFF, reproduce the pin's static forward on the SAME WEIGHTS? Where a change is
an identity only under a condition on the weights, which condition?

**Configuration** (both sides, `_static_build.static_args`, asserted on the BUILT modules): `production_args()`
(= a fresh `--arch production`) + `token_encoding static`, belief `fixed_mass` (a flag at the pin, the only
belief representation at HEAD), readout `tower`, `move_resolution off`, `value_threat_inject` on,
`speed_physics off`; at HEAD also `obs_facts off`, `op_reduction max`. Built as the K9 learner golden's seeded,
NAME-KEYED-perturbed learner (`learner_golden.build_learner(args, perturb_keyed=True)`), CPU, one thread,
torch 2.8.0+cu126. Rows: the pin's committed `learner_golden_buffer_fixed_mass.npz` (64 real rows, sha256
`4b48eaf0…`, obs 2761), fed to HEAD with the OBS-FACTS block appended.

**Method** (the precedent's, `version_break_identity_2026-10-07/`, adapted):

1. `dump_surface.py` (each side) + `diff_surfaces.py`: build facts, state_dict keys and shapes, the model's
   observation layout (`fe.layout`) and every `agents.observation.constants` int, and the static encoder's
   `s_in` column blocks located on real rows. `surface_diff_head.json` is the pin vs HEAD diff.
2. `capture_reference.py` (at the pin, run with `--src` so it imports the pin's code and writes nothing there):
   one seeded init, CONDITIONED into each identity's subspace, one variant per combination of conditions;
   for each, the conditioned `state_dict` and `evaluate_actions_functional` on the 64 rows; for `tied_pre1`,
   also one K9 `train()` with the op gains FROZEN. Weights in `~/.cache/gen3ai/static_port_identity/ref/`
   (97 MB, NOT committed); every file's sha256 is in `reference_meta.json`.
3. `compare_mapped.py` (at the target): build, MAP the reference through the declared mapping, then
   `load_state_dict(strict=True)` (no undeclared key may be dropped, missing or reshaped), then the forward under two
   OBS-FACTS fills (zeros and a seeded N(0, 9) fill), compared with `torch.equal`, plus the update with `--update`.
   `--readd-flat-bias` is the precedent's F16b CONTROL.
4. `emulate_port.py`: tests the `static_port` mapping before the parent's branch exists. It emulates c1 and c2
   in-process at HEAD and runs the same mapping and compare.

| condition | what it sets at the pin | why |
|---|---|---|
| `tie` (a) | each v145 tie group of the op's 138 `out_gain`s (`out_gain_groups_v145.json`, read off HEAD's `_out_gain_tie`, re-checked by the compare) = its float64 mean | v144 part 4 + v145 tie 138 → 29 gains. The per-slot inits are NOT group-equal (the perturbation), so `raw` cannot map exactly |
| `pre1` (b) | the 3 groups HEAD's part-5 consumer reads PRE-gain (`intent_conditional`: `out_move[1]` the high roll, `out_p_outspeed`, `out_secondary[4]` flinch; 9 positions) = 1.0 | the pin reads them post-gain; at gain 1 the two reads are bitwise equal |
| `one` (b') | every gain = 1.0 | the blunt form of tie + pre1 |
| `fb0` | `flat_intent_head.out.bias` = 0 | F16b's forward half: `s + 0 == s` exactly |
| `c1` (parent) | `pokemon_encoder.role_encoder.0.weight[:, 78:94]` (the type2 block) = `[:, 62:78]` (type1) | `s_in` = species 32 · stats 13 · item 16 · item-known 1 · **type1 16 @62** · **type2 16 @78** · … (`StaticTokenEncoder.encode`; both windows located on real rows) |
| `c2` (parent) | `op_content.outgoing_proj` weight and bias = 0 | its zero-init Deep-Sets replacement must then also contribute exactly 0 |

## Result (target = `2e357971`; `result_head.json`, `result_head_control.json`, `result_emulated_port_*.json`)

`surface_diff_head.json`: **the 2761-dim prefix is IDENTICAL.** Every shared layout entry and every observation
constant is equal. The only exceptions are the three entries that ARE the total width (`total_dim`, `base_dim`,
`parts.reactive.end`: 2761 → 2845). HEAD adds only `obs_facts*` / `OFFSET_OBS_FACTS = 2761`. State_dict: the pin
has exactly 11 keys HEAD lacks (×3 aliases) — F1's value tower (10 keys, 592,129 parameters) and F16b's bias (1).
HEAD has none the pin lacks. The ONE reshape is `damage_op.out_gain` 138 → 29. Policy parameters: 3,061,302 →
2,469,063 (−592,239 = 592,129 + 1 + 109).

| variant (`--mapping head`) | forward vs pin (64 rows) | with the F16b control | the update (gains frozen) |
|---|---|---|---|
| `raw` | values 1.7e-4, log π 2.3e-3, masked log π 3.3e-3, entropy 1.0e-4 (tie not exact) | same | — |
| `tied` | values BITWISE; log π 5.3e-5, masked 2.8e-4, entropy 1.4e-6 | same (part 5) | — |
| `tied_fb0` | as `tied` | as `tied` | — |
| `tied_pre1` | values BITWISE; log π / masked 2.4e-7, entropy 7.2e-7 (F16b) | **BITWISE** | 30 groups differ ≤ 2.4e-7 (≤ 3.0e-8 off `embeddings`), losses differ; **control: every parameter BITWISE, all 17 pinned losses EQUAL** |
| **`tied_pre1_fb0`** | **BITWISE** | BITWISE | — |
| **`tied_pre1_fb0_c1c2`** | **BITWISE** (c1 / c2 conditioning does not by itself break pin↔HEAD) | BITWISE | — |
| `one_fb0` | BITWISE | BITWISE | — |

In every variant the two OBS-FACTS fills gave **bitwise-equal** forwards, so with `obs_facts off` nothing reads
the appended block.

`emulate_port.py` (`static_port` mapping, variant `tied_pre1_fb0_c1c2`, HEAD + the emulated c1 and c2):
- `--c1-spelling concat` (the CONTROL, the pin's arithmetic on the mapped 162-wide weight): **BITWISE**. The
  mapping works, and c2 (a zero-output Deep-Sets in place of the zeroed `outgoing_proj`) is bitwise.
- `--c1-spelling sum` (`emb(t1)+emb(t2)`, the parent's form): values 1.2e-7, log π / masked 8.3e-7,
  entropy 6.0e-7. This is the FIRST Linear's summation order (178 vs 162 terms, `e1+e2` pre-added).

## Classification (the static path, these 64 rows)

| change | class | evidence |
|---|---|---|
| v144 part 1 (blob deleted; fixed_mass the only belief) | BITWISE on fixed_mass (the static arm never ran blob) | every bitwise row |
| F1 (value tower deleted) | BITWISE: no loss read it, and it received no gradient | the dropped keys, then the forward and update bitwise |
| F6a (`max_by_index` for every op max, pair_reduce, the hypothesis tails) | BITWISE forward; the update differs only on an EXACT tie with a nonzero upstream gradient (none here) | the forward and the control update bitwise |
| F7a (the discarded speed-spread lookups deleted) | BITWISE | as above |
| F16b (the flat pointer's shared bias deleted) | identity up to fp32 rounding (a logit shift, log π ~1 ulp) | `tied_pre1` 2.4e-7, bitwise under the control AND under `fb0`; update ≤ 2.4e-7, bitwise under the control |
| part 4 + v145 (`out_gain` tied 138 → 29) | BITWISE iff the gains are group-equal (`tie`). Not identity on unequal per-slot gains (`raw`: log π 2.3e-3). With TRAINABLE gains the update differs through the gains' own Adam steps (the v145 proof), so frozen here | `tied` values bitwise; `tie_exact` asserted |
| part 5 (consumers read PRE-gain) | BITWISE iff the 3 groups `intent_conditional` reads (out high roll, P(first), flinch) have gain 1.0 (`pre1`). Otherwise NOT identity; it cannot be made identical on trained gains ≠ 1 (a semantic change) | `tied` vs `tied_pre1`: log π 5.3e-5 → bitwise (control) |
| part 3 (OBS-FACTS appended, `--obs-facts off`) | BITWISE; the block is never read | prefix check + the two fills equal in every variant |
| v146 `--op-reduction max` | BITWISE | every bitwise row (`max` runs nothing new) |
| poke-env deletions (T27 P6) | no model change on this path: BITWISE | as above |
| `957d4dbd` (the update's host-sync batching) | BITWISE update on CPU | the control update: every parameter and every loss EQUAL |
| (parent) c2: `outgoing_proj` → zero-output Deep-Sets | BITWISE iff the pin's `outgoing_proj` is zero AND the new output layer is zero-init (`x + 0 == x`) | emulated, both spellings |
| (parent) c1: type concat → `emb(t1)+emb(t2)` | identity up to fp32 rounding iff type2 block == type1 block; cannot be bitwise (the first Linear's summation order) | emulated `sum` ≤ 8.3e-7 log π; `concat` control BITWISE |

**Verdict.** On conditions `tie` + `pre1` + `fb0`, static at HEAD reproduces the pin's static forward bit for
bit. With `fb0` replaced by the F16b control, one K9 update (gains frozen) is also bit for bit. Every post-pin
change is an exact identity on that subspace except F16b's mathematically exact shift invariance (~1 ulp,
isolated by a bitwise control). Part 5 is a semantic change, an identity only at gain 1 on the 3 channels it
re-routes.

## Rerun

Python `/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3` (`$PY`), run from anywhere. Use `-E`: the
`--src` flag puts the named checkout's `src/` first. The reference does not change; re-capture only to re-prove it.

```bash
D=designs/research_state/measurements/static_port_identity_2026-10-09   # in the checkout that holds this folder
REF=~/.cache/gen3ai/static_port_identity/ref
# (once) the reference, at the pin — writes only REF and $D/reference_meta.json:
$PY -E $D/capture_reference.py --src <pin 6c6d2e09 checkout>/src $REF
# the surface / prefix check of a target:
$PY -E $D/dump_surface.py --src <target>/src /tmp/surface_target.json
$PY $D/diff_surfaces.py ~/.cache/gen3ai/static_port_identity/surface_pin.json /tmp/surface_target.json
# HEAD-shaped target (no c1 / c2 in the code):
$PY -E $D/compare_mapped.py --src <target>/src $REF --mapping head --update [--readd-flat-bias]
# the PARENT's branch (c1 + c2 in the code; --mon-hazard-cost / --move-actor-state OFF, absent from the state_dict):
$PY -E $D/compare_mapped.py --src <branch>/src $REF --mapping static_port --variants tied_pre1_fb0_c1c2 \
    [--readd-flat-bias] --out /tmp/result_static_port.json
```

Under `static_port` only the c1+c2-conditioned variant maps, since the others refuse c1's assertion. Expected at
the parent's branch: c1's fp32 rounding (≤ ~1e-6 on log π; this folder's emulated `sum` read 8.3e-7) and
everything else bitwise. To isolate c1, the parent can add a concat spelling of its new Linear in a script, as
`emulate_port.py --c1-spelling concat` does. A new key under `op_content.` keeps the target's own init, except its
LAST `nn.Linear` (the output layer), which the mapping CONDITIONS to zero (the K9 build perturbs every parameter, so a
zero-init layer holds a perturbed copy in the target state_dict; zero is its code init and the twin of capturing the
reference's `outgoing_proj` at 0). Any other new key REFUSES the mapping (strict load). The compare does not support an `--update` under `static_port` (c1's assertion fails on an updated
state).

## Result at the port branch (`gen3_static_port_v1`, config v147; `result_static_port_branch.json`)

Run from the port's worktree after the build (`--mapping static_port --variants tied_pre1_fb0_c1c2`, both new facts
OFF so their projections are absent from the state_dict), CPU, one thread:

| variant | values | log π / masked log π | entropy | OBS-FACTS fills equal | tie exact |
|---|---|---|---|---|---|
| `tied_pre1_fb0_c1c2` | 1.19e-7 | 8.34e-7 | 5.96e-7 | yes | yes |

These are exactly `emulate_port.py --c1-spelling sum`'s numbers: c1's first-Linear summation order, whose concat
spelling is the BITWISE control. c2 (the real Deep-Sets module: a bias-free per-move `Linear(6 → 32)` + ReLU, summed
over our moves, then the zero-init bias-free `Linear(32 → 128)`) contributes exactly 0 here. So the port is the pin's
static forward on these weights up to c1's fp32 rounding, and bitwise on every other post-pin change under the
conditions above. (The mapping's c2 step was adjusted at this run to CONDITION the target's output layer to zero
rather than assert it, for the reason given under Rerun.)

## What it does not cover (FINDINGS)

- The proof covers only the 64 K9 rows and the perturbed init. F6a's exact-tie gradient rule and F16b's rounding
  can show elsewhere: the class is known and bounded, but it is not exhaustively enumerated.
- CPU eager only. CUDA and compiled numerics are not covered (CPU-only brief).
- The update is compared with the op gains FROZEN. Trainable gains differ by construction (v145's own proof).
- Part 5 cannot be an identity for a TRAINED pin checkpoint whose 3 consumer-read gains ≠ 1. A screen checkpoint
  moved to HEAD changes `intent_conditional`'s inputs.
- c1 is an identity up to fp32 summation order, never bitwise, so the parent's run will read ~1e-7 deltas on
  every output. That is expected, not a defect.
