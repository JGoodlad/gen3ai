# bf16 canary FATAL on trained weights — diagnostic verdict

Reads: 8. Arms: **E32** eager, trunk fp32 (the canary's fp32 REFERENCE, `fp32_scope`); **Ebf** eager, trunk bf16 (the canary's yardstick); **Cbf** compiled (Inductor), trunk bf16 — the production compile (`compile_regions.install`); **C32** compiled (Inductor), trunk fp32 — a second learner at fp32, the same weights; **Cbf_emul** compiled (Inductor) bf16 with `emulate_precision_casts` (rounds where eager rounds); **Cbf_aot** compiled bf16 through `aot_eager` (dynamo + AOTAutograd, eager kernels: no Inductor codegen); **Cbf_pheager** compiled (Inductor) bf16 with the POINTER HEAD run eager (`torch.compiler.disable` on its forward: a graph break, fullgraph=False) — a third learner, the same weights

## Per checkpoint (slice 0, `pointer_head.move_proj.weight`; d = relative L2 to the second arm)

| read | d(Ebf,E32) | d(Cbf,E32) | d(C32,E32) | d(Cbf_emul,Ebf) | d(Cbf_aot,Ebf) | d(Cbf_pheager,Ebf) | canary ratio | verdict | Cbf flips | Cbf frozen-branch d | min switching margin | relocated ratio / verdict | null P(ratio>K) Ebf pairs / Cbf | relocated null | kappa | pre-act p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu_15M | 0.0121 | 0.236 | 1.6e-06 | 0.233 | 0 | 0.233 | 19.5 | refused | 32 | 0.00734 | 0.00143 | 0.955 / pass | 0.214 / 0 | 0 / 0 | 2.93 | 6.26 |
| gpu_30M | 0.0165 | 0.0198 | 2.07e-06 | 0.0115 | 0 | 0.0115 | 1.2 | pass | 0 | 0.0129 | 0.0333 | 1.23 / pass | 0 / 0 | 0 / 0 | 3.05 | 7.17 |
| gpu_45M | 0.0201 | 0.0157 | 2.24e-06 | 0.0167 | 0 | 0.0167 | 0.778 | pass | 0 | 0.00536 | 0.0027 | 0.993 / pass | 0 / 0 | 0 / 0 | 2.62 | 8.61 |
| gpu_60M | 0.0189 | 0.0309 | 2.77e-06 | 0.0197 | 0 | 0.0197 | 1.64 | pass | 0 | 0.00476 | 0.0296 | 1.5 / pass | 0 / 0 | 0 / 0 | 3.15 | 10.1 |
| gpu_66M | 0.293 | 0.295 | 2.81e-06 | 0.0174 | 0 | 0.0174 | 1.01 | pass | 32 | 0.00661 | 0.00495 | 1.16 / pass | 0 / 0 | 0 / 0 | 2.55 | 11.2 |
| gpu_70M | 0.0289 | 0.0218 | 3.79e-06 | 0.0235 | 0 | 0.0235 | 0.754 | pass | 0 | 0.00878 | 0.0142 | 1.09 / pass | 0.107 / 0 | 0 / 0 | 2.32 | 10.6 |
| gpu_72M | 0.0278 | 0.0325 | 3.4e-06 | 0.0223 | 0 | 0.0223 | 1.17 | pass | 0 | 0.0117 | 0.0521 | 1.2 / pass | 0.00302 / 0 | 0 / 0 | 2.75 | 10.7 |
| gpu_fatal | 0.0282 | 0.403 | 2.48e-06 | 0.404 | 0 | 0.404 | 14.3 | refused | 32 | 0.0084 | 0.00451 | 0.655 / pass | 0.167 / 0.156 | 0 / 0 | 2.62 | 11 |

## Verdict (fill from the table; each line names the read that decides it)

- [ ] **Reproduced** on the canary's weights (`gpu_fatal`: `canary_replay` / `canary_verdict` refused, naming the run's parameters at its numbers): ...
- [ ] **The clip-boundary flip** (the CPU finding): Cbf flips a row's PPO clip branch against E32 (`implied.Cbf.branch_flips`), its FROZEN-branch error is in the no-flip mode (`frozen_branch_vs_ref`), the null's bad mode is the flip seeds (`null.flip_attribution`): ...
- [ ] **(i) precision-independent miscompile**: d(C32,E32) on the pointer head vs the fp32 bar (`fp32_rule.over_bar` empty = KILLED): ...
- [ ] **(ii) Inductor bf16 numerics**: Cbf's frozen-branch error and `implied.Cbf.self` (its backward against its own forward) vs Ebf's; Cbf_emul / Cbf_aot / Cbf_pheager vs Ebf: ...
- [ ] **(iii) a yardstick unusually small**: d(Ebf,E32) at the FATAL vs `null.errors.Ebf` (its no-flip mode): ...
- [ ] **(iv) the fixture**: the rows within rounding of their switching clip boundary (`ratio.switching_margin_logp`); the RELOCATED fixture (the T36 fix, production's) passes and its null has P(ratio > K) = 0: ...
- [ ] **Recommendation for X26**: ...
