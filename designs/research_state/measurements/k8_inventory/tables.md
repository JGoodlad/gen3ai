# K8 inventory — generated tables

`python -m main.compile_inventory report <the five dirs in README.md's Provenance>`, verbatim (absolute source paths are under `~/gen3ai_archive/k8_inventory/`; the JSON copies beside this file are the same results).

### By area, every trace given (graphs / ops in graphs / break sites (events) / skipped frames / eager fallbacks / recompiles)

| area | 2.5.1 today | 2.5.1 whole_step | 2.8.0 today | 2.8.0 whole_step |
|---|---|---|---|---|
| extractor | 4 g / 33317 ops / 2 brk (5 ev) / 0 skip / 0 fb / 5 rc | 10 g / 41539 ops / 2 brk (15 ev) / 0 skip / 0 fb / 8 rc | 3 g / 24933 ops / 1 brk (1 ev) / 0 skip / 0 fb / 2 rc | 1 g / 8311 ops / 0 brk (0 ev) / 0 skip / 0 fb / 0 rc |
| heads | — | 9 g / 144 ops / 0 brk (0 ev) / 0 skip / 0 fb / 4 rc | — | 7 g / 25090 ops / 0 brk (0 ev) / 0 skip / 0 fb / 1 rc |
| masking+distribution | — | 8 g / 19 ops / 3 brk (19 ev) / 0 skip / 0 fb / 7 rc | — | 8 g / 19 ops / 1 brk (17 ev) / 0 skip / 0 fb / 7 rc |
| train() fold (inline) | — | 0 g / 0 ops / 1 brk (1 ev) / 0 skip / 0 fb / 0 rc | — | 0 g / 0 ops / 2 brk (2 ev) / 0 skip / 0 fb / 0 rc |
| loss terms | — | 335 g / 1117 ops / 123 brk (492 ev) / 4 skip / 2 fb / 330 rc | — | 413 g / 1266 ops / 104 brk (357 ev) / 4 skip / 2 fb / 234 rc |
| diagnostics | — | 117 g / 4262 ops / 75 brk (154 ev) / 8 skip / 0 fb / 64 rc | — | 141 g / 2423 ops / 71 brk (154 ev) / 9 skip / 0 fb / 48 rc |
| optimizer | — | 3 g / 15120 ops / 1 brk (1 ev) / 0 skip / 0 fb / 0 rc | — | 3 g / 15120 ops / 0 brk (0 ev) / 0 skip / 0 fb / 2 rc |
| rollout buffer | — | 9 g / 93 ops / 0 brk (0 ev) / 0 skip / 0 fb / 7 rc | — | 9 g / 93 ops / 0 brk (0 ev) / 0 skip / 0 fb / 7 rc |
| other | — | 23 g / 0 ops / 1 brk (2 ev) / 0 skip / 4 fb / 0 rc | — | 64 g / 0 ops / 0 brk (0 ev) / 0 skip / 2 fb / 0 rc |

### Trace — torch 2.5.1+cu121 (cpu)
- source: `/home/goodlad/gen3ai_archive/k8_inventory/20260930_170750_trace_cpu_torch2.5.1/trace_result.json`
- torch `2.5.1+cu121`, device `cpu`, matmul `highest`, git `89679ddbdc`
- checkpoint `/home/goodlad/dev/gen3ai/models/ai_v14_02_lbat_ctrl/final_model.zip` (sha256 `06efae2c44d6`), buffer `/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl`
- geometry: 192 rows, micro 64 × 3/epoch, accum 2, epochs 2, diagnostics_every 1
- peak RSS by stage (MB; the 12 GB-capped scope): startup 873, today:update1 1192, today:update2 1186, today:rollout 1233, today_update3:profiled_update 2365, today_update3:profile_export 2503, whole_step:update1 2591, whole_step:update2 2612, whole_step:rollout 2666, whole_step:update3 2666, whole_step_update3:profiled_update 3300, whole_step_update3:profile_export 3300, whole_step_update3:gzip 3300; classify (separate process) 812
- ObservationDebugger: api present False, attached by the run False (dropped for both views, as the compile path drops it); micro-check: {"present": false}

**Totals per view** (all four calls):

| view | graphs | graphs ending in a break | break sites | break events | …in skipped frames | skipped frames | eager fallbacks | recompiles | frames traced | ops in graphs |
|---|---|---|---|---|---|---|---|---|---|---|
| today | 4 | 0 | 2 | 5 | 0 | 0 | 0 | 5 | 4 | 33317 |
| whole_step | 514 | 475 | 206 | 684 | 12 | 12 | 6 | 420 | 271 | 62294 |

**today: per call** (a NEW graph/break is counted at the call that first traced it; a cached call adds nothing):

| call | new graphs | break events | skipped frames | recompiles | eager fallbacks |
|---|---|---|---|---|---|
| update1:first(diagnostics) | 2 | 4 | 0 | 3 | 0 |
| update2:diag_skipped | 0 | 0 | 0 | 0 | 0 |
| rollout:eval_no_grad | 1 | 1 | 0 | 1 | 0 |
| update3:diag_skipped(steady) | 1 | 0 | 0 | 1 | 0 |

**today: by component**:

| component | ops in graphs | break sites | skipped-frame sites | eager-fallback sites |
|---|---|---|---|---|
| extractor | 33293 | 2 | 0 | 0 |
| model (other) | 24 | 0 | 0 | 0 |

**today: by source file** (every site; top reason by events):

| file | sites | events | kinds | top reason |
|---|---|---|---|---|
| agents/model/forward_guard.py | 1 | 4 | graph_break | 'inline in skipfiles: WeakKeyDictionary.get \| get /home/goodlad/miniconda3/envs/gen3ai_stable/lib/python3.11/w |
| agents/model/encoders.py | 1 | 1 | graph_break | unsupported operator: aten._native_multi_head_attention.default (see https://docs.google.com/document/d/1GgvOe |

**today: sites** (dynamo's reason, verbatim first line):

| component | kind | site | reason | events | calls | detail |
|---|---|---|---|---|---|---|
| extractor | graph_break | agents/model/forward_guard.py:57 forward_guard_for | 'inline in skipfiles: WeakKeyDictionary.get \| get /home/goodlad/miniconda3/envs/gen3ai_stable/lib/python3.11/weakref.py, skipped according trace_rules | 4 | update1 |  |
| extractor | graph_break | agents/model/encoders.py:287 forward | unsupported operator: aten._native_multi_head_attention.default (see https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/e | 1 | rollout |  |

**today: recompiles** (first failing guard):

| call | function | site | guard | n |
|---|---|---|---|---|
| update1:first(diagnostics) | forward | agents/model/features_extractor.py:267 | GLOBAL_STATE changed: grad_mode | 1 |
| update1:first(diagnostics) | forward_guard_for | agents/model/forward_guard.py:51 | GLOBAL_STATE changed: grad_mode | 1 |
| update1:first(diagnostics) | torch_dynamo_resume_in_forward_at_281 | agents/model/features_extractor.py:281 | GLOBAL_STATE changed: grad_mode | 1 |
| rollout:eval_no_grad | torch_dynamo_resume_in_forward_at_281 | agents/model/features_extractor.py:281 | GLOBAL_STATE changed: grad_mode | 1 |
| update3:diag_skipped(steady) | _forward_unguarded | agents/model/features_extractor.py:287 | tensor 'L['obs']['observation']' size mismatch at index 0. expected 48, actual 64 | 1 |

**today: steady-state update, host ATen ops compiled vs eager** (top-level ops; profiled no stacks — eager ops by phase only): compiled 62513, eager 62939 — compiled share 49.8%

| phase | compiled ops | eager ops | compiled host ms | eager host ms |
|---|---|---|---|---|
| backward | 15834 | 2309 | 85788.9 | 4605.6 |
| batch | 0 | 261 | 0.0 | 14.5 |
| capacity | 0 | 0 | 0.0 | 0.0 |
| epoch_end | 0 | 2986 | 0.0 | 3930.5 |
| forward | 35298 | 381 | 32055.2 | 482.8 |
| kl | 0 | 30 | 0.0 | 0.2 |
| logging | 0 | 11820 | 0.0 | 12655.9 |
| loss | 0 | 4090 | 0.0 | 4119.9 |
| noise_base | 0 | 992 | 0.0 | 315.0 |
| noise_probe | 0 | 0 | 0.0 | 0.0 |
| optim | 0 | 2738 | 0.0 | 3211.8 |
| probes | 11381 | 37332 | 13709.8 | 40660.4 |
| setup | 0 | 0 | 0.0 | 0.0 |
| tail | 0 | 0 | 0.0 | 0.0 |

Eager ops by component:

| component | eager ops | eager host ms |
|---|---|---|
| (no python stack) | 60636 | 65391.1 |
| backward (eager autograd) | 2303 | 4605.6 |

**whole_step: per call** (a NEW graph/break is counted at the call that first traced it; a cached call adds nothing):

| call | new graphs | break events | skipped frames | recompiles | eager fallbacks |
|---|---|---|---|---|---|
| update1:first(diagnostics) | 474 | 626 | 12 | 392 | 6 |
| update2:diag_skipped | 10 | 22 | 0 | 4 | 0 |
| rollout:eval_no_grad | 12 | 15 | 0 | 9 | 0 |
| update3:diag_skipped(steady) | 18 | 21 | 0 | 15 | 0 |

**whole_step: by component**:

| component | ops in graphs | break sites | skipped-frame sites | eager-fallback sites |
|---|---|---|---|---|
| distribution+masking | 35 | 3 | 0 | 0 |
| extractor | 25153 | 2 | 0 | 0 |
| heads | 227 | 0 | 0 | 0 |
| loss+diagnostics | 4782 | 145 | 11 | 1 |
| model (other) | 16766 | 52 | 1 | 0 |
| optimizer | 15120 | 1 | 0 | 0 |
| other | 118 | 3 | 0 | 1 |
| rollout buffer | 93 | 0 | 0 | 0 |

**whole_step: by source file** (every site; top reason by events):

| file | sites | events | kinds | top reason |
|---|---|---|---|---|
| agents/model/opp_intent.py | 53 | 251 | graph_break,skip_frame | dynamic shape operator: aten.bincount.default; to enable, set torch._dynamo.config.capture_dynamic_output_shap |
| agents/training/belief_bank.py | 58 | 165 | graph_break,skip_frame | Graph break due to unsupported builtin None.list.append. This function is either a Python builtin (e.g. _warni |
| agents/training/instrumented_ppo/value_terms.py | 16 | 80 | graph_break | Tensor.item |
| agents/training/instrumented_ppo/calibration.py | 27 | 50 | graph_break,skip_frame | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_ |
| agents/training/scaffolding.py | 25 | 36 | graph_break,skip_frame | Dynamic control flow is not supported at the moment. Please use functorch.experimental.control_flow.cond to ex |
| agents/training/rank_metrics.py | 5 | 30 | graph_break | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_ |
| agents/training/grad_balance.py | 13 | 25 | graph_break,skip_frame | Tensor.item |
| sb3_contrib/common/maskable/distributions.py | 3 | 19 | graph_break | call_method GetAttrVariable(UserDefinedObjectVariable(MaskableCategorical), __dict__) pop [ConstantVariable(), |
| agents/model/forward_guard.py | 1 | 12 | graph_break | 'inline in skipfiles: WeakKeyDictionary.get \| get /home/goodlad/miniconda3/envs/gen3ai_stable/lib/python3.11/w |
| agents/training/instrumented_ppo/metrics_export.py | 5 | 7 | graph_break,skip_frame | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_ |
| None | 2 | 6 | graph_break,recompile_limit | torch._dynamo hit config.cache_size_limit (8) |
| agents/training/instrumented_ppo/noise_scale_terms.py | 2 | 6 | graph_break | Tensor.item |
| agents/training/instrumented_ppo/signal_metrics.py | 4 | 6 | graph_break | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape |
| agents/model/encoders.py | 1 | 3 | graph_break | unsupported operator: aten._native_multi_head_attention.default (see https://docs.google.com/document/d/1GgvOe |
| agents/training/opp_intent_labels.py | 1 | 2 | convert_failed | WON'T CONVERT torch_dynamo_resume_in_train_at_176 /home/goodlad/dev/gen3ai-wt/k8-inventory/src/agents/training |
| stable_baselines3/common/utils.py | 2 | 2 | graph_break | generic_jump NumpyNdarrayVariable() |
| agents/training/instrumented_ppo/ppo.py | 1 | 1 | graph_break | Graph break due to unsupported builtin time.perf_counter. This function is either a Python builtin (e.g. _warn |
| torch/optim/optimizer.py | 1 | 1 | graph_break | 'skip function graph_break in file /home/goodlad/miniconda3/envs/gen3ai_stable/lib/python3.11/site-packages/to |

**whole_step: sites** (top 40 of 220 by events; the JSON has all) (dynamo's reason, verbatim first line):

| component | kind | site | reason | events | calls | detail |
|---|---|---|---|---|---|---|
| model (other) | graph_break | agents/model/opp_intent.py:233 info_gain_nats | dynamic shape operator: aten.bincount.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 21 | update1 |  |
| loss+diagnostics | graph_break | agents/training/rank_metrics.py:44 <lambda> | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_outputs = True | 18 | update1 | agents/training/rank_metrics.py:torch_dynamo_resume_in_effective_rank_at_42 |
| model (other) | graph_break | agents/model/opp_intent.py:313 _alpha_subset_metrics | Tensor.item | 17 | update1,update2,update3 |  |
| distribution+masking | graph_break | sb3_contrib/common/maskable/distributions.py:69 apply_masking | call_method GetAttrVariable(UserDefinedObjectVariable(MaskableCategorical), __dict__) pop [ConstantVariable(), ConstantVariable()] {} | 16 | update1,rollout |  |
| model (other) | graph_break | agents/model/opp_intent.py:516 torch_dynamo_resume_in_intent_losses_at_516 | Graph break due to unsupported builtin None.dict.update. This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python  | 15 | update1,update2,update3 |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:202 torch_dynamo_resume_in_move_belief_loss_at_203 | Graph break due to unsupported builtin None.list.append. This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python  | 15 | update1,update2,update3 | agents/training/belief_bank.py:torch_dynamo_resume_in_move_belief_loss_at_203 |
| extractor | graph_break | agents/model/forward_guard.py:57 forward_guard_for | 'inline in skipfiles: WeakKeyDictionary.get \| get /home/goodlad/miniconda3/envs/gen3ai_stable/lib/python3.11/weakref.py, skipped according trace_rules | 12 | update1,rollout |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:184 _vocab_check | builtin: bool [<class 'torch._dynamo.variables.tensor.TensorVariable'>] False | 12 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_move_belief_loss_at_194 |
| model (other) | graph_break | agents/model/opp_intent.py:237 torch_dynamo_resume_in_info_gain_nats_at_236 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:329 torch_dynamo_resume_in__alpha_subset_metrics_at_313 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:333 torch_dynamo_resume_in__alpha_subset_metrics_at_332 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:335 torch_dynamo_resume_in__alpha_subset_metrics_at_334 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:341 torch_dynamo_resume_in__alpha_subset_metrics_at_338 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:353 torch_dynamo_resume_in__alpha_subset_metrics_at_345 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:358 torch_dynamo_resume_in__alpha_subset_metrics_at_353 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:359 torch_dynamo_resume_in__alpha_subset_metrics_at_358 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:368 torch_dynamo_resume_in__alpha_subset_metrics_at_364 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:389 torch_dynamo_resume_in__beta_subset_metrics_at_382 | Tensor.item | 8 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:236 torch_dynamo_resume_in_info_gain_nats_at_236 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 7 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:337 torch_dynamo_resume_in__alpha_subset_metrics_at_335 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 7 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:338 torch_dynamo_resume_in__alpha_subset_metrics_at_336 | Tensor.item | 7 | update1 |  |
| loss+diagnostics | graph_break | agents/training/grad_balance.py:161 _cos_vs_policy | Tensor.item | 7 | update1 | agents/training/grad_balance.py:_cos_vs_policy |
| model (other) | graph_break | agents/model/opp_intent.py:332 torch_dynamo_resume_in__alpha_subset_metrics_at_331 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 6 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:334 torch_dynamo_resume_in__alpha_subset_metrics_at_333 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 6 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:343 torch_dynamo_resume_in__alpha_subset_metrics_at_341 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 6 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:515 torch_dynamo_resume_in_intent_losses_at_506 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 6 | update1 |  |
| model (other) | graph_break | agents/model/opp_intent.py:516 torch_dynamo_resume_in_intent_losses_at_514 | dynamic shape operator: aten.nonzero.default; to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True | 6 | update1 |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:125 torch_dynamo_resume_in_nature_ev_belief_loss_at_125 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_nature_ev_belief_loss_at_125 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:126 torch_dynamo_resume_in_nature_ev_belief_loss_at_125 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_nature_ev_belief_loss_at_125 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:126 torch_dynamo_resume_in_nature_ev_belief_loss_at_126 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_nature_ev_belief_loss_at_126 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:127 torch_dynamo_resume_in_nature_ev_belief_loss_at_126 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_nature_ev_belief_loss_at_126 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:306 torch_dynamo_resume_in_move_belief_latent_loss_at_305 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_move_belief_latent_loss_at_305 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:307 torch_dynamo_resume_in_move_belief_latent_loss_at_306 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_move_belief_latent_loss_at_306 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:85 torch_dynamo_resume_in_spread_belief_loss_at_85 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_spread_belief_loss_at_85 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:86 torch_dynamo_resume_in_spread_belief_loss_at_85 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_spread_belief_loss_at_85 |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:89 torch_dynamo_resume_in_spread_belief_loss_at_86 | Tensor.item | 6 | update1 | agents/training/belief_bank.py:torch_dynamo_resume_in_spread_belief_loss_at_86 |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:175 torch_dynamo_resume_in__fold_at_174 | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_outputs = True | 6 | update1,update3 | agents/training/instrumented_ppo/calibration.py:torch_dynamo_resume_in__fold_at_174 |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:176 torch_dynamo_resume_in__fold_at_175 | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_outputs = True | 6 | update1,update3 | agents/training/instrumented_ppo/calibration.py:torch_dynamo_resume_in__fold_at_175 |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:177 torch_dynamo_resume_in__fold_at_176 | data dependent operator: aten._local_scalar_dense.default; to enable, set torch._dynamo.config.capture_scalar_outputs = True | 6 | update1,update3 | agents/training/instrumented_ppo/calibration.py:torch_dynamo_resume_in__fold_at_176 |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/value_terms.py:205 torch_dynamo_resume_in__win_prob_loss_at_204 | Tensor.item | 6 | update1 | agents/training/instrumented_ppo/value_terms.py:torch_dynamo_resume_in__win_prob_loss_at_204 |

**whole_step: recompiles** (first failing guard):

| call | function | site | guard | n |
|---|---|---|---|---|
| update1:first(diagnostics) | torch_dynamo_resume_in__beta_subset_metrics_at_382 | agents/model/opp_intent.py:382 | L['sfx'] == '' | 4 |
| update1:first(diagnostics) | torch_dynamo_resume_in__beta_subset_metrics_at_389 | agents/model/opp_intent.py:389 | KeyError on L['out']['opp_intent/beta_recall_top1'] | 4 |
| update1:first(diagnostics) | torch_dynamo_resume_in__beta_subset_metrics_at_382 | agents/model/opp_intent.py:382 | L['sfx'] == '_pool' | 4 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_313 | agents/model/opp_intent.py:313 | L['sfx'] == '' | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_332 | agents/model/opp_intent.py:332 | len(L['out']) == 2 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_334 | agents/model/opp_intent.py:334 | len(L['out']) == 3 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_338 | agents/model/opp_intent.py:338 | len(L['out']) == 5 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_345 | agents/model/opp_intent.py:345 | len(L['out']) == 6 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_353 | agents/model/opp_intent.py:353 | len(L['out']) == 7 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_358 | agents/model/opp_intent.py:358 | len(L['out']) == 8 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_359 | agents/model/opp_intent.py:359 | len(L['out']) == 9 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_364 | agents/model/opp_intent.py:364 | len(L['out']) == 10 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_313 | agents/model/opp_intent.py:313 | L['sfx'] == '_bot' | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__beta_subset_metrics_at_389 | agents/model/opp_intent.py:389 | KeyError on L['out']['opp_intent/beta_recall_top1_pool'] | 3 |
| update1:first(diagnostics) | swap_and_flatten | stable_baselines3/common/buffers.py:62 | tensor '___from_numpy(L['arr'])' rank mismatch. expected 4, actual 3 | 2 |
| update1:first(diagnostics) | swap_and_flatten | stable_baselines3/common/buffers.py:62 | tensor '___from_numpy(L['arr'])' dtype mismatch. expected Float, actual Long | 2 |
| update1:first(diagnostics) | swap_and_flatten | stable_baselines3/common/buffers.py:62 | tensor '___from_numpy(L['arr'])' dtype mismatch. expected Long, actual Float | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_info_gain_nats_at_236 | agents/model/opp_intent.py:236 | tensor 'L['___stack1']' size mismatch at index 0. expected 4, actual 3 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_330 | agents/model/opp_intent.py:330 | KeyError on L['out']['opp_intent/alpha_acc'] | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_332 | agents/model/opp_intent.py:332 | len(L['out']) == 1 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_333 | agents/model/opp_intent.py:333 | L['___stack0'] == 3                                           # _dynamo/output_graph.py:463 in init_ambient_guards | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_334 | agents/model/opp_intent.py:334 | len(L['out']) == 2 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_341 | agents/model/opp_intent.py:341 | L['___stack0'] == 3                                           # _dynamo/output_graph.py:463 in init_ambient_guards | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_forward_at_281 | agents/model/features_extractor.py:281 | GLOBAL_STATE changed: grad_mode | 2 |
| update1:first(diagnostics) | <lambda> | agents/training/rank_metrics.py:44 | L['t'] == 0.9 | 2 |
| update1:first(diagnostics) | <lambda> | agents/training/rank_metrics.py:44 | L['t'] == 0.95 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_effective_rank_at_45 | agents/training/rank_metrics.py:45 | L['___stack1'] == 9.428796887747705 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_effective_rank_at_45 | agents/training/rank_metrics.py:45 | L['___stack1'] == 3.6118969066972473 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_effective_rank_at_45 | agents/training/rank_metrics.py:45 | L['___stack1'] == 4.803937917991325 | 2 |
| update1:first(diagnostics) | sq_norm | agents/training/instrumented_ppo/noise_scale_terms.py:68 | expected type of 'L['grads'][232]' to be a tensor type, ' but found <class 'NoneType'> | 2 |

**whole_step: steady-state update, host ATen ops compiled vs eager** (top-level ops; profiled no stacks — eager ops by phase only): compiled 88237, eager 78161 — compiled share 53.0%

| phase | compiled ops | eager ops | compiled host ms | eager host ms |
|---|---|---|---|---|
| backward | 16689 | 1821 | 45607.2 | 1790.9 |
| batch | 537 | 1061 | 48.4 | 0.1 |
| capacity | 0 | 0 | 0.0 | 0.0 |
| epoch_end | 5544 | 1774 | 1657.6 | 836.7 |
| forward | 35589 | 156 | 23185.5 | 141.5 |
| kl | 0 | 30 | 0.0 | 0.1 |
| logging | 11577 | 2183 | 3827.2 | 89.6 |
| loss | 1233 | 4555 | 713.4 | 2118.1 |
| noise_base | 0 | 1008 | 0.0 | 159.6 |
| noise_probe | 0 | 0 | 0.0 | 0.0 |
| optim | 5544 | 27807 | 2272.6 | 6070.3 |
| probes | 11505 | 37706 | 13571.1 | 49280.4 |
| setup | 19 | 60 | 0.2 | 0.0 |
| tail | 0 | 0 | 0.0 | 0.0 |

Eager ops by component:

| component | eager ops | eager host ms |
|---|---|---|
| (no python stack) | 76346 | 58697.4 |
| backward (eager autograd) | 1815 | 1789.9 |

**Static scan of `train()`'s minibatch loop** (AST, not dynamo; `/home/goodlad/dev/gen3ai-wt/k8-inventory/src/agents/training/instrumented_ppo/ppo.py` from line 364): host copy (.cpu/.numpy/.tolist) ×6, host read (.item) ×11, host read (float/int/bool of a value) ×49, inner python loop ×25, python branch on a value ×7

### Trace — torch 2.8.0+cu126 (cpu)
- source: `/home/goodlad/gen3ai_archive/k8_inventory/20260930_155733_trace_cpu_torch2.8.0/trace_result.json`
- torch `2.8.0+cu126`, device `cpu`, matmul `highest`, git `ecd2be0063`
- checkpoint `/home/goodlad/dev/gen3ai/models/ai_v14_02_lbat_ctrl/final_model.zip` (sha256 `06efae2c44d6`), buffer `/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl`
- geometry: 1536 rows, micro 512 × 3/epoch, accum 2, epochs 2, diagnostics_every 1
- ObservationDebugger: api present False, attached by the run False (dropped for both views, as the compile path drops it); micro-check: {"present": false}

**Totals per view** (all four calls):

| view | graphs | graphs ending in a break | break sites | break events | …in skipped frames | skipped frames | eager fallbacks | recompiles | frames traced | ops in graphs |
|---|---|---|---|---|---|---|---|---|---|---|
| today | 3 | 0 | 1 | 1 | 0 | 0 | 0 | 2 | 2 | 24933 |

**today: per call** (a NEW graph/break is counted at the call that first traced it; a cached call adds nothing):

| call | new graphs | break events | skipped frames | recompiles | eager fallbacks |
|---|---|---|---|---|---|
| update1:first(diagnostics) | 2 | 0 | 0 | 1 | 0 |
| update2:diag_skipped | 0 | 0 | 0 | 0 | 0 |
| rollout:eval_no_grad | 1 | 1 | 0 | 1 | 0 |
| update3:diag_skipped(steady) | 0 | 0 | 0 | 0 | 0 |

**today: by component**:

| component | ops in graphs | break sites | skipped-frame sites | eager-fallback sites |
|---|---|---|---|---|
| extractor | 24927 | 1 | 0 | 0 |
| model (other) | 6 | 0 | 0 | 0 |

**today: by source file** (every site; top reason by events):

| file | sites | events | kinds | top reason |
|---|---|---|---|---|
| agents/model/encoders.py | 1 | 1 | graph_break | Operator does not support running with fake tensors |

**today: sites** (dynamo's reason, verbatim first line):

| component | kind | site | reason | events | calls | detail |
|---|---|---|---|---|---|---|
| extractor | graph_break | agents/model/encoders.py:287 forward | Operator does not support running with fake tensors | 1 | rollout |  |

**today: recompiles** (first failing guard):

| call | function | site | guard | n |
|---|---|---|---|---|
| update1:first(diagnostics) | forward | agents/model/features_extractor.py:267 | GLOBAL_STATE changed: grad_mode | 1 |
| rollout:eval_no_grad | forward | agents/model/features_extractor.py:267 | GLOBAL_STATE changed: grad_mode | 1 |

**today: steady-state update, host ATen ops compiled vs eager** (top-level ops; profiled with Python stacks): compiled 62510, eager 26576 — compiled share 70.2%

| phase | compiled ops | eager ops | compiled host ms | eager host ms |
|---|---|---|---|---|
| backward | 15834 | 2522 | 5158.8 | 112.4 |
| batch | 0 | 261 | 0.0 | 6.1 |
| capacity | 0 | 0 | 0.0 | 0.0 |
| epoch_end | 0 | 2986 | 0.0 | 14.0 |
| forward | 35292 | 381 | 4367.0 | 17.6 |
| kl | 0 | 30 | 0.0 | 0.2 |
| logging | 0 | 11820 | 0.0 | 1373.1 |
| loss | 0 | 4797 | 0.0 | 48.9 |
| noise_base | 0 | 992 | 0.0 | 2.3 |
| noise_probe | 0 | 0 | 0.0 | 0.0 |
| optim | 0 | 2738 | 0.0 | 46.5 |
| probes | 11384 | 49 | 373.3 | 0.1 |
| setup | 0 | 0 | 0.0 | 0.0 |
| tail | 0 | 0 | 0.0 | 0.0 |

Eager ops by component:

| component | eager ops | eager host ms |
|---|---|---|
| extractor | 12022 | 1373.6 |
| loss+diagnostics | 6117 | 52.6 |
| optimizer | 5474 | 59.5 |
| backward (eager autograd) | 2516 | 112.3 |
| rollout buffer | 261 | 6.1 |
| heads | 99 | 15.8 |
| distribution+masking | 87 | 1.3 |

**Static scan of `train()`'s minibatch loop** (AST, not dynamo; `/home/goodlad/dev/gen3ai-wt/k8-inventory/src/agents/training/instrumented_ppo/ppo.py` from line 364): host copy (.cpu/.numpy/.tolist) ×6, host read (.item) ×11, host read (float/int/bool of a value) ×49, inner python loop ×25, python branch on a value ×7

### Trace — torch 2.8.0+cu126 (cpu)
- source: `/home/goodlad/gen3ai_archive/k8_inventory/20260930_170750_trace_cpu_torch2.8.0/trace_result.json`
- torch `2.8.0+cu126`, device `cpu`, matmul `highest`, git `89679ddbdc`
- checkpoint `/home/goodlad/dev/gen3ai/models/ai_v14_02_lbat_ctrl/final_model.zip` (sha256 `06efae2c44d6`), buffer `/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl`
- geometry: 192 rows, micro 64 × 3/epoch, accum 2, epochs 2, diagnostics_every 1
- peak RSS by stage (MB; the 12 GB-capped scope): whole_step:update1 1711, whole_step:update2 1848, whole_step:rollout 1922, whole_step:update3 1921, whole_step_update3:profiled_update 2464, whole_step_update3:gzip 2464; classify (separate process) 695
- ObservationDebugger: api present False, attached by the run False (dropped for both views, as the compile path drops it); micro-check: {"present": false}

**Totals per view** (all four calls):

| view | graphs | graphs ending in a break | break sites | break events | …in skipped frames | skipped frames | eager fallbacks | recompiles | frames traced | ops in graphs |
|---|---|---|---|---|---|---|---|---|---|---|
| whole_step | 646 | 600 | 178 | 530 | 6 | 13 | 4 | 299 | 263 | 52322 |

**whole_step: per call** (a NEW graph/break is counted at the call that first traced it; a cached call adds nothing):

| call | new graphs | break events | skipped frames | recompiles | eager fallbacks |
|---|---|---|---|---|---|
| update1:first(diagnostics) | 604 | 499 | 13 | 265 | 2 |
| update2:diag_skipped | 12 | 10 | 0 | 14 | 2 |
| rollout:eval_no_grad | 7 | 9 | 0 | 7 | 0 |
| update3:diag_skipped(steady) | 23 | 12 | 0 | 13 | 0 |

**whole_step: by component**:

| component | ops in graphs | break sites | skipped-frame sites | eager-fallback sites |
|---|---|---|---|---|
| distribution+masking | 46 | 1 | 0 | 0 |
| extractor | 33313 | 0 | 0 | 0 |
| heads | 59 | 0 | 0 | 0 |
| loss+diagnostics | 3035 | 176 | 12 | 1 |
| model (other) | 532 | 0 | 1 | 0 |
| optimizer | 15120 | 1 | 0 | 0 |
| other | 124 | 0 | 0 | 1 |
| rollout buffer | 93 | 0 | 0 | 0 |

**whole_step: by source file** (every site; top reason by events):

| file | sites | events | kinds | top reason |
|---|---|---|---|---|
| agents/model/opp_intent.py | 43 | 176 | graph_break,skip_frame | Dynamic shape operator |
| agents/training/belief_bank.py | 50 | 136 | graph_break,skip_frame | Unsupported Tensor.item() call with capture_scalar_outputs=False |
| agents/training/instrumented_ppo/calibration.py | 30 | 66 | graph_break,skip_frame | Data-dependent branching |
| agents/training/instrumented_ppo/value_terms.py | 15 | 49 | graph_break | Unsupported Tensor.item() call with capture_scalar_outputs=False |
| agents/training/scaffolding.py | 20 | 38 | graph_break,skip_frame | Data-dependent branching |
| agents/training/rank_metrics.py | 5 | 20 | graph_break | Data dependent operator |
| agents/training/grad_balance.py | 13 | 20 | graph_break,skip_frame | Data dependent operator |
| sb3_contrib/common/maskable/distributions.py | 1 | 17 | graph_break | Unsupported method call |
| agents/training/instrumented_ppo/metrics_export.py | 5 | 7 | graph_break,skip_frame | Data dependent operator |
| agents/training/instrumented_ppo/signal_metrics.py | 4 | 6 | graph_break | Dynamic shape operator |
| agents/training/instrumented_ppo/noise_scale_terms.py | 2 | 4 | graph_break | Attempted to call function marked as skipped |
| None | 1 | 2 | recompile_limit | torch._dynamo hit config.recompile_limit (8) |
| agents/training/opp_intent_labels.py | 1 | 2 | convert_failed | WON'T CONVERT torch_dynamo_resume_in_train_at_176 /home/goodlad/dev/gen3ai-wt/k8-inventory/src/agents/training |
| stable_baselines3/common/utils.py | 1 | 2 | graph_break | Data-dependent branching |
| agents/training/instrumented_ppo/ppo.py | 2 | 2 | graph_break | Call to `torch._dynamo.graph_break()` |

**whole_step: sites** (top 40 of 193 by events; the JSON has all) (dynamo's reason, verbatim first line):

| component | kind | site | reason | events | calls | detail |
|---|---|---|---|---|---|---|
| distribution+masking | graph_break | sb3_contrib/common/maskable/distributions.py:69 apply_masking | Unsupported method call | 17 | update1,rollout |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:233 info_gain_nats | Dynamic shape operator | 16 | update1 |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:126 nature_ev_belief_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 12 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/rank_metrics.py:44 <lambda> | Data dependent operator | 12 | update1 | agents/training/rank_metrics.py:rank_probe |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:313 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 10 | update1,update2 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:516 intent_losses | Dynamic shape operator | 10 | update1,update2 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:236 info_gain_nats | Dynamic shape operator | 9 | update1 |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:125 nature_ev_belief_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 8 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:202 move_belief_loss | Attempted to call function marked as skipped | 8 | update1,update2 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:85 spread_belief_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 8 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/value_terms.py:256 _win_prob_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 8 | update1 | agents/training/instrumented_ppo/value_terms.py:_win_prob_loss |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:338 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 7 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:358 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 7 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:359 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 7 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:368 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 7 | update1 |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:150 hp_type_belief_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 7 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:333 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:335 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:341 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:353 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 |  |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:184 _vocab_check | Failed to trace builtin operator | 6 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:306 move_belief_latent_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:307 move_belief_latent_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/belief_bank.py:86 spread_belief_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 | agents/training/belief_bank.py:compute |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:123 metrics | Data-dependent branching | 6 | update1 | agents/training/instrumented_ppo/metrics_export.py:_record_head_metrics |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:175 _fold | Data dependent operator | 6 | update1,update3 | agents/training/instrumented_ppo/metrics_export.py:_record_head_metrics |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:176 _fold | Data dependent operator | 6 | update1,update3 | agents/training/instrumented_ppo/metrics_export.py:_record_head_metrics |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/calibration.py:177 _fold | Data dependent operator | 6 | update1,update3 | agents/training/instrumented_ppo/metrics_export.py:_record_head_metrics |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/value_terms.py:205 _win_prob_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 | agents/training/instrumented_ppo/value_terms.py:_win_prob_loss |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/value_terms.py:207 _win_prob_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 | agents/training/instrumented_ppo/value_terms.py:_win_prob_loss |
| loss+diagnostics | graph_break | agents/training/instrumented_ppo/value_terms.py:208 _win_prob_loss | Unsupported Tensor.item() call with capture_scalar_outputs=False | 6 | update1 | agents/training/instrumented_ppo/value_terms.py:_win_prob_loss |
| loss+diagnostics | graph_break | agents/training/scaffolding.py:90 _rankdata | Data-dependent branching | 6 | update1 | agents/training/instrumented_ppo/metrics_export.py:_record_signal_metrics |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:331 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 5 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:329 _alpha_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 4 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:330 _alpha_subset_metrics | Dynamic shape operator | 4 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:332 _alpha_subset_metrics | Dynamic shape operator | 4 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:334 _alpha_subset_metrics | Dynamic shape operator | 4 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:370 _alpha_subset_metrics | Dynamic shape operator | 4 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:389 _beta_subset_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 4 | update1 |  |
| loss+diagnostics | graph_break | agents/model/opp_intent.py:428 switch_coverage_metrics | Unsupported Tensor.item() call with capture_scalar_outputs=False | 4 | update1 |  |

**whole_step: recompiles** (first failing guard):

| call | function | site | guard | n |
|---|---|---|---|---|
| update1:first(diagnostics) | torch_dynamo_resume_in_intent_losses_at_516 | agents/model/opp_intent.py:516 | Cache line invalidated because L['___stack1'] got deallocated | 10 |
| update1:first(diagnostics) | torch_dynamo_resume_in_move_belief_loss_at_203 | agents/training/belief_bank.py:203 | Cache line invalidated because L['___stack0'] got deallocated | 5 |
| update2:diag_skipped | torch_dynamo_resume_in_intent_losses_at_516 | agents/model/opp_intent.py:516 | Cache line invalidated because L['___stack1'] got deallocated | 5 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_353 | agents/model/opp_intent.py:353 | len(out) == 7 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_358 | agents/model/opp_intent.py:358 | len(out) == 8 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_359 | agents/model/opp_intent.py:359 | len(out) == 9 | 3 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_364 | agents/model/opp_intent.py:364 | len(out) == 10 | 3 |
| update2:diag_skipped | torch_dynamo_resume_in_move_belief_loss_at_203 | agents/training/belief_bank.py:203 | Cache line invalidated because L['___stack0'] got deallocated | 3 |
| update1:first(diagnostics) | swap_and_flatten | stable_baselines3/common/buffers.py:62 | tensor '___from_numpy(arr)' rank mismatch. expected 4, actual 3 | 2 |
| update1:first(diagnostics) | swap_and_flatten | stable_baselines3/common/buffers.py:62 | tensor '___from_numpy(arr)' dtype mismatch. expected Float, actual Long | 2 |
| update1:first(diagnostics) | swap_and_flatten | stable_baselines3/common/buffers.py:62 | tensor '___from_numpy(arr)' dtype mismatch. expected Long, actual Float | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_info_gain_nats_at_236 | agents/model/opp_intent.py:236 | tensor '___stack1' size mismatch at index 0. expected 4, actual 3 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_332 | agents/model/opp_intent.py:332 | len(out) == 2 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_334 | agents/model/opp_intent.py:334 | len(out) == 3 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_338 | agents/model/opp_intent.py:338 | len(out) == 5 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_345 | agents/model/opp_intent.py:345 | len(out) == 6 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_330 | agents/model/opp_intent.py:330 | KeyError on out['opp_intent/alpha_acc'] | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_332 | agents/model/opp_intent.py:332 | len(out) == 1 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_334 | agents/model/opp_intent.py:334 | len(out) == 2 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__beta_subset_metrics_at_382 | agents/model/opp_intent.py:382 | sfx == '' | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__beta_subset_metrics_at_389 | agents/model/opp_intent.py:389 | KeyError on out['opp_intent/beta_recall_top1'] | 2 |
| update1:first(diagnostics) | <lambda> | agents/training/rank_metrics.py:44 | ___as_tensor(t).item() == 0.95  # (unknown source ___as_tensor(t).item(), please file a bug) | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_effective_rank_at_45 | agents/training/rank_metrics.py:45 | ___stack1 == 9.428796887747705 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_intent_losses_at_500 | agents/model/opp_intent.py:500 | tensor '___stack1' size mismatch at index 0. expected 35, actual 46 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__alpha_subset_metrics_at_336 | agents/model/opp_intent.py:336 | len(out) == 4 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_switch_coverage_metrics_at_427 | agents/model/opp_intent.py:427 | ___stack0 == 17.0 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_nature_ev_belief_loss_at_117 | agents/training/belief_bank.py:117 | tensor '___stack1' size mismatch at index 0. expected 98, actual 89 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in_nature_ev_belief_loss_at_126 | agents/training/belief_bank.py:126 | ___stack1 == 0.3265306055545807 | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__win_prob_loss_at_205 | agents/training/instrumented_ppo/value_terms.py:205 | ___as_tensor(___stack2).item() == 0.828125  # (unknown source ___as_tensor(___stack2).item(), please file a bug) | 2 |
| update1:first(diagnostics) | torch_dynamo_resume_in__win_prob_loss_at_208 | agents/training/instrumented_ppo/value_terms.py:208 | ___as_tensor(___stack5).item() == 0.828125  # (unknown source ___as_tensor(___stack5).item(), please file a bug) | 2 |

**whole_step: steady-state update, host ATen ops compiled vs eager** (top-level ops; profiled no stacks — eager ops by phase only): compiled 89047, eager 43671 — compiled share 67.1%

| phase | compiled ops | eager ops | compiled host ms | eager host ms |
|---|---|---|---|---|
| backward | 16734 | 1797 | 156768.5 | 6267.2 |
| batch | 537 | 911 | 88.8 | 0.1 |
| capacity | 0 | 0 | 0.0 | 0.0 |
| epoch_end | 5544 | 1774 | 5035.5 | 827.9 |
| forward | 35568 | 129 | 86540.1 | 0.4 |
| kl | 0 | 30 | 0.0 | 0.1 |
| logging | 11592 | 3958 | 21441.0 | 145.7 |
| loss | 2009 | 5470 | 3329.1 | 6888.2 |
| noise_base | 0 | 1008 | 0.0 | 1974.2 |
| noise_probe | 0 | 0 | 0.0 | 0.0 |
| optim | 5544 | 27807 | 7503.0 | 6527.2 |
| probes | 11496 | 702 | 36273.9 | 1.2 |
| setup | 23 | 85 | 0.3 | 0.1 |
| tail | 0 | 0 | 0.0 | 0.0 |

Eager ops by component:

| component | eager ops | eager host ms |
|---|---|---|
| (no python stack) | 41880 | 16365.4 |
| backward (eager autograd) | 1791 | 6267.2 |

**Static scan of `train()`'s minibatch loop** (AST, not dynamo; `/home/goodlad/dev/gen3ai-wt/k8-inventory/src/agents/training/instrumented_ppo/ppo.py` from line 364): host copy (.cpu/.numpy/.tolist) ×6, host read (.item) ×11, host read (float/int/bool of a value) ×49, inner python loop ×25, python branch on a value ×7

### Time — torch 2.8.0+cu126 (CUDA), matmul `high`
- source: `/home/goodlad/gen3ai_archive/k8_inventory/20260930_174406_time_cuda_torch2.8.0/time_analysis.json`
- geometry: 98304 rows, micro 2048 × 48/epoch, accum 32, epochs 10
- full update, `diag_skipped`: unbracketed train_ms 53230, bracketed train_ms 57616

Bracketed full update, per phase (synchronised segment wall):

| phase | s | share | marks |
|---|---|---|---|
| backward | 25.93 | 45.0% | 480 |
| forward | 11.73 | 20.4% | 480 |
| probes | 8.89 | 15.4% | 960 |
| loss | 7.84 | 13.6% | 480 |
| batch | 2.95 | 5.1% | 480 |
| logging | 0.13 | 0.2% | 1 |
| epoch_end | 0.06 | 0.1% | 10 |
| optim | 0.04 | 0.1% | 480 |
| kl | 0.03 | 0.1% | 480 |
| setup | 0.01 | 0.0% | 1 |
| noise_base | 0.01 | 0.0% | 480 |
| noise_probe | 0.00 | 0.0% | 480 |
| capacity | 0.00 | 0.0% | 480 |

**Profiled `diag_skipped` update, 1 epoch(s)** — train span 9271 ms, GPU busy 3907 ms, idle 5364 ms; kernel time compiled 3210 ms vs eager 581 ms → **compiled share of kernel time 84.7%**, of train wall 34.6%; unmatched device events 0/234677

| phase | host wall ms | compiled kernel ms | eager kernel ms | memcpy/set ms | compiled share | compiled kernels | eager kernels |
|---|---|---|---|---|---|---|---|
| probes | 3437 | 9 | 16 | 1 | 37.8% | 375 | 253 |
| batch | 1868 | 0 | 0 | 66 | — | 0 | 0 |
| backward | 1269 | 2192 | 248 | 30 | 89.8% | 42720 | 42708 |
| forward | 1245 | 1008 | 24 | 1 | 97.7% | 26064 | 3696 |
| loss | 1155 | 0 | 262 | 18 | 0.0% | 0 | 84924 |
| logging | 170 | 0 | 27 | 1 | 0.0% | 0 | 6028 |
| epoch_end | 38 | 0 | 1 | 0 | 0.0% | 0 | 295 |
| optim | 37 | 0 | 1 | 0 | 0.0% | 0 | 47 |
| noise_base | 34 | 0 | 1 | 0 | 0.0% | 0 | 743 |
| setup | 7 | 0 | 0 | 0 | — | 0 | 0 |
| kl | 6 | 0 | 0 | 0 | 0.0% | 0 | 240 |
| tail | 1 | 0 | 0 | 0 | — | 0 | 0 |
| capacity | 0 | 0 | 0 | 0 | — | 0 | 0 |
| noise_probe | 0 | 0 | 0 | 0 | — | 0 | 0 |

### Time — torch 2.5.1+cu121 (CUDA), matmul `high`
- source: `/home/goodlad/gen3ai_archive/k8_inventory/20260930_180918_time_cuda_torch2.5.1/time_analysis.json`
- geometry: 98304 rows, micro 2048 × 48/epoch, accum 32, epochs 10
- full update, `diag_skipped`: unbracketed train_ms —, bracketed train_ms 53801

Bracketed full update, per phase (synchronised segment wall):

| phase | s | share | marks |
|---|---|---|---|
| backward | 26.63 | 49.5% | 480 |
| forward | 11.21 | 20.8% | 480 |
| loss | 7.88 | 14.7% | 480 |
| probes | 4.92 | 9.1% | 960 |
| batch | 2.88 | 5.4% | 480 |
| logging | 0.13 | 0.2% | 1 |
| epoch_end | 0.05 | 0.1% | 10 |
| optim | 0.04 | 0.1% | 480 |
| kl | 0.03 | 0.1% | 480 |
| noise_base | 0.01 | 0.0% | 480 |
| setup | 0.01 | 0.0% | 1 |
| ridealong | 0.01 | 0.0% | 480 |
| noise_probe | 0.00 | 0.0% | 480 |
| capacity | 0.00 | 0.0% | 480 |

**Profiled `diag_skipped` update, 1 epoch(s)** — train span 10045 ms, GPU busy 3992 ms, idle 6053 ms; kernel time compiled 3227 ms vs eager 632 ms → **compiled share of kernel time 83.6%**, of train wall 32.1%; unmatched device events 0/239422

| phase | host wall ms | compiled kernel ms | eager kernel ms | memcpy/set ms | compiled share | compiled kernels | eager kernels |
|---|---|---|---|---|---|---|---|
| probes | 4229 | 9 | 16 | 1 | 36.6% | 369 | 253 |
| batch | 1917 | 0 | 0 | 65 | — | 0 | 0 |
| backward | 1235 | 2213 | 293 | 47 | 88.3% | 43584 | 44340 |
| loss | 1231 | 0 | 269 | 18 | 0.0% | 0 | 84924 |
| forward | 1158 | 1006 | 24 | 1 | 97.7% | 28272 | 3696 |
| logging | 141 | 0 | 27 | 1 | 0.0% | 0 | 6028 |
| noise_base | 39 | 0 | 1 | 0 | 0.0% | 0 | 743 |
| epoch_end | 38 | 0 | 1 | 0 | 0.0% | 0 | 295 |
| optim | 38 | 0 | 1 | 0 | 0.0% | 0 | 47 |
| kl | 6 | 0 | 0 | 0 | 0.0% | 0 | 240 |
| setup | 6 | 0 | 0 | 0 | — | 0 | 0 |
| ridealong | 1 | 0 | 0 | 0 | — | 0 | 0 |
| tail | 1 | 0 | 0 | 0 | — | 0 | 0 |
| capacity | 0 | 0 | 0 | 0 | — | 0 | 0 |
| noise_probe | 0 | 0 | 0 | 0 | — | 0 | 0 |
