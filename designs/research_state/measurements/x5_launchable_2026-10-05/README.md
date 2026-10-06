# X5 fixed_mass made launchable: F-XC-3 fixed, and the REAL production launch (2026-10-05)

Under the GPU lease (owner `x5-perf-encode`). Base `065d2d7f` (`gen3_x5_hyp_gather_v1`). Box: RTX 3080 Ti
(12 GiB), torch 2.8.0+cu126, Triton 3.4.0, fp32 'highest', desktop STOPPED (T23). Tags: **MEASURED** unless
marked. The paired speed benchmark that follows this is `../x5_paired_speed_2026-10-05/`.

## Verdict

| refusal (`design_x5_belief_tokens.md` §7.9: FIXED, never overridden) | before | after | verdict |
|---|---|---|---|
| **F-XC-3**, a T2 slot load's growth vs the 1 MiB declared-load tolerance | ≈ 1.1 MiB in a launch (`889add9d`); 16.3 / 17.2 MB per load on the 2-slot probe | **1,024 B** per load (21 of 21 in the real launch, 0 B on one); 7,168 B on the probe | **FIXED** (`gen3_reference_state_released_v1`) |
| **F-XC-2**, `UpdateFit` headroom vs the declared 1,024 MiB | 1,192 MiB (`065d2d7f`, its launch ran under a lowered measurement floor) | **1,790 MiB** with NO override | **PASSES** (766 MiB margin) |
| F-XC-4, the R1 compile gate on CUDA | fixed at `c7b4d03e` | PASS (fresh weights, per-parameter 1.94e-5) | holds |
| F-XC-5, K9(b)'s tie share | fixed at `e8008c2d` | (not re-read here) | — |

## 1. F-XC-3: the mechanism

`scripts/slotprobe.py` (CUDA, graph backend, buckets 8 / 64, fixed_mass golden learner, a second perturbed
policy loaded into a slot three times; a gc census of live CUDA tensors after each load):
`results/slotprobe/slotprobe_fm_before.json` / `_after.json`.

- The parity gate's EAGER reference (`decision.policy_reference`) runs the slot replica's own forward, and a
  forward REPLACES the extractor's and the op's per-forward stashes (`ExtractorStashes`, `OpStashes`) and two
  plain attributes (`PokemonEncoder.last_move_tokens`, `EntitySeats.last_cand`; the probe also names
  `team_transformer.last_global_out`) at entry. Those tensors outlived the gate, sized by its LAST row count.
- A slot load gates every bucket, so the replica kept the stash of the last bucket's gate. Whenever that differed
  from what the startup gate had left, `rust_rollout.build.checked_slot_load` read the difference as a NEW
  allocation. Under fixed_mass the stash holds ≈ 0.38 MB per row (157 tensors, 24.6 MB at 63 rows); blob far less.
  Every T2 replica also held one for the life of the run.
- Before: loads 1 and 2 grew 16,336,896 and 17,214,976 B. After: 7,168 B each.

**FIX** (`agents/inference/service/decision.py`): `policy_reference` runs under `forward_state_released(policy)`,
which empties every module attribute the forward REPLACED when it returns (a stash to a fresh instance, a tensor /
tensor-tuple to `None`, an attribute the forward created is removed); untouched attributes are left alone. It is
a NO-GRAD eager path used only by the parity gate (`parity.judge`), so no compiled code and no served output moves.

## 2. Blob byte-identical (the nanfix unit's method)

`scripts/identity2.sh`: with FX / AOT caches off, blob and fixed_mass R1 (forward + backward) and T2 (`decide`)
compiled at the fix and with `decision.py` put back to HEAD, compared by the nanfix unit's `compare_dumps.py`
(`../x5_fxc4_nanfix_2026-10-05/scripts/`; two process-state comments normalised). `results/identity/`:

| arm | R1 modules | T2 modules | identical after normalisation | combined sha (R1 / T2) |
|---|---|---|---|---|
| blob | 32 / 32 | 3 / 3 | **yes / yes** | `dee5ac1f16a8…` / `4156f57a6ac8…` (= the gather and nanfix units') |
| fixed_mass | 16 / 16 | 3 / 3 | yes / yes | `689688d4e410…` / `583277c39109…` |

Expected: the change touches no compiled path. The served outputs are the reference's return values, unchanged.

## 3. The REAL production launch (no `PROF_*` override)

`scripts/queue3.sh` step 2: a fresh fixed_mass launch, `--arch production --device cuda --steps 15000000
--seed 1001`, the X26 ride-along heads, `--snapshot-ladder-games 0 --checkpoint-every-steps 1000000
--belief-tokens fixed_mass --allow-nonproduction-arch --eval-freq 400000 --promote-threshold 0.0`, into a seeded
20-snapshot pool (regime A; `scripts/build_pool.py` from a fixed_mass checkpoint). Run through the cost
ablation's phase-probe wrapper (the trainer's own `main()`); the ONLY `PROF_*` variables set were
`PROF_PHASE_LOG` and `PROF_LOG_SLOT_LOADS`, which LOG (the tolerance and the refusal unchanged). No
`PROF_FIT_HEADROOM_MIB`, no `PROF_SLOT_LOAD_TOL_MIB`. Stopped by SIGTERM to its PID after 12 update rows
(`results/launch/`: the read, the phase markers with every slot load, the key log lines).

| check | result |
|---|---|
| T2 startup (every slot gated per bucket) | up in 508.8 s, 31 slots, buckets (8, 64, 256), graph backend |
| startup slot loads (20 pool snapshots) | 20 of 20 grew **1,024 B** (tolerance 1,048,576) |
| R1 compile gate | PASS (fresh: per-parameter 1.94e-5, grad cosine 1.000000) |
| compile lock | 1 cache entry per code object, 4 graphs; no later compile |
| stage ledger, T2 (allocated) | **749 MiB** (the gather unit's launch `fmA_gather`: 1,264) |
| **`UpdateFit` (D-6)** | demand 9,134 MiB, headroom **1,790 MiB vs the declared 1,024: PASS** |
| pool refresh 1 (eval at step 400,052; promoted, slot 20 loaded) | grew **1,024 B** |
| pool refresh 2 (eval at step 800,220; promoted, slot 0 replaced) | grew **0 B** |
| update-10 compile canary | **PASS** (trained weights: loss rel 5.96e-8, grad cosine 1.000000, per-parameter 1.91e-5) |
| NaN | none in any loss / KL row (the only `nan` cells are the near-tie `rel_gap_b*` meters' empty bins, as in blob) |
| learner freeze | `cuda_reserved_after_freeze_mib` 0, `cuda_streams_after_freeze` 0 every update |
| exit | SIGTERM → checkpoint saved; host peak 16.68 GB (cap 48) |

**fixed_mass is LAUNCHABLE on its refusals**: no startup refusal fires and none is overridden.
