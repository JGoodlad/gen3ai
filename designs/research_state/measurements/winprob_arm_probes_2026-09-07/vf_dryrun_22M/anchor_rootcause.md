# cf_audit ANCHOR failure — ROOT CAUSE (2026-09-07)

**Verdict: the replay path is EXACT. The dry run broke the anchor itself by passing
`--checkpoint`.** cf_audit's anchor is not a pure replay — it scripts the recorded prefix, then
plays the rest of the battle LIVE with a *reloaded* greedy trainee. `--checkpoint` swaps that
trainee for a different network, so the continuation is a different game. Nothing is wrong with
the record, the dice, the rust driver or the traces.

## 1. Reproduction (arm = `models/ai_v12_02_winprob_critic`, traces `step_22000032`, seed 3)

The bot anchor pool is exactly **136 battles**, so `--anchors 150` takes all of them: every arm
below is the *same 136 anchors*, paired. Harness: `anchor_probe.py` (replicates cf_audit.py:783-808
verbatim, logging every anchor). Per-anchor rows in `arm_*.jsonl`.

| arm — only the replaying checkpoint differs | rate | flips |
|---|---|---|
| **default ladder** (resolves `step_22000032/snapshot.zip`, tier `exact`) | **136/136 = 100.0%**, 0 err | none |
| `--checkpoint checkpoint_22454016_steps.zip` (the dry run's, +454k steps) | 115/136 = **84.6%**, 0 err | 19 loss→win, 2 loss→tie |
| same override, **`--impl node`** | 115/136 = **84.6%**, 0 err | **identical failing set** |
| `--checkpoint checkpoint_9969408_steps.zip` (12M steps *earlier*) | 109/136 = **80.1%**, 0 err | 20 loss→win, 1 loss→tie, **6 win→loss** |

The dry run's 80.0% (n=20) and 87.5% (n=80) sit inside this; 84.6% at n=136 is the true rate for
that override.

## 2. Bisect

- **By impl:** rust and node agree on **136/136** anchors under the same override (fail sets
  identical). The rust replay driver is exonerated — suspect (c) is dead.
- **By run:** comparator `models/ai_v9_29_rev1_0823` (shaped critic, also `bridge_impl=rust`),
  default ladder, seed 3 → **132/133 = 99.2%**, 1 error. Not arm- or era-specific.
- The single comparator error is the known 250-turn-cap class, verbatim:
  `RuntimeError: replay driver [rust: …/search_driver] failed (rc=1): replayed all 506 commands but
  battle has not ended (turn 250) — corrupt or truncated record?` (`staller_v2/loss_s0_003`, inv 127,
  249 move decisions). Rate 1/133; **not** the cause of a 12% shortfall.
- **Direction test (the clincher):** a *weaker* checkpoint introduces 6 win→loss flips that the
  newer one never produces. Failures track *policy identity*, exactly as a "different policy plays
  the continuation" mechanism predicts, and not at all as a replay defect would.

## 3. Failing-anchor characterisation (override arm, n=136)

Flip rate by recorded outcome — `arm_rust` / `+454k` / `−12M`:

| recorded | default ladder | +454k ckpt | −12M ckpt |
|---|---|---|---|
| win (n=72) | 0.0% | 0.0% | 8.3% |
| loss (n=64) | 0.0% | **32.8%** | **32.8%** |

Every one of the 21 failures is a recorded **loss** turned into a win (19) or tie (2). No opponent
concentration (spread over 8 of 9 bots; `random` never fails). Failing anchors are marginally
longer (median turn 13 vs 11; median 11 plies still to play vs 10) and their recorded win_prob is
*indistinguishable* from the pool (median 0.914 vs 0.918). All 136 records carry a
`*_reconstruction.json` sibling with a `sodium,<hex>` seed. This is regression to the mean on a
loss-enriched sample (capture rate 0.097/win vs 0.732/loss), not a defect signature.

## 4. First divergence — 3 failing battles, snapshot vs override, `narrate=True`

In every case the sim is **byte-identical** (same opponent moves, same damage percentages, same
heal ticks) up to the divergence, and the first difference is **our own action**:

| battle | first divergence | snapshot | override |
|---|---|---|---|
| `aggressive_v2/loss_s0_005` inv 11 | turn 18 (turns 0-17 identical) | we used **Ice Punch** | we used **Body Slam** (→ paralysis) |
| `setup_sweep_v2/loss_s2_004` inv 10 | turn 9 forced switch-in | we sent in **Tyranitar** | we sent in **Jirachi** |
| `heuristic2/loss_s0_005` inv 10 | turn 10 forced switch-in | we sent in **Tyranitar** | we sent in **Swampert** |

**No mechanic is implicated** — there is no move, ability, item or engine rule at the divergence,
only a different policy choosing differently. That is the whole finding.

## 5. Suspects

| suspect | verdict |
|---|---|
| (a) recorded dice missing/partial (traces written under load) | **RULED OUT** — 136/136 records present, all `sodium,<hex>` seeds, and the default ladder reproduces 136/136 including all 64 losses |
| (b) forfeits / `/choose default` recorded as actions | **RULED OUT** — 0/136 records contain `default` or `forcelose` in their input log; 0 among the failures |
| (c) rust replay kernels ≠ rust bridge | **RULED OUT** — node and rust agree on 136/136 under the same override |
| (d) loss-enriched trace selection | **SURVIVES, as an amplifier not a cause** — 100% of failures are recorded losses; the quota makes the anchor pool ~47% losses, so a policy mismatch is maximally visible here |
| 250-turn-cap records are unreplayable | **SURVIVES, minor** — 1/133 on the comparator, 0/136 on the arm; `cf_producer.record_is_full_replay_anchorable` already excludes this class, `cf_audit` does not |

## 6. Recommendation

**No code change is required to unblock the 75M read: drop `--checkpoint`.** The default ladder
resolves each battle's `eval_traces/step_<N>/snapshot.zip` at tier `exact` — the frozen weights
that actually played the battle — and anchors at 100%. It is also the *more correct* setting for
the audit itself: the bias map compares the trace's recorded `win_prob` (the snapshot's) against
tight-MC, so the MC must be rolled out under the snapshot; the override was biasing labels with a
policy that never generated those states.

If a defensive fix is wanted, change **`src/agents/training/cf_audit.py:779-780`** to build the
*anchor* session with `ckpt_override=None` (the exact ladder) while the label pass keeps the
override, and print a loud line whenever the two differ — today the anchor silently measures
policy drift and reports it as replay infidelity, and the tolerance then refuses correct labels.
A second, smaller item: adopt `cf_producer.record_is_full_replay_anchorable`'s forfeit/250-cap
exclusion in the anchor pool at `cf_audit.py:784` so a capped record is a *declared* skip rather
than a counted failure. Do **not** touch `--anchor-tolerance` — it did its job.

Artifacts: `anchor_probe.py`, `arm_rust.jsonl`, `arm_rust_ckptoverride.jsonl`,
`arm_node_ckptoverride.jsonl`, `arm_rust_ckpt10M.jsonl`, `comparator_rust.jsonl` (+ `.log` each).
