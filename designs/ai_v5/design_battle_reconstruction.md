# Battle Reconstruction Layer (Phase 0) — design

**Status: BUILT + SHIPPED (`0f30c2e`, 2026-06-11).** The probe-agnostic observability /
replayability foundation for counterfactual battle analysis. Phase 1 (the
luck-vs-mistake falsifier — `design_falsifier_phase1.md`, or any other consumer:
distributional-critic validation, policy what-ifs, search) builds ON this; nothing
probe-specific is baked in here. **As-built record (files, gates, module map):
`impl_step9_battle_reconstruction_and_falsifier.md`.**

## What it gives us

1. **Every bridge battle is fully reconstructable after the fact.** The sim
   already mints a fresh random PRNG seed per battle (eval stays diverse — no
   pinned seed); we now *capture* it, with both packed teams and the complete
   command sequence, so any eval battle can be re-run bit-for-bit offline.
2. **A reusable reconstruct → re-roll → materialize-obs primitive.** Given a
   record: rebuild the battle to the start of any turn `t`, resolve that one
   turn under N fresh seeds (PRNG swap — all sim randomness routes through
   `battle.prng`), and rebuild the **agent's one-sided observation** of any
   point via the real encoder pipeline.

## The hard constraint (and how it's enforced structurally)

**One-sided (agent) and full-information (referee) views never mix.**

- The record (seed + both teams + dice) is captured only at the **bridge/sim
  layer** (`local_sim_bridge.js` emits a `__RECON__` frame at battle end) and
  persisted only as a **separate artifact** (`*_reconstruction.json` beside the
  forensic trace). `battle_recorder.py` / `states.npz` / the obs pipeline never
  touch it.
- Offline obs come from `agents.training.obs_materializer`, which is fed only
  the **per-side protocol chunks** the replay regenerates — the same bytes the
  live agent saw — and replays them through the unmodified poke-env →
  `Gen3Battle` → encoder stack (poke-env stays purely one-sided).
- Omniscient outputs of the re-roll API (`pre_state`, `outcome`, `turn_log`)
  are explicitly namespaced for ground-truth analysis; the materializer cannot
  consume them.

## Capture (what / where)

- **`src/utils/bridge/local_sim_bridge.js`** — records every command it
  processes; at battle end emits `__RECON__ <b64 json>` before `__END__`:
  `{v, format_id, prng_seed, input_log, commands}`.
  - `input_log` = `battle.inputLog` — the sim's normalized record (the `>start`
    line carries the **resolved** seed; `>player` lines carry both packed
    teams; committed choices only). State-faithful.
  - `commands` = the raw `CHOOSE`/`FORCELOSE` lines in processing order —
    **including attempts the sim refused**. The `[Unavailable choice]`
    maybe-trapped probe never reaches `input_log`, but its `|error|` +
    re-request round is part of the agent's protocol (and a rejection event in
    the event log) — replaying raw commands regenerates it exactly.
    Protocol-faithful.
- **`run_local_battles._demux`** (eval/fuzz path) routes the frame to a bounded
  registry; **`BridgeSession._dispatch`** (training transport) keeps a
  single-slot `last_recon` instead (training persists no traces; a 64-record
  registry × 64 env workers would pin hundreds of MB for nothing).
- **Join:** the frame arrives *after* the `|win|` chunks, i.e. after the eval
  forensic trace is written — so capture and trace-writing meet in
  `reconstruction.py`'s registry keyed by battle tag (`offer_record` /
  `register_trace_prefix`); whichever side lands second writes the artifact.
  `EvalRLPlayer._battle_finished_callback` registers the prefix (+
  `trainee_username`). Websocket eval produces no record → pending entries just
  evict (graceful).
- **`states.npz` gains an `actions` array** (agent-side, chosen action index per
  decision) — required to re-advance the turn-history tracker in replay.
  Additive; old traces simply lack it.

## Offline API (`src/utils/bridge/reconstruction.py` + `replay_driver.js`)

```python
record = ReconstructionRecord.load("…_reconstruction.json")

replay_battle(record) -> BattleReplay            # {p1_chunks, p2_chunks, outcome}

reroll_turn(record, t, seeds=[…],                # sim seeds: "a,b,c,d" | "sodium,<hex>"
            p1_action="recorded",                 # "recorded" | "random" | explicit "move 2"
            p2_action="recorded",
            followup="random")                    # mid-turn forced switches: "random" | "default"
  -> RerollResult{turn, pre_state, requests, recorded_choices,
                  prefix_p1_chunks, prefix_p2_chunks,
                  rerolls: [TurnReroll{seed, choices_used, outcome, turn_log,
                                       p1_chunks, p2_chunks}]}

# agents/training/obs_materializer.py — the one-sided wall's only crossing point
materialize_from_record(record, actions=npz["actions"]) -> MaterializedTrace
materialize_decisions(chunks, username=…, packed_team=…, side=…, actions=…)
```

- A **policy action source is composition, not new machinery**: materialize the
  obs at the decision point, run a policy, pass its pick as an explicit choice
  string. Counterfactual continuations likewise: feed
  `prefix_chunks + reroll.pN_chunks` to the materializer; one row past the last
  provided action is still valid (an obs depends on prior actions only) — which
  is exactly the V(s′) row a re-roll consumer needs.
- `replay_driver.js` is batch JSON-over-stdio (one request → one response, no
  server). Re-roll rebuilds the prefix per seed (~10–30 ms each);
  `State.serializeBattle` cloning is the known ~2–3× optimization if a consumer
  ever needs thousands of re-rolls per decision.

## Validation (all green, 2026-06-11)

- **Round-trip (the load-bearing proof)** —
  `agents/training/obs_roundtrip_fuzz_test.py`: real bridge battles, live obs
  recorded through the real `BattleRecorder` path; offline
  `replay_battle → materialize` must equal `states.npz` **bit-for-bit**.
  1124/1124 decisions over 6 sequential + 6 **concurrent** (`concurrency=3`,
  interleaved POKE_LOOP feeds — eval's latency-hiding mode) battles, plus
  463/463 + 188/188 in earlier runs. Proves the capture is complete, the replay
  regenerates the exact one-sided protocol, and the materializer mirrors the
  live decision cadence (stall check → embed → zero-mask deferral →
  `tracker.advance`), including HiddenPower/turn-history/progress-clock state.
- **Replay/re-roll invariants** — `utils/bridge/reconstruction_fuzz_test.py`:
  outcome matches the live result (turn + winner); replay and re-roll are
  deterministic functions of (record, seed); re-rolled timelines always advance
  (incl. forced switches the original timeline never had); distinct seeds split
  outcomes. The only nondeterminism found and excluded: `|t:|` wall-clock lines
  (in poke-env's `MESSAGES_TO_IGNORE` — state/obs-invisible).
- **Registry + record units** — `utils/bridge/reconstruction_test.py` (join in
  both arrival orders, FIFO bounds); `bridge_session_test.py` guards that a
  `__RECON__` frame can never be fed to a player client.

## Deferred (Phase 1+, deliberately not here)

- The falsifier itself: fix-both vs re-sample policy, what to measure
  (P(outcome), V-distribution vs dice distribution), seed counts, which
  decisions to sample. **→ BUILT 2026-06-11: `design_falsifier_phase1.md`**
  (which also added the `"original"` seed + recorded-queue driver semantics and
  the materializer's `map_actions_at`/`stop_after_decision`).
- The 3-way determinization (dice vs hidden-team-uncertainty vs critic-bias) —
  needs team-pool sampling for the opponent's unseen set; out of scope.
- Websocket-eval capture (needs a server-side hook; bridge eval is the
  supported path — `--use-showdown-bridge`).
- Clone-based re-roll fast path; re-anchoring a re-roll at a mid-turn
  forced-switch decision (currently anchors at start-of-turn move rounds).
