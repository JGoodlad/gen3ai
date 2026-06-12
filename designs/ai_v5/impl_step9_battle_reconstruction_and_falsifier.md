# Implementation: Step 9 — Battle reconstruction layer + dice-attribution falsifier

Separate **luck from reducible mistake** in the plateau's loss craters. The Step-3/triage tooling
showed ~49% of losses pass through value craters, but a crater is ambiguous: an **aleatoric** one (a
fair coinflip lost — irreducible; argues for a distributional critic that prices variance) and an
**epistemic/policy** one (a better line existed — argues obs/reward/policy) demand different levers,
and saved traces cannot tell them apart — the 3409-dim obs vector is not invertible back to tracker
state, and counterfactual next-states never happened, so nothing on disk could score "what would the
dice / another action have done". The fix is structural: make every eval battle **fully
reconstructable** (seed + teams + commands), add a **re-roll** primitive (replay to turn *t*, swap
the PRNG, resolve that one turn under N fresh seeds), and a **materializer** that rebuilds the
agent's one-sided obs through the real encoder — then build the falsifier on top.

> **Status: BUILT & SHIPPED**, two commits: `0f30c2e` (Phase 0 — reconstruction layer) +
> `2236b76` (Phase 1 — falsifier). Pure tooling: **no obs/arch/config change** → `ARCH_SIGNATURE`
> unchanged (`gen3_incoming_crit_split_v1`, obs 3409), `MODEL_CONFIG_VERSION` unchanged (11), no new
> training flags. Always-on at the bridge layer (capture is ~µs/battle); artifact persistence rides
> the existing forensic-trace quota. As-built record; the design docs are
> `design_battle_reconstruction.md` (Phase 0) and `design_falsifier_phase1.md` (Phase 1).
> **Data gate:** no run has reconstruction records yet — the live run picks the layer up at its next
> 3h launcher restart (`--sync-to-main`); run-level falsification waits on that.

---

## What shipped (one paragraph)

Every bridge battle now emits a `__RECON__` frame at battle end (resolved PRNG seed + both packed
teams + the sim's `inputLog` + the **raw command log**, including choices the sim refused); eval
forensic traces gain a fourth sibling `*_reconstruction.json` (joined by battle tag), and
`states.npz` gains the chosen-`actions` array. `replay_driver.js` + `reconstruction.py` replay a
battle verbatim or **re-roll one turn** under fresh seeds by swapping `battle.prng` in place;
`obs_materializer.py` replays the regenerated one-sided protocol through the real obs pipeline —
proven **bit-for-bit equal** to the live `states.npz` rows (1124/1124 decisions, sequential +
concurrent eval modes). On top, the prober gains `falsify` (`falsifier.py`, `ProbeSession.falsify`,
`python -m main.prober.query falsify`): per anchored decision, a **fix-both re-roll** places the
realized outcome in its dice distribution (`luck_percentile`) and a **paired alternative sweep**
(same seeds = common random numbers) measures whether a better legal action existed → verdicts
`LUCK` / `MISTAKE` / `MIXED` / `NEUTRAL`.

## The hard requirement: the one-sided / omniscient wall

The reconstruction record holds the opponent's full team and the dice — referee-view data. It is
captured only at the **bridge/sim layer** and persisted only as a **separate artifact**; nothing in
the obs/training path reads it (`battle_recorder.py` untouched by it; the obs pipeline consumes
poke-env's partial view exactly as live). Offline obs come only from the **per-side protocol
chunks** the replay regenerates — the same bytes the live agent saw. The falsifier's *margins* are
deliberately omniscient (what actually happened on the board is analysis, not model input); the wall
constrains what feeds the encoder/critic, never what a forensic report may read.

## Artifacts & record format

| Artifact | New? | Contents |
|---|---|---|
| `__RECON__ <b64 json>` (bridge stdout, before `__END__`) | NEW | `{v, format_id, prng_seed, input_log, commands}` |
| `*_reconstruction.json` (4th trace sibling) | NEW | the record + `battle_tag` + `trainee_username` |
| `*_states.npz` | +1 array | `actions` (chosen action idx per decision — replay needs it to re-advance the turn-history tracker) |
| `BridgeSession.last_recon` | NEW | single-slot `(tag, b64)`, training transport only (a bounded registry × 64 env workers would pin ~100s of MB for traces training never writes) |

Two logs on purpose: `input_log` (`battle.inputLog`) is **state-faithful** but logs only *committed*
choices; the raw `commands` log is **protocol-faithful** — a refused `[Unavailable choice]`
maybe-trapped probe never reaches `inputLog`, yet its `|error|` + re-request round is part of the
protocol the agent saw (and a rejection event in the event log → obs). Replaying raw commands
regenerates it exactly. The `>start` line carries the **resolved** seed (the sim mints a fresh
random one per battle when none is passed — eval stays diverse, reproducible after the fact, no
pinned seed).

**Capture join:** `__RECON__` arrives *after* the `|win|` chunks — i.e. after the forensic trace is
already written — so capture and trace-writing meet in a bounded (64, FIFO) registry keyed by battle
tag (`offer_record` from the demux / `register_trace_prefix` from `EvalRLPlayer`); whichever lands
second writes the artifact. Quota-dropped battles and websocket eval just evict (graceful).

## Offline API

| Primitive | What it does |
|---|---|
| `replay_battle(record)` | re-run verbatim → per-side chunks + final omniscient outcome. Byte-identical to live modulo `\|t:\|` wall-clock lines (the ONE nondeterminism found; in poke-env's `MESSAGES_TO_IGNORE` → state/obs-invisible) |
| `record.team_details(side)` / `decode_packed_team(packed)` | THE one decode home for review tooling: either side's full team (moves, EVs, IVs, nature, item, ability, level), omission-defaults applied, ids resolved through the sim's alias table. Delegates to poke-env's `parse_packed_team`; validated vs the sim's `Teams.unpack` over the ENTIRE pool (`packed_team_decode_integration_test.py`, 719 teams / 4314 mons — caught a real `wisp`→`willowisp` alias case on its first run). Replay/re-roll never decode: packed strings return to the sim verbatim |
| `reroll_turn(record, t, seeds=…, p1_action/p2_action, followup)` | rebuild to the start of turn *t*, swap `battle.prng` per seed, resolve that turn. Action sources per side: `recorded` (a QUEUE — a live refused-then-corrected sequence replays faithfully under fresh seeds) / `random` / explicit choice string. Special seed `"original"` = **no swap**; with both sides recorded it feeds the remaining recorded commands verbatim → the REALIZED line, scored through the same outcome pipeline |
| `materialize_decisions(chunks, …)` / `materialize_from_record(record, …)` | replay one side's chunks through the REAL obs pipeline (`_ReplayObsPlayer` mirrors `EvalRLPlayer`'s cadence: stall → embed → zero-mask deferral → `tracker.advance(actions[i])`). `map_actions_at=i` → legal action idx → sim choice string via the real mapper; `stop_after_decision=i` → cheap prefix replay. One row past the last provided action is still valid (an obs depends on *prior* actions only) — the V(s′) row a re-roll consumer needs |
| `ProbeSession.falsify(battle_id, invs=, worst=, n_seeds=, n_alts=, followup=)` | the Phase-1 probe (below); model-free, requires the `*_reconstruction.json` sibling |

Driver routing invariant (`resolveTurn`): within one turn a side gets at most ONE `'move'` request —
refusals re-ask as `'move'`, mid-turn follow-ups (forced switch after a faint) ask as `'switch'` —
so `'move'` → the configured source, anything else → the `followup` policy (`random` via an aux PRNG
derived from the re-roll seed — independent of the sim dice under study — or `default`).

## The falsifier (Phase 1)

Per anchored `move_selection` decision (re-rolls anchor at start-of-turn rounds; a forced-switch
crater attributes to its turn's move decision):

- **Luck axis** — fix both actions at the recorded picks, re-roll under N fresh seeds (default 40);
  the realized line comes from `"original"`. `luck_percentile` = midrank of the realized **material
  margin** (`(our_alive − opp_alive) + (our_hp_frac − opp_hp_frac)`) within the fresh-seed margins.
- **Mistake axis** — top-k alternative legal actions by the saved logits, each under the **same** N
  seeds (common random numbers → the advantage is a mean of *paired* per-seed differences; dice
  noise cancels within pairs, SE ~5–10× tighter than independent arms). Action idx → sim choice via
  `map_actions_at` (the live mapper, legality = the live mask). Alts the sim refuses (maybe-trapped,
  detected via `[Unavailable choice]` in the one-sided suffix) report `refused_frac` and are
  excluded from the verdict when >0.5 — a refused "alternative" was never available.
- **Anchors** — default: the `worst` most-negative-δ decisions on distinct turns
  (δ = r + γV(s′) − V(s), the prober's formula); explicit `--inv` overrides.

| Verdict | Condition (constants in `falsifier.py`, echoed in output) |
|---|---|
| `MISTAKE` | best paired advantage > `0.5` material AND > `1.96·SE` |
| `LUCK` | realized percentile ≤ `0.20`, no such alternative |
| `MIXED` | both |
| `NEUTRAL` | neither |

## Gates (all green at ship)

| Gate | Result |
|---|---|
| Obs round-trip (`obs_roundtrip_fuzz_test.py`) — THE load-bearing proof | **1124/1124** decisions bit-identical (6 sequential + 6 `concurrency=3` battles; +463 +188 in earlier runs). Proves capture completeness, replay protocol fidelity, and materializer cadence in one assertion |
| Replay/re-roll invariants (`reconstruction_fuzz_test.py`) | live outcome reproduced (turn + winner); replay & re-roll deterministic per (record, seed); re-rolled timelines always advance; distinct seeds split outcomes |
| Registry + dispatch units | join in both arrival orders, FIFO bounds; a `__RECON__` frame can never reach a player client (`bridge_session_test.py`) |
| Falsifier units (`falsifier_test.py`, 14) | margin antisymmetry, midrank percentile, paired stats (1 pair ⇒ ∞ SE ⇒ never significant), verdict matrix + threshold edges, seed determinism, δ-anchor ranking + forced-switch remap |
| Falsifier integration (real battle) | full pipeline valid + **deterministic on re-run**; sanity property: a RANDOM-policy trainee reads `MISTAKE` (e.g. `move hydropump` +0.89 ± 0.11 over the chosen switch, 24 paired seeds) while a bottom-decile dice outcome reads `LUCK` (pctl 0.083) |
| Packed-team decode sweep (`packed_team_decode_integration_test.py`) | `decode_packed_team` == the sim's `Teams.unpack`, field-by-field, over the **entire pool** (719 teams / 4314 mons, incl. every Hidden-Power IV spread); found + fixed one real alias case (`wisp`) on first run |
| Full unit suite | **2212 passed**, 2 skipped; integration suite + 30-episode persistent-bridge soak (incl. a 250-turn stall/forfeit) clean |

## Cost

Re-roll ≈ 10–30 ms/seed (fresh prefix rebuild; `State.serializeBattle` cloning is the known ~2–3×
fast path if ever needed); one falsified decision at 40 seeds × (1 + 3 alts) ≈ **5–10 s**; a battle
at `--worst 3` ≈ 20–30 s. Capture overhead on live eval/training: ~µs + ~tens of KB per battle.

## Limitations (known, accepted)

- **Bridge-eval only** — websocket eval has no capture point (degrades silently); full coverage
  needs `--use-showdown-bridge` (the live run's config).
- **Persistence rides the forensic quota** (5 wins + 10 losses per opponent per cycle of the 100
  played); capture itself is always-on, so widening is a one-constant change.
- **Trainee-side obs only** — the board replays fully for both sides, but the opponent's belief
  state would need its action indices (not saved; choice-string→idx inverse is buildable).
- A true `[Invalid choice]` error round can't be mirrored (poke-env answers with a coin flip);
  the server-authoritative mask makes it a non-event. Re-rolls anchor at move rounds only.
- Fixing the opponent's recorded action in the alt sweep leaks what they did that game
  (game-theoretic caveat; re-sampling needs an opponent policy source — deferred).

## Module map

| File | Change |
|---|---|
| `utils/bridge/local_sim_bridge.js` | command log + `__RECON__` emission (both child modes) |
| `utils/bridge/reconstruction.py` | **NEW** — record type, capture registry + artifact writer, `replay_battle` / `reroll_turn` |
| `utils/bridge/replay_driver.js` | **NEW** — batch JSON-over-stdio replay/re-roll driver (PRNG swap, `"original"` seed, recorded queues, followup policies) |
| `utils/bridge/local_battle_runner.py` / `bridge_session.py` | `__RECON__` demux/dispatch (registry vs single-slot) |
| `agents/training/battle_recorder.py` | `actions` array in `states_arrays()` |
| `agents/training/eval_callback.py` | `register_trace_prefix` after each persisted trace |
| `agents/training/obs_materializer.py` | **NEW** — one-sided obs replay through the real encoder (+ `map_actions_at`, `stop_after_decision`) |
| `agents/training/obs_roundtrip_fuzz_test.py` | **NEW** — the bit-for-bit round-trip gate (sequential + concurrent) |
| `utils/bridge/reconstruction_test.py` / `reconstruction_fuzz_test.py` | **NEW** — registry + decode units / replay+re-roll invariants |
| `utils/bridge/packed_team_decode_integration_test.py` | **NEW** — full-pool decode sweep vs the sim's `Teams.unpack` |
| `main/prober/falsifier.py` (+`falsifier_test.py`, `falsifier_integration_test.py`) | **NEW** — the dice-attribution engine + tests |
| `main/prober/session.py` / `query.py` | `falsify` method + CLI verb |
| docs | root + `training/` + `prober/` `CLAUDE.md`, bridge `README.md`, the two design docs, this doc |

## Forward design / context

- Design docs: `design_battle_reconstruction.md` (Phase 0 — the layer, the wall, capture mechanics)
  and `design_falsifier_phase1.md` (Phase 1 — the locked probe decisions).
- Motivated by the Step-3-era plateau diagnosis (2026-06-09): lever 2 (tail-calibrated critic) vs
  irreducible aleatoric variance was unresolved; the falsifier measures that split.
- **Next:** run-level `falsify-scan` (falsify the worst decision of every loss at a step → the
  run's luck-vs-mistake share — the number that decides how much of the td-tail a distributional
  critic can even recover), once the live run writes records. Then optionally the critic-scored arm
  (V(s′) over materialized one-sided obs — distributional-critic calibration) and TUI parity.
