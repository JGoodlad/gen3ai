# Self-Play Stale-Decision Race — context, fix, and verification

This documents the **self-play opponent stale-decision race**: what it is, why it was hard to
pin down, how it's fixed, and the **layered verification systems** that reproduce it and prove
the fix. If this class of bug ever resurfaces, start here — it took ~10 attempts the first time,
and almost all of that was *understanding the shape*, not writing the fix.

---

## TL;DR

- **Symptom:** during **self-play** training, a worker crashes with
  `StaleDecisionError: Mid-decision state change ... [moves] latched=(4 moves) server=()` (the
  move-face) or `... switch slot N (heracross) is not a currently available switch` (the
  switch-face). It is **rare** (a narrow timing window) and only at **self-play scale**.
- **Root cause:** the self-play opponent decides on the **training thread** while POKE_LOOP (a
  *separate* thread) mutates its battle. It latches a legal-actions snapshot, then serializes the
  chosen action against the **live** battle a moment later — and in between, a `parse_request`
  (usually a faint→force-switch) changes the request under it. The per-battle `asyncio` lock does
  **not** protect this: asyncio locks serialize *coroutines on the event loop*, not a *different
  thread*.
- **Fix (defense in depth):**
  1. **Prevent** — the wrapper drains POKE_LOOP's in-flight work for the opponent's battle before
     polling it (`SingleAgentWrapper._settle_opponent_battle`).
  2. **Detect** — `Gen3ActionMapper.assert_decision_current` compares **every** action-relevant
     axis (moves incl. disabled, switches incl. species, force_switch, trapped, maybe_trapped,
     wait, struggle).
  3. **Backstop** — the opponent is **strict**: a stale decision **crashes** (crash-over-
     corruption), exactly like the trainee. It never defers to a default move — in self-play the
     opponent *is* the trainee's training signal, so a garbage default is garbage-in.

---

## The mechanism

A `RLPlayer` (the trainee *and* the self-play opponent) decides in **two phases**:

1. **Latch** a `LegalActions` snapshot (`ctx.legal`) at observation time (`embed_battle`).
2. **Validate + serialize** the chosen action against the **live** battle later
   (`assert_decision_current` → `choice_to_order`).

The gap between phase 1 and phase 2 is the danger window. poke-env runs on `POKE_LOOP`, a daemon
thread that parses incoming Showdown messages. If, during that window, POKE_LOOP parses a message
that shifts the request — most commonly the active fainting, which flips the battle to a
**force-switch** (`available_moves → []`) — then the snapshot no longer matches the live battle.
Acting on it would send the wrong order, so we raise `StaleDecisionError`.

```
training thread                         POKE_LOOP (separate thread)
---------------                         ---------------------------
opponent.choose_move(battle2)
  embed_battle → latch ctx.legal
    (4 moves, force_switch=False)
                                        parse_request: aerodactyl fainted
                                        → battle2.available_moves = []
                                        → battle2.force_switch = True
  action_to_order:
    assert_decision_current(ctx, battle2)
    → latched (4 moves) != live ()  →  StaleDecisionError   ← the crash
```

---

## Why **self-play** specifically (and why vs-bots was fine for days)

- **The latch pattern is the ingredient, not "an RL model".** Heuristic bots read the live battle
  and emit an order in a single pass — no captured snapshot, nothing to diverge from. So vs-bots,
  the opponent side can't hit this. Self-play swaps that bot for a latch-pattern `RLPlayer`.
- **The trainee uses the same pattern but is gated.** The trainee's decision flows through
  poke-env's *request* path, whose drain (commit `199936d`) ensures the request's message batch is
  complete before the decision fires, and its window sits where the battle isn't being mutated.
  The self-play **opponent is polled out-of-band** by our own wrapper
  (`single_agent_wrapper.py`, `opponent.choose_move(battle2)`) on the training thread — it gets
  none of that gating. Same code, safe in one path, racy in the other.
- **`max_concurrent_battles` does NOT apply.** The env pins its connection agents to
  `max_concurrent_battles=1` (`env.py`), and the self-play opponent is a *brain*
  (`start_listening=False`, no connection — its limit is dead). The concurrency lever that drives
  the repro is `--n-envs` (SubprocVecEnv parallel battles), not per-player concurrent battles.

---

## Why this was so hard (read before re-debugging)

1. **It lives in a seam no single component owns** — poke-env (async, POKE_LOOP) × our wrapper
   (sync, training thread) × our latch decision pattern. Each piece is correct alone; the bug is
   in the *interaction*. Reading any one file doesn't reveal it.
2. **The lock gave false confidence.** "There's a per-battle `asyncio` lock, so reads are safe" is
   the natural — and wrong — assumption. An `asyncio.Lock` only serializes coroutines on its loop;
   it does **not** block another *thread*. Distrust "the lock protects it" across a thread boundary.
3. **We were fuzzing the wrong opponent.** A fresh run is ~80% heuristic bots, which don't touch
   the latch path. So our fuzzing "passed" while never exercising self-play. **The single biggest
   unlock was forcing 100% self-play** (`GEN3_FORCE_SELFPLAY=1`). Reproduce the *production
   condition* before theorizing.
4. **The asymmetry trap.** The trainee uses the same pattern and ran for *days* vs-bots, "proving"
   the code works — but it's gated and the opponent isn't. When the same pattern is safe in one
   path and racy in another, **map both decision paths explicitly** (who polls it, on what thread,
   gated by what).
5. **An earlier fix masked it.** The drain (`199936d`) genuinely fixed the *trainee's* incomplete-
   batch problem — "never saw it again." But it lives in poke-env's request path and doesn't cover
   the out-of-band opponent poll. A fix that makes the symptom you see disappear can hide the one
   you don't. Verify a fix covers **every caller** of the pattern.
6. **Rare + timing-dependent = unreproducible-on-demand.** Each "it didn't crash" was inconclusive
   (absence of evidence ≠ evidence of absence). The race needs scale + slow decisions to hit, so
   cheap single-env attempts told us nothing. There is **no seconds-fast shortcut** — the cheapest
   *reliable* check is the deterministic unit guard (tier 1); the real-race repro takes real time
   (tens of minutes). We tried to manufacture a fast version (random-policy + `widen`, multi-proc)
   and it was *worse* than the real model — the rarity is intrinsic.
7. **The crash pointed downstream.** It surfaced deep in `choice_to_order` ("switch not
   available"), which looks like an action-mapping bug; the cause is upstream (the battle mutated
   between snapshot and serialize). Symptom location ≠ cause location.

---

## The fix, in code

| Layer | Where | What |
|---|---|---|
| **Prevent** | `poke_env/environment/single_agent_wrapper.py` → `_settle_opponent_battle` | Before polling the opponent, run a coroutine on POKE_LOOP that **yields until the battle's decision signature is stable** across a scheduler tick (the server is waiting on our move, so once nothing changes, nothing else lands). Drains the in-flight `parse_request`. Bounded + best-effort. |
| **Detect** | `agents/action/mapper.py` → `assert_decision_current` | Compares **every** action-relevant axis of the snapshot vs the live request (moves with `disabled`, switches with `species`, `force_switch`, `trapped`, `maybe_trapped`, `wait`, `struggle`). Any divergence → `StaleDecisionError`. `action_to_order` is the final backstop for the residual window before serialization. |
| **Backstop** | `agents/inference/player.py` → `choose_move` | The opponent is **strict** — `StaleDecisionError` propagates and crashes the worker (launcher restarts from the last checkpoint). No tolerance path, no default-move substitution. |

---

## Verification systems (three tiers, fast → faithful)

> Live-server tiers need a Showdown server on a **private 9XXX port — NEVER 8000/8001**
> (`npm run showdown -- 9124`). **`GEN3_FORCE_SELFPLAY=1`** forces 100% self-play opponents (the
> only knob; the settle is **unconditional** in production — there is no "disable" switch). To
> demonstrate the FAIL on purpose, comment out the `self._settle_opponent_battle()` call in
> `SingleAgentWrapper.step` and re-run; `single_agent_wrapper_test.py` guards that the call stays.
>
> **Budget real time for the live validation — tens of minutes, up to ~half an hour or more.**
> The race is intrinsically rare (~one hit per ~6,000 self-play decisions), so reproducing it and
> then proving it's *gone* (a 5× clean run past that point) is **not** a seconds-fast loop. Tier 1
> is the instant reliable check; tier 3 is the real-race confidence. There is no cheaper shortcut —
> that rarity is the whole reason this was hard to pin down.

### 1. Deterministic unit guard — *instant, no server*
`agents/action/mapper_test.py::TestAssertDecisionDefensiveness` (+ `TestStaleDecisionStrict` in
`agents/inference/player_test.py`). Injects each divergence axis (headlined by the exact
faint→force-switch case) and asserts we raise; asserts the opponent crashes, never defaults. **If
the guard is ever narrowed back toward "move-ids only", or tolerance returns, these go red.**
```bash
pytest src/agents/action/mapper_test.py::TestAssertDecisionDefensiveness \
       src/agents/inference/player_test.py::TestStaleDecisionStrict -q
```

### 2. Single-env race fuzz — *real path, lightweight (~5–15 min)*
`agents/training/racing_player_fuzz_e2e_test.py`. One isolated env through the REAL wrapper +
RLPlayer path, random play, with a `--widen` sleep that holds the snapshot→serialize window open.
**`--widen 0.1` reproduces** (a divergence around ~4,000 decisions, verified); the old 3 ms default
saw nothing — which is exactly why early runs looked falsely clean. Reports snapshot-vs-live
divergences.
```bash
npm run showdown -- 9124
# REGRESSION CHECK (settle on, as shipped) -- expect 0 divergences:
python src/agents/training/racing_player_fuzz_e2e_test.py --port 9124 --widen 0.1 --episodes 250
# To see the FAIL on purpose: comment out the _settle_opponent_battle() call in
# SingleAgentWrapper.step, re-run the same command -> divergences > 0.
```
> A standalone *multi-process* "artificial" launcher was prototyped and **removed**: extra
> processes don't pool the per-battle hit (each still needs ~thousands of its *own* decisions), and
> a real model forward reproduces the timing *better* than a random-policy + `widen` sleep. The
> faithful stress (tier 3) is the better fast lever — just scale `--n-envs` on a box with spare
> cores/GPU.

### 3. Faithful stress — *production conditions, the gold standard*
The real trainer forced to 100% self-play — what originally reproduced the crash, and the
**fail→fix→pass A/B** that validates the settle end-to-end.
```bash
npm run showdown -- 9124
# As shipped (settle on): runs clean (confidence run reached 5x the repro point, 0 crashes).
GEN3_FORCE_SELFPLAY=1 python src/main/train_rl_agent.py \
  --self-play --device cpu --n-envs 12 --steps 50000000 \
  --n-steps 256 --batch-size 256 --n-epochs 1 --showdown-port 9124 --log-level periodic
# To reproduce the crash: comment out the _settle_opponent_battle() call in
# SingleAgentWrapper.step. Measured then: 12 envs -> ~6,144 steps / ~900 s -> StaleDecisionError.
```

**Calibrating `--n-envs` (measured).** The hit is throughput-bound: ~6,000 *total* self-play
decisions to the first crash, so time-to-repro ≈ 6,000 / (total decisions·s⁻¹).
- On a **free GPU box**, raise `--n-envs` (32–64): more decisions/sec pulls the repro toward a few
  minutes.
- On a **CPU-bound / shared machine** (e.g. a training run already saturating the cores) total
  throughput is fixed, so more envs **don't** help — they only compete. Measured here: **12 envs
  and 24 envs both ran at fps≈7** (identical total throughput → CPU-bound), so ~12 envs / ~15 min
  is the floor. Don't bother scaling `--n-envs` on a busy box.

**Confidence = 5× clean.** Proving a rare event is *gone* means running settle-ON well past where
it reliably appears — target **5× the repro point** (~30,720 steps, on the order of an hour at the
~7 fps CPU rate). Recorded A/B:
- **FAIL (settle off):** crash @ **6,144 steps / ~900 s** (turn-8 `[moves]`, fainted aerodactyl).
- **PASS (settle on):** clean @ **15,360 steps** (2.5×) and extended to **~30,720 steps (5×)** with
  **zero** stale crashes.

---

## If it recurs

1. **Reproduce the production condition first** — run tier 3 (`GEN3_FORCE_SELFPLAY=1`), or tier 2
   (`--widen 0.1`) for a lighter loop. Don't theorize from a vs-bots run; budget tens of minutes.
2. **Confirm which axis diverged** — the `StaleDecisionError` message names it (`[moves]`,
   `[switches]`, `[force_switch]`, …). That tells you *what* shifted under the decision.
3. **Check the settle is still wired** — `_settle_opponent_battle` is called at the top of
   `SingleAgentWrapper.step`, before any opponent read. `single_agent_wrapper_test.py` guards
   that deterministically; if it's bypassed, tier 2 (run as-is) starts finding divergences.
4. **A new decider on the training thread?** Any *new* place that reads a battle for a decision
   off POKE_LOOP needs the same settle gating. Map its thread + gating like the table above.
5. **Don't reach for tolerance.** The strict crash is intentional (gigo). The answer is to close
   the window (settle), not to act on a stale snapshot.
