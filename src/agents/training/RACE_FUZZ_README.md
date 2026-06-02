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
- **Confirmed mechanism (race trace, `GEN3_RACE_TRACE=1`):** the kill isn't just "a parse lands in
  the snapshot→serialize gap" — it's that POKE_LOOP parses an **in-flight turn-resolution *during the
  model forward***, so by serialize time `battle.turn` is one *ahead* of `ctx.turn`. The trace caught
  it: `EMBED battle.turn=2 → [Thread-2 RX: moves, |turn|3] → ASSERT ctx.turn=2 battle.turn=3`. Trigger
  in the wild: both players are Arena-Trap Dugtrio → mutual trap → the turn resolves while the opponent
  is still computing.
- **Fix — split by who *owns* the decision (SB3 has no failed-step path):**
  1. **Opponent re-decides (the fix).** Its decision is *internal* to `SingleAgentWrapper.step` — SB3
     never sees it — so `RLPlayer.choose_move` catches the `StaleDecisionError` and **re-decides on
     the now-current request** (bounded `_OPP_REDECIDE_MAX`; valid default only if it never settles).
     A raise would kill the SB3 worker, so it must always return a valid order.
  2. **Trainee crashes (unchanged).** Its action is *SB3's*, computed outside the step and not
     re-runnable mid-step, so a stale trainee decision crashes — acting on it would corrupt its
     `(obs, action) → (reward, next_obs)`. (It's gated by the env's `race_get` and doesn't hit this.)
  3. **Detect** — `assert_decision_current` compares **every** action axis (moves incl. disabled,
     switches incl. species, force_switch, trapped, maybe_trapped, wait, struggle).
  4. **Pre-drain (optimization, not the fix)** — `_settle_opponent_battle` drains *already-arrived*
     messages to trim the re-decide rate; it can't drain in-flight ones (hence re-decide is the fix).

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
| **Resolve** (the fix) | `agents/inference/player.py` → `choose_move` | On `StaleDecisionError` the opponent **RE-DECIDES** on the now-current request, bounded by `_OPP_REDECIDE_MAX`, falling back to a valid `choose_default_move()` only if the battle never settles. Its decision is internal to `SingleAgentWrapper.step` (SB3 never sees it), so it must always return a valid order — a raise would kill the worker. |
| **No phantom** | `agents/training/episode_tracker.py` → `snapshot`/`restore`, called from `choose_move` | Each attempt's `embed_battle()` records its would-be decision into the rolling turn-history. `choose_move` snapshots before the loop and `restore()`s on a stale re-decide, so a superseded attempt leaves **no phantom turn** — only the committed decision survives in the opponent's obs. The snapshot captures exact deque contents (incl. an entry `maxlen` would drop), so the rollback is exact even deep in a long game. |
| **Detect** | `agents/action/mapper.py` → `assert_decision_current` | Compares **every** action-relevant axis of the snapshot vs the live request (moves with `disabled`, switches with `species`, `force_switch`, `trapped`, `maybe_trapped`, `wait`, `struggle`). Any divergence → `StaleDecisionError`. `action_to_order` is the final backstop for the residual window before serialization. |
| **Pre-drain** (optimization) | `poke_env/environment/single_agent_wrapper.py` → `_settle_opponent_battle` | Before polling, yields on POKE_LOOP until the decision signature is stable, draining *already-arrived* messages to REDUCE how often the re-decide fires. It can't drain a turn-resolution still *in flight* during the model forward, so it's an optimization, **not** the fix (that's why the strict-crash version still crashed at 64-env scale). |
| **Trainee** | `agents/training/gen3_env.py` | Stays **strict** — a stale trainee decision crashes (crash-over-corruption). See the next section for why this asymmetry is correct. |

---

## Why the trainee AND the opponent are both safe

The race can fire for *either* decider, but the safe response differs — and the difference is
principled, set by **who owns the decision**.

**The opponent is safe because its decision is ours to retry.** The self-play opponent is polled
*inside* `SingleAgentWrapper.step`; SB3 never sees this call. So when the battle advances under it
we simply **re-decide on the now-current request** and return a valid order. Three properties make
that airtight:
- *Always valid.* The re-decide loops on the live request until `action_to_order` serializes
  cleanly (bounded by `_OPP_REDECIDE_MAX`; a valid default if it never settles). SB3 has **no
  failed-step path** — a raised exception kills the `SubprocVecEnv` worker — so "always returns a
  valid order" is the contract, and the re-decide meets it.
- *No corrupted learning.* The opponent is a frozen snapshot with no labels; nothing it does feeds
  a gradient. Re-deciding can't corrupt a transition because there is no opponent transition.
- *No corrupted observation.* The one thing it carries forward is its turn-history obs, and the
  `snapshot`/`restore` rollback guarantees the superseded attempt's `record()` is undone — a
  re-decide leaves the history exactly as a single clean decision would. Verified end-to-end: 244
  forced re-decides → 0 phantom turns; with `restore` disabled, 178/178 decisions go phantom.

**The trainee is safe because it never acts stale — and crashes rather than guess if it ever
would.** The trainee's action comes from *SB3*, computed outside the env and handed back into
`step`; it cannot be re-run mid-step, so it can't re-decide. Two layers keep it safe:
- *It doesn't hit the race.* The env gates the trainee's decision on `race_get` — poke-env's own
  request-wait — so by the time the trainee observes, its request has settled. Empirically: 17 h of
  vs-bots + self-play, zero trainee staleness.
- *If it ever did, it crashes — by design.* A stale trainee decision raises and takes the worker
  down (launcher restarts from the last checkpoint). Acting on the stale snapshot would pair the
  wrong action with the obs/reward and corrupt its `(obs, action) → (reward, next_obs)` transition,
  poisoning the policy gradient. **Crash-over-corruption:** a restart costs minutes; a corrupted
  transition silently degrades the model forever. The trainee is the one decider whose mistakes are
  permanent, so it gets the strict path.

In one line: **the opponent re-decides because it can and a crash would needlessly kill the worker;
the trainee crashes because it can't re-decide and acting stale would poison learning.**

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

### 1. Deterministic unit + rollback guards — *instant, no server*
`agents/action/mapper_test.py::TestAssertDecisionDefensiveness` (injects each divergence axis,
headlined by the exact faint→force-switch case, and asserts we raise) +
`agents/inference/player_test.py::TestStaleDecisionRedecide` (the opponent re-decides on a stale
attempt, rolls it back, exhausts to a valid default, and **never** crashes — SB3 has no failed-step
path) + `agents/training/episode_tracker_test.py` snapshot/restore tests (the rollback undoes a
record exactly, recovers a `maxlen`-dropped entry, and leaves no phantom turn). **If the detector
is narrowed toward "move-ids only", or the rollback regresses, these go red.**
```bash
pytest src/agents/action/mapper_test.py::TestAssertDecisionDefensiveness \
       src/agents/inference/player_test.py::TestStaleDecisionRedecide \
       src/agents/training/episode_tracker_test.py -k "snapshot or restore or redecide or phantom" -q
```

### 1b. Re-decide rollback fuzz — *real battles, no server (~1–2 min)*
`agents/training/redecide_rollback_fuzz_test.py`. Plays real bridge battles and **forces the
re-decide path on every decision** (injects a one-shot `StaleDecisionError` before each serialize),
asserting per-decision that the committed history grows by ≤ 1 transition — i.e. the rollback left
no phantom turn — and that every battle still finishes (the re-decide always returns a valid order).
`_n_transitions` (monotonic, never capped) is the load-bearing invariant, so the phantom is caught
even when `len(_history)` has saturated at its cap. **Self-checking negative control:** disabling
`EpisodeTracker.restore` flips it to 178/178 decisions phantom — the test is not vacuously green.
```bash
python src/agents/training/redecide_rollback_fuzz_test.py 8
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
