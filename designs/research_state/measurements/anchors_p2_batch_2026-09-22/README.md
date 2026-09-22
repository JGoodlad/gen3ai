# `main.anchors` + `checkargs`: five backlog rows, and two of them were the SAME defect

**2026-09-22.** Five registered tech-debt rows, landed as five commits. CPU only
(`CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, `OMP_NUM_THREADS=1`), ports 9500–9599 stopped by PID,
:8000/:8001 never touched, nothing written under `models/`, a GPU arm and two other agents sharing
the box (load 11–19 throughout).

> ## The verdict
> **Hazard H14 — "`--opponent foulplay` cannot run a multi-game `ours_challenge` half" — was
> OURS, and it is closed.** poke-env's challenge loop releases its battle semaphore when a battle
> *starts*, so it emitted challenge *k+1* ~0.4 s INTO battle *k*; Foul Play reads its PMs only
> between battles, dropped it, and waited forever for a challenge already consumed. One extra
> `await` fixes it: **a 10-game Foul Play cell now completes 10/10, both halves, `status: OK`**,
> with the five challenges **11–37 s apart** instead of 0.4 s.
>
> 🚨 **And the Metamon `RecursionError` (H-H) is the SAME defect, on their side of the wire.**
> Metamon serializes its ACCEPTOR role — its own `_accept_challenge_loop` "fully awaits the
> battle before accepting the next" — and drives its CHALLENGER role through poke-env's
> **pipelined** loop. That, not the regime, is why the failure has always been in the half where
> Metamon challenges. The fix is upstream; the tool now NAMES it rather than leaving an rc.

---

## 1. What landed

| # | commit | row |
|---|---|---|
| 1 | `9eea8b04` | `main.anchors` with no `--out` wrote `anchors_out/` into the CALLING DIRECTORY |
| 2 | `558586d0` | `--opponent foulplay` cannot run a multi-game `ours_challenge` half (SOP **H14**) |
| 3 | `ae64dfe0` | the Metamon `mixed · Metamon challenges` half-cell fails identically in three campaigns (**H-H**) |
| 4 | *(this batch)* | `checkargs --argv` leading-token strip **+** `checkargs` pinned checkout incomplete |
| 5 | *(this batch)* | `run_local_battles`' docstring stale about the rust bridge |

## 2. The gate table

| # | gate | result |
|---|---|---|
| **a** | anchors unit suite | **209 passed** (from 167), 0 failed |
| **b** | 🚨 **Foul Play, 10 games, `--server rust`, `--search-time-ms 100`, arm W @ 75,005,952** | **10/10, `status: OK`** — `ours_challenge` 5/5 (the half that could not run), `peer_challenge` 5/5; regime VERIFIED, `peer_clean` true, **0 ERRORs** in the front end's log, realized **84,205** visits/decision; W4/L6 → 0.400 [0.168, 0.687] |
| **b′** | the protocol evidence | the five `challenge gen3aianchor1 -> foulplay1` lines are **11–37 s apart** (`cells/fp10_serial/challenge_timing.txt`) against H14's **0.4 s** signature |
| **c** | Metamon control, 20 games, `SmallRL` greedy/away, seed 920001 | **20/20, `status: OK`**, `argmax_match_rate` **1.0000**, `peer_clean` true, **0.600 [0.387, 0.781]** |
| **d** | Metamon challenger-role gate, 20 games, seed 920077 | **20/20, `status: OK`**, 10 games in the challenger role, `argmax_match_rate` 1.0000, peer exited CLEAN, 0.500 [0.299, 0.701] |
| **e** | `checkargs --argv` on the archived `ai_v12_01_winprob_critic` command | `✗ argv REFUSED … unconsumed value '…/launcher/__main__.py'` → **`✅ argv validated against PINNED parser @e798c13a via the pinned build_parser()`** |
| **e′** | the pinned checkout, `designs/baselines.json` hidden | reproduces the recorded `BaselineError: no baseline registry at /tmp/pinned-argv-…/designs/baselines.json`, demoting to `ast_scan`; **with the fix the same commit reaches `build_parser`** and a future gap reads `🚨 incomplete_pinned_checkout` |
| **f** | routine gate, each landing | 11,093 → 11,107 → 11,144 → (this batch) passed, 0 failed |

⚠️ **The Foul Play cell is a PROTOCOL gate, not a strength read.** `team_source_asymmetry` is
true on the home set by construction (ours 719 / theirs 72), n = 10, and the budget is 100 ms
against the SOP's 1000 ms. No number here belongs in a strength comparison.

## 3. H14 — the mechanism, and why it was ours

`Player._send_challenges` is `challenge(); await _battle_semaphore.acquire()`. The semaphore is
released in `_create_battle`, i.e. when the battle **starts** — so the loop sends the next
challenge while the current battle is still being played. Foul Play's own log then sits at
`Waiting for a gen3ou challenge` for a PM it discarded.

`--challenge-mode serial` (the DEFAULT) adds one await: `_battle_count_queue` holds one unfinished
item per LIVE battle (`put` in `_create_battle` **before** the semaphore release, `get`/`task_done`
on `|win|`/`|tie|`), so `join()` after the acquire returns exactly when that battle has ENDED.

🚨 **Nothing about the play changes.** At `--concurrency 1` the queue's maxsize is 1, so
`_create_battle`'s `put` already blocked battle *k+1* from starting before battle *k* ended. Only
the moment the PM is emitted moves. Serialization is **REFUSED above concurrency 1** rather than
honouring that flag as its own opposite.

🚨 **The patch goes on poke-env's BASE `Player`, and that is correctness, not style.**
`Gen3Player._send_challenges` is a WRAPPER: it awaits `_await_connected` — the connect-or-raise
deadline that names a login `action.php` refused instead of spinning until someone else's timeout
— and only then calls `super()`. An override on the leaf class replaces that wrapper whole and
deletes the guard, silently, in exactly the configuration (a local `--no-security` server that
still authenticates a registered name) the guard exists for. One base patch also covers all nine
roster bots. A test stands on this specifically.

## 4. H-H — root-caused UPSTREAM, and why the row's own prescription was not taken

From `wcont_control_read_2026-09-20`'s challenger log: ~494 repetitions of

```
Battle is already finished, call reset
Force resetting due to long-tail error
```

then `RecursionError`. Two upstream facts:

1. **`MetamonAMAGOWrapper.step`** (`metamon/rl/metamon_to_amago.py`) answers **any** exception
   with `self.reset(); return self.step(action)` — no depth bound, no re-raise, no distinction
   between a transient and a permanent error. ~988 frames, then `RecursionError`.
2. 🚨 **Metamon serializes one role and leaves the other pipelined.** `metamon/env/wrappers.py`:
   the ACCEPTOR role runs its own `_accept_challenge_loop`, documented as *"Accepts one challenge
   at a time and fully awaits the battle before accepting the next … ensuring terminated/truncated
   signals propagate correctly"*; the CHALLENGER role runs
   `self.start_challenging(n_challenges=num_battles)` — poke-env's pipelined loop, the H14 class.
   The env's `current_battle` and the agent's desynchronise, and `openai_api.py:362` raises
   `RuntimeError("Battle is already finished, call reset")`.

That is precisely why F-D (`flywheel_wb_floor_read_2026-09-18`) found the common factor across all
four occurrences to be **Metamon CHALLENGING**, not Metamon sampling.

### The minimal upstream patch (documented, NOT applied — Metamon is not ours)

```python
# metamon/env/wrappers.py — mirror what the acceptor role already does
if role == "challenger":
    self._challenge_task = asyncio.run_coroutine_threadsafe(
        self._send_challenge_loop(num_battles), POKE_LOOP)
...
async def _send_challenge_loop(self, n_challenges: int):
    """One challenge at a time, fully awaiting the battle — the mirror of
    `_accept_challenge_loop`, which already does this and for the stated reason."""
    for _ in range(n_challenges):
        await self.agent.send_challenges(self._opponent_username, 1)

# metamon/rl/metamon_to_amago.py — bound the handler
except Exception as e:
    self._long_tail_retries = getattr(self, "_long_tail_retries", 0) + 1
    if self._long_tail_retries > 3:
        raise                     # a permanent error must not become a RecursionError
    ...
```

### 🚨 Why the backlog row's prescription is NOT taken

The row asked `main.anchors` to *"refuse that role/regime combination"*. Both spellings are wrong:

* **on the REGIME axis** it refuses on the axis F-D retired — the fourth occurrence was a MATCHED
  greedy cell, so `mixed` is not the property that predicts the crash;
* **on the ROLE axis** it refuses half of **every** standing read, for a teardown that happens
  after the last decision and costs no games. The 2026-09-20 re-derivation is the evidence: 31 of
  84 sub-cells, every one with `argmax_match_rate` 1.0000, all flip to verified, no win rate moves.

So the landing is **name it, do not refuse it**: a COMPLETE `peer_challenge` half whose Metamon
peer died of a `RecursionError` is stamped **`peer_recursion_upstream`** in `summary.json` and
printed with its cause. An INCOMPLETE half, the other half, a non-Metamon peer, or any other error
gets **no** excuse — the classifier must not launder a short series. The **`mixed` cell alone IS
refused** (`--allow-unmatched-regime` against metamon): three crashes, no recorded success, and
unreadable beside any other number by SOP rule 1.

## 5. Hazards — each one is a finding

| # | hazard | what it means |
|---|---|---|
| **H-1** | 🚨 **`Gen3Player` wraps `_send_challenges`** to add the connect-or-raise deadline. The first version of the H14 fix overrode `RLPlayer` and would have DELETED that guard in silence | patch the BASE, and keep a test that names the wrapper. Found only because a test asserted the wrapper survived |
| **H-2** | 🚨 **The incomplete-checkout detector's first version substring-matched the `pinned-argv-` prefix and FALSE-POSITIVED** on the genuine *"this commit predates `build_parser()`"* reason, which also names `<tmp>/src` | the claim is now CHECKED on disk — every path the reason names inside the tmp tree is tested, and the finding fires only for one that is NOT there. An existing test caught the false positive |
| **H-3** | ⚠️ **The 20-game challenger-role gate did NOT fire the upstream recursion** (peer exited clean, 0 forfeits) | it cannot be produced on demand — 4 occurrences in 800 games. The classifier is covered by unit tests against the exact recorded signature; the live cell shows only that the role runs and nothing regressed |
| **H-4** | ⚠️ **`designs/baselines.json` does not exist at every commit a pin can name** (it is absent at `e798c13a`) | `_tree_paths` already filters by existence, so naming it is safe — but it means a pin below that commit still cannot hit the `baselines.json` path, and that is a property of the pin, not a gap |
| **H-5** | ⚠️ **The Metamon control is n = 20, not a replicate of the banked n = 100** | 0.600 [0.387, 0.781] sits on top of the banked 0.590 [0.492, 0.681] for this cell and transport. It is a "did not regress" check and nothing more |
| **H-6** | ⚠️ **Three commits' hashes moved under a rebase** between writing the backlog row and landing | each was corrected in the FOLLOWING commit; the table in §1 is the landed truth |

## 6. What this cannot say

* **Nothing about strength.** Arm W is frozen; the three cells are protocol gates at n = 10–20.
* **Nothing about Foul Play at the SOP's 1000 ms budget**, and nothing about a 100-game Foul Play
  cell — only that the half that could not run now runs.
* **Nothing about whether the upstream recursion still fires** after the serialization: our
  challenge loop and Metamon's are independent, and the H-H mechanism is inside Metamon's
  challenger role, which our change does not touch.
* **Nothing about `SyntheticRLV2`** — only `SmallRL` and Foul Play were played.

## 7. What is in this directory

`cells/<label>/summary.json` + `games.jsonl` for all three cells; `cells/fp10_serial/`
additionally carries the front end's full log and `challenge_timing.txt`, the timing evidence.

---

## 8. Ready-to-append ledger paragraph

**2026-09-22 — OPS: the Foul Play multi-game half is FIXED and the defect was OURS; the Metamon
`RecursionError` is the SAME defect on their side of the wire; five anchors/checkargs backlog rows
closed.** Hazard **H14** — `--opponent foulplay` could not run a multi-game `ours_challenge` half,
reproduced 3/3 on Node (09-16) and 1/10 on the front end (09-20) and therefore called
transport-independent — is transport-independent because **it is ours**: poke-env's
`Player._send_challenges` releases its battle semaphore when a battle *starts*, so it emitted
challenge *k+1* ~0.4 s INTO battle *k*, and Foul Play, which reads its PMs only between battles,
dropped it and waited forever for a challenge already consumed. `--challenge-mode serial` (the
DEFAULT) adds one `_battle_count_queue.join()` after the acquire, so the next `/challenge` waits
for the previous battle to END. 🚨 **Nothing about the play changes** — at `--concurrency 1` the
queue's maxsize already prevented battle *k+1* from starting early, so only the moment the PM is
emitted moves — and the patch goes on poke-env's **BASE** `Player` because
`Gen3Player._send_challenges` is a wrapper carrying the connect-or-raise deadline that an override
on the leaf would silently delete. **GATE: a 10-game Foul Play cell on `--server rust` completes
10/10, `status: OK`, both halves (`ours_challenge` 5/5), 0 ERRORs, realized 84,205 visits/decision,
the five challenges 11–37 s apart against H14's 0.4 s** — plus a 20-game Metamon control at 20/20,
`argmax_match_rate` 1.0000, 0.600 [0.387, 0.781] on top of the banked 0.590 [0.492, 0.681].
**Second finding: Metamon's post-game `RecursionError` (H-H) is the SAME defect class.** Metamon
serializes its ACCEPTOR role (`_accept_challenge_loop`, documented as fully awaiting each battle)
and drives its CHALLENGER role through poke-env's pipelined `start_challenging()`; the env's and
the agent's `current_battle` desynchronise, `openai_api.py` raises `Battle is already finished,
call reset`, and `MetamonAMAGOWrapper.step`'s unbounded `self.reset(); return self.step(action)`
turns it into ~988 frames — which is exactly why F-D found the common factor to be **who
challenges**, not sampling. 🚨 **The backlog row's own prescription is NOT taken and the reason is
the finding**: refusing on the regime axis refuses on the axis F-D retired, and refusing on the
role axis would refuse half of every standing read for a teardown that costs no games (31 of 84
sub-cells, all `argmax_match_rate` 1.0000, all re-derived as verified). So the post-game recursion
is **NAMED** (`peer_recursion_upstream`, only on a COMPLETE `peer_challenge` half) and only the
`mixed` cell is refused; the minimal upstream patch is documented here. **Also closed:**
`main.anchors` with no `--out` no longer writes `anchors_out/` into the calling directory (it is
run-scoped under `$GEN3AI_ANCHORS_OUT_ROOT` and PRINTED — an anchor read is taken from the MAIN
checkout, so the relative default filled the repo it was measuring); `checkargs --argv` strips a
leading interpreter/script token, which had turned every pasted `original_command` into a phantom
`unconsumed value '…/launcher/__main__.py' — a flag's ARITY differs` (now
`✅ validated against PINNED parser @e798c13a`); the pinned checkout carries
`designs/{baselines,production_config}.json`, without which the probe died on a file that exists
at the commit and fell through to the non-authoritative AST scan — and that demotion is now the
LOUDEST line, **checked on disk** rather than matched on the path prefix, because the prefix match
false-positived on the genuine "this commit predates `build_parser()`" reason. ⚠️ The Foul Play
cell is a PROTOCOL gate, not a strength read (n = 10, 100 ms, `team_source_asymmetry` true by
construction), and the 20-game challenger-role gate did NOT fire the upstream recursion — it
cannot be produced on demand (4 in 800 games), so the classifier is unit-tested against the
recorded signature rather than exercised live. Tag: **OPS · H14 CLOSED, THE DEFECT WAS OURS ·
FOUL PLAY 10/10 MULTI-GAME · METAMON H-H ROOT-CAUSED UPSTREAM AS THE SAME PIPELINED-CHALLENGE
DEFECT · NAMED NOT REFUSED (the row's prescription was keyed to a retired axis) · `--out` NEVER
THE CWD · `checkargs` PINNED PATH AUTHORITATIVE AGAIN**.
