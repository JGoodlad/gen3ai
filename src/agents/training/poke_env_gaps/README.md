# poke-env Gaps — Research & Fuzz Coverage

This directory documents known gaps between what poke-env tracks and what we need for
accurate `TurnDelta.opp_move_id` in Gen 3 OU. It also contains the e2e fuzz test that
confirmed these behaviors at scale.

---

## Background

`TurnDelta` (in `turn_delta.py`; the per-decision `BattleContext` snapshot it reads lives in
`battle_snapshot.py`) exposes `opp_move_id` and `opp_move_known` to the reward function and the
observation encoder. In production these fold from the event log (`build_from_events`); the
legacy snapshot-diff `TurnDelta.build` these gap tests exercise reads
`battle.opponent_active_pokemon.last_move`, which poke-env populates from the Showdown
`|-move|` protocol.

poke-env's `Pokemon.last_move` iterates `self.moves.values()` and returns the `Move` whose
`_is_last_used` flag is `True`. That flag is set inside `Pokemon.moved()`:

```python
def moved(self, move_id, failed=False, use=True, reveal=True, pressure=False):
    move = None
    if reveal:
        move = self._add_move(move_id)   # adds to self.moves, returns the Move object
    if use:
        for m in self.moves.values():
            m._is_last_used = m is move  # exactly one gets True; None → all get False
```

The `reveal=False` path is the core of all gaps below: when `move=None`, every flag goes
to `False` and `last_move` returns `None`.

---

## Confirmed Behaviors (from fuzz test + poke-env source)

### 1. Normal moves
`last_move` is always set correctly after `|-move|`. No gap.

### 2. `|cant|` turns (paralysis, freeze, flinch, confusion, sleep-no-sleeptalk)
`cant_move()` does **not** clear `_is_last_used`. So:
- If the mon has used a move before this `|cant|` turn → `last_move` persists from last
  actual use (e.g., Hyper Beam recharge shows `"hyperbeam"` again). Correct and useful.
- If `|cant|` fires on the mon's **first active turn** (no prior move in this appearance)
  → `last_move = None`. Expected; the mon literally has no move history yet.

TurnDelta handles this via `opp_move_known = False` when `opp_last_move_id is None`.

### 3. Explosion / Self-Destruct (the attacker-fainted gap)
When the opponent uses Explosion and faints:
1. `|-move|` sets `last_move = "explosion"` on the attacker.
2. `|faint|` removes the attacker from `opponent_active_pokemon`.
3. The new switch-in becomes `opponent_active_pokemon`, whose `last_move = None`.

By the time `_get_observation()` reads `opponent_active_pokemon.last_move`, the attacker
is gone. `opp_last_move_id = None`.

TurnDelta still captures the outcome via `opp_fainted = True` and `opp_prev_active` (the
species that exploded). The reward function uses this correctly.

**Fix (if ever needed):** Intercept `|faint|` in `AbstractBattle._parse_message()` and
snapshot `last_move` before the active slot changes. Requires forking poke-env.

### 4. Sleep Talk — success path
Showdown sends two `|move|` messages:
```
|move|p1a: Suicune|Sleep Talk|p1a: Suicune
|move|p1a: Suicune|Surf|p2a: Blissey|[from] Sleep Talk
```
poke-env converts `[from] Sleep Talk` → `[from] move: Sleep Talk` → `pass` (no change to
`use`/`reveal`). The second call `moved("surf", reveal=True, use=True)` runs normally.
Result: `last_move = "surf"`. **Works correctly.**

### 5. Sleep Talk — delegation failure
When Sleep Talk cannot delegate (e.g., all moves at 0 PP, picks a move that can't execute),
only the first `|move|Sleep Talk` message fires.
Result: `last_move = "sleeptalk"`.

This is **correct and useful** — we know the opponent tried Sleep Talk and failed. The
model sees `opp_move_id = "sleeptalk"` with `opp_move_known = True`. Observed ~5-10% of
Sleep Talk uses in fuzz testing (159 of ~1,000+ Sleep Talk turns in scenario B).

### 6. Delegating moves with `reveal=False` — THE GAP

The following moves in `abstract_battle.py` set `reveal=False` for the delegated move:

```python
elif overridden_move in {"Copycat", "Metronome", "Nature Power", "Round"}:
    reveal = False
```

Because `reveal=False` → `move=None` → all `_is_last_used` flags cleared → `last_move = None`.

| Move | Gen 3 competitive viability | Gap severity |
|------|------|------|
| **Metronome** | Extremely rare (Clefable) | Low — too chaotic to be meta |
| **Nature Power** | Niche (becomes Swift in standard terrain) | Low |
| **Copycat** | Gen 4+ — not in Gen 3 | N/A |
| **Assist** | Niche (Delcatty, Persian) — not in handler, falls to warning | Low |
| **Mirror Move** | Niche (Pidgeot, Swellow) — not in handler | Low |

After any of these fires: `opp_last_move_id = None`, `opp_move_known = False`. The model
sees uncertainty for that turn but continues correctly.

### 7. Snatch — the snatcher's STOLEN move (fixed) + the now-strict parser

Snatch (priority +4) steals the target's self-targeting status move and makes the *snatcher*
execute it. Showdown emits it as a move line the snatcher does not own:

```
|move|p2a: Blissey|Calm Mind|p2a: Blissey|[from] Snatch
```

`[from] Snatch` was in none of poke-env's stripped-tag lists, so the event fell through to the
generic move-format branch: it (a) logged a per-occurrence "Unmanaged move message format
received" warning (the visible spam) and (b) ran `moved(..., reveal=True)`, **adding the stolen
move to the snatcher's revealed moveset** — which then leaked into the opponent's obs move slots
(a Snatch Blissey read as "knows Calm Mind"). Fixed in `abstract_battle.py` by treating Snatch
exactly like **Magic Coat / Mirror Move** (`use=False, reveal=False`, tag stripped): the actor
reveals *Snatch* (logged on its own line), never the move it stole.

Alongside the fix, the move-message handler is now **strict**: every previously-`logger.warning`
"Unmanaged …" branch (unhandled `[from] move:` / `[from] ability:` override, and the two
move-format fall-throughs) now **`raise ValueError`** instead — matching the existing
`raise ValueError("Unhandled item message")` precedent in the same parser. The contract is
crash-don't-drop: an unrecognised move message stops the run loudly (a real traceback) rather
than silently spamming a warning and recording possibly-wrong state, so new gaps are caught and
fixed instead of accumulating. `snatch_fuzz_test.py` exercises the Snatch path and doubles as
the regression guard that the strict parser does not mis-fire on normal play.

---

## Fuzz Test Results Summary (50 battles × 3 scenarios, ~30K transitions)

Run: `python src/agents/training/poke_env_gaps/transition_fuzz_test.py 50`

| Metric | A-Explosion | B-Rest/SleepTalk | C-HyperBeam | Total |
|--------|------------|------------------|-------------|-------|
| Total transitions | 5,602 | 17,540 | 7,294 | 30,436 |
| `our_move_slot_unknown` | 0 | 0 | 0 | **0** |
| Explosion gap | 124 | 778 | 135 | 1,037 |
| Cant-move (estimated) | 457 | 1,086 | 684 | 2,227 |
| True anomalies¹ | 24 | 96 | 45 | **165 (0.5%)** |
| `last_move == "sleeptalk"` | 0 | 159 | 0 | 159 |
| Two-turn same move | 203 | 34 | 301 | 538 |

¹ "True anomaly" = `revealed_moves` grew (a move was used) but `last_move = None`. These
  are almost certainly false positives from poke-env's `_update_from_request()` adding
  moves to a Pokémon via the `|request|` metadata path (not via `|-move|`), so
  `last_move = None` is actually correct in those cases. Not a training concern.

**Key finding:** `our_move_slot_unknown = 0` across all 30K transitions confirms
`active_move_ids` resolution in `BattleContext` is correct for all observable move actions.

---

## Proposed Fix for the `reveal=False` Gap

If we ever want accurate `last_move` for Metronome / Nature Power / Assist / Mirror Move,
the fix is small and targeted:

**In `src/poke_env/battle/pokemon.py`, `moved()` method:**

```python
def moved(self, move_id, failed=False, use=True, reveal=True, pressure=False):
    move = None
    if reveal:
        move = self._add_move(move_id)
    if use:
        self._last_delegated_move_id = to_id_str(move_id)  # NEW: always track the ID
        for m in self.moves.values():
            m._is_last_used = m is move
```

And update `last_move` to prefer `_last_delegated_move_id` when `_is_last_used` scan
returns None. Or simpler: store the move ID directly and expose `last_move_id: str | None`
as a separate property that bypasses the `moves` dict lookup entirely.

This is a **5-line change** but it modifies a core poke-env class with broad reach. Worth
doing only if these moves become relevant in training.

---

## Running the Fuzz Test

Runs battles **in-process via the local BattleStream bridge — no `npm run showdown`** (only the `deps/pokemon-showdown` `dist/` + `node_modules` symlinks from the root CLAUDE.md worktree setup):

```bash
# One-time worktree setup (per root CLAUDE.md — symlink the BUILD ARTIFACTS, never the whole
# submodule dir, which breaks git status):
git submodule update --init
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist         deps/pokemon-showdown/dist
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules deps/pokemon-showdown/node_modules

# Run (30 battles per scenario ≈ 2 min)
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python src/agents/training/poke_env_gaps/transition_fuzz_test.py 30

# More thorough (50 battles ≈ 5 min)
python src/agents/training/poke_env_gaps/transition_fuzz_test.py 50
```

Scenarios:
- **A — Explosion**: Gengar/Claydol/Metagross with Explosion — stresses the attacker-fainted gap
- **B — Rest/Sleep Talk**: Suicune/Snorlax with Rest+Sleep Talk, Smeargle/Gengar with sleep inducers
- **C — Hyper Beam**: Tyranitar/Regice with Hyper Beam — confirms recharge-turn `last_move` persistence
- **D — Roar/Whirlwind**: phaze recovery via the DamagingMoveEvent

---

## Fuzz Coverage Map (what's validated, and what to expand)

Two e2e fuzz tests run real `gen3ou` battles and validate the protocol →
poke-env → `BattleContext` → `TurnDelta` → encoded-obs pipeline. The matching
unit tests are `src/agents/training/move_attribution_test.py` (decision table)
and the per-encoder `*_test.py` files (dim/layout).

### `transition_fuzz_test.py` — move ATTRIBUTION (who used what)
| Validated | Notes |
|---|---|
| `our_move_slot_unknown == 0` | every action index maps to a known move slot |
| **Intent ↔ outcome** (our side) | the move the agent *pressed* equals the move the protocol says *fired*, with principled skips for **callers** (Sleep Talk/Metronome → the *called* move), **`\|cant\|`**, switch-out / two-turn charge, and **stale `last_move`** (no fresh `\|move\|` this turn). 0 real mismatches over ~30K turns. |
| Opp `last_move` attribution | Explosion gap, recharge persistence, phaze recovery, cant persistence — all classified as expected vs anomaly |

### `move_outcome_fuzz_test.py` — move OUTCOME (what happened)
Four layers per turn (raw protocol → poke-env props → `BattleContext` → `TurnDelta` → encoded vector). FAILS on any mismatch **or** missing coverage. Validates, for **both sides**:
| Validated | Notes |
|---|---|
| crit / miss / fail / hit / cant | outcome one-hot + crit bit, BattleContext vs TurnDelta vs encoding |
| **Explosion / Self-Destruct self-faint** | the move *landed* before the self-KO → `move_id=explosion`, outcome = **hit**. A stale miss/fail flag is overridden: an SE/resisted/immune hit promotes a damaging event, and a *neutral* hit (no event) is covered by treating `SELF_KO_MOVES` as always-connected. (Edge: a Protect-blocked Explosion reads `hit` rather than `fail` — extremely rare, documented below.) |
| **Switch-in death (e.g. Spikes)** | it's a *switch*, not a move → `switch_to` set, `move_id=None`, outcome `None` (a stale miss/fail flag does **not** leak) |
| **KO'd before acting** | nothing fired → `move_id=None`, outcome `None`, `cant_reason="fainted"` (distinct from a voluntary switch and from `\|cant\|`) |
| Edge cases run for opp too | `opp_explosion_self_faint`, `opp_faint_before_acting` |

Run it: `python src/agents/training/poke_env_gaps/move_outcome_fuzz_test.py 40`
(two teams: a variance team for crit/miss/fail/cant, and an Explosion+Spikes+frail
"FaintEdge" team so the faint edge cases actually fire under random play).

### `snatch_fuzz_test.py` — Snatch parsing (stolen-move attribution + strict parser)
Forcing players (a Snatch-user team vs a snatchable-setup team) make `[from] Snatch` lines fire
reliably every setup turn while both sides attack on the others so battles terminate. Validates,
against the raw protocol:
| Validated | Notes |
|---|---|
| **No parse crash** | the bridge propagates a `parse_message` raise, so an unhandled `[from] Snatch` — or any newly-unhandled move message under the now-strict parser — crashes the run. A clean run proves both that Snatch is handled and that the strict change does not mis-fire. |
| **Correct attribution** | for every observed snatch, the stolen move is NOT in the snatcher's revealed moveset (checked from the opponent's reveal-gated view — the one the obs encoder reads). The old `reveal=True` bug fails this; the test is sensitivity-checked against that regression. |
| **Coverage** | FAILS if no Snatch was ever observed (≈ 12 snatches/battle in practice). |

Run it: `python src/agents/training/poke_env_gaps/snatch_fuzz_test.py 20`

### Known gaps / to expand
- **Effectiveness on delegated damaging moves**: `our_last_damaging_event` can lag
  at a mon's *prior* damaging move when Sleep Talk calls a *different* damaging
  move (the called move's effectiveness isn't promoted). We currently drop to the
  "unknown" effectiveness sentinel via the alignment guard; a dedicated assertion
  that the encoded effectiveness matches the *called* move's protocol effectiveness
  would tighten this.
- **Protect-blocked Self-Destruct/Explosion**: `SELF_KO_MOVES` are treated as
  always-connected to recover the common neutral-hit case, so an Explosion that
  is fully blocked by Protect (no damage, no event) reads `outcome=hit` instead
  of `fail`. The `move_id` is still correct (not the critical "nothing happened"
  misrepresentation); the outcome bit is a rare secondary inaccuracy.
- **Non-damaging move then self-faint**: a status move used by a mon that then
  faints isn't recoverable from the damaging event (active slot has shifted) — it
  currently reads as `move_id=None`. Rare; not yet asserted.
- **Multi-faint / double-KO turns** (e.g. our Explosion KOs both): only spot-checked.
- **Baton Pass** chains: classified as a switch; the pass target attribution is not
  separately validated.
- The fuzz uses **random** action selection — it exercises legality and protocol
  attribution, not strategic sequences. Coverage of rare interactions is
  probabilistic; raise the battle count to harden.
