---
name: project_bridge_unique_battle_tags
description: "Bridge must give every battle a process-unique tag; per-call tags reused across run_local_battles calls caused the \"7th pokemon\" crash"
metadata: 
  node_type: memory
  type: project
  originSessionId: 4ec9eedc-4256-4a75-ad16-aaa453eee32b
---

The "intermittent poke-env crash" `ValueError: <side>'s team already has 6 pokemons: cannot add <side>: <Species> to ...` (via `Battle.switch → get_pokemon`) hit by the bridge fuzz at scale was **NOT** a poke-env gen3 tracking bug (not Ditto/Transform, not nicknames — the pool has zero nickname/forme collisions). Root cause: `src/utils/bridge/local_battle_runner.py` tagged battles `battle-{fmt}-{index+1}`, resetting to 1..N **every** `run_local_battles` call. The same `Player` objects persist across calls (their `_battles` dict is never cleared), and poke-env `_create_battle` returns the **existing** battle for a seen tag (`player.py`: `if battle_tag in self._battles: return self._battles[battle_tag]`). A **chunked / time-budget fuzz loop** (e.g. `event_log_fuzz_test.py <N>m` time-budget mode loops `run_local_battles(...,25)`) reuses `battle-{fmt}-1` each chunk → the new battle is parsed into the *previous* battle's full-team object → first differing `|switch|` overflows to a 7th → crash. The "phantom 7th" is just a mon from a different battle's matchup.

**Why it looked like a poke-env bug:** count-mode (single `run_local_battles(N)`, unique tags 1..N) never reproduces it — `event_log_fuzz_test.py 1500` passed 108k decisions clean. Only the chunked/repeated-call path collides tags.

**Fix:** a process-global `itertools.count` (`_BATTLE_SEQ`) so every tag is unique across all calls, like a real server's room ids. Single-call behaviour is unchanged. Do NOT instead pop finished battles from `_battles` — `local_sim_bridge_integration_test` reads `n_finished_battles`/`_battles` after a run. Regression guard: `test_repeated_calls_use_unique_tags` in `local_sim_bridge_integration_test.py`. See [[project_local_sim_bridge]].
