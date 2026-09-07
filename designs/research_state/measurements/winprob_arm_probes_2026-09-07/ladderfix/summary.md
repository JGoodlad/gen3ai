# Ladder fit: drop the eval SENTINEL edges; the gate refits both sides

Worktree `/home/goodlad/dev/gen3ai-wt-ladderfix`, branch `ladder-fit-eval-sentinel`,
commit `9476bb09` (**not pushed**; main checkout untouched and clean).

## 1. What changed

| file:line | change |
|---|---|
| `src/agents/training/snapshot_ladder.py:135-138` | `fit_ladder` docstring states the exclusion + the `eval_sentinel_edges_dropped` key |
| `src/agents/training/snapshot_ladder.py:178-201` | source (2) keeps the **bot** edges and drops `snap:`-vs-`snap:` rows (`elo_mod.is_snapshot(na) and elo_mod.is_snapshot(nb)` at `:196`), counting them; ~20 lines of comment carry the why, the **+8.9 pp [+7.0, +10.7]** measurement and the date 2026-09-07 |
| `src/agents/training/snapshot_ladder.py:230-233` | `eval_sentinel_edges_dropped` in the returned/written ladder dict |
| `src/main/critic_gate.py:284-329` | `ladder_section` now refits **both** sides from their own `games.jsonl` with this tree's `fit_ladder(first_n=n)` whenever that log exists (new local `_side` at `:308`), not only the longer side; a side that cannot be refit falls back to its committed `ladder.json` and lands in `fallbacks` |
| `src/main/critic_gate.py:352` | new `refit_fallbacks` JSON key |
| `src/main/critic_gate.py:359-372` | `fit_size_note` rewritten: names both refits + the sentinel reason, and appends `⚠️ FELL BACK to the COMMITTED ladder.json for: …` |
| `src/agents/training/snapshot_ladder_test.py:+4 tests` | see §2 |
| `src/main/critic_gate_test.py:+2 tests, 1 renamed` | see §2 |
| `src/agents/training/CLAUDE.md` | ladder section: the exclusion + the measurement + the per-run refit shift; "Reading an ELO" gains **rule 4** (a committed `ladder.json` is not evidence); the `critic_gate` table's `1 ladder` row rewritten |

**UNMATCHED behaviour preserved.** `unmatched` (and therefore the famine refusal) still fires only
when a side has MORE rated nodes than the matched count AND cannot be refit. A side whose count
already matches but has no pair log is a *fallback* — the SIZE is matched, only the SOURCE is the
committed file — and is labelled, not refused.

## 2. Tests added

`snapshot_ladder_test.py` — planted `eval_results.jsonl` (one bot edge `random` 0.9, one sentinel
edge 200-vs-100 at 0.95) beside a dense `games.jsonl` edge for that same pair at a dead-even
50/100:
- `..._keeps_bot_edges_and_drops_eval_sentinel_edges` — spies on the list handed to `fit_pairwise`:
  exactly **one** snap-vs-snap pair reaches the fit and it is the dense `(50, 100)` one; the bot
  edge `(90, 100)` does enter; `eval_sentinel_edges_dropped == 1`; the two nodes end level (<5 Elo).
- `..._sentinel_drop_changes_the_rating` — the same inputs refit with the sentinel edge re-admitted
  open a **>100 Elo** gap, so the filter is not a no-op.
- `..._first_n_and_steps_kwargs_still_work` — `first_n` restricts + never writes, `steps` slices,
  `write=True` default still writes, ordering preserved.
- `..._the_per_cycle_eval_elo_star_fit_still_counts_sentinel_edges` — **`elo.fit_from_run` is
  untouched**: `_rows_to_results` still yields the snap-vs-snap pair and the star fit still puts
  200 more than 100 Elo above 100, which is only reachable through that edge (identical bot rates).

`critic_gate_test.py`:
- `..._a_committed_ladder_is_NOT_used_when_its_own_games_log_can_be_refit` — both sides rate exactly
  2 nodes (so the old `len(nodes) > n` guard refit neither); the arm's committed 2nd node is planted
  at 2500 against a pair log implying 55/100. Asserts `refit_at_count == {run: True, parent: True}`,
  empty `refit_fallbacks`, both nodes equal to a fresh `fit_ladder(first_n=2)`, and `|delta| < 20`
  where the committed delta would have read **+650**.
- `..._a_side_with_no_pair_log_at_matched_count_is_LABELLED_as_a_fallback`.
- `test_equal_counts_need_no_refit_and_say_so` renamed to
  `test_equal_counts_with_no_pair_log_fall_back_to_the_committed_ladders_and_say_so` (its old
  "no refit" wording no longer exists).

## 3. Test + gate output

```
$ pytest src/agents/training/snapshot_ladder_test.py src/agents/training/elo_test.py \
         src/main/critic_gate_test.py -q -p no:cacheprovider
79 passed in 9.14s
```
Re-run after the doc edits, adding the CLAUDE.md-freshness and file-size gates: `99 passed in 6.87s`.

```
$ ruff check src/agents src/main src/utils --select F,E9 --exclude src/poke_env --exclude src/rust_sim
All checks passed!
$ python -m mypy
Success: no issues found in 87 source files
```

## 4. Per-node shift, committed `ladder.json` -> fixed `fit_ladder(write=False, steps=<rated>)`

Read-only on `models/`; nothing written there.

**`ai_v12_02_winprob_critic`** — 65 sentinel edges dropped, 120 frozen pairs, converged

| step | committed | fixed | shift | | step | committed | fixed | shift |
|---|---|---|---|---|---|---|---|---|
| 4,000,032 | 1697.1 | 1729.2 | **+32.1** | | 20,000,016 | 2000.4 | 1985.7 | −14.7 |
| 6,000,000 | 1845.4 | 1860.9 | +15.5 | | 22,000,032 | 2010.6 | 1995.8 | −14.8 |
| 8,000,016 | 1896.0 | 1914.3 | +18.3 | | 24,000,000 | 2014.3 | 2004.3 | −10.0 |
| 10,000,032 | 1930.2 | 1934.6 | +4.4 | | 26,000,016 | 2043.3 | 2023.3 | −20.0 |
| 12,000,000 | 1966.3 | 1962.8 | −3.5 | | 28,000,032 | 2056.6 | 2035.0 | −21.6 |
| 14,000,016 | 1986.1 | 1983.4 | −2.7 | | 30,000,000 | 2048.0 | 2026.9 | −21.1 |
| 16,000,032 | 2006.4 | 1998.9 | −7.5 | | 32,000,016 | 2051.6 | 2029.7 | −21.9 |
| 18,000,000 | 2027.8 | 2028.1 | +0.3 | | 34,000,032 | 2053.3 | 2019.7 | **−33.6** |

**`ai_v9_29_rev1_0823`** — 45 dropped, 66 frozen pairs, converged

| step | committed | fixed | shift | | step | committed | fixed | shift |
|---|---|---|---|---|---|---|---|---|
| 2,000,016 | 1692.5 | 1723.0 | **+30.5** | | 14,000,016 | 2044.9 | 2026.1 | −18.8 |
| 4,000,032 | 1879.0 | 1896.8 | +17.8 | | 16,000,032 | 2066.9 | 2034.9 | −32.0 |
| 6,000,000 | 1917.3 | 1948.9 | +31.6 | | 18,000,000 | 2057.8 | 2032.3 | −25.5 |
| 8,000,016 | 1957.9 | 1961.5 | +3.6 | | 20,000,016 | 2068.9 | 2026.1 | −42.8 |
| 10,000,032 | 2016.4 | 2018.0 | +1.6 | | 22,000,032 | 2083.2 | 2055.4 | −27.8 |
| 12,000,000 | 2032.6 | 2014.4 | −18.2 | | 24,000,000 | 2098.4 | 2053.3 | **−45.1** |

**`ai_v9_59_R2ACTION_0827`** — only 10 dropped, 21 frozen pairs, converged (a FORK: its eval rows
exist mostly at the late nodes, so the sentinel contamination is sparse and the shift is not
monotone)

| step | committed | fixed | shift | | step | committed | fixed | shift |
|---|---|---|---|---|---|---|---|---|
| 2,000,016 | 1553.5 | 1587.9 | +34.4 | | 16,000,032 | 1937.8 | 1937.4 | −0.4 |
| 4,000,032 | 1837.2 | 1836.8 | −0.4 | | 18,000,000 | 1926.7 | 1961.6 | +34.9 |
| 6,000,000 | 1817.7 | 1817.4 | −0.3 | | 20,000,016 | 1902.9 | 1934.4 | +31.5 |
| 8,000,016 | 1863.3 | 1888.3 | +25.0 | | 22,000,032 | 1925.3 | 1934.4 | +9.1 |
| 10,000,032 | 1913.4 | 1913.0 | −0.4 | | 24,000,000 | 1956.5 | 1982.8 | +26.3 |
| 12,000,000 | 1889.2 | 1916.5 | +27.3 | | 26,000,016 | 1939.2 | 1948.4 | +9.2 |
| 14,000,016 | 1939.7 | 1965.1 | +25.4 | | 28,000,032 | 1957.4 | 1947.3 | −10.1 |

**The pattern on the two dense runs is a compression, not a uniform drop**: the early nodes RISE
(+30 to +32) and the late nodes FALL (−20 to −45). The registered figure "+21..+29 on the newest
nodes" is the arm's late block; the deepest node of each run moves −33.6 and −45.1.

## 5. Famine comparison — registered vs fixed

Arm `ai_v12_02_winprob_critic` vs `ai_v9_29_rev1_0823` (registry `famine_comparator`), both refit
on a **strict prefix** (`fit_ladder(first_n=n, steps=<rated>)`), node = the n-th. "PRE-FIX" is the
identical code path with the sentinel edges re-admitted, and it **reproduces the registered numbers
exactly** (−30, −34), which is the control that the two arms differ only in the edge set.

| n | arm node | comparator node | delta (arm − comp) | combined SE | **trail** (positive = behind) |
|---|---|---|---|---|---|
| **4** PRE-FIX | 10,000,032 = 2022.1 (se 14.0) | 8,000,016 = 2052.0 (se 15.2) | −29.9 | 20.7 | **+29.9** (registered −30) |
| **4** FIXED | 10,000,032 = 1989.0 (se 15.0) | 8,000,016 = 2002.0 (se 16.2) | −13.0 | 22.1 | **+13.0** |
| **12** PRE-FIX | 26,000,016 = 2064.1 (se 10.2) | 24,000,000 = 2098.4 (se 10.6) | −34.3 | 14.7 | **+34.3** (registered −34) |
| **12** FIXED | 26,000,016 = 2032.4 (se 11.0) | 24,000,000 = 2053.3 (se 11.3) | −20.9 | 15.8 | **+20.9** |

6 sentinel edges were dropped per side at n=4; 45 per side at n=12.
**The verdict is unchanged** — every trail is inside the registry floor of 38 — but the trail
shrinks by ~17 Elo at n=4 and ~13 at n=12, and both fixed trails now sit inside their own 95% CI
of zero.

## 6. `python -m main.critic_gate` with the fixed fit

Command as briefed (`--parent v9_fold_parent --famine-comparator famine_comparator --control
<the three G5 plain arms> --skip-meter`); exit **1**, which is the pre-existing **G7 KILL**
(episode length), not a failure of this change. Ladder section reports
`refit_at_count = {run: True, parent: True}`, `refit_fallbacks = []`, `matched_fit_size = True`.

Famine sentence, verbatim:

> the arm TRAILS the comparator by 21 ELO at 12 snapshots (floor 38) — inside the floor: starvation is NOT demonstrated on the ladder half [floor from registry `famine_comparator`.floor_elo; signed trail +21, positive = behind]

Ladder `fit_size_note`, verbatim:

> BOTH sides were REFIT from their own games.jsonl on their first n snapshots (strict: every snapshot endpoint inside the prefix) with THIS tree's fit_ladder — because the newest node of a longer fit is inflated, AND because a committed ladder.json written before 2026-09-07 folded the eval cycles' greedy-vs-stochastic SENTINEL edges into the fit (+8.9 pp to the newer snapshot; +21..+29 Elo on the newest nodes) — refit: run=True, parent=True; the committed-ladder delta would have read +91.

Headline ladder read (arm vs `v9_fold_parent` = `ai_v9_59_R2ACTION_0827`), now **Δ at 14 snapshots
= +87 ELO [+50, +123]**, against the committed-ladder value of **+91**.
Overall verdict: `KILL` — `G7 breached: ep_len_bots 23.79 is 1.30x the era's 18.28 (> 1.25);
ep_len_bots 23.49 is 1.28x the era's 18.28 (> 1.25)`.

Artifacts: `gate_fixed.json`, `gate_fixed.md`, `gate_fixed.txt`, `shifts.json` in this directory.

## 7. Not done / caveats

- Nothing was blocked; every briefed step completed. No errors to report.
- The full routine suite was **not** run (a live training run shares the box) — the three briefed
  test files plus the CLAUDE.md-freshness and file-size gates were, along with both static gates.
- **No `ladder.json` on disk was rewritten** — every refit used `write=False` / `first_n`. The
  committed ladders of all three runs still carry the pre-fix numbers, and the live arm, pinned to
  an older commit, will keep writing them; that is exactly why the gate now refits rather than
  reads. Re-fitting them for real is a separate decision (it would need `--fit-only` per run and
  writes under `models/`).
- `first_n` fits still exclude any sentinel edge whose endpoint falls outside the prefix, so
  `eval_sentinel_edges_dropped` on a prefix fit counts only the edges that survived the prefix
  filter (asserted in the `first_n` test).
