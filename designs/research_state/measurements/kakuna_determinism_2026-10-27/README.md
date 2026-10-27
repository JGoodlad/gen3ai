# Kakuna replay determinism (2026-10-27)

**Question.** N1-P look 1's operator saw C0's re-played forum72 units match D1's banked games on 414 / 420 (6 games
differ, net +4 wins), and an uncounted p72 unit replay match 18 / 20. Is Kakuna play nondeterministic, and what does
that do to the identity units and determinism guards that assume exact replay?

**Answer.** Kakuna is not the cause. On a counted unit, with the engine held fixed, replay is exact: 0 games differ in
2,100 compared games. The 414 / 420 compared two ENGINES: D1 played on E1 and N1-P look 1 on E3
(`designs/tools/reads.md` §4). Between them, `d8d674b99` (`gen3_volatile_silent_ends_v1`) changed what OUR side
observes after a Destiny Bond or an Endure. Every one of the cross-engine divergences is our own policy choosing
differently from a byte-identical protocol. Kakuna's choices never differ first. The p72 replay's two games are
Metamon's known turn-250 fault (`EXTERNAL_ANCHORS_SOP.md` H5), which already drops the unit.

All read-only, CPU, on the archived captures (`--capture-dir`: per battle the seed, every `CHOOSE` / `FORCELOSE`
and every per-side protocol chunk). Scripts: `scripts/` (below).

## 1. What a game's identity is, and how a divergence is classified

The guards compare a game's 5-tuple `(half, index, our_team, result, turns)` (`main.reads.assemble`). The captures allow
a stricter check. For each pair of plays of one game, `scripts/survey.py`:
1. finds the first per-side protocol chunk that differs;
2. reads which side's ACTION in that chunk differs (the first `|move|` / `|switch|` / `|cant|` of each side);
3. checks whether a silently ending volatile was live in the turns before it. These are the cells
   `gen3_volatile_silent_ends_v1` changed: Destiny Bond (`-singlemove`), Endure (`-singleturn`), and Grudge, Rage,
   Snatch, Helping Hand.

It also separates two cases that change no outcome:
- **order-only:** the same choices, sent in another order (simultaneous choices race to the server). The protocol is
  identical.
- **choice-unexecuted:** our choice differed, but it never ran, for example because we were KO'd first.

The brief's classes:
- (a) the peer model's nondeterminism;
- (b) timing;
- (c) our side;
- (d) the peer's state carried across games.

## 2. The 414 / 420: an ENGINE change on our side (class c, deterministic), not Kakuna

| C0 units, D1 (E1) vs N1-P look 1 (E3) | games | 5-tuple differs | our choice differs (capture) | Kakuna's choice differs first | games with a silently ending volatile |
|---|---|---|---|---|---|
| forum72 (seeds 0-6 × 3 finals) | 420 | **6** (net +4 wins on E3) | 7 | **0** | 48 |
| p72 | 420 | 4 (net −1) | 7 | 0 | 45 |
| heldout72_hpfix | 420 | 1 (0) | 1 | 0 | 26 |
| **all** | **1,260** | **11 (0.87 %)** | **15 (1.19 %)** | **0** | 119 |

Model bytes are identical (`09a6a2df…` for c0_s3501). Seeds, the START (both packed teams) and the taskset are equal.
`OMP_NUM_THREADS=1` is set on both sides at both pins (`main.x35.kakuna.unit_env`).

Every divergence is OUR side choosing differently from an identical protocol and request. Because the policy is greedy
and single-threaded on a fixed input, a different choice means a different input: the obs row the E3 core serves.

**11 of the 12 that change the protocol follow a Destiny Bond or an Endure in the same stint**, and so does the
forum72 choice that never ran (c0_s3503 u02 g9). Examples:
- forum72 c0_s3501 u04 g8: E1 re-uses Destiny Bond, E3 plays Ice Punch, and the loss becomes a win.
- c0_s3503 u02 g19: after our Scizor's Endure, E1 switches to Magneton and E3 Endures again.

At E1 those cells stayed 1 for the whole stint (GIGO audit A-C1, `gigo_audit_obs_2026-10-26/`: the obs golden moved
on 36 / 991 decisions, all Destiny Bond). One p72 game (c0_s3502 u01 g9, Thunderbolt vs Thunder Wave, no such
volatile) is not attributed. **FINDING:** another play-path change between `6c428453a` and `de15dfc7b` is the likely
cause. It is not identified.

The divergence rate is a property of the ENGINE PAIR and the PANEL: it is set by how often the changed cell is live.
It is not a property of Kakuna.

## 3. Every replayed Kakuna unit in the archive, by engine pair

`scripts/inventory.py` groups all 1,094 complete Kakuna cells under `~/gen3ai_archive/` by what fixes their play:
- the model's sha256, the opponent and both regimes;
- both team digests;
- `--team-seed`, `--seed-base`, `--games`, the load and `--deal-game`.

`scripts/pairs.py` compares the 5-tuples of every pair. It uses whole 20-game units only, leaves out S-U's searched
cells, and takes engines from `main.reads.engine`. `scripts/x51cmp.py` compares later P\* re-plays with X51's
banked per-game rows (`x51_plateau_vs_kakuna_2026-10-20/rows.jsonl`, E0).

**Across `d8d674b99`** (one side on E0 / E0B / E1, the other on E2 / E3 / main-10-26 / the S-U engine):

| pair | panel | games | differ |
|---|---|---|---|
| E1 vs E3 (above) | forum72 / p72 / heldout72 | 1,260 | 11 |
| E1 vs E2 (D3 look 1's guard + its pre-run) | heldout72_hpfix | 280 | 0 |
| E1 vs S-U engine (`bdc5d2be/94`; H-CTX am2) | forum72 / home72 | 280 / 100 | 1 / 0 |
| E1 vs main-10-26 (`67a272e7/93`) | forum72 / heldout72 / home72 | 140 | 0 |
| E0 (X51) vs S-U engine (H-CTX am2) | forum72 / home72 | 480 | 1 (the same game as against E1) |
| E0 (X51) vs main-10-26 (H-CTX am1 + its guard) | forum72 / heldout72 / home72 | 80 | 0 |

The S-U-engine divergence (P\* forum72 u02 g19) is also ours, after an Endure, and so are its 3 capture-only
siblings.

**Not across it** (both sides on the same side of the change):

| pair | games | differ |
|---|---|---|
| E0 (X51) vs E1: x26c's P\* (1,200) and the belief-prior-mix guards (2 × 80) | 1,360 | **0** |
| E2 vs E3 | 280 | 0 |
| main-10-26 vs S-U engine | 120 | 0 |
| same engine (E1, E2, main-10-26, S-U engine) | 340 | 0 |
| **all counted units** | **2,100** | **0** (95 % upper bound 0.14 %) |

## 4. The p72 unit's 18 / 20: H5, a timing race that poisons the next two games (classes b then d)

`n1p_s3503__p72__03` played twice on E3 at the same pin. The first play's peer faulted; the re-run was counted.

**Game 15** reached turn 250 in BOTH plays, and our side forfeited (`FORCELOSE p2`) identically. Whether Metamon's
long-tail handler fires on that forfeit is a race (H5). Here it fired in one play and not the other. Across the
archive it fired on **3 of the 105** turn-250 forfeits in 1,117 Kakuna units (`scripts/h5.py`).

When it fires, the damage is fixed, and it falls on the next two games of the half:
- **Game n+1:** Kakuna forfeits at its first decision (`FORCELOSE p1`), a 1-turn "win" for us.
- **Game n+2:** Kakuna plays on carried state. Its first differing choice comes from an identical protocol (p72 g17,
  turn 22: Counter vs switch to Heracross).

**Deterministic once triggered.** P\* forum72 u00 faulted at game 16 in TWO reads on two engines (H-CTX am1, belief
prior mix l025). The two faulted plays are byte-identical on 20 / 20. Each differs from its clean re-run on exactly
games 17 (a forfeit) and 18 (Pursuit vs Hidden Power, a loss becomes a win).

All three faulted units were `peer_clean = False` and re-run with the same seed, so **no counted unit holds an H5 game.**
S-U's forfeit-win guard (`main.anchors.runner.forced_end_verdict`) would also fail game n+1 by name. Game n+2 is
caught only because the whole unit is dropped.

**FINDING:** a unit with a mid-unit peer fault must never be partly counted.

Separately, about 6 % of games are **order-only** (26 / 420 on forum72). This timing difference has no effect on the
protocol.

## 5. Reproduction

`scripts/replay.py` re-plays one N1-P look-1 unit from that read's own E3 pin worktree, on a claimed 95xx port, under
SCHED_IDLE. Its gate enqueues only when no other heavy job is queued. The unit was c0_s3501 forum72 u04, three
times.

**Not run (FINDING).** The heavy-slot queue held 4 to 5 read units (N1-P, H-CTX, X59-S) for the whole window, so
the gate never opened. Entering the FIFO would have put the replays ahead of those reads' later units, which the brief
ruled out. Each replay was withdrawn before it ran (its queued waiter was killed by PID), and no game was played.

**What stands in for it, from the archive:**
- **A replay at scale.** 2,100 counted games were played twice with no observation change between the two plays,
  including X59-S's own `repeat` unit. All 2,100 are identical.
- **Where the divergences fall.** The cross-engine divergences cluster where the changed cell is live. 119 of the
  1,260 E1-vs-E3 games contain a Destiny Bond or an Endure anywhere. If the 12 protocol-changing divergences fell on
  random games, P(≥ 11 of 12 in such games) = 6 × 10⁻¹¹. The test counts a divergence only when the volatile came in
  the few turns before it, so this null is conservative.

To run it when the box is quiet: `scripts/replay.py <units.jsonl> c0_s3501__forum72__04 <out>`. Expect N1-P's 20
games byte-identical: game 8 is Ice Punch and a win at 105 turns.

## 6. Classes, counts, fixes

| class | where | games | fix |
|---|---|---|---|
| **(c) our side: the ENGINE changed what we observe** (deterministic per engine) | every cross-engine divergence | E1 vs E3: 11 / 1,260 by 5-tuple, 15 by choice; S-U vs E1 / E0: 1 | **None in code.** The rule already in code holds: a determinism guard, an identity unit and a banked import never cross an engine (`main.reads` `_same_engine`; `reads.md` §4). A cross-engine replay is not a determinism measurement. |
| (b) → (d) timing: H5's race, then Kakuna's carried state | 3 faulted units in the archive | 2 games each (the forfeit, then the carried game) | Caught today: `peer_clean` drops the whole unit and it re-runs with the seed. Proposed (harness-side, needs the owner's go): fail the half at the FIRST "Force resetting due to long-tail error" line, so a faulted unit stops at once instead of playing its contaminated games. No counted game changes, so there is no regime change. |
| (b) order-only | about 6 % of games | 0 outcomes | none |
| (a) Kakuna model nondeterminism | — | **0** of 2,100 counted same-side games; 0 first-divergences in 1,500 cross-engine games | none: threads are pinned to 1 and `argmax_match_rate` is 1.0 |

**Proposed, not shipped:** where both plays carry captures, the determinism guard could compare choices instead of the
5-tuple. It would see 15 divergences where the 5-tuple sees 11 (E1 vs E3). The `order-only` case must stay equal.

## 7. What it means for past guards and identity claims

- **The E0 → E1 guards were exact, not lucky.** These are the belief-prior-mix guard's 80 / 80 per arm, the human-only
  and PastaMittens identity unit's 20 / 20, and x26c's 60 units. Measured game for game here: x26c's 1,200 and the
  guards' 160 equal X51's on all 1,360. E0 → E1 crosses no change of this kind.
- **D3 look 1's 140 / 140 (E2 vs E1) crossed `d8d674b99`.** It passed on a low-exposure panel: on heldout72_hpfix the
  E1-vs-E3 rate is 1 / 420, so P(140 / 140) ≈ 0.72; at the pooled cross-change rate of the archive pairs (12 / 2,060) it is ≈ 0.44. Its
  recorded bound of "< 2.1 % (95 %)" holds: the measured rate is 0.2-1.4 % by panel. On forum72 the shift was
  +4 wins in 420 (+0.95 pp) for the newer engine, the same order as D3 look 1's Δ̂ = +1.27 pp. This is why look 2
  plays every game on E1 (the orchestrator's 2026-10-26 ruling).
- **What a guard can prove.** On one engine it is exact, with 0 of 2,100 measured. Across an engine change it proves
  only "no divergence on this panel at this exposure": a passing guard bounds the rate, and a fail-on-any guard fails
  by chance once its sample is large. That is why it must never cross an engine.
- **FINDING: the engine key covers the Rust build only.** A Python-only change to our play path (the model or op code)
  between two commits on one engine changes no engine stamp. Only `repin`'s play-path identity (§3) or a
  game-for-game guard catches it.

## Files

- `scripts/survey.py`: the capture classifier (`python3 survey.py spec1.json`).
- `scripts/inventory.py`, `scripts/pairs.py`: the archive survey by engine pair.
- `scripts/x51cmp.py`: the comparison with X51's rows.
- `scripts/h5.py`: the turn-250 fault tally.
- `scripts/ctx.py`: prints the turns before a divergence.
- `scripts/replay.py`: the replay driver.
- `scripts/spec*.json`: the cell pairs compared.
- `results_*.txt`: their outputs at the time of writing.
