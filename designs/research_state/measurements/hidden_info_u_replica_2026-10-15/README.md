# Hidden-info study: the blind U-arm replication (red team E1, 2026-10-15)

## In plain language

The hidden-information study asks how much better a search would decide if it knew the opponent's hidden facts. Its
yardstick is **IMP**: how much the search's pick beats the policy's own pick, in win probability per decision,
judged by a referee that played every action out 24 times in the TRUE battle. The search plays its rollouts either
in the true battle (**O**, the oracle) or in **fair worlds** (**F**): copies of the battle in which every fact the
searched side has not seen is re-drawn from the Smogon usage prior, kept only if replaying the game so far shows the
searched side exactly what it really saw.

This folder adds the two arms that isolate **U**, the opponent's NEVER-SEEN team members (a mon the searched side
has not seen switch or be dragged in before the decision: its species, moves, item, ability, nature, EVs, IVs,
gender and slot):

- **F+U**: each fair world with the never-seen members replaced by the TRUE ones (revealing U alone).
- **O-U**: the TRUE battle with the never-seen members re-drawn from the prior (hiding U alone).

It was built by EXTENDING the independent replica's builder (`../hidden_info_replica_2026-10-11/replica/`),
blind to the first implementation's code and to every prior result. A third party compares it with the first
implementation, under the pre-registration in [`PREREGISTRATION.md`](PREREGISTRATION.md) (committed `8dd25833`
before any arm output existed). **This folder does not compare.**

## Status

DONE: the read ran on the complete 1,200 states (2026-10-16). **Results: filled below; one identity check reads FALSE on prod (2 states, a guard false positive on Hidden Power naming), see "Identities" in Results.**

**COMPARED by a third party (2026-10-16):** [`third_party_comparison.md`](third_party_comparison.md) / `.json`
(`third_party_compare.py`). The pre-registered criterion AGREES at LEAF-K8 on the 978 common states, and the identity issue is
ruled a guard-caused exclusion. This folder's own Results below are unchanged.

## What was built (`hiu/`)

| file | what it does |
|---|---|
| `fix.py` | installs two changes into the replica's modules at run time (the replica's files are unchanged): the CURRENT learnset table, and the never-seen draw placed uniformly |
| `arms.py` | the world sets: F / F+M (the replica's `fair.build`), F+U (F's worlds with the true never-seen entries), O-M and O-U (the true team with M or U re-drawn) |
| `guard.py` | the consistency guard: every world's contract (which slots are true, which drawn) and the engine's own view of the team, slot by slot; a mismatch RAISES |
| `run.py` | the resumable runner: per state, all six arms, one JSON row per arm, a done-marker per state |
| `score.py` | this build's own read: IMP per arm and scorer, s_U, n_U, s_M, n_M, R3's numerators with intervals, the identities, the exclusions, the per-state IMPs, the marginals |
| `run_worker.sh`, `snapshot.sh` | the detached worker (rule 5b) and the code snapshot it runs from |

### The learnset fix (every arm)

The replica's move prior and legality check read the learnset table at the checkpoints' PINS, which predates
`64a695dc` (2026-10-14) and lacks 1,500 legal (species, move) pairs: every move inherited from a pre-evolution
(Tyranitar Dragon Dance, Breloom Spore, Swampert Hydro Pump, Umbreon Wish). Measured here: the pinned table (both
pins identical) holds a strict subset of the current one, 1,500 pairs added and none removed. `hiu.fix` replaces
`replica.sampling.learnset` with the current table (`data/pokemon/gen3_learnset.json`, sha256
`c90202a1…a456596`, checked at load; the facade's semantics: an unknown species is unconstrained) for EVERY arm,
F, F+M and O-M included, and asserts at start that Tyranitar Dragon Dance is back in the prior. F, F+M and O-M are
therefore re-run here, paired with F+U and O-U on the same states, so every share is handicap-free.

### The U arms

- **F+U**: world j = this build's F world j with every never-seen slot replaced by the TRUE packed entry (its
  realized gender) at its TRUE slot; the seen mons' draws stay exactly F's (the guard checks byte equality). Gated
  exactly as F (lines AND rows byte-equal, legality, the playout tool's replay). F+U is F's 8 worlds or nothing: a
  world that fails fails the state (reason recorded), never a re-draw.
- **O-U**: the TRUE team; every never-seen slot re-drawn as F draws it (species from the Smogon teammate prior given
  the revealed species, the mean over the revealed species, then a whole set from the Smogon prior), no species
  twice and no gen-3 OU clause broken, the drawn mons placed on the never-seen slots in a uniformly random order;
  every seen mon verbatim true. Gated as F, up to 60 draws for 8 worlds.
- Seeds and scorer as §13.1: world j on dice seed j of `truth.turn_seeds(id, 48)`; LEAF-K1 / K4 / K8; IMP =
  T(argmax, ties toward the policy's order) − T(top1), the referee on truth seeds 16-39.

## Spec ambiguities, and the choice made for each

| # | where the spec is silent | this build's choice |
|---|---|---|
| U1 | the opponent's PENDING root choice (logged before the searched side's, not yet executed): the replica drops it in every constructed world (its A3), O keeps it | the arms built from the TRUE team (O-M, O-U) KEEP it, rendered for the world, as O does: an "O minus x" arm then differs from O only by x, and the registered identity O-U = O at k = 6 holds on the pending states too. The arms built from F (F, F+M, F+U) drop it, as the replica's F does. This changes O-M from the replica's convention on the pending states only (7 of 600 end2 states) |
| U2 | "place the drawn mons on the never-seen slots in a uniformly random order" vs the replica's slot-by-slot draw (not exchangeable: earlier slots get the popular species more often) | the mons are drawn as the replica draws them, then permuted uniformly onto the never-seen slots, in EVERY arm that draws U (F, F+M and their screens' contexts, O-U) |
| U3 | F+U's world that fails the gate | the state fails F+U with the reason; no re-draw (F+U must be F's world j) |
| U4 | the gender of a true never-seen entry | its REALIZED gender (the replica's seed alignment plays explicit genders) |
| U5 | the opponent population of a state (secondary read) | `sentinel_<i>` opponents are our own snapshots (the trainer's eval label for a snapshot); every other label is a scripted bot |

## Identities (the build is invalid unless they hold)

At k = 6 (every opponent mon seen) F+U ≡ F and O-U ≡ O, outcome for outcome (every action, every world).
`hiu.score` checks them on every k = 6 state and records each mismatch (`result.json` `identities`; the build is
INVALID if `HOLD` is false on either checkpoint). The k = 6 states: end2 129 / 600 (one with a pending choice),
prod 103 / 600.

**Verification run (2026-10-15 03:05-03:20, code identical to `21e7bb38`'s runner (built as `db8f6f30` before the rebase; same tree); 13 registered states chosen for
k = 6, a pending choice, low and mid k):** at the time of this commit, 4 k = 6 states had finished: F+U = F on all
4 (end2 1, prod 3) and O-U = O on all 3 that had O-U (prod), outcome for outcome. Every arm built 8 worlds on every
finished state; the guard checked every world (contract and engine) and raised NOTHING after one fix (below). The
end2 pending k = 6 state was still running; its identity is read by `hiu.score` with everything else.

**Guard finding during the build:** the first test raised on O-U because the engine's request names Return by its
happiness power (`return102`); the guard now strips that power for Return / Frustration (a guard false positive,
not a world defect).

## Runner and handoff

- **Rows:** `~/gen3ai_archive/hidden_info_u_replica/rows/<cp>/<arm>/w<pid>.jsonl` (one JSON line per state and arm:
  the 8 world teams, every action's 8 outcomes, the build bookkeeping, the guard's counts).
  **Done-markers:** `~/gen3ai_archive/hidden_info_u_replica/done/<cp>/` (one file per state). **Claims:**
  `.../claims/<cp>/`.
- **Code:** the workers run from the snapshot `~/gen3ai_archive/hidden_info_u_replica/code_db8f6f30/` (the tree of `21e7bb38`, named for its pre-rebase sha; made by
  `snapshot.sh`), not from any worktree. Two workers, one per checkpoint, each chained after its verification
  worker (`~/gen3ai_archive/hidden_info_u_replica/chain_full.sh`), each `run_worker.sh` (cores 5-7 / 13-15,
  nice 19, an 8 GB `mem_cap` scope, re-run up to 6 times after a crash). Logs: `full_<cp>.log`,
  `verify_<cp>.log` in the same folder.
- **More workers:** safe at any time (claims are exclusive): `setsid nohup <code>/hidden_info_u_replica_2026-10-15/run_worker.sh <cp> <code> ~/gen3ai_archive/hidden_info_u_replica > <log> 2>&1 &`
  — mind rule 5b's two-heavy-jobs cap while the GPU is leased.
- **Status:** `cd designs/research_state/measurements/hidden_info_u_replica_2026-10-15 && python -m hiu.status`.
- **Cost:** 50-480 s of wall per state for the six arms on a shared core (median about 2.5 min); 1,200 states at two
  workers is about 30-45 h of wall, about twice the brief's estimate (the six arms' playouts dominate: every legal
  action x 8 worlds to the end of the game, one torch thread).
- **The read (when every state has a done-marker):**
  `cd designs/research_state/measurements/hidden_info_u_replica_2026-10-15 && python -m hiu.score --root ~/gen3ai_archive/hidden_info_u_replica --out .`
  writes `result.json` (identities, exclusions, every read: end2 / prod / pooled, per population, LEAF-K8 / K4 /
  K1), `imp_per_state.json` and `marginals.json`; `hiu.score.tv_permutation_test` is the permutation-null TV test
  for a third party's marginal comparison. Then fill "Results" here. Do NOT compare with the first implementation.

## Blindness (what this build read and did not read)

- **Read:** the replica's `replica/` code and its README; `design_hidden_info_binding.md` §13.1 only (plus, once,
  the document's heading list to find §13.1; no other section's text); the registered states
  `value_disc_pilot_2026-10-11/states_<cp>.json`, the bank and the Phase 1 truth rows they point to (the rows for
  the referee only); the pin checkouts `~/.cache/gen3ai/vdp/pin_{end2,prod}`; `agents.gen3_data.learnset` (current)
  and the learnset fix commit's FILE LIST only (`git show --stat`, message not read); the Rust engine's request
  format (by running it).
- **Not read:** `src/main/value_search/`, `src/agents/info_levers/`, any `determinize.py`,
  `hidden_info_tier1_2026-10-13/`, `~/gen3ai_archive/hidden_info_tier1/`, `design_hidden_info_binding.md` §16 or
  any section but §13.1, `search_ledger.md`, `UNDERSTANDING.md`, `ledger.md`,
  `hidden_info_by_reveal_count_2026-10-12/`; the replica's own rows, `result.json`, `marginals.json` and
  `imp_per_state.json` were not opened. The replica's README holds its own final tables; it was read for the build
  description, and none of its numbers is used here.

## Results (this implementation's own numbers; NO comparison with any other implementation)

Read run 2026-10-16 with `python -m hiu.score --root ~/gen3ai_archive/hidden_info_u_replica --out .` on the complete row
set: 600 states per checkpoint, 1,200 done-markers, every arm present for every state, no duplicate rows, no `hiu.run`
alive. Files: `result.json` (identities, exclusions, every read), `imp_per_state.json`, `marginals.json`. Intervals are
95 %. "Boundary" in the JSON marks a bound within +-0.005 of 0 (rule 8: not clear).

### Identities (registered: at k = 6, F+U = F and O-U = O, outcome for outcome)

| checkpoint | k = 6 states | F+U = F | F+U or F missing | O-U = O | O-U or O missing | `HOLD` |
|---|---|---|---|---|---|---|
| end2 | 129 | 114 | 15 | 129 | 0 | true |
| prod | 103 | 94 | 9 | 101 | 0 | **false** |

**IDENTITY FAILURE on prod, 2 states (`HOLD` = false; per the registration the build is invalid if an identity fails).**
No outcome differs on any state where both arms exist (zero `FU_ne_F`; zero `OU_ne_O` on end2). The two prod
O-U states (`.../random/loss_s2_003#p1#22` and `.../sentinel_1/loss_s1_005#p1#25`) have O-U `ok` = false while O is ok:
the build's own consistency guard RAISED in O-U world 0, slot 4: the contract has `hiddenpowerdark`, the engine's view
has `hiddenpower` (the same Hidden Power move; the guard's name normalisation covers Return / Frustration but not Hidden
Power, a guard false positive of the kind already fixed once for Return). Both states have no never-seen slot (k = 6),
so the O-U team is the true team and the guard fired on the TRUE team. The same guard message also excluded
prod states from F (4 states), F+M (2) and O-M (2) too (see below). The failure is therefore a missing O-U row on 2 of 103 states, not
a differing outcome, but the registered check as scored reads FALSE; whether that invalidates the build is the third
party's call, and it is reported as a finding, not fixed here (the runner and its rows are frozen).

### Exclusions (states dropped from an arm; the common set is the states with every arm)

| checkpoint | F | F+M | F+U | O-M | O-U | common set |
|---|---|---|---|---|---|---|
| end2 | 43 (42 `no_consistent_context`, 1 HiddenPowerTracker exception) | 36 (35 `no_consistent_context`, 1 `only_7_worlds`) | 43 (`F_not_ok`) | 1 (`only_0_worlds`) | 2 (`only_0_worlds`) | 545 |
| prod | 38 (34 `no_consistent_context`, 4 guard Hidden Power naming) | 34 (32 `no_consistent_context`, 2 guard) | 38 (`F_not_ok`) | 2 (guard) | 3 (2 guard, 1 HiddenPowerTracker exception) | 544 |

### The headline read: LEAF-K8, IMP in win probability per decision

| population | states (end2 / prod) | s_U | n_U | s_M | n_M | R3 reveal-only | R3 hide-only |
|---|---|---|---|---|---|---|---|
| pooled (end2+prod) | 545 / 544 | 0.31 [0.17, 0.44] | 0.92 [0.77, 1.07] | 0.00 [-0.17, 0.14] | 0.55 [0.43, 0.69] | +0.020 [+0.009, +0.031] | +0.023 [+0.012, +0.034] |
| end2 | 545 / 0 | 0.23 [0.01, 0.43] | 0.86 [0.68, 1.09] | 0.06 [-0.18, 0.27] | 0.60 [0.41, 0.81] | +0.011 [-0.004, +0.027] | +0.016 [+0.001, +0.032] |
| prod | 0 / 544 | 0.39 [0.20, 0.57] | 0.98 [0.77, 1.24] | -0.06 [-0.32, 0.13] | 0.50 [0.34, 0.69] | +0.029 [+0.014, +0.044] | +0.030 [+0.015, +0.044] |
| scripted bots | 306 / 330 | 0.52 [0.38, 0.66] | 0.87 [0.71, 1.06] | 0.06 [-0.14, 0.22] | 0.39 [0.26, 0.53] | +0.032 [+0.020, +0.046] | +0.034 [+0.020, +0.048] |
| own snapshots | 239 / 214 | -0.07 [-0.44, 0.20] | 1.01 [0.76, 1.38] | -0.11 [-0.51, 0.15] | 0.85 [0.59, 1.18] | +0.002 [-0.016, +0.020] | +0.008 [-0.008, +0.026] |
| end2, bots | 306 / 0 | 0.46 [0.22, 0.69] | 0.84 [0.60, 1.15] | 0.15 [-0.19, 0.42] | 0.34 [0.12, 0.58] | +0.021 [+0.002, +0.041] | +0.032 [+0.011, +0.054] |
| end2, snapshots | 239 / 0 | -0.09 [-0.59, 0.22] | 0.90 [0.63, 1.29] | -0.06 [-0.52, 0.26] | 0.96 [0.62, 1.44] | -0.002 [-0.026, +0.023] | -0.004 [-0.028, +0.020] |
| prod, bots | 0 / 330 | 0.57 [0.39, 0.74] | 0.90 [0.69, 1.16] | -0.01 [-0.28, 0.20] | 0.43 [0.28, 0.60] | +0.043 [+0.026, +0.062] | +0.035 [+0.017, +0.055] |
| prod, snapshots | 0 / 214 | -0.05 [-0.76, 0.38] | 1.17 [0.74, 2.07] | -0.19 [-1.06, 0.18] | 0.68 [0.28, 1.28] | +0.006 [-0.021, +0.034] | +0.022 [-0.003, +0.046] |

Per-arm IMP (LEAF-K8, common set) and G (the oracle's gain over the policy's pick, as `hiu.score` defines it):

| population | F | F+M | F+U | O | O-M | O-U | G |
|---|---|---|---|---|---|---|---|
| pooled (end2+prod) | +0.0110 | +0.0110 | +0.0308 | +0.0742 | +0.0393 | +0.0161 | +0.0631 |
| end2 | +0.0153 | +0.0193 | +0.0300 | +0.0783 | +0.0404 | +0.0239 | +0.0630 |
| prod | +0.0067 | +0.0027 | +0.0316 | +0.0700 | +0.0382 | +0.0083 | +0.0633 |
| scripted bots | +0.0038 | +0.0079 | +0.0404 | +0.0739 | +0.0465 | +0.0128 | +0.0701 |
| own snapshots | +0.0212 | +0.0154 | +0.0172 | +0.0745 | +0.0292 | +0.0207 | +0.0533 |
| end2, bots | +0.0110 | +0.0208 | +0.0415 | +0.0767 | +0.0541 | +0.0217 | +0.0656 |
| end2, snapshots | +0.0207 | +0.0174 | +0.0153 | +0.0804 | +0.0228 | +0.0268 | +0.0596 |
| prod, bots | -0.0030 | -0.0040 | +0.0394 | +0.0713 | +0.0394 | +0.0046 | +0.0743 |
| prod, snapshots | +0.0217 | +0.0130 | +0.0195 | +0.0680 | +0.0363 | +0.0140 | +0.0462 |

Pooled paired differences (LEAF-K8): F+U - F = +0.0198 [+0.0095, +0.0303]; O - O-U = +0.0580 [+0.0479, +0.0683];
F+M - F = +0.0000 [-0.0099, +0.0094]; O - O-M = +0.0349 [+0.0261, +0.0434].
R3's two conditions on the pooled set: reveal-only +0.0198 [+0.0089, +0.0310] (excludes 0 clearly: True);
hide-only +0.0232 [+0.0121, +0.0344] (excludes 0 clearly: True).

### The other scorers (pooled)

| scorer | s_U | n_U | s_M | n_M | R3 reveal-only | R3 hide-only |
|---|---|---|---|---|---|---|
| LEAF-K8 | 0.31 [0.17, 0.44] | 0.92 [0.77, 1.07] | 0.00 [-0.17, 0.14] | 0.55 [0.43, 0.69] | +0.020 [+0.009, +0.031] | +0.023 [+0.012, +0.034] |
| LEAF-K4 | 0.38 [0.22, 0.52] | 0.88 [0.73, 1.06] | 0.10 [-0.07, 0.26] | 0.67 [0.53, 0.84] | +0.016 [+0.005, +0.026] | +0.012 [+0.002, +0.023] |
| LEAF-K1 | 0.37 [0.11, 0.61] | 0.89 [0.65, 1.22] | 0.04 [-0.34, 0.28] | 0.80 [0.56, 1.14] | +0.009 [+0.001, +0.018] | +0.002 [-0.008, +0.011] |

### Per-state IMPs and marginals

`imp_per_state.json` (per-state IMP per arm and scorer) and `marginals.json` (per checkpoint: the drawn species / nature
frequencies for class U and the seen mons' move / item / ability frequencies) are the inputs for a third party's
comparison. `hiu.score.tv_permutation_test` (the permutation-null TV) is NOT run here: it needs the other
implementation's marginals, and this folder does not compare.
