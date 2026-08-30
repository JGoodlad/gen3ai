# The GLOBAL-RANDOM COUPLING sweep — a genre census

**Date:** 2026-08-30 · **Scope:** `src/agents`, `src/main`, `src/utils`, `src/poke_env`
**Method:** exhaustive grep for module-state RNG draws (`random.<fn>(`, `np.random.<fn>(`,
`torch` global sampling) over every non-test module, then per-site inspection of WHO draws, WHEN,
and against WHAT other drawer.

---

## 0. The genre

A drawer that reaches into a **process-wide** RNG couples itself to every other drawer in the
process. Two players interleave their `choose_move` calls inside one battle; two paired **arms**
interleave them *differently* (one awaits an executor, the other runs inline). So a decision that
consumed the shared stream lands differently in the two arms **with no treatment involved** — and
the paired design's whole claim is that the two arms differ only by the treatment.

The founding specimen was `Gen3StallerPlayer`'s Protect coin, fixed opt-in at **4437c85**. It was
found by a **failed integrity check**, not by review: the transfer-coefficient cell
(`transfer_coefficient_cell_2026-08-29.md` §4) ran a paired falsifier whose zero-overrule units
must be the *same battle* in both arms. It came back **exactly 0.0000 over 2693 pairs** on the
deterministic bots and **non-zero on exactly the two stallers**. That cell explicitly deferred the
repair — "a change to the shared eval opponents [that] belongs to whoever owns that surface". This
sweep is that repair, plus the census that asks how many more there were.

**The answer is four, and the founding specimen was the smallest of them.**

---

## 1. Classification

| class | meaning |
|---|---|
| **(a)** | **cross-arm coupling in paired designs** — a mid-battle / per-episode draw, interleaved with another arm's draws. The staller class. Worst. |
| **(b)** | **irreproducibility only** — no pairing surface, but a fixed-seed claim that the seed does not actually deliver. |
| **(c)** | **benign** — one-shot setup, an already-private RNG, or a path this project never executes. |

---

## 2. The census

### Class (a) — FIXED (4 sites, all opt-in, all defaults byte-identical)

| # | site | who draws | when | flag / hook | landed as |
|---|---|---|---|---|---|
| a1 | `poke_env/player/player.py` — `choose_random_{move,singles_move,doubles_move}` + the `DEFAULT_CHOICE_CHANCE` coin + `random_teampreview`; and `baselines.py` — `SimpleHeuristicsPlayer` / `MaxBasePowerPlayer`'s `choose_singles_move` fallbacks | **every** player | per decision | `rng_seed=` · `$GEN3AI_PLAYER_SEED` | `gen3_player_choice_rng_v1` |
| a2 | `utils/teambuilder.py` — `Gen3Teambuilder._draw_team` (bias coin, bias choice, uniform choice, PFSP `choices`) | both teambuilders | per battle | `rng_seed=` · `$GEN3AI_TEAM_SEED` | `gen3_team_draw_rng_v1` |
| a3 | `agents/inference/player.py` — `RLPlayer._predict_best_action`'s `Categorical(...).sample()` (**torch**'s default generator, not `random`) | every RL player | per decision, when `stochastic` | `policy_seed=` · `$GEN3AI_POLICY_SEED` | `gen3_policy_sample_rng_v1` |
| a4 | `agents/training/snapshot_pool.py` — `SnapshotPool.sample` | the self-play env wrapper | per episode | `rng_seed=` · `$GEN3AI_POOL_SEED` | `gen3_pool_sample_rng_v1` |
| — | `agents/opponents.py` — the two stallers' Protect coin | *(prior art)* | conditional | `protect_seed=` · `$GEN3AI_STALLER_SEED` | 4437c85 |

**a1 is the worst finding, and it is worse than the specimen that started the sweep on three
counts.**

1. **Breadth.** `choose_random_move` is not a conditional coin — it is `RandomPlayer`'s **entire
   policy**, one draw per decision. It is also the fallback of all sixteen scripted bots in
   `agents/opponents.py` (~50 `self.choose_random_move(battle)` sites) and of `baitbot`; and
   `DEFAULT_CHOICE_CHANCE` fires inside the **RL players** too, so even an all-deterministic-bot
   roster has a shared-stream consumer in it.
2. **It was invisible to the check that found the specimen.** The transfer cell's falsifier
   conditions on **zero-overrule** units, and the measured overrule rate against `random` is
   **1.00** (`transfer_coefficient_cell_2026-08-29.md` §5). The `random` bot therefore contributed
   **no units at all**, and the same table's "7 deterministic bots · delta exactly 0.0000" row was
   read as covering it. *A falsifier that a subject contributes zero units to has not exonerated
   that subject.*
3. **It is the seam every other one-off fix would have had to re-invent.** Because it lives on
   `Player`, closing it also gave the three benchmarks below (`self._choice_rng`) and any future
   scripted bot a per-instance stream for free.

**a3 is the one a `random`-only grep never finds.** Torch has its own process-wide default
generator, `Categorical.sample()` draws from it, and `stochastic=True` is the **default** for the
self-play pool opponents and the stable cross-run opponents — so a self-play battle has *two*
RLPlayers sampling every decision off one shared torch stream, interleaved by the bridge.

**a4 is an internal inconsistency as much as a coupling.** The only caller of `sample()` is
`MaskableAgentWrapper`, which already owns a per-env `random.Random(rng_seed)` for *which bucket*
it picks — its own comment says "per-env seed → envs don't pick in lockstep" — and then reached
into the global module for *which snapshot*. A wrapper that looks seeded was not reproducible.

### Class (b) — FIXED where cheap, otherwise documented (6 sites)

| # | site | finding | action |
|---|---|---|---|
| b1 | `agents/training/live_view_build_benchmark.py` | `random.seed(seed)` + 2 teambuilders + `RandomPlayer` + a `_Capture` fallback on one stream, interleaved by the bridge. Its whole design is "capture ONE seeded board and freeze it", which is worth nothing if the board moves. **The `random.seed(k) is NOT enough` rule, in the benchmark that most needed it to be enough.** | FIXED — all four drawers seeded off distinct derived seeds |
| b2 | `agents/training/obs_build_benchmark.py` | same shape; the profiled battle was not reproducible from `--seed` | FIXED — same, plus the bench player's action pick moved to `self._choice_rng` |
| b3 | `agents/training/trainer_turn_benchmark.py` | same shape | FIXED — same |
| b4 | `agents/training/instrumented_ppo/aux_terms.py:73` | `np.random.default_rng(int(np.random.randint(...)))` — a private generator whose SEED comes from global numpy state. One-shot lazy init; the stream is private thereafter | documented; left (single learner process, no pairing surface) |
| b5 | `agents/training/teacher/buffer.py:54` | `np.random.default_rng()` with no seed — private but not reproducible | documented; left (offline teacher buffer) |
| b6 | `poke_env/player/baselines.py:70,93,95` | `random.choice` in `MaxBasePowerPlayer`'s **doubles** target-picking | left — gen3ou is singles; unreachable here |

`agents/training/golden_obs_capture.py` is the **reference implementation** of the b-class lesson
and needed nothing: it replaces the team draw with a `_CyclingTeambuilder`, the action with a
deterministic rule, and the dice with a fixed sim seed — "removes every randomness source so the
trajectory is a pure function of (teams, sim seed)". Its committed 991 per-decision hashes are also
the sharpest available proof that this sweep changed no default (§4).

### Class (c) — checked and benign (the rest)

| site | why benign |
|---|---|
| `poke_env/ps_client/account_configuration.py:33` | one-shot username suffix at construction; identity, not behaviour |
| `agents/training/team_completion/replay_parser.py:280` | `random.sample` in a `__main__` corpus subsample |
| `agents/action/fuzz_test_unit.py` (6 sites) | a test module's own mock-battle generator |
| `agents/training/capacity_overhead_benchmark.py:112` | `np.random.seed(7)` **deliberately** global — it exists to give both A/B arms the same minibatch permutation |
| `agents/training/winprob_finetune.py:857-859` | offline single-process CLI; seeds all three RNGs at entry and snapshots their state |
| `agents/baitbot.py:82` | already `random.Random(seed)` — private. Its two `choose_random_move` fallbacks were global and are covered by a1 |
| `agents/training/wrappers.py:166` | already `random.Random(rng_seed)`, per env. The leak was the pool draw *inside* it (a4) |
| `main/search_dividend/*` | already clean, and the best pattern in the tree: `battery.py:81` derives `random.Random(f"{salt}\|{opponent}\|{game}\|teams")` — a deterministic **string** seed per cell, so a re-run of one cell reproduces without replaying the others |
| `utils/bridge/{bridge_session,local_battle_runner}.py` | sim dice are an explicit `seed=` argument, producer-side validated (`validate_seed_spec`); a seedless START mints and **reports** a fresh seed |
| ~30 × `np.random.default_rng(seed)` / `RandomState(seed)` | `cf_audit`, `hodge`, `scaffolding`, `species_priors`, `capacity_probes`, `prober/engine/probes`, `harvest_meter`, `scaffolding_gauge`, `search_teacher_persistent_worker`, `audit_states`, `cf_label_buffer`, `winprob_finetune`, the three bridge benchmarks — all per-instance and explicitly seeded |

**Counts:** 4 class (a) · 6 class (b) · ~45 class (c). **Fixed: 4 + 3 = 7 sites.**

### Out of scope, recorded

SB3's own `Categorical` sampling inside `collect_rollouts` draws from torch's default generator.
It is third-party (not under `src/`) and the learner is not an arm of any paired battle design, so
it is noted rather than swept.

---

## 3. The shape of the fix — one pattern at all four seams

Deliberately one pattern, so a reader who has understood one has understood all of them, and so
the resolver contract can be asserted **once, parametrized over the three `random` seams**
(`global_random_coupling_test.py::TestTheResolverContract`). The torch seam (a3) has the same
three-way contract with one substitution — it resolves to a **seed or `None`**, and `None` means
"use torch's shared default generator" where the others mean "use the `random` module".

```
_resolve_X_rng(seed) →  the `random` MODULE     when seed is None and $ENV is unset/empty
                        random.Random(seed)     when a seed arrives by either route
                        raise ValueError        when $ENV is set but unparseable
```

Three properties are load-bearing:

* **The default is the module itself**, not a `Random` seeded from entropy. So the unseeded call
  site makes *the same call, on the same stream, in the same order* it always did — byte-identity
  by construction rather than by measurement.
* **An unseeded instance does not carry the attribute.** The stream lives on a **class attribute**
  (`_choice_rng` / `_rng`) and `__init__` writes an instance attribute only when a seed really
  arrived. Two consequences: an object built by bypassing `__init__` with `cls.__new__` — which
  several unit suites do — behaves exactly as before, and no `__dict__` ever holds an unpicklable
  module object (env workers each unpickle their own `Gen3Teambuilder`).
* **An unparseable env seed RAISES.** A seed that was meant to be set and silently was not makes a
  paired arm *look* reproducible while it is not — the exact failure mode the whole genre is about.

Two seams needed something extra:

**a1 needed a descriptor.** The three `choose_random_*` methods are `@staticmethod` upstream and
are called **both ways** across this tree: ~50 `self.choose_random_move(battle)` sites, plus
`Player.choose_random_singles_move(battle)` in `singles_env` / `doubles_env` / `baselines`.
Converting them to ordinary methods breaks the second spelling; leaving them static leaves the
first unable to reach the instance's RNG. `_rng_aware_static` keeps both and supplies the right
stream to each — the instance's when there is an instance, the shared module when the call goes
through the class, which has no player to be per-instance about. It is a **non-data** descriptor,
so `player.choose_random_move = MagicMock(...)` still shadows it (the pattern `opponents_test.py`
relies on).

The descriptor also closed a hole the seam alone would have left. `SimpleHeuristicsPlayer` — in
**both** the eval and the training roster — reaches the random fallback on two real decision-path
branches, and `MaxBasePowerPlayer` on one, but all three were written in the **class** form
(`Player.choose_random_singles_move(battle)`) inside methods that were themselves `@staticmethod`.
A class-form call cannot carry a per-instance stream, so seeding those players would have left
exactly those branches coupled — a fix that reports success and half-works. Both
`choose_singles_move` methods now take the same descriptor. (The transfer cell measured
`heuristic` at delta 0.0000 over 2693 pairs, so the branches are rare in practice. Rare is not
never, and this is the class of residue that makes the *next* falsifier ambiguous.)

**a3 needed a torch generator, and a care about byte-identity.** `Categorical.sample()` takes no
`generator=`, so the seeded branch calls `torch.multinomial(cat.probs, 1, True, generator=g)` —
sampling from **the same `cat.probs` tensor `Categorical.sample()` itself would feed to
`multinomial`**, never a re-derived softmax, so the only difference between the two branches is
which generator is drawn from. The unseeded branch still calls `cat.sample()` verbatim. Generators
are built lazily and **per device** (an opponent may be constructed on cpu while a trainee samples
on cuda), so an unseeded player allocates nothing.

---

## 4. Verification

Every seam gets the same five claims. The fifth is what keeps the suite honest.

1. the resolver contract (6 tests × 3 `random` seams, parametrized) — module when unseeded,
   private `Random` when seeded, explicit beats env, empty is not a seed, unparseable raises;
2. **default byte-identity at the call site** — the unseeded path reproduced against a hand-written
   replay of the pre-fix arithmetic on a re-seeded module stream, including the two-call ordering
   of the teambuilder's bias branch and the equality of the two torch sampling branches under one
   seed;
3. the class attribute answers for a `cls.__new__` instance, and an unseeded instance's `__dict__`
   is unchanged;
4. **paired arms are decision-identical when seeded**, while burning different, unpredictable
   amounts of the shared stream between decisions — the asymmetry a searched arm introduces;
5. **a revert arm**: unseeded, the same interleaving pulls the two arms apart. *If this ever
   passes, the per-instance RNG has stopped being the difference and claims 1-4 are asserting
   nothing.* The revert arm is not a simulation of the old code — it **is** the old code path,
   since unseeded is byte-identical to pre-fix.

`src/agents/global_random_coupling_test.py` — **47 tests, ~2 s**, unmarked (runs in the fast inner
loop).

### 4.1 The end-to-end measurement — how large this actually was

The unit suite proves the seams; this proves the stake. Two "arms", each playing **2 real bridge
battles** under the **same fixed sim seed `[1,2,3,4]`** and the same 8-team pool, with arm B
burning 1234 unrelated draws off the global stream before it starts — the crudest possible stand-in
for "the searched arm awaits an executor and the control runs inline".

| arm | teams + players | battle 1 | battle 2 |
|---|---|---|---|
| A | **unseeded** (pre-fix) | 84 turns, p1 **won** | 145 turns, p1 lost |
| B | **unseeded** (pre-fix) | 212 turns, p1 won | 233 turns, p1 **won** |
| A | seeded (`rng_seed=`) | 202 turns, p1 lost | 124 turns, p1 won |
| B | seeded (same seeds) | **202 turns, p1 lost** | **124 turns, p1 won** |

**Unseeded, a fixed sim seed bought nothing.** The two arms played *different games* — different
lengths, different winners — because the teams drawn and every action taken came off a stream the
rest of the process was also consuming. Seeded, the two arms are identical battle for battle.

This is the concrete form of the project's fuzz rule ("`random.seed(k)` is NOT enough — two
players share the global `random` and the bridge interleaves their `choose_move` calls"), and the
concrete form of the cost the transfer cell measured indirectly as 4 divergences in 755 pairs.
That cell saw the residue of this on *one conditional coin*; the table above is what the same
mechanism does when it owns the team draw and every action.

**Blast radius: none.** The committed obs golden's per-decision hashes are unchanged (its capture
path is deterministic by construction — see §2's note on `golden_obs_capture`), and the touched
suites pass unchanged: `opponents_test` · `snapshot_pool_test` · `wrappers_test` ·
`teambuilder_test` · `teambuilder_integration_test` · `baitbot_test` · all of `src/utils` · the
three static gates (ruff, mypy, file-size) · `packaging_gate` · `poke_env_fork_gate`.

---

## 5. What to do with it

Nothing, on a campaign run: every default is unchanged and no flag is on.

**When you next build a paired-arm design over battles**, set the four env hooks and the arms stop
sharing dice with each other:

```bash
GEN3AI_PLAYER_SEED=1  GEN3AI_TEAM_SEED=2  GEN3AI_POLICY_SEED=3  GEN3AI_POOL_SEED=4 \
GEN3AI_STALLER_SEED=5   <your harness>
```

Two cautions, both structural rather than incidental:

* **A flat seed makes two instances draw the same *sequence*.** That is reproducibility, not
  independence — and their decisions still differ, because their legal-order lists do. Where the
  two sides must be independent *as well as* reproducible, pass distinct `rng_seed=` values (the
  benchmarks in §2 b1-b3 derive four from one `--seed` for exactly this reason).
* **A seed is not a fixed battle.** The sim dice are a separate axis with a separate mechanism
  (`seed=` on the bridge, validated producer-side). The full recipe for a reproducible battle is
  still the one the project's fuzz rule states: **fixed teams, a per-player RNG, a fixed sim
  seed** — this sweep supplies the middle term, which is the one that was missing.
