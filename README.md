# Gen3AI: a reinforcement-learning agent for Gen 3 (ADV) OU

**Gen3AI teaches a neural network to play competitive Pokémon**, specifically Generation 3 OverUsed
(ADV OU) as played on [Pokémon Showdown](https://pokemonshowdown.com/). It learns by PPO self-play,
on our own Rust port of the Showdown battle engine. The network is an entity-token transformer: it
holds explicit guesses about the opponent's hidden team, computes the real Gen 3 damage formula
inside its forward pass, and has a critic whose output is the probability of winning. A forensic
battle viewer lets you read any game turn by turn and see what the model believed.

ADV OU is a great reinforcement-learning problem. Information is **imperfect**: you see the
opponent's Pokémon only as they come in, and their moves, items and spreads have to be inferred.
Games are **long**, since stall wars run for hundreds of turns. The **tactics are sharp**, because one
wrong switch into Spikes can lose the game. And the tier has a deep, human-tuned metagame to measure
against. There's no physical/special split, Spikes has a single answer, Pursuit traps and sand
chips. The generation rewards real strategic understanding over raw damage.

**This is an open research project, and contributors are very welcome**, whether you're an ML
person, a Rust or Python engineer, or a competitive player who can tell when a move was wrong.
Jump to **[Contributing](#contributing)**.

---

## Status at a glance (as of October 2026)

**Working today**

| | |
|---|---|
| **One Rust stack** | Training and evaluation run entirely **in process** on the Rust env core, a Rust port of the Gen 3 Showdown engine. No Node server is involved and no websockets are used. [poke-env](https://github.com/hsahovic/poke-env) is **retired** (finished 2026-10-08). Live play reads the Showdown protocol through the same Rust reader that produces training rows ([`designs/rust_sim/live_reader.md`](designs/rust_sim/live_reader.md)). |
| **Engine parity** | A 220-battle end-to-end golden on real teams, played to game end, must match Showdown bit for bit. An unbounded A/B fuzzer hunts for divergences ([`designs/rust_sim/e2e_capstone.md`](designs/rust_sim/e2e_capstone.md), [`designs/rust_sim/ab_fuzzer_findings.md`](designs/rust_sim/ab_fuzzer_findings.md)). |
| **Belief as hypothesis tokens** | The opponent's hidden Pokémon are represented as concrete, weighted guesses ("fixed-mass" hypothesis tokens, the X5 experiment). They were **adopted on 2026-10-07**. At matched training steps they were non-inferior to the older belief "blob" (Δ −0.99 pp [−2.92, +0.94], 8 seeds per arm) and better at predicting the opponent's next action (1.753 vs 1.904 nats). Per GPU-hour they currently cost strength ([`UNDERSTANDING.md` §4.5](designs/research_state/UNDERSTANDING.md)). |
| **Win-probability critic** | The win-probability critic is the only critic. Its value output *is* P(win), trained on the game result alone (win = 1, draw = 0, no discount); the hand-shaped reward path is deleted ([`designs/training/critic_and_value_losses.md`](designs/training/critic_and_value_losses.md)). |
| **The prober** | A forensic replay inspector with a new battle viewer, `/game`. It shows the turn's story, what the model expected the opponent to do, its hypothesis tokens, attention heat maps and the damage operator's facts. A live instance is at **[prober.g5d.io](https://prober.g5d.io)**, with no install needed. |

**Being researched now**

- **Static per-Pokémon tokens** (`--token-encoding static`). Each Pokémon's identity comes from a
  table, its battle state sits beside it, and attention plus the damage operator do the rest. This is
  a pre-registered non-inferiority screen. Look 2 (2026-10-09) read CONTINUE (Δ̂ −2.70 pp), so every
  strength claim for it is **still a hypothesis**
  ([`design_static_tokens.md`](designs/endstate/design_static_tokens.md)). Production is still the
  legacy encoding.
- **The static-recovery levers**: a third trunk round, an entry-hazard cost on the switch cell and a
  per-mon end-of-turn residual. They are built behind flags, off by default, and combined as the
  `--arch static_recovery` arm, to be bisected lever by lever.
- **What we hand-compute.** [`design_hand_computed_features.md`](designs/endstate/design_hand_computed_features.md)
  is the always-current ledger of every quantity our code derives for the network instead of letting
  it learn. It applies the **FACT vs JUDGMENT** rule: we hand the network facts and leave judgment to it.
- **Goals.** Near term, beat Metamon's best model on our teams. Long term, beat Foul Play
  ([`EXPERIMENT_BACKLOG.md`](designs/research_state/EXPERIMENT_BACKLOG.md)).

**What we do *not* claim:** a ladder rating. **We do not play against humans** (owner policy,
2026-10-07). Our public-server use is limited to low-volume self-vs-self challenges between our own
accounts. `--mode ladder` is refused in code
([`design_ladder_campaign.md`](designs/endstate/design_ladder_campaign.md), Decision record).

---

## How it works: a short tour

| Piece | What it is | Read more |
|---|---|---|
| **The Rust engine** | A port of Showdown's Gen 3 battle engine (`src/rust_sim/`) and the env core (`src/rust_env/`) that steps many battles at once and hands training its observation rows. It is deterministic and replayable from recorded seeds. | [`src/rust_sim/CLAUDE.md`](src/rust_sim/CLAUDE.md), [`design_three_tier_environment.md`](designs/endstate/design_three_tier_environment.md) |
| **The observation** | A flat 2845-dim `float32` vector plus an 11-dim action mask, written by the Rust encoder. Every offset is a named constant; nothing is hardcoded. | [`ARCHITECTURE.md` §1](designs/ARCHITECTURE.md) |
| **Entity-token transformer** | Each Pokémon, each of our moves, each opponent threat and the recent battle events become tokens (62 in production). Attention between them is **biased by computed physics**: 17 edge families covering damage, speed order, trapping and more. Each starts at zero, so the network learns how much to trust it. | [`ARCHITECTURE.md` §2 and §5](designs/ARCHITECTURE.md), [`designs/learning/entity_tokens_biases_pointers.md`](designs/learning/entity_tokens_biases_pointers.md) |
| **Differentiable damage operator** | The Gen 3 damage formula, P(KO), speed order and accuracy are computed inside the forward pass over the believed opponent sets. They're checked against the real engine. | [`ARCHITECTURE.md` §4](designs/ARCHITECTURE.md), [`designs/model/op_contracts.md`](designs/model/op_contracts.md) |
| **Beliefs and opponent intent** | Hypothesis tokens give concrete guesses at the opponent's hidden species and sets, starting from Smogon priors. An intent head predicts what the opponent will click next, and the policy and critic both read it. | [`design_x5_belief_tokens.md`](designs/endstate/design_x5_belief_tokens.md), [`designs/model/opponent_intent.md`](designs/model/opponent_intent.md) |
| **The win-prob critic** | `sigmoid(win_head(value_pooled))`. The value function is the probability of winning from this state. | [`designs/training/critic_and_value_losses.md`](designs/training/critic_and_value_losses.md) |
| **PPO self-play** | PPO against a pool of frozen snapshots of itself, with promotion gates, plus scripted bots. A pointer head picks the action. The production recipe lives in one mirrored file. | [`designs/training/self_play_and_pool.md`](designs/training/self_play_and_pool.md), [`design_learner_recipe.md`](designs/endstate/design_learner_recipe.md), [`design_league_decisions.md`](designs/endstate/design_league_decisions.md) |
| **The prober** | Reads a run's eval traces and a checkpoint. It attributes a loss to **luck or mistake by re-rolling the dice**, replays counterfactual moves, and searches for better lines. It runs in the browser and as a JSON CLI. | [`src/main/prober/CLAUDE.md`](src/main/prober/CLAUDE.md), [`designs/prober/battle_view_v2.md`](designs/prober/battle_view_v2.md) |

[`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md) is the one document that states the model **as
it is now**. [`designs/endstate/README.md`](designs/endstate/README.md) gives the end-state specs
in reading order. [`designs/research_state/UNDERSTANDING.md`](designs/research_state/UNDERSTANDING.md)
says what we currently believe, with an evidence tag on every claim. To click through the model's
data flow, see **[model.g5d.io](https://model.g5d.io)**, which is generated from the live code.

---

## Quick start

```bash
git clone https://github.com/JGoodlad/gen3ai.git && cd gen3ai
./scripts/bootstrap.sh --dry-run     # print the plan, change nothing
./scripts/bootstrap.sh --with-rust   # conda env, Showdown submodule, Rust build, self-check
conda activate gen3ai_torch28        # torch 2.8
```

`bootstrap.sh` is idempotent and fails loudly, so you can re-run it any time.
[`CONTRIBUTING.md`](CONTRIBUTING.md) explains each step. It installs `src/` in editable mode, so
`import agents` works from anywhere in the main checkout.

**A training smoke test** runs on the CPU, needs no server and takes about 2 minutes. Use a scratch
directory so the run doesn't land in `models/`:

```bash
export GEN3AI_MODELS_DIR=$(mktemp -d)
python src/main/train_rl_agent.py --debug --steps 10000
```

Look for `🦀 [ENV CORE] rust`, `[ModelVersion] Round-trip smoke test PASSED`, a metrics table per
update, and `Training complete`. To exercise the evaluation path as well, add
`--debug-eval --eval-freq 4000`.

**The prober.** The quickest look needs no install: open **[prober.g5d.io](https://prober.g5d.io)**,
pick a run, then open a battle in `/game`. To run it locally against your own runs:

```bash
python -m main.prober.web models/            # serves on :6008 with a run picker
python -m main.prober.query --help           # the same engine as a JSON CLI
```

**The tests** (tiers and timings in [`CONTRIBUTING.md`](CONTRIBUTING.md#tests--what-to-run-and-when)):

```bash
pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2   # inner loop
pytest src/ -m "not slow and not e2e" -q -n 6                                   # the routine gate
```

Full training, resuming, evaluation and the launcher are covered in **[docs/RUNNING.md](docs/RUNNING.md)**.
Real training needs a CUDA GPU; everything on this page runs on the CPU.

---

## Contributing

**You don't need a GPU, or a trained model, to make a real difference here.** Most of what makes
this project work is careful checking. Is the engine right? Is this feature a fact? Did the model
see what we think it saw? A lot of that needs Pokémon knowledge more than ML knowledge.

### Good first contributions

Every item below is real open work, linked to where it's tracked.

**1. Verify Gen 3 mechanics against Showdown (the GIGO hunt).** Our worst bugs have been quiet
mechanics errors in what the network is told. Two examples landed on 2026-10-09: the Spikes entry
rule read a Kecleon's *current* type instead of its base type, and it read the current ability
instead of the species' Levitate ([`CHANGELOG.md`](designs/CHANGELOG.md), v148 and v149). If you
know an ADV interaction cold, pick a hand-computed rule in
[`design_hand_computed_features.md`](designs/endstate/design_hand_computed_features.md) §2. Check it
against `deps/pokemon-showdown/data/mods/gen3/` (and the generations it inherits from), then send a test that pins the
case. Its §7 lists findings already made.

**2. Add a hand-computed FACT** from the ranked ADD list in
[`design_hand_computed_features.md` §4](designs/endstate/design_hand_computed_features.md). The
smaller items:
  - **Freeze Clause, per side** (rank 8, XS): the side token carries Sleep Clause but not Freeze Clause.
  - **Future Sight / Doom Desire pending on the *target* side** (rank 9, S): today it's a bit on the
    user, which is lost when the user switches.
  - **Phazing's entry damage** (rank 7, S): the expected Spikes chip of the random mon Roar or
    Whirlwind drags in.
  - **The exact KO ramp** (rank 3): P(KO) over the 16 discrete damage rolls plus the crit, replacing a
    smoothed approximation.

  Each one must pass the doc's §1 test: it's the probability or size of a **game event**, never a
  weighted judgment of what's good. If it's a prior, it's Smogon-derived. Each new fact is one lever
  with its own screen, so please open an issue first.

**3. Probe and prober views.** Make the prober usable on a phone (`/game`'s turn strip, intent bars
and attention heat map: [`TASK_BACKLOG.md`](designs/ops/TASK_BACKLOG.md) T29), or add a view that
answers a question the current ones can't. See [`src/main/prober/web/CLAUDE.md`](src/main/prober/web/CLAUDE.md).

**4. Rust engine coverage and fuzzing.** Run the A/B differential fuzzer in its broader team modes
(`node src/rust_sim/harness/ab_fuzz.js --mode random --battles 200`). Then turn each divergence into
a named, deterministic regression pin
([`designs/rust_sim/regression_pins.md`](designs/rust_sim/regression_pins.md)). Showdown's own
random-battle generator (`--mode randbats`) reaches far more species and moves than our team pool.

**5. Docs.** If a doc confused you, that's a bug in the doc. Fixing an always-current doc
(`ARCHITECTURE.md`, a `CLAUDE.md`, a `README.md`) is as valuable as fixing code.

**6. Competitive-player expertise.** You can contribute:
  - **"The model got this wrong" reports**: a [prober.g5d.io](https://prober.g5d.io) `/game` link,
    the turn, and what you'd have clicked and why. We have tooling built to turn "this move was bad"
    into a measured answer.
  - **Teams and sets**: good, legal ADV OU teams as Showdown exports (the pool lives under
    `data/teams/others/`, with its sources in `sources.json`), and corrections to set or spread
    assumptions.
  - **ADV theory arguments** in issues: why the agent under-switches, what a stall team needs to
    know, which interaction we're probably missing.

### Proposing an experiment

Open an issue that states one lever, the question, the meter you'd read and the decision rule
(what result would make you keep or kill it). Experiments here are **pre-registered** before their
first game and judged against named baselines. See the ranked queue in
[`EXPERIMENT_BACKLOG.md`](designs/research_state/EXPERIMENT_BACKLOG.md) and the evidence vocabulary
in [`UNDERSTANDING.md` §0](designs/research_state/UNDERSTANDING.md).

### What a pull request needs

- **Fork and open a pull request.** The maintainer pushes straight to `main` from worktrees, but
  outside contributions come in as PRs, and we'll review them.
- **A fix comes with a test that fails if the fix is reverted.** Prefer a targeted test, and name
  the scope you ran in the PR.
- **The routine gate passes**: `pytest src/ -m "not slow and not e2e" -q -n 6`. It includes the
  repo's static gates (mypy, ruff, file size, doc freshness and more).
- **Docs move with the code.** If your change makes a `CLAUDE.md`, a `README.md` or
  `designs/ARCHITECTURE.md` stale, update it in the same PR.

The mechanics (setup, test tiers, ports, worktrees, local conventions) are in
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

### Respect

Be kind in issues and reviews; disagreement about ideas is welcome, aimed at ideas. And **respect
the people on Pokémon Showdown**: this project never plays, challenges or chats with human players,
and contributions must not add ways to do so.

---

## Project layout

```
src/
  rust_sim/      the Rust Showdown port: engine, encoder, protocol reader, search/replay drivers
  rust_env/      the Rust env core: N battles stepped in process, columns out
  agents/        the model (model/), observation layout, data facade (gen3_data/), training
  main/          entry points: trainer, launcher, prober, live play, the offline meters
  utils/         paths, bridge, the Rust env's Python side, shared helpers
designs/         ARCHITECTURE.md, the end-state specs, research state, ops and training docs
data/            the source of truth (Smogon usage + priors, dex tables, the team pool), read via agents.gen3_data
tools/           acquisition: the only layer that talks to upstream sources
deps/            pokemon-showdown (git submodule, the reference engine)
```

Each major directory has a `CLAUDE.md` beside the code. It's the detailed, always-current
documentation for that area, written for human and AI contributors alike.

---

## Research philosophy

- **Measure, then believe.** Experiments are pre-registered with one lever, one meter and a
  decision rule. Strength reads use non-inferiority or superiority tests with stated margins.
  "No effect" is not a result; NOT DETECTED, EQUIVALENCE SUPPORTED and INDETERMINATE are.
- **Honest kills.** A failed idea is recorded as failed, and a retraction is recorded as a
  retraction. The research ledger is append-only.
- **Deterministic checks.** A test passes or fails the same way every run. Collected tests take
  seeded battles from the Rust core, and inputs within a rounding error of a decision boundary are
  excluded rather than left to chance.
- **Smogon-only priors.** Everything the network reads as a prior traces to Smogon usage data,
  ground-truth labels or ladder replays. Our training team pool may measure structure, but never
  ships as a prior.
- **Facts, not judgments.** We hand the network exact game facts (damage, P(KO), speed order,
  hazard chip). Judgment about what's *good* is the network's job.
- **Guards over good intentions.** When a silent bug is found, the fix ships with a structural check
  (a static gate, a throwing guard) that makes the whole bug class fail loudly.

---

## Acknowledgements

- **[Pokémon Showdown](https://github.com/smogon/pokemon-showdown)** and its maintainers. It's the
  reference engine our Rust port is held to, bit for bit, and the server this format lives on.
- **[Smogon](https://www.smogon.com/)**, whose usage statistics and decades of ADV theory are the
  only source of our priors.
- **Jett Wang's MIT MEng thesis**, *Winning at Pokémon Random Battles Using Reinforcement Learning*
  (MIT EECS, 2024). Its PPO + MCTS on gen4randombattles peaked at rank 8 on the official Showdown
  ladder. Gen3AI has since diverged (a different generation, team play, belief modeling, an
  in-network damage operator, its own simulator), but that work got this project started and shaped
  its framing. A copy is at `designs/references/wang2024_pokemon_rl.pdf`.
- **[poke-env](https://github.com/hsahovic/poke-env)**, the Python Showdown client this project began
  on, before the Rust stack replaced it.

## License

[MIT](LICENSE). Pokémon Showdown (a git submodule) is its own MIT-licensed project. Pokémon is
© Nintendo / Creatures / GAME FREAK; this is an unaffiliated fan research project.

<sub>Keywords: Pokémon AI · Pokémon Showdown bot · reinforcement learning · PPO · self-play · Gen 3
OU · ADV · imperfect-information games · transformer · belief modeling · opponent modeling · Rust
game engine · counterfactual analysis</sub>
