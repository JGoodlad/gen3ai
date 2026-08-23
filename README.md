# Gen3AI — a Pokémon battle AI for Generation 3 (ADV) OU

**A reinforcement-learning agent that plays competitive Pokémon** — Gen 3 OverUsed on
[Pokémon Showdown](https://pokemonshowdown.com/) — built from scratch: PPO self-play, an
entity-token transformer with learned beliefs over the opponent's hidden team, a differentiable
damage calculator inside the network, a byte-exact Rust reimplementation of the battle engine,
and a forensic toolchain that can tell you whether a lost game was bad luck or a bad decision —
by re-rolling the dice.

This is a serious research project, and a welcoming one. **Contributions of any size are
genuinely wanted** — an idea, a question, a single test, or a subsystem. See
[Contributing](#contributing--or-just-say-hi).

## Why Gen 3?

ADV OU is a beautiful problem: **imperfect information** (you see six Pokémon; you must *infer*
their moves, items, and spreads), **long horizons** (stall wars run hundreds of turns), sharp
tactical branches (one wrong switch loses the game), and a mature, human-optimized metagame to
measure against. No physical/special split, Spikes with only one answer, pursuit trapping, sand —
the generation rewards genuine strategic understanding rather than raw damage output.

## What's inside

- **The model** — an entity-token transformer: every Pokémon, move, and threat is a token;
  attention between them is *biased by computed physics* (damage rolls, speed order, trapping)
  rather than left to discover the game from scratch. Supervised **belief heads** infer the
  opponent's hidden species, movesets, items, EV spreads, and Hidden Power types from play, and
  an **intent model** predicts what they'll click next — consumed by both the policy and the
  critic. A **differentiable damage operator** computes the full Gen 3 damage formula on GPU,
  inside the forward pass, validated against the real simulator by constructed-scenario oracle
  fuzzing.
- **The simulator** — training runs against an **in-process Rust reimplementation of the Gen 3
  Showdown battle engine**: byte-for-byte protocol parity with the reference implementation,
  validated move-by-move; no server, no websockets, deterministic replay from recorded seeds.
- **Training** — PPO self-play with a frozen-opponent pool and promotion gates, distributional
  critic, non-blocking evaluation workers, and an **anchored Bradley–Terry ELO** that makes
  model generations comparable across runs. Thirteen generations and counting.
- **The prober** — a forensic replay inspector (web UI): for any lost game it can attribute the
  loss to **luck vs. mistake by re-rolling the actual dice**, replay counterfactual moves against
  the real opponent to a win/loss, and beam-search for a better line by cloning mid-battle
  simulator states. **A live instance browses real training runs at
  [prober.g5d.io](https://prober.g5d.io)** — pick a run, open a battle, and read a game
  turn-by-turn with what the model believed, what it expected the opponent to do, and what the
  critic thought of every decision. No install needed; it's the fastest way to see what this
  project actually does.

## Engineering culture

The part we're quietly proudest of. Every refactor is gated on **byte-identical model outputs**
(a sha over the forward pass); the physics are pinned by **oracle fuzz tests** against the real
engine; 5,000+ tests run in the routine gate with mypy and ruff enforced inside it; the
architecture diagram is **generated from the live code**, and a module without edges fails a
completeness test. When we found a silently-dead subsystem this month, the fix shipped with the
structural guard that makes the whole bug class unrepresentable. History is append-only, claims
carry their measurements, and retractions are recorded as retractions.

## Getting started

```bash
git clone git@github.com:JGoodlad/gen3ai.git && cd gen3ai
./scripts/bootstrap.sh              # conda env, submodule, sim build — then verifies itself

conda activate gen3ai_stable
# a 1-minute training smoke, no server and no GPU needed:
python src/main/train_rl_agent.py --debug --steps 10000
```

`bootstrap.sh` is idempotent and fail-loud — re-run it any time and it skips what is already
done; `--dry-run` prints the plan without touching anything. One of its steps is
`pip install -e .`, so `import agents` works from any directory with nothing exported. Skipped the
bootstrap, or working in a git worktree? `export PYTHONPATH=$PYTHONPATH:src` is the equivalent
fallback and is what you will still see at the top of many scripts.

How to work in this repo — tests, ports, the worktree flow: **[CONTRIBUTING.md](CONTRIBUTING.md)**.
Full training, evaluation, and test commands: **[docs/RUNNING.md](docs/RUNNING.md)**.
The architecture as it stands today: **[designs/ARCHITECTURE.md](designs/ARCHITECTURE.md)**.
How it got here, version by version: **[designs/CHANGELOG.md](designs/CHANGELOG.md)**.

## Contributing — or just say hi

You don't need to train a model to contribute here, and you don't need to contribute to be
welcome. **[CONTRIBUTING.md](CONTRIBUTING.md)** has the mechanics — setup, which test command to
run before you push, the port rules, and the worktree workflow. All of these are valued:

- **Ideas and questions.** Open an issue to argue about ADV theory, RL design, or why the agent
  under-switches. Half the good levers in this project started as a conversation.
- **Small things.** A failing-case report, a doc fix, one more fuzz scenario, a test for an edge
  you know from playing the tier. The test suite's whole philosophy is that small pins compound.
- **Medium things.** The prober web UI, the data tooling, benchmark harnesses, the Rust
  simulator's remaining coverage tails.
- **Big things.** If you want to own a research direction — opponent modeling, search, league
  training — open an issue and let's talk.

If you play ADV seriously and think the bot's play is wrong somewhere, that's not a complaint,
that's *data* — we have tooling specifically built to turn "this move was bad" into a measured
answer.

## Prior work and credit

This project stands on **Jett Wang's MIT MEng thesis** — *Winning at Pokémon Random Battles
Using Reinforcement Learning* (MIT EECS, 2024; PPO + MCTS on gen4randombattles, peaking at
**rank 8 on the official Showdown ladder** — the best known result by a non-human agent in that
format). While Gen3AI has since diverged substantially — different generation, team play rather
than randoms, belief modeling, an in-network damage operator, its own simulator — Wang's work
was foundational in getting this project set up, and its problem framing shaped ours. A copy
lives at `designs/references/wang2024_pokemon_rl.pdf`.

Also load-bearing: [poke-env](https://github.com/hsahovic/poke-env) (the Python Showdown client
this project forked and builds on) and [Pokémon Showdown](https://github.com/smogon/pokemon-showdown)
itself — the reference battle engine our Rust port is validated against, move by move.

## License

[MIT](LICENSE) — use it, fork it, build on it; keep the notice. The vendored
[poke-env](https://github.com/hsahovic/poke-env) fork under `src/poke_env/` retains its original
MIT notice, and [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (a git submodule)
is its own MIT-licensed project. Pokémon itself is © Nintendo/Creatures/GAME FREAK — this is an
unaffiliated fan research project.

## Keywords

Pokémon AI · Pokémon Showdown bot · reinforcement learning · PPO · self-play · Gen 3 OU · ADV ·
imperfect-information games · transformer · belief modeling · opponent modeling · Rust game
engine · counterfactual analysis
