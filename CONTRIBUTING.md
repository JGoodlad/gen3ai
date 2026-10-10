# Contributing to Gen3AI

Thanks for being here! 👋 Gen3AI is an open research project teaching a neural network to play
Gen 3 OU on Pokémon Showdown, and help of every size is welcome: a question, a bug report, a
single test, or a whole subsystem.

It's built for two kinds of people, and we need both:

- **ML and software folks** (Python, PyTorch, Rust, web), and
- **competitive Pokémon players**, who can tell when a move was wrong or a mechanic is off.

You don't need a GPU, a trained model, or to read everything in this repo. This page is all you
need for a first pull request.

## Ways to help

- **Report a bad play.** Open a battle on [prober.g5d.io](https://prober.g5d.io), find the turn,
  and tell us what you'd have clicked and why.
- **Check a mechanic.** Compare something we hand the network against Showdown's source, and pin
  any mistake with a test.
- **Fix something small.** The README has a [good first contributions](README.md#good-first-contributions)
  table, sorted by what you already know.
- **Fix a confusing doc.** If a page confused you, that's a bug, and the fix is a welcome PR.
- **Ask a question.** "I don't understand this part" is useful: it usually means the docs are wrong.

## Your first PR in 5 steps

1. **Fork** the repo on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<you>/gen3ai.git && cd gen3ai
   ```

2. **Set up** with one command (safe to re-run; `--dry-run` shows the plan first):

   ```bash
   ./scripts/bootstrap.sh
   conda activate gen3ai_torch28
   ```

3. **Make your change** on a branch (`git checkout -b my-fix`).

4. **Run the tests** (about 5 minutes, CPU only):

   ```bash
   pytest src/ -m "not slow and not e2e" -q -n 6
   ```

5. **Open a pull request** against `main`, saying what you changed and which tests you ran.

Stuck at any step, or not sure your idea fits? Open the PR or an issue anyway and ask. A maintainer
will help you get it over the line.

## What makes a PR easy to merge

- **A test that fails without your change.** For a bug fix, a small test that pins the exact case.
- **Docs updated in the same PR.** If your change makes a `README.md`, a `CLAUDE.md` or
  `designs/ARCHITECTURE.md` out of date, fix it there too.
- **Game-mechanics claims checked against Showdown's source.** Gen 3 rules live in
  [`data/mods/gen3/`](https://github.com/smogon/pokemon-showdown/tree/master/data/mods/gen3)
  (locally `deps/pokemon-showdown/data/mods/gen3/`). Name the file in your PR.
- **Keep it focused.** One fix or one idea per PR is much easier to review than several.

## Proposing an experiment or a bigger change

For anything bigger than a fix (a new feature for the network, a model change, an experiment),
please open an issue first so we can talk it through before you spend time on it. Experiments here
are planned before the first game is played, so a short proposal helps a lot:

```markdown
**The change:** the one thing you'd change (one lever per experiment).
**The question:** what you expect it to do, and why.
**How we'd measure it:** the metric you'd read.
**Keep or drop:** what result would make us keep it, and what would make us drop it.
```

The current queue is [`EXPERIMENT_BACKLOG.md`](designs/research_state/EXPERIMENT_BACKLOG.md), and
what we believe today is in [`UNDERSTANDING.md`](designs/research_state/UNDERSTANDING.md).

## Being kind

Be kind and direct in issues and reviews, and disagree with ideas, not people. And please be
respectful of the people on Pokémon Showdown: they're real players enjoying a game they love. See
the README's [Respect for players](README.md#respect-for-players).

## Going deeper

- [`docs/DEVELOPING.md`](docs/DEVELOPING.md): the detailed developer reference (setup in depth, the
  test tiers and markers, the static gates, ports, worktrees, code conventions).
- [`docs/RUNNING.md`](docs/RUNNING.md): training, resuming, evaluation and the launcher.
- [`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md): the model as it is today.

## For AI agents and power users

AI coding agents (for example Claude Code) working in this repo must follow the root
[`CLAUDE.md`](CLAUDE.md): its standing rules, the git workflow, the GPU lease and the test tiers.
They also follow the leaf `CLAUDE.md` in each directory they touch, and the `designs/` docs those
point to. These files are the agents' rulebook. Humans can ignore them unless you're curious,
though each leaf `CLAUDE.md` is also good, detailed documentation for its area.

One rule worth knowing even if you're human: `python src/main/play.py --mode ladder` (queueing a model
on the Showdown ladder) is allowed for people at one battle at a time, but the code refuses it in any AI
agent session unless the maintainer has written an approval token by hand. See the README's
[Respect for players](README.md#respect-for-players).
