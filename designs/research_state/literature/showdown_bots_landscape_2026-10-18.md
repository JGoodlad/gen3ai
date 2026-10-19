# Showdown bots: the autumn-2026 landscape (as of 2026-10-18)

> A dated literature snapshot, from web research on 2026-10-18 (public pages only, nothing played,
> no Showdown connection). It does not get updated; a later survey gets a new dated file.
> **Every ladder figure below is SELF-REPORTED by the bot's author.** I did not read the Showdown
> ladder itself (the brief rules out connecting to Showdown). "Primary" means the author's own post,
> repo or paper. "Secondary" means a third party or a search-engine summary. Smogon thread contents
> were read through a page summariser, so a quote here may be paraphrased; the thread URLs are the
> evidence.

## 1. TL;DR

- The "bot at the top of the ladder" is real but there is more than one. The ladder is **[Gen 9] Champions
  VGC 2026 Reg M-C** (doubles, Bo1), the Showdown format for the new *Pokémon Champions* game. The
  first public bot to reach #1 was **mikumiku37** (author "Shao"), on 2026-10-01: 1857 Elo, 81.2 % GXE.
- At least three more bots then claimed #1 or the top 20 within about ten days: Nessie123 (#1 on the Bo3
  open-team-sheet ladder, about 2026-09-28), Wolfe_Alt6767, qtipsa's bot and RotomDotExe (#16).
- The winning recipe is close to ours: **a small transformer (7–9M parameters), PPO self-play against a
  league of its own past versions, a custom Rust simulator, and a one-turn simultaneous-move search**
  (regret matching over sampled worlds). It trains on one RTX 5090 in about two days.
- Smogon/PS answered on **2026-10-09** with an initial bot policy: a mandatory bot registry
  (an unregistered bot is bannable), a user toggle to opt out of bot opponents, and AI help in tournaments
  treated as ghosting. Moderators were openly hostile to the wave.
- Randbats saw the same thing in August: **Jaxcalibur** (PPO self-play plus pUCT search, a JAX simulator,
  one H100 for a week) peaked at 2557 Elo, #1.

## 2. The doubles bot(s)

### mikumiku37: the first, and the one the rumour is about

**Who and when.** Its author is "Shao" (X @shaokthx). Shao announced it on X on 2026-10-01
(https://x.com/shaokthx/status/2106149313466564815). The page could not be fetched (HTTP 402), so the
date comes from the post ID and the search snippet. Shao posted the write-up on Smogon on 2026-10-02:
https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/
(primary; 31K views). Stuart Rutherford (@Stalruth) amplified it on X
(https://x.com/Stalruth/status/2106303935048982810, secondary).

**The ladder result (self-reported).** Format: Gen 9 Champions VGC 2026 Reg M-C, Bo1. It reached #1 on
its 358th ladder game, on 2026-10-01: 1857 Elo, 81.2 % GXE, Glicko 1775 ± 25, and a record of 248–110
(69.3 %). A later update says 775 games at about 72 % (search snippet, secondary).

**Method (primary, the Smogon thread):**
- **Model.** A transformer of about 8.7M parameters, trained from scratch.
- **Training.** PPO self-play against a league of its own past versions. Rewards are terminal only, and no
  human replays are used.
- **Training scale.** About 330M training games in 48.3 h on one RTX 5090: 8,192 parallel battles,
  32 steps per update (about 262k samples), about 100k steps/s.
- **Bottleneck.** The PPO gradient takes about 70 % of wall time, so the learner is GPU-compute-bound.
- **Simulator.** A custom **Rust battle engine** at about 7k battles/s per core. An earlier version ran
  on Showdown itself, managed about 700 battles/min and used under 0.5 % of the GPU, which is why Shao
  rewrote it.
- **Action head.** Each of the two slots predicts an action plus a target, and slot 2 is conditioned on
  slot 1 (autoregressive joint action).
- **Search.** An "AlphaZero-style" ONE-TURN simultaneous-move solver, not a deep tree. It takes the top 8
  actions per side and builds a payoff matrix. Each cell averages 16 sampled worlds with several RNG
  seeds, and the damage, crit and miss outcomes are real engine draws. The value net scores the leaves,
  and regret matching solves the matrix.
- **Teams.** It draws from about 1,260 public Reg M-B and M-C tournament teams, with some EV spreads
  estimated. On the ladder it rotates among the ten teams that did best in training.
- **Known weaknesses.** It had a Perish Song bug, since fixed. Early exploits were found with
  Kingambit, Volcarona and Raichu teams.
- **Release.** Shao will release neither weights nor engine, citing the possible impact on the ladder
  and on official *Champions* tournaments. The bot accepts Bo1 open-team-sheet challenges. Shao
  credits Jaxcalibur's write-up as a partial design reference.

**Status.** The claim is CONFIRMED by its author, and the thread drew no staff dispute of the rating. The
rating has NOT been independently verified here (FINDING F1).

### The others in the same fortnight (all from Smogon's PS! Bot Projects forum, primary unless noted)

The forum index is https://www.smogon.com/forums/forums/ps-bot-projects.1059/.

- **Nessie123** (nessie_dev, posted 2026-10-03):
  https://www.smogon.com/forums/threads/nessie123-an-ots-vgc-bot-that-topped-the-reg-m-c-bo3-ladder.3789213/
  - **Result.** It was briefly #1 on the Reg M-C **Bo3, open team sheet** ladder, around 2026-09-28. That
    is probably slightly before mikumiku37's Bo1 #1, though the dates are self-reported.
  - **Evaluator.** A network of about 1.5M parameters, trained on about 375k self-play games. Half of
    those games use mutated sets. It has auxiliary heads for weather, terrain, damage and moves.
  - **Search.** A **double-oracle** search seeded by the network. It branches on critical chance events
    (KOs, crits, secondary effects), solves the implied matrix games, and deepens selectively by
    reach probability × the entropy of the static evaluation.
  - **Compute.** About 20 h on an M4 Mac Mini (one CPU core) plus about $120 of cloud compute.
  - **Teams.** A pool of about 300 teams from the Baltimore Regional open sheets (6-2 or better), with
    the stat points imputed.
  - **Release.** It will not be open-sourced; an analysis tool is planned.
  - **Comparison.** The author says mikumiku37 makes "the opposite tradeoff" (wider one-turn solving
    versus deeper search) and counts three independent bots in one week.
- **Wolfe_Alt6767** ("BotMakerGuy", 2026-10-10):
  https://www.smogon.com/forums/threads/wolfe_alt6767-1-ranked-vgc-bot-what-do-i-do-with-this-thing-now.3789554/
  - **Result.** It reached #1 on Reg M-C Bo1 in about 155 games and holds a rating of about 1800.
  - **Model.** A 7.5M-parameter transformer, 8 layers, width 256.
  - **Training.** An imitation warm start, then PPO self-play.
  - **Search.** A one-turn simultaneous-move search with regret matching.
  - **Simulator.** A Rust engine at 38k battles/s on one thread (250k on 12), validated against Showdown.
  - **Teams and compute.** About 5,600 teams from VGC Pastes and Limitless. Development ran on a GTX 1070
    (the summariser wrote "RTX 1070"); training ran on a rented RTX 5090.
  - **Controversy.** The name evokes the pro player Wolfe Glick, and readers said the write-up was
    LLM-generated.
- **qtipsa's bot** (2026-10-10):
  https://www.smogon.com/forums/threads/reached-vgc-1-with-my-bot.3789558/
  - **Result.** An RL bot that reached #1, on public teams. Details are thin.
  - **Reaction.** A moderator (Posho) replied "then don't". The author then SHELVED the bot.
- **RotomDotExe** (Subby, 2026-10-09):
  https://www.smogon.com/forums/threads/rotomdotexe-a-vgc-bot-i-created-with-almost-no-machine-learning-background-and-hit-top-20-on-the-m-c-ladder.3789518/
  - **Result.** #16, about 1800.
  - **Method.** Self-play on 139M games in 4 days, then supervised training on about 6k human replays to
    model opponents, then lookahead.
- **philmantatsky / antonius1** (repo, primary):
  https://github.com/philmantatsky/VGC-Pokemon-Showdown-AI
  - **Format and result.** Reg M-B. 55–45 over its first 100 ladder games, peak 1365. Not top.
  - **Method.** A PPO transformer with fictitious play / double oracle against a league that includes a
    behaviour-cloned model of about 14k top-player games. It adds exact-simulation search and a
    calibrated value net for the leaves.
  - **Teams.** Experiments with 1, 4, 16 and 64 teams, plus held-out-team gates. It is built on
    VGC-Bench.

## 3. Landscape table

| name | format | method | claimed strength | source (type, date) | verified? |
|---|---|---|---|---|---|
| **mikumiku37** | Gen 9 Champions VGC Reg M-C, Bo1 doubles | 8.7M transformer; PPO self-play vs past-self league; terminal reward; Rust sim; one-turn regret-matching search over 16 sampled worlds | #1, 1857 Elo / 81.2 % GXE (2026-10-01) | Smogon thread, primary, 2026-10-02 (URL §2) | author-confirmed; not independently checked |
| **Nessie123** | Reg M-C **Bo3 OTS** | 1.5M evaluator from 375k self-play games; double-oracle search with chance branching | briefly #1 (about 2026-09-28) | Smogon thread, primary, 2026-10-03 | author-confirmed |
| **Wolfe_Alt6767** | Reg M-C Bo1 | 7.5M transformer; imitation warm start + PPO self-play; one-turn regret matching; Rust sim | #1 in about 155 games, about 1800 | Smogon thread, primary, 2026-10-10 | author-confirmed |
| qtipsa's bot | Reg M-C | RL, public teams | #1 | Smogon thread, primary, 2026-10-10 | author-confirmed, thin |
| RotomDotExe | Reg M-C | self-play (139M games) + BC on 6k replays + lookahead | #16, about 1800 | Smogon thread, primary, 2026-10-09 | author-confirmed |
| antonius1 | Reg M-B | PPO + FSP/DO league incl. a BC human model; exact-sim search | peak 1365 | GitHub repo, primary | author-confirmed |
| **Jaxcalibur** | Gen 9 Random Battle (singles) | 8.5M position-MoE transformer; PPO+GAE self-play on about 100M games (1 week, 1 H100); pure-JAX simulator; pUCT with regret-mixed edge sampling for simultaneous moves, 32 sampled worlds, depth 4; search adds about 100–150 Elo; search NOT used in training | #1, peak 2557 Elo / 95.3 GXE (Aug 2026); about 2400 without search | https://jaxcalibur.github.io/ (primary); Smogon thread 3787537 (primary, 2026-08-28) | author-confirmed; code withheld (2026-09-12) |
| Laplace | Gen 9 Random Battle | 8 sampled opponent teams × MCTS on poke-engine (Rust); a small net breaks ties on 5–10 % of turns | peak 2231, rank about 150 | https://www.trylaplaceai.com/ (primary; undated); Smogon thread 2026-07-14 | author-confirmed; open source |
| Foul Play | all-gen singles | search on poke-engine (Rust) | 2341 Elo / 88 GXE in randbats, per Jaxcalibur's write-up | https://github.com/pmariglia/foul-play (primary; no strength claim in the README) | strength figure is secondary |
| **Metamon** (UT Austin) | Gen 1–4 OU + Gen 9 OU singles | offline RL on transformers from reconstructed human replays, plus 22M+ self-play trajectories; no explicit search | best agents in the 90–99th percentile; "TaurosEnsemble has held #1" on Gen 1 OU; Kakuna (142M) 71 % GXE Gen 9 OU | arXiv 2504.04395 (RLC 2025, primary); https://github.com/UT-Austin-RPL/metamon (primary) | paper-reviewed; the #1 claim is from the README |
| VGC-Bench (UT Austin) | Gen 9 VGC doubles | benchmark: 700k+ human logs; heuristic, LLM, BC, and MARL baselines (SP / FSP / DO) | single-team mirror beats a pro; generalisation falls as the team count grows | arXiv 2506.10326 (AAMAS 2026, primary) | peer-reviewed |
| PokéChamp | Gen 8 randbats, Gen 9 OU | LLM (GPT-4o) for sampling, opponent model and value inside minimax | projected 1300–1500 Elo (top 30–10 %) | arXiv 2503.04094 (ICML 2025, primary) | peer-reviewed; projected, not laddered |
| PokéLLMon | Gen 8 randbats | LLM + RAG, in-context feedback | 49 % on the ladder | arXiv 2402.01118 (primary, 2024) | peer-reviewed |

Not read: PsyMew (open-source, 2026-04), Safari (Oak NN search, 2026-07), eevenator (randbats, 2026-10)
and "VGC Arena" (a bot-vs-bot venue, 2026-10-09). All four are listed on the forum index above.

## 4. Ladder bot policy

- **The policy, posted 2026-10-09.** "Initial policy on bots", by Django (Smogon senior staff):
  https://www.smogon.com/forums/threads/initial-policy-on-bots.3789822/ (primary, read via the
  summariser). It says:
  - Bots are inevitable, and staff would rather work with bot developers than drive them underground.
  - Human-vs-human play stays the default.
  - Playing a bot must be the user's CHOICE.
  - The policy is meant to protect suspect tests and usage statistics.
  - A **mandatory bot registry**: running an unregistered bot is bannable.
  - **AI assistance in tournaments counts as ghosting** and draws an immediate ban.
- **Implementation (Hecate, Battle Simulator admin, in the same thread).** A user opt-out toggle for bot
  opponents exists. Bots register through `/ht` → "Something else > I want to register my bot".
  Enforcement is case-by-case. **UNVERIFIED** in PS's own code or news page: whether the toggle covers
  ladder matchmaking, and whether registered bots are excluded from usage stats (FINDING F3).
- **Was the wave sanctioned?** No. Every #1 run above came BEFORE the registry existed, so none was
  sanctioned. None was called a rule breach either. The reaction was hostile:
  - Moderator Posho: "we never asked for these tools"; to qtipsa, "then don't".
  - Another moderator: "Sites fucked guys."
  - Players fear undetectable cheating in online play and in official *Champions* events.

  Several authors (Jaxcalibur, mikumiku37, Nessie) withheld their code for that reason, and qtipsa
  shelved his bot.
- **For us.** This tightens our stance; it does not loosen it. Our rule (no ladder without the owner's
  token; never play humans; root CLAUDE.md § Playing) is stricter than PS's new rule. If the owner
  ever approves a ladder session, the bot must **first be registered under the PS registry**. Our
  `ladder_guard` does not check registration today (FINDING F4).

## 5. Relevance to us (MY JUDGEMENT, not measured)

1. **Our architecture choice is validated.** Independent teams converged on our recipe: a small
   transformer, PPO self-play against a past-self league, a terminal win/loss reward, a Rust simulator,
   and **one-turn search** at decision time. They reached #1 within about 2 GPU-days. mikumiku37's
   bottleneck (PPO gradient 70 %, GPU-bound once the simulator is Rust) matches ours. The 48 h /
   330M-game scale is a useful yardstick for our step budgets (not compared here; their "game" and our
   "step" are different units).
2. **One-ply search should be a simultaneous-move MATRIX game, not maximin.** All three doubles bots and
   Jaxcalibur solve the turn as a matrix game:
   - top-k actions per side;
   - each cell averaged over sampled hidden-info worlds × RNG seeds, using real engine draws;
   - value-net leaves;
   - regret matching or double oracle to get a MIXED strategy.

   Jaxcalibur adds two cautions: search adds only about 100–150 Elo over the policy, and it truncates
   search after a "surprising" own action so that hidden information does not leak to the simulated
   opponent. This bears directly on `designs/endstate/search_ledger.md` and on the P0.1 hidden-info
   binding study (`designs/endstate/design_hidden_info_binding.md`). Their "16–32 sampled worlds" is
   the practical answer to hidden sets.
3. **Team generalisation is the open problem everywhere, and nobody solved it on the ladder.**
   - The #1 bots piloted a FEW well-trained teams: mikumiku37 rotated its best 10 of about 1,260;
     Nessie used about 300.
   - VGC-Bench measured the cost: the single-team best agent becomes more exploitable and weaker as the
     team count grows.
   - This matches our x37a finding that the held-out drop is PILOTING. For a strength read, a narrow
     ladder team set is defensible; for the research goal, our held-out-team meters measure exactly
     what these bots do not.
4. **Cheap ideas worth a look:**
   - mutated-set augmentation in self-play (Nessie: 50 % of games);
   - an imitation warm start before PPO (Wolfe_Alt6767, RotomDotExe);
   - auxiliary opponent-action and opponent-set heads (Jaxcalibur). We have belief heads; the
     opponent-action head may be the new part.

## 6. Open questions

- Who holds #1 on Reg M-C NOW, and are bots still on it after the registry? I did not read the ladder
  (FINDING F1).
- Did the opt-out toggle reach ladder matchmaking, and are bot games excluded from usage stats? There
  is no PS-side primary source (F3).
- mikumiku37's search depth and its Elo gain over the raw policy were not stated in the thread as
  summarised.
- Is there any Gen 3 OU (ADV) bot activity at all? None turned up. Metamon's Gen 1–4 work is still the
  only competitor in our era.
- Nothing above is peer-reviewed except Metamon, VGC-Bench, PokéChamp and PokéLLMon. Every bot result is
  a forum self-report.

## Findings (hazards and unverified claims)

- **F1.** Every ladder rating and rank is the author's self-report. The live ladder was not consulted
  (the brief forbids connecting to Showdown).
- **F2.** x.com returned HTTP 402 to the fetcher. The X posts' dates and text come from search snippets
  and from decoding the snowflake post ID (about 2026-10-01).
- **F3.** The PS-side implementation of the bot policy (the toggle's scope, the registry) is known only
  from the Smogon thread, not from PS's code or news.
- **F4.** Our `src/main/ladder_guard.py` does not require PS bot registration. Any future
  owner-approved ladder session must register the account first, or it violates PS policy.
- **F5.** The Smogon thread contents were read through an LLM page summariser, so quotes may be
  paraphrased. One summary misspelled mikumiku37 as "mikumiku39"; I corrected it here.
