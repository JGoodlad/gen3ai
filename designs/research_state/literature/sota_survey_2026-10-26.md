# How we compare with the state of the art, and what it does for unfamiliar teams that we don't (as of 2026-10-26)

> A dated literature survey, from web research on 2026-10-25/26, a read of the local Metamon checkout, and our own
> design docs and ledger. It is a SNAPSHOT and does not get updated; a later survey gets a new dated file. The
> always-current views are `../UNDERSTANDING.md`, `../../endstate/design_beat_metamon_path.md` and
> `../../endstate/search_ledger.md`. It builds on the two earlier surveys in this folder
> ([`showdown_bots_landscape_2026-10-18.md`](showdown_bots_landscape_2026-10-18.md) for the 2026 ladder bots and
> [`hidden_info_search_review_2026-10-20.md`](hidden_info_search_review_2026-10-20.md) for search under hidden
> information) and does not repeat them.
>
> **Evidence tags.** **VERIFIED** = read in the primary source (the paper's arXiv abstract or text, the author's own
> repository, or the Metamon code at local commit `0a00a759`), with a pointer. **UNVERIFIED** = from a secondary
> source, a search snippet, or not found. **MY JUDGEMENT** = an inference no source states. Every number about OUR
> stack names the doc or measurement that owns it; this file holds no measurement of its own. Bot ladder numbers
> are their authors' self-reports.
>
> **Abbreviations.** pp = percentage points. GXE = Showdown's "expected win rate against a random ladder player".
> BC = behaviour cloning (supervised imitation of logged decisions). IL = imitation learning. Offline RL =
> reinforcement learning from a fixed dataset, no new play. UED = unsupervised environment design (a generator
> that picks training environments). PLR = prioritised level replay. P\* = X26's plateau checkpoint
> (`rb_x26_s2601` @ 280M). U = the pooled score on our three unfamiliar panels (H = `heldout72_hpfix`, F =
> `forum72`, P72 = procedural). Goal (b) = a Wilson lower bound above 0.50 on U against Kakuna.

---

## 0. The answer, in plain language

**Where we stand.** On teams we trained on, our plateau policy beats Metamon's best public model, Kakuna, 0.598
[0.549, 0.644]. On teams we never trained on it scores 0.487 [0.459, 0.515], about 4 pp short of goal (b)
(`design_beat_metamon_path.md` §0, X51). Kakuna in turn plays Foul Play, the strongest search bot in the field, level
(0.490 [0.422, 0.559], ledger 2026-09-29 X22 (e)). By a model ~70 times smaller (2.03M parameters against Kakuna's
142.8M), trained only by self-play, we are at the state of the art on familiar teams. We are behind it on unfamiliar ones.

**What the strongest agents do for unfamiliar teams that we don't.** One thing stands out, and the rest is
secondary:

1. **They train on one to two orders of magnitude more distinct teams.**
   - Metamon's self-play draws a team at random each episode from sets of 11k procedurally generated teams
     (`paper_variety`) and 107k replay-reconstructed Gen 3 teams (`gl_05_26`). Its README says "almost all
     improvement in metamon's performance is driven by large and diverse datasets of agent vs. agent battles".
   - Every random-battle bot (Wang 2024, Jaxcalibur, Laplace) gets a fresh team every game by construction.
   - The general reinforcement-learning result is the same: Procgen's agents "strongly overfit to small training
     sets", and "to close the generalization gap, agents need access to as many as 10,000 levels".
   - We train on 719 teams. D1 added 1,981 forum teams at the plateau, for about 120 piloting games per team.
2. **They start from human behaviour.** Metamon (offline RL from 475k–5.3M reconstructed human battles), AlphaStar
   (supervised start plus a KL penalty toward the human policy), VGC-Bench (behaviour-cloning starts beat
   from-scratch ones, also on unseen teams) and two of the 2026 VGC bots all do. That lever is already on our shelf
   (X59 / X60), so it is not new here.
3. **They use much bigger networks.** Procgen also found that "larger models significantly improve both sample
   efficiency and generalization". Our evidence on capacity is a NULL from GROWING a plateaued toy (X47); a wide
   network trained from scratch is untested.

**What the field does NOT offer for this gap** (MY JUDGEMENT, with evidence):
- Leagues and population methods: VGC-Bench's game-theoretic methods do not fix team generalisation, and our own
  archive (A1) and fresher-pool (SPRT) reads are NULL.
- Language models: the PokéAgent organisers report that "RL agents far outpace harness LLM agents".
- Search: it wins where the opponent's team is nearly free to know. Gen 3 OU has no team preview, and our search is
  paused at ≈ 0 for that reason (`search_ledger.md`). Search does not touch PILOTING an unfamiliar team of our own,
  which is where the deficit lives.

**The ranked NEW levers (§5):**

| rank | lever |
|---|---|
| N1 | Procgen-scale team breadth from step 0 (≥ 10⁴ distinct teams), on the toy first |
| N2 | a noise or bottleneck regulariser on the own-team set channel |
| N3 | a regret-scored team curriculum (prioritised level replay / ACCEL over teams) |
| N4 | a WIDER network trained from scratch |
| N5 | team breadth through an offline, Metamon-style agent-vs-agent corpus |

N1 has the largest expected value, and its first test costs ≈ 13 GPU-h on the toy.

---

## 1. Our position, in the numbers each comparison needs

| quantity | value | owner |
|---|---|---|
| familiar teams vs Kakuna (home72, symmetric) | P\* **0.598** [0.549, 0.644] | X51; `design_beat_metamon_path.md` §3 |
| unfamiliar teams vs Kakuna (U, pooled) | P\* **0.487** [0.459, 0.515]; X26 @480M 0.4875 (flat) | X51; X26c |
| where the deficit lives | PILOTING an unfamiliar team, +0.123 [+0.039, +0.206] strength-corrected; FACING one costs +0.017 (n.d.); off-meta (FAR) teams cost most | X37a stage 1; TSR |
| the deficit is about teams unseen by US | forum72 and P72 drop as far as the replay panel (UNSEEN-DRIVEN) | X37a stage 2 |
| in-house meters see it? | no: in-house P\* pilots unseen teams about as well as their strength predicts | X37b |
| deployment-shaped headline | L = 0.630 [0.573, 0.686] (we bring our teams, Kakuna brings H) | X37a stage 1 |
| widening at the plateau (D1, 1,981 forum teams, 0.5 share, 20M) | U +2.10 pp [−0.92, +5.11], UNDECIDED; in-pool −1.61 [−2.90, −0.32] | X35 D1 look 2 |
| practising 36 teams for 5M (X52) | practised +3.2 [−0.2, +6.8], unpractised +1.3 [−3.5, +6.1], UNDECIDED | X52 |
| archive pool (A1) / fresher pool (SPRT) | Δ +0.19 / −0.25 pp, both NULL at look 1 | §4.10, §4.11 of `UNDERSTANDING.md` |
| a fresh best responder's headroom vs P\* | +11.1 pp [+8.3, +13.8], SATURATED | R0-prod (`UNDERSTANDING.md` §6) |
| toy depth growth 2 → 4 at the plateau (X47) | −0.88 pp [−3.20, +1.44], UNDECIDED, Δ̂ ≤ 0 | `UNDERSTANDING.md` §4.9 |
| honest one-ply search over the policy | ≈ 0; only the TRAINING POOL's team identification pays (+0.038) | `search_ledger.md` stage 1, 1c |
| human opponent model as search's judge | HOM-NONE (closed for search at P\*) | R-b2 |
| network size | production 2,031,460 parameters; the T-ENT toy 1.42M | `ARCHITECTURE.md` (2026-10-12); `UNDERSTANDING.md` TL;DR 22 |
| training teams | 719 pool teams; D1's arm adds 1,981 forum teams | `design_beat_metamon_path.md` §4 |
| piloting dose per pool team by P\* | ≈ 9,300 games (ESTIMATE); a D1 forum team ≈ 120 | the same §4 |
| learner throughput | ≈ 1,333 steps/s production (ESTIMATE); the toy ≈ 3.3× | `design_beat_metamon_path.md` §5; `UNDERSTANDING.md` TL;DR 22 |
| Foul Play vs us | 0.388 [0.288, 0.497] for us at 75M (PRE-ERA, n = 80); never read at P\* | ledger 2026-09-14 *FOUL PLAY DE-RISKED* |

---

## 2. The table: source × method, scale, generalisation mechanism, evaluation, transferable lever, our status

"Generalisation mechanism" means what, in that system, lets it play a team or situation it has not trained on.

| source | method | scale | generalisation mechanism | evaluation | transferable lever | our status |
|---|---|---|---|---|---|---|
| **Metamon** (Grigsby et al., RLC 2025, arXiv 2504.04395; repo `UT-Austin-RPL/metamon`) | first-person reconstruction of spectator replays; a turn-encoder transformer feeding a causal transformer over the battle; IL, then offline actor-critic (filtered / advantage-weighted BC + a Q term), then large self-play corpora | paper: 15M / 50M / 200M parameters, 475k battles (Gens 1–4), 8× A5000, ≥ 1M gradient steps; repo: 5.3M human trajectories + 22M self-play; Kakuna 142.8M | **team breadth in the DATA**: self-play teams drawn per episode from 11k procedural "variety" teams and later 107k replay-reconstructed Gen 3 teams (`gl_05_26`); the human prior | human ladder (GXE; Gen 3 ≈ 63 % for Kakuna on `competitive` teams); head-to-head vs Foul Play (a draw in Gens 3–4) and PokéLLMon; heuristic bots; NO systematic held-out-team test | N1 (team breadth at scale), N5 (an offline breadth corpus); the human prior = X59 / X60 | 719 + 1,981 teams; no human decisions in training; Kakuna is our external meter |
| **PokéAgent Challenge** (Karten et al., NeurIPS 2025 competition; arXiv 2603.15563) | competition; Gen 1 OU champion PA-Agent = Metamon's 200M model with 6 rounds of iterative offline RL, human share annealed 100 % → 10 %; Gen 9 OU champion Foul Play (MCTS) | 100+ teams, 100k+ battles; a released 200k+ team dataset | PA-Agent: none for teams; it SELECTED its deployment teams by a tournament ("> 60 % win rates") | best-of-99 single-elimination | annealing human data down across self-play rounds (part of X60's design space) | goal (a)-style team selection is irrelevant to (b) |
| **Foul Play** (pmariglia; described in the PokéAgent appendix) | root-parallel MCTS with decoupled UCB on the Rust poke-engine; damage-roll grouping; 10+ turn depth on promising lines | CPU, wall-clock budget | **set inference by mechanics** (damage reveals stats, priority ordering bounds speed, weather duration reveals items) over a set database from Smogon usage, forum builds and replays | ladder self-report: Gen 9 OU peak 1879, over 80 % GXE; randbats 2341 | none for PILOTING: inference covers the opponent's team, and facing costs us ≈ 0 | our determinizer already enforces damage / speed / item facts (`design_world_sampler.md`); Foul Play beat our pre-era 75M policy; unread at P\* (FINDING F4) |
| **PokéChamp** (Karten, Nguyen, Jin; arXiv 2503.04094) | minimax tree search with an LLM as action sampler, opponent model and value | GPT-4o or Llama 3.1 8B, no LLM training | the LLM's general Pokémon knowledge | Gen 9 OU: 76 % vs PokéLLMon, 84 % vs the strongest rule-based bot; projected Elo 1300–1500 | none at our compute | not pursued; the PokéAgent organisers report that RL and search beat LLM agents |
| **PokéLLMon** (Hu, Huang, Liu; arXiv 2402.01118) | an LLM with in-context feedback, retrieval of game knowledge, consistent action generation | GPT-class API | retrieval of game knowledge | 49 % on the ladder, 56 % in invited battles | none | not pursued |
| **Wang 2024** (MIT MEng thesis; `designs/references/wang2024_pokemon_rl.pdf`) | PPO self-play + MCTS at play time with the network as opponent model | 150M steps, 4 days on one A6000 + 80 CPU workers (memory note) | **a fresh random team every game** (Gen 4 random battles); search samples hidden sets from Showdown's own generator | ladder rank 8, 1693 Elo, 79.5 % GXE (VERIFIED, thesis p. 31); MCTS + NN vs NN 0.809, which the author calls "somewhat inflated" | breadth by construction (N1); search with a known generator (not available in OU) | our search pays only with the training pool's team identification (stage 1) |
| **2026 VGC and randbats bots** (the 10-18 survey) | PPO self-play vs past selves; Rust simulators; one-turn regret-matching search | 7–9M transformers; ≈ 330M games in 48 h on one RTX 5090 (mikumiku37) | randbats: fresh team per game; VGC: piloting a FEW teams (best 10 of 1,260); Nessie123: 50 % of games on MUTATED sets | ladder #1 self-reports | set mutation (= our D3 / X41); an imitation warm start | D3 BUILT, not launched; warm start = X60 |
| **VGC-Bench** (Angliss et al., AAMAS 2026, arXiv 2506.10326) | BC, self-play, fictitious play, double oracle, and BC-initialised variants | 700k+ human logs; 5,013,504 steps per run, 5 seeds | **the training team COUNT**: 1, 4, 16, 64 nested sets | on 72 unseen teams, the 64-team BC + self-play agent beats the 16-team BC + double-oracle agent 0.564; per-team strength falls as the count rises | breadth helps unseen teams at a per-team cost; a BC start helps | our D1 trade has the same shape (U +2.1 n.d., in-pool −1.6) |
| **AlphaStar** (Vinyals et al., *Nature* 2019; verified in `design_alphastar_league_path.md` §1.1) | supervised start on human replays, a KL penalty to the supervised policy, conditioning on a human strategy statistic z; a league with exploiters | 32 TPUv3 per agent for 44 days | the human prior plus z-conditioned diversity; exploiters | Grandmaster; ablations: exploiters were the largest league component | human prior (X60); z-style plan conditioning (not ranked: no Pokémon evidence) | A1 / SPRT NULL; exploiter headroom R0 +11 pp |
| **DeepNash** (Perolat et al., *Science* 2022, arXiv 2206.15378) | R-NaD: regularised Nash dynamics, model-free, no search | large | equilibrium-seeking training dynamics; no team (setup) generalisation claim | top-3 on Gravon | the trailing magnet (= LG-4 / X35-A4) | NOT BUILT (T56) |
| **ReBeL** (Brown et al., NeurIPS 2020, arXiv 2007.13544); **Student of Games** (Schmid et al., *Science Advances* 2023, arXiv 2112.03178) | search over public belief states, trained by self-play | poker-scale private state | a sound equilibrium | superhuman heads-up no-limit hold'em; strong Scotland Yard | none at 10²¹+ hidden teams (the 10-20 review §4) | outside our regime |
| **piKL / RL-DiL-piKL** (Jacob et al. 2022, arXiv 2112.07544; Bakhtin et al. 2022, arXiv 2210.05492) | search or self-play RL regularised by KL toward a human imitation policy | Diplomacy, chess, Go | the human anchor keeps play human-compatible | Diplodocus: 1st and 3rd by Elo in a 62-human, 200-game tournament | a KL toward a human policy in TRAINING (= H-KL in the human-replay plan) | the piKL anchor in SEARCH is built (L1): honest search reaches the policy, never above |
| **Procgen** (Cobbe et al., ICML 2020, arXiv 1912.01588); **CoinRun** (Cobbe et al., ICML 2019, arXiv 1812.02341) | PPO on procedurally generated levels | 200M steps per training-set size | **the number of distinct training levels**; regularisation (L2, dropout, augmentation, batch norm); bigger networks | held-out levels: overfit "to small training sets in almost all cases", the gap closes at "as many as 10,000 levels" | N1 (breadth scale), N2 (regularisers), N4 (width) | we are at 719–2.7k "levels" |
| **Selective noise injection + information bottleneck** (Igl et al., NeurIPS 2019, arXiv 1910.12911) | regularisers adapted to RL | CoinRun | noise in training only; a bottleneck on what the policy reads | held-out levels | N2 | untried |
| **PAIRED** (Dennis et al., NeurIPS 2020, arXiv 2012.02096); **PLR** (Jiang et al., ICML 2021, arXiv 2010.03934); **ACCEL** (Parker-Holder et al., ICML 2022, arXiv 2203.01302); **POET** (Wang et al. 2019, arXiv 1901.01753); **ADR** (OpenAI 2019, arXiv 1910.07113) | a generator chooses or edits training environments by regret or learning potential | grid worlds, Procgen, robotics | a curriculum at the frontier of the agent's ability | zero-shot transfer to held-out environments; PLR + others "over 76 % improvement in test return" on Procgen | N3 (a regret-scored team curriculum, team editing) | X30 (oversample worst-piloted teams) is on the backlog, not built; team-PFSP was deleted pre-era |
| **Card games with unseen cards** (Cardsformer, Hearthstone; generalised card representations, Magic: The Gathering; arXiv 2407.05879) | feature- or text-based card representations instead of ids | — | features shared by unseen cards | play with untrained cards | none: only 9 of heldout72's 72 teams carry an entity our pool never had (D0a) | our static token already reads move properties |

---

## 3. Per source: the claim, how it was measured, what transfers, what our evidence says

### 3.1 Metamon

**Claims (VERIFIED, arXiv 2504.04395 abstract and text; README at local `0a00a759`).**
- The best agents "climb to rankings inside the top 10 % of active players" (paper). The README now says "90–99th
  percentile".
- Kakuna is "the best public metamon agent": Superkazam (142M) fine-tuned on 7M more self-play battles collected
  "at higher sampling temperature for improved exploration and value estimation", with human replays down-weighted.
- Gen 3 GXE: Kakuna 63 %, SyntheticRLV2 55 % (README table, `competitive` team set).
- "TaurosEnsemble … reached #1" on Gen 1 OU, and the Kakuna router family was "the first Metamon agent to reach #1"
  (README; the format of the second claim is not stated).
- Against Foul Play: "We manage to play the best version of the bot to a draw in Gens 3 and 4" (paper §5; no Gen 3
  numbers in the text).

**How it was measured.**
- Anonymous ladder play (GXE and Glicko-1).
- Head-to-head on a local server.
- A composite heuristic-bot score.

The paper's team sets are three: "Variety" ("11k intentionally diverse teams per gen/tier procedurally generated …
to evaluate OOD gameplay and to generate unambiguous self-play data"), "Replay" and "Competitive" (10–20 sample
teams). The `competitive` set is the one Metamon says it overfit to; 19 of its 20 Gen 3 teams are in our pool
(ledger 2026-10-18).

**What it does for generalisation (VERIFIED, README "Self-Play Datasets" and "Team Sets").**
- Team sets "are randomly sampled between episodes".
- Self-play corpora were "generated by the ladder self-play launcher with various team sets and model pools".
- The variety set's moves were "generated by sampling from all-time usage stats", with "unrealistic OOD lead-off
  Pokémon" (code comment, `metamon/env/wrappers.py:44`).
- `gl_05_26` holds 107k Gen 3 teams "predicted from time- and rating-appropriate usage stats"; `hl_05_26` (22k)
  is its high-ladder subset. Our panel H is drawn from `hl_05_26`.
- The paper's stated reason for unrealistic teams: diversity "will cover replay reconstruction failures".
  **UNVERIFIED:** which team sets Kakuna's own self-play (`pac-base`, `pac-exploratory`) used (FINDING F3).

**Limits it states (VERIFIED).**
- "our agents … can begin to make nonsensical decisions in long battles — particularly when the opponent is playing
  with a rare team or uncommon strategy".
- A self-play-fine-tuned variant showed "inconsistent improvement against real players".
- The paper reports no evaluation on teams outside its training data. Its generalisation is ASSUMED from breadth,
  not measured as ours is.

**What transfers.**
- Team breadth at the scale of 10⁴–10⁵ teams (N1).
- Breadth carried by a large stored corpus rather than by on-policy steps (N5).
- Higher-temperature self-play for value estimates of sub-optimal actions. Our analogue is the starvation finding
  (`UNDERSTANDING.md` §4.4). Not ranked: our learner is on-policy PPO, so a behaviour temperature needs an
  importance correction.
- Its scale (142.8M parameters) contrasts with our 2.03M (N4).

**What our evidence says.**
- We beat Kakuna on familiar teams and lose on unfamiliar ones (§1).
- The two most direct breadth reads are small and unresolved: D1 +2.1 pp, X52 unpractised +1.3 pp.
- Both added breadth AT A PLATEAU, at a few percent of the pool's per-team dose. Neither added breadth on the
  Procgen scale or from step 0.

### 3.2 PokéAgent Challenge (NeurIPS 2025)

**Claims (VERIFIED, arXiv 2603.15563, results and Appendix E; the appendix entries are the participants' own
write-ups).**
- Gen 1 OU champion PA-Agent beat 4thLesson 50–28 in the final; Gen 9 OU champion Foul Play beat "Q" 50–14.
- "All top-performing submissions employed RL or search-based methods rather than pure LLM approaches."
- PA-Agent ran Metamon's 200M model through 6 rounds of inter-model battles, "gradually reducing human data
  proportion from 100 % to 10 %". It chose deployment teams by "a tournament selection mechanism" (> 60 % win rates).
- Team Q (Gen 9 finalist, 50M parameters) reported "the most dramatic performance jumps when transitioning from
  the v0 team set (random/heuristic initialization) to the v1 team set (curriculum-based)".

**How it was measured.** A best-of-99 bracket after qualifying ladders. GXE figures are self-reported.

**What transfers.** Annealing the human share across self-play rounds is a schedule for X60, not a new lever.
Team selection serves a deployment goal like (a), not (b). Team Q's team-set jump is anecdotal (one entry, no
ablation; MY JUDGEMENT).

**Our status.** Gen 3 was not a challenge format. Metamon's champion lineage is the one we already meter against.

### 3.3 Foul Play and EPOké

**Claims (VERIFIED, PokéAgent Appendix E; the Foul Play README states no strength).**
- Root-parallel MCTS with the decoupled UCB rule, "damage roll grouping", and "10+ turn depth on promising lines".
- Hidden information is inferred through mechanics: "damage calculations reveal stat distributions, move priority
  ordering constrains speed ranges, extended weather duration indicates specific items".
- It keeps "a comprehensive dataset of possible sets for each species, sourced from Smogon usage statistics,
  scraped team builds from forums, and public replay data".
- Self-reported peaks: Gen 9 OU 1879 (over 80 % GXE), randbats 2341.

**EPOké** (`pkmn/EPOke`, work in progress) and its user **0 ERROR** (`pkmn/0-ERROR`, no results published) do the
same inference: reverse damage calculation, speed ranges, usage statistics.

**What transfers.** For goal (b), nothing. Foul Play's machinery infers the OPPONENT's team, and facing an unfamiliar
team costs us ≈ 0 (FACE_c +0.017, X37a). Our determinizer already enforces damage, speed and item facts
(`design_world_sampler.md` §3).

**Our evidence.**
- Foul Play beat our pre-era 75M policy 0.612 from its side (n = 80).
- Kakuna plays it level on home72.
- It has never been read against P\* (FINDING F4).

### 3.4 PokéChamp and PokéLLMon

**Claims (VERIFIED, both abstracts).**
- PokéChamp (GPT-4o) wins 76 % against the best LLM bot and 84 % against the strongest rule-based bot in Gen 9 OU,
  with a PROJECTED Elo of 1300–1500.
- PokéLLMon wins 49 % in ladder games and 56 % in invited battles.
- **UNVERIFIED here:** PokéChamp's ICML 2025 venue and PokéLLMon's format (Gen 8 random battles), both taken from
  the 10-18 survey.

**What transfers.** Nothing at our compute. The organisers' own benchmark places these agents below the RL and search
entries (§3.2).

### 3.5 Wang 2024

**Claims (VERIFIED, thesis PDF pp. 30–31).**
- Rank 8 on gen4randombattles (1693 Elo, 1756 ± 28 Glicko-1, 79.5 % GXE).
- MCTS + NN beats NN alone 0.809. The author adds that he believes this "to be somewhat 'inflated'", because the
  search models its opponent as exactly that network.

**Its generalisation is structural.** Random battles hand each side a fresh generated team every game, so there is
no team to memorise. Its search samples hidden sets from the format's own generator, a belief that is right by
construction (the 10-20 review §0 point 2).

**What transfers.** The breadth argument (N1). Gen 3 OU has no generator to sample from.

### 3.6 VGC-Bench

**Claims (VERIFIED, arXiv 2506.10326 abstract and §5, Tables 3–4).**
- Teams were nested sets of 1, 4, 16 and 64.
- On 72 UNSEEN teams (Table 4; 1,000 games a cell, ±0.03) the 64-team agent beats the 16-team agent 0.564, the
  16-team agent beats the 4-team one 0.547, the 4-team agent beats the 1-team one 0.595, and 64 beats 1 by 0.669.
- On SEEN teams the order reverses: the smaller set wins (0.699, 0.594, 0.644 from the smaller set's side).
- "As the size of the set of teams used during training increases, the performance … on any one team decreases
  considerably and consistently".
- BC-initialised methods "generally outperform" the others.

**How it was measured.** 5,013,504 steps per run, 5 seeds, head-to-head on seen (Table 3) and on unseen (Table 4)
teams, win rates relative to the row player.

**What transfers.** The shape of the trade, and that breadth DOES buy unseen-team play.

**Our evidence.**
- Our D1 has the same shape at 1.5 orders more teams: U +2.1 n.d., in-pool −1.6.
- VGC-Bench's budget is 1–2 % of ours. Whether its slope persists past ≈ 10³ teams is open (MY JUDGEMENT).

### 3.7 AlphaStar, DeepNash, ReBeL, Student of Games

These were checked at source by the league and search docs.
- **AlphaStar** (`design_alphastar_league_path.md` §1.1): a supervised start, a KL toward the human policy, the human
  statistic z, exploiters as the largest league gain.
- **DeepNash** and **ReBeL / Student of Games** (the 10-20 review §4–§7): DeepNash's R-NaD is the trailing magnet,
  ReBeL's belief-state search is outside our regime.

Here they matter only for generalisation, and none of them addresses varying the PLAYER'S OWN loadout. DeepNash
deploys varied Stratego setups; AlphaStar has three races and a small set of maps. **Our evidence:** the league
levers are NULL at the plateau (A1, SPRT), and an exploiter still finds +11 pp (R0). That is exploitability, a
different axis from goal (b) (`design_beat_metamon_path.md` §5 ranks X35 +0 to +2 for (b)).

### 3.8 piKL and RL-DiL-piKL (Cicero's planning core)

**Claims (VERIFIED, abstracts).**
- KL-regularised search toward an imitation policy gives "higher human prediction accuracy and stronger performance
  than imitation learning alone".
- RL-DiL-piKL's Diplodocus ranked first and third by Elo in a 200-game, 62-human tournament.
- Cicero itself (*Science* 2022) was not fetched (FINDING F7).

**What transfers.** A human-anchored TRAINING objective (H-KL in `design_human_replay_lever.md`, ranked last there).
A human anchor is an anchor to policies that have piloted many more teams than we have: ≥ 1500 ladder players bring
a team distribution our pool does not cover (15.9 % of their full-six sides lie in our panels' region; the data
card).

**Our evidence.** The piKL anchor in SEARCH is measured (L1): it brings honest search up to the policy, never above.
The human model (HOM) predicts humans 0.58 nats better than P\* but does not help search (R-b2).

### 3.9 Unsupervised environment design and procedural generation

**Claims (VERIFIED, Procgen §3.1 and §4; CoinRun, Igl et al., PAIRED, PLR, ACCEL, POET and ADR abstracts).**
- Agents trained for 200M steps on 100 to 100,000 levels "strongly overfit to small training sets in almost all
  cases"; "to close the generalization gap, agents need access to as many as 10,000 levels".
- Past a threshold, TRAINING performance also rises with set size: "the implicit curriculum provided by the
  distribution of levels".
- Wider IMPALA networks (×2, ×4 channels) improve both training efficiency and generalisation.
- CoinRun: L2, dropout, augmentation and batch norm all improve generalisation.
- Igl et al.: noise injected in training only, plus an information bottleneck, improve on CoinRun.
- PLR replays levels by their temporal-difference error ("learning potential"); with other methods it reports
  "over 76 % improvement in test return".
- PAIRED and ACCEL pick or EDIT environments by regret, starting simple and compounding complexity.

**What transfers.** A team is our "level".
- The size of the team set (N1).
- Regularisers on the team channel (N2).
- A curriculum that replays and edits teams by learning potential (N3).
- Width (N4).

**Our evidence.**
- Breadth: D1 and X52 (above).
- Augmentation: set mutation (D3 / X41) is BUILT, not run. Its registered guess is +1.5 pp.
- Curricula: X30 is a backlog row, not built. Team-PFSP was deleted pre-era (`deleted_flags.md`, probe P).
- Regularisers and width: never tried.

---

## 4. What the strongest agents do that we don't, and how sure we are

| pattern | who does it | does it bear on unfamiliar-team PILOTING? | our status |
|---|---|---|---|
| **10⁴–10⁵ distinct training teams, or a fresh team per game** | Metamon, every randbats bot, Wang; Procgen's threshold | YES, directly: Procgen's held-out gap closes only at ≈ 10⁴ levels | 719 (+ 1,981 in D1). **The biggest gap between us and the field** |
| a human prior (BC start, offline RL from replays, KL to humans) | Metamon, AlphaStar, VGC-Bench, Wolfe_Alt6767, RotomDotExe | probably: humans have piloted far more teams than our pool holds (MY JUDGEMENT); VGC-Bench's BC starts generalise better | X59 (CPU) next, X60 after review; owner's grant 2026-10-25 |
| a large network | Metamon 15–200M; Procgen's wider networks | plausibly; untested on our side | 2.03M; X47 depth growth at the plateau UNDECIDED (Δ̂ ≤ 0) |
| mutated or augmented sets | Nessie123 (50 % of games); CoinRun's augmentation | yes, for the CORE stratum (familiar species, unfamiliar sets ≈ 11 pp) | D3 / X41 BUILT, awaiting the GO |
| test-time ensembles | Metamon's routers (first #1) | weakly: it averages members that all trained on the same teams | ranked low (+0 to +2 GUESS) |
| deep search with mechanics-based set inference | Foul Play, 0 ERROR | no: it serves FACING, which costs us ≈ 0 | search paused at ≈ 0 for Gen 3 OU |
| selected deployment teams | PA-Agent, mikumiku37, Nessie123 | no: it is goal (a)'s regime | L = 0.630 already |

**The asymmetry behind goal (b)** (MY JUDGEMENT, partly UNVERIFIED). Kakuna's training distribution almost
certainly overlaps our "unfamiliar" panels. H was carved from Metamon's own `hl_05_26`. P72-style procedural teams
resemble Metamon's `paper_variety`. Forum teams resemble its scraped `competitive` and replay sets. So the symmetric
U cells pit our unseen against Kakuna's probably-seen. X37a's assumption A-K (Kakuna's piloting and facing
familiarity are equal) is still UNVERIFIED. The third-source stage (X37a stage 2) killed "Kakuna's home turf" as
the CAUSE of OUR drop, but not as a component of Kakuna's own strength on those panels. Either way, goal (b) asks us
to match a model trained on 10⁴–10⁵ teams while we train on 10³. The literature predicts exactly that gap.

---

## 5. The NEW levers, ranked

"Expected value" is pp on U against Kakuna over P\* (or over the matched control), toward the ≈ +4.3 pp goal (b)
needs. Every such number is a **GUESS**. Every cost is an **ESTIMATE** from the throughput figures in §1. GPU-h
means production GPU-hours unless "toy" is written. A lever already on our shelf is not listed. Those are human BC
(X60), set mutation (D3), the trailing magnet (LG-4), ensembles, and search.

### N1. Procgen-scale team breadth from step 0 (≥ 10⁴ distinct teams), on the toy first

- **What it is.**
  - A fresh toy run whose self-play draws each side's team per episode from a corpus of ≥ 10⁴ distinct teams.
  - The corpus mixes procedural Smogon-usage teams (`src/rust_sim/harness/ou_random_teams.js`, seeded), forum teams,
    and Hidden-Power-repaired replay teams. It excludes the sealed and registered panels by
    `heldout_panels.json`'s ≥ 3-fit rule.
  - The pool is kept at a minority share (e.g. 0.3).
- **Why it is new.**
  - D1 widened at the PLATEAU with 1,981 teams.
  - X53 (conditional) is pool + forum (≈ 2.7k teams) from step 0.
  - The diversity doc's D3 option (c), "whole procedural teams", was a 0.3-probability augmentation of the pool.
  - None reaches the 10⁴ threshold where Procgen's gap closes. Metamon's per-episode draw from 10⁴–10⁵ teams has no
    counterpart here.
- **Evidence for it.**
  - Procgen §3.1, with the gap closing at ≈ 10⁴ levels.
  - Metamon's README: improvement "driven by large and diverse datasets".
  - VGC-Bench's monotone unseen-team gain from 1 to 64 teams.
  - Our own familiarity premium (≈ 11 pp) with the dose arithmetic: P\* gave each pool team ≈ 9,300 games.
- **Evidence against it.**
  - D1's +2.1 pp at a 0.5 share.
  - X52's unpractised +1.3 pp.
  - Procedural teams are weak (P72 rates −146 Elo, TSR), so the policy may learn to pilot bad teams.
  - X37b says nearly all in-house gain came before 100M. That argues FOR reading at 100M.
- **Expected value (GUESS).** +3 to +8 pp on U. In-pool strength −1 to −4 pp (VGC-Bench's per-team cost).
- **Cost (ESTIMATE).**
  - Corpus build ≈ 1 agent-day (generator, validator pass, HP repair, panel exclusion, a manifest).
  - Two toy arms to 100M ≈ 2 × 6.3 GPU-h (toy ≈ 4,400 steps/s).
  - Kakuna reads ≈ 2 × (1,200 U + 400 home72 games) ≈ 9 CPU lane-h.
- **Risks.**
  - Toy-to-production transfer is not calibrated: the toy's own familiarity premium is about half P\*'s on the
    logit scale (X51), which lowers the test's sensitivity.
  - Corpus coherence (legality, Species and Sleep Clause).
  - One seed per arm.
- **The falsifiable first test.**
  - Arm W (the ≥ 10⁴-team corpus, pool share 0.3) against arm C (pool only), same seed family, both fresh T-ENT to
    100M at the toy's production pin.
  - Primary: W − C on U, the symmetric Kakuna cells, n = 1,200 per arm, paired seeds.
  - **GO** (scale to a production X53-wide) if W − C ≥ +4 pp with its lower bound above 0.
  - **KILL** breadth-at-scale on the toy if the upper bound is below +2 pp.
  - Secondary: the familiarity premium (home72 − U) and the REVERSE cell (we pilot H against Kakuna on home72),
    the cleanest piloting meter.
  - Register before launch. The in-house meter cannot steer this (X37b).

### N2. A noise or bottleneck regulariser on the own-team set channel

- **What it is.**
  - In training only, with probability p per own mon per episode, the static token's SET fields (moves, item,
    spread) are replaced by a Smogon-usage draw for the species or masked to the species prior.
  - Evaluation is deterministic and noise-free (Igl et al.'s rule).
  - The idea: a policy that cannot key on the exact set cannot memorise a team by its set signature. It must pilot
    from what the species and the battle reveal.
- **Why it is new.** D3 changes the TEAM that is played (the battle is fought with the mutated set). N2 changes only
  what the POLICY READS about its own team, while the battle keeps the true sets. So it attacks recognition, not
  exposure. No regulariser of this kind (dropout, noise injection, bottleneck) has been tried on any channel here
  (MY JUDGEMENT, from the flag registry's absence of one; UNVERIFIED by a full search).
- **Evidence for it.**
  - CoinRun: L2, dropout, augmentation and batch norm all help.
  - Igl et al.: noise injection plus a bottleneck.
  - Ours: the belief heads "memorise the pool" (`UNDERSTANDING.md`, the era close-out), and the CORE-stratum deficit
    (familiar species, unfamiliar sets) is ≈ 11 pp (X37a).
- **Evidence against it.** Our own exact sets ARE information a real player has. Removing them costs in-pool play by
  construction.
- **Expected value (GUESS).** +1 to +3 pp.
- **Cost (ESTIMATE).**
  - The probe below ≈ 0.5 agent-day + a few CPU-h.
  - The flag ≈ 1 agent-day (a training-only column mask at token assembly, behind a flag, with a gate that evaluation
    is bit-identical with it off).
  - A P\* + 20M fork in the X35 frame ≈ 12 GPU-h, read by D1's rule against the shared C0.
- **Risks.**
  - Train / evaluation mismatch (the reason SNI exists).
  - Interaction with D3 if both run.
- **The falsifiable first test (zero GPU).** The EXACT-SET SENSITIVITY probe.
  - On banked P\* states, swap one own mon's set for a usage draw of the same species (D3's generator).
  - Measure the policy KL and the argmax flip rate, separately on pool-team states and on unfamiliar-team states.
  - **Prediction under set-keyed memorisation:** pool-team KL ≥ 1.5× unfamiliar-team KL.
  - **KILL** N2 (and weaken D3's registered mechanism) if the ratio's interval covers 1.0 and excludes 1.5.

### N3. A regret-scored team curriculum (prioritised level replay, then ACCEL-style edits)

- **What it is.**
  - Each team in the training corpus carries a running learning-potential score: PLR's mean positive
    temporal-difference error, or a regret proxy such as the gap between the team's in-house score and its
    Kakuna-free strength rating.
  - The builder replays high-score teams more often.
  - ACCEL's step: it EDITS high-score teams (mutating 1–2 sets) to make new corpus entries.
- **Why it is new.**
  - X30 (backlog, not built) oversamples the teams we pilot WORST, by win rate. PLR's score is learning potential,
    not difficulty, and ACCEL adds generation.
  - Team-PFSP (pre-era, deleted) weighted by win rate too.
- **Evidence for it.**
  - PLR on Procgen ("over 76 % improvement in test return" combined with other methods).
  - PAIRED / ACCEL's zero-shot transfer.
- **Evidence against it.** X37b found in-house meters blind to the piloting deficit. A curriculum scored in-house may
  chase the wrong teams. That is exactly what the first test checks.
- **Expected value (GUESS).** +0 to +3 pp. Larger only on top of N1, since a curriculum needs a corpus to choose from.
- **Cost (ESTIMATE).**
  - The probe ≈ 0.5 agent-day.
  - A prioritised team sampler on the Rust-core builder ≈ 1.5–2 agent-days.
  - A fork ≈ 12 GPU-h.
- **Risks.** It concentrates practice, the opposite of breadth, if the score is noisy (D1's per-team dose is already
  ≈ 120 games).
- **The falsifiable first test (zero GPU).**
  - On P\*'s banked traces, compute each team's mean positive advantage (or value loss).
  - Correlate it (Spearman) with the per-team reverse-cell Kakuna deficit from X37a and the TSR per-team piloting gap.
  - **KILL** if ρ < 0.2 with its interval covering 0: the in-house signal is blind to the deficit, and PLR has
    nothing to steer by.

### N4. A WIDER network, trained from scratch

- **What it is.** The toy (or production) at 2× trunk width, fresh, read at matched WALL CLOCK against the standard
  width.
- **Why it is new.**
  - X47 GREW a plateaued toy's depth (UNDECIDED, Δ̂ ≤ 0).
  - X47b / X48 are growth forks.
  - The T-ENT result (a smaller toy beats production at equal compute at 15M) is about a SMALL budget.
  - Width from scratch at a large budget is untested.
- **Evidence for it.**
  - Procgen §4: "larger models significantly improve both sample efficiency and generalization".
  - CoinRun: deeper networks generalise better.
  - Metamon trains 15M / 50M / 200M models and its later best models are 57–142M (paper §5, README). That larger
    is better there is UNVERIFIED here: I did not read its size comparison figure.
  - Ours: 100M more steps on the same network moved nothing (X26c), consistent with a capacity limit as well as a
    data limit.
- **Evidence against it.**
  - The learner is GPU-bound, so width costs steps per hour.
  - X47's growth NULL.
  - At equal compute a bigger network may be SLOWER to reach any level.
- **Expected value (GUESS).** +0 to +3 pp alone. Possibly more as a partner of N1: Procgen's two effects compound
  (MY JUDGEMENT).
- **Cost (ESTIMATE).** ≈ 10–15 toy GPU-h for one wide arm to the matched wall clock. It is cheapest as a third arm
  beside N1's two.
- **Risk.** Wide arms change throughput and compile behaviour (bf16 trunk, the compiled learner).
- **The falsifiable first test.** A third toy arm, W-wide (N1's corpus at 2× width), read on U at the wall clock of
  arm W's 100M.
  - **GO** if W-wide − W ≥ +3 pp with its lower bound above 0.
  - **KILL** width as a generalisation lever at this scale if the upper bound is below +1 pp.

### N5. Team breadth through an offline, Metamon-style agent-vs-agent corpus

- **What it is.**
  - Generate a large CPU self-play corpus with P\* and its archive as players, on ≥ 10⁴ teams (N1's corpus).
  - Train a student from P\* offline with Metamon's binary-advantage-filtered BC plus a critic (paper Eq. 2), on
    those trajectories.
  - Breadth then costs CPU battles, not on-policy GPU steps.
- **Why it is new.** We have never trained off-policy at scale. X59-S (the matched-student contest on human vs P\*
  labels) builds part of the student machinery for a different question.
- **Evidence for it.**
  - Metamon's whole recipe: "almost all improvement … driven by large and diverse datasets of agent vs. agent
    battles", 11M + 7M trajectories.
  - PA-Agent's six iterative offline rounds.
- **Evidence against it.**
  - A corpus played by P\* contains P\*'s own piloting mistakes on unfamiliar teams. The advantage filter is the only
    new signal, and it is noisy in a stochastic game.
  - Metamon's corpora were seeded by human data.
- **Expected value (GUESS).** +1 to +5 pp, the widest uncertainty of the five.
- **Cost (ESTIMATE).**
  - Build ≈ 3–5 agent-days (corpus writer, offline trainer, student evaluation).
  - Corpus ≈ tens of CPU lane-h (the env core runs ≈ 75 battles/s per busy core, throughput audit 2026-10-18; P\*'s
    CPU inference cost is UNVERIFIED).
  - Student training ≈ 5–10 GPU-h.
- **Risk.** High: a new training mode, and on-policy assumptions elsewhere in the stack.
- **The falsifiable first test.**
  - After X59-S lands its student machinery, train two students for an equal number of updates: one on 10⁴-team
    filtered corpus rows, one on pool-team rows.
  - Primary: their U against Kakuna.
  - **KILL** if the breadth student's U − the pool student's U has an upper bound below +2 pp.

### Not ranked, with the reason

- **AlphaStar's strategy statistic z, as a team-plan latent learned from human replays.** No Pokémon evidence, and a
  multi-day build.
- **Kakuna-style high-temperature collection.** It needs an off-policy correction in PPO. It is reachable through
  N5's corpus if N5 runs.
- **Card-game feature representations.** Unseen ENTITIES are not our gap (D0a).
- **Deeper or equilibrium search.** It serves facing, and search is paused (`search_ledger.md`).
- **LLM agents.** Beaten by RL and search in the field's own benchmark.

### The order, and how the levers interact

1. Run N2's and N3's zero-GPU probes first (≈ 1 agent-day together). They can each kill a lever before any GPU is
   spent.
2. Then run N1 with N4 as its third arm (≈ 23 toy GPU-h, ≈ 14 CPU lane-h of Kakuna reads).
3. N5 waits for X59-S's machinery.
4. D3 (built) and X60 (human data) stay where `design_beat_metamon_path.md` ranks them. N1 is complementary to both:
   D3 perturbs familiar cores, and X60 brings human piloting of human teams.

---

## 6. Where we stand (honest)

On the teams we built and trained on, our 2M-parameter self-play policy is as strong as anything public in Gen 3 OU.
It beats Metamon's 142M-parameter Kakuna 0.60. Kakuna in turn draws Foul Play, the PokéAgent Gen 9 OU champion and
the strongest search bot. That is a real result at about 1/70 of the parameters and with no human data. But "the
state of the art" in Pokémon means playing whatever team you are handed, and there we are not there: 0.487 against
Kakuna on teams we never saw. 100M more steps, an archive pool, a fresher pool, a 5M practice block and a 2.7k-team
widening have each moved that by ≤ 2 pp, none resolved. The literature's diagnosis is blunt and matches ours. Every
system that generalises across teams trained on vastly more of them: 10⁴–10⁵ for Metamon, a fresh one every game for
the random-battle bots. Procgen puts the threshold for closing a held-out gap at ≈ 10⁴ "levels". We have run every
breadth experiment at 10³ teams and at a plateau. The honest reading is that we have not yet tested the lever the
field relies on, at the scale the field uses. Our two other differences, human data and network size, are the next
most common features of the strong systems. Human data is queued (X59 / X60); size is untested. Two caveats keep this
from being a promise:
- Procgen's levels are not Pokémon teams, and VGC-Bench is the only team-count study in Pokémon.
- Goal (b)'s meter may carry Kakuna's own familiarity with our "unfamiliar" panels, which no read has removed.

Nothing here is a measured effect on our stack. Every expected value in §5 is a GUESS until N1's registered read.

---

## 7. FINDINGS (hazards, skips, and what could not be verified)

- **F1. `UNDERSTANDING.md` has no outside-view section.** The brief said to update one "if it has one"; none exists,
  so none was created (rule 9). This survey is indexed in `../README.md`'s literature list instead.
- **F2. Metamon's variety-set generator is not described in the paper.** Its procedure is VERIFIED only through one
  code comment (`metamon/env/wrappers.py:44` at local `0a00a759`): usage-sampled moves, unrealistic leads. The local
  checkout dates from 2026-05-22; upstream may have moved.
- **F3. The team sets of Kakuna's own self-play (`pac-base`, `pac-exploratory`) are UNVERIFIED** ("various team
  sets"). H comes from Metamon's `hl_05_26`, so Kakuna may have played those exact teams. Goal (b)'s symmetric cells
  may be FAMILIAR to Kakuna, and X37a's assumption A-K stays unverified.
- **F4. Foul Play has never been read against P\*.** The only read is the PRE-ERA 75M policy (n = 80). "At the state
  of the art on familiar teams" is a transitive inference through Kakuna ≈ Foul Play (on home72, at 1,000 ms per
  decision under load), not a measurement.
- **F5. The PokéAgent Appendix E entries are participants' self-descriptions** inside the organisers' paper. Their
  GXE and ladder figures are self-reported.
- **F6. Procgen was read through `pdftotext` of the arXiv PDF.** Text claims are VERIFIED; figure values were not
  read.
- **F7. Cicero (*Science* 2022) was not fetched.** RL-DiL-piKL (arXiv 2210.05492) stands in for the human-regularised
  family.
- **F8. Not re-verified here:** PokéChamp's venue (ICML 2025), PokéLLMon's format, and the 2026 bots' details. All
  come from the 10-18 survey. The conference venues of the general-RL papers (Procgen, CoinRun, PLR, ACCEL, PAIRED,
  ReBeL) are from my own knowledge, not from the fetched arXiv pages.
- **F9. Every cost in §5 is an ESTIMATE** from documented throughput (≈ 1,333 production steps/s, toy ≈ 3.3×, ≈ 360
  Kakuna games per CPU lane-hour). None was re-measured.
- **F10. Metamon reports no held-out-team evaluation.** Its generalisation across teams is inferred from breadth, not
  measured the way ours is. The Procgen analogy is MY JUDGEMENT.
- **F11. No EPOké competition exists as such.** "EPOké" is the `pkmn` set-inference library behind the 0 ERROR bot,
  neither with published results. The IEEE CoG VGC AI Competition (pokemon-vgc-engine) generates random teams per
  battle and swaps them between players. Its past winners were not found (UNVERIFIED).
- **F12. The N2 "never tried" claim rests on the absence of such a flag** in the flag registry as I read it, not on
  an exhaustive search of the ledger.

---

## 8. Sources

Primary, read 2026-10-25/26:

- Grigsby et al., "Human-Level Competitive Pokémon via Scalable Offline RL with Transformers", RLC 2025,
  https://arxiv.org/abs/2504.04395 (v2 HTML read). README and code: https://github.com/UT-Austin-RPL/metamon, local
  `/home/goodlad/dev/metamon` @ `0a00a759`.
- Karten et al., "The PokéAgent Challenge", https://arxiv.org/abs/2603.15563 (HTML, results + Appendix E).
- Foul Play, https://github.com/pmariglia/foul-play. EPOké, https://github.com/pkmn/EPOke. 0 ERROR,
  https://github.com/pkmn/0-ERROR.
- Karten, Nguyen, Jin, "PokéChamp", https://arxiv.org/abs/2503.04094. Hu, Huang, Liu, "PokeLLMon",
  https://arxiv.org/abs/2402.01118.
- Wang 2024, `designs/references/wang2024_pokemon_rl.pdf` (pp. 30–31).
- Angliss et al., "VGC-Bench", AAMAS 2026, https://arxiv.org/abs/2506.10326 (HTML §5).
- VGC AI Competition (IEEE CoG 2025), https://cog2025.inesc-id.pt/vgc/.
- Perolat et al., DeepNash, https://arxiv.org/abs/2206.15378. Brown et al., ReBeL, https://arxiv.org/abs/2007.13544.
  Schmid et al., Student of Games, https://arxiv.org/abs/2112.03178.
- Jacob et al., piKL, https://arxiv.org/abs/2112.07544. Bakhtin et al., RL-DiL-piKL,
  https://arxiv.org/abs/2210.05492.
- Cobbe et al., Procgen, https://arxiv.org/abs/1912.01588 (PDF §3.1, §4). Cobbe et al., CoinRun,
  https://arxiv.org/abs/1812.02341. Igl et al., https://arxiv.org/abs/1910.12911.
- Dennis et al., PAIRED, https://arxiv.org/abs/2012.02096. Jiang et al., PLR, https://arxiv.org/abs/2010.03934.
  Parker-Holder et al., ACCEL, https://arxiv.org/abs/2203.01302. Wang et al., POET, https://arxiv.org/abs/1901.01753.
  OpenAI, ADR, https://arxiv.org/abs/1910.07113.
- Card games with unseen cards (search results only, UNVERIFIED beyond titles): https://arxiv.org/html/2407.05879v1.

Ours, cited by section:
- `designs/endstate/design_beat_metamon_path.md` §0–§5.
- `designs/research_state/UNDERSTANDING.md` §3.1, §4.8–§4.11, §6.
- `designs/endstate/search_ledger.md`.
- `designs/endstate/human_data_architecture.md`.
- `designs/endstate/design_alphastar_league_path.md` §1.
- `designs/endstate/design_diversity_teams_opponents.md` §4.
- The two earlier surveys in this folder.
