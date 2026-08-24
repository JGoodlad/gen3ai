# Metamon, verified — and what human Showdown replays would cost US

**Written 2026-08-23.** Every claim below carries its source. Anything I could not verify against a
primary source is marked **UNVERIFIED** or **ESTIMATED**. Figure numbers read off the rendered
figure images are marked as such — I downloaded and read them rather than trusting a text summary,
because the numeric bar labels exist only inside the images.

Primary sources used:

| # | Source |
|---|---|
| S1 | arXiv **2504.04395**, *Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers*, Grigsby, Xie, Sasek, Zheng, Zhu (UT Austin). **v1 2025-04-06, v2 2025-07-30 (latest).** Venue: **Reinforcement Learning Conference 2025**. |
| S2 | Figure images from `arxiv.org/html/2504.04395v2/x11.png` (Fig. "Ladder Ratings in Online Battles vs. Humans") and `x13.png` (Fig. "Ladder Percentiles"), downloaded and read directly. |
| S3 | GitHub `UT-Austin-RPL/metamon` — README + source tree (`metamon/backend/replay_parser/`, `metamon/backend/team_prediction/`). Code license **MIT**. |
| S4 | HuggingFace `jakegrigsby/metamon-parsed-replays` (dataset card + HF API `/api/datasets/.../tree/main`). |
| S5 | HuggingFace `jakegrigsby/metamon-raw-replays` (dataset card + `datasets-server.huggingface.co/size` + `/first-rows`). |
| S6 | `https://replay.pokemonshowdown.com/search.json?format=gen3ou` — probed live 2026-08-23. |
| S7 | This repo, at `worktree-agent-a8f426ba420078642`. File paths + line numbers given inline. |

---

## Decision-relevant summary (2 minutes)

1. **They published everything.** Not just the paper — the **raw replay logs** (2,681,841 rows, with
   a full `log` column carrying the raw spectator protocol) and the **parsed first-person
   trajectories**, plus the revealed-teams record and the replay-derived team-prediction statistics.
   The parsed data is **CC-BY-NC-4.0**; the code is **MIT**. `gen3ou.tar.gz` is **2.691 GB — the
   largest gen1–4 slice in the corpus**, 2.1× gen1ou. (S4, S5)
2. **Gen3OU is their WEAKEST of gens 1–4, and it is the format where "top decile" is literally the
   number.** Best model (SynRL-V2) reaches **GXE 64%** and the **90.1st percentile** of 8,944 known
   active gen3ou players, versus **GXE 77%** and the **95.8th percentile** of 2,661 in gen1ou. Two
   appearances inside the top 300 in gen3ou; peak **#31 global** in gen1ou. (S2, S1 §5.5)
3. **The expensive half of the problem is already solved inside OUR tree, and we did not know it.**
   `src/agents/training/obs_materializer.py` is a **protocol replayer, not a state replayer** — its
   documented input is "only the per-side protocol chunks", and "the omniscient reconstruction data
   (seed, both teams, dice) never enters this module". The seed/team requirement lives one layer up,
   in `utils/bridge/reconstruction.py`, which *produces* those chunks by re-simulating. A public
   replay does not need re-simulation; it needs a **second producer of one-sided chunks**. That is a
   much smaller and much better-defined job than "build an offline obs pipeline".
4. **The one thing a spectator log structurally cannot give us is the `|request|` JSON, and our stack
   hard-fails without it** (`mask_generator.py:74` raises `STRICT MODE FAILURE: No last_request
   found`). Metamon's answer is exactly this: their parser is described in their own repo as *"a
   from-scratch python implementation of the sim protocol that **simulates requests** and predicts
   teams"* (S3). We would have to do the same. The good news: `LegalActions.from_battle`
   (`live_view.py:501`) reads a **bounded** set of request fields, so the synthesis target is small.
5. **Total honest lift: ~4–6 focused weeks to a first measured training result**; **~2–3 weeks to the
   decision point**, which is a self-validating round-trip gate we can build because we own both
   halves (see *Stage A*). Recommended first experiment is in §2.6.
6. **The single biggest risk is not noise, it is a directional confound**: we impute our own side's
   unrevealed set from priors, but the human chose *knowing* the true set — and the states where
   imputation error is largest (early battle, before reveals) are exactly the states where the label
   is most informative. Metamon does not quantify this anywhere. Our Stage A gate is the only
   instrument that can. (§2.7)

---

# Part 1 — the paper, verified

## 1.1 Provenance

- arXiv **2504.04395**. **v1 6 Apr 2025** (11.5 MB), **v2 30 Jul 2025** (15.9 MB) — v2 is the latest
  revision as of 2026-08-23. Comments field: *"Reinforcement Learning Conference 2025"*. (S1)
- Project code: `github.com/UT-Austin-RPL/metamon`, **MIT**. A personal mirror `jakegrigsby/jg-metamon`
  also exists. (S3)

## 1.2 Data

**In the paper (Sept 2024 cutoff):**

| Quantity | Value | Source |
|---|---|---|
| Battles reconstructed | **475k** | S1 §3 |
| POV trajectories (2 per battle) | **~950k** | S1 §3 |
| Timesteps | **38M** | S1 §3 |
| Generations | **1–4** | S1 §2 |
| Tiers | four (OU / UU / NU / Ubers); *"evaluations will focus on OverUsed (OU)"* | S1 §2 |
| Historical depth | *"historical Gen 1-4 battles dating back to 2014"* | S1 §3 |

Scraping: Pokémon Showdown *"creates a log ('replay') of every battle that expires after a brief
period unless saved"*, and their *"pipeline is now actively downloading new battles in the 15-minute
window before they are deleted"* (S1 §3). So the corpus is part historical-archive, part live
harvest of battles that would otherwise never have been saved — a detail that matters for us,
because it means the *public* archive is a strict subset of what Metamon holds.

**Why gens 1–4 only:** *"the size of the team space creates so much variance from Generation 5
onwards that PS adopts a mechanic called 'team preview'... For this reason, we focus on the first
four generations."* (S1 §2). Gen9 OU was added later, outside the paper. (S4 changelog v3)

**In the published dataset today (well beyond the paper):**

| Artifact | Value | Source |
|---|---|---|
| `metamon-raw-replays` rows | **2,681,841** | S5, HF `/size` API |
| Raw replay parquet bytes | 5.70 GB | S5 |
| Raw schema columns | `id, format, players, log, uploadtime, formatid, rating` — **`log` is the full raw spectator protocol** | S5 `/first-rows` |
| `metamon-parsed-replays` trajectories | **5.3M** (repo README) | S3 |
| Repo-stated self-play piles | 11M / 7M / 4M trajectories (`metamon-parsed-pile`) | S3 |

Per-format parsed tarball sizes (HF tree API, exact bytes → GB):

| format | GB | | format | GB |
|---|---|---|---|---|
| gen1ou | 1.260 | | gen3ou | **2.691** |
| gen1uu | 0.033 | | gen3uu | 0.033 |
| gen1nu | 0.022 | | gen3nu | 0.020 |
| gen1ubers | 0.045 | | gen3ubers | 0.081 |
| gen2ou | 0.723 | | gen4ou | 1.907 |
| gen2uu | 0.033 | | gen4uu | 0.081 |
| gen2nu | 0.019 | | gen4nu | 0.020 |
| gen2ubers | 0.034 | | gen4ubers | 0.132 |
| | | | gen9ou | 20.391 |
| | | | `revealed_teams.tar.gz` | 1.013 |
| | | | `replay_stats.tar.gz` | 0.004 |

**gen3ou is the largest gen1–4 slice by a wide margin.** (S4)

**Dataset format** (S4 card, verbatim structure):

```
{battle id}_{ELO rating}_{pov username}_vs_{opponent username}_{DD-MM-YYYY}_{WIN|LOSS}.json.lz4
  -> {"states": [UniversalState, ...], "actions": [...]}
```

and their loader yields, per battle:

```
actions["chosen"]  list[Any], len T  — the action label
actions["legal"]   list[set], len T  — valid actions at each step
actions["missing"] list[bool], len T — True if the action was NOT present in the replay
```

That third field is important for us: **they publish per-timestep "this label is imputed" flags**, so
a consumer can choose to mask rather than inherit their filling policy.

**Licenses — asymmetric, read carefully:**

- Code (`UT-Austin-RPL/metamon`): **MIT**. (S3)
- Parsed replays (`jakegrigsby/metamon-parsed-replays`): **`cc-by-nc-4.0`** — verified in the HF API
  `cardData.license` and in the `tags` array. **Non-commercial.** (S4)
- Raw replays (`jakegrigsby/metamon-raw-replays`): license **not stated on the card** —
  **UNVERIFIED**. The card does say the underlying battles are *"publicly available from the
  Showdown replay API"*, and usernames are anonymized pseudonyms consistent across both datasets. (S5)

**Version history** (S4 changelog): v0 = the paper's dataset (~1M replays, numpy, hardcoded obs); v1
(2025-04-25) switched to `UniversalState` JSON *and upgraded the team predictor from
`NaiveUsagePredictor` to `ReplayPredictor`*; v2 (2025-05-29) lz4 + an item/ability bug fix; v3-beta
(Jun 2025) backfilled gen9ou and **"has not yet been confirmed to replicate Gen 1-4 paper results"**;
v4 (2025-08-12); v5 (2026-01-18, subdirs → `genXou/YYYY/MM`); v6 (2026-05-19). The raw card notes a
**2021–2024 replay-database gap** caused by a Showdown infrastructure failure, with most recoverable
replays from that window being tournament rather than ladder play. (S5)

## 1.3 The replay → first-person reconstruction, in detail

This is the part the owner cares most about, so it is sourced from **code**, not prose.

**Architecture.** Their pipeline figure (`x14.png`, read directly) draws two lanes into one common
type:

```
ONLINE :  Pokémon Showdown --poke-env--> Battle --------> UniversalState
OFFLINE:  Raw Replay --ours--> Parsed Replay --> Replay State --> UniversalState
                                              (Datatypes for Tokenization <- Battle)
```

So they do **not** re-simulate a replay to recover state; they parse it into a state object that is
type-compatible with the live poke-env one. Their repo describes the parser as *"a from-scratch
python implementation of the sim protocol that **simulates requests** and predicts teams"* and its
job as converting *"from the spectator POV of raw Pokémon Showdown replays to the first-person POV of
RL agents"* (S3, `replay_parser/README.md`).

**Module inventory** (S3, GitHub tree API, sizes in bytes):

| file | size | role |
|---|---|---|
| `replay_parser/forward.py` | 62,113 | the from-scratch protocol sim (forward pass) |
| `replay_parser/replay_state.py` | 35,710 | `Pokemon` / `Turn` / `Action` / `ParsedReplay` / `BackwardMarkers` |
| `replay_parser/backward.py` | 15,201 | the backfill pass — `fill_missing_team_info`, `POVReplay`, `backward_fill` |
| `replay_parser/checks.py` | 13,714 | the conservative-discard consistency checks |
| `replay_parser/exceptions.py` | 7,900 | `BackwardException`, `InconsistentTeamPrediction`, … |
| `team_prediction/predictor.py` | 23,384 | `TeamPredictor` / `NaiveUsagePredictor` / `ReplayPredictor` |
| `team_prediction/team.py` | 43,522 | `TeamSet` / `PokemonSet` |
| `team_prediction/iterative_decoder.py` | 22,408 | MaskGIT-style parallel decoding (learned predictor) |
| `team_prediction/prediction_model.py` | 18,820 | the learned `TeamPredictionModel` (experimental) |
| `team_prediction/usage_stats/` | ~55 KB | Smogon usage-stat scraping/reading |

**The forward/backward structure.** Forward: simulate the battle from the spectator perspective,
narrowing both teams as information is revealed. Backward: at battle end, impute everything never
revealed, then *push the imputation back through every decision point* and slice out one player's
view. From `backward.py::fill_missing_team_info` (verbatim docstring):

> Team prediction works by:
> 1. Converting the team we've gathered here in the replay parser to the format expected by the team_prediction module
> 2. Predicting the team with a TeamPredictor
> 3. Filling missing information with the predicted team

and it raises `InconsistentTeamPrediction` if `revealed_team.is_consistent_with(predicted_team)`
fails. `POVReplay` carries `_resolve_transforms`, `_resolve_zoroark`, `_fill_one_side`,
`_align_states_actions`; module-level `backward_fill` and `add_filled_final_turn`.

### 🔑 How they handle the imitated player's OWN unrevealed information

**They treat it identically to the opponent's.** The parser never assumes it knows either team; it
reconstructs *both* from the public log and fills *both* at the end. S1 §3: *"we use incoming
information to estimate the initial configuration of **both unobserved teams**. At the end of the
battle, we infer any information that was never revealed."* Then `_fill_one_side` /
`backward_fill` inject the imputed team into every earlier decision point, so the agent's own
"knowledge" of its own team at turn 1 is a *guess made with hindsight from the whole battle*.

This is worth stating plainly because it is the design decision that makes the whole thing tractable
and is also the source of the deepest concern for us (§2.7): **the imitated player's own team is
imputed, not known.** They do not have a privileged channel for it.

**The two predictors** (verbatim from `team_prediction/predictor.py` docstrings):

- `NaiveUsagePredictor` — *"The original paper strategy. We use the names of the pokemon we have
  already know to guess the full team, then fill in the movesets using usage stats. **Every decision
  is made independently by sampling from unconditioned usage distributions.** This has some downsides
  that are explained / improved by the ReplayPredictor."*
- `ReplayPredictor` (dataset v1+) — *"matches the current revealed team to a set of candidates
  discovered from every replay in the dataset. If we think of revealed teams as nodes on a graph,
  where an edge A -> B exists if team B could have been made from team A by revealing more
  information, then these `candidate` teams are the leaf nodes. By sampling from the most commonly
  implied candidates, we are restricting our predictions to real + reasonably popular choices."*
  Defaults: `top_k_consistent_teams=20, top_k_consistent_movesets=15, top_k_scored_teams=10,
  top_k_scored_movesets=3`. **"Currently only supports gen1ou, gen2ou, gen3ou, and gen4ou"** — gen3ou
  is a first-class supported format.
- `score_pokemon`'s docstring names the sharpest failure mode of naive counting, and it generalises
  to any prior-based imputation we build: *"the difference between a moveset that is frequently
  **possible** vs. a moveset that is frequently **used**"* — with a worked gen1ou Tauros example where
  Stomp/Thunder/Swords Dance are *"dramatically overrepresented because many replays are consistent
  with these movesets, but in reality these would rarely be the 4th move."* The fix is to score the
  *diff* (`current_pokemon.additional_details(candidate_pokemon)`) against per-species usage
  likelihood. **We would hit this exact bug on day one if we counted naively.**
- An experimental learned predictor also exists (`prediction_model.py` + `iterative_decoder.py`,
  transformer masked-token prediction with MaskGIT-style parallel decoding) — not the paper's method.

### Reconstruction failures and missing action labels

- **Discards.** *"A long list of checks identifies trajectories that have entered ambiguous
  situations and conservatively discards them"* (S1 §3), implemented in `checks.py`. **They do not
  publish a discard rate** — **UNVERIFIED**, and I looked.
- **Two acknowledged residual gaps** (S1 App. D.1): illegal actions, and unrevealed moves. Both are
  *retained with mitigations* rather than discarded, because unrevealed moves occur *"too often to
  discard the trajectory"* — a qualitative statement with **no percentage attached** (UNVERIFIED).
- **Missing action labels** (S1 App. D.1): *"BC-RNN baselines... mask unrevealed action labels. When
  training offline RL... RL models fill action labels"*, with the stated rationale that *"when
  training offline RL along a full trajectory sequence in parallel, it is risky to back up the
  Q-values of timesteps where the actor or critic is not trained."* The filler is *"a small BC-RNN
  model trained on a much earlier version of the dataset"*. Their Figure 20 ablates improved
  missing-action labels on a 15M model.
- **No stated reconstruction error rate anywhere.** No "% of moves never revealed", no "% of imputed
  fields correct". This is the paper's weakest reproducibility surface, and it is precisely the
  quantity our Stage A gate would measure for ourselves (§2.6).

## 1.4 Algorithm

| Property | Value | Source |
|---|---|---|
| Backbone | **AMAGO** (Grigsby et al. 2024a) — causal Transformer, actor + critic heads | S1 §4 |
| Context | *"entire battle up until the current timestep, τ₀:ₜ"* | S1 §4 |
| Model scales | **15M / 50M / 200M** params | S1 §4 |
| Observation | **87 text "words" + 48 numerical features**; text tokenized over a Pokémon vocabulary with an `<unknown>` token | S1 §4 |
| Observation scope | ⚠ *"observations only include the **opponent's active Pokémon**"* — the model must *"rely entirely on memory to infer the opponent's team"* | S1 §4 |
| Action space (paper) | **9 discrete**: indices 0–3 = active mon's moves, 4–8 = the five switches | S1 §4 |
| Action space (repo, now) | 13 discrete, to cover gen9 gimmicks | S3 |
| Objective family (Table 1) | `IL` (w=1, λ=0) · `Exp` (w = exp(β·Aᴨ(h,a))) · `Binary` · `Binary+MaxQ` | S1 §4 |
| Self-play ladder | SynRL-V0 → V1 (adds gens 2+4, **3M trajectories**, 200M retrained from scratch) → V1+SP → V1++ → **V2** (a new 200M trained from scratch on V1++ data **+ 50k new human replays**) | S1 §5.3 |

Note the obs scope line: their agent sees only the opponent's **active** Pokémon and must remember
the rest. **Our 2501-dim obs carries a full 6-slot opponent team block with belief heads** — a
structurally richer input. That is a genuine architectural difference, not a detail.

## 1.5 Results — and the gen3ou answer

**Figure "Ladder Ratings in Online Battles vs. Humans"** plots official Glicko-1 with rating
deviations; the **bar labels are GXE %**. The numbers live only inside the image, so I downloaded
`arxiv.org/html/2504.04395v2/x11.png` and read the four panels directly (S2). Bars are in legend
order.

| model | Gen1OU | Gen2OU | Gen3OU | Gen4OU |
|---|---|---|---|---|
| PokeEnv Heuristic | 27 | *(cropped)* | **22** | 32 |
| Small-IL | 47 | 32 | **28** | 37 |
| Large-IL | 47 | 42 | **35** | 38 |
| Large-RL | 58 | 53 | **42** | 41 |
| SynRL-V0 | 69 | 59 | **52** | 48 |
| SynRL-V1 | 74 | 63 | **51** | 62 |
| SynRL-V1+SP | 70 | 63 | **58** | 53 |
| SynRL-V1++ | 70 | 63 | **60** | 58 |
| **SynRL-V2** | **77** | **68** | **64** | **66** |

Gen1OU Glicko-1 axis readings (only the Gen1 panel carries y-tick labels): heuristic ≈1310,
Small-IL ≈1475, Large-IL ≈1478, Large-RL ≈1565, SynRL-V0 ≈1655, SynRL-V1 ≈1698, V1+SP ≈1665,
V1++ ≈1655, **V2 ≈1723**, against an *"approximate leaderboard threshold"* dashed line at ≈1655.
Panels 2–4 appear to share that y-axis but carry no tick labels — **treat the non-Gen1 Glicko values
as UNVERIFIED**; the GXE bar labels are the trustworthy per-format number.

**Figure "Ladder Percentiles"** (`x13.png`, read directly):

| format | known active players | SynRL-V2 percentile | SynRL-V2 GXE |
|---|---|---|---|
| **Gen1OU** | **2,661** | **95.8%** | ~77 |
| **Gen3OU** | **8,944** | **90.1%** | **64** |

**Prose (S1 §5.5):** *"Our SynRL-V2 model is a reasonably advanced player estimated to be inside the
top decile across generations. Although ELO ratings are noisy, the SynRL-V1 and SynRL-V2 models
reach peak global rankings of #46 and #31 in Gen1OU, respectively, and SynRL-V2 makes **two
appearances inside the top 300 in Gen3OU**. All RL models sit inside the top 500 in Gen2OU."*
Footnote 4: *"SynRL-V2 plays 613 human battles and settles at a Gen1OU GXE of 79.9% (Glicko-1
1761±35)."* Evaluations ran late Dec 2024 → late Mar 2025.

### ➡ The gen3ou answer, plainly

**Gen3OU is Metamon's weakest generation among gens 1–4, and it is the format where the paper's
headline "top decile" claim is literally the measurement rather than a floor.** Best model: **GXE
64%, 90.1st percentile of 8,944 known active players, two appearances inside the top 300**. Compare
gen1ou: **GXE 77%, 95.8th percentile of 2,661, peak #31 global**. Every rung of their ladder is
lower in gen3 than gen1 — heuristic 22 vs 27, Large-IL 35 vs 47, Large-RL 42 vs 58, V2 64 vs 77.

**Their stated explanation for the gen3 shortfall: there isn't one.** The paper explains why it stops
at gen 4 (team-space variance and team preview from gen 5 on) but I found **no sentence explaining
gen3 < gen1** — **UNVERIFIED / absent**. Structural facts that are in the paper and are consistent
with it, offered as inference not as their claim: gen3ou's active pool is **3.4× larger** (8,944 vs
2,661), so the same percentile is a harder climb; gen3 adds abilities and held items to gen1's
mechanics, enlarging the hidden-information space; and their observation shows **only the opponent's
active mon**, pushing all of that hidden state onto sequence memory.

## 1.6 Corrections to our stated beliefs

| our belief | verdict | correction |
|---|---|---|
| *"hundreds of thousands to ~1M+ battles"* | **✅ right for the paper, badly understated for today** | The paper is 475k battles / 950k trajectories / 38M timesteps at a Sept-2024 cutoff (S1 §3). The **published** corpus is now **2,681,841 raw replays** (S5) and **5.3M parsed trajectories** (S3), plus 22M self-play trajectories. |
| *"top-decile of active ranked players in gen1 OU"* | **🔴 WRONG — attached to the wrong generation** | Top-decile is the **across-generation floor**, and the generation that *sets* that floor is **gen3ou (90.1st percentile)**. In **gen1ou** they are far better: **95.8th percentile, peak global #31, GXE 79.9% / Glicko-1 1761±35 over 613 battles**. Reading "top decile" as a gen1 result understates gen1 and overstates gen3 — and gen3 is the only one we care about. |
| *"offline RL beat plain BC in their ablations"* | **✅ right, and consistently — but the margin shrinks in the later gens** | Large-RL − Large-IL, in GXE points: gen1 **+11** (58 vs 47), gen2 **+11** (53 vs 42), **gen3 +7** (42 vs 35), gen4 **+3** (41 vs 38). Text: *"RL updates significantly outperform the pure-BC Transformers"* (S1 §5.2). Note the margins are on top of a **BC base that is itself weak in gen3** (35 GXE). Neither BC nor offline-RL-on-BC-data is what got them to 64; **self-play on top of it was** (52 → 64 across the SynRL series). |
| *"human-competitive, not superhuman"* | **✅ right, and the paper agrees** | *"a reasonably advanced player estimated to be inside the top decile"*, *"Our best agents climb to rankings inside the top 10% of active players"* (abstract). Peak #31 in gen1ou is strong but is a peak, not a settled rating. |
| *"they may have published the parsed first-person dataset"* | **✅ CONFIRMED, and it is better than we assumed** | Published: raw logs, parsed POV trajectories, revealed-teams records, replay-derived team-prediction stats, teams, usage stats, and 40+ model checkpoints. **Data is CC-BY-NC-4.0; code is MIT.** The raw-replay card states no license (**UNVERIFIED**). |

One further correction worth flagging because it changes how a reader should weigh their numbers:
**their headline agent is not a replay-BC model.** SynRL-V2 is a 200M transformer trained from
scratch on a **5M-trajectory mostly-self-play** dataset, with human replays as the seed and a 50k-replay
top-up. The human corpus is the *bootstrap*, not the *product*. Anyone reading "replays got them to
top 10%" is reading it wrong.

---

# Part 2 — feasibility for our pipeline

## 2.0 What our stack actually needs (the crux, verified in code)

The load-bearing discovery is in `src/agents/training/obs_materializer.py:1–20` (verbatim):

> *"The way to get the obs of ANY point — recorded or re-rolled — is to **replay the one-sided
> protocol stream** through the *real* obs pipeline from battle start, rebuilding the tracker state
> exactly as the live player did. That replay is what this module does."*
>
> *"**The one-sided / omniscient wall (hard requirement).** Input here is only the per-side protocol
> chunks regenerated by `utils.bridge.reconstruction`... **The omniscient reconstruction data (seed,
> both teams, dice) never enters this module**; the encoder reads the same partial poke-env view it
> reads live."*

So the interface our entire offline obs stack is built on is:

```
one-sided protocol chunks  ──►  obs_materializer  ──►  2501-dim obs (+ tracker state, bit-faithful)
```

The `__RECON__` record (seed + both teams + committed choices) is required by
`utils/bridge/reconstruction.py::replay_battle`, whose job is to **produce** those chunks by
re-simulating. **A public replay does not need to be re-simulated.** It needs a *second producer* of
one-sided chunks. That reframing turns "build an offline obs pipeline for replays" (months) into
"build a spectator→one-sided transcoder" (weeks), and it is the reason this memo does not conclude
"infeasible".

*(Housekeeping: that same docstring says "the 3457-dim output vector". The obs is **2501-dim**
(root `CLAUDE.md`, `designs/ARCHITECTURE.md`). Stale by two generations — corrected in this commit.)*

## 2.1 Gap 1 — parsing a spectator log into ONE SIDE's poke-env battle state

**What already works.** A spectator log is, for the most part, a *subset* of a player log: every
`|move|`, `|switch|`, `|-damage|`, `|faint|`, `|-status|`, `|-boost|`, `|turn|`, weather and hazard
line is present and identical. poke-env's `Battle._parse_message` (which `Gen3Battle.parse_message`
wraps — it classifies the line into a `BattleEvent` *before* state mutates, then delegates) will
consume those lines happily.

**What breaks — and it is one thing, not many.**

| break | where | severity |
|---|---|---|
| **No `\|request\|` JSON.** `LegalActions.from_battle` (`src/agents/battle/live_view.py:501–544`) builds `move_slots` from `request["active"][0]["moves"]` (`id`, `pp`, `maxpp`, `disabled`, `target`), and reads `battle.available_switches`, `battle.force_switch`, `battle.trapped`, `battle.maybe_trapped`, `battle.wait`, `battle.available_moves`. `mask_generator.py:74` raises `RuntimeError("STRICT MODE FAILURE: No last_request found in battle. Cannot mask.")` | `live_view.py`, `mask_generator.py` | **BLOCKING** |
| **p1/p2 role assignment.** Spectator logs name both players but do not mark "you". | — | **already solved**: `src/main/prober/loops.py:273::identify_our_side` + `:136::players_from_protocol` |
| **HP granularity.** Replays give `x/100`; live gives exact `x/y` on own side. | — | **mostly a non-issue** (below) |
| **`\|request\|`-derived volatiles** (Choice lock, Disable target, Encore, Taunt turns) that poke-env reads from the request rather than the log. | poke-env | **partial** — most have a `\|-start\|`/`\|-activate\|` line in the log; PP does not |

**The HP finding is better than expected.** The per-mon obs HP channel is a **fraction**
(`live_view.py:107` `hp_fraction: float  # 0.0–1.0`, populated at `:214` from
`mon.current_hp_fraction`; consumed at `src/agents/observation/pokemon.py:156`). Exact integer
`current_hp`/`max_hp` appear on `LiveMon` (`live_view.py:160–161`) but the **only** obs consumer is
`src/agents/observation/incoming_damage_encoder.py:265`. With an imputed spread we compute exact
`max_hp` from base stats + EVs/IVs/level and set `current_hp = round(frac · max_hp)` — a ≤0.5%
rounding error on one narrow feature block, not a structural blocker. **The subagent-level intuition
that "our obs needs exact HP" is wrong; I checked.**

**The fix is request synthesis, exactly as Metamon does it.** We must emit a synthetic `|request|`
before each of our decisions, containing only what `LegalActions` reads. That requires tracking, for
our imputed active mon: PP per move (decrement on use, Pressure doubles it), `disabled` (Disable /
Encore / Taunt / Torment / Choice lock / no-PP), plus `force_switch`, `trapped` / `maybe_trapped`
(Mean Look, Block, Spider Web, Arena Trap, Shadow Tag, the wrap family), and `wait`. **Bounded list.**

> **Lift: 6–10 days.** Reuses `prober/loops.py` line parsers and `identify_our_side` (~0 days).
> The high-fidelity alternative — a *forced-outcome* re-simulation that drives `src/rust_sim` and
> resolves every RNG call against the recorded log — would give a perfect request but is **15–25
> days and high-risk**; it is the fallback, not the plan.

## 2.2 Gap 2 — imputing the imitated player's own unrevealed details

**The opponent half is already free.** Our encoder's opponent path is public-information-only by
construction: `spread_known=0` distinguishes "unknown opponent" from "own mon with 0 EVs", item and
ability are `None` until revealed, and `moves` holds revealed moves only. Ability priors are already
consumed live. **Zero work.**

**The self half is the work**, and our priors already cover Metamon's v0 (`NaiveUsagePredictor`)
exactly. In `data/pokemon/`, reached through the `agents.gen3_data` facade:

| artifact | shape | fills |
|---|---|---|
| `gen3_spread_priors.json` | species → top-k `[nature, [6 IVs], prob]` | EVs / IVs / nature → exact stats → `max_hp` |
| `gen3_move_priors.json` | species → {move_id: prob} | the unrevealed 1–3 move slots |
| `gen3_item_priors.json` | species → {item_id: prob} | held item |
| `gen3_ability_priors.json` | species → {ability_id: prob} | ability |
| `gen3_hidden_power_priors.json` | species → {hp_type: prob} | HP type (with our live `HiddenPowerTracker`) |
| `gen3_teammate_priors.json` | species → {teammate: prob} | the unrevealed team slots — *the one species×species joint Smogon publishes* |
| `gen3_learnset.json` | species → legal gen3 movepool | the hard legality gate on any imputed move |
| `data/teams/gen3_team_archetypes.json` | 719 pool teams → pace class + style tags | archetype-consistency scoring (optional) |

**Priors-policy check.** Root `CLAUDE.md` requires anything the network reads to trace to *"Smogon
(or ground-truth labels / **ladder replays**)"*. Human ladder replays are explicitly sanctioned, so a
replay-derived candidate-set prior (Metamon's `ReplayPredictor` upgrade) is **in policy**. The pool
remains off-limits as a prior source.

**Take their `score_pokemon` lesson for free.** Naive per-slot sampling produces
*frequently-possible* rather than *frequently-used* sets — their Tauros example — and we would ship
that bug by default. Score the *diff* between revealed and candidate against per-species usage.

> **Lift: 1–2 days** for a Naive-equivalent (wiring existing priors + learnset legality).
> **+4–6 days** for a `ReplayPredictor`-equivalent, which needs a candidate corpus — and Metamon
> **publishes** one (`replay_stats.tar.gz`, 4 MB, "rosters and sets with frequencies") under
> CC-BY-NC, or we build our own from scraped logs.

## 2.3 Gap 3 — action-label extraction

Our action space (`src/agents/action/constants.py`, verified):

```
ACTION_SPACE_SIZE = 11
0–5   switches, team slot 0–5   (indexed against list(battle.team.values()) order)
6–9   moves, REQUEST slot 0–3
10    struggle
```

The memory note *"action-aligned consumers MUST source `legal.move_slots`"* is the whole difficulty —
and **synthesizing the request dissolves it.** Because we author the `|request|`, we author the move
order; the label is simply the index of the used move in the array we emitted. There is no order to
"recover". Remaining cases:

| case | handling |
|---|---|
| a move is used that our imputed set lacks | repair the imputation online and **re-emit prior requests** (Metamon's forward pass narrows the same way). Cheapest correct policy: two-pass — narrow forward, impute at end, re-render requests backward, exactly their forward/backward split |
| Hidden Power | `gen3_hidden_power_priors.json` + our live `HiddenPowerTracker` |
| forced / non-decisions (recharge, charge turn 2, partial trap, post-faint replacement) | **already handled** — `obs_materializer`'s documented "all-zero mask ⇒ no decision (no row, no tracker advance)" deferral |
| Struggle | visible in the log; maps to action 10 |
| turns whose action left no visible trace | mark `missing` and mask, as Metamon's BC-RNN baselines do; **do not** fill (we have no BC-RNN filler and their own ablation says filling exists for offline-RL Q-backup reasons that do not apply to a BC aux term) |

> **Lift: 3–5 days**, dominated by the backward re-render.

## 2.4 Gap 4 — where it plugs into training

The seam exists. `src/agents/training/teacher/buffer.py`:

```python
@dataclass
class Correction:
    obs: np.ndarray          # [obs_dim] float32 — the one-sided policy obs
    action_mask: np.ndarray  # [n_actions] int8/bool
    better_action: int       # action index (6+k == move slot k)
    advantage: float
    confirmed_value: float
    step_produced: int
    opponent: str
    pi_target: Optional[np.ndarray] = None   # [11] float32, 0 on illegal
```

`CorrectionBuffer` is a bounded recency ring on the model (`model._correction_buffer`), sampled with
its **own policy forward** on each rollout minibatch inside `train()` — deliberately **not** the PPO
rollout buffer, because off-policy states must never enter GAE or the clip objective. Existing flags:
`--search-teacher` (`parser.py:594`), `--opd-coef` (`:616`), `--distill-coef` (`:801`).

Three options, cheapest first:

1. **BC as an aux CE term over the existing buffer.** Fill `Correction`s from replays with
   `better_action` = the human action and `pi_target` = one-hot. Add one loss term
   (`--replay-bc-coef`, masked cross-entropy) and one disk loader (`--replay-bc-dir`). **This is the
   recommendation.** ~2–3 days. Do **not** smuggle it through AWR by faking `advantage` — that field
   means "confirmed dwin margin" and overloading it would corrupt the search-teacher's metrics.
2. **Offline pretrain, then RL.** Closest to Metamon, and by far the most expensive: our PPO stack
   has no offline pretraining path, and a fresh generation is a multi-week commitment. Not first.
3. **Distillation reuse (`--distill-coef`).** Wrong shape — it distils a *live teacher policy's*
   distribution, not a fixed dataset.

> **Lift: 2–3 days** for option 1.

## 2.5 Gap 5 — volume

**Metamon's corpus** (S4/S5): 2,681,841 raw replays; `gen3ou.tar.gz` = **2.691 GB parsed**, the
largest gen1–4 slice. The exact gen3ou battle count is **UNVERIFIED** — the HF filter API's DuckDB
index was corrupted when I queried it. **ESTIMATED:** gen3ou is 37.7% of gen1–4 parsed bytes
(2.691 / 7.134 GB); applied to the paper's 475k gen1–4 battles that is **~180k gen3ou battles at the
Sept-2024 cutoff**, and today's v6 corpus is larger. Treat as an order-of-magnitude only.

**Our own scrape** (S6, measured live 2026-08-23):
`https://replay.pokemonshowdown.com/search.json?format=gen3ou` returns **51 results/page**;
`page=` caps at **100** (page 101 → `[]`) for 5,100 via paging, but **`before=<uploadtime>` walks
arbitrarily deep** (verified back to Aug 2024). Pages 1→100 spanned uploadtime 1787540952 →
1785795586 = 1,745,366 s = **20.2 days**, so **≈252 public gen3ou replays/day ≈ 92k/year**. Ladder
games carry a `rating` field (e.g. 1472); tournament/unrated ones show `null`, which gives us a free
quality filter. Note the raw-replay card's **2021–2024 gap**.

**So volume is not the constraint.** Even a self-scrape reaches Metamon-paper scale for gen3ou in
well under a year, and the first experiment needs 5k.

> **Lift: 2–3 days** for a scraper + local cache. **0 days** to instead take Metamon's tarball — at
> the cost of **CC-BY-NC-4.0**, which is fine for research and forecloses anything else.

## 2.6 The cheapest credible first experiment

**Stage A — the round-trip gate (this is the decision point, and it costs ~2–3 weeks total).**

We can do something Metamon structurally could not, because we own a simulator *and* a ground-truth
obs producer: **manufacture a spectator log from one of our own bridge battles and measure the
imputation error against the truth.**

1. Take a recorded bridge battle with its `__RECON__` and its true `states.npz`.
2. **Degrade** it to spectator-equivalent: delete every `|request|` line, coarsen all HP to `x/100`,
   and strip our side's unrevealed moves/item/ability/spread.
3. Run the degraded log through the new replay path (§2.1–2.3).
4. Diff the materialized obs against the **known-true** obs, **per block**, and diff the recovered
   action labels against the true `actions` array.

This is a direct sibling of `src/agents/training/obs_roundtrip_fuzz_test.py`, which already asserts
that replaying a recorded battle reproduces `states.npz` **bit-for-bit** — so the harness, the
fixtures (`record_fixture_battle`), and the comparison machinery all exist. The output is a number
nobody has: **per-obs-block imputation error and label accuracy, with ground truth.** If label
accuracy is poor or the error concentrates in blocks the policy leans on, we stop here having spent
weeks instead of months.

**Stage B — the experiment (only if Stage A passes).**

- **5,000 gen3ou replays**, rating-filtered (prefer `rating ≥ ~1400`), ≈250k labeled decisions.
- **Arm:** fork the current generalist, `--replay-bc-coef` small, 2–3M steps.
- **Control:** the identical fork at coef 0 (byte-identical path, per our standing flag discipline).
- **Primary gate:** anchored ELO from `<run>/snapshot_ladder/ladder.json` **at matched snapshot
  count**, at run end — never a mid-run `eval/elo` delta (BT re-solves every node on every add and
  the newest node is systematically inflated).
- **Secondary, and arguably the more informative one:** the human-agreement probe. We already
  measured that our policy **under-switches versus humans (30% vs 16%)** with ~35% action match. If
  BC-on-replays is doing what it claims, that gap must close. If agreement rises but ELO does not,
  we have learned something specific and real about the imitation/strength gap rather than nothing.

**Total honest lift to a measured result:**

| item | days |
|---|---|
| spectator→one-sided transcoder + request synthesis (§2.1) | 6–10 |
| team imputation, Naive-equivalent (§2.2) | 1–2 |
| action labeling + backward re-render (§2.3) | 3–5 |
| Stage A round-trip gate | 3–5 |
| BC aux term + disk loader (§2.4) | 2–3 |
| corpus acquisition (§2.5) | 0–3 |
| **subtotal → decision point (Stage A)** | **~13–22 days** |
| Stage B run + analysis (mostly machine time) | 5–8 |
| **total** | **~18–30 engineering days ≈ 4–6 focused weeks** |

Optional upgrades, deliberately *not* in the first pass: `ReplayPredictor`-equivalent candidate
matching (+4–6 d), forced-outcome rust re-simulation for a perfect request (+15–25 d).

**Component coverage — what we already own:**

| gap | covered by |
|---|---|
| protocol line parsing, side identification | `src/main/prober/loops.py` (`parse_events`, `split_turns`, `players_from_protocol`, **`identify_our_side`**, `hp_frac`, `species_of`) |
| obs production from one-sided chunks | `src/agents/training/obs_materializer.py` — **unchanged** |
| ground-truth comparison harness | `src/agents/training/obs_roundtrip_fuzz_test.py` (+ `record_fixture_battle`) |
| self-side imputation priors | `agents.gen3_data.priors` + `gen3_learnset.json` |
| HP-type inference | `HiddenPowerTracker` |
| BC injection seam | `teacher/buffer.py` (`Correction`, `CorrectionBuffer`) + the AWR/OPD term plumbing |
| opponent-side encoding | already public-info-only — **no work** |
| corpus | Showdown replay search API, or Metamon's tarball (CC-BY-NC) |
| **not** reusable | `utils/bridge/reconstruction.py` (needs seed + both teams) |

## 2.7 The single biggest risk

**Imputation is a directional confound, not noise — and it points the wrong way.**

We fill our own side's unrevealed moves, item and spread from priors. The human chose **knowing the
true set**. Worse, the two error sources are *anti-correlated in the state*: imputation error is
largest **early in the battle, before reveals**, and that is exactly where the human's choice carries
the most information (lead selection, turn-1 read, the first pivot). Metamon's backward fill makes
this concrete — the team injected at turn 1 is a guess made with *hindsight from the whole battle*,
so a state at turn 1 is labeled with the action of a player who knew their set, while being
*described* by a set inferred partly from moves that had not yet been revealed. That is not i.i.d.
label noise a BC loss averages away; it is a systematic incentive to click moves the policy has no
grounds to believe it has, and it will show up as exactly the kind of over-confident coverage-move
behaviour our critic already struggles to price.

**Metamon does not quantify this anywhere** — no reconstruction error rate, no per-field imputation
accuracy, no discard rate (§1.3, all UNVERIFIED and searched for). It is the least-defended claim in
an otherwise strong paper.

**We can measure it, and they could not.** That asymmetry is the whole reason Stage A comes first:
we own a simulator that produces ground truth, so we can degrade our own battles and read the error
off directly, per obs block, with labels to check against. **Build the meter before building the
lever.**

Secondary risks, in order: (2) **license** — CC-BY-NC-4.0 on the parsed data forecloses any
non-research use, and the raw-replay license is unstated; a self-scrape avoids both. (3) **request
synthesis fidelity** — a wrong `disabled` or `trapped` bit silently changes the legal mask, which is
a GIGO class our own rules say to treat as drop-everything; the Stage A gate catches it, which is
another reason not to skip it. (4) **the ceiling may simply be low** — Metamon's own gen3ou BC
result is **GXE 35**, and offline RL on the same data only reached **42**; everything above that came
from **self-play**. Replays are a bootstrap, not a strength lever, and we should expect them to move
the human-agreement metric more convincingly than they move ELO.
