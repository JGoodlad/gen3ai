# What does the model represent? The first read of the representation probe battery (2026-10-09)

**DESCRIPTIVE / exploratory.** No registered test; every number is a seed mean with a 95 % t-interval over the screen's
finished seeds (legacy s1001–s1008, static s1001–s1007, all at 15M, pin `6c6d2e09`). The tool is
`python -m main.probe_battery` ([`designs/prober/probe_battery.md`](../../../prober/probe_battery.md)).

Owner's request (2026-10-09): *"a very rigorous linear probe decodability battery, or whatever you think is important.
I really want to understand what the model represents easily and what it doesn't, so we can go hunt this if needed.
... And do we understand the value of a phazer, or the response to an opponent's phazer?"* The orchestrator added
depth-use and capacity-use diagnostics the same evening.

## In plain language

- **A linear read tells what the network makes EASY to use, not what it uses.** "R²" below is the share of a
  quantity's variation a straight-line read of the representation recovers; "AUC" is how well it separates a yes from
  a no (0.5 = chance, 1 = perfect). Both are measured on battles the read never saw.
- **The next Spikes-like gaps.** Five facts a gen-3 decision needs are poorly represented where decisions are made,
  in BOTH arms: (1) the **end-of-turn race** — how much HP each active gains or loses at the end of the turn
  (Leftovers, poison, sand, Leech Seed): R² 0.18–0.22; (2) **how big a stat boost is** — the Atk / Def / Spe stage:
  R² ≤ 0.10 even at the input, though "is it boosted at all" reads at AUC 0.97 and the policy clearly uses boosts;
  (3) **the opponent's HP at our decision tokens** — R² 0.26–0.34 in the policy state, against 0.85 at the
  opponent's own input token; (4) **the speed margin** (R² 0.41 at our active, 0.28 in the policy state); (5) **Spikes on THEIR side** at our move
  seats and policy state (R² 0.35 / 0.20 legacy, 0.25 / 0.17 static), the fact that makes a phaze or a forced switch
  chip them.
- **Static-specific gaps reproduce the diagnostic on fresh, current-era games:** Spikes on our side at our tokens
  (R² 0.25 static vs 0.45 legacy; 0.24 vs 0.59 at our move seats; 0.26 vs 0.44 in the policy state), the Spikes chip
  our bench mon would take (0.31 vs 0.42), and our active's HP on its move seats (0.43 vs 0.77). Static is BETTER at
  "can we KO them" (AUC 0.94 vs 0.92 at our move seats) and the Toxic counter.
- **Represented easily:** weather, Spikes presence at the board tokens, status, choice lock, "is boosted", "can they
  KO us" (AUC 0.92), "would our bench mon be KO'd coming in" (AUC 0.91), alive counts and HP totals at the critic
  pool (R² 0.90), and the opponent's unrevealed moves and items (AUC 0.88–1.0 — but this is SPECIES memory: in this
  team pool a species nearly fixes its set).
- **Phazing.** The model KNOWS who can phaze (our active's Roar / Whirlwind AUC 0.96; their active's 0.94, species
  memory) and USES the boost half of the phazing value: with a phazer available, it puts 29 % of its mass on the
  phaze against a boosted foe vs 8 % against an unboosted one, and erasing the foe's boosts from the input drops that
  by 8.6 pp (legacy) / 6.4 pp (static). The **Spikes half** — dragging a foe in through Spikes — is used a little by
  legacy (−1.9 pp when their Spikes are removed; −4.4 pp on boosted states) and **not at all by static** (−0.4 pp,
  interval through 0), although both critics value their Spikes (−6 pp win probability when removed). Against an
  opponent's phazer, a boosted mon does NOT boost less: it sets up 16 % (no phazer on their team), 22 % (one hidden),
  28 % (one revealed, 79 states) — descriptive and confounded by matchup, but no sign of the "don't boost into Roar"
  response.
- **Depth looks saturated; width does not look under-used.** Skipping the second (last) trunk round changes the
  chosen action on ~40 % of states (top-1 agreement 61 % legacy / 58 % static; KL 0.48 nats); each round rewrites a
  token by about its own size. Static leans more on the last round's attention (dropping it: top-1 agreement 58 % vs
  67 %). Tokens spread over ~18–22 effective dimensions of 128 (90 % of variance in ~47), every attention head
  matters, no FFN unit is dead, and no low-rank truncation down to rank 96 leaves the policy unchanged. **The lever
  this points at is depth, not width** (with a caveat on the truncation test, below).

## The hunt list (ranked by decision relevance × how poorly it is represented)

| # | fact (seat) | legacy / static after the trunk, at its best decision site | at the input token | likely cause | cheapest fix |
|---|---|---|---|---|---|
| 1 | **end-of-turn net HP change** of our active / theirs; the race | R² 0.18 / 0.19 (ours) · 0.22 / 0.21 (theirs) · race 0.15 / 0.15 | 0.36 / 0.37 | a DERIVED quantity (Leftovers + status tick + weather chip + Leech Seed + Wish); it reaches tokens only as the `g` edge's ratio (legacy) or OPC's amounts (static) and is diluted by the trunk; a random network reads it as well | the residual ledger per mon as D content / on the switch cell (hand-computed doc §4 row 2, BUILT as N4 `--eot-residual`, OFF) |
| 2 | **stage magnitude** Atk / Def / Spe (their active; ours) | R² 0.06 / 0.05 (TA Atk), 0.10 / 0.09 (TA Spe); ours ≤ 0.08 | 0.09–0.23 | the 14 boost columns enter through the role MLP (E2 injection) and are squashed into "boosted or not" (AUC 0.97); a random network is no worse | stage-applied stats per active mon as D columns (hand-computed doc NEW row 12; exact physics, XS) |
| 3 | **their active's HP** at our decision sites | R² 0.27 / 0.34 (policy state), 0.26 / 0.34 (our active's token), 0.44 / 0.45 (our move seats) | 0.85 at THEIR token | the fact stays home (their token, 0.68 after the trunk) and reaches our side diluted; decisions get it mainly through the op's KO cells (we_can_ko AUC 0.92–0.94) | the target's HP on our move seats (NEW row 13, XS) — or depth |
| 4 | **speed margin** (log of our effective Speed over theirs) | R² 0.41 / 0.40 at our token; 0.28 / 0.24 policy state | 0.56 | the exact who-is-faster bit reads well (AUC 0.89 at our token) but the margin is lost through the trunk | exact speed physics S2 (already in the bundle) |
| 5 | **Spikes on THEIR side at our decision sites** (both arms) | R² 0.35 / **0.25** at our move seats (best decision site; gap −0.10 [−0.19, −0.02]), 0.20 / 0.17 policy state, 0.23 / 0.20 our active | 0.38 / 0.13 at our active | the fact sits in the board token (0.72 legacy global, 0.81 static THEIR SIDE) and barely reaches the tokens that decide; it is what makes a phaze or a forced switch chip — and static's policy does not use it when phazing (below) | the phaze cell's entry damage (hand-computed doc §4 row 7); N1 covers THEIR mons too (their own side's Spikes on each of their tokens) |
| 6 | **Spikes on our side at our tokens** (static) | R² 0.45 / **0.25** our active, 0.46 / **0.28** bench, 0.59 / **0.24** move seats, 0.44 / **0.26** policy state | 0.65 / 0.10 | static's mon tokens carry no board fact by design; the OUR SIDE token holds it (0.84) but it reaches our tokens only at the last round (the diag's H3) | N1 `--mon-hazard-cost` + N3 `--switch-hazard-cost` (both built, OFF) |
| 7 | **Spikes chip our bench mon would take** | R² 0.42 / **0.31** (gap −0.10 [−0.20, −0.01]) | 0.57 / 0.27 | as 6 | as 6 |
| 8 | **our active's HP on its move seats** (static) | R² 0.77 / **0.43** (first seat), 0.87 / **0.51** (all four) | — | static's move tokens carry no HP (the diag) | N2 `--move-actor-state` (built OFF) |
| 9 | **weather at our tokens / policy state** (static) | AUC 0.87 / **0.82** (our active), policy state 0.87 / **0.78** | 0.97 / 0.69 | as 6 (FIELD token, last round) | depth |
| 10 | **Toxic counter, sleep turns** | R² 0.37 / 0.46 (ours), 0.33 / 0.44 (theirs); sleep ~0 | 0.42–0.60 | rare states (900 / 700 rows), lost through the trunk; static BETTER (+0.10) | low priority |
| 11 | **a phazer anywhere on their team** (alive), true / revealed | AUC 0.83 / 0.83 (critic pool), revealed 0.82 / 0.83 | 0.70–0.80 | team-level belief, about what a random network gives | the hidden-species belief carries it; low priority (no USE evidence of a "don't boost into Roar" response either way) |

Not represented beyond a random network (so the trunk adds no LINEAR information): the opponent's next action
(switch AUC 0.84, attack 0.77 in the policy state, random network 0.82 / 0.72) — the flat intent head is its reader
and was not probed. Too rare to read in 25k decisions: Substitute, confusion, screens and their turns, Safeguard,
rain / sun / hail, temporary-weather turns.

## Represented easily (`easy` / high `ok` in [`catalogue.md`](catalogue.md))

AUC at the best decision site, legacy / static: they_can_ko 0.92 / 0.92 (our token); our bench mon KO'd on entry
0.91 / 0.91; we_can_ko 0.92 / 0.94 (move seats); is-boosted (their active) 0.97 / 0.97; status 0.88–0.98; choice
lock 0.98 / 0.99; weather 0.87–1.0 (legacy) at our tokens; our / their alive counts and HP totals R² 0.90–0.91 at the
critic pool; the opponent's unrevealed moves / items 0.88–1.0 (species memory — the species lookup alone gives
0.90–1.0). The critic's own win-prob separates wins from losses at AUC 0.766 / 0.763 (Brier 0.205 / 0.205,
[`critic_auc.json`](critic_auc.json)); a linear read of the critic pool gets 0.73.

## Phazing in detail ([`behaviour.json`](behaviour.json))

**(a) Phazing value** — states where our active has a legal Roar / Whirlwind (3,049; 184 with their active boosted):

| | legacy | static | static − legacy |
|---|---|---|---|
| P(phaze), their active boosted, no Spikes (112) | 0.288 [0.216, 0.361] | 0.280 [0.227, 0.333] | −0.009 [−0.090, +0.073] |
| P(phaze), their active boosted, Spikes up (72) | 0.228 [0.187, 0.268] | 0.195 [0.148, 0.242] | −0.033 [−0.089, +0.023] |
| P(phaze), not boosted, no Spikes / Spikes up | 0.076 / 0.110 | 0.083 / 0.111 | ~0 |
| Δ P(phaze), their boosts ZEROED (184) | **−0.086 [−0.112, −0.061]** | **−0.064 [−0.087, −0.041]** | +0.022 [−0.009, +0.054] |
| Δ P(phaze), +2 Atk / +2 SpA INJECTED (2,865) | +0.021 [+0.016, +0.025] | +0.014 [+0.010, +0.019] | −0.006 [−0.012, −0.001] |
| Δ P(phaze), their Spikes REMOVED (1,591) | **−0.019 [−0.030, −0.009]** | −0.004 [−0.011, +0.003] | **+0.016 [+0.004, +0.028]** |
| ... on boosted states (72) | −0.044 [−0.065, −0.024] | −0.006 [−0.023, +0.011] | +0.038 [+0.014, +0.062] |
| Δ win-prob, boosts zeroed / Spikes removed / boosts injected | +0.061 / −0.059 / −0.047 | +0.060 / −0.056 / −0.045 | ~0 |

**(b) Phazing threat** — our active boosted with a legal setup move (2,384): mass on setup / attack / switch by what
the viewer knows of the opponent's phazers.

| opponent's phazers | n | P(setup) L / S | P(attack) L / S | P(switch) L / S |
|---|---|---|---|---|
| none on their team | 1,392 | 0.158 / 0.167 | 0.547 / 0.536 | 0.158 / 0.156 |
| held, unrevealed | 913 | 0.221 / 0.220 | 0.538 / 0.517 | 0.101 / 0.114 |
| revealed | 79 | 0.279 / 0.281 | 0.460 / 0.435 | 0.059 / 0.082 |

Zeroing OUR boosts: P(attack) −0.208 / −0.200, P(switch) +0.141 / +0.127, P(setup) +0.056 / +0.045, win-prob −0.061 /
−0.064 — the policy knows it is boosted and cashes it in by attacking. The strata are natural, not edited: revealed-
phazer states are mostly stall mirrors, so the rise in setup is confounded (no edit removing a revealed phazer was
built — FINDING 4).

**Answer.** Decodable: yes (who can phaze, and the boosted / Spikes situations, AUC 0.85–0.96 at the relevant token).
Used: the BOOST half causally, in both arms; the SPIKES half weakly by legacy and not by static. The response to an
opponent's phazer: no evidence the policy avoids boosting into one.

## Depth and capacity use ([`depth_summary.json`](depth_summary.json))

On a 4,000-decision subsample (every 7th bank row), per checkpoint; seed means [95 % CI]:

| | legacy | static | random network |
|---|---|---|---|
| logit lens: heads read layer 1 (KL nats / top-1 agree / win-prob MAE) | 0.48 [0.42, 0.55] / 0.61 / 0.13 | 0.49 [0.36, 0.61] / 0.58 / 0.13 | — (its policy is uniform) |
| no layer-2 attention (KL / top-1) | 0.30 / 0.67 | **0.42 / 0.58** (gap +0.12 [+0.01, +0.24]) | — |
| no layer-2 FFN (KL / top-1) | 0.22 / 0.74 | 0.21 / 0.73 | — |
| skip layer 1 instead (KL / top-1) | 0.77 / 0.49 | 0.91 / 0.45 | — |
| ‖Δx‖ / ‖x‖ per round (our mons, their mons, moves, board) | 0.82–0.98 | 0.82–0.99 | 0.76–0.95 (legacy's shape) |
| one head ablated (KL, the 8 heads) | 0.05–0.14 | 0.06–0.13 | — |
| participation ratio of the tokens after the trunk (of 128): our mons / their mons / move seats / events | 18 / 22 / 22 / 22 | 19 / 22 / 22 / 21 | 15 / 15 / 11 / 20 |
| dims holding 90 % of the variance (same) | 47 / 49 / 47 / 47 | 46 / 49 / 48 / 48 | 39 / 34 / 31 / 43 |
| board tokens' participation ratio | 13 (the global token) | **5** (the three board tokens) | 6 / 4 |
| trunk weights truncated to rank 16 / 32 / 64 / 96 (KL) | 0.66 / 0.53 / 0.31 / 0.17 | 0.69 / 0.68 / 0.34 / 0.16 | — |
| input projections truncated to rank 32 / 96 (KL) | 0.27 / 0.08 | 0.24 / 0.13 | — |
| dead FFN units / mean fraction active | 0 / 0.45 | 0 / 0.44 | ≤ 0.008 / 0.49 |
| weight-matrix participation ratio, q·k·v / out / FFN (of 128) | 73–75 / 122–123 / 120–124 | 72–74 / 122–123 / 119–124 | 76–77 / 128 / 128 |

**Verdict.** Depth: SATURATED by the brief's sign — the last round changes the decision on ~40 % of states and the
first round's removal is worse still, so neither round is near-identity. Width: NOT shown to be under-used — tokens
occupy ~20 effective dimensions (like a random network of the same shape), every head and FFN unit is used, and no
rank-r truncation is free. ⚠️ Two cautions: the logit lens feeds the heads a distribution they were not trained on,
so its KL is an upper bound on what the last round contributes; and the trained matrices' spectra are still close to
their random-init shape (training moved them little), so the truncation test cannot tell learned width from init
energy the network adapted to. Read together: **depth is the better-supported lever** (the diagnostic's
recommendation of a third trunk round under static stands — now built as `--trunk-layers 3` inside the `static_recovery` arm; this read says the depth question is not static-specific).

## Label validity ([`label_check.json`](label_check.json))

The damage physics behind the KO facts is a small gen-3 calculator on TRUE stats (approximate: no crits, multi-hit
at a fixed count). Checked against what happened: when the opponent used a damaging move and our active stayed in
(and the foe did not fall first), our active fainted on **81.7 %** of the 3,932 "sure KO" predictions (the min roll
kills; the rest are misses, Protect, a Substitute), **50.4 %** of the 1,350 in-between, and **3.0 %** of the 13,132
"no KO" predictions (crits, residual chip). The labels are sound to that precision.

## What was read

- **Bank** (`~/gen3ai_archive/probe_battery/bank_v1/`, [`manifest.json`](manifest.json)): 704 games (22 cells × 16
  mirrored pairs: each legacy seed vs the static seed of the same index, and a ring inside each arm; greedy vs greedy,
  ONE CPU engine at the pin) → 65,489 decisions replayed through `core_events --views --trackers --obs` at the pin →
  **25,000 selected**, 5,000 per viewer-team archetype (stall, semi-stall, balance, offense, hyper-offense), phases in
  proportion (opening 1,837, midgame 17,714, endgame 5,449), 704 battles. Content hashes in the manifest; every step
  re-runs byte for byte.
- **Facts**: 114 ([`facts.json`](facts.json): name, family, kind, the mon it is about, its decision sites, valid
  rows, base rate, the species-lookup baseline); 14 too rare to read.
- **Captures**: one eager CPU forward per checkpoint over the whole bank (4 threads, nice 19) at the pin: tokens at
  10 seats × 3 depths, the policy state (the actor latent, 512) and the critic pool (512 at the pin).
- **Probes**: ridge with a fold-internal leave-one-out α, five folds by BATTLE; 26 slots per checkpoint; the random
  networks are 3 fresh builds per architecture.

## FINDINGS (standing rule 7)

1. **The H&L control task cannot gate here.** A mon token carries its species almost perfectly and there are ~150
   species, so a species-keyed random label reads at AUC ~0.99 at our active's input token: selectivity (real −
   control) is low or negative for most token-level facts. The tool therefore reports it, adds a SPECIES-LOOKUP
   baseline (how much of the fact the species alone fixes), and cautions a fact only when both say "species"; the
   tier never rests on selectivity (`designs/prober/probe_battery.md` §4).
2. **The team pool nearly fixes a species' set** (species lookup AUC 0.90–1.0 for the opponent's hidden moves and
   items), so belief facts' decodability here is species memory, not inference; an off-pool bank (ladder teams)
   would be needed to read belief.
3. **A first version of the archetype stratification was WRONG and was caught before use:** `classify_team` reads a
   team EXPORT, and handed a packed string it classified all 704 games' teams as semi-stall. The archetype table is
   now built by the worker through the pin's teambuilder (the h2h table's rule) and refuses any unmapped team.
4. **Not built:** an edit that removes a REVEALED phazer from the opponent's move slots (the cleanest "response to an
   opponent's phazer" test) — the move-slot encoding interacts with the belief machinery; the natural strata are
   reported instead, confounded.
5. **An edit is a lower bound:** it writes the encoder's columns (the active-context boost pairs, the global Spikes
   scalar) but the 32-event window still remembers the boost / Spikes events. The layout was self-checked against the
   views on all 25,000 rows before any edit.
6. **A near-constant feature column blew up a held-out fold** (a random network's critic pool; R² −300): fixed by
   zeroing training-constant columns, pinned by a test; every probe was re-run on the fixed code.
7. **Busy windows (CPU, nice 19, ≤ 4 threads, one heavy job at a time, each under `mem_cap.sh`):** bank play
   17:28–17:41 PDT, captures 17:44–17:54, probes 17:58–19:05, depth 19:06–19:45, behaviour 19:45–19:50, probes
   re-run on the fixed code 19:51–20:13 and 20:14–21:03. The live screen's speed read discards contended cycles; it may have lost quiet cycles in
   these windows. The box's load was 7–11 from other jobs during 18:30–19:30.
8. **Scope limits:** the bank is the screen arms' own greedy play at 15M (other policies visit other states); linear
   decodability is not use (only phazing has a use read); the policy readout of these checkpoints is `tower`, so there
   is no state-query site.

## Files

| file | what |
|---|---|
| [`catalogue.md`](catalogue.md) | every fact ranked (tier, best decision site, legacy / static, input, L1, random, selectivity, species lookup, gap, notes) |
| [`summary.json`](summary.json) | every fact × slot × arm (mean, CI, per seed), the gaps |
| [`behaviour.json`](behaviour.json) · [`depth_summary.json`](depth_summary.json) | the phazing use read; depth / capacity use with the verdict |
| [`facts.json`](facts.json) · [`manifest.json`](manifest.json) · [`cells.json`](cells.json) | the facts; the bank's manifest and cells |
| [`label_check.py`](label_check.py) → [`label_check.json`](label_check.json) | the damage labels against what happened |
| [`critic_auc.py`](critic_auc.py) → [`critic_auc.json`](critic_auc.json) | the critic's own win-prob vs the outcome |
| [`run_all.sh`](run_all.sh) | the commands as run |

Bulk (bank rows, captures, per-checkpoint probe and depth JSON, logs): `~/gen3ai_archive/probe_battery/`.
