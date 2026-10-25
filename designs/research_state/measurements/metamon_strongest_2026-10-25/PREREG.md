# P\* against Metamon's strongest gen3ou agent (the `PastaMittens` ensemble): feasibility and PREREGISTRATION, 2026-10-25

**The owner's ask (2026-10-25):** "try and run against Metamon's largest or most powerful model for Gen 3 OU."
This file records the feasibility check (no full read played) and registers the read **before any counted game**.

Abbreviations: P\* = `rb_x26_s2601` @280,000,170 (X26's plateau); H = `heldout72_hpfix`; F = `forum72`; U = the three
unfamiliar panels H, F, P72 at equal weight; CI = 95 % interval; LB / UB = its bounds; n.d. = not detected; pp =
percentage points; GXE = Glicko-X expected win probability vs a random ladder player.

## 1. Which Metamon agents can play gen3ou (checkout `/home/goodlad/dev/metamon` @ `0a00a759`)

Read from `metamon/rl/pretrained.py` (the registry), `README.md` and `metamon/rl/experimental/ensemble/`. Nothing new
was fetched from the web. "Local GXE" is upstream's internal round robin (README § Internal Leaderboards), relative to
the listed models only; "ladder" is upstream's human-ladder GXE with sample teams (README § Featured Policies).

| agent | params | gen3ou evidence (upstream's own) | weights in our cache |
|---|---:|---|---|
| **`PastaMittens`** (`KakunaEnsemble` router, preset `dt4_k34_midjudge_switchk28`) | 5 members: Kakuna ckpt 28 (×2 roles), 34, 30 + Alakazam ckpt 8 | **none for gen3**. README: the Kakuna router family is "the first Metamon agent to reach #1 on the human Showdown ladder", format NOT stated; its member `gxe` priors (0.78 / 0.75 / 0.72 / 0.64) match the gen1 column, and the code comment says "these gxe scores are incorrect" | ckpt 28, 34 yes; **ckpt 30 and Alakazam 8 NO** |
| **`Kakuna`** ckpt 34 (default) | 142.8M | docstring "the current best Metamon policy"; ladder gen3ou ~63 %; local GXE G3 **63 % competitive / 67 % modern replays, rank 1 of 18** | yes (and ckpt 28) |
| `Superkazam` | ≈142M | local G3 59 % / 62 % (rank 2) | no |
| `Kadabra4` / `Kadabra3` / `Kadabra2` | ≈57M | local G3 58 / 66, 57 / 60, 58 / 59 % | no |
| `Alakazam` ckpt 8 | ≈50M (README) | local G3 56 % / 61 % | no |
| **`SyntheticRLV2`** (the LARGEST) | **200.9M** | ladder gen3ou 64 % (2024 ladder); local G3 **55 % / 55 %** (rank 7) | yes |
| `SyntheticRLV1(+SP/++)`, `SyntheticRLV0`, `LargeRL` / `LargeIL`, `Medium*`, `Small*`, `Minikazam`, `Abra`, `Kadabra` | 4.7M–200M | local G3 34–48 % where listed | `SmallRL` only |
| `TaurosV0`, `TaurosEnsemble` / `Exeggcute`, `V2A*` | 12–62M | **gen1ou only** ("does not play other rulesets") | — |

## 2. The pick: `PastaMittens`, with Kakuna ckpt 34 as the standing comparator

- **Largest ≠ strongest.** `SyntheticRLV2` (200.9M) is the largest single model and ranks 7th on upstream's own gen3
  board (55 % vs Kakuna's 63–67 %); an older checkpoint of ours already reads it (SOP §4). Not the target.
- **Strongest single policy = Kakuna ckpt 34**: rank 1 on every gen3 column upstream reports, and in OUR regime ckpt
  28 is not detectably stronger (0.515 [0.466, 0.564], X51). P\* already has its numbers (X51): **home72 0.598
  [0.549, 0.644]; U 0.487 [0.459, 0.515]** (H 0.497, F 0.530, P72 0.433).
- **Most powerful deployed agent = the `PastaMittens` ensemble**: upstream's public eval agent built on Kakuna, the
  only Metamon agent that could plausibly exceed Kakuna on gen3ou. Its strength there is **UNVERIFIED** (no gen3
  number exists; its router was tuned with gen1 priors). The read below is what settles whether it is stronger than
  Kakuna for us; if not, Kakuna stays the goal's opponent.

## 3. Feasibility, measured (CPU, rule 5b cores, `mem_cap.sh` short lane, ports claimed by `kakuna.claim_port`)

Two 2-game smokes against `bot:heuristic` on the `away` set (our side a bot, so the peer's time is isolated), local
server only (ports 9542 / 95xx). Raw dirs: `~/gen3ai_archive/metamon_strongest_2026-10-25/runs/` (not committed).

| peer | median / p90 per decision | first decision (warm-up) | peak RSS (whole scope) | regime check |
|---|---|---|---|---|
| Kakuna ckpt 34 | **65 ms / 70–77 ms** (34 decisions) | 3.8–4.0 s | 2.1 GB | verified, argmax 1.0000 |
| ensemble COST PROXY: the `PastaMittens` preset's 5 members and router, with the two uncached members swapped for Kakuna ckpt 34 (`METAMON_ENSEMBLE_MEMBER_SPECS`) | **0.23–0.28 s / 0.37–0.68 s** (35 decisions) | 5.4–7.3 s | 4.9 GB | **not verifiable today** (FINDING 2) |

- **CPU is usable; no GPU needed.** The ensemble costs about 4× Kakuna per decision. The proxy is an upper bound:
  Alakazam (8 layers, d_model 768) is smaller than the Kakuna it stands in for (10 layers, 900). At ≈ 35 peer decisions per
  game that adds ≈ 7 s to X51's 11.8 s / game for P\* vs Kakuna: **≈ 19–20 s / game, ≈ 9 lane-hours of play for
  1,600 games**, plus the heavy-slot queue (X51: 18.5 h wall for 8 h of play).
- **Download needed (NOT done; awaiting the owner's OK):** from the Hugging Face repo `jakegrigsby/metamon` (the
  snapshot our cache holds is `ac8e88e7`), into Metamon's cache (`/home/goodlad/dev/metamon/cache`, not `models/` or
  `data/`):
  - `kakuna/ckpts/policy_weights/policy_epoch_30.pt`: **571,581,627 bytes** (the same architecture as our cached ckpt 28 / 34);
  - `alakazam/ckpts/policy_weights/policy_epoch_8.pt`: **≈ 0.2–0.35 GB, UNVERIFIED** (estimated from its config);
  - URLs `https://huggingface.co/jakegrigsby/metamon/resolve/<rev>/<path>`. Pin `<rev>` = `ac8e88e7` if it holds both files,
    else record the revision used and both sha256 values.

## 4. Preconditions (a separate harness unit, before any counted game)

1. **The two weights above**, sha256 recorded here.
2. **An `anchors.json` agent `PastaMittens`** (`"policy": "PastaMittens"`, no checkpoint), with the stamp
   `opponent_version = PastaMittens@preset:dt4_k34_midjudge_switchk28` and the preset digest (§6).
3. **A regime instrument for a router** (`metamon_side.py`). Today's probe wraps `amago.agent.Agent.get_actions`,
   which the ensemble never calls, and compares the emitted action with the LAST member's argmax; so every unit
   would read `regime_verified_decisions = false` and count nothing. The build: time and record `sample` on
   `HeuristicRouterEnsemblePolicy.get_actions`, and verify DETERMINISM per decision (the router draws no random
   number: the torch and Python RNG states are byte-equal before and after each call, rate exactly 1.0000; the
   router does not read the `sample` keyword at all). A test pins it, and the 2-game ensemble smoke must pass it.
4. **Harness identity with X51**: one P\* vs Kakuna ckpt 34 unit (home72, seed 0, 20 games) at the read's commit. It
   must reproduce X51's `pstar_home72_00` rows exactly (winner and turn count, 20 / 20). If any game differs, the
   full P\* vs Kakuna home72 control (400 games, seeds 0–19) is played at the read's commit, and §5's Δ on home72
   is read against THAT cell. U is then reported cross-harness (the panels' Kakuna cells are not re-played).
5. The T28 halt marker clear; ports 95xx only; `CUDA_VISIBLE_DEVICES=""`; `models/` read-only (P\* is a sha-checked copy).

## 5. The read

**Our side:** P\* = `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed`
(full: `814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed`), `bare` load, greedy, the in-process Rust core
slot. `rb_x26c_s2601` (live, last checkpoint 482,000,079, no final) has no reading that puts it above P\*, so P\* is the
best checkpoint we have (FINDING 4).

**Cells:** D1 look 2's four panels, each at X51's committed symmetric config
(`../x51_plateau_vs_kakuna_2026-10-20/configs/anchors_x51_sym_{home72,heldout72_hpfix,forum72,p72}.json`, plus the
`PastaMittens` agent entry), **400 games per panel = 20 units × 20 games**, role-balanced halves, `--team-seed =
--seed-base` = the unit seed **0–19**. So every game twins X51's P\* vs Kakuna game of the same seed, half and index:
both sides draw the same team files, and only the opponent differs. 1,600 counted games in all.

**Counting:** X51's rule. `main.anchors.results.unit_counts` (undeclared, symmetric), both content digests equal, the
capture check, AND precondition 3's determinism rate = 1.0000 with `sample_kwarg_values == [False]` on both halves.
A failed unit is re-run with the same seed; never replaced by a new seed.

**Statistics** (`main.x35.d1.newcombe_strat` for U, as X51):

| quantity | statistic | classes |
|---|---|---|
| **E-U (primary)** | P\* vs `PastaMittens` on U, equal weight, Kish n_eff | BETTER (LB > 0.50) / WORSE (UB < 0.50) / n.d. |
| **E-home** | the same on home72, Wilson | the same |
| **Δ-U (primary comparison)** | E-U − X51's 0.487 (Newcombe, stratified) | **ENSEMBLE STRONGER** (UB < 0) / **KAKUNA STRONGER** (LB > 0) / n.d. |
| **Δ-home** | E-home − X51's 0.598 (or precondition 4's control) | the same |
| paired descriptor (reported) | per twin: mean, normal and seed-bootstrap 95 %, discordant counts | none |

**What follows:**
- Δ-U = ENSEMBLE STRONGER ⇒ the orchestrator decides whether `PastaMittens` replaces Kakuna as goal (b)'s opponent
  (`design_beat_metamon_path.md` §2); the SOP's goal row names it.
- Δ-U n.d. or KAKUNA STRONGER ⇒ Kakuna ckpt 34 stays the goal's opponent (4× cheaper); the ensemble is reported.
- E-U BETTER would put P\* above Metamon's strongest gen3ou agent on unfamiliar teams. Expected: not.

**Power:** each panel's Wilson half-width at n = 400 is ≈ ±4.9 pp; Δ on one panel ≈ ±6.9 pp; Δ-U ≈ ±4.0 pp. A
2–4 pp ensemble gain will most likely read n.d.; the paired descriptor is reported to show its direction.

**Predictions (registered, not read by any rule):** the ensemble is mildly stronger than Kakuna in our regime.
E-home **0.56** (plausible 0.50–0.62), E-U **0.46** (0.42–0.50; WORSE or n.d.), Δ-U **−2.5 pp** (−7 to +2; n.d. most
likely), and on each panel the same order as against Kakuna (F > H > P72).

## 6. The regime and its id

Greedy (P\*) vs the ensemble's own DETERMINISTIC router (the ensemble forces `sample_actions_val = False`; its members'
`action_temperature` values only reshape the probabilities the router scores). Metamon `0a00a759`, CPU, one torch
thread per side, Showdown pin `e0551883f`, the 250-turn forfeit, server `rust`, transport `rust_core_slot`. Preset
`dt4_k34_midjudge_switchk28`, canonical-JSON sha256 `1b0d180e82207414…` (file `ensemble_presets.json` sha256
`249dcbc4dab1e7b5…`). **Regime id `5c1ea828e951126b`** (`agents.training.eval_ledger.schema.regime_id`, v1 rule)
over: `battle_format gen3ou, forfeit_turn_limit 250, mirrored_pairs false, model_loader bare, model_step 280000170,
model_sha256 814823c5…, opponent metamon:PastaMittens, opponent_commit 0a00a759, opponent_version
PastaMittens@preset:dt4_k34_midjudge_switchk28, opponent_preset_sha256 1b0d180e82207414, our_regime greedy,
their_regime greedy, our_side model, our_transport rust_core_slot, regime_matched true, server_impl rust,
showdown_pin e0551883f, units "20 x 20 games, role-balanced, --team-seed = --seed-base = unit seed 0-19"`. The same
fields with X51's opponent (`metamon:Kakuna`, `Kakuna@ckpt34`, no preset) give `e30dd0b859e45753`: the two cells
differ in the opponent only. Every row also carries `main.anchors`' own per-row stamp (SOP rule 1).

## 7. FINDINGS (feasibility)

1. **`main.anchors` cannot count an ensemble unit today** (precondition 3): the proxy smoke played 2 / 2 games cleanly
   (`status OK`) but read `regime_verified_decisions = false`, `argmax_match_rate = None`. The timings above came from a
   scratch probe on the router, which was never committed.
2. The proxy is a cost measurement, not a strength one (its members are not the preset's, and n = 2).
3. **`design_beat_metamon_path.md` named the ensemble's fourth member as Superkazam**; the preset (and `KakunaEnsemble`'s
   default) uses **Alakazam ckpt 8**. Corrected there in this commit.
4. **No `rb_x26c` final exists** (the run is live at 482M of its continuation). "Our best" is P\*. A later rb_x26c final
   would be read in its own unit.
5. **The ladder #1 claim has no format**, and nothing upstream reports the ensemble on gen3ou: "most powerful" is
   upstream's deployment claim, not a gen3 measurement.
6. The worktree needed `scripts/bootstrap.sh` (Showdown symlinks) before a smoke: the first attempt failed at team
   validation, no game played. A 19-character peer username (`MetaKakunaEnsemble1`) is refused by `check_username`;
   the agent name must be ≤ 13 characters (`PastaMittens` fits).
7. No other hazard: no GPU, no public server, local 95xx ports only, `models/` and `data/` untouched, every process
   stopped by its own exit.
