# heldout72_hpfix — heldout72 with Hidden Power repaired, and X26 / K2 vs Kakuna on it — 2026-10-18

**Why.** `heldout72` (`../heldout_teams_kakuna_2026-10-18/`) inherited the replay archive's Hidden Power artifact:
`hl_05_26` writes `Hidden Power` with no type and all-31 IVs, which is Hidden Power **Dark** under the gen-3 formula, on
164 of its 432 mons. A gen-3 replay never shows the type, so the real players' choices are lost. The owner's 2026-09-25
spec (`designs/ops/TECH_DEBT_BACKLOG.md`, "The Metamon ladder-usage corpus's HIDDEN fields are artifacts") says how to
repair it. `heldout72_hpfix` is the same 72 teams by id with only Hidden Power repaired.

**Answer (n = 400 each, Kakuna greedy, our side greedy, paired seeds 0–19).** On heldout72_hpfix **X26 reads 0.448
[0.400, 0.496] and K2 0.367 [0.322, 0.416]: both still LOSE to Kakuna** (WORSE, X26 only just). **X26 − K2 = +0.080
[+0.012, +0.147], X26 BETTER** (heldout72: +0.038, NOT DETECTED). Against heldout72 (HP Dark) the repair moves X26 by
+0.052 [−0.016, +0.120] and K2 by +0.010 [−0.056, +0.076], both NOT DETECTED, and the change in X26 − K2 between the two
sets is +0.043 (paired [−0.040, +0.125]), NOT DETECTED. **Repairing the HP Dark artifact does not close the held-out gap:**
X26 is still −0.127 [−0.195, −0.058] below home72; the repair's own effect (+0.052, upper bound +0.120) is neither
detected nor excluded. PROVISIONAL.

## The repair (`agents.gen3_data.hp_repair`, `gen3_hp_repair_v1`)

- **Target.** Only an UNTYPED `Hidden Power` whose IVs are absent or all 31. A typed one (`Hidden Power [Fire]`), or one
  with any IV below 31, stays byte for byte. 🚨 In `hl_05_26` EVERY set carries an explicit `IVs: 31 / 31 / …` line,
  which is the same default fill as no line; the spec's "do not touch a set that already has explicit IVs" is read as
  "IVs the player chose", i.e. any IV below 31 (FINDING 1).
- **The side** (gen 3: a move's category follows its TYPE — Fire / Water / Grass / Ice / Electric / Dark / Psychic / Dragon
  special, the rest physical; `deps/pokemon-showdown/data/mods/gen3/scripts.ts` and `moves.ts` `hiddenpower`). Three
  votes, each −1 / 0 / +1 (+1 = physical): the **EVs** (sign of Atk EV − SpA EV), the **nature** (+1 if it raises Atk or
  lowers SpA, −1 if it raises SpA or lowers Atk), and the set's **other attacking moves** (sign of #physical − #special
  by the type rule; status, fixed-damage and OHKO moves do not vote; Return, Frustration, Flail, Reversal, Low Kick,
  Magnitude, Present and Spit Up do). The sign of the sum is the side; a zero sum falls to the set's real Atk vs SpA
  stat (base stats, IVs, EVs, nature); an exact tie would leave both sides open (none occurred).
- **The type.** Sampled from Smogon's per-species Hidden Power usage, `agents.gen3_data.priors.hidden_power`
  (`data/pokemon/gen3_hidden_power_priors.json`: the chaos `Moves` typed `hiddenpower<type>` keys over 12 months, built by
  `tools/smogon_stats_downloader/compute_priors.py`), restricted to the side and renormalised. A species with no Smogon
  mass on its side falls back to the format-wide Smogon usage on that side (never needed here). Smogon only; nothing from
  our pool. `data/` was not touched (frozen while X26 runs): the source was already there.
- **The IVs** are what Showdown's validator writes (`sim/team-validator.ts` ~1154–1164: `fillStats(HPivs, 31)` from
  `data/typechart.ts`), so the power stays 70. 🚨 They are NOT `utils.gen3_utils.GEN3_HP_IVS`, whose Ghost row differs
  (FINDING 2).
- **Deterministic.** One `random.Random` per team, seeded `int(sha256("gen3_hp_repair_v1|20261018|<team id>")[:8 bytes])`,
  one draw per target mon in slot order over the types in alphabetical order. The seed, the votes, the restricted
  distribution, the uniform draw and the result of every repaired mon are in `heldout72_hpfix_manifest.json`
  (`teams.<id>.repairs`). Re-running `scripts/make_heldout72_hpfix.py` reproduces the manifest byte for byte.

## The set (`scripts/make_heldout72_hpfix.py`)

Per team: the `hl_05_26` SOURCE file named by `heldout72_manifest.json` (sha256-checked) → `hp_repair.repair_team` →
heldout72's own sanitiser (`make_heldout72.fix_team`, with the validator's IV table). Hard checks, all passed:

1. every line outside a repaired set's Hidden Power move line and its IVs line is byte-identical to the heldout72 file,
   and every repaired set is one heldout72 played as HP Dark;
2. the pinned validator accepts all 72 (`validate_teams_locally`, gen3ou, Showdown pin `e0551883f`);
3. `utils.bridge.hp_set_check`: every Hidden Power set (all 168) plays the type its line names at power 70
   (`dex.getHiddenPower`), and the validator leaves its IVs unchanged;
4. the overlap rules of the heldout72 draw still hold (below).

**Hidden Power, before → after (168 HP sets on 432 mons):**

| type | heldout72 | heldout72_hpfix | side |
|---|---:|---:|---|
| Dark | **164** | **0** | special |
| Grass | 0 | 64 | special |
| Bug | 1 | 39 | physical |
| Flying | 2 | 16 | physical |
| Ice | 0 | 15 | special |
| Fire | 1 | 14 | special |
| Ghost | 0 | 6 | physical |
| Rock | 0 | 6 | physical |
| Fighting | 0 | 5 | physical |
| Electric | 0 | 1 | special |
| Ground | 0 | 1 | physical |
| Steel | 0 | 1 | physical |

- **164 / 164 HP Dark sets repaired; 71 of them (43 %) changed side** (special HP Dark → a physical type): the HP Dark
  fill was on the wrong side of the set for 71 mons. The 4 typed sets (Bug 1, Fire 1, Flying 2) are unchanged.
- **The side was decided by the votes for 160 sets and by the stat tie-break for 4.** All three votes agreed (or
  abstained) on 148; on 16 the moves disagreed with the EVs and nature, and the EVs and nature won.
- **HP Dark is never drawn:** Smogon's Dark share is at most 0.23 % for every repaired species (Salamence: Flying 0.652, Grass
  0.337, Dark 0.00006).
- **The speed tie (the 30 Speed IV of Fire / Psychic / Rock / Fighting):** 24 repaired sets carry it (Fire 13, Rock 6,
  Fighting 5), each 1 Speed point below its heldout72 twin (pinned in `hp_repair_showdown_integration_test.py`: a Jolly
  252-Speed Salamence is 328 with HP Dark / Flying, 327 with HP Fire / Psychic / Rock / Fighting).

**Overlap** (`gen3_team_overlap_v1`, Hidden Power matched by type, so the repair moves it):

| heldout72_hpfix vs | exact team (strict) | same 6 species | mon sets seen (strict) | ≥ 4 shared | max shared |
|---|---:|---:|---:|---:|---:|
| our training pool (719) | 0 (0) | 21 | 164 / 432 (45) | **0** | 3 (1 team) |
| `competitive` (20) | 0 (0) | 9 | 61 / 432 (4) | 0 | 2 |
| (heldout72 vs the pool, for comparison) | 0 (0) | 21 | 104 / 432 (27) | 0 | 2 |

Repairing the type makes 60 more mon sets match a pool mon set, because the pool's Hidden Powers are real types; the set
is still held out by the draw rule (0 exact, 0 sharing ≥ 4).

**Files.** `teams_heldout72_hpfix/team_NNNNNN.txt` (the same 72 stems), `heldout72_hpfix_manifest.json` (per team: the
source file and its sha256, the heldout72 file's sha256, the new file's sha256, the team seed and every repair; the
summary and the overlap). **Installed for Metamon** as `~/dev/metamon/cache/teams/gen3ai_heldout72_hpfix/gen3ou/
team_NNNNNN.gen3ou_team`, byte copies; the read's own copy is `~/dev/gen3ai-reads/heldout72hp_kakuna_2026-10-18/
teams_heldout72_hpfix/`. The queue sha256-checks every file on both sides, and the config, before every unit. Metamon's
poke-env packs a repaired set exactly as our teambuilder does (checked on `team_000389`: `hiddenpowerflying` with IVs
30/30/30/30/30/31, `hiddenpowerghost` with 31/31/30/31/30/31).

## The read — X26 and K2 vs Kakuna on heldout72_hpfix (2026-10-18 11:13–12:53 PT)

| model | n | W–L–T | win rate [Wilson 95%] | vs 0.50 |
|---|---:|---|---|---|
| X26 `rb_x26_s2601` @161,000,169 | 400 | 179–220–1 | **0.448 [0.400, 0.496]** | **WORSE** |
| K2 `ai_v14_07_g0p_k2` @91,127,808 | 400 | 147–252–1 | **0.367 [0.322, 0.416]** | **WORSE** |

| | Newcombe 95% (headline) | paired descriptor (`paired.json`) |
|---|---|---|
| X26 − K2 on heldout72_hpfix | **+0.080 [+0.012, +0.147] X26 BETTER** | +0.080; normal [+0.017, +0.143], seed bootstrap [+0.007, +0.155]; 101 / 69 discordant |
| X26: heldout72_hpfix − heldout72 | +0.052 [−0.016, +0.120] NOT DETECTED | +0.052; normal [−0.008, +0.113], seed bootstrap [−0.010, +0.113]; 87 / 66 |
| K2: heldout72_hpfix − heldout72 | +0.010 [−0.056, +0.076] NOT DETECTED | +0.010; normal [−0.045, +0.065], seed bootstrap [−0.035, +0.055]; 66 / 62 |
| change of X26 − K2 (hpfix − heldout72) | | +0.043; normal [−0.040, +0.125], seed bootstrap [−0.045, +0.125] NOT DETECTED |

The cross-set pairs are exact twins: the same model, unit seed, half and game index draw the same team ids on both
sides in both reads (our team per game and the peer's `team_draws` per unit, checked by `paired_hpfix.py`), so a pair
differs only in the Hidden Power lines.

**Beside the other cells** (Newcombe, independent samples):

| model | home72 (72 / 72 in pool) | away (19 / 20 in pool) | heldout72 (HP Dark) | **heldout72_hpfix** | hpfix − home72 | hpfix − away |
|---|---|---|---|---|---|---|
| X26 | 0.575 [0.526, 0.623] | 0.497 [0.449, 0.546] | 0.395 [0.348, 0.444] | **0.448 [0.400, 0.496]** | **−0.127 [−0.195, −0.058] LOWER** | −0.050 [−0.118, +0.019] |
| K2 | 0.487 [0.439, 0.536] | 0.355 [0.310, 0.403] | 0.357 [0.312, 0.406] | **0.367 [0.322, 0.416]** | **−0.120 [−0.187, −0.051] LOWER** | +0.013 [−0.054, +0.079] |
| X26 − K2 | +0.087 [+0.018, +0.155] | +0.143 [+0.074, +0.209] | +0.038 [−0.030, +0.104] | **+0.080 [+0.012, +0.147]** | | |

**Reading.**

- **The HP Dark artifact is not the main reason our models lose on held-out teams.** The repair moved X26 by +5 pp and
  K2 by +1 pp, neither detected, and X26 is still 13 pp below home72 (DETECTED) and below even against Kakuna. The
  18 pp held-out drop measured on heldout72 is mostly still there. The upper bound (+12 pp) does not exclude a
  real part-effect: under `designs/endstate/design_diversity_teams_opponents.md`'s rule for H6 (KILLED iff not
  detected AND upper bound < +5 pp) the Hidden Power hypothesis is NOT killed, and not supported.
- **X26 − K2 on held-out teams is now DETECTED (+0.080), where heldout72 read it NOT DETECTED (+0.038).** The change
  between the two sets (+0.043) is itself not detected, and the lower bound (+0.012) is near zero, so read this as "X26's
  lead over K2 on held-out teams is probably real and probably smaller than on training teams (+0.087 / +0.143)",
  not as "the repair revealed it".
- **heldout72_hpfix is the held-out set of record from here on** (`designs/ops/EXTERNAL_ANCHORS_SOP.md` §2): its
  Hidden Powers are Smogon-plausible, so a read on it no longer carries a known-wrong move on 38 % of the mons.
- **Still confounded,** as heldout72 was: unseen by us AND a replay-derived, usage-filled team population of the kind
  Kakuna's data may cover (the heldout72 README's Kakuna caveat). The EVs and natures are still the predictor's fill.
- **PROVISIONAL.** X26 is one checkpoint of a live, unplateaued run (161M of a 500M cap), one seed.

**Integrity.** All 40 units counted on their first try (`status OK`, `n == 20`, `regime_verified_decisions`, no
`team_source_asymmetry`, `peer_clean`; `skipped_units` empty; the queue logged `fails: {}`). Kakuna's argmax rate was
1.0000 in every unit. One regime stamp per cell over its 400 rows (`cells.json` `regime_stamps`). Per role (our challenge
/ theirs): X26 0.445 / 0.450, K2 0.405 / 0.330. The 250-turn forfeit fired 0 (X26) and 1 (K2) times; one tie per cell
(heldout72 had none). Median length 38 / 39 turns. Both sides drew 69 of the 72 teams (the same 69 as heldout72).

**Regime stamp.** heldout72's, except the team set: Kakuna ckpt 34, Metamon `0a00a759`, CPU, `OMP_NUM_THREADS=1`,
greedy / greedy, Showdown pin `e0551883f`, 250-turn forfeit, 20-game role-balanced units with `--team-seed =
--seed-base = unit seed` (paired seeds 0–19). X26: `checkpoint_161000169_steps.zip` sha256 `bc6050d1…2992` (the
read-only copy in `~/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17/model/`, re-hashed before every unit), commit
`63f97a57` in a fresh detached checkout, `gen3ai_torch28`, `bare`, `rust_core_slot`, `ws_frontend[in-process]@63f97a57`.
K2: `final_model.zip` sha256 `cc5d5502…07c4` (+ its `model_config.json` / `metadata.json`), copied read-only from
`models/ai_v14_07_g0p_k2/`, its training pin `a5b2fdba` in a fresh detached checkout, `gen3ai_stable` (torch 2.5.1),
`bare`, the pin's websocket client, `ws_frontend@a5b2fdba` (port 9576). Config `anchors_heldout72_hpfix_kakuna.json` (committed here)
(sha256 `dcd99308…3916`): heldout72's with `home` = heldout72_hpfix on every side.

**Cost.** One lane on rule 5b's cores (`nice -n 19 taskset -c 5-7,13-15`, `mem_cap.sh 12`) beside the live X26 trainer
and the maturity battery. X26 8.5 s/game, K2 6.4 s/game; load1 mean 6.1 (max 12.5); 800 games in 1 h 40 min wall.

## FINDINGS

1. **Every `hl_05_26` set has an explicit all-31 `IVs:` line**, so the spec's "do not touch a set that already has
   explicit IVs" cannot mean "an IVs line is present" here (it would repair nothing). The repair treats all-31 as the
   default fill (the same Dark as no line) and leaves any set with an IV below 31 alone. Our `ladder_corpus` sets have no
   IVs line (the 2026-09-25 measurement), so the rule reads the same on both.
2. **`utils.gen3_utils.GEN3_HP_IVS`'s Ghost row is not Showdown's** (30 Atk / 30 SpD vs `HPivs` 30 Def / 30 SpD; both Ghost
   at power 70), found by the repair's pin to `data/typechart.ts`. `Gen3Teambuilder` applies ours before the validator,
   so the 37 pool HP Ghost sets with no IVs line play 1 Atk lower than on Showdown. Not fixed here (a training-input
   change while X26 is live); a TECH_DEBT row. heldout72 had no HP Ghost, and heldout72_hpfix writes Showdown's IVs.
3. **The played HP type on the server was not observed in a battle stream** (the read keeps no protocol log). It rests
   on the paste carrying both the typed move and matching IVs, the validator check, and both clients' packing (checked
   on one team, above).
4. No hazard hit: the T28 marker was clear before every unit; the tool's own server on 9576; nothing under `models/` or
   `data/` written; no GPU (`CUDA_VISIBLE_DEVICES=""`).

## Files

- `teams_heldout72_hpfix/`, `heldout72_hpfix_manifest.json`: the set.
- `cells.json` (both cells, every comparison, the overlap block), `rows.jsonl` (800 game rows + `read_cell`, `unit`,
  `unit_seed`, `source_read`), `paired.json` (X26 − K2, each model's hpfix − heldout72, and the change of X26 − K2).
- `scripts/make_heldout72_hpfix.py` (the set), `scripts/hq_hpfix.py` (the queue: heldout72's `hq.py`, set and pins
  changed), `scripts/aggregate_hpfix.py`, `scripts/paired_hpfix.py`. Raw unit dirs, logs, `units.jsonl`, the config and the
  read-only K2 copy live in `/home/goodlad/dev/gen3ai-reads/heldout72hp_kakuna_2026-10-18/`.
