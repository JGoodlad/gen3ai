# EXTERNAL ANCHORS — the SOP

**How our strength is read against an agent NOBODY here trained.** Every other number in this
project is measured against ourselves: `ladder.json` is our own snapshots, `eval/elo` is our own
pool, `win_rate_vs_pool` is our own bots. All of them can move together and say nothing. An
external anchor is the instrument that cannot: a third-party gen3ou agent, at a fixed protocol,
whose difficulty does not drift when our training does.

**The tool is `python -m main.anchors`. This document is the procedure.** The tool enforces what
can be enforced in code (the port refusal, the regime verification, the schema, the named
failures); this file holds what cannot — which read to take when, what the bar is, and what the
numbers have been so far.

---

## 0. Is this comparison valid? The two rules, the regime, the guards

Two numbers are comparable only when their REGIMES match. `main.anchors` stamps the regime on every row, the guards in
§ 0.2 police it, and this section is the one place that says what each covers. How a registered read records the id, and
which banked read sat on which engine, is `designs/tools/reads.md` §4.

### 🚨 RULE 1 — A number never leaves without its regime

*"Temperature 1.0" is not one setting across models.* Measured 2026-09-14 over 14,085 sampled decisions:

| at T = 1.0 | plays its own argmax | so the perturbation is |
|---|---:|---:|
| `SmallRL` (13.9M) | **64.7%** | 35% of decisions |
| `SyntheticRLV2` (200.9M) | **88.0%** | 12% of decisions |
| `Kakuna` (142.8M, ckpt 34) | **85–93%** per half | 7–15% of decisions (2026-09-28, X22 (c)/(f1), home72) |
| **OUR N0** (`ai_v14_01_base` final) | **62.8%** (5,287 / 8,418, vs Kakuna T = 1.0); 63.4% (2,605 / 4,108, vs Kakuna greedy) | **~37%** of decisions (2026-09-28, X22 (f3), `--our-temperature 1.0`, counted per decision) |

The same nominal knob is a large behavioural change for one policy and a small one for the other, and the two models'
temperature effects come out with **opposite signs** because of it. The de-risk's mixed-regime headline of **0.742**
against `SmallRL` became **0.520** on the like-for-like matched cell: **−22.2 pp [−34.1, −9.4]**, CI clear of zero.

So a win rate is reported with its regime, its team set and that set's overlap with our pool (§ 2 "The team-set
overlap"), the opponent's version and commit, our checkpoint's step, and (for a search opponent) the realized search
width. `main.anchors` stamps all of that on **every row of `games.jsonl`**, not once per file, and
`src/main/anchors/results_test.py` reads the written file back to prove it.

### 🚨 RULE 2 — The recurring read is greedy-vs-greedy

Both sides move together, and both sides are **verified per decision**, never assumed. In order of weight:

1. It is the protocol every other strength number here is taken under: `ladder.json` is greedy-vs-greedy with
   symmetric builders, `--temperature 0` is `play.py`'s documented measurement setting, and the eval sentinels went
   greedy on 2026-09-07. A number taken at T = 1.0 cannot be read beside any of them.
2. T = 1.0 is not a fixed yardstick across opponents (the table above), and if Metamon ships a new policy the regime
   moves under us with no flag to read. Greedy is the same operation for every policy.
3. Determinism narrows a fixed-n read, which matters when the point is to detect movement in our own model between
   milestones.

The cost was measured: greedy cost 2 forfeits in 400 greedy games (0.5%), T = 1.0 cost 2 in 400, and neither regime
collapsed the effective sample (87–100 distinct team pairs per 100-game cell in both).

**The standing caveat.** A deterministic policy is exploitable in principle, but only by an opponent that ADAPTS, and
neither side here does: Metamon's hidden state gives it memory *within* a battle, `amago`'s rollout loop resets it on
`done`, so it carries nothing across battles and cannot mine a fixed opponent over 100 games. Our model conditions only
on the observation. Greedy-vs-greedy is safe **for this pair**, and the property that makes it safe is an
implementation fact about Metamon that could change in a release. The one carry-over is the long-tail fault
(H5 · H17 · H21): the game after it is played on carried state, so that unit is VOID.

### 0.1 The regime: what the id covers, and what it does not

A registered read's id is `main.reads.assemble.anchor_regime_id`: a digest of the cell identity `summary.json` stamps
(`ANCHOR_REGIME_KEYS` plus the three conditional rows). A difference in any row is a different regime: never pooled
(inside one read the group is INCONCLUSIVE); an import refuses outright on a different engine, team pack, peer code or
policy mode (G7, G9, G14), and `import_off` on the regime id it registers.

| component | enters the id | stamped as | note |
|---|---|---|---|
| the opponent, its version and checkpoint | always | `opponent`, `opponent_kind`, `opponent_version` (`Kakuna@ckpt28`, `PastaMittens@preset:<name>`) | |
| both sampling regimes, with our load options | always | `our_regime`, `their_regime`, `regime_matched`; suffixes `sample:T=<T>`, `bot:<name>`, `+belief_prior_mix=<L>`, `+search=<preset>@<turn>` | a load option never pools with the unchanged model; verified per decision (G1) |
| our side and how it is served | always | `our_side`, `our_transport`, `server_impl`, `model_loader` | the two transports play the same games (§ 2 "`--our-transport`"), yet a registered read gives them different ids |
| the team cell | always | `teamset`, `our_teamset`, both file counts, both content digests (`gen3_team_set_digest_v1`), `declared_team_cell` | |
| the protocol | always | `battle_format`, `forfeit_turn_limit`, `mirrored_pairs`, `search_time_ms` / `search_parallelism` (Foul Play), `showdown_pin` | |
| the Metamon weights | only when NOT `assemble.LEGACY_WEIGHTS` | `hf_revision`, `weights_sha256` (`our_…` for a Metamon our-side) | another hub revision or any other digest is a new regime (§ "Weight pins") |
| the engine | only when NOT a banked engine | `engine_stamp` = `port=<src>/<n>;env=<src>/<n>`: the build stamps of `sim_bridge` and the env core (`main.anchors.server.engine_stamp`; empty on Node, on an external server and under `POKESIM_ALLOW_STALE_BUILD=1`; the dry-run plan prints it) | the per-read engine table is `designs/tools/reads.md` §4 |
| the Metamon peer's team pack | only when True | `peer_pack_changed` (`our_peer_pack_changed`) | H22 |
| Foul Play's played team | only when True | `frontend_pack_changed` (`our_frontend_pack_changed`): the front end plays the validator's re-pack, and a validator-accepted file Foul Play may draw plays a different set re-packed than as Foul Play packed it (`main.anchors.frontend_pack`) | H22 |
| the Metamon peer's code | only when NOT a banked one (`side_identity.LEGACY_PEER_CODE`) | `peer_code` (`our_peer_code`): the digest of what the peer executed (`peer_scripts/peer_code.py`): our peer scripts, the Metamon checkout's HEAD and any local change, its interpreter and every installed package | another checkout, env or peer script is a new regime (G14) |
| our policy's load mode | only when NOT `train` | `policy_mode` = `eval` / `train` / `mixed`, read back from the loaded modules (`core_side.policy_mode`); the loader's name is `model_loader` above | `eval` since `141f36f7f`; every banked read played `train` but N1-P look 1 (`eval`) |

**The LEGACY rule.** A conditional row enters the id only when it differs from what EVERY banked read played, so every
banked id is unchanged (`assemble_test._BANKED_ID`) and a new read on the banked weights, engine or pack keeps the same
id and is comparable with them. The price: ids cannot tell two banked engines apart, so every import compares the
engines and the sides themselves (G7, G14). `assemble.LEGACY_WEIGHTS`, `engine.LEGACY_ENGINE` and `side_identity`'s
legacy tables are FROZEN: a repin never edits them.

**Not in the id, so "the same id" is necessary and not sufficient:**
- The playing commit and the harness files (`server_version` is recorded per row). A Python-only change to OUR play
  path between two commits on one engine (the observation encoder, the front end, our packer) moves no id; only
  `repin`'s play-path identity (G8) or a game-for-game guard (G10) sees one. The peer's own Python is `peer_code`.
- **OUR side's pack regime (H24).** A play commit before `d8d674b99` packs forum72's `sft431` Gyarados with
  Frustration at base power 1; from `d8d674b99` on, at 102. No other panel differs. The id does not see it. The read's
  play pin decides it: never split one read's arms across it on forum72, and quote it beside any forum72 number.
- **The X35-frame shared play pin**, `x35-shared-play-86cfdf7e` (FROZEN; `designs/tools/reads.md` §4), the play pin of
  every X35-frame arm read (PR-1′, X62, X63, X48's C0 replay). Its regime:
  - engine E1;
  - Kakuna's pack post-H22;
  - peer code `9cc9caacde85f8ea`, policy mode `train`;
  - OUR side's pack the legacy one (H24, as the arms trained);
  - the front end plays the validator's re-pack, which OUR side's sets survive unchanged.

  Its ids are new on every panel, so no banked row imports into it. A later harness fix reaches it only as a new,
  named pin.
- Our checkpoint: its sha256 is checked per unit by `main.reads` and its step is stamped, but neither is in the id.
- What git does not hold: the Metamon cache, panel team dirs outside the repo, our toolchain and env. Each unit verifies
  the first two at play (weight pins, team digests); the Metamon checkout and env are `peer_code`.
- Seeds, n, wall time.

### 0.2 The guards, and what each proves

A guard turns a silent difference into a named refusal or a VOID unit. "Proves" is what a pass licenses; "not" is what
it leaves open.

| # | guard (code) | fires on | proves | not |
|---|---|---|---|---|
| G1 | per-decision regime check, `regime_verified_decisions` (`peers.py`, `runner.py`) | greedy: the peer's `argmax_match_rate` is not exactly 1.0; t1: not materially below 1.0 (the positive control); no decision at all (the vacuous pass); an ensemble router that drew a random number; our `stochastic` kwarg off the cell's; the load option read back wrong (λ, search stats) | the regime took effect on every decision of both sides | that the peer's process was healthy (G2) or its weights the pinned ones (G3) |
| G2 | peer cleanliness, `peer_clean` (`session.PEER_FAULT_LINE`, `session.watch`, `runner.apply_peer_fault`) | a peer rc ≠ 0, a recorded peer error, or ANY `Force resetting due to long-tail error` line in its log (H5 · H17 · H21): the half fails at the FIRST line (`peer_long_tail_fault`), the unit is VOID and re-run with the same seed | no counted game was played after a peer fault | that the peer's decisions were good |
| G3 | weight pins (`peers.MetamonPeer._check_weight_pin`, `registry_pins_test`) | the peer is online, runs another hub revision, hashes a pinned file differently or loads an unpinned one (`weights_pin_mismatch`); an agent with no pin is refused at plan; a derived config is re-hashed per unit | the opponent's bytes are the registry's | that the registry's revision is what upstream serves today |
| G4 | team pack (the peer: `peer_scripts/team_pack.py`; our side: `utils.teambuilder` → `utils.bridge.team_validator.check_pack_identity`, `gen3_team_pack_oracle_v1`) | a file whose Showdown pack the oracle disputes (the peer exits naming it; our teambuilder raises); a report without a verified pack | each file was played as Showdown reads it | that the file is the team intended (H3, H19) |
| G5 | unit counting, `results.unit_counts` | status ≠ OK, n ≠ games, G1 / G2 / G6 failing, the team condition (§ "Counting a unit") | the unit may enter a pooled read | anything across units: that is the id |
| G6 | forfeit-win guard, `runner.forced_end_verdict` | a forced end other than OUR turn-limit forfeit (a peer's drop or `/forfeit`, our disconnect): the row is dropped (`non_battle_termination`); `blind` (no record, or a record the front end could not write: it is moved aside, `ws_frontend._poison_forced_ends`) fails the unit | no peer drop was booked as our win | anything on `--server node` / `--server-uri` (`unavailable`: the older judgment) |
| G7 | engine identity (`main.reads.engine`) | a group or family spanning two engines is INCONCLUSIVE; an import across engines, or from one nobody can name, refuses; `repin` refuses commits on different engines | one Rust build played the cell | that the Python play path matched (G8) |
| G8 | pin guard and `repin` (`main.reads.runner`, `repin`, `playpath`) | the playing checkout is not `code.json`'s commit (or a repinned one) or its tree is dirty; `repin` accepts another commit only if the tracked play path is byte-identical | the code that played is the registered code | the untracked inputs above |
| G9 | peer-pack import check (`imports.check_peer_pack`) | an import whose counted units on a shared team set ran another pack regime (Kakuna's pack, Foul Play's played team) | no import crosses either split | |
| G10 | game-for-game guard (`rules.guard_games`, `assemble.game_choices`) | a banked unit re-played: any game's 5-tuple, or either player's `CHOOSE`-sequence digest, differs; a capture the choice comparison expects (the bank names its source) missing on either side is `inconclusive`, never a pass on the 5-tuple (a spec's `choices_guard: five_tuple` opts out, recorded) | the cell replays exactly on ONE engine | across engines (§ 0.3): our own policy diverges |
| G14 | the guards RE-DERIVED at assemble (`main.reads.rederive`) | a unit of ANY pin whose capture holds a `FORCELOSE` that is not our turn-limit cap, or whose peer log holds the long-tail line: under the spec's `rederived_guards: count` (every spec registered after 2026-10-27) it does not count; under `report` (the reads registered before, N1-P, H-CTX, X59-S, D3, S-U) counting is unchanged and `decide` HOLDS for the orchestrator | G6 and G2 hold for units a pre-guard pin played | a unit whose captures or peer logs are gone (`unavailable`, judged as before) |
| G11 | G-PREFIX (`main.reads.prefix`) | a searched game and its raw twin differ before the search starts (through `\|turn\|10`): the lane HOLDS and the group stays inconclusive | the pair is a valid pairing | |
| G12 | search replay guard (`main.anchors.search_side`) | the replay of the true input log fails to reproduce the live decision's tokens and row byte for byte: the game fails by name | the searcher saw the state we played | a gain without the leak: searched gains are leak-inclusive |
| G13 | mirrored-pair check (`main.anchors.mirrored`) | a pair whose sides did not play the planned teams in order, or whose two battles did not share a seed: VOIDed with its reason | team-draw luck cancelled inside the pair | |
| G14 | side identity (`main.reads.side_identity`; `peers.MetamonPeer._check_peer_code`, `runner.check_side_halves`) | a peer report without its code record, or whose scripts are not this checkout's, or two halves that differ: the unit is unverified; a group spanning two peer codes or policy modes is INCONCLUSIVE; an import, `import_off`, the determinism guard or `prior_mix_ab`'s bank across either, or from an unknown one, refuses | every counted unit names the peer code and policy mode it played, and no comparison crosses them | that the legacy checkout and env were the same at every banked read (reconstructed from the box on 2026-10-27; reads.md §4) |

Read-level gates (the VOID gate, `guards_not_read`, the reduced read at a registered stop) are `designs/tools/reads.md`
§5 and §6.

### 0.3 Is A comparable with B? In this order

1. Both carry a regime (RULE 1)? A number without one is not comparable with anything.
2. Same id? If not, the differing row of § 0.1 is the regime difference: quote each number beside its own regime and
   never pool them (an import of the other regime's rows is INCONCLUSIVE, or refuses on an engine or pack difference).
3. Same id: walk the "not in the id" list. Same engine by value, same pin (or a `repin` that proved the play path
   identical)? Two commits on one engine with different Python harnesses need a guard unit (G10) on the cells that
   matter. A difference measured through two pins (K2 by its own pin `a5b2fdba`, X26 by `63f97a57`) crosses two
   harnesses; say so beside it.
4. Two engines are comparable only as a recorded, guarded exception (`designs/tools/reads.md` §4's shipped records),
   never as a new read's design: a registered read plays on ONE engine, named at registration.
5. Replay exactness is evidence on one engine only. On one engine a counted Kakuna unit replays exactly; across an
   engine that changed our observation our OWN greedy policy diverges, on a rate that belongs to the engine pair and
   the panel, not to Kakuna (`designs/tools/reads.md` §4 "Replay determinism, measured"). An identity unit or a
   determinism guard (G10) is therefore evidence on one engine and nowhere else.

---

## 1. The three tiers

| tier | when | what | cost |
|---|---|---|---|
| **A — per promotion** | every time a snapshot is promoted | the **dense ladder** (`<run>/snapshot_ladder/ladder.json`), greedy-vs-greedy, compared at **matched snapshot COUNT** | already paid by the run |
| **B — per milestone** | each eval milestone | `metamon:SmallRL` **greedy, BOTH team sets, 100 games each** + **ONE t1 cell** as the distribution check | ~15 min |
| **C — era gate** | opening or closing an era | `metamon:SyntheticRLV2` **greedy, away** — bar: **Wilson lower bound > 0.50**; and **Foul Play** at a fixed `--search-time-ms` with its realized visit count | ~2 h |
| **goal read** | the near-term goal (X22) | `metamon:Kakuna`, Metamon's best public policy (142.8M, ckpt 34, upstream's stated gen3ou GXE ~63%). 🏁 **At the plateau (X51, 2026-10-21, `measurements/x51_plateau_vs_kakuna_2026-10-20/`): P\* (X26 @280,000,170) reads 0.598 [0.549, 0.644] on home72 (goal (a) MET) and 0.487 [0.459, 0.515] pooled on the unfamiliar panels (goal (b) NOT met); Kakuna ckpt 28 is not detectably stronger than ckpt 34 (0.515 [0.466, 0.564]), so ckpt 34 stays the goal's opponent.** home72 and `away` are training data, so quote the goal as "met on our training teams", never as "met"; every cell, comparison and the held-out reads are in § 4. The 2026-09-28 read of N0 (`measurements/kakuna_goal_read_2026-09-28/`, home72) took both Kakuna temperatures (`--opponent-temperature 1.0` and greedy) because the difference was NOT DETECTED: greedy 0.460 [0.392, 0.529], T = 1.0 0.380 [0.316, 0.449] (n = 200 each, goal NOT met); our side stays GREEDY (N0 sampling at T = 1.0 reads 0.310 and 0.280, X22 (f)) | **~2 min / 20 games** on CPU (6 s/game, load ~25) |

### Tier A — per promotion: the dense ladder

**No external process.** The headline strength number stays `<run>/snapshot_ladder/ladder.json`
(dense, ±10), never `eval/elo` (±29). Three standing rules apply unchanged:

* a rating is only final once the RUN is — the newest Bradley–Terry node is systematically inflated;
* a cross-run comparison is at matched snapshot **COUNT**, not matched step;
* `win_rate_vs_pool` / `eval/elo` carry an **opponent-regime boundary at 2026-09-07** and are not
  comparable across it (eval sentinels went greedy and draw the trainee's own teams; the old
  asymmetry was worth **+8.9 pp**). `ladder.json` and every bot edge are UNAFFECTED.

An external anchor is not run per promotion. It is the instrument that says whether the ladder's
axis has drifted, and that question is asked at milestones, not at every snapshot.

### Tier B — per milestone: `SmallRL` greedy on both team sets

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
    --regime greedy --teamset home --games 100 --out <out>/smallrl_greedy_home
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
    --regime greedy --teamset away --games 100 --out <out>/smallrl_greedy_away
# the standing cheap check that determinism is not costing us something — NOT the headline
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
    --regime t1 --teamset away --games 100 --out <out>/smallrl_t1_away
```

**Report BOTH team sets, always.** They are not redundant: they disagreed in SIGN for `SmallRL`
(−0.030 pooled, point estimate on the *away* side). 🚨 **Neither set is unseen by our models** (2026-10-18): `home`
IS the training pool, and 19 of `away`'s 20 teams are exact copies of pool teams. The two sets differ in WHICH
teams are drawn (20 Metamon-chosen sample teams vs 719), not in whether we trained on them. The held-out set is
`heldout72_hpfix` (§2 "The team sets"). `SmallRL` is the recurring anchor because it is cheap (3.0 s/game, ~5 min for 100) and at
0.520 home / 0.650 away it is a **live signal in both directions**, not a floor.

### Tier C — era gates: `SyntheticRLV2`, and Foul Play

```bash
# the era bar: the Wilson LOWER BOUND must exceed 0.50. A point estimate above 0.50 is not a pass.
python -m main.anchors --model models/<run> --opponent metamon:SyntheticRLV2 \
    --regime greedy --teamset away --games 100 --out <out>/synthv2_greedy_away

# the search opponent. The budget is WALL CLOCK, so the realized visit count is part of the result.
python -m main.anchors --model models/<run> --opponent foulplay \
    --regime greedy --teamset home --games 100 --search-time-ms 1000 --out <out>/foulplay_1000ms
```

✅ **A multi-game Foul Play cell WORKS as of 2026-09-22** (`gen3_anchor_serial_challenge_v1`;
hazard **H14** closed). The one-game-per-half limit is gone: `--challenge-mode serial` is the
default and a 10-game cell completes 10/10 on `--server rust`. Every Foul Play number in the
ledger predating this came from runs where Foul Play was the CHALLENGER, so it is not directly
comparable to a role-balanced cell taken now.

`SyntheticRLV2` is the milestone reference because it is *ahead of us* — which is what a reference
should be. Foul Play is a different axis entirely: a search bot, not a learned policy, so it probes
whatever a tree finds and a network does not.

**Read the bar honestly.** `Wilson lower bound > 0.50` is BETTER. A CI that straddles 0.50 is
**NOT DETECTED** — never "equal", and never rounded up to a pass because the point estimate is on
the right side of even.

---

## 2. The exact command, and what it does

```
python -m main.anchors [--model <run dir | .zip | run@step> | --our-side <who>] \
    --opponent {metamon:SmallRL | metamon:SyntheticRLV2 | metamon:Kakuna | foulplay} \
    --regime {greedy | t1} --teamset {home | away} --games N --out <dir>
```

### `--our-side` — WHO plays our half

An anchor read does not have to put one of our checkpoints on the board. Three values:

| `--our-side` | our half is | why it exists |
|---|---|---|
| `model` (default) | the `--model` checkpoint (the core slot, or the live client — see `--our-transport`) | the ordinary read |
| `bot:<name>` | one of the **nine PINNED eval bots** — since P6 (2026-10-07) the **Rust port** of that bot (`src/rust_env/src/bots/`, gated action-equal to the Python bot per decision), deciding on our side's own reading of the battle as a live client (`main.live.bot_reader`) | it puts an external anchor on the **ABSOLUTE** scale without routing through any checkpoint of ours: the bots carry fixed ratings from the bot-vs-bot round robin, so an anchor-vs-bot edge is an edge to a pinned node |
| `metamon:<Agent>` | a SECOND external peer | anchor-vs-anchor — the transitivity check on the fit |

🚨 **A bot cell is stamped `regime_matched = false`, and that is the honest label rather than a
defect.** A bot has no sampling knob to set to the peer's regime — the same shape as Foul Play —
and its policy IS the one the round robin pinned. `--regime t1` with a bot our-side is REFUSED.
`our_regime` on those rows reads `bot:<name>`.

**A bot's randomness is DECLARED (P6).** `--bot-seed S` (default: `--team-seed`) seeds the bot exactly as the env
core's bot route does: half `h` (0 = `ours_challenge`, 1 = `peer_challenge`) draws `random.Random(stream_seed(S, h, k))`
for its `choice` (k = 0) and the stallers' `protect` (k = 1) streams, which RUN ACROSS the half's games (one bot per
half, as one `Player` per half was). Every row stamps `our_bot_seed`; each half's report carries both stream seeds
(`our_bot_streams`). The Python bot drew from the process-wide, UNSEEDED `random` — the same distribution, a
different (unrepeatable) stream. A bot never forfeits at the 250-turn limit (the Python bots carried no stall check,
nor does the env core's bot route); its rows still stamp the plan's `forfeit_turn_limit` and `hit_forfeit_limit`
reads `turns >= limit`.

An anchor-vs-anchor cell IS matched (both peers take `--regime` and both verify it per decision),
and its per-game record comes from **Metamon's own battle CSV** rather than from our own side's
record, because there is no player of ours in the process. That instrument books a TIE as a LOSS
(its `won` field is a boolean), so the rows say so; ties ran 0–1 per 100 games in the 2026-09-14
batteries.

🚨 **The CSV's `Team File` column is NOT the team played** (H-CTX review F1, fixed 2026-10-27, `main.anchors.peer_teams`).
Metamon writes the row after the peer has drawn the NEXT battle's team, so the column names the next draw in 82 % of games
(1,886 of 2,302 captured pair games) and the played one in the rest. A pair cell's per-game `our_team` is therefore the
team OUR peer played, read from its own draw log (`team_draws`), and the row says so (`our_team_from: peer_draw_log`).
Two guards fail the half by name (`peer_team_attribution`): a CSV row that is neither draw `i` nor draw `i + 1`
(a previous attempt's rows appended, a draw that never became a battle), and, when `--capture-dir` caught every battle
of the half, a START command whose six species differ from the team file the log names (an incomplete capture set is
skipped, and the half's report says so). **A pair cell played before this carries the CSV's label, unstamped:** take its
per-game teams from the captures (the team-strength lane and the forensics already do); its win rates are unaffected.

### `--opponent-a X --opponent-b Y` — OPPONENT-vs-OPPONENT (a pair cell)

Two external anchors head to head — e.g. `--opponent-a foulplay --opponent-b
metamon:SyntheticRLV2 --regime greedy --teamset home --search-time-ms 1000` — replacing `--model`,
`--our-side` and `--opponent`. It is the anchor-vs-anchor path above, spelled without pretending
one side is ours and extended to Foul Play: the same server started and stopped by PID, the same
shared team set, the same role-balanced halves, the regime stamp on every row, and hazard **H17**
named for whichever Metamon process challenged in that half. Three facts a reader needs:

* 🚨 **The cell is reported FROM the Metamon side**, whatever order the flags came in — its battle
  CSV is the only per-game record (a tie is booked as a loss there). Foul Play vs Foul Play is
  refused: it has no record.
* 🚨 **Nothing forfeits at 250 turns.** That rule lives in our side's client (the core slot / the live client), which a pair
  cell never runs; the rows carry `forfeit_turn_limit = 0` and the sim's **1000-turn TIE** is the only cap.
  A pair cell is therefore NOT under the forfeit rule its model-vs-anchor neighbours are, and a
  long stall between two non-forfeiting bots needs a `--progress-timeout` above the 900 s default.
* A Foul Play pair cell's `team_source_asymmetry` compares the two PEERS' directories — both must
  be the same files (see **H19**).

### A second CHECKPOINT of one Metamon policy — an agent ALIAS (X51, 2026-10-20)

An agent in `designs/ops/anchors.json` may carry `"policy"`: the Metamon pretrained model it plays, when that
differs from its name. `"Kakuna28": {"policy": "Kakuna", "checkpoint": 28}` is Kakuna at checkpoint 28 (Metamon's
ensemble presets rate it above the default 34). The peer loads the POLICY (`--agent Kakuna`) at the alias's pinned
checkpoint, and the rows' `opponent_version` names what was played (`Kakuna@ckpt28`), never the alias. This is what
lets a pair cell play two checkpoints of one policy: `--opponent-a metamon:Kakuna28 --opponent-b metamon:Kakuna`.
🚨 The self-pair refusal reads `(policy, checkpoint)`, not the names, so an alias with the other side's policy AND
checkpoint is still refused as a policy against itself. A new checkpoint's weights are NOT downloaded by a read (the peer is offline): fetch them once by hand at the pinned
revision and pin their sha256 first (§ Weight pins).

### An ENSEMBLE agent — `metamon:PastaMittens` (2026-10-25)

An agent with `"preset"` + `"preset_sha256"` (and no checkpoint) is a Metamon nickname ENSEMBLE: `PastaMittens` is
the Kakuna router over preset `dt4_k34_midjudge_switchk28` (Kakuna ckpts 28 ×2, 34, 30 + Alakazam ckpt 8). The rows'
`opponent_version` is `PastaMittens@preset:<name>`; the peer refuses to play unless the preset file's entry AND the member
specs the built model will route over both hash (canonical JSON, sha256) to the pin. 🚨 **The router has no argmax**, so its
regime is verified differently: it is DETERMINISTIC (it ignores `sample` and draws no random number), and the peer
checks exactly that per router call — torch's and NumPy's RNG states byte-equal before and after, and no Python
`random` draw from the router's own thread (poke-env's loop thread shares that state and draws mid-call; a plain
state compare blamed the router for 1 call in ~23). The rate is `their_router_no_rng_rate` on every row (exactly
1.0 in greedy); `--regime t1` never verifies for a router. Pinned by
`src/main/anchors/ensemble_router_regime_integration_test.py`. CPU cost ≈ 4× Kakuna (§ Cost per read).

### Weight pins — the Metamon weights are one hub revision, offline, sha256-verified (FR-1 FINDING 3, 2026-10-25)

🚨 **The defect.** Metamon fetches a checkpoint with `huggingface_hub.hf_hub_download(repo_id, filename, cache_dir)`: no
`revision` (the hub's moving `main`) and online. A read could silently load newer upstream weights, changing the
anchor under a regime id that claimed the same policy.

**The rule.** `designs/ops/anchors.json` carries `opponents.metamon.hf_repo`, a FULL 40-hex `hf_revision` (the commit
in the box's cache: `ac8e88e7…`; a branch or short sha is refused) and, per agent, `weights`: `{path in the repo:
sha256}` for EVERY file it may load (a single policy: its one `policy_epoch_<ckpt>.pt`, checked against `checkpoint`; an
ensemble: every member checkpoint, `PastaMittens` = Kakuna 28 / 30 / 34 + Alakazam 8). An agent with no pin is refused at
plan time (`MetamonOpponent.pin`). **The pins belong to the opponent, not to a panel's config**: a config without them (every
sha-pinned panel config that `main.x35.kakuna` passes via `$GEN3AI_ANCHORS_CONFIG`, which must never be edited) takes
`hf_repo` / `hf_revision` and each agent's `weights` from this registry by agent name, when the registry's agent plays the same
policy and checkpoint; pins a config carries itself win; `registry_pins_test` loads EVERY committed anchors config. An agent
a panel config does not name at all (`PastaMittens`, `LargeRL`, `LargeIL`) is played by `main.x35.kakuna` (and so
`main.reads`) under a DERIVED config written once at plan: the committed panel config plus that one agent, exactly the
registry's entry; each unit re-checks the derived file's sha256 and that it is the committed config plus that agent
(`kakuna.opponent_config` / `check_derived_config`). The peer (`peer_scripts/metamon_side.py`, helpers in the stdlib-only
`peer_scripts/weight_pins.py`, shared with `main.anchors`):

1. forces `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` at import, over whatever the caller exported (the plan's env sets them too),
   and refuses unless the imported hub library really is offline;
2. replaces `huggingface_hub.hf_hub_download` with a downloader that serves ONLY a pinned file of the pinned repo, adds
   `revision=<pin>, local_files_only=True`, and hashes the file it returns;
3. hashes EVERY pinned file through it BEFORE the model is built, and exits `REFUSING TO PLAY` on a missing file, an
   unpinned file, a revision absent from the cache or any sha256 difference;
4. reports `hf_hub_offline`, `hf_revision`, `weights_verified`, `weights_loaded`; `MetamonPeer.read_report` FAILS CLOSED on
   them (no report of the pin, a different revision or sha, not offline, or a load outside the pin ⇒
   `regime_verified_decisions = false`, so `unit_counts` does not count the unit; the reason is `weights_pin_mismatch`).

Every row and summary stamps `hf_revision` and `weights_sha256` (`main.anchors.peer_scripts.weight_pins.weights_digest`
of the verified `{path: sha256}`), and `our_hf_revision` / `our_weights_sha256` for a Metamon our-side peer. Teams are
also fetched from the hub by Metamon, but only into the cache (their content digest is already on the cell); a team set
absent from the cache now fails offline instead of downloading.

**To pin or move a checkpoint** (a network step, not a read; the orchestrator's go): fetch with
`hf_hub_download(repo_id, filename, revision=<full sha>, cache_dir=<cache>/pretrained_models)`, `sha256sum` the blob
(the blob's file name is its sha256), put revision and sha in `anchors.json`, and play a smoke. A new revision is a NEW regime.

**Regime ids.** The weights enter the id only when they are not the legacy ones (§ 0.1): a new revision or digest is a
new regime, while a read of the banked commit `ac8e88e7…` keeps the banked id and is comparable with the banked reads.

**The engine stamp** (REVIEW F F-M2, 2026-10-26). Kakuna games run on OUR in-process Rust engine, and the commit in
`server_version` does not say which engine that was (the 10-26 parity batches changed OU-visible mechanics), so each
Rust-served cell stamps `engine_stamp` (`main.anchors.server.engine_stamp`). Its place in the id and the import rule are
§ 0.1 and G7; the per-read engine table is `designs/tools/reads.md` §4.

### `--our-teamset {home,away}` — an ASYMMETRIC pair cell (B-D9, 2026-10-18)

PAIR CELLS ONLY (a `metamon:` our-side, or `--opponent-a/-b`): our peer draws from the Metamon
config's `team_sets[--our-teamset]` while the opponent peer keeps `--teamset`. Default: the same as
`--teamset`. Refused, with its reason, for a checkpoint or `bot:` our-side (those draw from
`our_team_sources[--teamset]`, so the flag would be stamped and never applied).

* 🚨 **It is the one way to play a policy against ITSELF**: the self-pair refusal is lifted only
  when `--our-teamset` resolves to a DIFFERENT Metamon team-set directory than `--teamset` — the
  strength calibration K(P, H), e.g. `--opponent-a metamon:Kakuna --opponent-b metamon:Kakuna
  --teamset home --our-teamset away --declare-asymmetric-teams`. The same agent on one set stays
  refused. The two sides then log in under DISTINCT names, each carrying the set it pilots
  (`MetaKakunaAway1` / `MetaKakunaHome1`, ≤ 18 characters with the per-half suffix); each peer writes
  its own battle CSV under its own `--results-dir`, and the cell reads OUR peer's.
* Stamped on the plan (`--dry-run` prints `OUR peer draws <set>`), on every row (`our_teamset`), and
  in the default out-dir slug (`…_ours-<set>`).
* 🚨 **The two role halves REPLAY ONE team sequence** (measured 2026-10-18, the team-strength read's first unit):
  with one `--team-seed`, battle `10 + i` plays battle `i`'s two teams again with the SEATS swapped, on another
  battle seed. A 20-game unit is 10 matchups × 2 seats, not 20 independent draws. A set-level win rate is unaffected;
  a per-team or per-matchup bootstrap must cluster by (unit, matchup), and a game-only interval is optimistic.
  `main.team_strength` clusters this way (F-DV-18 of `design_diversity_teams_opponents.md`).
* **Which team each side played** comes from `--capture-dir`: each battle's START command carries both packed teams
  and both names (`…Away…` = `--our-teamset`, `…Home…` = `--teamset`), and its log names the winner.
  `main.team_strength.kakuna_lane.parse_unit` reads it and cross-checks every winner against `games.jsonl`.

### `--declare-asymmetric-teams` — a DECLARED asymmetric cell (B-D6)

Stamps `declared_team_cell: "asymmetric"` on the plan and the summary (default `"symmetric"`). It
changes NO play: it says the two sides draw from different team sets ON PURPOSE, which decides how
the unit is counted.

### Counting a unit — `main.anchors.results.unit_counts` (THE one rule, B-D6)

Every summary stamps each side's team-set CONTENT digest, `our_team_digest` / `their_team_digest`
(`gen3_team_set_digest_v1`, `main.anchors.session.team_set_digest`: sha256 over the sorted
sha256s of every allowlisted team file's `.strip()`ed text — names, extensions and order never
enter it, so a `.txt` copy and a `.gen3ou_team` copy of one set digest the same), and
`team_content_asymmetry` (the digests differ; null when either is unknown) beside the count-based
`team_source_asymmetry` (unchanged). `unit_counts(summary, *, games, declared_digests=None,
verified_digests=None) -> UnitCount(counts, reasons)` counts a unit iff `status == "OK"`,
`n == games`, `regime_verified_decisions is True`, `peer_clean is True`, no game of it ended by a non-battle
termination (below), and:

* **undeclared** (`declared_digests=None`): `team_source_asymmetry is False` and
  `team_content_asymmetry` is not true — a 72-vs-72 cell of DIFFERENT teams no longer counts on its
  equal counts (F-DV-5);
* **declared** (`declared_digests=(ours, theirs)` from a registration): the summary's own digests
  (else the caller's `verified_digests`, computed from file sets it sha-checked itself — a summary
  from an older pinned tree carries none) must EQUAL the declared pair; no digest at all ⇒ it does
  NOT count. `team_source_asymmetry` is not a reason here, so a declared 72-vs-16 cell CAN count.

**The forfeit-win guard (2026-10-27, after the S-U build's negative control booked a peer's keepalive drop as OUR
win).** After a FORCELOSE the sim emits the same `|win|` batch a played-out win does, so only the front end can tell
them apart. Our front end records every FORCELOSE that decides a battle (`ws_frontend.FORCED_END_SCHEMA`: the battle,
the slot, the user, `cause` = `forfeit` or `disconnect`, the turn) in `<out>/forced_ends.jsonl`, in both placements
(in process and as our subprocess). `main.anchors.runner.forced_end_verdict` then allows exactly ONE forced end as an
outcome, our side's own turn-limit forfeit (the cap, a row stamped `hit_forfeit_limit`). Any other (the peer's drop or
`/forfeit`, our side's drop, a forfeit before the cap) is VOID: its row is dropped, the summary lists it in
`non_battle_endings` and fails with `non_battle_termination`, and the unit never counts. `forced_end_guard` is `armed`
(the record was read), `blind` (our front end served but no record: FAILED, `forced_end_guard_blind`) or
`unavailable` (`--server node` / `--server-uri`: the guard cannot see a forced end there); a summary that carries the
field counts only when it is `armed`. An older summary, without it, is judged as before HERE; a registered read
re-derives this guard and G2 from the unit's captures and peer logs at assemble (G14, `main.reads.rederive`). A
forced end the record cannot store moves the record aside, so the guard reads `blind` (fail closed). The live
negative control is `peer_patience_integration_test.py` (slow); `guard_wiring_test.py` pins the wiring fast.

Every failing reason is listed. `main.x35.kakuna` counts through it (undeclared), and so does a registered read's
anchor lane (`python -m main.reads`, `designs/tools/reads.md`), which plays kakuna's units and records each unit's
regime id.

### `--opponent-temperature T` — the Metamon peer SAMPLES, our side stays greedy

The one sanctioned MIXED Metamon cell (added 2026-09-28 for X22(b), "at which temperature is
Kakuna strongest?"): `--opponent metamon:<Agent> --regime greedy --opponent-temperature 0.5`. The
peer runs `t1` at `action_temperature` T and verifies it samples (`argmax_match_rate` < 1); our
side is greedy exactly as in a greedy cell. Every row is stamped
`regime_matched = false`, `their_regime = sample:T=<T>`. It is refused with any other opponent,
our-side or `--regime`, and for T ≤ 0 (greedy is `--regime greedy` alone — H7). It is **not** the
recurring protocol (RULE 2) and it does NOT reopen `--allow-unmatched-regime` against metamon,
which stays refused; hazard **H17** (the Metamon-challenges post-game `RecursionError`) is expected
in this shape, and a unit that hits it is VOID and re-run like any long-tail fault (§ 3.1).

### `--our-temperature T` — OUR checkpoint SAMPLES (the mirror)

Added 2026-09-28 for X22(f) ("does our model play better greedy?"). Same one shape — a metamon
`--opponent`, a checkpoint our-side, `--regime greedy` — and it composes with
`--opponent-temperature`: our side samples at T (`core_side.our_stochastic`), the peer follows `--regime greedy` or
its own `--opponent-temperature`. Every row is stamped `our_regime = sample:T=<T>`;
`regime_matched` is true only when both nominal temperatures are equal. Our half is VERIFIED per
decision: every decision must receive `stochastic=True` (else `regime_verified_decisions = false`),
and each row carries `our_argmax_matches` / `our_argmax_decisions` — the chosen action against the
argmax of that same decision's masked logits — pooled to the summary's `our_argmax_match_rate`
(OUR side of the §0 argmax table). Seed our sampling with `$GEN3AI_POLICY_SEED` for a repeatable
series. Refused with any other opponent, our-side or `--regime`, and for T ≤ 0.

### `--our-belief-prior-mix L` — OUR opponent belief anchored to its Smogon prior

Added 2026-10-26 (`agents.model.belief_prior_mix`, D0f at L = 1). A MODEL-LOAD option of a checkpoint our-side: the
loaded policy's species presences, move posterior and item posterior become `(1 − L)·b + L·prior` before anything
reads them (inference only). The loader reads L back from the policy into the half's report
(`our_belief_prior_mix`); a mismatch with the cell's L sets `regime_verified_decisions = false`. Every row is stamped
`our_regime = <regime>+belief_prior_mix=<L>` (so it never pools with the unchanged model); L = 0 is the unchanged
model and stamps nothing. Refused for a bot or peer our-side and outside [0, 1]. `main.x35.kakuna` units pass it
from a final's `belief_prior_mix`, `main.reads` from a model's `load`.

### `--our-search PRESET[@TURN]` — OUR checkpoint plus a registered one-ply search (S-U, 2026-10-27)

`main.anchors.search_side`, for S-U (`designs/endstate/design_search_u_headroom.md`). At each of our decisions from
turn TURN (default 10) with ≥ 2 legal actions, the checkpoint plus the registered search (`main.anchors.search_presets`;
`r0s4` is R0 §4's: 8 honest worlds `dice="seed"`, every legal action played once in each world 4 turns deep then the
critic, piKL λ 0.015) picks the action; every other decision is the checkpoint's greedy choice, exactly the raw
cell's. The search replays the battle's TRUE input log in process, cut honestly (the opponent's choice of our own
update is never in it) and guarded: the log's replay must reproduce the live decision's tokens and row byte for byte,
else the game fails by name. A searched gain is LEAK-INCLUSIVE (the worlds replay the true seed, `search_ledger.md`
§6.1). Every decision is logged to `<out>/search_decisions.jsonl` (`python -m main.reads search-cost` reads it). Needs a
checkpoint our-side on the core slot, `--regime greedy` with no `--our-temperature`, and a `--seed-base`. Stamps
`our_regime = <regime>+search=<preset>@<turn>`. The search leafs on rust_env's RELEASE `libpokesim_env.so`, which only
`scripts/bootstrap.sh --with-rust` builds (a selfcheck build is not it): `main.anchors` loads it, stale-build guard included,
while it plans, and refuses by name BEFORE any unit starts (`--dry-run` prints the refusal and goes on). Unchecked, each
unit died with rc 2 about 6 CPU-s in.

**The peer's patience (S-U FINDING S-3, read and tested 2026-10-27).** Our front end has no battle timer, and the
Metamon peer waits on its observation queue with no timeout; its ONE clock is poke-env's websocket keepalive (a ping
every 20 s, the connection closed when a pong is 20 s late). A searched decision (30-90 s) therefore runs OFF the event
loop, in a worker thread, so the front end keeps answering pings: a 70-s off-loop decision kept Kakuna connected
(`src/main/anchors/peer_patience_integration_test.py`, slow tier). A decision that BLOCKS the loop that long drops the
peer, and the front end then forfeits it; the sim books OUR win, which the forfeit-win guard (§ Counting a unit)
voids: never decide on the loop. Our own watchdog is the other
clock: size `--first-game-timeout` / `--progress-timeout` for a searched game (`main.x35.kakuna` does).

### `--deal-game G` — ONE game of a seeded series (S-U's per-game units)

Plays only game G (0-based) of the `--games` series: its half (the role), both sides' team draws (our team source and
the Metamon peer's seeded draw, `metamon_side.py --team-skip`, each skip the half's earlier draws) and its battle seed
(the front end counts battles from G, so game G gets `derive_seed_from_base(--seed-base, G)`). The summary counts 1
game and records the deal under `provenance.deal`. Needs a metamon `--opponent`, the core slot, a `--seed-base` and no
`--mirrored-pairs`. `main.x35.kakuna` plans a `per_game` read as 20 such units per seed (`designs/tools/reads.md`).
**Verified (2026-10-28, against Kakuna on the engine `e00a212a…/3d0d5c9c…`):** a lone game G replays game G of a whole
20-game unit byte for byte. Two null branches (`--our-force` with π's own action) reproduced games 0 and 10 of the S-U
gate's units whole: the same protocol, result and length (`designs/research_state/measurements/dlive_read_2026-10-28/PREREG.md` §9). A one-game unit's
battle TAG is still `battle-gen3ou-1`, because the front end's tag counter is per process; match a game by its series
index, never by its tag.

### `--keep-input-logs FROM-TO` — a RAW cell keeps its decisions' input logs (the S-U gate test, 2026-10-27)

`main.anchors.input_log`. A raw checkpoint our-side on the core slot (with a `--seed-base`, never with
`--our-search`) plays exactly as without the flag; at each of our decisions in turns FROM..TO with ≥ 2 legal actions
it keeps the battle's honest input log, cut exactly as the live searched player cuts it (`search_side.honest_log`),
with the decision's live search key, tokens, greedy choice and the live row's sha256, in `<out>/input_logs.jsonl`.
The offline scorer (`main.value_search.sign_gate`) re-runs the registered search there and refuses any kept log whose
replay does not reproduce the decision's tokens and row. No regime stamp (play is unchanged); the half's report carries
the recorder's counters under `input_logs`, and a window that did not take effect fails G1. `main.reads` passes it as
the label's `load.input_log`.

### `--our-force SPEC` — a FORCED BRANCH of a raw game (the deviation-only live read, 2026-10-28)

`main.anchors.forced_branch`. A `--deal-game G` unit of a RAW greedy checkpoint on the core slot (no `--our-search`,
no `--keep-input-logs`, no `--our-temperature`) replays game G of the raw series and plays the checkpoint greedy at
every decision but ONE: the spec's named decision (turn, core frame index), where it plays the spec's action, then
greedy again. The spec (`gen3_forced_branch_v1`) is built from the raw game's KEPT decision (`--keep-input-logs`): its
key, the series index G, the action, π's greedy action, the choice tokens, the live row's sha256 and the sha256 of the
honest input log at that decision. **G-BRANCH**, live, before the action is sent: the branch game's series index,
honest-log digest, tokens, row digest and greedy action must all equal the raw decision's, or the unit fails by name
(the record in `<out>/branch.jsonl` says which); a game that never reaches the decision fails the half's verification.
The battle TAG is a per-process counter (a one-game unit's first battle is `battle-gen3ou-1` whatever G is), so it is
never compared. Stamps `our_regime = <regime>+force=<kind>:a<action>@t<turn>n<n>` (`kind` = `branch`, or `null` when
the action is π's own: a replay control). `main.value_search.live_branch` plans these units for `main.x35.kakuna run`.

### `--mirrored-pairs` — MIRRORED TEAM PAIRS (T17, `gen3_mirrored_pairs_v1`)

Within each half, battle `2k` pits our team `A_k` (drawn from OUR team source) against the peer's
`B_k` (drawn from ITS team set), and battle `2k+1` HANDS THE TEAMS OVER — we pilot `B_k`, the peer
pilots `A_k` — on ONE seed: the front end (`--pair-seeds`) keys each battle's seed by its UNORDERED,
canonicalized team pair and that pair's occurrence // 2. Team-draw luck cancels inside the pair.
**Needs** `--server rust` started by this tool, a `metamon:` opponent (its driver plays a planned
`--team-sequence`), a checkpoint or `bot:` our-side, and `--games` a multiple of 4; anything else is
REFUSED. With no `--seed-base` the read is seeded with `--team-seed` (printed).

**Nothing is trusted on faith.** A pair counts only when BOTH sides played the planned teams in the
planned order (our draw log; the peer's `team_draws`), both battles finished, and the two shared one
seed (the front end's `pair_seeds.jsonl`); otherwise it is VOIDED with its reason. A battle our side
forfeits at the turn limit is a DRAW (the trainer's convention). The summary's **PAIR SCORE** carries the
pentanomial 95% interval over verified PAIRS — the read's headline; the Wilson line is labelled per-game,
NOT the unit. Every row stamps `mirrored_pairs`: a mirrored cell half-plays our side on the PEER's team
draw, so it is never quoted beside an unmirrored one. Every team file handed to the peer spells out its
gen-3 Hidden Power IVs (`mirrored.with_hp_ivs`): the first mirrored read died on the peer's Hidden Power rejection
(2026-10-01). Since H22 the peer packs every file through Showdown anyway, so the IV line is belt and braces.

### `--model-load {auto,bare,foreign}` — HOW the checkpoint is loaded

🚨 **A cross-run frozen snapshot FAILS a bare `MaskablePPO.load`.** The extractor is rebuilt from
the zip's own `policy_kwargs` and handed to the CURRENT `ExtractorBuild`, so every
`ai_v9_29_rev1_0823` node dies with `unexpected keyword argument 'threat_prob_outspeed'` while
every `ai_v12`/`ai_v13` node loads fine — a campaign spanning eras hits it on SOME cells only.
`foreign` uses `load_foreign_opponent`, which reads the zip's own `model_config.json` and checks
the **`arch_signature`** (observation-family compatibility, the property that actually matters for
a frozen opponent). `auto` tries `bare`, falls back, and PRINTS which ran; the winner is stamped as
`model_loader` on every row.

⚠️ **`MIGRATION_FLOOR` is a hard wall, not a hint.** A checkpoint below it — every `ai_v8` snapshot,
config version 44/45 under `ARCH_SIGNATURE = gen3_opp_hp_typed_candidates_v1` — is refused as
PRE-GENERATION by BOTH loaders. There is no way to play a pre-generation node with current code, so
an era-spanning campaign's oldest reachable node is whatever the floor admits.

| step | what happens |
|---|---|
| 1 | starts a server on an auto-picked **9500–9599** port, and stops it on exit or failure. **8000 and 8001 are refused in code.** 🚨 **That server is the in-repo websocket FRONT END over the Rust bridge (`--server rust`, the DEFAULT): NO Node process is started at all.** With our CORE slot (the default, see `--our-transport` below) it runs **inside the anchors process** — closed, and every battle child reaped, at the end; with the live client (a `bot:` our-side) it is a subprocess, PID recorded and **exactly that PID** stopped. `--server node` is the explicit opt-out that starts `deps/pokemon-showdown`; `--server-uri` uses an existing server, starts nothing, and stamps the rows `server_impl=external`. The transport and its version are on **every row** |
| 2 | resolves `--model` through `resolve_model_ref` — 🚨 **a bare run directory means that run's LAST SNAPSHOT**; name the `.zip` or `@step` to pin a file. The rung it resolved by is recorded on every row |
| 3 | plays our checkpoint as an **in-process slot of the front end on the Rust core's own row** (the default), or as a **websocket client on the Rust stack** (`--our-transport live`: `--server node`, `--server-uri`, a `bot:` our-side) — no poke-env on our side either way (`--device cpu`, greedy, the trainer's 250-turn forfeit limit — `--forfeit-turn-limit N` may only LOWER it, for a deliberately shorter series such as the capped smoke `anchors_integration_test::test_one_capped_real_game_against_metamon_smallrl` (`slow` since 2026-10-19), never a strength read; every row stamps `forfeit_turn_limit`, and a forced forfeit is H5's trigger), role-balanced across two half-series |
| 4 | verifies the regime **per decision on both sides**, writes the `argmax_match_rate`, and records the verdict as **two** fields — `regime_verified_decisions` (the regime) and `peer_clean` (the exit). Never one; see hazard **H16** |
| 5 | writes `games.jsonl` + `summary.json` with the Wilson CI and the full provenance |
| 6 | on a dead peer or a stalled series, **FAILS with a named cause** and exit code 2 |

🚨 **`--out` has NO relative default.** With it omitted the read goes to a **run-scoped** directory
`<root>/<YYYYmmdd_HHMMSS>_<opponent>_<regime>_<teamset>` under `$GEN3AI_ANCHORS_OUT_ROOT` (default:
the system temp dir), and the path is PRINTED before anything starts. The old default was
`./anchors_out`, and an anchor read is taken from the MAIN checkout because that is the only tree
with `models/` — so the default filled the repo it was measuring, one `git clean` from a lost
measurement and one `git status` from a landing that stops. An EXPLICIT `--out` is taken verbatim,
because a directory under `designs/research_state/measurements/` is a deliberate destination.
**Pass `--out` for anything you intend to bank.**

Other useful flags: `--dry-run` (prints the plan — the transport, both peer commands verbatim,
both team sources with their counts, every deadline — and starts nothing), `--show-config`
(resolves `designs/ops/anchors.json` and says which paths exist), `--games`, `--search-time-ms`,
`--progress-timeout` / `--first-game-timeout` / `--peer-ready-timeout`, and — **`--server rust`
only** — `--seed-base` / `--capture-dir`, the reproducibility pair (the Node server mints its own
seed per battle and has no capture, so both are REFUSED there rather than ignored).

### The transport — `--server {rust,node}`

| | **`rust` (the DEFAULT)** | **`node` (the opt-out)** |
|---|---|---|
| what runs | `utils.bridge.ws_frontend --impl rust` — IN the anchors process with our core slot (the default), its own subprocess with the live client; each battle is backed by ONE `sim_bridge` child | `deps/pokemon-showdown` — a whole Showdown server |
| memory | **40 MB mean / 166 MB peak** server tree over 100 games | **3,227 MB mean / 3,551 MB peak** — 80× the mean |
| wall, 100 games | **126 s** | 203 s |
| needs `node` on PATH | only for the `/utm` team validation (a short-lived process per distinct team, cached) | yes, plus the submodule's `dist/` + `node_modules/` |
| reproducible | `--seed-base` / `--capture-dir` | no counterpart |
| stamped on every row | `server_impl=rust`, `server_version=ws_frontend[in-process]@<head>+rust:<bridge binary>` (core slot) or `ws_frontend@…` (subprocess) | `server_impl=node`, `server_version=showdown:<pin>` |

**They agree.** 100 games of `metamon:SmallRL` greedy/away at one team seed, arm W @ 75,005,952:
**0.590 [0.492, 0.681]** through the front end against **0.580 [0.482, 0.672]** through Node,
**Δ +0.010 [−0.124, +0.144] — NOT DETECTED** (`anchors_rust_frontend_2026-09-20`, on top of the
200-battle side-by-side in `foulplay_axes_and_frontend_validation_2026-09-16`). 🚨 **An
agreement at n = 100 is not byte equality**, and the two paths are not replicates — the instrument
for protocol equality is `src/utils/bridge/ws_frontend_byte_identity_integration_test.py`, which
compares the front end's per-side bytes against the Node bridge's and is green on both arms.

**When to reach for `--server node`:** a differential like the one above, and anything the front
end deliberately does NOT implement (`designs/rust_sim/ws_frontend.md`'s deferral list —
reconnection, a battle timer, `/search`, replays). Nothing in the standing procedure needs it.

### `--our-transport {auto,core,live}` — WHO BUILDS OUR OBSERVATION (P3 + P6, 2026-10-07)

| value | our side is | needs |
|---|---|---|
| `core` | an **in-process slot** of the Rust front end (`main.anchors.core_side`): it decides on `sim_bridge`'s core frame — the row, mask and choice tokens training's reader builds — and sends `/choose <token>\|<rqid>` into the front end by function call | `--server rust` started by this tool, a checkpoint our-side |
| `live` | a **websocket client on the Rust stack** (`main.anchors.live_side` over `main.live.client`): the checkpoint decides on `main.live`'s reader frame (the `live_reader` binary: the SAME `SideReader` chain `sim_bridge`'s core frame comes from) with the core slot's own arithmetic; a `bot:` our-side is the Rust port of that bot deciding on our side's reading (`main.live.bot_reader`, the `bot_reader` binary) | — (serves `--server node`, `--server-uri`, a `bot:` our-side) |
| `auto` (DEFAULT) | `core` wherever it can serve, else `live`; the plan PRINTS which and why | — |

**No side of ours imports poke-env** (`src/poke_env_free_entry_points_test.py`: the import closure of the whole
package, a blocked Rust-bot battle; the slow reads run every transport under the import blocker). The legacy
`main.play` → `RLPlayer` client (`--our-transport poke-env`) was **DELETED in P6** and a typed one is refused with
its reason (`designs/deleted_flags.md`); so was poke-env's pipelined challenge loop (`--challenge-mode pipelined`).

Every row stamps **`our_transport`**: `rust_core_slot` · `rust_live_reader` · `rust_live_bot` · `peer`. History: a
row written before 2026-10-07 has no field and was `poke_env_rlplayer` (or a bot / peer by its `our_side`); rows
from 2026-10-07 until P6 also wrote `poke_env_rlplayer` / `poke_env_bot` for the legacy client.

**They play the SAME games** (byte-identical protocol and choices on both paths):

* **P3, the core slot vs the legacy client:** 100 / 100 seeded battles of one checkpoint vs `metamon:SmallRL` (11,108
  choices, shift 0.000, 0 protocol failures; `measurements/pokeenv_p3_anchors_2026-10-07/`).
* **P6, the live client and the Rust bots vs the legacy client** (`measurements/pokeenv_p6_anchors_2026-10-07/`): all
  nine roster bots, 12 seeded games each, the old path's Python bot seeded as the Rust bot is: 108 / 108 byte-identical
  (a seed-mismatched TEETH cell differs on 12 / 12). The checkpoint on the live client: 19 / 20 byte-identical; the 20th
  diverged when METAMON forfeited at its first decision on the old path only (H5's desync after our stall forfeit,
  timing-dependent inside the peer). On a NODE Showdown (no seed), with the live reader as a SHADOW: 1,347 compared
  decisions + 2 stall forfeits, 0 row / mask / action / token differences, and 0 protocol failures on Node and on a
  master-built Node via `--server-uri`.

So the transports are one behaviour and `main.anchors` refuses no mix, but `our_transport` IS part of `main.reads`' regime id (§ 0.1), so a registered read never pools two transports in one cell. ⚠️ Two fields changed meaning:
a legacy row's `n_decisions` is CUMULATIVE over its half (a pre-existing `RLPlayer` counter; `RLPlayer` is deleted, T27 P6) where every Rust-stack
row's is that game's; and a legacy `poke_env_bot` row's bot drew from the process-wide, UNSEEDED `random` where a
`rust_live_bot` row's draws are the declared `--bot-seed` streams (`our_bot_seed`).

What our side does NOT do on either transport: re-decide or fall back to a default move (it decides exactly where
the reader opens a decision, so there is nothing stale to re-decide; `n_redecides` / `n_defaults` are 0 by
construction). Any protocol surprise FAILS the read by name with the games that finished attached: on the core slot
as `core_slot_error` (a live request with no frame, a frame nobody consumed, a NaN cell, a masked choice,
`[Invalid choice]`, a popup); on the live client as **`live_parse_halt`** — P4's T28 HALT (a reader refusal, an
illegal choice, `[Invalid choice]`): the halt marker is RECORDED (`python -m main.live.halt`; every live entry point,
this one included, refuses to start past it) and the read exits `FATAL_LIVE_PARSE` (7) — or `live_client_error` (a
login that never completes, a rejected team, a dropped socket).

### The team sets

| | **home** | **away** | **heldout72_hpfix** (a config, 2026-10-18; the held-out set of record) |
|---|---|---|---|
| what | our 719-team gen3ou pool, nickname-free, with explicit Hidden Power IVs (or `home72`: 72 of those pool teams, sanitised) | Metamon's 20-team `competitive` gen3ou set, each file `.strip()`ed | 72 teams drawn with a fixed seed from Metamon's replay-derived `hl_05_26/gen3ou`, rejecting any with an exact team match, or ≥ 4 of 6 mon sets shared with one team, in our pool or `competitive` (`heldout72`); then each untyped all-31 Hidden Power (= HP Dark, the replay artifact, 164 / 432 mons) re-sampled by `agents.gen3_data.hp_repair` (Smogon species usage on the set's side, the validator's IVs) |
| why | it is our training distribution | the set Metamon's README names as its human-ladder evaluation set and says Metamon has overfit to (`metamon/README.md:865`) | the one set NEITHER side trained on as a set |
| 🚨 **in OUR training pool** (`gen3_team_overlap_v1`) | all of it | **19 / 20 teams EXACT copies** (119 / 120 mon sets) | **0 / 72** exact, 0 sharing ≥ 4 mon sets, 164 / 432 mon sets (heldout72: 104) |
| in Metamon's `competitive` | home72: 8 / 72 exact | all of it | 0 / 72 exact |
| how to run it | `--teamset home` | `--teamset away` | `--teamset home` with `$GEN3AI_ANCHORS_CONFIG` = `measurements/heldout72_hpfix_kakuna_2026-10-18/anchors_heldout72_hpfix_kakuna.json` (Metamon's cache set `gen3ai_heldout72_hpfix`) |

🚨 **`away` is NOT held out from us.** 19 of its 20 teams are exact copies of teams in our 719-team training pool (23 pool
entries; the pool holds four duplicates): it is Metamon's choice of teams, which Metamon's README says it has overfit
to. `home72` is 72 pool teams, so it is training data too, by design. Every `away` number stands as a measurement (only
its description was corrected, 2026-10-18), and X26 − K2 comparisons on it stay valid because both models trained on
the same pool. The held-out read is `heldout72_hpfix` (`measurements/heldout72_hpfix_kakuna_2026-10-18/`).
🚨 **A held-out read uses the HP-REPAIRED set:** `heldout72` (`measurements/heldout_teams_kakuna_2026-10-18/`) is the same
72 teams with the archive's Hidden Power Dark fill on 164 / 432 mons, a known-wrong move on 71 of them; it stays as a
record, not a read target. A NEW held-out set from replay-derived teams is repaired the same way
(`hp_repair.repair_team`, a seed per team id, recorded) before any game. Kakuna's own exposure to those teams is in the
heldout72 README's caveat.

### The evaluation MATRIX and THE external headline (ADOPTED 2026-10-18, the orchestrator; Q1 revised)

**No team set is neutral**, so no number is quoted as if one were. Every Kakuna read is quoted with its row of the
FAMILIARITY MATRIX: home72 leans toward us, `away` toward Kakuna, heldout72 toward Kakuna, smogon_forum is
near-neutral and P72 is unclear.

- **THE external headline is L:** the LADDER-WEIGHTED deployment estimate. Our side pilots home72, and Kakuna's side
  pilots heldout72_hpfix teams by species stratum, weighted by `ladder_weights_vN` (v1: CORE 0.176 / NEAR 0.675 /
  FAR 0.149, from 22,862 `hl_05_26` teams).
- **Beside it, always:** the sample ground (home72, `away`), the unfamiliar panel (heldout72_hpfix, smogon_forum,
  P72), the in-house meter GEN-M, and L_c1.
- The design is `endstate/design_diversity_teams_opponents.md` §5.
- 🏁 **Standing (X37a stage 1, 2026-10-18, `measurements/x37a_eval_matrix_2026-10-18/`, PROVISIONAL): L
  (ladder_weights_v1) = 0.630 [0.573, 0.686]** for X26 @161,000,169 vs Kakuna greedy (D_CORE 0.610, D_NEAR 0.623,
  D_FAR 0.690; stratified team-clustered bootstrap); L_c1 = 0.624 [0.570, 0.676]. L leans toward us: Kakuna's
  deployment teams are usage-filled replay reconstructions, weaker in anyone's hands (K(P, H) = 0.573).
- **The asymmetric cells and how to run them:** a per-cell config whose `our_team_sources.home` and Metamon
  `team_sets.home` name DIFFERENT sets, declared in a registration and counted by `unit_counts` against the
  registered per-side digests; the Kakuna-vs-Kakuna calibration with `--opponent-a/-b metamon:Kakuna --our-teamset away
  --declare-asymmetric-teams`. Pass `--capture-dir` to keep the opponent's team per game (the rows do not carry it).
- **The unfamiliar panel's other two members are frozen** (X37a stage 2, `measurements/x37a_eval_matrix_2026-10-18/`):
  `forum72` (Metamon set `gen3ai_forum72`, `forum72_manifest.json`) and `P72` (`gen3ai_p72`, `p72_manifest.json`), each
  run like heldout72_hpfix with its own config in that directory's `configs/`. X26 reads 0.470 and 0.453 on them:
  the held-out drop is about teams unseen by us, not about the replay population.

### The team-set overlap — a REQUIRED field of every read (2026-10-18)

`main.anchors` measures how much of the read's team set our training pool and the opponent's KNOWN team sets
already hold (`src/main/anchors/team_overlap.py`, `gen3_team_overlap_v1`), BEFORE any game, prints it in the plan
and the result, and writes it to `summary.json["team_overlap"]`. A read whose team set cannot be measured fails
before anything starts.

* **Matched on** each mon's normalised species + sorted moves (the `mon` tier), and a `strict` tier that adds
  item, ability, nature and EVs. Hidden Power is matched by its TYPE: a typeless `Hidden Power` takes the type its
  IVs give (all 31 = Dark). A forme is matched by its own id, except a forme with its base's stats and types (the
  Unown letters), which is the base. Nothing is keyed by `num`.
* **Counted, per reference:** `exact_team` / `exact_team_strict`; `same_species6`; `mons_seen` /
  `mons_seen_strict`; and per read team the most mon sets shared with any ONE reference team
  (`max_shared_mons_hist`, `teams_sharing_ge4`).
* **References:** `training_pool` (this checkout's `data/teams`) always; for a Metamon opponent, the config's
  `opponents.metamon.reference_team_dirs` (`competitive` and the archived `hl_05_26`), or every non-`gen3ai_*` set in
  its cache when the config names none. A configured reference that is absent on the box is listed under
  `missing_references`, never a silent zero. Foul Play has no team set of its own.
* **Quote it with the number.** A win rate on a team set is reported with that set's overlap with our pool,
  exactly as it is reported with its regime (RULE 1).

Both sides draw from the SAME set in a cell. The summary carries `team_source_asymmetry`, which
goes true the moment the two sides' file counts differ; a cell that trips it mixes skill with
matchup and is not a clean read.

### Where the external checkouts live

`designs/ops/anchors.json`, with a per-key env override (`GEN3AI_METAMON_DIR`,
`GEN3AI_METAMON_PYTHON`, `GEN3AI_METAMON_CACHE_DIR`, `GEN3AI_FOULPLAY_DIR`,
`GEN3AI_FOULPLAY_PYTHON`, `GEN3AI_SHOWDOWN_NODE`), or `$GEN3AI_ANCHORS_CONFIG` for the whole file.
A path literal may not live under `src/` — `src/utils/paths_test.py` fails it, because a `/home/…`
literal is correct on one box and an invisible permanent skip on every other. A missing checkout is
a **named refusal** naming the key and its env var, never a skip and never a zero.

### Cost per read

Measured under a load average of 25–31 on 16 cores, with a live training arm on the box. CPU only;
nothing here touches the GPU.

| opponent | s/game | 100 games | notes |
|---|---:|---:|---|
| `metamon:SmallRL` | **3.0** (🚨 **7.2–7.9 with an end-state checkpoint on our side**, 2026-10-17) | **~5 min** (**~13 min** on rule 5b's agent cores, 2026-10-17) | 12–17 ms/decision; seconds to load. At the end-state network our in-process side (one torch thread) dominates the game time; Kakuna's cell then costs only a little more (8.8 s/game) On `--server rust` (the default) the same cell took **126 s** on a quiet box against **203 s** on Node, 2026-09-20 |
| `metamon:SyntheticRLV2` | **6.1** | **~10 min** | 56–113 ms/decision, **plus a multi-minute build per half** (200M params, 804 MB) |
| `metamon:Kakuna` | **~10–18** (smoke, n = 4) | **~15–25 min** (est.) | **median 73–78 ms, p90 86–88 ms/decision** over 157 greedy decisions; the FIRST decision of each peer is a 4–19 s warm-up; build ~10 s per half; **peak RSS ~2.0 GB per peer** (571 MB fp32 weights). Load 21–27 on 16 cores, 2026-09-28 (`measurements/kakuna_wiring_2026-09-28/`). **Measured 2026-09-28 over 500 games vs N0: ~6 s/game** (a 20-game unit in ~120 s wall, load 24–25); a Kakuna-vs-SyntheticRLV2 pair cell ~13–16 s/game (50 games in 620–815 s, load ~28) |
| `metamon:PastaMittens` | ≈ 19–20 with P\* on our side (est. from X51's 11.8 + the per-decision delta) | ≈ 33 min | **median 0.22–0.23 s, p90 0.28 s per router call** (95 calls, 2026-10-25, `measurements/metamon_strongest_2026-10-25/`); first call 5–33 s; **peak ~3.9–4.9 GB per scope** (5 members on CPU) |
| `foulplay` @ 1000 ms | **~88** | **~2.5 h** | **~80 core-seconds/game** — ~17× ours; cost is linear in `--search-time-ms` × `--search-parallelism` × the ×2 world multiplier while the opponent's active has <3 revealed moves |

🚨 **ONE THREAD PER PEER, and it is worth ~20×.** Measured 2026-09-14 on a box at load 63: each
Metamon peer was burning **210% CPU** on B = 1 CPU inference — two cores of thread synchronisation
per peer, buying nothing — and three parallel cells ran at **33 s/game**. With
`OMP_NUM_THREADS = 1` (now set in the peer env, and defensively inside `metamon_side.py` before
torch imports) the same three lanes run a 100-game cell in **~150 s, i.e. ~1.5 s/game**. Put the
parallelism ACROSS cells, never inside one forward pass. `src/main/anchors/peers_test.py` asserts
the pin on the env the plan actually carries.

---

## 3. The known hazards — every one has cost a session

The ids are stable (code and measurement records cite them). **Standing** hazards are still true of the world: a reader
accounts for them. **Closed** ones are fixed or refused in code and kept as one line.

### 3.1 The Metamon peer's long-tail fault — H5 · H17 · H21 (guarded; upstream unfixed)

One upstream defect, met three ways. `MetamonAMAGOWrapper.step` (`metamon_to_amago.py:323`) answers ANY exception with
`self.reset(); return self.step(action)`, with no depth bound: an error `reset()` cannot clear prints `Force resetting
due to long-tail error` about 985 times and ends in a `RecursionError` that kills the process.

| id | trigger | what it costs |
|---|---|---|
| **H5** | our side forfeits at turn 250, ending the battle mid-step ("Battle is already finished, call reset"). A RACE: it fired on 3 of the 105 turn-250 forfeits in 1,117 archived Kakuna units (`measurements/kakuna_determinism_2026-10-27/` §4), and the same unit re-run forfeited cleanly | when it fires mid-half the process lives on and spoils the NEXT TWO games: game n+1 is a 1-turn Kakuna `FORCELOSE` (a "win" for us), and game n+2 is played on carried state (its first differing choice comes from an identical protocol). Once triggered it is deterministic: two faulted plays of one unit on two engines are byte-identical. Forfeits ran 4 in 800 games (0.5 %); the watchdog's named `peer_exited` (rc and log tail, finished games attached) replaced a silent hang |
| **H17** | the half where METAMON challenges. Metamon serialises the acceptor role (its own `_accept_challenge_loop`) but drives the challenger role through poke-env's pipelined `start_challenging`: the same defect class as H14, on their side of the wire. The env's `current_battle` and the agent's then disagree | four recorded post-game occurrences. Three were `mixed`-regime, so the `mixed` cell is refused outright (`--allow-unmatched-regime` against metamon); the fourth was matched greedy, so that refusal is not the fix. The minimal upstream patch is in `measurements/anchors_p2_batch_2026-09-22/README.md` |
| **H21** | the peer's LAST battle: the crash comes after every game is recorded | 3 of 240 units of `measurements/human_only_vs_pstar_2026-10-25/`, one of them twice at the identical log line for the same seed |

**The rule (G2): a long-tail line anywhere in a unit voids the unit.** The watchdog (`session.watch`, and
`watch_peer_pair` for a pair cell) reads the peer's log on every poll and fails the half at the FIRST line
(`session.PEER_FAULT_LINE`) as `peer_long_tail_fault`, so a contaminated game is never played out (at most the game
already running when the line lands). After the half, `runner.apply_peer_fault` reads the whole log again and sets
`peer_clean` False on any line, so the verdict depends neither on the peer also recording an `error` or a nonzero rc nor
on whether a poll saw the line before the last record. `unit_counts` drops the unit and the lane's retry re-runs it with
the same seed. Never salvage a game played after a mid-unit fault. The rule costs nothing already banked: of 1,249
archived anchor units, none of the 950 counted has the line, and the 28 that have it were already uncounted.

`peer_recursion_upstream` (`summary.json["peer_exit_notes"]`) names a COMPLETE `peer_challenge` half whose peer died of
the `RecursionError`. It is a diagnosis, not a rescue: the long-tail lines have already voided that unit, so it never
counts. Cells banked under the older rule (H16's 31 sub-cells, where the fault fired after the last game) stand as recorded;
today's tool would have re-run them.

### 3.2 Teams: the played set must be the file's set

| id | status | hazard and handling |
|---|---|---|
| **H22** | fixed; a regime split | The Metamon peer packed its teams with upstream poke-env's `Teambuilder`, not as Showdown reads them (found 2026-10-27 by the H-CTX Hidden Power FINDING's sweep). It (1) filled a named Hidden Power with all-31 IVs from `HP_TO_IVS`, a table in the formula's bit order (hp, atk, def, spe, spa, spd) packed as (hp, atk, def, spa, spd, spe): 12 of the 16 types read as another type, Flying keeps its type with other IVs; (2) packed `Happiness: 0` as the default 255 and lacks Showdown's "Frustration ⇒ happiness 0" import rule, so Frustration hit at base power 1, not 102; (3) ignored a `Hidden Power: <Type>` line on a mon without the move. Silent except (1)'s refusals by the server (rc72: "Aerodactyl has Hidden Power Bug, but its IVs are for Hidden Power Water"). **Fix, `gen3_peer_showdown_pack_v1`:** the peer packs every file it may draw through `utils/bridge/showdown_pack.js` (import → validator → `Teams.pack`, what a real server plays), and `pack_oracle.js` must agree at startup or it exits naming the file and the set (`peer_scripts/team_pack.py`). The report's `peer_pack_changed` (True iff a file plays differently from the legacy pack) enters the id (§ 0.1); a report without the pack never counts. The local front end likewise PLAYS the validator's re-packed team now (`validate_team.js` returns it, `ws_frontend._cmd_utm` stores it, as `server/ladders.ts` plays `valResult.slice(1)`): OUR side's packs play the same sets (`main/anchors/our_side_played_team_integration_test.py`), but **FOUL PLAY'S play CHANGED** (`gen3_frontend_validated_pack_v1`, measured on its actual team sources 2026-10-27): its own packer writes no Hidden Power IVs (H19's bracketless moves), so a `Hidden Power: <Type>` mon played at the all-31 IV type (Dark 70) where the validator writes the named type's IVs; 11 of 72 files of its `gen3ai_pool` and `away`'s `team_23` play another set. `FoulPlayPeer.read_report` stamps `frontend_pack_changed` (`main/anchors/frontend_pack.py`: Foul Play's own packer under its interpreter, the validator's verdict, `showdown_pack.js`'s field-for-field diff over every file its `load_team` may draw; a file the validator refuses never plays and is no change; fail closed: a record that cannot be computed voids the regime verification), it enters the id only when True (§ 0.1), and an import across it refuses (`peer_pack` state `frontend_changed`). A banked Foul Play read of `gen3ai_pool` is its own regime; no Foul Play read of `away` is banked, and one could not run (H23) |
| **H23** | standing; nothing banked rests on it | **Foul Play on `away` cannot start 19 of its 21 draws, and the tool's guards do not see it** (audit 2026-10-27). Foul Play loads the RAW `competitive` directory and its `export_to_packed` splits a paste on `"\n\n"`, so a team file ending `\n\n\n` (18 of the 20) packs a seventh EMPTY mon and the validator refuses it ("You may only bring up to 6 Pokémon (your team has 7)"); `index.csv`, which its `load_team` also draws, packs one mon named `filename` (refused). Only `team_22` (four newlines) and `team_23` start: P(a draw starts) = 2/21 = 9.5 %, so a unit of k Foul Play draws starts all k with probability (2/21)^k. The front end drops a refused `/utm` (`conn.team = None` and a popup Foul Play has no handler for), so the next `/accept` or `/challenge` is refused and Foul Play waits for a battle that never begins: the half ends `no_first_game` / `no_progress` with `status: FAILED`. NO rule counts a refused start as anything: it makes no battle, no row and no `forced_ends.jsonl` record (G6), and `unit_counts` drops the FAILED unit (G5), so it is neither a win, a loss nor a game (read from `ws_frontend._cmd_utm` / `_cmd_accept` and Foul Play's client; **UNVERIFIED live**). **The guards are blind:** `their_team_count` and the team digests go through the `TEAM_FILE_GLOBS` allowlist and `.strip()`, so `--opponent foulplay --teamset away --dry-run` plans a clean `symmetric` 20-vs-20 cell (`peer teams: 20 files`) while Foul Play draws from 21 files; `FoulPlayPeer.read_report` records `frontend_pack_refused_files` and acts on none. **True coverage: 2 of 20 teams.** The same defect as H4, on Foul Play's own parser (the 2026-09-14 hang was our vendored fork's). **Nothing banked touches it:** all 50 recorded Foul Play `main.anchors` cells (817 games: the 09-16 axes, the P2 / Rust-front-end gates, the N0 end-of-run, the Kakuna goal read, the N0 smokes) and the pre-tool de-risk and flywheel campaigns drew from `gen3ai_pool` or `home72`, where 72 of 72 files start (`home72` re-packs unchanged; the pool changes 11 of 72, H22). Fix: TECH_DEBT (d) |
| **H24** | standing; a regime split, bounded | **OUR side's pack before `d8d674b99`** (the 10-27 X35 pack-regime audit). The packer dropped `Happiness: 0` and lacked Showdown's "Frustration ⇒ happiness 0" rule, so every Frustration set played base power 1, and it ignored a `Hidden Power: <Type>` line on a mon without the move (±1 stat point). On the panels only forum72's `sft431` is material: our side piloted it in 72 of 9,583 banked forum72 games, all at pre-fix play pins except N1-P look 1 (2) and H-CTX (4). home72, p72, heldout72_hpfix and rc72b pack identically. The playing commit decides the regime, which no id carries: a read's arms share ONE play pin, so they share it. D3 look 1 crossed it with 0 `sft431` games on either side. Training at `f4b25238`, its corpus-only pins and P\*'s `9c033489` carries the same defect (0.13–0.38 % of draws; `designs/training/team_curriculum.md`). Record: `designs/research_state/measurements/x35_pack_regime_2026-10-27/` |
| **H3** | standing | Our pool pastes omit the `IVs:` line and `Gen3Teambuilder` patches it in from `GEN3_HP_IVS` at pack time; Foul Play packs the paste verbatim, so `Hidden Power [Grass]` with no IV line becomes Hidden Power Dark: a DIFFERENT team, silently. The export WRITES the IV line into the paste; our pastes are only complete under our own teambuilder |
| **H19** | worked around, not fixed | Foul Play's stock `home` directory is NOT the pool, and 11 of its 72 files are incomplete. `designs/ops/anchors.json`'s `foulplay.team_dirs.home` holds the 72 `data/teams/sample` teams while our `home` source is the 719-team pool, so every stock Foul Play `home` cell is `team_source_asymmetry = true` by construction; the files keep nicknames (H2) and 23 bracketless `- Hidden Power Grass`-style moves on 11 teams with no `IVs:` line (the 2026-09-14 exporter matched only `Hidden Power [Grass]`), so Foul Play plays a different Hidden Power from the one our teambuilder patches in (H3). Found 2026-09-27. `measurements/n0_endofrun_2026-09-27/scripts/make_home72.py` writes a sanitized, validator-checked copy of the 72 that all three clients read in that campaign; the stock directory is unchanged |
| H2 | closed | A nickname (`Airmure (Skarmory)`) packs an EMPTY species in upstream poke-env 0.8.3.3 and hangs the read. Every team file handed to a third-party poke-env is nickname-free; the pool export strips them |
| H4 | closed | A trailing blank line became an empty 7th Pokemon in the (since deleted) vendored fork's parser and hung the read; `validate_teams_locally` passes it, the pack does not. The parser is `utils.team_packing`, the pack carries a throwing guard (> 6 mons raises) and the tool `.strip()`s (every Metamon `competitive` file ends `\n\n\n`), but Foul Play reads its team directory raw (H23); pinned by `src/poke_env_teambuilder_gate_test.py` |

**Which sets H22 moved.** Unchanged (the legacy pack played every file as Showdown does): home72, p72,
heldout72_hpfix, rc72b, the x37a and `tsr_k_*` sets except the two below, `tsr_ref24`. Changed, so a NEW regime, never
pooled with a banked read of the same set:

| team set | the file | Kakuna played before | Kakuna plays now |
|---|---|---|---|
| **forum72** (and `tsr_k_forum72`) | `sft431` Gyarados, `Happiness: 0`, Frustration | Frustration at base power **1** (a small constant handicap on Kakuna's side, about 1 team in 72) | base power 102 |
| away (`competitive`, `tsr_k_away`) | `team_23` Gyarados, Hidden Power Flying, no IV line | IVs 31/31/31/30/30/30 | the validator's 30/30/30/30/30/31 |
| the pool (`gen3ai_pool`) | 4 teams: 2 Frustration sets (an explicit `Happiness: 0`, or none, which Showdown's import reads as 0 for Frustration) and 2 `Hidden Power: <Type>` lines on mons without the move | Frustration at 1; all-31 IVs | 102; the type's IVs |

### 3.3 The opponents are not what their authors publish

| id | status | hazard and handling |
|---|---|---|
| **H7** | standing | No temperature can express Metamon's greedy: `MetamonDiscrete` clips probabilities to [0.001, 0.99] AFTER the temperature division, so the argmax tops out near 0.992 however cold and a random non-argmax move comes about 1 decision in 125, silently. Greedy is `Agent.get_actions(sample=False)` and nothing else, verified by `argmax_match_rate == 1.0000` (G1) |
| **H8** | standing | Foul Play's only budget is WALL CLOCK (no iteration, visit or depth budget anywhere), so the opponent is a width meter, not a setting. The realized visit count is scraped from its own `Iterations N: <visits>` lines and recorded per cell; a log with none reads UNVERIFIED (§ 4 "Foul Play") |
| **H12** | standing | The Showdown pin. Ours is `e0551883f`, 13 commits behind Metamon's bundled submodule. Irrelevant while BOTH clients play on ours, which the tool guarantees; relevant the moment anyone compares against numbers a third party produced on its own server. `showdown_pin` is on every row (§ 6 for the 20 / 20 byte-identical replay across the two pins) |
| **H18** | standing, sized | Metamon reads a Baton Pass wrong: upstream poke-env 0.8.3.3 clears the entrant's boosts on `\|switch\|…\|[from] Baton Pass`, and the protocol emits nothing else about the pass (`sim/pokemon.ts:1249`). Measured 2026-09-22 over 871 decision points in 20 greedy `SmallRL` battles: 13 mismatches in 3 battles (Metamon read `atk 0` where the sim held −1 and +2). It is NOT our server: the same battles through Metamon's own bundled Showdown (`d62d3a398`) are byte-identical on 20 / 20, so it fires wherever Metamon plays, its published numbers included. Our anchor win rate is OPTIMISTIC on Baton Pass teams (Metamon under-reads passed setup on both sides), and nothing warns (0 log warnings, 0 `UNKNOWN_TOKEN`s). **Unfixed, deliberately:** patching a third-party checkout would make the anchor a different opponent from the one its authors publish. Exposure: 8 of 21 `competitive` teams and 157 of 719 pool teams carry Baton Pass. **Sized 2026-09-22** (`measurements/h18_baton_pass_bias_2026-09-22/`): \|bias\| < 0.02 on both team sets, NOT DETECTED (paired, 2×2×400); the volatile half of the carry-over (Substitute etc.) is the larger half on a Baton-Pass-enriched set, per-exposure bound ±0.14–0.22. Anchor numbers stand with this bound attached |
| **H20** | fixed; one caveat | A gin file beats the `gin_overrides` dict, and the newer Metamon models SLIDE A WINDOW. `amago.cli_utils.use_config` binds the dict first and then parses the model's `.gin` files; `superkazam.gin` (Kakuna, Superkazam), `alakazam*.gin` and `smaller_multitaskagent*.gin` bind `attention_type = @FlashAttention` themselves, so the CPU swap silently lost, and the same files set `FlashAttention.window_size = (96, 0)` (or `(32, 0)`), which plain `VanillaAttention` would not honour: a different function from the trained one, on exactly the long games. Fixed 2026-09-28: `metamon_side.install_cpu_attention` re-binds AFTER the files parse and reads the file's window (none → `VanillaAttention`, so `SmallRL` / `SyntheticRLV2` are unchanged; a window → `WindowedVanillaAttention`, key `j` visible to query `i` iff `i − left ≤ j ≤ i`); class and window go into every peer report (`attention`, `attention_window`); pinned by `src/main/anchors/windowed_attention_integration_test.py` (equality with a naive sliding-window reference; Kakuna → window 96, SyntheticRLV2 → plain). Caveat: upstream's GPU path runs flash attention in bf16, ours is fp32 (the same small numeric gap SyntheticRLV2 always had) |

### 3.4 The harness, the transport and the clock

| id | status | hazard and handling |
|---|---|---|
| H1 | closed | A 19-character username hangs both clients (Smogon's guest login refuses it even against a `--no-security` local server; Foul Play logs "Successfully logged in" and blocks): `main.anchors.peers.check_username`, both names, before anything starts |
| H6 | closed | Metamon hardcodes `ws://localhost:8000` (`PokeEnvWrapper.server_configuration`), our shared dev port. No flag, env var or constructor argument exists, so the driver rebinds the module global before any env is built and refuses 8000 / 8001 in its own code |
| H9 | closed | Metamon's unflushed prints hid its readiness banner and the peer looked dead: `PYTHONUNBUFFERED=1` in the peer env |
| H10 | closed | A lingering websocket between halves: half 1's client still held the name, Showdown answered `\|nametaken\|`, the fork continued as a guest and the peer challenged a user that no longer existed (a ~8 min hang). The socket closes between halves AND each half plays under its own username suffix |
| H11 | closed | Two `poke_env` packages in one process broke silently. The peers are SUBPROCESSES with `PYTHONPATH=""`, nothing in this repo imports Metamon, and since P6 no side of ours imports poke-env, so the peer's upstream copy is the only one in any read |
| H13 | closed | Torch's default thread pool on B = 1 inference burned 210 % CPU per peer, about 20× slower than necessary. `OMP_NUM_THREADS=1` (+ MKL / OpenBLAS / numexpr / torch) in the peer env and in `metamon_side.py` before torch imports, pinned by `src/main/anchors/peers_test.py`; parallelise ACROSS cells, never inside one forward pass |
| H14 | closed | `--opponent foulplay` could not run a multi-game `ours_challenge` half: our poke-env sent challenge k+1 about 0.4 s into battle k and Foul Play, which reads its PMs only between battles, dropped it and waited forever (`no_progress` after 1 game). The defect was OURS. `--challenge-mode serial` is the default (`gen3_anchor_serial_challenge_v1`), both transports challenge serially by construction, and `--challenge-mode pipelined` is refused with its reason |
| H15 | closed | A bare TCP readiness probe fabricates an ERROR in the front end's log (`websockets` answers connect-then-close with a three-deep traceback), and "0 ERRORs in the server log" is the validation criterion: a server that PRINTS a readiness line (`[ws_frontend] READY`) is never TCP-probed (the Node server, which prints none we parse, still is) |
| H16 | closed | A composite verification flag hid which half failed: `regime_verified` ANDed "did the regime take effect?" with "did the peer exit cleanly?", and read false on 31 of 84 sub-cells of the 2026-09-20 continuation campaign whose every `argmax_match_rate` was 1.0000. `regime_verified_decisions` and `peer_clean` are separate fields on every row and summary (`gen3_anchor_regime_split_v1`); `regime_verified` survives as their AND, deprecated. The 84 sub-cells re-derive as verified with no win rate or verdict changed (`measurements/anchor_ab_continuation_2026-09-20/regime_split_rederived.json`) |

### 3.5 About us

A long campaign that imports the MAIN checkout is not pinned, and `main` moves on a multi-session box: the parallel
Metamon de-risk landed `--forfeit-turn-limit` mid-run on 2026-09-14 and an `AttributeError` killed a Foul Play session
silently after 40 games. Run a campaign from its own worktree, or pin the commit it imports (a registered read does:
G8).

---

## 4. The standing numbers, to date

### The matched-regime 2×2 — `ai_v12_02_winprob_critic` @ 75,005,952 steps

800 games on our pinned Showdown (`e0551883f`), Metamon @ `0a00a759`, CPU-only, 100 games per cell,
role balanced inside every cell, **both regimes verified on every decision on both sides**. Win
rate is OURS; ties count in the denominator and not the numerator. Full artifact:
`designs/research_state/measurements/metamon_matched_regime_2026-09-14/`.

**`SmallRL` (ckpt 40, 13.9M)**

| cell | n | W/L/T | win rate | Wilson 95% | mean turns |
|---|---:|---|---:|---|---:|
| greedy · home | 100 | 52/48/0 | 0.520 | [0.423, 0.615] | 36.3 |
| **greedy · away (PRIMARY)** | 100 | 65/35/0 | **0.650** | **[0.553, 0.736]** | 48.8 |
| t1.0 · home | 100 | 63/37/0 | 0.630 | [0.532, 0.718] | 41.2 |
| t1.0 · away | 100 | 56/44/0 | 0.560 | [0.462, 0.653] | 42.3 |

**PRIMARY VERDICT: BETTER** — the lower bound 0.553 clears 0.50. Pooled 236/400 = 0.590.

**`SyntheticRLV2` (ckpt 48, 200.9M)**

| cell | n | W/L/T | win rate | Wilson 95% | mean turns |
|---|---:|---|---:|---|---:|
| greedy · home | 100 | 50/49/1 | 0.500 | [0.404, 0.596] | 40.4 |
| **greedy · away (PRIMARY)** | 100 | 42/58/0 | **0.420** | **[0.328, 0.518]** | 47.9 |
| t1.0 · home | 100 | 41/59/0 | 0.410 | [0.319, 0.508] | 42.7 |
| t1.0 · away | 100 | 31/69/0 | 0.310 | [0.228, 0.406] | 44.8 |

**PRIMARY VERDICT: NOT DETECTED** — lower bound 0.328, and the point estimate is below even.
Pooled 164/400 = 0.410. `SyntheticRLV2` leads us in three of four cells and is level in the fourth.

**The three effects, each with its own Newcombe CI** (an effect whose CI covers zero is NOT
DETECTED, never "equal"):

| effect | `SmallRL` | `SyntheticRLV2` |
|---|---|---|
| temperature (greedy − t1), pooled | **−0.010 [−0.105, +0.086]** NOT DETECTED | **+0.100 [+0.004, +0.194]** DETECTED |
| team set (home − away), pooled | **−0.030 [−0.125, +0.066]** NOT DETECTED | **+0.090 [−0.006, +0.184]** NOT DETECTED |
| role (we challenge − they challenge) | +0.050 [−0.046, +0.145] | −0.020 [−0.115, +0.076] |

Two results worth carrying forward: our own **+8.9 pp** eval-regime figure is an ASYMMETRY term and
does **not** transfer to a symmetric regime change; and our 719-team training pool buys **no
detectable home advantage** over Metamon's 20-team set. 🚨 **CORRECTED 2026-10-18:** that set is NOT one we never
trained on (19 of its 20 teams are exact pool copies), so this is not evidence about unseen teams, and it is not
the "count dominates conditioning" corroboration it was read as.

### Foul Play — same checkpoint

| | |
|---|---|
| our win rate @ `--search-time-ms 1000`, `--search-parallelism 1` | **0.388 (31/80)**, Wilson 95% **[0.288, 0.497]** |
| **realized search width** | **1.40 M** MCTS visits/decision; session means **1.21–1.53 M** at a CONSTANT nominal budget |
| turns | mean 34.2, median 28, max 110; 0/80 hit the 250-turn forfeit |
| protocol | 0 parse failures either way over 3,040 decisions; the two clients agreed on the winner 80/80 |
| team set | Foul Play drew its 72-file `gen3ai_pool` here, as in every Foul Play cell to date (`home72` in the N0 and Kakuna-goal cells); 72 of 72 files start. **No Foul Play cell on `away` exists** (H23) |

**The external bot is AHEAD of the 75M win-prob arm at that budget, and the interval excludes
parity.** ⚠️ The team distribution in that campaign was ASYMMETRIC (we pinned one pool team per
10-game session, Foul Play redrew from all 72); `main.anchors` draws both sides from the same set
and flags any residual asymmetry, so a rerun through the tool is not directly comparable to the
0.388.

### X26 — `rb_x26_s2601` @ 161,000,169 (2026-10-17; the run was still live)

1,080 games, our side greedy on the in-process Rust core slot, Metamon @ `0a00a759` on CPU, 20-game
role-balanced units, every counted unit `status OK`, `regime_verified_decisions` true,
`team_source_asymmetry` false, `peer_clean` true. Full artifact:
`designs/research_state/measurements/x26_metamon_anchor_2026-10-17/`.

| cell | n | X26 win rate [Wilson 95%] | vs 0.50 | vs the previous best of ours on the same cell (Newcombe) |
|---|---:|---|---|---|
| Kakuna greedy · home72 | 240 | **0.600 [0.537, 0.660]** | **BETTER** | N0: 0.460 → **+0.140 [+0.046, +0.230]** |
| Kakuna T = 1.0 · home72 (`regime_matched = false`) | 240 | **0.600 [0.537, 0.660]** | **BETTER** | N0: 0.380 → **+0.220 [+0.127, +0.308]** |
| Kakuna greedy · away (19 / 20 teams in our pool) | 400 | 0.497 [0.449, 0.546] | NOT DETECTED | K2 (G0′): 0.355 → **+0.143 [+0.074, +0.209]** X26 BETTER (seeds 0–4 of the first read + 5–19 of `k2_x26_kakuna_both_teamsets_2026-10-17`, one identical regime stamp; the first 100 read 0.570) |
| Kakuna greedy · home72, topped up | 400 | **0.575 [0.526, 0.623]** | **BETTER** | K2 (G0′): 0.487 [0.439, 0.536] → **+0.087 [+0.018, +0.155]** X26 BETTER; N0 → +0.115 [+0.030, +0.198] (seeds 0–11 of row 1 + 12–19 of `k2_x26_kakuna_both_teamsets_2026-10-17`, one identical regime stamp) |
| **Kakuna greedy · heldout72 (0 / 72 teams in our pool; 2026-10-18)** | 400 | **0.395 [0.348, 0.444]** | **WORSE** | K2 (G0′): 0.357 [0.312, 0.406] WORSE → **+0.038 [−0.030, +0.104] NOT DETECTED**; X26 heldout72 − home72 **−0.180 [−0.247, −0.111]**, − away −0.102 [−0.170, −0.034] (`heldout_teams_kakuna_2026-10-18`, same regime stamp as the 10-17 cells except the team set) |
| **Kakuna greedy · heldout72_hpfix (the held-out set of record; HP repaired; 2026-10-18)** | 400 | **0.448 [0.400, 0.496]** | **WORSE** | K2 (G0′): 0.367 [0.322, 0.416] WORSE → **+0.080 [+0.012, +0.147] X26 BETTER**; vs heldout72 (HP Dark): X26 +0.052 [−0.016, +0.120], K2 +0.010, both NOT DETECTED; X26 hpfix − home72 **−0.127 [−0.195, −0.058]** (`heldout72_hpfix_kakuna_2026-10-18`, same regime stamp) |
| **Kakuna greedy · DEPLOYMENT, X26 pilots home72, Kakuna pilots heldout72_hpfix by stratum (X37a, 2026-10-18)** | 600 | **L (ladder_weights_v1) = 0.630 [0.573, 0.686]** (CORE 0.610 n = 100, NEAR 0.623 n = 400, FAR 0.690 n = 100) | **BETTER** | L_c1 0.624 [0.570, 0.676]; the reweighted cell S(P, H) 0.631 [0.592, 0.670] (`x37a_eval_matrix_2026-10-18`, same regime stamp; declared asymmetric) |
| **Kakuna greedy · REVERSE, X26 pilots heldout72_hpfix, Kakuna pilots home72 (X37a)** | 400 | **0.380 [0.334, 0.428]** | **WORSE** | Kakuna vs Kakuna K(home72, heldout72_hpfix) 0.573 [0.524, 0.620]; PILOT_c +0.123 [+0.039, +0.206], FACE_c +0.017 [−0.062, +0.095]: the held-out drop is PILOTING; H-PILOT UNDECIDED, H-PILOT-OFFMETA SUPPORTED |
| **Kakuna greedy · forum72 (72 human-built Smogon-forum teams, 0 / 72 in our pool; X37a stage 2)** | 400 | **0.470 [0.422, 0.519]** | NOT DETECTED | − heldout72_hpfix +0.022 n.d.; − home72 **−0.105 [−0.173, −0.036]** ⇒ third-source class **UNSEEN-DRIVEN** (forum72 governs) |
| **Kakuna greedy · P72 (72 procedural Smogon-usage teams, 0 / 72 in our pool; X37a stage 2)** | 400 | **0.453 [0.404, 0.501]** | NOT DETECTED | − heldout72_hpfix +0.005 n.d.; − home72 **−0.123 [−0.190, −0.053]** ⇒ **UNSEEN-DRIVEN** (agrees) |
| SmallRL greedy · away | 200 | 0.645 [0.577, 0.708] | BETTER | K2 (G0′): 0.625 → +0.020 [−0.058, +0.094] NOT DETECTED |
| SmallRL greedy · home (pool) | 200 | 0.785 [0.723, 0.836] | BETTER | K2: 0.755 → +0.030 [−0.040, +0.093] NOT DETECTED |
| SmallRL t1 · away | 100 | 0.730 [0.636, 0.807] | BETTER | `ai_v12_02`: 0.560 → +0.170 [+0.037, +0.295] |

**Reading the table.** The near-term goal (Kakuna at its strongest temperature, lower bound > 0.50) is MET at this
checkpoint on our TRAINING teams (home72, at both Kakuna settings: greedy and T = 1.0 read identically, Δ 0.000
[−0.087, +0.087], so "Kakuna is strongest at T = 1.0" is not reproduced). On `away` (19 of 20 teams in our pool) it is
NOT DETECTED, and on held-out teams it is not met: X26 loses 18 pp from home72 on `heldout72`. **Quote the goal as "met
on our training teams", never as "met".** Against SmallRL X26 is level with K2, the previous lineage's plateau.

* **K2 vs X26 crosses two harnesses** (`measurements/k2_x26_kakuna_both_teamsets_2026-10-17/`): K2 is played by its own
  training pin `a5b2fdba` on `gen3ai_stable` (torch 2.5.1, the pin's poke-env client; it does not load at HEAD), X26 by
  the X26 read's `63f97a57` (that read's § Comparability). K2 loses to Kakuna on `away` (0.355, WORSE) and X26 is ahead of it on away (+0.143), home72 (+0.087) and
  `heldout72_hpfix` (+0.080), DETECTED; on `heldout72` (+0.038) NOT DETECTED, and the change between those two held-out
  deltas is not detected either. K2 − N0 is +0.027 (NOT DETECTED), so the
  gain over N0 on our teams belongs to the K2 → X26 step.
* **`heldout72`** (`measurements/heldout_teams_kakuna_2026-10-18/`) is 72 replay-derived `hl_05_26` teams with 0 exact
  matches and 0 sharing ≥ 4 mon sets with our pool or `competitive`. It is both unseen by us and a different population
  (high-ladder replays, usage-sampled fills, Hidden Power Dark on 164 / 432 mons), possibly of the kind Kakuna's replay
  data covers; this read does not separate the two. The Hidden Power repair (`heldout72_hpfix`,
  `measurements/heldout72_hpfix_kakuna_2026-10-18/`) moves neither model detectably (X26 +0.052, K2 +0.010) and does not
  close the held-out gap (its own effect is neither detected nor excluded).

PROVISIONAL (one checkpoint, one seed, a run 161M into a 500M cap). 🚨 **Run `main.anchors` from the checkout ROOT**: the
pool (`home`) source resolves `data/teams` against the CWD and, from `src/`, finds 0 teams and dies as `our_side_error:
IndexError` (the read's FINDING 1).

**The home72 config is COMMITTED** (2026-10-17, T55, F21): `measurements/x35_breakers_stage_2026-10-17/anchors_home72_kakuna.json`
is byte-identical to the one this read used (sha256 `e8a064db…`), and `home72_manifest.json` beside it pins all 72
team files on BOTH sides (our committed pastes and Metamon's cache copy, byte-identical to each other). X35's G-K
read runs through `python -m main.x35.kakuna` (plan / run / status / gk / smoke), which re-checks that manifest, the
T28 halt marker and the model's sha256 before every unit, runs from the checkout root under rule 5b, picks its own
95xx port, and counts a unit only on `status OK`, `n == 20`, `regime_verified_decisions`, no
`team_source_asymmetry` and `peer_clean`.

**The UNFAMILIAR panels (X38 / D1's PRIMARY H, 2026-10-19).** `python -m main.x35.kakuna plan --teamsets unfamiliar`
plans the same 7 shared-seed 20-game units per seed final on each of `heldout72_hpfix`, `forum72` and `p72` (any comma
list of `home72`, `away` and those panels is accepted; the default stays `home72,away`, G-K's cell, unchanged). Each
panel unit runs on that panel's COMMITTED symmetric config (`anchors_heldout72_hpfix_kakuna.json`,
`x37a_eval_matrix_2026-10-18/configs/anchors_x37a_forum72.json`, `…_p72.json`, each pinned by sha256), and before EVERY
unit the runner checks the config's sha256, the panel's registry entry and committed copy, and BOTH sides' team dirs
(our byte copy outside the repo and Metamon's cache set) by count and `gen3_team_set_digest_v1` content digest against
`heldout_panels.json`; `python -m main.x35.kakuna panels` runs that check alone (all three OK on 2026-10-19). A panel unit
also passes `--capture-dir <unit>/capture`, so each game's two teams are on record (F-DV-17). A DIAGNOSTIC cell
(`kakuna.DIAGNOSTIC_CELLS`: `rc72b`, the H-CTX read's recombined pool sets, config
`measurements/hctx_rc72_2026-10-26/anchors_rc72b_kakuna.json`; `rc72` is its retired first draw) runs the same way; its panel is registered in
`diagnostic_panels.json`, which the exclusion rule never reads. `python -m main.x35.kakuna
d1h --on <arm> --off C0` writes D1's H input (per seed final and panel, wins / n, ties beside); `python -m
main.x35.readout decide-d1` reads it (module `main.x35.d1`: the equal-weight panel mean, its stratified variance, the
effective-size Newcombe interval and the registered classes).

### P\* — `rb_x26_s2601` @ 280,000,170, X26's plateau (X51, 2026-10-20 / 21)

2,800 games at the X26 reads' own harness (a detached `63f97a57`), Kakuna ckpt 34 greedy, paired seeds 0–19, every
unit counted on its first try. Full artifact: `designs/research_state/measurements/x51_plateau_vs_kakuna_2026-10-20/`.

| cell | n | win rate [Wilson 95%] | vs 0.50 | P\* − X26 @161M (Newcombe) |
|---|---:|---|---|---|
| home72 | 400 | **0.598 [0.549, 0.644]** | **BETTER** | +0.023 [−0.046, +0.090] n.d. |
| heldout72_hpfix | 400 | 0.497 [0.449, 0.546] | NOT DETECTED | +0.050 [−0.019, +0.118] n.d. |
| forum72 | 400 | 0.530 [0.481, 0.578] | NOT DETECTED | +0.060 [−0.009, +0.128] n.d. |
| P72 | 400 | 0.433 [0.385, 0.481] | WORSE | −0.020 [−0.088, +0.049] n.d. |
| the three unfamiliar panels, equal weight (Kish) | n_eff 1,208 | **0.487 [0.459, 0.515]** | NOT DETECTED | +0.030 [−0.010, +0.070] n.d. |
| Kakuna ckpt 28 (`metamon:Kakuna28`, the agent `policy` alias) vs ckpt 34, home72, a pair cell at `b9c75884` | 400 | 0.515 [0.466, 0.564] (ckpt 28's side) | NOT DETECTED | — |
| the toy T\* (`rb_toy_ent_s4002_deep` @290,000,087) · home72 / forum72 | 400 / 400 | 0.493 [0.444, 0.541] / 0.458 [0.409, 0.506] | NOT DETECTED | T\* − P\*: −0.105 / −0.073, both WORSE |
| P\* vs **`metamon:SmallRL`** (human-only, ckpt 40), the same twins, at `d32ffcee` (FR-1, 2026-10-25, `measurements/human_only_vs_pstar_2026-10-25/`) · home72 / H / F / P72 / U | 400 each; U n_eff 1,200 | 0.785 [0.742, 0.822] / 0.678 / 0.663 / 0.658 / **0.666 [0.639, 0.692]** | all BETTER | vs Kakuna on the twins: U +0.179 [+0.140, +0.218] (SmallRL WEAKER) |
| P\* vs **`metamon:LargeRL`** / **`metamon:LargeIL`** (human-only, ckpt 40, hub `ac8e88e7`), the same twins (FR-3, 2026-10-26) · home72 / H / F / P72 / U | 400 each | LargeRL 0.700 / 0.640 / 0.613 / 0.640 / **0.631 [0.603, 0.658]**; LargeIL 0.785 / 0.810 / 0.770 / 0.790 / **0.790 [0.766, 0.812]** | all BETTER | vs Kakuna on the twins, U: LargeRL +0.144 [+0.105, +0.183], LargeIL +0.303 [+0.266, +0.339] (both WEAKER) |

PROVISIONAL (one seed). The REVERSE cell (we pilot heldout72_hpfix, Kakuna pilots home72) was not read at P\*.

### P\* vs `PastaMittens`, Metamon's ensemble (2026-10-26)

1,600 games at `d32ffcee4`, every game a twin of X51's (the identity unit replayed X51 20 / 20), the router verified
deterministic on all 80,893 calls. Full artifact: `designs/research_state/measurements/metamon_strongest_2026-10-25/`.
P\* scores **home72 0.550 [0.501, 0.598]** (BETTER), H 0.488, F 0.520, P72 0.445 (WORSE), **U 0.484 [0.456, 0.512]**.
Against Kakuna's cells that is **Δ-U −0.25 pp [−4.23, +3.73]** and Δ-home −4.75 pp [−11.5, +2.1], both NOT DETECTED:
the ensemble is not detectably stronger than Kakuna, and Kakuna ckpt 34 stays the goal's opponent.

### 🚨 Numbers that must NOT be quoted without their regime

`0.742` (vs `SmallRL`) and `0.583` (vs `SyntheticRLV2`) are **mixed-regime reads on home teams** —
us greedy, them at T = 1.0 — from the de-risk. Matched, the same cells read **0.520** and **0.500**.
The honest summary of that checkpoint against Metamon is the 2×2 above.

---

## 5. Reading a result

`summary.json` carries `status`, the cell (regime, team set, opponent version and commit, our
checkpoint and how it resolved, the search budget), the Wilson interval, the per-role split, the
integrity counters (`hit_forfeit_limit`, `n_defaults`, `n_redecides`, `distinct_our_teams`), the
realized visit count and `their_argmax_match_rates`. `python -m main.anchors` prints the same block
at the end, regime first.

**Before believing a number, check four things:**

1. `status == "OK"`. A `FAILED` read still carries its games, deliberately, so a partial n is never mistaken for a full
   one, but it is not a measurement. The named causes are `peer_never_ready`, `peer_exited`, `no_first_game`,
   `no_progress`, `short_series`, `peer_long_tail_fault` (H5 · H17 · H21), `non_battle_termination` and
   `forced_end_guard_blind` (G6), `peer_team_attribution` (a pair cell), `core_slot_error`, `live_parse_halt`,
   `live_client_error` and `our_side_error`.
2. 🚨 **`regime_verified_decisions == true`** — the field the verification reads (`gen3_anchor_regime_split_v1`, G1). It
   is true when BOTH halves' `sample` kwarg matched the regime AND the per-decision `argmax_match_rate` was
   regime-appropriate over at least one decision: exactly `1.0` in a greedy cell, materially **below** 1.0 in a t1 cell.
   The sampling cell is the POSITIVE CONTROL (an instrument reading 1.000 in both regimes would have no power) and the
   "at least one decision" clause closes the vacuous pass, because a check that never ran reads exactly like one that
   passed. `their_argmax_match_rates` is printed beside it; read the flag, then look at the rates.
   - **`peer_clean` is a SEPARATE line** (H16, G2): every peer process exited 0, recorded no error and printed no
     long-tail line. A false `peer_clean` does not say the regime failed, so read the peer log for the cause. But
     `unit_counts` counts a unit only when it is true, so such a unit is no measurement for a read, and a long-tail line
     (H5 · H17 · H21) also fails the half (`peer_long_tail_fault`, VOID). For a one-off cell with `status OK` and no
     long-tail line, a nonzero rc alone does not invalidate a verified regime: quote the number with the flag stated,
     never pool it.
   - **A `peer_exit_notes` entry means the tool RECOGNISED the dirty exit.** `peer_recursion_upstream` is Metamon's
     recursion in the half where Metamon challenges (H17), stamped only when every game of that half is present. It is a
     diagnosis: its long-tail lines have already voided the unit. A dirty exit with NO note is one nobody has diagnosed;
     read the log.
   - `regime_verified` is DEPRECATED (one release): it is the AND of the two, and reading it is how 31 of 84 sub-cells
     got flagged unverified when nothing was wrong with them.
3. The TEAM condition, and for Foul Play a non-null `realized_visits_per_decision_mean`. `team_source_asymmetry`
   compares only the two sides' file COUNTS; `team_content_asymmetry` compares their CONTENT digests
   (`our_team_digest` / `their_team_digest`, null = unknown). An UNDECLARED cell needs `team_source_asymmetry == false`
   and `team_content_asymmetry` not true; a cell with `declared_team_cell == "asymmetric"` needs instead its two digests
   to equal the ones its registration declared. Decide it with `main.anchors.results.unit_counts` (§ 2), never by eye.
4. 🚨 **`team_overlap`** (2026-10-18): how much of the team set our training pool holds. A read on teams the pool
   already contains is a read on training data, whatever the set is called. A summary without the block predates
   `gen3_team_overlap_v1`; compute it with the module for any number you quote.

**Then write it down with its regime.** A row in `designs/research_state/ledger.md` names the
opponent, its version and commit, the regime, the team set AND its overlap with our pool, our checkpoint's step, n, the win rate
and its Wilson interval — and for a search opponent the realized visit count.

---

## 6. What an anchor read cannot say

* **Nothing about Metamon on Metamon's own Showdown pin.** Both clients play on ours.
  ⚠️ Amended 2026-09-22: it can now say ONE thing about it — the same seeded battle replayed
  through both pins is **byte-identical on 20/20** (`metamon_obs_faithfulness_2026-09-22`), so a
  protocol-level pin difference is ruled out for gen3ou at this sample. Strength on their pin is
  still unmeasured.
* **Nothing about exploitation.** Neither side adapts, so nothing here is evidence that greedy is
  safe against an adapting opponent.
* **Nothing about which model is stronger in general.** 100 games resolves ~±10 pp at best; every
  cell near even stays NOT DETECTED however the point estimate reads.
* **Nothing about the open ladder.** Both clients speak to a PINNED local server, and the two
  clients have OPPOSITE drift postures — ours raises on an unknown protocol keyword by design,
  Foul Play silently ignores it. A green anchor read says nothing about either on the public
  server; `python src/main/ladder_drift_scan.py` remains the instrument for that, and
  `designs/research_state/ladder_readiness.md` the audit.

## Amendment 2026-09-16 — the per-promotion read is BUDGET-CONDITIONAL, and the anchor `--model` spelling

**Scope today:** Tier A stands. No code or schedule runs an external anchor per promotion; an anchor is read at milestones (§ 1). What below still holds is the cell's measured noise floors, the fire rule if someone does read this cell per promotion, and the `--model` spelling warning.

**The per-promotion read, if you take it (from `ai_v13_03_fork` on):** `python -m main.anchors --model models/<run>/snapshots/snapshot_<12-digit step>.zip --opponent metamon:SmallRL --regime greedy --teamset away --games 100 --device cpu` with `OMP_NUM_THREADS=1`, ONE cell at a time in the foreground (batched anchor runs were killed by a session harness's memory accounting with 67 GB free). 🚨 `--model <run>@<promoted step>` FAILS — `resolve_model_ref` does not search `snapshots/`, and checkpoint steps are on a different cadence from promotions (backlog row).

**Floors, measured 2026-09-16 on this cell:** three-seed RUN-level floor **0.110** (`ctrl10M` / `_b` / `_c` read 0.440 / 0.380 / 0.330, each ±0.095); **eval-draw floor 0.020** (four team-seed draws of the same `fork@10M` snapshot: 0.530 / 0.510 / 0.530 / 0.510). The run-level variance on this cell belongs to the RUN, not the draw.

**The re-registered fire rule:** (1) WITHIN-RUN — the newest promoted node's Wilson upper bound falls below the previous node's point by more than **0.02 + the CI half-width** (≈ 12 pp at n=100), a drop the eval draw cannot explain; (2) the ABSOLUTE bar (upper bound < 0.50 fires; lower bound > 0.50 is BETTER) applies at ≥ 75M only, where the 0.63–0.65 band lives — at 10M an unremarkable control (`ctrl10M_c`, upper 0.427) trips it; (3) every 10M anchor number is banked beside the three-seed band **0.33–0.44** with the words "not separated" unless it clears 0.110 against ALL THREE controls. **The budget effect on this cell (~0.38 at 10M → ~0.64 at 75M) dwarfs every lever the ladder measured**; never compare a 10M arm's anchor to a 75M arm's.
