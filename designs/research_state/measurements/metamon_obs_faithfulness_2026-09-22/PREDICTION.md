# PRE-REGISTRATION — is Metamon's OWN OBSERVATION of our stream faithful? (2026-09-22)

**The owner's question, verbatim:** *"are we confident we don't have a GIGO with Metamon?"*

**What is already verified, and is NOT re-litigated here.** The teams we hand Metamon
(nickname-free, explicit Hidden Power IVs, trailing newline stripped — de-risk H1/H3/H4), the
forfeit limit (`--forfeit-turn-limit`, `StallConfig().threshold` = 250), the Showdown pin
(`e0551883f`, recorded on every anchor row), the per-decision greedy regime
(`regime_verified_decisions`, `argmax_match_rate == 1.0000`), and byte-identical transport
(`src/utils/bridge/ws_frontend_byte_identity_integration_test.py`).

**What is NOT verified, and is the subject of this record.** That *Metamon's own parser*, reading
*our* server's protocol stream, builds the same picture of the battle a Metamon eval on *its own*
server would build. Every number in `designs/ops/EXTERNAL_ANCHORS_SOP.md` §4 assumes it. A win rate
cannot reveal a violation: a Metamon that silently reads a wrong item, a missing move or an unknown
token plays worse and the harness books it as our win.

**The surface, located before any game was played** (cited so the checks below are falsifiable):

* `/home/goodlad/dev/metamon/metamon/interface.py:571` — `UniversalState.from_Battle(battle)`, the
  ONE function that turns a poke-env `Battle` into Metamon's state. It reads: `battle.battle_tag`,
  `battle.weather`, `battle.fields`, `battle.side_conditions`, `battle.opponent_side_conditions`,
  `battle.active_pokemon`, `battle.opponent_active_pokemon`, `battle.team`, `battle.opponent_team`,
  `battle.reviving`, `battle.force_switch`, `battle.won` / `battle.lost`, `battle.can_tera`,
  `battle.teampreview_opponent_team`.
* `/home/goodlad/dev/metamon/metamon/interface.py:409` — `UniversalPokemon.from_Pokemon(pokemon)`,
  reading `species`, `base_species`, `current_hp_fraction`, `types`, `item`, `ability`, `level`,
  `status`, `effects`, `moves` (`.values()[:4]`, **NOT** `available_moves`), `boosts`, `base_stats`.
* `/home/goodlad/dev/metamon/metamon/tokenizer/tokenizer.py:76` — `PokemonTokenizer.tokenize`,
  which maps any word absent from the frozen vocabulary to `UNKNOWN_TOKEN = -1` **and prints
  nothing**. This is the exact "silently substitutes an unknown token" failure the question names.
* The backend for `SmallRL` / `SyntheticRLV2` is `poke-env`
  (`metamon/rl/pretrained.py`), so the `Battle` is an **upstream poke-env 0.8.3.3** object, a
  different package from our vendored fork (root `CLAUDE.md`; anchors hazard H11).

---

## The four checks and their bars

| # | check | instrument | **BAR** |
|---|---|---|---|
| **1** | **Parser parity on our stream.** 20 greedy games of `metamon:SmallRL` vs arm **W** (`models/ai_v13_02_flywheel_winprob/final_model.zip` @ 75,005,952), `--server rust`, `--seed-base` + `--capture-dir` so the protocol is replayable. The Metamon side's captured chunks are fed to an UPSTREAM poke-env `Battle` (+ `UniversalState.from_Battle`) in the Metamon interpreter, and the SAME chunks to our `Gen3Battle`/`LiveView`/`LegalActions` in `gen3ai_stable`. Compared at every decision point: active species, HP fractions, statuses, boosts, known moves, revealed team, weather/field, and the request's legal actions | two JSONL state dumps + a comparator | **mismatch classes of type (b) = 0** |
| **2** | **Silent-substitution census.** Metamon's stderr across the captured cells, grepped for unparsed lines / unknown move/species/item / `UnknownPokemon` / `NotImplemented` / fallback tokenization / `KeyError` recoveries — **plus** the count that no warning would ever produce: `UNKNOWN_TOKEN` (−1) occurrences in the token observation Metamon actually builds on our stream, and any `UniversalPokemon` field reading `unknownitem` / `unknownability` where the protocol had revealed the truth | peer stderr + a tokenizer census over the replayed states | **warning count = 0** and **avoidable UNKNOWN_TOKEN = 0** (a token that is unknown because the *information* is genuinely hidden is not a defect; one that is unknown because the *parse* lost it is) |
| **3** | **Strength cross-check against a fixed reference.** A published, per-generation win rate for `SmallRL` / `SyntheticRLV2` against a NAMED fixed baseline was searched for first (Metamon `README.md` ladder-GXE table L255–300; "Early Gen OU Local GXE" L398–760; the RLC 2025 paper's Fig. 6 / Fig. 16 / §5.3). **If none is reproducible, the registered substitute is `SyntheticRLV2` vs `SmallRL` head-to-head on OUR server, greedy-vs-greedy, away teams, ≥ 200 games** (anchors `--our-side metamon:SmallRL`), read against the ORDERING the paper and the README imply (`SyntheticRLV2` stronger) | `python -m main.anchors` | **the published number inside our Wilson CI**, or — on the substitute — **`SyntheticRLV2` > `SmallRL` with the Wilson interval clear of 0.50** |
| **4** | **Decision-level agreement.** For 200 decisions sampled from the captured games, Metamon's policy is run offline on (i) its observation built from OUR stream and (ii) its observation built from the same battle replayed through **Metamon's own bundled Showdown** (`/home/goodlad/dev/metamon/server/pokemon-showdown` @ `d62d3a398`, 13 commits ahead of our pin). Argmax disagreements are counted. **If that server is not buildable inside 30 minutes, the check is SKIPPED and said to be skipped** — a skip is reported as a skip, never as a pass | offline policy replay | **argmax disagreements = 0** |

## The verdict rule, fixed in advance

* **All four pass ⇒ NO GIGO DETECTED.** Stated with its scope: 20 captured games, one team set, one
  opponent, our pin.
* **Any type-(b) parser mismatch, or any nonzero warning class ⇒ GIGO CANDIDATE**, with the class
  NAMED, a repro turn attached, and `designs/ops/EXTERNAL_ANCHORS_SOP.md` §3 gaining a hazard row.
* A type-(c) mismatch (a field OUR side reads differently) is **our** bug and is reported as one; it
  does not make Metamon a GIGO but it is not swept under the rug either.
* A type-(a) mismatch (a presentation rule with no decision impact — e.g. one side spells a status
  `BRN` and the other `brn`) is recorded with its count and is **not** a failure.
* Check 4 SKIPPED (server not buildable) ⇒ the overall verdict may still be NO GIGO DETECTED, but
  the skip is carried in the headline, not buried.
* 🚨 **A check that could not run is not a check that passed.** A comparator that finds zero
  mismatches because it compared zero decision points FAILS; every script asserts its own
  denominator (`n_decision_points >= 1`, `n_battles >= 1`) rather than branching on it.

## What this record will NOT be able to say

* Nothing about Metamon on the **public** ladder or on a third party's pin.
* Nothing about team sets other than the one captured, or opponents other than `SmallRL`.
* Nothing about whether Metamon's observation is a *good* observation — only whether it is the
  observation Metamon would have built elsewhere.

**Registered before the first capture.** Arm W, `metamon @ 0a00a759`, Showdown `e0551883f`,
CPU-only (`CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1`), own servers on 9500–9599
stopped by PID, nothing written under `models/`.
