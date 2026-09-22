# IS METAMON'S OWN OBSERVATION OF OUR STREAM FAITHFUL? — 2026-09-22

**The owner's question, verbatim:** *"are we confident we don't have a GIGO with Metamon?"*

**The answer, in one sentence:**

> 🚨 **Nothing our server does corrupts Metamon's view — but Metamon's own parser DOES lose one
> thing, and it is not a spelling: every Baton-Passed stat stage.** Over **871 decision points in
> 20 greedy anchor battles**, the picture Metamon builds from our protocol stream and the picture
> our `Gen3Battle`/`LiveView` builds from the *same bytes* agree on **858 of 871 (98.51%)** — and
> the 13 that disagree are **all one mechanism**: upstream poke-env 0.8.3.3 clears the entrant's
> boosts on a `|switch|…|[from] Baton Pass`, so Metamon read `atk 0` where the simulator held
> `atk −1` and `atk +2`. **Our fork fixed exactly this on 2026-08-23**
> (`src/poke_env/battle/baton_pass_carryover_test.py`); Metamon runs the unfixed upstream.
> It is therefore **Metamon's defect, not ours** — check 4 proves the point by replaying every
> captured battle through **Metamon's own bundled Showdown** (`d62d3a398`, 13 commits ahead of our
> pin) and getting a **byte-identical** stream on 20 of 20, so Metamon has this blind spot on its
> own server too, including in the numbers it publishes. The silent-substitution census that no
> win rate and no stderr grep could find — `UNKNOWN_TOKEN` in the tokens the policy actually ate —
> is **0 over all 871 decisions**, and `Average Valid Actions` is **1.0000** on both halves.

**Registered verdict, by the rule fixed in [`PREDICTION.md`](PREDICTION.md) before the first
capture: GIGO CANDIDATE — class named: `BATON PASS BOOST CARRY-OVER`.** Three of the four bars are
clean; the type-(b) bar is not, and the pre-registration says any type-(b) mismatch names the class
and buys the SOP a hazard row (added as **H18**). The honest reading of the class is in §3: it
biases the anchor **against Metamon**, by an amount this pass measured the EXPOSURE of and did not
measure the SIZE of.

Pre-registered in [`PREDICTION.md`](PREDICTION.md), committed **`e5f31d53`** before the first
battle was played.

---

## 0. THE FOUR VERDICTS

| # | check | bar | result | verdict |
|---|---|---|---|---|
| **1** | parser parity on our stream | type-(b) mismatch classes = **0** | **1 class, 13 / 871 points (1.49%), 3 / 20 battles** — all `active.boosts` after a Baton Pass | 🚨 **FAIL** |
| **2** | silent substitution + warnings | warnings = 0, avoidable `UNKNOWN_TOKEN` = 0 | **0 in every protocol class; 0 UNKNOWN_TOKENs over 871 decisions; `Average Valid Actions` 1.0000 / 1.0000** | ✅ **PASS** |
| **3** | strength cross-check | published number inside our CI, else `SyntheticRLV2` > `SmallRL` with the interval clear of 0.50 | no reproducible published per-generation number exists (§4.1); on the registered substitute `SyntheticRLV2` beats `SmallRL` **0.650, Wilson 95% [0.582, 0.713], n = 200** | ✅ **PASS** |
| **4** | decision-level agreement across the two Showdown pins | argmax disagreements = **0** | **20 / 20 battles byte-identical** between our pin and Metamon's bundled one, with the live-stream reproduction control passing 20 / 20 ⇒ **0 disagreements by construction** | ✅ **PASS** |

---

## 1. SETUP — every hash

| Thing | Value |
|---|---|
| Our checkout (worktree) | `metamon_gigo`, from `e5f31d53` |
| Our Showdown pin | `deps/pokemon-showdown` @ **`e0551883f`** = `v0.11.10-1224-ge0551883f` |
| **Metamon's bundled Showdown** | `/home/goodlad/dev/metamon/server/pokemon-showdown` @ **`d62d3a398`** — built for check 4 |
| Metamon checkout | `/home/goodlad/dev/metamon` @ `0a00a759` |
| `poke-env` (Metamon env) | **0.8.3.3, upstream from PyPI** — a *different package* from ours |
| `poke-env` (ours) | the vendored fork at `src/poke_env/` |
| Arm **W** | `models/ai_v13_02_flywheel_winprob/final_model.zip` @ **75,005,952** steps, rung `explicit_zip`, loader `bare` |
| Metamon policies | `SmallRL` ckpt 40 (13.9M) · `SyntheticRLV2` ckpt 48 (200.9M), `MinimalActionSpace`, `poke-env` backend |
| Obs space / tokenizer | the policies' own — `DefaultObservationSpace` + `allreplays-v3` (read off `get_pretrained_model(...)`, never named by hand) |
| Transport | `--server rust`, `ws_frontend@80560c85+rust:sim_bridge`, ports 9500–9599, each stopped by its own PID. **:8000 and :8001 were never touched** |
| Compute | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, `OMP_NUM_THREADS=1`. **Nothing written under `models/`** |

### The capture

```
python -m main.anchors --model models/ai_v13_02_flywheel_winprob/final_model.zip \
  --opponent metamon:SmallRL --regime greedy --teamset away --games 20 \
  --device cpu --server rust --seed-base 20260922 --capture-dir <dir>/captures --out <dir>/cell
```

`status OK` · `regime_verified_decisions true` · `peer_clean true` · `their_argmax_match_rates
[1.0]` · `team_source_asymmetry false` · `hit_forfeit_limit 0` · `n_defaults 0` ·
**W 13 / L 7 / T 0 = 0.650, n = 20**, mean 37.75 turns. (At n = 20 that number carries ±0.2 and is
not a strength claim; it is the byproduct of getting 20 replayable captures. The cell's standing
value at n = 1200 is `0.5533` — `anchor_ab_continuation_2026-09-20`.)

---

## 2. CHECK 1 — PARSER PARITY ON OUR STREAM

### How it was done, and why this is the only shape that answers the question

Each `ws_frontend` capture holds `chunks` — **every per-side chunk the front end relayed**, as
`(slot, text)`. We take the chunks **Metamon's client received** and feed those exact bytes to two
parsers that never meet:

* `replay_metamon_state.py`, under the **Metamon interpreter** (upstream poke-env 0.8.3.3), through
  `Player._handle_battle_message`'s own dispatch → an upstream `Battle` → `UniversalState.from_Battle`
  (`metamon/interface.py:571`) and `UniversalAction.definitely_valid_actions`, snapshotted at every
  non-`wait` `|request|` — which is exactly where `PokeEnvWrapper.embed_battle` runs;
* `replay_ours_state.py`, under `gen3ai_stable` (the vendored fork), through the same dispatch →
  `Gen3Battle` → `LiveView` + `LegalActions`, snapshotted at the same points.

Two interpreters, because one process resolves `import poke_env` to exactly one package and the
other half then breaks *silently* (root `CLAUDE.md`; anchors **H11**). `replay_common.py` — the one
shared module — imports nothing but the standard library for the same reason.

**The alignment is checked, not assumed.** A battle whose two dumps carry different decision counts
is recorded as an `alignment_failure` and skipped rather than compared row by row; there were
**0 of 20**, and every `seq` matched. And the comparator refuses a zero denominator: *a comparator
that compared nothing is a failed check, not a passing one*.

### The result

| | |
|---|---:|
| battles compared | **20** |
| decision points compared | **871** |
| alignment failures | **0** |
| **type-(b) mismatches — a field Metamon READS** | **13** |
| type-(c) mismatches — a field only WE read | 13 *(the same fact seen through the raw team dict, not a second defect)* |
| type-(a) presentation rules, resolved by declared normalization | 8 rules (below) |

| mismatch class | count | battles | what it is |
|---|---:|---:|---|
| **`b:active.boosts`** | **13** | 3 (`-7`, `-12`, `-17`) | 🚨 the Baton Pass carry-over — §3 |
| `c:our_team.boosts` | 13 | the same 3 | the same 13 decisions, read off the raw team dict instead of the active slot |

**Everything else agreed on all 871 points**: active and opponent species, HP fractions, statuses,
types, levels, revealed movesets, items, abilities, volatiles, bench membership and bench HP,
weather, field, both sides' side conditions, `forced_switch`, `opponents_remaining`, `won`/`lost`,
and the decision surface (`available_moves`, `available_switches`, `force_switch`, `trapped`).

### The (a)-class rules, each declared rather than discovered

A normalization that is invented after seeing a mismatch is a normalization that can absorb a real
one. Each of these is a **stated representation choice of one side or the other**, written into
`compare_states.py` with the line of code it comes from:

| # | rule | whose choice | why it cannot hide a real difference |
|---|---|---|---|
| 1 | side conditions compared by **containment** | Metamon keeps only the most recent (`universal_conditions`) | a name Metamon holds that we do not is still a mismatch |
| 2 | `unknown_item` / `unknownitem` / `None` folded to one sentinel | poke-env vs Metamon vs our `LiveView` | a *revealed* item one side lacks is still a mismatch |
| 3 | `opponents_remaining` reconstructed from our raw `opp_team` | the two quantities differ by definition (`LiveSide.remaining` is a lower bound) | comparing them as written would be a category error, not a check |
| 4 | `nostatus` ≡ `None` | Metamon's spelling | a real status is untouched |
| 5 | the `notype` pad dropped | `universal_types(force_two=True)` | a real second type is untouched |
| 6 | `hiddenpower*` folded to `hiddenpower` | **ours** — `LegalActions` keeps the request's wire-truth (`own_hp_typed_id` exists for this) | a different move still mismatches |
| 7 | move legality compared on the **enabled** subset | ours keeps every request slot with `LegalMove.disabled`; upstream's `available_moves` is already filtered | a Taunt or a Choice lock is not a parser disagreement |
| 8 | volatiles compared by **containment** | Metamon keeps only the most recent effect | a volatile Metamon holds that we do not is still a mismatch |

Before these eight were applied the raw count was 2,271; after, 13. **Every one of the 2,258
differences removed is a spelling or a slice, and every one is named above** — the count did not
fall because the comparator got quieter, but because it stopped reporting `BRN` vs `brn`.

---

## 3. THE FINDING — `BATON PASS BOOST CARRY-OVER`

### The repro, exactly

`battle-gen3ou-12`, Metamon on **p1**, turns 0–4 (`captures/battle-gen3ou-12.json`):

```
|switch|p1a: Zapdos|Zapdos|321/321
|switch|p2a: Salamence|Salamence, F|100/100
|-ability|p2a: Salamence|Intimidate|boost
|-unboost|p1a: Zapdos|atk|1              <- Zapdos is now atk -1
...
|move|p1a: Zapdos|Baton Pass|p1a: Zapdos
|switch|p1a: Hariyama|Hariyama, F|429/429|[from] Baton Pass
```

At the very next decision (`seq 4`, turn 4):

| | `active.boosts["atk"]` |
|---|---:|
| **Metamon reads** | **0** |
| **we read** | **−1** |
| **the simulator holds** | **−1** |

`battle-gen3ou-17` (7 points) and `battle-gen3ou-7` (2 points) are the same mechanism with the sign
the other way: a Celebi Swords-Dances to **+2** and passes, and Metamon's entrant reads **0**.

### Ground truth, cited rather than argued

`deps/pokemon-showdown/sim/pokemon.ts:1249`, `copyVolatileFrom`:

```ts
if (switchCause !== 'shedtail') this.boosts = pokemon.boosts;
```

The entrant inherits the passer's boosts **and the protocol emits nothing about it** — the
`|switch|…|[from] Baton Pass` line is the entire trace. A client that clears boosts on every switch
therefore loses the pass, with nothing to warn it.

### Whose defect it is

**Ours is fixed; theirs is not.** `src/poke_env/battle/baton_pass_carryover_test.py` pins the
carry-over in our fork and its docstring records the find (2026-08-23, from a live probe trace,
present since the fork was vendored 2026-05-12) in exactly these terms: *"That is a GIGO defect,
not a display one."* Upstream 0.8.3.3's `Pokemon.switch_out` — the code Metamon runs — still calls
`clear_boosts()` and `_clear_effects()` unconditionally, and its `abstract_battle` has no mention of
Baton Pass at all (`grep -rn 'Baton Pass' <upstream>/poke_env/environment/` → nothing).

### 🚨 Why this is NOT a GIGO we introduced

Check 4 replays all 20 captured battles through **Metamon's own bundled Showdown** and gets a
**byte-identical** stream. The line `|switch|…|[from] Baton Pass` is standard Showdown emission on
both pins, so **Metamon reads a Baton Pass wrong wherever it plays** — on our server, on its own,
and in the evaluations behind its published numbers. Our server introduces nothing.

### The exposure, measured; the size, not

| | |
|---|---:|
| decision points affected, this capture | **13 / 871 = 1.49%** |
| battles affected | **3 / 20 = 15%** |
| Baton Pass **switch events** in the 20 battles | 24, of which **3 carried a nonzero boost** |
| `competitive` (away) teams containing Baton Pass | **8 / 21** |
| our 719-team pool (home) containing Baton Pass | **157 / 719 (21.8%)** |

**Direction:** Metamon under-reads boosts on **both** sides — its own passed setup and ours. It
therefore plays a Baton Pass line weaker than it otherwise would, which makes our anchor win rate
against it **optimistic** on exactly the battles where a pass lands with stages on it.
🚨 **The SIZE of that bias is NOT measured here** and must not be guessed: 13 mis-read decisions in
871 is not a win-rate delta, because a mis-read boost matters enormously on one turn and not at all
on another. The measurement that would settle it is named in §8.

**The volatile half of the same mechanism** (Substitute, Leech Seed and the rest of
`BATON_PASS_COPIED_EFFECTS`, which `copyVolatileFrom` also carries) has the same cause and **did not
fire in these 20 battles** — the three boost-carrying passes carried no copyable volatile. The
instrument for it is in place (`(a)8`, containment over `LiveView.volatiles`) and read 0. That is a
zero from a check that ran, not from a check that was not written.

---

## 4. CHECK 2 — SILENT SUBSTITUTION AND WARNINGS

**A Metamon that silently substitutes an unknown token is a GIGO no win rate reveals** — and, it
turns out, no stderr grep either: `PokemonTokenizer.tokenize`
(`metamon/tokenizer/tokenizer.py:76`) maps any out-of-vocabulary word to `UNKNOWN_TOKEN = -1` and
**prints nothing**. So the census is taken in the array itself, on the observation
`obs_space.state_to_obs(state)` actually produces, at every decision point.

| class | pattern searched | acceptor half | challenger half | replay stderr |
|---|---|---:|---:|---:|
| unparsed / unhandled line | `unparsed\|cannot parse\|unhandled\|NotImplemented` | 0 | 0 | 0 |
| unknown species | `UnknownPokemon\|Unknown species\|not exist\|KeyError` | 0 | 0 | 0 |
| unknown move | `unknown move\|invalid move\|no move named` | 0 | 0 | 0 |
| unknown item / ability | `unknown item\|unknown ability` | 0 | 0 | 0 |
| fallback tokenization | `` Adding: ` ``\|`UNKNOWN_TOKEN` | 0 | 0 | 0 |
| exception / traceback | `Traceback\|Exception\|RecursionError\|AssertionError` | 0 | 0 | 0 |
| invalid action fallback | `invalid\|choose_random_move` | 0 | 0 | 0 |
| any `WARNING` | `WARNING\|Warning:\|warn(` | **1** | **1** | 0 |

The single `Warning:` on each half is
`You are sending unauthenticated requests to the HF Hub` — a Hugging Face rate-limit notice emitted
during weight download, with nothing to do with the protocol. Reported rather than filtered out,
because a grep that quietly drops its only hit is not a grep.

| the census a grep cannot take | |
|---|---:|
| decision points instrumented | **871** |
| **`UNKNOWN_TOKEN` (−1) in the tokens the policy consumed** | **0** |
| distinct out-of-vocabulary words | **0** |
| Metamon's own `Average Valid Actions` (acceptor / challenger) | **1.0000 / 1.0000** |
| `n_defaults` (our illegal-action fallbacks) | **0** |

`Average Valid Actions = 1.0000` is the independent corroboration from Metamon's own scoreboard:
every action its policy chose was legal under its own reading of our request, over all 871
decisions. **VERDICT: PASS.**

---

## 5. CHECK 3 — THE STRENGTH CROSS-CHECK

### 5.1 No reproducible published number exists — where that was looked for

| source | line | what it carries | why it is not reproducible here |
|---|---|---|---|
| Metamon `README.md` | L255–300, "Ladder Ratings with Sample Teams (GXE)" | `SyntheticRLV2` G3 = **64%** | a **human-ladder** GXE on the public server; the README's own tip says these "have no connection to ratings on the public ladder" for the local table, and this one cannot be reproduced without laddering (which `CLAUDE.md` forbids from this box's egress) |
| Metamon `README.md` | L398–760, "Early Gen OU Local GXE" | `SynRLV2` Competitive/G3 = **55%** | a GXE **relative to the 15 listed models** on a local ladder. Reproducing it means running that whole population, and `SmallRL` is not in the table at all |
| RLC 2025 paper (arXiv 2504.04395) | Fig. 6, Fig. 16, §5.3 | heuristic composite scores; a Gen1-4OU-**aggregated** heuristic round robin | no per-generation win rate for `SmallRL` / `SyntheticRLV2` against a named baseline. The only gen3 figure is **BC-RNN** at "as high as 85% in Gen3OU", which is a different model |

So the registered substitute was taken.

### 5.2 The substitute — `SyntheticRLV2` vs `SmallRL`, head to head, on OUR server

```
python -m main.anchors --our-side metamon:SmallRL --opponent metamon:SyntheticRLV2 \
  --regime greedy --teamset away --games 200 --device cpu --server rust --out <dir>
```

An **anchor-vs-anchor** cell: no checkpoint of ours is on the board, both peers take `--regime` and
**both verified it per decision**.

| | n | W / L / T | win rate | Wilson 95% |
|---|---:|---|---:|---|
| `SmallRL` (ckpt 40, 13.9M) | 200 | 70 / 130 / 0 | 0.350 | [0.287, 0.418] |
| **`SyntheticRLV2` (ckpt 48, 200.9M)** | 200 | 130 / 70 / 0 | **0.650** | **[0.582, 0.713]** |

`status OK` · `regime_verified_decisions true` · `peer_clean true` · `their_argmax_match_rates
[1.0]` · `team_source_asymmetry false` · 20 distinct teams · mean 56.8 turns · 0 forfeits.

**VERDICT: PASS.** The lower bound **0.582 clears 0.50**, and the ordering is the one the paper and
the README both imply — `SyntheticRLV2` is the stronger policy, by 30 pp, measured on our pin with
our transport and our forfeit rule. A parser that had lost the battle would not produce a 30-pp
ordered separation between two policies whose only difference is training.

⚠️ **What this check can and cannot say.** It shows the two policies retain their *relative* order
and a large, well-resolved gap on our server. It does not put either on an absolute published
scale, because no such scale is reproducible (§5.1). It also inherits the §3 defect on both sides
symmetrically, which is a reason to read it as an ordering rather than as a calibration.

---

## 6. CHECK 4 — THE SAME BATTLE ON BOTH SHOWDOWN PINS

Metamon's bundled Showdown is **13 commits ahead** of ours (anchors **H12**), and the honest form
of "run the policy on both servers" is not two *different* battles — it is the **same** battle
twice. A capture makes that possible: `commands` is the seeded `START` plus every committed
`CHOOSE`, and replaying it regenerates the battle exactly.

`pin_differential.py` replays each capture's commands through `local_sim_bridge.js` twice — once
against `deps/pokemon-showdown` (`e0551883f`), once against
`/home/goodlad/dev/metamon/server/pokemon-showdown` (`d62d3a398`, built for this check) — and
compares the per-side protocol text under the two legitimate normalizations (the `|t:|` timestamp
and the `rqid` counter).

| | |
|---|---:|
| battles replayed on both pins | **20** |
| **byte-identical per-side streams** | **20 / 20** |
| divergent | **0** |
| replay errors | **0** |
| 🚨 **control: our replay reproduced the LIVE captured stream** | **20 / 20** |

**The control is the whole point.** A "no divergence" verdict from a harness that produced nothing
on either pin is indistinguishable from a real agreement, so the run **refuses to write a summary**
unless its own replay reproduces the live capture at least once; it reproduced it every time.

**Argmax disagreements: 0.** Not sampled — *by construction*. Metamon's policy in this regime is a
deterministic function of the stream it receives (greedy, `Agent.get_actions(sample=False)`,
`argmax_match_rate = 1.0000` on every decision of both halves), and the streams are byte-identical,
so no 200-decision sample could find a disagreement that exists. **VERDICT: PASS**, and it is the
check that converts §3 from "our server confuses Metamon" into "Metamon reads Baton Pass wrong
everywhere".

---

## 7. HAZARDS — each is a finding

### H-A 🚨 `BATON PASS BOOST CARRY-OVER` — Metamon's observation loses every passed stat stage

Fully in §3. **Goes to `designs/ops/EXTERNAL_ANCHORS_SOP.md` as H18.** The exact repro is
`captures/battle-gen3ou-12.json`, Metamon slot `p1`, decision `seq 4` (turn 4):

```bash
python designs/research_state/measurements/metamon_obs_faithfulness_2026-09-22/replay_metamon_state.py \
    --capture <captures>/battle-gen3ou-12.json --slot p1 --username MetaSmallRL2 --out /tmp/meta.jsonl   # metamon env
python designs/research_state/measurements/metamon_obs_faithfulness_2026-09-22/replay_ours_state.py  \
    --capture <captures>/battle-gen3ou-12.json --slot p1 --username MetaSmallRL2 --out /tmp/ours.jsonl   # gen3ai_stable
python designs/research_state/measurements/metamon_obs_faithfulness_2026-09-22/compare_states.py \
    --meta-dir /tmp --ours-dir /tmp --out /tmp/parity.json
```

**Not fixed, deliberately.** The defect is in a third-party checkout we do not own, and patching
Metamon's `poke_env` would make our anchor a *different opponent* from the one its authors publish
— which is the one property an external anchor exists to have. The remediation worth having is a
**measurement**, not a patch: §8's backlog row.

### H-B ⚠️ A comparator's normalizations are where a real mismatch goes to die

The raw comparison produced **2,271** differences; eight declared representation rules took it to
**13**. Had those rules been written *after* looking at the output, they would have been
indistinguishable from a filter tuned until the answer came out clean. They are therefore each
pinned to the line of third-party or first-party code that makes the choice (§2), and two of them
(side conditions, volatiles) are **containment** tests rather than equalities precisely because
Metamon's narrowing is lossy in one direction only. This is the `vacuous_tests_and_guards.md`
failure class, and it nearly bit here.

### H-C ⚠️ `UNKNOWN_TOKEN` is unreachable by every instrument the task originally named

Metamon's tokenizer substitutes `-1` for an unknown word and prints nothing — no warning, no
exception, no log line, no effect on `Average Valid Actions`. A stderr grep over a 20-game cell
would have returned a clean sheet whether the count was 0 or 800. The only instrument that can see
it is a census over the array the policy consumes, which is why `replay_metamon_state.py` builds the
real observation rather than re-deriving the text.

### H-D ⚠️ Our `LegalActions.move_ids` is NOT upstream's `available_moves`

Ours is every slot the request listed, **disabled ones included** (`LegalMove.disabled` carries the
server's own word so the masker need not re-derive it); upstream's is the enabled subset. Comparing
them as written reported a Taunt and a Choice lock as parser disagreements (42 of them). Anyone
diffing our decision surface against a third-party client's must compare
`[m.id for m in move_slots if not m.disabled]`, not `move_ids`.

### H-E ℹ️ Metamon's bundled Showdown builds clean, and is worth keeping built

`node build` in `/home/goodlad/dev/metamon/server/pokemon-showdown` succeeded from a bare checkout
in a single pass (no `node_modules` beforehand). It is what makes the pin differential possible, and
`pin_differential_bridge.js` — a copy of `local_sim_bridge.js` whose only change is that the
Showdown checkout comes from `$GEN3AI_PS_PATH` — is what points a replay at it. The real bridge
stays pinned; a differential tool is not a reason to make the production path configurable.

---

## 8. THE BACKLOG ROW

**Measure the SIZE of the Baton Pass bias, don't guess it.** The exposure is known (§3); the
win-rate effect is not. The measurement: run the standing Tier-B cell twice at matched team seeds —
once as shipped, once with the Metamon peer's process carrying a `sitecustomize.py` that restores
the passed boosts in upstream `Pokemon.switch_out` (our fork's fix, ported) — and read the delta
with its own Newcombe CI. Two 400-game arms, CPU, ~30 min. Until that runs, **every anchor number
against a Baton Pass team set is optimistic by an unmeasured amount**, and the honest statement is
that sentence, not a number.

---

## 9. WHAT IS IN THIS DIRECTORY

| File | What |
|---|---|
| `PREDICTION.md` | the pre-registration, committed `e5f31d53` before the first battle |
| `replay_common.py` | the ONE shared normalization — standard library only, because the two replays run under different `poke_env` packages |
| `replay_metamon_state.py` | our stream → upstream poke-env → `UniversalState` + the real tokenized observation + the `UNKNOWN_TOKEN` census (Metamon interpreter) |
| `replay_ours_state.py` | the same stream → `Gen3Battle` → `LiveView` + `LegalActions` (`gen3ai_stable`) |
| `compare_states.py` | the per-decision comparator, the three classes, and the eight declared (a)-rules |
| `pin_differential.py` + `pin_differential_bridge.js` | check 4 — the same seeded battle through both Showdown pins |
| `warning_census.sh` | check 2's log grep, by class, reporting every class including the zeros |
| `parity_summary.json` | check 1's full output: counts, classes, examples with repro turns |
| `pin_diff.json` | check 4's per-battle result |
| `anchor_cell_smallrl_greedy_away_summary.json` | the 20-game capture cell's `summary.json` |
| `anchor_h2h_synthv2_vs_smallrl_summary.json` | check 3's 200-game head-to-head |

The captures themselves (20 battles, ~40 MB) live under the session scratch and are not committed;
`--seed-base 20260922` regenerates them exactly.

---

## 10. READY-TO-APPEND LEDGER PARAGRAPH

> **2026-09-22 — METAMON OBSERVATION FAITHFULNESS: no GIGO from our server; ONE real defect in
> Metamon's own parser — `BATON PASS BOOST CARRY-OVER`.** Pre-registered (`e5f31d53`) before the
> first capture, four checks with four bars. **(1) Parser parity:** 20 greedy `metamon:SmallRL`
> anchor battles vs arm W (`ai_v13_02_flywheel_winprob` @ 75,005,952), `--server rust`,
> `--seed-base 20260922`, captured and replayed through BOTH parsers on the same bytes — upstream
> poke-env 0.8.3.3 + `UniversalState.from_Battle` on one side, `Gen3Battle`/`LiveView`/
> `LegalActions` on the other. **858 of 871 decision points agree (98.51%), 0 alignment failures**;
> the 13 that disagree are one class — after `|switch|…|[from] Baton Pass` Metamon reads the
> entrant's boosts as 0 where the sim holds −1 and +2 (3 of 20 battles; repro
> `battle-gen3ou-12` seq 4). **Our fork fixed this 2026-08-23**
> (`poke_env/battle/baton_pass_carryover_test.py`); Metamon runs the unfixed upstream, and
> `sim/pokemon.ts:1249` (`this.boosts = pokemon.boosts`, emitting nothing) is the ground truth.
> **(2) Silent substitution:** `UNKNOWN_TOKEN` in the tokens the policy actually consumed = **0 over
> 871 decisions**; 0 hits in seven protocol warning classes (the one `Warning:` is a Hugging Face
> rate-limit notice); Metamon's own `Average Valid Actions` = **1.0000 / 1.0000**. **(3) Strength
> cross-check:** no reproducible published per-generation number exists (README L255–300 is a human
> ladder GXE, L398–760 is a population-relative local GXE without `SmallRL`, the RLC 2025 paper's
> heuristic figures are Gen1-4-aggregated), so the registered substitute ran — `SyntheticRLV2` beats
> `SmallRL` head to head on our server, greedy-vs-greedy, away teams, **0.650 [0.582, 0.713],
> n = 200**, lower bound clear of 0.50, both regimes verified per decision. **(4) Pin differential:**
> all 20 captured battles replayed through Metamon's OWN bundled Showdown (`d62d3a398`, 13 commits
> ahead of `e0551883f`) produce **byte-identical per-side streams, 20/20**, with the live-stream
> reproduction control passing 20/20 — so argmax disagreements are **0 by construction**, and
> Metamon reads a Baton Pass wrong on its own server too, including behind its published numbers.
> **Registered verdict: GIGO CANDIDATE, class `BATON PASS BOOST CARRY-OVER`** — it is Metamon's
> defect and not ours, it biases the anchor in OUR favour (Metamon under-reads passed setup on both
> sides), the exposure is 13/871 decisions, 3/20 battles, 8/21 `competitive` and 157/719 pool teams
> carrying Baton Pass, and **the SIZE of the bias is NOT measured** — the backlog row names the
> patched-vs-unpatched 2×400 cell that would settle it. `EXTERNAL_ANCHORS_SOP.md` gains hazard
> **H18**. Artifact:
> `designs/research_state/measurements/metamon_obs_faithfulness_2026-09-22/`.
