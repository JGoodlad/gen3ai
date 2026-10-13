# Hidden-info study: the independent replication of the fair-world instrument (2026-10-11 / 12)

## In plain language

The hidden-information study (`designs/endstate/design_hidden_info_binding.md`) asks how much better a search
would decide if it knew the opponent's hidden facts (its unseen moves, its never-seen team members, items,
abilities, stat spreads). Its yardstick is **IMP**: how much the search's pick beats the policy's own pick, in
win probability per decision, judged by a referee that played every action out 24 times in the TRUE battle.
The search plays its rollouts either in the true battle (**O**, the oracle) or in **fair worlds**: copies of the
battle where every fact the searched side has not seen is re-drawn from the Smogon usage prior, kept only if
replaying the game so far shows the searched side exactly what it really saw.

The owner's worry was that a bug in the fair-world builder would quietly BE the result. So this folder is a
SECOND builder, written by a different agent from the spec text alone (§13.1), without looking at the first
builder, its scripts or any of its results. Its numbers go to a third party, who compares the two. This folder
does NOT compare them.

**What we found while building it (the short version):** see FINDINGS below. The most important for the third
party: (1) reusing the true battle's dice seed lets the turn-0 gender dice leak which never-seen species the
opponent has, and we fixed that by aligning the seed; (2) the same seed reuse still lets any hidden fact that
spends dice without printing a line (Focus Band, King's Rock, a speed tie) be "seen" by the gate; (3) the pool
teams use Hidden Power IV patterns and spreads the Smogon top-25 spread table does not hold, so a faithful
builder needs the full spread table and every max-power IV pattern, or it cannot rebuild about half the truths.

## INTERIM REGISTRATION (2026-10-12 ~10:00, before any F / F+M outcome was read)

The CPU allowance was cut to 3 workers (a GPU run took the box at 09:29), so the full 600 + 600 states will take
about a day. On the orchestrator's instruction an INTERIM subset is registered here, and its numbers go out first:

- **Subset:** per checkpoint, the first 40 states of each of the pilot's five gap quintiles, in the pilot's own
  registered sampling order (`sha256("value_disc_pilot_2026:" + decision_id)`, `design_value_discrimination.md`
  §7.2): 200 states per checkpoint, listed in `interim_ids_end2.txt` / `interim_ids_prod.txt`
  (`replica/interim.py`). The states file lists states sorted by id, so a plain file-order prefix would hold one
  cycle and a few opponents only; hence the stratified hash order.
- **Read:** `replica/score.py` unchanged, run on these ids for every arm (O, P, F, F+M, O-M), and its
  states-included rule unchanged (states where all five arms built 8 worlds).
- **What had been seen when this was written:** O rows (the 8,914 / 8,914 match with Phase 1's seeds 0-7, no IMP
  computed); P / O-M row COUNTS and failure reasons; F / F+M row counts and failure reasons only (no outcome,
  no IMP, no contrast). The scorer was run once at ~23:30 on partial O rows only (own-set IMP of O; no other arm).
- The final read uses all 600 states per checkpoint when they complete.

## INTERIM RESULTS (the registered subset, read 2026-10-12 17:10; `interim/result.json`, `interim/tables.md`)

The read is LEAF-K8, with IMP on the win scale. CIs are battle-clustered bootstraps (2,000 draws, rng 20261011), with
every contrast paired on the same draws. The common set holds the states where all five arms built 8 worlds:
end2 180 states / 159 battles, prod 181 / 163. **This replication does not compare these numbers with the first
implementation's; a third party does.**

| read | states | IMP_O | IMP_P | IMP_F | IMP_F+M | IMP_O-M |
|---|---|---|---|---|---|---|
| end2 | 180 | +0.0569 [+0.0335, +0.0810] | +0.0037 [-0.0178, +0.0264] | +0.0147 [-0.0052, +0.0360] | +0.0116 [-0.0152, +0.0361] | +0.0237 [+0.0013, +0.0461] |
| prod | 181 | +0.0642 [+0.0385, +0.0922] | -0.0218 [-0.0462, +0.0019] | -0.0152 [-0.0387, +0.0088] | +0.0132 [-0.0092, +0.0378] | +0.0363 [+0.0111, +0.0621] |
| pooled | 180 / 181 | +0.0606 [+0.0430, +0.0787] | -0.0091 [-0.0258, +0.0081] | -0.0003 [-0.0159, +0.0157] | +0.0124 [-0.0045, +0.0298] | +0.0300 [+0.0124, +0.0476] |

| read | G = O - F | G_P = O - P | F+M - F | O - (O-M) | s_M | n_M | B2 | B3 | B4 |
|---|---|---|---|---|---|---|---|---|---|
| end2 | +0.0422 [+0.0182, +0.0659] | +0.0532 [+0.0285, +0.0783] | -0.0031 [-0.0252, +0.0180] | +0.0332 [+0.0113, +0.0566] | not computed (B2) | not computed (B2) | no | no | no |
| prod | +0.0794 [+0.0503, +0.1106] | +0.0860 [+0.0592, +0.1110] | +0.0284 [+0.0047, +0.0544] | +0.0280 [+0.0113, +0.0447] | +0.358 [+0.065, +0.650] | +0.352 [+0.153, +0.630] | yes | no | no |
| pooled | +0.0609 [+0.0422, +0.0797] | +0.0697 [+0.0517, +0.0881] | +0.0127 [-0.0036, +0.0295] | +0.0306 [+0.0170, +0.0453] | +0.209 [-0.067, +0.457] | +0.502 [+0.290, +0.758] | yes | no | yes (point 0.5024, only 0.0019 above the 0.5005 boundary) |

LEAF-K1 / K4 and the own-set IMP of every arm are in `interim/tables.md` and `interim/result.json`.
**Exclusions** (no consistent F world found by the screen, all with reason `no_consistent_context`): end2 F 17 / 200,
end2 F+M 11, prod F 13, prod F+M 12. O, P and O-M lost no interim state. Every arm is under §6.1's 10 % line.

## STATUS AND HANDOFF (2026-10-12 ~20:05; the agent stopped under the quota rule; the workers keep running)

The full read (600 + 600) is IN PROGRESS. Three unattended, resumable workers run on cores 5-7 / 13-15 at nice 19,
each under `mem_cap`, and write rows to `~/gen3ai_archive/hidden_info_replica/rows/<cp>/<arm>/*.jsonl`:

| worker | now | then |
|---|---|---|
| W1 | prod F (rest) | end2 F, half A of the non-interim states (`interimRestA_ids_end2.txt`) |
| W2 | end2 F+M (rest) | end2 O-M (rest) |
| W3 | prod F+M (rest) | end2 F, half B (`interimRestB_ids_end2.txt`) |

Done at 20:00: O 600 + 600; P 600 + 600 (end2: 2 failed states remain after one re-run); O-M prod 600, end2 316;
F end2 257, prod 346; F+M end2 387, prod 430. At about 1.3 states a minute in all, roughly 15 h remain.
**To finish:**
1. When every arm holds 600 rows, run `replica.remaining` per (cp, arm). Any listed state is unfinished: run it with
   `seq_remaining.sh <new tag> 5-7,13-15 <cp>:<arm>`.
2. Run `replica.audit` per checkpoint with `--rows` (identity + playout replay, teeth, parser, harness vs Phase 1).
3. Run `replica.score` (all states), `replica.report`, `replica.marginals` and `replica.pack --out <this folder>/rows`.
4. Put the final tables here beside the interim ones. Do NOT compare with the first implementation.
A state run twice keeps its ok row (`score.load_rows`); two ok rows for one state is an error by design.

## Blindness (what this agent read and did not read)

- **Read:** `design_hidden_info_binding.md` §1.1, §1.2, §4, §5 (the audit definitions), §6.1-§6.3, §12.1-§12.3,
  §12.10, §12.12 (Amendment 1), §13, §15; `design_value_discrimination.md` §2.1, §5.1, §7.1-§7.4 (the pilot's
  definitions: the referee, the seeds, the checkpoint, the continuation). The registered states
  (`value_disc_pilot_2026-10-11/states_<cp>.json`), the banks (`~/gen3ai_archive/value_disc_pilot/<cp>/bank`) and
  the Phase 1 truth rows (`.../<cp>/rows/`, used only for the referee's outcome strings and, as an audit of this
  harness, their seeds 0-7). General infrastructure at the pins: `utils.rust_env.successors.play_out`,
  `agents.battle.core_replay.run_core`, `main.policy_spectrum.{bank,reader,replay,truth}` (the truth tool's
  `turn_seeds`, `to_log`, `cmd_index`, `MicroBatcher`), `agents.gen3_data.*`, the Rust engine source (gender
  construction, the gen-5 LCG, the move panics) and its `scan_move_probe` coverage oracle.
- **Not read:** `src/main/value_search/`, `src/main/hidden_info_study/`, `determinize.py`, every script and
  result file in `value_disc_pilot_2026-10-11/` and `hidden_info_binding_2026-10-11/` other than the two states
  files, the ledger entries about them, `design_hidden_info_binding.md` §0 and §16 (results), the pilot's §7.10-
  §7.13 (amendments and results), the pilot's `det*` / `qrows` archives, and `~/.cache/gen3ai/vdp/support_*`
  (the first study's engine probe; this folder ran its own: `engine_probe_*.jsonl`).
- The pin checkouts `~/.cache/gen3ai/vdp/pin_{end2,prod}` were used as built, after a byte check that every
  tracked file under `src/` and `data/` equals the pin commit.

## What was built (`replica/`)

| file | what it does |
|---|---|
| `env.py` | puts the pin first on the path, names its binaries, loads the registered states |
| `view.py` | reads the SEARCHED side's own protocol lines up to the root: the opponent mons seen, every move a public line names for them, items and abilities named by `-item` / `-enditem` / `-ability` or a `[from] item:` / `[from] ability:` tag (owner resolution written out) |
| `sampling.py` | the Smogon prior filtered to what the pinned engine runs (own probe) and the pinned gen3ou rules allow; conditional-Poisson move draws; the never-seen species draw; IV patterns; legality |
| `builder.py` | per state: the truth replay, the pending opponent choice, seed alignment, world assembly per arm, rendering of past opponent choices, the GATE (searched side's lines AND rows byte-equal) |
| `screen.py` | the F / F+M screen of the seen mons' item, ability, spread, IVs and Hidden Power |
| `fair.py` | the 8 accepted worlds per state and arm (≤ 60 draws) |
| `play.py` | the playouts (every legal action x 8 worlds, world j on dice seed j; O: the true battle on seeds 0-7) |
| `score.py` | the referee, IMP per arm and scorer, the contrasts, §12.3's B2-B4 for M |
| `audit.py` | builder identity, the planted-defect teeth, the parser against the truth, the harness against Phase 1 |
| `marginals.py` | per-class marginals of the drawn worlds (for §12.2's TV check) |

## Spec ambiguities, and the choice made for each (most literal reading; none peeked at the first builder)

| # | where the spec is silent or self-overlapping | this replication's choice |
|---|---|---|
| A1 | worlds reuse the TRUE battle's dice seed (needed for the prefix to replay), but the engine's turn-0 construction draws one die per blank-gender, gendered-species mon (p1's team, then p2's): a world whose never-seen species are genderless where the truth's are not shifts EVERY later die, so the gate would accept only worlds whose never-seen species match the truth's gender pattern (a leak about U) | every world plays with EXPLICIT genders on both teams (ours: as realized; a seen opponent mon: as shown; a never-seen one: M or F with probability 1/2 each, independent of the species, blank for a fixed-gender or genderless species) and with the raw seed advanced past the truth's gender draws (`builder.lcg_advance`, the gen-5 LCG of `rust_sim/src/prng/gen5.rs`). The identity audit passes on every state with it, and fails when the seed is one draw off (tooth T4) |
| A2 | the same seed reuse lets ANY hidden fact that spends dice without printing a line (Focus Band's roll on every hit to its holder, King's Rock, a speed tie's shuffles) change later dice, so the gate "sees" it | NOT fixed (it is inherent to replaying with the true seed; the first builder shares it). Reported as a finding |
| A3 | the opponent's root-turn choice, when the log recorded it before the searched side's (it is pending, not yet executed, so not seen) | DROPPED in every constructed world: the world's greedy opponent answers its root decision, identically across the searched side's actions. Kept in O (the referee's convention). Phase 1's rows say 7 end2 states and 1 prod state are of this kind |
| A4 | slot positions (team order, move-slot order) are not a class of §6.1 | kept from the truth: a seen mon in its true slot, a revealed move in its true move slot; the drawn moves fill the other slots |
| A5 | "past opponent choices that name a re-drawn move or a never-seen species must be rendered" | a `move X` whose X is not in the world's set (a choice that never executed, so never shown) becomes the world's move in X's TRUE slot; a typed Hidden Power becomes the world's Hidden Power; a `switch S` to a re-drawn slot becomes the world's mon in that slot |
| A6 | "species ∝ the teammate prior given the revealed species" with several revealed species | the MEAN over the revealed species of the Smogon P(teammate \| revealed), each never-seen slot in turn, excluding a species whose base is already on the team (Species Clause), banned species, and species with no move / item / ability prior |
| A7 | "every unrevealed fact re-drawn from the Smogon prior": the spread prior | `priors.all_spreads` (every chaos Spreads entry). The top-25 file `priors.spreads` holds only ≈ 49 % of the bank's opponent spreads (2,435 of 4,998), so with it about half the truths could never be rebuilt |
| A8 | Smogon publishes no IVs | the attack IV is "max" or "min" (0, or the lowest value keeping a Hidden Power's two low bits) with prior 1/2 each; with a Hidden Power of type t, every max-power pattern (each IV 30 or 31) whose type is t, uniformly. The pool uses non-canonical patterns (e.g. Hidden Power Ice as Spe 30 alone) and the gate is exact to one HP point |
| A9 | §6.1's classes OVERLAP in gen 3: a Hidden Power's type (M) is a function of the IVs (S) | where S is TRUE (P, O-M) the IVs are kept, so a revealed bare Hidden Power has its true type and an M draw may add a Hidden Power only of the IVs' type; in F an UNREVEALED Hidden Power's presence and type are screened together with S (prior: its conditional-Poisson inclusion probability); in F+M the true moves carry the true type and the IV pattern is drawn among that type's patterns |
| A10 | "each candidate in the prior's support is screened by a replay with the other mons at their F draws" | the support is thousands of spreads per species, so the screen is: a context search (one consistent world, found by replays), per-mon windows over the support at that context, then per-mon REJECTION sampling of joint (spread, IV key, item, ability, Hidden Power) prior draws with the other mons at the context; ≤ 192 draws per mon, else the context alone (flagged `context_only`). Worlds combine the mons' accepted draws and pass the full gate (≤ 60 tries) |
| A11 | which line types "name" a move / item / ability for a mon | written out in `view.py`: `move` (not a `[from]` bounced / called move, not Struggle / Recharge), `cant` with a move, `-singleturn`, `-singlemove`, Spite's `-activate`, Disable's `-start`; items: `-item` (Thief / Trick handled), `-enditem`, `[from] item:`, a positional `item:`; abilities: `-ability` (Trace handled), `[from] ability:` with the owner resolved by who can hold it |
| A12 | legality of a world team | the pinned format's rules (`team_legality`, ladder-only rules excluded), ≤ 1 Hidden Power, learnset, Hidden Power type = the IVs' type, engine support (own `scan_move_probe` run). A problem the TRUE team also has (our learnset table rejects some pool moves, e.g. Dragon Dance Tyranitar) is excused |
| A13 | the scorers | LEAF-K8 as registered (LEAF-K1 / K4 reported from the same playouts). The critic and TRUNC scorers are not run: §13.1 names only LEAF-K8 |
| A14 | the referee's tie / truncation | an outcome '0' in the Phase 1 strings is 1/2 |
| A15 | "F+M (F with the true seen moves)" | the TRUE full movesets of the seen mons (class M revealed), everything else as F |
| A16 | public details beyond the species | a seen mon's shown gender, shiny flag and level are carried into every world (a shiny seen mon otherwise fails every world's gate) |

## Reproduce

```bash
D=designs/research_state/measurements/hidden_info_replica_2026-10-11
$D/run_chain.sh <cp> <shard> <nshards> <cpus> O P F F+M O-M        # rows -> ~/gen3ai_archive/hidden_info_replica/rows
PYTHONPATH=$D python -m replica.audit --cp <cp> --out $D --rows ~/gen3ai_archive/hidden_info_replica/rows
PYTHONPATH=$D python -m replica.score --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D
PYTHONPATH=$D python -m replica.marginals --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D
```
