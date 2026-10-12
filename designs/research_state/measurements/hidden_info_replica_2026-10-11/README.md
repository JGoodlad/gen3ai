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

RESULTS_PLACEHOLDER

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

AMBIGUITIES_PLACEHOLDER

## Reproduce

```bash
D=designs/research_state/measurements/hidden_info_replica_2026-10-11
$D/run_chain.sh <cp> <shard> <nshards> <cpus> O P F F+M O-M        # rows -> ~/gen3ai_archive/hidden_info_replica/rows
PYTHONPATH=$D python -m replica.audit --cp <cp> --out $D --rows ~/gen3ai_archive/hidden_info_replica/rows
PYTHONPATH=$D python -m replica.score --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D
PYTHONPATH=$D python -m replica.marginals --rows ~/gen3ai_archive/hidden_info_replica/rows --out $D
```
