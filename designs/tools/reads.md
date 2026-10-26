# `main.reads` — one module for registered reads

Always current. Owns `src/main/reads/`. A **registered read** measures trained models against a rule written
before any counted game (a PREREG): an arm against its control, a final against a comparator, a model against an
external anchor. Before this module (2026-10-25) every read re-implemented the same seven pieces as one-off scripts
under `designs/research_state/measurements/<read>/`. The same bugs recurred (§8). New reads use this module. The old
folders stay as FROZEN records: their scripts are never edited or re-run as tools.

## 1. The interface

A read is one **spec** file (`read_spec.json`, committed with the PREREG) and one **state** directory. The state
lives outside the repo and outside `models/`, for example `~/gen3ai_archive/<read>/`.

```bash
python -m main.reads check-spec --spec read_spec.json
python -m main.reads plan     --spec read_spec.json --state DIR    # copy settled models; plan units
python -m main.reads run      --state DIR --lane anchor --hours 20 # detached, resumable; one per lane
python -m main.reads run      --state DIR --lane s      --hours 20
python -m main.reads status   --state DIR                          # resume = run again (or `resume`)
python -m main.reads record   --state DIR [--out <stage>/reads/<arm>_look<k>.json]
```

`record` runs `assemble` (writes `cells.json`) and then `decide` (writes `inputs.json` and `decision.json`). Both
are also verbs on their own. Exit 0 means success. Exit 2 is a typed refusal, named on stderr. Exit 3 means R moved
under an S lane. In Python, `main.reads.api` exposes the same verbs as functions.

**The spec** (`main.reads.spec`, schema `gen3_read_spec_v1`). Every field is validated, and an unknown field is a
refusal.

| field | holds |
|---|---|
| `models` | `{label: {arm, seed, ref, load?}}`. `ref` is `{"kind": "pinned", path, sha256}` (a frozen file) or `{"kind": "final", run, min_step, quiet_s, done_marker?}` (§2). `load` holds model-LOAD options every unit of the label passes (`spec.LOAD_OPTIONS`: `belief_prior_mix` λ ∈ (0, 1], `main.anchors --our-belief-prior-mix`), so two labels may play one file two ways |
| `anchor` | `{group: {opponent, teamsets, unit_seeds, models}}`: the Kakuna cells (§3) |
| `s` | `{group: {refpop, refpop_content_sha256, panel, pairs, batch_pairs, family, models, regime_id?}}`: S against a frozen R. **One eval-ledger family per group**, because a family holds one regime |
| `decision` | `{rule, params}`: a rule from `main.reads.rules.RULES`, with its registered parameters checked at load |
| `pin` | optional. The commit the playing checkout must be at, with a clean tracked tree |

Example. This is x26c's read written as a spec; `golden_test.x26c_spec` parses it.

```json
{"schema": "gen3_read_spec_v1", "read": "x26c_continuation_2026-10-25", "pin": "e05c45eeb",
 "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": {"kind": "pinned",
              "path": "models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip", "sha256": "8148…"}},
            "x26c_480m": {"arm": "x26c_480m", "seed": 2601, "ref": {"kind": "final",
              "run": "models/rb_x26c_s2601", "min_step": 480000000, "quiet_s": 300}}},
 "anchor": {"unfamiliar": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix", "forum72", "p72"],
                           "unit_seeds": "0-19", "models": ["pstar", "x26c_480m"]}},
 "s": {"pool": {"refpop": "designs/…/reference_population_production.json", "refpop_content_sha256": "cef0960e…",
                "panel": null, "pairs": 500, "batch_pairs": 100, "family": "x26c_s_pool_2026-10-25",
                "models": ["x26c_480m"], "regime_id": "adfdbee824c9eb0a"}},
 "decision": {"rule": "anchor_trajectory", "params": {"on": "x26c_480m", "off": "pstar",
   "anchor_group": "unfamiliar", "goal_bar": 0.5, "min_unit_frac": 0.9, "subset_below": 7,
   "s1": {"group": "pool", "label": "x26c_480m"}, "s_baseline": {"s_pp": 58.23, "se_pp": 0.55, "source": "…"},
   "k9b": "designs/…/k9b_at_end.json"}}}
```

How paths resolve: an absolute path is used as written; `models/…` resolves under the main checkout's run archive
(`utils.paths.main_models_dir`); anything else is repo-relative.

## 2. Which file is read (`main.reads.models`)

- **The settled rule.** A `final` ref reads `<run>/final_model.zip` and nothing else. It is settled when four
  things hold: the file exists, its recorded step is at least `min_step`, nothing has written it for `quiet_s`
  seconds (minimum 60), and the `done_marker` line is present if the spec names one. Until then the model is
  WAITING, and its units are not even planned.
- **No checkpoint fallback.** The x26c rule fell back to "the newest quiet checkpoint ≥ 480M". Twice, the Kakuna
  chain started on a checkpoint 81k steps short while the end chain was still writing the final (x26c FINDING 2).
  A read that wants a checkpoint names it as a `pinned` ref, with its sha256.
- **The copy.** Each settled model is copied into `<state>/ckpts/<label>/`. The checks are:
  - the copy must hash to the source, and the source to a pinned sha256;
  - the copy carries its `.json` sidecar and the run's `model_config.json`;
  - a missing `model_config.json` refuses at copy time (D1 look 2's void `h_d`, d1_read FINDING 4);
  - every unit re-hashes the copy before it plays.

## 3. Lanes, state, resume (`main.reads.runner`, `main.reads.layout`)

**The anchor lane** plays `main.x35.kakuna` units unchanged, through its own functions (`plan_units`,
`run_unit`, `is_done`, `MAX_ATTEMPTS`):

- 20-game units on the heavy slots' short lane (`mem_cap.sh --max-s 600`);
- 95xx ports claimed with a held flock;
- each unit preflighted: the T28 halt marker, the committed team manifest or panel cell, and the model bytes.

There is one kakuna state per (group, model), `<state>/anchor/<group>/<label>/`, planned when that model
settles. Unit seeds are the outer loop, so a stopped queue stays balanced by seed.

**The S lane** plays `main.h2h play-many` cells:

- `pairs` mirrored pairs per cell, recorded in batches of `batch_pairs`, into the group's own ledger and family
  (`<state>/s/<group>/ledger`);
- one request per node, `<group>_<label>_b<batch_pairs>`;
- a unit is up to `--workers` engines in ONE heavy slot (`mem_cap.sh --slot`), time-boxed to `--unit-s` (default
  2,400 s), at nice 19 on cores 5-7,13-15;
- R is re-verified before every unit (every member's sha256 and `content_sha256`). If R moved, the lane stops
  with exit 3 (the drift pre-read's hazard 1);
- the box ends at the registered stop at the latest (the slot wait does not extend it); an engine that exits
  non-zero on its own is reported even when another hits the box; a unit that ends OK with no new pair is a
  named refusal, never a loop; a cell counts only at the spec's `pairs` (>= 500 for `x35_r1` / `d1`).

**Both lanes:**

- one flock supervisor per lane per state;
- a registered stop (`--hours` or `--stop-at-epoch`, never open-ended);
- a counted unit or a recorded batch is never replayed;
- every child gets THIS checkout's `src` on `PYTHONPATH`;
- the playing checkout must stay at the commit `plan` wrote to `code.json`, and a dirty tree at the same
  `git diff HEAD` digest. This addresses "the playing code is not
  the pin" (d1_read 5, sprt 1, g1 2, x47 F-X47L1-4), which before was only ever checked by hand.

Run each lane detached, one process per lane:

```bash
setsid nohup python -m main.reads run --state DIR --lane s --hours 20 > DIR/s.log 2>&1 &
```

## 4. Assemble: the regime id is ALWAYS recorded (`main.reads.assemble`)

Five S drivers wrote `regime_ids: [""]` (x26c FINDING 4, g1 FINDING 3), and the regime was then checked by hand.
Here every group carries an id, or it is not conclusive.

**An S group's id** is the eval-ledger read's own (`LedgerRead.regime_id`), recorded with the read's digest. A group
is INCONCLUSIVE if any of these holds:

- it registers a `regime_id` and the ledger holds another;
- it names a `panel` and the rows' team set is a different one;
- a request has no row.

`s_node` is `readout.population_score` plus the per-member SE and se(S) = √Σse²/K.

**An anchor unit's id** is `anchor_regime_id`: a digest over the cell identity that `summary.json` stamps:

- the opponent and its version;
- both regimes;
- the transport and the server;
- the format and the forfeit limit;
- both team-content digests;
- the declared team cell;
- a Metamon peer's verified weights (`their_weights` / `our_weights`: hub revision + `weights_sha256`), included only when they
  differ from `assemble.LEGACY_WEIGHTS` (the one commit every pre-pin read played), so banked ids are unchanged and a
  re-read of the same weights keeps the id (`EXTERNAL_ANCHORS_SOP.md` § Weight pins).

Each (group, team set) must share ONE id. The group's id is `group_regime_id` over its team sets. The playing commit
and file paths are recorded per unit but are not part of the id. A unit played on bytes other than the resolved
model's is a refusal, never a pooled row. A model-load option changes the id through `our_regime` (a
`belief_prior_mix` unit is stamped `greedy+belief_prior_mix=<λ>`), so each option needs its own group. Every unit
row also carries its `games` (`[half, index, our_team, result, turns]` from `games.jsonl`), which is what a
game-for-game determinism guard compares.

## 5. Decide: the registered rules (`main.reads.rules`)

A rule is two pure halves: `inputs(cells, params, spec)` produces the rule's input document, and `decide(doc,
params)` produces the verdict as recorded. A decision can therefore be re-checked from `inputs.json` alone.

| rule | is | from |
|---|---|---|
| `x35_r1` | X35 R1 / R2 / guards / verdict on S per seed final (SPRT, LRF, A1 …); `import_off` takes a banked control from another read's committed `results.json`, sha-checked | `main.x35.readout.read` |
| `d1` | D1's H (the arm's primary Kakuna panels, `d1.arm_panels`: D1's / D3's three, X61P's `forum72` alone at 60 units; widened by AM-5's φ̂) and P (S on home72); an optional `reported_group` (an anchor group disjoint from the primary, same models) adds per-panel rows, U and Δ_H − Δ_F to `reported.reported_panels`, which no rule reads | `main.x35.d1.read_d1` |
| `anchor_trajectory` | x26c PREREG §5: the goal bar on the ON score's Wilson LB; CLIMBING / FLAT on the Newcombe-of-strata Δ; S1 RISING against a banked node; VOID on K9(b) or under-counted units | ported from x26c `verdict.py` |
| `prior_mix_ab` | the belief-prior-mix read (`measurements/belief_prior_mix_2026-10-26/PREREG.md` §5): each λ arm (its own anchor group, one regime) against a BANKED λ = 0 control (another read's committed per-game rows, sha-checked, never re-played; `banked_units`), unit-paired on the same seeds; VOID unless a guard group re-playing banked units at λ = 0 reproduces them GAME FOR GAME; PRIOR-HELPS / -AT-A-COST / -HURTS / NOT DETECTED; D0f's DiD and H3 kill rule on a λ = 1 arm | new (2026-10-26) |

**The estimators are not re-derived.** Each is the one function a read already used, re-exported under a name:

- `welch_interval` and `classify` (readout);
- `strat_score`, `newcombe_strat`, `overdispersion`, `widen` and `wilson_pn` (d1);
- `wilson` and `newcombe` at z = 1.959963985 (anchors);
- `newcombe_diff_ci` at z = 1.96 (`best_response_gap`);
- rule 8 as `clears`.

Migrated here from the scripts, which had no home in `src/`:

- `cell_se` and `s_se`;
- `paired_unit_diff`;
- `guards_not_read`, the orchestrator's A1 ruling (a): guards gate only a QUALIFYING primary, so an unplayed guard on
  a non-qualifying primary is recorded INCONCLUSIVE with the read's reason.

A new rule is a new `Rule` entry with a test. It never replaces an existing rule's numbers.

## 6. The record (`main.reads.record`, schema `gen3_read_record_v1`)

The record holds:

- `read`, `rule`, `spec_sha256`, `prereg`, `verdict` and `created`;
- `models` (each with the sha256 played);
- `regimes` (`{anchor: {group: id}, s: {group: id}}`);
- `ledgers` (each S group's family, digest and row count);
- `code` (the playing commit);
- `cells_inconclusive`;
- the rule's `inputs` and `decision`.

The record is refused if any regime id is missing, a model was never resolved, a ledger digest or the playing
commit is missing, `cells_inconclusive` is not empty (a verdict decided on cells assemble could not count: a wrong
regime or panel, unplayed units, a cell short of the spec's `pairs`), or the verdict is not the decision's. An X35-stage rule (`x35_r1`, `d1`) also carries the
decision's keys at the top level. When written to `<stage>/reads/<arm>_look<k>.json`, the file therefore loads in
`main.x35.batch.read_record` and `final_class` unchanged. A ledger entry cites the record by its path and
`record.summary_line`.

## 7. Golden reproductions (`src/main/reads/golden_test.py`)

Three shipped reads, re-decided by the module, reproduce their recorded verdicts and intervals EXACTLY, by float
equality.

| read | recorded | committed half (always runs) | archive half (owner's box) |
|---|---|---|---|
| SPRT look 1 | NULL, Δ −0.25 pp [−1.65, +1.16] | the input → `x35_r1`: `r1`, `r2`, `guards`, `reasons` and verdict equal `reads/sprt_look1.json` | re-assembled from `~/gen3ai_archive/x35_sprt_read/ledger`; C0 from `x35_g1/ledger` equals the imported `g1/results.json`; regime `adfdbee824c9eb0a`; the record loads in `batch.read_record` |
| D1 look 2 | UNDECIDED, H +2.10 pp [−0.92, +5.11] | the input → `d1`: `h`, `p`, `guards` and verdict equal `reads/d1_look2.json` | H from the five kakuna states `h_a`…`h_e`, P from `p/ledger`, both equal the committed input; regimes: Kakuna `09f297883701a35a`, home72 `d4de0fd7c16a2658` |
| x26c | goal (b) not met; FLAT; Δ +0.08 pp [−3.90, +4.06]; S1 −0.01 [−1.54, +1.52] | the committed unit rows + S1 → `anchor_trajectory`: the `kakuna` block and every decision value equal `verdict.json` | the two kakuna states reproduce the committed `kakuna_units.jsonl` row for row; the three S nodes equal `s_results.json` (S, se, CI, per-member); regimes recorded: Kakuna `09f297883701a35a`, pool `adfdbee824c9eb0a`, home72 `d4de0fd7c16a2658` |

D1's and x26c's Kakuna units, played at different commits on the same committed configs, carry the SAME group regime
id. This matches x26c FINDING 1.

The archive half reads `~/gen3ai_archive` read-only (`$GEN3AI_READS_ARCHIVE` overrides). Where it is absent, it
skips and names why.

## 8. The inventory it replaces, and the bug classes it closes

| read (frozen) | scripts | pieces it hand-rolled |
|---|---|---|
| X26 drift pre-read | `x35_breakers_stage_2026-10-17/x26_drift_preread/preread.py` | S cells on slots, copy_checked, se(S), ledger read |
| G1, D1 (P, pool S), SPRT | `g1/g1_read.py`, `d1_read/d1{p,s}_read.py`, `sprt_read/s_read.py` | near-verbatim copies of the pre-read machinery |
| D1, SPRT assembly | `d1_read/assemble.py`, `sprt_read/assemble.py` | merging kakuna states and requests, the A1 guard ruling, imported C0 |
| x26c | `x26c_continuation_2026-10-25/{finals,s_read,verdict}.py` | the final-selection rule, S cells, the trajectory rule |
| X47 look 1 | `x47_look1/{play_cells,assemble_read}.py` | per-cell play-many, population score |
| Kakuna-era queues | `x26_metamon_anchor/xq.py`, `x51…/x51q.py`, `x52…/x52q.py`, `x37a…/xq.py`, `heldout*/hq*.py`, `k2…/kq2.py` + their `aggregate.py` | ports, units, Wilson / Newcombe tables |
| Metamon strongest, human-only FR (RUNNING 10-25; untouched) | `metamon_strongest_2026-10-25/scripts/{mmq,aggregate}.py`, `human_only_vs_pstar_2026-10-25/scripts/{hrrq,aggregate}.py` | copies of `claim_port`, other opponents' units, per-read classes |

| bug class (where recorded) | recurs | closed by |
|---|---|---|
| the final-selection race (x26c F2) | three readiness rules in use | the settled rule, no checkpoint fallback (§2) |
| hand-counted `parents[N]` (x26c F3; xq.py `parents[5]`) | copies at different depths | `utils.paths` only; no script paths |
| empty `regime_ids` (x26c F4, g1 F3; 5 copies) | yes | regimes always recorded; a record without one refuses (§4, §6) |
| port race (d1_read F3; x51 F6) | `claim_port` re-copied in mmq / hrrq | kakuna's held-flock claim, reused |
| copied finals without `model_config.json` (d1_read F4; refpop F-R4) | yes | refused at copy time (§2) |
| R moving under a read (drift hazard 1) | every S driver | re-verified before each unit, exit 3 (§3) |
| playing code ≠ pin (d1 5, sprt 1, g1 2, x47 L1-4, hr 1, x51 9) | the most common | `code.json` + `pin` (§3) |
| heavy-slot starvation (d1_read F2; drift 2-3; x51 F5) | yes | short lane for Kakuna units; time-boxed S units in one slot each |
| rule-8 and statistic copies (≈10 scripts' `EPS`; x26c's `clears`, hand-copied baselines) | yes | one `clears`; baselines in the spec with their source |
| a missing panel argument to `strat_score` (x26c `verdict.py`, correct only because its panels were d1's default) | latent | panels always passed (rules_test caught it) |

## 9. What was migrated, wrapped, left

- **Wrapped, numbers unchanged:**
  - `main.x35.readout` (R1 / R2 / guards / verdict);
  - `main.x35.d1` (H / P / φ̂, `refpop_members`);
  - `main.x35.kakuna` (units, ports, preflight, counting);
  - `main.anchors.results` (`wilson`, `newcombe`, `unit_counts`).

  `main.x35.batch.read_record` loads the module's X35-stage records as they are.
- **Migrated:**
  - from the per-read scripts: the S-cell runner, `copy_checked`, the per-cell and per-final SE, the trajectory rule,
    the unit-paired difference, the A1 ruling and the final-selection rule;
  - from `d1.p_input` / `kakuna.d1h_input`: their equivalents over assembled cells (`rules.p_input`,
    `rules.h_input`).
- **Left:**
  - every frozen measurement folder;
  - the reads RUNNING on 2026-10-25 (Metamon strongest, the human-only FR-1 / FR-3 reads, the X59-S build, D3's
    arm-end). They finish on their own scripts.

## 10. Not built (each a FINDING of the build)

- **Anchor opponents other than Kakuna.** `mmq.py` / `hrrq.py` play PastaMittens, SmallRL, LargeRL and LargeIL with
  their own configs. Adding them is an `ANCHOR_OPPONENTS` entry plus a config-and-manifest check in the kakuna unit
  path (`HF_HUB_OFFLINE`, per-opponent configs).
- **Reported-only extras** (SPRT's mechanism block, dose from launcher summaries) stay per-read. A rule's `params`
  can carry them, but no assembler reads run dirs.
- **No VOID gate in `x35_r1`.** SPRT's assembly stopped on a seed whose K9(b) or V-S1 / V-S2 was not PASS
  (`batch2_void.read_void`); the rule does not read run dirs, so a read that registers such a gate checks it
  before `record` until a `void` parameter is built.
- **The always-run D1 golden re-decides only.** `rules.h_input` / `p_input` are compared with the shipped
  input only in the archive half (D1's per-unit rows are not committed); off the owner's box that half skips.
- **Never run end to end on real play.** The verbs are tested with a stubbed unit runner and stubbed heavy slots
  (`runner_test`, `pipeline_test`). The play-many and anchor argv are checked against the real parsers. The first
  new read is the module's first live run.
