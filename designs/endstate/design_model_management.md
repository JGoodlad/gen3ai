# Design — Model management, the end state

**Status: DESIGN, not scheduled.** Authored 2026-09-22 at the owner's request. It states what the
model-side of the system looks like when the environment side of
[`design_three_tier_environment.md`](design_three_tier_environment.md) is in place — but most of it
does not depend on that document and several parts are already true. Where a part exists today it is
named with its record; where it does not, the gap is named.

---

## 0. The one-paragraph version

A **model** is a named, immutable artifact with a recorded identity (architecture signature, config
version, weights hash, lineage, the exact commit and command that produced it, the dose it trained
at, and the eval regime its numbers were measured under). Every consumer — the trainee, the self-play
pool, the sentinels, the baselines, the anchors, the search leaf, the ladder client — obtains a model
**by name through one registry**, never by path, and runs it **through one inference tier**, never by
loading its own copy. Promotion, retention, and every cross-run comparison are policies over the
registry, executed by tools that print the ledger line they imply. The registry's authority is the
run directory's own recorded metadata; nothing is re-derived.

---

## 1. What exists today, and what each piece already gets right

| piece | what it does | record |
|---|---|---|
| `model_config.json` (per run) | the weight-SHAPE record; `check_compatible` makes a mismatch a `[ModelVersion] FATAL` at startup | root `CLAUDE.md`; `src/agents/model/model_version/` |
| `metadata.json` (per run) | provenance; three IMMUTABLE blocks written once — `original_command`, `lineage` (who forked whom, with the file every reference RESOLVED to), `pin_history` (which commit ran which steps); the `matchup` hash; `eval_sentinel_greedy` (the eval regime) | root `CLAUDE.md`; `designs/training/matchup_and_lineage.md` |
| `agents.training.lineage` / `main.lineage` / `main.sidecar_audit` | the only readers of those blocks; never re-derive | same |
| `resolve_model_ref` | one choke point turning a spec (`run`, `run@step`, `run/file.zip`) into a `ResolvedModel`; a BARE run directory means the run's LAST SNAPSHOT | `designs/training/matchup_and_lineage.md` |
| `designs/baselines.json` + `agents.training.baselines` | NAMED baselines (`production`, `v9_long_baseline`, `v9_fold_parent`, `famine_comparator`, `untaught_meter_opponent`, …), each pinning an EXPLICIT checkpoint so the last-snapshot rule cannot move it; `baselines.load(name)` is THE by-name load (sanitizes deleted kwargs; raises a typed `BaselineLoadError` with `.reason` ∈ {pre_generation, arch_drift, unresolvable, not_a_model} naming the fix); `python -m main.baselines set … --reason` is the only mutator and prints the ledger line | landed `80560c85` (2026-09-22) |
| the self-play pool | snapshots promoted at a win-rate gate (`--promote-threshold`, regime-following 0.55 / 0.65), a pool of the run's own past selves; a FORK auto-seeds its parent's pool and exits `FATAL_CONFIG` with none | `designs/training/self_play_and_pool.md` |
| the dense snapshot ladder | `snapshot_ladder/ladder.json` — Bradley–Terry over frozen pairs; the headline rating; now carries a `recipe` block and every reader refuses or refits on a stale stamp | landed `0f230405`; audit `measurements/ladder_refit_audit_2026-09-22/` |
| `models/` retention | a written policy plus a dry-run tool; nothing has been deleted; baselines are a protected keep-list | `designs/research_state/models_retention_policy.md` |
| `main.dose` | reads a run's realized step size (lr × epochs / (batch × accum)) from the run — the only honest dose | `designs/training/step_size_and_batch.md` |
| the fork-lr guard | a fork of a FROZEN parent must NAME its own dose or refuse (`gen3_fork_lr_inherit_guard_v1`); 3 of 162 archived forks fire it, exactly the three era-2 exploiters whose 4.5× gap cost a campaign | landed `35258dcc` |
| the untaught-teacher guard | a pinned trainee team in the untaught-8 manifest is a startup FATAL; 5 of 89 archived pinned runs fire it | landed `1393d192` |
| `main.best_response_gap` | the population loop's meter; REFUSES unmatched budget / dose / regime | landed `5a738ec1` |
| `main.anchors` | external anchors by name (`metamon:SmallRL`, `metamon:SyntheticRLV2`, `foulplay`), regime verified per decision, transport stamped on every row, server = the Rust front end by default | `designs/ops/EXTERNAL_ANCHORS_SOP.md` |

**What these already get right, and the end state keeps:** immutability of provenance; by-name
resolution through one choke point; explicit pins so nothing moves under a reader; refusal over
silent fallback; every mutation printing its ledger line; the regime travelling with the number.

---

## 2. What is missing — the gaps this month exposed

Each row is a thing that went wrong in September because the piece did not exist.

| gap | what happened | date |
|---|---|---|
| **No model catalogue at runtime** | six consumers load their own copy (per-env opponents at B = 1 on CPU; each eval worker its own sentinel; the search leaf its own materializer + forward); the sum is the box's CPU budget | standing |
| **The optimisation block is not in the identity** | `model_config.json` (144 keys) carries no lr / batch / epochs / accum; the dose lives only in `metadata.json`'s `original_command`; a fork inherited a frozen NUMBER without the FREEZE and its controller annealed 3× mid-run, unrecorded until `main.dose` read it after 14 GPU-h | 2026-09-20 |
| **The regime is not part of the number** | every banked vs-target rate for the exploiters was greedy-vs-greedy eval, not the training regime; discovered by reading `eval_worker`'s fixed-opponent branch | 2026-09-22 |
| **A committed rating file carried no recipe** | 68 of 93 `ladder.json` files move on refit under the current fitter; the newest node reads high in 65; the v9 generation ladder reverses 21 of 153 orderings | 2026-09-22 |
| **A registry entry could point at an unloadable node** | `production` pointed at a pre-generation node; the bare loader failed on every current entry too | 2026-09-14 → 22 |
| **Teacher admission was by head-to-head, not by transfer** | three exploiters at 0.66 / 0.53 / 0.46 vs their target were LEVEL with it on their own teams against a third party; the fold that distilled them cost ~10 pp untaught | 2026-09-21 |
| **A fourth lever went unregistered** | `--team-block-episodes 64` differed between a fold and its control; nobody diffed the argvs; the config records the flag but no tool compared two runs' resolved configs | 2026-09-20 |
| **The comparator moved** | a 75M win-prob parent gains +15.5 pp untaught from plain continuation; every delta-against-a-frozen-parent on that side was inflated; the fix was a plateau parent, established by a run | 2026-09-20 |

---

## 3. The end state

### 3.1 The model record — one identity, complete

A model is a directory (today's run directory, unchanged) plus a **`model_record.json`** that is the
union of what `model_config.json`, `metadata.json` and `main.dose` know, written once at each
snapshot and never edited:

```
{
  "id":            "ai_v13_12_plateau@95158272",         # run @ step — the NAME
  "arch": {        "signature": "gen3_…", "config_version": 121, "weights_sha256": "…" },
  "provenance": {  "commit": "…", "original_command": "…", "pin_history": [...],
                   "lineage": { "parent": "ai_v13_09_wcont@87097344", "fork_step": …, "resolved_file": "…" } },
  "optimisation": { "lr_at_save": 2.8e-5, "lr_frozen": true, "batch": 2048, "accum": 32,
                    "epochs": 10, "dose": 4.272e-9, "dose_x_v8": 0.20 },   # THE MISSING BLOCK
  "ecology": {     "matchup": "ef5242cffd", "team_block_episodes": 1, "trainee_teams": [...],
                   "distill": { "coef": 0.0, "target": null, "topk": null, "teachers": [] },
                   "pool": { "seeded_from": "…", "n": 20 }, "exploiter_target": null },
  "eval_regime": { "sentinel_greedy": true, "fixed_opponent_greedy": true,   # named, per branch
                   "trainee_greedy_in_eval": true },
  "ratings": {     "ladder": { "recipe": {...}, "newest_node": 2019.1, "n_nodes": 20 },
                   "anchors": [ { "opponent": "metamon:SmallRL", "regime": "greedy", "teamset": "away",
                                  "n": 1200, "p": 0.553, "ci": [0.525, 0.581], "server": "rust",
                                  "hazards": ["H18"] } ] }
}
```

Three rules. **(1) Every block is written by the thing that knows it** — the trainer writes
`optimisation` and `ecology` at save, the ladder fitter writes `ratings.ladder`, `main.anchors`
appends to `ratings.anchors`; nothing is re-derived. **(2) A number never appears without its
regime and its hazards** — the field shape forbids it. **(3) A diff of two records is a tool**
(`python -m main.model_diff A B`) that prints every resolved difference, so the "fourth lever" class
cannot recur: a fold registration includes the diff against its control, generated.

### 3.2 The registry — names, roles, policies

`designs/baselines.json` grows into **`designs/model_registry.json`**, still named entries pinning
explicit checkpoints, with three additions:

- **Roles**, not only names: `trainee`, `pool_member`, `sentinel`, `baseline`, `teacher`,
  `anchor_reference`, `plateau_parent`. A role has an **admission policy** (below); a model holds a
  role only after its policy's tool has printed the ledger line.
- **Admission policies as code**, each an offline meter that REFUSES rather than guesses:
  - `teacher` ← `main.admission`: OUTSIDE the floor above the *continuation or plateau parent* (never
    the frozen parent) on its own pinned teams against `untaught_meter_opponent`, 800 games/team, the
    rule `|Δ| > floor AND CI excludes the floor point`. This is the gate that refused four exploiters
    in September and is the reason none was folded.
  - `pool_member` ← the promotion gate as today, plus a recorded regime.
  - `sentinel` ← matched-count ladder position, refit under the current recipe.
  - `plateau_parent` ← a continuation block whose untaught Δ is WITHIN the floor (the 2026-09-21
    stop rule), confirmed once externally at ≥ 1,200 anchor games.
  - `baseline` ← loads through `baselines.load` or raises the typed error; era recorded.
- **The catalogue manifest for Tier 2**: which named models the inference tier should hold
  resident (the trainee, the pool, the sentinels, the baselines a running read needs) and the
  eviction order. Today each consumer loads its own copy; in the end state the manifest is the only
  place a model is "loaded", and a consumer asks by name.

### 3.3 The pool and the population loop

The self-play pool becomes a **population** with two kinds of member and one meter:

- **past selves** (today's snapshots, promoted at the gate) and **exploiters** (best responses
  trained against a named target, admitted as *opponents* — never as teachers unless they clear the
  teacher policy separately). The lesson of 2026-09-21 is written into the roles: an exploiter's
  head-to-head win rate licenses `pool_member`; only the transfer gate licenses `teacher`.
- **weights**, not a flat pool: an exploiter enters at a weight that bites (the split arm showed two
  specialists at pool weight left entropy and untaught indistinguishable from doing nothing), with
  the weight recorded in `ecology.pool`.
- **the meter** is `main.best_response_gap`: gap(t) = P(a fresh exploiter beats G_t) − 0.5 at
  matched budget / dose / regime, per round × archetype; the loop is working iff it falls. The
  registry records each round's exploiters and the gap they read, so the loop's history is a table,
  not a ledger search.

### 3.4 The comparator rule, made structural

Every delta is against a **named comparator with a named role**, and the tool refuses a frozen
parent as the comparator on the win-prob side unless the parent holds the `plateau_parent` role.
`main.untaught_meter`, `main.admission`, the fold reads and `main.best_response_gap` all take
`--comparator <name>` and print its record's `id` beside every number. The +15.5 pp inflation of
2026-09-20 becomes impossible to bank unnoticed, because the comparator's role is printed.

### 3.5 Ratings — one recipe, one stamp, one refit tool

Already landed in shape (`0f230405`, `7db504ac`): every `ladder.json` carries a `recipe`; readers
refuse or refit; `main.elo refit --apply` re-stamps. The end state adds: the refit is applied to
the archive once (93 runs, the owner's or the Training Run session's call — the audit says 68 move,
newest node high in 65), the v9 generation-ladder entries are re-read under the current recipe
(their orderings reverse 21 of 153 pairs; the re-read is not done), and a live arm is refit at run
END as policy.

### 3.6 Anchors — the external scale, with its hazards attached

`main.anchors` already stamps regime, transport and server on every row. The end state attaches
**hazards** as data (`H18`: Metamon drops Baton Pass boosts; bias favourable to us; size from
`measurements/h18_baton_pass_bias_2026-09-22/` when it lands) so a strength claim carries the
correction, and reads by name against `anchor_reference` models whose own faithfulness record
(`metamon_obs_faithfulness_2026-09-22/`) is linked from the registry. A new external opponent gets
the same four checks before it is named.

### 3.7 Retention, made a function of the registry

The retention policy already protects the baselines' files. The end state protects every model that
holds ANY role, every model a banked ledger entry names by id, and every plateau parent; everything
else follows the written tiers. The dry-run tool prints what it would groom and which ledger entries
would lose a resolvable reference — and refuses to groom a file the registry names.

---

## 4. The interfaces, stated

```
registry.get(name) -> ModelRecord                 # by NAME; never a path
registry.resolve(spec) -> ResolvedModel           # run | run@step | run/file.zip — unchanged
registry.roles(name) -> {role: admitted_at_commit}
registry.admit(name, role, --reason) -> ledger line   # runs the role's policy; refuses or prints
registry.diff(a, b) -> every resolved difference   # config ∪ optimisation ∪ ecology ∪ regime
inference.score(model_id, rows, masks, mode)      # Tier 2 — the only way a model runs
inference.manifest() / .evict(name)               # what is resident
```

Every offline meter (`main.elo`, `main.untaught_meter`, `main.critic_gate`, `main.exploitability`,
`main.best_response_gap`, `main.anchors`, `main.dose`, `main.lineage`) reads through `registry` and
runs through `inference`. None loads a network itself.

---

## 5. Ordering

| step | what | depends on | status |
|---|---|---|---|
| 1 | `optimisation` + `ecology` + `eval_regime` blocks written at save; `main.model_diff` | nothing | **not started** — cheapest, closes the two September classes (dose, fourth lever) |
| 2 | `main.admission` as the teacher gate tool (the harness in `admission_artifacts/` made a CLI); the `--comparator` rule with role printing | 1 | harness exists; CLI not started |
| 3 | roles in the registry; `registry.admit` printing the ledger line | 1, 2 | not started |
| 4 | apply the ladder refit to the archive; re-read the v9 entries | nothing | tool landed; apply is the owner's call |
| 5 | H18's size attached to every anchor row | the bias cell | cell in flight |
| 6 | the Tier 2 catalogue manifest; opponents move first | the inference tier (companion doc §4) | not started |
| 7 | retention as a function of roles | 3 | not started |

Steps 1–3 are Python, need no parity oracle, and land between reads. Step 6 is the companion
document's step 4.

---

## 6. What this document does not decide

- Whether the population's exploiter weight is a fixed share or PFSP-style (`--pfsp-scale` exists,
  OFF) — an experiment.
- Whether a `teacher` role should exist at all in the next era, given four refusals at 0.39× dose
  and one pending at 1.78× (`ai_v13_18_teach5_offense_hidose`, live) — the hidose read decides.
- The registry's storage: one JSON (today) or one record per model with an index; the latter when
  the count passes a few hundred.
