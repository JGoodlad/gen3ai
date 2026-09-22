# The search-teacher COMPOSITION on rust — re-measured, and three assertions it was missing

**Date** 2026-09-22 · **Branch** `stc` · **Box** 16 cores, load ~19 at launch, a GPU arm and
several agents live · **Run dir** `/home/goodlad/.claude/jobs/9ab51de6/tmp/stc/basetemp/` (never
under `models/`) · **Engine** `--use-bridge rust`, main's prebuilt `sim_bridge` + `search_driver`.

## 0. What was already true before this pass

Both backlog rows this pass was dispatched against were **already CLOSED**, and the premises they
were dispatched with are stale:

* *"candidate SELECTION ran INLINE in `_on_step`"* — fixed at **`e22c1236`** (2026-09-08). A cycle
  is two phases and **both are children** (`_pending["phase"]` is `select` then `search`);
  `main.search_teacher_select_worker` publishes `teacher_cycle/candidates.json` atomically.
* *"a full multi-cycle run end-to-end on rust is GATED, the composition is not"* — the composition
  gate landed at **`49fb43ce`** (2026-09-07) as
  `src/main/train/search_teacher_composition_test.py`.

So there was **no inline selection to fix** and **no composition gate to build**. What this pass
actually found and closed is listed below.

## 1. The real hole: the gate's verdict was never RECORDED

`designs/ops/slow_tier_status.json` contained **no row** for
`test_search_teacher_runs_multiple_cycles_end_to_end_on_rust`. The gate is `sim` + `slow`, so the
ROUTINE gate deselects it — and per the root `CLAUDE.md`, *a deselected test cannot fail*. The
slow-tier status mechanism exists precisely to close that, and this test had never been run under
it. It now has a row:

```
status   pass
commit   eda0d0a372ea73361672b079b5abf6ab95cd6f8d
at       2026-09-22T21:16:28Z
duration 572.45 s
contention 2.1
```

## 2. The per-cycle table (this run, 30,000 steps)

| launched @ step | selected | **searched** | labelled | wall (s) | worker status histogram |
|---|---|---|---|---|---|
| 6,373 | 8 | 8 | 2 | 38.1 | `{gate_failed: 5, already_known: 1, ok: 2}` |
| 10,617 | 3 | 3 | 0 | 16.0 | `{gate_failed: 3}` |
| 15,997 | 8 | 8 | 1 | 36.1 | `{gate_failed: 7, ok: 1}` |
| 24,103 | 8 | 8 | 2 | 34.3 | `{gate_failed: 5, ok: 2, already_known: 1}` |
| 29,470 | 8 | — | — | **PENDING** | *(the run ended while this cycle was live — a SIZING fact, not a defect: nothing waits for a per-cycle worker at `_on_training_end`)* |

**5 launches · 4 collects · 5 corrections · 572 s wall.** Wall resolution is ±2 s (the harness
samples marker arrival at its poll interval; the child log carries no timestamps of its own and
three regexes match the marker format, so adding a clock to the print would break them).

Every worker config carried `"impl": "rust"`. `teacher/loss`, `teacher/ce`, `teacher/n` and
`grad/searchteacher_share` were all present and non-zero.

## 3. `_on_step` wall — before and after

The honest before/after is **not** an A/B of total step wall on this box. Under a load average of
19 on 16 cores, the between-run variance of a step timing swamps the quantity; and the teacher's
contribution to `_on_step` is already instrumented directly at the only three places it can enter
(`teacher/step_block_ms`, labelled by path). A direct measurement of the addend beats a differenced
measurement of the total whenever the addend is instrumented — which is why the callback instruments
it.

| | selection's cost to the training step |
|---|---|
| **BEFORE** (`49fb43ce`, selection inline in `_on_step`) | **48.1 s** over 9 loss traces, **350.2 s** over the default 60 — *per cycle*, blocking the SB3 loop |
| **AFTER** (`e22c1236`, both phases are children) — re-measured here | **0.46 s over the whole 30,000-step run**: `select-launch` n=7 max **28.0 ms**, `worker-spawn` n=5 max **115.9 ms** (that one is `model.save`, the freeze a cycle always paid), `collect` n=4 max **0.9 ms** |

~15 µs per step on average, against a defect that cost 48–350 s per cycle: five orders of
magnitude. The gate asserts the worst single step against 5 s × contention scale — two orders
above the measurement, two below the defect, so it can only be crossed by a regression.

## 4. Three assertions the gate was missing, now added

1. **The SEARCHED count, per cycle.** `produce_correction` returns exactly one status reason per
   candidate and the worker increments its histogram once per return, so the histogram's TOTAL *is*
   the searched count and must equal what selection offered. Without this, a cycle that selected 8
   and silently searched 2 reads **identically** to a healthy zero-yield cycle: `n_ok` can be 0 for
   an honest reason (`gate_failed`), and the missing candidates produce **no key at all**, so the
   `error:*` check sees nothing. Absence is not a zero. Held on all four cycles (5+1+2, 3, 7+1,
   5+2+1).
2. **The AWR loss is NON-ZERO, not merely present.** The gate asserted the tag existed. A
   `teacher/loss` recorded as identically 0.0 — a coef silently resolved to 0, a CE against the
   action already taken — writes the tag and passes. `--search-teacher-coef 0.5` is in the argv so
   the loss half is gated, and it is only gated if the number is asserted rather than the key.
   Same for `teacher/ce` and `grad/searchteacher_share > 0`.
3. **The per-cycle wall**, printed as the table above, previously available only as a run total.

**A naming correction recorded in the test itself:** the dispatch asked for `grad/distill_share`.
That is a **different term** — the EXPLOITER-distillation KL's shared-trunk share, from
`--distill-teacher` — and it is correctly **ABSENT** from this run, which passes no such flag. The
search-teacher's share is `grad/searchteacher_share`. A note in the assertion message says so, so
the next reader does not re-raise it.

## 5. `search_impl_parity` — the verdict is NOT "byte-identical", and cannot be

The dispatch asked to confirm *"a search-teacher cycle's labels are byte-identical between
`--search-impl node` and `rust` on the same candidates and seed"*. Two findings:

**(a) There is no `--search-impl` flag.** The engine is not separately selectable: it flows
`--use-bridge` → `args.bridge_impl` → the callback's `impl` → the worker config's `"impl"` key →
`ProbeSession._impl` → `better_line_decision(impl=…)` and `replay_counterfactual_battle(impl=…)`.
The composition gate asserts `"impl": "rust"` on every worker config, which is that seam.

**(b) Byte-identical labels are impossible BY CONSTRUCTION whenever the confirm opponent is a
checkpoint — and the reason is not the engine.** A label is
`(obs, mask, better_action, advantage)`. Tracing each field:

* `better_action` (A\*) comes from `better_line`, whose node≡rust candidate values are **already
  gated bit-identical**.
* `advantage` = `confirm.win_rate − 0.0`, from `replay_counterfactual`'s Monte-Carlo rollouts. The
  **sim dice are deterministic**: `fresh_seeds(n, salt=f"{battle_tag}:{inv_index}:cf")` is a
  sha256 of the salt, so the PRNG seeds reproduce exactly. The **trainee plays greedy**
  (`stochastic=False`).
* **But the reloaded checkpoint opponent plays STOCHASTIC at temp 1.0**
  (`_RECORDED_CKPT_STOCHASTIC = True`, matching `eval_worker`'s sentinel regime) — and there is
  **no `torch.manual_seed` / `np.random.seed` anywhere in the confirm path**
  (`prober/replay.py`, `prober/falsifier.py`, `teacher/*.py`, `main/search_teacher_worker.py`).

So the sampled opponent actions come from torch's unseeded global RNG. The consequence is stronger
than a parity gap: **`advantage`, and the `ok` / `gate_failed` verdict that depends on the Wilson
bound, are not reproducible run-to-run even at FIXED impl** against a checkpoint opponent.
"Same candidates and seed" does not pin the label, so no cross-impl byte-identity claim can be
made or tested in that configuration.

Where it **is** deterministic and therefore testable: a **BOT** opponent (rebuilt reproducibly,
deterministic policy) with the greedy trainee and the sha256 dice. That is the case a future
cross-impl label-identity gate should be built on, and it is a narrower claim than the row implied.

**(c) The existing harness could not be run.** `src/rust_sim/harness/search_impl_parity.py` gates
node vs rust at the driver WIRE level (18,877 leaf fields, allowlist forgiving only the absence of
the port-only `view_pN` payload). It requires `tmp/search_golden_node.json`, a **scratch artifact
that is absent** from both the main checkout and this worktree — as the ledger already records.
Capturing a fresh golden needs node plus `tmp/search_golden.py`, also absent; out of scope here and
recorded as such.

## 6. What is open

* **The unseeded confirm RNG** (§5b) — filed as a new backlog row. It makes every `advantage` a
  non-reproducible draw and is a prerequisite for any label-level parity gate.
* **Selection still falsify-gates on NODE under `--use-bridge rust`** — the pre-existing P2 row,
  untouched here; §5b compounds it (one half of the teacher is on the other engine *and* the
  confirm half is unseeded).
* **A cross-impl label-identity gate on the BOT-opponent configuration** — buildable, not built.

---

## Ready-to-append ledger paragraph

> **2026-09-22 — the search-teacher composition gate's verdict was never RECORDED, and its
> labels are not reproducible.** The two backlog rows dispatched for this pass were already closed
> (`49fb43ce` built the composition gate 2026-09-07; `e22c1236` moved selection out of `_on_step`
> 2026-09-08), so there was no inline selection to fix and no gate to build. The real hole was one
> layer out: `designs/ops/slow_tier_status.json` carried **no row** for
> `test_search_teacher_runs_multiple_cycles_end_to_end_on_rust`, so the `sim`+`slow` gate that
> proves the whole ExIt pipeline composes on rust had never been run under the recorder — and a
> deselected test cannot fail. It now records **pass, 572.45 s, contention 2.1**: 30,000 steps,
> **5 cycle launches / 4 collects / 5 corrections**, every worker config `"impl": "rust"`, and the
> teacher's total cost to the training step re-measured at **0.46 s over the whole run**
> (`worker-spawn` worst 115.9 ms, which is `model.save`) against the **48.1 s / 350.2 s per cycle**
> the inline defect cost. Three assertions were added that the gate lacked: the **SEARCHED count
> per cycle** (the status histogram's total must equal the candidates selection offered — a cycle
> that quietly searched 2 of 8 was previously indistinguishable from an honest zero-yield cycle,
> because missing candidates emit no status key at all), the AWR loss being **non-zero rather than
> merely present**, and a per-cycle wall table. **The parity half returned a negative result that
> is worth more than the gate**: there is no `--search-impl` flag (the engine flows from
> `--use-bridge`), and byte-identical cross-impl labels are impossible by construction — the
> confirm rollouts' sim dice ARE deterministic (`fresh_seeds` is a sha256 of
> `battle_tag:inv:cf`) and the trainee plays greedy, but a reloaded **checkpoint opponent plays
> stochastic at temp 1.0 with no `torch.manual_seed` anywhere in the confirm path**, so a label's
> `advantage` — and the `ok`/`gate_failed` verdict the Wilson bound produces — is a fresh draw on
> every run at FIXED impl. A label-identity gate is therefore only well-posed on the BOT-opponent
> configuration, which is deterministic end to end. `search_impl_parity.py` itself could not be
> run: its `tmp/search_golden_node.json` is absent from every checkout, as the ledger already
> notes.
