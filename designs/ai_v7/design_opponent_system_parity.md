# Design Note — Opponent-System Parity (training vs eval opponent selection)

**Status:** PROPOSAL / not built. Deferred until the live `ai_v7_05` specialist run finishes.
**Trigger:** launching the `ai_v7_05` exploiter-mode specialist, we found `--exploiter <T>` does **not**
auto-evaluate its target — you must *also* pass `--stable-opponents <T>` to get a
`win_rate_vs_ext_<T>` verdict. That footgun prompted the question: *is the opponent-flag surface
(`--self-play` / `--stable-opponents` / `--exploiter` / `--exploiter-keep-bots`) accreting non-orthogonal
edge cases, and should we unify it?*
**Basis:** a 6-agent audit (map training selection · map eval assembly · catalogue constraints · minimal
design · unified design · adversarial critique). This note is the durable record of that audit's verdict.
Line anchors are as-of-audit and may drift — re-grep before acting.

---

## 1. The core diagnosis — two axes entangled into one

"Opponent" is really **two orthogonal concepts** that the code only ever unified in *self-play* mode:

- **Axis A — the TRAINING MIXTURE.** A weighted set of opponent *sources* the trainee plays each
  episode: `{heuristic-bots, self-play-pool, frozen-external(s)}`, each with a selection probability.
- **Axis B — the EVAL SET.** Which frozen references the trainee is *scored* against each cycle —
  independent of who it trains on. Anchored refs feed ELO/`win_rate_vs_bots`; unanchored are yardsticks
  kept out of the aggregates.

`--stable-opponents` wires **both axes at once, but only under `--self-play`** (the injection is gated
`if self_play and stable_opponents:` — `train_rl_agent.py:~2273`). `--exploiter` was added later wiring
**only Axis A** (it sets the training opponent via `exploiter_player`, `train_rl_agent.py:~2138–2158`,
`~2292–2304`) and never registers its target for Axis B. Hence the footgun: training-registration ≠
eval-registration, and for the exploiter specifically "I train on T" trivially implies "I want my WR vs T."

### Current state, compressed

**Training selection** — one function, strict branch precedence (`wrappers.py:185–207`):

| # | Branch | Guard | Returns early? |
|---|---|---|---|
| B1 | exploiter short-circuit → `exploiter_player` | `_exploiter_player is not None` | **YES, unconditional** (unless `keep_bots` flips a coin to a floor bot first) |
| B2 | challenge bucket → self-pool / un-mastered stable | `rng < self_play_fraction` **and** `_pick_challenge_opponent()` non-None | yes if non-None |
| B3 | floor bucket → bots + mastered stable | fall-through / default | terminal |

- B1 is an absolute short-circuit: pool, stable, `self_play_fraction` are all dead code in exploiter mode.
- `--exploiter` is mutually exclusive with `--self-play` (`parser.error`, `~1690`).
- `--trainee-team` touches the **teambuilder only** (`~2037–2049`), never opponent selection — orthogonal.

**Eval assembly** — `_fixed_opponents` comes *only* from `resolve_stable_opponents(--stable-opponents)`
(`~2099–2106`); consumed by `PerOpponentEvalCallback` (`~2698`). `is_external` (the `ext_` prefix forced
in `fixed_opponent_pool.py:~182–186`) gates every aggregate — `bot_wr` excludes ext, ELO excludes ext,
best-model selection uses `bot_wr` only — so an ext entry is additive telemetry by construction. The
duplicate-label `ValueError` lives **inside** `resolve_stable_opponents` (`fixed_opponent_pool.py:~191`),
scoped to one `--stable-opponents` spec — it does **not** protect a Python-side append elsewhere.

---

## 2. Debt-or-defensible verdict

Most of what *looks* like edge-case sprawl is load-bearing separation-of-concerns:

| Edge case | Verdict | Why |
|---|---|---|
| Two loader gates (`check_compatible` vs `check_opponent_compatible`) | **Defensible & important** | full weight-shape for pool/sentinel vs obs-family-only for foreign; unifying false-rejects league play |
| Mastery flip (eval→training promotion) | **Defensible** | a real curriculum feedback loop |
| `--exploiter` XOR `--self-play` | **Defensible** | a pool beside a sole-target exploiter is dead code |
| Eval-callback *class* chosen by mode (`SelfPlayCallback` vs `PerOpponentEvalCallback`) | **Defensible** | self-play eval genuinely differs (sentinels, 2× workers, promotion) |
| Stable trains only under `--self-play` | **Defensible** | stable *rides* the pool-vs-heuristic split; no pool ⇒ no training slot |
| Mix-fraction mirror in the eval callback | **Defensible-with-guard** | duplicated model, but drift-guarded by a test; threading live wrapper state into eval is worse |
| **Exploiter target not auto-evaluated** | **Genuine debt (low severity)** | the one real footgun; ~4-line fix |
| **Stable share/mastery/temp silently no-op without `--self-play`** | **Genuine debt (trivial)** | silent no-ops; fix = a warning |

**Net: two real debts, both low-severity, both additive ≤5-line fixes. Everything else needs docs, not a rewrite.**

*(Fact check: `--exploiter-pool` and `--trainee-archetype` — referenced in older notes — are **not on main**.
They were built on a branch and never shipped. Current main: `--exploiter`, `--exploiter-keep-bots`,
`--exploiter-bot-fraction`, `--trainee-team`.)*

---

## 3. Proposal A — minimal cleanup (the recommended path)

Two additive changes, no version bump, byte-identical-off preserved:

1. **Exploiter auto-eval (~4 lines, `train_rl_agent.py` after ~2165).** Append `_exploiter_entry` to
   `_fixed_opponents` so `--exploiter <T>` auto-produces `win_rate_vs_ext_<T>` + `elo_vs_ext_<T>`.
   **MANDATORY dedup guard:** `if not any(e.label == _exploiter_entry.label for e in _fixed_opponents): _fixed_opponents.append(_exploiter_entry)`.
   Without it, `--exploiter T --stable-opponents T` (today's documented workaround, and exactly what
   `ai_v7_05` runs) puts **two** `ext_<T>` entries in `_fixed_opponents` → the eval work-steals **both**,
   both write the same `win_rate_vs_ext_<T>`/`elo_vs_ext_<T>` keys (a label-keyed dict, so **last-writer-wins**,
   not a clean double-count), **doubling the eval battles vs T** and stealing shard budget from real
   opponents. `is_external` keeps it out of `bot_wr`/ELO/best-model regardless, so training/promotion are safe —
   the only harm is wasted eval compute + a confusing TB surface. The guard removes it.
2. **Diagnostic warning (~3 lines, `train_rl_agent.py` ~2122).** When `--stable-opponent-selfplay-share` /
   `-mastered-wr` / `-temp` are user-set but `--self-play` is off, `emit()` a
   `⚠️ [STABLE] … IGNORED without --self-play`. Detect user-set-vs-default via a `sys.argv` substring scan
   (argparse can't distinguish default from an explicit equal-to-default). Pure diagnostics — zero behavior,
   RNG, version, or resume impact.

**Do NOT touch:** the wrapper hot path, the two loader gates, the mutual-exclusion guard, the eval-class
selection, `--bot-weights`/PFSP/`--trainee-team`, or **any existing flag name** (renaming breaks the
launcher's same-command resume).

---

## 4. Proposal B — full two-axis unification (deferred; only if a real need appears)

Separate Axis A and Axis B explicitly. Every mode becomes a preset over a `{source → weight}` mixture; the
five source types are `bots`, `self_pool`, `external` (one frozen model), `external_pool` (a set), and the
exploiter's `external`-pinned-to-1.0. New surface:

```
--opponents SPEC        # Axis A: comma-sep sources TYPE[:REF][@weight][:temp=T][:eval=MODE]
                        #   e.g. external:models/ai_v7_02@0.5,bots@0.5   (== today's exploiter+keep-bots)
--eval-opponents SPEC   # Axis B extras (eval-only refs); every --opponents source auto-evals
```

Old flags become aliases/presets (`--self-play` → `self-pool@…`, `--exploiter T` → `external:T@1.0`,
`--exploiter-keep-bots`/`-bot-fraction` → the two source weights, `--stable-opponents` → an `external`
source). **The payoff that shows the abstraction is right:** exploiter-auto-eval and a multi-target
`exploiter-pool` (unbuilt today) fall out with **zero new selection machinery** — `external` sources just
compose, unlike the current single-target short-circuit.

**Why deferred — the risks the critic found (all training-only, none catastrophic, but real for a live run):**
- **RNG-stream drift (highest risk).** A weighted `rng.choices(sources)` consumes a draw **every episode**
  that today's `self_play_fraction` branch does not. The proposed "skip the draw for a degenerate single
  source" fix is **incomplete** — it misses the *dominant early-training state*: self-play with `f≈0` is a
  2-source `[bots, self_pool≈0]` mixture that still draws. That offset permanently shifts the RNG stream
  feeding every downstream bot/PFSP pick → **a different training distribution on a resumed segment**, caught
  by no existing test.
- **Telemetry mirror.** `SelfPlayCallback._opponent_mix_fractions` (`~2684–2686`) is a hand-derived analytic
  mirror of the selection weights, drift-guarded by `test_mix_fractions_match_actual_sampling`. Any selection
  rewrite forces rewriting it; get it wrong and `train/selfplay_fraction` silently lies.
- **Legacy-semantics shim.** `--stable-opponents X` without `--self-play` is eval-only today; the more-orthogonal
  model could let an `external` source train without a pool. One shim bug = a quietly redefined training
  distribution.

**Migration cost: MEDIUM.** Entirely training-only — no forward pass, obs vector, weight shape, version gate,
or resume-immutable hparam — so the worst failure is an opponent-mix regression (caught by WR/ELO canaries),
**never** a wrong-arch load or checkpoint break. Sequencing if ever done: (1) `opponent_config.py` +
desugaring shim + test, selection **untouched** (byte-identical); (2) rewire `wrappers.py` behind a fast path
**with the RNG-sequence test written first**; (3) rewrite the mix mirror + drift guard; (4) add `--opponents`,
drop the mutual-exclusion; (5) `exploiter-pool` + auto-eval then need **zero** new code.

---

## 5. The `--sync-to-main` constraint (why not now)

`ai_v7_05` runs with **`--sync-to-main`**, so its 6h launcher restarts pull `origin/main` **HEAD** — meaning
*any* opponent-system commit lands on the live experiment at its next restart, not just flag changes. **Do not
ship opponent-system changes to main while a `--sync-to-main` run is live.** Land them after it finishes, or in
the branch that launches the next run.

Note `ai_v7_05` already gets its verdict metric via the `--stable-opponents models/ai_v7_02_critic_shape_0627`
we added — so even Proposal A Change 1 is a **no-op for the live run** (the dedup guard skips it, since the
stable entry already registered the label). Change 1 helps only *future* exploiter runs that forget the flag.

---

## 6. Recommendation

1. **Now:** ship nothing. Capture this note (done) + memory.
2. **Next exploiter/fork launch (between runs):** land Proposal A — exploiter auto-eval (dedup-guarded) +
   the inert-stable-knob warning. Additive, byte-identical-off, no version bump.
3. **Only if a real multi-target league need appears:** do Proposal B as a dedicated between-runs pass, with
   the two missing tests (single-source **and** near-zero-weight RNG-sequence assertion; rewritten mix-fraction
   drift guard) written **first**. Never overlapping a live run.

**One line:** the real pain (exploiter needs `--stable-opponents T` to be evaluated; three stable knobs
silently no-op) is genuine but low-severity and fully fixable with one `print` + four guarded lines — none of
which justifies perturbing the RNG stream / selection hot path of a live from-scratch run on one GPU, which is
exactly what the only "clean" option (B) would risk.
