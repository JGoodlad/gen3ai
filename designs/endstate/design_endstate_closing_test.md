# The CLOSING TEST: the END STATE against PRODUCTION (`--arch endstate` vs `--arch production`)

🚨 **ALWAYS-CURRENT (owner, 2026-09-27; `README.md` in this directory).** A decision or build that differs from this
doc updates it in the same commit, and the Decision record at the end says what changed and why.

**Status (2026-10-09): REGISTERED, before any seed.** §3 is the pre-registration; the ledger entry
`2026-10-09 · REGISTRATION · THE END-STATE CLOSING TEST` is its record. The PRODUCTION arm may launch at once (its pin is
fixed below). The END-STATE arm is BLOCKED until §2.3's preconditions hold (the compiled-learner startup gate fails on
the end-state graph today, §8 F-ES-1). Nothing has been read.
**Amended before any end-state data (2026-10-10, Decision record):** P_prod re-pinned to `95d014fa`; the end-state
overlay gains `--effective-stats on --move-target-state on` (v153) and `--move-set-closure on` (v154, the owner's "Do
A"); **P_end = `b132b099`**; precondition E at P_end is the first end-state seed's own startup gates.

## 0. Summary

| | |
|---|---|
| question | is the end state (`--arch endstate`, `design_static_tokens.md` §14) NOT DETECTABLY WORSE than production? |
| arms | PRODUCTION = `--arch production` pinned at **P_prod = `95d014fa`** (re-pinned pre-data from `c0f528b4`); END STATE = `--arch endstate` pinned at a later **P_end = `b132b099`** at which production is byte-identical to P_prod (§2.3; Decision record) |
| budget | 15M steps per seed, the production recipe, **8 seeds per arm** (ids 2001–2008), fixed n, ONE look |
| meter | the mirrored head-to-head CROSS at P_end: 8 × 8 cells × 1,000 mirrored pairs (`main.h2h play-many`) |
| statistic | X5 §7.4's: Δ̂ = mean(h) − 50, V̂ = (s²_R + s²_C)/8 on 14 df; the two-sided 90 % interval Δ̂ ± 1.761 √V̂ |
| **rule (owner)** | **PASS iff the interval includes 0 or lies above it AND its lower end > −2.0 pp; FAIL otherwise** |
| on PASS | adopt the end state; delete legacy and the judgments (§6) |
| on FAIL | BISECT along the registered ladder (§5) |
| speed | reported, never decides (the owner's rule has no speed clause) |
| cost | ≈ 47 GPU-h (§7) |

## 1. The question, and why this rule

The plateau-first roadmap (owner, 2026-10-09; memory `project_plateau_first_funnel`, item 4) runs ONE big test before
the deep run: the full end state against production. The static-token screen's look 3 (FINAL, 2026-10-09) read NOT
DETECTED for `static` vs `legacy` (Δ̂ −2.32 pp, 90 % [−3.60, −1.04]); the owner kept legacy "until a CLOSING test"
(`design_static_tokens.md` Decision record) and bundled every recovery and fact-completion lever into one speculative
arm, to be bisected only if it fails.

**The owner's rule** (2026-10-09): the end state must be NOT DETECTABLY WORSE than production. Unlike the screen's
non-inferiority margin (δ = 3.5 pp, under which "worse but inside δ" adopted), this rule has two clauses:
1. **No detectable deficit:** the two-sided 90 % interval includes 0 or lies above it. A deficit the data can see fails.
2. **Enough precision:** the interval's lower end is above −2.0 pp. A read too noisy to exclude a 2-pp deficit fails.

PASS needs both. Read on the static screen's look-3 matrix, this rule FAILS on both clauses (`closing_rule.py
self-check`, §3.3): the bar is deliberately stricter than the screen's.

## 2. The arms, the pins and the preconditions

### 2.1 PRODUCTION

`--arch production` at **P_prod = `c0f528b4`** (origin/main HEAD at registration: the fact-completion build, config
v152; it is after the v151 Toxic / Wish fix `58f8149a`). Measured at P_prod (CPU, this registration):

| identity fact at P_prod | value |
|---|---|
| production extractor dynamo graph (`static_recovery_2026-10-09/graph_sha.py '{}'`) | **graph_sha `421c6b98ce7937f4`**, state_sha `749c56159ab028f4`, out_sha `51c02c6c6342a045`, 1 graph, 20,148 lines, 1,937,942 params |
| K9 learner golden `src/agents/training/learner_golden.json` | git blob **`9ef44772a3f76fcfa4de7e59590aec7be6b05812`** (sha256 `ce21cd2d97cc4870…`) |
| K9 golden buffer `src/agents/training/learner_golden_buffer.npz` | git blob **`9f13350daec82f90f37cf557b776646c316e6415`** (sha256 `66a1439218721fca…`) |
| production mirror `designs/production_config.json` | git blob `1f454b35e69cf925de0f01eda3c7399f86d10ffa` (checkargs tag `production_config@1f454b35e69c`) |
| obs golden `src/agents/training/golden_obs_fixture.json` | git blob `09733f8cef9a42be6b7eb03e8f327ab20153e953` |

The graph / state / outputs hashes equal the fact-completion build's own record (`measurements/endstate_facts_2026-10-09/
shas.out`, production "build"): the measurement is reproducible.

### 2.2 END STATE

`--arch endstate` (`src/main/train/arch_arms.py`; at P_end `b132b099` the checkargs tag is
`endstate@production_config@49cd523f38ee+overlay@2b359668`): `static_recovery` (static tokens + `--mon-hazard-cost`,
`--move-actor-state`, `--trunk-layers 3`, `--switch-hazard-cost`, `--eot-residual`) + `--move-resolution on
--speed-physics on --value-threat-inject off --op-reduction principled --obs-facts v1` + `--move-resolution-facts full
--status-facts exact --ko-ramp exact --drop-progress-clock on --g-ledger eot` + `--effective-stats on
--move-target-state on` (v153) + `--move-set-closure on` (v154). The arm is the CLOSED declaration in `arch_arms.py` at
P_end. It was widened twice BEFORE any end-state seed (the Decision record's 2026-10-10 rows, the second the owner's
pre-data amendment); once an end-state seed has run it is never widened for this test (a changed overlay is then a new
registration).

### 2.3 Preconditions on the END-STATE arm (before its first seed; each recorded in the ledger)

**P_end** is the first commit at which (A)–(E) all hold. Run from a checkout of P_end (a launcher pin worktree or a
detached checkout), `PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3`:

```bash
# (A) production's extractor is byte-identical: the SAME three hashes as P_prod (§2.1)
cd <P_end checkout>/src && $PY -c "import runpy,sys; sys.argv=['g','{}']; \
  runpy.run_path('../designs/research_state/measurements/static_recovery_2026-10-09/graph_sha.py', run_name='__main__')"
#   must print n_graphs 1, graph_sha 421c6b98ce7937f4, state_sha 749c56159ab028f4, out_sha 51c02c6c6342a045, n_params 1937942

# (B) the K9 golden is byte-identical AND production still reproduces it at P_end
git -C <P_end checkout> rev-parse <P_end>:src/agents/training/learner_golden.json        # 9ef44772a3f76fcfa4de7e59590aec7be6b05812
git -C <P_end checkout> rev-parse <P_end>:src/agents/training/learner_golden_buffer.npz  # 9f13350daec82f90f37cf557b776646c316e6415
cd <P_end checkout>/src && $PY -m pytest agents/training/learner_golden_test.py -q          # green

# (C) the production mirror and the obs golden are byte-identical, and the Rust core still reproduces the obs golden
git -C <P_end checkout> rev-parse <P_end>:designs/production_config.json                  # 1f454b35e69cf925de0f01eda3c7399f86d10ffa
git -C <P_end checkout> rev-parse <P_end>:src/agents/training/golden_obs_fixture.json     # 09733f8cef9a42be6b7eb03e8f327ab20153e953
cd <P_end checkout>/src && $PY -m agents.training.golden_obs_core --check                  # green

# (D) what changed between the pins, classified (CUDA-only and Rust changes are invisible to (A)-(C))
git -C <P_end checkout> diff --name-only 95d014fa <P_end> -- src/ data/ ':!*_test.py'
```

(The commands' expected values are P_prod's; after the 2026-10-10 re-pin the diff in (D) starts at `95d014fa`, and the
mirror blob in (C) is `5ad40492fe21` at P_prod.)

- **(C) as AMENDED (2026-10-10, pre-data):** the production mirror is identical to P_prod's **modulo NEW keys whose
  production value is OFF**, each key listed in the ledger entry, with (A) + (B) the proof that production builds and
  computes nothing new; the obs golden stays byte-identical and `golden_obs_core --check` green.

- **(D)'s rule:** every listed file is classified in the ledger entry as (a) gated on a flag production keeps OFF,
  (b) tests / docs / tooling no run imports, or (c) on production's training path. **ANY `data/` file, any (c) file,
  or any change under `src/rust_sim/` / `src/rust_env/` voids the identity**, because the CPU proofs (A)–(C) cannot
  see a CUDA- or Inductor-only change (the very kind the F-ES-1 fix may be) or a battle-mechanics change. A voided
  identity means: re-shape the change so production's path is untouched, or re-run the production arm at P_end; it is
  never waived in silence.
- **(E) the end-state build is GPU-checked at P_end:** the fact-completion GPU checks (`design_static_tokens.md` §14.1:
  CUDA compile parity forward + backward, T2's CUDA-graph build, K9(b) on a real first update, a real two-minute
  launch, the cost read) and the compiled-learner startup gate (R1) PASS on `--arch endstate`, which needs the F-ES-1
  fix. Recorded with their measurement dir.
- **The argv re-validates at P_end:** `python -m main.checkargs --argv "<§4.2's argv without the launcher flags>"`
  shows `✓ every ARCH-surface key matches the production mirror + the arm 'endstate'`, `✓ every RECIPE knob matches
  recipe.fresh` and `✓ this command still launches`; `python -m main.launcher --dry-run <§4.2's argv>` reads
  `✓ DRY RUN — this command would launch`.

### 2.4 Preconditions on BOTH arms

- One commit per arm: every production seed `pin_history` = [`95d014fa`] only, every end-state seed [`b132b099`]
  only.
- `metadata.json` `init_num_threads` equal across all sixteen; torch 2.8.0+cu126 in all.
- **`data/` frozen** from the first production launch to the last end-state seed's end (a pin isolates code, not
  data: pinned children read `data/` from MAIN).
- The production arm's first seed passes its own startup gates (R1, the K6 freeze, K9(b) on update 1) at P_prod; that
  launch is the GPU check of production at P_prod (it has run no 15M seed since v148 / v149 changed the `x` cell).

## 3. The registration

### 3.1 Meter: the mirrored head-to-head CROSS

`python -m main.h2h play-many` on the Rust eval core, **played at P_end, never at HEAD** (both architectures load there;
production's weights from P_prod run on P_end's byte-identical production graph). Every end-state seed's 15M final
(`final_model.zip`) plays every production seed's: **8 × 8 = 64 cells, 1,000 mirrored pairs per cell** (≈ 2,000 games;
meter SE ≈ 1.1 pp per cell), both sides greedy, schedule seed 0, `--oracle-reveal-mode off`, purpose `ab`, family
`es_closing_strength_steps`, request `es_closing_steps`, label `es_closing`. h_ij = end-state seed i's score against
production seed j, in pp (a draw ½); the player keeps seat p1 and the mirror swaps the teams, as `main.h2h` does. The
seat effect is not subtracted (X5 §7.4).

### 3.2 Statistic (X5 §7.4's, unchanged: `main.h2h.cross.cross_stat`)

Δ̂ = mean_ij(h_ij) − 50; row means R_i (end-state seeds), column means C_j (production seeds); **V̂ = (s²_R + s²_C)/8 on
df = 14**; the two-sided **90 % interval Δ̂ ± t_{0.95,14} √V̂, t_{0.95,14} = 1.7613**. Seeds do not pair runs.

### 3.3 Decision rule (fixed n = 8, one look)

| outcome | rule | what follows |
|---|---|---|
| **INCONCLUSIVE** | an input is incomplete or invalid: a run short of 15M or out of restarts (after the §3.7 amendments), a broken §2 precondition, a cell with < 1,000 completed pairs, or timeouts > 25 % of attempted battles | repair and re-read (a failed run is replaced by the next seed id, 2009, 2010, …; never dropped); never interpreted |
| **PASS** | upper end Δ̂ + 1.7613 √V̂ **> 0** (the interval includes 0 or lies above it) **AND** lower end Δ̂ − 1.7613 √V̂ **> −2.0 pp** | §6: adopt the end state |
| **FAIL** | anything else | §5: bisect |

**Rule 8:** an interval end within 1e-9 pp of its threshold (0 for the first clause, −2.0 for the second) does NOT
satisfy that clause, and the read records it as `near_boundary`. A tie never passes.

The rule is CODED before any seed: `research_state/measurements/endstate_closing_test_2026-10-09/closing_rule.py`
(`decide`, on `cross_stat`). Its `self-check` reads the static screen's look-3 matrix and must return FAIL on both
clauses with that README's interval [−3.60, −1.04] to 1e-9 (it does).

### 3.4 No early look (the owner's optional 4-seed look: NOT taken)

The owner was open to an optional look at 4 seeds per arm. Designed and priced, then **not registered**:
- **It could only FAIL early, never PASS.** At n = 4 the interval's half-width alone is ≈ 2.0 pp (t_{0.95,6} = 1.943 ×
  √V̂ ≈ 1.03 at the screen's variances), so the precision clause cannot be met. The only sound early stop is for HARM:
  a two-look one-sided O'Brien–Fleming test of "Δ < 0" at information fraction ½ (C = 1.678, z₁ = 2.373, nominal
  p 0.0088), mapped to t on 6 df: **stop iff Δ̂ / √V̂ ≤ −3.242** on the 4 × 4 cross. As a non-binding futility stop it
  spends none of the PASS error.
- **What it buys is small** (`closing_rule.py simulate`, 20,000 reps, at the screen's variances): it stops 0.9 % of the
  time at Δ = 0 (PASS 0.835 → 0.832), 18–29 % at Δ = −2 to −2.5, 57 % at Δ = −3.5. Each stop saves end-state seeds
  2005–2008 (≈ 12 GPU-h); each look costs a GPU pause for the 16-cell cross (≈ 0.5 GPU-h with the engine's 13-min
  start; the cells are reused at the final). At the screen's own deficit (−2.3) the expected saving is ≈ 3 GPU-h on
  ≈ 47.
- **And it weakens the FAIL branch:** §5's bisection reuses the end-state arm's 8 seeds as its top rung; an arm stopped
  at 4 gives that rung, and every later comparison against it, half the seeds.

Fixed n = 8, one look, is the registration.

### 3.5 Operating characteristics (SIMULATED, `simulate.json`, additive model h_ij = 50 + Δ + a_i − b_j + e_ij)

σ_e = 1.1 pp per cell; σ_E / σ_P the end-state / production run-strength SDs.

| true Δ (pp) | P(PASS), σ 1.65 / 1.10 (the screen's look 3) | σ 1.65 / 1.65 | σ 2.5 / 2.5 |
|---|---|---|---|
| +1.0 | 0.989 | 0.959 | 0.728 |
| 0.0 | **0.835** | 0.726 | 0.437 |
| −0.5 | 0.608 | 0.515 | 0.299 |
| −1.0 | 0.346 | 0.294 | 0.184 |
| −1.5 | 0.146 | 0.136 | 0.098 |
| −2.0 | **0.044** | 0.046 | 0.051 |
| −2.5 | 0.008 | 0.012 | 0.022 |

The false-PASS rate at a true 2-pp deficit is ≈ 0.05 by construction (the lower end's one-sided 95 %). **A truly equal
end state FAILS 17 % of the time at the screen's variances, and 56 % if both arms' run-to-run SD is 2.5 pp** (F-ES-3):
the precision clause is a real bar at n = 8.

### 3.6 Speed: REPORTED, never decides

s = (pooled median quiet update-cycle wall of the end state) / (that of production) − 1, by the static screen's
`speed_read.py` method (quiet cycles only: each child's first 11 updates, non-advancing records, compile canaries,
in-loop evals and cycles at contention factor ≥ 1.05 excluded), plus the end-to-end wall per run. No speed clause: the
owner's rule is strength only. **Context:** the 2026-10-09 GPU checks measured `--arch static_recovery` (R) against
production (P) at **+14.8 % `train_ms` and −11.4 % throughput** (**UNVERIFIED here:** the brief's figures; the GPU-check
measurement dir had not landed at registration); the end state adds the fact-completion levers (+12.4 % eager CPU
forward over E, `endstate_facts_2026-10-09/`), so s is expected near or above R's. Perf work is interleaved on the end
state only (roadmap item 5).

### 3.7 The K9(b) behaviour check: strict, with the static screen's amendments 1–2 carried over

Both arms run the strict K9(b) check (`--behaviour-check` at its production default, FATAL), at their pins (both
carry the flip-judge, `718c0adf`). Carried over verbatim from `design_static_tokens.md`'s Decision record (2026-10-08):
1. **Amendment 1:** a K9(b) stop whose ONLY cause is the tie-share ceiling (excluded share > 0.15), with max |d log π|
   < 1e-4, is a known false alarm. The seed resumes from its own latest checkpoint (the launcher's crash restart; once
   its 3-crash budget is spent, a `--model` resume of its own latest checkpoint, same pin, same argv), WITHOUT limit,
   and is NOT INCONCLUSIVE; each stop is banked. Any stop with max |d log π| ≥ 1e-4, or any other failure, counts as
   before. The in-arm speed s excludes each resume window.
2. **Amendment 2:** once a seed's launcher crash budget is spent on tie-only stops, its `--model` resume of its own
   latest checkpoint (same pin, same argv) runs with `--behaviour-check warn`. The run is NOT INCONCLUSIVE provided
   EVERY probe after the switch keeps max |d log π| < 1e-4 (verified from the log by the training agent); any probe
   ≥ 1e-4 makes the seed INCONCLUSIVE (replaced by the next seed id). Banked as a deviation.

### 3.8 Secondary (REPORTED, never gated)

The bots panel (each run's last in-loop eval, the 8 training bots, random excluded) and sentinel MONOTONICITY, by the
screen's `panel_bots.py`; a HARM note when the end state's bots mean is below production's by > 4 pp (2 × the 2-pp
floor), told to the owner before adoption; `train_ms`, the T2 flush; every seed's deviations by the screen's
`deviations.py` method.

## 4. The argv and the chain

### 4.1 PRODUCTION (validated at registration)

```bash
cd /home/goodlad/dev/gen3ai && PYTHONPATH=src python -m main.launcher --restart-interval-hours 6 --pin-commit 95d014fa \
  --arch production --steps 15000000 --seed 2001 \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all \
  --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda --run-name rb_es_prod_s2001
```

Seeds 2002–2008: change `--seed` and `--run-name` together (`rb_es_prod_s200N`). The ride-along heads and
`--snapshot-ladder-games 0` / `--checkpoint-every-steps 1000000` are the static screen's (X5 Amendment 1's reason; the
1M checkpoints serve a matched-wall descriptor); the ride-along keys are critic READOUTS outside the ARCH surface, so
no `--allow-nonproduction-arch` is needed. (`--belief-tokens fixed_mass`, typed by the screen, is DELETED: X5 is the
only belief representation.) Validated on 2026-10-09 at `c0f528b4`: `python -m main.checkargs --argv "…"` → 13 flags,
0 unrecognized, `✓ every ARCH-surface key matches the production mirror`, `✓ every RECIPE knob matches recipe.fresh
(28 knobs)`, `✓ this command still launches`; `python -m main.launcher --dry-run …` → FRESH, pin `c0f528b4cfab…`,
torch 2.8.0+cu126, disk ✓ (8.95 GiB required), desktop GPU ✓, `✓ DRY RUN — this command would launch`.

### 4.2 END STATE (at P_end, after §2.3)

The same argv with `--pin-commit b132b099 --arch endstate` and `--run-name rb_es_end_s200N`:

```bash
cd /home/goodlad/dev/gen3ai && PYTHONPATH=src python -m main.launcher --restart-interval-hours 6 --pin-commit b132b099 \
  --arch endstate --steps 15000000 --seed 2001 \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all \
  --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda --run-name rb_es_end_s2001
```

Re-validated at P_end (§2.3; `research_state/measurements/endstate_closing_test_2026-10-09/p_end_2026-10-10/`):
`checkargs` and `--dry-run` as recorded there.

### 4.3 Chain order

1. **Production first, tonight:** `rb_es_prod_s2001` → `s2002` → … → `s2008`, one at a time (the GPU lease is the
   training agent's).
2. **Then the end state:** `rb_es_end_s2001` → … → `s2008`, once §2.3 is recorded as met.
3. If §2.3 is met while production seeds remain, the chain MAY alternate (E, P, E, P, …) for the remaining slots,
   the switch recorded in the ledger. Order does not enter the read (seeds do not pair; one read after all sixteen).
4. A seed made INCONCLUSIVE is replaced by the next id of its arm (2009, …) at the end of that arm's chain.
5. The cross (§3.1) plays once all sixteen 15M finals are banked, under its own GPU lease, at P_end.

## 5. On FAIL: the BISECTION (registered order)

The levers are grouped, in the owner's order, into a CUMULATIVE ladder from production. Each rung adds the next group:

| rung | adds | the levers (`arch_arms.py` keys) |
|---|---|---|
| B1 | **encoding + recovery facts** | `token_encoding static`, `mon_hazard_cost`, `move_actor_state`, `switch_hazard_cost`, `eot_residual` |
| B2 | **trunk depth** | `trunk_layers 3` (B2 = `--arch static_recovery` exactly) |
| B3 | **move resolution + status facts** | `move_resolution on`, `move_resolution_facts full`, `status_facts exact` |
| B4 | **speed physics** | `speed_physics on` |
| B5 | **critic route** | `value_threat_inject off` |
| B6 | **op reduction** | `op_reduction principled`, `ko_ramp exact`, `g_ledger eot` |
| B7 | **probe-battery facts** | `effective_stats on`, `move_target_state on` (the v153 levers; owner 2026-10-10: their own rung) |
| B8 | **obs facts** | `obs_facts v1`, `drop_progress_clock on`, `move_set_closure on` (B8 = `--arch endstate`, the FAILED arm) |

(`ko_ramp` and `g_ledger` are the damage operator's math, so they ride with the op group; `drop_progress_clock` removes
a read of the observation, so it rides with the obs facts; `move_set_closure` (the 2026-10-10 amendment) sits in the
top rung as the last lever added. Those three placements are this registration's call: the
owner may move them before the first bisection seed.)

**Search:** each tested rung is a new 8-seed arm (ids 2001–2008, run names `rb_es_b<k>_s200N`) at its own pin under
§2.3's identity preconditions, crossed against the SAME eight production seeds under §3's rule. A rung that PASSES puts
the culprit above it; one that FAILS puts it at or below it. Order:
1. **B2 first** (`static_recovery`): the static encoding alone was −2.32 pp at the screen's look 3, so groups 1–2 carry
   the highest prior, and the arm is already declared.
2. B2 FAILS → test **B1**: B1 FAILS ⇒ the culprit is group 1 (encoding + recovery facts); B1 PASSES ⇒ group 2 (trunk
   depth).
3. B2 PASSES → test **B5**: B5 FAILS → test **B3** (FAILS ⇒ group 3) then **B4** (FAILS ⇒ group 4, PASSES ⇒ group 5);
   B5 PASSES → test **B6** (FAILS ⇒ group 6); B6 PASSES → test **B7** (FAILS ⇒ group 7, PASSES ⇒ group 8 by elimination against the failed B8).

At most five arms (≈ 25 GPU-h each). **Result:** the first failing group, named with the interaction caveat (a cumulative
ladder attributes a deficit to the group that reveals it given everything below it). The highest PASSING rung is
adoptable under the same rule; the groups above the culprit return in a later closing test without it. Rungs that are
not yet declared arms (B1, B3–B7) are declared in `arch_arms.py` before their first seed (a bisection of `endstate`,
never a typed flag list). Each bisection read is its own ledger REGISTRATION before its seeds.

## 6. On PASS

1. **Adopt:** `designs/production_config.json` takes the end-state surface (one unit; `ARCHITECTURE.md` and the
   CHANGELOG in the same pass; the K9 golden re-baked for the new production).
2. **Delete legacy and the judgments** (owner): the legacy token encoding, and every JUDGMENT the end state replaces
   (`design_hand_computed_features.md`'s judgment rows: `neutralization` / `tempo_cost`, `turns_since_progress`, the
   critic's threat injection, the non-principled op reductions, the coarse KO ramp, the second end-of-turn rule), each
   flag with its OFF path; a deletion manifest written at adoption, one unit per lane.
3. The deep plateau run (X26) starts from the adopted end state (roadmap items 5–7).

## 7. Cost

Production ≈ 2.6 h per seed (the screen's legacy end-to-end 2.557 h) ⇒ ≈ 21 GPU-h; end state ≈ 3.0–3.2 h per seed
(+14.8 % `train_ms` for R, plus the facts) ⇒ ≈ 25 GPU-h; the cross 128,000 games ≈ 1.2 GPU-h (look 3 played 78,000 in
41 min, engine start 13 min). **≈ 47 GPU-h**, of which production's 21 h can run tonight.

## 8. Findings

- **F-ES-1 (BLOCKER for the end-state arm).** At P_prod the end state's predecessor E (`static_recovery` + the bundle)
  FAILS the compiled-learner startup gate: an Inductor SDPA memory-efficient backward numerics defect, being pinned now.
  The end-state arm waits for that fix + the GPU checks (§2.3 (E)).
- **F-ES-2 (HAZARD for the fix).** The F-ES-1 fix is likely CUDA / Inductor-level, which §2.3 (A)–(C)'s CPU proofs
  cannot see. If it touches production's path (both arms share the SDPA sites, `dense_attn_bias`), §2.3 (D) voids the
  identity and the eight production seeds run tonight do not count. **The fix must be gated on the end-state graph, or
  shown not to change production's compiled path**, or production re-runs at P_end (≈ 21 GPU-h).
- **F-ES-3 (POWER).** A truly equal end state FAILS 17 % of the time at the screen's variances and 56 % at a 2.5-pp
  run SD (§3.5). A FAIL is therefore not proof of a deficit; §5's first rung reads whether the encoding group alone
  carries it.
- **F-ES-4.** The cross plays at P_end: production checkpoints from P_prod must load there through the strict loader
  (the config version may move between pins). The cross's plan refuses on a load failure, which makes the read
  INCONCLUSIVE, never silently re-pinned.
- **F-ES-5.** `data/` is frozen for the test's whole window (≈ 2 days of chain + the cross), because pinned children read
  `data/` from MAIN.

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-09 | **The closing test REGISTERED** (before any seed) | END STATE (`--arch endstate`, P_end) vs PRODUCTION (`--arch production`, P_prod = `c0f528b4`); 15M, the production recipe, the screen's ride-along heads; 8 seeds per arm (2001–2008), fixed n, one look; the 8 × 8 mirrored cross at P_end; X5's Δ̂ / V̂ on 14 df; **PASS iff the 90 % interval's upper end > 0 and lower end > −2.0 pp** (owner's rule), rule 8 against PASS; K9(b) strict with the screen's amendments 1–2; speed reported only | the screen's δ-margin non-inferiority (owner: not detectably worse, with a precision floor); a group-sequential design | owner 2026-10-09; §1–§3 |
| 2026-10-09 | The optional 4-seed look | NOT taken | a two-look OBF harm stop (t₆ ≤ −3.242): it cannot PASS early, saves ≈ 3 GPU-h in expectation at Δ = −2.3, and halves the bisection's top rung when it fires | §3.4; `simulate.json` |
| 2026-10-09 | Identity of production across the pins | the CPU hashes (graph, state, outputs; K9 golden + `learner_golden_test`; the mirror; the obs golden + `golden_obs_core --check`) AND a classified `src/` + `data/` diff in which no production-path, Rust or data change is allowed | the CPU hashes alone (blind to a CUDA / Inductor-only or mechanics change) | §2.3; F-ES-2 |
| 2026-10-09 | The bisection | a cumulative ladder from production in the owner's group order, B2 (`static_recovery`) tested first, binary search after it; `ko_ramp` / `g_ledger` with the op group, `drop_progress_clock` with the obs facts | leave-one-group-out from the end state (7 arms; and legacy compositions of static-only levers); a pure midpoint start (ignores the screen's −2.32 on the encoding) | §5 |
| 2026-10-09 | Chain order | production 2001–2008 first (runnable tonight), then the end state; alternation allowed once the end state is unblocked | waiting for both arms to interleave from the start (idles the GPU) | §4.3 |
| 2026-10-10 | **P_prod RE-PINNED to `95d014fa` (pre-data; no seed of either arm had launched)** | The end-state overlay gains `--effective-stats on` + `--move-target-state on` (`79da8cd3`, config v153; the probe battery's rows 12 / 13), per the owner's 'run all levers speculatively, bisect after'. P_prod moves from `c0f528b4` to `95d014fa` so that P_prod's production mirror already carries the two new keys (both OFF in production). Re-measured identity at `95d014fa` (CPU): production graph `421c6b98ce7937f4`, state `749c56159ab028f4`, outputs `51c02c6c6342a045`, 1,937,942 params (UNCHANGED from `c0f528b4`); K9 golden blobs `9ef44772…` / `9f13350d…` (UNCHANGED); `production_config.json` blob `5ad40492fe21` (was `1f454b35e69c`: two new OFF keys); obs golden blob unchanged. Precondition C now reads against `5ad40492fe21`. The END-STATE arm's identity at `95d014fa`: graph `55c7f5c009d4595b`, 2,031,460 params. The production argv is unchanged except `--pin-commit 95d014fa`. The end-state arm still awaits the SDPA compile fix (gated to the end-state arm only, never production's compiled path) + its GPU checks | launching production at `c0f528b4` and amending C after the fact (a cleaner pre-data re-pin was available) | orchestrator, under the owner's 10-09 delegation |
| 2026-10-10 | **Precondition E MET for the END-STATE arm at `95d014fa` (GPU check, before any end-state seed)** | One real `--arch endstate` launch at `95d014fa` (`rb_es_gpucheck_end`, seed 2001, NOT a closing-test seed; run dir kept as evidence): startup R1 / compile-region parity PASS (loss rel 0, features max \|Δ\| 0, grad cosine 1.000000); K6 freeze armed; UpdateFit headroom 2,058 MiB (≥ 1,024); K9(b) strict 10 probes, excluded_frac max 0.031, max \|d log π\| 7.15e-7; the update-10 canary compiled == eager. `train_ms` ~53.8 s / update, fps ~1,685–1,868 (descriptive). The earlier E defect at `43a59bbd` (an Inductor graph-shape interaction needing obs_facts + move_resolution + principled + vti-off together) does NOT form at `95d014fa` (`53412b77`). The production chain `rb_es_prod_s2001..8` started 00:44:34 at `95d014fa`; `data/` is frozen. If P_end = `95d014fa`, preconditions A–D hold trivially (same commit); the end-state arm may then run at P_prod's own commit | none (the registered precondition, now met) | orchestrator |
| 2026-10-10 | **END-STATE overlay + `--move-set-closure on`; P_end → `b132b099` (pre-end-state-data; owner 2026-10-10)** | Owner, 2026-10-10: "Do A". `--arch endstate` gains `--move-set-closure on` (the four-move closure, v154, `a9d6fd79`; `research_state/measurements/belief_closure_2026-10-10/`) as a PRE-DATA AMENDMENT: no end-state seed had run (only the throwaway GPU check `rb_es_gpucheck_end`). Production is untouched: P_prod stays `95d014fa` (`rb_es_prod_s2001` banked, `s2002` running). **P_end = `b132b09952bafb846198466af81fffcb979c12b1`** (the commit landing the overlay, on top of the infra commits since `95d014fa`). Re-run at P_end on CPU (`research_state/measurements/endstate_closing_test_2026-10-09/p_end_2026-10-10/`): **(A)** production graph `421c6b98ce7937f4`, state `749c56159ab028f4`, outputs `51c02c6c6342a045`, 1 graph, 20,148 lines, 1,937,942 params: EQUAL to P_prod; **(B)** `learner_golden.json` `9ef44772a3f76fcfa4de7e59590aec7be6b05812`, `learner_golden_buffer.npz` `9f13350daec82f90f37cf557b776646c316e6415`: equal, `learner_golden_test` 6 passed; **(C), AMENDED to "identical modulo new keys that are OFF in production"**: `production_config.json` blob `49cd523f38ee` vs P_prod's `5ad40492fe21`, the ONE new key `move_set_closure` = `"off"`, proven inert by (A) + (B); obs golden blob `09733f8cef9a42be6b7eb03e8f327ab20153e953` equal, `golden_obs_core --check` 991 / 991; **(D)** the `95d014fa..b132b099` `src/` + `data/` diff classified file by file (the measurement README): the closure build (gated OFF, or its record / parser surface with production's value OFF), the `endstate` overlay, the eval-snapshot hard link (`52c2cdd9`: on-disk STORAGE only, sha-verified, after the cycle's games), the launcher parent process (restart default, submodule preflight), live-play / anchor / prober tooling and docs; `data/` EMPTY; under `src/rust_sim/` two Markdown files only, no Rust code; no file on production's training path. Identity HOLDS. **The new end-state identity:** graph `42ee361b11fa5a8d`, state `df71da47741a8a0f`, outputs `5adfaa6c40a7a71e`, 1 graph, 35,662 lines, 2,031,460 params (the closure adds no parameter; the `95d014fa` overlay in the same tree reproduces `55c7f5c009d4595b` / `c78ca89766000c71`); the K9(b) recorder sees no undeclared op; the CPU `--debug` smoke PASSED (1,010 s). **(E) at P_end = the FIRST end-state seed's own startup gates** (R1 compile parity forward + backward, the K6 freeze, K9(b) at update 1, the update-10 canary): the `95d014fa` GPU check does not cover the closure's new graph path. **If its startup gate refuses, the chain HOLDS** (the training session's follower holds on a non-DONE ending) **and no data counts.** In §5's ladder the closure sits in B7. The argv: §4.2 (`--pin-commit b132b099 --arch endstate`, the rest as staged); `checkargs` and `--dry-run` ✓ at P_end | keeping the closure out of the closing test (its own screen after the read; the end state then read with a measured leak open); re-running production at P_end (not needed: (A)–(D) hold) | owner 2026-10-10; ledger AMENDMENT 2026-10-10 |
| 2026-10-10 | **OWNER RULINGS: (1) precondition (D)'s standing interpretation; (2) the v153 levers get their own bisect rung** | Owner, 2026-10-10, "1 yes and 2 put it in". (1) Files that only RECORD or PARSE a new OFF-by-default flag (`model_version/*`, `snapshot.py`, the parser, `flag_registry.py`) do NOT void identity under (D) when (A) graph / state / outputs sha and (B) the K9 golden prove production unchanged; this is the STANDING reading of (D) for this test and future closing tests. Markdown under `src/rust_sim/` is docs, not compiled code, and does not void. (2) `effective_stats` + `move_target_state` become their own rung **B7 (probe-battery facts)**; obs facts move to **B8** (= `--arch endstate`). The search's B6-PASSES branch now tests B7 before attributing to obs facts; at most five arms | leaving the v153 levers in the top rung by default | owner, 2026-10-10 |
| 2026-10-10 | **OWNER: COARSER bisect ladder — 'do more per rung and we can bisect after' (supersedes the B1–B8 ladder above and the B7 split of `dd841d64`)** | On FAIL, the first-level search uses THREE cumulative rungs: **C1 = `--arch static_recovery`** (encoding + recovery facts + trunk depth; already declared); **C2 = C1 + `move_resolution on` + `move_resolution_facts full` + `status_facts exact` + `speed_physics on` + `value_threat_inject off`** (declared in `arch_arms.py` before its first seed); **C3 = `--arch endstate`** (C2 + `op_reduction principled` + `ko_ramp exact` + `g_ledger eot` + `effective_stats on` + `move_target_state on` + `obs_facts v1` + `drop_progress_clock on` + `move_set_closure on`; the FAILED arm). Search: test **C1** first (FAILS ⇒ group 1; PASSES ⇒ test **C2**: FAILS ⇒ group 2, PASSES ⇒ group 3 by elimination). At most TWO new 8-seed arms. The culprit GROUP is then bisected lever by lever as its own later registration (the B1–B8 groupings above remain the reference for that within-group split). Same rule, same production seeds, §2.3 identity per arm | the finer B1–B8 ladder first (up to five arms before any adoption) | owner, 2026-10-10 |
