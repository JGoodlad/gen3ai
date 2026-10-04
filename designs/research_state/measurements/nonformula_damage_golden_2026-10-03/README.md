# The K9 learner-golden re-bake proof for `gen3_nonformula_damage_v1` (X5 Tier 0 F8), 2026-10-03

**Claim.** The learner golden moved (on the pre-fix buffer: post-update params `bbfab0a0…` → `e3a9c53b…`, 36
of 41 parameter groups, 16 of 19 pinned losses; init `f476942c…` unchanged; recorded on the rebuilt buffer:
`ef8a1b7c…`) ONLY because of the non-formula damage
class's table entries. Nothing else in the code-path change moves an element.

**Method** (`scripts/proof.py`, torch `2.8.0+cu126`, CPU, one thread, the golden's own `compute`):
run the golden update on the PARENT tree (`cda60edb`) and on the FIX tree, each with and without the
class NEUTRALISED in the op's tables:

* parent, neutralised: `MOVE_FIXED_DAMAGE := 0` (the only class table the old op had);
* fix, neutralised: every `gen3_nonformula_damage_v1` table `:= 0`, `MOVE_BP` at the declared-BP rows
  back to the dex 0, `MOVE_PHYS` at the dex-damaging BP-0 rows back to the facade-derived 0.

If the two neutralised runs are byte-identical, the fix's code paths (the new `damage_kinds`
override, the `usable` gate, the effective BP, every kernel's new plumbing) are a no-op except
through those table entries, and every changed element is caused by a class move.

**Result** (`out/*.json`):

| run | post_params_sha256 |
|---|---|
| parent | `bbfab0a0f3164d82…` (= the banked golden) |
| parent, neutralised | `9a171336e2545162…` |
| fix | `e3a9c53b8b96c7ef…` (re-recorded) |
| fix, neutralised | `9a171336e2545162…` |

Parent-neutralised == fix-neutralised: params, all 41 group hashes and all 19 losses identical.
(The parent itself moves when neutralised: its INCOMING fixed-damage override was live — only the
outgoing one was dead.)

**The rebuilt buffer.** The committed buffer's stored behaviour log-probs were the pre-fix op's, so the K9(b)
probe on it failed after the fix. `python -m agents.training.learner_golden rebuild-buffer` re-recorded it from
the same seeds: observations, actions, masks, rewards and episode starts are BYTE-IDENTICAL (the same games);
only `values` (64/64, max |Δ| 5.9e-3), `log_probs` (57/64, 8.1e-3), `advantages` and `returns` moved — the
behaviour policy's outputs under the fixed op. The proof was repeated on it (`out/rebuilt_*.json`, `--buffer`):

| run (rebuilt buffer) | post_params_sha256 |
|---|---|
| parent | `682d45e70b8b0f71…` |
| parent, neutralised | `3e394b6228ee3d8e…` |
| fix | `ef8a1b7c9e5ced1c…` (the recorded golden) |
| fix, neutralised | `3e394b6228ee3d8e…` |

Neutralised runs identical again (params, groups, losses).

**Which class moves the buffer reaches** (the op's forwards during the update): Seismic Toss and Hidden
Power in visible move slots; Counter, Endeavor and Beat Up in the opp active's believed top-K; Beat Up in
our request slots. Of these, Seismic Toss (now priced in every kernel), Endeavor (newly priced) and the
`MOVE_PHYS` / `MOVE_BP` rows move the features; Counter is `unmodelled` (0 before and after); Beat Up's
table is unchanged.

**Reproduce:**

    git worktree add --detach <old> cda60edb
    PYTHONPATH=<old>/src  python scripts/proof.py --label old [--neutralise] --out out/old[_neutralised].json
    PYTHONPATH=<fix>/src  python scripts/proof.py --label new [--neutralise] --out out/new[_neutralised].json
    # the rebuilt buffer: add --buffer <fix>/src/agents/training/learner_golden_buffer.npz (out/rebuilt_*.json);
    # the pre-fix buffer is cda60edb's src/agents/training/learner_golden_buffer.npz

`OMP_NUM_THREADS=1` (the golden builds and updates at one thread anyway).

**Cost** (`scripts/perf.py`, `out/perf_cpu.txt`): the production forward + backward on the 64 golden rows, CPU
eager fp32, one thread, the two trees interleaved: +1.0 to +1.3 % on the median (the non-formula override now runs
in every kernel). The GPU compiled path is **UNVERIFIED**.
