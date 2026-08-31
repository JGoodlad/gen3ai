# TWO CORRECTNESS REPAIRS — the stale exclusion list, and the uncommitted admission artifacts

**Status: COMPLETE.** Both are repairs of ARTIFACTS against recorded reality. No policy changed, no
model touched, no battle played, nothing under `data/teams/` created, moved or deleted. Found in
passing by `exploitability_taught_untaught_2026-08-31.md` (§Headline 5, §"What to change" 3) and by
its own Reproduce section.

---

## FIX 1 — `designs/ai_v12/promotion_exclusions.json` was built from a PLAN, not a RECORD

### What was wrong

The artifact is the STANDING exclusion union for `python -m main.promote_teams`, the seeded random
draw that will pick the 40-team fleet. It was generated on 2026-08-30 from **frozen argv files** in
a session-scoped job directory — before the rev-4 runs had launched, which is why its `rev4_pending`
blocks carried `"run_dir_present": false, "metadata_verified": false`.

The launched runs did not deal the teams those argvs did. `--verify-exclusions` fails on **all
three** rev-4 arms:

| arm | run | named but NEVER PINNED | PINNED but missing |
|---|---|---|---|
| R4S3a | `ai_v9_73_R4S3a_0829` | `1c4e182530` `564b9be3ae` `bcd4d09ee9` | `3495ef83ef` `6a49f096f0` `a7bb29d48c` |
| R4S3b | `ai_v9_74_R4S3b_0829` | `55ff6899a2` `a7bb29d48c` `c90e782cad` `fed4eee838` | `21022d30fb` `78b3b6f4a6` `a7406f6c97` `c84f2b64a2` |
| R4S3c | `ai_v9_75_R4S3c_0829` | `6a49f096f0` `78b3b6f4a6` | `1c4e182530` `bcd4d09ee9` |

**Five** of those per-arm errors are teams that merely moved *between* rev-4 arms — `1c4e182530`
(a→c), `bcd4d09ee9` (a→c), `a7bb29d48c` (b→a), `6a49f096f0` (c→a), `78b3b6f4a6` (c→b) — and cancel
in the union. **Four are genuinely wrong in the union, in each direction.**

`taught_F5` (5 arms) and `taught_F6` (6 arms) were **re-verified CORRECT** — 11 of 11 arms match
their runs' `metadata.json` exactly. That is a confirmation of the decomposition's own verification,
not a second opinion overturning it, and it is why the taught/untaught result stands unaltered.

### Before → after, exactly

The **union stayed 26 and the eligible count stayed 693** — 4 teams left, 4 entered. That is the
whole reason this survived: `promote_teams_test.py` already asserted
`len(excl.union) == 26`, `counts == {taught_F5: 9, taught_F6: 12, rev4_pending: 24,
held_out_instruments: 2}` and `719 − 26 == 693`, and **every one of those assertions passed while
the list was wrong.** A count-shaped check cannot see a membership error.

| | before | after |
|---|---|---|
| pool | 719 | 719 |
| union | 26 | **26** |
| **eligible** | **693** | **693** |
| `taught_F5` / `taught_F6` / `rev4_pending` / `held_out_instruments` | 9 / 12 / 24 / 2 | 9 / 12 / 24 / 2 |

**NOW EXCLUDED (were eligible for the "fresh" draw, but rev-4 had pinned them):**

| sha | file | pinned by |
|---|---|---|
| `3495ef83ef` | `data/teams/sample/e11829f0561ef5a9.txt` | R4S3a (`ai_v9_73_R4S3a_0829`) |
| `21022d30fb` | `data/teams/sample/b89e1e37caa40e6a.txt` | R4S3b (`ai_v9_74_R4S3b_0829`) |
| `a7406f6c97` | `data/teams/sample/a04c29cf769e9a11.txt` | R4S3b (`ai_v9_74_R4S3b_0829`) |
| `c84f2b64a2` | `data/teams/sample/9f27f5d3e34021a7.txt` | R4S3b (`ai_v9_74_R4S3b_0829`) |

**NOW ELIGIBLE (were excluded, but no rev-4 arm ever trained on them):**

| sha | file | claimed by the frozen argv of |
|---|---|---|
| `564b9be3ae` | `data/teams/sample/9d5f845869e899ee.txt` | R4S3a — never ran |
| `55ff6899a2` | `data/teams/sample/9283210847f806ee.txt` | R4S3b — never ran |
| `c90e782cad` | `data/teams/sample/90b94599967c6b77.txt` | R4S3b — never ran |
| `fed4eee838` | `data/teams/sample/61590463ee85d456.txt` | R4S3b — never ran |

Both directions were live defects: the first four would have let genuinely-taught teams back into a
draw whose entire value is that it is untaught, and the last four were wasting eligible curated
teams — the scarcest resource in the whole 40-team plan, where §1 of the slate shows only 8 curated
teams remain untaught at all.

**The `--dry-run` eligible count is unchanged (693 → 693), but the DRAW is not.** Re-running the
committed demo at its own seed `20260830` moves **21 of the 40 drawn positions**, with 20 teams in
and 20 out. That is not an error, it is what a seeded shuffle of a *sorted* eligible list does when
4 of its 693 members are swapped: every element between a removal and an insertion shifts index. A
seeded draw is reproducible against a FIXED eligible set; it is not stable across a change to that
set. `designs/ai_v12/promotion_dry_run_demo.{json,md}` are re-drawn at the same seed and say so.

### The regeneration path, so it cannot rot again

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.promote_teams --verify-exclusions      # report (exit 1 on drift)
python -m main.promote_teams --regenerate-exclusions  # repair, from run metadata alone
```

`--regenerate-exclusions` rebuilds every run-derived block from that run's own `metadata.json`
(`cli_args.trainee_teams` → `team_sha`), prints the before/after team ids in both directions, and
rewrites the file. It carries verbatim exactly two things it must not invent: a category with no
runs (`held_out_instruments` — held out by DESIGN, so no metadata could confirm it) and an
unlaunched run's block (the case the artifact marks `metadata_verified: false`).

Verify, repair and the gate share **one** derivation, `promote_teams.recorded_provenance` — so a
check and a repair cannot disagree about what "recorded" means.

### The gate

`src/main/promote_teams_test.py::test_the_committed_exclusions_agree_with_recorded_run_provenance`
fails on any per-arm disagreement and **names the offending team ids in both directions**
(`in_artifact_never_pinned` / `pinned_but_missing`), because the count is what was already checked
and what already passed. It skips when there is no `models/` archive (`main_models_dir()` is
`None`) — and asserts, before the real assertion, that at least one named run is actually present,
so it cannot pass vacuously on a box where every run is missing.

Verified failing on the pre-repair file and passing on the repaired one.

### Downstream

`designs/ai_v12/team_slate_build.py` had the same root defect — it read the frozen `*.argv` files
directly. It now sources `TAUGHT`/`REV4` from `promotion_exclusions.json` (metadata-derived, gated),
falling back to an argv only for a fleet with no run block, and its §1 "curated and still untaught
after rev-4" row is **derived** instead of the hardcoded literal it was. That literal named the
right COUNT (8) and the wrong SET:

| | teams |
|---|---|
| was | `ce35b736 · b89e1e37 · 9909f2e9 · e11829f0 · f7ba5702 · dbf81d8e · a04c29cf · 9f27f5d3` |
| **is** | `61590463 · 90b94599 · 92832108 · 9909f2e9 · 9d5f8458 · ce35b736 · dbf81d8e · f7ba5702` |

`team_slate_40.{md,json}` are corrected **in place on the exclusion-derived fields only**, with a
banner saying so and a `_meta.corrections` entry recording the deltas. A full re-cut was tried and
deliberately reverted: the `headroom_screen` the slate's §7 was waiting on has since completed, so
regenerating now moves tier A from 18 to 32 teams, re-fits the generation offset −0.0100 → −0.0239
and changes every baseline. That is a research re-cut, not a correctness repair, and mixing the two
would make neither reviewable.

---

## FIX 2 — the admission artifacts are now in the tree

### What was wrong

`fleet_admission.json`, `r3_admission.json` and `r4_admission.json` are the entire evidential basis
of the exploitability curve, the teacher-ceiling reframe and the taught/untaught decomposition — and
they lived **only** in `~/.claude/jobs/1046b1d6/tmp/probes/`, a session-scoped job directory. Every
exploitability claim in the ledger was one cleanup away from un-reproducible.

### What was done

Copied byte-identically (sha256-verified against the originals) to
**`designs/research_state/measurements/admission_artifacts/`**, with a `README.md` recording
provenance, per-fleet shape and budget, the `fleet_admission` schema they satisfy, and the reproduce
commands. Total 19 KB across three files — no gzip; the project's row-level-data convention does not
bind at this size, and a plain JSON stays greppable and diffable.

`exploitability_taught_untaught.py`'s `PROBE_DIR` default now points at the committed directory
(`--probe-dir` still overrides), so the decomposition reproduces with no job directory at all.

### Reproduce — verified 2026-08-31

```bash
export PYTHONPATH=$PYTHONPATH:src

nice -n 15 python -m main.exploitability \
  rev-2=designs/research_state/measurements/admission_artifacts/fleet_admission.json \
  rev-3=designs/research_state/measurements/admission_artifacts/r3_admission.json \
  rev-4=designs/research_state/measurements/admission_artifacts/r4_admission.json

nice -n 15 python designs/research_state/measurements/exploitability_taught_untaught.py \
  --out /tmp/tu_check
```

* `main.exploitability` from the committed copies is **byte-identical** to the same command run in
  the job directory — `diff` clean, so the published rev-2 **+0.1165** / rev-3 **+0.1350** / rev-4
  **+0.1252** and both generation deltas reproduce exactly. It did **not** refuse: all three pass
  `load_artifact`'s schema validation and all three recompute the
  `net = teacher − reference = ordered − seniority` identity against their own per-team cells.
* The decomposition rebuilt from the committed copies matches
  `exploitability_taught_untaught_2026-08-31.json` **exactly**, the only difference in the whole
  document being the three recorded `artifact` path strings.

### What is deliberately NOT committed

The piloting/coverage probes `team_slate_40.md` §8 cites (`pilot_R2ACTION_n300.json`,
`cov_R2ACTION.json`, `cov_rev1fin.json`, `headroom_screen.json`, `coverage_sample.json`,
`coverage_sweep.json`) stay in the job directory. Their numbers are already distilled into the
committed `team_slate_40.json`, which is what makes the slate survive that directory's deletion.
The three files copied here are the ones a live TOOL reads.

---

## The durable lesson

**A FROZEN ARGV IS A PLAN; `metadata.json` IS THE RECORD.** Both were available and only one is
authoritative, and the artifact that got it wrong said so in its own fields
(`run_dir_present: false`) for a day and then nobody re-read it once the runs existed.

Its companion: **a count-shaped check cannot see a membership error.** Four in, four out, union 26
throughout — every committed assertion about this file passed the entire time it was wrong. That is
the `vacuous_tests_and_guards.md` "presence-not-value" class, one level up: the assertions were not
vacuous, they were about the wrong quantity.

## Cross-references

* `exploitability_taught_untaught_2026-08-31.md` — the probe that found both, §Headline 5 and
  §"What to change" 3 (now struck through).
* `admission_artifacts/README.md` — the artifacts' own provenance record.
* ledger `6990-7007` — S5, `python -m main.promote_teams` landing (union 26, eligible 693 of 719;
  both figures still hold).
* `designs/ai_v12/team_slate_40.md` §1 — the curated-32 cap the corrected untaught set feeds.
