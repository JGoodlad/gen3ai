# REVIEW (look 2): adversarial review of D3's final look, before any look-2 game (2026-10-27)

**Under review:** [`PREREG.md`](PREREG.md) through AM-D3-ENG and AM-D3-L2 (`35f60839a`);
[`read/read_spec_look2.json`](read/read_spec_look2.json) (sha256 `2e98bf74…`, equal to the planned state's
`spec.json`); the play pin `c4a5c37b6` (tag `x41-d3-look2-play-c4a5c37b`) with [`read/look2_pin.json`](read/look2_pin.json)
and [`read/look2_identity.json`](read/look2_identity.json); [`designs/tools/reads.md`](../../../tools/reads.md) §4;
[`kakuna_determinism_2026-10-27/README.md`](../kakuna_determinism_2026-10-27/README.md).

**Method.** CPU only, nice 19 on cores 5-7,13-15, no heavy slot.
- I read the pin's code from git objects and from the pin worktree (`PYTHONDONTWRITEBYTECODE=1`, read-only).
- I ran the registered rule (`main.x35.d1`, byte-identical at the pin and at main) in
  [`read/review_look2_checks.py`](read/review_look2_checks.py). Its output is
  [`read/review_look2_checks.json`](read/review_look2_checks.json) (seed 20261027).
- I read the archived captures under `~/gen3ai_archive/` (look 1's D3 units, D1's banked C0 units, the identity units).
- **No look-2 game exists.** The state `~/gen3ai_archive/x41_d3_read/look2/` holds plan files only. Its queue starter
  (pid 3714879) is still waiting on the H-CTX supervisor.

Abbreviations: as in the PREREG. **Bank** = D1's committed `d1_read/read_look2_input.json` (`8c4884aa…`).

## Verdict

**The pin is sound: it plays E1's game on every path the verdict reads.** One registration defect will make the final
`decide` refuse. Fix it before the first game (B1). Two MAJORs change what the read can conclude. Register both before
the read starts:
- NULL cannot be reached at look 2 (M1).
- Look 2 has 3–7 % conditional power for GAIN (M2).

| # | rank | finding | fix before the read starts |
|---|---|---|---|
| B1 | **BLOCKER** | the registered VOID gate reads `K9(b) MISSING` for d3_s3505 at the pin, so `record` / `decide` refuses | re-register `void.k9b_dir` (or the decide checkout), re-plan |
| M1 | MAJOR | NULL is unreachable at 2,100 games per arm (half-width ≥ 3.0 pp at every plausible rate) | declare it, or amend the band, in an AM note |
| M2 | MAJOR | P(GAIN at look 2 \| what is known) ≈ 3–7 %; P(UNDECIDED) ≈ 93–97 % | confirm the spend; register "look 2 is terminal, no pooled look 3" |
| m1–m8 | MINOR | guard ordering, choice-level guard, the loader, the S lane's identity, unpinned external state, descriptive rows, the lane horizon, the AM's prose | each a one-line registration or a record-time check |

---

## BLOCKER

### B1. The registered VOID gate cannot pass at the pin

`params.void.k9b_dir` is repo-relative
(`designs/research_state/measurements/x35_breakers_stage_2026-10-17/k9b`).
- `rules.void_gate` resolves it through `models.spec_path`, which is `repo_path`, so it reads the CHECKOUT THAT RUNS
  `record`.
- s3505's K9(b) record shipped in `28711af06`, after the pin's base `a9c09711d`. It is not in `c4a5c37b6`'s tree.

Measured with the pin's own code on the planned state's `spec.json`:

| checkout | d3_s3501 | d3_s3502 | d3_s3503 | d3_s3504 | d3_s3505 | `void_failures` |
|---|---|---|---|---|---|---|
| pin worktree `d3l2_pin` | PASS | PASS | PASS | PASS | **MISSING** | `{'d3_s3505': 'K9(b) MISSING'}` |
| main | PASS | PASS | PASS | PASS | PASS | `{}` |

`api.void_gate` runs only at `record` / `decide`. So the read would play its whole ≈ 20 h and then refuse. The
obvious workaround would be to decide from main, but that is a post-data choice of the code that decides:
- main's `main.reads.assemble` differs from the pin's by +186 lines, and `rules` by +92 (the choice-level guard, the
  peer-pack id, H21);
- nothing in AM-D3-L2 says which checkout decides.

The play is unaffected. This is a registration defect, and fail-closed. It is a BLOCKER only because the cheap fix
must be registered before any look-2 data exists.

**Fix (≈ 10 min, before the first game; one of):**
- (a) **Preferred.** Re-register `void.k9b_dir` as an ABSOLUTE path to main's checkout
  (`spec_path` honours absolute paths), commit the spec, and re-plan the state. Only plan files exist, so this costs
  nothing.
- (b) Register that `record` / `decide` run from the pin worktree, with main's
  `k9b/rb_x35_d3mut_s3505.json` blob copied there untracked and its sha256 checked. The deciding code stays the pin's;
  the queue starter's `--untracked-files=no` precondition is unaffected.

Either way, an AM-D3-L2 addendum names the deciding checkout.

---

## MAJOR

### M1. NULL is unreachable at look 2, and was at D1's look 2

H's interval is the Newcombe hybrid of two Wilson intervals at the stratified n_eff (`d1.strat_score`,
`newcombe_strat`), widened by √max(1, φ̂). At 700 games per panel per arm, the UNWIDENED half-width is:

| every panel's rate | 0.40 | 0.44 | 0.45 | 0.48 | 0.50 |
|---|---|---|---|---|---|
| half-width (pp) | 2.96 | 3.000 | 3.006 | 3.019 | 3.022 |

NULL needs the whole interval inside (−3.0, +3.0). That requires panel rates below about 0.44 (or above 0.56), and
every arm here sits at 0.45–0.50. Two checks:
- **The best case:** ON set equal to the bank's own counts (Δ̂ = 0 exactly) reads **[−3.015, +3.015] → UNDECIDED**.
- **Simulation:** NULL never appeared in any of the ≈ 168,000 simulated reads (every scenario in the JSON).

D1's own final read +2.10 [−0.92, +5.11], a half-width of 3.015, so D1's NULL row was dead too.

The registration predicted NULL with probability 0.15. §6's NULL row ("exact-set identification on familiar cores is not
the binding channel") cannot be issued by this rule at this n.

**Fix (an AM note before any look-2 game; the orchestrator chooses):**
- (a) **No rule change.** Declare NULL unreachable. Register that the final interval is reported as the bound on D3's
  H effect. For example, "an UNDECIDED whose interval lies inside [−3.1, +3.1] is reported as *no effect beyond
  ≈ 3 pp*", as a reported line, not a class.
- (b) Amend the band (e.g. ±3.5 pp). This is a rule change, and it breaks D1's verbatim comparability. AM-D3-L2 says
  "no decision-rule change".

### M2. Conditional power at look 2 is 3–7 %; register that look 2 is terminal

Most of what decides look 2 is already known:
- the bank fixes C0 s3501–s3505 (per panel 0.451 / 0.489 / 0.473, mean 0.471);
- look 1 fixes D3 s3501–s3503 up to the E1 re-play, which moves about 1 % of games.

Only D3 s3504–s3505 are random. Under the registered rule:

| D3 s3504–s3505 true rate | P(GAIN) | P(UNDECIDED) | P(NULL / LOSS) |
|---|---|---|---|
| C0 + 0 | 0.003 | 0.997 | 0 |
| C0 + 1.27 pp (look 1's Δ̂) | 0.025 | 0.975 | 0 |
| D3's look-1 per-panel rates | 0.070 | 0.930 | 0 |
| C0 + 3 pp | 0.175 | 0.825 | 0 |
| C0 + 5 pp | 0.575 | 0.425 | 0 |

GAIN needs both new seeds at ≥ 73 / 140 on every panel (0.521, ≈ 5 pp above C0); that reads Δ̂ +3.24 [+0.22, +6.25].

Unconditionally (both arms binomial, one look), P(GAIN) at true Δ = +1.5 pp is 0.14, and at +3 pp it is 0.42. The
read costs 2,240 Kakuna games (≈ 16 lane-h quiet, up to ≈ 58 h at look 1's contended 9–31 min a unit) plus 17,500
S pairs (≈ 7.5 h).

**The likely final is UNDECIDED with a ≈ ±3 pp interval.** That still has value: it bounds D3's effect on H, it
measures P at 5 seeds, and it closes the family. But register it as the expected outcome.

**Fix:**
- (1) The orchestrator confirms the spend with this table in hand.
- (2) Register in the AM that look 2 is TERMINAL. No further D3 units or seeds are pooled with these rows. Any later
  D3 read is a new registration on new seeds. An UNDECIDED followed by "a few more seeds" is the one path that would
  inflate the error rate.

---

## MINOR

- **m1. The guard does not run first.** §3 says the C0 guard is checked BEFORE any D3 unit. At the pin,
  `runner._anchor_todo` interleaves by unit seed: guard unit s plays after the 15 D3 units of seed s, so the last guard
  unit plays near the end. There is no contamination, because a failed guard re-plays only C0. Fix: register that the
  80 / 80 identity check, run before any D3 game, discharges §3's ordering at look 2. Or pre-run the 7 guard units
  (≈ 1 h).
- **m2. The guard compares 5-tuples, not choices.** H counts results only, so a choice-only divergence cannot bias
  Δ̂. But it would signal a play-path difference. The units capture (`--capture-dir` on panel cells). Fix: before
  `record`, run [`read/look2_identity_compare.py`](read/look2_identity_compare.py) on the 7 guard units against the
  bank. Register that any choice difference is a FINDING that stops pooling.
- **m3. Assert the loader.** Our side must load `bare` (`load_checkpoint_strict`, restored, training mode), as the bank
  and look 1's 70 / 70 D3 units did. The pin's `foreign` fallback (`session.foreign_loader`, a harness file from
  `141f36f7f`) loads in eval mode, unlike the bank. Fix: a record-time check that every counted unit stamps
  `model_loader: bare`. This also corrects AM-D3-L2's wording: only `snapshot.py`'s half of the eval-mode change is
  restored.
- **m4. The S lane has no executed identity check.** `main.reads.playpath` covers the anchor lane only. I computed
  `main.h2h`'s import closure at the pin and at `6c428453a` / `5e389ddac` / `f4b25238d`:
  - 43–44 of ≈ 417 files differ, all of them training-only, launcher, or anchors-harness modules
    (`agents/training/*` loss and rollout code, `main/train/*`, `main/anchors/*`, the lock, port and procfs helpers);
  - `main/h2h`, `agents/{model,observation,inference}`, `rust_env` and the packer are identical.

  I accept this as declared. If S is deterministic per seed, re-playing one banked C0 home72 batch (100 pairs,
  minutes) would close the gap.
- **m5. External state that the pin does not pin.** The Metamon checkout `/home/goodlad/dev/metamon` is at
  `0a00a759` (= the bank's `opponent_commit`, which is in the id), clean except an untracked `cache/`. The `metamon`
  env is unchanged since 2026-09-14. The ids see neither a dirty checkout nor an env change, and neither does
  `gen3ai_torch28`. Fix: snapshot `git status` and a `conda list --explicit` digest of both envs at the start and end
  of the read, and record them.
- **m6. Report two descriptive rows** (no rule reads them):
  - D3 s3501–s3503 on E1 vs look 1's E2, game for game. This is a free measurement of the engine effect on the ON
    arm.
  - Look 1's Δ̂ recomputed on E1. AM-D3-ENG already says the CONTINUE stands. A sign flip would need a net −16 wins
    from roughly 1 % of games diverging, so it is not expected.
- **m7. The anchor lane's horizon.** The starter passes `--hours 40`. At look 1's contended pace the lane can stop
  with units left (`stop_at`). Resuming it is supported; put that in the operator's checklist.
- **m8. Peer pack: the brief's question has a stronger answer.** The pre-`b0441623` pack touches NO H game. Neither
  side ever plays the forum72 Frustration set (`sft431`) at unit seeds 0–6:
  - 0 of 700 banked C0 forum72 games and 0 of 420 look-1 D3 forum72 games contain Frustration;
  - the 7 unit seeds draw 52 of the 72 forum72 teams, identical across finals;
  - the Hidden Power IV cases are away and pool only, and home72 (P) is unchanged.

  Both arms share the regime regardless. The split is bookkeeping, and the id already keeps these rows apart from
  post-`b0441623` forum72 reads.

---

## Checked and sound (no finding)

- **The hybrid pin reproduces E1 play.**
  - Every restored tree equals `f4b25238d`'s, and equals both bank commits' (`src/rust_sim` differs from them only in
    `CLAUDE.md`).
  - `engine_at` and the built stamps are E1. The stamp refuses a binary built from other sources.
  - Dropout is absent, so the bare (training-mode) load is deterministic.
  - I read the anchor play path's harness diffs (`core_side`, `session`, `runner`, `results`, `live_side`,
    `heldout_panels`, `kakuna`). None changes a decision: the belief-prior-mix path is inert at `None`, the label guard
    only refuses, and the stamps are metadata.
  - All three panel configs load at the pin.
- **Is 80 / 80 enough? Yes, as a backstop to source identity, which carries the proof.**
  - The 80 games hold 4,910 of our decisions, all identical. That bounds a systematic per-decision divergence below
    about 0.06 % (95 %).
  - Three units are positive controls that diverge on E3.
  - Same-engine replay is exact (0 / 2,100).
  - What 80 / 80 alone does not cover:
    - heldout72_hpfix: the in-state guard covers it;
    - D3 models: they use the same loader and code;
    - Frustration / Hidden Power packer games: none occur in H (m8).
- **Look 1's E2 rows are excluded and every D3 unit is re-played on E1.** Look 1's record is untouched. The
  pinned shas equal look 1's resolved finals and the files on disk. All finals are at ≥ 300,000,170 steps.
- **The C0 import:**
  - `8c4884aa…` and `be989806…` match the files;
  - `LEGACY_BANK_ENGINE` maps both to E1, in both lanes for the bank;
  - `import_off` checks the regime ids (`09f297883701a35a`, `d4de0fd7c16a2658`), both engines, P's panel / team set
    and the guard.
- **The pin's rule reproduces look 1:** +1.27 [−2.63, +5.16], CONTINUE.
- **Error rates.** There is no alpha spending; each look is 95 % with a sign rule.
  - Under Δ = 0, the two-look false-GAIN rate is 2.4 % (1.5 % at look 1 + 0.9 % at look 2).
  - With a 1.5 pp seed SD it is 2.9 % for one look. This is about a one-sided 2.5 %, consistent with `5b84bca6`'s
    3.5–7.4 % at larger seed SDs.
- **Pairing.** Both arms play the same teams and seeds. On look 1 the paired-by-game SE is 0.89× the unpaired one, so
  the registered unpaired statistic gives up little power.
- **The VOID records:** K9(b) PASS for all five D3 seeds in main's `k9b/` (B1 is about where the record is read, not
  what it says).
