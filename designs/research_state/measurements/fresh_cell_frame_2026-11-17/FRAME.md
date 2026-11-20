# One fresh cell against a banked BOUNDARY-1 cell: the frame (2026-11-17; ADOPTED 2026-11-20)

**Status: ADOPTED 2026-11-20, rules unchanged since `52353b1b7`, by two FULL registrations at width 128:
[`../q6n_policy_readout_fresh_2026-11-20/PREREG.md`](../q6n_policy_readout_fresh_2026-11-20/PREREG.md) (first in the
queue) and [`../q5n_epochs_fresh_2026-11-20/PREREG.md`](../q5n_epochs_fresh_2026-11-20/PREREG.md). The 256 arms (Q5-W,
Q6-W) were NOT registered (the orchestrator's ruling after B1's 50M look: width is not ahead). § 4's σ amendment is
Q6-N's §5. From here this file changes only by a dated amendment that both registrations cite. By itself it
registers nothing and launches nothing.** It is the shared part of every later registration that
reads ONE new three-seed fresh cell against ONE banked BOUNDARY-1 (B1) cell. Such a read can adopt a recipe or an
architecture, so each lever is registered in the **FULL tier** (`../../README.md` § Registration tiers). Its
`PREREG.md` adopts this frame by reference, names its card's choices, and lands on main before its first step. Two
lever cards sit beside this file: [`CARD_Q5_epochs.md`](CARD_Q5_epochs.md) (fewer PPO epochs at matched dose) and
[`CARD_Q6_policy_readout.md`](CARD_Q6_policy_readout.md) (the flat policy tower retired). Source of the unit: the
queue proposal `~/.cache/gen3ai/handoffs/post_b1_queue_proposal_2026-11-16.md` §1 D4, §2 rows Q5 / Q6, §3, §4 P5.

**Blind (as written, 2026-11-17).** No B1 look has been planned or played, and no B1 head-to-head or Kakuna number was read to write this. What
was read, none of it a score: B1's PREREG (§2 to §11, Amendments 1 to 6), `python -m main.dose` on the six C and W
runs, and the sibling analysis of the wide arm's learning rate (`~/.cache/gen3ai/handoffs/b1_wide_lr_floor_2026-11-17.md`).

Abbreviations: B1 = BOUNDARY-1 (`../boundary1_scratch_2026-11-09/PREREG.md`); pp = percentage points of win rate;
SD = standard deviation; df = degrees of freedom; Δ̂ = the estimated effect; lo / hi = the ends of a two-sided 90 %
interval; σ = the seed SD (the SD across seeds of one arm, game noise removed). Views and panels are B1's (§5.1):
**N** = ½ home72 + ⅙ H + ⅙ F + ⅙ P72. GPU-h = hours of the one training card.

## 1. The design in one paragraph

A **banked cell** is the three runs of one B1 arm (C at width 128, or W at 256), trained at B1's launch pin
`15136b61634b667bb9da5ccc6647ff9265238829`. A **fresh cell** is three new runs that type the banked arm's argv token
for token, plus ONE lever, a new seed and a new name. Both cells stop at 50M steps, are read by a head-to-head cross
(every fresh run against every banked run, 9 cells), and the fresh cell continues to 100M only if it is ahead at
50M. The 100M cross decides, with B1's own statistic and classes. The banked cell costs nothing: that is what
makes a one-lever fresh arm cost one cell (≈ 28.6 GPU-h narrow, ≈ 39.6 wide, to 50M; queue proposal, from B1's run
log) instead of two.

## 2. When a banked cell may be the control (every item, else § 7)

1. **The same pin.** Every fresh run types `--pin-commit 15136b61634b667bb9da5ccc6647ff9265238829`, and every look
   plays at that pin, as B1's looks do (B1 §3). The lever's flags exist in THAT commit's parser: `python -m
   main.checkargs --argv "<argv>"` prints `parser : the PINNED commit 15136b61's` and `unrecognized : 0`.
2. **Exactly one lever in the resolved config.** The fresh argv is the banked arm's B1 argv
   (`main.b1.arms.build_argv(<arm>, <its first seed>, <pin>, <target>)`, which B1 §10 already holds equal to each
   banked run's immutable `original_command`) with the lever's tokens inserted before `--checkpoint-every-steps`,
   and a new `--seed` / `--run-name`. The check uses B1's own helpers (`main.b1.arms.resolved_namespace`, the
   trainer's parser then `resolve_config`, and `namespace_diff`), run in a clean detached checkout of the pin so the
   parser is the pin's, at both step targets (50M and 100M). It PASSES iff the keys that differ are exactly the
   card's declared keys plus `seed` and `run_name`. No new tool is needed: `main.dose` reads finished runs only,
   and `checkargs` reads one argv against production, so neither diffs two argvs; the kit's functions do. The
   command (the card's records are its output):

   ```bash
   # in a clean detached checkout of the pin, PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
   python - <arm> <seed> "<lever tokens>" <fresh seed> <fresh name> <<'EOF'
   import shlex, sys
   from main.b1.arms import build_argv, resolved_namespace, namespace_diff
   PIN = "15136b61634b667bb9da5ccc6647ff9265238829"
   arm, seed, lever, fseed, fname = sys.argv[1], int(sys.argv[2]), shlex.split(sys.argv[3]), sys.argv[4], sys.argv[5]
   for steps in (50_000_000, 100_000_000):
       c = build_argv(arm, seed, PIN, steps); a = list(c); k = a.index("--checkpoint-every-steps")
       a[k:k] = lever; a[a.index("--seed") + 1] = fseed; a[a.index("--run-name") + 1] = fname
       nc, na = resolved_namespace(c), resolved_namespace(a)
       print(f"@{steps}:", shlex.join(a))
       for key in namespace_diff(nc, na):
           print(f"   {key}: control={nc.get(key)!r} arm={na.get(key)!r}")
   EOF
   ```
3. **The same base and tail.** `--arch production --ent-coef 0.0125 --behaviour-check warn`, the 6 h restart, a
   checkpoint every 1,000,000 steps, `--device cuda --log-level periodic`; phase 1 fresh to 50,000,000, phase 2 a
   same-run resume to 100,000,000 by B1's resume form (§14: `--model models/<run>/final_model.zip --steps
   100000000`, nothing else typed). The 50M stop is then the same "one more restart" for both cells (B1 §4).
4. **The same runtime.** The run's torch is `2.8.0+cu126` from step 0 and its pin history is ONE entry, the pin, from
   step 0 (B1 Amendment 6 item 8: a save under another torch appends a second entry and B1's guard then refuses the
   run). `--allow-torch-switch` is never typed. The core split is recorded at each launch (`python -m utils.core_split
   show`); B1 launched every run at `5+3: trainer 0-4,8-12, agents 5-7,13-15` (B1 §18 slot S8). A different split is
   recorded as a deviation; it does not void the read (**UNVERIFIED:** that the split moves strength; B1 §13 item 9
   froze it without measuring that).
5. **Seeds and names free.** B1's own scan (`main.b1.arms.scan_collisions` over the main `models/` and
   `~/gen3ai_archive`) pointed at the card's seeds and names finds no hit; it is re-run immediately before the first
   launch. The cards' seed blocks are disjoint from B1's (5101 to 5113) and from each other.
6. **The banked cell is still banked at the look.** Its three runs pass B1 §10's run guards at the look's step, and
   their look files are on disk, byte-identical to what B1 read.
   **THE RETENTION RULE: never thin these.** The 50M and 100M look checkpoints of B1's C runs (`rb_b1c_s5101`,
   `rb_b1c_s5102`, `rb_b1c_s5110`) and of W's (`rb_b1w_s5103`, `rb_b1w_s5104`, `rb_b1w_s5111`), each with its
   sidecar `.json`, plus every other checkpoint a registered card names (its card says which), are never thinned,
   moved or rewritten by any plan or agent, after B1 §9 closes as before it. B1 §4 forbids thinning only "before
   §9's reading is closed", so without this rule nothing keeps them (finding F2). Deleting run data is the owner's
   alone; this rule asks nothing of the owner and grants no deletion. It is written into B1's slot S9 and
   `../../models_retention_policy.md` by the first card's registration commit.
7. **The banked cell reaches the look.** C runs to 100M under every B1 outcome. W, Br and WB do not after a 50M
   STOP-HARM of their factor (B1 §8): then a fresh arm at that width has no 100M control and is read at 50M only
   (REPORTED, no class), or buys its own control (§ 7).

## 3. The meter: a 3 × 3 head-to-head cross

- **Cells:** every fresh run against every banked run, once: 9 cells a look. **Each cell is B1's** (§5.1): 1,000
  mirrored pairs on home72 and 500 on each of H, F and P72; both sides greedy; one engine, the pin's; `--seed 0`;
  oracle reveal off; ONE device for the whole look. 45,000 games a look, a sixth of a B1 look.
- **The file each run plays** at a look is B1's rule: the first `checkpoints/checkpoint_<N>_steps.zip` with N at or
  after the look's step and below the look's step + 1,000,000; never `final_model.zip`.
- **The statistic is B1's simple effect, as coded:** `main.b1.cross.simple_effect(design, scores, "<fresh>",
  "<banked>", view="N")`. Δ̂ = the mean of the 9 cell scores − 50; V̂ = s²_fresh / 3 + s²_banked / 3, where each s² is
  the variance of that arm's three row means, **floored at 1.76²** (B1 §5.2's floor, unchanged); 4 df, t = 2.132;
  the interval Δ̂ ± t·√V̂, two-sided 90 %. Its half-width is never below 3.06 pp. A bound within 0.005 pp of a
  threshold fails the comparison (`at_boundary`, root rule 8).
- **The deciding view is N** for both cards (neither lever is a breadth lever, so B1's F-only rule for breadth does
  not apply). Every other view (home72, U, F, H, P72, HF) is REPORTED.
- **The classes at 100M** are `main.b1.cross.classify("100M", stat)`, B1 §5.3's table in its order: **WIN** (lo > 0),
  **LOSS** (hi < 0), **NO-USEFUL-GAIN** (hi < +4.0), else **UNDECIDED**.
- **Validity (B1 §10, applied to both cells):** a run counts iff its look checkpoint exists with a matching sidecar
  step, its pin history is one entry from step 0, it is `fresh` with no fork parent, its `original_command` is its
  registered argv token for token, its last process ran the card's epochs at `ent_coef` 0.0125, and its K9(b) read
  through the look step is PASS. A look is INCONCLUSIVE (never a class) if a cell is missing, duplicated, or holds
  other than the registered pairs; if more than a quarter of a cell's games are drawn; or if its rows name more than
  one commit, team set per panel, device or backend.
- **The registered rule is `b1_fresh_cell`, and the kit is `python -m main.b1.fresh`** (finding F1, BUILT
  2026-11-17). The rule (`main.b1.fresh_rule`, registered in `main.reads.rules.RULES`; `designs/tools/reads.md` §5)
  is `main.b1.cross.simple_effect` on N, `classify("100M", …)` at 100M and § 4's Δ̂ > +0.005 rule at 50M, every other
  view REPORTED, with B1's draw and INCONCLUSIVE rules (`scores_from_cells`, `row_problems`) unchanged. `main.reads`
  plays no head-to-head cross, so a `main.reads` spec naming the rule is refused and the kit decides it; its two
  halves are pure, so `recheck` re-decides a look from `result/inputs.json` alone. Each card's fresh-arm table is a
  file beside it (`fresh_arm_q5n.json`, `_q5w`, `_q6n`, `_q6w`: argv tokens, seeds, run names, declared keys; DRAFT
  with its card). The commands, run from a kit checkout of main, with a clean detached checkout of the pin (Rust env
  core built) as `PIN_DIR`:

  ```bash
  python -m main.b1.fresh describe    --arm <card>/fresh_arm_q5n.json        # the fresh argvs and their resumes
  python -m main.b1.fresh lever-check --arm <J> --pin-tree $PIN_DIR          # § 2 item 2 at both targets; exit 3 FAIL
  python -m main.b1.fresh check-run   --arm <J> --run rb_q5n_s5201 --pin-tree $PIN_DIR   # one run, first checkpoint
  python -m main.b1.fresh plan  --arm <J> --look 50M --pin-tree $PIN_DIR \
      --b1-look <B1's 50M work dir>/look.json --out <W>                     # exit 3 = REFUSED (the six runs' guards)
  python -m main.b1.fresh play  --work <W> --pin-dir $PIN_DIR --device cuda # four panels; the caller holds the lease
  python -m main.b1.fresh read  --work <W>                                  # <W>/result/{inputs,verdict}.json
  python -m main.b1.fresh recheck --work <W>                                # re-decide from inputs.json; exit 3 differs
  python -m main.b1.fresh bootstrap --work <W>                              # REPORTED: B1's team-cluster bootstrap
  ```

  `plan` refuses a fresh run that is not exactly its card's lever over the banked argv (its `original_command`, the
  pin's resolved-config diff, its last process's declared keys, `n_epochs` and B1's lever keys, any undeclared
  `cli_args` drift), any run on another pin (an arm on another pin buys its own control: `own_control` in its
  arm file, § 7), and a banked run that fails B1's own `main.b1.look.check_run` or whose look file is not the bytes
  B1's `look.json` at that step records (§ 2 item 6). The parser the lever check uses is the PIN's, in `$PIN_DIR`; the
  rest of the kit (the K9(b) judge, the ledger reader) is the kit commit's. Tests: `src/main/b1/fresh_rule_test.py`
  (a hand-worked 3 × 3, every 0.005 pp boundary of both rules, label order, the INCONCLUSIVE rules),
  `fresh_test.py` (the arm files against the cards' recorded argvs, the guards on fixture runs, play's commands,
  a real-ledger read), and the opt-in CPU rehearsal `fresh_rehearsal_e2e_test.py`.

## 4. The looks: 50M decides whether to continue, 100M decides

| look | the files | what is read | consequence |
|---|---|---|---|
| **50M** | both cells at 50M | Δ̂ on N | **CONTINUE** iff Δ̂ > +0.005 pp, else **STOPPED-AT-50M** |
| **100M** | both cells at 100M | the class on N | § 5 |

- **The 50M rule only stops.** It reads the point estimate, not the interval: a fresh arm that is not ahead after half
  the budget does not get the other half. A STOPPED-AT-50M arm is reported with its 50M interval on every view and
  carries NO class: it is never a LOSS, never "no effect", never retired. If its upper end is below −2.0 pp (B1's
  harm-stop line) the record also says HARM-AT-50M, as a description.
- **Its operating characteristics, exact under B1's score model.** In B1's simulation model (`power_matrix.py`
  `simulate`: a cell is 50 + s_i − s_j + e_ij, each run's s ~ N(arm effect, σ²), cell noise e of SD 0.68 pp on N),
  Δ̂ over a 3 × 3 cross is normal with mean δ and SD √(2σ²/3 + 0.68²/9). So P(CONTINUE) = Φ((δ − 0.005)/SD), at a
  true 50M effect δ:

  | σ at 50M | SD of Δ̂ | δ = −4 | −2 | 0 | +2 | +4 | +6 pp |
  |---|---|---|---|---|---|---|---|
  | 1.76 (B1's floor) | 1.46 | 0.003 | 0.08 | 0.50 | 0.92 | 0.997 | 1.00 |
  | 2.5 | 2.05 | 0.03 | 0.16 | 0.50 | 0.83 | 0.97 | 1.00 |
  | 3.1 | 2.54 | 0.06 | 0.22 | 0.50 | 0.78 | 0.94 | 0.99 |

  An arm with no effect is continued half the time; one that is +4 pp ahead at 50M is lost at most one time in
  seventeen. It is not free for a slow starter: an arm behind at 50M and ahead at 100M is stopped (§ 8).
- **The 100M classes at three runs against three** (B1's helpers `simulate` + `simple_np`, every draw also classed by
  `main.b1.cross.simple_effect` + `classify`, 0 mismatches in 30,000; 50,000 draws a row, `numpy` seed 20261117; the
  command is in § 9). P(WIN) / P(LOSS) / P(NO-USEFUL-GAIN) / P(UNDECIDED) on N at a true 100M effect of:

  | σ at 100M | median half-width | 0 | +2 | +4 | +6 | +8 pp |
  |---|---|---|---|---|---|---|
  | 1.76 | 3.29 | 0.011 / 0.011 / 0.60 / 0.38 | 0.16 / 0.00 / 0.16 / 0.68 | **0.61** / 0 / 0.01 / 0.38 | 0.93 / 0 / 0 / 0.07 | 1.00 |
  | 2.5 | 4.21 | 0.028 / 0.027 / 0.41 / 0.54 | 0.15 / 0.00 / 0.15 / 0.70 | **0.43** / 0 / 0.03 / 0.54 | 0.75 / 0 / 0 / 0.25 | 0.93 WIN |
  | 3.1 | 5.09 | 0.037 / 0.037 / 0.30 / 0.62 | 0.14 / 0.01 / 0.13 / 0.73 | **0.34** / 0 / 0.04 / 0.62 | 0.60 / 0 / 0.01 / 0.40 | 0.81 WIN |

  These reproduce B1 §11.5's survivor table (0.61 at +4, NO-USEFUL-GAIN 0.60 at 0, σ 1.76) and the X70 follow-up
  drafts' independent simulation (`../x70_followups_drafts_2026-11-17/power_fresh.log`: 3.28, 0.61, 0.60). By symmetry a true
  −4 pp reads LOSS with the probabilities of the WIN column. **In words: a fresh cell finds a +6 pp lever nine times
  in ten, a +4 pp lever six times in ten, and a +2 pp lever one time in six; a +2 pp lever reads UNDECIDED about
  seven times in ten.** A lever of the size we have detected at forks (+2 to +4 pp) is likelier UNDECIDED than WIN.
- **σ is the unknown, and it is UNVERIFIED for this frame.** B1 measured 1.76 pp on fresh 15M runs at `ent_coef`
  0.05 (B1 §11.1); nothing has measured it at 50M or 100M at 0.0125. **UNVERIFIED:** every σ column above. B1's 50M
  look (expected ≈ 2026-11-18) prints each arm's within-arm variance of row means; the first card registered after it
  records the banked cell's at 50M by a dated amendment before its own 50M look and keeps all three columns. The
  1.76 floor stays whatever is measured (a small measured σ must not narrow the interval).
- **What a fresh cell costs.** To 50M: ≈ 28.6 GPU-h narrow, ≈ 39.6 wide; to 100M: ≈ 58.7 / ≈ 81.6 (queue proposal,
  B1's measured 9.55 h and 13.2 h a run per 50M). With no true effect the expected cost is 28.6 + ½ × 30.1 ≈ 43.7
  GPU-h narrow. A look is 45,000 games, ≈ 0.33 GPU-h on the card if B1's ≈ 2 GPU-h for 270,000 games scales
  linearly (**UNVERIFIED:** a fixed engine start does not scale).

## 5. What each outcome does

A WIN is not an adoption: no recipe, architecture, `designs/production_config.json` or baseline moves on a fresh
cell's read. The owner's rule (strong claims need replication at the plateau) holds.

| outcome | action |
|---|---|
| **WIN** at 100M | The lever is a CANDIDATE for the next fresh lineage. The orchestrator decides whether it enters B1's continuation (Q1) or a later lineage; the claim is IN-HOUSE until a Kakuna confirming read exists (queued behind B1's and Q1's on the lanes, B1 Amendment 6 item 6). |
| **LOSS** at 100M | Retired from scratch AS SPECIFIED: this lever, at this width, under this recipe and controller. Never quoted wider. |
| **NO-USEFUL-GAIN** | Not shown at this size (no +4 pp at 100M with three runs). Not carried forward, never "no effect" (B1 §9). |
| **UNDECIDED** | **Default, acting after B1 Amendment 6's 72-hour veto window: the lever is not adopted**, the current recipe stays, and the result is recorded with its interval. No fourth seed is registered. |
| **STOPPED-AT-50M** | Not continued. Reported with its interval; no class (§ 4). |
| **INCONCLUSIVE** | No adoption, no retirement. Whether to replay is the orchestrator's, written before any replay. |

- **Several fresh cells against ONE banked cell share its three seeds.** Their verdicts are correlated: a banked cell
  that drew lucky seeds pulls every lever read against it down. Each verdict stands alone; a ranking of two levers by
  their Δ̂ against the same banked cell is REPORTED, never a class.
- **No multiplicity correction** across cards: each card is its own registration, read two-sided at 90 %.

## 6. The checks before a card's first launch

1. The card's `PREREG.md` (FULL tier) is on main, adopting this frame and naming its lever, banked arm, seeds,
   deciding comparison and any amendment of § 4's σ.
2. § 2 items 1, 2 and 5 at the pin (`checkargs`, the resolved-config diff, the seed scan), re-run that day.
3. `python -m main.launcher --dry-run <argv>` from a shell at nice ≤ 10 with the card visible passes every preflight.
   From an agent's nice-19 shell with `CUDA_VISIBLE_DEVICES=""` it refuses for exactly B1's two reasons, the CUDA
   preflight and the inherited-nice gate, and nothing else (the cards' transcripts).
4. The card's own GPU check, if it has one, is green and recorded.
5. The kit of § 3 is on main, and the banked cell's look files are held (finding F2).

## 7. When the banked cell's pin cannot be matched

The banked cell is a control ONLY at its own pin, and the pin is not negotiable: an arm that cannot launch at
`15136b616`, or that would train other code than it, does not read against B1's cell. This covers:

- the lever's flag is missing at the pin, or it exists but needs a fix landed after it;
- main's launcher can no longer launch that pin (a torch or environment change, a refused preflight), or the run's
  torch would differ from the banked runs' `2.8.0+cu126`;
- the banked cell does not reach the look (§ 2 item 7), or loses a run beyond B1 §10's one replacement.

**Then the arm BUYS ITS OWN CONTROL:** three control runs typing the banked arm's argv at the new pin (new seeds, a new
scan), launched round-robin with the arm's three (control, arm, control, arm, ...), read with this frame's statistic
and rules at that pin. Price: +28.6 GPU-h narrow or +39.6 wide to 50M, about twice that to 100M. A cross between the
new cell and B1's banked cell across the two pins may be played and REPORTED; it never decides. "Only files outside
the trainer changed" is not an exemption: a re-pin that reads against the old cell would be the registrar's
judgment, which this frame does not allow.

## 8. What the frame cannot say

- **It reads at 100M from scratch, not at the plateau.** A lever that pays only late (a slower starter) can read
  NO-USEFUL-GAIN, UNDECIDED or STOPPED-AT-50M here and still win at the plateau. None of those is a retirement.
- **It cannot resolve effects below ≈ 4 pp** (§ 4). Showing that a cheaper lever is "no worse" at a useful margin
  needs far more runs: with σ 1.76, a non-inferiority margin of 2 pp at 80 % power needs about ten runs an arm
  (√(2σ²/n)·(1.7 + 0.84) ≈ 2 gives n ≈ 10; arithmetic, not simulated).
- **Time is a confound only through the box.** The fresh cell trains weeks after the banked one on the same code and
  card; anything that moves strength between the two (a driver update, a different core split, contention that
  changes nothing deterministic) is not modelled. § 2 item 4 records what can be recorded.
- **The Kakuna meter is not part of the frame.** A WIN is in-house only until a separately registered Kakuna read
  confirms it.
- **Realized entropy is an outcome, never part of a dose.** Both cells type `--ent-coef 0.0125`. Fresh B1 runs at
  that coefficient settle at 0.56–0.61 nats against ≈ 0.68 for P\* forks at the same coefficient (X70 follow-up
  drafts, `271a0c4b6`: B1 C's three runs end at 0.558–0.587 at 50M,
  [`b1_entropy_read.log`](../x70_followups_drafts_2026-11-17/b1_entropy_read.log); the fork figure is theirs, not
  re-read here). A fresh cell is
  compared with a fresh cell, so the fork level never enters; a lever that moves realized entropy (Q5 can: fewer
  passes per batch sharpen the policy less per update) is reported with it, and neither card matches on it. Main has
  no entropy-schedule flag, so no card can hold realized entropy by a schedule without a build.

## 9. Records and reproduction

- The 100M power table was computed 2026-11-17 in the B1 folder with B1's committed helpers (no new script):

  ```bash
  cd designs/research_state/measurements/boundary1_scratch_2026-11-09
  PYTHONPATH=<checkout>/src:. CUDA_VISIBLE_DEVICES="" python - <<'EOF'
  import numpy as np, power_matrix as P
  from main.b1 import cross as X
  rng = np.random.default_rng(20261117); n = 3
  for s in (1.76, 2.5, 3.1):
      for d in (0.0, 2.0, 4.0, 6.0, 8.0):
          f = P.simulate(rng, n, s, P.cell_sd("N"), {"C": 0.0, "W": d, "Br": 0.0, "WB": 0.0})
          dd, half = P.simple_np(f, n, "W", "C"); c = P.classes_100(dd, half); P.check_simple(f, n, "W", "C", c)
          print(s, d, np.median(half), [P.frac(c, k) for k in (X.WIN, X.LOSS, X.NO_USEFUL_GAIN, X.UNDECIDED)])
  print("mismatch", P.MISMATCH)
  EOF
  ```
  The same loop on `P.cell_sd("forum72")` gives F (median half-width 3.46 / 4.35 / 5.21; P(WIN) at +4 of 0.57 / 0.41
  / 0.33), REPORTED.
- The 50M table is the closed form of § 4 (`scipy.stats.norm`), no simulation.

## 10. FINDINGS

- **F1. No registered rule applied B1's statistic to one arm a side, and no kit could play or read a fresh-vs-banked
  cross** (§ 3). BUILT 2026-11-17: the rule `b1_fresh_cell` (`main.b1.fresh_rule`) and the kit `main.b1.fresh`, with
  tests under the routine gate (§ 3 holds the commands).
- **F2. Nothing kept the banked cell's checkpoints past B1 §9** (≈ 11-27 by the queue proposal), while a fresh
  cell's 100M look is weeks later (Segment C, to 12-16). § 2 item 6 now states the rule, "never thin these"; it binds
  once the first card registers, and until then B1 §4's hold covers them. **UNVERIFIED:** whether any thinning is
  scheduled before then.
- **F3. B1's simple effect is REPORTED-only inside B1** (`read_survivor` never classes it). `b1_fresh_cell` classes
  the same statistic with B1's `classify`; `fresh_rule_test.py` holds a classed 3 × 3 read worked by hand.
- **F4. Main has no entropy-schedule flag** (the orchestrator's input from the X70 follow-up drafts). Neither card
  needs one; § 8 says why realized entropy is reported, not matched.
