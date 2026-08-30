# Coverage / teams-per-teacher breadth — the flywheel's last standing supply axis

**Status:** 🟡 **ACTIVE — the era's main line** · **Ledger:** `e4c13a9` (rev-2 capstone) · `d4d551d`
(probe B) · `61608ac` (rev-3 admission / the ceiling) · `ade78c1` (rev-3 recap / the treadmill) ·
`36fc4af` (the dilution correction) · `91d5125` (the v8 narrative of record)

One-line claim: *the exploiter→distill flywheel transfers real, teacher-specific content, but only
where headroom exists and only to the teams it covers — so the binding constraint is COVERAGE
(and the breadth-per-teacher that makes broad coverage affordable), not teacher quality, budget,
dose, curriculum or ecology, all of which are now measured dead.*

## Known (cleared the honesty gates)

- **Content transfers, and it is teacher-SPECIFIC.** The ZapDug natural experiment (`d4d551d`) —
  team `eccfe630` pinned by two teachers but taught by only one, because `_distill_mask()` breaks
  on first match — gives a difference-in-differences of **ACTION +0.0400 SIG · KL +0.0408 SIG ·
  CTRL −0.0050 n.s.**: **≥+4.0pp of teacher-specific content**, ~48% of the on-slice gain, and a
  LOWER bound because shared content cancels in a DiD. *The flywheel has demonstrably transferred
  content at least once.*
- **ALIGNMENT ≠ BENEFIT** (`d4d551d`). R2-KL absorbed the MOST teacher shift (0.492 vs ACTION's
  0.188, below even the no-teacher control's 0.347) and finished **4.8pp behind** R2-ACTION.
  Copying the teacher's DIRECTION is what hurts; copying its DECISION is what pays. The same
  ordering appears in the dark-knowledge decomposition (`92ad277`): benefit is monotone in how
  *little* tail shape a form copies (+2.6 full-KL / +3.4 top-3 / +7.4 action), sign-reversed
  against dose.
- **The TEACHER CEILING (`61608ac`).** Six teachers converge to a **~0.6881 [0.672, 0.704]**
  set-mean, INVARIANT to per-team budget (1.5M vs 2.5M: **+0.0019, z=0.16**) and to target start
  (0.46–0.61). Target rose +0.0587 while extraction fell −0.0569 — the same number. **Extraction
  is headroom to a fixed ceiling, never a teacher property. The budget law is DISSOLVED**, and its
  celebrated prospective confirmation (rev-2's sd 0.0098 cluster across five teachers) was constant
  headroom in disguise. Both registrations were ill-posed; both authors own it.
- **Transfer is LOCAL and REPLICATED (`ade78c1`).** rev-3's absolute improvement bar MISSED
  (R3-ACTION +0.0174 z=1.29 · HI +0.0211 z=1.57 vs rev-1, no optional stopping) **while +6pp on
  the 3 coverage teams cleared z=2.57 and replicated at a second dose.** The discriminator is
  headroom: coverage teams 25.7pp vs meter teams 7.7pp. The bar had been pointed at mined-out
  teams — which the ceiling account predicts.
- **THE TREADMILL (`ade78c1`).** R2-ACTION **lost −5.9pp (z=2.5) on untaught coverage teams** while
  gaining on the taught ones; rev-3 then recovered exactly that. Each narrow fold damages the
  uncovered and the next revolution repairs the last. This makes breadth an **anti-treadmill**
  lever, not merely an adder.
- **The steering equation (`36fc4af`), and why the h2h can't see any of this.** General gain ≈
  per-team gain × coverage fraction × retention. Taught teams are 9–12 of 719 and drawn ~1.7% of
  games per side, so **even COMPLETE transfer of +8pp/team moves full-pool h2h by ~0.1–0.2pp** —
  the measured 0.4717 [0.432, 0.512] is an arithmetic non-event, useful only as a ≥4pp regression
  guard. Asking h2h > 0.5 would be requiring the impossible.
- **Plain continuation REDISTRIBUTES; it does not decay** (`61608ac` §6). R2-PLAIN is −2.1pp on the
  meter yet **+1.5pp (n.s.) at free draws** — so the fold's "anchor" value re-reads as *retention of
  meter-team competence*, the R2-CTRL −5.8pp anomaly shrinks further, and R2-PLAIN-LOWLR was demoted
  (the overshoot question survives; the general-decay emergency does not).
- **Teachers genuinely diverge** (`61608ac` §5): the controlled discriminator is **+0.2058** — a
  different slice costs 36 points of agreement where SGD noise costs 16 — so `--distill-team-bias`
  is what makes a 6-teacher fold coherent at all.

## The seven DEAD accounts of v8's +69 (each with what killed it)

The ledger's own count (`91d5125`): six settled with measurements, one interim.

| Account | Killed by | Number |
|---|---|---|
| Parent RIGIDITY / plasticity | `181c1d5` | P1 & P3 **refuted-opposite**: v8's 277M parent Lyle **1.154** (no capacity loss) vs plastic rev-1 **0.948**; v8's deltas were TRUNK-heavy 0.47 vs 0.28 |
| Dark-knowledge TAILS | `92ad277`, `2122878` | Inter-fork tail cosine **0.327** vs a no-fork control's **0.306** — dark NOISE, measured. And v8's tails are not special: **0.349 vs 0.344** like-for-like, CI [−0.021, +0.033] |
| BREADTH → differentiation | `673a694` | Slope **+0.0003 ± 0.0013 (z=0.23)** over a 2/3/4/9-team recipe-controlled ladder |
| Opponent CURRICULUM (C1) | F6-CURR, in `ade78c1` | **NULL, z=−1.40** (power rules out >4.5pp only) — the then-top-ranked candidate |
| BUDGET per team | `61608ac` | **+0.0019, z=0.16** across 1.5M→2.5M |
| DOSE | `ade78c1` | R3-ACTION-HI at 0.35 bought nothing for 4.5M steps |
| ECOLOGY / adaptive team-PFSP | probe P interim (P2) | The adaptive-repair signature is not in v8's artifacts. **Interim** — P-final and its off-lineage arm are still running; this is the row most likely to move |

## Not-known

- **Does breadth-per-teacher drive TRANSFER?** Probe A killed breadth for *differentiation*
  (a different metric), and that tension is the reconciliation bet rev-4 adjudicates. This is the
  last untested supply axis and the axis v8 differs on most (**23 distinct teams via 3 teachers at
  10/3/10** vs rev-3's **12 via 6 at 2**, zero team overlap).
- **Is the treadmill intrinsic or manufactured?** Our folds concentrate a FIXED team-bias 0.4 on the
  taught slices — actively manufacturing redistribution — while v8 ran ADAPTIVE one-sided team-PFSP
  across its whole span. Probe P's P1/P2 decide whether the missing ingredient is a repair-aware
  bias or whether coverage alone is the lever.
- **rev-3's OWN untaught pull-down** — never measured; probe Q is the third point of that series.
- **Whether "v8 worked better" is even true.** Established: v8 was structured differently.
  NOT established: that it worked better — different lineage, different meter (`ade78c1`).
- Whether the ceiling itself can be lifted. **F6-CURR is the first ceiling manipulation and it
  already ran** (`61608ac` §9) — its absolute-vs-0.69 row is requested before any new ceiling
  experiment is designed.

## Pros

- The only lever with a *demonstrated* content transfer (DiD ≥+4.0pp, confound-free) and a
  *durable* prior payoff (retention ~76%, `project_distill_retention_ablation`).
- Breadth is a pure cost knob under the ceiling result — extraction size tracks steps-per-team, so a
  broad teacher is not a weaker teacher, just a differently-spent one.
- It is the only lever that addresses the treadmill; every other supply knob is measured flat.
- The instruments already exist: `main.exploitability` prints per-class headroom, the coverage board
  enumerates the legal universe, and archetype labels give the spread constraint.

## Cons

- **The road is long by arithmetic.** At 12/719 per revolution and full transfer, general strength
  needs ≈60 revolutions. Usage-weighting shortens it only if taught teams are common archetypes.
- The improvement bar has now been MISSED once at the absolute standard, with the local gain real —
  a pattern that can absorb a lot of GPU while looking almost-positive.
- The exploitability curve's first cross-generation reading is **FLAT** (+0.0185 [−0.0100, +0.0470],
  `4867537`). Consistent with the ceiling account, and also exactly what "no progress" looks like.
- Coverage costs teacher-training compute linearly, and the ceiling means each teacher's *product*
  is capped — so the lever buys breadth, never depth.
- Probe A's flat differentiation is a live tension with the breadth-for-transfer bet, not a
  resolved one.

## Next test

**rev-4, the shape discriminator** (`ade78c1` §5): FIXED total compute, **3 teachers × 8 teams vs
the 6 × 2 shape**, teams drawn from the COVERAGE class ("mine where the ore is"), reading **teacher
ABSOLUTE + the coverage cut — never extraction** (extraction is headroom and would mislead by
construction). Gate: broad teachers must reach the same ~0.69 ceiling with **reduced or reversed**
robbery on the untaught. Pass ⇒ the 40-team revolution launches in the winning shape with teams
selected by headroom; fail ⇒ coverage alone is the lever and the fold's bias design is next.

Two cheap inputs land first and can invalidate the framing: **R3-SELF** (if the self-anchor
reproduces the +6pp, the gain is folding-plus-steps and not teacher content — *nothing scales until
that reads*) and **probe Q** (rev-3's own untaught pull-down; ≈0-or-positive refutes both the
share-constant and content-externality models and reopens the mechanism).
