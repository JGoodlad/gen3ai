# BUILD ITEM — a batch pins ONCE (queued 2026-09-04, dispatch after the 2×2 freeze lifts)

## The defect

A multi-arm batch launched from a chain script pins **per arm**, not per batch. Every arm carries
`--sync-to-main`, so each one records whatever `main` is at *its* launch instant. A batch that runs
22 hours across four arms therefore has four independent opportunities to inherit a code change —
and the arms of a batch are precisely the things that must not differ.

Observed 2026-09-04: `TC_FUND_A` pinned `0c76e2ee`, `TC_UNF_A` pinned `52ab5914` six hours later,
because commits landed in between. Verified inert *that time*; the mechanism guarantees nothing.

## Why the existing guard did not help

`chain_teacher_content.sh` records `EXPECT_PIN` and compares each arm's recorded `git_hash` against
it — **after the arm's launcher returns**. By then the arm has run to completion on the wrong
commit and the *next* arm has already launched on the same wrong commit. The check annotates
history; it cannot prevent propagation.

## The fix

1. **Resolve the pin ONCE, at batch start**, and record it in the batch's own log.
2. **Launch every arm on that recorded commit.** Either drop `--sync-to-main` inside a batch, or
   keep it and have the chain pass the resolved commit explicitly.
3. **Move the drift check BEFORE each launch.** A mismatch should refuse to start the arm (and say
   so) rather than record a note after it finishes.
4. Optional but cheap: have the chain assert `git rev-parse HEAD` still equals the recorded pin
   before each arm, so a human or peer pushing mid-batch is caught at the boundary.

## Acceptance

A test that simulates a mid-batch commit and asserts the second arm either launches on the recorded
pin or refuses to launch — not that it merely records a mismatch afterwards.

## Second defect from the same root: the "N/4 arms clean" counter

The chain's closing summary counts `tc_failed_arms.txt` entries as failures. Because `EXPECT_PIN`
names the *first arm's* commit rather than the batch's, every arm that correctly ran on the frozen
commit is recorded as a mismatch — so a batch where 4/4 arms produced `final_model.zip` closed with
the line `TEACHER-CONTENT 2x2 COMPLETE — 1/4 arms clean`.

Two things to fix together, since one change addresses both:

- the expectation must belong to the **batch**, resolved once at start (see above), so that
  "matches the batch pin" is the same statement for every arm;
- the summary must count the **artifact gate** (`final_model.zip`) separately from pin conformance,
  and report them as two numbers. "3 arms differ from the recorded pin" and "0 arms failed to
  produce a model" are different facts and must not collapse into one score.

**Acceptance:** a batch in which every arm completes reports 4/4 on the artifact gate regardless of
what the pin bookkeeping says, and any pin mismatch is reported as its own line with the commits
named.
