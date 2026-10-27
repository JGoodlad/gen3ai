# Second adversarial review of the all-null strategy and its PR-1, 2026-10-27

**Scope.** [`strategy_if_levers_null_2026-10-26.md`](../../../endstate/strategy_if_levers_null_2026-10-26.md) ("the
strategy") as revised at `7574743d` and edited since (`fd4160d76` N2, `8bebdfe31` plasticity, `8ed8164ea` N4), and
above all **PR-1**, its zero-GPU precision read (§3.1). The question: is PR-1 still right, and ready to register?

**Read for this review.** The first review ([`REVIEW.md`](REVIEW.md), `6fca2c06`) and the revision's scripts
([`../strategy_revision_2026-10-26/`](../strategy_revision_2026-10-26/)). Then everything that landed after the
revision:
- the plasticity check (`8bebdfe31`, NO-LOSS) and N2's identity test (`fd4160d76`, COMPOSITIONAL);
- the piloting-gap forensics, H-MID and their review (`90708bff`);
- the context-generalisation literature (`5d554c239`);
- Kakuna replay determinism (`21fa891f`) and the peer-pack regime split (`b0441623`, SOP H22);
- D3 look 2's addendum, its review and its hybrid pin (`35f60839a`, `71d74e5c1`, `0e32c5002`);
- the designs and registrations for X48 (`fc888c18f`), N4 (`8ed8164ea`), X62 (and its review `d997a1462`) and X64
  (`17b31f98a`);
- X63 ([`design_x63_long_breadth_continuation.md`](../../../endstate/design_x63_long_breadth_continuation.md)),
  read from its branch at `0de47dc24` and landed on `main` as `49fbf75a7` during this review (FINDING 1).
- The live box: `/proc` and every `units.jsonl` under `~/gen3ai_archive`, read-only.

**Method.** One deterministic CPU Monte Carlo, [`power_review2.py`](power_review2.py) →
[`power_review2.log`](power_review2.log). It reuses the revision's simulator unchanged (200,000 draws a cell). It ran
in the short lane (nice 19, cores 5-7 and 13-15, `mem_cap.sh --max-s 400`, peak 0.15 GB). No game was played, and
no GPU was used.

Abbreviations are the strategy's: pp, LB / UB, P\*, C0, F = `forum72`, H, P72, U, σ_s, lane-h, GUESS, ESTIMATE and
MY JUDGEMENT. E1 / E3 are the engines of [`designs/tools/reads.md`](../../../tools/reads.md) §4. **H22** is the
Kakuna team-pack regime `gen3_peer_showdown_pack_v1` (`b0441623`). The **legacy pack** is the regime before it. Box-h
is one wall-hour of the box's whole Kakuna capacity.

---

## Verdict

**PR-1 is NOT READY TO REGISTER as written.** There is one BLOCKER, four MAJORs and seven MINORs.

- **What still holds.** A zero-GPU read of the banked 20M breadth finals, at one engine with new unit seeds, is the
  right instrument for the +1–2 pp question, and pooling is what makes it affordable.
- **What changed.** It is no longer the strategy's first branch.
  - **Its primary estimand is confounded** (B1). D1 plays F inside its own corpus population, and that alone can
    produce a BREADTH-GAIN.
  - **It names no engine or pack regime** (M1). The only E1 pin on disk would leave its C0 bank unusable by every
    read after it.
  - **Its cost adds a third to a half to the box's queued Kakuna demand** (M2).
  - **Its decision value has shrunk** (M3). The literature puts dose before more 20M forks, and X63 is that
    experiment. PR-1's GAIN licenses a 20M stack fork that X63 supersedes. Its KILL does not reach X62 or X63.
- **Re-scoped as PR-1′** (§6): D3 and N1-P against C0 on F at one post-H22 "tree pin", with D1 reported only. At
  equal games it has the same power, or 0.68 at +1.5 pp for 30,000 games. It runs as a CPU filler after X48's read,
  or on a rented CPU box.

## Findings at a glance

| id | rank | finding | fixed in the strategy doc? |
|---|---|---|---|
| B1 | **BLOCKER** | D1's own population carries PR-1's GAIN: BREADTH-GAIN 0.46–0.65 when only D1 moves F (+4 pp, its point estimate) | flagged in §3.1; the re-registration is the owner's (orchestrator's) |
| M1 | MAJOR | No engine and no pack regime are chosen. "The tree's eval engine" is false: X48 registered its own playing commit, and X62 / X63 / X64 play H22. D3's E1 pin is the legacy pack and lacks the H5 / H21 harness fixes | §3.1 corrected |
| M2 | MAJOR | The cost is in a stale unit. 38,000 games are 70–97 h of the box's WHOLE Kakuna capacity (measured 390–545 games/h). The heavy-slot queue is full. Placed first, PR-1 delays X48's read | §0 / §3.1 note |
| M3 | MAJOR | Decision value shrank. GAIN → a 20M stack that X63 supersedes. KILL does not reach X62 / X63, so §5's "no cheap breadth lever remains" is false. The PR-1 → stack chain scores ≈ 0.02 per resource-day, the worst path on the doc's own metric | §5 corrected; §3.2 pointer |
| M4 | MAJOR | The ranking and tree are not the queue the box runs. X48 (production capacity, registered, GPU-next) has no row, and T3 names the N4 toy first. X62 (+ its review's B2 conflict with PR-1's KILL), X63, X64 and LR-F are missing | §3.2 pointer |
| m1 | MINOR | "C0 at √3× an arm minimises the variance" is false for a pooled mean. The optimum is C0 = 3× an arm, which reaches the registered variance at 35,400 games | corrected |
| m2 | MINOR | Replay is exact on one engine, so a reused unit seed is a duplicate game. PR-1's seeds must be disjoint from X48's C0 seeds 0–49 at a shared pin, and every later C0 top-up from PR-1's | — |
| m3 | MINOR | The E1 → E3 "+0.95 pp" F bias is 4 net of 6 discordant games: not distinguishable from 0. But it is 0.6× the KILL bar, and a same-seed re-play on a new engine is ≈ 98.6 % the same games | — |
| m4 | MINOR | Stale lines: FINDINGS 2–3, the header, D3's seed count, "the D3 / N1-P regime" | corrected |
| m5 | MINOR | LR-F (the plateau as the learning-rate controller's) is a sixth explanation, funded and running, absent from §2 | — |
| m6 | MINOR | §4's routing on the piloting gap can now be resolved: the critic branch does not fire | — |
| m7 | MINOR | Whether a v163+ commit loads the `f4b25238`-era finals is UNVERIFIED (no read plays at one yet) | — |

---

## 1. The engine and regime: which, and what each costs and means (M1)

**The facts.**
- **All 18 finals are on disk:** `models/rb_x35_{c0,d1wide,d3mut}_s3501…s3505` and `rb_x35_n1p_s3501…s3503`. Every
  one trained at the X35 pin `f4b25238`, which is E1's game code.
- **One engine is replay-exact.** On one engine a Kakuna unit replays exactly (0 of 2,100 games). Across E1 → E3,
  F differs on 6 of 420 games, by OUR choice after a Destiny Bond or Endure (`kakuna_determinism_2026-10-27/`).
- **The pack regime is machine-checked** since H22: `peer_pack_changed` enters the regime id, and an import across it
  REFUSES on forum72.
- **Every read registered or designed after PR-1 plays H22:**
  - X48: "any `main` commit carrying the rule", C0 s3501–s3503 re-played at unit seeds 0–49;
  - X64: the post-change regime;
  - X63: "one post-`b0441623` engine", preferring PR-1's;
  - X62: C0 "on the X62 read's engine", importing PR-1's C0 if it is byte-identical.

| option | engine · pack · harness | what PR-1 then measures | cost | who can reuse its C0 | verdict |
|---|---|---|---|---|---|
| **(A)** a frozen post-`b0441623` `main` commit, tagged, ideally X48's playing commit | an unbanked engine (≥ the BF4 parity engine; encoder at config v163) · H22 · the current harness (H5 fail-fast, H21, the choice-level guard) | every final plays OFF its training engine. The encoder changes since `f4b25238` (`d8d674b99`'s silent volatile ends, the v163 Hidden Power belief, the consumed-item rule, the eval-mode load, our packer) hit every arm alike. The contrast is valid; the level is "as played today" | no build. One smoke unit per final (m7). No game-for-game check against a bank is possible: any difference from E1 is the engine's | X48 (same commit), X62, X63, X64's regime | **ACCEPTABLE** (the fallback) |
| **(B)** E1 through D3's hybrid `c4a5c37b6` | E1 · legacy pack (Kakuna's `sft431` Gyarados hits Frustration at base power 1, about 1 team in 72) · the `b0441623^` harness, which lacks `e28318c82` (H5 fail-fast, H21: any peer fault voids the unit) and `52ff8cb1f` (liveness from `/proc`), checked by ancestry | the native engine, in the regime of every banked F row. But PR-1 imports none of them, so that buys nothing | no build. Identity shown for D3's units (80 / 80 games) | **nobody downstream.** X48, X62, X63 and X64 are H22, and an import across the pack regime refuses on forum72. The 14,000 C0 games bank for no one, and X62's PR-rule read pays a fresh C0 (187–269 lane-h instead of 126–181) | **REJECT for PR-1** |
| **(C)** a NEW hybrid: E1's game paths (the `f4b25238` bytes, as AM-D3-L2) under a post-`b0441623` harness with the H22 peer pack | E1 · H22 · the current harness | the native engine AND goal (b)'s current regime. It is the only option **verifiable game for game**: against banked E1 units on panels the pack change does not touch (home72, H, P72), and on F units that never draw a changed set | ≈ 0.5 agent-day plus ≈ 1–2 lane-h of identity units. The H22 files (`showdown_pack.js`, `pack_norm.js`, `pack_oracle.js`) live in `src/utils/bridge/`, which AM-D3-L2 pins to `f4b25238`. They must enter on the peer side only, and `playpath_test` must still hold (FINDING 3) | every X35-frame arm (C0, D1, D3, N1-P, LR-F, X48, X62, X63 all train at `f4b25238`); X63 lists it as its option (b). Not the N4 / X64 toys: their builds are at new pins, and they bring their own controls | **PREFERRED** |

**The ruling this review recommends.**
- **One "tree pin", tagged once.** Option (C), built after T69 lands, so that X48's `n4_screen` rule is in its
  harness. If (C) cannot be ready before X48's first K game, use (A) at X48's playing commit.
- **Who plays at it:** X48's read (an addendum before its first game; it registered only "any `main` commit carrying
  the rule"), PR-1′, X62 and X63.
- **(B) never serves PR-1.** It is right for D3 look 2, which imports banked E1 rows. PR-1 imports nothing.

---

## 2. B1 (BLOCKER). D1's own population can carry the pooled estimand

- **The setup.** The registered primary is the mean of three contrasts on F. One of the three arms trained on F's
  population: D1's corpus `w_forum_v1@ABC` is forum-sourced, and F is `forum72`. The panel exclusion removes F's
  exact teams, not their community, sets or near-copies.
- **The evidence it matters.** D1's final put its point gain on its own population: F +4.0, H +2.1, P72 +0.1
  (D1 read README; D3 PREREG line 18). The strategy itself says so (§3.1 meter row) and keeps the leave-D1-out
  contrast as a REPORTED line only.
- **The power when only D1 moves F** (`power_review2.log` §1; the registered 3 × 8,000 + C0 14,000; seeds
  5 / 5 / 3 / 5):

  | D1 / D3 / N1-P on F | BREADTH-GAIN σ_s 0 | σ_s 0.5 | σ_s 1 |
  |---|---:|---:|---:|
  | +2 / 0 / 0 | 0.20 | 0.20 | 0.18 |
  | +3 / 0 / 0 | 0.42 | 0.39 | 0.31 |
  | **+4 / 0 / 0** | **0.65** | **0.60** | **0.46** |

- **Why it is a BLOCKER, not a caveat.** Each class licenses something.
  - **GAIN.** In a world where neither set mutation nor procedural breadth transfers, PR-1 reads BREADTH-GAIN about
    half the time. It then licenses a stack fork whose union includes D1's forum corpus. That stack cannot be read on
    any current panel (strategy FINDING 6).
  - **KILL.** It loses power symmetrically.
  - **So the registered rule cannot say "breadth helps unfamiliar teams"**, the thing goal (b) asks.
- **The fix costs nothing at equal games.** For a pooled mean of k arms against C0, the binomial variance is
  1/(k n_b) + 1/n_c. At fixed total games its minimum is 4/N, whatever k is. Dropping D1 and re-allocating therefore
  loses only seed-level averaging (`power_review2.log` §2, D3 at 5 seeds, N1-P at 3):

  | primary | games | +1.5 pp each: GAIN σ_s 0 / 0.5 / 1 | null: false GAIN / KILL (σ_s 0) |
  |---|---:|---|---|
  | registered, three arms (revision log) | 38,000 | 0.764 / 0.709 / 0.566 | 0.021 / 0.789 |
  | **D3 + N1-P, C0 = 2 × an arm** | **38,000** (2 × 9,500 + 19,000) | **0.781 / 0.703 / 0.522** | 0.019 / 0.808 |
  | D3 + N1-P, C0 = 2 × an arm | 30,000 (2 × 7,500 + 15,000) | 0.680 / 0.620 / 0.482 | 0.019 / 0.714 |

- **Re-register PR-1 with:**
  - **the primary** = D3 and N1-P against C0 on F;
  - **D1 on F** REPORTED as "breadth within the panel's population";
  - **D1's transfer** is already read: D1's final, U +2.10 [−0.92, +5.11].
- **Neither remaining arm is hazarded on F.** N1-P's PREREG §8 hazards H and P72 only. D3's corpus is pool teams
  with Smogon re-draws, the set-level exposure every lever shares by design.
- **The strategy's outcome odds** (BREADTH-GAIN ≈ 0.42) used a prior pooled with D1. For D3 + N1-P alone they are
  lower: ≈ 0.2–0.3 (MY JUDGEMENT; D3's own look-1 F was +0.0 [−6.7, +6.7]).

---

## 3. M2 (MAJOR). The cost, against a saturated box and the queue

- **The stale unit.** The strategy prices PR-1 in lane-h at `--concurrency 1`: 165–238 lane-h, "≈ 7–10 days".
  Reads no longer run that way. N1-P look 1 and H-CTX rc72b run at K = 2, and three to five Kakuna units share the
  two heavy slots.
- **The measured box rate.** Counted Kakuna games across every read (`~/gen3ai_archive/**/units.jsonl`, 20-game units;
  S-U's per-game units counted as one):

  | window | games/h |
  |---|---:|
  | last 6 h | 544 |
  | last 12 h | 520 |
  | last 24 h | 393 |

  **38,000 games is ≈ 70–97 h of the box's WHOLE Kakuna capacity.** Shared fairly with the two or three reads
  running, the strategy's 7–10 days of wall time stands.
- **The box is saturated now.** At 11:20 PDT the heavy-slot queue held 7–8 waiting jobs, and this review's 18-second
  Monte Carlo waited 495 s for a slot (rule 5b: LR-F s3502 is live). Load average was 8.9.
- **The queue already holds ≈ 84,000–122,000 Kakuna games before PR-1** (ESTIMATE; the H-CTX and X59-S remainders
  are not counted):

  | read | Kakuna games |
  |---|---|
  | N1-P look 1 remainder (209 of 486 units counted) | ≈ 5,500 |
  | D3 look 2 (≈ 16 lane-h) | ≈ 2,600–3,700 |
  | H-CTX rc72b (`--hours 72`) and X59-S's pilot (`--hours 160`) | running |
  | X48's K | 6,000 (+ 2,400 home72 if there is room) |
  | X62's read, if registered | 20,000–43,000 |
  | X63 Stage A (66–95 lane-h) | ≈ 11,000–22,000 |
  | X64 | ≈ 39,000 toy games |

  PR-1's 38,000 adds a third to a half to that.
- **What follows:**
  - **Order.** PR-1 must not run ahead of X48's read. The card produces finals faster than the lane reads them (GPU
    plan FINDING 6), and an unread final is GPU work idling.
  - **The rented CPU box** (the strategy's owner question) is now the deciding variable for PR-1, not a convenience.
    No rented CPU has played a Kakuna game, and its price and rate are UNVERIFIED.
  - **B1's 30,000-game form** saves ≈ 15–20 box-h at a power of 0.68.

---

## 4. M3 (MAJOR). The literature and X63: what PR-1 still decides

**What each class now licenses.**
- **BREADTH-GAIN ⇒ the stack: a 20M P\*-fork on the union of forum, d3 and n1.**
  - The literature's recommendation 2 ("put dose before more 20M plateau forks", `context_generalisation_2026-10-27.md`
    §7) says the union should run LONG, not 20M.
  - X63 is that experiment: the same union minus the forum teams, at +100M from X26 @380M. It is designed not to be
    gated on PR-1.
  - So GAIN's licensed action is superseded. Its real use would be to raise X63's prior.
- **KILL ⇒ "breadth at the plateau, ≤ 20M, at these corpora" retired.**
  - Under the dose hypothesis a 20M KILL is EXPECTED even if breadth works at 10⁸ steps. KILL therefore retires 20M
    forks, which the literature already ranks last, not breadth.
  - X63 says KILL does not reach it.
  - X62's review (B2) shows the strategy and X62 route the same likely world in opposite directions. PR-1 tested
    set-novel corpora, while X62 is context recombination.
  - **§5's trigger text "no cheap breadth lever remains" is now false.** X62 (gated on rc72b) and X63 (a DRAFT) are
    cheap breadth levers outside KILL's scope, and X48 (capacity at the plateau) is registered. The trigger's
    routing is the orchestrator's to rule on; the false clause is marked in place.
- **UNDECIDED ⇒ nothing changes.**

**On the doc's own metric.** The stack's ≈ 0.12 per resource-day is CONDITIONAL on PR-1's GAIN.
- The chain PR-1 → stack has P(GAIN) ≈ 0.42 × 0.62 ≈ 0.26 (the doc's own numbers).
- It costs (165–238 + 107 lane-h + 20.5 GPU-h) / 24 ≈ 12–15 resource-days.
- That is **≈ 0.017–0.021 per resource-day**, the worst path in the table (rows 4–9 sit at 0.06–0.09). With B1's
  lower odds it is ≈ 0.01.
- PR-1's rank 1 therefore rests on information value, not on the table.

**The information PR-1 still buys** (each worth a CPU filler; none needs to come first):
- **the 20M point of a dose curve** for set-novel breadth, from P\*. X63's 400M read point has another parent;
- **σ_s at 20M**, the unknown in every PR-rule read. X63's false GAIN rises to 0.057–0.069 at σ 1.5–2 pp;
- **a C0 bank at the tree pin**, which saves X62 ≈ 60–90 lane-h, and only if M1's ruling holds.

**Answer to the brief: yes.** For goal (b), PR-1 is now less valuable than X63, on two conditions:
- X63 survives its own adversarial review. It is a DRAFT with no review of its own, and its 1 v 1 screen
  against a one-seed X26c is fragile under σ_s.
- Its gates read first (H-CTX, N1-P look 1, D3 look 2).

## 5. M4 (MAJOR). The tree is not the queue

- **The strategy's T3** reads "NO-LOSS ⇒ the N4 toy screen, then N4-P".
- **The GPU plan and STATUS** run differently:
  - **X48 is queued right after LR-F s3501:** production capacity at the plateau, registered under N4's one-look
    screen at ≈ 13 GPU-h + 26–38 lane-h;
  - the N4 toy waits on T69.
- **The ranking table (§3.2) has no row for:**
  - X48;
  - X62, whose review `d997a1462` B2 asks for a row and a PR-1 node;
  - X63 (DRAFT);
  - X64 (the literature's recommendation 5);
  - LR-F (m5).
- **Fix:** re-rank in the doc (the orchestrator's call), or mark §3.2 superseded by the GPU plan. This review adds a
  pointer only.

---

## 6. The top three, re-ranked with everything new

MY JUDGEMENT. Each lane is named, because the GPU and the Kakuna lane are separate scarce resources.

1. **X48, then its read at the tree pin.**
   - The capacity branch the plasticity NO-LOSS chose, at production scale.
   - Registered, kit-ready and GPU-next: ≈ 13 GPU-h + 26–38 lane-h, ≈ 1.6–2.1 resource-days, the cheapest new
     read.
   - Its K screen is weak (GO-K ≈ 0.6 at +3 pp). Its S read is strong.
2. **X63 (dose, the union without forum teams, +100M), after its adversarial review and its gates.**
   - The literature's recommendation 2, and the form the stack fork should take.
   - ≈ 20 GPU-h to its screen and 66–95 lane-h at Stage A.
   - Not gated on PR-1.
   - **If rc72b supports H-CTX, X62 (after its review's B1 / B2 fixes) enters level with it.** The literature's
     recommendation 1 is conditional on exactly that read.
3. **PR-1′, the re-scoped precision read.**
   - The primary is D3 + N1-P vs C0 on F, with C0 = 2 × an arm; D1 is REPORTED.
   - 30,000 games (0.68 at +1.5 pp, σ_s 0) or 38,000 (0.78).
   - It plays at the tree pin, with unit seeds disjoint from X48's C0 seeds 0–49.
   - It runs after X48's read in the Kakuna queue, or at once on a rented CPU box if the owner approves.
   - It is information only: σ_s, the 20M dose point and the C0 bank. It gates nothing in the top two.

Out of the top three, with the reason:
- **The N4 toy:** it waits on T69, and X48 asks the production version first.
- **X64:** ≈ 50 GPU-h, a measurement for a curve, with low transfer to production.
- **The stack fork:** superseded by X63.
- **X53-N1:** still behind N1-P's GO.

---

## 7. MINOR

- **m1. The C0 allocation.**
  - √k is the many-to-one optimum (Dunnett: each arm read against C0). For the pooled mean, the optimum is n_c = k ×
    n_b, half the games.
  - The registered 3 × 8,000 + 14,000 has an SE 3.7 % above that optimum. The optimum reaches the same variance at
    35,400 games (`power_review2.log` §3).
  - Corrected in §3.1. The extra C0 may still be wanted for the bank; if so, say so.
- **m2. Unit seeds and determinism.**
  - On one engine a unit replays exactly, so the same (final, unit seed, pin) is the same game.
  - At a shared tree pin, PR-1's C0 seeds must be disjoint from X48's (0–49 for s3501–s3503), and any later C0 top-up
    (X62) disjoint from PR-1's.
  - Conversely, the SAME unit seeds across arms pair the team draws, a free variance reduction that makes the
    binomial model conservative. Register both.
- **m3. The engine bias.**
  - E1 → E3 moved C0's F by +4 wins in 420 (+0.95 pp). That is 4 net of 6 discordant games, a sign test p ≈ 0.2: not
    distinguishable from 0.
  - Bounded by the ≈ 1.4 % divergence, it is still 0.6× PR-1's KILL bar. That is why one engine is non-negotiable.
  - It also means a same-seed re-play on a new engine adds almost no information: new seeds are the read.
- **m4. Stale statements, corrected in place:**
  - the header and FINDINGS 2–3 (the forensics and the engine audit have landed: `main.reads.engine` makes a group
    spanning two engines INCONCLUSIVE, and the pack regime is in the id);
  - D3 is at five seeds (`a9c09711d`, `28711af06`);
  - "the D3 / N1-P regime": they are two engines, both on the legacy pack.
- **m5. A sixth explanation.**
  - LR-F ("is the plateau the learning-rate controller's halving?", STAGED_PLAN §4.1.2) is funded and running, and
    reads before T1 ends.
  - §2's five explanations omit optimisation. A LR-F GAIN would re-price every 20M fork read at the decayed
    learning rate, PR-1's arms included.
- **m6. §4's routing can be resolved.** The forensics and H-MID landed:
  - the gap shows after turn 10, from level positions;
  - team vs set is OPEN until rc72b;
  - V is blind to the premium, as a self-play critic must be (REVIEW M1 of `90708bff`);
  - the literature's recommendation 3: no critic-calibration lever.

  So the "localises to the critic" branch does not fire, and rc72b routes X62.
- **m7. Loading at v163.**
  - No read yet plays at a commit carrying `5aa64372b` (config v163; `main` is at v164 since `c35ffab9e`). All ten
    read worktrees are older.
  - Option (A) needs one smoke unit per final at the chosen commit before registration. Option (C) does not, by
    construction.

---

## 8. What this review changed in the strategy doc (always-current; minimal, cited)

- **The status line:** REVIEW_2's verdict, with a pointer.
- **The header:** the forensics and the engine audit have landed.
- **§3.1:**
  - arms: D3 at five seeds;
  - meter: one pack regime (H22), the D1 hazard flagged as B1;
  - games: the C0 allocation sentence (m1);
  - ONE engine: the peer pack in the id, no engine chosen yet, and never the legacy D3 pin (M1).
- **§3.2:** a pointer to the missing rows (M4).
- **§5:** "no cheap breadth lever remains" marked false (M3).
- **§8:** FINDINGS 2 and 3 rewritten as landed.
- **§10:** a decision-record row.
- **Elsewhere:** STATUS §6 risk 1 and UNDERSTANDING's PR-1 row carry the one-line correction.

## FINDINGS (standing rule 7)

1. **X63 was read from its branch** (`0de47dc24`) and landed as `49fbf75a7` while this review was written. The landed
   diff only adds X62-review notes (near-duplicates, rc72b distance, the home72 floor's rationale); §4 and §6 rest on
   the unchanged scope (no forum teams, not gated on PR-1, one post-H22 engine). It is still a DRAFT with no review.
2. **The box rate is approximate.** Some counted units are not 20 games, and the panels are mixed (rc72b, home72).
   The 390–545 games/h is a box-wide ESTIMATE, not a per-panel F rate.
3. **Option (C) is UNBUILT.** Whether a post-H22 harness runs over the `f4b25238` game paths (`src/utils/bridge/`
   holds both our packer and the H22 peer pack files) is UNVERIFIED. AM-D3-L2 did the same over `b0441623^`'s
   harness, and the drift since then is two days of harness commits.
4. **X59-S's pilot anchor supervisor runs at nice 5** (pid 3240975, observed in `/proc`), not 19. This review did
   not check whether it carries a `GEN3AI_BLOCKING` sanction (rule 5b). Reported, not touched.
5. **No game was played, no GPU used, nothing in `models/` or `~/gen3ai_archive` written.** One Monte Carlo ran in
   the short lane.
