# The information-lever audit, first read (2026-10-12)

The audit suite of [`design_information_levers.md`](../../../endstate/design_information_levers.md) §10, run on the
honest search world builder the same day it was built.

- **Command:** `python -m main.info_audit --n 8 --worlds 4 --specs off,pilot,oracle,F+M,D50,W90 --out audit_8x4.json`
  (8 decisions of the committed Lane S bank `main.policy_spectrum.bank_test.BANK_V1`, spread over its fixed subset
  order; 4 worlds per decision and spec; one seen-mon screen shared by every spec of a decision).
- **Code:** the builder of the commit that lands this folder (`main.value_search.worlds` / `world_draws` /
  `world_screen`, the gender pad `98146ddb`).
- **Box:** a live GPU training run beside it; the audit ran at `nice -n 19` on three agent cores, so the seconds are
  contended wall time.

**Result:** 41 checks, all PASS (template identity, oracle identity, the strict re-gate, legality, genders,
provenance per spec; the leak reads |z| ≤ 1; the dose reads of the told move levers |z| ≤ 1.2; the log source against the Rust reading over 98 decisions, 0 facts the
source claims that the reading does not show). No decision excluded under any spec. The per-class truth-match table
(the pilot's builder vs the honest one) and the cost table are in the design doc §7.1 / §7.4 (`audit_8x4.json` holds
every count).

## The seed-reuse residual (`seed_reuse_200.json`, same day)

The blind replica's finding 2 (hidden facts that spend a die without printing a line stay visible to a gate that
replays on the true seed), MEASURED by counterfactual flips (`main.info_audit.checks.seed_reuse`; design §7.5).

- **Command:** a driver over `main.info_audit.select(200, min_turn=3, max_turn=40, min_opp_alive=1)` (169 decisions
  qualify) calling `checks.seed_reuse(prepare_root(...))` per decision and `checks.merge` over them; 1,662
  counterfactual replays, ~200 s contended (run twice, identical counts).
- **Result:** a seen mon's Speed ±1 point is rejected for 34 of 674 (decision, seen mon) pairs (5.0 %), in 34 of 169
  decisions (20.1 %; 6 with both neighbours rejected: the truth at a tie). King's Rock / Focus Band on a hidden item
  are rejected in 125 of 318 tests (39 %), but the prior's mass on them is 0.0075 summed (4.7·10⁻⁵ per mon), and no
  mon of the bank's 231 teams holds one. No seen mon had an unrevealed dice-rolling ability. Every decision's
  unflipped control passed.

## A wider honest read (`audit_off_40.json`, same day)

`python -m main.info_audit --n 40 --worlds 2 --specs off --min-turn 3 --max-turn 40 --out audit_off_40.json`, after
the playout-tool check, the Hidden Power IV patterns and the `seed_reuse` / `gender_independence` / `playout_open`
checks landed: 12 checks, all PASS; 39 of 40 decisions built, 1 EXCLUDED (a Tyranitar on a pool spread, 184 HP / 104
Atk / 220 Spe Adamant, all-31 IVs, that the chain did not fit in `MAX_STEPS`); 1,298 s contended.
