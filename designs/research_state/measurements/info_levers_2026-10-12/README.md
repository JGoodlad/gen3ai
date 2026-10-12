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
