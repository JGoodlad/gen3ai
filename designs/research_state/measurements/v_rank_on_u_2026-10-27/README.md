# V-RANK — sizing probe (2026-10-27); the read is NOT registered

The design is [`design_v_rank_on_u.md`](../../../endstate/design_v_rank_on_u.md): does the search's Q̂ rank the
moves of one position as well on unfamiliar teams as on home72, with live P\* vs Kakuna games as the truth. Nothing
here is a result about Kakuna or U. The PREREG waits on the design's builds (its §6).

**`scripts/power_probe.py` → `power_probe.json`** (sha256 `ff847a4c…ee40`). It is a
SELF-REFEREE proxy, used to size the read (design §5.1–§5.2). It reads the maturity battery's plateau bank
(`~/gen3ai_archive/hidden_info_maturity/plateau/`, read-only) through L1's registered loader
(`../search_l1_plateau_2026-10-25/l1_plateau.py`, which checks the bank's table and state-set sha256). The input is
P\*'s 719 states: every legal action played to the end in the true world with P\* on both sides, plus the honest F
worlds' TRUNC-4-K8 values. Run it with the gen3ai env and `PYTHONPATH=<checkout>/src`, at nice 19 on the agent cores.
It takes < 1 CPU-min and 1.1 GB, and plays nothing.

Headline (self-referee, pool teams):
- Q̂'s within-node slope against the 24-seed truth is 0.54 [0.43, 0.64]; with one seed an action it is 0.57, SE 0.089.
- Inside the contenders the slope is 0.30. The true advantage of the contenders Q̂ prefers over π's top is +0.0006
  [−0.015, +0.016].
- So the slope is real while search gains nothing (L1 at P\*: +0.0001).

The `power` block is the design's §5.2 table. One caution on it: the 31 searched Kakuna decisions banked by S-U (3
games, 2 deals) show a Q̂ spread 0.2–0.4× this bank's (design FINDING V-2).
