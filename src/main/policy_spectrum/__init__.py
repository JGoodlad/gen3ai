"""The POLICY-SPECTRUM instrument (M5 Lane S, owner 2026-09-29).

Save ≥ 10,000 turns ONCE, then read any policy on the SAME turns: how much probability goes to its
own 1st, 2nd, … nth choice, per stratum (game phase, forced switch vs free choice, legal-action
count, opponent class, our team, the move categories of the legal actions).

- :mod:`.bank`       the TURN BANK: re-encodable inputs (sim input logs + a decision index), stamped;
                     gate ① (re-encoding reproduces the recorded obs byte-equal) runs at build.
- :mod:`.replay`     input log → rows, masks, tokens through the Rust core (``core_events --obs``).
- :mod:`.categories` attack / status / setup / hazard / recovery / switch for a legal action.
- :mod:`.spectrum`   the rank-mass spectrum, entropy, per-category mass; battle-clustered intervals.
- :mod:`.reader`     any checkpoint ``.zip`` on the whole bank, forward passes only, CPU.
- :mod:`.report`     several reads side by side, with paired deltas on the same turns.

The committed v1 bank: ``designs/research_state/measurements/m5_laneS/bank_v1/``; progress, the
baseline read and the gate ④ / ⑤ specification: ``designs/research_state/measurements/m5_laneS/PROGRESS.md``.
"""
