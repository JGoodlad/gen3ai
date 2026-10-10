# `designs/observation/` — the observation layer's topic docs

Detail lifted OUT of [`src/agents/observation/CLAUDE.md`](../../src/agents/observation/CLAUDE.md) on 2026-10-10.
Each doc OWNS its subject and is ALWAYS-CURRENT like that leaf: update it in the same pass as the code.

What the observation IS (the block table, the per-mon slot, the event-record schema) is
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) §1 — it wins any disagreement. The Rust encoder that WRITES the row
(its gates, the layout ownership, the wire, the benchmark) is [`../rust_sim/encoder.md`](../rust_sim/encoder.md); its
trackers are [`../rust_sim/trackers.md`](../rust_sim/trackers.md).

| Doc | Holds |
|---|---|
| [`per_block_reference.md`](per_block_reference.md) | what each field of the row MEANS and where it is sourced: the per-mon slot (recency, protect odds, last action, toxic stage, sleep-wake belief, move slot, spread), the event window and `EventCol`, the OBS-FACTS block, the board (reactive) block and the deadline clock, `active_req_moves` and the two move orders, the deleted move-effect / incoming-damage blocks, the archived TurnDelta slot and its embedded-ID manifest |
| [`volatile_vocabulary.md`](volatile_vocabulary.md) | the source-derived volatile class (`gen3_effect_sources.py`), the classification (`GEN3_VOLATILE_TO_SLOT`, `NOT_A_VOLATILE`, the field sports), its gates, and the poke-env reading findings it surfaced (history) |
| [`typing.md`](typing.md) | the package's mypy scope and the obs layer's typing idioms |

Frozen snapshot of the leaf before the move: `../research_state/claude_md_archive/src_agents_observation_CLAUDE_2026-10-10.md`.
