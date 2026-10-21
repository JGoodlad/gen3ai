# The OFF-POOL BANK at P\*, and L3 and its decoy test replicated on it (2026-10-20)

**Status: BANK DRAWN, not yet scored.** Registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) (`7642df0f6`; the
registered files' sha256 in [`offpool_registration.json`](offpool_registration.json)). No battle had been played and no
likelihood computed at registration.

**The bank** ([`states_offpool.json`](states_offpool.json), committed before any likelihood):
- **Battles:** 312, all three parts complete. bots: 162 battles in 4.4 min. sentA: 90 in 26.7 min. sentB: 60 in 7.0 min.
- **Gates:** the off-pool gate passed on 312 / 312 battles (every opponent a heldout team, every traced side a pool
  team). Gate ①: 0 failures on 10,395 decisions; the turn decode agrees on all of them.
- **Draw:** 8,047 eligible decisions; every cell filled from the cycle alone, so the extension was not needed. **320
  states (160 snapshot, 160 bot; 64 per k bin)** from 223 battles and 68 of the 72 opponent teams. ids sha256
  `db5edcdc…29f6`, bank content sha256 `18bec65b…3725`.
- **Outcome mix (hazard 2, as declared):** bot stratum 154 wins / 6 losses; snapshot stratum 92 wins / 65 losses /
  3 draws. L3's bank was loss-enriched.

Stage 1c's A4 asks any L3 or 1c claim to replicate on an OFF-POOL opponent bank at the plateau checkpoint before it
steers anything. This unit builds the bank: X26's P\* (280,000,170) plays the plateau cycle's own roster, every
opponent piloting a `heldout72_hpfix` ladder team that X26's lineage never trained against ([`offpool_proof.json`](offpool_proof.json)).
It then runs the two cheapest reads on it: L3 (truth vs honest worlds) and the decoy test with OFF-POOL ladder decoys
(pool memory or real-team coherence?).

| file | what |
|---|---|
| `offpool_common.py` | the constants: models, sentinels, parts, seeds, cells, salts |
| `offpool_refs.py` | `proof` (the off-pool proof, run) and `corpus` (the off-pool decoy corpus, built) |
| `gen_cycle.py`, `run_bank.sh` | the battles: `main.ops.eval_trace_gen` with three patches (§3), three parts, one heavy slot each |
| `make_bank.py` | the maturity battery's draw over the generated cycle (§4) |
| `offpool_run.py`, `run_score.sh` | the scoring: L3's candidates + off-pool decoys under P\* (§5–§6), G3's recheck |
| `offpool_read.py` | the registered read: L3's and the decoy test's registered reads, reused by path and sha-checked (§7–§9) |
| `offpool_test.py` | the tests of the pure parts and the registration digests |

**Checkouts and archive.** Every engine, world and network step runs from a detached checkout of L3's code commit
`8fe4fb89` (`~/.cache/gen3ai/offpool/code`), with `--code`. The archive is `~/gen3ai_archive/search_offpool_bank/`:
`ckpt/`, `traces/`, `bank/`, `census.npz`, `decoy_corpus.json`, `rows.jsonl`, `recheck.jsonl`, and `smoke/` (the
engineering smokes, never read as output).
