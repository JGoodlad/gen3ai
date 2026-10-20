# The OFF-POOL BANK at P\*, and L3 and its decoy test replicated on it (2026-10-20)

**Status: REGISTERED** ([`PREREGISTRATION.md`](PREREGISTRATION.md); the registered files' sha256 in
[`offpool_registration.json`](offpool_registration.json)). No battle of the bank had been played and no likelihood of a
bank state computed at registration.

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
