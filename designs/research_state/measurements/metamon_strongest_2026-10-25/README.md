# P\* vs Metamon's `PastaMittens` ensemble: the record (2026-10-25)

The rules are in [`PREREG.md`](PREREG.md), committed before any counted game. This file records how its
preconditions were met; the result is added when the read ends.

## Precondition 1: the two downloaded weights (owner's OK, 2026-10-25)

Plain Hugging Face hub downloads (`huggingface_hub.hf_hub_download`, repo `jakegrigsby/metamon`, revision
`ac8e88e7da8dc2a2b85176d88f81fd8854b1445b`, the snapshot our cache already held) into Metamon's own cache
(`/home/goodlad/dev/metamon/cache/pretrained_models`; not `models/`, not `data/`). The hub stores each file under its
sha256, and both blob names equal the sha256 we computed.

| file | bytes | sha256 |
|---|---:|---|
| `kakuna/ckpts/policy_weights/policy_epoch_30.pt` | 571,581,627 (as predicted) | `0d66a65ea74f1b02a6d3cf540d22a6ae14d77f9bd2d30e568062569415e9adc5` |
| `alakazam/ckpts/policy_weights/policy_epoch_8.pt` | 304,154,601 (inside the 0.2–0.35 GB estimate) | `47a98516df7b346118fa824f9137e6aaaa1428d5a7150af01bf606c1ea935944` |

## Preconditions 2 and 3: the agent entry and the router's regime instrument

`designs/ops/anchors.json` gains `PastaMittens` (preset `dt4_k34_midjudge_switchk28`, sha256 `1b0d180e…0a9b`), and
`metamon_side.py` gains the router instrument (SOP § "An ENSEMBLE agent"). **The first live smoke caught a false
alarm:** a state compare on Python's `random` moved on 1 of 23 router calls (and 1 of 46, 1 of 45 on a rerun). The
moving RNG was always Python's, never torch's or NumPy's. Poke-env's event-loop thread shares that state and draws
while the router computes. The check is now per thread: no Python draw from the router's own thread, and torch and
NumPy states byte-equal. On the 4-game smoke against `bot:heuristic` (`away`), it read **1.0000 on 95 / 95 calls**
and `regime_verified_decisions = true`. A router call takes 0.22–0.23 s at the median and 0.28 s at p90. Peak memory
is 3.9 GB.
