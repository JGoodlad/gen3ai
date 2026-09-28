# Kakuna wiring smoke — X22(a), 2026-09-28

**Verdict: `metamon:Kakuna` loads and plays through `python -m main.anchors`, greedy regime verified
on every decision.** This is a WIRING check (n = 4), not a strength read — the win rate below
carries no information about the goal.

## What was wired

- `designs/ops/anchors.json`: `Kakuna` at **checkpoint 34** (upstream's own `default_checkpoint`,
  `metamon/rl/pretrained.py:879` @ `0a00a759`; the hub carries epochs up to 40), **142,832,563
  params** (counted from the state dict; the peer's own `Checkpoint validated` line agrees).
- Weights: `~/dev/metamon/cache/pretrained_models/.../kakuna/ckpts/policy_weights/policy_epoch_34.pt`,
  571,581,627 bytes fp32, sha256 `7e7474d5b2ec6624afdeb29a7cf1dc5c84817d5aee8ec39c1d35a153b18df26c`.
- 🚨 **Hazard H20 (new, SOP §3):** the first attempt died at build — `superkazam.gin` binds
  `attention_type = @FlashAttention` itself and gin files beat the override dict. The same file
  sets a **(96, 0) sliding window**, so the fix re-binds after the files parse to a
  `WindowedVanillaAttention` with flash-attn's mask. The peer reports carry
  `attention = WindowedVanillaAttention`, `attention_window = [96, 0]`.

## The smoke

```
python -m main.anchors --model models/ai_v14_01_base/final_model.zip --opponent metamon:Kakuna \
    --regime greedy --teamset away --games 4 --device cpu --port 9547 --out <archive>
```

CPU only (`CUDA_VISIBLE_DEVICES=""` on both sides; a learner-battery arm held the GPU), rust
front end, `OMP_NUM_THREADS=1`, box load 21–27 on 16 cores. Our side: `ai_v14_01_base` @ 75,005,952.

| field | value |
|---|---|
| status | OK, `regime_verified_decisions` true, both peers exited cleanly |
| argmax_match_rate | 1.0000 on both halves (157 decisions) |
| result | W1 / L3 (0.250, Wilson [0.046, 0.699]) — **n = 4, not a read** |
| Kakuna per-decision, CPU | **median 78 / 73 ms, p90 86 / 88 ms** (acceptor / challenger half) |
| first-decision warm-up | 18.6 s / 4.2 s (the `max_s`; inflates the means to 355 / 121 ms) |
| peak RSS per peer | **~2.0 GB** (VmHWM 2011 MB, sampled every 2 s — `rss_samples.tsv`) |
| cell wall time | 72 s for 4 games, including two peer builds |

## GPU per-decision estimate (NOT measured)

142.8M params, B = 1, KV-cached decode: ~0.6 GB of fp32 weights read per step is ~1 ms of memory
bandwidth on a 3080 Ti, so the step is **kernel-launch bound** — ~20 transformer/perceiver layers
plus the actor and six critics, a few hundred small kernels — **~10–20 ms/decision** eager.
🚨 The tool cannot run a Metamon peer on GPU today: `peers.py` and `metamon_side.py` both force
`CUDA_VISIBLE_DEVICES=""`, and the metamon env has no flash-attn wheel.

## Files

`summary.json`, `games.jsonl` (4 rows), the two `peer_*_report.json`, `rss_samples.tsv`
(`epoch  pid  VmRSS_kB  VmHWM_kB`). The full cell (logs, Metamon's own CSVs) and the failed first
attempt are at `~/gen3ai_archive/kakuna_wiring_2026-09-28/`.
