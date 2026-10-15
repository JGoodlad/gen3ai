# The ride-along reader on an END-STATE checkpoint — a plumbing smoke (T32, 2026-10-15)

**What this is.** Evidence for Amendment 2 of [`../PREREGISTRATION.md`](../PREREGISTRATION.md): that the registered
instrument, `python -m main.ridealong_read`, still RUNS at HEAD on a checkpoint of the end-state graph (static
per-mon tokens, trunk depth 3, X5 tokens, move resolution, the bf16 trunk recipe) that carries TRAINED heads. It is a
**plumbing smoke, not a read**: 40 of the bank's 580 battles, one checkpoint, no fresh-heads floor, no verdict. No
number in the JSONs is a result about the heads.

**The command** (reader at `264a35e3`; the `+dirty` stamp in the JSON is this unit's uncommitted TEST edits —
no reader file differed):

```bash
PYTHONPATH=src nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 12 python -m main.ridealong_read \
  --checkpoint models/rb_bf_eb_s2001/checkpoints/checkpoint_9000048_steps.zip=eb_s2001_9M \
  --out <scratch> --limit-battles 40 --no-rnd-choice --threads 3 --workers 1 --archive ''
```

**What it showed.**
- The bank (`m5_laneS/bank_v1`) re-encodes at HEAD's 2845-dim layout: 1,172 decisions / 40 battles in 4.8 s. "obs as
  recorded: False (0/995)" is expected and harmless: the bank was recorded under an older layout, and the reader
  re-encodes every state rather than reading recorded observations.
- The checkpoint (config v155, `rb_bf_eb_s2001` at 9.0M, bf16 trunk) loads at HEAD (config v157) as an inference
  model; its critic passes the win-prob check; its heads are read as `"trained"` with the registered spec (ensemble
  5, RND, A 5, B 5, every variant).
- Every meter family the reader implements produced output: (i) the uncertainty-vs-|V − z| table, (ii) A and the
  policy logits against all three `truth_v2` continuations, (iii) the starved-action spread, the RND variants'
  (a) / (b) / (e) and the series file. (iv) is absent as registered (no own-continuation truth). 50 s on 3 cores;
  peak memory 1.23 GB.
- **Not produced, because NOT BUILT:** R1×R2's paired AUROC difference (ensemble − RND) and R5's Brier(V) −
  Brier(Q_played). The JSON's `opp_plumbing` counts B's finite rows only. Amendment 2 records both as offline builds
  that must land before the first VERDICT read (FINDING A2-F2).

**Files.**
- `eb_s2001_9M.json`, `rnd_variants_series.json`: the reader's output.
- `x26_argv_checkargs_dryrun_264a35e3.log`: `python -m main.checkargs --argv` and `python -m main.launcher --dry-run`
  on X26's argv at `264a35e3` (`GEN3AI_MODELS_DIR` a scratch dir), pasted into Amendment 2 §A2.3.
