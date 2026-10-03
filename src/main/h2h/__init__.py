"""``python -m main.h2h`` — the CHECKPOINT-VS-CHECKPOINT mirrored head-to-head meter (X5 design §7, P0 / U0).

  python -m main.h2h play --player <ckpt.zip|run dir|run@step> --opponent <...> --pairs 5000 --out <dir> \\
        [--label <run label>] [--seed 0] [--schedule-key K] [--batch-pairs 500] [--purpose audit] \\
        [--device cpu|cuda] [--backend eager|graph] [--n-envs 64] [--threads 4] [--torch-threads 4]
  python -m main.h2h read <dir> [--json] [--regime ID]

``play`` plays N MIRRORED team pairs of the player against the opponent on the Rust eval core in the GREEDY eval
regime (module ``play`` says exactly what is mirrored, what is not, and how a re-run reproduces), appends one
§0b COUNT row per batch to ``<dir>/ledger.<writer>.jsonl`` (``agents.training.eval_ledger``; REFUSED under
``models/`` until the archive ledger exists) and prints the pooled win rate with its PAIR-clustered 95 %
interval. ``read`` pools a directory's rows per edge — it refuses to mix regimes — and validates every row.
``runfloor`` (module) turns a round-robin of edges into the run-to-run SD on this scale.

Detail: ``designs/training/eval_and_rating.md`` "The checkpoint-vs-checkpoint head-to-head".
"""
