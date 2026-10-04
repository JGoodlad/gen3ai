"""``python -m main.h2h`` — the CHECKPOINT-VS-CHECKPOINT mirrored head-to-head meter (X5 design §7, P0 / U0).

  python -m main.h2h play --player <ckpt.zip|run dir|run@step> --opponent <...> --pairs 5000 [--out <root>] \\
        [--label <run label>] [--seed 0] [--schedule-key K] [--batch-pairs 500] [--purpose audit] \\
        [--request ID] [--family F] [--request-kind K] \\
        [--device cpu|cuda] [--backend eager|graph] [--n-envs 64] [--threads 4] [--torch-threads 4]
  python -m main.h2h play-many (--cells <cells.json> | --players P… --opponents O…) --pairs N [the play flags]
  python -m main.h2h read [<root>] [--json] [--regime ID]

``play`` plays N MIRRORED team pairs of the player against the opponent on the Rust eval core in the GREEDY eval
regime (module ``play`` says exactly what is mirrored, what is not, and how a re-run reproduces), appends one
§0b COUNT row per batch, under a claim for a request, to the eval ledger (``agents.training.eval_ledger``, v2):
by default the run archive's ``<archive>/_ledger/``, else the root ``--out`` names outside ``models/``. It prints
the pooled win rate with its PAIR-clustered 95 % interval. ``read`` pools a ledger's h2h rows per edge, one regime
at a time, through a declared read (a legacy flat directory of v1 rows is read upgraded).
``play-many`` (module ``many``) plays MANY cells on ONE engine (one architecture; weights loaded into the declared slots
per cell, every load byte-checked and parity-gated), each cell exactly as ``play`` plays it — the same games, the same
rows. ``runfloor`` (module) turns a round-robin of edges into the run-to-run SD on this scale.

Detail: ``designs/training/eval_and_rating.md`` "The checkpoint-vs-checkpoint head-to-head".
"""
