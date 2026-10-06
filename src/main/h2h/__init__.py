"""``python -m main.h2h`` — the CHECKPOINT-VS-CHECKPOINT mirrored head-to-head meter (X5 design §7, P0 / U0).

  python -m main.h2h play --player <ckpt.zip|run dir|run@step> --opponent <...> --pairs 5000 [--out <root>] \\
        [--label <run label>] [--seed 0] [--schedule-key K] [--batch-pairs 500] [--purpose audit] \\
        [--request ID] [--family F] [--request-kind K] \\
        [--device cpu|cuda] [--backend eager|graph] [--n-envs 64] [--threads 4] [--torch-threads 4] \\
        [--oracle-reveal-mode off|one_sided|both_sided]
  python -m main.h2h play-many (--cells <cells.json> | --players P… --opponents O…) --pairs N [the play flags]
  python -m main.h2h read [<root>] [--json] [--regime ID]

``play`` plays N MIRRORED team pairs of the player against the opponent on the Rust eval core in the GREEDY eval
regime (module ``play`` says exactly what is mirrored, what is not, and how a re-run reproduces), appends one
§0b COUNT row per batch, under a claim for a request, to the eval ledger (``agents.training.eval_ledger``, v2):
by default the run archive's ``<archive>/_ledger/``, else the root ``--out`` names outside ``models/``. It prints
the pooled win rate with its PAIR-clustered 95 % interval. ``read`` pools a ledger's h2h rows per edge, one regime
at a time, through a declared read (a legacy flat directory of v1 rows is read upgraded).
``play-many`` (module ``many``) plays MANY cells on ONE engine (weights loaded into the declared slots per cell, every
load byte-checked and parity-gated), each cell exactly as ``play`` plays it — the same games, the same rows. An engine
serves up to TWO architectures (module ``arch``: one slot group each, the X5 cross), so ``play`` and ``play-many`` both
play a ``fixed_mass``-vs-``blob`` cell. An ORACLE checkpoint plays only under a reveal mode (module ``reveal``: the
PER-SIDE oracle reveal, one-sided clairvoyance or both-sided, each its own eval protocol stamped on every row). Run from the repo root (the team pool is read cwd-relative; elsewhere refused). ``runfloor`` (module) turns a round-robin of edges into the run-to-run SD on this scale. ``cross`` (module) is the
X5 A/B's REGISTERED seed × seed cross statistic and decision rule (§7.4: Δ̂, V̂, t on 2(n − 1) df, the look
boundaries, rule 8, the outcome table) plus the oracle reads (C, C_full − C_species, the floor, H) and the declared
family reads that build its matrix from the ledger.

Detail: ``designs/training/eval_and_rating.md`` "The checkpoint-vs-checkpoint head-to-head".
"""
