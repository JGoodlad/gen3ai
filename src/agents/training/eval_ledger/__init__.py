"""THE EVAL COUNT LEDGER (v2) — ``designs/endstate/design_evaluation.md`` §0b, built by eval unit U1.

One append-only, archive-level ledger of COUNTS (one row per batch x matchup), its three companion streams
(``requests/`` — the claim queue, ``decisions/``, ``references/``), and ONE declared way to read it.

=================  ===========================================================================================
module             what
=================  ===========================================================================================
``schema``         the v2 row, the v1 row and its deterministic upgrade on read, the companion records, the
                   closed lists, the batch key and seed block (uniqueness), the outcome digest
``store``          ``<archive>/_ledger/`` and its layout, the one file lock, the append-only files, the scans,
                   closing (gzipping) a shard, ``close-stale``
``queue``          the request queue as a deterministic fold of the requests stream (``fold`` is built of
                   ``apply_event``, the one implementation of what an event means); the void rule
``incremental``    reading an append-only stream from a per-file cursor; the SQLite plumbing the two indexes share
``event_index``    the persisted INCREMENTAL fold of ``requests/`` (``.ledger_index/events.sqlite3``): a claim or an
                   append costs O(new events + its request's records), not O(archive) (F-ED-22)
``row_index``      the persisted index of the row shards: a read of ONE request or family costs its rows (F-ED-22)
``writer``         ``LedgerWriter``: requests, families, CLAIMS, claimed row appends, decisions, references
``reader``         ``ReaderDecl`` / ``RegimeFilter`` / ``read`` — every consumer declares what it reads
``cells``          per-cell INCONCLUSIVE, a family read across its looks, pair-level (conditional) estimates
``audit``          ``audit`` / ``verify`` / ``show`` (``python -m main.eval_ledger``); ``audit`` also checks the indexes
=================  ===========================================================================================

``src/eval_ledger_reader_gate_test.py`` is the static gate: every module outside this package that reads the
ledger does so through :func:`read` with a spelled-out :class:`ReaderDecl`, and the closed lists equal §0b.2's
tables. Pure stdlib (plus ``agents.training.mirrored_pairs``): every producer and reader imports it.
"""
from __future__ import annotations

from agents.training.eval_ledger.cells import Cell, InferenceScopeError, cells, looks, pooled_pairs
from agents.training.eval_ledger.reader import (ALL_PURPOSES, LedgerRead, MixedRegimeError, ReaderDecl,
                                                ReaderDeclError, RegimeFilter, read, read_by_regime,
                                                read_decisions)
from agents.training.eval_ledger.schema import (BOT_NATIVE_TEMP, DECISION_KINDS, FLAGS, GROUP_SEQUENTIAL_KINDS,
                                                PLAYER_KINDS, PROTOCOLS, PURPOSES, PURPOSES_V1, REQUEST_KINDS,
                                                SCHEMA, SCHEMA_V1, SEAT_RULES, LedgerSchemaError, as_v2, batch_key,
                                                check_row, outcome_digest, regime_id, seed_key, team_id, team_set_id,
                                                upgrade_v1, utc_now, validate_row, validate_row_v1, with_regime_id)
from agents.training.eval_ledger.store import (LedgerLockTimeout, LedgerPathError, archive_ledger_root,
                                               check_write_root)
from agents.training.eval_ledger.writer import (AlreadyRecordedError, ClaimHeldError, ClaimVoidedError,
                                                DecisionExistsError, DuplicateBatchError, LedgerClaimError,
                                                LedgerWriter, RequestSpecError)

__all__ = [
    "ALL_PURPOSES", "AlreadyRecordedError", "BOT_NATIVE_TEMP", "Cell", "ClaimHeldError", "ClaimVoidedError",
    "DECISION_KINDS", "DecisionExistsError", "DuplicateBatchError", "FLAGS", "GROUP_SEQUENTIAL_KINDS",
    "InferenceScopeError",
    "LedgerClaimError", "LedgerLockTimeout", "LedgerPathError", "LedgerRead", "LedgerSchemaError", "LedgerWriter",
    "MixedRegimeError", "PLAYER_KINDS", "PROTOCOLS", "PURPOSES", "PURPOSES_V1", "REQUEST_KINDS", "ReaderDecl",
    "ReaderDeclError", "RegimeFilter", "RequestSpecError", "SCHEMA", "SCHEMA_V1", "SEAT_RULES", "archive_ledger_root",
    "as_v2", "batch_key", "cells", "check_row", "check_write_root", "looks", "outcome_digest", "pooled_pairs", "read", "read_by_regime",
    "read_decisions",
    "regime_id", "seed_key", "team_id", "team_set_id", "upgrade_v1", "utc_now", "validate_row", "validate_row_v1",
    "with_regime_id",
]
