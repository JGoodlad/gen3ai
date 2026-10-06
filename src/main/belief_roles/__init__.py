"""The X5 BELIEF READERS (U7; `designs/endstate/design_x5_belief_tokens.md` §4, §7.4).

Any checkpoint ``.zip`` of EITHER X5 arm (``--belief-tokens blob`` or ``fixed_mass``), read on the
fixed, committed M5 Lane S bank (20,712 re-encodable decisions, both full teams per battle), CPU
forward passes only; the output never goes under ``models/``.

- :mod:`.roles`     the pre-registered role set ("carries move m", Smogon expected carriers ≥ 0.25) and
                    the R3 substitute pairs — from SMOGON data only (never the pool).
- :mod:`.bank_rows` the bank re-encoded through this checkout's Rust core + every truth: the opponent's
                    true team and movesets, its realised action on the COMMON EVENT SPACE, on-/off-pool.
- :mod:`.forward`   one checkpoint (THE strict loader) → per-decision presence for three columns (the
                    arm's own, the Smogon prior), the intent probability of the realised event, role masses.
- :mod:`.eset`      Amendment 3(b): blob's named set E_row (a fixed_mass run is scored on EVERY blob
                    run of the look, ``read --reference``, its value the mean) and each arm's full distribution on the dense event space.
- :mod:`.metrics`   the purpose metrics (the ADOPTION GATE ``intent_logloss_conditional`` — renormalised
                    over E_row — with its coverage; the as-built intent log loss + its MISS column, descriptive;
                    presence Brier / log score,
                    OTHER calibration R4) and the role reads R1–R3, on-pool primary; ``per_run`` = one
                    value per run.
- :mod:`.infer`     §7.4's across-seed two-sample t over seeds, against a boundary the caller passes.

CLI: ``python -m main.belief_roles roles | read | infer`` (see ``__main__``).
"""
