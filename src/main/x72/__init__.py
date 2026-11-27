"""X72, the swap forks carried toward their plateau: the tools its registration owes.

The registration is ``designs/research_state/measurements/x72_swap_plateau_2026-11-25/PREREG.md`` (§15 lists these
as "NOT built"). Its rule file ``x72_rule.py`` is a FROZEN record: no module here imports it, each restates the
tables it needs, and the tests pin every restated value to the frozen files (the end crosses RUN X70's frozen rule
file for its ``rect_stat``, the statistic §9 registers). Nothing here changes a number, a class or a rule of the
registration.

* :mod:`main.x72.finals`: the ARCHIVE COPIES of the round finals (§3): the copy step with its ``finals.json``, and
  the sha256-and-record verification every cross and read goes through.
* :mod:`main.x72.cross`: the round cross (§5): plan, driver and reader; its ``h.json`` is the input of
  ``x72_rule.py decide-round``. With ``--end``, §9's two REPORTED end crosses (the read finals against the banked
  swap finals and the eight C0 finals): a ``report.json``, no class.
* :mod:`main.x72.floor`: the entropy floor's reader (§7): marks, the statistic at each, the comparison of
  ``x72_rule.py floor``.
* :mod:`main.x72.probe`: the exploitability probe at a FIFTH target (§10): X71's rule and reader with the read
  final of lineage 1 added, X71's banked cells reused.
"""
