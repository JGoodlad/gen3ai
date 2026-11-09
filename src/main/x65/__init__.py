"""X65 ExIt-0 and X66 ExIt-0H: distil the one-ply search into P\\* from a searched read's banked decisions
(``designs/endstate/design_x65_exit0_search_distillation.md``; the registrations are
``designs/research_state/measurements/x65_exit0_2026-11-05/PREREG.md``, S′-LIVE's leaky labels, and
``designs/research_state/measurements/x66_exit0h_2026-11-09/PREREG.md``, S′-HONEST's honest labels: the bank's
``source``, the student's ``recipe``).

B1 the label bank (:mod:`.bank`), B2 the fine-tune (:mod:`.student`), B3 the export at the play pin (:mod:`.export`),
B4's delivery meter (:mod:`.delivery`; the read rule is ``main.reads.rules_distill_play``). Research students only: no
run, eval, play or production path imports this package.
"""
