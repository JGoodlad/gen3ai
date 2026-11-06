"""X65 ExIt-0: distil the one-ply search S′ into P\\* from S′-LIVE's banked searched decisions
(``designs/endstate/design_x65_exit0_search_distillation.md``; the registration is
``designs/research_state/measurements/x65_exit0_2026-11-05/PREREG.md``).

B1 the label bank (:mod:`.bank`), B2 the fine-tune (:mod:`.student`), B3 the export at the play pin (:mod:`.export`),
B4's delivery meter (:mod:`.delivery`; the read rule is ``main.reads.rules_distill_play``). Research students only: no
run, eval, play or production path imports this package.
"""
