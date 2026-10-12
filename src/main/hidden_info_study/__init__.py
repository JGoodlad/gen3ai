"""The hidden-information binding study's HEAD-side instruments (`designs/endstate/design_hidden_info_binding.md`).

- :mod:`.hand_eval`   Foul Play's hand-written leaf evaluator (poke-engine ``gen3/evaluate.rs``, MIT, © 2024
                      pmariglia) ported to read ONLY the searched side's observation row (§6.5, backlog X3).

The measurement scripts that run these at the study's pins live in
``designs/research_state/measurements/hidden_info_binding_2026-10-11/`` and import the module BY FILE PATH
(``hand_eval_pin.py``), so the pinned runs execute this exact source.
"""
