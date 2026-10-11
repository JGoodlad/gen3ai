"""ONE-PLY VALUE SEARCH over determinized worlds — offline / evaluation use only.

    from main.value_search.search import SearchConfig, search_decision
    from main.value_search.playout import LeafEngine
    from main.value_search.worlds import WorldSampler

    engine = LeafEngine(forward)               # forward(rows, masks) -> [k, 12]: logits + V (win prob)
    res = search_decision(engine, log, at, side, root_row, root_mask, SearchConfig(per_action=4, depth=4),
                          key="<decision id>", sampler=WorldSampler())
    res.action, res.policy_target, res.value_target

Modules: ``rules`` (Gumbel's prior-aware selection, sequential halving, the completed-Q policy target;
pure numpy), ``worlds`` (the honest-information world sampler: Smogon prior + a byte-equal observation
replay gate), ``playout`` (the lockstep leaf engine: greedy playouts, optionally truncated and scored by
the critic), ``search`` (``search_decision``). Design, measurements and the bootstrap plan:
`designs/endstate/design_value_discrimination.md`.
"""
