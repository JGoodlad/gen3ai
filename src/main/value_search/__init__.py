"""ONE-PLY VALUE SEARCH over determinized worlds — offline / evaluation use only.

    from main.value_search.search import SearchConfig, search_decision
    from main.value_search.playout import LeafEngine
    from main.value_search.worlds import WorldSampler

    engine = LeafEngine(forward)               # forward(rows, masks) -> [k, 12]: logits + V (win prob)
    res = search_decision(engine, log, at, side, root_row, root_mask, SearchConfig(per_action=4, depth=4),
                          key="<decision id>", sampler=WorldSampler())
    res.action, res.policy_target, res.value_target

Modules: ``rules`` (Gumbel's prior-aware selection, sequential halving, the completed-Q policy target, and L1's
anchored PIKL rule ``SearchConfig(rule="pikl")``; pure numpy), ``worlds`` (the world sampler: HONEST by default — every hidden fact drawn from the Smogon prior and
gated on the searched side's byte-equal protocol and rows — with the extra information an INFORMATION-LEVER spec
grants, ``WorldSampler(reveal=...)``; `designs/endstate/design_information_levers.md` §7; ``dice="rolls"`` lets each
damage roll be any roll that prints what the searched side saw and weights the worlds, F1 tier A of
`designs/endstate/design_world_sampler.md` §3.3.1), ``world_draws`` (each
fact's draw under the spec), ``world_screen`` (the seen mons' item / ability / spread posterior draws: prior
rejection, then an observation-guided Metropolis-Hastings chain), ``playout`` (the lockstep leaf engine: greedy
playouts, optionally truncated and scored by the critic), ``search`` (``search_decision``; ``SearchConfig(weights=
"likelihood")`` averages a ``dice="rolls"`` sampler's worlds by their likelihood weights), ``opp_likelihood`` (OFF by
default: each world's likelihood of the opponent's OBSERVED past choices, the network seated as the opponent; candidate
(f) of `designs/endstate/design_world_sampler.md` §4; and stage 1c's IMPORTANCE RESAMPLE, ``resample`` /
``resample_by_opponent``: keep k of a larger proposal set by each world's damage-roll weight × that likelihood, each
kept world carrying its own rolls, ``Worlds.select``; ``SearchConfig(proposals=N)`` turns it on in a search, default
off). Design, measurements and the bootstrap plan: `designs/endstate/design_value_discrimination.md`. The LIVE
player that searches inside a game against an external anchor (S-U) is ``main.anchors.search_side``.
"""
