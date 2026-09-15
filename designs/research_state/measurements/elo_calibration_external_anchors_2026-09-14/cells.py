"""THE CELL LIST — the campaign's design, as data.

One row per `python -m main.anchors` invocation. Kept as a module rather than inlined in the
driver so the fit, the README's tables and the driver all read the SAME list, and a cell that was
never run cannot quietly disappear from the analysis.

Node ids follow the ladder's own keying so they can be joined to an existing fit without a
translation table: ``bot:<name>`` (PINNED, from ``data/gen3_bot_elo_anchors.json``),
``snap:<run>@<step>`` (free; a RUN-QUALIFIED snap key, because this campaign fits four runs at
once and a bare step collides across them), ``ext:metamon:<Agent>`` (free — the new nodes).
"""
from __future__ import annotations

MODELS_ROOT = "/home/goodlad/dev/gen3ai/models"

#: The nine PINNED eval bots, in roster order (`agents.training.eval_callback._EVAL_ROSTER`).
BOTS = ["random", "heuristic", "heuristic2", "staller", "staller_v2",
        "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"]

#: The two external anchors. `--opponent` spec -> node id.
ANCHORS = {"metamon:SmallRL": "ext:metamon:SmallRL",
           "metamon:SyntheticRLV2": "ext:metamon:SyntheticRLV2"}

#: (run, step) for every frozen snapshot this campaign plays. Chosen for ERA SPREAD across the
#: loadable archive and because every one already carries historical bot edges in its run's
#: `eval_results.jsonl`, so the external edges attach to a node of the existing anchored fit.
#:
#: 🚨 `ai_v8_03_zarch_control_0718` is ABSENT and cannot be added: its snapshots are config
#: version 44/45 under ARCH_SIGNATURE `gen3_opp_hp_typed_candidates_v1`, below MIGRATION_FLOOR 96,
#: so `load_foreign_opponent` refuses them as PRE-GENERATION. The weakest node here
#: (`ai_v9_29_rev1_0823` @ 2,000,016, refit 1723) stands in for the low end.
SNAPSHOTS = [
    ("ai_v9_29_rev1_0823", 2000016),
    ("ai_v9_29_rev1_0823", 8000016),
    ("ai_v9_29_rev1_0823", 24000000),
    ("ai_v12_11_ladder_ctrl10M", 10000032),
    ("ai_v12_02_winprob_critic", 36000000),
    ("ai_v12_02_winprob_critic", 56000016),
    ("ai_v12_02_winprob_critic", 74000016),
    ("ai_v13_01_flywheel_shaped", 22000032),
    ("ai_v13_01_flywheel_shaped", 48000000),
    ("ai_v13_01_flywheel_shaped", 72000000),
]

GAMES_PER_CELL = 100

#: The ANCHOR-vs-ANCHOR cell. The two anchors are already joined through 19 shared opponents, so
#: this edge is not needed to place them — it is the TRANSITIVITY CHECK on the whole joint fit: if
#: 200 direct games order the two one way and the fit orders them the other, that disagreement is
#: itself the finding, and a fitted ladder that cannot reproduce a 200-game head-to-head is a
#: scalar summary of a non-transitive population.
H2H_GAMES = 200


def snap_node(run: str, step: int) -> str:
    return f"snap:{run}@{step}"


def snapshot_zip(run: str, step: int) -> str:
    return f"{MODELS_ROOT}/{run}/snapshots/snapshot_{step:012d}.zip"


def cells() -> "list[dict]":
    """Every cell, in the order the driver should attempt them.

    BOT cells first on purpose: they are the cheapest (a bot decides in microseconds) and they are
    the ones that place the anchors on the PINNED frame. If the campaign has to stop early, the
    half that survives should be the half that does the anchoring.
    """
    out = []
    for opp, node in ANCHORS.items():
        for bot in BOTS:
            out.append({
                "id": f"bot_{bot}__{node.split(':')[-1]}",
                "kind": "bot",
                "our_node": f"bot:{bot}",
                "their_node": node,
                "argv": ["--our-side", f"bot:{bot}", "--opponent", opp],
            })
    for opp, node in ANCHORS.items():
        for run, step in SNAPSHOTS:
            out.append({
                "id": f"snap_{run}_{step}__{node.split(':')[-1]}",
                "kind": "snapshot",
                "our_node": snap_node(run, step),
                "their_node": node,
                "argv": ["--model", snapshot_zip(run, step), "--model-load", "auto",
                         "--opponent", opp],
            })
    out.append({
        "id": "h2h_SyntheticRLV2__SmallRL",
        "kind": "h2h",
        "our_node": "ext:metamon:SyntheticRLV2",
        "their_node": "ext:metamon:SmallRL",
        "games": H2H_GAMES,
        "argv": ["--our-side", "metamon:SyntheticRLV2", "--opponent", "metamon:SmallRL"],
    })
    return out
