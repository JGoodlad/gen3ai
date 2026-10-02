"""The eval TRAINEE's teambuilder — one rule for both eval paths, free of either eval worker.

Deletion pass U2: the Rust env core's eval table (``rust_eval.build.eval_builders``) imported this from
the Python-path eval worker (``main.eval_worker``, which loads poke-env players and the local battle
runner). It lives here now; ``main.eval_worker`` imports it from here.
"""
from typing import Any


def build_trainee_tb(cfg: dict, all_teams: Any, sample_teams: Any) -> Any:
    """The TRAINEE's eval teambuilder. When the run pins the trainee to one team
    (``--trainee-team`` → ``cfg['trainee_team_str']``, the raw Showdown export), eval MUST measure
    the model piloting THAT team — the worker used to hardcode the default full-pool builder here,
    so every specialist run's eval (win rates, ELO, vs-ext verdicts) measured the model piloting
    RANDOM teams it never trained on (pure out-of-distribution; the ai_v7_05–08 "plateau" was this
    gap, not the training). No pin → the default pool builder, byte-identical to the old behavior."""
    from utils.teambuilder import Gen3Teambuilder

    team_str = cfg.get("trainee_team_str")
    if team_str:
        # a LIST = the distillation/multi-team case (sample among the taught teams, as training does);
        # a plain str = the single --trainee-team pin.
        return Gen3Teambuilder(list(team_str) if isinstance(team_str, (list, tuple)) else [team_str])
    return Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
