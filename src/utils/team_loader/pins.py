"""Explicit, COMMITTED team lists for everything whose output must not move when a team folder does.

A golden, a benchmark or a cross-run measurement instrument that reads ``get_sample_teams()`` is
silently re-defined every time the curated set changes. On 2026-09-23 it did
(``gen3_curated_sample_split_v1``): ``data/teams/sample/`` went from 72 teams (32 Smogon + 40
promoted) back to exactly Smogon's 32, and the Curse RestLax paste was replaced by the thread's
newer one. Every consumer that must stay byte-identical across that change reads a list HERE, by
strip-normalized content sha, instead.

Keys are ``sha1(team_text.strip())[:10]`` — the convention of ``team_archetypes.team_sha``,
``matchup_spec.sample_team_shas`` and ``main.promote_teams``.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional

#: ``TeamLoader().get_sample_teams()`` EXACTLY as it stood before the curated/promoted split
#: (commit ``1ae1e04a``), in order: the 32 Smogon teams in thread order (index 17 is the
#: SUPERSEDED Curse RestLax paste ``0972146213a667c9``, sha ``45995e432f``), then the 40 teams
#: promoted on 2026-08-31 in promotion order. Because the pre-split pool was ``sample + other``,
#: this is ALSO the first 72 entries of the pre-split ``get_all_teams()``.
#:
#: Consumers (each must stay byte-identical across the split): the obs golden
#: (``golden_obs_capture``, first 6), the reward golden (``reward_golden_test``) and the
#: reproducible fixture battle (``obs_roundtrip_fuzz_test.record_fixture_battle``), the benchmarks
#: that drew from ``get_sample_teams()``, and the cross-run measurement instruments whose trainee
#: builder biased toward it (``snapshot_ladder``, ``bot_elo_calibration``, ``bot_matchup_matrix``).
PRE_SPLIT_SAMPLE_72 = (
    'bcd4d09ee9', 'a7406f6c97', 'c6e5e12614', 'c84f2b64a2', '21022d30fb', '3495ef83ef',
    '324235812b', '55ff6899a2', '569ebae46d', '1c4e182530', 'f36747ae7e', '01cd428c76',
    'c90e782cad', 'a12e56dca3', '4771662cf7', '78b3b6f4a6', '6a49f096f0', '45995e432f',
    'a2d1246e4b', '01cb64e16c', 'e050d23078', 'a7bb29d48c', 'fffd943e9e', '8aa51ef85c',
    'fed4eee838', 'b08909ab04', '41357610f2', '95e69273f8', 'bd4af7191a', 'f23f64a6b2',
    '69af2f1507', '564b9be3ae', '8bdb5796b9', '436335607f', '11fb5d2fc2', 'c6b830a3f0',
    '6916e13879', 'e702a104eb', 'ac17a9dde5', 'd1ed25a242', '64a691c473', 'e28069562f',
    '4bb434f8ba', '3650f09b2f', '9e95fb59d7', '9d8391d864', 'a185b2d193', 'dbbfac7bd5',
    '750d056194', 'a9831a5f5e', 'c9a2cef359', '3a83154c2a', '6ebe9ebc4d', '9b454d9ea7',
    '422a4745c5', '8b81c129de', '009e3d0244', 'b904dbe059', '9ba039ba8a', '63f81c5c58',
    'aea50f207e', '37d717a93a', '4239fc5ba2', '713403f9e9', '7d0337af97', '6a634b6281',
    'fda8ad353a', 'a5d6b3f271', 'af923978cf', '9c44a67be3', 'a6252f809d', '46d3021c99',
)


def team_sha(text: str) -> str:
    """Strip-normalized ``sha1[:10]`` — identical to ``agents.training.team_archetypes.team_sha``
    (not imported: ``utils`` must not depend on ``agents``; ``pins_test`` asserts they agree)."""
    return hashlib.sha1(text.strip().encode()).hexdigest()[:10]


def teams_by_sha(shas: Iterable[str], loader=None) -> List[str]:
    """The team texts for ``shas``, IN THAT ORDER, looked up across the pool AND the superseded
    pastes. Raises ``KeyError`` naming every sha it cannot find — a pinned list that silently
    shrank would re-define the golden it exists to freeze."""
    if loader is None:
        from utils.team_loader import TeamLoader
        loader = TeamLoader()
    index = {}
    for t in list(loader.get_all_teams()) + list(loader.get_superseded_teams()):
        index.setdefault(team_sha(t), t)
    shas = list(shas)
    missing = [s for s in shas if s not in index]
    if missing:
        raise KeyError(f"{len(missing)} pinned team sha(s) are not under data/teams "
                       f"(pool + superseded): {missing}")
    return [index[s] for s in shas]


def pre_split_sample_teams(loader: Optional[object] = None) -> List[str]:
    """The 72 texts of :data:`PRE_SPLIT_SAMPLE_72`, in order."""
    return teams_by_sha(PRE_SPLIT_SAMPLE_72, loader)


def measurement_bias_teams(loader: Optional[object] = None) -> List[str]:
    """The bias set of every cross-run MEASUREMENT's default trainee builder — the eval worker, the
    dense snapshot ladder, the bot ELO calibration and the bot matchup matrix — FROZEN at the
    pre-split 72 (:data:`PRE_SPLIT_SAMPLE_72`).

    Why frozen rather than following training's curated 32: these instruments' outputs are compared
    across the whole archive (``eval/elo``, ``win_rate_vs_pool``, ``ladder.json``, the bot anchors),
    and the ladder REUSES eval-measured pairs only because both play the same builder. Moving them
    with training would put a team-regime boundary into every one of those series at 2026-09-23 and
    mix regimes inside any ladder extended across it. Freezing keeps them one regime; the only
    difference from training is WHICH teams fill the 10% bias draw (the 90% pool draw is shared)."""
    return pre_split_sample_teams(loader)
