"""MatchupSpec — the ONE explicit declaration of what a run's battles look like.

Design: `designs/ai_v8/design_matchup_config.md` (P0). One week produced four independent failures
with a shared root — *the matchup a run plays is assembled implicitly across seams that nothing
forces to agree*: the eval worker rebuilt its own default teams (specialists measured OOD), the
env's single `team=` fed BOTH sides (the training mirror), and training/eval play modes drifted
(stochastic noise-farming). This module makes the matchup EXPLICIT: built ONCE in `train_rl_agent`
from the CLI (`MatchupSpec.from_args`), then CONSUMED — never re-derived — by the consumers
(the `plan.json` pattern: the parent writes the single source of truth; consumers read it).

P0 scope (this module): the spec + team-source builders + the startup echo + provenance
(`to_dict`/`spec_hash`, stamped into `metadata.json` beside `cli_args`). The team builders the env
factory uses come FROM the spec (`trainee_teams.build()` / `opponent_teams.build()`), so the two
sides are independent BY CONSTRUCTION, and the realized-matchup fuzz
(`poke_env_gaps/matchup_realized_fuzz_test.py`) asserts real battles match the declaration.
P1+ (not built): controllers keyed on eval play modes, per-row regime tags, per-opponent team pools.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# The default trainee mix (the historical behavior): full pool with a 10% bias draw from the
# curated sample teams. Kept as named constants so the spec and the legacy construction can't drift.
DEFAULT_TRAINEE_BIAS_PROB = 0.1


@dataclass(frozen=True)
class TeamSource:
    """Where one side's teams come from. `build()` is the ONLY constructor of that side's
    teambuilder — the env factory must not assemble builders inline anymore.

    kinds:
      * ``pool``           — uniform draws from the full team pool (the opponent default)
      * ``default_biased`` — full pool with a ``bias_prob`` draw from the sample teams (the
                             trainee default; reproduces the historical builder byte-for-byte)
      * ``pinned``         — ONE fixed team (``--trainee-team``); ``pin_str`` is the raw
                             Showdown export, ``pin_file`` its provenance
      * ``pin_biased``     — the pinned team with prob ``bias_prob``, else a pool draw (the
                             future ``--trainee-team-prob`` 50/50; supported here, no CLI yet)
    """
    kind: str
    pin_file: "str | None" = None
    pin_str: "str | None" = None
    bias_prob: float = 0.0

    def __post_init__(self):
        if self.kind not in ("pool", "default_biased", "pinned", "pin_biased"):
            raise ValueError(f"unknown TeamSource kind {self.kind!r}")
        if self.kind in ("pinned", "pin_biased") and not self.pin_str:
            raise ValueError(f"TeamSource kind {self.kind!r} needs pin_str (the team export)")

    def build(self, all_teams, sample_teams, team_pfsp="off",
              team_pfsp_cap=3.0, team_pfsp_floor=0.05):
        """The side's Gen3Teambuilder. Parity contract: ``pool`` == the historical opponent
        builder, ``default_biased`` == the historical trainee builder, ``pinned`` == the
        --trainee-team builder — each byte-for-byte (pinned by matchup_spec_test).

        team_pfsp/cap/floor: threaded ONLY for the TRAINEE side (opponent teams are not
        win-rate-sampled). Default "off" is byte-identical to the pre-PFSP construction."""
        from utils.teambuilder import Gen3Teambuilder
        _tp = dict(team_pfsp=team_pfsp, team_pfsp_cap=team_pfsp_cap, team_pfsp_floor=team_pfsp_floor)
        if self.kind == "pool":
            return Gen3Teambuilder(all_teams, **_tp)
        if self.kind == "default_biased":
            return Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=self.bias_prob, **_tp)
        if self.kind == "pinned":
            return Gen3Teambuilder([self.pin_str], **_tp)
        # pin_biased: the pinned team bias_prob of the time, else the full pool.
        return Gen3Teambuilder(all_teams, bias_teams=[self.pin_str], bias_prob=self.bias_prob, **_tp)

    def describe(self) -> str:
        if self.kind == "pool":
            return "full pool"
        if self.kind == "default_biased":
            return f"full pool + {self.bias_prob:.0%} sample-team bias"
        if self.kind == "pinned":
            return f"PINNED {self.pin_file or '<inline>'}"
        return f"PINNED {self.pin_file or '<inline>'} @ {self.bias_prob:.0%}, else pool"


@dataclass(frozen=True)
class PlayMode:
    """How a (frozen NN) opponent selects actions. Descriptive in P0 — the executors (RLPlayer
    stochastic/temp, the anneal/ratchet callbacks) already exist; the spec records the intent so
    the echo/provenance say what a metric was measured under. Bots are deterministic (n/a)."""
    kind: str = "stochastic"            # greedy | stochastic
    temperature: float = 1.0
    schedule: str = "fixed"             # fixed | anneal | ratchet (the exploiter-temp modes)

    def describe(self) -> str:
        if self.kind == "greedy":
            return "greedy"
        sched = "" if self.schedule == "fixed" else f", {self.schedule}"
        return f"stochastic@{self.temperature:g}{sched}"


@dataclass(frozen=True)
class MatchupSpec:
    """The whole matchup, declared once. Training + eval, both sides."""
    trainee_teams: TeamSource
    opponent_teams: TeamSource
    # -- training opponent mix (descriptive; the wrapper executes it) --
    mix_kind: str = "bots"              # bots | self_play | exploiter
    bot_weights: "str | None" = None
    exploiter_target: "str | None" = None
    exploiter_keep_bots: bool = False
    exploiter_bot_fraction: float = 0.5
    opponent_play: PlayMode = field(default_factory=PlayMode)
    # -- eval --
    eval_trainee_teams: TeamSource = None  # type: ignore[assignment]  # defaults to trainee_teams
    eval_opponent_play: PlayMode = field(default_factory=lambda: PlayMode(kind="greedy"))

    def __post_init__(self):
        if self.eval_trainee_teams is None:
            # The #1-bug fix made structural: eval pilots what training pilots, by DEFAULT.
            object.__setattr__(self, "eval_trainee_teams", self.trainee_teams)

    # -- provenance --
    def to_dict(self) -> dict:
        def ts(t: TeamSource) -> dict:
            return {"kind": t.kind, "pin_file": t.pin_file, "bias_prob": t.bias_prob,
                    # provenance keeps a short fingerprint of the pinned team, not the full text
                    "pin_sha": (hashlib.sha1(t.pin_str.encode()).hexdigest()[:10]
                                if t.pin_str else None)}
        return {
            "trainee_teams": ts(self.trainee_teams),
            "opponent_teams": ts(self.opponent_teams),
            "mix_kind": self.mix_kind,
            "bot_weights": self.bot_weights,
            "exploiter_target": self.exploiter_target,
            "exploiter_keep_bots": self.exploiter_keep_bots,
            "exploiter_bot_fraction": self.exploiter_bot_fraction,
            "opponent_play": vars(self.opponent_play).copy(),
            "eval_trainee_teams": ts(self.eval_trainee_teams),
            "eval_opponent_play": vars(self.eval_opponent_play).copy(),
        }

    def spec_hash(self) -> str:
        """Stable short hash of the declared matchup — the measurement-regime tag. Two runs (or two
        eras of one run) with different hashes are NOT metric-comparable (the OOD-era lesson)."""
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]

    def summary_lines(self) -> "list[str]":
        """The startup echo — one glance at what this run actually plays (Events panel)."""
        lines = [
            f"🧭 [MATCHUP {self.spec_hash()}] trainee teams: {self.trainee_teams.describe()}",
            f"   opponent teams: {self.opponent_teams.describe()} | mix: {self.mix_kind}"
            + (f" (weights {self.bot_weights})" if self.bot_weights else ""),
        ]
        if self.mix_kind == "exploiter":
            lines.append(
                f"   exploiter target: {self.exploiter_target} ({self.opponent_play.describe()})"
                + (f" | bots mixed in {self.exploiter_bot_fraction:.0%}" if self.exploiter_keep_bots
                   else " | sole opponent"))
        lines.append(
            f"   eval: trainee on {self.eval_trainee_teams.describe()} vs opponents "
            f"{self.eval_opponent_play.describe()}")
        return lines

    # -- construction --
    @classmethod
    def from_args(cls, args) -> "MatchupSpec":
        """The single CLI → matchup mapping. Reads the pinned team file here (one place)."""
        pin_str = None
        if getattr(args, "trainee_team", None):
            with open(args.trainee_team, "r", encoding="utf-8") as f:
                pin_str = f.read()
            trainee = TeamSource(kind="pinned", pin_file=args.trainee_team, pin_str=pin_str)
        else:
            trainee = TeamSource(kind="default_biased", bias_prob=DEFAULT_TRAINEE_BIAS_PROB)
        opponents = TeamSource(kind="pool")

        if getattr(args, "exploiter", None):
            mix_kind = "exploiter"
        elif getattr(args, "self_play", False):
            mix_kind = "self_play"
        else:
            mix_kind = "bots"

        # The frozen-NN opponents' play mode (bots are deterministic — n/a): the exploiter target's
        # temp curriculum when set, else the stable/self-play temperature.
        if getattr(args, "exploiter_temp_start", None) is not None:
            play = PlayMode(kind="stochastic", temperature=float(args.exploiter_temp_start),
                            schedule=str(getattr(args, "exploiter_temp_mode", "fixed") or "fixed")
                            if getattr(args, "exploiter_temp_mode", "fixed") != "fixed" else "anneal")
        else:
            play = PlayMode(kind="stochastic",
                            temperature=float(getattr(args, "stable_opponent_temp", 1.0)))

        return cls(
            trainee_teams=trainee,
            opponent_teams=opponents,
            mix_kind=mix_kind,
            bot_weights=getattr(args, "bot_weights", None),
            exploiter_target=getattr(args, "exploiter", None),
            exploiter_keep_bots=bool(getattr(args, "exploiter_keep_bots", False)),
            exploiter_bot_fraction=float(getattr(args, "exploiter_bot_fraction", 0.5)),
            opponent_play=play,
        )


def sample_team_shas(sample_teams) -> "set[str]":
    """Strip-normalized sha1[:10] of each curated sample team — the fingerprint set an exploiter
    trainee must belong to. Strip-normalized because ``TeamLoader`` strips files but a pin is read
    raw (a trailing newline must not spoof a mismatch); matches ``team_archetypes.team_sha``."""
    return {hashlib.sha1(t.strip().encode()).hexdigest()[:10] for t in sample_teams}


def validate_exploiter_trainee_is_sample(spec: "MatchupSpec", sample_teams) -> None:
    """The EXPLOITER team-source guarantee: an exploiter must ever pilot only a VETTED sample team —
    the curated, tournament-proven set (``data/teams/sample/``) — never an arbitrary or
    bulk-downloaded ``other`` team. Raises ``ValueError`` when a ``mix_kind == 'exploiter'`` run
    pins a trainee team whose (strip-normalized) fingerprint is not in the sample set; the caller
    turns it into a startup FATAL. Out of scope (returns quietly): a non-exploiter run, or an
    exploiter with an UNPINNED trainee (a full-pool generalist exploiter, not a single-team
    specialist — it isn't "using a team" to constrain). Covers the single-pin ``pinned`` /
    ``pin_biased`` kinds; a future multi-team exploiter pool must validate every member likewise."""
    if spec.mix_kind != "exploiter":
        return
    ts = spec.trainee_teams
    if ts.kind not in ("pinned", "pin_biased") or not ts.pin_str:
        return
    shas = sample_team_shas(sample_teams)
    pin = hashlib.sha1(ts.pin_str.strip().encode()).hexdigest()[:10]
    if pin not in shas:
        raise ValueError(
            f"exploiter trainee team {ts.pin_file or '<inline>'!r} (sha {pin}) is NOT one of the "
            f"{len(shas)} curated SAMPLE teams. Exploiters must only ever pilot a vetted, "
            "tournament-proven sample team (a subset of data/teams/sample/) — bulk-downloaded / "
            "hand-crafted teams are not allowed here. Pick a sample team, or promote this one into "
            "the sample set first if it is proven.")


def describe_drift(recorded: "dict | None", current: "dict | None") -> "list[str]":
    """Field-level diff of two ``MatchupSpec.to_dict()``s — the resume drift guard's payload.

    A resume whose declared matchup differs from what the run last recorded is legitimate
    (a deliberate mid-run curriculum change) but must never be SILENT: the caller emits these
    lines loudly and the new era lands in the metadata ``matchup_history``. Returns one
    ``key: recorded → current`` line per differing top-level field ([] = no drift). Pure."""
    recorded, current = recorded or {}, current or {}
    lines = []
    for key in sorted(set(recorded) | set(current)):
        a, b = recorded.get(key), current.get(key)
        if a != b:
            lines.append(f"{key}: {a!r} → {b!r}")
    return lines
