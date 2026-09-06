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
      * ``pin_multi``      — a SMALL FIXED SET of teams, sampled uniformly (``--trainee-teams``);
                             ``pin_strs`` are the raw exports, ``pin_files`` their provenance. For
                             a z-near multi-team exploiter (the 1-vs-3-team A/B). ``pin_str`` mirrors
                             ``pin_strs[0]`` so single-team consumers (eval pin, provenance) still work.
    """
    kind: str
    pin_file: "str | None" = None
    pin_str: "str | None" = None
    bias_prob: float = 0.0
    pin_strs: "tuple[str, ...] | None" = None
    pin_files: "tuple[str, ...] | None" = None

    def __post_init__(self):
        if self.kind not in ("pool", "default_biased", "pinned", "pin_biased", "pin_multi"):
            raise ValueError(f"unknown TeamSource kind {self.kind!r}")
        if self.kind in ("pinned", "pin_biased") and not self.pin_str:
            raise ValueError(f"TeamSource kind {self.kind!r} needs pin_str (the team export)")
        if self.kind == "pin_multi":
            if not self.pin_strs:
                raise ValueError("TeamSource kind 'pin_multi' needs pin_strs (the team exports)")
            # mirror the first team into pin_str so single-team consumers keep working
            if self.pin_str is None:
                object.__setattr__(self, "pin_str", self.pin_strs[0])
            if self.pin_file is None and self.pin_files:
                object.__setattr__(self, "pin_file", self.pin_files[0])

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
        if self.kind == "pin_multi":
            # the fixed set, sampled uniformly per episode (like a mini-pool of just these teams)
            return Gen3Teambuilder(list(self.pin_strs), **_tp)
        # pin_biased: the pinned team bias_prob of the time, else the full pool.
        return Gen3Teambuilder(all_teams, bias_teams=[self.pin_str], bias_prob=self.bias_prob, **_tp)

    def describe(self) -> str:
        if self.kind == "pool":
            return "full pool"
        if self.kind == "default_biased":
            return f"full pool + {self.bias_prob:.0%} sample-team bias"
        if self.kind == "pinned":
            return f"PINNED {self.pin_file or '<inline>'}"
        if self.kind == "pin_multi":
            names = ", ".join(self.pin_files or [f"<team {i}>" for i in range(len(self.pin_strs))])
            return f"PINNED {len(self.pin_strs)} teams: {names}"
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
            d = {"kind": t.kind, "pin_file": t.pin_file, "bias_prob": t.bias_prob,
                 # provenance keeps a short fingerprint of the pinned team, not the full text
                 "pin_sha": (hashlib.sha1(t.pin_str.encode()).hexdigest()[:10]
                             if t.pin_str else None)}
            if t.kind == "pin_multi":
                d["pin_shas"] = [hashlib.sha1(s.encode()).hexdigest()[:10] for s in t.pin_strs]
            return d
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
        if getattr(args, "trainee_teams", None):
            # a small FIXED SET, sampled uniformly (the multi-team exploiter, --trainee-teams f1,f2,..)
            files = [f.strip() for f in args.trainee_teams.split(",") if f.strip()]
            strs = []
            for fp in files:
                with open(fp, "r", encoding="utf-8") as f:
                    strs.append(f.read())
            trainee = TeamSource(kind="pin_multi", pin_strs=tuple(strs), pin_files=tuple(files))
        elif getattr(args, "trainee_team", None):
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

        # DISTILLATION: training biases `--distill-team-bias` of episodes onto the TEACHER teams, so
        # eval must measure the trainee ON THOSE TEAMS — otherwise `win_rate_vs_ext_<teacher>` compares a
        # random-pool trainee against a teacher piloting its own pinned teams, which mostly measures the
        # teacher's TEAM ADVANTAGE, not whether the distillation transferred. (This is the same
        # eval-pilots-what-training-pilots invariant the single-team pin already enforces; the distill
        # path silently violated it — the eval read 0.36 while an offline per-team probe read 0.710.)
        # Keyed on the PAIRS, so it follows the team bias exactly — including a `--distill-coef 0`
        # CONTROL arm, which since `gen3_distill_bias_at_coef0_v1` IS biased onto the teacher teams
        # and so must be MEASURED on them too, or the invariant would hold for one arm of a pair only.
        eval_trainee = None
        _dp = getattr(args, "_distill_pairs", None)
        if _dp:
            _strs, _files = [], []
            for _t, _teams in _dp:
                for _f in _teams:
                    with open(_f, "r", encoding="utf-8") as _fh:
                        _strs.append(_fh.read())
                    _files.append(_f)
            if _strs:
                eval_trainee = TeamSource(kind="pin_multi", pin_strs=tuple(_strs),
                                          pin_files=tuple(_files))

        return cls(
            trainee_teams=trainee,
            opponent_teams=opponents,
            eval_trainee_teams=eval_trainee,
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
    shas = sample_team_shas(sample_teams)
    # each pinned member (single or multi) must be a vetted sample team
    if ts.kind == "pin_multi":
        members = list(zip(ts.pin_files or [None] * len(ts.pin_strs), ts.pin_strs))
    elif ts.kind in ("pinned", "pin_biased") and ts.pin_str:
        members = [(ts.pin_file, ts.pin_str)]
    else:
        return
    for pin_file, pin_str in members:
        pin = hashlib.sha1(pin_str.strip().encode()).hexdigest()[:10]
        if pin not in shas:
            raise ValueError(
                f"exploiter trainee team {pin_file or '<inline>'!r} (sha {pin}) is NOT one of the "
                f"{len(shas)} curated SAMPLE teams. Exploiters must only ever pilot a vetted, "
                "tournament-proven sample team (a subset of data/teams/sample/) — bulk-downloaded / "
                "hand-crafted teams are not allowed here. Pick a sample team, or promote this one into "
                "the sample set first if it is proven.")


def read_recorded_trainee_teams(path: str, *, require_teams: bool = False) -> "list[str]":
    """THE single provenance reader: which team files did the run at ``path`` train its trainee on?

    ``path`` is a run dir, a checkpoint ``.zip``, or a ``model_config.json`` — the run's
    ``metadata.json`` is searched next to it and one level up (mirroring ``_read_source_elo``).
    Reads ``cli_args.trainee_teams`` (the multi-team ``pin_multi`` form) or ``cli_args.trainee_team``
    (the single pin); a generalist run (neither) returns ``[]``.

    🚨 **A PATH THAT DOES NOT EXIST RAISES** (`gen3_run_spec_split_v1`, 2026-09-05). It used to
    return ``[]`` — the same answer a real generalist run gives — so a caller that handed this a
    RUN SPEC rather than a run dir got a wrong answer on a success path::

        read_recorded_trainee_teams('models/ai_v9_92_R5F00_0831')          -> 2 teams
        read_recorded_trainee_teams('models/ai_v9_92_R5F00_0831@26267760') -> 0 teams   # was []

    A `--distill-teacher '<run>@<step>:*'` fold therefore reported teachers that taught nothing,
    diagnosed by the wrong message ("that run recorded NO trainee teams"). The producer side is
    fixed by `agents.training.run_spec.split_run_spec`; this raise is the consumer-side guard, so
    the silence cannot come back through some other caller. ``require_teams=True`` additionally
    raises when the run exists but recorded no pin — for a caller (the ``'TEACHER:*'`` wildcard)
    whose whole request is "the teams this run trained on".

    TWO consumers share this so producer and consumer cannot drift:
      * ``--distill-teacher '<model>:*'`` — distil a teacher over EXACTLY the teams it trained on.
        Hand-typing that list risks a mismatch, which would fire the distill mask on states where
        the teacher is OFF-DISTRIBUTION — silently, since nothing checks it.
      * the fold-back contract (``fixed_opponent_pool._read_trainee_pin``) — a specialist used as an
        OPPONENT must pilot its OWN team(s), not the shared pool.

    FAIL-LOUD, never silently degrade: a recorded team file that is missing raises
    ``FileNotFoundError``; a file whose content no longer matches the run's recorded fingerprint
    (``pin_shas`` / ``pin_sha`` from the MatchupSpec provenance, when present) raises ``ValueError``
    — the file changed since that run trained on it, so distilling/piloting it would be a lie.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"read_recorded_trainee_teams: {path!r} does not exist — refusing to read it as a run "
            "with no recorded teams. A run SPEC carries an optional '@<step>' suffix that a "
            "directory reader must not see: split it with "
            "`agents.training.run_spec.split_run_spec` first.")
    d = path if os.path.isdir(path) else os.path.dirname(path)
    meta = None
    for cand in (os.path.join(d, "metadata.json"),
                 os.path.join(os.path.dirname(d), "metadata.json")):
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                break
            except (OSError, ValueError):
                continue
    if not isinstance(meta, dict):
        if require_teams:
            raise ValueError(
                f"read_recorded_trainee_teams: {path!r} exists but no metadata.json was found "
                "beside it or one level up — it is not a run dir / run artifact.")
        return []
    cli = meta.get("cli_args") or {}
    raw = cli.get("trainee_teams") or cli.get("trainee_team")
    if not raw:
        if require_teams:
            raise ValueError(
                f"read_recorded_trainee_teams: run {path!r} recorded NO trainee teams "
                "(cli_args has neither trainee_teams nor trainee_team) — it was not a "
                "specialist/exploiter run.")
        return []
    files = [x.strip() for x in str(raw).split(",") if x.strip()]
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"run {path!r} recorded trainee team {f!r} but the file no longer exists — refusing "
                "to silently fall back (a specialist must train/pilot ITS OWN teams).")
    # Fingerprint check against the recorded MatchupSpec, when present. TWO locations, newest first:
    # `matchup_history[-1].spec` (the append-only era record) and the older
    # `cli_args._matchup_spec` stamp — older runs carry only the latter.
    hist = meta.get("matchup_history") or []
    spec_ts = (hist[-1].get("spec", {}).get("trainee_teams") or {}) if hist else {}
    if not spec_ts:
        spec_ts = ((cli.get("_matchup_spec") or {}).get("trainee_teams") or {})
    recorded = spec_ts.get("pin_shas") or ([spec_ts["pin_sha"]] if spec_ts.get("pin_sha") else [])
    if recorded:
        # UNSTRIPPED, matching how `MatchupSpec.to_dict` records pin_sha/pin_shas (raw file content).
        got = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                got.append(hashlib.sha1(fh.read().encode()).hexdigest()[:10])
        if set(got) != set(recorded):
            raise ValueError(
                f"run {path!r}: the team file(s) no longer match the pin_sha its run recorded "
                f"(recorded {sorted(recorded)}, on disk {sorted(got)}) — a team file changed since "
                "that run trained on it.")
    return files


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
