"""Parsing for the ``--distill-teacher`` spec (``gen3_exploiter_distill_v1``, multi-team teachers).

One TEACHER may own MANY teams. A multi-team exploiter (``--trainee-teams``) is a SINGLE frozen model
that pilots a whole z-cluster, so binding a teacher to one team would force the same teacher to be
repeated N times — N *identical* teacher forward passes per batch, for nothing. The spec therefore
GROUPS teams under their teacher::

    'TEACHER1:teamA.txt,teamB.txt;TEACHER2:teamC.txt'

``;`` separates TEACHERS, ``,`` separates that teacher's TEAMS.

The legacy comma-separated pair form (``'T1:a.txt,T2:b.txt'``) still parses, so in-flight runs and
existing scripts keep working: a comma segment that ITSELF contains ``':'`` starts a NEW teacher.
That rule is unambiguous because a team file path never contains a colon.

**The ``*`` (or ``auto``) wildcard** — ``'TEACHER:*'`` resolves to EXACTLY the teams that teacher was
trained on, read from its own recorded provenance
(``matchup_spec.read_recorded_trainee_teams``). Prefer it: hand-typing the list risks a MISMATCH with
what the teacher actually trained on, which would fire the distill mask on states where the teacher is
OFF-DISTRIBUTION — silently, since nothing cross-checks it. ``auto`` is the glob-proof spelling for
unquoted shell use (``*`` must be quoted or the shell may expand it).

**A TEACHER IS A RUN SPEC, NOT A PATH** (`gen3_run_spec_split_v1`, 2026-09-05). The grammar is
``<run|zip>[@<step>]:<teams>`` — the same ``path[@step]`` half `--stable-opponents` has always
taken, pinning a specific ``<run>/checkpoints/checkpoint_<step>_steps.zip`` instead of the run's
``best_model``. It is split by the ONE canonical splitter (`agents.training.run_spec`), because it
was NOT split before: `'<run>@<step>:*'` handed the whole string to
``matchup_spec.read_recorded_trainee_teams``, which found no ``metadata.json`` beside a directory
that does not exist and answered ``[]`` — indistinguishable from a generalist run, and diagnosed as
one ("that run recorded NO trainee teams"). The reader now raises on a missing path, this parser
splits the suffix off before calling it, and :func:`check_teacher_spec` is the launch-time refusal
that both `resolve_config` and `main.checkargs` read.
"""
from agents.training.run_spec import split_run_spec

WILDCARDS = ("*", "auto")


def parse_distill_teacher_spec(spec: str, resolve_wildcard=None):
    """``spec`` → ``[(teacher_path, [team_file, ...]), ...]``.

    Pure + total: raises ``ValueError`` (never ``SystemExit``) on a malformed spec so the caller can
    turn it into a ``parser.error``. Order is preserved — the teacher's 1-indexed position IS its
    teacher-id in the env's ``distill_mask``.

    ``resolve_wildcard(teacher_path) -> [team_file, ...]`` expands a ``'TEACHER:*'`` group from the
    teacher's OWN recorded provenance. Omitted (None) → ``*`` raises, so the pure parser stays pure
    and the filesystem read is the caller's choice.
    """
    out: "list[tuple[str, list[str]]]" = []
    for group in (spec or "").split(";"):
        for seg in group.split(","):
            seg = seg.strip()
            if not seg:
                continue
            if ":" in seg:                       # starts a NEW teacher
                teacher, _, team = seg.partition(":")
                teacher, team = teacher.strip(), team.strip()
                if not teacher:
                    raise ValueError(f"--distill-teacher item {seg!r} has an empty teacher path")
                out.append((teacher, [team] if team else []))
            else:                                # another TEAM for the current teacher
                if not out:
                    raise ValueError(
                        f"--distill-teacher item {seg!r} has no teacher — the spec must start with "
                        "'TEACHER:TEAM' (';' separates teachers, ',' separates that teacher's teams)")
                out[-1][1].append(seg)
    for teacher, teams in out:
        if not teams:
            raise ValueError(
                f"--distill-teacher: teacher {teacher!r} has no team file (expected "
                "'TEACHER:TEAM[,TEAM...]' or 'TEACHER:*')")
        # Syntax-check the `path[@step]` half here, so a malformed step is a parse error rather
        # than a "no such checkpoint" hundreds of lines later (or, before the fix, silence).
        split_run_spec(teacher, what=f"--distill-teacher teacher {teacher!r}")
    # Expand the '*'/'auto' wildcard from each teacher's OWN recorded trainee teams.
    expanded = []
    for teacher, teams in out:
        if any(t in WILDCARDS for t in teams):
            if len(teams) != 1:
                raise ValueError(
                    f"--distill-teacher: teacher {teacher!r} mixes the '*' wildcard with explicit "
                    "team files — use one or the other")
            if resolve_wildcard is None:
                raise ValueError(
                    f"--distill-teacher: teacher {teacher!r} used '*' but no resolver was provided")
            # THE `@step` SPLIT: the resolver reads a run DIRECTORY's recorded provenance, and a
            # spec's optional step suffix names a checkpoint inside it. Handing the suffix through
            # is the defect this split exists to close.
            got = resolve_wildcard(split_run_spec(
                teacher, what=f"--distill-teacher teacher {teacher!r}")[0])
            if not got:
                raise ValueError(
                    f"--distill-teacher: teacher {teacher!r} used '*' but that run recorded NO "
                    "trainee teams (it was not a specialist/exploiter run) — list its teams "
                    "explicitly, or point at a run that pinned --trainee-team(s).")
            expanded.append((teacher, list(got)))
        else:
            expanded.append((teacher, teams))
    return expanded


def check_teacher_spec(spec, *, resolve_wildcard=None, check_paths: bool = True) -> "list[str]":
    """Every reason a ``--distill-teacher`` spec would fail or teach NOTHING. ``[]`` = it is fine.

    THE ONE DECLARATION, read by BOTH surfaces — `main.train.config.resolve_config` turns the first
    finding into a `parser.error` (a `FATAL_CONFIG`-class refusal at launch) and `main.checkargs`
    prints every one offline. Neither owns the rule, so they cannot drift; this is the same contract
    `main.train.combination_checks` has, kept in a separate function because that module explicitly
    excludes anything that touches the filesystem or a teacher spec, and this needs both.

    It answers the question the `🧪 [DISTILL]` startup line's team count was the only witness to:
    *does every named teacher resolve to at least one team?* A teacher that resolves to ZERO teams
    folds no loss and biases no team draw while every log line still reads as a running fold.

    ``check_paths`` selects HOW MUCH of the answer each surface wants, and the asymmetry is
    deliberate rather than a hedge:

      * ``False`` (what `resolve_config` passes) — the STRUCTURAL findings only. At a real launch
        the path questions are already answered LOUDLY downstream: `main.train.model_build` exits
        `FATAL_CONFIG` naming the teacher it could not load (at ``--distill-coef > 0``, the only
        coefficient that loads one) and `matchup_setup.apply_distill_team_bias` raises on a team
        file it cannot open. Re-asking here would duplicate a check that exists, and would newly
        refuse a coef-0 CONTROL arm whose teacher run has since been archived — a shape that works
        today and that nothing about this defect argues against.
      * ``True`` (what `main.checkargs` passes) — path findings too. OFFLINE there is no downstream:
        the whole job is to answer "would this command launch?" before anything runs, and a
        reporting tool that reads the filesystem is that tool's normal mode.

    Total either way: it never raises, it returns messages.
    """
    text = (spec or "").strip()
    if not text:
        return []
    import os

    try:
        pairs = parse_distill_teacher_spec(text, resolve_wildcard=resolve_wildcard)
    except (ValueError, FileNotFoundError) as e:
        return [str(e)]

    out: "list[str]" = []
    for teacher, teams in pairs:
        try:
            run_path, _step = split_run_spec(teacher, what=f"--distill-teacher teacher {teacher!r}")
        except ValueError as e:
            out.append(str(e))
            continue
        if check_paths and not os.path.exists(run_path):
            out.append(
                f"--distill-teacher: teacher {teacher!r} names {run_path!r}, which does not exist "
                "— the teacher would resolve to nothing. (A run spec is 'path[@step]'; the '@step' "
                "suffix is part of the SPEC, never part of the directory name.)")
        if not teams:
            out.append(
                f"--distill-teacher: teacher {teacher!r} resolved to ZERO teams — it would teach "
                "nothing while every log line still reads as a running fold. Name its team files, "
                "or point ':*' at a run that pinned --trainee-team(s).")
        for team_file in (teams if check_paths else ()):
            if not os.path.isfile(team_file):
                out.append(
                    f"--distill-teacher: teacher {teacher!r} names team file {team_file!r}, which "
                    "does not exist.")
    return out
