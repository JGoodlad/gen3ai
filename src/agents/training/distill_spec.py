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
"""

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
            got = resolve_wildcard(teacher)
            if not got:
                raise ValueError(
                    f"--distill-teacher: teacher {teacher!r} used '*' but that run recorded NO "
                    "trainee teams (it was not a specialist/exploiter run) — list its teams "
                    "explicitly, or point at a run that pinned --trainee-team(s).")
            expanded.append((teacher, list(got)))
        else:
            expanded.append((teacher, teams))
    return expanded
