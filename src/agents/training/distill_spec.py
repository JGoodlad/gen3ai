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
"""


def parse_distill_teacher_spec(spec: str):
    """``spec`` → ``[(teacher_path, [team_file, ...]), ...]``.

    Pure + total: raises ``ValueError`` (never ``SystemExit``) on a malformed spec so the caller can
    turn it into a ``parser.error``. Order is preserved — the teacher's 1-indexed position IS its
    teacher-id in the env's ``distill_mask``.
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
                "'TEACHER:TEAM[,TEAM...]')")
    return out
