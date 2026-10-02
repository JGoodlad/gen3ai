"""THE EVAL DECLARATION and its startup (M5 Lane H; the declared lifecycle — nothing after it).

``EvalDecl`` is everything a Rust eval cycle will ever use, named at the trainer's startup beside the
collector's own declaration (``rust_rollout.build.RustEnvDecl``):

* the eval CORE — its env count (``--rust-eval-envs``), threads, front end and build (the collector's);
* the eval SLOTS in the trainer's T2 service (``eval_extra_slots`` → ``build_collector(extra_slots=)``):
  the trainee's EVAL slot and one slot per declared sentinel (``--n-sentinels`` under self-play), both
  in the trainee's slot group (the same compiled buckets — eval rides the rollout's), plus one slot
  per FIXED opponent that the training plan does not already serve (a stable opponent under self-play
  REUSES its training slot: the same weights, played greedy). 🚨 A slot holds its GROUP TEMPLATE's
  weights at startup (T2 deep-copies the template into every slot), and a same-architecture fixed
  slot joins the TRAINEE's group — so :func:`build_eval_core` LOADS each non-reused fixed opponent's
  weights into its slot (:func:`load_fixed_slots`), once, at startup. Without that load a
  ``--stable-opponents`` run without ``--self-play`` (no stable training route, so no reuse) measured
  its ``ext_`` opponent against the TRAINEE's startup weights; an ``--exploiter`` target was right only
  by accident (its training route is the group's first slot, so the template IS the target) — found
  by the Lane H fixed-opponent gate row;
* the eval ROUTE TABLE (``executor.EvalTable``): the nine roster bots in the core with per-episode
  streams, the sentinel and fixed policy routes, the filler route;
* the eval TEAM TABLE: the eval trainee builder (``eval_teams.build_trainee_tb`` — the specialist
  pin, else the pool builder with its 10 % sample-team bias), the flat pool builder, each fixed
  opponent's pinned builder — every team validated by use in ``Core::new``.

Every later cycle only LOADS weights into those slots and plays; ``RustEvalCore.check_lifecycle``
raises on any ``*_after_freeze`` counter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training.rust_eval.executor import EvalTable, RustEvalCore

#: The eval core's player names (p1 = the trainee). Distinct from the collector's so a trace names
#: which core played it.
NAMES = ("rhone", "rhtwo")


@dataclass(frozen=True)
class EvalDecl:
    n_envs: int = 64
    n_sentinels: int = 0
    fixed_labels: Tuple[str, ...] = ()
    #: fixed labels the TRAINING plan already serves (label -> its T2 slot)
    reused_fixed: Tuple[Tuple[str, int], ...] = ()


def eval_extra_slots(decl: EvalDecl, trainee_policy: Any, fixed_policies: Mapping[str, Any]
                     ) -> List[Tuple[str, Any]]:
    """The ``extra_slots`` the collector's T2 declaration appends for eval (module docs), in order:
    the trainee's eval slot, the sentinel slots, then every fixed opponent not reused."""
    reused = dict(decl.reused_fixed)
    out: List[Tuple[str, Any]] = [("evaltrainee", trainee_policy)]
    out += [("evalsentinel", trainee_policy)] * int(decl.n_sentinels)
    out += [("evalfixed", fixed_policies[lab]) for lab in decl.fixed_labels if lab not in reused]
    return out


def eval_table(decl: EvalDecl, extra_ids: Sequence[int], bots: Sequence[str]) -> EvalTable:
    reused = dict(decl.reused_fixed)
    ids = list(extra_ids)
    trainee = ids.pop(0)
    sentinels = tuple(ids[:decl.n_sentinels])
    ids = ids[decl.n_sentinels:]
    fixed: List[Tuple[str, int]] = []
    for lab in decl.fixed_labels:
        fixed.append((lab, reused[lab]) if lab in reused else (lab, ids.pop(0)))
    return EvalTable(bots=tuple(bots), trainee_slot=int(trainee), sentinel_slots=sentinels, fixed_slots=tuple(fixed))


def eval_builders(trainee_team_str: Any, fixed_entries: Sequence[Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    """``(trainee builder, flat pool builder, {fixed label: pinned builder})`` — the eval WORKER's own
    construction (``agents.training.eval_teams``, shared with ``main.eval_worker``), so both eval paths
    draw from the same distributions."""
    from agents.training.eval_teams import build_trainee_tb as _build_trainee_tb
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    loader = TeamLoader()
    all_teams, sample_teams = loader.get_all_teams(), loader.get_sample_teams()
    trainee = _build_trainee_tb({"trainee_team_str": trainee_team_str}, all_teams, sample_teams)
    flat = Gen3Teambuilder(all_teams)
    fixed: Dict[str, Any] = {}
    for e in fixed_entries:
        strs = list(getattr(e, "team_strs", ()) or ())
        one = getattr(e, "team_str", None)
        if strs:
            fixed[e.label] = Gen3Teambuilder(strs)
        elif one:
            fixed[e.label] = Gen3Teambuilder([one])
    return trainee, flat, fixed


def load_fixed_slots(svc: Any, decl: EvalDecl, table: EvalTable, fixed_policies: Mapping[str, Any]
                     ) -> List[Tuple[str, int]]:
    """LOAD every fixed opponent the eval core does NOT reuse from the training plan into its declared
    slot (module docs: a slot starts with its group template's weights, which for a same-architecture
    fixed opponent are the TRAINEE's). A reused label's slot is the training plan's, loaded by its own
    host. A missing policy is refused. Returns the ``(label, slot)`` pairs loaded."""
    reused = dict(decl.reused_fixed)
    done: List[Tuple[str, int]] = []
    for lab, slot in table.fixed_slots:
        if lab in reused:
            continue
        pol = fixed_policies.get(lab)
        if pol is None:
            raise RuntimeError(f"--env-core rust eval: fixed opponent {lab!r} has a declared eval slot ({slot}) "
                               "but no policy to load into it")
        svc.load(int(slot), pol, f"eval:fixed:{lab}")
        done.append((lab, int(slot)))
    return done


def build_eval_core(decl: EvalDecl, *, collector_decl: Any, svc: Any, extra_ids: Sequence[int],
                    trainee_builder: Any, opp_builder: Any, fixed_builders: Mapping[str, Any],
                    turn_limit: int, terminal: Optional[Dict[str, Any]] = None,
                    fixed_policies: Optional[Mapping[str, Any]] = None,
                    emit: Callable[[str], None] = print) -> RustEvalCore:
    """Acquire the eval core (module docs) over an already STARTED T2 service whose declaration carried
    :func:`eval_extra_slots`; ``extra_ids`` are those slots' global ids (``col.extra_slots``).
    ``fixed_policies`` (label -> policy, the ``eval_extra_slots`` mapping) are LOADED into the
    non-reused fixed slots here (:func:`load_fixed_slots`)."""
    import time

    from agents.training.eval_callback import eval_opponent_names
    from agents.training.rust_rollout.build import open_core
    from agents.training.rust_rollout.teams import TeamTable
    from utils.git import get_git_hash
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P
    from utils.rust_env.bot_inventory import by_name

    bots = eval_opponent_names()
    missing = [b for b in bots if not by_name().get(b) or not by_name()[b].rust]
    if missing:
        raise RuntimeError(f"--env-core rust eval: roster bots with no Rust port (Lane F): {missing}")
    table = eval_table(decl, extra_ids, bots)
    loaded = load_fixed_slots(svc, decl, table, dict(fixed_policies or {}))
    teams = TeamTable()
    for b in [trainee_builder, opp_builder, *fixed_builders.values()]:
        teams.add_builder(b)
    spec = P.spec_json(n=int(decl.n_envs), threads=int(collector_decl.threads), teams=list(teams.teams), names=NAMES,
                       decision_tense=bool(collector_decl.decision_tense),
                       switch_freeze=bool(collector_decl.switch_freeze), turn_limit=int(turn_limit),
                       refusal_budget=int(collector_decl.refusal_budget), bank_dir=None,
                       terminal=terminal, opponents=table.spec_rows())
    t0 = time.perf_counter()
    core = open_core(collector_decl, spec)
    lib = ffi.load(ffi.default_path(collector_decl.profile))
    try:
        commit = get_git_hash() or ""
    except Exception:
        commit = ""
    ev = RustEvalCore(core=core, svc=svc, lib=lib, table=table, team_table=teams, turn_limit=int(turn_limit),
                      trainee_builder=trainee_builder, opp_builder=opp_builder, fixed_builders=fixed_builders,
                      names=NAMES, commit=commit, emit=emit)
    emit(f"🦀 [RUST EVAL] eval core up in {time.perf_counter() - t0:.1f}s: {decl.n_envs} envs, {len(teams)} teams "
         f"validated by use, routes: {len(bots)} bots (in core, per-episode streams) + {len(table.sentinel_slots)} "
         f"sentinel slot(s) {list(table.sentinel_slots)} + fixed {list(table.fixed_slots)} (loaded {loaded}) + "
         f"filler; trainee eval slot {table.trainee_slot}")
    return ev
