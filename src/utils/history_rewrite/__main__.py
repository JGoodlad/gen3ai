"""``python -m utils.history_rewrite`` — the dry-run toolkit for the account-name purge.

DRY RUN ONLY: no subcommand pushes or touches a ref of the real repository. ``rewrite`` and ``dryrun`` work in a
throwaway CLONE under ``--work``; every other subcommand reads. The names never appear on stdout (identities are ``#NN``).
Runs at ``nice 19`` (and on ``--cpus`` if given), so it is safe beside a live training run (standing rule 5b).

Subcommands (a typical order; ``dryrun`` runs all of it on a clone):

  fetch-filter-repo --dest D           download + checksum git_filter_repo.py (pinned) into D (outside the repo)
  names   --out F                      derive the identities from the introducing commit; write F (mode 0600, out of repo)
  audit   --names F                    scan ALL history for the names: counts by identity index and by file
  rules   --names F --out R            audit, then write the filter-repo rules file R (mode 0600)
  plan    --work W --rules R           print the exact commands the rewrite will run
  rewrite --work W --names F           clone + filter-repo into W (a fresh dir); copies the commit-map / ref-map
  map     --work W --out TSV           the old->new SHA map of the CHANGED commits (the in-repo file)
  cite    --tree T --map TSV           rewrite SHA citations in T's tracked text files (dry unless --apply)
  verify  --work W --names F           names gone? graph intact? changed set = descendants? tip trees equal?
  report  --work W --tree T            what cannot be re-pointed (refs, worktrees, models/ sidecars, archives, GitHub)
  push-plan --map TSV --clone C        print (never run) the one atomic `git push --force-with-lease` of the moved refs
  local-plan --map TSV                 print (never run) the commands that move local branches / worktrees onto the new SHAs
  leakcheck --tree T                   no file of T holds a name (before every commit of this tooling; waived paths allowed)
  gates   --clone C                    the static tier (utils.static_gates) inside a clone
  dryrun  --work W --names F           everything above on a clone (then run `gates` on it before and after `cite`)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from utils.git import get_main_repo_root
from utils.history_rewrite import cite as C
from utils.history_rewrite import names as N
from utils.history_rewrite import report as R
from utils.history_rewrite import rewrite as W
from utils.history_rewrite import shamap as S
from utils.history_rewrite import verify as V
from utils.history_rewrite.gitobjs import commit_oids
from utils.history_rewrite.rules import SCOPE_DIR, Rules, build_rules

DEFAULT_NAMES = os.path.expanduser("~/.cache/gen3ai/history_rewrite/names.json")
DEFAULT_FR_DIR = os.path.expanduser("~/.cache/gen3ai/history_rewrite/filter_repo")
INTRO = ",".join(N.INTRODUCING)


def _low_priority(cpus: str | None) -> None:
    os.nice(19)
    if cpus:
        want: set[int] = set()
        for part in cpus.split(","):
            lo, _, hi = part.partition("-")
            want |= set(range(int(lo), int(hi or lo) + 1))
        os.sched_setaffinity(0, want & os.sched_getaffinity(0) or os.sched_getaffinity(0))


def _tiers(arg: str | None) -> list[str] | None:
    return [t for t in arg.split(",") if t] if arg else None


def _intro(a: argparse.Namespace) -> list[str]:
    """The introducing commits (``--introducing``, comma list); the FIRST is the base of the rewrite."""
    return [c for c in a.introducing.split(",") if c]


def _load_ids(args: argparse.Namespace) -> list[N.Identity]:
    return N.load_identities(args.names, _tiers(args.tiers))


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_private(path: str, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(p, 0o600)


def cmd_names(a: argparse.Namespace) -> int:
    ids = N.derive_identities(a.source, _intro(a)[0])
    N.save_identities(a.out, ids, a.source)
    flagged = sum(1 for i in ids if i.tier == "flagged")
    print(f"derived {len(ids)} identities from {_intro(a)[0]}: {flagged} flagged + {len(ids) - flagged} only in top25 -> {a.out} (0600, outside the repo)")
    if len(ids) != flagged:
        print(f"FINDING: {len(ids) - flagged} identities are in the top-25 list but not in the owner-approved 61; "
              f"--tiers flagged restricts the purge to the 61 (the other {len(ids) - flagged} stay in history)")
    return 0


def _scope(a: argparse.Namespace) -> set[str]:
    return {p for p in N.introducing_files(a.source, _intro(a)) if p.startswith(SCOPE_DIR)}


def cmd_audit(a: argparse.Namespace) -> int:
    ids = _load_ids(a)
    audit = N.audit_history(a.source, ids, before=_intro(a)[0], progress=lambda n: _say(f"  {n} blobs"))
    print("\n".join(audit.summary_lines(ids)))
    print(_rules_text(build_rules(audit, ids, _scope(a))))
    return 0


def _build_rules(a: argparse.Namespace, out: str) -> tuple[N.Audit, Rules]:
    """Audit the source, build the rules, write ``out`` (rules), ``callback.py`` and ``rules.meta.json`` beside it (0600)."""
    ids = _load_ids(a)
    audit = N.audit_history(a.source, ids, before=_intro(a)[0], progress=lambda n: _say(f"  {n} blobs"))
    rules = build_rules(audit, ids, _scope(a))
    _write_private(out, rules.render(ids))
    _write_private(os.path.join(os.path.dirname(out), "callback.py"), rules.render_callback(ids).encode())
    _write_private(os.path.join(os.path.dirname(out), "rules.meta.json"),
                   json.dumps({"waived": {str(k): list(v) for k, v in rules.waived.items()}, "scope_paths": list(rules.scope_paths)}).encode())
    return audit, rules


def _rules_text(rules: Rules) -> str:
    lit, rx, sc = rules.counts
    waived = "; ".join(f"#{i:02d}: {len(p)} path(s)" for i, p in sorted(rules.waived.items()))
    return f"rules: {lit} global literals + {rx} global regexes + {sc} scoped regexes (account-data files only: {len(rules.scope_paths)}); WAIVED occurrences: {waived or 'none'}"


def cmd_rules(a: argparse.Namespace) -> int:
    audit, rules = _build_rules(a, a.out)
    print("\n".join(audit.summary_lines(_load_ids(a))))
    print(f"{_rules_text(rules)} -> {a.out} (0600)")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    fr = a.filter_repo or os.path.join(DEFAULT_FR_DIR, "git_filter_repo.py")
    steps = W.plan_steps(a.source, a.work, a.rules, os.path.join(os.path.dirname(a.rules), "callback.py"), fr,
                         ["<oid>"] if a.show_keep else [])
    for i, s in enumerate(steps, 1):
        cwd = f"(cd {s.cwd}) " if s.cwd else ""
        argv = [x if len(x) < 200 else x[:60] + "..." for x in s.argv]
        print(f"{i}. {s.label}\n   {cwd}{' '.join(argv)}")
    print("\nThen: map (commit-map -> TSV), cite --apply on the clone's work tree, verify, report. "
          "NOT executed by this tool, ever: the force-push (designs/ops/history_rewrite.md).")
    return 0


def cmd_rewrite(a: argparse.Namespace) -> int:
    fr = a.filter_repo or str(W.fetch_filter_repo(DEFAULT_FR_DIR))
    rules = os.path.join(os.path.dirname(a.names), "rules.txt")
    _say("audit + rules")
    _, built = _build_rules(a, rules)
    _say(_rules_text(built))
    clone = W.run_rewrite(a.source, a.work, rules, os.path.join(os.path.dirname(rules), "callback.py"), fr,
                          introducing=_intro(a)[0], log=_say)
    print(f"clone: {clone}")
    return 0


def _pairs(work: str) -> tuple[dict[str, str], dict[str, str]]:
    full = S.read_commit_map(os.path.join(work, "commit-map"))
    return S.changed_only(full), full


def cmd_map(a: argparse.Namespace) -> int:
    pairs, full = _pairs(a.work)
    S.write_tsv(a.out, pairs, [
        f"gen3ai history rewrite: old -> new commit SHAs (40 hex, tab separated), CHANGED commits only; {len(pairs)} of {len(full)} commits",
        f"status: {a.status}",
        f"tool: git filter-repo {W.FILTER_REPO_VERSION}; resolve with `python -m utils.git_sha <sha>`; runbook designs/ops/history_rewrite.md"])
    print(f"{len(pairs)} changed commits of {len(full)} -> {a.out}")
    return 0


def _rewriter(pairs: dict[str, str], universe: str, tree: str) -> C.Rewriter:
    """The old universe is every commit of ``universe`` PLUS the map's old column, so a citation still resolves after the
    old objects are collected (Phase C)."""
    return C.Rewriter(pairs, C.CommitIndex([*commit_oids(universe), *pairs]), C.CommitIndex(commit_oids(tree)))


def cmd_cite(a: argparse.Namespace) -> int:
    pairs = S.read_tsv(a.map)
    rw = _rewriter(pairs, a.source, a.tree)
    res = C.rewrite_tree(a.tree, rw, apply=a.apply)
    print(_cite_summary(res))
    if a.apply and (probs := C.verify_edits(res, rw, a.tree)):
        print("\n".join(probs))
        return 1
    return 0


def _cite_summary(res: C.CiteResult) -> str:
    k = res.kinds
    return (f"files scanned {res.files_scanned} (binary skipped {res.files_binary}); changed {res.files_changed}{'' if res.written else ' (DRY: nothing written)'}; "
            f"citations rewritten {res.rewritten_tokens} ({len(res.edits)} distinct, {len(res.lengthened)} lengthened); "
            f"refused ambiguous {len(res.refusals)}; embedded-left {k['embedded']}; frozen files {len(res.frozen)}; pinned files {len(res.pinned)}; "
            f"other tokens: not-a-commit {k['not_a_commit']}, unchanged-commit {k['unchanged_commit']}, ambiguous-unchanged {k['ambiguous_unchanged']}")


def cmd_verify(a: argparse.Namespace) -> int:
    ids = _load_ids(a)
    pairs, full = _pairs(a.work)
    clone = os.path.join(a.work, "clone")
    meta = json.loads(Path(os.path.join(os.path.dirname(a.names), "rules.meta.json")).read_text())
    waived = {int(k): tuple(v) for k, v in meta["waived"].items()}
    checks = [V.check_names(clone, ids, waived), V.check_variants(clone, a.source, ids, tuple(_intro(a))), V.check_graph(clone, a.source, pairs, full),
              V.check_changed_set(a.source, pairs, full, _intro(a)[0]), V.check_tip_trees(clone, a.source, S.read_ref_map(os.path.join(a.work, "ref-map")))]
    for c in checks:
        print(c.line())
        for item in c.items[:25]:
            print(f"    {N.redact(item, ids)}")
    return 0 if all(c.ok or c.informational for c in checks) else 1


def _models(a: argparse.Namespace) -> str:
    return a.models or os.path.join(get_main_repo_root(a.source), "models")


def _scan_specs(a: argparse.Namespace) -> list[R.ScanSpec]:
    specs = []
    specs.append(R.ScanSpec("models/ sidecars", _models(a), ("*/metadata.json", "*/model_config.json", "*/checkpoints/*.json")))
    for raw in a.scan or []:
        label, _, rest = raw.partition("=")
        root, _, pats = rest.partition(":")
        specs.append(R.ScanSpec(label, root, tuple(pats.split(",")) if pats else tuple(f"*{s}" for s in R.TEXT_SUFFIXES)))
    return specs


def cmd_report(a: argparse.Namespace) -> int:
    full: dict[str, str] | None = None
    if a.map:
        pairs = S.read_tsv(a.map)
    else:
        pairs, full = _pairs(a.work)
    known = set(full) if full else None   # with --work the clone's snapshot: a ref that moved since reads as stale
    rw = _rewriter(pairs, a.source, a.tree)
    res = C.rewrite_tree(a.tree, rw, apply=False)
    scans = [R.scan(s, rw) for s in _scan_specs(a)]
    text = R.render_report(res, refs=R.ref_rows(a.source, pairs, rw, known), worktrees=R.worktree_rows(a.source, pairs, _models(a), known),
                           scans=scans, mapped_commits=len(pairs))
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


def _changed_files_summary(work: str, res: C.CiteResult) -> None:
    """Write ``cite_changed_files.tsv`` and print the changed files by extension; list every non-document file by name."""
    rows = sorted((p, k["rewritten"] + k["lengthened"]) for p, k in res.per_file.items())
    Path(work, "cite_changed_files.tsv").write_text("".join(f"{p}\t{n}\n" for p, n in rows))
    by_ext: dict[str, int] = {}
    for p, _ in rows:
        by_ext[os.path.splitext(p)[1] or "(none)"] = by_ext.get(os.path.splitext(p)[1] or "(none)", 0) + 1
    print("changed files by extension: " + ", ".join(f"{e} {n}" for e, n in sorted(by_ext.items(), key=lambda kv: -kv[1])))
    code = [p for p, _ in rows if os.path.splitext(p)[1] not in (".md", ".txt", ".json", ".jsonl", ".tsv", ".csv", ".log")]
    print(f"non-document files changed ({len(code)}): " + ", ".join(code[:40]))


def run_static_gates(clone: str, workers: int = 2) -> tuple[int, str]:
    """The static tier (``python -m utils.static_gates --paths``) inside ``clone``, with ``clone/src`` first on PYTHONPATH."""
    env = {**os.environ, "PYTHONPATH": os.path.join(clone, "src"), "CUDA_VISIBLE_DEVICES": "", "GEN3AI_SKIP_DEPS_GUARD": "1"}
    paths = subprocess.run([sys.executable, "-m", "utils.static_gates", "--paths"], cwd=clone, env=env,
                           capture_output=True, text=True, check=True).stdout.split()
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", f"-n{workers}", "-p", "no:cacheprovider", "--no-header"],
                              cwd=clone, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return 124, "static tier timed out after 1800 s (INCONCLUSIVE: a timeout is never a semantic outcome)"
    return proc.returncode, "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])


def cmd_gates(a: argparse.Namespace) -> int:
    rc, tail = run_static_gates(a.clone)
    print(tail)
    return rc


def cmd_local_plan(a: argparse.Namespace) -> int:
    pairs = S.read_tsv(a.map)
    print("\n".join(R.local_plan(a.source, pairs, _models(a))))
    return 0


def cmd_push_plan(a: argparse.Namespace) -> int:
    print("\n".join(R.push_plan(a.source, S.read_tsv(a.map), a.clone)))
    return 0


def cmd_leakcheck(a: argparse.Namespace) -> int:
    ids = _load_ids(a)
    meta = Path(os.path.join(os.path.dirname(a.names), "rules.meta.json"))
    allow = frozenset(p for v in json.loads(meta.read_text())["waived"].values() for p in v) if meta.exists() else frozenset()
    check = V.check_tree_names(a.tree, ids, allow)
    print(check.line())
    for item in check.items[:40]:
        print(f"    {N.redact(item, ids)}")
    return 0 if check.ok else 1


def cmd_fetch(a: argparse.Namespace) -> int:
    print(W.fetch_filter_repo(a.dest))
    return 0


def cmd_dryrun(a: argparse.Namespace) -> int:
    t0 = time.time()
    fr = a.filter_repo or str(W.fetch_filter_repo(DEFAULT_FR_DIR))
    ids = _load_ids(a)
    _say(f"dry run on a throwaway clone under {a.work}; {len(ids)} identities")
    rules = os.path.join(os.path.dirname(a.names), "rules.txt")
    audit, built = _build_rules(a, rules)
    print("\n".join(audit.summary_lines(ids)))
    _say(_rules_text(built))
    clone = W.run_rewrite(a.source, a.work, rules, os.path.join(os.path.dirname(rules), "callback.py"), fr,
                          introducing=_intro(a)[0], log=_say)
    pairs, full = _pairs(a.work)
    tsv = os.path.join(a.work, "history_rewrite_sha_map.tsv")
    S.write_tsv(tsv, pairs, [f"DRY RUN map: {len(pairs)} of {len(full)} commits", "status: DRY RUN"])
    _say(f"map: {len(pairs)} changed of {len(full)}")
    ns = argparse.Namespace(**{**vars(a), "work": a.work})
    rc = cmd_verify(ns)
    rw = _rewriter(pairs, a.source, clone)
    if a.gates:
        _say("static tier on the clone AFTER the history rewrite, BEFORE the citation rewrite")
        rc1, tail1 = run_static_gates(clone)
        print(f"static tier (before cite): exit {rc1}\n{tail1}")
    _say("citation rewrite (apply, on the clone)")
    res = C.rewrite_tree(clone, rw, apply=True)
    print(_cite_summary(res))
    _changed_files_summary(a.work, res)
    probs = C.verify_edits(res, rw, clone)
    _say(f"every rewritten citation resolves in the clone: {'YES' if not probs else 'NO: ' + str(probs[:5])}")
    if a.gates:
        _say("static tier on the clone AFTER the citation rewrite")
        rc2, tail2 = run_static_gates(clone)
        print(f"static tier (after cite): exit {rc2}\n{tail2}")
    again = C.rewrite_tree(clone, rw, apply=False)
    _say(f"second pass rewrites {again.rewritten_tokens} more (idempotent: expect 0)")
    scans = [R.scan(s, rw) for s in _scan_specs(a)]
    known = set(full)
    text = R.render_report(res, refs=R.ref_rows(a.source, pairs, rw, known), worktrees=R.worktree_rows(a.source, pairs, _models(a), known),
                           scans=scans, mapped_commits=len(pairs))
    Path(a.work, "cannot_rewrite_report.md").write_text(text)
    _say(f"report -> {a.work}/cannot_rewrite_report.md; total {time.time() - t0:.0f}s")
    return rc or (1 if probs or again.rewritten_tokens else 0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m utils.history_rewrite", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpus", default=None, help="pin to these CPUs, e.g. 5-7,13-15 (rule 5b agent cores)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, **kw):
        p = sub.add_parser(name, help=kw.pop("help", None))
        p.set_defaults(fn=fn)
        p.add_argument("--source", default=None, help="the ORIGINAL repository (read-only; default: this checkout's main)")
        p.add_argument("--introducing", default=INTRO, help="comma list of the commits that introduced the names; the first is the base")
        return p

    p = add("fetch-filter-repo", cmd_fetch, help="download the pinned git-filter-repo")
    p.add_argument("--dest", default=DEFAULT_FR_DIR)
    p = add("names", cmd_names, help="derive the identities")
    p.add_argument("--out", default=DEFAULT_NAMES)
    for name, fn in (("audit", cmd_audit), ("rules", cmd_rules), ("rewrite", cmd_rewrite), ("verify", cmd_verify), ("dryrun", cmd_dryrun)):
        p = add(name, fn)
        p.add_argument("--names", default=DEFAULT_NAMES)
        p.add_argument("--tiers", default=None, help="flagged,top25 (default: both; `flagged` = the owner-approved 61 only)")
        if name == "rules":
            p.add_argument("--out", required=True)
        if name in ("rewrite", "verify", "dryrun"):
            p.add_argument("--work", required=True)
        if name in ("rewrite", "dryrun"):
            p.add_argument("--filter-repo", default=None)
        if name == "dryrun":
            p.add_argument("--models", default=None)
            p.add_argument("--scan", action="append", metavar="LABEL=ROOT[:PAT,PAT]")
            p.add_argument("--gates", action="store_true", help="run the static tier on the clone before and after `cite`")
    p = add("plan", cmd_plan, help="print the commands")
    p.add_argument("--work", required=True)
    p.add_argument("--rules", required=True)
    p.add_argument("--filter-repo", default=None)
    p.add_argument("--show-keep", action="store_true")
    p = add("gates", cmd_gates, help="run the static tier inside a clone")
    p.add_argument("--clone", required=True)
    p = add("map", cmd_map, help="write the SHA map TSV")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--status", default="REWRITTEN")
    p = add("cite", cmd_cite, help="rewrite SHA citations")
    p.add_argument("--tree", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--apply", action="store_true")
    p = add("local-plan", cmd_local_plan, help="print the commands that move local refs + worktrees (never runs them)")
    p.add_argument("--map", required=True)
    p.add_argument("--models", default=None)
    p = add("leakcheck", cmd_leakcheck, help="no file of a work tree holds a name (run before every commit of this tooling)")
    p.add_argument("--tree", required=True)
    p.add_argument("--names", default=DEFAULT_NAMES)
    p.add_argument("--tiers", default=None)
    p = add("push-plan", cmd_push_plan, help="print (never run) the one atomic force-push with leases")
    p.add_argument("--map", required=True)
    p.add_argument("--clone", required=True)
    p = add("report", cmd_report, help="what cannot be rewritten")
    p.add_argument("--tree", required=True)
    p.add_argument("--map", default=None)
    p.add_argument("--work", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--models", default=None)
    p.add_argument("--scan", action="append", metavar="LABEL=ROOT[:PAT,PAT]")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    _low_priority(a.cpus)
    if getattr(a, "source", None) is None:
        a.source = get_main_repo_root()
    try:
        return a.fn(a)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"REFUSED: {N.redact(str(e), _safe_ids(a))}", file=sys.stderr)
        return 2


def _safe_ids(a: argparse.Namespace) -> list[N.Identity]:
    try:
        return N.load_identities(a.names) if getattr(a, "names", None) and os.path.exists(a.names) else []
    except (ValueError, OSError):
        return []


if __name__ == "__main__":
    sys.exit(main())
