"""Promote a seed-recorded RANDOM draw of validated pool teams into the curated sample set.

The 40-team fleet needs 40 legal exploiter trainees, and ``--exploiter`` refuses any trainee that
is not in ``data/teams/sample/`` (``matchup_spec.validate_exploiter_trainee_is_sample``). The owner
ruling (ledger 2026-08-30) is that the fleet is drawn **at random** from the validated pool rather
than ranked or hand-picked, so the result is an unbiased estimate of pool-wide transferability.
This tool is that draw, made reproducible and auditable:

    exclusions (taught ∪ rev-4-pending ∪ held-out)  →  seeded shuffle  →  local validation
        →  copy into data/teams/sample/  →  PROMOTION_MANIFEST.{md,json}

Usage:
  python -m main.promote_teams --dry-run                 # plan only, touches nothing
  python -m main.promote_teams --draw-only --seed 1234   # manifest only, no promotion
  python -m main.promote_teams --seed 1234               # the real thing
  python -m main.promote_teams --verify-exclusions       # re-derive exclusions from run metadata
  python -m main.promote_teams --regenerate-exclusions   # REWRITE them from run metadata

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

THREE things here are load-bearing and easy to get wrong:

* **The pool universe is the MANIFESTS, not the .txt files.** ``TeamLoader`` loads only teams listed
  in a ``teams.json``, deduped by resolved path. So promoting = copying the file into ``sample/``,
  ADDING it to ``sample/teams.json``, and REMOVING it from the source manifest. Skip that last step
  and the team is loaded twice — once as `sample`, once as `other` — which doubles exactly the fleet
  teams' opponent-draw weight. That is the ``yak_attack`` 66%-of-draws defect, re-created on the
  very teams the experiment measures. The source ``.txt`` is left on disk (nothing else reads it);
  only the manifest entry moves, so the change is reversible from the manifest alone.
* **An invalid team is REPLACED, never dropped** — by the next candidate in the same seeded shuffle,
  and the replacement is recorded. Dropping would silently shrink the fleet.
* **A validator that is broken reports every team as invalid**, which is indistinguishable from a
  pool of 693 bad teams. A known-good positive control rides in every batch; if it fails, this
  aborts instead of "replacing" the entire draw.

A FOURTH thing, learned the expensive way (2026-08-31): **the exclusion artifact ROTS.** It was
built from FROZEN ARGV FILES in a session-scoped job directory, before the runs they describe had
launched — and the launched runs did not deal the same teams those argvs did. Its ``rev4_pending``
block was stale on all three arms: it named 4 teams rev-4 never pinned and missed 4 it did, while
the union SIZE stayed 26, so no count-shaped check could see it. ``--verify-exclusions`` detects
that; ``--regenerate-exclusions`` REPAIRS it, from each named run's own ``metadata.json`` and
nothing else. A frozen argv is a plan; ``metadata.json`` is what ran.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import secrets
import shutil
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from agents.training.team_archetypes import load_team_archetypes, team_sha
from utils.paths import repo_path, repo_root

MANIFEST_JSON = "PROMOTION_MANIFEST.json"
MANIFEST_MD = "PROMOTION_MANIFEST.md"
DEFAULT_EXCLUSIONS = os.path.join("designs", "ai_v12", "promotion_exclusions.json")
DEFAULT_FORMAT = "gen3ou"
DEFAULT_N = 40

#: substrings that mark a ``validate_teams_locally`` result as an INFRASTRUCTURE failure rather than
#: a verdict about the team. The function returns the same ``{"valid": False}`` shape for both.
_INFRA_MARKERS = ("Node execution error", "Cannot find the Node bridge script",
                  "Subprocess failure", "No output from Node script", "cannot find module")


# ── the pool ────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoolTeam:
    """One team as the LOADER sees it — the manifest entry is the identity, the file is the payload."""
    sha: str
    text: str
    rel_path: str          # repo-relative, e.g. data/teams/others/giraffe/abc.txt
    manifest: str          # repo-relative path of the teams.json that lists it
    category: str          # "sample" | "other" — TeamLoader's own rule: "sample" in the manifest dir
    entry: Dict[str, Any]  # the manifest entry verbatim


@contextmanager
def _chdir(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_pool(root: str) -> Dict[str, PoolTeam]:
    """Every team ``TeamLoader`` would load, keyed by ``team_sha``, but carrying its FILE PATH.

    A verbatim mirror of ``utils.team_loader.TeamLoader._load_teams`` (manifest walk, ``valid: False``
    skip, dedup by resolved path with first-occurrence-wins) — ``TeamLoader`` returns only text, and
    promotion needs to know which file and which manifest a team came from. ``_cross_check_pool``
    proves the mirror still agrees at runtime, so a drift is caught rather than assumed away.
    """
    teams_dir = os.path.join(root, "data", "teams")
    pool: Dict[str, PoolTeam] = {}
    seen: set = set()
    for cur, _dirs, files in sorted(os.walk(teams_dir)):
        if "teams.json" not in files:
            continue
        json_path = os.path.join(cur, "teams.json")
        with open(json_path) as fh:
            meta = json.load(fh)
        for entry in meta:
            if entry.get("valid") is False:
                continue
            rel = entry.get("file")
            if not rel:
                continue
            full = os.path.join(root, "data", rel)
            if not os.path.exists(full):
                continue
            resolved = os.path.realpath(full)
            if resolved in seen:
                continue
            seen.add(resolved)
            with open(full) as fh:
                text = fh.read().strip()
            pool[team_sha(text)] = PoolTeam(
                sha=team_sha(text), text=text,
                rel_path=os.path.relpath(full, root), manifest=os.path.relpath(json_path, root),
                category="sample" if "sample" in cur else "other", entry=entry)
    return pool


def _cross_check_pool(root: str, pool: Dict[str, PoolTeam]) -> None:
    """The universe this tool draws from must be exactly the one the samplers draw from."""
    from utils.team_loader import TeamLoader
    with _chdir(root):
        loader_shas = {team_sha(t) for t in TeamLoader().get_all_teams()}
    if loader_shas != set(pool):
        only_loader = sorted(loader_shas - set(pool))[:5]
        only_here = sorted(set(pool) - loader_shas)[:5]
        raise RuntimeError(
            f"pool mirror DRIFT: TeamLoader sees {len(loader_shas)} teams, load_pool sees "
            f"{len(pool)} (loader-only e.g. {only_loader}, here-only e.g. {only_here}). "
            "load_pool must mirror TeamLoader._load_teams exactly — fix it before drawing.")


def _cross_check_archetypes(pool: Dict[str, PoolTeam]) -> Dict[str, Any]:
    """The archetype artifact must cover the pool exactly (S1's 719/719 check, re-run every draw)."""
    arch = load_team_archetypes()["teams"]
    missing = sorted(set(pool) - set(arch))
    if missing:
        raise RuntimeError(
            f"{len(missing)} pool teams have no entry in data/teams/gen3_team_archetypes.json "
            f"(e.g. {missing[:5]}) — regenerate it with `python -m agents.training.team_archetypes` "
            "before drawing; the manifest reports archetype composition and cannot do so blind.")
    return arch


# ── exclusions ──────────────────────────────────────────────────────────────────────────────────

@dataclass
class Exclusions:
    union: List[str]
    counts: Dict[str, int]
    path: str
    raw: Dict[str, Any]

    def as_set(self) -> set:
        return set(self.union)


def load_exclusions(path: str) -> Exclusions:
    with open(path) as fh:
        raw = json.load(fh)
    cats = raw["categories"]
    union = sorted({s for c in cats.values() for s in c["shas"]})
    if set(union) != set(raw.get("union", union)):
        raise RuntimeError(f"{path}: the recorded `union` disagrees with the per-category shas")
    return Exclusions(union=union, counts={k: len(v["shas"]) for k, v in cats.items()},
                      path=path, raw=raw)


@dataclass(frozen=True)
class ArmProvenance:
    """What ONE named run actually recorded — the authority a category block is checked against."""
    category: str
    tag: str
    run: str
    present: bool                       # is there a run dir with metadata to read?
    files: List[str] = field(default_factory=list)   # recorded --trainee-teams, verbatim
    shas: List[str] = field(default_factory=list)    # team_sha of each, sorted


def recorded_provenance(excl: Exclusions, models_dir: Optional[str],
                        root: Optional[str] = None) -> List[ArmProvenance]:
    """Read every RUN named by the artifact and return what its ``metadata.json`` actually says.

    THE one derivation. ``verify_exclusions`` (report), ``regenerate_exclusions`` (repair) and the
    drift test all consume this, so a check and a repair can never disagree about what "recorded"
    means. Recorded team paths are repo-relative, so the read happens with cwd at the repo root.

    A run that has no run dir (an argv frozen before launch) yields ``present=False`` and no shas —
    UNVERIFIABLE, never a mismatch and never an empty-set "repair".
    """
    from agents.training.matchup_spec import read_recorded_trainee_teams
    out: List[ArmProvenance] = []
    if not models_dir:
        return out
    with _chdir(root or str(repo_root())):
        for cat, blob in excl.raw["categories"].items():
            for tag, rec in sorted((blob.get("runs") or {}).items()):
                run_dir = os.path.join(models_dir, rec["run"])
                if not os.path.isdir(run_dir):
                    out.append(ArmProvenance(cat, tag, rec["run"], present=False))
                    continue
                files = read_recorded_trainee_teams(run_dir)
                out.append(ArmProvenance(
                    cat, tag, rec["run"], present=True, files=list(files),
                    shas=sorted(team_sha(open(f).read()) for f in files if os.path.exists(f))))
    return out


def exclusion_drift(excl: Exclusions, prov: Sequence[ArmProvenance]) -> List[Dict[str, Any]]:
    """The per-arm disagreements between the committed artifact and recorded provenance.

    Each row NAMES the offending team ids in both directions — a count-shaped answer would have
    missed the 2026-08-31 defect outright, where 4 teams went out and 4 came in and the union
    stayed 26.
    """
    rows: List[Dict[str, Any]] = []
    for a in prov:
        if not a.present:
            continue
        want = sorted(excl.raw["categories"][a.category]["runs"][a.tag]["shas"])
        if want == a.shas:
            continue
        rows.append({"category": a.category, "arm": a.tag, "run": a.run,
                     "artifact": want, "metadata": a.shas,
                     "in_artifact_never_pinned": sorted(set(want) - set(a.shas)),
                     "pinned_but_missing": sorted(set(a.shas) - set(want))})
    return rows


def verify_exclusions(excl: Exclusions, models_dir: Optional[str],
                      root: Optional[str] = None) -> int:
    """Re-derive every category that names a RUN from that run's own ``metadata.json``.

    The artifact was built from frozen argv files in a session-scoped job directory; this is the
    durable cross-check against the only copy that outlives it. A run that has not been launched
    yet has no metadata and is reported as UNVERIFIABLE, never as a mismatch.
    """
    if not models_dir:
        print("⚠️  no models/ archive on this box (utils.paths.main_models_dir() is None) — "
              "the run-metadata cross-check cannot run here. The artifact is unchanged.")
        return 0
    prov = recorded_provenance(excl, models_dir, root)
    drift = {(d["category"], d["arm"]): d for d in exclusion_drift(excl, prov)}
    for cat, blob in excl.raw["categories"].items():
        if not (blob.get("runs") or {}):
            print(f"  {cat:24s} {len(blob['shas']):3d} shas — no run to verify against "
                  f"({blob['reason'].split('—')[0].strip()})")
    for a in prov:
        if not a.present:
            print(f"  {a.category:24s} {a.tag:6s} UNVERIFIABLE — {a.run} has not been launched")
        elif (a.category, a.tag) not in drift:
            print(f"  {a.category:24s} {a.tag:6s} ✓ {len(a.shas)} teams match {a.run}/metadata.json")
        else:
            d = drift[(a.category, a.tag)]
            print(f"  {a.category:24s} {a.tag:6s} ✗ MISMATCH vs {a.run}/metadata.json\n"
                  f"      artifact: {d['artifact']}\n      metadata: {d['metadata']}\n"
                  f"      never pinned: {d['in_artifact_never_pinned']}   "
                  f"missing: {d['pinned_but_missing']}")
    return len(drift)


def regenerate_exclusions(excl: Exclusions, models_dir: Optional[str], pool_total: int,
                          stamp: str, root: Optional[str] = None) -> Dict[str, Any]:
    """Rebuild the artifact from recorded run provenance. Returns the NEW blob (does not write).

    Only the run-derived halves are rebuilt. A category with no ``runs`` (``held_out_instruments``
    — held out by DESIGN, not by having been trained) is carried verbatim, because no metadata
    exists that could confirm or deny it; regenerating it from nothing would silently empty it.
    An unlaunched run keeps its recorded block for the same reason.
    """
    if not models_dir:
        raise RuntimeError(
            "no models/ archive on this box (utils.paths.main_models_dir() is None) — the exclusion "
            "artifact is DERIVED from run metadata and cannot be regenerated without it. "
            "Set $GEN3AI_MODELS_DIR, or run this on the box that holds the runs.")
    prov = {(a.category, a.tag): a for a in recorded_provenance(excl, models_dir, root)}
    new = json.loads(json.dumps(excl.raw))          # deep copy; never mutate the loaded artifact
    for cat, blob in new["categories"].items():
        for tag, rec in (blob.get("runs") or {}).items():
            a = prov[(cat, tag)]
            rec["run_dir_present"] = a.present
            rec["metadata_verified"] = a.present
            if a.present:
                rec["teams"], rec["shas"] = a.files, a.shas
        if blob.get("runs"):
            blob["shas"] = sorted({s for r in blob["runs"].values() for s in r["shas"]})
    union = sorted({s for c in new["categories"].values() for s in c["shas"]})
    new["union"] = union
    from utils.git import get_git_hash
    new["_meta"] = dict(new["_meta"], generated_at=stamp, git_hash=get_git_hash(),
                        pool_total=pool_total, union_total=len(union),
                        eligible_after_exclusions=pool_total - len(union),
                        argv_source="DERIVED from models/<run>/metadata.json cli_args.trainee_teams "
                                    "by `python -m main.promote_teams --regenerate-exclusions`. The "
                                    "frozen argvs this file was FIRST built from are a plan, not a "
                                    "record — they disagreed with what ran (see the rev4_pending "
                                    "repair, 2026-08-31).")
    return new


# ── the draw ────────────────────────────────────────────────────────────────────────────────────

@dataclass
class DrawResult:
    seed: int
    n: int
    accepted: List[str] = field(default_factory=list)
    replacements: List[Dict[str, Any]] = field(default_factory=list)
    eligible_count: int = 0
    considered: int = 0


Validator = Callable[[List[str]], List[Dict[str, Any]]]


def make_validator(fmt: str = DEFAULT_FORMAT, control: Optional[str] = None) -> Validator:
    """``validate_teams_locally`` with a POSITIVE CONTROL riding in every batch.

    It returns ``{"valid": False}`` for a broken node bridge just as it does for an illegal team, so
    a missing ``deps/pokemon-showdown/dist`` would silently "replace" all 693 eligible teams and draw
    nothing. A team known to be legal is validated alongside every batch; if it fails, the validator
    is broken and this raises.
    """
    from utils.bridge.team_validator import validate_teams_locally

    def _validate(teams: List[str]) -> List[Dict[str, Any]]:
        if not teams:
            return []
        batch = list(teams) + ([control] if control else [])
        out = validate_teams_locally(fmt, batch)
        if len(out) != len(batch):
            raise RuntimeError(f"team validator returned {len(out)} results for {len(batch)} teams")
        for r in out:
            for e in r.get("errors") or []:
                if any(m.lower() in str(e).lower() for m in _INFRA_MARKERS):
                    raise RuntimeError(
                        f"the local team validator is BROKEN, not the teams: {e!r}. "
                        "Check node and deps/pokemon-showdown (in a fresh worktree: "
                        "`git submodule update --init` + the dist/node_modules symlinks).")
        if control:
            ctl = out[-1]
            if not ctl.get("valid"):
                raise RuntimeError(
                    "the positive control (a curated sample team) FAILED validation "
                    f"({ctl.get('errors')}) — the validator is broken; refusing to call the pool bad.")
            out = out[:-1]
        return out

    return _validate


def draw_teams(pool: Dict[str, PoolTeam], excluded: Iterable[str], n: int, seed: int,
               validator: Optional[Validator]) -> DrawResult:
    """A seeded shuffle of the eligible pool; walk it and keep the first ``n`` that validate.

    ``random.Random(seed)`` per call — never the module-level ``random``, whose stream any other
    import can perturb. The eligible list is SORTED before the shuffle so the permutation is a
    function of the seed alone.
    """
    excluded = set(excluded)
    eligible = sorted(sha for sha in pool if sha not in excluded)
    if n > len(eligible):
        raise ValueError(f"asked for {n} teams but only {len(eligible)} are eligible "
                         f"({len(pool)} pool − {len(excluded & set(pool))} excluded)")
    order = list(eligible)
    random.Random(seed).shuffle(order)

    res = DrawResult(seed=seed, n=n, eligible_count=len(eligible))
    cursor = 0
    while len(res.accepted) < n and cursor < len(order):
        need = n - len(res.accepted)
        chunk = order[cursor:cursor + need]
        cursor += len(chunk)
        res.considered += len(chunk)
        if validator is None:
            res.accepted.extend(chunk)
            continue
        verdicts = validator([pool[s].text for s in chunk])
        for sha, v in zip(chunk, verdicts):
            if v.get("valid"):
                res.accepted.append(sha)
            else:
                res.replacements.append({
                    "rejected_sha": sha, "path": pool[sha].rel_path,
                    "draw_position": res.considered - len(chunk) + chunk.index(sha),
                    "errors": list(v.get("errors") or [])})
    if len(res.accepted) < n:
        raise RuntimeError(f"exhausted {len(order)} eligible teams with only {len(res.accepted)} "
                           f"valid — {len(res.replacements)} failed validation")
    return res


# ── promotion ───────────────────────────────────────────────────────────────────────────────────

@dataclass
class Action:
    sha: str
    kind: str              # "copy" | "already_curated" | "already_promoted"
    src: str
    dest: str
    source_manifest: Optional[str]


def plan_promotion(root: str, pool: Dict[str, PoolTeam], accepted: Sequence[str]) -> List[Action]:
    sample_dir = os.path.join(root, "data", "teams", "sample")
    actions: List[Action] = []
    for sha in accepted:
        t = pool[sha]
        if t.category == "sample":
            actions.append(Action(sha, "already_curated", t.rel_path, t.rel_path, None))
            continue
        dest = os.path.join(sample_dir, f"{sha}.txt")
        rel_dest = os.path.relpath(dest, root)
        if os.path.exists(dest):
            existing = open(dest).read()
            if team_sha(existing) != sha:
                raise RuntimeError(
                    f"REFUSING to overwrite {rel_dest}: it exists and holds a DIFFERENT team "
                    f"(sha {team_sha(existing)}, wanted {sha}). Move it aside and re-run.")
            actions.append(Action(sha, "already_promoted", t.rel_path, rel_dest, t.manifest))
            continue
        actions.append(Action(sha, "copy", t.rel_path, rel_dest, t.manifest))
    return actions


def apply_promotion(root: str, pool: Dict[str, PoolTeam], actions: Sequence[Action],
                    arch: Dict[str, Any], seed: int, stamp: str) -> None:
    """Copy the files, then MOVE each team's manifest entry from its source into sample/teams.json."""
    sample_manifest = os.path.join(root, "data", "teams", "sample", "teams.json")
    with open(sample_manifest) as fh:
        sample_meta = json.load(fh)
    have = {e.get("file") for e in sample_meta}

    by_source: Dict[str, List[Action]] = {}
    for a in actions:
        if a.kind == "already_curated":
            continue
        shutil.copyfile(os.path.join(root, a.src), os.path.join(root, a.dest))
        rel = a.dest.split("data/", 1)[1]           # teams.json `file` is relative to data/
        if rel not in have:
            src_entry = pool[a.sha].entry
            sample_meta.append({
                "id": a.sha,
                "name": src_entry.get("name") or f"promoted {a.sha}",
                "format": src_entry.get("format", DEFAULT_FORMAT),
                "category": (arch[a.sha]["archetype"] if a.sha in arch else "unknown"),
                "valid": True, "errors": [],
                "file": rel,
                "source": src_entry.get("source") or src_entry.get("url"),
                "promoted": {"from": a.src, "seed": seed, "at": stamp,
                             "by": "python -m main.promote_teams"},
            })
        if a.source_manifest:
            by_source.setdefault(a.source_manifest, []).append(a)

    # ORDER MATTERS: ADD to sample/teams.json first, DE-LIST from the sources after. A crash between
    # the two then leaves a team in BOTH manifests — a duplicate, which `check_invariants` shouts
    # about. The other order leaves it in NEITHER, which is a team silently gone from the pool.
    _write_json(sample_manifest, sample_meta)
    for manifest_rel, group in by_source.items():
        path = os.path.join(root, manifest_rel)
        with open(path) as fh:
            meta = json.load(fh)
        drop = {pool[a.sha].entry.get("file") for a in group}
        kept = [e for e in meta if e.get("file") not in drop]
        _write_json(path, kept)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as fh:
        fh.write(json.dumps(obj, indent=2))


def check_invariants(root: str, expect_sample: int, expect_total: int) -> Dict[str, int]:
    """Re-load through ``TeamLoader`` and prove the promotion moved teams instead of copying them.

    The failure this exists for is silent: a promoted team left in BOTH manifests is loaded twice and
    draws twice as often as its neighbours — on exactly the teams the fleet measures.
    """
    from utils.team_loader import TeamLoader
    with _chdir(root):
        loader = TeamLoader()
    allt, samp = loader.get_all_teams(), loader.get_sample_teams()
    shas = [team_sha(t) for t in allt]
    dupes = [s for s, c in Counter(shas).items() if c > 1]
    if dupes:
        raise RuntimeError(f"{len(dupes)} team(s) are now loaded TWICE (e.g. {dupes[:5]}) — a promoted "
                           "team was left in its source manifest, doubling its draw weight.")
    if len(allt) != expect_total or len(samp) != expect_sample:
        raise RuntimeError(f"post-promotion counts wrong: {len(allt)} total (want {expect_total}), "
                           f"{len(samp)} sample (want {expect_sample})")
    return {"total": len(allt), "sample": len(samp), "other": len(loader.get_other_teams())}


# ── manifest ────────────────────────────────────────────────────────────────────────────────────

def _source_folder(rel_path: str) -> str:
    """`data/teams/others/giraffe/x.txt` -> `giraffe`; `data/teams/sample/x.txt` -> `sample`.

    The AUTHOR folder, not `others` — a fleet drawn deep into one author inherits that author's
    building habits as surely as it would inherit one archetype (S1 §"source-folder spread").
    """
    parts = rel_path.replace(os.sep, "/").split("/")
    return parts[3] if len(parts) > 3 and parts[2] == "others" else parts[2]


def build_manifest(res: DrawResult, pool: Dict[str, PoolTeam], arch: Dict[str, Any],
                   excl: Exclusions, actions: Sequence[Action], stamp: str,
                   fmt: str, validated: bool) -> Dict[str, Any]:
    rows = []
    for i, sha in enumerate(res.accepted):
        a = next(x for x in actions if x.sha == sha)
        rows.append({"rank": i + 1, "sha": sha, "source": pool[sha].rel_path, "dest": a.dest,
                     "action": a.kind, "folder": _source_folder(pool[sha].rel_path),
                     "archetype": arch.get(sha, {}).get("archetype", "unknown"),
                     "tags": arch.get(sha, {}).get("tags", [])})
    from utils.git import get_git_hash
    return {
        "_meta": {
            "tool": "python -m main.promote_teams",
            "generated_at": stamp,
            "git_hash": get_git_hash(),
            "key_convention": "sha1(team_text.strip())[:10] — agents.training.team_archetypes.team_sha "
                              "(STRIPPED; the unstripped variant is a recorded derived-key defect)",
            "selection": "UNIFORM RANDOM over the eligible pool (owner ruling, ledger 2026-08-30) — "
                         "not ranked, not curated. Archetype composition below is REPORTED, "
                         "never corrected; correcting it would reintroduce the selection confound.",
            "validated": validated, "format": fmt,
        },
        "seed": res.seed, "n_requested": res.n,
        "pool_total": len(pool), "eligible": res.eligible_count, "considered": res.considered,
        "exclusions": {"source": excl.path, "total": len(excl.union), "by_category": excl.counts,
                       "shas": excl.union},
        "draw": rows,
        "replacements": res.replacements,
        "archetype_composition": dict(Counter(r["archetype"] for r in rows).most_common()),
        "folder_composition": dict(Counter(r["folder"] for r in rows).most_common()),
        "action_composition": dict(Counter(r["action"] for r in rows).most_common()),
    }


def render_manifest_md(m: Dict[str, Any]) -> str:
    L = [f"# Team promotion — {m['n_requested']} teams, seed `{m['seed']}`", "",
         f"Generated {m['_meta']['generated_at']} at `{m['_meta']['git_hash'][:10]}` "
         f"by `{m['_meta']['tool']}`.", "",
         "**Selection is UNIFORM RANDOM over the eligible pool** (owner ruling, ledger 2026-08-30) "
         "— not ranked, not curated. That is what makes the fleet's result an unbiased estimate of "
         "pool-wide transferability. Archetype and folder composition below are **REPORTED, never "
         "corrected**; correcting them would put the selection confound back.", "",
         "| | |", "|---|---|",
         f"| pool | {m['pool_total']} |",
         f"| exclusions | {m['exclusions']['total']} |",
         f"| eligible | {m['eligible']} |",
         f"| drawn | {m['n_requested']} |",
         f"| candidates considered | {m['considered']} |",
         f"| replaced (failed validation) | {len(m['replacements'])} |",
         f"| validated | {m['_meta']['validated']} (`{m['_meta']['format']}`) |", "",
         "Keys are `sha1(team_text.strip())[:10]` — the strip-normalized convention shared by "
         "`team_archetypes.team_sha`, `MatchupSpec` pins and `TeamWinRateCallback`. "
         "(An unstripped `sha1(text)` is a recorded derived-key defect and is NOT used here.)", "",
         "## Exclusions applied (before the draw)", "", "| category | teams |", "|---|---|"]
    for k, v in m["exclusions"]["by_category"].items():
        L.append(f"| `{k}` | {v} |")
    L += [f"| **union (deduped)** | **{m['exclusions']['total']}** |", "",
          f"Source: `{m['exclusions']['source']}`.", "",
          "## The draw", "", "| # | sha | source | action | archetype | folder |", "|---|---|---|---|---|---|"]
    for r in m["draw"]:
        L.append(f"| {r['rank']} | `{r['sha']}` | `{r['source']}` | {r['action']} | "
                 f"{r['archetype']} | {r['folder']} |")
    L += ["", "## Replacements", ""]
    if m["replacements"]:
        L += ["A drawn team that fails local validation is REPLACED by the next candidate in the "
              "same seeded shuffle — never silently dropped.", "",
              "| rejected sha | draw position | errors |", "|---|---|---|"]
        for r in m["replacements"]:
            L.append(f"| `{r['rejected_sha']}` | {r['draw_position']} | {'; '.join(r['errors'])[:160]} |")
    else:
        L.append("None — every drawn team validated on the first pass.")
    L += ["", "## Composition (REPORTED, not corrected)", "", "| archetype | n |", "|---|---|"]
    for k, v in m["archetype_composition"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "| source folder | n |", "|---|---|"]
    for k, v in m["folder_composition"].items():
        L.append(f"| `{k}` | {v} |")
    L += ["", "A random draw reproduces the pool's own composition in expectation. A skew here is a "
          "property of the draw, and rebalancing it would put the selection confound back.", ""]
    return "\n".join(L)


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.promote_teams", description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=DEFAULT_N, help=f"how many teams to draw (default {DEFAULT_N})")
    p.add_argument("--seed", type=int, default=None,
                   help="the draw seed. Omitted = minted from os entropy and RECORDED; a manifest "
                        "always names the seed that produced it.")
    p.add_argument("--exclusions", default=None,
                   help=f"exclusion artifact (default {DEFAULT_EXCLUSIONS})")
    p.add_argument("--format", default=DEFAULT_FORMAT, help="validation format (default gen3ou)")
    p.add_argument("--dry-run", action="store_true", help="print the whole plan; touch nothing")
    p.add_argument("--draw-only", action="store_true",
                   help="write the manifest but do NOT promote (review-then-promote)")
    p.add_argument("--manifest-dir", default=None,
                   help="where the manifest is written (default data/teams/sample/)")
    p.add_argument("--no-validate", action="store_true",
                   help="skip local validation (a DRAW-SHAPE check only — never for a real promotion)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing manifest that records a different seed")
    p.add_argument("--verify-exclusions", action="store_true",
                   help="re-derive the exclusion artifact from run metadata and exit")
    p.add_argument("--regenerate-exclusions", action="store_true",
                   help="REWRITE the exclusion artifact from run metadata (the repair for the "
                        "check above) and exit. Prints the before/after team ids.")
    p.add_argument("--root", default=None,
                   help="operate on a COPY of the tree instead of this checkout — a full rehearsal "
                        "of the real promotion (copies, manifest surgery, invariant check) with "
                        "nothing at stake. Default: the repo root.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root) if args.root else str(repo_root())
    excl_path = args.exclusions or repo_path(*DEFAULT_EXCLUSIONS.split(os.sep))
    excl = load_exclusions(str(excl_path))

    if args.verify_exclusions or args.regenerate_exclusions:
        from utils.paths import main_models_dir
        md = main_models_dir()
        print(f"Re-deriving {excl.path} from run metadata"
              f"{f' under {md}' if md else ''}:\n")
        bad = verify_exclusions(excl, str(md) if md else None, root)
        print(f"\n{'✗ ' + str(bad) + ' MISMATCH(ES)' if bad else '✓ no mismatches'}"
              f" — union {len(excl.union)} teams: {excl.counts}")
        if not args.regenerate_exclusions:
            return 1 if bad else 0

        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        pool = load_pool(root)
        _cross_check_pool(root, pool)
        new = regenerate_exclusions(excl, str(md) if md else None, len(pool), stamp, root)
        before, after = set(excl.union), set(new["union"])
        print(f"\nREGENERATED — union {len(before)} → {len(after)} teams, "
              f"eligible {len(pool) - len(before)} → {len(pool) - len(after)} of {len(pool)}")
        print(f"  now excluded (were eligible): {sorted(after - before) or 'none'}")
        print(f"  now eligible (were excluded): {sorted(before - after) or 'none'}")
        if before == after and not bad:
            print("  (byte-identical membership — the artifact was already correct)")
        _write_json(excl.path, new)
        print(f"wrote {excl.path}")
        return 0

    if args.dry_run and args.draw_only:
        print("--dry-run and --draw-only are mutually exclusive (dry-run writes nothing).")
        return 2

    seed = args.seed if args.seed is not None else secrets.randbelow(2 ** 31)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    manifest_dir = args.manifest_dir or os.path.join(root, "data", "teams", "sample")

    pool = load_pool(root)
    _cross_check_pool(root, pool)
    arch = _cross_check_archetypes(pool)
    print(f"pool {len(pool)} teams (TeamLoader-verified, {len(arch)} archetype rows) · "
          f"exclusions {len(excl.union)} {excl.counts} · eligible "
          f"{len(pool) - len(excl.as_set() & set(pool))}")

    validator = None
    if not args.no_validate:
        control = next((t.text for t in pool.values() if t.category == "sample"), None)
        validator = make_validator(args.format, control)
    else:
        print("⚠️  --no-validate: this draw is a SHAPE CHECK, not a promotion-grade draw.")

    res = draw_teams(pool, excl.as_set(), args.n, seed, validator)
    actions = plan_promotion(root, pool, res.accepted)
    man = build_manifest(res, pool, arch, excl, actions, stamp, args.format, not args.no_validate)

    print(f"\nseed {seed} · drew {len(res.accepted)} of {res.eligible_count} eligible "
          f"(considered {res.considered}, {len(res.replacements)} replaced)")
    print(f"archetypes {man['archetype_composition']}\nfolders    {man['folder_composition']}"
          f"\nactions    {man['action_composition']}\n")
    for r in man["draw"]:
        print(f"  {r['rank']:3d}  {r['sha']}  {r['archetype']:14s} {r['action']:16s} {r['source']}")
    for r in res.replacements:
        print(f"  REPLACED {r['rejected_sha']} @{r['draw_position']}: {'; '.join(r['errors'])[:120]}")

    if args.dry_run:
        print(f"\n--dry-run: nothing written. The manifest would go to "
              f"{os.path.relpath(manifest_dir, root)}/{MANIFEST_MD}")
        return 0

    mpath = os.path.join(manifest_dir, MANIFEST_JSON)
    if os.path.exists(mpath) and not args.force:
        with open(mpath) as fh:
            prev = json.load(fh)
        if prev.get("seed") != seed or [r["sha"] for r in prev.get("draw", [])] != res.accepted:
            print(f"\nREFUSING: {mpath} already records a DIFFERENT draw (seed {prev.get('seed')}). "
                  "A promotion is pre-registered by its manifest; re-rolling the seed until the "
                  "composition looks good is the selection confound this tool exists to avoid. "
                  "Pass --force if you really mean to replace it.")
            return 1

    os.makedirs(manifest_dir, exist_ok=True)
    _write_json(mpath, man)
    with open(os.path.join(manifest_dir, MANIFEST_MD), "w") as fh:
        fh.write(render_manifest_md(man))
    shown = os.path.relpath(mpath, root)
    print(f"\nwrote {mpath if shown.startswith('..') else shown} + {MANIFEST_MD}")

    if args.draw_only:
        print("--draw-only: manifest written, no teams promoted. Re-run without --draw-only "
              f"(and with --seed {seed}) to promote.")
        return 0

    n_new = sum(1 for a in actions if a.kind == "copy")
    before = len(pool), sum(1 for t in pool.values() if t.category == "sample")
    apply_promotion(root, pool, actions, arch, seed, stamp)
    counts = check_invariants(root, expect_sample=before[1] + n_new, expect_total=before[0])
    print(f"promoted {n_new} team(s) into data/teams/sample/ — TeamLoader now sees "
          f"{counts['total']} total / {counts['sample']} sample / {counts['other']} other, no duplicates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
