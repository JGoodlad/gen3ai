"""The cannot-rewrite report: refs, worktrees, sidecar scans by JSON field, and the rendered sections."""
from __future__ import annotations

import json

from utils.history_rewrite import cite as C
from utils.history_rewrite import report as R
from utils.history_rewrite.synthetic_repo_testkit import git, make_repo

OLD = "abcdef1" + "0" * 33
NEW = "9876543" + "1" * 33
RW = C.Rewriter({OLD: NEW}, C.CommitIndex([OLD, "bcdef12" + "2" * 33]), C.CommitIndex([NEW]))


def test_scan_attributes_citations_to_json_fields_and_ignores_the_rest(tmp_path):
    run = tmp_path / "runA"
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "git_hash": OLD, "original_command": f"python x --pin {OLD[:8]}", "pin_history": [{"sha": OLD[:12]}, {"sha": OLD[:12]}],
        "other": "bcdef12" + "2" * 33}))
    (run / "model_config.json").write_text(json.dumps({"fine": 1}))
    (run / "checkpoints" / "c.json").write_text("not json " + OLD[:9])
    (tmp_path / "runA" / "blob.bin").write_bytes(b"\0" + OLD.encode())
    spec = R.ScanSpec("models", str(tmp_path), ("*/metadata.json", "*/model_config.json", "*/checkpoints/*.json"))
    s = R.scan(spec, RW)
    assert s.files_scanned == 3 and s.files_hit == 2
    assert s.by_field["git_hash"] == 1 and s.by_field["original_command"] == 1 and s.by_field["pin_history[].sha"] == 2
    assert s.by_field["(unparseable json)"] == 1 and s.tokens_hit == 5 and s.commits == {OLD}
    text = "\n".join(R.render_scan(s))
    assert "5 citations" in text and "`pin_history[].sha`: 2" in text


def test_scan_of_a_missing_root_is_empty_not_an_error(tmp_path):
    assert R.scan(R.ScanSpec("x", str(tmp_path / "nope")), RW).files_scanned == 0


def test_refs_and_worktrees_say_what_moves(tmp_path):
    r = make_repo(tmp_path)
    git(r["path"], "branch", "keeps", r["root"])
    git(r["path"], "tag", f"pin-{r['intro'][:8]}", r["intro"])
    wt = tmp_path / f"pin_{r['cite'][:8]}"
    git(r["path"], "worktree", "add", "-q", "--detach", str(wt), r["cite"])
    git(r["path"], "worktree", "add", "-q", str(tmp_path / "safe"), "keeps")
    pairs = {r["intro"]: NEW, r["cite"]: "8" * 40, r["msg"]: "7" * 40}
    rw = C.Rewriter(pairs, C.CommitIndex([r["intro"], r["cite"], r["msg"], r["root"]]), C.CommitIndex([NEW]))
    refs = R.ref_rows(r["path"], pairs, rw)
    assert "refs/heads/main" in refs["moved"] and "refs/heads/keeps" in refs["unchanged"]
    assert "refs/tags/v-intro" in refs["moved"]                       # the annotated tag peels to a rewritten commit
    assert refs["tag names that embed an old SHA (the name is kept)"] == [f"refs/tags/pin-{r['intro'][:8]}"]
    rows = {w["worktree"]: w for w in R.worktree_rows(r["path"], pairs)}
    assert "primary checkout" in rows[r["path"]]["action"]
    assert rows[str(wt)]["action"] == "move in place: checkout --detach 888888888888" and rows[str(wt)]["embeds"] == [r["cite"][:8]]
    assert rows[str(tmp_path / "safe")]["action"].startswith("none")


def test_render_report_has_every_section_and_counts(tmp_path):
    tree = tmp_path / "t"
    tree.mkdir()
    import subprocess
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    (tree / "designs" / "research_state").mkdir(parents=True)
    (tree / "designs" / "research_state" / "ledger.md").write_text(f"entry {OLD[:8]}\n")
    (tree / "a.md").write_text(f"see {OLD[:8]} and pin_{OLD[:8]}\n")
    subprocess.run(["git", "-C", str(tree), "add", "-A"], check=True)
    res = C.rewrite_tree(str(tree), RW, apply=False)
    text = R.render_report(res, refs={"moved": ["refs/heads/main"]}, worktrees=[], scans=[], mapped_commits=1)
    for needle in ("## 1. Tracked files left on old SHAs", "frozen (append-only", "`designs/research_state/ledger.md`: 1 rewritable",
                   "identifier-embedded", ": 1\n", "## 2. Refs", "## 3. Worktrees", "## 4. Files outside git", "## 5. Outside this machine",
                   "GitHub keeps every old commit"):
        assert needle in text, needle


def test_worktree_busy_sees_a_process_inside_and_a_models_symlink_into_it(tmp_path):
    import os
    import subprocess
    wt = tmp_path / "wt"
    wt.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    os.symlink(wt, models / "run_x")
    os.symlink(tmp_path / "elsewhere", models / "run_y")
    sleeper = subprocess.Popen(["sleep", "30"], cwd=wt)
    try:
        reasons = R.worktree_busy(str(wt), str(models))
    finally:
        sleeper.kill()
        sleeper.wait()
    assert f"pid {sleeper.pid} has its cwd inside" in reasons and "models/run_x links into it" in reasons
    assert not any("run_y" in r for r in reasons)
    assert R.worktree_busy(str(wt), str(models)) == ["models/run_x links into it"]      # the sleeper is gone


def test_local_plan_moves_refs_with_a_lease_and_worktrees_in_place_and_never_removes_one(tmp_path):
    r = make_repo(tmp_path)
    git(r["path"], "branch", "feat", r["cite"])
    git(r["path"], "branch", "keeps", r["root"])
    pin = tmp_path / "pin"
    git(r["path"], "worktree", "add", "-q", "--detach", str(pin), r["cite"])
    git(r["path"], "worktree", "add", "-q", str(tmp_path / "bound"), "-b", "bound", r["intro"])
    pairs = {r["intro"]: "1" * 40, r["cite"]: "2" * 40, r["msg"]: "3" * 40}
    text = "\n".join(R.local_plan(r["path"], pairs, str(tmp_path / "models")))
    assert f"git update-ref refs/heads/feat {'2' * 40} {r['cite']}" in text and "refs/heads/keeps" not in text
    assert f"git -C {pin} checkout --detach {'2' * 40}" in text and f"git -C {tmp_path / 'bound'} reset --keep {'1' * 40}" in text
    assert f"git -C {r['path']} reset --keep {'3' * 40}" in text and "PRIMARY CHECKOUT" in text
    assert "update-ref refs/heads/bound" not in text            # a branch checked out in a worktree moves with that worktree
    assert "worktree remove" not in text and "rm -rf" not in text
    assert "PHASE 2" in text and "git reflog expire" in text and text.startswith("# generated by")


def test_push_plan_leases_every_moved_ref_on_its_old_value_and_pushes_nothing_local(tmp_path):
    import subprocess
    r = make_repo(tmp_path)
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    git(r["path"], "remote", "add", "origin", str(origin))
    git(r["path"], "push", "-q", "origin", "main", "side", "v-intro")
    git(r["path"], "fetch", "-q", "origin")
    git(r["path"], "update-ref", "refs/history-rewrite/keep/x", r["pin"])        # local bookkeeping: never pushed
    tag_obj = git(r["path"], "rev-parse", "refs/tags/v-intro")
    pairs = {r["intro"]: "1" * 40, r["cite"]: "2" * 40, r["msg"]: "3" * 40, r["pin"]: "4" * 40}
    text = "\n".join(R.push_plan(r["path"], pairs, "/work/clone"))
    assert f"--force-with-lease=refs/heads/main:{r['msg']}" in text and f"--force-with-lease=refs/tags/v-intro:{tag_obj}" in text
    assert "refs/heads/side" not in text and "history-rewrite" not in text and "--force " not in text
    assert text.count("git push --atomic") == 1 and f"git remote add origin {origin}" in text
    assert text.splitlines()[0].startswith("# generated by") and "NOT RUN BY ANY TOOL" in text


def test_a_ref_or_worktree_the_snapshot_never_saw_is_reported_as_stale_not_as_unchanged(tmp_path):
    r = make_repo(tmp_path)
    git(r["path"], "worktree", "add", "-q", "--detach", str(tmp_path / "late"), r["root"])
    pairs = {r["intro"]: "1" * 40}
    known = {r["root"], r["intro"]}                               # `cite`, `msg`, `side`... were created after the snapshot
    rw = C.Rewriter(pairs, C.CommitIndex([r["intro"], r["root"]]), C.CommitIndex([NEW]))
    refs = R.ref_rows(r["path"], pairs, rw, known)
    assert "refs/heads/main" in refs[R.STALE] and "refs/heads/main" not in refs["unchanged"]
    rows = {w["worktree"]: w for w in R.worktree_rows(r["path"], pairs, known=known)}
    assert rows[str(tmp_path / "late")]["action"].startswith("none") and rows[r["path"]]["action"] == R.STALE
