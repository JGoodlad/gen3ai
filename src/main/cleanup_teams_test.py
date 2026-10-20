"""`main.cleanup_teams` imports (it carried `src.` imports that only resolved from the repo root with `src` a package),
finds `data/` through `utils.paths`, and rewrites a `teams.json` to the valid, present entries."""
import json

from main import cleanup_teams as ct


def test_it_imports_as_main_cleanup_teams_and_defaults_to_this_checkouts_data():
    import inspect
    from utils.bridge import team_validator
    assert ct.validate_team_locally is team_validator.validate_team_locally
    assert inspect.signature(ct.cleanup_teams).parameters["data_root"].default is None
    assert "from src." not in inspect.getsource(ct) and "import src." not in inspect.getsource(ct)


def test_it_keeps_only_present_valid_teams_under_the_given_data_root(tmp_path):
    folder = tmp_path / "teams" / "sample"
    folder.mkdir(parents=True)
    (folder / "good.txt").write_text("GOOD")
    (folder / "bad.txt").write_text("BAD")
    entries = [{"name": n, "file": f"teams/sample/{n}.txt"} for n in ("good", "bad", "gone")]
    (folder / "teams.json").write_text(json.dumps(entries))
    ct.cleanup_teams(tmp_path, validate=lambda fmt, text: {"valid": text == "GOOD", "errors": ["illegal"]})
    assert [e["name"] for e in json.loads((folder / "teams.json").read_text())] == ["good"]
