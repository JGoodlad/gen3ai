import subprocess


def get_git_hash(short: bool = False) -> str:
    """Return the current git HEAD hash, or 'unknown' if git is unavailable."""
    args = ["git", "rev-parse", "HEAD"] if not short else ["git", "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_repo_root() -> str:
    """Return the absolute path of the git repo root."""
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
    ).strip()
