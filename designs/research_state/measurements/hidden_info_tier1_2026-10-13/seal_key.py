"""Write the SEALED arm key (§12.10): ``<archive>/arm_key.json`` (NOT committed until every arm output exists) and
this folder's ``arm_key.sha256`` (committed with the scripts, BEFORE any arm output exists). ``--unseal`` copies the
archived key into this folder after checking it against the committed hash.

    python -I seal_key.py [--unseal]
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseal", action="store_true")
    a = ap.parse_args()
    sha_file = T.HERE / "arm_key.sha256"
    if a.unseal:
        b = T.key_path().read_bytes()
        want = sha_file.read_text().split()[0]
        got = hashlib.sha256(b).hexdigest()
        if got != want:
            raise SystemExit(f"the archived key's sha256 {got} != the committed {want}")
        (T.HERE / "arm_key.json").write_bytes(b)
        print("unsealed:", T.HERE / "arm_key.json")
        return
    b = T.key_bytes()
    T.key_path().parent.mkdir(parents=True, exist_ok=True)
    if T.key_path().exists() and T.key_path().read_bytes() != b:
        raise SystemExit(f"{T.key_path()} exists with different bytes")
    T.key_path().write_bytes(b)
    sha_file.write_text(hashlib.sha256(b).hexdigest() + "  arm_key.json (sealed in the archive)\n")
    print(sha_file.read_text().strip())


if __name__ == "__main__":
    main()
