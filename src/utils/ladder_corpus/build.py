"""(Re)build the LADDER-USAGE corpus from a Metamon team directory (`gen3_ladder_usage_corpus_v1`).

    python -m utils.ladder_corpus.build [--src DIR] [--check]

``--src`` defaults to the archived Metamon cache (``~/gen3ai_archive/metamon_cache_2026-09-24/
teams/hl_05_26/gen3ou``, ``teams`` revision v5, downloaded 2026-09-24). ``--check`` rebuilds in
memory and exits non-zero unless the result is byte-identical to the committed files.

The filter (see the package docstring): Showdown reads it, six Pokemon, ``TeamValidator('gen3ou')``
legal (``src/rust_sim/harness/ladder_corpus_filter.js``), and the ENGINE runs every one of its
species / moves / items / abilities (``scan_move_probe``, all four ``PROBE_KIND``s). Exact
duplicates (same packed string) keep their first file. Then one seeded permutation; the tiers are
its prefixes. The build is deterministic: the same source, engine and seed give the same bytes.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from utils.ladder_corpus import (COMMIT_N, DATA, MANIFEST, MILESTONE_N, SCHEMA, SEED, tier_sha256)
from utils.paths import repo_path

DEFAULT_SRC = Path.home() / "gen3ai_archive" / "metamon_cache_2026-09-24" / "teams" / "hl_05_26" / "gen3ou"
FILTER_JS = repo_path("src", "rust_sim", "harness", "ladder_corpus_filter.js")
CRATE = repo_path("src", "rust_sim")
KINDS = (("species", "species"), ("move", "moves"), ("item", "items"), ("ability", "abilities"))


def showdown_rows(src: Path) -> List[dict]:
    p = subprocess.run(["node", str(FILTER_JS), str(src)], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"ladder_corpus_filter.js failed: {p.stderr.strip()[-2000:]}")
    return [json.loads(line) for line in p.stdout.splitlines() if line.strip()]


def scan_probe_bin() -> str:
    """The engine's coverage oracle, built into THIS checkout's own target (never main's)."""
    subprocess.run(["cargo", "build", "--release", "--bin", "scan_move_probe"], cwd=CRATE,
                   check=True, capture_output=True)
    return str(CRATE / "target" / "release" / "scan_move_probe")


def engine_verdicts(rows: List[dict]) -> Dict[str, Dict[str, str]]:
    """``{kind: {id: verdict}}`` for every id any legal team carries; ``ran`` = playable."""
    binary = scan_probe_bin()
    out: Dict[str, Dict[str, str]] = {}
    for kind, key in KINDS:
        ids = sorted({i for r in rows if r["verdict"] == "ok" for i in r[key]})
        env = dict(os.environ, PROBE_KIND=kind)
        p = subprocess.run([binary], input="\n".join(ids) + "\n", capture_output=True, text=True,
                           env=env, check=True)
        got = {}
        for line in p.stdout.splitlines():
            d = json.loads(line)
            got[d.get("id", d.get("move"))] = d["verdict"]
        if set(got) != set(ids):
            raise RuntimeError(f"scan_move_probe answered {len(got)} of {len(ids)} {kind} ids")
        out[kind] = got
    return out


def build(src: Path) -> tuple:
    rows = showdown_rows(src)
    verdicts = engine_verdicts(rows)
    unplayable = {kind: sorted(i for i, v in verdicts[kind].items() if v != "ran") for kind, _ in KINDS}

    counts = collections.Counter(r["verdict"] for r in rows)
    why_engine: collections.Counter = collections.Counter()
    gen_cov: collections.Counter = collections.Counter()
    kept, seen = [], set()
    dup = 0
    for r in rows:
        if r["verdict"] != "ok":
            continue
        if r.get("generator_coverage"):
            gen_cov[r["generator_coverage"]] += 1
        bad = [f"{kind}:{i}" for kind, key in KINDS for i in r[key] if i in unplayable[kind]]
        if bad:
            for b in bad:
                why_engine[b] += 1
            counts["engine_unplayable"] += 1
            continue
        if r["packed"] in seen:
            dup += 1
            continue
        seen.add(r["packed"])
        kept.append(r)
    counts["duplicate"] = dup

    order = list(range(len(kept)))
    random.Random(SEED).shuffle(order)
    out_rows = [{"i": n, "sha": kept[j]["sha"], "file": kept[j]["file"], "packed": kept[j]["packed"]}
                for n, j in enumerate(order)]
    blob = "".join(json.dumps(r, sort_keys=True) + "\n" for r in out_rows).encode()
    data = gzip.compress(blob, compresslevel=9, mtime=0)

    source_fp = hashlib.sha256("".join(f"{r['file']}\t{r['sha']}\n" for r in rows).encode()).hexdigest()
    packed = [r["packed"] for r in out_rows]
    heal_bell = sum(1 for r in kept if "healbell" in r["moves"])
    total = len(rows)
    manifest = {
        "schema": SCHEMA,
        "source": {
            "what": "Metamon hl_05_26 gen3ou teams (the public ladder), `teams` revision v5, "
                    "downloaded 2026-09-24",
            "archive": "~/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou",
            "files": total,
            "fingerprint_sha256": source_fp,
        },
        "filter": {
            "rule": "Teams.import + six Pokemon + TeamValidator('gen3ou') legal + every species / "
                    "move / item / ability RUNS in the engine (scan_move_probe); exact packed "
                    "duplicates keep their first file. Nothing else is dropped (Heal Bell teams "
                    "and pool species-set matches stay IN).",
            "counts": {"total": total, "import_fail": counts["import_fail"], "not_six": counts["not_six"],
                       "validator_fail": counts["validator_fail"],
                       "engine_unplayable": counts["engine_unplayable"], "duplicate": dup,
                       "kept": len(out_rows)},
            "share_kept": round(len(out_rows) / total, 6) if total else 0.0,
            "engine_unplayable_ids": unplayable,
            "engine_unplayable_by_id": dict(why_engine.most_common()),
            "heal_bell_teams_kept": heal_bell,
            "for_information_generator_coverage_rejects": dict(gen_cov.most_common()),
        },
        "seed": SEED,
        "tiers": {
            "commit": {"n": COMMIT_N, "sha256": tier_sha256(packed[:COMMIT_N])},
            "milestone": {"n": MILESTONE_N, "sha256": tier_sha256(packed[:MILESTONE_N])},
            "full": {"n": len(packed), "sha256": tier_sha256(packed)},
        },
        "data_file": DATA.name,
        "data_sha256": hashlib.sha256(data).hexdigest(),
    }
    return data, manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    if not a.src.is_dir():
        print(f"no Metamon team directory at {a.src}", file=sys.stderr)
        return 2
    data, manifest = build(a.src)
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if a.check:
        same = DATA.exists() and DATA.read_bytes() == data and MANIFEST.read_text() == text
        print("reproduces the committed corpus" if same else "DOES NOT reproduce the committed corpus")
        return 0 if same else 1
    DATA.write_bytes(data)
    MANIFEST.write_text(text)
    print(json.dumps(manifest["filter"]["counts"]), manifest["tiers"]["full"]["n"], "teams,",
          len(data), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
