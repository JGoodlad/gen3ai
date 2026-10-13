"""Dump the namespace the closing test's end-state argv RESOLVES to (parser + resolve_config), plus
the ResolvedRunConfig, as sorted JSON.  cd <tree>/src && PYTHONPATH=<tree>/src python resolved.py <out.json>"""
import contextlib
import io
import json
import os
import sys
sys.path.insert(0, os.getcwd())

os.environ.setdefault("GEN3AI_MODELS_DIR", "/tmp/adopt_scripts/res/models_scratch")
os.makedirs(os.environ["GEN3AI_MODELS_DIR"], exist_ok=True)
ARGV = ("--arch production --steps 15000000 --seed 2001 --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 "
        "--ridealong-opp 5 --ridealong-rnd-variants all --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 "
        "--device cuda").split()
from main.train_rl_agent import build_parser  # noqa: E402
from main.train.config import resolve_config  # noqa: E402

parser = build_parser()
args = parser.parse_args(ARGV)
raw = {k: v for k, v in vars(args).items()}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = resolve_config(args, parser)


def j(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


out = {"parsed_raw": {k: j(v) for k, v in sorted(raw.items())},
       "resolved": {k: j(v) for k, v in sorted(vars(args).items()) if not k.startswith("_")},
       "resolved_private": {k: j(v) for k, v in sorted(vars(args).items()) if k.startswith("_")},
       "rc": {k: j(v) for k, v in sorted((getattr(rc, "__dict__", None) or rc._asdict()).items())}
       if rc is not None else None,
       "stdout": buf.getvalue()[-6000:]}
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print("ok", len(out["resolved"]))
