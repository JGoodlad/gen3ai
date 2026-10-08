"""K9(b) static-screen u1480 replay: the k9_early_probe harness + a static x fixed_mass arm.

Run with cwd = the checkout under test (its designs/ and src/)."""
import importlib.util
import os
import sys

ROOT = os.getcwd()
p = os.path.join(ROOT, "designs/research_state/measurements/k9_early_probe_2026-10-06/measure.py")
spec = importlib.util.spec_from_file_location("k9_early", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.old.ARMS["static_fm"] = {"belief_tokens": "fixed_mass", "token_encoding": "static"}
sys.exit(m.main())
