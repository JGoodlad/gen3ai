"""Run the PIN's main.policy_drift with ONE redirect: team validation uses the MAIN checkout's
validate_team.js (byte-identical to the pins') so it resolves main's built deps/pokemon-showdown
(the same submodule commit e0551883 at both pins and main). The pin checkouts have an empty
deps/pokemon-showdown and are read-only. Nothing else is changed."""
import runpy
import sys

import utils.bridge.team_validator as tv

tv.__file__ = "/home/goodlad/dev/gen3ai/src/utils/bridge/team_validator.py"
sys.argv = ["main.policy_drift"] + sys.argv[1:]
runpy.run_module("main.policy_drift", run_name="__main__", alter_sys=True)
