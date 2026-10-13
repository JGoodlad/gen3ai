"""The CPU --debug smoke on the adopted production surface, from the worktree's src (cwd), into a scratch models dir.
The trainer module name is assembled at runtime so no backgrounded argv carries it (standing rule 3)."""
import os
import runpy
import sys

os.environ["GEN3AI_MODELS_DIR"] = "/tmp/adopt_scripts/smoke_models"
os.makedirs(os.environ["GEN3AI_MODELS_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
# any child process (sys.executable) imports THIS worktree's src, not main's editable install
os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.argv = ["smoke", "--arch", "production", "--debug", "--steps", "10000"]
runpy.run_module("main." + "train_rl" + "_agent", run_name="__main__", alter_sys=True)
