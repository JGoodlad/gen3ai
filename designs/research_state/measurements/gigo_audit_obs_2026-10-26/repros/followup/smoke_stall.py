"""GIGO audit B-C2 by EXECUTION: the --debug smoke at a tiny stall threshold. Run from the repo root as
`GEN3AI_MODELS_DIR=<scratch> GIGO_STALL_TURNS=6 GIGO_STEPS=1500 python <this file>` (CPU; ~1 minute). Patches (in the PARENT only) the
production turn limit to GIGO_STALL_TURNS and prints, for every ended game, its collector rows vs the p1 decisions
the core actually FED (the CHOOSE p1 lines of the finished game's input log)."""
import os, re, runpy, sys
WT = os.getcwd()   # run from the repo root (a checkout or a worktree)
sys.path.insert(0, WT + "/src")


def install():
    from utils.rust_env import episode as EP
    limit = int(os.environ.get("GIGO_STALL_TURNS", "6"))
    EP.stall_threshold = lambda: limit
    from agents.training.rust_rollout import collector as C
    from agents.training.rust_rollout import store as S

    orig_after = C.RustCollector._after_step
    orig_end = S.GameLog.end
    state = {"fin": {}}

    def after(self):
        c = self.cols
        done = [int(e) for e in (c["done"] == 1).nonzero()[0]]
        state["fin"] = {}
        if done:
            for f in self.core.finished():
                state["fin"][(int(f["env"]), int(f["episode"]))] = f
        return orig_after(self)

    def end(self, env, **kw):
        rows = list(self.cur[env])
        g = orig_end(self, env, **kw)
        f = state["fin"].get((int(env), int(kw["episode"])))
        fed = len(re.findall(r"^CHOOSE p1 ", f["script"], re.M)) if f else None
        # a forced replacement is also a p1 CHOOSE; the stall FORFEIT is a FORCELOSE line instead of a CHOOSE
        forced = bool(f and "FORCELOSE p1" in f["script"])
        kept = int(g.length) if g is not None else 0
        print(f"[gigo-smoke] env {env} ep {kw['episode']}: forfeit_flag={kw['forfeit']} trunc={kw['truncated']} "
              f"rows_written={len(rows)} rows_kept={kept} p1_CHOOSE_fed={fed} FORCELOSE_p1={forced} "
              f"core_forfeit_side={f['forfeit'] if f else None}", flush=True)
        return g

    C.RustCollector._after_step = after
    S.GameLog.end = end


if __name__ == "__main__":
    install()
    sys.argv = ["train_rl_agent.py", "--debug", "--steps", os.environ.get("GIGO_STEPS", "1500"), "--rust-env-profile", "selfcheck"]
    runpy.run_path(WT + "/src/main/train_rl_agent.py", run_name="__main__")
