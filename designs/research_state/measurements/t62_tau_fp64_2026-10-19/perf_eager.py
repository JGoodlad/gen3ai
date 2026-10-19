"""T62 COST, the clean CPU read: the eager T2 decision forward at ONE torch thread (what the CPU meters run), OLD vs NEW
interleaved, at X26 checkpoint_280000170 (perf.py multi-thread eager timings were contended by other agents' jobs).

    PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= python3 perf_eager.py <this dir> 1 perf_eager_1thread.json
"""
import sys, importlib.util, numpy as np, torch as th, time, json
D=sys.argv[1]; th.set_num_threads(int(sys.argv[2]))
spec=importlib.util.spec_from_file_location("pf", D+"/perf.py"); PF=importlib.util.module_from_spec(spec); spec.loader.exec_module(PF)
EQ=PF.EQ
from agents.inference.service.decision import DecisionModule
from main.capacity import load_policy
from utils.rust_env.fixture_battles import play_rows
rows=play_rows(8, seed=6201); fo=np.concatenate([o for o,_ in rows]).astype(np.float32); fm=np.concatenate([m for _,m in rows]).astype(bool)
m,_c,_=load_policy(EQ.P280, EQ.P280_DIR, "cpu"); dm=DecisionModule(m.policy).eval()
out={}
for B in (1, 16, 64, 256):
    idx=np.arange(B)%len(fo); o,mk=th.as_tensor(fo[idx]),th.as_tensor(fm[idx])
    def run(dt):
        def r():
            PF._set(dt); return dm(o,mk)
        return r
    with th.no_grad():
        run(th.float32)(); run(th.float64)()
        out[B]=PF.abab({"old":run(th.float32),"new":run(th.float64)}, 21 if B<256 else 9)
    print(B, out[B], "ratio %.3f"%(out[B]["new"]/out[B]["old"]), flush=True)
json.dump({"threads": th.get_num_threads(), "eager_ms": out}, open(sys.argv[3],"w"), indent=1)
