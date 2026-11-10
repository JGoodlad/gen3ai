"""X66 Amendment 1's CPU smoke: the x66 trainer on the registered bank, fold H, 3 steps (student.train has no step-limit flag; this is the function the kit's argv calls). Run at the side pin under mem_cap, nice 19, the agent cores."""
import sys
from pathlib import Path
from main.x65 import student as ST
sys.exit(ST.train(Path("/home/goodlad/gen3ai_archive/x66_exit0h_2026-11-09/bank"), "0812768c9fcecf586c5973e23ba600cc9b3da99a704b2aeab567e89e7a185344", "heldout72_hpfix", 1e-4,
                  Path("/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"), Path("/home/goodlad/gen3ai_archive/x66_exit0h_2026-11-09/smoke/s_h"), device="cpu", threads=3, limit_steps=3, recipe="x66"))
