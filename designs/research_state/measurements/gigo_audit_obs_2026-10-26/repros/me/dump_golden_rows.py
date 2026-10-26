"""Dump the golden battles' p1 rows from whichever tree is on PYTHONPATH."""
import sys
import numpy as np
from agents.training.golden_obs_core import core_rows
np.save(sys.argv[1], np.stack(core_rows()).astype(np.float32))
print("saved", sys.argv[1])
