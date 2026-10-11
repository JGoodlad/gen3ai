"""Load check (no outcome): CP loads at its pin through the strict loader, the critic is winprob, the
layout's clock offset, and the CPU forward cost per row at playout batch sizes."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals

    out = {"cp": c}
    out["clock_offset"] = C.clock_offset()
    with inference_globals(C.threads()):
        t0 = time.monotonic()
        m = C.load_model(c)
        out["load_s"] = round(time.monotonic() - t0, 2)
        out["critic_mode"] = str(getattr(m.policy, "_critic_mode", None))
        dim = int(m.policy.observation_space["observation"].shape[0])
        out["obs_dim"] = dim
        rng = np.random.default_rng(0)
        for n in (64, 256, 448):
            rows = rng.random((n, dim), dtype=np.float32) * 0.1
            masks = np.ones((n, 11), dtype=np.uint8)
            logits_and_values(m, rows, masks, batch=1024)
            t = C.cpu_seconds(); w = time.monotonic()
            for _ in range(3):
                logits_and_values(m, rows, masks, batch=1024)
            out[f"fwd_ms_per_row_b{n}"] = {"cpu": round((C.cpu_seconds() - t) / (3 * n) * 1e3, 3),
                                           "wall": round((time.monotonic() - w) / (3 * n) * 1e3, 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
