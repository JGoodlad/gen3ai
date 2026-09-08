# Architecture drift — the three walls a stale checkpoint hits

Owned by this tree. The RULE is in `src/main/prober/CLAUDE.md`: a model-loading probe works only on
a run at the CURRENT architecture, the failure is an `ArchDriftError` diagnosis, and a surface that
loads a model renders that message rather than collapsing it to "analysis failed". This file holds
what the diagnosis is made of.


**The failure is now a DIAGNOSIS** (`ProbeModel.load` → `ArchDriftError`, `model.py`). Three walls
were measured, each of which used to surface as a raw error from inside SB3:

1. a DELETED flag still baked into the zip's `features_extractor_kwargs` → `TypeError: unexpected
   keyword argument 'spread_belief_nature_marginalize'`. **Recovered**: unknown kwargs are dropped
   (`_accepted_extractor_kwargs` introspects the live `__init__` signature) and *which* ones is
   reported — a dropped flag means the rebuilt extractor is not the one that played.
   ⚠ **Recovery rate on today's archive is ZERO** — every run carrying a deleted flag also differs
   in obs dim, so the drop alone never rescues one. It is unit-tested, not archive-proven, and it
   exists for the *next* pure-flag deletion (v66 was exactly that shape).
2. a value the code now VALIDATES → `move_candidate_floor=0.0`, legal when trained, rejected since
   the v65 legality guard. **Not** recovered: relaxing a correctness guard to probe would answer
   with a different model than the one under the microscope.
3. weight SHAPES that no longer fit → `mat1 and mat2 shapes cannot be multiplied (12x380 and
   386x256)`. Not recoverable in principle.

The error names the saved-vs-current obs dim, the saved-vs-current `arch_signature`, the dropped
flags, the underlying cause, **the exact `git checkout <hash>`** to re-probe from (read from the run
`metadata.json` — the sidecar search walks up THREE levels, because an eval snapshot sits at
`<run>/eval_traces/step_<N>/snapshot.zip` and a two-level search silently lost the hash on every one
of them), and which model-free views still work on that run. `peek_checkpoint` reads the arch
fingerprint from the zip's JSON `data` member in **~5 ms** — that is what makes diagnosing cheap
enough to do before the load is even attempted. `analyze`'s `model_resolution.dropped_kwargs`
carries the drop to a surface. Tests: `model_test.py`.

