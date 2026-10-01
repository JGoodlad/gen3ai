"""THE K8 COMPILE INVENTORY — what dynamo makes of the WHOLE PPO update, not just the extractor
(M5 Lane K8, `designs/endstate/program_rust_core.md`; readout under
`designs/research_state/measurements/k8_inventory/`).

The owner's goal (2026-09-30): "optimal compiling using Dynamo wherever possible as an explicit end
goal". K8 declares compiled REGIONS, each `fullgraph=True`, with a break inside a region a startup
error. Before anyone declares a region, this tool measures what is there:

  (a) every graph and every break — dynamo's reason, the site, the component — over a real update
      (`capture.InventoryCapture`; `attribution` maps a stack to extractor / heads /
      distribution+masking / loss+diagnostics / optimizer / buffer / logging);
  (b) which parts run compiled and which eager — graphs' op counts per component, and a profiled
      steady-state update's top-level ATen ops split by the compiled-graph markers
      (`trace_classify`);
  (c) recompiles and the guard that failed, across a first (diagnostic) update, a cadence-skipped
      update, an eval-mode rollout forward and a steady-state update;
  (d) on the GPU, the production compiled learner's kernel time inside compiled graphs vs eager
      ATen, per phase of `train()` (the `phase_hook` segments as profiler ranges).

It runs the REAL trainer on a real checkpoint and the learner benchmark's pinned rollout buffer, in
a subprocess (`worker`), and changes no production module. See `__main__` for the commands.
"""
