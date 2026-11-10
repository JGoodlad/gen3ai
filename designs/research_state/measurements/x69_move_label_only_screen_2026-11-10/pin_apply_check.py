"""At the X69 pin, on CPU: P*'s extractor as a fork load rebuilds it (from the PARENT's zip: `shaping`), then the
trainer's own post-load call (`reapply_resume_immutable`) with the arm's resolved mode."""
import sys
import torch as th
from agents.model.resume_reapply import reapply_resume_immutable
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings

th.set_num_threads(1)
model, _ = load_foreign_opponent(sys.argv[1], current_model_version(load_mappings()), device="cpu")
fe = model.policy.features_extractor
flags = lambda: {n: bool(getattr(getattr(fe, n, None), "publish_detach", False))      # noqa: E731
                 for n in ("hypothesis_builder", "move_belief", "item_belief_head", "spread_belief")
                 if getattr(fe, n, None) is not None}
print("loaded :", fe.belief_grad_mode, "| move publication cut:", fe._move_publish_cut, "|", flags())
out = reapply_resume_immutable(fe, {"belief_grad_mode": "move_label_only", "oracle_reveal": "off",
                                    "tau_solve": getattr(fe, "tau_solve", None) or "fp32_alt"})
print("applied:", out["belief_grad_mode"], "| move publication cut:", fe._move_publish_cut, "|", flags())
t = th.ones(3, requires_grad=True)
print("publish boundary detaches:", not fe._publish_move_belief(t).requires_grad)
assert out["belief_grad_mode"] == ("shaping", "move_label_only") and fe._move_publish_cut is True
assert not fe._publish_move_belief(t).requires_grad
print("PIN APPLY CHECK OK")
