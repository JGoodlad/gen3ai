"""Export the Gen3 feature-extractor architecture for visualization.

Produces an ONNX file you can drag-and-drop into https://netron.app to explore
the network interactively (zoom into every Linear / MultiheadAttention / embedding,
see tensor shapes on each edge). No Showdown server or checkpoint required — the
graph topology is independent of the trained weights.

Usage:
    export PYTHONPATH=$PYTHONPATH:src
    python3 src/main/visualize_arch.py                 # -> models/_arch/gen3_arch.onnx
    python3 src/main/visualize_arch.py --out /tmp/x.onnx
    python3 src/main/visualize_arch.py --torchview      # also emit a graphviz .gv source

Then open https://netron.app and drop the .onnx file in.
"""
from __future__ import annotations

import argparse
import os

import torch
from gymnasium import spaces

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.model.features_extractor import Gen3FeaturesExtractor


class _ExtractorWrapper(torch.nn.Module):
    """Adapts the dict-input extractor to flat named tensors for a clean ONNX graph."""

    def __init__(self, extractor: Gen3FeaturesExtractor):
        super().__init__()
        self.extractor = extractor

    def forward(self, observation: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        return self.extractor({"observation": observation, "action_mask": action_mask})


def build_extractor() -> tuple[Gen3FeaturesExtractor, spaces.Dict]:
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    obs_dim = encoder.dimension
    obs_space = spaces.Dict({
        "observation": spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype="float32"),
        "action_mask": spaces.Box(0, 1, shape=(11,), dtype="int8"),
    })
    kwargs = encoder.get_features_extractor_kwargs()
    kwargs.pop("log_level", None)
    extractor = Gen3FeaturesExtractor(obs_space, **kwargs)
    extractor.eval()
    return extractor, obs_space


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/_arch/gen3_arch.onnx")
    ap.add_argument("--torchview", action="store_true", help="also write a graphviz .gv source")
    args = ap.parse_args()

    extractor, obs_space = build_extractor()
    obs_dim = obs_space["observation"].shape[0]
    print(f"[arch] obs_dim={obs_dim}  features_out={extractor.features_dim}")

    wrapper = _ExtractorWrapper(extractor)
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    dummy_mask = torch.ones(1, 11, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_mask),
        args.out,
        input_names=["observation", "action_mask"],
        output_names=["features"],
        dynamic_axes={"observation": {0: "batch"}, "action_mask": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[arch] wrote ONNX -> {args.out}")
    print("[arch] open https://netron.app and drag this file in.")

    if args.torchview:
        from torchview import draw_graph
        g = draw_graph(
            wrapper,
            input_data=(dummy_obs, dummy_mask),
            depth=4,
            expand_nested=True,
            graph_name="Gen3FeaturesExtractor",
            save_graph=False,
        )
        gv_path = os.path.splitext(args.out)[0] + ".gv"
        with open(gv_path, "w") as f:
            f.write(g.visual_graph.source)
        print(f"[arch] wrote graphviz source -> {gv_path}")
        print("[arch] paste its contents into https://dreampuf.github.io/GraphvizOnline/")


if __name__ == "__main__":
    main()
