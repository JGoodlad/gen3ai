"""The declarative observation SCHEMA — Stage 3 groundwork for the ai_v9 entity generation
(`designs/ai_v9/design_generation_roadmap.md` §3 Stage 3: "one declarative schema from which the
env packer, the model unpacker, the gym spaces, and the dimension tests are all generated").

THIS module is deliberately the migration's FIRST half: a validated, self-checking **view** over
the existing `Gen3ObservationEncoder.get_layout()` — one place that names every contiguous block
of the flat observation, proves the blocks tile the vector exactly (no gaps, no overlaps, totals
match), and prints the canonical table. The second half (generating the packer/unpacker/spaces
FROM the schema and re-homing the flat vector into entity blocks) builds on this view; starting
as a view means the schema can never drift from the live encoder while both exist — it is
DERIVED from the same source every consumer already reads, and its invariants throw on any
inconsistency the layout itself carries.

Usage:
  python -m agents.observation.schema          # print the block table
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Block:
    name: str
    offset: int
    dim: int
    doc: str = ""
    children: "List[Block]" = field(default_factory=list)

    @property
    def end(self) -> int:
        return self.offset + self.dim


@dataclass(frozen=True)
class ObsSchema:
    total_dim: int
    blocks: List[Block]

    def validate(self) -> "ObsSchema":
        """The tiling proof: blocks are ordered, contiguous, gap-free and sum to total_dim; every
        block's children (when present) tile their parent the same way. Throws on any violation —
        a schema that cannot prove itself is worse than no schema."""
        cur = 0
        for b in self.blocks:
            if b.offset != cur:
                raise ValueError(f"block {b.name!r}: offset {b.offset} != expected {cur} "
                                 f"(gap or overlap with the previous block)")
            if b.dim <= 0:
                raise ValueError(f"block {b.name!r}: non-positive dim {b.dim}")
            cur = b.end
            if b.children:
                ccur = 0
                for c in b.children:
                    if c.offset != ccur:
                        raise ValueError(f"{b.name}.{c.name}: child offset {c.offset} != {ccur}")
                    ccur = c.end
                if ccur != b.dim:
                    raise ValueError(f"{b.name}: children tile {ccur} of {b.dim} dims")
        if cur != self.total_dim:
            raise ValueError(f"blocks tile {cur} dims but total_dim is {self.total_dim}")
        return self

    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    # --- the GENERATOR half (Stage 3, second act): consumers DERIVE from the schema ------------
    def slices(self) -> "Dict[str, slice]":
        """name → `slice` for every top-level block AND every child (child keys are
        `parent.child`, absolute offsets). The one mapping an unpacker should read instead of
        hand-holding offsets — validated by construction (the tiling proof ran in build)."""
        out: Dict[str, slice] = {}
        for b in self.blocks:
            out[b.name] = slice(b.offset, b.end)
            for c in b.children:
                out[f"{b.name}.{c.name}"] = slice(b.offset + c.offset, b.offset + c.end)
        return out

    def gym_space(self):
        """The flat observation's gym `Box`, generated FROM the schema (the exact params
        `Gen3Env.vector_space` uses today — one source for the shape, so an obs-dim change
        can never desync the env's space from the encoder's vector)."""
        import numpy as np
        from gymnasium import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_dim,), dtype=np.float32)

    def describe(self) -> str:
        lines = [f"{'block':<22} {'offset':>7} {'dim':>6}", "-" * 38]
        for b in self.blocks:
            lines.append(f"{b.name:<22} {b.offset:>7} {b.dim:>6}")
            for c in b.children:
                lines.append(f"  .{c.name:<19} {b.offset + c.offset:>7} {c.dim:>6}")
        lines.append("-" * 38)
        lines.append(f"{'TOTAL':<22} {'':>7} {self.total_dim:>6}")
        return "\n".join(lines)


def _sub_blocks(sub_layout: Dict) -> List[Block]:
    kids = [(k, v["offset"], v["dim"]) for k, v in sub_layout.items()
            if isinstance(v, dict) and "offset" in v and "dim" in v]
    kids.sort(key=lambda t: t[1])
    return [Block(name=k, offset=o, dim=d) for k, o, d in kids]


def build_schema(layout: Dict) -> ObsSchema:
    """Derive the schema from the live `get_layout()` dict. The top-level tiling is rebuilt from
    the layout's own scalars (NOT the `parts` table verbatim — its `reactive` entry's `end` spans
    to the vector end as a slicing convenience, which is exactly the kind of ambiguity this schema
    exists to name away)."""
    team = layout["parts"]["our_team"]
    n_mons, mon_dim = team["reshape"]
    ctx_dim = 2 * layout["active_context_dim"]
    global_dim = layout["parts"]["global"]["dim"]
    reactive_dim = layout["base_dim"] - layout["parts"]["reactive"]["start"]
    blocks = [
        Block("our_team", 0, n_mons * mon_dim, doc=f"{n_mons} × {mon_dim}-dim mon slots"),
        Block("opp_team", n_mons * mon_dim, n_mons * mon_dim),
        Block("active_context", 2 * n_mons * mon_dim, ctx_dim, doc="2 × active ctx"),
        Block("global_env", 2 * n_mons * mon_dim + ctx_dim, global_dim,
              children=_sub_blocks(layout.get("global_layout", {}))),
        Block("reactive", layout["parts"]["reactive"]["start"], reactive_dim,
              children=_sub_blocks(layout.get("reactive_layout", {}))),
        Block("prev_action_mask", layout["base_dim"], layout["prev_mask_dim"]),
        Block("turn_history", layout["turn_history_offset"],
              layout["n_history_turns"] * layout["turn_delta_dim"],
              doc=f"{layout['n_history_turns']} × {layout['turn_delta_dim']}-dim TurnDelta"),
    ]
    return ObsSchema(total_dim=layout["total_dim"], blocks=blocks).validate()


def main() -> None:
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    print(build_schema(layout).describe())


if __name__ == "__main__":
    main()
