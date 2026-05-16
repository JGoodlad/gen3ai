# Model Directory — Contributor Notes

## Keep the architecture digraph in sync

`designs/ai_v3/README.md` contains a Mermaid digraph and dimension reference table for the network.

**Update it whenever you change `features_extractor.py`**, specifically:

- Adding, removing, or resizing any layer (Linear dims, attention heads, embedding dims)
- Changing what gets concatenated into move processor input, role encoder input, or the final aggregation
- Changing the observation dimension (`base_dimension` or `dimension` in `state_encoder.py`)
- Adding or removing an attention path (Pressure / Safety / Synergy)
- Changing how `prev_mask` or future turn-history features are routed

The digraph is the fastest way for a new contributor (or Claude) to understand data flow. A stale digraph is worse than none.
