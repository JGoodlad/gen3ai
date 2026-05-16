# ai_v3 TODO

## Observation / Encoding

- **Volatile count encoding** — `active_context.py` encodes volatiles as binary flags only.
  Two cases where count matters:
  - **Perish Song**: 0–3 turns remaining, encoded as a single bit regardless.
  - **Sleep**: 1–7 turns remaining per Gen 3 mechanics; you cannot reset the counter by
    switching. The network cannot learn sleep-turn-aware switch timing without this signal.
  Not critical for early training, but affects late-game decision quality.
