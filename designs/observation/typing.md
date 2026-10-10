# Static typing in `src/agents/observation/` — mypy

**Moved from `src/agents/observation/CLAUDE.md` on 2026-10-10.** ALWAYS-CURRENT like that leaf. The shared config and
its strictness tier are [`../model/typing.md`](../model/typing.md)'s. Index: [`README.md`](README.md).

## Moved from the leaf (2026-10-10)

This package is **type-checked at ZERO errors**, on the same config and the same strictness tier as
`src/agents/model/` — one `mypy.ini` at the repo root, one `files =` naming both. New code here must
pass before it lands:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m mypy   # scope from mypy.ini; must be clean
```

The gate is `src/agents/model/mypy_gate_test.py`, which runs bare `python -m mypy` (no path) so the
scope is the config's, and separately asserts that `files =` still NAMES both packages — mypy exits
0 just as happily on a scope of nothing, and "checked less" is otherwise indistinguishable from
"everything is clean". **The two packages share one config, so a loosening to clear something here
silently de-tiers the model package.** Narrow with a targeted ignore instead.

**The obs layer's own idioms come first; types complement them, never replace them:**

- **The offset/layout constants discipline is unchanged.** A slice is still written from a named
  constant and read back through `get_layout()` — mypy types the *array*, not the *index*, so it
  cannot catch a wrong offset and must never be mistaken for a check that it does. The shape and
  block comments stay.
- **`np.ndarray` carries no shape.** Same rule as the model package's `[B, 6, K]` comments: the
  dimension lives in the comment and the `*_DIM` constant, the checker only knows "an array".

- **`# type: ignore` always carries a code and a reason.** The three compact-string `describe_vector`
  sub-encoders (types / items / abilities) return a string where the base declares a dict — their
  output is embedded as a dict VALUE by `PokemonEncoder` — so the divergence is declared at each
  override.

- ⚠️ **A standalone comment that starts with `# type: ignore` IS a directive.** mypy parses it
  wherever it sits and rejects it as malformed, so an explanatory line above an ignore must not
  begin with those words — this file's convention is `# Why the \`type: ignore[...]\` below — …`.

