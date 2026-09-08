# Static typing in `src/agents/model/` — the mypy scope and the annotation rules

**Lifted out of `src/agents/model/CLAUDE.md` on 2026-09-08.** This doc OWNS its subject and carries
the same ALWAYS-CURRENT obligation as that leaf — update it in the same pass as the code.
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) is the doc of record for what the model IS; where the
two disagree, ARCHITECTURE.md wins.

**Config: `mypy.ini` at the repo root**, deliberately SCOPED to this package. Everything else in
the tree is read for types via `follow_imports = silent` but its errors are not reported, so this
package can be tightened without waiting on the others.

**Strictness landed:** `disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`,
`no_implicit_optional`, `strict_equality`, `warn_return_any`, `warn_redundant_casts`,
`warn_unused_ignores`. NOT `strict` (nor `disallow_any_generics` / `disallow_untyped_calls`) — those
fire almost entirely on the untyped third-party boundary, where a targeted `# type: ignore[code]`
says more than a guessed annotation. **Excluded:** `*_test.py`, the GENERATED `extractor_arch.py`,
and `feature_coverage/` (its probes are all `*_test.py`, and `_support.py` is their fixture).

**The rules for the annotations themselves:**

- **The `[B, 6, K]` shape comments are the shape documentation and mypy does not replace them** —
  a tensor is `torch.Tensor` to the checker and its shape lives in the comment. Keep both.
- **Registered buffers and mixin surfaces are DECLARED under `if TYPE_CHECKING:`, not ignored.**
  `register_buffer` in a loop (the whole `damage_tables` set, PopArt's `mu`/`sigma`, `Embeddings`'
  index maps) exists only dynamically, so `Module.__getattr__` types every read `Any` and the `Any`
  then leaks into every expression downstream. A TYPE_CHECKING block of `NAME: torch.Tensor`
  declarations is the fix — no runtime effect, and it keeps arithmetic typed.
- `damage_op_pairwise.py` / `damage_op_blocks.py` are **mixins composed into an `nn.Module`**, so
  they additionally declare `def __getattr__(self, name: str) -> Any: ...` (mirroring what
  `torch.nn.Module` gives the composed class) plus the sibling/composed-class methods they call.
- `typing.cast` is the narrowing tool where a value is **provably** present — chiefly the
  `gen3_data` facade, whose `.get()` is `Optional` but is keyed by the facade's own `raw()` /
  `base_form_ids()` list. `cast` returns its argument unchanged, so the forward stays bit-identical.
- **`# type: ignore` always carries a specific code**, and a one-line reason where the cause is not
  obvious. Two causes dominate and are worth knowing before you read one as a smell:
  1. **torch types `Module.__call__` as `Callable[..., Any]`**, so every `return self.some_layer(x)`
     in a forward trips `warn_return_any`. That is a stub defect, not an unknown in our code.
  2. **Flag-gated submodules are `Optional`, and their reads sit under a *correlated* guard** —
     `Gen3FeaturesExtractor.damage_op` is read under `edge_bias is not None` /
     `damage_block is not None` / `intent_* is not None`, implications the constructor enforces with
     a raise. That invariant spans two objects, so no narrowing expresses it.
- **A `1.0 - <Tensor>` reads as `Any`** through torch's stubs (the reflected `__rsub__` return is
  lost). Pin it at the binding — `revealed: torch.Tensor = 1.0 - mask` — never with an ignore.

**Two annotations mypy proved were FALSE** are recorded here rather than silently widened:
`Gen3FeaturesExtractor.__init__` declares `observation_space: spaces.Dict` but never reads the
parameter, and three serverless probe paths pass a flat `Box`; and its `layout` is
`Optional[...] = None` while the first phase module built indexes it unconditionally, so `None` is
not a usable argument.
