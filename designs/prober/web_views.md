# `/battle` — merged into `/game` (2026-10-09)

Owned by this tree (always current). The classic turn-by-turn replay that lived at `/battle` was
MERGED into the battle viewer on 2026-10-09: `/battle` now answers **307** to `/game`, keeping `run`
and `battle` and mapping `start=N` (the replay's turn window) to `turn=N` (the first decision at or
after that game turn). `/api/battle-turns` — the CLI's `turns` contract — is unchanged.

- What the viewer shows, field by field: [`battle_view_v2.md`](battle_view_v2.md).
- Why the two pages were merged, the audit behind it, and what each `/battle` element became (kept,
  moved, or removed with its reason): [`battle_viewer_ux_2026-10-09.md`](battle_viewer_ux_2026-10-09.md)
  §1 and §13.
