"""ProberApp — the Textual forensic-replay inspector.

Layout: a trace browser (Tree) | an invocation list (ListView) | four analysis
tabs (DataTables). Navigation is instant; the slow torch work — loading the
checkpoint and running the per-invocation forward/backward passes — runs in
``@work(thread=True)`` workers so the asyncio event loop (and the UI) never
blocks. Workers compute and hand results back via ``call_from_thread``; widgets
are only ever touched on the event loop.
"""

from __future__ import annotations

import json
import os

import numpy as np
from rich.text import Text
from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Collapsible,
    DataTable,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    Tree,
)

# Analysis sections (Collapsible id, title, toggle key) — multiple can be open at once.
# Keys are 1-indexed in DISPLAY order (no 0 — awkward on a laptop) and shown in each title.
_SECTIONS = [
    ("sec-summary", "Summary", "1"),
    ("sec-team", "Team", "2"),
    ("sec-review", "Review", "3"),
    ("sec-board", "Board", "4"),
    ("sec-faith", "Faithfulness", "5"),
    ("sec-matchups", "Matchups", "6"),
    ("sec-sweep", "Intervention", "7"),
    ("sec-saliency", "Saliency", "8"),
    ("sec-flow", "Flow", "9"),
    ("sec-outcome", "Outcome", "0"),
]
# Title shown on each Collapsible — "1  Summary" — so the hotkey is always visible.
_SEC_TITLE = {sid: f"{key}  {title}" for sid, title, key in _SECTIONS}
_OPEN_BY_DEFAULT = {"sec-summary", "sec-board", "sec-faith", "sec-flow", "sec-outcome"}


class NavTree(Tree):
    """Tree with file-explorer left/right keys: right expands (then descends to the
    first child), left collapses (then ascends to the parent)."""

    BINDINGS = [
        Binding("right", "expand_or_child", "Expand", show=False),
        Binding("left", "collapse_or_parent", "Collapse", show=False),
    ]

    def action_expand_or_child(self) -> None:
        node = self.cursor_node
        if node is None:
            return
        if node.allow_expand and not node.is_expanded:
            node.expand()
        elif node.children:
            self.move_cursor(node.children[0])

    def action_collapse_or_parent(self) -> None:
        node = self.cursor_node
        if node is None:
            return
        if node.allow_expand and node.is_expanded:
            node.collapse()
        elif node.parent is not None:
            self.move_cursor(node.parent)


class PaneSplitter(Static):
    """A 1-cell draggable divider; dragging resizes the pane to its left.

    The right-most pane is `1fr` so it absorbs the slack as a fixed-width pane to
    the splitter's left grows/shrinks."""

    DEFAULT_CSS = """
    PaneSplitter { width: 1; height: 1fr; background: $primary 20%; }
    PaneSplitter:hover { background: $accent; }
    """

    def __init__(self, target: str, min_w: int = 14, max_w: int = 160, **kw) -> None:
        super().__init__("", **kw)
        self._target_sel = target
        self._min, self._max = min_w, max_w
        self._dragging = False
        self._start_x = 0
        self._start_w = 0

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._start_w = self.app.query_one(self._target_sel).region.width
        self._start_x = event.screen_x
        self._dragging = True
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging:
            return
        w = self._start_w + (event.screen_x - self._start_x)
        self.app.query_one(self._target_sel).styles.width = max(self._min, min(self._max, int(w)))
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            event.stop()

from main.prober.discovery import (
    TIERS,
    BattleTrace,
    ModelChoice,
    TraceTree,
    build_trace_tree,
    resolve_model_for_step,
)
from main.prober.engine import (
    InvocationAnalysis, analyze_invocation, build_result_timeline,
    parse_protocol_log, protocol_for_turn, summary_flags,
)
from main.prober.review import ReviewStore
from main.tui import THEME_PATH, Gen3App, gradient_color

_DEFAULT_GAMMA = 0.99   # used to compute the Outcome panel's TD residual when metadata.json is absent

# Glyphs for the flags shown in the invocation list (switch is omitted — already
# visible in the "switch:…" label). n/N jumps to the discrete, selective events
# (faints, switches); "uncertain" is shown as a glyph but not jumped to — for a
# low-confidence policy it's the norm, not a needle worth jumping to.
_FLAG_GLYPH = {"uncertain": "?", "faint": "✗", "disagree": "≠"}
_JUMP_FLAGS = ("faint", "switch")
# User manual-review annotations (distinct from the auto summary_flags above).
_REVIEW_FLAG_GLYPH = "⚑"
_REVIEW_NOTE_GLYPH = "✎"


class ProberApp(Gen3App):
    """Interactive inspector for saved forensic battle traces."""

    CSS_PATH = [str(THEME_PATH), "prober.tcss"]
    TITLE = "Gen3AI Prober"

    BINDINGS = Gen3App.BINDINGS + [
        ("j", "next_inv", "Next inv"),
        ("k", "prev_inv", "Prev inv"),
        ("n", "next_flagged", "Next flag"),
        ("N", "prev_flagged", "Prev flag"),
        ("f", "cycle_filter", "Filter"),
        ("r", "reanalyze", "Re-analyze"),
        ("m", "cycle_model", "Model tier"),
        ("R", "reload_model", "Reload ckpt"),
        # Manual review (model's own games): mark a decision funky + jot why.
        ("space", "toggle_review_flag", "⚑ Flag"),
        ("e", "edit_note", "Note"),
        ("right_square_bracket", "next_annotated", "Next note"),
        ("left_square_bracket", "prev_annotated", "Prev note"),
        ("E", "export_notes", "Export notes"),
        ("y", "copy_replay_path", "Replay path"),
        # Section toggles — generated from _SECTIONS so the key/title/binding never drift.
        *[Binding(key, f"toggle_section('{sid}')", title, show=False)
          for sid, title, key in _SECTIONS],
    ]

    def __init__(
        self,
        root: "str | None" = None,
        ckpt_override: "str | None" = None,
        preselect_inv: "int | None" = None,
        injected_model=None,
    ) -> None:
        super().__init__()
        self._root = root
        self._ckpt_override = ckpt_override
        self._preselect_inv = preselect_inv
        self.sub_title = root or ""

        self._tree_model: "TraceTree | None" = None
        self._summary_cache: "dict[str, dict]" = {}
        self._opp_team_cache: "dict[str, tuple | None]" = {}   # privileged opp team per trace (reconstruction.json)
        self._our_hp_cache: "dict[str, dict | None]" = {}      # our typed Hidden Power per trace (reconstruction.json)
        self._protocol_cache: "dict[str, tuple]" = {}          # raw Showdown protocol lines per trace (replay.html)
        self._current_battle: "BattleTrace | None" = None
        self._current_summary: "dict | None" = None
        self._analyze_token = 0
        self._pending_inv: "int | None" = None
        self._flagged: "list[int]" = []          # invocation indices with an auto summary-flag
        self._battle_filter = "all"              # cycled by `f`: all/loss/win
        self._review_store: "ReviewStore | None" = None   # manual flags/notes (per run dir)
        self._current_inv: "int | None" = None   # the highlighted invocation index
        self._last_analysis: "InvocationAnalysis | None" = None  # for cheap review-card re-render on note add

        # Per-battle model resolution. The model that re-runs a trace depends on the
        # trace's eval step (exact snapshot → nearest checkpoint → most recent), so we
        # resolve + (lazily) load per selected battle, caching loaded models by path.
        self._gamma = _DEFAULT_GAMMA              # set from run metadata in on_mount; for TD residual
        self._tier = "auto"                       # cycled by `m` (auto/nearest/recent)
        self._injected_model = injected_model     # tests: stand in for every path
        self._model_cache: "dict[str, object]" = {}
        self._model = None                        # currently active ProbeModel
        self._active_path: "str | None" = None
        self._current_choice: "ModelChoice | None" = None

    @property
    def _model_ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose_body(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("⏳ no checkpoint", id="ckpt-badge")
                tree: Tree = NavTree("traces", id="trace-tree")
                tree.root.expand()
                yield tree
            yield PaneSplitter("#sidebar", id="split-sidebar")
            with Vertical(id="middle"):
                yield Static("select a battle", id="battle-header")
                yield ListView(id="invocation-list")
            yield PaneSplitter("#middle", id="split-middle")
            with Vertical(id="analysis"):
                # Collapsible sections (not exclusive tabs) so several can be open at
                # once; toggle by clicking a title or pressing its number key (1–8, shown
                # in each title). Titles come from _SEC_TITLE so key+label never drift.
                with VerticalScroll(id="analysis-scroll"):
                    # Decision dashboard: the one-glance "funky turn" view — what it
                    # chose, the move/switch probabilities WITH their effectiveness +
                    # incoming KO-risk, and the threat/critic context, all grouped so a
                    # turn can be judged without scanning Board+Faith+Matchups+Outcome.
                    with Collapsible(title=_SEC_TITLE["sec-summary"], collapsed=False, id="sec-summary"):
                        yield Static("", id="summary-head")
                        with Horizontal(id="summary-tables"):
                            with Vertical(classes="summary-col"):
                                yield Static("MOVES", classes="summary-col-label")
                                yield DataTable(id="summary-moves")
                            with Vertical(classes="summary-col"):
                                yield Static("SWITCHES", classes="summary-col-label")
                                yield Static("", id="summary-switches")
                            with Vertical(classes="summary-col"):
                                yield Static("OPP TEAM (revealed)", classes="summary-col-label")
                                yield Static("", id="summary-opp")
                    # Team details: every mon's moveset (ours full; opp's revealed-only) + hp/
                    # status/item — decoded from the obs, so it needs captured state.
                    with Collapsible(title=_SEC_TITLE["sec-team"], collapsed=True, id="sec-team"):
                        with Horizontal(id="team-tables"):
                            with Vertical(classes="summary-col"):
                                yield Static("our team", classes="board-label")
                                yield Static("", id="team-our")
                            with Vertical(classes="summary-col"):
                                yield Static("opp team (revealed)", classes="board-label")
                                yield Static("", id="team-opp")
                    # Manual-review card: what the model EXPECTED vs what HAPPENED, plus the
                    # human's funky-flag + note (space=flag, e=note, [ ]=jump, E=export).
                    with Collapsible(title=_SEC_TITLE["sec-review"], collapsed=False, id="sec-review"):
                        yield Static("", id="review-card")
                        yield Static("", id="review-status")
                        yield Input(placeholder="add note — Enter to append, timestamped (e to focus)…",
                                    id="review-note")
                    with Collapsible(title=_SEC_TITLE["sec-board"], collapsed=False, id="sec-board"):
                        yield Static("", id="board-summary")
                        yield Static("", id="board-field")
                        yield Static("our team", classes="board-label")
                        yield DataTable(id="board-our")
                        yield Static("opp team (revealed)", classes="board-label")
                        yield DataTable(id="board-opp")
                    with Collapsible(title=_SEC_TITLE["sec-faith"], collapsed=False, id="sec-faith"):
                        yield DataTable(id="faith-table")
                    with Collapsible(title=_SEC_TITLE["sec-matchups"], collapsed=True, id="sec-matchups"):
                        yield DataTable(id="matchups-table")
                        yield Static("", id="matchups-threat")
                    with Collapsible(title=_SEC_TITLE["sec-sweep"], collapsed=True, id="sec-sweep"):
                        yield DataTable(id="sweep-table")
                    with Collapsible(title=_SEC_TITLE["sec-saliency"], collapsed=True, id="sec-saliency"):
                        yield DataTable(id="saliency-table")
                    with Collapsible(title=_SEC_TITLE["sec-flow"], collapsed=False, id="sec-flow"):
                        yield Static("", id="flow-legend")
                        yield Static("", id="flow-flow")
                    with Collapsible(title=_SEC_TITLE["sec-outcome"], collapsed=False, id="sec-outcome"):
                        yield Static("", id="outcome-summary")
                        yield DataTable(id="reward-table")
                        yield Static("", id="outcome-events")
                        yield Static("", id="outcome-log")
        # `y` reveals the current battle's replay path here — a full-width bar so the path sits on its
        # OWN line (not wrapped inside a toast), trivially selectable under `v` copy mode. Hidden until
        # used.
        yield Static("", id="replay-path-bar")

    def on_mount(self) -> None:
        self.query_one("#summary-moves", DataTable).add_columns("move", "eff", "prob")
        # summary-switches / summary-opp / team-our / team-opp are custom-rendered Static panels
        # (not DataTables) so the moveset sub-row can span the full width under each mon.
        self.query_one("#faith-table", DataTable).add_columns("action", "valid", "recorded", "re-run")
        self.query_one("#matchups-table", DataTable).add_columns("move", "×mult")
        self.query_one("#sweep-table", DataTable).add_columns("×mult", "P(chosen)", "P(switches)")
        self.query_one("#reward-table", DataTable).add_columns("reward component", "value")
        self.query_one("#saliency-table", DataTable).add_columns("obs block", "|grad|/dim", "sum")
        for tid in ("#board-our", "#board-opp"):
            self.query_one(tid, DataTable).add_columns("pokémon", "hp", "status")
        # Flow: a Static box-art dataflow diagram (obs → trunk bus → ⑂ fork → two π/V lanes).
        self.query_one("#flow-legend", Static).update(Text(
            "the model's forward DATAFLOW (obs → shared trunk → ⑂ fork → π/V), then what each head "
            "reads — bar = |∂output/∂obs| per obs block, normalized within head; green = high "
            "use, dim = barely read. SENSITIVITY, not proof of use.",
            style="dim italic"))

        self._build_tree()
        if self._injected_model is None:
            self.query_one("#ckpt-badge", Static).update(
                Text("select a battle to load its model", style="dim"))

    # ------------------------------------------------------------------
    # Trace tree
    # ------------------------------------------------------------------

    def _build_tree(self) -> None:
        tree = self.query_one("#trace-tree", Tree)
        tree.clear()
        if not self._root:
            tree.root.label = "no path given"
            return
        self._tree_model = build_trace_tree(self._root)
        self._gamma = self._read_run_gamma(self._tree_model.run_dir)
        self._review_store = ReviewStore(self._tree_model.run_dir)
        if self._tree_model.is_empty:
            tree.root.label = "no traces found"
            return
        flt = self._battle_filter
        shown = 0
        for step in self._tree_model.steps:
            step_node = tree.root.add(f"step {step.step:,}", expand=False)
            for opp in step.opponents:
                battles = [b for b in opp.battles if flt == "all" or b.outcome == flt]
                if not battles:
                    continue
                opp_node = step_node.add(f"{opp.name} ({len(battles)})", expand=False)
                for battle in battles:
                    opp_node.add_leaf(battle.label, data=battle)
                    shown += 1
            if not step_node.children:
                step_node.remove()
        suffix = "" if flt == "all" else f" · {flt} only"
        tree.root.label = f"traces ({shown}{suffix})"
        # Surface the most recent step so there's something to click immediately.
        if tree.root.children:
            tree.root.children[-1].expand()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        battle = event.node.data
        if isinstance(battle, BattleTrace):
            self._select_battle(battle)

    # ------------------------------------------------------------------
    # Battle / invocation selection
    # ------------------------------------------------------------------

    def _load_summary(self, battle: BattleTrace) -> dict:
        summ = self._summary_cache.get(battle.summary_path)
        if summ is None:
            with open(battle.summary_path) as f:
                summ = json.load(f)
            self._summary_cache[battle.summary_path] = summ
        return summ

    def _load_opp_team(self, battle: BattleTrace) -> "tuple | None":
        """The opponent's FULL team (species ids) from the trace's `reconstruction.json` sibling —
        PRIVILEGED referee data (bridge-eval traces only). `None` when the sibling is absent or
        unparseable (websocket/older traces) — the Summary then shows only the revealed mons + the
        anonymous belief. Cached per battle (file IO kept out of the pure engine)."""
        key = battle.summary_path
        if key in self._opp_team_cache:
            return self._opp_team_cache[key]
        team = None
        recon = (key[: -len("_summary.json")] + "_reconstruction.json"
                 if key.endswith("_summary.json") else None)
        if recon and os.path.exists(recon):
            try:
                from utils.bridge.reconstruction import ReconstructionRecord
                rec = ReconstructionRecord.load(recon)
                side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
                if side:
                    opp = "p2" if side == "p1" else "p1"
                    team = tuple(m["species"] for m in rec.team_details(opp))
            except Exception:  # noqa: BLE001 — privileged team is best-effort; degrade to no truth
                team = None
        self._opp_team_cache[key] = team
        return team

    def _load_our_hp_types(self, battle: BattleTrace) -> "dict | None":
        """OUR team's typed Hidden Power per species (`{norm_species: 'hiddenpower(bug)'}`) from the
        trace's `reconstruction.json` sibling — so our own mons show their HP TYPE even before they've
        revealed it (Showdown's request carries only the bare id, the type being IV-derived; an
        OPPONENT's un-revealed HP stays bare, no leak). `None` for websocket/older traces. Cached per
        battle (file IO kept out of the pure engine, mirroring `_load_opp_team`)."""
        key = battle.summary_path
        if key in self._our_hp_cache:
            return self._our_hp_cache[key]
        hp_map = None
        recon = (key[: -len("_summary.json")] + "_reconstruction.json"
                 if key.endswith("_summary.json") else None)
        if recon and os.path.exists(recon):
            try:
                from utils.bridge.reconstruction import ReconstructionRecord
                from main.prober.engine import build_our_hp_types
                rec = ReconstructionRecord.load(recon)
                side = rec.side_of(rec.trainee_username) if rec.trainee_username else None
                if side:
                    hp_map = build_our_hp_types(rec.team_details(side))
            except Exception:  # noqa: BLE001 — privileged team is best-effort; degrade to bare HP
                hp_map = None
        self._our_hp_cache[key] = hp_map
        return hp_map

    def _load_protocol(self, battle: BattleTrace) -> tuple:
        """The battle's raw Showdown protocol lines from its `*_replay.html` sibling (the same log a
        browser replay shows), parsed once and cached per trace. Empty tuple when the file is absent
        or unreadable — the Outcome panel then shows no raw log."""
        key = battle.summary_path
        if key in self._protocol_cache:
            return self._protocol_cache[key]
        lines: tuple = ()
        replay = (key[: -len("_summary.json")] + "_replay.html"
                  if key.endswith("_summary.json") else None)
        if replay and os.path.exists(replay):
            try:
                with open(replay, encoding="utf-8") as f:
                    lines = parse_protocol_log(f.read())
            except Exception:  # noqa: BLE001 — best-effort; degrade to no raw log
                lines = ()
        self._protocol_cache[key] = lines
        return lines

    def _select_battle(self, battle: BattleTrace) -> None:
        self._current_battle = battle
        self.query_one("#replay-path-bar", Static).display = False   # drop any stale yanked path
        self._current_summary = self._load_summary(battle)
        invs = self._current_summary["invocations"]
        meta = self._current_summary.get("meta", {})
        self.query_one("#battle-header", Static).update(
            f"{battle.opponent} · {meta.get('result', '?')} · "
            f"{meta.get('turns', '?')}t · {len(invs)} inv"
        )
        lv = self.query_one("#invocation-list", ListView)
        lv.clear()
        for i, inv in enumerate(invs):
            lv.append(ListItem(Label(self._inv_label(inv, i))))
        self._flagged = [i for i, inv in enumerate(invs)
                         if set(summary_flags(inv)) & set(_JUMP_FLAGS)]
        # Resolve + ensure the right model for THIS trace's step before selecting an
        # invocation (the model can differ per battle). If it must load, _analyze
        # queues via _pending_inv and runs when the model is ready.
        self._ensure_model_for(battle.step)
        if invs:
            target = 0
            if self._preselect_inv is not None and 0 <= self._preselect_inv < len(invs):
                target = self._preselect_inv
                self._preselect_inv = None
            # Setting the index fires ListView.Highlighted (index was None after
            # clear()), which drives _analyze — so we don't call it here too.
            lv.index = target

    def _inv_label(self, inv: dict, index: int) -> str:
        phase = str(inv.get("phase", "")).replace("_selection", "")
        glyphs = "".join(_FLAG_GLYPH.get(f, "") for f in summary_flags(inv))
        # User review annotations take a leading column so flagged decisions stand out.
        bid = self._battle_id()
        mark = ""
        if bid is not None and self._review_store is not None:
            if self._review_store.flag(bid, index):
                mark = _REVIEW_FLAG_GLYPH
            elif self._review_store.note(bid, index):
                mark = _REVIEW_NOTE_GLYPH
        head = f"{mark or ' '} "
        tail = f"  {glyphs}" if glyphs else ""
        return f"{head}t{inv.get('turn', '?'):>3} {phase:<6} {inv.get('chosen', '')}{tail}"

    def _battle_id(self) -> "str | None":
        """Stable per-trace key for the review store: the trace path relative to the run dir."""
        if self._current_battle is None or self._tree_model is None:
            return None
        try:
            return os.path.relpath(self._current_battle.summary_path, self._tree_model.run_dir)
        except ValueError:
            return self._current_battle.summary_path

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "invocation-list" and event.list_view.index is not None:
            self._current_inv = event.list_view.index
            self._analyze(event.list_view.index)

    def action_next_inv(self) -> None:
        self.query_one("#invocation-list", ListView).action_cursor_down()

    def action_prev_inv(self) -> None:
        self.query_one("#invocation-list", ListView).action_cursor_up()

    def action_next_flagged(self) -> None:
        self._jump_flagged(+1)

    def action_prev_flagged(self) -> None:
        self._jump_flagged(-1)

    def _jump_flagged(self, direction: int) -> None:
        """Move the invocation cursor to the next/prev flagged decision."""
        if not self._flagged:
            return
        lv = self.query_one("#invocation-list", ListView)
        cur = lv.index if lv.index is not None else -1
        if direction > 0:
            nxt = next((i for i in self._flagged if i > cur), self._flagged[0])
        else:
            nxt = next((i for i in reversed(self._flagged) if i < cur), self._flagged[-1])
        lv.index = nxt

    def action_cycle_filter(self) -> None:
        """Cycle the battle-outcome filter (all → loss → win) and rebuild the tree."""
        order = ("all", "loss", "win")
        self._battle_filter = order[(order.index(self._battle_filter) + 1) % len(order)]
        self._build_tree()

    def action_toggle_section(self, sec_id: str) -> None:
        """Open/close an analysis section (multiple can be open at once)."""
        sec = self.query_one(f"#{sec_id}", Collapsible)
        sec.collapsed = not sec.collapsed
        if not sec.collapsed:
            sec.scroll_visible()

    def action_reanalyze(self) -> None:
        lv = self.query_one("#invocation-list", ListView)
        if lv.index is not None:
            self._analyze(lv.index)

    # ------------------------------------------------------------------
    # Per-battle model resolution (exact → nearest → most recent)
    # ------------------------------------------------------------------

    def _ensure_model_for(self, step: int) -> None:
        """Resolve and activate the model for an eval `step`, loading it if needed.

        Updates the badge, then: injected (tests) → use it; cached → activate; else
        kick a background load. Analysis for the current invocation runs once the
        model is active (queued via _pending_inv while loading)."""
        if self._tree_model is None:
            return
        choice = resolve_model_for_step(
            self._tree_model, step, self._ckpt_override, self._tier)
        self._current_choice = choice

        if self._injected_model is not None:
            self._model = self._injected_model
            self._active_path = choice.path or "<injected>"
            self._update_badge(choice, loading=False)
            return

        if choice.path is None:
            self._model = None
            self._active_path = None
            self._update_badge(choice, loading=False)
            return

        if choice.path == self._active_path and self._model is not None:
            self._update_badge(choice, loading=False)
            return

        cached = self._model_cache.get(choice.path)
        if cached is not None:
            self._model = cached
            self._active_path = choice.path
            self._update_badge(choice, loading=False)
            return

        # Must load — clear the active model so analysis queues until it's ready.
        self._model = None
        self._active_path = None
        self._update_badge(choice, loading=True)
        self._load_model_worker(choice.path)

    def _update_badge(self, choice: ModelChoice, loading: bool) -> None:
        m = choice.manifest or {}
        git = (m.get("git_hash") or "")[:7]
        arch = m.get("arch_signature") or ""
        ident = "  ·  ".join(p for p in (f"git {git}" if git else "", arch) if p)
        tier_style = {"exact": "bold green", "override": "bold cyan",
                      "nearest": "yellow", "recent": "yellow", "none": "bold red"}.get(
                          choice.tier, "white")
        name = os.path.basename(choice.path) if choice.path else "—"
        head = Text()
        if loading:
            head.append("⏳ ", style="yellow")
        head.append(f"[{choice.tier}] ", style=tier_style)
        head.append(f"{name}  ", style="bold")
        head.append(choice.detail, style="dim")
        if ident:
            head.append(f"   trace: {ident}", style="dim")
        self.query_one("#ckpt-badge", Static).update(head)

    def action_cycle_model(self) -> None:
        """Cycle the resolution preference (auto → nearest → recent) and reload."""
        self._tier = TIERS[(TIERS.index(self._tier) + 1) % len(TIERS)]
        if self._current_battle is not None:
            self._ensure_model_for(self._current_battle.step)
            lv = self.query_one("#invocation-list", ListView)
            if self._model_ready and lv.index is not None:
                self._analyze(lv.index)

    def action_reload_model(self) -> None:
        if self._active_path and self._active_path in self._model_cache:
            del self._model_cache[self._active_path]
        self._model = None
        self._active_path = None
        if self._current_battle is not None:
            self._ensure_model_for(self._current_battle.step)

    @work(thread=True, exclusive=True, group="load")
    def _load_model_worker(self, ckpt_path: str) -> None:
        from main.prober.model import ProbeModel

        try:
            model = ProbeModel.load(ckpt_path)
        except Exception as e:  # noqa: BLE001 — surface load/arch errors in the UI
            self.call_from_thread(self._on_model_error, ckpt_path, str(e))
            return
        self.call_from_thread(self._on_model_ready, ckpt_path, model)

    def _on_model_ready(self, ckpt_path: str, model) -> None:
        self._model_cache[ckpt_path] = model
        # Only activate if this is still the path the current battle wants.
        if self._current_choice is not None and self._current_choice.path == ckpt_path:
            self._model = model
            self._active_path = ckpt_path
            self._update_badge(self._current_choice, loading=False)
            if self._pending_inv is not None:
                inv, self._pending_inv = self._pending_inv, None
                self._analyze(inv)

    def _on_model_error(self, ckpt_path: str, msg: str) -> None:
        self.query_one("#ckpt-badge", Static).update(
            Text(f"✗ load failed ({os.path.basename(ckpt_path)}): {msg}", style="bold red"))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self, inv_index: int) -> None:
        if self._current_battle is None or self._current_summary is None:
            return
        if not self._model_ready:
            self._pending_inv = inv_index
            return
        self._analyze_token += 1
        self._set_analysis_status(f"computing inv {inv_index}…")
        self._analyze_worker(self._current_battle, inv_index, self._analyze_token)

    @work(thread=True, exclusive=True, group="analyze")
    def _analyze_worker(self, battle: BattleTrace, inv_index: int, token: int) -> None:
        try:
            npz = self._load_npz(battle)
            analysis = analyze_invocation(
                self._model, self._load_summary(battle), npz, inv_index,
                summary_path=battle.summary_path, npz_path=battle.npz_path,
                opp_team=self._load_opp_team(battle),
                our_hp_types=self._load_our_hp_types(battle),
            )
        except Exception as e:  # noqa: BLE001 — render analysis errors, don't crash
            self.call_from_thread(self._on_analysis_error, str(e), token)
            return
        self.call_from_thread(self._render_analysis, analysis, token)

    def _load_npz(self, battle: BattleTrace):
        """Load the per-invocation obs eagerly into a dict so we never hold an open
        npz handle while the user browses dozens of battles."""
        if battle.npz_path is None:
            return {}
        with np.load(battle.npz_path) as z:
            return {k: z[k] for k in z.files}

    def _render_analysis(self, a: InvocationAnalysis, token: int) -> None:
        if token != self._analyze_token:
            return  # a newer selection superseded this result
        self._set_analysis_status("")
        self._last_analysis = a   # cached so a note-add can cheaply re-render the review card
        if self._current_battle is not None:
            meta = a.meta
            hdr = Text()
            # Always-visible obs-mismatch flag (above the invocation list) — so it's clear no matter
            # which analysis section is on screen, not just the Summary banner.
            if a.obs_mismatch is not None:
                hdr.append(f"⚠ OBS MISMATCH {a.obs_mismatch[0]}≠{a.obs_mismatch[1]}  ",
                           style="bold white on red")
            hdr.append(f"{a.our_species} vs {a.opp_species} · {meta.result} · "
                       f"turn {a.turn} · inv {a.inv_index}/{meta.n_invocations}")
            self.query_one("#battle-header", Static).update(hdr)
        self._render_summary(a)
        self._render_team(a)
        self._render_review(a)
        self._render_board(a)
        self._render_faithfulness(a)
        self._render_matchups(a)
        self._render_sweep(a)
        self._render_saliency(a)
        self._render_flow(a)
        self._render_outcome(a)

    def _on_analysis_error(self, msg: str, token: int) -> None:
        if token != self._analyze_token:
            return
        self._set_analysis_status(f"✗ {msg}")

    def _set_analysis_status(self, msg: str) -> None:
        # Reuse the battle header line as a transient status when analyzing.
        if msg:
            self.query_one("#battle-header", Static).update(Text(msg, style="yellow"))

    # ---- per-panel renderers ----

    def _td_residual(self, a: InvocationAnalysis) -> "float | None":
        """γ-discounted TD residual δ = r + γV(s′) − V(s) — the critic-surprise that
        flags a funky turn. None when value / next-value / reward aren't all captured."""
        if a.value is None or a.value.next_recorded is None:
            return None
        reward = (a.outcome or {}).get("reward")
        rtotal = reward.get("total") if isinstance(reward, dict) else None
        if rtotal is None:
            return None
        return float(rtotal) + self._gamma * a.value.next_recorded - a.value.recorded

    def _render_summary(self, a: InvocationAnalysis) -> None:
        """The decision dashboard: a context header (matchup+status+item · field · chose ·
        critic · threat) over two side-by-side tables — MOVES (effectiveness + prob) and
        SWITCHES (prob · hp · status · item · incoming KO-risk) — so a turn is judgeable
        in one glance."""
        head = Text()
        # Obs-version mismatch banner — the trace's obs predates a later obs change, so every
        # obs-OFFSET panel (THREAT incoming-eff/P(KO)/outspeed, RESULT crit/boost/move-order,
        # Matchups incoming, Saliency) is misaligned and UNRELIABLE. Board / items / movesets still hold.
        if a.obs_mismatch is not None:
            tdim, edim = a.obs_mismatch
            head.append("⚠ OBS MISMATCH ", style="bold white on red")
            head.append(f" trace obs {tdim} ≠ code {edim} — incoming/threat/crit/boost/saliency "
                        "panels are UNRELIABLE (this model predates a later obs change). "
                        "Board · items · movesets still hold.\n", style="bold red")
        # SITUATION group — line 1 matchup: each active's species + HP bar + status/volatiles
        # ("TOX(5)|SUB") + boosts ({atk:-1 spa:+6}) + held item (opp's once revealed); then FIELD,
        # THREAT. A blank line separates this from the DECISION group, then the OUTCOME group —
        # the three-way chunking keeps the dense header scannable.
        bd = a.board
        _append_summary_active(head, a.our_species, bd.ours.active_hp if bd else "",
                               bd.ours.status if bd else "", bd.ours.boosts if bd else "",
                               bd.ours.item if bd else "")
        head.append(" vs ", style="dim")
        _append_summary_active(head, a.opp_species, bd.opp.active_hp if bd else "",
                               bd.opp.status if bd else "", bd.opp.boosts if bd else "",
                               bd.opp.item if bd else "")
        head.append(f"   ·   turn {a.turn}", style="dim")
        result = (a.meta.result if a.meta is not None else None) or "?"
        head.append("   ·   ", style="dim")
        head.append(str(result).upper(),
                    style={"win": "bold green", "loss": "bold red"}.get(str(result).lower(), "dim"))
        # FIELD — weather / hazards / screens / turn (the highlighted Board line).
        head.append("\nFIELD   ", style="dim")
        head.append(_field_text(a.field))
        # THREAT — the danger it faced, STACKED so the Summary is self-sufficient: line 1 the
        # incoming-damage KO belief + speed; line 2 the incoming type-effectiveness (folded in
        # from Matchups). (P(KO) reds with danger, matching Matchups.)
        inc = a.incoming
        th = a.threats
        if (inc is not None and inc.active_pko is not None) or th is not None:
            head.append("\nTHREAT  ", style="dim")
            if inc is not None and inc.active_pko is not None:
                head.append(f"incoming P(KO) {inc.active_pko * 100:.0f}%",
                            style=gradient_color(1.0 - inc.active_pko))
                if inc.active_outspeed is not None:
                    head.append(f"   ·   we outspeed {inc.active_outspeed * 100:.0f}%", style="dim")
                head.append(f"   ·   worst-on-team {inc.max_pko * 100:.0f}%", style="dim")
                if inc.recovery_known or inc.recovery_rate > 0:
                    head.append(f"   ·   opp recovery {inc.recovery_rate * 100:.0f}%"
                                + ("✓" if inc.recovery_known else "?"), style="dim")
            if th is not None:
                head.append("\n        incoming eff (opp→us): ", style="dim")
                if not th.present:
                    head.append("BLANK — opp coverage unrevealed (priors only)", style="bold yellow")
                else:
                    head.append(f"worst {th.max_incoming:.2f}×", style=_mult_color(th.max_incoming))
                    head.append(f"   ·   revealed {th.revealed_frac * 100:.0f}%", style="dim")
        # DECISION group (blank line above) — what it chose + confidence (+ a disagree flag).
        chosen_p = _chosen_prob(a)
        head.append("\n\nCHOSE   ", style="dim")
        head.append("▶ " + (a.chosen or "?"), style="bold")
        if chosen_p is not None:
            head.append(f"  {chosen_p * 100:.1f}%", style=gradient_color(chosen_p))
        if a.rerun_argmax is not None and not a.agrees:
            head.append("   ⚠ now prefers ", style="yellow")
            head.append(str(a.rerun_argmax), style="bold yellow")
        # OUTCOME group (blank line above) — RESULT (what happened + events), AFTER, REWARD, CRITIC.
        _append_happened(head, a, "\n\nRESULT  ")
        # AFTER — the RESOLVED board at the start of the next decision, so before (matchup line) →
        # after reads at a glance. Mirror the matchup line via _append_summary_active so it carries
        # the SAME species + HP bar + [status] + {boosts} + @item (a freshly applied PAR/SLP/boost
        # shows here, not just on the before line) — a switch/faint shows the new mon on the field.
        nb = a.next_board
        if nb is not None:
            head.append("\nAFTER   ", style="dim")
            _append_summary_active(head, nb.ours.active_species, nb.ours.active_hp,
                                   nb.ours.status, nb.ours.boosts, nb.ours.item)
            head.append("   vs   ", style="dim")
            _append_summary_active(head, nb.opp.active_species, nb.opp.active_hp,
                                   nb.opp.status, nb.opp.boosts, nb.opp.item)
        # REWARD — the reward the env actually assigned (total + per-component breakdown).
        reward = (a.outcome or {}).get("reward")
        if isinstance(reward, dict):
            head.append("\nREWARD  ", style="dim")
            total = reward.get("total")
            if total is not None:
                head.append(f"{float(total):+.4f}",
                            style=gradient_color(max(0.0, min(1.0, (float(total) + 3) / 6))))
            for k, val in reward.items():
                if k != "total":
                    head.append(f"   ·   {k}: {val}", style="dim")
        elif reward is not None:
            head.append(f"\nREWARD  {reward}", style="dim")
        # CRITIC — last: WHY this turn is worth a look (ΔV / TD-surprise spikes).
        if a.value is not None:
            head.append("\nCRITIC  ", style="dim")
            head.append(f"V {a.value.recorded:+.2f}", style="bold")
            if a.value.delta is not None:
                head.append("   ΔV ", style="dim")
                head.append(f"{a.value.delta:+.2f}", style=("green" if a.value.delta >= 0 else "red"))
                _append_surprise(head, self._td_residual(a))
        # WIN-PROB — the calibrated P(win) the head reads + ΔP(win), how much this move moved the win
        # odds (the interpretable [0,1] complement to CRITIC's shaped-return V). Only on a
        # --win-prob-mode run (None otherwise). Greener = better odds (red at low P(win), like THREAT).
        if a.win_prob is not None:
            head.append("\nWIN-PROB  ", style="dim")
            head.append(f"P(win) {a.win_prob.recorded * 100:.0f}%",
                        style=gradient_color(max(0.0, min(1.0, a.win_prob.recorded))))
            if a.win_prob.delta is not None:
                head.append("   ΔP ", style="dim")
                head.append(f"{a.win_prob.delta * 100:+.0f}%",
                            style=("green" if a.win_prob.delta >= 0 else "red"))
        # VALUE-DIST — the distributional critic's predicted RETURN DISTRIBUTION (v29): a one-line
        # histogram + its shape, the interpretability read the scalar V collapses (sharp = confident,
        # wide = uncertain, bimodal = the critic sees a coinflip). Only on a --value-dist-mode run.
        if a.value_dist is not None:
            vd = a.value_dist
            head.append("\nVALUE-DIST  ", style="dim")
            head.append(f"E[Z] {vd.mean:+.2f}", style="bold")
            if vd.mean_real is not None:
                head.append(f" (real {vd.mean_real:+.1f})", style="dim")
            head.append(f"   σ {vd.std:.2f}   P10–90 [{vd.p10:+.1f}, {vd.p90:+.1f}]   H {vd.entropy:.2f}",
                        style="dim")
            if vd.bimodality > 0.35:
                head.append("   ⑂ bimodal", style="bold yellow")
            head.append("\n  ", style="dim")
            _append_dist_hist(head, vd)
        self.query_one("#summary-head", Static).update(head)

        # MOVES — fuse type-effectiveness (Matchups) with the policy prob (Faithfulness),
        # ranked by prob so the policy's preference order reads top-down.
        mt = self.query_one("#summary-moves", DataTable)
        mt.clear()
        # A non-damaging move (Spikes/Toxic/Protect) carries a phantom type multiplier in the
        # obs — meaningless, so map those slots to None → an "—" eff cell, not a misleading ×mult.
        mult_by_label = ({l: (m if ap else None)
                          for l, m, ap in zip(a.matchups.move_labels, a.matchups.multipliers,
                                              a.matchups.applicable)}
                         if a.matchups is not None else {})
        move_rows = [r for r in (a.actions or []) if not r.label.startswith("switch")]
        if not move_rows:
            mt.add_row(_warn_cell(a), "", "")
        for r in sorted(move_rows, key=lambda r: r.recorded, reverse=True):
            mult = mult_by_label.get(r.label)
            eff = (Text(f"{mult:4.2f}×", style=_mult_color(mult)) if mult is not None
                   else Text("—", style="dim"))
            label = ("▶ " if r.is_chosen else "  ") + r.label
            lstyle = "bold" if r.is_chosen else ("" if r.valid else "dim")
            # Disabled (no-PP / locked) moves render grey, not the red of a low prob.
            prob_style = (_DISABLED_GREY if not r.valid
                          else "bold" if r.is_chosen else gradient_color(r.recorded))
            mt.add_row(Text(label, style=lstyle), eff,
                       Text(f"{r.recorded * 100:5.1f}%", style=prob_style))

        # SWITCHES — a custom panel (not a DataTable) so each pivot's moveset can span the FULL
        # width below it: prob · hp · status · risk-in (item inlined into the name as "(item)").
        # The i-th switch action == team slot i == per_slot_pko[i] (fixed obs action layout), so
        # pair BEFORE sorting by prob.
        per_slot = list(inc.per_slot_pko) if (inc is not None and inc.per_slot_pko) else []
        switch_rows = [r for r in (a.actions or []) if r.label.startswith("switch")]
        paired = [(r, per_slot[i] if i < len(per_slot) else None)
                  for i, r in enumerate(switch_rows)]
        attrs = _side_attr_map(a.board.ours) if a.board is not None else {}
        sw = Text()
        sw.append(f"{'target':<{_PANEL_NAME_W}}{'prob':<8}{'hp':<{_PANEL_HP_W}}"
                  f"{'status':<{_PANEL_STAT_W}}risk-in", style="bold")
        if not switch_rows:
            sw.append("\nno switch available", style="dim")
        for r, pko in sorted(paired, key=lambda rp: rp[0].recorded, reverse=True):
            target = r.label.split(":", 1)[-1]
            disabled = not r.valid   # illegal switch (fainted / the active mon) → grey, not red
            hp, status, item, moves = attrs.get(target.lower(), (None, "", "", ()))
            prob_style = (_DISABLED_GREY if disabled
                          else "bold" if r.is_chosen else gradient_color(r.recorded))
            hp_cell = _hp_bar(hp, disabled=disabled) if hp is not None else Text("?", style="dim")
            if disabled:
                risk = Text("—", style="dim")
            elif pko is None:
                risk = Text("?", style="dim")
            else:
                risk = Text(f"{pko * 100:.0f}%", style=gradient_color(1.0 - pko))
            row = Text()
            row.append_text(_col(_mon_label(target, item, chosen=r.is_chosen, disabled=disabled),
                                 _PANEL_NAME_W))
            row.append_text(_col(Text(f"{r.recorded * 100:.1f}%", style=prob_style), 8))
            row.append_text(_col(hp_cell, _PANEL_HP_W))
            row.append_text(_col(_status_cell(status), _PANEL_STAT_W))
            row.append_text(risk)
            sw.append("\n")
            sw.append_text(row)
            ml = _moves_line(moves)
            if ml:
                sw.append("\n")
                sw.append_text(ml)
        self.query_one("#summary-switches", Static).update(sw)

        # OPP TEAM — the opponent's revealed mons (the mirror of our switches), shared team panel;
        # plus the model's belief about the hidden mons. With the PRIVILEGED truth (reconstruction.json)
        # show the slot-MATCHED truth-vs-guess (✓/✗ per hidden mon); otherwise the anonymous belief.
        opp_panel = _team_panel_text(a.board.opp if a.board is not None else None)
        if a.belief_truth is not None:
            _append_belief_truth(opp_panel, a.belief_truth)
        else:
            _append_belief(opp_panel, a.belief)
        self.query_one("#summary-opp", Static).update(opp_panel)

    def _render_team(self, a: InvocationAnalysis) -> None:
        """Full team detail: every mon's moveset (ours complete; opp's revealed-only) + hp ·
        status · item, decoded from the obs. The one place to read 'what does each mon do'."""
        self.query_one("#team-our", Static).update(
            _team_panel_text(a.board.ours if a.board is not None else None))
        self.query_one("#team-opp", Static).update(
            _team_panel_text(a.board.opp if a.board is not None else None))

    def _render_board(self, a: InvocationAnalysis) -> None:
        summ = self.query_one("#board-summary", Static)
        our_t = self.query_one("#board-our", DataTable)
        opp_t = self.query_one("#board-opp", DataTable)
        our_t.clear()
        opp_t.clear()
        bd = a.board
        if bd is None:
            summ.update("")
            return
        line = Text()
        _append_active(line, "▶ ", bd.ours)
        line.append("   vs   ", style="dim")
        _append_active(line, "", bd.opp)
        if bd.ours.moves:
            line.append("\nmoves: ", style="dim")
            line.append(", ".join(bd.ours.moves))
        if bd.ours.boosts or bd.opp.boosts:
            line.append("\nboosts: ", style="dim")
            line.append(f"ours {bd.ours.boosts or '—'}  ·  opp {bd.opp.boosts or '—'}",
                        style="magenta")
        summ.update(line)
        self.query_one("#board-field", Static).update(_field_text(a.field))
        self._fill_team(our_t, bd.ours)
        self._fill_team(opp_t, bd.opp)

    @staticmethod
    def _fill_team(table: DataTable, side) -> None:
        table.add_row(Text("▶ " + side.active_species, style="bold"),
                      _hp_text(side.active_hp), _status_cell(side.status))
        for m in side.bench:
            sp_style = "dim" if m.fainted else ""
            hp = Text("faint", style="dim red") if m.fainted else _hp_text(m.hp)
            table.add_row(Text(m.species, style=sp_style), hp, _status_cell(m.status))

    def _render_faithfulness(self, a: InvocationAnalysis) -> None:
        t = self.query_one("#faith-table", DataTable)
        t.clear()
        if not a.actions:
            t.add_row(_warn_cell(a), "", "", "")
            return
        for row in a.actions:
            label = ("▶ " if row.is_chosen else "  ") + row.label
            drift = abs(row.recorded - row.rerun)
            rerun_col = gradient_color(max(0.0, 1.0 - drift * 10))  # green when faithful
            style = "bold" if row.is_chosen else ("" if row.valid else "dim")
            t.add_row(
                Text(label, style=style),
                Text("✓" if row.valid else "·", style=style or "dim"),
                Text(f"{row.recorded * 100:5.1f}%", style=style),
                Text(f"{row.rerun * 100:5.1f}%", style=rerun_col),
            )

    def _render_matchups(self, a: InvocationAnalysis) -> None:
        t = self.query_one("#matchups-table", DataTable)
        t.clear()
        if a.matchups is not None:
            for label, mult, ap in zip(a.matchups.move_labels, a.matchups.multipliers,
                                       a.matchups.applicable):
                if not ap:  # non-damaging move → the obs multiplier is a phantom, show n/a
                    t.add_row(label, Text("—  n/a (non-damaging)", style="dim"))
                    continue
                bar = "█" * int(round(mult / 4.0 * 12))
                t.add_row(label, Text(f"{mult:4.2f}× {bar}", style=_mult_color(mult)))
        # Incoming threat (opp → us), decoded from their_matchups. The key tell is
        # `present`: it's blank for an opponent whose moves aren't revealed yet (a
        # just-switched-in mon), so the policy is pricing it from priors alone.
        threat = self.query_one("#matchups-threat", Static)
        th = a.threats
        lines = Text()
        if a.obs_mismatch is not None:   # the incoming/their_matchups offsets are misaligned here
            lines.append(f"⚠ OBS MISMATCH {a.obs_mismatch[0]}≠{a.obs_mismatch[1]} — the incoming "
                         "lines below are UNRELIABLE\n", style="bold red")
        if th is None:
            lines.append("incoming eff (opp→us): ", style="dim")
            lines.append("n/a", style="dim")
        else:
            lines.append("incoming eff (opp→us): ", style="dim")
            if not th.present:
                lines.append("BLANK — opp coverage unrevealed (priors only)", style="bold yellow")
            else:
                lines.append(f"worst {th.max_incoming:.2f}×", style=_mult_color(th.max_incoming))
                lines.append(f"  ·  revealed {th.revealed_frac * 100:.0f}%", style="dim")
        # The calibrated P(KO) belief (incoming_damage_v1) — prices power×Atk·Def×HP×roll, unlike
        # the raw-effectiveness line above. active_pko near 1.0 = obs says our on-field mon is
        # likely KO'd this turn; pair it with outspeed (are we even slower) and max_pko (worst on
        # the board) to read "should it switch?".
        inc = a.incoming
        if inc is not None:
            lines.append("\nincoming P(KO): ", style="dim")
            if inc.active_pko is None:
                lines.append("n/a", style="dim")
            else:
                # red = high incoming-KO danger (1 − pko), matching the Summary THREAT colour.
                lines.append(f"active {inc.active_pko * 100:.0f}%", style=gradient_color(1.0 - inc.active_pko))
                lines.append(f"  ·  outspd {inc.active_outspeed * 100:.0f}%", style="dim")
                lines.append(f"  ·  worst-on-team {inc.max_pko * 100:.0f}%", style="dim")
            if inc.recovery_known or inc.recovery_rate > 0:
                lines.append(f"  ·  opp-recovery {inc.recovery_rate * 100:.0f}%"
                             + ("✓" if inc.recovery_known else "?"), style="dim")
        # Unified DamageOperator (--unified-damage both): the model's OUTGOING per-move damage (our active
        # → opp active, action-aligned), so you can see how it ranks moves of EQUAL type-effectiveness —
        # the Earthquake-vs-Brick-Break tie-break. Each move's max-roll %HP + P(KO) (the realized,
        # accuracy-folded KO this turn). Only present on a --damage-op-with-outgoing checkpoint.
        dop = a.damage_op
        if dop is not None:
            # gen3_unified_move_system_v1: the opp active's per-status SECONDARY threat (its damaging
            # moves' para/flinch/freeze/burn — the axis the binary status flag missed), accuracy-folded +
            # ×Serene Grace. Show only the non-trivial ones.
            isec = dop.get("incoming_secondary") or {}
            inc_shown = [f"{c} {p * 100:.0f}%" for c, p in isec.items() if p > 0.05]
            if inc_shown:
                lines.append("\nopp 2ndary: ", style="dim")
                lines.append("  ·  ".join(inc_shown), style="yellow")
        if dop is not None and dop.get("outgoing"):
            moves = dop["outgoing"]["moves"]
            osec = dop["outgoing"].get("secondary") or [{} for _ in moves]
            labels = (list(a.matchups.move_labels) if a.matchups is not None
                      else [f"m{k}" for k in range(len(moves))])

            def _top_secondary(sc: dict) -> str:
                """The single most-likely secondary effect our move causes (what status + probability)."""
                items = [(c, p) for c, p in sc.items() if p > 0.05]
                if not items:
                    return ""
                c, p = max(items, key=lambda cp: cp[1])
                return f" ({c} {p * 100:.0f}%)"
            shown = [f"{lab} {mv['high'] * 100:.0f}%" + (f"→KO{mv['pko'] * 100:.0f}%" if mv["pko"] > 0.05 else "")
                     + _top_secondary(sc)
                     for lab, mv, sc in zip(labels, moves, osec) if mv["high"] > 0]
            lines.append("\nour damage (op): ", style="dim")
            lines.append("  ·  ".join(shown) if shown else "n/a", style="cyan")
        # gen3 has no team preview: the model BELIEVES the revealed opp's still-UNSEEN moves (--move-belief).
        # Show what it thinks (per revealed opp mon) + the op's per-OUR-mon incoming damage from that belief —
        # "what it thinks the damage is", including the moves it's only guessing.
        mb = a.move_belief
        if mb is not None and mb.opp:
            lines.append("\n\n— damage op belief (unseen moves) —", style="bold cyan")
            for ob in mb.opp:
                lines.append(f"\n{ob.species} ", style=_MON_COLOR)
                if ob.believed:
                    lines.append("≈ ", style="dim")
                    lines.append("  ·  ".join(f"{n} {p * 100:.0f}%" for n, p in ob.believed), style="magenta")
                else:
                    lines.append("(moveset fully revealed)", style="dim")
        # Per-OUR-mon believed incoming damage from the op (TEAM-SLOT order, labeled by species; the
        # active mon marked ▶). Worst channel's high-roll %HP + P(KO) — what it thinks the opp does to us.
        if dop is not None and mb is not None and dop.get("incoming"):
            parts = []
            for (slot, sp, act), row in zip(mb.our_labels, dop["incoming"]):
                if not sp:
                    continue
                high = max(row["phys"]["high"], row["spec"]["high"])
                pko = max(row["phys"]["pko"], row["spec"]["pko"])
                if high <= 0:
                    continue
                seg = f"{'▶' if act else ''}{sp} {high * 100:.0f}%" + (f"→KO{pko * 100:.0f}%" if pko > 0.05 else "")
                parts.append((seg, pko, act))
            if parts:
                lines.append("\nincoming (op): ", style="dim")
                for k, (seg, pko, act) in enumerate(parts):
                    if k:
                        lines.append("  ·  ", style="dim")
                    lines.append(seg, style=gradient_color(1.0 - pko) if (act or pko > 0.05) else "dim")
        threat.update(lines)

    def _render_sweep(self, a: InvocationAnalysis) -> None:
        t = self.query_one("#sweep-table", DataTable)
        t.clear()
        if a.sweep is None or not a.sweep.applicable:
            t.add_row(Text("chosen action is not a move", style="dim"), "", "")
            return
        for r in a.sweep.rows:
            t.add_row(
                f"{r.multiplier:.0f}×",
                Text(f"{r.p_chosen * 100:5.1f}%", style=gradient_color(r.p_chosen)),
                Text(f"{r.p_switches * 100:5.1f}%", style="dim"),
            )

    def _render_saliency(self, a: InvocationAnalysis) -> None:
        t = self.query_one("#saliency-table", DataTable)
        t.clear()
        if a.obs_mismatch is not None:   # obs-offset blocks are misaligned → saliency is on wrong dims
            t.add_row(Text(f"⚠ OBS MISMATCH {a.obs_mismatch[0]}≠{a.obs_mismatch[1]} — UNRELIABLE",
                           style="bold red"), "", "")
        # Two heads: π = policy logit saliency (what the ACTOR reads), V = critic value
        # saliency (what the VALUE head reads — the lens for OHKO tail-blindness). Each group is
        # normalized to its OWN peak so the bars compare blocks within a head. Watch the
        # incoming_damage block's V row vs its peers — that's "does the critic use the belief?".
        for tag, sal, style in (("π", a.saliency, "cyan"), ("V", a.value_saliency, "magenta")):
            if sal is None:
                continue
            peak = max((b.total_abs for b in sal.blocks), default=1.0) or 1.0
            for b in sal.blocks:
                bar = "█" * int(round(b.total_abs / peak * 14))
                t.add_row(f"{tag} {b.name}", f"{b.mean_abs:.4f}",
                          Text(f"{b.total_abs:6.2f} {bar}", style=style))

    def _render_flow(self, a: InvocationAnalysis) -> None:
        """Draw the model as a DATAFLOW diagram in the Static `#flow-flow` (not a list): a single
        left rail is the forward spine; phases tee off it in stage BANDS (ENCODE / BELIEF / ⑂ FORK)
        — active numbered, optional-off dim `⌀`, side readouts as `┄▷ … ✗→heads`; the CLSPool fork
        SPLITS the rail into two SIDE-BY-SIDE π/V lanes (so their attribution bars sit at the same
        row for read-across). Below ~88 cols the lanes stack. SENSITIVITY, not proof of causal use."""
        static = self.query_one("#flow-flow", Static)
        if a.obs_mismatch is not None:   # offsets misaligned → the per-block split is on wrong dims
            static.update(Text(f"⚠ OBS MISMATCH {a.obs_mismatch[0]}≠{a.obs_mismatch[1]} — "
                               "attribution UNRELIABLE", style="bold red"))
            return
        # Forward pipeline, introspected live (empty for a stub/fake model → heads only, no spine).
        arch = []
        model = getattr(self, "_model", None)
        if model is not None and hasattr(model, "architecture"):
            try:
                arch = model.architecture() or []
            except Exception:  # noqa: BLE001 — introspection is best-effort; never break the panel
                arch = []
        reads = {ph["stage"]: ph["role"] for ph in arch if ph["stage"] in ("policy", "value")}
        # Width-aware role budget — use the panel's real width so descriptions aren't needlessly cut.
        role_w = max(40, min(96, self._flow_width() - 33))
        lines = list(self._flow_pipeline_lines(arch, role_w)) if arch else []
        pol = self._flow_head_lane("π POLICY", "policy", a.saliency, "bold cyan",
                                   self._flow_policy_caption(a), reads)
        val = self._flow_head_lane("V VALUE", "value", a.value_saliency, "bold magenta",
                                   self._flow_value_caption(a), reads)
        if pol is None and val is None:          # model-free trace / no state → no per-decision grads
            lines.append(Text("(no per-decision attribution — model-free trace / no captured state)",
                              style="dim"))
        else:
            if arch:
                lines += self._flow_fork_lines()
            if pol is not None and val is not None and self._flow_width() >= MIN_TWO_LANE:
                lines += self._flow_combine_lanes(pol, val)     # side-by-side, read-across
            else:                                                # narrow / one head → stack vertically
                for lane in (pol, val):
                    if lane:
                        lines += lane
                        lines.append(Text(""))
        lines.append(Text(""))
        lines.append(self._flow_legend())
        static.update(Text("\n").join(lines))

    def _flow_width(self) -> int:
        """Available width for the Flow Static (→ two-lane vs stacked). Big default when unknown."""
        try:
            w = self.query_one("#flow-flow", Static).container_size.width
        except Exception:  # noqa: BLE001
            return 999
        return w if w and w > 0 else 999

    def _flow_pipeline_lines(self, arch: "list[dict]", role_w: int) -> "list[Text]":
        """The forward spine: `obs` source, then each non-head phase grouped under a stage band that
        the rail flows DOWN into (`▼ BAND → <what it produces>`), so the dataflow reads top-to-bottom.
        Numbering runs across active non-side phases; CLSPool=⑂, assembler=◆, attention layers get ⊛."""
        # What each stage PRODUCES — a high-level edge label on the rail so you see the tensor move.
        produces = {"ENCODE": "→ role tokens, self-attended",
                    "BELIEF": "→ opp tokens enriched (moves / spread)",
                    "FORK": "→ pooled, then SPLIT → π · V"}
        out = [Text("obs vec(3457)", style="bold")]
        n = [0]
        bands = {b: [] for b in _FLOW_BAND_ORDER}
        for ph in arch:
            if ph["stage"] in ("policy", "value"):
                continue                         # the π/V heads render as the two lanes below
            bands[_flow_band(ph)].append(ph)
        for band in _FLOW_BAND_ORDER:
            phs = bands[band]
            if not phs:
                continue
            out.append(Text("│", style=_RAIL))                       # rail flows DOWN…
            hdr = Text("▼ ", style="bold cyan")                      # …into the next stage
            hdr.append("⑂ FORK" if band == "FORK" else band, style="bold cyan")
            hdr.append(f"  {produces.get(band, '')}", style="dim")   # what the stage produces
            out.append(hdr)
            for ph in phs:
                out.append(self._flow_phase_row(ph, n, role_w))
        return out

    def _flow_phase_row(self, ph: dict, n: "list[int]", role_w: int) -> Text:
        """One phase row tee'd off the rail: category glyph+colour (① required · green optional-on ·
        dim `·` off · `└┄▷` side), an ⊛ marker on a LIVE attention layer, then the (width-aware) role.
        No text tier-tag — the glyph/colour already encodes it (see the legend)."""
        name, active, stage = ph["name"], ph["active"], ph["stage"]
        line = Text("│  ", style=_RAIL)
        if not active:
            line.append("·  ", style="dim"); line.append(name, style="dim")
        elif stage == "side":               # branches OFF the trunk, never feeds the heads
            line.append("└┄▷ ", style="yellow"); line.append(name, style="yellow")
        elif stage == "fork":               # the CLSPool split
            line.append("⑂ ", style="bold cyan"); line.append(name, style="bold cyan")
        elif name == "ProjectionAssembler":
            line.append("◆ ", style="bold"); line.append(name, style="bold")
        else:
            n[0] += 1
            line.append(f"{_circled(n[0])} ", style="bold")
            line.append(name, style="bold green" if ph["optional"] else "bold")
        line = _pad(line, 27)               # align the ⊛ attention column across glyph/name widths
        line.append("⊛ " if (active and ph.get("attn")) else "  ", style="bold magenta")
        line.append(_trunc(ph["role"], role_w), style="dim")
        if stage == "side":
            line.append("  ✗→heads", style="dim italic yellow")
        return line

    def _flow_fork_lines(self) -> "list[Text]":
        """The split: the rail tees into the two lane starts (col 0 and col LANE_W+gutter)."""
        rcol = LANE_W + len(GUTTER)         # column where the right lane begins
        return [
            Text("│", style=_RAIL),
            Text("├" + "─" * (rcol - 1) + "┐", style=_RAIL),
            Text("▼" + " " * (rcol - 1) + "▼", style=_RAIL),
        ]

    def _flow_head_lane(self, tag: str, stage: str, sal, style: str, caption: str,
                        reads: dict) -> "list[Text] | None":
        """One head's lane (list of lines): caption · `↳ reads …` · the attribution bars (sorted
        most-read-first, smooth bar + abbreviated block + share %, dominant bold, <8% greyed).
        None when the head has no saliency (model-free / no state)."""
        if sal is None:
            return None
        head = Text(tag, style=style)
        if caption:
            head.append(f"  {caption}", style="dim")
        out = [head]
        if reads.get(stage):
            out.append(Text(f"↳ {_trunc(reads[stage], LANE_W - 2)}", style="dim italic"))
        blocks = sorted(sal.blocks, key=lambda b: b.total_abs, reverse=True)
        peak = max((b.total_abs for b in blocks), default=1.0) or 1.0
        for i, b in enumerate(blocks):
            share = b.total_abs / peak
            faint = share < 0.08
            row = Text()
            row.append(f"{_flow_bar(share, LANE_BAR):<{LANE_BAR}} ",
                       style="dim" if faint else gradient_color(share))
            row.append(_trunc(_abbr_block(b.name), 16).ljust(16),
                       style="bold" if i == 0 else "dim" if faint else "")
            row.append(f"{share * 100:3.0f}%", style="dim")
            out.append(row)
        return out

    def _flow_combine_lanes(self, left: "list[Text]", right: "list[Text]") -> "list[Text]":
        """Zip the two lanes side-by-side with a full-height gutter, padding each left line to
        LANE_W so the π/V bars sit at the same column for read-across comparison."""
        out = []
        for i in range(max(len(left), len(right))):
            lhs = left[i] if i < len(left) else Text("")
            rhs = right[i] if i < len(right) else Text("")
            line = _pad(lhs, LANE_W)
            line.append(GUTTER, style="dim")
            line.append_text(rhs)
            out.append(line)
        return out

    @staticmethod
    def _flow_legend() -> Text:
        t = Text("legend  ", style="dim")
        t.append("① required  ", style="cyan")
        t.append("① optional-on  ", style="green")
        t.append("· off  ", style="dim")
        t.append("┄▷ side (✗→heads)  ", style="yellow")
        t.append("⊛ attention  ", style="bold magenta")
        t.append("⑂ fork  ◆ assembler", style="cyan")
        return t

    def _flow_policy_caption(self, a: InvocationAnalysis) -> str:
        p = _chosen_prob(a)
        return f"chose {a.chosen} ({p * 100:.0f}%)" if p is not None else f"chose {a.chosen}"

    @staticmethod
    def _flow_value_caption(a: InvocationAnalysis) -> str:
        if a.value is None:
            return ""
        v = a.value
        # Show both: the de-normalized real-return V and (on a PopArt run) the normalized V —
        # the critic's own learning scale, comparable across the run's return-scale drift.
        if v.normalized_recorded is not None:
            return f"V(s) {v.recorded:+.2f}  ·  norm {v.normalized_recorded:+.2f}"
        return f"V(s) {v.recorded:+.2f}"

    @staticmethod
    def _read_run_gamma(run_dir: "str | None") -> float:
        """γ from the run's metadata.json (same source as ProbeSession), for the TD residual."""
        if not run_dir:
            return _DEFAULT_GAMMA
        try:
            with open(os.path.join(run_dir, "metadata.json")) as f:
                return float(json.load(f).get("gamma", _DEFAULT_GAMMA))
        except (OSError, ValueError, TypeError):
            return _DEFAULT_GAMMA

    def _render_outcome(self, a: InvocationAnalysis) -> None:
        # Value + model-agreement summary line(s).
        summary = Text()
        if a.value is not None:
            v = a.value
            summary.append(f"V(s) recorded {v.recorded:+.2f}", style="bold")
            # PopArt-normalized companion (the critic's own learning scale) when available.
            if v.normalized_recorded is not None:
                summary.append(f" (norm {v.normalized_recorded:+.2f})", style="dim")
            if v.rerun is not None:
                summary.append(f"  ·  re-run {v.rerun:+.2f}", style="dim")
                if v.normalized_rerun is not None:
                    summary.append(f" (norm {v.normalized_rerun:+.2f})", style="dim")
            if v.delta is not None:
                d_style = "green" if v.delta >= 0 else "red"
                summary.append("  ·  ΔV ", style="dim")
                summary.append(f"{v.delta:+.2f}", style=d_style)
                summary.append(f" → {v.next_recorded:+.2f}", style="dim")
                # TD residual δ = r + γV(s') − V(s): how surprised the critic was. Parity with
                # the CLI's overview/analyze td_residual (the decisive metric in loss forensics).
                td = self._td_residual(a)
                if td is not None:
                    summary.append("  · ", style="dim")
                    _append_surprise(summary, td, term="TD δ")
            summary.append("\n")
        # Win-probability head's read (calibrated P(win) + ΔP to the next decision) — the [0,1]
        # complement to V above; only on a --win-prob-mode run.
        if a.win_prob is not None:
            wp = a.win_prob
            summary.append(f"P(win) {wp.recorded * 100:.0f}%", style="bold")
            if wp.delta is not None:
                summary.append("  ·  ΔP ", style="dim")
                summary.append(f"{wp.delta * 100:+.0f}%", style=("green" if wp.delta >= 0 else "red"))
                if wp.next_recorded is not None:
                    summary.append(f" → {wp.next_recorded * 100:.0f}%", style="dim")
            summary.append("\n")
        agree_style = "green" if a.agrees else "bold red"
        summary.append("model: ", style="dim")
        summary.append(f"chosen={a.chosen}", style="bold")
        if a.rerun_argmax is not None:
            verdict = "agrees" if a.agrees else f"DISAGREES → {a.rerun_argmax}"
            summary.append(f"  ·  re-run {verdict}", style=agree_style)
        out = a.outcome or {}
        our, opp = out.get("our") or {}, out.get("opp") or {}
        if our or opp:
            summary.append(
                f"\nour: {our.get('action','?')} ({our.get('hp_delta','?')})"
                f"   opp: {opp.get('action','?')} ({opp.get('hp_delta','?')})", style="dim")
        self.query_one("#outcome-summary", Static).update(summary)

        # Reward breakdown (total first, then the component strings).
        rt = self.query_one("#reward-table", DataTable)
        rt.clear()
        reward = out.get("reward")
        if isinstance(reward, dict):
            total = reward.get("total")
            if total is not None:
                col = gradient_color(max(0.0, min(1.0, (float(total) + 3) / 6)))
                rt.add_row(Text("total", style="bold"), Text(f"{float(total):+.4f}", style=col))
            for k, val in reward.items():
                if k != "total":
                    rt.add_row(k, str(val))
        elif reward is not None:
            rt.add_row("total", f"{reward}")

        events = out.get("events") or []
        ev = self.query_one("#outcome-events", Static)
        ev.update(Text("events: " + (", ".join(map(str, events)) if events else "—"),
                       style="yellow" if events else "dim"))

        # Raw Showdown protocol for this decision's turn (parsed from the replay.html log) — so the
        # exact events the summary collapses (a |-miss|, the per-hit |-damage|, a switch-in) are
        # visible in-prober without opening the browser replay.
        lines = (protocol_for_turn(self._load_protocol(self._current_battle), a.turn)
                 if self._current_battle is not None else ())
        self.query_one("#outcome-log", Static).update(_protocol_text(lines, a.turn))

    # ------------------------------------------------------------------
    # Manual review (model's own games): expectation card + flag/note
    # ------------------------------------------------------------------

    def _render_review(self, a: InvocationAnalysis) -> None:
        """One-glance 'what the model EXPECTED → what it DID → what HAPPENED → how surprised'
        card, then sync the flag/note widgets to this decision."""
        card = Text()
        chosen_p = _chosen_prob(a)
        card.append("chose ", style="dim")
        card.append(a.chosen or "?", style="bold")
        if chosen_p is not None:
            card.append(f" ({chosen_p * 100:.0f}%)", style="dim")
        inc = a.incoming
        if inc is not None and inc.active_pko is not None:
            card.append("   expected ", style="dim")
            card.append(f"P(KO on us) {inc.active_pko * 100:.0f}%",
                        style=gradient_color(1.0 - inc.active_pko))
            card.append(f" · outspd {inc.active_outspeed * 100:.0f}%", style="dim")
        if a.value is not None:
            card.append("\nV(s) ", style="dim")
            card.append(f"{a.value.recorded:+.2f}", style="bold")
            if a.value.delta is not None:
                card.append("  ΔV ", style="dim")
                card.append(f"{a.value.delta:+.2f}",
                            style=("green" if a.value.delta >= 0 else "red"))
                _append_surprise(card, self._td_residual(a))
        if a.rerun_argmax is not None and not a.agrees:
            card.append("\n⚠ model now prefers ", style="yellow")
            card.append(str(a.rerun_argmax), style="bold yellow")
        _append_happened(card, a, "\nhappened ")

        # The timestamped note log for this decision (append-only — keys on the LIST POSITION
        # self._current_inv, set on highlight, NOT a.inv_index which can differ).
        bid = self._battle_id()
        idx = self._current_inv
        log = (self._review_store.notes(bid, idx)
               if (bid and self._review_store and idx is not None) else [])
        for ts, text in log:
            card.append("\n  • ", style="dim")
            if ts:
                card.append(f"{ts}  ", style="cyan")
            card.append(text)
        self.query_one("#review-card", Static).update(card)

        self._render_review_status()
        # The input adds a NEW log entry (append, not overwrite) — keep it empty.
        note_w = self.query_one("#review-note", Input)
        if note_w.value:
            note_w.value = ""

    def _render_review_status(self) -> None:
        st = self.query_one("#review-status", Static)
        bid = self._battle_id()
        if bid is None or self._review_store is None or self._current_inv is None:
            st.update("")
            return
        flagged = self._review_store.flag(bid, self._current_inv)
        n_ann = len(self._review_store.annotated_invs(bid))
        line = Text()
        line.append(f"{_REVIEW_FLAG_GLYPH} FLAGGED" if flagged else "unflagged",
                    style="bold red" if flagged else "dim")
        line.append(f"   ·   {n_ann} annotated here", style="dim")
        line.append("   (space=flag · e=note · [ ]=jump · E=export)", style="dim")
        st.update(line)

    def action_toggle_review_flag(self) -> None:
        bid = self._battle_id()
        if bid is None or self._review_store is None or self._current_inv is None:
            return
        self._review_store.toggle_flag(bid, self._current_inv)
        self._render_review_status()
        self._refresh_list_item(self._current_inv)

    def action_edit_note(self) -> None:
        try:
            self.query_one("#review-note", Input).focus()
        except Exception:  # noqa: BLE001 — no input mounted yet
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "review-note":
            return
        bid = self._battle_id()
        if bid is None or self._review_store is None or self._current_inv is None:
            return
        self._review_store.add_note(bid, self._current_inv, event.value)   # APPEND a timestamped entry
        event.input.value = ""
        # Re-render the card so the new (timestamped) entry shows in the log immediately.
        if self._last_analysis is not None:
            self._render_review(self._last_analysis)
        else:
            self._render_review_status()
        self._refresh_list_item(self._current_inv)
        self.query_one("#invocation-list", ListView).focus()

    def _refresh_list_item(self, index: int) -> None:
        invs = (self._current_summary or {}).get("invocations", [])
        if not (0 <= index < len(invs)):
            return
        try:
            item = self.query_one("#invocation-list", ListView).children[index]
            item.query_one(Label).update(self._inv_label(invs[index], index))
        except Exception:  # noqa: BLE001
            pass

    def action_next_annotated(self) -> None:
        self._jump_annotated(+1)

    def action_prev_annotated(self) -> None:
        self._jump_annotated(-1)

    def _jump_annotated(self, direction: int) -> None:
        bid = self._battle_id()
        if bid is None or self._review_store is None:
            return
        ann = self._review_store.annotated_invs(bid)
        if not ann:
            return
        lv = self.query_one("#invocation-list", ListView)
        cur = lv.index if lv.index is not None else 0
        if direction > 0:
            nxt = next((i for i in ann if i > cur), ann[0])
        else:
            nxt = next((i for i in reversed(ann) if i < cur), ann[-1])
        lv.index = nxt

    def action_export_notes(self) -> None:
        if self._review_store is None:
            return
        path = self._review_store.export_markdown()
        self.notify(f"exported → {os.path.basename(path)}" if path else "nothing to export")

    def action_copy_replay_path(self) -> None:
        """Reveal the current battle's browser-watchable Showdown replay path (`*_replay.html`) on a
        dedicated full-width bar — the path on its OWN line, so it's cleanly selectable under `v` copy
        mode (the portable path that works on Terminal.app). Also best-effort copies it to the
        clipboard for terminals that honour OSC-52 (kitty / iTerm2 / WezTerm)."""
        if self._current_battle is None:
            self.notify("no battle selected")
            return
        path = self._current_battle.summary_path.replace("_summary.json", "_replay.html")
        exists = os.path.exists(path)
        try:
            self.copy_to_clipboard(path)   # bonus on OSC-52 terminals; the bar is the portable path
        except Exception:  # noqa: BLE001 — clipboard may be unavailable / unsupported over ssh
            pass
        bar = self.query_one("#replay-path-bar", Static)
        hint = Text("replay path — press v, then select the line below to copy"
                    + ("" if exists else "   ⚠ file missing") + "\n", style="dim italic")
        hint.append(path, style="bold")
        bar.update(hint)
        bar.display = True
        self.notify("replay path shown below — v to copy" + ("" if exists else " (file missing!)"),
                    severity="information" if exists else "warning", timeout=4)


def _warn_cell(a: InvocationAnalysis):
    msg = a.warnings[0] if a.warnings else "no analysis"
    return Text(msg, style="yellow")


def _mult_color(mult: float) -> str:
    # 0× bad (red) … 4× great (green), centered on 1× neutral.
    return gradient_color(min(1.0, mult / 4.0))


# ---- shared cell/decode helpers (one source for Summary / Board / Review) ----

def _chosen_prob(a: InvocationAnalysis) -> "float | None":
    """The recorded policy probability of the action the model actually took."""
    return next((r.recorded for r in (a.actions or []) if r.is_chosen), None)


_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫"   # step numbers for the forward-pipeline chain


def _circled(n: int) -> str:
    """A circled step number (①..⑫) for the pipeline chain; falls back to '(n)' past the table."""
    return _CIRCLED[n - 1] if 1 <= n <= len(_CIRCLED) else f"({n})"


_FLOW_BAR_PARTS = " ▏▎▍▌▋▊▉"   # 8 sub-cell levels (index 0..7) for a smooth bar end


def _flow_bar(frac: float, width: int = 16) -> str:
    """A smooth (eighth-block) horizontal bar of ``width`` cells for a fraction in [0, 1] —
    full `█` cells plus a partial end cell, so small differences in attribution stay visible."""
    frac = max(0.0, min(1.0, frac))
    units = frac * width
    full = int(units)
    s = "█" * full
    if full < width:
        rem = int((units - full) * 8)   # 0..7 → never indexes past the 8-char parts string
        if rem:
            s += _FLOW_BAR_PARTS[rem]
    return s


# ── Flow dataflow-diagram layout (a Static box-art panel, not a Tree) ──────────────────────────
# A single left rail (│) is the forward spine; phases group into stage BANDS that tee off it; the
# CLSPool fork SPLITS the rail into two SIDE-BY-SIDE π/V lanes so the attribution bars sit at the
# same vertical level (read-across) and stay flush-left. Below MIN_TWO_LANE cols the lanes stack.
LANE_W = 38                          # cell width of each π/V head lane
LANE_BAR = 12                        # attribution bar width inside a lane (narrower than the table's 16)
GUTTER = " │ "                       # full-height divider drawn between the two lanes (dim)
MIN_TWO_LANE = 2 * LANE_W + len(GUTTER) + 1   # below this available width, stack the heads vertically
_RAIL = "dim cyan"                   # the flow spine / connectors / stage rules

# Which stage BAND each phase renders under (display grouping; forward order preserved within a band).
_FLOW_BAND = {
    "Embeddings": "ENCODE", "ObsUnpack": "ENCODE", "PokemonEncoder": "ENCODE",
    "MoveLatentEncoder": "ENCODE", "BeliefSlots": "ENCODE", "TeamTransformer": "ENCODE",
    "BeliefHead": "BELIEF", "MoveBelief": "BELIEF", "SpreadBelief": "BELIEF",
    "CLSPool": "FORK", "WinProbHead": "FORK", "ValueDistHead": "FORK", "HiddenOppBeliefPool": "FORK",
    "DamageOperator": "FORK", "ProjectionAssembler": "FORK",
}
_FLOW_BAND_ORDER = ("ENCODE", "BELIEF", "FORK")


def _flow_band(ph: dict) -> str:
    """Band for a phase — by name, else by stage (so a future phase still lands sensibly)."""
    b = _FLOW_BAND.get(ph["name"])
    if b:
        return b
    s = ph["stage"]
    return "FORK" if s in ("fork", "shared") else "BELIEF" if s == "side" else "ENCODE"


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _abbr_block(name: str) -> str:
    """Shorten the long obs-block names so a row fits a half-width lane (full names live in Saliency)."""
    return (name.replace("our active pokemon block", "our active")
                .replace("active move_multipliers", "move_mults"))


def _pad(t: Text, width: int) -> Text:
    """Right-pad a Text to a fixed CELL width (so two lanes / a role column align). Box/bar glyphs
    are single-cell, so cell_len is the true column count."""
    gap = width - t.cell_len
    if gap > 0:
        t = t.copy()
        t.append(" " * gap)
    return t


_DIST_BLOCKS = " ▁▂▃▄▅▆▇█"   # 0..8 eighths (index 0 = empty) — the value-dist histogram bars


def _append_dist_hist(line: Text, vd) -> None:
    """Append the value-dist head's per-atom return distribution as an inline eighth-block histogram
    (one bar per atom, height ∝ prob/peak, heat-coloured by magnitude so the mass reads at a glance;
    the modal bin bold-magenta). The shape IS the interpretability read — a single tall spike =
    confident, a broad hump = uncertain, two humps = a coinflip. No-op on an empty distribution."""
    probs = vd.probs
    if not probs:
        return
    pmax = max(probs) or 1.0
    peak = max(range(len(probs)), key=lambda k: probs[k])
    for k, p in enumerate(probs):
        ch = _DIST_BLOCKS[int(round((p / pmax) * (len(_DIST_BLOCKS) - 1)))]
        line.append(ch, style="bold magenta" if k == peak else gradient_color(p / pmax))


def _surprise_phrase(td: float) -> str:
    """Plain-language reading of the TD-surprise (the critic's prediction error, δ = r + γV(s′) −
    V(s)), so the ML term is self-explaining when reviewing: negative δ = the turn turned out worse
    than the critic predicted. Magnitude → 'much' for the big craters."""
    mag = abs(td)
    if mag < 0.5:
        return "about what the critic expected"
    much = "much " if mag >= 3.0 else ""
    return f"{much}{'better' if td > 0 else 'worse'} than the critic expected"


def _append_surprise(line: Text, td: "float | None", term: str = "surprise(TDδ)") -> None:
    """Append the TD-surprise value + its plain-language gloss (always paired, per the
    'keep ML terms human-friendly' rule). No-op when δ isn't computable."""
    if td is None:
        return
    line.append(f"   {term} ", style="dim")
    line.append(f"{td:+.2f}", style=("green" if td >= 0 else "red"))
    line.append(f" — {_surprise_phrase(td)}", style="dim")


def _item_style(item: str) -> str:
    # Choice items lock the moveset / boost a stat — high decision impact, so highlight them.
    return "bold magenta" if "choice" in (item or "").lower() else "cyan"


def _item_cell(item: str) -> Text:
    """A held-item table cell — '—' when unknown/none, Choice items highlighted."""
    return (Text(item, style=_item_style(item))
            if item and item.lower() != "none" else Text("—", style="dim"))


def _status_cell(status: str) -> Text:
    """A status/volatiles table cell (e.g. 'TOX(5)|SUB') — '—' when none."""
    return Text(status, style="yellow") if status else Text("—", style="dim")


_DISABLED_GREY = "grey50"   # disabled/fainted slots → neutral grey (NOT red — red means "low HP, real danger")


def _hp_bar(hp: str, width: int = 6, disabled: bool = False) -> Text:
    """A compact colour-graded health bar + percentage, e.g. green '████▌░ 76%'. ``disabled``
    (a fainted mon / an illegal switch) renders grey rather than the HP gradient, so 'dead/
    unavailable' reads differently from 'alive but low' (which stays red). Unparseable → dim."""
    f = _hp_frac(hp)
    if f is None:
        return Text(str(hp), style="dim")
    filled = int(round(f * width))
    col = _DISABLED_GREY if disabled else gradient_color(f)
    t = Text()
    t.append("█" * filled + "░" * (width - filled), style=col)
    t.append(f" {hp}", style=col)
    return t


def _append_summary_active(line: Text, species: str, hp: str, status: str,
                           boosts: str, item: str) -> None:
    """Append 'species <hp-bar> [status] {boosts} @item' for one active mon to the Summary header
    line — status (yellow), boosts (magenta, distinct), item shown only when present."""
    line.append(species, style="bold")
    if hp:
        line.append(" ")
        line.append_text(_hp_bar(hp))
    if status:
        line.append(f" [{status}]", style="yellow")
    if boosts:
        line.append(f" {{{boosts}}}", style="magenta")   # e.g. {atk:-1 spa:+6} — boost stages
    if item and item.lower() != "none":
        line.append(f" @{item}", style=_item_style(item))


def _side_attr_map(side) -> "dict[str, tuple]":
    """species(lower) → (hp, status, item, moves) for a side's active + bench — the per-mon facts
    the SWITCHES table shows. Keyed lower-case (board species are id-form, matching switch labels)."""
    out = {side.active_species.lower(): (side.active_hp, side.status, side.item, tuple(side.moves))}
    for m in side.bench:
        out[m.species.lower()] = (m.hp, m.status, m.item, tuple(m.moves))
    return out


_MON_COLOR = "deep_sky_blue1"   # mon names pop in blue


def _col(cell, width: int) -> Text:
    """A `Text` padded to `width` visual cells (left-aligned) for manual column layout."""
    t = cell if isinstance(cell, Text) else Text(str(cell))
    gap = width - t.cell_len
    if gap > 0:
        t.append(" " * gap)
    return t


def _mon_label(species: str, item: str = "", *, chosen: bool = False, disabled: bool = False) -> Text:
    """'▶ donphan (leftovers)' — name blue (grey if disabled / bold if chosen), item inline in
    dim lowercase parens."""
    t = Text("▶ " if chosen else "  ")
    t.append(species, style=(_DISABLED_GREY if disabled
                             else f"bold {_MON_COLOR}" if chosen else _MON_COLOR))
    if item and item.lower() != "none":
        t.append(f" ({item.lower()})", style="dim")
    return t


def _moves_line(moves) -> "Text | None":
    """The full-width moveset sub-line under a mon — '⮡ m1 · m2 · …', spanning all columns."""
    return Text("     ⮡ " + " · ".join(moves), style=_DISABLED_GREY) if moves else None


_PANEL_NAME_W, _PANEL_HP_W, _PANEL_STAT_W = 30, 14, 10   # shared column widths for the mon panels


def _team_panel_text(side, *, name_w: int = _PANEL_NAME_W, hp_w: int = _PANEL_HP_W) -> Text:
    """A whole team panel (active ▶ then revealed bench) as `pokémon (item) | hp | status`, with a
    full-width moveset sub-line under each mon. Used by the Summary OPP TEAM + both Team-tab tables."""
    out = Text()
    out.append(f"{'pokémon':<{name_w}}{'hp':<{hp_w}}status", style="bold")
    if side is None:
        out.append("\n—", style="dim")
        return out
    mons = [(side.active_species, side.active_hp, side.status, side.item, side.moves, False, True)]
    mons += [(m.species, m.hp, m.status, m.item, m.moves, m.fainted, False) for m in side.bench]
    for sp, hp, status, item, moves, fainted, active in mons:
        row = Text()
        row.append_text(_col(_mon_label(sp, item, chosen=active, disabled=fainted), name_w))
        row.append_text(_col(_hp_bar(hp, disabled=fainted), hp_w))
        row.append_text(_status_cell(status))
        out.append("\n")
        out.append_text(row)
        ml = _moves_line(moves)
        if ml:
            out.append("\n")
            out.append_text(ml)
    return out


def _append_belief(out: Text, belief) -> None:
    """Append the model's species guess for each still-HIDDEN opp slot below the revealed OPP TEAM —
    the believed complement of the revealed mons (Gen3 has no team preview, so the rest of the
    opponent's team is unseen). Each hidden slot is one line of `species NN%` guesses (blue species +
    confidence-graded prob, most-likely first). A no-op unless the hidden-opponent belief was enabled
    for the run (the trace carries a `belief` block) — so off-runs show only the revealed mons."""
    if belief is None or not belief.slots:
        return
    out.append("\n\nbelieved hidden", style="bold")
    out.append("  (model's guess)", style="dim")
    for i, slot in enumerate(belief.slots, 1):
        out.append(f"\n{i:>2}  ", style="dim")
        for j, (species, prob) in enumerate(slot.top):
            if j:
                out.append(" · ", style="dim")
            out.append(species, style=_MON_COLOR)
            out.append(f" {prob * 100:.0f}%", style=gradient_color(prob))


def _append_belief_truth(out: Text, view) -> None:
    """Append the PRIVILEGED belief-vs-truth (from the reconstruction.json referee record + a belief-on
    checkpoint): the opponent's FULL team — revealed mons listed, then each STILL-HIDDEN mon with the
    model's species guess for it, the believed slot Hungarian-matched to the true mon (the same
    matching training uses). A marker scores each: `✓` top-1 right · `≈` the true mon IS in the belief
    but not top-1 (a near-miss) · `✗` not in the top-k at all; the true species is highlighted within
    the guess list, with its rank `(#k)` when it wasn't the top pick."""
    if view is None or not view.mons:
        return
    seen = [m.species for m in view.mons if m.revealed]
    # Best match first: ✓ top-1 (rank 1) → ≈ by ascending rank → ✗ not-in-belief (rank -1 → last).
    hidden = sorted((m for m in view.mons if not m.revealed),
                    key=lambda m: m.true_rank if m.true_rank > 0 else float("inf"))
    out.append("\n\ntruth + belief", style="bold")
    out.append(f"  ({view.n_correct}/{view.n_hidden} top-1)", style="dim")
    if seen:
        out.append("\nseen   ", style="dim")
        for j, sp in enumerate(seen):
            if j:
                out.append(" · ", style="dim")
            out.append(sp, style=_MON_COLOR)
    for m in hidden:
        out.append("\n")
        # ✓ top-1 right · ≈ true mon IS in the belief but not top-1 (a near-miss) · ✗ not in top-k.
        if m.guessed_right:
            marker, mstyle = "✓ ", "green"
        elif m.true_rank > 0:
            marker, mstyle = "≈ ", "yellow"
        else:
            marker, mstyle = "✗ ", "red"
        out.append(marker, style=mstyle)
        out.append_text(_col(Text(m.species, style="bold " + _MON_COLOR), 14))
        if not m.guess:
            out.append("(no guess)", style="dim")
            continue
        for j, (sp, prob) in enumerate(m.guess):
            if j:
                out.append(" · ", style="dim")
            hit = sp == m.species
            out.append(sp, style=("bold green" if hit else _MON_COLOR))
            out.append(f" {prob * 100:.0f}%", style=gradient_color(prob))
        if not m.guessed_right and m.true_rank > 0:
            out.append(f"   (#{m.true_rank})", style="dim")


# Plain-language for a "couldn't move" (cant) reason decoded from the TurnDelta.
_CANT_PHRASE = {"slp": "asleep", "frz": "frozen", "par": "fully paralyzed", "flinch": "flinched",
                "recharge": "recharging", "nopp": "no PP", "truant": "loafing", "attract": "immobilized",
                "taunt": "taunted", "disable": "disabled", "flinched": "flinched"}


def _cant_phrase(cant: str) -> str:
    return _CANT_PHRASE.get(str(cant).lower(), str(cant))


# Per-line tint for the raw protocol log (first matching prefix wins; default dim).
_PROTO_STYLE = (
    ("|faint|", "bold red"), ("|-crit|", "bold yellow"), ("|-miss|", "yellow"),
    ("|-immune|", "yellow"), ("|-supereffective|", "bold green"), ("|-resisted|", "dim"),
    ("|-damage|", "red"), ("|-heal|", "green"), ("|move|", "bold"), ("|cant|", "yellow"),
    ("|switch|", "cyan"), ("|drag|", "cyan"), ("|-status|", "magenta"),
    ("|-boost|", "magenta"), ("|-unboost|", "magenta"), ("|turn|", "bold dim"),
)


def _protocol_text(lines: "tuple[str, ...]", turn: int) -> Text:
    """Render a turn's raw Showdown protocol lines, lightly tinted by event kind."""
    out = Text()
    out.append(f"raw log · turn {turn}\n", style="dim italic")
    if not lines:
        out.append("  (no replay.html log for this trace)", style="dim")
        return out
    for ln in lines:
        style = next((st for pre, st in _PROTO_STYLE if ln.startswith(pre)), "dim")
        out.append("  " + ln + "\n", style=style)
    return out


_SIDE_STYLE = {"we": "bold green", "opp": "bold red"}
# Why a move did nothing visible (engine `no_effect`) — so a blank line never reads as missing data.
_NO_EFFECT_TEXT = {"immune": "no effect (immune)", "missed": "missed", "failed": "no effect"}


def _append_timeline_entry(line: Text, e: dict) -> None:
    """Render one RESULT timeline entry (engine ``build_result_timeline``) as a battle-log line:
    ``we thunderbolt did 31% (suicune 31% → faint)`` · ``opp rockslide did 73% (salamence 100% →
    27%)`` · ``we switch tyranitar → skarmory`` · ``opp sends in metagross`` · a couldn't-move note."""
    side = e.get("side", "")
    line.append(f"{side} ", style=_SIDE_STYLE.get(side, "dim"))
    kind = e.get("kind")
    if kind == "switch":
        line.append("switch ", style="dim")
        line.append(str(e.get("actor", "")), style=_MON_COLOR)
        line.append(" → ", style="dim")
        line.append(str(e.get("switch_to", "")), style="bold")
        return
    if kind == "send_in":
        line.append("send in " if side == "we" else "sends in ", style="dim")
        line.append(str(e.get("sent_in", "")), style="bold")
        return
    if kind == "faint":
        line.append(str(e.get("actor", "")), style=_MON_COLOR)
        line.append(" fainted", style="bold red")
        return
    # kind == "move"
    line.append(str(e.get("move") or "?"), style="bold")
    if e.get("cant"):
        line.append(f" — couldn't move ({_cant_phrase(e['cant'])})", style="bold yellow")
    elif e.get("damage"):
        line.append(f" did {e['damage']}", style="bold")
        line.append("  (", style="dim")
        line.append(str(e.get("target", "")), style=_MON_COLOR)
        line.append(f" {e.get('hp_before', '')} → ", style="dim")
        aft = str(e.get("hp_after", ""))
        line.append("faint" if aft == "faint" else aft, style=("bold red" if aft == "faint" else "dim"))
        line.append(")", style="dim")
    elif e.get("resulting"):                     # hit a switch-IN; only the resulting HP is known
        line.append(" → ", style="dim")
        line.append(str(e.get("target", "")), style=_MON_COLOR)
        line.append(f" (now {e.get('hp_after', '')})", style="dim")
    elif e.get("status"):
        line.append(" → ", style="dim")
        line.append(str(e.get("target", "")), style=_MON_COLOR)
        line.append(f" {e['status']}", style="bold yellow")
    elif e.get("no_effect"):                     # nothing happened — say why (missed / immune / failed)
        line.append(f" — {_NO_EFFECT_TEXT.get(e['no_effect'], 'no effect')}", style="yellow")
    if e.get("boost"):
        line.append(f"  ·  {e['boost']}", style="magenta")
    if e.get("crit"):
        line.append("  ⚡CRIT", style="bold yellow")


def _append_happened(line: Text, a: InvocationAnalysis, label: str) -> None:
    """Append what ACTUALLY happened after this decision as an ORDERED, one-line-per-action battle
    log — each mon's HP loss attributed to the OPPONENT's move that dealt it (``opp rockslide did 73%
    (salamence 100% → 27%)``), in execution order so move-order reads top-to-bottom (no '«1st»' tag).
    ``label`` carries its own leading newline + the panel name; continuations align under it. No-op
    when the outcome isn't recorded yet."""
    out = a.outcome or {}
    timeline = out.get("timeline")
    if timeline is None:   # defensive: an analysis built before the engine attached it
        timeline = build_result_timeline(
            out, a.our_species, a.opp_species, a.phase,
            our_hp_after=(a.next_board.ours.active_hp if a.next_board else None),
            opp_hp_after=(a.next_board.opp.active_hp if a.next_board else None),
        )
    if not timeline:
        return
    line.append(label, style="dim")
    indent = "\n" + " " * len(label.lstrip("\n"))
    # When both sides moved but move_order wasn't recorded (a no-state / model-free decision), the
    # engine flags the move entries order_certain=False: render an UNORDERED set (neutral bullets + a
    # note) rather than implying a top-to-bottom sequence we can't actually ground in the log.
    uncertain = any(e.get("kind") == "move" and not e.get("order_certain", True) for e in timeline)
    for i, e in enumerate(timeline):
        if i:
            line.append(indent)
        if uncertain:
            line.append("· ", style="dim")
        _append_timeline_entry(line, e)
    if uncertain:
        line.append(indent)
        line.append("(move order not recorded)", style="dim italic")


def _hp_frac(hp: str) -> "float | None":
    s = str(hp).strip()
    if "faint" in s.lower():
        return 0.0
    if s.endswith("%"):
        try:
            return max(0.0, min(1.0, float(s[:-1]) / 100.0))
        except ValueError:
            return None
    return None


def _hp_text(hp: str) -> Text:
    f = _hp_frac(hp)
    return Text(str(hp), style=gradient_color(f) if f is not None else "dim")


def _append_active(line: Text, prefix: str, side) -> None:
    line.append(prefix, style="bold")
    line.append(f"{side.active_species} ", style="bold")
    line.append(_hp_text(side.active_hp))
    if side.status:
        line.append(f" {side.status}", style="yellow")
    if side.boosts:
        line.append(f" [{side.boosts}]", style="magenta")


def _screens_str(field: dict, side: str) -> str:
    names = [("reflect", "Reflect"), ("light_screen", "LightScreen"),
             ("safeguard", "Safeguard"), ("mist", "Mist")]
    on = [label for key, label in names if field.get(f"{side}_{key}")]
    return ", ".join(on) if on else "—"


def _field_text(field: "dict | None") -> Text:
    """One-line field summary: weather, hazards (spikes), screens, turn."""
    if not field:
        return Text("field: —", style="dim")
    t = Text()
    weather = field.get("weather", "NONE")
    t.append("weather: ", style="dim")
    if weather and weather != "NONE":
        turns = field.get("weather_turns_left")
        suffix = " (∞)" if field.get("weather_permanent") else (f" ({turns:g} left)" if turns else "")
        t.append(f"{weather}{suffix}", style="cyan")
    else:
        t.append("none", style="dim")
    t.append("   spikes: ", style="dim")
    t.append(f"our {field.get('our_spikes', 0)} / opp {field.get('opp_spikes', 0)}")
    t.append("   screens: ", style="dim")
    t.append(f"our {_screens_str(field, 'our')} | opp {_screens_str(field, 'opp')}")
    # gen3_wish_wired_v1: a pending Wish heals the slot mon ~50% at the END of this turn (floating heal).
    wish_sides = [s for s, k in (("our", "wish_our"), ("opp", "wish_opp")) if field.get(k)]
    if wish_sides:
        t.append("   💧wish: ", style="dim")
        t.append(" & ".join(wish_sides) + " (~50% end-of-turn)", style="green")
    if field.get("turn") is not None:
        t.append(f"   turn {field['turn']:g}", style="dim")
    return t
