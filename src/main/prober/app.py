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
_SECTIONS = [
    ("sec-summary", "Summary", "8"),
    ("sec-review", "Review", "7"),
    ("sec-board", "Board", "1"),
    ("sec-faith", "Faithfulness", "2"),
    ("sec-matchups", "Matchups", "3"),
    ("sec-sweep", "Intervention", "4"),
    ("sec-saliency", "Saliency", "5"),
    ("sec-outcome", "Outcome", "6"),
]
_OPEN_BY_DEFAULT = {"sec-summary", "sec-board", "sec-faith", "sec-outcome"}


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
from main.prober.engine import InvocationAnalysis, analyze_invocation, summary_flags
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
        Binding("1", "toggle_section('sec-board')", "Board", show=False),
        Binding("2", "toggle_section('sec-faith')", "Faith", show=False),
        Binding("3", "toggle_section('sec-matchups')", "Matchups", show=False),
        Binding("4", "toggle_section('sec-sweep')", "Interv", show=False),
        Binding("5", "toggle_section('sec-saliency')", "Saliency", show=False),
        Binding("6", "toggle_section('sec-outcome')", "Outcome", show=False),
        Binding("7", "toggle_section('sec-review')", "Review", show=False),
        Binding("8", "toggle_section('sec-summary')", "Summary", show=False),
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
        self._current_battle: "BattleTrace | None" = None
        self._current_summary: "dict | None" = None
        self._analyze_token = 0
        self._pending_inv: "int | None" = None
        self._flagged: "list[int]" = []          # invocation indices with an auto summary-flag
        self._battle_filter = "all"              # cycled by `f`: all/loss/win
        self._review_store: "ReviewStore | None" = None   # manual flags/notes (per run dir)
        self._current_inv: "int | None" = None   # the highlighted invocation index

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
                # once; toggle by clicking a title or pressing its number key (1–6).
                with VerticalScroll(id="analysis-scroll"):
                    # Decision dashboard: the one-glance "funky turn" view — what it
                    # chose, the move/switch probabilities WITH their effectiveness +
                    # incoming KO-risk, and the threat/critic context, all grouped so a
                    # turn can be judged without scanning Board+Faith+Matchups+Outcome.
                    with Collapsible(title="Summary", collapsed=False, id="sec-summary"):
                        yield Static("", id="summary-head")
                        with Horizontal(id="summary-tables"):
                            with Vertical(classes="summary-col"):
                                yield Static("MOVES", classes="summary-col-label")
                                yield DataTable(id="summary-moves")
                            with Vertical(classes="summary-col"):
                                yield Static("SWITCHES", classes="summary-col-label")
                                yield DataTable(id="summary-switches")
                    # Manual-review card: what the model EXPECTED vs what HAPPENED, plus the
                    # human's funky-flag + note (space=flag, e=note, [ ]=jump, E=export).
                    with Collapsible(title="Review", collapsed=False, id="sec-review"):
                        yield Static("", id="review-card")
                        yield Static("", id="review-status")
                        yield Input(placeholder="note — Enter to save (e to focus)…",
                                    id="review-note")
                    with Collapsible(title="Board", collapsed=False, id="sec-board"):
                        yield Static("", id="board-summary")
                        yield Static("", id="board-field")
                        yield Static("our team", classes="board-label")
                        yield DataTable(id="board-our")
                        yield Static("opp team (revealed)", classes="board-label")
                        yield DataTable(id="board-opp")
                    with Collapsible(title="Faithfulness", collapsed=False, id="sec-faith"):
                        yield DataTable(id="faith-table")
                    with Collapsible(title="Matchups", collapsed=True, id="sec-matchups"):
                        yield DataTable(id="matchups-table")
                        yield Static("", id="matchups-threat")
                    with Collapsible(title="Intervention", collapsed=True, id="sec-sweep"):
                        yield DataTable(id="sweep-table")
                    with Collapsible(title="Saliency", collapsed=True, id="sec-saliency"):
                        yield DataTable(id="saliency-table")
                    with Collapsible(title="Outcome", collapsed=False, id="sec-outcome"):
                        yield Static("", id="outcome-summary")
                        yield DataTable(id="reward-table")
                        yield Static("", id="outcome-events")

    def on_mount(self) -> None:
        self.query_one("#summary-moves", DataTable).add_columns("move", "eff", "prob")
        self.query_one("#summary-switches", DataTable).add_columns(
            "target", "prob", "hp", "status", "item", "risk-in")
        self.query_one("#faith-table", DataTable).add_columns("action", "valid", "recorded", "re-run")
        self.query_one("#matchups-table", DataTable).add_columns("move", "×mult")
        self.query_one("#sweep-table", DataTable).add_columns("×mult", "P(chosen)", "P(switches)")
        self.query_one("#reward-table", DataTable).add_columns("reward component", "value")
        self.query_one("#saliency-table", DataTable).add_columns("obs block", "|grad|/dim", "sum")
        for tid in ("#board-our", "#board-opp"):
            self.query_one(tid, DataTable).add_columns("pokémon", "hp", "status")

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

    def _select_battle(self, battle: BattleTrace) -> None:
        self._current_battle = battle
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
        if self._current_battle is not None:
            meta = a.meta
            self.query_one("#battle-header", Static).update(
                f"{a.our_species} vs {a.opp_species} · {meta.result} · "
                f"turn {a.turn} · inv {a.inv_index}/{meta.n_invocations}"
            )
        self._render_summary(a)
        self._render_review(a)
        self._render_board(a)
        self._render_faithfulness(a)
        self._render_matchups(a)
        self._render_sweep(a)
        self._render_saliency(a)
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
        # Line 1 — matchup: each active's species + status/volatiles ("TOX(5)|SUB") + held item
        # (the opp's once revealed — decoded from the obs) + outcome.
        bd = a.board
        _append_summary_active(head, a.our_species,
                               bd.ours.status if bd else "", bd.ours.item if bd else "")
        head.append(" vs ", style="dim")
        _append_summary_active(head, a.opp_species,
                               bd.opp.status if bd else "", bd.opp.item if bd else "")
        head.append(f"   ·   turn {a.turn}", style="dim")
        result = (a.meta.result if a.meta is not None else None) or "?"
        head.append("   ·   ", style="dim")
        head.append(str(result).upper(),
                    style={"win": "bold green", "loss": "bold red"}.get(str(result).lower(), "dim"))
        # Line 2 — field: weather / hazards / screens / turn (the highlighted Board line).
        head.append("\nFIELD   ", style="dim")
        head.append(_field_text(a.field))
        # Line 3 — what it chose + confidence (+ a disagree flag if the model now prefers else).
        chosen_p = _chosen_prob(a)
        head.append("\nCHOSE   ", style="dim")
        head.append("▶ " + (a.chosen or "?"), style="bold")
        if chosen_p is not None:
            head.append(f"  {chosen_p * 100:.1f}%", style=gradient_color(chosen_p))
        if a.rerun_argmax is not None and not a.agrees:
            head.append("   ⚠ now prefers ", style="yellow")
            head.append(str(a.rerun_argmax), style="bold yellow")
        # Line 3 — critic context: WHY this turn is worth a look (ΔV / TD-surprise spikes).
        if a.value is not None:
            head.append("\nCRITIC  ", style="dim")
            head.append(f"V {a.value.recorded:+.2f}", style="bold")
            if a.value.delta is not None:
                head.append("   ΔV ", style="dim")
                head.append(f"{a.value.delta:+.2f}", style=("green" if a.value.delta >= 0 else "red"))
                td = self._td_residual(a)
                if td is not None:
                    head.append("   surprise(TDδ) ", style="dim")
                    head.append(f"{td:+.2f}", style=("green" if td >= 0 else "red"))
        # Line 4 — the danger it faced: incoming KO belief + speed (the switch-or-not signal).
        inc = a.incoming
        if inc is not None and inc.active_pko is not None:
            head.append("\nTHREAT  ", style="dim")
            head.append(f"incoming P(KO) {inc.active_pko * 100:.0f}%",
                        style=gradient_color(1.0 - inc.active_pko))
            if inc.active_outspeed is not None:
                head.append(f"   ·   we outspeed {inc.active_outspeed * 100:.0f}%", style="dim")
            head.append(f"   ·   worst-on-team {inc.max_pko * 100:.0f}%", style="dim")
            if inc.recovery_known or inc.recovery_rate > 0:
                head.append(f"   ·   opp recovery {inc.recovery_rate * 100:.0f}%"
                            + ("✓" if inc.recovery_known else "?"), style="dim")
        self.query_one("#summary-head", Static).update(head)

        # MOVES — fuse type-effectiveness (Matchups) with the policy prob (Faithfulness),
        # ranked by prob so the policy's preference order reads top-down.
        mt = self.query_one("#summary-moves", DataTable)
        mt.clear()
        mult_by_label = (dict(zip(a.matchups.move_labels, a.matchups.multipliers))
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
            prob_style = "bold" if r.is_chosen else gradient_color(r.recorded)
            mt.add_row(Text(label, style=lstyle), eff,
                       Text(f"{r.recorded * 100:5.1f}%", style=prob_style))

        # SWITCHES — prob next to the incoming KO-risk on the switch-in (per_slot_pko, the
        # single most switch-relevant fact). The i-th switch action == team slot i ==
        # per_slot_pko[i] (fixed obs action layout: 6 switches in team order, then moves),
        # so pair BEFORE sorting by prob.
        st = self.query_one("#summary-switches", DataTable)
        st.clear()
        per_slot = list(inc.per_slot_pko) if (inc is not None and inc.per_slot_pko) else []
        switch_rows = [r for r in (a.actions or []) if r.label.startswith("switch")]
        paired = [(r, per_slot[i] if i < len(per_slot) else None)
                  for i, r in enumerate(switch_rows)]
        # Each pivot's hp · status/volatiles · held item — "how healthy / how crippled / what
        # does it hold" next to its risk-in. Looked up by species from our board (active + bench).
        attrs = _side_attr_map(a.board.ours) if a.board is not None else {}
        if not switch_rows:
            st.add_row(Text("no switch available", style="dim"), "", "", "", "", "")
        for r, pko in sorted(paired, key=lambda rp: rp[0].recorded, reverse=True):
            target = r.label.split(":", 1)[-1]
            label = ("▶ " if r.is_chosen else "  ") + target
            lstyle = "bold" if r.is_chosen else ("" if r.valid else "dim")
            prob_style = "bold" if r.is_chosen else gradient_color(r.recorded)
            hp, status, item = attrs.get(target.lower(), (None, "", ""))
            hp_cell = _hp_text(hp) if hp is not None else Text("?", style="dim")
            if not r.valid:
                risk = Text("—", style="dim")          # fainted / the active mon: can't switch in
            elif pko is None:
                risk = Text("?", style="dim")
            else:
                risk = Text(f"{pko * 100:.0f}%", style=gradient_color(1.0 - pko))
            st.add_row(Text(label, style=lstyle),
                       Text(f"{r.recorded * 100:5.1f}%", style=prob_style),
                       hp_cell, _status_cell(status), _item_cell(item), risk)

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
            for label, mult in zip(a.matchups.move_labels, a.matchups.multipliers):
                bar = "█" * int(round(mult / 4.0 * 12))
                t.add_row(label, Text(f"{mult:4.2f}× {bar}", style=_mult_color(mult)))
        # Incoming threat (opp → us), decoded from their_matchups. The key tell is
        # `present`: it's blank for an opponent whose moves aren't revealed yet (a
        # just-switched-in mon), so the policy is pricing it from priors alone.
        threat = self.query_one("#matchups-threat", Static)
        th = a.threats
        lines = Text()
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
                lines.append(f"active {inc.active_pko * 100:.0f}%", style=gradient_color(inc.active_pko))
                lines.append(f"  ·  outspd {inc.active_outspeed * 100:.0f}%", style="dim")
                lines.append(f"  ·  worst-on-team {inc.max_pko * 100:.0f}%", style="dim")
            if inc.recovery_known or inc.recovery_rate > 0:
                lines.append(f"  ·  opp-recovery {inc.recovery_rate * 100:.0f}%"
                             + ("✓" if inc.recovery_known else "?"), style="dim")
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
            if v.rerun is not None:
                summary.append(f"  ·  re-run {v.rerun:+.2f}", style="dim")
            if v.delta is not None:
                d_style = "green" if v.delta >= 0 else "red"
                summary.append("  ·  ΔV ", style="dim")
                summary.append(f"{v.delta:+.2f}", style=d_style)
                summary.append(f" → {v.next_recorded:+.2f}", style="dim")
                # TD residual δ = r + γV(s') − V(s): how surprised the critic was. Parity with
                # the CLI's overview/analyze td_residual (the decisive metric in loss forensics).
                td = self._td_residual(a)
                if td is not None:
                    summary.append("  ·  TD δ ", style="dim")
                    summary.append(f"{td:+.2f}", style=("green" if td >= 0 else "red"))
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
                td = self._td_residual(a)
                if td is not None:
                    card.append("  surprise(TDδ) ", style="dim")
                    card.append(f"{td:+.2f}", style=("green" if td >= 0 else "red"))
        if a.rerun_argmax is not None and not a.agrees:
            card.append("\n⚠ model now prefers ", style="yellow")
            card.append(str(a.rerun_argmax), style="bold yellow")
        out = a.outcome or {}
        our, opp = out.get("our") or {}, out.get("opp") or {}
        if our or opp:
            card.append("\nhappened ", style="dim")
            card.append(f"we {our.get('action', '?')} ({our.get('hp_delta', '?')}) · "
                        f"opp {opp.get('action', '?')} ({opp.get('hp_delta', '?')})")
        if out.get("events"):
            card.append("  [" + ", ".join(map(str, out["events"])) + "]", style="yellow")
        self.query_one("#review-card", Static).update(card)

        # Store keys on the LIST POSITION (self._current_inv, set on highlight) so the glyph,
        # flag, and note all align — NOT a.inv_index (the trace's "i", which can differ).
        self._render_review_status()
        bid = self._battle_id()
        idx = self._current_inv
        note = (self._review_store.note(bid, idx)
                if (bid and self._review_store and idx is not None) else "")
        note_w = self.query_one("#review-note", Input)
        if note_w.value != note:
            note_w.value = note

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
        self._review_store.set(bid, self._current_inv, note=event.value.strip())
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


def _append_summary_active(line: Text, species: str, status: str, item: str) -> None:
    """Append 'species [status] @item' for one active mon to the Summary header line
    (status/volatiles and item shown only when present)."""
    line.append(species, style="bold")
    if status:
        line.append(f" [{status}]", style="yellow")
    if item and item.lower() != "none":
        line.append(f" @{item}", style=_item_style(item))


def _side_attr_map(side) -> "dict[str, tuple]":
    """species(lower) → (hp, status, item) for a side's active + bench — the per-mon facts the
    SWITCHES table shows. Keyed lower-case (board species are id-form, matching switch labels)."""
    out = {side.active_species.lower(): (side.active_hp, side.status, side.item)}
    for m in side.bench:
        out[m.species.lower()] = (m.hp, m.status, m.item)
    return out


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
    if field.get("turn") is not None:
        t.append(f"   turn {field['turn']:g}", style="dim")
    return t
