const D = JSON.parse(document.getElementById('payload').textContent);
const $ = s => document.querySelector(s);
const CLS_VAR = {absolute:'--absolute', ratio:'--ratio', none:'--none'};
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

if (typeof cytoscape === 'undefined') {
  /* Marked on the BODY, not just written into it: the failure text also appears verbatim in this
     script's own source, so anything reading the DOM cannot tell "the page failed" from "the page
     contains the code that would say so". The attribute is unambiguous. */
  document.body.dataset.cytoscape = 'missing';
  document.body.innerHTML = '<div id="err"><b>cytoscape failed to load.</b><br>' +
    'This page needs the one CDN &lt;script&gt; tag at the top (cytoscape). ' +
    'You are probably offline. The data itself is embedded and intact — open the ' +
    '<code>#payload</code> script block, or use <code>designs/architecture_graph.dot</code>.</div>';
  throw new Error('cytoscape missing');
}

/* ---------- measurement lookup ------------------------------------------------------------ */
let OV = D.overlays.length ? D.overlays[D.overlays.length - 1] : null;
let METRIC = 'flip_rate';
/* Map an edge to its measured value under the selected overlay.
   bias   -> the family arm (each family's map zeroed).
   concat -> the op-block table's FULL_CONCAT, shuffle arm (width-fair); falls back to the
             edge-audit 'concat' arm when this checkpoint only has that.
   cell   -> only measured jointly as 'concat_cells'; reported, never faked per-edge. */
function measure(e) {
  if (!OV) return null;
  if (e.type === 'bias' && e.family && OV.families[e.family])
    return {v: OV.families[e.family][METRIC], src: 'family ' + e.family + ' (map zeroed)',
            bad: (OV.contaminated || []).includes(e.family)};
  if (e.type === 'concat' && e.src === 'damage_op') {
    const b = OV.blocks && OV.blocks.FULL_CONCAT;
    if (b && b.shuffle_all && b.shuffle_all[METRIC] != null)
      return {v: b.shuffle_all[METRIC], src: 'op block FULL_CONCAT (shuffle arm)', bad: false};
    if (OV.arms.concat) return {v: OV.arms.concat[METRIC], src: 'concat arm (zeroed)', bad: false};
  }
  if (e.type === 'cell' && OV.arms.concat_cells)
    return {v: OV.arms.concat_cells[METRIC], src: 'concat_cells arm (joint, not per-edge)',
            bad: false};
  return null;
}
/* measured -> px. sqrt so the small families stay visible against a max ~400x larger, and a
   deliberately modest ceiling: the widest channel here (the op's per-action cell route) is real
   and IS the policy's biggest dependency, but at 10px it becomes an opaque band that hides the
   seats it crosses. 6.4px still reads as "much the fattest" without erasing its neighbours. */
function scaleW(v) {
  if (v == null) return 1.1;
  const all = D.edges.map(measure).filter(Boolean).map(m => m.v).filter(x => x != null);
  const mx = Math.max(...all, 1e-9);
  return 1.0 + 5.4 * Math.sqrt(Math.max(v, 0) / mx);
}

/* ---------- graph ------------------------------------------------------------------------- */
const els = [
  ...D.nodes.map(n => ({data: {...n, label: n.id}})),
  ...D.edges.map(e => ({data: {...e, source: e.src, target: e.dst}})),
];
/* The stylesheet is a FUNCTION of the current CSS variables, not a literal, because cytoscape
   resolves `cssv()` once at construction and then owns those values on its own canvas. Baked in,
   a theme applied after construction — which is what a `#dark=1` deep link does — leaves every
   node painted in the other theme's palette on the new background. Rebuilt, both entry points
   (toggle and deep link) go through the same path. */
const sheet = () => [
    {selector: 'node', style: {
      'shape': 'data(shape)', 'label': 'data(label)', 'font-size': 11,
      'text-valign': 'center', 'color': cssv('--text-primary'),
      'background-color': cssv('--surface-2'), 'border-width': 1.4,
      'border-color': cssv('--line'), 'width': 'label', 'height': 'label',
      'padding': '7px', 'text-wrap': 'wrap', 'text-max-width': '104px'}},
    {selector: 'node[kind="head"], node[kind="operator"]', style: {
      'border-width': 2.6, 'font-weight': 'bold', 'font-size': 12}},
    {selector: 'node[kind="logit"]', style: {'font-size': 10}},
    {selector: 'edge', style: {
      'curve-style': 'bezier', 'width': 1.1, 'opacity': .82,
      'target-arrow-shape': 'triangle', 'arrow-scale': .55,
      'line-color': cssv('--unmeasured'), 'target-arrow-color': cssv('--unmeasured')}},
    {selector: 'edge[cls="absolute"]', style: {'line-color': cssv('--absolute'),
      'target-arrow-color': cssv('--absolute')}},
    {selector: 'edge[cls="ratio"]', style: {'line-color': cssv('--ratio'),
      'target-arrow-color': cssv('--ratio')}},
    {selector: 'edge[cls="none"]', style: {'line-color': cssv('--none'),
      'target-arrow-color': cssv('--none')}},
    /* Opacity states are CLASSES, deliberately ordered: an element `.style()` bypass outranks
       every stylesheet rule, so anything that fades an edge has to live here or focus-dimming
       silently loses to it. Later rule wins on a tie, so `.dim` is last. */
    {selector: 'edge.contam', style: {'opacity': .34}},
    {selector: 'edge.faint', style: {'opacity': .2}},
    {selector: 'edge.faint.contam', style: {'opacity': .1}},
    {selector: '.dim', style: {'opacity': .05, 'text-opacity': .12}},
    {selector: '.hid', style: {'display': 'none'}},
    {selector: '.sel', style: {'border-width': 3, 'border-color': cssv('--text-primary')}},
];
const cy = cytoscape({
  container: $('#cy'), elements: els, wheelSensitivity: .2, layout: {name: 'preset'},
  style: sheet(),
});
function applyTheme() { cy.style().fromJson(sheet()).update(); dashes(); restyle(); }
function dashes() {
  D.edges.forEach(e => {                   // dash = sub-channel (secondary encoding)
    const d = D.edge_dash[e.type];
    if (d) cy.$id(e.id).style({'line-style': 'dashed', 'line-dash-pattern': d});
  });
}
dashes();

/* ---------- layout -------------------------------------------------------------------------
   Hand-placed columns, NOT a layered ranker. 388 of the 487 edges are attention-bias edges
   running seat->seat WITHIN one 36-token sequence: peer edges, not pipeline stages. Handing
   those to dagre asks it to put 36 mutual peers in a rank order, and it answers with a diagonal
   smear whose fit-zoom drives every label sub-pixel. The pipeline the model actually has is
   short and known, so the columns are assigned outright and the seats pack into three bands.

   Bands come from a RULE on `kind`, never from an id list: a node added to the snapshot lands
   somewhere deliberate, and anything the rule does not recognise goes to a trailing MISC band
   that the header calls out rather than being quietly dropped or piled on the origin. This also
   covers the seats that carry no edge at all — `history[0..6]` have NONE, and `global` only
   RECEIVES bias (28 in, 0 out) — which a ranker has no opinion about and would drop wherever. */
const SEAT_BANDS = [['our_mon', 'opp_mon'],
                    ['E3_move', 'E4_threat', 'E5_tail'],
                    ['global', 'history', 'event']];
const MISC_BAND = 9;
const seatGroup = id => id.replace(/\[\d+\]$/, '');
/* Bands read LEFT-TO-RIGHT in pipeline order. The two leading bands are the T0 front end: the obs
   SUBSTRATE (the schema's validated tiling) and then the phases that consume it. They were added
   when the early stage was expanded from 4 nodes to 20 — before that the picture began at the role
   tokens, which made the tier the contract now enforces invisible in the one diagram people read. */
function bandOf(n) {
  if (n.kind === 'obs_block' || n.kind === 'obs_group') return 0;
  if (n.kind === 'phase') return 1;
  if (n.kind === 'input' || n.kind === 'belief_head') return 2;
  if (n.kind === 'operator') return 3;
  if (n.kind === 'seat') {
    const i = SEAT_BANDS.findIndex(b => b.includes(seatGroup(n.id)));
    return i < 0 ? MISC_BAND : 4 + i;
  }
  if (n.kind === 'head') return 7;
  if (n.kind === 'logit' || n.kind === 'aux_loss') return 8;
  return MISC_BAND;
}
const BANDS = Array.from({length: MISC_BAND + 1}, () => []);
D.nodes.forEach(n => BANDS[bandOf(n)].push(n));
/* Reading order inside a band: sources before beliefs, seats in declared group order, forward
   sinks before the training-only losses. */
const ordKey = n => (n.kind === 'aux_loss' ? '9' : n.kind === 'belief_head' ? '5' : '1') +
  (n.kind === 'seat' ? String(SEAT_BANDS.flat().indexOf(seatGroup(n.id))).padStart(2, '0') : '') +
  n.id;
BANDS.forEach(b => b.sort((a, c) => ordKey(a) < ordKey(c) ? -1 : 1));
const NARROW = () => innerWidth < 760;
function layoutColumns() {
  /* The three seat bands are ONE logical column (the 36-token sequence) split only because 36
     rows would not fit; they get a tight gap so they read as one block, and the stage boundaries
     around them get a wide one. */
  const GAP_WIDE = 92, GAP_TIGHT = 34, ROW = 13, GROUP_GAP = 26;
  let x = 0, prevBand = -1;
  BANDS.forEach((band, bi) => {
    if (!band.length) return;
    if (prevBand >= 0) x += (prevBand >= 2 && prevBand <= 3 && bi <= 4) ? GAP_TIGHT : GAP_WIDE;
    prevBand = bi;
    const w = Math.max(...band.map(n => cy.$id(n.id).outerWidth()));
    let h = 0, prev = null;
    const ys = band.map(n => {                       // stack, with a gap where the group changes
      const g = n.kind === 'seat' ? seatGroup(n.id) : n.kind;
      if (prev !== null && g !== prev) h += GROUP_GAP;
      prev = g;
      const nh = cy.$id(n.id).outerHeight(), y = h + nh / 2;
      h += nh + ROW;
      return y;
    });
    const off = -(h - ROW) / 2;                      // every band centred on the same axis
    band.forEach((n, i) => cy.$id(n.id).position({x: x + w / 2, y: off + ys[i]}));
    x += w;
  });
}
function layoutStacked() {
  /* Phone shape: bands run TOP-TO-BOTTOM, so the flow still reads sources -> sinks — down the
     screen instead of across it — and each band WRAPS into three columns.

     Wrapping is the part that matters. A plain transpose sounds right and is worse: band 3 holds
     16 seats, and 16 abreast is wider than the landscape layout it was meant to replace. Wrapped,
     the drawing is roughly 360 x 840 instead of 1200 x 600, which on a 390px-wide screen takes
     the fit-zoom from ~0.33 to ~0.83 — the difference between 3.6px labels and 9px ones. */
  const COLS = 3, CELL_W = 126, ROW_GAP = 8, BAND_GAP = 26;
  let y = 0;
  BANDS.forEach(band => {
    if (!band.length) return;
    for (let i = 0; i < band.length; i += COLS) {
      const row = band.slice(i, i + COLS);
      const h = Math.max(...row.map(n => cy.$id(n.id).outerHeight()));
      const xoff = -(row.length - 1) * CELL_W / 2;     // centre short rows under full ones
      row.forEach((n, c) => cy.$id(n.id).position({x: xoff + c * CELL_W, y: y + h / 2}));
      y += h + ROW_GAP;
    }
    y += BAND_GAP;
  });
}
function layout() {
  if (NARROW()) layoutStacked(); else layoutColumns();
  cy.resize();                      // measure the container as it IS, not as it was at construction
  cy.fit(undefined, NARROW() ? 12 : 24);
}
layout();
/* One more fit after the first paint. On a phone the URL bar and the safe-area insets settle
   after layout, and a fit computed against the pre-settle height clips the drawing. */
addEventListener('load', () => { cy.resize(); cy.fit(undefined, NARROW() ? 12 : 24); });

/* Which bias family is on show: 'all' (every family, faint, so the skeleton reads through 388
   overlapping edges), 'none' (hidden), or one family id at full strength. The family IS the unit
   the measurement is defined on, so this is the control that turns the overlay from a wash of
   colour into a readable per-family answer. */
let BIASSEL = 'all';
function restyle() {
  cy.batch(() => {
    D.edges.forEach(e => {
      const m = measure(e), el = cy.$id(e.id);
      el.style('width', scaleW(m ? m.v : null));
      const c = m ? cssv(CLS_VAR[e.cls]) : cssv('--unmeasured');
      el.style('line-color', c); el.style('target-arrow-color', c);
      const solo = e.type === 'bias' && BIASSEL !== 'all' && BIASSEL !== 'none';
      el.toggleClass('contam', !!(m && m.bad));
      el.toggleClass('faint', e.type === 'bias' && BIASSEL === 'all');
      el.toggleClass('hid', e.type === 'bias' && (BIASSEL === 'none'
                                                  || (solo && e.family !== BIASSEL)));
    });
  });
}

/* ---------- focus / path filter ----------------------------------------------------------- */
let FOCUS = null, DIR = 'anc';
function applyFocus() {
  cy.elements().removeClass('dim sel');
  if (!FOCUS) return;
  const n = cy.$id(FOCUS); if (!n.length) return;
  let keep = n.union(DIR === 'anc' ? n.predecessors()
                   : DIR === 'desc' ? n.successors()
                   : n.predecessors().union(n.successors()));
  cy.elements().difference(keep).addClass('dim');
  n.addClass('sel');
}

/* ---------- side panel -------------------------------------------------------------------- */
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function row(k, v) { return v == null || v === '' ? '' : `<dt>${esc(k)}</dt><dd>${v}</dd>`; }
const fmt = v => METRIC === 'flip_rate' ? (v * 100).toFixed(2) + '%' : v.toFixed(5);

/* A bias family is never shown as a bare code. `d2` means nothing to a reader; "d2 — our bench's
   offense vs their ACTIVE" means something immediately, and the full cell definition is one hover
   away. Every place a family id appears goes through one of these. */
const FAM = D.families || {};
const famLabel = f => (FAM[f] && FAM[f].label) || '';
const famName = f => famLabel(f) ? `${f} — ${famLabel(f)}` : f;
const famCells = f => {
  const m = FAM[f];
  if (!m || !m.cells) return '';
  return `[${m.cells}]` + (m.per ? ` per (${m.per})` : '') + (m.width ? ` · ${m.width} dims` : '');
};
const famTitle = f => {
  const m = FAM[f];
  if (!m) return f;
  return [famName(f), famCells(f), m.note].filter(Boolean).join('\n');
};

/* Forward reachability, by channel class. Built once; 58 nodes / 487 edges makes this trivial.
   `aux` is excluded from every route set — a training-only edge reaching a forward sink is the
   leak the badge exists to rule out, so it must never count as a way to get somewhere. */
const ADJ = {};
D.edges.forEach(e => (ADJ[e.src] = ADJ[e.src] || []).push(e));
const CONTENT_ROUTES = new Set(['content', 'concat', 'cell']);
const ANY_FORWARD = new Set(['content', 'concat', 'cell', 'bias']);
function reach(id, types) {
  const seen = new Set(), stack = [id];
  while (stack.length) {
    for (const e of ADJ[stack.pop()] || []) {
      if (!types.has(e.type) || seen.has(e.dst)) continue;
      seen.add(e.dst); stack.push(e.dst);
    }
  }
  return seen;
}
/* Where this token's information can actually GO, split by channel class. A bias edge DOES move
   information — it changes the attention weights, so it changes the pooled token — it just
   cannot carry a magnitude, so collapsing the two into one "reachable" would lose the whole
   point of the diagram.

   The `no route` verdict needs its caveat stated IN THE PANEL, not just here. This graph models
   EXPLICIT channels; ordinary self-attention among the 36 seats is not an edge in it. So a seat
   with no modelled channel (`history[0..6]` have no edge at all; `global` only receives bias) is
   "not tracked by this graph", NOT "unused by the model" — and on an artifact whose job is to be
   read aloud in a design conversation, that difference is exactly the kind of thing that would
   otherwise get repeated back as a finding. */
function delivery(id) {
  const viaContent = reach(id, CONTENT_ROUTES), viaAny = reach(id, ANY_FORWARD);
  const hit = (s, k) => k.endsWith('.') ? [...s].some(x => x.startsWith(k)) : s.has(k);
  const out = [['pi_projection', 'pi_projection'], ['vf_projection', 'vf_projection'],
               ['pointer logits', 'pointer.']]
    .filter(([, k]) => k !== id)
    .map(([label, k]) => row('→ ' + label,
      hit(viaContent, k) ? '<b>content</b> — concat / cell / token'
      : hit(viaAny, k) ? 'attention <b>bias only</b> — steers, carries no magnitude'
      : '<span class="mut">no route</span>')).join('');
  return out ? `<div class="sec">what it can deliver to</div><dl>${out}</dl>` : '';
}
/* Said in the panel, because "no route" on its own reads as "this token does nothing". */
const NO_CHANNEL_NOTE = '<div class="note">This seat carries <b>no modelled channel</b>. It still '
  + 'sits in the 36-token sequence and takes part in ordinary self-attention — the delivery graph '
  + 'tracks explicit channels (content / concat / cell / bias / aux) and does not model plain '
  + 'attention as an edge. Read this as <b>“not tracked here”</b>, not “unused by the model”.</div>';
/* Every bias family acting on this token, ranked by measured dependence at the selected
   checkpoint. This is the per-token form of the overlay: "what conditions this seat, and how
   much does the policy actually move when you switch it off". */
function familiesOn(id) {
  const n = {};
  D.edges.forEach(e => {
    if (e.type === 'bias' && (e.src === id || e.dst === id)) n[e.family] = (n[e.family] || 0) + 1;
  });
  const keys = Object.keys(n);
  if (!keys.length) return '';
  const rows = keys.map(f => ({
    f, n: n[f],
    v: OV && OV.families[f] ? OV.families[f][METRIC] : null,
    bad: !!(OV && (OV.contaminated || []).includes(f)),
  })).sort((a, b) => (b.v == null ? -1 : b.v) - (a.v == null ? -1 : a.v));
  return `<div class="sec">bias families on this token (${keys.length})</div>
    <div class="elist">${rows.map(r => `<div title="${esc(famTitle(r.f))}">
      <b>${esc(r.f)}</b> <span class="mut">×${r.n}</span> —
      ${r.v == null ? '<i>unmeasured here</i>'
        : fmt(r.v) + (r.bad ? ' <span class="mut">⚠ contaminated</span>' : '')}
      <div class="mut">${esc(famLabel(r.f))}</div></div>`).join('')}
    </div>`;
}
/* On a phone the panel is a bottom sheet, so it needs an explicit open and an explicit way out
   — a fixed overlay with no dismiss is a trap. On desktop these are no-ops: the class does
   nothing outside the media query, and the panel is always visible. */
function openSheet() { document.body.classList.add('sheet'); closeCtls(); }
function closeSheet() { document.body.classList.remove('sheet'); }
/* The controls sheet. On desktop `#ctls` is `display:contents`, so none of this is visible there
   — the class toggles nothing outside the phone media query. */
function openCtls() { document.body.classList.add('ctls'); closeSheet(); }
function closeCtls() { document.body.classList.remove('ctls'); }

/* What the panel is currently showing, so a checkpoint or metric change re-renders it. Left
   static, the panel keeps displaying the numbers of a checkpoint you already navigated away
   from — stale figures beside a selector that says otherwise. */
let SEL = null;
function refreshPanel() {
  if (!SEL) return;
  (SEL.kind === 'node' ? showNode : showEdge)(SEL.id);
}
function showNode(id) {
  const n = D.nodes.find(x => x.id === id); if (!n) return;
  SEL = {kind: 'node', id}; openSheet();
  const ins = D.edges.filter(e => e.dst === id), outs = D.edges.filter(e => e.src === id);
  /* One row per edge, and NO cap: the list scrolls. The previous `.slice(0, 14)` printed a
     "×15" chip above 14 rows, which is the exact shape of a silent truncation. */
  const grp = es => D.class_order.map(c => {
    const k = es.filter(e => e.cls === c); if (!k.length) return '';
    return `<div class="ghdr"><span class="chip"><span class="sw"
      style="background:var(${CLS_VAR[c]})"></span>${c} ×${k.length}</span></div>` +
      k.map(e => `<div>${esc(e.type)} ${e.src === id ? '→' : '←'}
        <b>${esc(e.src === id ? e.dst : e.src)}</b></div>`).join(''); }).join('');
  $('#sidebody').innerHTML = `<h2>${esc(id)}</h2><div class="kind">${esc(n.kind)}</div><dl>
    ${row('token index', n.index)}${row('token type', n.token_type)}
    ${row('action index', n.action_index)}${row('out dim', n.out_dim)}
    ${row('in / out', n.in_features ? n.in_features + ' → ' + n.out_features : null)}
    ${row('incoming dim', n.incoming_dim)}${row('label key', n.label_key)}</dl>
    ${ins.length + outs.length ? delivery(id) + familiesOn(id)
        : '<div class="sec">what it can deliver to</div>' + NO_CHANNEL_NOTE}
    <div class="sec">incoming (${ins.length})</div><div class="elist">${grp(ins) || '—'}</div>
    <div class="sec">outgoing (${outs.length})</div><div class="elist">${grp(outs) || '—'}</div>
    ${n.doc_section ? `<div class="sec">reference</div>
      <div>designs/ARCHITECTURE.md § <b>${esc(n.doc_section)}</b></div>` : ''}
    <div class="sec">focus</div>
    <div><button onclick="setFocus('${id}','anc')">what feeds it</button>
         <button onclick="setFocus('${id}','desc')">what it feeds</button></div>`;
}
function showEdge(id) {
  const e = D.edges.find(x => x.id === id); if (!e) return;
  SEL = {kind: 'edge', id}; openSheet();
  const m = measure(e);
  $('#sidebody').innerHTML = `<h2>${esc(e.src)} → ${esc(e.dst)}</h2>
    <div class="kind">${esc(e.type)} · carries ${esc(e.carries)}</div>
    <dl>${row('channel', esc(e.blurb))}${row('width', e.width)}
    ${row('family', e.family ? `<b>${esc(e.family)}</b> — ${esc(famLabel(e.family))}` : null)}
    ${row('cells', e.family && famCells(e.family)
      ? `<code>${esc(famCells(e.family))}</code>` : null)}
    ${row('family note', e.family && FAM[e.family] && FAM[e.family].note
      ? esc(FAM[e.family].note) : null)}
    ${row('via', e.via ? esc(e.via) : null)}
    ${row('source constant', e.source_constant ? '<code>' + esc(e.source_constant) + '</code>' : null)}
    ${row('bidirectional', e.bidirectional ? 'yes' : null)}
    ${row('zero-init', e.zero_init ? 'yes — identity at init' : null)}
    ${row('note', e.note ? esc(e.note) : null)}</dl>
    ${m ? `<div class="sec">measured</div><dl>
       ${row(METRIC, fmt(m.v))}
       ${row('arm', esc(m.src))}${row('checkpoint', esc(OV.id))}</dl>
       ${m.bad ? `<div class="warn"><b>contaminated.</b> ${esc(OV.contamination_note)}</div>` : ''}`
     : '<div class="sec">measured</div><div class="kind">no measurement for this edge at the ' +
       'selected checkpoint — rendered in the neutral “unmeasured” style, never as zero.</div>'}
    <div class="sec">concept</div><div>${esc(e.concept_file)}<br>§ <b>${esc(e.concept_heading)}</b></div>`;
}

/* ---------- legend, table, controls -------------------------------------------------------- */
function legend() {
  const line = (c, dash) => `<svg width="34" height="8"><line x1="1" y1="4" x2="33" y2="4"
    stroke="var(${CLS_VAR[c]})" stroke-width="2.4" ${dash ? `stroke-dasharray="${dash.join(' ')}"` : ''}/></svg>`;
  $('#legend').innerHTML = `<b>Edge colour = what the channel physically carries</b>
    <div class="row">${line('absolute', null)}<span><b>absolute</b> — ${esc(D.class_carries.absolute)}</span></div>
    <div class="row" style="margin-left:16px">${line('absolute', D.edge_dash.concat)}<span>concat (head input) · ${line('absolute', D.edge_dash.cell)} cell (per-action) · solid = token content</span></div>
    <div class="row">${line('ratio', D.edge_dash.bias)}<span><b>ratio</b> — ${esc(D.class_carries.ratio)}</span></div>
    <div class="row">${line('none', D.edge_dash.aux)}<span><b>training-only</b> — ${esc(D.class_carries.none)}</span></div>
    <div class="row"><svg width="34" height="8"><line x1="1" y1="4" x2="33" y2="4"
      stroke="var(--unmeasured)" stroke-width="2.4"/></svg><span>no measurement at this checkpoint</span></div>
    <div class="row"><span class="sub">thickness = <span id="lgm"></span>; node shape = kind</span></div>
    ${BIASSEL !== 'all' && BIASSEL !== 'none' ? `<div class="row famdef"><span>
      showing <b>${esc(BIASSEL)}</b> — ${esc(famLabel(BIASSEL))}<br>
      <span class="mut">${esc(famCells(BIASSEL))}</span></span></div>` : ''}`;
  $('#lgm').textContent = $('#mt').selectedOptions[0].textContent;
}
function table() {
  const rows = D.edges.map(e => ({e, m: measure(e)}))
    .sort((a, b) => ((b.m && b.m.v) || -1) - ((a.m && a.m.v) || -1));
  $('#tbl').innerHTML = `<table><thead><tr><th>from</th><th>to</th><th>channel</th>
    <th>carries</th><th>width</th><th>family</th><th>${esc(METRIC)}</th><th>arm</th></tr></thead>
    <tbody>${rows.map(({e, m}) => `<tr><td>${esc(e.src)}</td><td>${esc(e.dst)}</td>
    <td>${esc(e.type)}</td><td>${esc(e.cls)}</td><td>${e.width ?? ''}</td>
    <td title="${esc(e.family ? famTitle(e.family) : '')}">${e.family
      ? esc(e.family) + ' <span class="mut">' + esc(famLabel(e.family)) + '</span>' : ''}</td>
    <td>${m ? fmt(m.v) : '—'}</td>
    <td>${m ? esc(m.src) + (m.bad ? ' ⚠ contaminated' : '') : ''}</td></tr>`).join('')}</tbody></table>`;
}
function header() {
  const m = D.meta;
  $('#hdr').textContent = `${m.arch_signature} · cfg v${m.config_version} · obs ${m.obs_dim} · ` +
    `${m.n_tokens} tokens · op ${m.op_out_dim} · ${D.nodes.length} nodes / ${D.edges.length} edges` +
    (BANDS[MISC_BAND].length ? ` · ⚠ ${BANDS[MISC_BAND].length} unplaced (` +
      BANDS[MISC_BAND].map(n => n.kind).join(', ') + ') — extend bandOf()' : '');
  /* Provenance, so staleness is something you SEE rather than something you have to reason
     about. The hash identifies the architecture (same hash = same graph as last time). The age
     only appears when served, because a static file has no way to know it — and it is the
     snapshot's age, NOT the page's: the page rebuilds per request, the graph underneath it does
     not. 14 days is a nudge, not a rule. */
  const prov = D.snapshot_built;
  $('#prov').innerHTML = `graph <code>${esc(D.graph_sha256 || '?')}</code>` + (prov
    ? ` · snapshot <span class="${prov.days >= 14 ? 'aged' : ''}">${prov.days}d old</span>` : '');
  $('#prov').title = 'graph_sha256 identifies the architecture itself.\n' + (prov
    ? `The delivery-graph snapshot was last rebuilt ${prov.iso} (${prov.source}).\n` +
      'The page re-renders on every request; the snapshot under it only changes when someone ' +
      'regenerates it after an architecture change.'
    : 'Snapshot age is only shown when served — a file:// copy cannot know it.');
  const L = D.leak_safety;
  $('#leak').innerHTML = `<span class="badge ${L.ok ? 'ok' : 'no'}">leak-safety ` +
    `${L.ok ? 'PASS' : 'FAIL (' + L.violations.length + ')'}</span>`;
  $('#leak').title = 'computed from the embedded data: no aux edge may terminate at ' +
    'pi_projection, vf_projection, or a pointer logit';
  /* Same three facts, re-rendered for the phone: the bar hides them at 390px, and the leak badge
     and snapshot age are exactly what you want when reading this away from the machine. */
  $('#ctlmeta').innerHTML = $('#leak').innerHTML + '<br>' + esc($('#hdr').textContent) +
    '<br>' + $('#prov').innerHTML;
}
function setFocus(id, dir) { FOCUS = id; DIR = dir || DIR; $('#fo').value = id || '';
  $('#dir').value = DIR; applyFocus(); hash(); }
function hash() {
  const p = new URLSearchParams();
  if (FOCUS) { p.set('focus', FOCUS); p.set('dir', DIR); }
  p.set('ck', OV ? OV.id : ''); p.set('m', METRIC); p.set('bias', BIASSEL);
  p.set('theme', document.documentElement.dataset.theme);
  location.hash = p.toString();
}
function fromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  /* The theme is pinned EXPLICITLY, not as a "dark=1" flag on top of a default. Dark is the
     default here, and a flag-over-default link silently changes meaning the day the default
     moves. */
  const th = p.get('theme') === 'light' ? 'light' : p.get('theme') === 'dark' ? 'dark' : null;
  if (th) { document.documentElement.dataset.theme = th;
            $('#dk').setAttribute('aria-pressed', String(th === 'dark')); }
  if (p.get('m')) METRIC = p.get('m');
  if (p.get('bias')) BIASSEL = p.get('bias');
  const ck = p.get('ck'); if (ck) { const o = D.overlays.find(x => x.id === ck); if (o) OV = o; }
  FOCUS = p.get('focus') || null; DIR = p.get('dir') || 'anc';
}

$('#ck').innerHTML = D.overlays.map(o => `<option value="${o.id}">${o.generation || o.id} @ ` +
  `${o.step ? (o.step / 1e6).toFixed(1) + 'M' : '?'} · n=${o.n_states || '?'} · ` +
  `${o.date || ''}</option>`).join('');
const FAMS = [...new Set(D.edges.filter(e => e.type === 'bias').map(e => e.family))].sort();
const famCount = f => D.edges.filter(e => e.type === 'bias' && e.family === f).length;
$('#bf').innerHTML = `<option value="all">all ${FAMS.length} families (faint)</option>` +
  FAMS.map(f => `<option value="${f}">${esc(famName(f))} (${famCount(f)})</option>`).join('') +
  '<option value="none">none (skeleton only)</option>';
$('#fo').innerHTML = '<option value="">— none —</option>' +
  D.nodes.map(n => `<option value="${n.id}">${n.id}</option>`).join('');
fromHash();
$('#ck').value = OV ? OV.id : ''; $('#mt').value = METRIC; $('#fo').value = FOCUS || '';
$('#dir').value = DIR; $('#bf').value = BIASSEL;
$('#bf').onchange = e => { BIASSEL = e.target.value; restyle(); legend(); hash(); };
$('#ck').onchange = e => { OV = D.overlays.find(o => o.id === e.target.value); restyle(); legend();
  table(); refreshPanel(); hash(); };
$('#mt').onchange = e => { METRIC = e.target.value; restyle(); legend(); table(); refreshPanel();
  hash(); };
$('#fo').onchange = e => setFocus(e.target.value || null);
$('#dir').onchange = e => { DIR = e.target.value; applyFocus(); hash(); };
$('#clr').onclick = () => setFocus(null);
$('#tg').onclick = e => { const on = e.target.getAttribute('aria-pressed') !== 'true';
  e.target.setAttribute('aria-pressed', on); $('#tbl').style.display = on ? 'block' : 'none';
  $('#cy').style.display = on ? 'none' : 'block'; $('#legend').style.display = on ? 'none' : 'block';
  if (on) table(); };
$('#dk').onclick = e => { const on = e.target.getAttribute('aria-pressed') !== 'true';
  e.target.setAttribute('aria-pressed', on);
  document.documentElement.dataset.theme = on ? 'dark' : 'light';
  applyTheme(); applyFocus(); legend(); hash(); };
$('#sheetclose').onclick = closeSheet;
$('#ctlclose').onclick = closeCtls;
$('#ctlbtn').onclick = () => {
  const on = !document.body.classList.contains('ctls');
  $('#ctlbtn').setAttribute('aria-pressed', on);
  on ? openCtls() : closeCtls();
};
$('#lgt').onclick = e => { const on = e.target.getAttribute('aria-pressed') !== 'true';
  e.target.setAttribute('aria-pressed', on); $('#legend').classList.toggle('on', on); };
/* Re-lay out when the viewport crosses the phone breakpoint — a rotation is exactly that, and a
   graph laid out for one shape is unreadable in the other. Re-fit either way, since the canvas
   size changed regardless. */
let WAS_NARROW = NARROW(), rt = null;
addEventListener('resize', () => {
  clearTimeout(rt);
  rt = setTimeout(() => {
    if (NARROW() !== WAS_NARROW) { WAS_NARROW = NARROW(); layout(); }
    else cy.fit(undefined, NARROW() ? 12 : 24);
  }, 180);
});
cy.on('tap', 'node', ev => showNode(ev.target.id()));
cy.on('tap', 'edge', ev => showEdge(ev.target.id()));
cy.on('tap', ev => { if (ev.target === cy) { closeSheet(); closeCtls();
    $('#ctlbtn').setAttribute('aria-pressed', 'false'); }
  if (ev.target === cy) if (ev.target === cy) $('#sidebody').innerHTML =
  '<div class="kind">Click a node or an edge. Use <b>focus</b> to isolate everything that feeds ' +
  'a sink — e.g. <code>vf_projection</code> answers “what does the critic see”.</div>'; });
cy.on('mouseover', 'edge', ev => { const e = ev.target.data(), m = measure(e);
  const t = $('#tip'); t.style.display = 'block';
  t.innerHTML = `<b>${esc(e.src)} → ${esc(e.dst)}</b><br>${esc(e.type)} · ${esc(e.carries)}` +
    (e.width ? `<br>width ${e.width}` : '') +
    (e.family ? `<br>family <b>${esc(e.family)}</b> — ${esc(famLabel(e.family))}` +
      (famCells(e.family) ? `<br><span class="mut">${esc(famCells(e.family))}</span>` : '') : '') +
    (m ? `<br><b>${esc(METRIC)}</b> ${fmt(m.v)}${m.bad ? ' ⚠' : ''}`
       : '<br><i>unmeasured here</i>'); });
cy.on('mouseout', 'edge', () => $('#tip').style.display = 'none');
document.addEventListener('mousemove', e => { const t = $('#tip');
  t.style.left = Math.min(e.clientX + 14, innerWidth - 345) + 'px';
  t.style.top = (e.clientY + 16) + 'px'; });
header(); applyTheme(); legend(); applyFocus();   // applyTheme covers the `#dark=1` deep link
/* A deep link restores the panel too, not just the dimming — otherwise a shared
   `#focus=vf_projection` lands on a filtered graph with an empty panel beside it. */
if (FOCUS) showNode(FOCUS);
else $('#sidebody').innerHTML = '<div class="kind">Click any node — a token seat, the operator, a ' +
  'head, a logit — for its channels, what it can deliver to, and the bias families acting on ' +
  'it. Use <b>focus</b> to isolate everything that feeds a sink: <code>vf_projection</code> ' +
  'answers “what does the critic see”.</div>';

/* A machine-readable record that the page actually RENDERED, for the headless check in
   `build_arch_viewer_render_integration_test.py`. Without it a render test can only assert the
   file was served; with it, it can assert the script ran to completion, that every node got a
   position, and — the bug this exists for — that the theme reached the CYTOSCAPE CANVAS and not
   merely the CSS, which is a separate palette that a deep-linked theme once missed. */
document.body.dataset.ready = '1';
document.body.dataset.nodes = String(cy.nodes().length);
document.body.dataset.positioned =
  String(cy.nodes().filter(n => n.position('x') !== 0 || n.position('y') !== 0).length);
document.body.dataset.nodeBg = cy.nodes()[0].style('background-color');
