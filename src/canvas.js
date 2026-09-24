// Node-based pipeline canvas (n8n style): blocks, connections, drag, auto layout.
import { S, emit, node, nodeOf, columns } from './store.js';
import { BLOCKS, MODEL, MODELS, blockDefaults, modelDefaults, FAMILY_COLORS, DEFAULT_MODELS } from './registry.js';
import { $, h, toast, clamp, fmt } from './util.js';
import { icon as ic, MODEL_ICON } from './icons.js';

const MODEL_LIKE = new Set(['model', 'zoo', 'voting', 'stacking']);
export const isModelLike = n => MODEL_LIKE.has(n.type);
const wrank = n => (isModelLike(n) ? 5 : BLOCKS[n.type].rank);
const lrank = n => (n.type === 'model' ? 5 : BLOCKS[n.type].rank);
let seq = 1, eseq = 1;
const W = 1080;

export function nodeColor(n) { return n.type === 'model' ? FAMILY_COLORS[MODEL[n.key]?.fam] || '#16a34a' : BLOCKS[n.type].c; }
export function nodeIcon(n) { return n.type === 'model' ? MODEL_ICON[MODEL[n.key]?.fam] || 'cpu' : BLOCKS[n.type].i; }
export function nodeTitle(n) { return n.type === 'model' ? MODEL[n.key]?.name || n.key : BLOCKS[n.type].t; }
export function nodeDesc(n) {
  if (n.type === 'model') { const m = MODEL[n.key]; return m ? m.fam + (modelAvailable(n.key) ? '' : ' · not available') : ''; }
  if (n.type === 'zoo') return `${(n.cfg.models || []).length} models`;
  if (n.type === 'dataset') return S.dataName ? `${S.dataName}${n.cfg.target ? ' → ' + n.cfg.target : ''}` : BLOCKS.dataset.d;
  if (n.type === 'split') { const c = n.cfg; return `Test ${Math.round(c.test_size * 100)}% · Val ${Math.round(c.val_size * 100)}% · ${c.cv_strategy === 'none' ? 'no CV' : c.folds + '-fold CV'}`; }
  return BLOCKS[n.type].d;
}
export function modelAvailable(key) {
  const m = MODEL[key]; if (!m) return false;
  const lib = String(typeof m.cls === 'string' ? m.cls : m.cls.classification || m.cls.regression).split(':')[0];
  return S.engine.available?.[lib] !== false;
}

export function reachable(start) {
  const seen = new Set([start]), q = [start];
  while (q.length) { const id = q.shift(); for (const e of S.edges) if (e.from === id && !seen.has(e.to)) { seen.add(e.to); q.push(e.to); } }
  return seen;
}
export function link(a, b) {
  if (!a || !b || a === b || b.type === 'dataset') return false;
  if (S.edges.some(e => e.from === a.id && e.to === b.id)) return false;
  if (reachable(b.id).has(a.id)) return false;
  S.edges.push({ id: 'e' + eseq++, from: a.id, to: b.id });
  return true;
}
export function addNode(type, key = null, cfg = null) {
  const n = { id: 'n' + seq++, type, key, x: 40, y: 40, w: 180, h: 84, cfg: {}, status: 'idle', notes: '', chat: [], mode: 'manual' };
  if (type === 'model') { n.cfg = cfg || modelDefaults(key, taskGuess()); n.grid = ''; }
  else n.cfg = { ...blockDefaults(type), ...(cfg || {}) };
  if (type === 'dataset' && S.profile && !cfg) { n.cfg.target = S.profile.target_guess || ''; n.cfg.exclude = S.profile.cols.filter(c => c.id_like).map(c => c.name); }
  if (type === 'zoo' && !cfg) n.cfg.models = MODELS.filter(m => m.tasks.includes(taskGuess()) && !(m.flags || {}).slow).map(m => m.key).filter(modelAvailable);
  S.nodes.push(n);
  return n;
}
function autowire(n) {
  const r = wrank(n), others = S.nodes.filter(x => x !== n);
  const lower = others.filter(x => wrank(x) < r), higher = others.filter(x => wrank(x) > r);
  const maxL = Math.max(-1, ...lower.map(wrank)), minH = Math.min(99, ...higher.map(wrank));
  const preds = lower.filter(x => wrank(x) === maxL), succs = higher.filter(x => wrank(x) === minH);
  if (!isModelLike(n)) S.edges = S.edges.filter(e => !(preds.some(p => p.id === e.from) && succs.some(s => s.id === e.to)));
  preds.forEach(p => link(p, n));
  succs.forEach(s => link(n, s));
}
export function insertNode(type, key = null) {
  if (type !== 'model' && type !== 'zoo' && nodeOf(type)) return nodeOf(type);
  if (type === 'model' && S.nodes.some(n => n.type === 'model' && n.key === key)) return S.nodes.find(n => n.type === 'model' && n.key === key);
  const n = addNode(type, key);
  autowire(n);
  tidy();
  S.dirty = true;
  return n;
}
export function deleteNode(id) {
  const n = node(id); if (!n) return;
  const preds = S.edges.filter(e => e.to === id).map(e => node(e.from)), succs = S.edges.filter(e => e.from === id).map(e => node(e.to));
  S.edges = S.edges.filter(e => e.from !== id && e.to !== id);
  S.nodes = S.nodes.filter(x => x !== n);
  if (!isModelLike(n)) preds.forEach(p => succs.forEach(s => { if (!(isModelLike(p) && isModelLike(s))) link(p, s); }));
  if (S.sel === id) S.sel = null;
  S.dirty = true;
}
export function taskGuess() {
  const ds = nodeOf('dataset');
  if (S.results) return S.results.task;
  const t = ds?.cfg.task;
  if (t && t !== 'auto') return t;
  const c = S.profile?.cols.find(x => x.name === ds?.cfg.target);
  if (!c) return 'classification';
  if (c.kind !== 'numeric') return 'classification';
  return c.unique <= 20 && c.integer ? 'classification' : 'regression';
}

let CH = 470;
export function setSeq() {
  seq = 1 + Math.max(0, ...S.nodes.map(n => parseInt(String(n.id).slice(1)) || 0));
  eseq = 1 + Math.max(0, ...S.edges.map(e => parseInt(String(e.id).slice(1)) || 0));
}
export function tidy() {
  const top = S.nodes.filter(n => lrank(n) < 5 && !isModelLike(n)).sort((a, b) => lrank(a) - lrank(b));
  const models = S.nodes.filter(isModelLike).sort((a, b) => (a.type === 'model' ? 0 : a.type === 'zoo' ? 1 : 2) - (b.type === 'model' ? 0 : b.type === 'zoo' ? 1 : 2));
  const bottom = S.nodes.filter(n => lrank(n) > 5 && !isModelLike(n)).sort((a, b) => lrank(a) - lrank(b));
  const row = (arr, y, maxW, hh) => {
    const gap = 26, n = arr.length; if (!n) return;
    const w = Math.min(maxW, (W - 40 - gap * (n - 1)) / n), total = n * w + gap * (n - 1);
    let x = (W - total) / 2;
    for (const nd of arr) { nd.w = w; nd.h = hh; nd.x = x; nd.y = y; x += w + gap; }
  };
  row(top, 24, 196, 88);
  let y = 24 + 88 + 64;
  const per = 6;
  for (let i = 0; i < models.length; i += per) { row(models.slice(i, i + per), y, 164, 78); y += 78 + 34; }
  if (!models.length) y += 20;
  row(bottom, y + 26, 250, 80);
  CH = y + 26 + 80 + 26;
}
export function buildPipeline(spec) {
  S.nodes = []; S.edges = []; S.sel = null;
  for (const it of spec) { const n = typeof it === 'string' ? addNode(it) : addNode(it.type, it.key, it.cfg); autowire(n); }
  tidy();
}
export function defaultPipeline(task = taskGuess()) {
  const models = DEFAULT_MODELS[task].filter(modelAvailable);
  if (!models.includes('xgb') && modelAvailable('hgb')) models.splice(2, 0, 'hgb');
  buildPipeline(['dataset', 'split', 'preprocess', 'fe', 'fs', ...models.map(key => ({ type: 'model', key })), 'eval', 'tuning', 'deploy']);
}

// ---------- rendering
let scale = 1;
export function renderCanvas() {
  const wrap = $('#cwrap'); if (!wrap) return;
  scale = clamp((wrap.clientWidth - 2) / W, 0.5, 1);
  $('#stage').style.width = W * scale + 'px';
  $('#stage').style.height = CH * scale + 'px';
  const w = $('#world');
  w.style.width = W + 'px'; w.style.height = CH + 'px'; w.style.transform = `scale(${scale})`;
  renderNodes();
}
function edgePath(a, b) {
  if (b.y > a.y + a.h - 4) { const sx = a.x + a.w / 2, sy = a.y + a.h, tx = b.x + b.w / 2, ty = b.y, my = sy + (ty - sy) / 2; return `M${sx},${sy} V${my} H${tx} V${ty - 3}`; }
  if (b.y + b.h < a.y + 4) { const sx = a.x + a.w / 2, sy = a.y, tx = b.x + b.w / 2, ty = b.y + b.h, my = ty + (sy - ty) / 2; return `M${sx},${sy} V${my} H${tx} V${ty + 3}`; }
  const sx = a.x + a.w, sy = a.y + a.h / 2, tx = b.x, ty = b.y + b.h / 2;
  if (tx >= sx) { const mx = sx + (tx - sx) / 2; return `M${sx},${sy} H${mx} V${ty} H${tx - 3}`; }
  const my = Math.max(a.y + a.h, b.y + b.h) + 16;
  return `M${sx},${sy} h14 V${my} H${tx - 14} V${ty} H${tx - 3}`;
}
export function renderEdges() {
  const svg = $('#edges'); if (!svg) return;
  svg.setAttribute('width', W); svg.setAttribute('height', CH);
  let p = '';
  for (const e of S.edges) {
    const a = node(e.from), b = node(e.to); if (!a || !b) continue;
    const d = edgePath(a, b), live = S.running && b.status === 'running';
    p += `<g class="edge${live ? ' live' : ''}" data-id="${e.id}"><title>Click to remove this connection</title><path class="ehit" d="${d}"/><path class="eline" d="${d}" marker-end="url(#arrow)"/></g>`;
  }
  svg.innerHTML = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--edge)"/></marker></defs>${p}<path id="tmpEdge" class="eline tmp" d=""/>`;
}
function nodeEl(n) {
  const stat = n.status === 'done' ? ic('check', 11, 3) : n.status === 'error' ? '!' : '';
  const unavailable = n.type === 'model' && !modelAvailable(n.key);
  const el = h('div', { class: 'node' + (isModelLike(n) ? ' model' : '') + (S.sel === n.id ? ' sel' : '') + (unavailable ? ' unavail' : ''), 'data-id': n.id, tabindex: '0', role: 'button', 'aria-label': `${nodeTitle(n)} block, ${n.status}`, title: unavailable ? 'This library is not available in the browser engine' : '', style: `--nc:${nodeColor(n)};left:${n.x}px;top:${n.y}px;width:${n.w}px;height:${n.h}px` },
    n.type !== 'dataset' ? h('span', { class: 'port in' }) : null,
    h('span', { class: 'nicon', html: ic(nodeIcon(n), isModelLike(n) ? 16 : 19) }),
    h('div', { class: 'ntext' }, h('div', { class: 'nt' }, nodeTitle(n)), n.metric ? h('div', { class: 'nm' }, n.metric) : h('div', { class: 'nd' }, nodeDesc(n))),
    h('span', { class: 'nstat ' + n.status, html: stat, title: n.statusText || n.status }),
    h('span', { class: 'port out', title: 'Drag to connect' }));
  el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(n.id); } if (e.key === 'Delete') { deleteNode(n.id); tidy(); renderCanvas(); emit('pipeline'); } });
  return el;
}
export function renderNodes() {
  const layer = $('#nodes'); if (!layer) return;
  layer.innerHTML = '';
  for (const n of S.nodes) layer.append(nodeEl(n));
  renderEdges();
}
export function select(id) { S.sel = id; S.ui.cfgTab = 'config'; renderNodes(); emit('select', id); }

export function bindCanvas() {
  const world = $('#world');
  let drag = null;
  const pt = e => { const r = world.getBoundingClientRect(); return { x: (e.clientX - r.left) / scale, y: (e.clientY - r.top) / scale }; };
  world.addEventListener('pointerdown', e => {
    const port = e.target.closest('.port.out'), nd = e.target.closest('.node');
    if (port && nd) { e.preventDefault(); drag = { mode: 'link', from: nd.dataset.id }; world.setPointerCapture(e.pointerId); return; }
    if (nd) { const n = node(nd.dataset.id), p = pt(e); drag = { mode: 'move', id: n.id, ox: p.x - n.x, oy: p.y - n.y, sx: e.clientX, sy: e.clientY, moved: false, el: nd }; world.setPointerCapture(e.pointerId); return; }
    const eg = e.target.closest('.edge');
    if (eg) { const ed = S.edges.find(x => x.id === eg.dataset.id); if (ed) { S.edges = S.edges.filter(x => x !== ed); renderEdges(); emit('pipeline'); toast('Connection removed', { label: 'Undo', fn: () => { S.edges.push(ed); renderEdges(); emit('pipeline'); } }); } }
  });
  world.addEventListener('pointermove', e => {
    if (!drag) return;
    const p = pt(e);
    if (drag.mode === 'move') {
      if (!drag.moved && Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) < 4) return;
      drag.moved = true; drag.el.classList.add('drag');
      const n = node(drag.id);
      n.x = clamp(p.x - drag.ox, 0, W - n.w); n.y = clamp(p.y - drag.oy, 0, CH - n.h);
      drag.el.style.left = n.x + 'px'; drag.el.style.top = n.y + 'px';
      renderEdges();
    } else { const a = node(drag.from); $('#tmpEdge')?.setAttribute('d', `M${a.x + a.w},${a.y + a.h / 2} L${p.x},${p.y}`); }
  });
  const end = e => {
    if (!drag) return;
    const d = drag; drag = null;
    if (d.mode === 'move') { d.el.classList.remove('drag'); if (!d.moved) select(d.id); return; }
    $('#tmpEdge')?.setAttribute('d', '');
    const tEl = document.elementFromPoint(e.clientX, e.clientY)?.closest('.node');
    if (tEl && tEl.dataset.id !== d.from) {
      const a = node(d.from), b = node(tEl.dataset.id);
      if (b.type === 'dataset') toast('The Dataset block has no input.');
      else if (link(a, b)) { renderEdges(); emit('pipeline'); toast(`Connected ${nodeTitle(a)} → ${nodeTitle(b)}`); }
      else toast('That connection already exists or would create a loop.');
    }
  };
  world.addEventListener('pointerup', end);
  world.addEventListener('pointercancel', () => { drag = null; renderNodes(); });
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => renderCanvas()).observe($('#cwrap'));
}

export function renderAddMenu(filter = '') {
  const m = $('#addMenu');
  const task = taskGuess();
  const q = filter.toLowerCase();
  m.innerHTML = '';
  m.append(h('input', { type: 'search', placeholder: 'Search blocks and models…', value: filter, 'aria-label': 'Search blocks', oninput: e => { renderAddMenu(e.target.value); m.querySelector('input').focus(); } }));
  const item = (label, color, iconName, disabled, fn, title = '') => h('button', { class: 'addItem', disabled, title, style: `--nc:${color}`, onclick: () => { fn(); m.hidden = true; } }, h('span', { class: 'nicon', html: ic(iconName, 14) }), label);
  const place = n => { renderCanvas(); select(n.id); emit('pipeline'); toast(`Added ${nodeTitle(n)}`); };
  const blocks = ['dataset', 'split', 'preprocess', 'fe', 'fs', 'zoo', 'voting', 'stacking', 'eval', 'tuning', 'deploy'].filter(t => !q || BLOCKS[t].t.toLowerCase().includes(q));
  if (blocks.length) m.append(h('h4', {}, 'Blocks'), h('div', { class: 'addGrid' }, blocks.map(t => item(BLOCKS[t].t, BLOCKS[t].c, BLOCKS[t].i, t !== 'zoo' && !!nodeOf(t), () => place(insertNode(t))))));
  const fams = {};
  for (const mdl of MODELS) { if (q && !(mdl.name + ' ' + mdl.fam + ' ' + mdl.key).toLowerCase().includes(q)) continue; (fams[mdl.fam] ||= []).push(mdl); }
  for (const [fam, list] of Object.entries(fams)) {
    m.append(h('h4', {}, fam), h('div', { class: 'addGrid' }, list.map(mdl => {
      const wrong = !mdl.tasks.includes(task), exists = S.nodes.some(n => n.type === 'model' && n.key === mdl.key), avail = modelAvailable(mdl.key);
      return item(mdl.name + (wrong ? ` (${mdl.tasks[0]})` : ''), FAMILY_COLORS[fam], MODEL_ICON[fam] || 'cpu', exists || !avail, () => place(insertNode('model', mdl.key)), !avail ? 'Not available in this browser engine' : wrong ? `Only for ${mdl.tasks.join(' / ')}` : '');
    })));
  }
}
export function modelIdsOf(n) {
  if (n.type === 'zoo') return (n.cfg.models || []).map(k => `${n.id}__${k}`);
  return [n.id];
}
export function nodeForModelId(mid) {
  if (!mid) return null;
  const base = mid.replace(/__tuned$/, '');
  return node(base) || node(base.split('__')[0]);
}
export { W as CANVAS_W, fmt };
