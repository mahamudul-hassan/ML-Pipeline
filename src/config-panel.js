// Block configuration panel (Manual / AI Powered) + per-block AI chat.
import { S, emit, node, columns, colInfo } from './store.js';
import { BLOCKS, MODEL, MODELS, optLabel, paramOptions, paramDefault, sanitizeBlock, sanitizeModelParams, FAMILY_COLORS } from './registry.js';
import { h, $, toast, esc } from './util.js';
import { icon, MODEL_ICON } from './icons.js';
import { nodeTitle, nodeColor, nodeIcon, deleteNode, tidy, renderCanvas, select, taskGuess, modelAvailable, isModelLike } from './canvas.js';
import { AI, aiReady, blockAI } from './ai.js';
import { ChatView, newThread } from './chat.js';
import { A } from './actions.js';

const PLACEHOLDER = {
  dataset: 'e.g. Predict "pass" as classification and ignore the id column.', split: 'e.g. Use 10-fold stratified cross-validation and a 15% test set.',
  preprocess: 'e.g. Median imputation, robust scaling because of outliers, and SMOTE for the imbalance.', fe: 'e.g. Add a study/sleep ratio and polynomial features for attendance.',
  fs: 'e.g. Keep the 10 most informative features using mutual information.', eval: 'e.g. Recall matters most; explain the top 3 models.', tuning: 'e.g. Tune the top 3 models with a random search of 30 candidates.',
  model: 'e.g. Make it less overfit / faster.', zoo: 'e.g. Pick fast models that suit this dataset.', voting: 'e.g. Combine the best 4 models.', stacking: 'e.g. Stack the top 3 models.', deploy: 'e.g. What do I need to serve this as an API?',
};
const changed = () => { S.dirty = true; emit('pipeline'); };

function inputFor(f, n) {
  const v = n.cfg[f.k];
  const set = val => { n.cfg[f.k] = val; changed(); };
  const cols = columns();
  switch (f.f) {
    case 'target': case 'col': {
      const opts = f.f === 'col' ? ['', ...cols] : cols;
      return h('select', { 'aria-label': f.l, onchange: e => { set(e.target.value); if (f.k === 'target') { A.refreshTargetDependent?.(); } } }, opts.map(c => h('option', { value: c, selected: c === v }, c || '(none)')));
    }
    case 'cols': case 'numcols': {
      const list = f.f === 'numcols' ? S.profile?.cols.filter(c => c.kind === 'numeric').map(c => c.name) || [] : cols;
      const target = n.cfg.target;
      if (!list.length) return h('div', { class: 'xs muted' }, 'Load a dataset to choose columns.');
      return h('div', { class: 'colbox' }, list.filter(c => c !== target || f.f === 'numcols').map(c => h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: (v || []).includes(c), onchange: e => set(e.target.checked ? [...new Set([...(n.cfg[f.k] || []), c])] : (n.cfg[f.k] || []).filter(x => x !== c)) }), c, colInfo(c) ? h('span', { class: 'chip ' + colInfo(c).kind, style: 'margin-left:auto' }, colInfo(c).kind) : null)));
    }
    case 'cat': return h('select', { 'aria-label': f.l, onchange: e => { const hit = f.o.find(o => String(o) === e.target.value); set(hit); } }, f.o.map(o => h('option', { value: String(o), selected: String(o) === String(v) }, optLabel(o))));
    case 'int': case 'float': return h('input', { type: 'number', value: v ?? '', min: f.min, max: f.max, step: f.step || (f.f === 'int' ? 1 : 'any'), 'aria-label': f.l, onchange: e => { const s = sanitizeBlock(n.type, { [f.k]: e.target.value }); if (f.k in s) { set(s[f.k]); e.target.value = s[f.k]; } } });
    case 'feats': {
      const box = h('div', {});
      const draw = () => {
        box.innerHTML = '';
        (n.cfg.custom_features || []).forEach((ft, i) => box.append(h('div', { class: 'row', style: 'margin-bottom:6px;flex-wrap:nowrap' },
          h('input', { type: 'text', value: ft.name, placeholder: 'name', style: 'width:38%', 'aria-label': 'Feature name', onchange: e => { ft.name = e.target.value.replace(/[^\w]/g, '_'); changed(); } }),
          h('input', { type: 'text', value: ft.expr, placeholder: 'study_hours / (sleep_hours + 1)', 'aria-label': 'Expression', onchange: e => { ft.expr = e.target.value; changed(); } }),
          h('button', { class: 'ib sm', 'aria-label': 'Remove feature', html: icon('x', 14), onclick: () => { n.cfg.custom_features.splice(i, 1); changed(); draw(); } }))));
        box.append(h('button', { class: 'btn sm', onclick: () => { (n.cfg.custom_features ||= []).push({ name: 'new_feature', expr: '' }); draw(); } }, '+ Add feature'),
          h('div', { class: 'xs muted', style: 'margin-top:6px' }, 'pandas expressions over column names: + − * / **, comparisons, log, sqrt, abs, exp.'));
      };
      draw(); return box;
    }
    case 'json': return h('textarea', { rows: 4, class: 'mono', style: 'font-size:12px', 'aria-label': f.l, placeholder: '{"rf": {"n_estimators": [100, 300], "max_depth": [null, 10]}}', onchange: e => { try { set(e.target.value.trim() ? JSON.parse(e.target.value) : {}); e.target.style.borderColor = ''; } catch { e.target.style.borderColor = 'var(--err)'; toast('That is not valid JSON.'); } } }, Object.keys(v || {}).length ? JSON.stringify(v, null, 1) : '');
    case 'models': return zooPicker(n);
  }
  return h('span', {}, '');
}
function zooPicker(n) {
  const task = taskGuess();
  const box = h('div', {});
  const draw = () => {
    box.innerHTML = '';
    const sel = new Set(n.cfg.models || []);
    const set = arr => { n.cfg.models = arr; changed(); draw(); };
    const ok = MODELS.filter(m => m.tasks.includes(task) && modelAvailable(m.key));
    box.append(h('div', { class: 'row', style: 'margin-bottom:6px' },
      h('button', { class: 'btn sm', onclick: () => set(ok.map(m => m.key)) }, `All ${ok.length}`),
      h('button', { class: 'btn sm', onclick: () => set(ok.filter(m => !(m.flags || {}).slow).map(m => m.key)) }, 'Fast only'),
      h('button', { class: 'btn sm', onclick: () => set([]) }, 'None')));
    const fams = {};
    for (const m of ok) (fams[m.fam] ||= []).push(m);
    const list = h('div', { class: 'colbox', style: 'max-height:300px' });
    for (const [fam, ms] of Object.entries(fams)) {
      list.append(h('div', { class: 'xs muted', style: 'margin:8px 0 2px;font-weight:700' }, fam));
      for (const m of ms) list.append(h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: sel.has(m.key), onchange: e => set(e.target.checked ? [...sel, m.key] : [...sel].filter(k => k !== m.key)) }), m.name, (m.flags || {}).slow ? h('span', { class: 'chip', style: 'margin-left:auto' }, 'slow') : null));
    }
    box.append(list, h('div', { class: 'xs muted', style: 'margin-top:6px' }, `${sel.size} selected for ${task}. Slow models train on a subsample of large data.`));
  };
  draw(); return box;
}
function paramInput(n, p) {
  const task = taskGuess();
  const opts = paramOptions(p, task);
  const v = n.cfg[p.k] !== undefined ? n.cfg[p.k] : paramDefault(p, task);
  const set = val => { const s = sanitizeModelParams(n.key, { [p.k]: val }, task); if (p.k in s) { n.cfg[p.k] = s[p.k]; changed(); } else toast(`Invalid value for ${p.k}`); };
  if (p.t === 'bool') return h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: !!v, onchange: e => set(e.target.checked) }), h('span', { class: 'mono sm' }, p.k));
  let inp;
  if (opts) inp = h('select', { 'aria-label': p.k, onchange: e => set(e.target.value === 'null' ? null : opts.find(o => String(o) === e.target.value)) }, opts.map(o => h('option', { value: String(o), selected: String(o) === String(v) }, o === null ? 'None' : String(o))));
  else if (p.t === 'tuple') inp = h('input', { type: 'text', value: v ?? '', placeholder: '100 or 128,64', 'aria-label': p.k, onchange: e => set(e.target.value) });
  else inp = h('input', { type: 'number', value: v ?? '', step: 'any', placeholder: p.t === 'nint' || p.nullable ? 'None' : '', 'aria-label': p.k, onchange: e => set(e.target.value === '' ? null : e.target.value) });
  return h('div', { class: 'fld' }, h('label', { class: 'fl mono' }, p.k), inp);
}

function fieldsBlock(n) {
  const out = [];
  if (n.type === 'model') {
    const m = MODEL[n.key];
    const task = taskGuess();
    if (!m.tasks.includes(task)) out.push(h('div', { class: 'note warn' }, `${m.name} is for ${m.tasks.join(' / ')}. It will be skipped for this ${task} task.`));
    if (!modelAvailable(n.key)) out.push(h('div', { class: 'note warn' }, 'This library did not load in the browser engine, so this model will fail. It still works in the exported Python project.'));
    out.push(h('div', { class: 'sec' }, 'Hyperparameters'), ...m.params.filter(p => !(p.task && p.task !== task)).map(p => paramInput(n, p)));
    if (m.note) out.push(h('div', { class: 'note' }, m.note));
    if ((m.flags || {}).slow) out.push(h('div', { class: 'note' }, `Slow on big data: trains on up to ${S.settings.slowModelRows} rows (change in Settings).`));
    const adv = h('details', { class: 'adv' }, h('summary', {}, 'Tuning search space', h('span', { html: icon('chev', 16) })));
    const gridText = n.grid || '';
    adv.append(h('textarea', { rows: 5, class: 'mono', style: 'font-size:12px', placeholder: JSON.stringify(m.grid), 'aria-label': 'Search space JSON', onchange: e => { const t = e.target.value.trim(); if (!t) { n.grid = ''; changed(); return; } try { JSON.parse(t); n.grid = t; e.target.style.borderColor = ''; changed(); } catch { e.target.style.borderColor = 'var(--err)'; toast('That is not valid JSON.'); } } }, gridText),
      h('div', { class: 'xs muted', style: 'margin-top:6px' }, 'Leave empty to use the default shown as the placeholder. Lists are tried as given by grid search; Optuna and Bayesian search treat numeric lists as ranges. You can also give ranges: {"n_estimators": {"low": 50, "high": 600, "type": "int"}, "learning_rate": {"low": 0.01, "high": 0.3, "log": true}}.'));
    out.push(adv);
    return out;
  }
  if (n.type === 'deploy') return deployBlock();
  const F = BLOCKS[n.type].fields || [];
  const main = F.filter(f => !f.op && !f.adv), ops = F.filter(f => f.op), adv = F.filter(f => f.adv);
  if (n.type === 'dataset') out.push(datasetSource());
  for (const f of main) out.push(h('div', { class: 'fld' }, h('label', { class: 'fl' }, f.l), inputFor(f, n)));
  if (ops.length) out.push(h('div', { class: 'sec' }, 'Available operations'), ...ops.map(f => h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: !!n.cfg[f.k], onchange: e => { n.cfg[f.k] = e.target.checked; changed(); } }), f.l)));
  if (adv.length) {
    const d = h('details', { class: 'adv' }, h('summary', {}, 'Advanced options', h('span', { html: icon('chev', 16) })));
    for (const f of adv) d.append(f.f === 'bool' ? h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: !!n.cfg[f.k], onchange: e => { n.cfg[f.k] = e.target.checked; changed(); } }), f.l) : h('div', { class: 'fld' }, h('label', { class: 'fl' }, f.l), inputFor(f, n)));
    out.push(d);
  }
  if (n.type === 'split') out.push(h('div', { class: 'note' }, 'Models are ranked by cross-validation on the training set (or the validation set when CV is off). The test set is only used for the final, unbiased report.'));
  if (n.type === 'tuning') out.push(h('div', { class: 'note' }, 'Optuna (TPE Bayesian optimisation) is the default: it learns from earlier trials which hyperparameters work. Numeric lists in a model\'s search space become continuous ranges (log scale for C, alpha, learning rate…). Each model block has its own search space (open a model → "Tuning search space"); ranges look like {"low": 0.01, "high": 10, "log": true}. Tuned models appear in the leaderboard as "(tuned)".'));
  return out;
}
function datasetSource() {
  const box = h('div', {});
  if (S.profile) box.append(h('div', { class: 'fileRow' }, h('span', { class: 'fic', html: icon('file', 18) }), h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname' }, S.dataName), h('div', { class: 'fmeta' }, `${S.profile.rows.toLocaleString()} rows · ${S.profile.cols.length} columns`))));
  box.append(h('div', { class: 'row' }, h('button', { class: 'btn sm', html: icon('upload', 14) + ' Upload file', onclick: () => A.openFile() }), h('button', { class: 'btn sm', html: icon('db', 14) + ' Sample datasets', onclick: () => A.sampleMenu() })));
  return box;
}
function deployBlock() {
  const R = S.results;
  const best = R ? R.models[R.best] : null;
  return [
    h('div', { class: 'note' }, best ? `Best model: ${best.name}. Exports include the fitted model (joblib), a Python project that re-trains the same pipeline with native scikit-learn, and a FastAPI server.` : 'Run the pipeline to enable model exports. The pipeline itself can be exported any time.'),
    h('div', { class: 'cfgFoot' },
      h('button', { class: 'btn primary', disabled: !R, html: icon('download', 15) + ' Download project (.zip)', onclick: () => A.exportThing('project_zip') }),
      h('button', { class: 'btn', disabled: !R, html: icon('cpu', 15) + ' Model file (.joblib)', onclick: () => A.exportThing('model_file') }),
      h('button', { class: 'btn', disabled: !R, html: icon('list', 15) + ' Test-set predictions (.csv)', onclick: () => A.exportThing('test_predictions') }),
      h('button', { class: 'btn', html: icon('save', 15) + ' Pipeline (.json)', onclick: () => A.exportThing('pipeline_json') })),
  ];
}

let aiDraft = { id: null, text: '', result: null, busy: false };
function aiPowered(n) {
  if (aiDraft.id !== n.id) aiDraft = { id: n.id, text: '', result: null, busy: false };
  const box = h('div', {});
  const ta = h('textarea', { rows: 4, placeholder: PLACEHOLDER[n.type] || 'Describe what you want…', 'aria-label': 'Describe the configuration', oninput: e => { aiDraft.text = e.target.value; } }, aiDraft.text);
  const btn = h('button', { class: 'btn violet wide', disabled: aiDraft.busy, html: icon('sparkle', 15) + (aiDraft.busy ? ' Thinking…' : ' Generate settings with AI'), onclick: async () => {
    if (!aiReady()) { A.openSettings('ai'); return; }
    if (!aiDraft.text.trim()) { toast('Describe what you want first.'); return; }
    aiDraft.busy = true; render();
    try {
      const r = await blockAI(n, aiDraft.text);
      const patch = n.type === 'model' ? sanitizeModelParams(n.key, r.settings, taskGuess()) : sanitizeBlock(n.type, r.settings, columns());
      aiDraft.result = { explanation: r.explanation, patch, dropped: Object.keys(r.settings).filter(k => !(k in patch)) };
    } catch (e) { aiDraft.result = { error: e.message }; }
    aiDraft.busy = false; render();
  } });
  box.append(h('div', { class: 'fld' }, h('label', { class: 'fl' }, 'Tell the AI what this block should do'), ta), btn);
  const r = aiDraft.result;
  if (r) {
    if (r.error) box.append(h('div', { class: 'note warn' }, r.error));
    else {
      const keys = Object.keys(r.patch);
      box.append(h('div', { class: 'aiOut' }, h('div', {}, r.explanation || 'Suggested settings:'),
        keys.length ? h('table', { class: 'tbl', style: 'margin-top:8px' }, h('tr', {}, h('th', {}, 'Setting'), h('th', {}, 'Now'), h('th', {}, 'Suggested')), keys.map(k => h('tr', {}, h('td', { class: 'mono' }, k), h('td', {}, fmtVal(n.cfg[k])), h('td', {}, h('b', {}, fmtVal(r.patch[k])))))) : h('div', { class: 'xs muted' }, 'No changes suggested.'),
        r.dropped.length ? h('div', { class: 'xs muted', style: 'margin-top:6px' }, `Ignored unknown or invalid settings: ${r.dropped.join(', ')}`) : null,
        keys.length ? h('div', { class: 'mact' }, h('button', { class: 'btn sm primary', onclick: () => { Object.assign(n.cfg, r.patch); aiDraft.result = null; n.mode = 'manual'; changed(); render(); toast('Settings applied'); } }, 'Apply'), h('button', { class: 'btn sm', onclick: () => { aiDraft.result = null; render(); } }, 'Discard')) : null));
    }
  }
  return box;
}
const fmtVal = v => (v === null || v === undefined ? 'None' : Array.isArray(v) ? (v.length ? v.map(x => (typeof x === 'object' ? x.name : x)).join(', ') : '—') : typeof v === 'object' ? JSON.stringify(v) : optLabel(v));

let blockChat = null;
export function render() {
  const body = $('#cfgBody');
  if (!body) return;
  body.innerHTML = '';
  const n = node(S.sel);
  if (!n) {
    body.append(h('p', { class: 'muted sm', style: 'margin:4px 0' }, 'Select a block on the canvas to configure it.'),
      h('div', { class: 'blist' }, S.nodes.map(x => h('button', { class: 'bpill', style: `--nc:${nodeColor(x)}`, onclick: () => select(x.id) }, h('span', { class: 'd' }), nodeTitle(x)))));
    return;
  }
  const desc = n.type === 'model' ? `${MODEL[n.key].fam} · scikit-learn ${String(typeof MODEL[n.key].cls === 'string' ? MODEL[n.key].cls : MODEL[n.key].cls[taskGuess()] || '').split(':')[1] || ''}` : BLOCKS[n.type].d;
  body.append(h('div', { class: 'cfgHead', style: `--nc:${nodeColor(n)}` }, h('span', { class: 'nicon', html: icon(nodeIcon(n), 20) }), h('div', { style: 'min-width:0' }, h('div', { class: 'cht' }, nodeTitle(n)), h('div', { class: 'chd' }, desc))));
  const tab = (l, k) => h('button', { class: 'tab' + (S.ui.cfgTab === k ? ' on' : ''), onclick: () => { S.ui.cfgTab = k; render(); } }, l);
  body.append(h('div', { class: 'tabs' }, tab('Configuration', 'config'), tab('AI Assistant', 'ai')));
  if (S.ui.cfgTab === 'ai') {
    if (!n.chat || !n.chat.items) n.chat = newThread();
    const holder = h('div', { style: 'display:flex;flex-direction:column;height:520px;margin:0 -14px' });
    body.append(holder);
    blockChat = new ChatView(holder, () => n.chat, { focus: () => n, compact: true, placeholder: `Ask about ${nodeTitle(n)}…`, chips: n.type === 'model' ? ['Explain these hyperparameters', 'Make it less overfit', 'How did this model do?'] : ['What do these options do?', 'Suggest good settings for my data'] });
    return;
  }
  body.append(h('div', { class: 'sec' }, 'Configuration Mode'),
    h('div', { class: 'seg', role: 'tablist' }, ['manual', 'ai'].map(m => h('button', { class: n.mode === m ? 'on' : '', role: 'tab', 'aria-selected': n.mode === m, onclick: () => { n.mode = m; render(); } }, m === 'manual' ? 'Manual' : 'AI Powered'))));
  if (n.mode === 'ai') body.append(aiPowered(n));
  body.append(h('div', { class: 'fld', style: 'margin-top:14px' }, h('label', { class: 'fl' }, 'Instructions / Notes'), h('textarea', { rows: 2, maxlength: 500, placeholder: PLACEHOLDER[n.type] || '', 'aria-label': 'Block notes', oninput: e => { n.notes = e.target.value; e.target.nextSibling.textContent = `${e.target.value.length}/500`; } }, n.notes || ''), h('div', { class: 'cnt' }, `${(n.notes || '').length}/500`)));
  for (const el of fieldsBlock(n)) body.append(el);
  const order = [...S.nodes].sort((a, b) => a.y - b.y || a.x - b.x);
  const next = order[order.indexOf(n) + 1];
  body.append(h('div', { class: 'cfgFoot' },
    h('button', { class: 'btn primary', onclick: () => { toast(`${nodeTitle(n)} saved`); if (next) select(next.id); } }, next ? 'Apply & Continue' : 'Apply'),
    n.type !== 'dataset' ? h('button', { class: 'btn ghost', html: icon('trash', 14) + ' Remove block', onclick: () => { deleteNode(n.id); tidy(); renderCanvas(); emit('pipeline'); render(); toast('Block removed'); } }) : null));
}
export function refreshBlockChat() { if (blockChat && S.ui.cfgTab === 'ai') blockChat.render(); }
