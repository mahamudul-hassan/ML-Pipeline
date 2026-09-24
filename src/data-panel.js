// Dataset details panel and the Extra Documents / Instructions panel.
import { S, emit, nodeOf, store } from './store.js';
import { h, $, fmt, fmtBytes, esc, toast } from './util.js';
import { icon } from './icons.js';
import { A } from './actions.js';

export const SAMPLES = [
  { key: 'student_performance', name: 'Student performance', d: '10,000 rows · pass / fail and test score' },
  { key: 'breast_cancer', name: 'Breast cancer (Wisconsin)', d: '569 rows · 30 features · binary' },
  { key: 'wine', name: 'Wine', d: '178 rows · 13 features · 3 classes' },
  { key: 'iris', name: 'Iris', d: '150 rows · 4 features · 3 classes' },
  { key: 'digits', name: 'Digits', d: '1,797 rows · 64 pixels · 10 classes' },
  { key: 'diabetes', name: 'Diabetes progression', d: '442 rows · 10 features · regression' },
];

export function renderData() {
  const root = $('#dsBody'); if (!root) return;
  root.innerHTML = '';
  if (!S.profile) {
    const drop = h('div', { class: 'drop', tabindex: '0', role: 'button', onclick: () => A.openFile(), onkeydown: e => { if (e.key === 'Enter') A.openFile(); } }, h('div', { html: icon('upload', 22) }), h('div', {}, h('b', {}, 'Drop a CSV, Excel or JSON file'), ' or click to browse'), h('div', { class: 'xs' }, S.engine.status === 'ready' ? 'Processed locally in your browser.' : 'The Python engine is still loading…'));
    bindDrop(drop);
    root.append(drop, h('div', { class: 'sec' }, 'Or try a sample dataset'), ...SAMPLES.map(s => h('button', { class: 'tplCard', style: 'width:100%;text-align:left;cursor:pointer', onclick: () => A.loadSample(s.key) }, h('span', { class: 'fic', html: icon('db', 16) }), h('div', {}, h('b', {}, s.name), h('span', {}, s.d)))));
    return;
  }
  const p = S.profile;
  const missing = p.cols.reduce((a, c) => a + c.missing, 0);
  const fileRow = h('div', { class: 'fileRow' }, h('span', { class: 'fic', html: icon('file', 18) }), h('div', { style: 'min-width:0;flex:1' }, h('div', { class: 'fname' }, S.dataName), h('div', { class: 'fmeta' }, `${S.dataSize ? fmtBytes(S.dataSize) + ' · ' : ''}${p.rows.toLocaleString()} rows · ${p.cols.length} columns${missing ? ` · ${missing.toLocaleString()} missing values` : ''}`)),
    h('button', { class: 'btn sm', onclick: () => A.openFile() }, 'Change'), h('button', { class: 'btn sm', onclick: () => A.sampleMenu() }, 'Samples'));
  bindDrop(fileRow);
  const tabs = ['preview', 'summary', 'info', 'columns'];
  const labels = { preview: 'Preview', summary: 'Summary', info: 'Data Info', columns: 'Columns' };
  root.append(fileRow, h('div', { class: 'tabs', role: 'tablist' }, tabs.map(t => h('button', { class: 'tab' + (S.ui.dataTab === t ? ' on' : ''), role: 'tab', onclick: () => { S.ui.dataTab = t; renderData(); } }, labels[t]))));
  const body = h('div', { class: 'dsBody' });
  root.append(body);
  const target = nodeOf('dataset')?.cfg.target;
  if (S.ui.dataTab === 'preview') {
    if (!S.preview) { body.append(h('div', { class: 'muted sm', style: 'padding:10px' }, 'Loading preview…')); A.loadPreview(0); return; }
    const pv = S.preview;
    const head = h('tr', {}, h('th', {}, '#'), pv.columns.map(c => h('th', {}, c === target ? '🎯 ' + c : c)));
    const cell = v => h('td', { class: typeof v === 'number' ? 'num' : '' }, v == null ? h('span', { class: 'muted' }, 'NaN') : typeof v === 'number' ? fmt(v, 3) : String(v));
    const rows = pv.rows.map((r, i) => h('tr', {}, h('td', { class: 'muted' }, pv.start + i + 1), r.map(cell)));
    body.append(h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, head, rows)));
    body.append(h('div', { class: 'row', style: 'margin-top:8px' }, h('span', { class: 'xs muted' }, `Rows ${pv.start + 1}–${pv.start + pv.rows.length} of ${pv.total.toLocaleString()}`), h('span', { class: 'sp' }),
      h('button', { class: 'btn sm', disabled: pv.start === 0, onclick: () => A.loadPreview(Math.max(0, pv.start - 50)) }, 'Previous'),
      h('button', { class: 'btn sm', disabled: pv.start + 50 >= pv.total, onclick: () => A.loadPreview(pv.start + 50) }, 'Next')));
  } else if (S.ui.dataTab === 'summary') {
    const num = p.cols.filter(c => c.kind === 'numeric');
    body.append(h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['Column', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max', 'Skew'].map(x => h('th', {}, x))),
      num.map(c => h('tr', {}, h('td', {}, c.name), [c.mean, c.std, c.min, c.q1, c.median, c.q3, c.max, c.skew].map(v => h('td', { class: 'num' }, fmt(v, 3))))))),
      h('div', { class: 'sec' }, 'Categorical columns'), h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['Column', 'Unique', 'Top values'].map(x => h('th', {}, x))), p.cols.filter(c => c.kind !== 'numeric').map(c => h('tr', {}, h('td', {}, c.name), h('td', { class: 'num' }, c.unique), h('td', {}, c.top.slice(0, 4).map(([k, n]) => `${k} (${n})`).join(', ')))))));
  } else if (S.ui.dataTab === 'info') {
    body.append(h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['Column', 'Kind', 'dtype', 'Missing', 'Unique', 'Notes'].map(x => h('th', {}, x))),
      p.cols.map(c => h('tr', {}, h('td', {}, c.name), h('td', {}, h('span', { class: 'chip ' + c.kind }, c.kind)), h('td', { class: 'mono xs' }, c.dtype), h('td', { class: 'num' }, c.missing ? `${c.missing} (${(c.missing_pct * 100).toFixed(1)}%)` : '0'), h('td', { class: 'num' }, c.unique), h('td', { class: 'xs muted' }, [c.id_like && 'id-like', c.constant && 'constant', c.name === target && 'target', c.kind === 'numeric' && Math.abs(c.skew) > 1 && 'skewed'].filter(Boolean).join(', ')))))),
      h('div', { class: 'xs muted', style: 'margin-top:8px' }, `Memory: ${fmtBytes(p.memory)}`));
  } else {
    const ds = nodeOf('dataset');
    body.append(h('div', { class: 'xs muted', style: 'margin-bottom:6px' }, 'Pick the target and the columns used for training.'),
      h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['Column', 'Kind', 'Target', 'Use'].map(x => h('th', {}, x))),
        p.cols.map(c => h('tr', {}, h('td', {}, c.name), h('td', {}, h('span', { class: 'chip ' + c.kind }, c.kind)),
          h('td', {}, h('input', { type: 'radio', name: 'tgt', checked: ds?.cfg.target === c.name, 'aria-label': `Use ${c.name} as target`, onchange: () => { ds.cfg.target = c.name; ds.cfg.exclude = (ds.cfg.exclude || []).filter(x => x !== c.name); S.dirty = true; emit('pipeline'); A.refreshTargetDependent?.(); renderData(); } })),
          h('td', {}, h('input', { type: 'checkbox', disabled: ds?.cfg.target === c.name, checked: ds?.cfg.target !== c.name && !(ds?.cfg.exclude || []).includes(c.name), 'aria-label': `Use ${c.name} for training`, onchange: e => { ds.cfg.exclude = e.target.checked ? ds.cfg.exclude.filter(x => x !== c.name) : [...ds.cfg.exclude, c.name]; S.dirty = true; emit('pipeline'); } })))))));
  }
}
function bindDrop(el) {
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', e => { e.preventDefault(); el.classList.remove('over'); const f = e.dataTransfer.files?.[0]; if (f) A.loadFile(f); });
}

export function renderDocs() {
  const root = $('#docsBody'); if (!root) return;
  root.innerHTML = '';
  const input = h('input', { type: 'file', multiple: true, hidden: true, accept: '.txt,.md,.csv,.json,.py,.yaml,.yml,.html,.tsv,.log', onchange: async e => { for (const f of e.target.files) await addDoc(f); e.target.value = ''; } });
  const drop = h('div', { class: 'drop', tabindex: '0', role: 'button', onclick: () => input.click(), onkeydown: e => { if (e.key === 'Enter') input.click(); } }, h('div', { html: icon('upload', 20) }), h('div', {}, h('b', {}, 'Upload documents'), ' (txt, md, csv, json, py)'), h('div', { class: 'xs' }, 'The AI assistant reads them as context.'));
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', async e => { e.preventDefault(); drop.classList.remove('over'); for (const f of e.dataTransfer.files) await addDoc(f); });
  root.append(input, drop);
  if (S.docs.length) root.append(h('div', { style: 'display:flex;flex-direction:column;gap:6px;margin-top:10px' }, S.docs.map((d, i) => h('div', { class: 'docItem' }, h('span', { html: icon('file', 15) }), h('span', { class: 'n', title: d.name }, d.name), h('span', { class: 'xs muted' }, fmtBytes(d.size)), h('button', { class: 'ib sm', 'aria-label': 'Remove ' + d.name, html: icon('x', 13), onclick: () => { S.docs.splice(i, 1); renderDocs(); } })))));
  const ta = h('textarea', { rows: 6, maxlength: 3000, placeholder: 'Add instructions for the AI, e.g. "Focus on recall, missing a failing student is costly. Explain results simply."', 'aria-label': 'Instructions for the AI', oninput: e => { S.instructions = e.target.value; store.set('instructions', S.instructions); cnt.textContent = `${e.target.value.length}/3000`; } }, S.instructions);
  const cnt = h('div', { class: 'cnt' }, `${S.instructions.length}/3000`);
  root.append(h('div', { class: 'fld', style: 'margin-top:12px' }, h('label', { class: 'fl' }, 'Instructions for the AI'), ta, cnt));
}
async function addDoc(f) {
  if (f.size > 2e6) { toast(`${f.name} is larger than 2 MB and was skipped.`); return; }
  const text = await f.text();
  S.docs.push({ name: f.name, size: f.size, text: text.slice(0, 60000) });
  renderDocs();
  toast(`Added ${f.name}`);
}
