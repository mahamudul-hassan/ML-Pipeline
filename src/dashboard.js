// Results dashboard: every model's metrics, curves, CV folds, importances, tuning, SHAP, EDA and predictions.
import { S, emit, nodeOf, columns, colInfo } from './store.js';
import { optLabel, LOWER_BETTER, CLS_METRICS, REG_METRICS, FAMILY_COLORS } from './registry.js';
import { h, $, fmt, fmtTime, pct, esc, toast, toCSV, download } from './util.js';
import { icon } from './icons.js';
import { plot, PALETTE } from './charts.js';
import { A } from './actions.js';
import { pythonOutput } from './chat.js';

export const TABS = [['overview', 'Overview', 'home'], ['doctor', 'Diagnostics', 'alert'], ['leaderboard', 'Leaderboard', 'list'], ['model', 'Model Details', 'cpu'], ['compare', 'Compare', 'bars'], ['tuning', 'Tuning', 'sliders'], ['explain', 'Explainability', 'sparkle'], ['data', 'Data Insights', 'db'], ['predict', 'Predict', 'target'], ['code', 'Python Lab', 'code'], ['logs', 'Pipeline & Logs', 'file'], ['runs', 'Runs', 'history']];
const NEEDS_RESULTS = new Set(['overview', 'leaderboard', 'model', 'compare', 'tuning', 'explain', 'predict', 'logs']);
const short = (s, n = 28) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
const ml = m => optLabel(m);
const R = () => S.results;
const lower = m => LOWER_BETTER.has(m);
export const okModels = () => (R() ? R().ranking.map(id => R().models[id]) : []);
const allModels = () => {
  if (!R()) return [];
  const rest = Object.values(R().models).filter(m => !R().ranking.includes(m.id));
  return [...okModels(), ...rest.filter(m => m.status === 'ok'), ...rest.filter(m => m.status !== 'ok')];
};
const metricsOf = () => (R()?.task === 'regression' ? REG_METRICS : CLS_METRICS).filter(m => Object.values(R().models).some(x => x.test && x.test[m] != null));
const val = (m, split, metric) => (split === 'cv' ? m.cv?.[metric]?.mean : m[split]?.[metric]);
const className = i => (R()?.classes ? R().classes[i] : String(i));
const modelColor = (m, i) => (m.baseline ? '#8591ad' : PALETTE[i % PALETTE.length]);
const card = (title, ...kids) => h('div', { class: 'card' }, h('div', { class: 'ct' }, title), ...kids);
const chart = (cls = 'chart') => h('div', { class: cls });
const later = fn => requestAnimationFrame(() => { try { fn(); } catch (e) { console.error(e); } });

function modelSelect(current, onChange, list = allModels().filter(m => m.status === 'ok')) {
  return h('select', { style: 'width:auto;max-width:320px', 'aria-label': 'Model', onchange: e => onChange(e.target.value) }, list.map(m => h('option', { value: m.id, selected: m.id === current }, `${m.name}${m.id === R().best ? ' ★' : ''}`)));
}
function currentModel() {
  const r = R(); if (!r) return null;
  if (!S.ui.model || !r.models[S.ui.model] || r.models[S.ui.model].status !== 'ok') S.ui.model = r.best;
  return r.models[S.ui.model];
}
export function openModel(id, tab = 'model') { S.ui.model = id; S.ui.dashTab = tab; renderDashboard(); $('#dash')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }

// ---------------------------------------------------------------- shell
export function renderDashboard() {
  const tabs = $('#dashTabs'), root = $('#dashBody');
  if (!root) return;
  tabs.innerHTML = '';
  for (const [k, l, i] of TABS) tabs.append(h('button', { class: 'tab' + (S.ui.dashTab === k ? ' on' : ''), role: 'tab', 'aria-selected': S.ui.dashTab === k, onclick: () => { S.ui.dashTab = k; renderDashboard(); } }, h('span', { html: icon(i, 14) }), l));
  root.innerHTML = '';
  root.append(runBar());
  const tab = S.ui.dashTab;
  if (NEEDS_RESULTS.has(tab) && !R()) {
    root.append(h('div', { class: 'empty', style: 'margin-top:12px' }, S.running ? 'Training… results will appear here as soon as the run finishes.' : 'Run the pipeline to see the analysis of every model.', h('div', { class: 'mact', style: 'justify-content:center' }, !S.running ? h('button', { class: 'btn run', html: icon('play', 12) + ' Run pipeline', onclick: () => A.run() }) : null)));
    return;
  }
  const body = h('div', { class: 'dashBody' });
  root.append(body);
  ({ overview, doctor, code: codeTab, leaderboard, model: modelTab, compare, tuning, explain, data: dataTab, predict: predictTab, logs, runs })[tab](body);
}

function runBar() {
  if (S.running) {
    const p = S.progress || {};
    const frac = p.n ? Math.min(1, (p.i - (p.stage === 'model' ? 1 : 0)) / p.n) : 0;
    return h('div', { class: 'runBar busy' }, h('span', { class: 'runIc', html: icon('cpu', 16) }), h('div', { style: 'flex:1;min-width:0' }, h('div', { class: 'rt' }, p.message || 'Running…'), h('div', { class: 'row', style: 'margin-top:4px' }, h('div', { class: 'progress' }, h('i', { style: `width:${(frac * 100).toFixed(0)}%` })), h('span', { class: 'xs muted' }, p.n ? `${p.i}/${p.n}` : ''), h('span', { class: 'xs muted' }, fmtTime((Date.now() - S.runStart) / 1000)))),
      h('button', { class: 'btn sm danger', html: icon('stop', 11) + ' Stop', onclick: () => A.stop() }));
  }
  if (S.lastError) return h('div', { class: 'runBar bad' }, h('span', { class: 'runIc', html: icon('alert', 16) }), h('div', { style: 'flex:1;min-width:0' }, h('div', { class: 'rt' }, 'Run failed'), h('div', { class: 'rs' }, S.lastError)), h('button', { class: 'btn sm violet', html: icon('wand', 12) + ' Diagnose & fix with AI', onclick: () => A.fixWithAI('run') }), h('button', { class: 'btn sm', onclick: () => { S.ui.dashTab = 'doctor'; renderDashboard(); } }, 'Diagnostics'));
  const r = R();
  if (!r) return h('div', { class: 'runBar idle' }, h('span', { class: 'runIc', html: icon('play', 14) }), h('div', { style: 'flex:1' }, h('div', { class: 'rt' }, 'Not run yet'), h('div', { class: 'rs' }, S.engine.status === 'ready' ? 'Configure the blocks, then press Run Pipeline.' : S.engine.message)), h('button', { class: 'btn run sm', disabled: S.engine.status !== 'ready', html: icon('play', 11) + ' Run', onclick: () => A.run() }));
  const best = r.models[r.best];
  const nOk = Object.values(r.models).filter(m => m.status === 'ok').length, nErr = Object.values(r.models).length - nOk;
  return h('div', { class: 'runBar ok' }, h('span', { class: 'runIc', html: icon('check', 16, 2.4) }), h('div', { style: 'flex:1;min-width:0' },
    h('div', { class: 'rt' }, `Completed in ${fmtTime(r.duration)} · ${nOk} models trained${nErr ? `, ${nErr} failed` : ''}`),
    h('div', { class: 'rs' }, `Best: ${best.name} — ${ml(r.primary)} ${fmt(best.score)} (${best.score_source.toUpperCase()}), test ${fmt(best.test[r.primary])}${S.dirty ? ' · pipeline changed since this run' : ''}`)),
    issueButton(),
    h('button', { class: 'btn sm', onclick: () => A.askAI('Summarise the results of all models and recommend what to do next.') }, h('span', { html: icon('sparkle', 13) }), 'AI summary'));
}

// ------------------------------------------------------------- overview
function overview(body) {
  const r = R(), best = r.models[r.best], pm = r.primary;
  const all = Object.values(r.models), ok = all.filter(m => m.status === 'ok');
  const second = r.task === 'classification' ? (pm === 'roc_auc' ? 'f1' : 'roc_auc') : (pm === 'rmse' ? 'mae' : 'rmse');
  const kpi = (l, v, s) => h('div', { class: 'kpi' }, h('div', { class: 'l' }, l), h('div', { class: 'v', title: String(v) }, v), h('div', { class: 's', title: s }, s));
  body.append(h('div', { class: 'kpis' },
    kpi('Best model', best.name, `${best.family}${best.tuned_from ? ' · tuned' : ''}`),
    kpi(`${ml(pm)} (${best.score_source.toUpperCase()})`, fmt(best.score), best.cv?.[pm] ? `± ${fmt(best.cv[pm].std)} over ${best.cv[pm].folds.length} folds` : 'no cross-validation'),
    kpi(`Test ${ml(pm)}`, fmt(best.test[pm]), `${ml(second)}: ${fmt(best.test[second])}`),
    kpi('Models trained', `${ok.length}`, `${all.length - ok.length} failed · ${all.filter(m => m.tuning).length} tuned`),
    kpi('Data split', `${r.sizes.train}/${r.sizes.val}/${r.sizes.test}`, r.cv ? `${r.cv.name}, ${r.cv.splits} splits` : 'train / val / test'),
    kpi('Run time', fmtTime(r.duration), `${r.features.input.length} inputs → ${r.features.processed.length} features`)));
  const lb = chart('chart tall'), gap = chart(), ts = chart('chart short'), feat = chart('chart short');
  body.append(h('div', { class: 'grid2' }, card(`All models by ${ml(pm)} (${best.score_source.toUpperCase()}, mean ± std)`, lb), card(`Train vs validation vs test — ${ml(pm)}`, gap)));
  body.append(h('div', { class: 'grid3' }, card('Score vs training time', ts), card(`What drives ${short(best.name, 22)}`, feat), card('Insights', insights())));
  const list = [...okModels(), ...all.filter(m => m.baseline && m.status === 'ok')];
  later(() => {
    const rows = [...list].reverse();
    plot(lb, [{ type: 'bar', orientation: 'h', y: rows.map(m => short(m.name, 30)), x: rows.map(m => m.score), error_x: { type: 'data', array: rows.map(m => m.cv?.[pm]?.std || 0), visible: true, color: '#8591ad' }, marker: { color: rows.map(m => (m.id === r.best ? '#22c55e' : m.baseline ? '#8591ad' : '#3d6df5')) }, text: rows.map(m => fmt(m.score)), textposition: 'auto', hovertemplate: '%{y}: %{x:.4f}<extra></extra>' }],
      { margin: { l: 170, r: 14, t: 8, b: 36 }, xaxis: { title: ml(pm) }, height: Math.max(300, rows.length * 24 + 60) });
    lb.style.height = Math.max(300, rows.length * 24 + 60) + 'px';
    const top = okModels().slice(0, 10);
    const splits = [['train', 'Train'], ['val', 'Validation'], ['cv', 'CV'], ['test', 'Test']].filter(([s]) => top.some(m => val(m, s, pm) != null));
    plot(gap, splits.map(([s, l], i) => ({ type: 'bar', name: l, x: top.map(m => short(m.name, 16)), y: top.map(m => val(m, s, pm)), marker: { color: ['#a855f7', '#f59e0b', '#3d6df5', '#22c55e'][i] } })), { barmode: 'group', yaxis: { title: ml(pm) }, margin: { l: 50, r: 10, t: 8, b: 80 } });
    plot(ts, [{ type: 'scatter', mode: 'markers+text', x: ok.map(m => Math.max(1e-3, m.fit_time + (m.cv_time || 0))), y: ok.map(m => m.score), text: ok.map(m => short(m.name, 14)), textposition: 'top center', textfont: { size: 9 }, marker: { size: 9, color: ok.map(m => FAMILY_COLORS[m.family] || '#64748b') }, hovertemplate: '%{text}<br>%{x:.2f}s · %{y:.4f}<extra></extra>' }], { xaxis: { type: 'log', title: 'Fit + CV time (s)' }, yaxis: { title: ml(pm) } });
    const imp = best.permutation ? best.permutation.slice(0, 12).map(x => [x.feature, x.mean]) : best.shap ? best.shap.order.slice(0, 12).map(j => [best.shap.features[j], best.shap.global[j]]) : (best.native_importance?.items || []).slice(0, 12).map(x => [x.feature, x.value]);
    const kind = best.permutation ? 'Permutation importance' : best.shap ? 'mean |SHAP|' : best.native_importance?.kind || '';
    if (imp.length) plot(feat, [{ type: 'bar', orientation: 'h', y: imp.map(x => short(x[0], 22)).reverse(), x: imp.map(x => x[1]).reverse(), marker: { color: '#a855f7' } }], { margin: { l: 130, r: 10, t: 8, b: 36 }, xaxis: { title: kind } });
    else feat.innerHTML = '<div class="empty">No importance available for this model.</div>';
  });
}
function insights() {
  const r = R(), pm = r.primary, ok = okModels(), best = r.models[r.best];
  const items = [];
  if (ok[1]) items.push(`<b>${esc(best.name)}</b> leads with ${ml(pm)} ${fmt(best.score)}; runner-up <b>${esc(ok[1].name)}</b> is ${fmt(Math.abs(best.score - ok[1].score))} behind${best.cv?.[pm] && Math.abs(best.score - ok[1].score) < best.cv[pm].std ? ' — within one CV standard deviation, so the difference may not be meaningful' : ''}.`);
  const base = Object.values(r.models).find(m => m.baseline && m.status === 'ok');
  if (base) items.push(`Baseline (dummy) scores ${fmt(base.score)}; the best model improves on it by ${fmt(Math.abs(best.score - base.score))}.`);
  const over = ok.filter(m => m.overfit_gap != null).sort((a, b) => b.overfit_gap - a.overfit_gap)[0];
  if (over && over.overfit_gap > 0.05) items.push(`<b>${esc(over.name)}</b> overfits the most: train ${fmt(over.train?.[pm])} vs test ${fmt(over.test?.[pm])}.`);
  const fast = ok.filter(m => Math.abs(m.score - best.score) < 0.01 * Math.max(1, Math.abs(best.score))).sort((a, b) => a.fit_time - b.fit_time)[0];
  if (fast && fast.id !== best.id) items.push(`<b>${esc(fast.name)}</b> is almost as good and trains in ${fmtTime(fast.fit_time)} (best: ${fmtTime(best.fit_time)}).`);
  const tuned = Object.values(r.models).filter(m => m.tuning);
  for (const t of tuned.slice(0, 2)) items.push(`Tuning ${esc(t.name.replace(' (tuned)', ''))}: ${fmt(t.tuning.before)} → ${fmt(t.tuning.after)} (${t.tuning.n_trials} candidates, ${t.tuning.method}).`);
  if (best.permutation?.length) items.push(`Most important inputs for the best model: ${best.permutation.slice(0, 3).map(x => `<b>${esc(x.feature)}</b>`).join(', ')}.`);
  const errs = Object.values(r.models).filter(m => m.status !== 'ok');
  if (errs.length) items.push(`<span class="err">${errs.length} model(s) failed: ${errs.map(m => esc(m.name)).join(', ')}.</span> See Leaderboard for the errors.`);
  const warn = (r.log || []).filter(l => l.level === 'warn').length;
  if (warn) items.push(`${warn} warning(s) in the run log (Pipeline & Logs).`);
  return h('ul', { class: 'insights', html: items.map(i => `<li>${i}</li>`).join('') });
}

// ---------------------------------------------------------- leaderboard
function leaderboard(body) {
  const r = R(), mets = metricsOf();
  const split = S.ui.lbSplit || 'cv';
  const splits = [['cv', 'Cross-validation'], ['val', 'Validation'], ['test', 'Test'], ['train', 'Train']].filter(([s]) => s !== 'cv' || r.cv).filter(([s]) => s !== 'val' || r.sizes.val);
  const sp = splits.find(s => s[0] === split) ? split : splits[0][0];
  const sort = S.ui.lbSort || { key: sp + ':' + r.primary, dir: lower(r.primary) ? 1 : -1 };
  const rows = allModels();
  const getv = (m, key) => { const [s, k] = key.split(':'); if (s === 'meta') return k === 'fit' ? m.fit_time : k === 'gap' ? m.overfit_gap : k === 'name' ? m.name : k === 'std' ? m.cv?.[r.primary]?.std : null; return val(m, s, k); };
  rows.sort((a, b) => (a.status !== 'ok') - (b.status !== 'ok') || cmp(getv(a, sort.key), getv(b, sort.key)) * sort.dir);
  const th = (label, key, title) => h('th', { class: 'sortable', title: title || '', onclick: () => { S.ui.lbSort = { key, dir: sort.key === key ? -sort.dir : (key.endsWith('name') || LOWER_BETTER.has(key.split(':')[1]) || key === 'meta:fit' ? 1 : -1) }; renderDashboard(); } }, label, sort.key === key ? (sort.dir > 0 ? ' ▲' : ' ▼') : '');
  const cmpSet = new Set(S.ui.compare);
  body.append(h('div', { class: 'row', style: 'margin-bottom:8px' }, h('span', { class: 'sm muted' }, 'Scores on'), h('select', { style: 'width:auto', onchange: e => { S.ui.lbSplit = e.target.value; S.ui.lbSort = null; renderDashboard(); } }, splits.map(([s, l]) => h('option', { value: s, selected: s === sp }, l))), h('span', { class: 'sp' }),
    h('button', { class: 'btn sm', disabled: cmpSet.size < 2, onclick: () => { S.ui.dashTab = 'compare'; renderDashboard(); } }, `Compare selected (${cmpSet.size})`),
    h('button', { class: 'btn sm', html: icon('download', 13) + ' CSV', onclick: () => downloadLeaderboard() })));
  const tbl = h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, ''), h('th', {}, '#'), th('Model', 'meta:name'), h('th', {}, 'Family'), mets.map(m => th(ml(m), sp + ':' + m)), r.cv ? th(`CV std (${ml(r.primary)})`, 'meta:std') : null, th('Train−test gap', 'meta:gap', `${ml(r.primary)} on train minus test`), th('Fit time', 'meta:fit')));
  rows.forEach((m, i) => {
    if (m.status !== 'ok') { tbl.append(h('tr', { class: 'errrow' }, h('td', {}), h('td', {}, '—'), h('td', {}, m.name), h('td', {}, m.family), h('td', { colspan: mets.length + 3, style: 'white-space:normal' }, short(m.error || 'failed', 160), ' ', h('button', { class: 'btn sm violet', html: icon('wand', 12) + ' Fix with AI', onclick: () => A.fixWithAI('model', { id: m.id, name: m.name, error: m.error }) })))); return; }
    tbl.append(h('tr', { class: 'click' + (m.id === r.best ? ' hl' : ''), onclick: e => { if (e.target.closest('input')) return; openModel(m.id); } },
      h('td', {}, h('input', { type: 'checkbox', checked: cmpSet.has(m.id), 'aria-label': 'Compare ' + m.name, onchange: e => { if (e.target.checked) S.ui.compare = [...cmpSet, m.id]; else S.ui.compare = [...cmpSet].filter(x => x !== m.id); renderDashboard(); } })),
      h('td', {}, m.baseline ? '—' : r.ranking.indexOf(m.id) + 1), h('td', {}, m.name, m.tuned_from ? h('span', { class: 'chip', style: 'margin-left:6px' }, 'tuned') : null, m.ensemble ? h('span', { class: 'chip', style: 'margin-left:6px' }, 'ensemble') : null), h('td', { class: 'muted' }, m.family),
      mets.map(k => h('td', { class: 'num' }, fmt(val(m, sp, k)))), r.cv ? h('td', { class: 'num' }, fmt(m.cv?.[r.primary]?.std)) : null, h('td', { class: 'num' + (m.overfit_gap > 0.1 ? ' warnc' : '') }, fmt(m.overfit_gap)), h('td', { class: 'num' }, fmtTime(m.fit_time))));
  });
  body.append(h('div', { class: 'scrollx', style: 'max-height:560px' }, tbl), h('div', { class: 'xs muted', style: 'margin-top:6px' }, `Ranked by ${ml(r.primary)} on ${r.cv ? 'cross-validation' : r.sizes.val ? 'validation' : 'test'} data. Click a row for details; tick rows to compare.`));
}
const cmp = (a, b) => (a == null && b == null ? 0 : a == null ? 1 : b == null ? -1 : typeof a === 'string' ? a.localeCompare(b) : a - b);
export function downloadLeaderboard() { const csv = leaderboardCSV(); if (csv) download('leaderboard.csv', csv, 'text/csv'); }
export function updateRunBar() { const el = $('#dashBody > .runBar'); if (el) el.replaceWith(runBar()); }
export function leaderboardCSV() {
  const r = R(); if (!r) return '';
  const mets = metricsOf(), splits = ['cv', 'val', 'test', 'train'];
  const cols = ['rank', 'id', 'model', 'family', 'status', ...splits.flatMap(s => mets.map(m => `${s}_${m}`)), 'cv_std_' + r.primary, 'fit_time_s', 'error'];
  const rows = allModels().map(m => [r.ranking.indexOf(m.id) + 1 || '', m.id, m.name, m.family, m.status, ...splits.flatMap(s => mets.map(k => val(m, s, k) ?? '')), m.cv?.[r.primary]?.std ?? '', m.fit_time ?? '', m.error || '']);
  return toCSV(cols, rows);
}

// -------------------------------------------------------------- model tab
function modelTab(body) {
  const r = R(), m = currentModel(), pm = r.primary, mets = metricsOf();
  body.append(h('div', { class: 'row', style: 'margin-bottom:10px' }, modelSelect(m.id, id => { S.ui.model = id; renderDashboard(); }), h('span', { class: 'chip' }, m.family), m.id === r.best ? h('span', { class: 'chip okc' }, 'best') : null, h('span', { class: 'sp' }),
    h('button', { class: 'btn sm', html: icon('sparkle', 13) + ' Ask AI', onclick: () => A.askAI(`Analyse the model "${m.name}" (id ${m.id}) in detail: strengths, weaknesses, overfitting and how to improve it.`) }),
    h('button', { class: 'btn sm', html: icon('download', 13) + ' Predictions', onclick: () => A.exportThing('test_predictions', m.id) }),
    h('button', { class: 'btn sm', html: icon('cpu', 13) + ' Model file', onclick: () => A.exportThing('model_file', m.id) })));
  const cols = [['cv', 'CV mean'], ['train', 'Train'], ['val', 'Validation'], ['test', 'Test']].filter(([s]) => s === 'cv' ? m.cv : m[s]);
  const mt = h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, 'Metric'), cols.map(([, l]) => h('th', {}, l))), mets.map(k => h('tr', { class: k === pm ? 'hl' : '' }, h('td', {}, ml(k)), cols.map(([s]) => h('td', { class: 'num' }, s === 'cv' && m.cv?.[k] ? `${fmt(m.cv[k].mean)} ± ${fmt(m.cv[k].std, 3)}` : fmt(val(m, s, k)))))));
  const params = h('div', { class: 'kv' }, Object.entries(m.params || {}).flatMap(([k, v]) => [h('b', { class: 'mono' }, k), h('span', { class: 'mono' }, Array.isArray(v) ? v.join(', ') : v === null ? 'None' : String(v))]));
  const pipeBtn = h('button', { class: 'btn sm', onclick: async () => { const x = await A.engine('model_info', m.id); pipeBox.textContent = x.pipeline; pipeBox.hidden = false; } }, 'Show sklearn pipeline');
  const pipeBox = h('pre', { class: 'box', hidden: true });
  body.append(h('div', { class: 'grid2', style: 'margin-top:0' }, card('Metrics', h('div', { class: 'scrollx' }, mt), h('div', { class: 'xs muted', style: 'margin-top:6px' }, `Fit ${fmtTime(m.fit_time)} · CV ${fmtTime(m.cv_time)} · predict ${fmtTime(m.predict_time)}`)),
    card('Hyperparameters', params, m.tuning ? h('div', { class: 'note' }, `Tuned with ${m.tuning.method} search: ${fmt(m.tuning.before)} → ${fmt(m.tuning.after)}`) : null, h('div', { class: 'mact' }, pipeBtn), pipeBox)));
  const d = m.details || {};
  const grid = h('div', { class: 'grid3' });
  body.append(grid);
  const add = (title, cls = 'chart', extra = null) => { const c = chart(cls); grid.append(card(title, extra, c)); return c; };
  if (r.task === 'classification') {
    let norm = false;
    const cmBox = add('Confusion matrix (test)', 'chart', h('label', { class: 'chk xs', style: 'margin:0 0 0 auto' }, h('input', { type: 'checkbox', onchange: e => { norm = e.target.checked; drawCM(); } }), 'normalize'));
    const drawCM = () => {
      const z = d.confusion.map(row => { const s = row.reduce((a, b) => a + b, 0) || 1; return norm ? row.map(v => v / s) : row; });
      const labels = r.classes.map(c => short(String(c), 14));
      plot(cmBox, [{ type: 'heatmap', z, x: labels, y: labels, colorscale: 'Blues', showscale: false, texttemplate: norm ? '%{z:.2f}' : '%{z}', hovertemplate: 'Actual %{y}<br>Predicted %{x}: %{z}<extra></extra>' }], { xaxis: { title: 'Predicted', type: 'category' }, yaxis: { title: 'Actual', autorange: 'reversed', type: 'category' }, margin: { l: 80, r: 10, t: 8, b: 50 } });
    };
    grid.append(card('Per-class report (test)', h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['Class', 'Precision', 'Recall', 'F1', 'Support'].map(x => h('th', {}, x))), (d.report || []).map((c, i) => h('tr', {}, h('td', {}, className(i)), h('td', { class: 'num' }, fmt(c.precision)), h('td', { class: 'num' }, fmt(c.recall)), h('td', { class: 'num' }, fmt(c.f1)), h('td', { class: 'num' }, c.support)))))));
    const roc = d.roc?.length ? add('ROC curve (test)') : null, pr = d.pr?.length ? add('Precision-recall curve (test)') : null;
    const cal = d.calibration ? add('Calibration (reliability)') : null, thr = d.threshold ? add('Decision threshold (positive class: ' + className(1) + ')') : null, sh = d.score_hist ? add('Predicted probability by true class') : null;
    later(() => {
      drawCM();
      if (roc) plot(roc, [...d.roc.map((c, i) => ({ type: 'scatter', mode: 'lines', name: `${r.classes.length > 2 ? className(c.cls) + ' ' : ''}AUC ${fmt(c.auc, 3)}`, x: c.x, y: c.y, line: { width: 2 } })), { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dot', color: '#8591ad' }, showlegend: false }], { xaxis: { title: 'False positive rate', range: [0, 1] }, yaxis: { title: 'True positive rate', range: [0, 1.02] } });
      if (pr) plot(pr, d.pr.map(c => ({ type: 'scatter', mode: 'lines', name: `${r.classes.length > 2 ? className(c.cls) + ' ' : ''}AP ${fmt(c.ap, 3)}`, x: c.x, y: c.y })), { xaxis: { title: 'Recall', range: [0, 1] }, yaxis: { title: 'Precision', range: [0, 1.02] } });
      if (cal) plot(cal, [{ type: 'scatter', mode: 'lines+markers', name: 'Model', x: d.calibration.prob_pred, y: d.calibration.prob_true }, { type: 'scatter', mode: 'lines', name: 'Perfect', x: [0, 1], y: [0, 1], line: { dash: 'dot', color: '#8591ad' } }], { xaxis: { title: 'Mean predicted probability', range: [0, 1] }, yaxis: { title: 'Fraction positive', range: [0, 1] } });
      if (thr) plot(thr, ['precision', 'recall', 'f1', 'accuracy'].map(k => ({ type: 'scatter', mode: 'lines', name: ml(k), x: d.threshold.map(t => t.threshold), y: d.threshold.map(t => t[k]) })), { xaxis: { title: 'Threshold' }, yaxis: { range: [0, 1.02] } });
      if (sh) { const x = [...Array(20).keys()].map(i => (i + 0.5) / 20); plot(sh, [{ type: 'bar', name: className(0), x, y: d.score_hist.neg, opacity: 0.7 }, { type: 'bar', name: className(1), x, y: d.score_hist.pos, opacity: 0.7 }], { barmode: 'overlay', xaxis: { title: `P(${className(1)})` }, yaxis: { title: 'Rows' } }); }
    });
  } else if (d.scatter) {
    const pa = add('Predicted vs actual (test)'), rs = add('Residuals vs predicted'), rh = add('Residual distribution'), qq = add('Residual Q-Q plot');
    later(() => {
      const s = d.scatter, lo = Math.min(...s.y, ...s.pred), hi = Math.max(...s.y, ...s.pred);
      plot(pa, [{ type: 'scatter', mode: 'markers', x: s.y, y: s.pred, marker: { size: 5, opacity: 0.6 }, name: 'rows' }, { type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi], line: { dash: 'dot', color: '#8591ad' }, name: 'ideal' }], { xaxis: { title: 'Actual' }, yaxis: { title: 'Predicted' }, showlegend: false });
      plot(rs, [{ type: 'scatter', mode: 'markers', x: s.pred, y: s.res, marker: { size: 5, opacity: 0.6, color: '#a855f7' } }], { xaxis: { title: 'Predicted' }, yaxis: { title: 'Residual (actual − predicted)', zeroline: true } });
      const e = d.res_hist.edges; plot(rh, [{ type: 'bar', x: d.res_hist.counts.map((_, i) => (e[i] + e[i + 1]) / 2), y: d.res_hist.counts, marker: { color: '#22c55e' } }], { xaxis: { title: 'Residual' }, yaxis: { title: 'Rows' }, bargap: 0.02 });
      plot(qq, [{ type: 'scatter', mode: 'markers', x: d.qq.theory, y: d.qq.sample, marker: { size: 5 } }, { type: 'scatter', mode: 'lines', x: [-2.5, 2.5], y: [-2.5, 2.5], line: { dash: 'dot', color: '#8591ad' } }], { xaxis: { title: 'Normal quantiles' }, yaxis: { title: 'Standardised residuals' }, showlegend: false });
    });
  }
  if (m.cv?.[pm]) { const c = add(`CV folds — ${ml(pm)}`, 'chart short'); later(() => plot(c, [{ type: 'bar', x: m.cv[pm].folds.map((_, i) => 'Fold ' + (i + 1)), y: m.cv[pm].folds, marker: { color: '#3d6df5' } }, { type: 'scatter', mode: 'lines', x: m.cv[pm].folds.map((_, i) => 'Fold ' + (i + 1)), y: m.cv[pm].folds.map(() => m.cv[pm].mean), line: { dash: 'dash', color: '#22c55e' }, name: 'mean' }], { showlegend: false, yaxis: { title: ml(pm) } })); }
  if (m.native_importance) { const c = add(`Model importance (${m.native_importance.kind})`, 'chart'); later(() => { const it = m.native_importance.items.slice(0, 15).reverse(); plot(c, [{ type: 'bar', orientation: 'h', y: it.map(x => short(x.feature, 24)), x: it.map(x => x.value), marker: { color: '#06b6d4' } }], { margin: { l: 140, r: 10, t: 8, b: 36 } }); }); }
  const permC = chart();
  grid.append(card('Permutation importance (test)', m.permutation ? null : h('button', { class: 'btn sm', style: 'margin-left:auto', onclick: () => A.analyze('permutation', m.id) }, 'Compute'), permC));
  if (m.permutation) later(() => { const it = m.permutation.slice(0, 15).reverse(); plot(permC, [{ type: 'bar', orientation: 'h', y: it.map(x => short(x.feature, 24)), x: it.map(x => x.mean), error_x: { type: 'data', array: it.map(x => x.std) }, marker: { color: '#a855f7' } }], { margin: { l: 140, r: 10, t: 8, b: 36 }, xaxis: { title: `Drop in ${ml(pm)}` } }); });
  else permC.innerHTML = '<div class="empty">How much the test score drops when each input column is shuffled.</div>';
  const lcC = chart();
  grid.append(card('Learning curve', m.learning_curve ? null : h('button', { class: 'btn sm', style: 'margin-left:auto', onclick: () => A.analyze('learning_curve', m.id) }, 'Compute'), lcC));
  if (m.learning_curve) later(() => { const l = m.learning_curve; plot(lcC, [{ type: 'scatter', mode: 'lines+markers', name: 'Train', x: l.sizes, y: l.train_mean, error_y: { type: 'data', array: l.train_std } }, { type: 'scatter', mode: 'lines+markers', name: 'Validation', x: l.sizes, y: l.val_mean, error_y: { type: 'data', array: l.val_std } }], { xaxis: { title: 'Training rows' }, yaxis: { title: ml(pm) } }); });
  else lcC.innerHTML = '<div class="empty">Shows whether more data would help (3-fold CV at 6 training sizes).</div>';
  const feats = r.features.input;
  const pdC = chart();
  const pdSel = h('select', { style: 'width:auto;margin-left:auto', 'aria-label': 'Feature for partial dependence' }, feats.map(f => h('option', { value: f, selected: f === m._pdp?.feature }, f)));
  grid.append(card('Partial dependence + ICE', pdSel, h('button', { class: 'btn sm', onclick: async () => { m._pdp = await A.analyze('pdp', m.id, { feature: pdSel.value }, false); renderDashboard(); } }, 'Show'), pdC));
  if (m._pdp) later(() => drawPDP(pdC, m._pdp));
  else pdC.innerHTML = '<div class="empty">Pick a feature to see how the prediction changes with it.</div>';
}
function drawPDP(el, p) {
  const r = R();
  const traces = (p.ice || []).map(row => ({ type: 'scatter', mode: 'lines', x: p.values, y: row, line: { width: 0.7, color: 'rgba(133,145,173,.35)' }, showlegend: false, hoverinfo: 'skip' }));
  traces.push({ type: 'scatter', mode: 'lines+markers', x: p.values, y: p.mean, line: { width: 3, color: '#f59e0b' }, name: 'average' });
  plot(el, traces, { xaxis: { title: p.feature, type: p.kind === 'categorical' ? 'category' : 'linear' }, yaxis: { title: r.task === 'classification' ? `P(${className(p.class)})` : 'Prediction' }, showlegend: false });
}

// --------------------------------------------------------------- compare
function compare(body) {
  const r = R(), pm = r.primary, mets = metricsOf();
  const ok = allModels().filter(m => m.status === 'ok');
  let sel = S.ui.compare.filter(id => r.models[id]?.status === 'ok');
  if (sel.length < 2) sel = ok.filter(m => !m.baseline).slice(0, 6).map(m => m.id);
  const metric = S.ui.cmpMetric && mets.includes(S.ui.cmpMetric) ? S.ui.cmpMetric : pm;
  const picker = h('div', { class: 'colbox', style: 'max-height:120px;display:flex;flex-wrap:wrap;gap:4px 14px' }, ok.map(m => h('label', { class: 'chk', style: 'margin:3px 0' }, h('input', { type: 'checkbox', checked: sel.includes(m.id), onchange: e => { S.ui.compare = e.target.checked ? [...sel, m.id] : sel.filter(x => x !== m.id); renderDashboard(); } }), short(m.name, 30))));
  body.append(h('div', { class: 'row', style: 'margin-bottom:8px' }, h('span', { class: 'sm muted' }, 'Metric'), h('select', { style: 'width:auto', onchange: e => { S.ui.cmpMetric = e.target.value; renderDashboard(); } }, mets.map(k => h('option', { value: k, selected: k === metric }, ml(k))))), picker);
  const models = sel.map(id => r.models[id]);
  const bars = chart('chart'), box = chart('chart'), heat = chart('chart tall'), third = chart('chart');
  body.append(h('div', { class: 'grid2' }, card(`${ml(metric)} on every split`, bars), card(`Cross-validation spread — ${ml(pm)} (all models)`, box)));
  body.append(h('div', { class: 'grid2' }, card('All test metrics (colour = rank within column)', heat), card(r.task === 'classification' ? (r.classes.length === 2 ? 'ROC curves (test)' : 'Metric profile') : 'Absolute error distribution (test)', third)));
  later(() => {
    const splits = [['train', 'Train'], ['cv', 'CV'], ['val', 'Validation'], ['test', 'Test']].filter(([s]) => models.some(m => val(m, s, metric) != null));
    plot(bars, splits.map(([s, l]) => ({ type: 'bar', name: l, x: models.map(m => short(m.name, 16)), y: models.map(m => val(m, s, metric)) })), { barmode: 'group', yaxis: { title: ml(metric) }, margin: { l: 50, r: 10, t: 8, b: 80 } });
    const withCv = okModels().filter(m => m.cv?.[pm]);
    if (withCv.length) plot(box, withCv.map((m, i) => ({ type: 'box', name: short(m.name, 16), y: m.cv[pm].folds, boxpoints: 'all', jitter: 0.3, pointpos: 0, marker: { size: 4, color: modelColor(m, i) }, line: { color: modelColor(m, i) } })), { showlegend: false, yaxis: { title: ml(pm) }, margin: { l: 50, r: 10, t: 8, b: 80 } });
    else box.innerHTML = '<div class="empty">Cross-validation is off (Validation & CV block).</div>';
    const z = models.map(m => mets.map(k => m.test[k]));
    const rank = mets.map((k, j) => { const col = models.map(m => m.test[k]).filter(v => v != null); const lo = Math.min(...col), hi = Math.max(...col); return models.map(m => (m.test[k] == null || hi === lo ? 0.5 : lower(k) ? (hi - m.test[k]) / (hi - lo) : (m.test[k] - lo) / (hi - lo))); });
    plot(heat, [{ type: 'heatmap', z: models.map((_, i) => mets.map((_, j) => rank[j][i])), x: mets.map(ml), y: models.map(m => short(m.name, 22)), text: z.map(row => row.map(v => fmt(v, 3))), texttemplate: '%{text}', colorscale: [[0, '#7f1d1d'], [0.5, '#a16207'], [1, '#15803d']], showscale: false, hovertemplate: '%{y}<br>%{x}: %{text}<extra></extra>' }], { margin: { l: 150, r: 10, t: 8, b: 70 }, yaxis: { autorange: 'reversed' } });
    if (r.task === 'classification' && r.classes.length === 2) plot(third, [...models.filter(m => m.details?.roc?.length).map((m, i) => ({ type: 'scatter', mode: 'lines', name: `${short(m.name, 16)} ${fmt(m.details.roc[0].auc, 3)}`, x: m.details.roc[0].x, y: m.details.roc[0].y })), { type: 'scatter', mode: 'lines', x: [0, 1], y: [0, 1], line: { dash: 'dot', color: '#8591ad' }, showlegend: false }], { xaxis: { title: 'False positive rate' }, yaxis: { title: 'True positive rate' } });
    else if (r.task === 'classification') { const ks = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc'].filter(k => mets.includes(k)); plot(third, models.map(m => ({ type: 'scatterpolar', fill: 'toself', opacity: 0.55, name: short(m.name, 16), r: [...ks.map(k => m.test[k]), m.test[ks[0]]], theta: [...ks.map(ml), ml(ks[0])] })), { polar: { bgcolor: 'rgba(0,0,0,0)', radialaxis: { range: [0, 1], gridcolor: '#8591ad33' }, angularaxis: { gridcolor: '#8591ad33' } } }); }
    else plot(third, models.filter(m => m.details?.scatter).map(m => ({ type: 'box', name: short(m.name, 16), y: m.details.scatter.res.map(Math.abs), boxpoints: false })), { showlegend: false, yaxis: { title: '|residual|' }, margin: { l: 50, r: 10, t: 8, b: 80 } });
  });
}

// ---------------------------------------------------------------- tuning
function tuning(body) {
  const r = R();
  const tuned = Object.values(r.models).filter(m => m.tuning);
  if (!tuned.length) { body.append(h('div', { class: 'empty' }, 'No tuning results. Add the Hyperparameter Tuning block (and keep "Tune automatically" on) or ask the AI to tune specific models.')); return; }
  const pm = r.primary;
  body.append(h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['Model', 'Method', 'Candidates', 'Time', `Before (${ml(pm)})`, 'After', 'Change', 'Test after', 'Best parameters'].map(x => h('th', {}, x))),
    tuned.map(m => { const dlt = (m.tuning.after ?? 0) - (m.tuning.before ?? 0); const good = lower(pm) ? dlt < 0 : dlt > 0; return h('tr', { class: 'click', onclick: () => { S.ui.tuneModel = m.id; renderDashboard(); } }, h('td', {}, m.name), h('td', {}, m.tuning.method === 'optuna' ? `Optuna (${String(m.tuning.sampler || 'tpe').toUpperCase()})` : optLabel(m.tuning.method)), h('td', { class: 'num' }, m.tuning.n_trials), h('td', { class: 'num' }, fmtTime(m.tuning.time)), h('td', { class: 'num' }, fmt(m.tuning.before)), h('td', { class: 'num' }, fmt(m.tuning.after)), h('td', { class: 'num ' + (good ? 'okc' : dlt === 0 ? '' : 'err') }, (dlt > 0 ? '+' : '') + fmt(dlt)), h('td', { class: 'num' }, fmt(m.test[pm])), h('td', { class: 'mono xs', style: 'white-space:normal' }, Object.entries(m.tuning.best_params).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(',') : v}`).join('  '))); }))));
  const m = r.models[S.ui.tuneModel] && r.models[S.ui.tuneModel].tuning ? r.models[S.ui.tuneModel] : tuned[0];
  const t = m.tuning;
  const par = chart('chart tall'), trials = chart(), hist = chart(), imp = chart();
  body.append(h('div', { class: 'row', style: 'margin-top:12px' }, h('span', { class: 'sm muted' }, 'Search details for'), modelSelect(m.id, id => { S.ui.tuneModel = id; renderDashboard(); }, tuned)));
  const methodLabel = t.method === 'optuna' ? `Optuna · ${optLabel(t.sampler || 'tpe')} sampler` : optLabel(t.method);
  body.append(h('div', { class: 'note', style: 'margin-top:8px' }, `${methodLabel}: ${t.n_trials} trials${t.n_failed ? ` (${t.n_failed} failed)` : ''} in ${fmtTime(t.time)}, ${t.folds}-fold CV each. Best: ${Object.entries(t.best_params).map(([k, v]) => `${k}=${fmtParam(v)}`).join(', ')}.${t.note ? ' ' + t.note : ''}`));
  body.append(h('div', { class: 'grid2' }, card('Optimisation history (best so far)', hist), card('Hyperparameter importance', imp)));
  body.append(h('div', { class: 'grid2' }, card('Candidates (parallel coordinates, colour = score)', par), card(`Candidates ranked — CV ${ml(pm)}`, trials)));
  body.append(card('Top candidates', h('div', { class: 'scrollx', style: 'max-height:320px' }, h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, 'Rank'), h('th', {}, 'Trial'), Object.keys(t.grid).map(k => h('th', { class: 'mono' }, k)), h('th', {}, `CV ${ml(pm)}`), h('th', {}, 'Std'), h('th', {}, 'Fit time')),
    t.trials.slice(0, 25).map(tr => h('tr', { class: tr.rank === 1 ? 'hl' : '' }, h('td', {}, tr.rank), h('td', { class: 'muted' }, (tr.number ?? 0) + 1), Object.keys(t.grid).map(k => h('td', { class: 'mono' }, fmtParam(tr.params[k]))), h('td', { class: 'num' }, fmt(tr.mean)), h('td', { class: 'num' }, fmt(tr.std, 3)), h('td', { class: 'num' }, fmtTime(tr.fit_time))))))));
  later(() => {
    if (t.history?.length) plot(hist, [{ type: 'scatter', mode: 'markers', name: 'trial', x: t.history.map(x => x.number + 1), y: t.history.map(x => x.value), marker: { size: 7, color: '#3d6df5', opacity: 0.7 } }, { type: 'scatter', mode: 'lines', name: 'best so far', line: { shape: 'hv', width: 3, color: '#22c55e' }, x: t.history.map(x => x.number + 1), y: t.history.map(x => x.best) }, { type: 'scatter', mode: 'lines', name: 'before tuning', x: [1, t.history.length], y: [t.before, t.before], line: { dash: 'dash', color: '#f59e0b' } }], { xaxis: { title: 'Trial' }, yaxis: { title: `CV ${ml(pm)}` } });
    else hist.innerHTML = '<div class="empty">No history for this run.</div>';
    if (t.importance?.length) plot(imp, [{ type: 'bar', orientation: 'h', y: t.importance.map(x => x.param).reverse(), x: t.importance.map(x => x.importance).reverse(), marker: { color: '#a855f7' } }], { margin: { l: 150, r: 10, t: 8, b: 36 }, xaxis: { title: 'Importance (random forest on trials)' } });
    else imp.innerHTML = '<div class="empty">Needs at least 6 successful trials with varying scores.</div>';
    const keys = Object.keys(t.grid);
    const dims = keys.map(k => {
      const vals = t.trials.map(x => x.params[k]);
      const numeric = vals.every(v => typeof v === 'number');
      if (numeric && t.grid[k] && !Array.isArray(t.grid[k]) && t.grid[k].log) return { label: k + ' (log10)', values: vals.map(v => Math.log10(v)) };
      if (numeric) return { label: k, values: vals };
      const cats = [...new Set(vals.map(fmtParam))];
      return { label: k, values: vals.map(v => cats.indexOf(fmtParam(v))), tickvals: cats.map((_, i) => i), ticktext: cats };
    });
    const scores = t.trials.map(x => x.mean ?? NaN);
    dims.push({ label: ml(pm), values: scores });
    plot(par, [{ type: 'parcoords', line: { color: scores, colorscale: 'Viridis', showscale: true, reversescale: lower(pm) }, dimensions: dims }], { margin: { l: 60, r: 60, t: 40, b: 20 } });
    const sorted = [...t.trials].sort((a, b) => a.rank - b.rank);
    plot(trials, [{ type: 'scatter', mode: 'markers', x: sorted.map((_, i) => i + 1), y: sorted.map(x => x.mean), error_y: { type: 'data', array: sorted.map(x => x.std) }, marker: { color: '#3d6df5', size: 7 } }, { type: 'scatter', mode: 'lines', x: [1, sorted.length], y: [t.before, t.before], line: { dash: 'dash', color: '#f59e0b' }, name: 'before tuning' }], { xaxis: { title: 'Candidate (by rank)' }, yaxis: { title: ml(pm) }, showlegend: false });
  });
}
const fmtParam = v => (v === null ? 'None' : Array.isArray(v) ? v.join(',') : String(v));

// ---------------------------------------------------------- explainability
function explain(body) {
  const r = R(), m = currentModel();
  body.append(h('div', { class: 'row', style: 'margin-bottom:10px' }, modelSelect(m.id, id => { S.ui.model = id; renderDashboard(); }), h('span', { class: 'sp' }),
    h('button', { class: 'btn sm', html: icon('sparkle', 13) + (m.shap ? ' Recompute SHAP' : ' Compute SHAP'), onclick: () => A.analyze('shap', m.id) }),
    h('button', { class: 'btn sm', onclick: () => A.askAI(`Explain which features drive the predictions of "${m.name}" (id ${m.id}) and whether that makes sense.`) }, 'Ask AI to interpret')));
  if (!m.shap) body.append(h('div', { class: 'empty' }, `SHAP values show how each input pushes a prediction up or down. Press "Compute SHAP" for ${m.name} (model-agnostic permutation SHAP on test rows).`));
  else {
    const sh = m.shap, top = sh.order.slice(0, 14);
    const bar = chart('chart tall'), swarm = chart('chart tall'), dep = chart(), wf = chart('chart tall');
    const featSel = h('select', { style: 'width:auto;margin-left:auto', 'aria-label': 'Dependence feature', onchange: () => drawDep() }, sh.order.map(j => h('option', { value: j }, sh.features[j])));
    const rowSel = h('select', { style: 'width:auto;margin-left:auto', 'aria-label': 'Row to explain', onchange: () => drawWf() }, sh.rows.map((row, i) => h('option', { value: i }, `Test row ${row}`)));
    const tgt = r.task === 'classification' ? `P(${className(sh.class)})` : 'prediction';
    body.append(h('div', { class: 'grid2', style: 'margin-top:0' }, card(`Mean |SHAP| — impact on ${tgt}`, bar), card('SHAP summary (colour = feature value, low → high)', swarm)));
    body.append(h('div', { class: 'grid2' }, card('Dependence', featSel, dep), card('Why this prediction? (waterfall)', rowSel, wf)));
    const drawDep = () => { const j = +featSel.value; const raw = sh.raw_values.map(x => x[j]); const numeric = raw.every(v => v === '' || v === 'nan' || isFinite(Number(v))); plot(dep, [{ type: 'scatter', mode: 'markers', x: numeric ? raw.map(Number) : raw, y: sh.phi.map(x => x[j]), marker: { size: 7, color: sh.norm_values.map(x => x[j]), colorscale: [[0, '#3b82f6'], [1, '#ec4899']] } }], { xaxis: { title: sh.features[j], type: numeric ? 'linear' : 'category' }, yaxis: { title: 'SHAP value' } }); };
    const drawWf = () => {
      const i = +rowSel.value, phi = sh.phi[i];
      const ord = [...phi.keys()].sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a])).slice(0, 12);
      const rest = phi.reduce((a, v, j) => a + (ord.includes(j) ? 0 : v), 0);
      const labels = [...ord.map(j => `${short(sh.features[j], 18)} = ${short(String(sh.raw_values[i][j]), 10)}`), ...(ord.length < phi.length ? ['other features'] : [])].reverse();
      const vals = [...ord.map(j => phi[j]), ...(ord.length < phi.length ? [rest] : [])].reverse();
      plot(wf, [{ type: 'waterfall', orientation: 'h', base: sh.base, y: ['base value', ...labels, 'prediction'], x: [0, ...vals, 0], measure: ['absolute', ...vals.map(() => 'relative'), 'total'], text: ['', ...vals.map(v => (v > 0 ? '+' : '') + fmt(v, 3)), fmt(sh.fx[i], 3)], increasing: { marker: { color: '#ec4899' } }, decreasing: { marker: { color: '#3b82f6' } }, totals: { marker: { color: '#a855f7' } }, connector: { line: { color: '#8591ad55' } } }], { margin: { l: 190, r: 20, t: 8, b: 36 }, xaxis: { title: tgt } });
    };
    later(() => {
      plot(bar, [{ type: 'bar', orientation: 'h', y: top.map(j => short(sh.features[j], 24)).reverse(), x: top.map(j => sh.global[j]).reverse(), marker: { color: '#a855f7' } }], { margin: { l: 150, r: 10, t: 8, b: 36 } });
      const traces = top.slice().reverse().map((j, k) => ({ type: 'scatter', mode: 'markers', x: sh.phi.map(p => p[j]), y: sh.phi.map((_, i) => k + ((i * 7919) % 100 / 100 - 0.5) * 0.5), marker: { size: 5, color: sh.norm_values.map(v => v[j]), colorscale: [[0, '#3b82f6'], [1, '#ec4899']], cmin: 0, cmax: 1 }, showlegend: false, hovertemplate: `${sh.features[j]}: %{x:.3f}<extra></extra>` }));
      plot(swarm, traces, { margin: { l: 150, r: 10, t: 8, b: 36 }, xaxis: { title: 'SHAP value', zeroline: true }, yaxis: { tickvals: top.map((_, k) => k), ticktext: top.slice().reverse().map(j => short(sh.features[j], 24)), zeroline: false } });
      drawDep(); drawWf();
    });
  }
  const perm = okModels().filter(x => x.permutation);
  if (perm.length > 1) {
    const hm = chart('chart tall');
    body.append(h('div', { class: 'grid2' }, h('div', { class: 'card span2' }, h('div', { class: 'ct' }, 'Permutation importance across models'), hm)));
    later(() => {
      const feats = [...new Set(perm.flatMap(x => x.permutation.slice(0, 10).map(p => p.feature)))];
      plot(hm, [{ type: 'heatmap', z: feats.map(f => perm.map(x => x.permutation.find(p => p.feature === f)?.mean ?? 0)), x: perm.map(x => short(x.name, 16)), y: feats.map(f => short(f, 22)), colorscale: 'YlOrRd', hovertemplate: '%{y} · %{x}: %{z:.4f}<extra></extra>' }], { margin: { l: 150, r: 10, t: 8, b: 90 } });
    });
  }
}

// -------------------------------------------------------------- data tab
function dataTab(body) {
  if (!S.profile) { body.append(h('div', { class: 'empty' }, 'Load a dataset first.')); return; }
  const target = nodeOf('dataset')?.cfg.target;
  if (!S.eda || S.eda._target !== target) { body.append(h('div', { class: 'empty' }, 'Computing data insights…')); A.loadEDA(); return; }
  const E = S.eda, p = S.profile;
  const cols = Object.keys(E.columns);
  S.ui.edaCol = cols.includes(S.ui.edaCol) ? S.ui.edaCol : cols.find(c => c !== target) || cols[0];
  const rels = Object.keys(E.target?.relations || {});
  S.ui.edaRel = rels.includes(S.ui.edaRel) ? S.ui.edaRel : rels[0];
  const miss = chart(), tgt = chart(), dist = chart(), corr = chart('chart tall'), rel = chart();
  const colSel = h('select', { style: 'width:auto;margin-left:auto', onchange: e => { S.ui.edaCol = e.target.value; renderDashboard(); } }, cols.map(c => h('option', { value: c, selected: c === S.ui.edaCol }, c)));
  const relSel = h('select', { style: 'width:auto;margin-left:auto', onchange: e => { S.ui.edaRel = e.target.value; renderDashboard(); } }, rels.map(c => h('option', { value: c, selected: c === S.ui.edaRel }, c)));
  body.append(h('div', { class: 'grid3', style: 'margin-top:0' }, card('Missing values per column', miss), card(`Target: ${target}`, tgt), card('Distribution', colSel, dist)));
  body.append(h('div', { class: 'grid2' }, card('Correlation (numeric columns)', corr), card(`Feature vs target`, relSel, rel)));
  later(() => {
    const mc = Object.entries(E.missing).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
    if (mc.length) plot(miss, [{ type: 'bar', x: mc.map(x => short(x[0], 16)), y: mc.map(x => x[1] / p.rows), marker: { color: '#f59e0b' }, hovertemplate: '%{x}: %{y:.1%}<extra></extra>' }], { yaxis: { tickformat: '.0%' }, margin: { l: 50, r: 10, t: 8, b: 80 } });
    else miss.innerHTML = '<div class="empty">No missing values.</div>';
    const drawCol = (el, c) => { const d = E.columns[c]; if (!d) return; if (d.kind === 'numeric') plot(el, [{ type: 'bar', x: d.counts.map((_, i) => (d.edges[i] + d.edges[i + 1]) / 2), y: d.counts, marker: { color: '#3d6df5' } }], { bargap: 0.03, xaxis: { title: c }, yaxis: { title: 'Rows' } }); else plot(el, [{ type: 'bar', x: d.labels.map(l => short(l, 14)), y: d.counts, marker: { color: '#ec4899' } }], { xaxis: { type: 'category' }, yaxis: { title: 'Rows' }, margin: { l: 50, r: 10, t: 8, b: 70 } }); };
    if (target) drawCol(tgt, target);
    drawCol(dist, S.ui.edaCol);
    if (E.corr) plot(corr, [{ type: 'heatmap', z: E.corr.values, x: E.corr.cols.map(c => short(c, 14)), y: E.corr.cols.map(c => short(c, 14)), zmin: -1, zmax: 1, colorscale: 'RdBu', reversescale: true, hovertemplate: '%{y} × %{x}: %{z:.2f}<extra></extra>' }], { margin: { l: 110, r: 10, t: 8, b: 90 }, yaxis: { autorange: 'reversed' } });
    else corr.innerHTML = '<div class="empty">Needs at least two numeric columns.</div>';
    const rr = E.target?.relations?.[S.ui.edaRel];
    if (rr?.type === 'scatter') plot(rel, [{ type: 'scatter', mode: 'markers', x: rr.x, y: rr.y, marker: { size: 4, opacity: 0.5 } }], { xaxis: { title: S.ui.edaRel }, yaxis: { title: target } });
    else if (rr?.type === 'box') plot(rel, Object.entries(rr.groups).map(([g, q]) => ({ type: 'box', name: short(g, 14), q1: [q[1]], median: [q[2]], q3: [q[3]], lowerfence: [q[0]], upperfence: [q[4]], mean: [q[5]], x: [short(g, 14)] })), { showlegend: false, yaxis: { title: S.ui.edaRel }, xaxis: { title: target, type: 'category' } });
    else rel.innerHTML = '<div class="empty">Pick a numeric feature.</div>';
  });
}

// ------------------------------------------------------------- predict tab
function predictTab(body) {
  const r = R(), m = currentModel();
  const target = nodeOf('dataset')?.cfg.target, excl = new Set(nodeOf('dataset')?.cfg.exclude || []);
  const feats = S.profile.cols.filter(c => c.name !== target && !excl.has(c.name));
  S.ui.form ||= {};
  for (const c of feats) if (!(c.name in S.ui.form)) S.ui.form[c.name] = c.kind === 'numeric' ? (c.integer ? Math.round(c.median) : +(+c.median).toFixed(3)) : c.top?.[0]?.[0] ?? '';
  const form = h('div', { class: 'grid3', style: 'margin-top:0' });
  for (const c of feats) {
    const v = S.ui.form[c.name];
    const inp = c.kind === 'numeric' ? h('input', { type: 'number', step: 'any', value: v ?? '', onchange: e => { S.ui.form[c.name] = e.target.value === '' ? null : Number(e.target.value); } })
      : c.unique <= 30 ? h('select', { onchange: e => { S.ui.form[c.name] = e.target.value; } }, c.top.map(([k]) => h('option', { value: k, selected: String(v) === k }, k)), !c.top.some(([k]) => k === String(v)) && v != null ? h('option', { value: v, selected: true }, v) : null)
        : h('input', { type: 'text', value: v ?? '', onchange: e => { S.ui.form[c.name] = e.target.value; } });
    form.append(h('div', { class: 'fld', style: 'margin:0' }, h('label', { class: 'fl' }, c.name, h('span', { class: 'chip ' + c.kind, style: 'margin-left:6px' }, c.kind)), inp));
  }
  const out = h('div', {});
  const doPredict = async () => {
    out.innerHTML = '<div class="muted sm">Predicting…</div>';
    try {
      const res = await A.engine('predict', m.id, JSON.stringify([S.ui.form]));
      S.ui.lastPred = { model: m.name, pred: res.prediction[0], proba: res.probabilities?.[0], base: S.ui.lastPred?.baseRow === S.ui.baseRow ? S.ui.lastPred?.base : null, baseRow: S.ui.baseRow };
      if (S.ui.baseRow != null && !S.ui.lastPred.base) S.ui.lastPred.base = S.ui.basePred;
      drawOut();
    } catch (e) { out.innerHTML = `<div class="note warn">${esc(e.message)}</div>`; }
  };
  const drawOut = () => {
    out.innerHTML = '';
    const lp = S.ui.lastPred; if (!lp) return;
    out.append(h('div', { class: 'kpi', style: 'margin-top:10px' }, h('div', { class: 'l' }, `Prediction by ${lp.model}`), h('div', { class: 'v' }, typeof lp.pred === 'number' ? fmt(lp.pred) : String(lp.pred)), lp.base != null ? h('div', { class: 's' }, `Original row: ${typeof lp.base.pred === 'number' ? fmt(lp.base.pred) : lp.base.pred}${lp.base.proba && lp.proba ? ` · P(${lp.pred}) ${pct(lp.base.proba[lp.pred])} → ${pct(lp.proba[lp.pred])}` : ''}`) : null));
    if (lp.proba) { const c = chart('chart short'); out.append(c); later(() => { const ks = Object.keys(lp.proba); plot(c, [{ type: 'bar', x: ks, y: ks.map(k => lp.proba[k]), marker: { color: ks.map(k => (k === String(lp.pred) ? '#22c55e' : '#3d6df5')) }, text: ks.map(k => pct(lp.proba[k])), textposition: 'auto' }], { yaxis: { range: [0, 1], tickformat: '.0%' }, xaxis: { type: 'category' } }); }); }
  };
  body.append(h('div', { class: 'row', style: 'margin-bottom:10px' }, modelSelect(m.id, id => { S.ui.model = id; S.ui.lastPred = null; renderDashboard(); }), h('span', { class: 'sp' }),
    h('button', { class: 'btn sm', onclick: () => A.askAI(`Using model ${m.id}, predict for these values and explain the result: ${JSON.stringify(S.ui.form)}`) }, h('span', { html: icon('sparkle', 13) }), 'Explain with AI')));
  body.append(h('div', { class: 'grid2', style: 'margin-top:0' }, card('Input values (edit to run what-if scenarios)', form, h('div', { class: 'mact' }, h('button', { class: 'btn primary', onclick: doPredict }, 'Predict'), h('button', { class: 'btn', onclick: () => { S.ui.form = {}; S.ui.baseRow = null; S.ui.lastPred = null; renderDashboard(); } }, 'Reset to typical values')), out), testRowsCard(m)));
  drawOut();
  const file = h('input', { type: 'file', accept: '.csv,.txt', hidden: true, onchange: async e => { const f = e.target.files[0]; if (!f) return; try { const res = await A.engine('predict_csv', m.id, await f.text()); download(f.name.replace(/\.\w+$/, '') + '_predictions.csv', res.csv, 'text/csv'); toast(`Predicted ${res.rows} rows`); } catch (err) { toast(err.message); } e.target.value = ''; } });
  body.append(card('Batch prediction', file, h('p', { class: 'sm muted', style: 'margin:0 0 8px' }, 'Upload a CSV with the same columns as the training data (the target may be missing). You get the file back with a prediction column and class probabilities.'), h('button', { class: 'btn', html: icon('upload', 14) + ' Upload CSV to predict', onclick: () => file.click() })));
}
function testRowsCard(m) {
  const box = h('div', {});
  const c = card('Test-set rows (pick one to start a what-if)', box);
  const load = async start => {
    try {
      const d = await A.engine('test_rows', start, 8);
      box.innerHTML = '';
      box.append(h('div', { class: 'scrollx', style: 'max-height:330px' }, h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, ''), h('th', {}, '#'), d.columns.map(x => h('th', {}, x))), d.rows.map((row, i) => h('tr', {}, h('td', {}, h('button', { class: 'btn sm', onclick: async () => {
        const ob = {}; d.columns.forEach((k, j) => { ob[k] = row[j]; }); S.ui.form = ob; S.ui.baseRow = start + i;
        const res = await A.engine('predict', m.id, JSON.stringify([ob])); S.ui.basePred = { pred: res.prediction[0], proba: res.probabilities?.[0] }; S.ui.lastPred = { model: m.name, pred: res.prediction[0], proba: res.probabilities?.[0], base: S.ui.basePred, baseRow: S.ui.baseRow }; renderDashboard();
      } }, 'Use')), h('td', { class: 'muted' }, start + i), row.map(v => h('td', {}, v == null ? '' : typeof v === 'number' ? fmt(v, 3) : String(v))))))),
        h('div', { class: 'row', style: 'margin-top:6px' }, h('span', { class: 'xs muted' }, `${start + 1}–${start + d.rows.length} of ${d.total}`), h('span', { class: 'sp' }), h('button', { class: 'btn sm', disabled: start === 0, onclick: () => load(Math.max(0, start - 8)) }, 'Previous'), h('button', { class: 'btn sm', disabled: start + 8 >= d.total, onclick: () => load(start + 8) }, 'Next')));
    } catch (e) { box.innerHTML = `<div class="note warn">${esc(e.message)}</div>`; }
  };
  load(S.ui.testStart || 0);
  return c;
}

// ------------------------------------------------------------------- logs
function logs(body) {
  const r = R(), f = r.features;
  body.append(h('div', { class: 'grid3', style: 'margin-top:0' },
    card(`Input columns (${f.input.length})`, h('div', { class: 'blist' }, f.input.map(c => h('span', { class: 'bpill' }, c)))),
    card(`Features after preprocessing (${f.processed.length}, best model)`, h('div', { class: 'colbox', style: 'max-height:220px' }, f.processed.map(c => h('div', { class: 'xs mono', style: 'padding:2px 0' }, c)))),
    card('Left out and created', f.dropped.length ? h('ul', { class: 'insights' }, f.dropped.map(([c, why]) => h('li', {}, h('b', {}, c), ` — ${why}`))) : h('div', { class: 'xs muted' }, 'No columns were dropped.'), f.custom.length ? h('ul', { class: 'insights' }, f.custom.map(c => h('li', {}, h('b', {}, c.name), ' = ', h('code', {}, c.expr)))) : null)));
  body.append(h('div', { class: 'grid2' }, card('Run log', h('div', { style: 'max-height:380px;overflow:auto' }, (r.log || []).map(l => h('div', { class: 'logLine ' + l.level }, h('time', {}, new Date(l.t * 1000).toLocaleTimeString()), h('span', {}, l.message))))),
    card('scikit-learn pipeline of the best model', h('pre', { class: 'box' }, r.pipeline_text || ''))));
}

// ------------------------------------------------------------------- runs
function runs(body) {
  if (!S.history.length) { body.append(h('div', { class: 'empty' }, 'Every run is recorded here so you can compare experiments and restore an earlier pipeline.')); return; }
  const c = chart('chart short');
  body.append(h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, ['#', 'Time', 'Dataset', 'Target', 'Models', 'Best model', 'Metric', 'Score', 'Test', 'Duration', ''].map(x => h('th', {}, x))),
    S.history.map((x, i) => h('tr', {}, h('td', {}, i + 1), h('td', {}, new Date(x.time).toLocaleTimeString()), h('td', {}, x.dataset), h('td', {}, x.target), h('td', { class: 'num' }, x.models), h('td', {}, x.best), h('td', {}, ml(x.metric)), h('td', { class: 'num' }, fmt(x.score)), h('td', { class: 'num' }, fmt(x.test)), h('td', { class: 'num' }, fmtTime(x.duration)), h('td', {}, h('button', { class: 'btn sm', onclick: () => A.restoreRun(i) }, 'Restore pipeline')))))),
  card('Best score per run', c));
  later(() => plot(c, [{ type: 'scatter', mode: 'lines+markers+text', x: S.history.map((_, i) => 'Run ' + (i + 1)), y: S.history.map(x => x.score), text: S.history.map(x => short(x.best, 14)), textposition: 'top center', textfont: { size: 9 } }], { yaxis: { title: 'Best score' } }));
}

// --------------------------------------------------------- pipeline doctor
function issueButton() {
  const is = S.diag?.issues || [];
  const n = is.filter(i => i.severity !== 'info').length;
  if (!n) return null;
  const errs = is.filter(i => i.severity === 'error').length;
  return h('button', { class: 'btn sm' + (errs ? ' danger' : ''), html: icon('alert', 13) + ` ${n} issue${n > 1 ? 's' : ''}`, onclick: () => { S.ui.dashTab = 'doctor'; renderDashboard(); } });
}
export function describeFix(fix) {
  return (fix || []).map(p => {
    if (p.add_block) return `add a ${p.add_block} block`;
    if (p.remove) return `remove block ${p.block}`;
    if (p.add_exclude) return `exclude ${p.add_exclude.join(', ')} from training`;
    if (p.remove_feature) return `delete custom feature ${p.remove_feature}`;
    return `${p.block}: ${Object.entries(p.settings || {}).map(([k, v]) => `${k} = ${Array.isArray(v) ? v.join(', ') : optLabel(v)}`).join(', ')}`;
  }).join(' · ');
}
function doctor(body) {
  const dg = S.diag;
  const fixable = (dg?.issues || []).filter(i => i.fix && !dg.applied?.has(i.id));
  body.append(h('div', { class: 'row', style: 'margin-bottom:10px' },
    h('button', { class: 'btn sm', html: icon('refresh', 13) + ' Run checks', onclick: async () => { await A.diagnose(); renderDashboard(); } }),
    dg ? h('span', { class: 'xs muted' }, `Checked ${new Date(dg.time).toLocaleTimeString()}${S.results ? ' · includes results of the last run' : ''}`) : null, h('span', { class: 'sp' }),
    h('button', { class: 'btn sm', disabled: !fixable.length, onclick: () => { A.applyFixes(['all']); renderDashboard(); } }, `Apply all fixes (${fixable.length})`),
    h('button', { class: 'btn sm violet', disabled: !dg?.issues?.length, html: icon('wand', 12) + ' Fix everything with AI', onclick: () => A.fixWithAI('doctor') })));
  if (!dg) { body.append(h('div', { class: 'empty' }, 'Checking the pipeline…')); A.diagnose().then(() => { if (S.ui.dashTab === 'doctor') renderDashboard(); }); return; }
  if (dg.error) body.append(h('div', { class: 'note warn' }, dg.error));
  const count = s => dg.issues.filter(i => i.severity === s).length;
  body.append(h('div', { class: 'kpis', style: 'grid-template-columns:repeat(3,minmax(0,1fr))' },
    h('div', { class: 'kpi' }, h('div', { class: 'l' }, 'Errors'), h('div', { class: 'v err' }, count('error')), h('div', { class: 's' }, 'break the run or invalidate the results')),
    h('div', { class: 'kpi' }, h('div', { class: 'l' }, 'Warnings'), h('div', { class: 'v warnc' }, count('warning')), h('div', { class: 's' }, 'likely conceptual mistakes')),
    h('div', { class: 'kpi' }, h('div', { class: 'l' }, 'Notes'), h('div', { class: 'v' }, count('info')), h('div', { class: 's' }, 'suggestions'))));
  if (!dg.issues.length) { body.append(h('div', { class: 'empty', style: 'margin-top:10px' }, '✓ No problems found in the data, the pipeline settings or the last results.')); return; }
  const list = h('div', { style: 'display:flex;flex-direction:column;gap:8px;margin-top:10px' });
  for (const i of dg.issues) {
    const done = dg.applied?.has(i.id);
    list.append(h('div', { class: 'card issue ' + i.severity },
      h('div', { class: 'ct' }, h('span', { class: 'chip sev ' + i.severity }, i.severity), i.title, done ? h('span', { class: 'chip okc' }, 'fix applied') : null),
      h('div', { class: 'sm' }, i.detail),
      i.fix ? h('div', { class: 'xs muted', style: 'margin-top:6px' }, 'Fix: ' + describeFix(i.fix)) : null,
      h('div', { class: 'mact' }, i.fix && !done ? h('button', { class: 'btn sm primary', onclick: () => { A.applyFixes([i.id]); renderDashboard(); } }, 'Apply fix') : null,
        h('button', { class: 'btn sm', html: icon('wand', 12) + ' Fix with AI', onclick: () => A.fixWithAI('issue', { issue: i }) }),
        i.model && S.results?.models[i.model] ? h('button', { class: 'btn sm ghost', onclick: () => openModel(i.model) }, 'Open model') : null)));
  }
  body.append(list, h('div', { class: 'xs muted', style: 'margin-top:8px' }, 'After applying fixes, run the pipeline again. The checks re-run automatically after every run.'));
}

// --------------------------------------------------------------- python lab
function codeTab(body) {
  body.append(h('div', { class: 'row', style: 'margin-bottom:8px' },
    h('button', { class: 'btn sm', html: icon('plus', 13) + ' Add cell', onclick: () => { A.addCell(''); renderDashboard(); } }),
    h('button', { class: 'btn sm', html: icon('play', 11) + ' Run all', onclick: async () => { for (const c of S.cells) { const r = await A.runCell(c.id); if (r?.exception || r?.error) break; } } }),
    h('span', { class: 'sp' }),
    h('button', { class: 'btn sm violet', html: icon('wand', 12) + ' Ask AI to write code', onclick: () => { const t = prompt('What should the code do?'); if (t) A.askAI(`${t}\nWrite it as a Python Lab cell with write_code_cell, run it, and fix it until it works.`); } })));
  body.append(h('div', { class: 'note', style: 'margin:0 0 10px' }, 'Available: df, X_train, X_val, X_test, y_train, y_val, y_test, models (fitted pipelines), results, best_model_id, classes, task, register_model(name, estimator), make_preprocessor(), predict(model_id, rows), compute_metrics(y, pred), np, pd, plt. Shift+Enter runs a cell. Cells share one namespace.'));
  for (const c of S.cells) {
    const status = c.running ? 'running…' : c.out ? (c.out.exception || c.out.error ? 'error' : 'ok') : 'not run';
    const ta = h('textarea', { class: 'mono cell', rows: Math.min(24, Math.max(3, c.code.split('\n').length + 1)), spellcheck: 'false', 'aria-label': `Code of cell ${c.id}`,
      oninput: e => { c.code = e.target.value; A.saveCells(); },
      onkeydown: e => {
        if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); A.runCell(c.id); }
        if (e.key === 'Tab') { e.preventDefault(); const s = e.target.selectionStart; e.target.setRangeText('    ', s, e.target.selectionEnd, 'end'); c.code = e.target.value; A.saveCells(); }
      } }, c.code);
    const err = status === 'error', warn = c.out?.warnings?.length;
    body.append(h('div', { class: 'card', style: 'margin-bottom:10px' },
      h('div', { class: 'ct' }, h('span', { class: 'mono' }, `[${c.id}]`), h('span', { class: 'chip ' + (err ? 'sev error' : status === 'ok' ? 'okc' : '') }, status), warn ? h('span', { class: 'chip sev warning' }, `${warn} review warning${warn > 1 ? 's' : ''}`) : null, h('span', { class: 'sp' }),
        h('button', { class: 'btn sm run', disabled: c.running, html: icon('play', 10) + ' Run', onclick: () => A.runCell(c.id) }),
        err || warn ? h('button', { class: 'btn sm violet', html: icon('wand', 12) + ' Fix with AI', onclick: () => A.fixWithAI('cell', { id: c.id }) }) : null,
        h('button', { class: 'btn sm', onclick: () => A.fixWithAI('review', { id: c.id }) }, 'Review with AI'),
        h('button', { class: 'ib sm', 'aria-label': 'Delete cell', html: icon('trash', 14), onclick: () => { S.cells = S.cells.filter(x => x !== c); A.saveCells(); renderDashboard(); } })),
      ta, c.out ? h('div', { style: 'margin-top:8px' }, pythonOutput(c.out)) : null));
  }
}
