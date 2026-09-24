// App entry: boots the Python engine, wires UI modules, runs pipelines and exposes the agent API.
import { S, emit, on, node, nodeOf, columns, store } from './store.js';
import { BLOCKS, MODEL, MODELS, optLabel, sanitizeBlock, sanitizeModelParams, modelSpec, modelDefaults, blockDefaults, gridForEngine } from './registry.js';
import { $, $$, h, toast, modal, download, loadScript, b64ToBytes, fmt, fmtTime, esc, debounce, toCSV } from './util.js';
import { icon, paintIcons } from './icons.js';
import { Engine } from './engine-client.js';
import * as C from './canvas.js';
import { render as renderConfig, refreshBlockChat } from './config-panel.js';
import { renderData, renderDocs, SAMPLES } from './data-panel.js';
import { renderDashboard, updateRunBar, leaderboardCSV, openModel, describeFix } from './dashboard.js';
import { ChatView, newThread, clearThread } from './chat.js';
import { AI, saveAI, listModels, chat, FALLBACK_MODELS, setAgentAPI, resolveBlock, leaderboard, aiReady } from './ai.js';
import { A } from './actions.js';

Object.assign(S.settings, store.get('settings', {}));
S.instructions = store.get('instructions', '');
S.chat = newThread();
S.liveLog = [];
S.autoFixCount = 0;
const saved = store.get('state', null);
const short = (s, n = 18) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

// ------------------------------------------------------------------ engine
const engine = new Engine({ transport: window.__ENGINE_TRANSPORT__ || null, onStatus, onProgress });
A.engine = (fn, ...args) => engine.call(fn, ...args);
let firstReady = true;
function onStatus(m) {
  S.engine.status = m.status;
  S.engine.message = m.message;
  if (m.status === 'ready') {
    S.engine.versions = m.versions;
    S.engine.available = { xgboost: !!m.optional?.xgboost, lightgbm: !!m.optional?.lightgbm };
    if (firstReady) {
      firstReady = false;
      if (saved?.source?.type === 'sample') loadSample(saved.source.key, { restore: saved.pipeline, quiet: true });
      else loadSample('student_performance', { quiet: true, restore: saved?.pipeline && saved.dataName === 'student_performance.csv' ? saved.pipeline : null });
    } else if (S.dataSource) reloadData();
  }
  renderEngine();
  renderData();
  renderDashboard();
  C.renderNodes();
}
let uiTimer = 0;
function scheduleRunUI() {
  if (uiTimer) return;
  uiTimer = setTimeout(() => { uiTimer = 0; updateRunBar(); C.renderNodes(); renderRunStatus(); }, 200);
}
function markStage(types, status) { for (const n of S.nodes) if (types.includes(n.type) && n.status !== 'skipped') n.status = status; }
function onProgress(p) {
  if (!S.running) return;
  if (p.stage === 'log') { S.liveLog.push(p); return; }
  S.progress = p;
  if (p.stage === 'prepare') markStage(['split', 'preprocess', 'fe', 'fs'], 'running');
  if (p.stage === 'model') {
    markStage(['split', 'preprocess', 'fe', 'fs'], 'done');
    const n = C.nodeForModelId(p.model);
    if (n) n.status = 'running';
  }
  if (p.stage === 'model_done') { const n = C.nodeForModelId(p.model); if (n && n.type !== 'zoo') n.status = p.status === 'ok' ? 'done' : 'error'; }
  if (p.stage === 'tune') markStage(['tuning'], 'running');
  if (p.stage === 'explain') markStage(['eval'], 'running');
  scheduleRunUI();
}

// -------------------------------------------------------------------- data
async function loadFile(file) {
  if (S.engine.status !== 'ready') { toast('The Python engine is still loading. Try again in a moment.'); return; }
  const ext = file.name.split('.').pop().toLowerCase();
  let text;
  try {
    if (ext === 'xlsx' || ext === 'xls') {
      if (!window.XLSX) await loadScript('https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js');
      const wb = window.XLSX.read(await file.arrayBuffer(), { type: 'array' });
      text = window.XLSX.utils.sheet_to_csv(wb.Sheets[wb.SheetNames[0]]);
    } else if (ext === 'json') {
      let rows = JSON.parse(await file.text());
      if (!Array.isArray(rows)) rows = rows.data || rows.rows || Object.values(rows)[0];
      if (!Array.isArray(rows) || !rows.length || typeof rows[0] !== 'object') throw new Error('JSON must be an array of row objects.');
      const cols = [...new Set(rows.flatMap(r => Object.keys(r)))];
      text = toCSV(cols, rows.map(r => cols.map(c => r[c])));
    } else text = await file.text();
    toast(`Reading ${file.name}…`);
    const res = await engine.call('load_csv', text, file.name);
    afterLoad(res, { type: 'text', text: text.length < 30e6 ? text : null, name: file.name }, file.size);
    toast(`Loaded ${file.name}: ${res.profile.rows.toLocaleString()} rows × ${res.profile.cols.length} columns`);
  } catch (e) { toast('Could not read the file: ' + e.message, null, 6000); }
}
async function loadSample(key, opts = {}) {
  if (S.engine.status !== 'ready') { toast('The Python engine is still loading.'); return; }
  try {
    const res = await engine.call('load_sample', key);
    afterLoad(res, { type: 'sample', key }, 0, opts);
    if (!opts.quiet) toast(`Loaded sample: ${SAMPLES.find(s => s.key === key)?.name || key}`);
  } catch (e) { toast(e.message); }
}
async function reloadData() {
  const src = S.dataSource;
  try {
    const res = src.type === 'sample' ? await engine.call('load_sample', src.key) : src.text ? await engine.call('load_csv', src.text, src.name) : null;
    if (res) { S.profile = res.profile; toast('Engine restarted and the data was reloaded.'); }
  } catch (e) { toast('Reloading data failed: ' + e.message); }
}
function taskFromProfile(p) {
  const c = p.cols.find(x => x.name === p.target_guess);
  if (!c || c.kind !== 'numeric') return 'classification';
  return c.unique <= 20 && c.integer ? 'classification' : 'regression';
}
function afterLoad(res, source, size, opts = {}) {
  S.profile = res.profile; S.dataName = res.name; S.dataSize = size || 0; S.dataSource = source;
  S.results = null; S.preview = null; S.eda = null; S.lastError = null;
  S.ui.form = null; S.ui.lastPred = null; S.ui.model = null; S.ui.compare = [];
  if (opts.restore) { restorePipeline(opts.restore); S.customized = true; }
  else if (!S.nodes.length || !S.customized) C.defaultPipeline(taskFromProfile(res.profile));
  else {
    const ds = nodeOf('dataset'), cols = columns();
    if (ds) {
      if (!cols.includes(ds.cfg.target)) ds.cfg.target = res.profile.target_guess;
      ds.cfg.exclude = [...new Set([...(ds.cfg.exclude || []).filter(c => cols.includes(c)), ...res.profile.cols.filter(c => c.id_like).map(c => c.name)])].filter(c => c !== ds.cfg.target);
    }
    for (const n of S.nodes) { n.status = 'idle'; n.metric = ''; }
    adaptToTask(taskFromProfile(res.profile));
  }
  if (!S.sel) S.sel = nodeOf('dataset')?.id;
  autosave();
  renderAll();
}
// When new data changes the task, swap task-specific models to their counterpart (e.g. SVC ↔ SVR).
const COUNTERPART = { logreg: 'ridge', ridge_clf: 'ridge', sgd_clf: 'sgd_reg', pa_clf: 'pa_reg', svc: 'svr', nusvc: 'nusvr', linear_svc: 'linear_svr', gp_clf: 'gp_reg', ridge: 'logreg', linreg: 'logreg', lasso: 'logreg', elasticnet: 'logreg', svr: 'svc', nusvr: 'nusvc', linear_svr: 'linear_svc', gp_reg: 'gp_clf', sgd_reg: 'sgd_clf', pa_reg: 'pa_clf' };
function adaptToTask(task) {
  const swapped = [];
  for (const n of S.nodes) {
    if (n.type === 'model' && MODEL[n.key] && !MODEL[n.key].tasks.includes(task)) {
      const to = COUNTERPART[n.key];
      if (to && MODEL[to].tasks.includes(task) && !S.nodes.some(x => x.type === 'model' && x.key === to)) { swapped.push(`${MODEL[n.key].name} → ${MODEL[to].name}`); n.key = to; n.cfg = modelDefaults(to, task); n.grid = ''; }
    }
    if (n.type === 'zoo') n.cfg.models = (n.cfg.models || []).filter(k => MODEL[k]?.tasks.includes(task)).length ? n.cfg.models.filter(k => MODEL[k]?.tasks.includes(task)) : MODELS.filter(m => m.tasks.includes(task) && !(m.flags || {}).slow && C.modelAvailable(m.key)).map(m => m.key);
    if (n.type === 'eval' && n.cfg.primary_metric !== 'auto') n.cfg.primary_metric = 'auto';
  }
  if (swapped.length) toast(`Switched models for ${task}: ${swapped.join(', ')}`, null, 5000);
}
async function loadPreview(start) {
  try { const p = await engine.call('preview', start, 50); S.preview = { ...p, start }; } catch (e) { S.preview = { columns: [], rows: [], total: 0, start: 0 }; toast(e.message); }
  renderData();
}
let edaBusy = false;
async function loadEDA() {
  if (edaBusy) return; edaBusy = true;
  const target = nodeOf('dataset')?.cfg.target;
  try { S.eda = await engine.call('eda', target || ''); S.eda._target = target; } catch (e) { toast(e.message); S.eda = { columns: {}, missing: {}, _target: target }; }
  edaBusy = false;
  if (S.ui.dashTab === 'data') renderDashboard();
}
function refreshTargetDependent() { S.eda = null; S.ui.form = null; renderData(); if (S.ui.dashTab === 'data' || S.ui.dashTab === 'predict') renderDashboard(); C.renderNodes(); }

// --------------------------------------------------------------- run config
export function buildConfig() {
  const ds = nodeOf('dataset');
  if (!ds) throw new Error('Add a Dataset block first.');
  if (!S.profile) throw new Error('Load a dataset first.');
  if (!ds.cfg.target) throw new Error('Choose a target column in the Dataset block.');
  const reach = C.reachable(ds.id);
  const inP = t => S.nodes.find(n => n.type === t && reach.has(n.id));
  const task = C.taskGuess();
  const sp = { ...blockDefaults('split'), ...(inP('split')?.cfg || {}) };
  const models = [], ensembles = [];
  for (const n of S.nodes.filter(x => reach.has(x.id))) {
    if (n.type === 'model' && MODEL[n.key] && C.modelAvailable(n.key)) {
      const spec = modelSpec(n.id, n.key, sanitizeModelParams(n.key, n.cfg, task), task);
      if (n.grid) { try { spec.grid = gridForEngine(n.key, JSON.parse(n.grid)); } catch { /* keep default */ } }
      models.push(spec);
    } else if (n.type === 'zoo') {
      for (const k of n.cfg.models || []) if (MODEL[k] && C.modelAvailable(k)) models.push(modelSpec(`${n.id}__${k}`, k, modelDefaults(k, task), task));
    } else if (n.type === 'voting' || n.type === 'stacking') {
      ensembles.push({ id: n.id, key: n.type, name: BLOCKS[n.type].t, top_k: n.cfg.top_k, params: n.type === 'voting' ? { voting: n.cfg.voting } : { cv: n.cfg.cv } });
    }
  }
  const t = inP('tuning');
  const ev = inP('eval');
  return {
    dataset: { target: ds.cfg.target, task: ds.cfg.task, exclude: ds.cfg.exclude || [], group_col: ds.cfg.group_col || '', time_col: ds.cfg.time_col || '' },
    split: { test_size: sp.test_size, val_size: sp.val_size, stratify: sp.stratify, shuffle: sp.shuffle, seed: sp.seed, refit_train_val: sp.refit_train_val },
    cv: { strategy: inP('split') ? sp.cv_strategy : 'stratified_kfold', folds: sp.folds, repeats: sp.repeats },
    preprocess: inP('preprocess')?.cfg || { num_impute: 'median', encoding: 'onehot', scaling: 'standard' },
    fe: inP('fe')?.cfg || { date_parts: true },
    fs: inP('fs')?.cfg || {},
    models, ensembles,
    eval: ev ? ev.cfg : { primary_metric: 'auto', baseline: true, permutation: 'best', shap: 'none', learning_curve: 'none' },
    tuning: t ? { ...t.cfg, grids: t.cfg.grids || {} } : { enabled: false },
    advanced: { max_train_rows: +S.settings.maxTrainRows || 0, slow_model_rows: +S.settings.slowModelRows || 3000 },
  };
}
async function run(opts = {}) {
  if (S.running) return { error: 'A run is already in progress.' };
  if (opts.source !== 'agent' && opts.source !== 'auto') S.autoFixCount = 0;
  if (S.engine.status !== 'ready') { toast('The Python engine is still loading.'); return { error: 'The Python engine is still loading.' }; }
  let cfg;
  try { cfg = buildConfig(); } catch (e) { toast(e.message); return { error: e.message }; }
  if (!cfg.models.length) { const m = 'Connect at least one model block (or a Model Zoo) to the pipeline.'; toast(m); return { error: m }; }
  const reach = C.reachable(nodeOf('dataset').id);
  for (const n of S.nodes) { n.status = reach.has(n.id) ? 'pending' : 'skipped'; n.metric = ''; n.statusText = ''; }
  nodeOf('dataset').status = 'done';
  S.running = true; S.runStart = Date.now(); S.progress = { message: 'Preparing data…' }; S.lastError = null; S.liveLog = [];
S.autoFixCount = 0;
  renderAll();
  try {
    const res = await engine.call('run', JSON.stringify(cfg));
    S.results = res; S.dirty = false;
    applyStatuses(res, reach);
    const best = res.models[res.best];
    S.history.push({ time: Date.now(), dataset: S.dataName, target: cfg.dataset.target, task: res.task, models: Object.values(res.models).filter(m => m.status === 'ok').length, best: best.name, metric: res.primary, score: best.score, test: best.test[res.primary], duration: res.duration, pipeline: snapshot() });
    S.ui.model = res.best; S.ui.compare = []; S.ui.lastPred = null; S.ui.form = null;
    toast(`Done in ${fmtTime(res.duration)} — best: ${best.name} (${optLabel(res.primary)} ${fmt(best.score)})`);
    diagnose().then(() => setTimeout(() => maybeAutoFix(opts), 50));
    return leaderboard();
  } catch (e) {
    const stopped = e.message === 'Stopped';
    S.lastError = stopped ? null : e.message;
    for (const n of S.nodes) if (n.status === 'pending' || n.status === 'running') n.status = stopped ? 'idle' : 'error';
    if (!stopped) { toast('Run failed: ' + e.message, null, 6000); diagnose().then(() => setTimeout(() => maybeAutoFix(opts), 50)); }
    return { error: stopped ? 'Stopped by the user.' : e.message };
  } finally { S.running = false; S.progress = null; renderAll(); }
}
function applyStatuses(res, reach) {
  const pm = res.primary;
  for (const n of S.nodes) {
    if (!reach.has(n.id)) continue;
    if (n.type === 'model' || n.type === 'voting' || n.type === 'stacking') {
      const r = res.models[n.id];
      if (!r) { n.status = 'skipped'; n.statusText = 'Not trained (wrong task or unavailable)'; continue; }
      if (r.status !== 'ok') { n.status = 'error'; n.statusText = r.error; n.metric = 'failed'; continue; }
      n.status = 'done';
      const tr = res.models[n.id + '__tuned'];
      n.metric = `${optLabel(pm)} ${fmt(r.score, 3)}${tr?.status === 'ok' ? ` → ${fmt(tr.score, 3)}` : ''}${res.best === n.id || res.best === n.id + '__tuned' ? ' ★' : ''}`;
    } else if (n.type === 'zoo') {
      const mine = Object.values(res.models).filter(m => m.id.startsWith(n.id + '__') && m.status === 'ok');
      if (!mine.length) { n.status = 'error'; n.metric = 'failed'; continue; }
      const top = mine.sort((a, b) => res.ranking.indexOf(a.id) - res.ranking.indexOf(b.id))[0];
      n.status = 'done'; n.metric = `${mine.length} ok · best ${short(top.name, 12)} ${fmt(top.score, 3)}`;
    } else n.status = 'done';
  }
}
function stop() {
  if (!S.running) return;
  engine.restart();
  S.engine.status = 'loading'; S.engine.message = 'Restarting the Python engine…';
  S.results = null;
  toast('Stopped. The Python engine is restarting (trained models from this session are cleared).', null, 5000);
  renderEngine();
}

// ---------------------------------------------------------------- analysis
async function analyze(kind, id, params = {}, rerender = true) {
  try {
    toast(`Computing ${kind.replace('_', ' ')}…`, null, 1800);
    const out = await engine.call('analyze', kind, id, JSON.stringify(params));
    const m = S.results?.models[id];
    if (m && ['shap', 'permutation', 'learning_curve'].includes(kind)) m[kind] = out;
    if (rerender) renderDashboard();
    return out;
  } catch (e) { toast(e.message, null, 5000); throw e; }
}

// ------------------------------------------------------------------ exports
const TRAIN_PY = `"""Re-train the ML Agent pipeline with native scikit-learn and save the best model.

    python train.py [data.csv]
"""
import base64
import json
import sys

import pandas as pd

import engine


def main():
    cfg = json.load(open("pipeline_config.json"))
    data = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    info = json.loads(engine.set_frame(pd.read_csv(data), data))
    print(f"Loaded {info['profile']['rows']} rows from {data}")
    engine.set_progress(lambda m: print("  " + json.loads(m)["message"]) if json.loads(m)["stage"] == "log" else None)
    res = json.loads(engine.run(json.dumps(cfg)))
    if "error" in res:
        sys.exit(res["error"])
    pm = res["primary"]
    print(f"\\n{'model':42s} {pm + ' (rank)':>16s} {'test ' + pm:>14s}")
    for mid in res["ranking"]:
        m = res["models"][mid]
        print(f"{m['name'][:42]:42s} {m['score']:16.4f} {(m['test'].get(pm) or float('nan')):14.4f}")
    art = json.loads(engine.export_model(res["best"]))
    with open("model.joblib", "wb") as f:
        f.write(base64.b64decode(art["b64"]))
    print(f"\\nSaved the best model ({art['name']}) to model.joblib")


if __name__ == "__main__":
    main()
`;
const SERVE_PY = `"""REST API for the exported model.

    uvicorn serve:app --reload
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '[{"feature": 1}]'
"""
from typing import Any, Dict, List, Union

from fastapi import FastAPI

import engine

art = engine.load_artifact("model.joblib")
app = FastAPI(title=f"ML Agent model: {art['name']}")


@app.get("/")
def info():
    return {"model": art["name"], "task": art["task"], "classes": art["classes"], "columns": art["features"], "trained_with_sklearn": art["sklearn"]}


@app.post("/predict")
def predict(rows: Union[Dict[str, Any], List[Dict[str, Any]]]):
    return engine.predict_with_artifact(art, rows)
`;
const PREDICT_PY = `"""Batch predictions from a CSV file: python predict.py input.csv [output.csv]"""
import sys

import pandas as pd

import engine

art = engine.load_artifact("model.joblib")
df = pd.read_csv(sys.argv[1])
out = engine.predict_with_artifact(art, df.to_dict("records"))
df["prediction"] = out["prediction"]
for c in (out.get("probabilities") or [{}])[0]:
    df[f"prob_{c}"] = [p[c] for p in out["probabilities"]]
dest = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
df.to_csv(dest, index=False)
print(f"Wrote {len(df)} predictions to {dest}")
`;
function requirements() {
  const v = S.engine.versions || {};
  return [`scikit-learn==${v.sklearn || '1.6.1'}`, `numpy`, `pandas`, `scipy`, `joblib`, v.xgboost ? `xgboost==${v.xgboost}` : 'xgboost', v.lightgbm ? `lightgbm==${v.lightgbm}` : 'lightgbm', 'matplotlib', 'optuna', 'fastapi', 'uvicorn[standard]'].join('\n') + '\n';
}
function exportReadme() {
  const R = S.results, best = R?.models[R.best];
  return `# ML Agent export — ${S.dataName}\n\n${best ? `Best model: **${best.name}** (${optLabel(R.primary)} ${fmt(best.score)} ${best.score_source.toUpperCase()}, test ${fmt(best.test[R.primary])}).\n\n` : ''}Files:\n\n- \`engine.py\` – the same scikit-learn engine that ran in the browser\n- \`pipeline_config.json\` – the pipeline you built on the canvas\n- \`train.py\` – re-train everything natively: \`python train.py data.csv\`\n- \`model.joblib\` – the fitted best pipeline (load with \`engine.load_artifact\`)\n- \`serve.py\` – REST API: \`uvicorn serve:app\` then POST rows to \`/predict\`\n- \`predict.py\` – batch predictions: \`python predict.py new.csv\`\n- \`leaderboard.csv\`, \`report.json\` – results of every model\n\n\`\`\`bash\npython -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt\npython train.py data.csv\nuvicorn serve:app --reload\n\`\`\`\n\n\`model.joblib\` was pickled with scikit-learn ${S.engine.versions?.sklearn || ''}; install the pinned version to load it, or run \`train.py\` to rebuild it with any recent version.\n`;
}
function slimReport(R) {
  const models = {};
  for (const [id, m] of Object.entries(R.models)) models[id] = { name: m.name, family: m.family, status: m.status, error: m.error, params: m.params, score: m.score, cv: m.cv && Object.fromEntries(Object.entries(m.cv).map(([k, v]) => [k, { mean: v.mean, std: v.std }])), train: m.train, val: m.val, test: m.test, fit_time: m.fit_time, tuning: m.tuning && { method: m.tuning.method, best_params: m.tuning.best_params, before: m.tuning.before, after: m.tuning.after }, permutation: m.permutation?.slice(0, 20) };
  return { dataset: S.dataName, task: R.task, classes: R.classes, primary: R.primary, best: R.best, sizes: R.sizes, cv: R.cv, features: R.features, models };
}
function pipelineJSON() { return JSON.stringify({ app: 'ml-agent', version: 2, dataName: S.dataName, source: S.dataSource?.type === 'sample' ? S.dataSource : null, ...snapshot(), instructions: S.instructions }, null, 1); }
async function exportThing(what, modelId) {
  const R = S.results;
  const need = () => { if (!R) throw new Error('Run the pipeline first.'); };
  try {
    if (what === 'pipeline_json') { download('ml-agent-pipeline.json', pipelineJSON(), 'application/json'); return { ok: true, file: 'ml-agent-pipeline.json' }; }
    if (what === 'leaderboard_csv') { need(); download('leaderboard.csv', leaderboardCSV(), 'text/csv'); return { ok: true, file: 'leaderboard.csv' }; }
    if (what === 'test_predictions') { need(); const id = modelId || R.best; const r = await engine.call('test_predictions', id); download(`test_predictions_${id}.csv`, r.csv, 'text/csv'); return { ok: true, file: `test_predictions_${id}.csv` }; }
    if (what === 'model_file') { need(); const id = modelId || R.best; const r = await engine.call('export_model', id); download(`${id}.joblib`, new Blob([b64ToBytes(r.b64)])); return { ok: true, file: `${id}.joblib`, model: r.name }; }
    if (what === 'project_zip') {
      toast('Building the project zip…');
      if (!window.JSZip) await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
      const zip = new window.JSZip();
      zip.file('engine.py', await (await fetch(new URL('./engine.py', import.meta.url))).text());
      zip.file('pipeline_config.json', JSON.stringify(buildConfig(), null, 2));
      zip.file('train.py', TRAIN_PY); zip.file('serve.py', SERVE_PY); zip.file('predict.py', PREDICT_PY);
      zip.file('requirements.txt', requirements()); zip.file('README.md', exportReadme());
      zip.file('ml-agent-pipeline.json', pipelineJSON());
      const data = await engine.call('frame_csv');
      zip.file('data.csv', data.csv);
      if (R) {
        const art = await engine.call('export_model', modelId || R.best);
        zip.file('model.joblib', b64ToBytes(art.b64));
        zip.file('leaderboard.csv', leaderboardCSV());
        zip.file('report.json', JSON.stringify(slimReport(R), null, 1));
      }
      const blob = await zip.generateAsync({ type: 'blob' });
      download('ml-agent-project.zip', blob);
      return { ok: true, file: 'ml-agent-project.zip' };
    }
    throw new Error('Unknown export ' + what);
  } catch (e) { toast(e.message, null, 5000); return { error: e.message }; }
}

// --------------------------------------------------------- save / restore
function snapshot() {
  return JSON.parse(JSON.stringify({ nodes: S.nodes.map(({ chat: _c, status: _s, metric: _m, statusText: _t, ...n }) => n), edges: S.edges }));
}
function restorePipeline(p) {
  if (!p?.nodes?.length) return false;
  S.nodes = p.nodes.filter(n => BLOCKS[n.type] && (n.type !== 'model' || MODEL[n.key])).map(n => ({ status: 'idle', metric: '', notes: '', mode: 'manual', ...n, chat: newThread() }));
  const ids = new Set(S.nodes.map(n => n.id));
  S.edges = (p.edges || []).filter(e => ids.has(e.from) && ids.has(e.to));
  C.setSeq(); C.tidy();
  S.sel = null;
  const ds = nodeOf('dataset'), cols = columns();
  if (ds && cols.length && !cols.includes(ds.cfg.target)) ds.cfg.target = S.profile.target_guess;
  return true;
}
const autosave = debounce(() => { store.set('state', { pipeline: snapshot(), source: S.dataSource?.type === 'sample' ? S.dataSource : null, dataName: S.dataName }); }, 600);
function loadPipelineFile(file) {
  file.text().then(t => {
    const p = JSON.parse(t);
    if (!restorePipeline(p)) throw new Error('No pipeline found in this file.');
    if (p.instructions) S.instructions = p.instructions;
    S.customized = true; S.dirty = true; autosave(); renderAll();
    toast('Pipeline loaded' + (p.dataName && p.dataName !== S.dataName ? ` (it was built for ${p.dataName})` : ''));
  }).catch(e => toast('Could not load pipeline: ' + e.message));
}
function restoreRun(i) { const r = S.history[i]; if (!r) return; restorePipeline(r.pipeline); S.customized = true; S.dirty = true; autosave(); renderAll(); toast(`Restored the pipeline of run ${i + 1}`); }

const TEMPLATES = [
  { key: 'default', name: 'Standard comparison', d: '5 popular models, CV, tuning of the top 3' },
  { key: 'zoo', name: 'Full model zoo + ensembles', d: 'Every fast model for the task, voting and stacking' },
  { key: 'boosting', name: 'Gradient boosting focus', d: 'HistGB, XGBoost, LightGBM, GB, RF + stacking, deeper tuning' },
  { key: 'linear', name: 'Interpretable linear models', d: 'Regularised linear models, SHAP and permutation importance' },
  { key: 'quick', name: 'Quick baseline', d: '3 models, 3-fold CV, no tuning — fastest' },
];
function applyTemplate(key) {
  const task = C.taskGuess(), ok = k => MODEL[k] && MODEL[k].tasks.includes(task) && C.modelAvailable(k);
  const m = keys => keys.filter(ok).map(k => ({ type: 'model', key: k }));
  const base = ['dataset', 'split', 'preprocess', 'fe', 'fs'];
  const cls = task === 'classification';
  if (key === 'default') C.defaultPipeline(task);
  else if (key === 'zoo') C.buildPipeline([...base, 'zoo', 'voting', 'stacking', 'eval', 'tuning', 'deploy']);
  else if (key === 'boosting') { C.buildPipeline([...base, ...m(['hgb', 'xgb', 'lgbm', 'gb', 'rf']), 'stacking', 'eval', { type: 'tuning', cfg: { n_iter: 20, models: 'top3' } }, 'deploy']); }
  else if (key === 'linear') C.buildPipeline([...base, ...m(cls ? ['logreg', 'ridge_clf', 'lda', 'linear_svc', 'sgd_clf'] : ['linreg', 'ridge', 'lasso', 'elasticnet', 'bayes_ridge', 'huber']), { type: 'eval', cfg: { permutation: 'all', shap: 'top3' } }, 'deploy']);
  else if (key === 'quick') C.buildPipeline(['dataset', { type: 'split', cfg: { folds: 3 } }, 'preprocess', ...m(cls ? ['logreg', 'rf', 'hgb'] : ['ridge', 'rf', 'hgb']), { type: 'eval', cfg: { permutation: 'best', shap: 'none', learning_curve: 'none' } }, 'deploy']);
  S.customized = true; S.dirty = true; S.sel = nodeOf('dataset')?.id;
  autosave(); renderAll();
  toast(`Pipeline created: ${TEMPLATES.find(t => t.key === key).name}`);
}
function templatesModal() {
  const md = modal('New pipeline from a template', '');
  for (const t of TEMPLATES) md.el.append(h('div', { class: 'tplCard' }, h('span', { class: 'fic', html: icon('layout', 18) }), h('div', {}, h('b', {}, t.name), h('span', {}, t.d)), h('button', { class: 'btn sm primary', onclick: () => { applyTemplate(t.key); md.close(); } }, 'Use')));
  md.el.append(h('p', { class: 'xs muted' }, 'The current pipeline is replaced. Save it first if you want to keep it.'));
}
function sampleMenu() {
  const md = modal('Sample datasets', '');
  for (const s of SAMPLES) md.el.append(h('div', { class: 'tplCard' }, h('span', { class: 'fic', html: icon('db', 18) }), h('div', {}, h('b', {}, s.name), h('span', {}, s.d)), h('button', { class: 'btn sm primary', onclick: () => { md.close(); loadSample(s.key); } }, 'Load')));
}

// ----------------------------------------------------------------- settings
async function openSettings(section = 'ai') {
  const md = modal('Settings', '', { wide: true });
  const el = md.el;
  const provider = h('select', {}, h('option', { value: 'cloud', selected: AI.provider === 'cloud' }, 'Ollama Cloud (via this app\'s /api/ollama proxy)'), h('option', { value: 'local', selected: AI.provider === 'local' }, 'Local Ollama (http://localhost:11434, direct from the browser)'));
  const key = h('input', { type: 'password', value: AI.apiKey || '', placeholder: 'Paste your Ollama API key', autocomplete: 'off', 'aria-label': 'Ollama API key' });
  const showKey = h('button', { class: 'btn sm', type: 'button', onclick: () => { key.type = key.type === 'password' ? 'text' : 'password'; } }, 'Show');
  const remember = h('input', { type: 'checkbox', checked: AI.remember });
  const base = h('input', { type: 'text', value: AI.baseUrl, 'aria-label': 'Local Ollama URL' });
  const modelSel = h('select', { 'aria-label': 'Model' });
  const custom = h('input', { type: 'text', placeholder: 'or type any model name, e.g. qwen3-coder:480b', 'aria-label': 'Custom model name' });
  const temp = h('input', { type: 'number', min: 0, max: 2, step: 0.1, value: AI.temperature });
  const think = h('select', {}, ['default', 'off', 'on', 'low', 'medium', 'high'].map(v => h('option', { value: v, selected: AI.think === v }, v === 'default' ? 'Model default' : v)));
  const confirmBox = h('input', { type: 'checkbox', checked: AI.confirm });
  const autoFixBox = h('input', { type: 'checkbox', checked: AI.autoFix });
  const status = h('div', { class: 'sm muted', style: 'margin-top:6px' });
  const fillModels = (list, note) => {
    modelSel.innerHTML = '';
    const names = list.length ? list : FALLBACK_MODELS.map(n => ({ name: n }));
    if (!names.some(m => m.name === AI.model) && AI.model) names.unshift({ name: AI.model });
    for (const m of names) modelSel.append(h('option', { value: m.name, selected: m.name === AI.model }, m.name + (m.size ? ` (${(m.size / 1e9).toFixed(0)} GB)` : '')));
    if (note) status.textContent = note;
  };
  const collect = () => ({ provider: provider.value, apiKey: key.value.trim(), remember: remember.checked, baseUrl: base.value.trim() || 'http://localhost:11434', model: custom.value.trim() || modelSel.value, temperature: +temp.value || 0, think: think.value, confirm: confirmBox.checked, autoFix: autoFixBox.checked });
  const refresh = async () => {
    saveAI(collect()); status.textContent = 'Loading models…';
    try { const list = await listModels(); fillModels(list, `${list.length} models available.`); } catch (e) { fillModels([], 'Could not list models (' + e.message + '). Showing common cloud models.'); }
  };
  fillModels([], '');
  const maxRows = h('input', { type: 'number', min: 0, step: 500, value: S.settings.maxTrainRows });
  const slowRows = h('input', { type: 'number', min: 200, step: 500, value: S.settings.slowModelRows });
  const theme = h('select', {}, ['dark', 'light'].map(t => h('option', { value: t, selected: S.settings.theme === t }, t === 'dark' ? 'Dark' : 'Light')));
  const name = h('input', { type: 'text', value: S.settings.name, maxlength: 60 });
  const v = S.engine.versions || {};
  const sec = (title, ...kids) => h('div', { style: 'margin-bottom:18px' }, h('div', { class: 'sec', style: 'font-size:14px;margin-top:0' }, title), ...kids);
  const fld = (label, ...kids) => h('div', { class: 'fld' }, h('label', { class: 'fl' }, label), ...kids);
  el.append(
    sec('AI assistant (Ollama)',
      fld('Provider', provider),
      fld('Ollama API key', h('div', { class: 'row', style: 'flex-wrap:nowrap' }, key, showKey)),
      h('div', { class: 'xs muted' }, 'Create a free key at ', h('a', { href: 'https://ollama.com/settings/keys', target: '_blank', rel: 'noopener' }, 'ollama.com/settings/keys'), '. It is stored only in this browser and sent to Ollama through this app\'s proxy.'),
      h('label', { class: 'chk' }, remember, 'Remember the key on this device (otherwise only for this tab)'),
      fld('Local Ollama URL (local provider only)', base),
      fld('Model', h('div', { class: 'row', style: 'flex-wrap:nowrap' }, modelSel, h('button', { class: 'btn sm', onclick: refresh, html: icon('refresh', 14) + ' Refresh' })), custom),
      h('div', { class: 'xs muted' }, 'Tool calling works best with gpt-oss, qwen3, kimi, glm, deepseek and minimax cloud models.'),
      h('div', { class: 'grid2', style: 'margin-top:0' }, fld('Temperature', temp), fld('Thinking', think)),
      h('label', { class: 'chk' }, confirmBox, 'Ask me before the AI changes the pipeline, trains or runs code'),
      h('label', { class: 'chk' }, autoFixBox, 'When a run has errors, let the AI diagnose and fix them automatically (at most 2 attempts)'),
      h('div', { class: 'row' }, h('button', { class: 'btn', onclick: async () => {
        saveAI(collect()); status.textContent = 'Testing…';
        try { const r = await chat({ messages: [{ role: 'user', content: 'Reply with exactly: OK' }], stream: false }); status.innerHTML = `<span class="okc">Connected to ${esc(AI.model)}</span> — replied “${esc(r.content.trim().slice(0, 60))}”`; } catch (e) { status.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
        renderEngine();
      } }, 'Test connection')), status),
    sec('Python engine & performance',
      h('div', { class: 'kv' }, h('b', {}, 'Status'), h('span', {}, S.engine.message), h('b', {}, 'Python'), h('span', {}, v.python || '—'), h('b', {}, 'scikit-learn'), h('span', {}, v.sklearn || '—'), h('b', {}, 'pandas / numpy / scipy'), h('span', {}, `${v.pandas || '—'} / ${v.numpy || '—'} / ${v.scipy || '—'}`), h('b', {}, 'XGBoost / LightGBM'), h('span', {}, `${v.xgboost || 'not loaded'} / ${v.lightgbm || 'not loaded'}`)),
      h('div', { class: 'grid2', style: 'margin-top:0' }, fld('Max training rows (0 = all)', maxRows), fld('Rows for slow models (SVM, GP, kernel)', slowRows)),
      h('button', { class: 'btn sm', onclick: () => { engine.restart(); S.engine.status = 'loading'; S.results = null; renderEngine(); renderDashboard(); toast('Restarting the engine…'); } }, 'Restart engine')),
    sec('Appearance', h('div', { class: 'grid2', style: 'margin-top:0' }, fld('Theme', theme), fld('Display name', name))),
    h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: () => {
      saveAI(collect());
      Object.assign(S.settings, { maxTrainRows: +maxRows.value || 0, slowModelRows: +slowRows.value || 3000, theme: theme.value, name: name.value.trim() || 'Guest' });
      store.set('settings', S.settings); applyTheme(); renderEngine(); md.close(); toast('Settings saved');
    } }, 'Save settings'), h('span', { class: 'sp' }), h('button', { class: 'btn ghost', onclick: () => { store.del('state'); store.del('ai'); store.del('settings'); store.del('instructions'); toast('Saved data cleared. Reload the page to start fresh.'); } }, 'Clear saved data')));
  if (AI.apiKey || AI.provider === 'local') refresh();
  if (section === 'ai') key.focus();
}

// ---------------------------------------------------------------- agent API
function findModel(ref) {
  const R = S.results; if (!R || !ref) return R ? R.models[R.best] : null;
  const r = String(ref);
  if (R.models[r]) return R.models[r];
  const low = r.toLowerCase();
  const byName = Object.values(R.models).find(m => m.name.toLowerCase() === low) || Object.values(R.models).find(m => m.key === r && !m.tuned_from);
  if (byName) return byName;
  const n = resolveBlock(r);
  return n ? R.models[n.id] : null;
}
const r4 = v => (typeof v === 'number' ? Math.round(v * 1e4) / 1e4 : v);
const top = (arr, n = 15) => (arr || []).slice(0, n);
setAgentAPI({
  updateBlock(ref, settings) {
    const n = resolveBlock(ref);
    if (!n) return { error: `No block "${ref}". Blocks: ${S.nodes.map(x => `${x.id}=${C.nodeTitle(x)}`).join(', ')}` };
    const task = C.taskGuess();
    let applied;
    if (n.type === 'model') {
      const { search_space, grid, ...rest } = settings;
      applied = sanitizeModelParams(n.key, rest, task);
      if (search_space || grid) { n.grid = JSON.stringify(search_space || grid); applied.search_space = search_space || grid; }
      Object.assign(n.cfg, Object.fromEntries(Object.entries(applied).filter(([k]) => k !== 'search_space')));
    } else {
      applied = sanitizeBlock(n.type, settings, columns());
      Object.assign(n.cfg, applied);
    }
    const ignored = Object.keys(settings).filter(k => !(k in applied));
    S.customized = true; S.dirty = true; autosave(); C.renderNodes();
    if (S.sel === n.id) renderConfig();
    if (n.type === 'dataset') refreshTargetDependent();
    return { ok: true, block: n.id, title: C.nodeTitle(n), applied, ignored: ignored.length ? ignored : undefined, now: n.cfg };
  },
  addBlock(type, key, settings) {
    if (!BLOCKS[type]) return { error: `Unknown block type ${type}` };
    if (type === 'model' && !MODEL[key]) return { error: `Unknown model key "${key}". Valid keys: ${MODELS.map(m => m.key).join(', ')}` };
    if (type === 'model' && !C.modelAvailable(key)) return { error: `${MODEL[key].name} is not available in this browser engine.` };
    const n = C.insertNode(type, key);
    let applied = {};
    if (settings) applied = type === 'model' ? sanitizeModelParams(key, settings, C.taskGuess()) : sanitizeBlock(type, settings, columns());
    Object.assign(n.cfg, applied);
    S.customized = true; S.dirty = true; autosave(); C.renderCanvas();
    return { ok: true, id: n.id, title: C.nodeTitle(n), settings: n.cfg };
  },
  removeBlock(ref) {
    const n = resolveBlock(ref);
    if (!n) return { error: `No block "${ref}"` };
    if (n.type === 'dataset') return { error: 'The Dataset block cannot be removed.' };
    const t = C.nodeTitle(n);
    C.deleteNode(n.id); C.tidy(); S.customized = true; autosave(); C.renderCanvas(); renderConfig();
    return { ok: true, removed: t };
  },
  runPipeline: () => run({ source: 'agent' }),
  modelDetails(ref) {
    const R = S.results; if (!R) return { error: 'No results yet. Run the pipeline first.' };
    const m = findModel(ref);
    if (!m) return { error: `Unknown model "${ref}". Models: ${Object.values(R.models).map(x => `${x.id} (${x.name})`).join(', ')}` };
    if (m.status !== 'ok') return { id: m.id, name: m.name, status: m.status, error: m.error };
    const d = m.details || {};
    const res = d.scatter ? (() => { const a = d.scatter.res.map(Math.abs).sort((x, y) => x - y); return { mean_abs: r4(a.reduce((s, x) => s + x, 0) / a.length), p50_abs: r4(a[Math.floor(a.length / 2)]), p90_abs: r4(a[Math.floor(a.length * 0.9)]), max_abs: r4(a[a.length - 1]) }; })() : undefined;
    return {
      id: m.id, name: m.name, family: m.family, params: m.params, ranked_by: m.score_source, score: r4(m.score), is_best: m.id === R.best, rank: R.ranking.indexOf(m.id) + 1,
      cv: m.cv && Object.fromEntries(Object.entries(m.cv).map(([k, v]) => [k, { mean: r4(v.mean), std: r4(v.std) }])), train: m.train, val: m.val, test: m.test,
      fit_time_s: r4(m.fit_time), overfit_gap: r4(m.overfit_gap), classes: R.classes, confusion_matrix_test: d.confusion, per_class_test: d.report, roc_auc_per_class: d.roc?.map(c => ({ class: R.classes?.[c.cls], auc: r4(c.auc) })),
      threshold_table: d.threshold?.filter((_, i) => i % 2 === 0), residuals_test: res,
      native_importance: m.native_importance && { kind: m.native_importance.kind, top: top(m.native_importance.items) }, permutation_importance: top(m.permutation),
      shap_global: m.shap ? top(m.shap.order.map(j => ({ feature: m.shap.features[j], mean_abs_shap: r4(m.shap.global[j]) }))) : undefined,
      tuning: m.tuning && { method: m.tuning.method, n_trials: m.tuning.n_trials, best_params: m.tuning.best_params, before: m.tuning.before, after: m.tuning.after, top_trials: m.tuning.trials.slice(0, 5) },
      learning_curve: m.learning_curve,
    };
  },
  async predict(ref, rows) { const m = findModel(ref); if (!m) return { error: 'Run the pipeline first or give a valid model id.' }; const out = await engine.call('predict', m.id, JSON.stringify(rows)); return { model: m.name, model_id: m.id, ...out }; },
  async whatIf(ref, row, changes) { const m = findModel(ref); if (!m) return { error: 'Run the pipeline first.' }; return { model: m.name, row, ...(await engine.call('what_if', m.id, row, JSON.stringify(changes || {}))) }; },
  async explain(ref, kind, args) {
    const m = findModel(ref); if (!m) return { error: 'Unknown model or no results.' };
    const params = kind === 'pdp' ? { feature: args.feature } : kind === 'local_shap' ? { row: args.row || 0 } : {};
    if (kind === 'pdp' && !args.feature) return { error: 'Give a feature for partial dependence.' };
    const out = await analyze(kind, m.id, params, false);
    S.ui.model = m.id;
    S.ui.dashTab = ['shap', 'local_shap'].includes(kind) ? 'explain' : 'model';
    if (kind === 'pdp') m._pdp = out;
    renderDashboard();
    if (kind === 'shap') return { model: m.name, class: out.class, base_value: r4(out.base), top_features: out.order.slice(0, 15).map(j => ({ feature: out.features[j], mean_abs_shap: r4(out.global[j]) })) };
    if (kind === 'local_shap') return { model: m.name, row: args.row || 0, base_value: r4(out.base), prediction: r4(out.fx[0]), contributions: out.phi[0].map((v, j) => ({ feature: out.features[j], value: out.raw_values[0][j], shap: r4(v) })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, 12) };
    if (kind === 'permutation') return { model: m.name, importance: top(out) };
    if (kind === 'pdp') return { model: m.name, feature: out.feature, values: out.values, average_prediction: out.mean.map(r4) };
    return { model: m.name, ...out };
  },
  async runPython(code) {
    const out = await engine.call('run_code', code);
    if (out.results_changed) { S.results = await engine.call('get_summary'); renderDashboard(); toast('Leaderboard updated from custom code'); }
    return out;
  },
  show(tab, ref) { if (ref) { const m = findModel(ref); if (m) S.ui.model = m.id; } S.ui.dashTab = tab; renderDashboard(); $('#dash')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); return { ok: true, tab }; },
  testRows: (start, count) => engine.call('test_rows', start, count),
  exportThing,
  async diagnose() { const dg = await diagnose(); return { issues: dg.issues.map(i => ({ id: i.id, severity: i.severity, title: i.title, detail: i.detail, fix: i.fix || undefined, model: i.model || undefined })), error: dg.error }; },
  applyFixes: ids => applyFixes(ids),
  retrainModels: refs => retrainModels(refs),
  listCells: () => listCells(),
  writeCell: (id, code, runIt) => writeCell(id, code, runIt !== false),
});

// ------------------------------------------------ doctor, fixes, python lab
async function diagnose() {
  let cfg;
  try { cfg = buildConfig(); } catch (e) { S.diag = { issues: [{ id: 'config', severity: 'error', code: 'config', title: 'The pipeline cannot run', detail: e.message, fix: null }], time: Date.now(), applied: new Set() }; afterDiag(); return S.diag; }
  try { const r = await engine.call('diagnose', JSON.stringify(cfg)); S.diag = { issues: r.issues || [], time: Date.now(), applied: new Set() }; }
  catch (e) { S.diag = { issues: [], error: e.message, time: Date.now(), applied: new Set() }; }
  afterDiag();
  return S.diag;
}
function afterDiag() { if (S.ui.dashTab === 'doctor') renderDashboard(); else updateRunBar(); }
function applyPatch(p) {
  if (p.add_block) {
    const n = C.insertNode(p.add_block);
    if (p.settings) Object.assign(n.cfg, sanitizeBlock(p.add_block, p.settings, columns()));
    return `added ${C.nodeTitle(n)}`;
  }
  let n = resolveBlock(p.block);
  if (!n && BLOCKS[p.block] && p.block !== 'model') n = C.insertNode(p.block);
  if (!n) throw new Error(`Block ${p.block} not found`);
  const title = C.nodeTitle(n);
  if (p.remove) { C.deleteNode(n.id); C.tidy(); return `removed ${title}`; }
  if (p.add_exclude) { n.cfg.exclude = [...new Set([...(n.cfg.exclude || []), ...p.add_exclude])]; return `excluded ${p.add_exclude.join(', ')}`; }
  if (p.remove_feature) { n.cfg.custom_features = (n.cfg.custom_features || []).filter(f => f.name !== p.remove_feature); return `deleted feature ${p.remove_feature}`; }
  const applied = n.type === 'model' ? sanitizeModelParams(n.key, p.settings || {}, C.taskGuess()) : sanitizeBlock(n.type, p.settings || {}, columns());
  Object.assign(n.cfg, applied);
  return `${title}: ${Object.entries(applied).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(',') : v}`).join(', ')}`;
}
function applyFixes(ids) {
  const all = S.diag?.issues || [];
  const pick = ids.includes('all') ? all.filter(i => i.fix) : all.filter(i => ids.includes(i.id));
  const applied = [], skipped = [];
  for (const i of pick) {
    if (!i.fix) { skipped.push({ id: i.id, reason: 'no automatic fix; use update_block or explain it' }); continue; }
    if (S.diag.applied?.has(i.id)) { skipped.push({ id: i.id, reason: 'already applied' }); continue; }
    try { applied.push({ id: i.id, title: i.title, changes: i.fix.map(applyPatch) }); S.diag.applied.add(i.id); }
    catch (e) { skipped.push({ id: i.id, reason: e.message }); }
  }
  const unknown = ids.filter(x => x !== 'all' && !all.some(i => i.id === x));
  if (applied.length) { S.customized = true; S.dirty = true; autosave(); C.renderCanvas(); renderConfig(); toast(`Applied ${applied.length} fix${applied.length > 1 ? 'es' : ''}. Run the pipeline to see the effect.`); }
  return { applied, skipped, unknown_ids: unknown.length ? unknown : undefined, next: applied.length ? 'Run the pipeline (or retrain_models for model-only changes) to verify.' : undefined };
}
async function retrainModels(refs) {
  if (!S.results) return { error: 'Run the pipeline first.' };
  if (S.running) return { error: 'A run is in progress.' };
  let cfg;
  try { cfg = buildConfig(); } catch (e) { return { error: e.message }; }
  const ids = [];
  for (const r of refs || []) {
    const s = String(r);
    const hit = cfg.models.find(m => m.id === s) || cfg.models.find(m => m.id === resolveBlock(s)?.id) || cfg.models.find(m => m.id === S.results.models[s]?.tuned_from) || cfg.models.find(m => m.key === s);
    if (hit && !ids.includes(hit.id)) ids.push(hit.id);
  }
  if (!ids.length) return { error: `No matching model blocks. Model ids: ${cfg.models.map(m => `${m.id} (${m.name})`).join(', ')}` };
  S.running = true; S.runStart = Date.now(); S.progress = { message: 'Retraining…' };
  for (const id of ids) { const n = node(id); if (n) n.status = 'running'; }
  renderRunStatus(); C.renderNodes(); updateRunBar();
  try {
    const res = await engine.call('retrain_models', JSON.stringify(cfg), JSON.stringify(ids));
    const report = res.retrained; delete res.retrained;
    S.results = { ...res, duration: S.results.duration };
    applyStatuses(S.results, C.reachable(nodeOf('dataset').id));
    diagnose();
    return { retrained: report, leaderboard: leaderboard() };
  } catch (e) { for (const id of ids) { const n = node(id); if (n) n.status = 'error'; } return { error: e.message }; }
  finally { S.running = false; S.progress = null; renderAll(); }
}
function maybeAutoFix(opts = {}) {
  if (opts.source === 'agent' || !AI.autoFix || !aiReady() || !mainChat || mainChat.busy) return;
  const failed = S.results ? Object.values(S.results.models).filter(m => m.status !== 'ok') : [];
  const errs = (S.diag?.issues || []).filter(i => i.severity === 'error');
  if (!S.lastError && !failed.length && !errs.length) return;
  if (S.autoFixCount >= 2) return;
  S.autoFixCount++;
  toast('Problems found — the AI is diagnosing and fixing them…', null, 4000);
  fixWithAI('auto');
}

const DEFAULT_CELL = `# Python Lab: run any code on your data and trained models (Shift+Enter).
# Available after a run: X_train, X_test, y_train, y_test, models, results, best_model_id, register_model(...)
print(df.shape)
df.describe().T.head(10)`;
S.cells = (store.get('cells', null) || [{ id: 'c1', code: DEFAULT_CELL }]).map(c => ({ id: c.id, code: c.code }));
const saveCells = debounce(() => store.set('cells', S.cells.map(c => ({ id: c.id, code: c.code }))), 400);
function newCellId() { let i = 1; while (S.cells.some(c => c.id === 'c' + i)) i++; return 'c' + i; }
function addCell(code = '') { const c = { id: newCellId(), code }; S.cells.push(c); saveCells(); return c; }
async function runCell(id) {
  const c = S.cells.find(x => x.id === id);
  if (!c) return { error: `No cell ${id}. Cells: ${S.cells.map(x => x.id).join(', ')}` };
  c.running = true;
  if (S.ui.dashTab === 'code') renderDashboard();
  try {
    c.out = await engine.call('run_code', c.code);
    if (c.out.results_changed) { S.results = await engine.call('get_summary'); toast('Leaderboard updated from custom code'); }
  } catch (e) { c.out = { error: e.message }; }
  c.running = false;
  if (S.ui.dashTab === 'code') renderDashboard();
  return c.out;
}
async function writeCell(id, code, runIt = true) {
  let c = id ? S.cells.find(x => x.id === String(id)) : null;
  if (!c) { c = addCell(''); if (id && /^c\d+$/.test(String(id)) && !S.cells.some(x => x.id === String(id))) c.id = String(id); }
  c.code = String(code ?? ''); c.out = null; saveCells();
  S.ui.dashTab = 'code'; renderDashboard();
  if (runIt === false) return { cell_id: c.id, saved: true };
  const out = await runCell(c.id);
  return { cell_id: c.id, status: out.exception || out.error ? 'error' : 'ok', ...out };
}
function listCells() {
  return S.cells.map(c => ({ cell_id: c.id, code: c.code, status: c.out ? (c.out.exception || c.out.error ? 'error' : 'ok') : 'not run', error: c.out?.error_info || c.out?.error || undefined, stdout: c.out?.stdout ? c.out.stdout.slice(-600) : undefined, warnings: c.out?.warnings?.length ? c.out.warnings : undefined }));
}
function fixWithAI(kind, p = {}) {
  const fence = code => '```python\n' + code + '\n```';
  let q;
  if (kind === 'cell' || kind === 'review') {
    const c = S.cells.find(x => x.id === p.id); if (!c) return;
    const e = c.out?.error_info;
    q = kind === 'cell'
      ? `Python Lab cell ${c.id} has a problem.\n\n${fence(c.code)}\n${e ? `Error${e.line ? ` on line ${e.line}` : ''}: ${e.type}: ${e.message}\nHint: ${e.hint}\n` : c.out?.error ? `Error: ${c.out.error}\n` : ''}${c.out?.warnings?.length ? `Review warnings: ${c.out.warnings.join(' ')}\n` : ''}\nFind the root cause, correct the code with write_code_cell (cell_id "${c.id}"), run it again and repeat until it works. Fix conceptual ML mistakes too, then explain briefly what was wrong.`
      : `Review Python Lab cell ${c.id} for bugs and conceptual ML mistakes (data leakage, fitting or selecting on test data, wrong metric or target encoding, wrong pairing of X and y).\n\n${fence(c.code)}\n${c.out?.warnings?.length ? `Static review warnings: ${c.out.warnings.join(' ')}\n` : ''}If you find problems, fix them with write_code_cell (cell_id "${c.id}") and run it to verify. Explain what you changed, or confirm the code is correct.`;
  } else if (kind === 'code') {
    q = `This code has a problem:\n${fence(p.code)}\n${p.error}\nFix it: write the corrected code with write_code_cell, run it until it works, and explain the fix.`;
  } else if (kind === 'model') {
    q = `Model "${p.name}" (id ${p.id}) failed with this error:\n${p.error}\nDiagnose the root cause, fix the model block with update_block (or apply_fixes), verify with retrain_models, and explain what was wrong.`;
  } else if (kind === 'issue') {
    const i = p.issue;
    q = `The Pipeline Doctor reports [${i.severity}] ${i.title}: ${i.detail}${i.fix ? ` Suggested fix: ${describeFixText(i.fix)}.` : ''} (issue id ${i.id})\nIs this a real problem for my data? If yes, fix it (apply_fixes or update_block), verify, and explain.`;
  } else if (kind === 'doctor') {
    const list = (S.diag?.issues || []).map(i => `- [${i.severity}] ${i.title} (id ${i.id}): ${i.detail}`).join('\n');
    q = `The Pipeline Doctor found these problems:\n${list}\n\nFix the real problems (apply_fixes where a fix exists, otherwise update_block), say which were conceptual ML mistakes, then run the pipeline to verify and summarise the before / after.`;
  } else {
    const failed = S.results ? Object.values(S.results.models).filter(m => m.status !== 'ok') : [];
    const errs = (S.diag?.issues || []).filter(i => i.severity === 'error' && i.code !== 'model_failed');
    const parts = [S.lastError ? `The run failed: ${S.lastError}` : '', ...failed.map(m => `Model ${m.name} (id ${m.id}) failed: ${m.error}`), ...errs.map(i => `Pipeline Doctor [error] ${i.title} (id ${i.id}): ${i.detail}`)].filter(Boolean);
    q = `Something went wrong in the last run:\n${parts.map(x => '- ' + x).join('\n')}\n\nFollow the debugging protocol: diagnose, fix the root causes, verify (retrain_models for model settings, run_pipeline for data or preprocessing changes), then explain what was wrong and what you changed.`;
  }
  askAI(q);
}
const describeFixText = fix => fix.map(x => JSON.stringify(x)).join('; ');

// ---------------------------------------------------------------- rendering
function applyTheme() {
  document.documentElement.dataset.theme = S.settings.theme === 'light' ? 'light' : 'dark';
  const b = $('#btnTheme'); if (b) b.innerHTML = icon(S.settings.theme === 'light' ? 'moon' : 'sun', 18);
}
function renderEngine() {
  const st = S.engine.status, v = S.engine.versions;
  const ep = $('#enginePill');
  if (ep) ep.innerHTML = `<span class="dot ${st === 'ready' ? 'ok' : st === 'error' ? 'bad' : 'busy'}"></span>${st === 'ready' ? `Python · scikit-learn ${esc(v?.sklearn || '')}` : st === 'error' ? 'Engine error' : 'Loading engine…'}`;
  const ap = $('#aiPill');
  if (ap) ap.innerHTML = `<span class="dot ${aiReady() ? 'ok' : ''}"></span>${aiReady() ? esc(AI.model) : 'Connect AI'}`;
  const cm = $('#chatModel'); if (cm) cm.textContent = aiReady() ? `${AI.provider === 'local' ? 'Local' : 'Ollama Cloud'} · ${AI.model}` : 'not connected';
  const nm = $('#userName'); if (nm) nm.textContent = S.settings.name;
  const av = $('#userAv'); if (av) av.textContent = S.settings.name.split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0].toUpperCase()).join('') || 'U';
  const boot = $('#boot');
  if (boot) { boot.hidden = st === 'ready'; boot.innerHTML = st === 'error' ? `<b class="err">Engine failed to start</b><div class="xs muted" style="margin-top:4px">${esc(S.engine.message)}</div>` : `<div class="row" style="flex-wrap:nowrap"><span class="dot busy"></span><b>Starting the Python ML engine</b></div><div class="xs muted" style="margin-top:4px">${esc(S.engine.message)}</div>`; }
  const run = $('#btnRun'); if (run) run.disabled = st !== 'ready';
}
function renderRunStatus() {
  const el = $('#runStatus'); if (!el) return;
  const btn = $('#btnRun');
  if (S.running) {
    const p = S.progress || {};
    el.innerHTML = `<b>${esc(p.message || 'Running…')}</b>${p.n ? ` · ${p.i}/${p.n}` : ''} · ${fmtTime((Date.now() - S.runStart) / 1000)}`;
    btn.className = 'btn danger'; btn.innerHTML = icon('stop', 12) + 'Stop';
  } else {
    el.textContent = S.results ? `Last run: ${fmtTime(S.results.duration)} · best ${S.results.models[S.results.best].name}${S.dirty ? ' · pipeline changed since' : ''}` : "Drag from a block's right dot to connect · click a line to remove it";
    btn.className = 'btn run'; btn.innerHTML = icon('play', 12) + 'Run Pipeline';
  }
}
function renderAll() { C.renderCanvas(); renderConfig(); renderData(); renderDocs(); renderDashboard(); renderRunStatus(); renderEngine(); }

let mainChat;
function askAI(text) {
  $('#chatPanel')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (mainChat.busy) { toast('The assistant is still working on the previous request.'); return; }
  mainChat.send(text);
}

Object.assign(A, {
  run, stop, analyze, exportThing, openSettings, diagnose, applyFixes, retrainModels, fixWithAI, runCell, writeCell, addCell, saveCells, sampleMenu, loadSample, loadFile, loadPreview, loadEDA, refreshTargetDependent, askAI, restoreRun,
  openFile: () => $('#fileIn').click(),
});

on('pipeline', () => { S.customized = true; S.dirty = true; C.renderNodes(); autosave(); renderRunStatus(); });
on('select', () => renderConfig());
on('open-settings', s => openSettings(s));
on('ai-settings', () => { saveAI({}); renderEngine(); });
on('chat-done', () => { C.renderNodes(); refreshBlockChat(); });

function init() {
  paintIcons();
  applyTheme();
  C.bindCanvas();
  mainChat = new ChatView($('#chatBox'), () => S.chat, { chips: ['Run the pipeline and summarise all models', 'Why is the best model the best?', 'Add LightGBM and a stacking ensemble, then run', 'Plot the errors of the best model on the test set'] });
  $('#chatClear').onclick = () => { clearThread(S.chat); mainChat.render(); };
  $('#btnRun').onclick = () => (S.running ? stop() : run({ source: 'user' }));
  $('#btnSave').onclick = () => { autosave(); download('ml-agent-pipeline.json', pipelineJSON(), 'application/json'); toast('Pipeline saved (also kept in this browser)'); };
  $('#btnLoad').onclick = () => $('#pipeIn').click();
  $('#pipeIn').onchange = e => { const f = e.target.files[0]; if (f) loadPipelineFile(f); e.target.value = ''; };
  $('#fileIn').onchange = e => { const f = e.target.files[0]; if (f) loadFile(f); e.target.value = ''; };
  $('#btnSettings').onclick = () => openSettings('ai');
  $('#aiPill').onclick = () => openSettings('ai');
  $('#btnTheme').onclick = () => { S.settings.theme = S.settings.theme === 'light' ? 'dark' : 'light'; store.set('settings', S.settings); applyTheme(); renderDashboard(); };
  $('#btnTidy').onclick = () => { C.tidy(); C.renderCanvas(); };
  $('#btnAdd').onclick = e => { e.stopPropagation(); const m = $('#addMenu'); m.hidden = !m.hidden; if (!m.hidden) { C.renderAddMenu(); m.querySelector('input')?.focus(); } };
  $('#btnMore').onclick = e => { e.stopPropagation(); $('#moreMenu').hidden = !$('#moreMenu').hidden; };
  document.addEventListener('click', e => {
    if (!e.target.closest('#addMenu') && !e.target.closest('#btnAdd')) $('#addMenu').hidden = true;
    if (!e.target.closest('.menuWrap')) $('#moreMenu').hidden = true;
  });
  $('#moreMenu').addEventListener('click', e => {
    const act = e.target.closest('[data-act]')?.dataset.act; if (!act) return;
    $('#moreMenu').hidden = true;
    ({ templates: templatesModal, exportZip: () => exportThing('project_zip'), exportJson: () => exportThing('pipeline_json'), lb: () => exportThing('leaderboard_csv'), reset: () => { S.customized = false; C.defaultPipeline(); autosave(); renderAll(); } })[act]?.();
  });
  for (const b of $$('[data-go]')) b.addEventListener('click', () => {
    $$('.nv').forEach(x => x.classList.toggle('on', x === b));
    if (b.dataset.tab) { S.ui.dashTab = b.dataset.tab; renderDashboard(); }
    $('#' + b.dataset.go)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  for (const b of $$('[data-act]')) if (!b.closest('#moreMenu')) b.addEventListener('click', () => ({ settings: () => openSettings('ai'), ai: () => openSettings('ai'), templates: templatesModal, upload: () => $('#fileIn').click(), samples: sampleMenu, export: () => exportThing('project_zip') })[b.dataset.act]?.());
  $('#askInput').addEventListener('keydown', e => { if (e.key === 'Enter' && e.target.value.trim()) { askAI(e.target.value.trim()); e.target.value = ''; } });
  window.addEventListener('load', () => renderDashboard());
  renderAll();
  engine.start();
}
init();
