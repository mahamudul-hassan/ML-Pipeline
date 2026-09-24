// Ollama Cloud / local Ollama client and the tool-calling agent that drives the app.
import { S, store, node, nodeOf, colInfo } from './store.js';
import { BLOCKS, MODELS, MODEL, optLabel, LOWER_BETTER } from './registry.js';
import { nodeTitle } from './canvas.js';

export const FALLBACK_MODELS = ['gpt-oss:120b', 'gpt-oss:20b', 'qwen3-coder-next', 'kimi-k2.6', 'glm-4.7', 'deepseek-v3.2', 'minimax-m2.7', 'qwen3-next:80b', 'gemma4:31b'];
const DEFAULTS = { provider: 'cloud', apiKey: '', remember: true, model: 'gpt-oss:120b', baseUrl: 'http://localhost:11434', temperature: 0.3, confirm: true, autoFix: true, think: 'default', proxy: '/api/ollama' };
export const AI = { ...DEFAULTS, ...store.get('ai', {}) };
if (!AI.remember) AI.apiKey = sessionStorage.getItem('mlagent.key') || '';
export function saveAI(patch) {
  Object.assign(AI, patch);
  const persisted = { ...AI };
  if (!AI.remember) { persisted.apiKey = ''; try { sessionStorage.setItem('mlagent.key', AI.apiKey || ''); } catch { /* ignore */ } }
  store.set('ai', persisted);
}
export const aiReady = () => (AI.provider === 'local' ? !!AI.model : !!(AI.apiKey && AI.model));

function endpoint(path) { return AI.provider === 'local' ? AI.baseUrl.replace(/\/$/, '') + '/api/' + path : AI.proxy + '/' + path; }
function headers() {
  const hd = { 'Content-Type': 'application/json' };
  if (AI.provider !== 'local' && AI.apiKey) hd['x-ollama-key'] = AI.apiKey;
  return hd;
}
async function errorText(res) {
  const t = await res.text().catch(() => '');
  let msg = t;
  try { msg = JSON.parse(t).error || t; } catch { /* plain text */ }
  if (res.status === 401 || res.status === 403) return `Ollama rejected the API key (${res.status}). Check the key in Settings → AI.`;
  if (res.status === 404 && /model/i.test(msg)) return `Model not found: ${msg}. Pick another model in Settings → AI.`;
  if (res.status === 429) return 'Ollama Cloud rate limit or free usage limit reached. Wait a bit or choose a smaller model.';
  if (res.status === 404 && AI.provider !== 'local') return 'The /api/ollama proxy was not found. Deploy on Vercel or run `npm run dev` (see README).';
  return `Ollama error ${res.status}: ${String(msg).slice(0, 400)}`;
}

export async function listModels() {
  const res = await fetch(endpoint(AI.provider === 'local' ? 'tags' : 'models'), { headers: headers() });
  if (!res.ok) throw new Error(await errorText(res));
  const data = await res.json();
  return (data.models || []).map(m => ({ name: m.name || m.model, size: m.size, details: m.details || {} })).filter(m => m.name).sort((a, b) => a.name.localeCompare(b.name));
}

// Streams /api/chat NDJSON. Returns { content, thinking, tool_calls }.
export async function chat({ messages, tools, format, signal, onDelta, stream = true }) {
  const body = { model: AI.model, messages, stream, options: { temperature: Number(AI.temperature) } };
  if (tools && tools.length) body.tools = tools;
  if (format) body.format = format;
  if (AI.think !== 'default') body.think = ['low', 'medium', 'high'].includes(AI.think) ? AI.think : AI.think === 'on';
  const res = await fetch(endpoint('chat'), { method: 'POST', headers: headers(), body: JSON.stringify(body), signal });
  if (!res.ok) throw new Error(await errorText(res));
  const out = { content: '', thinking: '', tool_calls: [] };
  const take = obj => {
    if (obj.error) throw new Error('Ollama: ' + obj.error);
    const m = obj.message || {};
    if (m.content) out.content += m.content;
    if (m.thinking) out.thinking += m.thinking;
    if (m.tool_calls?.length) out.tool_calls.push(...m.tool_calls);
    onDelta && onDelta(out);
  };
  if (!stream || !res.body) { take(await res.json()); return out; }
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (line) take(JSON.parse(line)); }
  }
  if (buf.trim()) take(JSON.parse(buf));
  return out;
}

// ---------------------------------------------------------------- context
const r4 = v => (v == null ? null : Math.round(v * 10000) / 10000);
export function dataSummary(maxCols = 60) {
  if (!S.profile) return 'No dataset loaded.';
  const p = S.profile;
  const cols = p.cols.slice(0, maxCols).map(c => `${c.name} (${c.kind}${c.missing ? `, ${c.missing} missing` : ''}${c.kind === 'numeric' ? `, mean ${r4(c.mean)}` : `, ${c.unique} values`}${c.id_like ? ', id-like' : ''})`);
  return `Dataset "${S.dataName}": ${p.rows} rows, ${p.cols.length} columns.\nColumns: ${cols.join('; ')}${p.cols.length > maxCols ? ` … and ${p.cols.length - maxCols} more` : ''}`;
}
export function pipelineSummary() {
  return S.nodes.map(n => {
    const cfg = n.type === 'model' ? { key: n.key, params: n.cfg } : n.cfg;
    return `- ${n.id} ${n.type === 'model' ? 'model' : n.type} "${nodeTitle(n)}": ${JSON.stringify(cfg)}`;
  }).join('\n') + `\nConnections: ${S.edges.map(e => `${e.from}→${e.to}`).join(', ')}`;
}
export function leaderboard(metric = null, limit = 60) {
  const R = S.results; if (!R) return null;
  const pm = metric || R.primary;
  const rows = Object.values(R.models).map(m => ({
    id: m.id, name: m.name, status: m.status, error: m.error,
    [`cv_${pm}`]: r4(m.cv?.[pm]?.mean), [`cv_${pm}_std`]: r4(m.cv?.[pm]?.std), [`val_${pm}`]: r4(m.val?.[pm]), [`test_${pm}`]: r4(m.test?.[pm]), [`train_${pm}`]: r4(m.train?.[pm]),
    fit_s: r4(m.fit_time), tuned_from: m.tuned_from, baseline: m.baseline || undefined,
  }));
  const lb = LOWER_BETTER.has(pm);
  const key = R.cv ? `cv_${pm}` : R.sizes.val ? `val_${pm}` : `test_${pm}`;
  rows.sort((a, b) => (a.status !== 'ok') - (b.status !== 'ok') || ((a[key] ?? (lb ? 1e18 : -1e18)) - (b[key] ?? (lb ? 1e18 : -1e18))) * (lb ? 1 : -1));
  return { primary: R.primary, ranked_by: key, best: R.best, best_name: R.models[R.best]?.name, task: R.task, classes: R.classes, sizes: R.sizes, cv: R.cv, rows: rows.slice(0, limit) };
}
function resultsSummary() {
  const lb = leaderboard(); if (!lb) return 'No results yet: the pipeline has not been run.';
  const lines = lb.rows.slice(0, 15).map(r => `${r.id} | ${r.name} | ${r.status === 'ok' ? `${lb.ranked_by}=${r[lb.ranked_by]} test=${r[`test_${lb.primary}`]} fit=${r.fit_s}s` : 'ERROR ' + r.error}`);
  return `Task ${lb.task}${lb.classes ? ` (classes: ${lb.classes.join(', ')})` : ''}. Primary metric ${lb.primary}. Split train/val/test = ${lb.sizes.train}/${lb.sizes.val}/${lb.sizes.test}${lb.cv ? `, CV ${lb.cv.name} (${lb.cv.splits} splits)` : ', no CV'}.\nBest model: ${lb.best} (${lb.best_name}).\nLeaderboard (id | name | scores):\n${lines.join('\n')}`;
}
function blockReference() {
  const lines = Object.entries(BLOCKS).filter(([, b]) => b.fields?.length).map(([t, b]) => `${t}: ` + b.fields.map(f => `${f.k}${f.o ? `(${f.o.map(o => o === null ? 'null' : o).join('|')})` : `(${f.f})`}`).join(', '));
  const models = MODELS.map(m => `${m.key}=${m.name} [${m.tasks.map(t => t[0]).join('')}] params: ${m.params.map(p => p.k).join(',')}`);
  return lines.join('\n') + '\nModel keys ([c]lassification/[r]egression):\n' + models.join('\n');
}
export function systemPrompt(focus) {
  const docs = S.docs.filter(d => d.text).map(d => `### ${d.name}\n${d.text.slice(0, 4000)}`).join('\n\n');
  return `You are ML Agent, the AI assistant of a visual machine-learning pipeline builder (n8n-style canvas). The app runs real scikit-learn, XGBoost and LightGBM in the user's browser via Pyodide. Through tools you can inspect and change the pipeline, train models, and do anything with the trained models.

How to work:
- Use tools to get facts; never invent numbers. Call get_leaderboard / get_model_details before comparing models.
- To change the pipeline use update_block / add_block / remove_block, then run_pipeline when the user wants new results.
- For anything else (custom analysis or plots, error analysis, new or custom models, statistics, filtering data, threshold choices, saving outputs) use run_python. Python namespace: df (raw data), X_train, X_val, X_test (raw feature DataFrames that the fitted pipelines accept), y_train, y_val, y_test (class indices for classification), classes, task, models (dict model_id → fitted sklearn Pipeline), results (dict model_id → metrics), best_model_id, register_model(name, estimator) (wraps it in the current preprocessing, trains on the training set, evaluates with CV/val/test and adds it to the leaderboard), make_preprocessor(), predict(model_id, rows), compute_metrics(y_true, y_pred), np, pd, plt (matplotlib figures are shown to the user), sklearn. print() output and the last expression are returned.
- Answer briefly and concretely: key numbers, what they mean, a recommended next step. Use markdown tables for comparisons.
- Reply in the user's language (Bengali or English); keep ML terms in English.
- The user may have to approve pipeline changes and code before they run.

Debugging and quality protocol (use it whenever something fails, looks wrong, or the user asks you to check or fix something):
1. Gather facts: diagnose (Pipeline Doctor), get_model_details, list_code_cells, and the error_info of run_python / write_code_cell (failing line, hint, available names or columns).
2. Find the root cause and say whether it is a code error or a conceptual ML error (leakage, wrong validation, wrong metric, overfitting …).
3. Fix it with the smallest correct change: apply_fixes for doctor issues, update_block for settings and hyperparameters, write_code_cell with the same cell_id to correct code. Never hide errors with bare try/except, never delete a model, column or cell just to make an error disappear unless that is the right fix, and never train, tune or select models on the test set.
4. Verify: re-run the cell, retrain_models for changed model blocks, or run_pipeline when data, split, preprocessing or features changed. If it still fails, repeat (up to 3 attempts), then explain what blocks it.
5. Report briefly: what was wrong, what you changed, and the result after the fix.
Code you run also gets static-review warnings (for example fitting on the test set); treat them as bugs and fix them.
Python Lab: the user's notebook cells (list_code_cells / write_code_cell). Cells and run_python share one namespace.

## Current state
${dataSummary()}
Target: ${nodeOf('dataset')?.cfg.target || 'not set'}.
Pipeline blocks:
${pipelineSummary()}
Results:
${resultsSummary()}
${S.diag?.issues?.length ? `\nPipeline Doctor (last check): ${S.diag.issues.filter(i => i.severity !== 'info').slice(0, 8).map(i => `[${i.severity}] ${i.title} (id ${i.id})`).join('; ') || 'only informational notes'}.` : ''}
${focus ? `\nThe user is looking at block ${focus.id} (${nodeTitle(focus)}). Focus on it.` : ''}
${S.instructions ? `\n## User instructions\n${S.instructions.slice(0, 3000)}` : ''}
${docs ? `\n## Extra documents\n${docs.slice(0, 12000)}` : ''}

## Block settings reference
${blockReference()}`;
}

// ------------------------------------------------------------------ tools
const fn = (name, description, properties = {}, required = []) => ({ type: 'function', function: { name, description, parameters: { type: 'object', properties, required } } });
const BLOCK_TYPES = ['dataset', 'split', 'preprocess', 'fe', 'fs', 'model', 'zoo', 'voting', 'stacking', 'eval', 'tuning', 'deploy'];
export const TOOLS = [
  fn('get_state', 'Get the dataset profile, all pipeline blocks with their settings, and the results summary.'),
  fn('update_block', 'Change settings of a pipeline block. For model blocks the settings are scikit-learn hyperparameters. For tuning search spaces put {"grids": {"rf": {"n_estimators": [100, 300]}}} on the tuning block.', {
    block: { type: 'string', description: 'Block id (e.g. "n3"), block type (dataset, split, preprocess, fe, fs, eval, tuning, zoo, voting, stacking) or model key (e.g. "rf").' },
    settings: { type: 'object', description: 'Only the settings to change.' },
  }, ['block', 'settings']),
  fn('add_block', 'Add a block. Use type "model" with model_key to add a model, or "zoo" with models to train many at once.', {
    type: { type: 'string', enum: BLOCK_TYPES }, model_key: { type: 'string', description: 'For type=model' }, settings: { type: 'object', description: 'Optional initial settings' },
  }, ['type']),
  fn('remove_block', 'Remove a block from the pipeline.', { block: { type: 'string', description: 'Block id, type or model key' } }, ['block']),
  fn('run_pipeline', 'Train and evaluate the whole pipeline now (all models, CV, tuning, explanations). Returns the leaderboard when finished.'),
  fn('get_leaderboard', 'Leaderboard of all trained models with CV, validation and test scores.', { metric: { type: 'string', description: 'Metric name, default is the primary metric' } }),
  fn('get_model_details', 'All results for one model: metrics on train / val / test / CV, confusion matrix and per-class report or residual statistics, feature importances, tuning trials and hyperparameters.', { model_id: { type: 'string' } }, ['model_id']),
  fn('predict', 'Predict with a trained model for rows of raw feature values (column → value). Missing columns are imputed.', { model_id: { type: 'string', description: 'Omit for the best model' }, rows: { type: 'array', items: { type: 'object' } } }, ['rows']),
  fn('what_if', 'Take a test-set row, change some feature values and compare the prediction before and after.', { model_id: { type: 'string' }, row: { type: 'integer', description: 'Test-set row index' }, changes: { type: 'object' } }, ['row', 'changes']),
  fn('explain_model', 'Compute an explanation for a model and show it in the dashboard: SHAP values, permutation importance, learning curve, partial dependence for a feature, or SHAP for one test row.', {
    model_id: { type: 'string' }, kind: { type: 'string', enum: ['shap', 'permutation', 'learning_curve', 'pdp', 'local_shap'] }, feature: { type: 'string', description: 'For pdp' }, row: { type: 'integer', description: 'For local_shap' },
  }, ['model_id', 'kind']),
  fn('run_python', 'Run Python code in the in-browser engine with the data and the trained models (see the namespace in the system prompt). Returns printed output, the last expression, tables and figures.', { code: { type: 'string' } }, ['code']),
  fn('show_in_dashboard', 'Open a dashboard tab for the user, optionally for a specific model.', { tab: { type: 'string', enum: ['overview', 'leaderboard', 'model', 'compare', 'tuning', 'explain', 'data', 'predict', 'logs', 'runs'] }, model_id: { type: 'string' } }, ['tab']),
  fn('get_test_rows', 'Get rows of the test set (raw columns) to pick examples for predictions or what-if analysis.', { start: { type: 'integer' }, count: { type: 'integer' } }),
  fn('diagnose', 'Run the Pipeline Doctor: checks the current pipeline, data and last results for code-level and conceptual ML problems (target leakage, ID columns, wrong task or metric, selecting models on the test set, time data shuffled, missing scaling, class imbalance, feature explosion, failed models with hints, overfitting, suspiciously perfect scores, no better than baseline, unstable CV). Each issue has an id and, when possible, a machine-applicable fix.'),
  fn('apply_fixes', 'Apply the suggested fixes of Pipeline Doctor issues (from diagnose) to the pipeline.', { issue_ids: { type: 'array', items: { type: 'string' }, description: 'Issue ids from diagnose, or ["all"] for every issue that has a fix.' } }, ['issue_ids']),
  fn('retrain_models', 'Retrain only the given model blocks on the existing data split (fast way to verify a hyperparameter fix). If data, split or preprocessing changed, use run_pipeline instead.', { model_ids: { type: 'array', items: { type: 'string' }, description: 'Model block ids (e.g. "n7") or model keys (e.g. "rf").' } }, ['model_ids']),
  fn('list_code_cells', 'List the Python Lab cells (the user\'s notebook) with their code, status, errors and warnings.'),
  fn('write_code_cell', 'Create or replace a Python Lab cell and (by default) run it. Use it to write analysis code and to correct cells that fail: pass the same cell_id to replace the broken code. Returns output, error_info (failing line, hint) and static-review warnings.', { cell_id: { type: 'string', description: 'Existing cell id to replace, or omit for a new cell' }, code: { type: 'string' }, run: { type: 'boolean', description: 'Default true' } }, ['code']),
  fn('export', 'Download something for the user.', { what: { type: 'string', enum: ['project_zip', 'model_file', 'test_predictions', 'leaderboard_csv', 'pipeline_json'] }, model_id: { type: 'string' } }, ['what']),
];
const MUTATING = new Set(['update_block', 'add_block', 'remove_block', 'run_pipeline', 'run_python', 'apply_fixes', 'retrain_models', 'write_code_cell']);
let API = null;
export function setAgentAPI(api) { API = api; }

export function resolveBlock(ref) {
  if (!ref) return null;
  const r = String(ref).trim();
  return node(r) || S.nodes.find(n => n.type === r) || S.nodes.find(n => n.type === 'model' && n.key === r) || S.nodes.find(n => nodeTitle(n).toLowerCase() === r.toLowerCase()) || null;
}
function compact(obj, max = 9000) {
  let s = JSON.stringify(obj, (k, v) => (typeof v === 'number' ? Math.round(v * 1e5) / 1e5 : v));
  if (s.length > max) s = s.slice(0, max) + '…(truncated)';
  return s;
}

async function execTool(name, args) {
  if (!API) throw new Error('Agent API not ready');
  switch (name) {
    case 'get_state': return { data: dataSummary(120), target: nodeOf('dataset')?.cfg.target, pipeline: pipelineSummary(), results: leaderboard() };
    case 'update_block': return API.updateBlock(args.block, args.settings || {});
    case 'add_block': return API.addBlock(args.type, args.model_key, args.settings);
    case 'remove_block': return API.removeBlock(args.block);
    case 'run_pipeline': return API.runPipeline();
    case 'get_leaderboard': return leaderboard(args.metric) || { error: 'No results yet. Run the pipeline first.' };
    case 'get_model_details': return API.modelDetails(args.model_id);
    case 'predict': return API.predict(args.model_id, args.rows);
    case 'what_if': return API.whatIf(args.model_id, args.row, args.changes);
    case 'explain_model': return API.explain(args.model_id, args.kind, args);
    case 'run_python': return API.runPython(args.code);
    case 'show_in_dashboard': return API.show(args.tab, args.model_id);
    case 'get_test_rows': return API.testRows(args.start || 0, Math.min(50, args.count || 10));
    case 'export': return API.exportThing(args.what, args.model_id);
    case 'diagnose': return API.diagnose();
    case 'apply_fixes': return API.applyFixes(args.issue_ids || ['all']);
    case 'retrain_models': return API.retrainModels(Array.isArray(args.model_ids) ? args.model_ids : [args.model_ids]);
    case 'list_code_cells': return API.listCells();
    case 'write_code_cell': return API.writeCell(args.cell_id, args.code, args.run);
    default: throw new Error('Unknown tool ' + name);
  }
}

function parseArgs(a) { if (typeof a === 'string') { try { return JSON.parse(a); } catch { return { code: a }; } } return a || {}; }

// Runs one user turn: streams the reply, executes tool calls (asking for approval if configured) and loops.
export async function runAgent(thread, text, { focus = null, onUpdate = () => {}, signal } = {}) {
  thread.items.push({ role: 'user', content: text });
  thread.api.push({ role: 'user', content: text });
  let useTools = thread.noTools ? false : true;
  for (let round = 0; round < 10; round++) {
    const item = { role: 'assistant', content: '', thinking: '', tools: [], pending: true };
    thread.items.push(item);
    onUpdate();
    const history = thread.api.slice(-40);
    while (history.length && history[0].role === 'tool') history.shift();
    const messages = [{ role: 'system', content: systemPrompt(focus) + (useTools ? '' : '\n\nTools are not available with this model. To run Python, reply with a ```python code block; the user can press Run.') }, ...history];
    let res;
    try {
      res = await chat({ messages, tools: useTools ? TOOLS : null, signal, onDelta: o => { item.content = o.content; item.thinking = o.thinking; onUpdate(); } });
    } catch (e) {
      if (useTools && /does not support tools|tool/i.test(e.message) && /support/i.test(e.message)) {
        thread.items.pop(); useTools = false; thread.noTools = true;
        thread.items.push({ role: 'note', content: `${AI.model} does not support tool calling, so the assistant will answer with code you can run. Pick gpt-oss or qwen3 models for full control.` });
        round--; continue;
      }
      item.pending = false; item.error = e.name === 'AbortError' ? 'Stopped.' : e.message; onUpdate();
      return;
    }
    item.pending = false;
    item.content = res.content; item.thinking = res.thinking;
    const calls = res.tool_calls || [];
    thread.api.push({ role: 'assistant', content: res.content || '', ...(calls.length ? { tool_calls: calls } : {}) });
    if (!calls.length) { onUpdate(); return; }
    for (const call of calls) {
      const name = call.function?.name, args = parseArgs(call.function?.arguments);
      const t = { name, args, status: 'running', result: null };
      item.tools.push(t);
      if (AI.confirm && MUTATING.has(name)) {
        t.status = 'awaiting'; onUpdate();
        const ok = await new Promise(resolve => { t.resolve = resolve; signal?.addEventListener('abort', () => resolve(false)); });
        delete t.resolve;
        if (!ok) { t.status = 'denied'; thread.api.push({ role: 'tool', tool_name: name, content: '{"denied": "The user did not approve this action."}' }); onUpdate(); continue; }
        t.status = 'running';
      }
      onUpdate();
      try {
        const out = await execTool(name, args);
        t.status = out && out.error ? 'error' : 'done';
        t.result = out;
        thread.api.push({ role: 'tool', tool_name: name, content: compact(stripForModel(name, out)) });
      } catch (e) {
        t.status = 'error'; t.result = { error: e.message };
        thread.api.push({ role: 'tool', tool_name: name, content: compact({ error: e.message }) });
      }
      onUpdate();
      if (signal?.aborted) return;
    }
  }
  thread.items.push({ role: 'note', content: 'Stopped after 10 tool rounds.' });
  onUpdate();
}
function stripForModel(name, out) {
  if ((name === 'run_python' || name === 'write_code_cell') && out) return { ...out, figures: out.figures ? `${out.figures.length} figure(s) shown to the user` : undefined };
  return out;
}

// Code blocks from models without tool support.
export async function runCodeBlock(code) { return API.runPython(code); }

// "AI Powered" block configuration: returns { explanation, settings }.
export async function blockAI(n, instruction) {
  const fields = n.type === 'model' ? MODEL[n.key].params.map(p => ({ k: p.k, t: p.t, o: p.o })) : (BLOCKS[n.type].fields || []).map(f => ({ k: f.k, type: f.f, options: f.o, min: f.min, max: f.max }));
  const target = nodeOf('dataset')?.cfg.target;
  const sys = `You configure one block of a scikit-learn pipeline. Reply with JSON only, in the form {"explanation": "<one or two sentences in the user's language>", "settings": {<only settings to change>}}.
Block: ${nodeTitle(n)} (${n.type}${n.type === 'model' ? ', key ' + n.key : ''}).
Allowed settings: ${JSON.stringify(fields)}.
Current settings: ${JSON.stringify(n.cfg)}.
${dataSummary(80)}
Target: ${target}${colInfo(target) ? ` (${colInfo(target).kind})` : ''}.
${S.results ? 'Latest results: ' + resultsSummary().slice(0, 2500) : ''}`;
  const res = await chat({ messages: [{ role: 'system', content: sys }, { role: 'user', content: instruction }], format: 'json', stream: false });
  let j;
  try { j = JSON.parse(res.content.replace(/^```(json)?|```$/g, '').trim()); } catch { throw new Error('The model did not return valid JSON. Try again or pick another model.'); }
  return { explanation: j.explanation || '', settings: j.settings || {} };
}
export { optLabel };
