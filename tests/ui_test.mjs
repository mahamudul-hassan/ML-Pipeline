// End-to-end UI test: real index.html + modules in jsdom, real Python engine over a pipe, mocked Ollama.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import readline from 'node:readline';
const require = createRequire(process.env.JSDOM_PATH || '/home/claude/mla/');
const { JSDOM } = require('jsdom');
const root = new URL('..', import.meta.url);
const html = fs.readFileSync(new URL('index.html', root), 'utf8').replace(/<script[^>]*><\/script>/g, '');
const dom = new JSDOM(html, { url: 'http://localhost:3000/', pretendToBeVisual: true });
const w = dom.window;
for (const k of ['window', 'document', 'localStorage', 'sessionStorage', 'getComputedStyle', 'requestAnimationFrame', 'cancelAnimationFrame', 'HTMLElement', 'Node', 'CustomEvent']) globalThis[k] = k === 'window' ? w : typeof w[k] === 'function' && !/^[A-Z]/.test(k) ? w[k].bind(w) : w[k];
w.HTMLElement.prototype.scrollIntoView = () => {};
w.document.elementFromPoint = () => null;
const errors = [];
const origErr = console.error;
console.error = (...a) => { errors.push(a.map(String).join(' ')); origErr(...a); };
let plots = 0;
w.Plotly = { react: (el, data) => { plots++; if (!Array.isArray(data)) throw new Error('bad plot data'); for (const t of data) if (!t.type) throw new Error('trace without type'); el.classList.add('js-plotly-plot'); }, Plots: { resize() {} } };
const downloads = [];
w.URL.createObjectURL = () => 'blob:x'; w.URL.revokeObjectURL = () => {};
w.HTMLAnchorElement.prototype.click = function () { downloads.push(this.download); };
const zipped = {};
w.JSZip = class { file(n, d) { zipped[n] = d; } async generateAsync() { return new Blob(['zip']); } };

// Python engine over stdin/stdout
const py = spawn('python3', [new URL('tests/rpc_server.py', root).pathname], { stdio: ['pipe', 'pipe', 'inherit'] });
const queue = [];
const transport = { onmessage: null, send: m => py.stdin.write(JSON.stringify(m) + '\n'), start() { for (const m of queue.splice(0)) this.onmessage(m); } };
readline.createInterface({ input: py.stdout }).on('line', l => { const m = JSON.parse(l); transport.onmessage ? transport.onmessage(m) : queue.push(m); });
w.__ENGINE_TRANSPORT__ = transport;

// Mock Ollama: scripted tool-calling conversation
const calls = [];
let script = [];
globalThis.fetch = async (url, opts = {}) => {
  const u = String(url);
  if (u.startsWith('file:')) return new Response(fs.readFileSync(new URL(u)));
  if (u.includes('/api/ollama/chat')) {
    const body = JSON.parse(opts.body); calls.push(body);
    if (opts.headers['x-ollama-key'] !== 'test-key') return new Response('{"error":"unauthorized"}', { status: 401 });
    const step = script.shift() || { content: 'Done.' };
    const lines = step.tool_calls ? [{ message: { role: 'assistant', content: '', tool_calls: step.tool_calls }, done: false }, { done: true }] : [...step.content.match(/.{1,12}/g)].map(c => ({ message: { role: 'assistant', content: c }, done: false })).concat([{ done: true }]);
    return new Response(lines.map(l => JSON.stringify(l)).join('\n') + '\n', { status: 200 });
  }
  if (u.includes('/api/ollama/models')) return new Response(JSON.stringify({ models: [{ name: 'gpt-oss:120b' }, { name: 'qwen3-coder-next' }] }));
  throw new Error('unexpected fetch ' + u);
};

const until = async (fn, ms = 600000, label = '') => { const t0 = Date.now(); while (!fn()) { if (Date.now() - t0 > ms) throw new Error('timeout ' + label); await new Promise(r => setTimeout(r, 50)); } };
const flush = () => new Promise(r => setTimeout(r, 60));
const ok = (c, m) => { if (!c) { console.log('FAIL', m); process.exitCode = 1; } else console.log('ok  ', m); };

localStorage.setItem('mlagent.settings', JSON.stringify({ maxTrainRows: 1500, slowModelRows: 800, theme: 'dark', name: 'Mahamudul Hassan Siddique' }));
const { S } = await import(new URL('src/store.js', root));
const { A } = await import(new URL('src/actions.js', root));
await import(new URL('src/main.js', root));
await until(() => S.profile, 60000, 'data');
ok(S.nodes.length === 13, `default pipeline has ${S.nodes.length} blocks: ${S.nodes.map(n => n.key || n.type).join(', ')}`);
ok(document.querySelectorAll('.node').length === S.nodes.length, 'canvas renders all nodes');
await until(() => S.preview, 20000, 'preview');
ok(document.querySelector('#dsBody table') !== null, 'dataset preview table renders');
// config panel for every block
for (const n of S.nodes) { A.run.name; (await import(new URL('src/canvas.js', root))).select(n.id); await flush(); ok(document.querySelector('#cfgBody .cfgHead') !== null, `config panel renders for ${n.key || n.type}`); }

const t0 = Date.now();
const lb = await A.run();
ok(!lb.error && S.results, `run finished in ${((Date.now() - t0) / 1000).toFixed(1)}s: ${lb.error || lb.rows.map(r => `${r.name}=${r[lb.ranked_by]}`).join(', ')}`);
ok(S.nodes.filter(n => n.type === 'model').every(n => n.status === 'done' && n.metric), 'model nodes show status + metric');
const tabs = ['overview', 'leaderboard', 'model', 'compare', 'tuning', 'explain', 'data', 'predict', 'logs', 'runs'];
const { renderDashboard } = await import(new URL('src/dashboard.js', root));
for (const tab of tabs) {
  S.ui.dashTab = tab; plots = 0; renderDashboard(); await flush();
  if (tab === 'data') { await until(() => S.eda, 20000, 'eda'); await flush(); await flush(); }
  ok(document.querySelector('#dashBody .dashBody, #dashBody .empty') && errors.length === 0, `dashboard tab ${tab}: ${plots} charts, errors ${errors.length}`);
}
for (const id of Object.keys(S.results.models).filter(id => S.results.models[id].status === 'ok')) { S.ui.dashTab = 'model'; S.ui.model = id; plots = 0; renderDashboard(); await flush(); ok(errors.length === 0 && plots >= 4, `model details ${id}: ${plots} charts`); }
const best = S.results.best;
await A.analyze('shap', S.results.ranking[1]); S.ui.dashTab = 'explain'; S.ui.model = S.results.ranking[1]; renderDashboard(); await flush();
ok(errors.length === 0 && S.results.models[S.results.ranking[1]].shap, 'SHAP computed for a non-best model');
const pdp = await A.analyze('pdp', best, { feature: 'study_hours' }, false); ok(pdp.mean.length > 3, 'PDP');
await A.exportThing('project_zip');
ok(['engine.py', 'train.py', 'serve.py', 'model.joblib', 'data.csv', 'leaderboard.csv', 'pipeline_config.json', 'requirements.txt'].every(f => f in zipped), 'project zip has all files: ' + Object.keys(zipped).join(', '));
fs.writeFileSync('/tmp/export_model.joblib', zipped['model.joblib']);
fs.writeFileSync('/tmp/export_cfg.json', zipped['pipeline_config.json']);
await A.exportThing('test_predictions'); await A.exportThing('pipeline_json');
ok(downloads.length >= 3, 'downloads: ' + downloads.join(', '));

// Agent with tools
const { saveAI } = await import(new URL('src/ai.js', root));
saveAI({ apiKey: 'test-key', model: 'gpt-oss:120b', confirm: false, provider: 'cloud' });
script = [
  { tool_calls: [{ function: { name: 'get_leaderboard', arguments: {} } }] },
  { tool_calls: [{ function: { name: 'update_block', arguments: { block: 'rf', settings: { n_estimators: 50, max_depth: 8, bogus: 1 } } } }, { function: { name: 'add_block', arguments: { type: 'model', model_key: 'lgbm' } } }] },
  { tool_calls: [{ function: { name: 'run_python', arguments: { code: "from sklearn.ensemble import ExtraTreesClassifier\nr = register_model('Extra Trees 300', ExtraTreesClassifier(n_estimators=60))\nimport matplotlib.pyplot as plt\nplt.hist(y_test)\nprint(r['test']['accuracy'])\npd.DataFrame(results[best_model_id]['test'], index=[0])" } } }] },
  { tool_calls: [{ function: { name: 'predict', arguments: { rows: [{ study_hours: 9, attendance: 95 }] } } }, { function: { name: 'what_if', arguments: { row: 0, changes: { study_hours: 0 } } } }, { function: { name: 'explain_model', arguments: { model_id: best, kind: 'local_shap', row: 2 } } }, { function: { name: 'get_model_details', arguments: { model_id: best } } }] },
  { content: 'The **best model** is ready. Here is a summary.' },
];
await A.askAI('Improve the random forest and add LightGBM');
await until(() => S.chat.items.at(-1)?.role === 'assistant' && !S.chat.items.at(-1).pending && S.chat.items.at(-1).content.includes('summary'), 120000, 'agent');
const tools = S.chat.items.flatMap(i => i.tools || []);
ok(tools.length === 8 && tools.every(t => t.status === 'done'), 'agent tools: ' + tools.map(t => `${t.name}:${t.status}${t.result?.error ? '(' + t.result.error + ')' : ''}`).join(', '));
ok(S.nodes.some(n => n.key === 'lgbm') && S.nodes.find(n => n.key === 'rf').cfg.n_estimators === 50, 'agent changed the pipeline');
ok(tools[3].result.figures.length === 1 && tools[3].result.table && S.results.models.custom_extra_trees_300, 'run_python registered a model + figure + table');
ok(calls[1].messages.some(m => m.role === 'tool' && m.tool_name === 'get_leaderboard'), 'tool results sent back to the model');
ok(calls[0].tools.length === 14 && calls[0].messages[0].content.includes('Leaderboard'), 'system prompt + 14 tools sent');
ok(document.querySelectorAll('#chatBox .tool').length === 8, 'tool cards rendered in chat');
// Approval flow
saveAI({ confirm: true });
script = [{ tool_calls: [{ function: { name: 'remove_block', arguments: { block: 'lgbm' } } }] }, { content: 'Removed.' }];
A.askAI('remove lightgbm');
await until(() => S.chat.items.at(-1)?.tools?.[0]?.status === 'awaiting', 20000, 'approval');
document.querySelector('#chatBox .approve .btn.primary').click();
await until(() => S.chat.items.at(-1)?.content === 'Removed.', 20000, 'after approval');
ok(!S.nodes.some(n => n.key === 'lgbm'), 'approved tool ran');

// Regression dataset through a template
await A.loadSample('diabetes');
ok(S.nodes.some(n => n.key === 'ridge') && S.nodes.some(n => n.key === 'svr') && !S.nodes.some(n => n.key === 'logreg'), 'customised pipeline adapted to regression: ' + S.nodes.filter(n => n.type === 'model').map(n => n.key).join(', '));
const lb2 = await A.run();
ok(!lb2.error && S.results.task === 'regression', `regression run: ${lb2.error || lb2.rows.slice(0, 4).map(r => `${r.name}=${r[lb2.ranked_by]}`).join(', ')}`);
for (const tab of tabs) { S.ui.dashTab = tab; S.ui.model = null; renderDashboard(); await flush(); if (tab === 'data') { await until(() => S.eda, 20000); await flush(); } }
ok(errors.length === 0, 'regression dashboard renders without errors');
ok(S.history.length === 2, 'runs history');
console.log(errors.length ? 'ERRORS:\n' + errors.join('\n') : 'no console errors');
py.kill();
process.exit(process.exitCode || 0);
