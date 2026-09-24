/* Classic Web Worker that hosts Pyodide + scikit-learn and the ML Agent engine (engine.py). */
const PYODIDE_VERSION = '0.27.7';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
let py = null, engine = null, ready = null;
const post = m => self.postMessage(m);
const status = (s, message, extra = {}) => post({ type: 'status', status: s, message, ...extra });

async function boot() {
  status('loading', 'Downloading Python runtime (first visit ≈ 30–40 MB, cached afterwards)…');
  importScripts(INDEX_URL + 'pyodide.js');
  py = await self.loadPyodide({ indexURL: INDEX_URL });
  status('loading', 'Loading numpy, pandas, SciPy and scikit-learn…');
  await py.loadPackage(['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib'], { messageCallback: () => {} });
  const optional = {};
  for (const p of ['xgboost', 'lightgbm']) {
    status('loading', `Loading ${p}…`);
    try { await py.loadPackage([p], { messageCallback: () => {} }); optional[p] = true; } catch { optional[p] = false; }
  }
  status('loading', 'Starting the ML engine…');
  const src = await (await fetch(new URL('./engine.py', self.location.href))).text();
  py.FS.mkdirTree('/home/pyodide/app');
  py.FS.writeFile('/home/pyodide/app/engine.py', src);
  py.runPython('import sys\nsys.path.insert(0, "/home/pyodide/app")\nimport engine');
  engine = py.pyimport('engine');
  engine.set_progress(msg => post({ type: 'progress', data: msg }));
  const versions = JSON.parse(engine.versions());
  status('ready', 'Python engine ready', { versions, optional });
  // matplotlib is only needed for custom code; fetch it quietly in the background.
  py.loadPackage(['matplotlib'], { messageCallback: () => {} }).catch(() => {});
}

self.onmessage = async e => {
  const { id, fn, args } = e.data;
  try {
    ready ||= boot();
    await ready;
    if (fn === '__ping') return post({ type: 'result', id, result: '{}' });
    if (fn === 'run_code') { try { await py.loadPackagesFromImports(args[0] + '\nimport matplotlib'); } catch { /* optional */ } }
    const f = engine[fn];
    if (!f) throw new Error('Unknown engine function ' + fn);
    const result = f(...args);
    post({ type: 'result', id, result: typeof result === 'string' ? result : JSON.stringify(result ?? null) });
  } catch (err) {
    post({ type: 'result', id, result: JSON.stringify({ error: String(err && err.message || err) }) });
    if (!engine) status('error', 'The Python engine failed to start: ' + (err && err.message || err));
  }
};
