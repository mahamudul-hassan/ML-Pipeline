# ML Agent — AI-powered ML pipeline builder

An n8n-style visual builder for machine-learning pipelines. **Real scikit-learn, XGBoost and LightGBM run in the browser** (Pyodide in a Web Worker), and an **Ollama Cloud AI agent** can build, change, run, tune and explain the pipeline, and work with the trained models through tool calling.

It is a static site with two tiny serverless functions, so it deploys to Vercel for free with no ML server.

## Features

- **Canvas:** drag, connect and auto-layout blocks: Dataset → Validation & CV → Preprocessing → Feature Engineering → Feature Selection → models / Model Zoo / Voting / Stacking → Evaluation → Hyperparameter Tuning → Deployment. Every block has Manual and AI Powered configuration and its own AI chat.
- **48 models:** linear, robust and generalised linear, discriminant, naive Bayes, neighbours, SVM, kernel and Gaussian-process models, trees, random forest, extra trees, bagging, AdaBoost, gradient boosting, HistGradientBoosting, XGBoost, LightGBM and MLP, plus voting and stacking ensembles and a dummy baseline.
- **Preprocessing:** mean, median, most-frequent, constant, KNN and iterative (MICE) imputation. One-hot, ordinal, target and frequency encoding, with separate handling of high-cardinality columns. Standard, MinMax, Robust, MaxAbs, Normalizer, quantile and power scaling. log1p, Yeo-Johnson, Box-Cox and quantile transforms. Outlier clipping (IQR, z-score, percentile) or Isolation Forest row removal. Class weights, over-sampling, under-sampling or SMOTE, applied inside CV folds. Duplicate, constant and high-missing column removal.
- **Feature engineering:** custom pandas expressions, date parts, polynomial and interaction features, splines, binning.
- **Feature selection:** variance filter, correlation filter, SelectKBest (F-test, mutual information, chi²), SelectPercentile, RFE, RFECV, L1-based and tree-importance SelectFromModel, sequential forward and backward selection. Reduction with PCA, Truncated SVD, FastICA, Kernel PCA or LDA.
- **Validation:** train / validation / test split (stratified or time-ordered). K-fold, stratified, repeated, shuffle, time-series and group K-fold CV. Models are ranked by CV score, and the test set is kept for the final report.
- **Tuning:** **Optuna Bayesian optimisation** (TPE, random or quasi-Monte-Carlo sampler, optional time limit), a built-in Gaussian-process Bayesian optimiser (used automatically if Optuna cannot be installed), grid, random, halving-grid and halving-random search. Per-model search spaces accept lists or ranges such as `{"low": 0.01, "high": 10, "log": true}`. The dashboard shows the optimisation history, hyperparameter importance, parallel coordinates and every trial.
- **Dashboard for every model:**
  - Overview with KPIs and insights.
  - Sortable leaderboard across all metrics and splits.
  - Per-model details: confusion matrix, per-class report, ROC, PR, calibration, threshold, residual and Q-Q plots, CV folds, native and permutation importance, learning curve, PDP + ICE.
  - Comparison: box plots, heatmap, radar, ROC overlay.
  - Tuning: parallel coordinates and trials.
  - SHAP: bar, beeswarm, dependence, waterfall.
  - Data insights, a predict / what-if form, batch CSV predictions, logs and run history.
- **AI agent (Ollama):**
  - 14 tools: read state, update / add / remove blocks, run the pipeline, leaderboard, model details, predict, what-if, explanations, run Python on the trained models, open dashboard tabs, read test rows and export.
  - Pipeline changes and code need your approval (can be turned off).
  - `run_python` can register brand-new models into the leaderboard.
- **Export:** a Python project with `engine.py`, `pipeline_config.json`, `train.py`, `model.joblib`, a FastAPI `serve.py`, `predict.py`, `requirements.txt` and the results.

## Run locally

Requirements: Node 18+ (no npm install needed).

```bash
npm run dev          # or: node dev-server.mjs
# open http://localhost:3000
```

The first visit downloads the Python runtime and libraries (about 30–40 MB). After that the browser caches them.

## Push to GitHub

```bash
cd ml-agent
git init
git add .
git commit -m "ML Agent: visual ML pipeline builder"
git branch -M main
git remote add origin https://github.com/<your-username>/ml-agent.git
git push -u origin main
```

## Deploy on Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import the GitHub repository.
2. Framework preset: **Other**. Leave the build command and output directory empty.
3. Click **Deploy**. `index.html` is served as a static site, and `api/ollama/chat.js` and `api/ollama/models.js` become Edge Functions.

Or with the CLI: `npm i -g vercel && vercel --prod`.

Every `git push` to `main` redeploys automatically.

## Connect Ollama Cloud

1. Create a free API key at <https://ollama.com/settings/keys>.
2. In the app open **Settings → AI assistant**, paste the key and press **Refresh** to load the models available to your account.
3. Pick a model and press **Test connection**. `gpt-oss:120b` and `gpt-oss:20b` work well for tool calling. qwen3, kimi, glm, deepseek and minimax cloud models also support tools.

How the key is handled: it is stored only in your browser. By default it goes to localStorage; untick "Remember" to keep it only for the tab. It is sent in an `x-ollama-key` header to this app's own `/api/ollama/*` function, which forwards it to `https://ollama.com/api/chat` and never stores or logs it. The browser cannot call ollama.com directly because of CORS, and Ollama advises keeping keys out of browser code, hence the proxy.

Optional shared key: set `OLLAMA_API_KEY` and `ALLOW_SERVER_KEY=true` in the Vercel project's environment variables to let visitors use your key without entering one. Anyone who can open the site can then spend your quota, so only do this for private deployments (for example behind Vercel password protection).

**Local Ollama:** choose "Local Ollama" in Settings. Because the app then calls `http://localhost:11434` from the browser, start Ollama with your site allowed:

```bash
OLLAMA_ORIGINS="https://your-app.vercel.app,http://localhost:3000" ollama serve
```

## Project structure

```
index.html              layout
src/main.js             boot, run, agent API, settings, save/load, export
src/engine.py           scikit-learn engine (also used by the exported project)
src/engine.worker.js    Pyodide Web Worker hosting engine.py
src/engine-client.js    promise RPC to the worker
src/registry.js         blocks, settings schema and the 48 models
src/canvas.js           node editor
src/config-panel.js     block configuration (Manual / AI Powered / block chat)
src/dashboard.js        results dashboard (Plotly)
src/ai.js               Ollama client, system prompt, tools, agent loop
src/chat.js             chat UI with tool cards and approvals
src/data-panel.js       dataset details, documents and instructions
api/ollama/chat.js      Vercel Edge Function: streaming chat proxy
api/ollama/models.js    Vercel Edge Function: model list
dev-server.mjs          local static server + the same API functions
tests/                  engine tests (native Python) and an end-to-end UI test
```

## Tests

```bash
pip install scikit-learn pandas scipy xgboost lightgbm matplotlib joblib optuna
python3 tests/test_engine.py        # every model, every preprocessing / selection / CV option
node tests/ui_test.mjs              # needs jsdom (npm i jsdom); real engine + mocked Ollama
```

## Limits

- Training runs single-threaded WebAssembly in the browser. That is usually 2–5× slower than native Python, and there is roughly a 2–4 GB memory ceiling. Datasets up to a few hundred thousand cells train quickly.
- Slow models (SVM, Gaussian processes, kernel ridge, Theil-Sen, ARD) train on at most 3,000 rows by default. You can cap all training rows in Settings.
- For big data, export the project and run `train.py` natively; it uses the same engine and gives the same results.
- **Stop** terminates the Python worker and restarts it, so models trained in that session are cleared.
- `model.joblib` is pickled with the browser's scikit-learn version (pinned in `requirements.txt`).
- Optuna is installed from PyPI with micropip the first time an Optuna search runs (a few MB, then cached).
- The XGBoost bundled with Pyodide (2.1.2) predates scikit-learn 1.6's estimator tags; `engine.fix_sklearn_tags` patches it at load time, so XGBoost works on its own and inside voting / stacking ensembles.
- SHAP values are computed with a model-agnostic permutation estimator on the original columns, so they are approximations.

## License

MIT
