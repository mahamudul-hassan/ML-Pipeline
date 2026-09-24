// Chat UI shared by the main assistant panel and the per-block AI tab.
import { S, emit } from './store.js';
import { AI, aiReady, runAgent, runCodeBlock } from './ai.js';
import { h, md, esc, toast } from './util.js';
import { icon } from './icons.js';
import { A } from './actions.js';

const TOOL_LABEL = { diagnose: 'Pipeline Doctor', apply_fixes: 'Apply fixes', retrain_models: 'Retrain models', list_code_cells: 'Read code cells', write_code_cell: 'Write & run code cell', get_state: 'Read pipeline state', update_block: 'Update block', add_block: 'Add block', remove_block: 'Remove block', run_pipeline: 'Run pipeline', get_leaderboard: 'Read leaderboard', get_model_details: 'Read model details', predict: 'Predict', what_if: 'What-if analysis', explain_model: 'Explain model', run_python: 'Run Python', show_in_dashboard: 'Open dashboard', get_test_rows: 'Read test rows', export: 'Export' };

export function newThread() { return { items: [], api: [] }; }

function argSummary(t) {
  const a = t.args || {};
  if (t.name === 'update_block') return `${a.block}: ${Object.keys(a.settings || {}).join(', ')}`;
  if (t.name === 'add_block') return a.model_key || a.type;
  if (t.name === 'explain_model') return `${a.kind} · ${a.model_id}`;
  if (t.name === 'get_model_details' || t.name === 'predict') return a.model_id || '';
  if (t.name === 'show_in_dashboard') return a.tab;
  if (t.name === 'export') return a.what;
  return '';
}
export function pythonOutput(r) {
  const box = h('div', {});
  if (!r) return box;
  if (r.stdout) box.append(h('pre', {}, r.stdout));
  if (r.result) box.append(h('pre', {}, '→ ' + r.result));
  if (r.table) {
    const t = r.table;
    box.append(h('div', { class: 'scrollx' }, h('table', { class: 'tbl' }, h('tr', {}, h('th', {}, ''), t.columns.map(c => h('th', {}, c))), t.rows.slice(0, 30).map((row, i) => h('tr', {}, h('td', { class: 'muted' }, t.index[i]), row.map(v => h('td', {}, v == null ? '' : typeof v === 'number' ? +v.toFixed(5) : String(v))))))));
  }
  for (const f of r.figures || []) box.append(h('img', { src: 'data:image/png;base64,' + f, alt: 'Figure from Python code' }));
  if (r.error_info) box.append(h('div', { class: 'note warn' }, h('b', {}, `${r.error_info.type}${r.error_info.line ? ` on line ${r.error_info.line}` : ''}: `), r.error_info.message, r.error_info.code_line ? h('pre', { style: 'margin:6px 0' }, r.error_info.code_line) : null, h('div', { class: 'xs' }, 'Hint: ' + r.error_info.hint)));
  if (r.exception) box.append(h('details', {}, h('summary', { class: 'xs muted' }, 'Full traceback'), h('pre', { class: 'err' }, r.exception)));
  if (r.warnings?.length) box.append(h('div', { class: 'note' }, h('b', {}, 'Review: '), r.warnings.join(' ')));
  if (r.error) box.append(h('pre', { class: 'err' }, r.error));
  if (r.results_changed) box.append(h('div', { class: 'xs okc' }, 'Leaderboard and dashboard updated.'));
  return box;
}
function toolCard(t, rerender) {
  const st = { running: 'running…', done: 'done', error: 'error', awaiting: 'needs approval', denied: 'not approved' }[t.status] || t.status;
  const d = h('details', { class: 'tool ' + (t.status === 'awaiting' ? 'pending' : t.status), open: t.status === 'awaiting' || (['run_python', 'write_code_cell', 'diagnose'].includes(t.name) && t.status !== 'denied') || t.status === 'error' },
    h('summary', {}, h('span', { html: icon(['run_python', 'write_code_cell'].includes(t.name) ? 'code' : t.name === 'run_pipeline' || t.name === 'retrain_models' ? 'play' : t.name === 'diagnose' ? 'alert' : 'wand', 14) }), `${TOOL_LABEL[t.name] || t.name}`, h('span', { class: 'muted', style: 'font-weight:500' }, argSummary(t)), h('span', { class: 'sp' }), h('span', { class: 'xs muted' }, st)));
  const body = h('div', { class: 'tb' });
  if (t.name === 'run_python' || t.name === 'write_code_cell') body.append(h('pre', {}, (t.args.cell_id ? `# cell ${t.args.cell_id}\n` : '') + (t.args.code || '')));
  else if (Object.keys(t.args || {}).length) body.append(h('pre', {}, JSON.stringify(t.args, null, 1).slice(0, 1500)));
  if (t.status === 'awaiting') {
    body.append(h('div', { class: 'approve' }, ['run_python', 'write_code_cell'].includes(t.name) ? 'Run this code in the Python engine?' : t.name === 'run_pipeline' ? 'Train all models now?' : t.name === 'retrain_models' ? 'Retrain these models now?' : 'Apply this change to the pipeline?',
      h('div', { class: 'mact' }, h('button', { class: 'btn sm primary', onclick: () => t.resolve && t.resolve(true) }, 'Allow'), h('button', { class: 'btn sm', onclick: () => t.resolve && t.resolve(false) }, 'Deny'),
        h('label', { class: 'chk xs', style: 'margin:0 0 0 6px' }, h('input', { type: 'checkbox', onchange: e => { if (e.target.checked) { AI.confirm = false; emit('ai-settings'); t.resolve && t.resolve(true); } } }), 'Always allow'))));
  }
  if (t.result) {
    if (t.name === 'run_python' || t.name === 'write_code_cell') body.append(pythonOutput(t.result));
    else if (t.name === 'diagnose' && t.result.issues) body.append(h('ul', { class: 'insights' }, t.result.issues.slice(0, 12).map(i => h('li', {}, h('b', { class: i.severity === 'error' ? 'err' : i.severity === 'warning' ? 'warnc' : 'muted' }, i.severity + ' '), i.title))));
    else if (t.result.error) body.append(h('pre', { class: 'err' }, t.result.error));
    else if (t.name !== 'get_state') body.append(h('pre', {}, JSON.stringify(t.result, (k, v) => (typeof v === 'number' ? Math.round(v * 1e4) / 1e4 : v), 1).slice(0, 2500)));
  }
  d.append(body);
  return d;
}

export class ChatView {
  constructor(root, getThread, { focus = () => null, chips = [], placeholder = 'Ask anything about your data, pipeline or trained models…', compact = false } = {}) {
    this.root = root; this.getThread = getThread; this.focus = focus; this.chips = chips; this.compact = compact;
    this.ctrl = null; this.raf = 0;
    root.innerHTML = '';
    this.list = h('div', { class: 'msgs', 'aria-live': 'polite' });
    this.chipBox = h('div', { class: 'chips' });
    this.input = h('textarea', { placeholder, 'aria-label': 'Message the AI assistant', rows: 1 });
    this.sendBtn = h('button', { class: 'btn primary icon', 'aria-label': 'Send', html: icon('send', 16), onclick: () => this.onSend() });
    this.input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.onSend(); } });
    this.input.addEventListener('input', () => { this.input.style.height = '40px'; this.input.style.height = Math.min(140, this.input.scrollHeight) + 'px'; });
    root.append(this.list, this.chipBox, h('div', { class: 'chatIn' }, this.input, this.sendBtn));
    this.render();
  }
  get busy() { return !!this.ctrl; }
  onSend() { if (this.busy) { this.ctrl.abort(); return; } const t = this.input.value.trim(); if (!t) return; this.input.value = ''; this.input.style.height = '40px'; this.send(t); }
  async send(text) {
    const thread = this.getThread();
    if (!aiReady()) {
      thread.items.push({ role: 'user', content: text }, { role: 'note', content: 'Connect the AI first: open Settings → AI, paste your Ollama API key and pick a cloud model.', action: 'settings' });
      this.render(); return;
    }
    this.ctrl = new AbortController();
    this.sendBtn.innerHTML = icon('stop', 14); this.sendBtn.setAttribute('aria-label', 'Stop');
    try { await runAgent(thread, text, { focus: this.focus(), onUpdate: () => this.schedule(), signal: this.ctrl.signal }); }
    catch (e) { thread.items.push({ role: 'note', content: 'Error: ' + e.message }); }
    this.ctrl = null;
    this.sendBtn.innerHTML = icon('send', 16); this.sendBtn.setAttribute('aria-label', 'Send');
    this.render();
    emit('chat-done');
  }
  schedule() { if (this.raf) return; this.raf = requestAnimationFrame(() => { this.raf = 0; this.render(); }); }
  render() {
    const thread = this.getThread();
    const nearBottom = this.list.scrollHeight - this.list.scrollTop - this.list.clientHeight < 80;
    this.list.innerHTML = '';
    if (!thread.items.length) {
      this.list.append(this.msg('bot', h('div', {}, h('div', { class: 'who' }, 'ML Agent AI'), h('div', { class: 'bubble', html: md(this.compact ? 'Ask me to configure this block, explain its options, or change it for you.' : `Hi! I can build and change your pipeline, train and tune models, compare them, explain predictions and run Python on your trained models.\n\nTry: *"Add LightGBM and a stacking ensemble, then run"* or *"Which features drive the best model?"*`) }))));
    }
    for (const it of thread.items) {
      if (it.role === 'user') this.list.append(this.msg('user', h('div', { class: 'bubble' }, it.content)));
      else if (it.role === 'note') this.list.append(h('div', { class: 'note' + (it.action ? ' warn' : '') }, it.content, it.action === 'settings' ? h('div', { class: 'mact' }, h('button', { class: 'btn sm primary', onclick: () => emit('open-settings', 'ai') }, 'Open AI settings')) : null));
      else {
        const box = h('div', {}, h('div', { class: 'who' }, 'ML Agent AI', h('span', { class: 'xs muted', style: 'font-weight:500' }, AI.model)));
        if (it.thinking) box.append(h('details', {}, h('summary', { class: 'xs muted' }, 'Reasoning'), h('div', { class: 'think' }, it.thinking)));
        if (it.content) { const b = h('div', { class: 'bubble', html: md(it.content) }); this.decorateCode(b); box.append(b); }
        for (const t of it.tools) box.append(toolCard(t, () => this.render()));
        if (it.pending && !it.content && !it.tools.length) box.append(h('div', { class: 'bubble thinking' }, it.thinking ? 'Thinking…' : 'Working…'));
        if (it.error) box.append(h('div', { class: 'note warn' }, it.error));
        this.list.append(this.msg('bot', box));
      }
    }
    this.chipBox.innerHTML = '';
    if (!this.busy) for (const c of this.chips) this.chipBox.append(h('button', { class: 'chipBtn', onclick: () => this.send(c) }, c));
    if (nearBottom || this.busy) this.list.scrollTop = this.list.scrollHeight;
  }
  msg(who, content) { return h('div', { class: 'msg ' + who }, who === 'bot' ? h('span', { class: 'bav', html: icon('bot', 16) }) : null, content); }
  decorateCode(b) {
    for (const pre of b.querySelectorAll('pre[data-lang="python"],pre[data-lang="py"]')) {
      const out = h('div', {});
      const btn = h('button', { class: 'btn sm', html: icon('play', 11) + ' Run in engine', onclick: async () => {
        btn.disabled = true; out.innerHTML = '<div class="xs muted">Running…</div>';
        try {
          const r = await runCodeBlock(pre.textContent); out.innerHTML = ''; out.append(pythonOutput(r));
          if (r.exception || r.warnings?.length) out.append(h('div', { class: 'mact' }, h('button', { class: 'btn sm violet', html: icon('wand', 12) + ' Fix with AI', onclick: () => A.fixWithAI('code', { code: pre.textContent, error: r.error_info ? `${r.error_info.type} on line ${r.error_info.line}: ${r.error_info.message}` : r.warnings.join(' ') }) })));
        } catch (e) { out.innerHTML = `<pre class="err">${esc(e.message)}</pre>`; }
        btn.disabled = false;
      } });
      pre.after(h('div', { class: 'mact' }, btn), out);
    }
  }
}
export function clearThread(thread) { thread.items = []; thread.api = []; thread.noTools = false; }
export { toast };
