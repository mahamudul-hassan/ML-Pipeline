// Promise-based RPC to the Pyodide worker. A custom transport can be injected (used by tests).
export class Engine {
  constructor({ onStatus = () => {}, onProgress = () => {}, transport = null } = {}) {
    this.onStatus = onStatus; this.onProgress = onProgress; this.transport = transport;
    this.seq = 0; this.pending = new Map(); this.w = null;
  }
  start() {
    if (this.transport) { this.transport.onmessage = m => this._msg(m); this.transport.start?.(); return; }
    this.w = new Worker(new URL('./engine.worker.js', import.meta.url));
    this.w.onmessage = e => this._msg(e.data);
    this.w.onerror = e => this.onStatus({ status: 'error', message: 'Engine worker error: ' + (e.message || 'unknown') });
    this.call('__ping').catch(() => {});
  }
  _msg(m) {
    if (m.type === 'status') return this.onStatus(m);
    if (m.type === 'progress') { try { this.onProgress(JSON.parse(m.data)); } catch { /* ignore */ } return; }
    if (m.type === 'result') {
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      let r;
      try { r = JSON.parse(m.result); } catch { r = { error: 'Invalid engine response' }; }
      if (r && typeof r === 'object' && !Array.isArray(r) && typeof r.error === 'string') { const err = new Error(r.error); err.trace = r.trace; p.reject(err); } else p.resolve(r);
    }
  }
  call(fn, ...args) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const msg = { id, fn, args };
      if (this.transport) this.transport.send(msg); else this.w.postMessage(msg);
    });
  }
  // Stopping a long computation means terminating the worker; state is lost and the engine restarts.
  restart() {
    for (const p of this.pending.values()) p.reject(new Error('Stopped'));
    this.pending.clear();
    if (this.w) this.w.terminate();
    if (this.transport) { this.transport.restart?.(); return; }
    this.start();
  }
}
