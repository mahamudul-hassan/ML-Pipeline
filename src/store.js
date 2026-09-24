// Shared application state and a tiny event bus.
export const S = {
  profile: null, dataName: null, dataText: null, dataSize: 0,
  nodes: [], edges: [], sel: null,
  results: null, running: false, progress: null, runStart: 0, lastError: null,
  engine: { status: 'loading', message: 'Starting Python engine…', versions: null, available: {} },
  chat: [], docs: [], instructions: '', history: [],
  ui: { dashTab: 'overview', model: null, compare: [], dataTab: 'preview', cfgTab: 'config', lbSort: null, lbSplit: 'cv' },
  settings: { name: 'Mahamudul Hassan Siddique', theme: 'dark', maxTrainRows: 0, slowModelRows: 3000 },
};
const L = {};
export const on = (ev, fn) => { (L[ev] ||= []).push(fn); };
export const emit = (ev, data) => { for (const fn of L[ev] || []) { try { fn(data); } catch (e) { console.error(ev, e); } } };
export const node = id => S.nodes.find(n => n.id === id);
export const nodesOf = t => S.nodes.filter(n => n.type === t);
export const nodeOf = t => S.nodes.find(n => n.type === t);
export const columns = () => (S.profile ? S.profile.cols.map(c => c.name) : []);
export const colInfo = name => S.profile?.cols.find(c => c.name === name);
export const store = {
  get(k, d) { try { const v = localStorage.getItem('mlagent.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('mlagent.' + k, JSON.stringify(v)); return true; } catch { return false; } },
  del(k) { try { localStorage.removeItem('mlagent.' + k); } catch { /* ignore */ } },
};
