// Plotly helpers with theme-aware layout.
import { h } from './util.js';
export const PALETTE = ['#3d6df5', '#22c55e', '#f59e0b', '#a855f7', '#ec4899', '#06b6d4', '#ef4444', '#84cc16', '#f97316', '#14b8a6', '#6366f1', '#eab308'];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
export function plot(el, data, layout = {}, config = {}) {
  if (!window.Plotly) { el.innerHTML = '<div class="empty">Charts need Plotly, which could not be loaded (offline?).</div>'; return; }
  const text = css('--muted') || '#999', line = css('--line') || '#333';
  const ax = a => ({ gridcolor: line, zerolinecolor: line, linecolor: line, automargin: true, ...a });
  const L = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: text, family: 'Manrope, system-ui, sans-serif', size: 11 },
    margin: { l: 50, r: 14, t: 16, b: 42 }, colorway: PALETTE, hoverlabel: { font: { family: 'Manrope, system-ui' } },
    legend: { orientation: 'h', y: -0.2, font: { size: 10.5 } }, ...layout,
    xaxis: ax(layout.xaxis), yaxis: ax(layout.yaxis),
  };
  if (layout.yaxis2) L.yaxis2 = ax(layout.yaxis2);
  window.Plotly.react(el, data, L, { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'], ...config });
}
export function chartBox(cls = 'chart') { return h('div', { class: cls }); }
export function resizeAll(root) { if (window.Plotly) for (const el of root.querySelectorAll('.js-plotly-plot')) window.Plotly.Plots.resize(el); }
