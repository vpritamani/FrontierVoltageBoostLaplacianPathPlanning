'use strict';

/* ===================== state ===================== */

const S = {
  registry: [],
  system: { cpu_count: 1, torch: false, cuda: false, cuda_device: null },
  mapSets: [],
  runs: [],
  jobs: [],
  tab: 'run',
  openRunId: null,   // run detail view when set
  runDetail: null,
  pollTimer: null,
};

/* Run-form state — survives tab switches and re-renders. */
const F = {
  source: 'new',                    // 'new' | 'existing'
  gen: { name: '', mode: '2', width: 100, height: 100, depth: 30,
         shape: '12,12,12,8',
         num_maps: 5, obstacles_min: 5, obstacles_max: 30, seed: '' },
  mapSetId: null,
  mapSel: null,                     // null = all maps, else Set of map names
  algoList: [],                     // [{id, params}] — instances, duplicates allowed
  addForm: null,                    // {id, params} being configured
  runName: '',
  timeout: 120,
  workers: 1,                       // parallel solves (each in its own subprocess)
};

const METRIC_OPTIONS = [
  { key: 'steering_penalty', label: 'Steering penalty @ θ (recommended)', src: 'metrics', steer: true },
  { key: 'time_s', label: 'Solve time (s)', src: 'rec' },
  { key: 'path_length_euclidean', label: 'Path length (Euclidean)', src: 'rec' },
  { key: 'path_steps', label: 'Path steps', src: 'rec' },
  { key: 'pathbench', label: 'PathBench smoothness', src: 'metrics' },
  { key: 'angle_change', label: 'Angle change / length', src: 'metrics' },
  { key: 'discrete', label: 'Discrete curvature²', src: 'metrics' },
  { key: 'deriv_v1_total', label: 'Derivative total (v1)', src: 'metrics' },
  { key: 'deriv_v2_total', label: 'Derivative total (v2)', src: 'metrics' },
];
let compareMetric = 'steering_penalty';

/* Steering θ used by comparison views (run detail + compare tab + graph). */
const STEER = { theta: 30, sweep: 5 };

/* Cache of recomputed steering values: `${runId}|${theta}|${sweep}` -> results[] */
const steerCache = new Map();

/* Metrics-lab state: preview holds an unsaved recompute for the open run. */
const LAB = { theta: 30, sweep: 5, preview: null, runId: null };

/* Cross-run compare state. */
const CMP = { sel: new Set(), metric: 'steer|30|5', cache: new Map(),
              graph: { mode: 'scatter', x: 'mean_time_s', y: 'steer|30|5', z: 'none' },
              // manual axis limits ('' = autofit) — editable + wheel-zoomable
              limits: { xmin: '', xmax: '', ymin: '', ymax: '', zmin: '', zmax: '' },
              // hyperparameter sweep panel state (param2 = optional 3rd dimension)
              hp: { algo: '', param: '', param2: 'none', y: 'steer|30|5',
                    limits: { xmin: '', xmax: '', ymin: '', ymax: '', zmin: '', zmax: '' },
                    view: { yaw: 0.7, pitch: 0.45, zoom: 1.0 } },
              lastRows: [] };

/* Decoded grid cache for viewers: key "msId/mapName" -> {shape, data}. */
const GRIDS = new Map();

const TERMINAL = ['done', 'error', 'interrupted', 'cancelled'];

/* ===================== helpers ===================== */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmt(v, digits = 3) {
  if (v === null || v === undefined || v === '' || Number.isNaN(v)) return '—';
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (Number.isInteger(n) && digits <= 3) return String(n);
  return n.toFixed(digits);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) {
    let msg;
    try { msg = (await res.json()).error; } catch (e) { msg = res.statusText; }
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json();
}

function chipStyle(algoId) {
  const idx = S.registry.findIndex(r => r.id === algoId);
  const slot = (idx >= 0 ? idx : 0) % 8 + 1;
  return `background: var(--cat-${slot})`;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chipHex(algoId) {
  const idx = S.registry.findIndex(r => r.id === algoId);
  const slot = (idx >= 0 ? idx : 0) % 8 + 1;
  return cssVar(`--cat-${slot}`) || '#2a78d6';
}

function statusChip(st) {
  return `<span class="status st-${esc(st)}">${esc(st)}</span>`;
}

function msLabel(ms) {
  const size = (ms.shape && ms.shape.length)
    ? ms.shape.join('×')
    : (ms.dims === 3 ? `${ms.width}×${ms.height}×${ms.depth}` : `${ms.width}×${ms.height}`);
  return `${ms.name} — ${ms.dims}D · ${ms.num_maps} maps · ${size}`;
}

function paramsSummary(params) {
  return Object.entries(params || {})
    .map(([k, v]) => `${k}=${typeof v === 'boolean' ? (v ? 'on' : 'off') : v}`)
    .join(', ');
}

/* Instance key of a result record or manifest algorithm entry (back-compat). */
function recKey(r) { return r.algorithm_key || r.algorithm_id; }
function algoKey(a) { return a.key || a.id; }

function systemLine() {
  const s = S.system;
  if (s.cuda) return `GPU: ${s.cuda_device} available — “Use GPU (PyTorch)” options run on CUDA.`;
  if (s.torch) return 'No CUDA GPU detected — “Use GPU (PyTorch)” options run on CPU via PyTorch.';
  return 'PyTorch not installed — runs with “Use GPU (PyTorch)” ticked will fail (pip install torch).';
}

/* ---- theme: explicit toggle overriding the system preference ---- */

function applyTheme() {
  const saved = localStorage.getItem('bench-theme');
  const theme = saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.dataset.theme = theme;
}

function refreshCurrentView() {
  if (S.tab === 'run') renderRunTab();
  else if (S.tab === 'maps') renderMapSets();
  else if (S.tab === 'results') { if (S.openRunId) renderRunDetail(); else renderRunsList(); }
  else if (S.tab === 'compare') renderCompareTab();
}

applyTheme();
$('#theme-toggle').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('bench-theme', next);
  document.documentElement.dataset.theme = next;
  refreshCurrentView();   // charts read theme colors at draw time
});

/* Instance display label.

   A user-set graph_label wins for chart/table readability; otherwise the
   algorithm name plus a full hyperparameter summary. The hyperparameters
   themselves are ALWAYS stored on the instance and shown alongside (tooltips,
   params columns) — a custom label never replaces that data. */
function instLabel(algorithms, entry) {
  if (entry.graph_label) return entry.graph_label;
  const ps = paramsSummary(entry.params);
  return ps ? `${entry.label} (${ps})` : entry.label;
}

/* ===================== modal ===================== */

let modalConfirmCb = null;

function showModal({ title, bodyHTML, confirmLabel = 'Delete', onConfirm,
                     confirmClass = 'btn-danger', hideConfirm = false,
                     wide = false, cancelLabel = 'Cancel', onOpen }) {
  $('#modal-title').textContent = title;
  $('#modal-body').innerHTML = bodyHTML;
  const confirm = $('#modal-confirm');
  confirm.textContent = confirmLabel;
  confirm.className = 'btn ' + confirmClass;
  confirm.classList.toggle('hidden', hideConfirm);
  $('#modal-cancel').textContent = cancelLabel;
  $('.modal').classList.toggle('wide', wide);
  modalConfirmCb = onConfirm;
  $('#modal-backdrop').classList.remove('hidden');
  if (onOpen) onOpen();
}

function hideModal() {
  $('#modal-backdrop').classList.add('hidden');
  modalConfirmCb = null;
}

$('#modal-cancel').addEventListener('click', hideModal);
$('#modal-backdrop').addEventListener('click', e => {
  if (e.target === $('#modal-backdrop')) hideModal();
});
$('#modal-confirm').addEventListener('click', async () => {
  if (modalConfirmCb) {
    try { await modalConfirmCb(); } catch (e) { alert(e.message); return; }
  }
  hideModal();
});

/* ===================== data loading & polling ===================== */

async function refreshLists() {
  [S.mapSets, S.runs, S.jobs] = await Promise.all([
    api('/api/map_sets'), api('/api/runs'), api('/api/jobs'),
  ]);
}

function anyActivity() {
  return S.jobs.some(j => j.status === 'queued' || j.status === 'running')
      || S.runs.some(r => !TERMINAL.includes(r.status))
      || S.mapSets.some(m => m.status === 'generating' || m.status === 'pending');
}

function ensurePolling() {
  if (S.pollTimer) return;
  S.pollTimer = setInterval(async () => {
    try {
      await refreshLists();
      renderJobsIndicator();
      if (S.tab === 'maps') renderMapSets();
      if (S.tab === 'results') {
        if (S.openRunId) {
          const st = (S.runDetail || {}).status;
          if (!TERMINAL.includes(st)) await openRun(S.openRunId, true);
        } else {
          renderRunsList();
        }
      }
      if (!anyActivity()) { clearInterval(S.pollTimer); S.pollTimer = null; renderJobsIndicator(); }
    } catch (e) { /* transient poll errors are ignored */ }
  }, 1500);
}

function renderJobsIndicator() {
  const active = S.jobs.filter(j => j.status === 'queued' || j.status === 'running');
  const box = $('#jobs-indicator');
  if (!active.length) { box.classList.add('hidden'); return; }
  const j = active[0];
  const p = j.progress || {};
  const pct = p.total ? ` ${p.current}/${p.total}` : '';
  box.innerHTML = `<span class="spinner"></span> ${esc(j.type)}${pct} — ${esc(p.message || '')}` +
    (active.length > 1 ? ` (+${active.length - 1} queued)` : '');
  box.classList.remove('hidden');
}

/* ===================== tabs ===================== */

$$('.tab').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

function switchTab(name) {
  S.tab = name;
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  ['run', 'maps', 'results', 'compare'].forEach(t =>
    $(`#tab-${t}`).classList.toggle('hidden', t !== name));
  if (name === 'run') renderRunTab();
  if (name === 'maps') renderMapSets();
  if (name === 'results') { S.openRunId = null; renderRunsList(); }
  if (name === 'compare') renderCompareTab();
}

/* ===================== RUN TAB ===================== */

function genDims() {
  if (F.gen.mode === 'nd') {
    const parts = String(F.gen.shape).split(',').map(s => s.trim()).filter(Boolean);
    return parts.length || null;
  }
  return Number(F.gen.mode);
}

function currentDims() {
  if (F.source === 'new') return genDims();
  const ms = S.mapSets.find(m => m.id === F.mapSetId);
  return ms ? ms.dims : null;
}

function renderRunTab() {
  const readySets = S.mapSets.filter(m => m.status === 'ready');
  if (F.source === 'existing' && !F.mapSetId && readySets.length) F.mapSetId = readySets[0].id;

  const dims = currentDims();
  const root = $('#tab-run');
  root.innerHTML = `
    <div class="card">
      <h2>1 · Maps</h2>
      <div class="radio-row">
        <label class="check"><input type="radio" name="src" value="new" ${F.source === 'new' ? 'checked' : ''}> Generate new maps</label>
        <label class="check"><input type="radio" name="src" value="existing" ${F.source === 'existing' ? 'checked' : ''} ${readySets.length ? '' : 'disabled'}> Use existing map set${readySets.length ? '' : ' (none yet)'}</label>
      </div>
      <div id="maps-config"></div>
    </div>

    <div class="card">
      <h2>2 · Algorithms ${dims ? `<span style="font-weight:400;color:var(--muted)">(for ${dims}D maps)</span>` : ''}</h2>
      <div id="algo-builder"></div>
    </div>

    <div class="card">
      <h2>3 · Execute</h2>
      <div class="row">
        <div class="field" style="min-width:220px">
          <label>Run name (optional)</label>
          <input type="text" id="run-name" value="${esc(F.runName)}" placeholder="e.g. fvb sweep, 100x100">
        </div>
        <div class="field">
          <label>Per-solve timeout (s)</label>
          <input type="number" id="run-timeout" value="${F.timeout}" min="5" step="5">
          <span class="hint">Hard kill per (map, algorithm)</span>
        </div>
        <div class="field">
          <label>Parallel workers</label>
          <input type="number" id="run-workers" value="${F.workers}" min="1" max="${Math.max(1, S.system.cpu_count)}" step="1">
          <span class="hint">of ${S.system.cpu_count} CPU cores</span>
        </div>
        <button class="btn btn-primary" id="start-run">Start run</button>
      </div>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:8px 0 0">
        ${esc(systemLine())}
      </p>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:4px 0 0">
        Parallel workers run (map, algorithm) cells concurrently — each still in its own
        watchdog subprocess. Concurrent solves share the CPU, so wall-clock timings can be
        inflated; use 1 worker for the fairest timing comparisons.
        Solved paths are stored with the run — smoothness hyperparameters (steering θ, sweep)
        are chosen later in Results / Compare without re-running the planners.
      </p>
      <div id="run-msg" style="margin-top:10px"></div>
    </div>`;

  $$('input[name="src"]', root).forEach(r => r.addEventListener('change', () => {
    F.source = r.value; renderRunTab();
  }));
  $('#run-name').addEventListener('input', e => { F.runName = e.target.value; });
  $('#run-timeout').addEventListener('input', e => { F.timeout = Number(e.target.value); });
  $('#run-workers').addEventListener('input', e => { F.workers = Number(e.target.value); });
  $('#start-run').addEventListener('click', startRun);

  renderMapsConfig();
  renderAlgoBuilder();
}

function renderMapsConfig() {
  const box = $('#maps-config');
  if (F.source === 'new') {
    const g = F.gen;
    const is3d = g.mode === '3';
    const isNd = g.mode === 'nd';
    box.innerHTML = `
      <div class="row">
        <div class="field" style="min-width:200px">
          <label>Set name (optional)</label>
          <input type="text" data-g="name" value="${esc(g.name)}" placeholder="e.g. dense 100x100">
        </div>
        <div class="field"><label>Dimensions</label>
          <select data-g="mode">
            <option value="2" ${g.mode === '2' ? 'selected' : ''}>2D</option>
            <option value="3" ${g.mode === '3' ? 'selected' : ''}>3D</option>
            <option value="nd" ${isNd ? 'selected' : ''}>ND (custom shape)</option>
          </select>
        </div>
        ${isNd ? `
          <div class="field" style="min-width:200px">
            <label>Shape (x, y, z, …)</label>
            <input type="text" data-g="shape" value="${esc(g.shape)}" placeholder="12,12,12,8">
            <span class="hint">${genDims() || '?'}D — one size per axis</span>
          </div>`
        : `
          <div class="field"><label>Width</label><input type="number" data-g="width" value="${g.width}" min="4"></div>
          <div class="field"><label>Height</label><input type="number" data-g="height" value="${g.height}" min="4"></div>
          ${is3d ? `<div class="field"><label>Depth</label><input type="number" data-g="depth" value="${g.depth}" min="4"></div>` : ''}
        `}
        <div class="field"><label># Maps</label><input type="number" data-g="num_maps" value="${g.num_maps}" min="1"></div>
        <div class="field"><label>Obstacles min</label><input type="number" data-g="obstacles_min" value="${g.obstacles_min}" min="0"></div>
        <div class="field"><label>Obstacles max</label><input type="number" data-g="obstacles_max" value="${g.obstacles_max}" min="0"></div>
        <div class="field"><label>Seed (optional)</label><input type="number" data-g="seed" value="${esc(g.seed)}" placeholder="random"></div>
      </div>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:8px 0 0">
        2D/3D maps are verified solvable with A*; ND maps with face-neighbour BFS.
        Start/end points are placed away from the border.
      </p>`;
    $$('[data-g]', box).forEach(inp => inp.addEventListener('change', () => {
      F.gen[inp.dataset.g] = inp.type === 'number' && inp.value !== '' ? Number(inp.value) : inp.value;
      if (inp.dataset.g === 'mode' || inp.dataset.g === 'shape') renderRunTab();
    }));
  } else {
    const readySets = S.mapSets.filter(m => m.status === 'ready');
    const ms = readySets.find(m => m.id === F.mapSetId);
    box.innerHTML = `
      <div class="row">
        <div class="field" style="min-width:340px">
          <label>Map set</label>
          <select id="ms-select">
            ${readySets.map(m => `<option value="${esc(m.id)}" ${m.id === F.mapSetId ? 'selected' : ''}>${esc(msLabel(m))}</option>`).join('')}
          </select>
        </div>
        <label class="check" style="margin-bottom:6px">
          <input type="checkbox" id="all-maps" ${F.mapSel === null ? 'checked' : ''}> All maps
        </label>
      </div>
      <div id="map-picker" style="margin-top:10px"></div>`;
    $('#ms-select').addEventListener('change', e => {
      F.mapSetId = e.target.value; F.mapSel = null; renderRunTab();
    });
    $('#all-maps').addEventListener('change', e => {
      F.mapSel = e.target.checked ? null : new Set((ms ? ms.maps : []).map(m => m.name));
      renderMapPicker(ms);
    });
    renderMapPicker(ms);
  }
}

function renderMapPicker(ms) {
  const box = $('#map-picker');
  if (!ms) { box.innerHTML = '<p class="empty">No map set selected.</p>'; return; }
  box.innerHTML = `<div class="map-pick-grid">` + ms.maps.map(m => {
    const sel = F.mapSel === null || F.mapSel.has(m.name);
    const thumb = ms.dims === 2
      ? `<img loading="lazy" src="/api/map_sets/${esc(ms.id)}/thumb/${esc(m.stem)}.png" alt="">`
      : `<div style="height:40px;display:flex;align-items:center;justify-content:center;color:var(--muted)">${ms.dims}D</div>`;
    return `<div class="map-pick ${sel ? 'selected' : ''}" data-name="${esc(m.name)}">${thumb}${esc(m.stem)}</div>`;
  }).join('') + `</div>`;
  $$('.map-pick', box).forEach(node => node.addEventListener('click', () => {
    if (F.mapSel === null) F.mapSel = new Set(ms.maps.map(m => m.name));
    const name = node.dataset.name;
    if (F.mapSel.has(name)) F.mapSel.delete(name); else F.mapSel.add(name);
    $('#all-maps').checked = F.mapSel.size === ms.maps.length;
    node.classList.toggle('selected');
  }));
}

/* ---- algorithm builder: add any number of instances, duplicates allowed ---- */

function specSupports(spec, dims) {
  return !dims || spec.dims === null || spec.dims === undefined || spec.dims.includes(dims);
}

function renderAlgoBuilder() {
  const dims = currentDims();
  const box = $('#algo-builder');

  const listHTML = F.algoList.length ? F.algoList.map((inst, i) => {
    const spec = S.registry.find(r => r.id === inst.id);
    const ok = spec && specSupports(spec, dims);
    const name = spec ? spec.label : inst.id;
    return `<div class="algo-inst ${ok ? '' : 'bad'}">
      <span class="chip" style="${chipStyle(inst.id)}"></span>
      <b>${esc(inst.graph_label || name)}</b>
      ${inst.graph_label ? `<span style="font-size:11.5px;color:var(--muted)">(${esc(name)})</span>` : ''}
      <span class="inst-params">${esc(paramsSummary(inst.params))}</span>
      ${ok ? '' : `<span class="inst-warn">not available for ${dims}D</span>`}
      <button class="btn btn-sm" data-edit-inst="${i}">Edit</button>
      <button class="btn btn-sm" data-del-inst="${i}">Remove</button>
    </div>`;
  }).join('') : `<p class="empty" style="text-align:left;padding:6px 0">No algorithms added yet.</p>`;

  const addOpen = F.addForm !== null;
  let addHTML;
  if (!addOpen) {
    addHTML = `<button class="btn" id="open-add">+ Add algorithm</button>`;
  } else {
    const spec = S.registry.find(r => r.id === F.addForm.id) || S.registry[0];
    const supported = specSupports(spec, dims);
    const paramsHTML = spec.params.map(p => {
      const val = F.addForm.params[p.name];
      if (p.type === 'bool') {
        return `<label class="check" style="margin-top:18px"><input type="checkbox" data-ap="${esc(p.name)}" ${val ? 'checked' : ''}> ${esc(p.label)}</label>`;
      }
      return `<div class="field">
        <label title="${esc(p.help || '')}">${esc(p.label)}</label>
        <input type="number" data-ap="${esc(p.name)}" value="${esc(val)}"
               ${p.min != null ? `min="${p.min}"` : ''} ${p.step != null ? `step="${p.step}"` : (p.type === 'float' ? 'step="any"' : '')}>
      </div>`;
    }).join('');
    addHTML = `<div class="add-form">
      <div class="row">
        <div class="field" style="min-width:260px">
          <label>Algorithm</label>
          <select id="add-algo-sel">
            ${S.registry.map(r => `<option value="${esc(r.id)}" ${r.id === spec.id ? 'selected' : ''}>
              ${esc(r.label)} (${r.dims === null ? 'any-D' : r.dims.join('/') + 'D'})</option>`).join('')}
          </select>
          ${supported ? '' : `<span class="hint" style="color:var(--status-critical)">not available for ${dims}D maps</span>`}
        </div>
        <div class="field" style="min-width:240px">
          <label>Graph label (optional)</label>
          <input type="text" id="add-graph-label" value="${esc(F.addForm.graph_label || '')}"
                 placeholder="e.g. FVB fine (n_l=50)">
          <span class="hint">Display name for charts — hyperparameters stay stored &amp; visible</span>
        </div>
      </div>
      <div class="algo-params">${paramsHTML}</div>
      <div class="row" style="margin-top:10px">
        <button class="btn btn-primary btn-sm" id="add-confirm" ${supported ? '' : 'disabled'}>
          ${F.addForm.editIndex != null ? 'Save changes' : 'Add to run'}</button>
        <button class="btn btn-sm" id="add-cancel">Cancel</button>
      </div>
    </div>`;
  }

  box.innerHTML = `<div class="algo-inst-list">${listHTML}</div><div style="margin-top:10px">${addHTML}</div>`;

  $$('[data-del-inst]', box).forEach(b => b.addEventListener('click', () => {
    F.algoList.splice(Number(b.dataset.delInst), 1);
    renderAlgoBuilder();
  }));
  $$('[data-edit-inst]', box).forEach(b => b.addEventListener('click', () => {
    const i = Number(b.dataset.editInst);
    F.addForm = { id: F.algoList[i].id, params: { ...F.algoList[i].params },
                  graph_label: F.algoList[i].graph_label || '', editIndex: i };
    renderAlgoBuilder();
  }));

  const openAdd = $('#open-add', box);
  if (openAdd) openAdd.addEventListener('click', () => {
    const spec = S.registry[0];
    F.addForm = { id: spec.id, params: defaultParams(spec), graph_label: '', editIndex: null };
    renderAlgoBuilder();
  });

  const sel = $('#add-algo-sel', box);
  if (sel) {
    sel.addEventListener('change', () => {
      const spec = S.registry.find(r => r.id === sel.value);
      F.addForm.id = spec.id;
      F.addForm.params = defaultParams(spec);
      renderAlgoBuilder();
    });
    $('#add-graph-label', box).addEventListener('input', e => {
      F.addForm.graph_label = e.target.value;
    });
    $$('[data-ap]', box).forEach(inp => inp.addEventListener('change', () => {
      F.addForm.params[inp.dataset.ap] = inp.type === 'checkbox' ? inp.checked : Number(inp.value);
    }));
    $('#add-confirm', box).addEventListener('click', () => {
      const inst = { id: F.addForm.id, params: { ...F.addForm.params },
                     graph_label: (F.addForm.graph_label || '').trim() || null };
      if (F.addForm.editIndex != null) F.algoList[F.addForm.editIndex] = inst;
      else F.algoList.push(inst);
      F.addForm = null;
      renderAlgoBuilder();
    });
    $('#add-cancel', box).addEventListener('click', () => { F.addForm = null; renderAlgoBuilder(); });
  }
}

function defaultParams(spec) {
  return Object.fromEntries(spec.params.map(p => [p.name, p.default]));
}

async function startRun() {
  const msg = $('#run-msg');
  msg.innerHTML = '';
  try {
    const dims = currentDims();
    const algos = F.algoList.filter(inst => {
      const spec = S.registry.find(r => r.id === inst.id);
      return spec && specSupports(spec, dims);
    });
    if (!F.algoList.length) throw new Error('Add at least one algorithm.');
    if (!algos.length) throw new Error(`None of the added algorithms support ${dims}D maps.`);
    if (algos.length < F.algoList.length) {
      throw new Error(`Some added algorithms do not support ${dims}D maps — remove them first.`);
    }

    // Metric hyperparameters are deliberately not set here — stored metrics use
    // the defaults (θ=30, sweep=5) and are recomputable at any θ afterwards.
    const body = {
      name: F.runName || null,
      solve_timeout_s: F.timeout,
      parallel_workers: Math.max(1, F.workers || 1),
      algorithms: algos,
    };
    if (F.source === 'new') {
      const g = { ...F.gen };
      if (F.gen.mode === 'nd') {
        g.shape = String(F.gen.shape).split(',').map(s => Number(s.trim())).filter(n => n > 0);
        if (g.shape.length < 4) throw new Error('ND shape needs at least 4 sizes (use 2D/3D modes otherwise).');
        g.dims = g.shape.length;
      } else {
        g.dims = Number(F.gen.mode);
        delete g.shape;
      }
      delete g.mode;
      body.new_map_set = g;
    } else {
      if (!F.mapSetId) throw new Error('Select a map set.');
      if (F.mapSel !== null && F.mapSel.size === 0) throw new Error('Select at least one map.');
      body.map_set_id = F.mapSetId;
      body.map_names = F.mapSel === null ? null : [...F.mapSel];
    }

    const res = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) });
    await refreshLists();
    ensurePolling();
    renderJobsIndicator();
    switchTab('results');
    await openRun(res.run_id);
  } catch (e) {
    msg.innerHTML = `<span style="color:var(--status-critical)">✕ ${esc(e.message)}</span>`;
  }
}

/* ===================== MAP SETS TAB ===================== */

function renderMapSets() {
  const root = $('#tab-maps');
  const header = `<div class="card" style="display:flex;align-items:center;gap:12px">
    <span style="font-weight:600">Map sets</span>
    <span style="color:var(--muted);font-size:12.5px">${S.mapSets.length} set(s)</span>
    <div style="flex:1"></div>
    <button class="btn btn-sm" id="import-ms">Import from folder…</button>
  </div>`;
  if (!S.mapSets.length) {
    root.innerHTML = header + `<div class="card"><p class="empty">No map sets yet — generate one from the Run tab, or import an existing folder of .npy maps.</p></div>`;
    $('#import-ms').addEventListener('click', importMapSetFlow);
    return;
  }
  root.innerHTML = header + `<div class="ms-grid">` + S.mapSets.map(ms => {
    const thumbs = ms.dims === 2
      ? (ms.maps || []).slice(0, 6).map(m =>
          `<img loading="lazy" src="/api/map_sets/${esc(ms.id)}/thumb/${esc(m.stem)}.png" alt="" title="${esc(m.stem)} — start ${m.start} → end ${m.end}">`).join('') +
        ((ms.maps || []).length > 6 ? `<span class="more">+${ms.maps.length - 6} more</span>` : '')
      : `<span class="more">${ms.dims}D set — no thumbnails</span>`;
    const size = (ms.shape && ms.shape.length) ? ms.shape.join('×')
      : `${ms.width}×${ms.height}${ms.dims === 3 ? '×' + ms.depth : ''}`;
    return `<div class="card ms-card">
      <div style="display:flex;align-items:center;gap:8px">
        <h2 style="margin:0">${esc(ms.name)}</h2>
        ${statusChip(ms.status)}
      </div>
      <div class="ms-meta">
        ${ms.dims}D · ${(ms.maps || []).length}/${ms.num_maps} maps · ${size}
        · obstacles ${ms.gen_params.obstacles_min ?? '?'}–${ms.gen_params.obstacles_max ?? '?'}
        ${ms.gen_params.seed != null ? `· seed ${ms.gen_params.seed}` : ''}<br>
        <span style="color:var(--muted)">${esc(ms.id)} · ${esc(ms.created)}</span>
      </div>
      <div class="ms-thumbs">${thumbs}</div>
      <div class="ms-actions">
        <button class="btn btn-sm" data-runon="${esc(ms.id)}" ${ms.status !== 'ready' ? 'disabled' : ''}>Run on this set</button>
        ${ms.dims >= 3 ? `<button class="btn btn-sm" data-view3d="${esc(ms.id)}" ${ms.status !== 'ready' ? 'disabled' : ''}>View maps</button>` : ''}
        <button class="btn btn-sm btn-danger" data-del="${esc(ms.id)}">Delete…</button>
      </div>
    </div>`;
  }).join('') + `</div>`;

  $('#import-ms').addEventListener('click', importMapSetFlow);
  $$('[data-runon]', root).forEach(b => b.addEventListener('click', () => {
    F.source = 'existing'; F.mapSetId = b.dataset.runon; F.mapSel = null;
    switchTab('run');
  }));
  $$('[data-view3d]', root).forEach(b => b.addEventListener('click', () => {
    const ms = S.mapSets.find(m => m.id === b.dataset.view3d);
    if (ms) openViewerModal(ms, null, []);
  }));
  $$('[data-del]', root).forEach(b => b.addEventListener('click', () => deleteMapSetFlow(b.dataset.del)));
}

function importMapSetFlow() {
  showModal({
    title: 'Import map set from folder',
    confirmLabel: 'Import',
    confirmClass: 'btn-primary',
    bodyHTML: `
      <p style="font-size:13px;color:var(--ink-2)">
        Point at a folder containing <code>.npy</code> map files and a
        <code>start_end_points.csv</code> listing each map's name and its
        start/end points. Files are copied — the source folder is not modified.
      </p>
      <div class="field" style="margin-top:10px">
        <label>Folder path</label>
        <input type="text" id="imp-path" placeholder="/path/to/maps" style="width:100%">
      </div>
      <div class="field" style="margin-top:10px">
        <label>Set name (optional)</label>
        <input type="text" id="imp-name" placeholder="imported maps">
      </div>`,
    onConfirm: async () => {
      const path = $('#imp-path').value.trim();
      if (!path) throw new Error('Enter a folder path.');
      await api('/api/map_sets/import', {
        method: 'POST',
        body: JSON.stringify({ path, name: $('#imp-name').value.trim() || null }),
      });
      await refreshLists();
      ensurePolling();
      renderJobsIndicator();
      renderMapSets();
    },
  });
}

async function deleteMapSetFlow(msId) {
  const detail = await api(`/api/map_sets/${msId}`);
  const deps = detail.dependent_runs || [];
  const depHTML = deps.length
    ? `<p>${deps.length} run(s) reference this map set:</p>
       <ul class="warn-list">${deps.map(r => `<li>${esc(r.name)} (${esc(r.status)})</li>`).join('')}</ul>
       <p style="font-size:13px;color:var(--ink-2)">Their stored results and images remain viewable,
       but you will not be able to re-run on these maps.</p>
       <label class="check" style="margin-top:8px"><input type="checkbox" id="cascade-del">
         Also delete these ${deps.length} run(s) and all their results</label>`
    : '';
  showModal({
    title: `Delete map set “${detail.name}”?`,
    bodyHTML: `
      <p>This permanently deletes:</p>
      <ul class="warn-list">
        <li>${(detail.maps || []).length} map file(s) (.npy)</li>
        <li>All thumbnails for this set</li>
        <li>Its generation manifest</li>
      </ul>
      ${depHTML}
      <p class="danger-note">This cannot be undone.</p>`,
    confirmLabel: 'Delete map set',
    onConfirm: async () => {
      const cascade = $('#cascade-del') && $('#cascade-del').checked ? '1' : '0';
      await api(`/api/map_sets/${msId}?cascade=${cascade}`, { method: 'DELETE' });
      await refreshLists();
      renderMapSets();
    },
  });
}

/* ===================== RESULTS TAB ===================== */

function renderRunsList() {
  const root = $('#tab-results');
  if (!S.runs.length) {
    root.innerHTML = `<div class="card"><p class="empty">No runs yet — start one from the Run tab.</p></div>`;
    return;
  }
  root.innerHTML = `<div class="card">
    <h2>Runs</h2>
    <div style="overflow-x:auto"><table class="data">
      <thead><tr>
        <th>Name</th><th>Status</th><th>Created</th><th>Map set</th><th>Algorithms</th>
        <th class="num">Solved</th><th></th>
      </tr></thead>
      <tbody>
        ${S.runs.map(r => {
          const algos = (r.algorithms || []).map(a =>
            `<span class="algo-cell" title="${esc(paramsSummary(a.params))}"><span class="chip" style="${chipStyle(a.id)}"></span>${esc(instLabel(r.algorithms, a))}</span>`).join('<br>');
          const solved = (r.summary || []).length
            ? r.summary.map(s => `${s.solved}/${s.total}`).join(' · ')
            : '—';
          return `<tr>
            <td><a class="plain" href="#" data-open="${esc(r.id)}">${esc(r.name)}</a></td>
            <td>${statusChip(r.status)}</td>
            <td>${esc(r.created)}</td>
            <td>${esc(r.map_set_name || r.map_set_id || '(generating)')}</td>
            <td>${algos}</td>
            <td class="num">${solved}</td>
            <td style="text-align:right">
              <button class="btn btn-sm btn-danger" data-delrun="${esc(r.id)}"
                ${!TERMINAL.includes(r.status) ? 'disabled' : ''}>Delete…</button>
            </td>
          </tr>`;
        }).join('')}
      </tbody>
    </table></div>
  </div>`;

  $$('[data-open]', root).forEach(a => a.addEventListener('click', e => {
    e.preventDefault(); openRun(a.dataset.open);
  }));
  $$('[data-delrun]', root).forEach(b => b.addEventListener('click', () => deleteRunFlow(b.dataset.delrun)));
}

async function openRun(runId, silent) {
  try {
    S.runDetail = await api(`/api/runs/${runId}`);
    S.openRunId = runId;
    if (LAB.runId !== runId) {
      LAB.preview = null;
      LAB.runId = runId;
      const mp = S.runDetail.metric_params || {};
      LAB.theta = mp.steering_theta_degrees ?? 30;
      LAB.sweep = mp.steering_sweep_range ?? 5;
    }
    renderRunDetail();
    if (!TERMINAL.includes(S.runDetail.status)) ensurePolling();
  } catch (e) {
    if (!silent) alert(e.message);
  }
}

function meanBy(records, getter) {
  const vals = records.map(getter).filter(v => v != null && Number.isFinite(Number(v)));
  return vals.length ? vals.reduce((a, b) => a + Number(b), 0) / vals.length : null;
}

/* Fetch (and cache) recomputed steering penalties for a run at (theta, sweep).
   Returns a Map: `${algorithm_key}|${map_stem}` -> steering value. */
async function steeringAt(runDetail, theta, sweep) {
  const saved = runDetail.metric_params || {};
  const useSaved = Number(saved.steering_theta_degrees) === Number(theta)
                && Number(saved.steering_sweep_range) === Number(sweep);
  let results;
  if (useSaved) {
    results = runDetail.results || [];
  } else {
    const key = `${runDetail.id}|${theta}|${sweep}`;
    if (!steerCache.has(key)) {
      const out = await api(`/api/runs/${runDetail.id}/metrics`, {
        method: 'POST',
        body: JSON.stringify({ theta_degrees: theta, sweep_range: sweep, save: false }),
      });
      steerCache.set(key, out.results);
    }
    results = steerCache.get(key);
  }
  const map = new Map();
  for (const r of results) {
    if (r.metrics) map.set(`${recKey(r)}|${r.map_stem}`, r.metrics.steering_penalty);
  }
  return map;
}

function renderRunDetail() {
  const d = S.runDetail;
  const root = $('#tab-results');
  const previewOn = LAB.preview && LAB.runId === d.id;
  const results = previewOn ? LAB.preview.results : (d.results || []);
  const mp = previewOn ? LAB.preview.metric_params
                       : (d.metric_params || { steering_theta_degrees: 30, steering_sweep_range: 5 });
  const active = ['pending', 'running', 'cancelling'].includes(d.status);
  const algos = d.algorithms || [];

  let progressHTML = '';
  if (active) {
    const job = S.jobs.find(j => j.result && j.result.run_id === d.id);
    const p = (job && job.progress) || {};
    const pct = p.total ? Math.round(100 * p.current / p.total) : 0;
    const msg = d.status === 'pending' && !(p.total)
      ? 'Waiting (generating maps or queued)…' : (p.message || 'starting…');
    progressHTML = `<div class="progress-wrap">
      <div class="progress-bar"><div class="fill" style="width:${pct}%"></div></div>
      <div class="progress-text">${p.total ? `${p.current}/${p.total} · ` : ''}${esc(msg)}</div>
    </div>`;
  }

  // ---- summary tiles (one per algorithm instance) ----
  const tiles = algos.map(a => {
    const key = algoKey(a);
    const recs = results.filter(r => recKey(r) === key);
    const solved = recs.filter(r => r.solved);
    const mTime = meanBy(solved, r => r.time_s);
    const mPen = meanBy(solved, r => r.metrics && r.metrics.steering_penalty);
    const mLen = meanBy(solved, r => r.path_length_euclidean);
    return `<div class="tile">
      <div class="t-head" title="${esc(paramsSummary(a.params))}"><span class="chip" style="${chipStyle(a.id)}"></span>${esc(instLabel(algos, a))}</div>
      <div class="t-stats">
        <span>solved <b>${solved.length}/${recs.length || '—'}</b></span>
        <span>mean time <b>${fmt(mTime, 3)}${mTime != null ? ' s' : ''}</b></span>
        <span>mean steering penalty <b>${fmt(mPen, 2)}</b></span>
        <span>mean path length <b>${fmt(mLen, 1)}</b></span>
      </div>
    </div>`;
  }).join('');

  // ---- per-map table ----
  const rows = results.map((r, i) => {
    const pen = r.metrics ? r.metrics.steering_penalty : null;
    const metricsHTML = r.metrics
      ? Object.entries(r.metrics).map(([k, v]) =>
          `<div><span class="k">${esc(k)}</span><span class="v">${fmt(v, 5)}</span></div>`).join('')
      : '';
    const entry = algos.find(a => algoKey(a) === recKey(r)) || { id: r.algorithm_id, label: r.algorithm_label, params: r.params };
    return `<tr>
      <td>${esc(r.map_stem || r.map_name)}</td>
      <td><span class="algo-cell" title="${esc(paramsSummary(r.params))}"><span class="chip" style="${chipStyle(r.algorithm_id)}"></span>${esc(instLabel(algos, entry))}</span></td>
      <td>${r.solved ? '<span class="solved-yes">✓ solved</span>'
                     : `<span class="solved-no" title="${esc(r.error || '')}">✕ failed</span>`}</td>
      <td class="num">${fmt(r.time_s, 3)}</td>
      <td class="num">${fmt(r.path_steps, 0)}</td>
      <td class="num">${fmt(r.path_length_euclidean, 1)}</td>
      <td class="num">${fmt(pen, 2)}</td>
      <td>${r.metrics ? `<button class="btn btn-sm" data-tgl="${i}">metrics</button>` : (r.error ? `<span style="font-size:11.5px;color:var(--status-critical)">${esc(String(r.error).slice(0, 60))}</span>` : '')}</td>
    </tr>
    ${r.metrics ? `<tr class="hidden" id="mrow-${i}"><td colspan="8"><div class="metrics-sub">${metricsHTML}</div></td></tr>` : ''}`;
  }).join('');

  // ---- images ----
  const byMap = {};
  results.filter(r => r.has_image).forEach(r => {
    (byMap[r.map_stem] = byMap[r.map_stem] || []).push(r);
  });
  const imagesHTML = Object.keys(byMap).length
    ? Object.entries(byMap).map(([stem, recs]) => `
        <div class="img-map-block">
          <h4>${esc(stem)}</h4>
          <div class="img-row">
            ${recs.map(r => {
              const entry = algos.find(a => algoKey(a) === recKey(r)) || { id: r.algorithm_id, label: r.algorithm_label, params: r.params };
              return `<figure>
              <img loading="lazy" src="/api/runs/${esc(d.id)}/image/${esc(recKey(r))}/${esc(stem)}_path.png" alt="">
              <figcaption><span class="chip" style="${chipStyle(r.algorithm_id)}"></span>${esc(instLabel(algos, entry))}</figcaption>
            </figure>`;
            }).join('')}
          </div>
        </div>`).join('')
    : `<p class="empty">${d.dims >= 3 ? 'No 2D images for ' + d.dims + 'D runs — use the path viewer above.' : 'No path images (no solved maps yet).'}</p>`;

  root.innerHTML = `
    <div class="card">
      <div class="detail-head">
        <button class="btn btn-sm" id="back-runs">← All runs</button>
        <h2>${esc(d.name)}</h2>
        ${statusChip(d.status)}
        <div class="spacer"></div>
        ${active ? `<button class="btn btn-sm btn-danger" id="cancel-run" ${d.status === 'cancelling' ? 'disabled' : ''}>${d.status === 'cancelling' ? 'Cancelling…' : 'Cancel run'}</button>` : ''}
        ${d.status === 'done' || d.status === 'cancelled' ? `<a class="btn btn-sm" href="/api/runs/${esc(d.id)}/results.csv">Download CSV</a>` : ''}
        <button class="btn btn-sm" id="rerun" ${d.map_set_exists ? '' : 'disabled title="map set was deleted"'}>Re-run config</button>
        <button class="btn btn-sm btn-danger" id="del-run" ${active ? 'disabled' : ''}>Delete…</button>
      </div>
      <div class="meta-line">
        ${esc(d.created)} · map set: ${esc(d.map_set_name || d.map_set_id || '(generating)')}${d.map_set_exists ? '' : ' (deleted)'}
        · ${d.dims ? d.dims + 'D' : ''} · timeout ${fmt(d.solve_timeout_s, 0)}s
        ${(d.parallel_workers || 1) > 1 ? `· ${d.parallel_workers} workers` : ''}
        · metrics: θ=${fmt(mp.steering_theta_degrees, 0)}° sweep=${fmt(mp.steering_sweep_range, 0)}${previewOn ? ' (preview)' : ''}
        ${d.error ? `<br><span style="color:var(--status-critical)">error: ${esc(String(d.error).slice(0, 300))}</span>` : ''}
      </div>
      ${previewOn ? `<div class="lab-banner">Showing recomputed metrics (θ=${fmt(mp.steering_theta_degrees, 0)}°, sweep=${fmt(mp.steering_sweep_range, 0)}) — not saved.
        <button class="btn btn-sm" id="lab-reset">Back to saved</button></div>` : ''}
      ${progressHTML}
      <div class="tile-row">${tiles}</div>
    </div>

    ${results.some(r => r.solved) && !active ? `
    <div class="card">
      <h2>Metrics lab <span style="font-weight:400;color:var(--muted)">(recompute from stored paths — no planner re-run)</span></h2>
      <div class="row">
        <div class="field">
          <label>Steering θ (°)</label>
          <input type="number" id="lab-theta" value="${LAB.theta}" min="0" max="180" step="5">
        </div>
        <div class="field">
          <label>Sweep range</label>
          <input type="number" id="lab-sweep" value="${LAB.sweep}" min="1" step="1">
        </div>
        <button class="btn" id="lab-preview">Recompute (preview)</button>
        <button class="btn btn-primary" id="lab-save">Save to run</button>
      </div>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:8px 0 0">
        Preview updates every table, tile and comparison below without touching stored
        results. “Save to run” overwrites the run's stored metrics, summary and CSV.
      </p>
    </div>` : ''}

    ${d.dims >= 3 && results.some(r => r.solved) ? `
    <div class="card">
      <h2>Path viewer (${d.dims}D)</h2>
      ${d.map_set_exists ? `
        <div class="row">
          <div class="field" style="min-width:240px">
            <label>Map</label>
            <select id="v3d-map">
              ${[...new Set(results.filter(r => r.solved).map(r => r.map_stem))].map(s =>
                `<option value="${esc(s)}">${esc(s)}</option>`).join('')}
            </select>
          </div>
          <button class="btn" id="v3d-open">Open viewer</button>
        </div>
        <p class="hint" style="color:var(--muted);font-size:12px;margin:8px 0 0">
          ${d.dims === 3
            ? 'Slice-by-slice view plus a rotatable 3D view (drag to rotate, scroll to zoom) with each algorithm’s path overlaid.'
            : 'Coordinate-trace chart (each axis vs. step) plus a rotatable 3D projection of any three chosen dimensions.'}
        </p>`
        : `<p class="empty">The map set was deleted — the viewer needs the map voxels and is unavailable.</p>`}
    </div>` : ''}

    ${results.some(r => r.solved) ? `
    <div class="card">
      <h2>Compare algorithms</h2>
      <div class="row">
        <div class="field" style="min-width:280px">
          <label>Metric (mean over solved maps)</label>
          <select id="cmp-metric">
            ${METRIC_OPTIONS.map(m => `<option value="${m.key}" ${m.key === compareMetric ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}
          </select>
        </div>
        ${(METRIC_OPTIONS.find(m => m.key === compareMetric) || {}).steer ? `
        <div class="field">
          <label>θ (°)</label>
          <input type="number" id="cmp-theta" value="${STEER.theta}" min="0" max="180" step="5">
        </div>
        <div class="field">
          <label>Sweep</label>
          <input type="number" id="cmp-sweep" value="${STEER.sweep}" min="1" step="1">
        </div>` : ''}
      </div>
      <div class="bars" id="detail-bars"><p class="empty">…</p></div>
      <div class="bar-note">Lower is better for all listed metrics.</div>
    </div>` : ''}

    <div class="card">
      <h2>Results by map</h2>
      ${results.length ? `<div style="overflow-x:auto"><table class="data">
        <thead><tr>
          <th>Map</th><th>Algorithm</th><th>Outcome</th>
          <th class="num">Time (s)</th><th class="num">Steps</th>
          <th class="num">Length</th><th class="num">Steering pen. @${fmt(mp.steering_theta_degrees, 0)}°</th><th></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table></div>` : '<p class="empty">No results yet.</p>'}
    </div>

    <div class="card">
      <h2>Path images</h2>
      ${imagesHTML}
    </div>`;

  $('#back-runs').addEventListener('click', () => { S.openRunId = null; renderRunsList(); });
  $('#del-run').addEventListener('click', () => deleteRunFlow(d.id, true));
  $('#rerun').addEventListener('click', () => prefillFromRun(d));
  const cmp = $('#cmp-metric');
  if (cmp) cmp.addEventListener('change', e => { compareMetric = e.target.value; renderRunDetail(); });
  const cmpTheta = $('#cmp-theta');
  if (cmpTheta) {
    cmpTheta.addEventListener('change', e => { STEER.theta = Number(e.target.value); renderDetailBars(); });
    $('#cmp-sweep').addEventListener('change', e => { STEER.sweep = Number(e.target.value); renderDetailBars(); });
  }
  $$('[data-tgl]', root).forEach(b => b.addEventListener('click', () =>
    $(`#mrow-${b.dataset.tgl}`).classList.toggle('hidden')));

  const cancelBtn = $('#cancel-run');
  if (cancelBtn) cancelBtn.addEventListener('click', async () => {
    cancelBtn.disabled = true;
    cancelBtn.textContent = 'Cancelling…';
    try {
      await api(`/api/runs/${d.id}/cancel`, { method: 'POST' });
      ensurePolling();
      await openRun(d.id, true);
    } catch (e) { alert(e.message); }
  });

  const labTheta = $('#lab-theta');
  if (labTheta) {
    labTheta.addEventListener('input', e => { LAB.theta = Number(e.target.value); });
    $('#lab-sweep').addEventListener('input', e => { LAB.sweep = Number(e.target.value); });
    $('#lab-preview').addEventListener('click', () => labRecompute(d.id, false));
    $('#lab-save').addEventListener('click', () => labRecompute(d.id, true));
  }
  const labReset = $('#lab-reset');
  if (labReset) labReset.addEventListener('click', () => { LAB.preview = null; renderRunDetail(); });

  const v3dOpen = $('#v3d-open');
  if (v3dOpen) v3dOpen.addEventListener('click', () => {
    openRunViewer(d, $('#v3d-map').value);
  });

  renderDetailBars();
}

/* Comparison bars in run detail — steering values honor the STEER θ/sweep. */
async function renderDetailBars() {
  const d = S.runDetail;
  const box = $('#detail-bars');
  if (!d || !box) return;
  const previewOn = LAB.preview && LAB.runId === d.id;
  const results = previewOn ? LAB.preview.results : (d.results || []);
  const algos = d.algorithms || [];
  const mo = METRIC_OPTIONS.find(m => m.key === compareMetric) || METRIC_OPTIONS[0];

  let steerMap = null;
  if (mo.steer) {
    try { steerMap = await steeringAt(d, STEER.theta, STEER.sweep); }
    catch (e) { box.innerHTML = `<p class="empty">${esc(e.message)}</p>`; return; }
  }

  const barData = algos.map(a => {
    const key = algoKey(a);
    const solved = results.filter(r => recKey(r) === key && r.solved);
    const val = mo.steer
      ? meanBy(solved, r => steerMap.get(`${key}|${r.map_stem}`))
      : meanBy(solved, r => mo.src === 'metrics' ? (r.metrics && r.metrics[mo.key]) : r[mo.key]);
    return { id: a.id, label: instLabel(algos, a), val };
  }).filter(b => b.val != null);

  const maxVal = Math.max(...barData.map(b => b.val), 0) || 1;
  box.innerHTML = barData
    .sort((x, y) => x.val - y.val)
    .map(b => `<div class="bar-row">
      <div class="b-label"><span class="chip" style="${chipStyle(b.id)}"></span>${esc(b.label)}</div>
      <div class="b-track"><div class="b-fill" style="width:${Math.max(2, 100 * b.val / maxVal)}%"></div></div>
      <div class="b-val">${fmt(b.val, 3)}</div>
    </div>`).join('') || '<p class="empty">No solved results to compare.</p>';
}

async function labRecompute(runId, save) {
  try {
    const out = await api(`/api/runs/${runId}/metrics`, {
      method: 'POST',
      body: JSON.stringify({ theta_degrees: LAB.theta, sweep_range: LAB.sweep, save }),
    });
    CMP.cache.delete(runId);
    for (const k of [...steerCache.keys()]) if (k.startsWith(runId + '|')) steerCache.delete(k);
    if (save) {
      LAB.preview = null;
      await openRun(runId, true);
    } else {
      LAB.preview = out;
      LAB.runId = runId;
      renderRunDetail();
    }
  } catch (e) { alert(e.message); }
}

function prefillFromRun(d) {
  F.source = 'existing';
  F.mapSetId = d.map_set_id;
  F.mapSel = null;
  const ms = S.mapSets.find(m => m.id === d.map_set_id);
  if (ms && d.map_names && d.map_names.length < ms.maps.length) F.mapSel = new Set(d.map_names);
  F.algoList = (d.algorithms || []).map(a => ({ id: a.id, params: { ...a.params },
                                                graph_label: a.graph_label || null }));
  F.addForm = null;
  F.runName = d.name + ' (re-run)';
  F.timeout = d.solve_timeout_s || 120;
  F.workers = d.parallel_workers || 1;
  switchTab('run');
}

async function deleteRunFlow(runId, fromDetail) {
  const run = S.runs.find(r => r.id === runId) || S.runDetail || {};
  showModal({
    title: `Delete run “${run.name || runId}”?`,
    bodyHTML: `
      <p>This permanently deletes:</p>
      <ul class="warn-list">
        <li>All result records and smoothness metrics</li>
        <li>The exported results.csv</li>
        <li>All stored paths and rendered path images</li>
      </ul>
      <p style="font-size:13px;color:var(--ink-2)">The map set it ran on is <b>not</b> deleted.</p>
      <p class="danger-note">This cannot be undone.</p>`,
    confirmLabel: 'Delete run',
    onConfirm: async () => {
      await api(`/api/runs/${runId}`, { method: 'DELETE' });
      CMP.cache.delete(runId);
      CMP.sel.delete(runId);
      await refreshLists();
      S.openRunId = null;
      renderRunsList();
    },
  });
}

/* ===================== COMPARE TAB ===================== */

/* Steering configs the user wants to compare simultaneously — each becomes its
   own metric/axis (e.g. put "@20°" on X and "@30°" on Y to compare them). */
CMP.steerConfigs = [{ theta: 30, sweep: 5 }];

function steerKey(c) { return `steer|${c.theta}|${c.sweep}`; }
function steerOptions() {
  return CMP.steerConfigs.map(c => ({
    key: steerKey(c), label: `Steering penalty @${c.theta}° (sweep ${c.sweep})`, steer: c,
  }));
}

const BASE_VAL_OPTIONS = [
  { key: 'mean_time_s', label: 'Mean solve time (s)' },
  { key: 'path_length_euclidean', label: 'Mean path length' },
  { key: 'path_steps', label: 'Mean path steps' },
  { key: 'pathbench', label: 'PathBench smoothness', metric: true },
  { key: 'angle_change', label: 'Angle change / length', metric: true },
  { key: 'discrete', label: 'Discrete curvature²', metric: true },
  { key: 'deriv_v1_total', label: 'Derivative total (v1)', metric: true },
  { key: 'deriv_v2_total', label: 'Derivative total (v2)', metric: true },
  { key: 'solved_rate', label: 'Solved rate' },
];

/* Run-level (per map set) properties — useful as line-graph X axes. */
const RUN_PROP_OPTIONS = [
  { key: 'map_max_side', label: 'Map size (largest side)' },
  { key: 'map_cells', label: 'Map total cells' },
  { key: 'dims', label: 'Map dimensionality' },
];

function cmpMetricOptions() { return [...steerOptions(), ...BASE_VAL_OPTIONS]; }
function axisOptions() { return [...steerOptions(), ...BASE_VAL_OPTIONS, ...RUN_PROP_OPTIONS]; }

async function renderCompareTab() {
  const root = $('#tab-compare');
  const eligible = S.runs.filter(r =>
    ['done', 'cancelled', 'interrupted'].includes(r.status) && (r.summary || []).length);

  if (!eligible.length) {
    root.innerHTML = `<div class="card"><p class="empty">No completed runs to compare yet.</p></div>`;
    return;
  }
  [...CMP.sel].forEach(id => { if (!eligible.find(r => r.id === id)) CMP.sel.delete(id); });

  root.innerHTML = `
    <div class="card">
      <h2>Pick runs to compare</h2>
      <div class="cmp-runs">
        ${eligible.map(r => `
          <label class="check">
            <input type="checkbox" data-cmp="${esc(r.id)}" ${CMP.sel.has(r.id) ? 'checked' : ''}>
            <b>${esc(r.name)}</b>
            <span style="color:var(--muted);font-size:12px">
              ${esc(r.created)} · ${esc(r.map_set_name || '?')} · ${r.dims ? r.dims + 'D' : ''}
              · ${(r.algorithms || []).map(a => instLabel(r.algorithms, a)).join(', ')}
            </span>
          </label>`).join('')}
      </div>
    </div>
    <div id="cmp-body"></div>`;

  $$('[data-cmp]', root).forEach(cb => cb.addEventListener('change', () => {
    if (cb.checked) CMP.sel.add(cb.dataset.cmp); else CMP.sel.delete(cb.dataset.cmp);
    renderCompareBody();
  }));
  renderCompareBody();
}

async function compareRows() {
  const details = [];
  for (const id of CMP.sel) {
    if (!CMP.cache.has(id)) {
      try { CMP.cache.set(id, await api(`/api/runs/${id}`)); }
      catch (e) { CMP.sel.delete(id); continue; }
    }
    details.push(CMP.cache.get(id));
  }

  const rows = [];
  for (const d of details) {
    // One steering map per configured (θ, sweep).
    const steerMaps = {};
    for (const c of CMP.steerConfigs) {
      try { steerMaps[steerKey(c)] = await steeringAt(d, c.theta, c.sweep); }
      catch (e) { steerMaps[steerKey(c)] = null; }
    }
    // Map-set geometry (for line graphs vs map size); may be deleted → null.
    const ms = S.mapSets.find(m => m.id === d.map_set_id);
    const shape = ms && ms.shape && ms.shape.length ? ms.shape
      : (ms && ms.width ? [ms.width, ms.height, ...(ms.dims === 3 ? [ms.depth] : [])] : null);
    const mapMaxSide = shape ? Math.max(...shape) : null;
    const mapCells = shape ? shape.reduce((a, b) => a * b, 1) : null;

    for (const a of (d.algorithms || [])) {
      const key = algoKey(a);
      const recs = (d.results || []).filter(r => recKey(r) === key);
      const solved = recs.filter(r => r.solved);
      const vals = {
        mean_time_s: meanBy(solved, r => r.time_s),
        path_length_euclidean: meanBy(solved, r => r.path_length_euclidean),
        path_steps: meanBy(solved, r => r.path_steps),
        dims: d.dims,
        map_max_side: mapMaxSide,
        map_cells: mapCells,
        solved_rate: recs.length ? solved.length / recs.length : null,
      };
      for (const c of CMP.steerConfigs) {
        const sk = steerKey(c);
        vals[sk] = steerMaps[sk]
          ? meanBy(solved, r => steerMaps[sk].get(`${key}|${r.map_stem}`)) : null;
      }
      for (const mo of BASE_VAL_OPTIONS.filter(o => o.metric)) {
        vals[mo.key] = meanBy(solved, r => r.metrics && r.metrics[mo.key]);
      }
      rows.push({
        runId: d.id, runName: d.name, algoId: a.id, algoKey: key,
        algoLabel: instLabel(d.algorithms, a),
        algoName: a.label,
        graphLabel: a.graph_label || null,
        params: a.params, mapSet: d.map_set_name || d.map_set_id, dims: d.dims,
        total: recs.length, solved: solved.length, vals,
      });
    }
  }
  return { rows, details };
}

async function renderCompareBody() {
  const body = $('#cmp-body');
  if (!body) return;
  if (CMP.sel.size < 1) {
    body.innerHTML = `<div class="card"><p class="empty">Select at least one run above.</p></div>`;
    return;
  }
  body.innerHTML = `<div class="card"><p class="empty">Loading…</p></div>`;

  const { rows, details } = await compareRows();
  CMP.lastRows = rows;
  const metricOpts = cmpMetricOptions();
  if (!metricOpts.find(m => m.key === CMP.metric)) CMP.metric = metricOpts[0].key;
  const mo = metricOpts.find(m => m.key === CMP.metric);
  const barKey = mo.key;

  const mapSets = [...new Set(details.map(d => d.map_set_id))];
  const warns = [];
  if (mapSets.length > 1) warns.push('These runs used different map sets — metric differences may come from the maps, not the algorithms.');

  const withVal = rows.filter(r => r.vals[barKey] != null);
  const maxVal = Math.max(...withVal.map(r => r.vals[barKey]), 0) || 1;
  const barsHTML = withVal
    .slice().sort((x, y) => x.vals[barKey] - y.vals[barKey])
    .map(r => `<div class="bar-row">
      <div class="b-label" title="${esc(paramsSummary(r.params))}">
        <span class="chip" style="${chipStyle(r.algoId)}"></span>
        <span>${esc(r.runName)} · ${esc(r.algoLabel)}</span>
      </div>
      <div class="b-track"><div class="b-fill" style="width:${Math.max(2, 100 * r.vals[barKey] / maxVal)}%"></div></div>
      <div class="b-val">${fmt(r.vals[barKey], 3)}</div>
    </div>`).join('');

  // Steering-config chips (each is a metric/axis of its own).
  const steerChipsHTML = CMP.steerConfigs.map((c, i) => `
    <span class="steer-chip">θ=${c.theta}° · sweep ${c.sweep}
      ${CMP.steerConfigs.length > 1 ? `<button data-del-steer="${i}" title="remove">×</button>` : ''}
    </span>`).join('');

  const isLine = CMP.graph.mode === 'line';
  const xOpts = isLine ? RUN_PROP_OPTIONS : axisOptions();
  const axisSel = (id, current, opts) => `<select id="${id}">
      ${opts.map(o => `<option value="${o.key}" ${o.key === current ? 'selected' : ''}>${esc(o.label)}</option>`).join('')}
    </select>`;
  if (!xOpts.find(o => o.key === CMP.graph.x)) CMP.graph.x = xOpts[0].key;
  if (!axisOptions().find(o => o.key === CMP.graph.y)) CMP.graph.y = axisOptions()[0].key;

  // Details table shows every configured steering column.
  const steerCols = steerOptions();

  body.innerHTML = `
    <div class="card">
      <h2>Comparison</h2>
      ${warns.map(w => `<div class="cmp-note">${esc(w)}</div>`).join('')}

      <h3 style="margin-top:0">Smoothness hyperparameters</h3>
      <div class="row" style="align-items:center">
        <div class="steer-chips">${steerChipsHTML}</div>
        <div class="field"><label>θ (°)</label><input type="number" id="steer-new-theta" value="20" min="0" max="180" step="5" style="width:80px"></div>
        <div class="field"><label>Sweep</label><input type="number" id="steer-new-sweep" value="5" min="1" step="1" style="width:70px"></div>
        <button class="btn btn-sm" id="steer-add">+ Add steering config</button>
      </div>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:6px 0 12px">
        Each config becomes its own metric — selectable below and usable as any graph axis
        (e.g. X = @20°, Y = @30°). All are recomputed from stored paths; no planner re-runs.
      </p>

      <div class="row">
        <div class="field" style="min-width:300px">
          <label>Metric (mean over solved maps)</label>
          <select id="cmp-tab-metric">
            ${metricOpts.map(m => `<option value="${m.key}" ${m.key === CMP.metric ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}
          </select>
        </div>
        <a class="btn btn-sm" style="align-self:flex-end"
           href="/api/compare/results.csv?runs=${[...CMP.sel].join(',')}">Download raw CSV (selected runs)</a>
      </div>
      <div class="bars">${barsHTML || '<p class="empty">No solved results among selected runs.</p>'}</div>
      <div class="bar-note">Lower is better for all listed metrics.</div>

      <h3>Metric graph</h3>
      <div class="row">
        <div class="field"><label>Chart</label>
          <select id="gx-mode">
            <option value="scatter" ${!isLine ? 'selected' : ''}>Scatter (2D / 3D)</option>
            <option value="line" ${isLine ? 'selected' : ''}>Lines vs map property</option>
          </select>
        </div>
        <div class="field" style="min-width:210px"><label>X axis</label>${axisSel('gx-x', CMP.graph.x, xOpts)}</div>
        <div class="field" style="min-width:210px"><label>Y axis</label>${axisSel('gx-y', CMP.graph.y, axisOptions())}</div>
        ${!isLine ? `<div class="field" style="min-width:210px"><label>Z axis (optional)</label>
          ${axisSel('gx-z', CMP.graph.z, [{ key: 'none', label: '— none (2D chart) —' }, ...axisOptions()])}</div>` : ''}
      </div>
      <div class="row" style="align-items:flex-end">
        ${['x', 'y'].concat(is3dGraph() ? ['z'] : []).map(ax => `
          <div class="field" style="min-width:150px">
            <label>${ax.toUpperCase()} limits (min / max)</label>
            <div style="display:flex;gap:4px">
              <input type="text" data-lim="${ax}min" value="${esc(CMP.limits[ax + 'min'])}" placeholder="auto" style="width:70px">
              <input type="text" data-lim="${ax}max" value="${esc(CMP.limits[ax + 'max'])}" placeholder="auto" style="width:70px">
            </div>
          </div>`).join('')}
        <button class="btn btn-sm" id="graph-zoom-in" title="Zoom in">＋ zoom</button>
        <button class="btn btn-sm" id="graph-zoom-out" title="Zoom out">－ zoom</button>
        <button class="btn btn-sm" id="graph-reset">Reset view</button>
        <button class="btn btn-sm" id="graph-png">Download PNG (current view, 2×)</button>
      </div>
      <div class="graph-wrap">
        <canvas id="cmp-graph" width="920" height="560"></canvas>
        <div class="graph-legend" id="graph-legend"></div>
      </div>
      <div class="bar-note" id="graph-hint"></div>

      <h3>Hyperparameter sweep <span style="font-weight:400;color:var(--muted)">(same algorithm, varying one parameter)</span></h3>
      <div id="hp-panel"></div>

      <h3>Details</h3>
      <div style="overflow-x:auto"><table class="data">
        <thead><tr>
          <th>Run</th><th>Label</th><th>Hyperparameters</th><th>Map set</th>
          <th class="num">Solved</th><th class="num">Mean time (s)</th>
          ${steerCols.map(c => `<th class="num">${esc(c.label)}</th>`).join('')}
        </tr></thead>
        <tbody>
          ${rows.map((r, ri) => `<tr>
            <td><a class="plain" href="#" data-goto="${esc(r.runId)}">${esc(r.runName)}</a></td>
            <td><span class="algo-cell"><span class="chip" style="${chipStyle(r.algoId)}"></span>
              <span>${esc(r.graphLabel || r.algoName)}</span>
              <button class="btn btn-sm" data-edit-label="${ri}" title="Edit chart label">✎</button></span></td>
            <td style="font-size:11.5px;color:var(--ink-2);max-width:320px">${esc(paramsSummary(r.params))}</td>
            <td>${esc(r.mapSet)} (${r.dims}D)</td>
            <td class="num">${r.solved}/${r.total}</td>
            <td class="num">${fmt(r.vals.mean_time_s, 3)}</td>
            ${steerCols.map(c => `<td class="num">${fmt(r.vals[c.key], 2)}</td>`).join('')}
          </tr>`).join('')}
        </tbody>
      </table></div>
    </div>`;

  $('#cmp-tab-metric').addEventListener('change', e => { CMP.metric = e.target.value; renderCompareBody(); });
  $('#steer-add').addEventListener('click', () => {
    const theta = Number($('#steer-new-theta').value);
    const sweep = Number($('#steer-new-sweep').value);
    if (!CMP.steerConfigs.find(c => c.theta === theta && c.sweep === sweep)) {
      CMP.steerConfigs.push({ theta, sweep });
      renderCompareBody();
    }
  });
  $$('[data-del-steer]', body).forEach(b => b.addEventListener('click', () => {
    CMP.steerConfigs.splice(Number(b.dataset.delSteer), 1);
    renderCompareBody();
  }));
  $$('[data-goto]', body).forEach(a => a.addEventListener('click', e => {
    e.preventDefault(); switchTab('results'); openRun(a.dataset.goto);
  }));
  $$('[data-edit-label]', body).forEach(b => b.addEventListener('click', () =>
    editGraphLabelFlow(rows[Number(b.dataset.editLabel)])));
  $('#gx-mode').addEventListener('change', e => { CMP.graph.mode = e.target.value; renderCompareBody(); });
  ['gx-x', 'gx-y', 'gx-z'].forEach((id, i) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('change', e => {
      CMP.graph[['x', 'y', 'z'][i]] = e.target.value;
      renderCompareBody();   // limits row depends on 2D/3D mode
    });
  });
  $$('[data-lim]', body).forEach(inp => inp.addEventListener('change', () => {
    CMP.limits[inp.dataset.lim] = inp.value.trim();
    drawCompareGraph(CMP.lastRows);
  }));
  $('#graph-zoom-in').addEventListener('click', () => graphZoom(0.8));
  $('#graph-zoom-out').addEventListener('click', () => graphZoom(1.25));
  $('#graph-reset').addEventListener('click', () => {
    CMP.limits = { xmin: '', xmax: '', ymin: '', ymax: '', zmin: '', zmax: '' };
    GVIEW.yaw = 0.7; GVIEW.pitch = 0.45; GVIEW.zoom = 1.0;
    renderCompareBody();
  });
  $('#graph-png').addEventListener('click', () => {
    // Export exactly the current view (limits, rotation, zoom) at 2× resolution.
    const off = document.createElement('canvas');
    off.width = 1840; off.height = 1120;
    drawCompareGraph(CMP.lastRows, off);
    const a = document.createElement('a');
    a.href = off.toDataURL('image/png');
    a.download = 'metric_graph.png';
    a.click();
  });

  renderHpPanel(rows, details);
  drawCompareGraph(rows);
}

function is3dGraph() {
  return CMP.graph.mode === 'scatter' && CMP.graph.z !== 'none';
}

/* Zoom the 2D chart by scaling the effective limits about their center
   (3D scatter zooms the projection instead). */
function graphZoom(factor) {
  if (is3dGraph()) {
    GVIEW.zoom = Math.max(0.4, Math.min(3, GVIEW.zoom / factor));
    drawCompareGraph(CMP.lastRows);
    return;
  }
  const eff = CMP.lastEffLimits;   // set by the last draw
  if (!eff) return;
  for (const ax of ['x', 'y']) {
    const [mn, mx] = eff[ax];
    const c = (mn + mx) / 2, half = (mx - mn) / 2 * factor;
    CMP.limits[ax + 'min'] = String(+(c - half).toPrecision(6));
    CMP.limits[ax + 'max'] = String(+(c + half).toPrecision(6));
  }
  syncLimitInputs();
  drawCompareGraph(CMP.lastRows);
}

function syncLimitInputs() {
  $$('[data-lim]').forEach(inp => { inp.value = CMP.limits[inp.dataset.lim]; });
}

/* Edit the persisted chart label of one algorithm instance (cosmetic only —
   the hyperparameters stay stored and displayed). */
function editGraphLabelFlow(row) {
  showModal({
    title: 'Edit chart label',
    confirmLabel: 'Save label',
    confirmClass: 'btn-primary',
    bodyHTML: `
      <p style="font-size:13px;color:var(--ink-2)">
        <span class="chip" style="${chipStyle(row.algoId)}"></span>
        <b>${esc(row.algoName)}</b> in run “${esc(row.runName)}”<br>
        Hyperparameters (always kept): <span style="font-size:12px">${esc(paramsSummary(row.params))}</span>
      </p>
      <div class="field" style="margin-top:10px">
        <label>Chart label (empty = default name + hyperparameters)</label>
        <input type="text" id="edit-label-input" value="${esc(row.graphLabel || '')}"
               placeholder="${esc(row.algoName)}" style="width:100%">
      </div>`,
    onConfirm: async () => {
      await api(`/api/runs/${row.runId}/graph_label`, {
        method: 'POST',
        body: JSON.stringify({ algorithm_key: row.algoKey,
                               graph_label: $('#edit-label-input').value }),
      });
      CMP.cache.delete(row.runId);
      for (const k of [...steerCache.keys()]) if (k.startsWith(row.runId + '|')) steerCache.delete(k);
      if (S.runDetail && S.runDetail.id === row.runId) S.runDetail = null;
      await refreshLists();
      renderCompareBody();
    },
  });
}

/* ---- hyperparameter sweep: same algorithm, X = one numeric parameter ---- */

function hpParamLabel(name) {
  const spec = S.registry.find(r => r.id === CMP.hp.algo);
  const p = spec && spec.params.find(p => p.name === name);
  return p ? p.label : name;
}

function hpIs3d() { return CMP.hp.param2 && CMP.hp.param2 !== 'none'; }

function renderHpPanel(rows, details) {
  const panel = $('#hp-panel');
  if (!panel) return;

  const algoIds = [...new Set(rows.map(r => r.algoId))];
  if (!CMP.hp.algo || !algoIds.includes(CMP.hp.algo)) CMP.hp.algo = algoIds[0] || '';
  const spec = S.registry.find(r => r.id === CMP.hp.algo);
  const numericParams = spec ? spec.params.filter(p => p.type === 'int' || p.type === 'float') : [];
  if (!numericParams.find(p => p.name === CMP.hp.param)) {
    CMP.hp.param = numericParams.length ? numericParams[0].name : '';
  }
  if (CMP.hp.param2 !== 'none' &&
      (!numericParams.find(p => p.name === CMP.hp.param2) || CMP.hp.param2 === CMP.hp.param)) {
    CMP.hp.param2 = 'none';
  }
  const metricOpts = cmpMetricOptions();
  if (!metricOpts.find(m => m.key === CMP.hp.y)) CMP.hp.y = metricOpts[0].key;

  if (!spec || !numericParams.length) {
    panel.innerHTML = `<p class="empty">No numeric hyperparameters to sweep for the selected runs.</p>`;
    return;
  }

  const mapSets = [...new Set(details.map(d => d.map_set_id))];
  const note = mapSets.length > 1
    ? `<div class="cmp-note">Selected runs span ${mapSets.length} map sets — for a clean sweep, compare runs on the same maps.</div>` : '';

  const is3d = hpIs3d();
  // Axis meaning: 2D → X=param, Y=metric. 3D → X=param, Y=param 2, Z=metric.
  const limitDefs = is3d
    ? [['x', hpParamLabel(CMP.hp.param)], ['y', hpParamLabel(CMP.hp.param2)], ['z', 'metric']]
    : [['x', hpParamLabel(CMP.hp.param)], ['y', 'metric']];

  panel.innerHTML = `
    ${note}
    <div class="row">
      <div class="field" style="min-width:220px"><label>Algorithm</label>
        <select id="hp-algo">${algoIds.map(id => {
          const s = S.registry.find(r => r.id === id);
          return `<option value="${esc(id)}" ${id === CMP.hp.algo ? 'selected' : ''}>${esc(s ? s.label : id)}</option>`;
        }).join('')}</select>
      </div>
      <div class="field" style="min-width:170px"><label>Sweep parameter (X)</label>
        <select id="hp-param">${numericParams.map(p =>
          `<option value="${esc(p.name)}" ${p.name === CMP.hp.param ? 'selected' : ''}>${esc(p.label)}</option>`).join('')}</select>
      </div>
      <div class="field" style="min-width:170px"><label>2nd parameter (optional)</label>
        <select id="hp-param2">
          <option value="none" ${!is3d ? 'selected' : ''}>— none (2D chart) —</option>
          ${numericParams.filter(p => p.name !== CMP.hp.param).map(p =>
            `<option value="${esc(p.name)}" ${p.name === CMP.hp.param2 ? 'selected' : ''}>${esc(p.label)}</option>`).join('')}
        </select>
      </div>
      <div class="field" style="min-width:250px"><label>Metric (mean over solved maps)</label>
        <select id="hp-y">${metricOpts.map(m =>
          `<option value="${m.key}" ${m.key === CMP.hp.y ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}</select>
      </div>
    </div>
    <div class="row" style="align-items:flex-end">
      ${limitDefs.map(([ax, lbl]) => `
        <div class="field" style="min-width:150px">
          <label>${ax.toUpperCase()} · ${esc(lbl)} (min / max)</label>
          <div style="display:flex;gap:4px">
            <input type="text" data-hplim="${ax}min" value="${esc(CMP.hp.limits[ax + 'min'])}" placeholder="auto" style="width:70px">
            <input type="text" data-hplim="${ax}max" value="${esc(CMP.hp.limits[ax + 'max'])}" placeholder="auto" style="width:70px">
          </div>
        </div>`).join('')}
      <button class="btn btn-sm" id="hp-zoom-in" title="Zoom in">＋ zoom</button>
      <button class="btn btn-sm" id="hp-zoom-out" title="Zoom out">－ zoom</button>
      <button class="btn btn-sm" id="hp-reset">Reset view</button>
      <button class="btn btn-sm" id="hp-png">Download PNG (current view, 2×)</button>
    </div>
    <div class="graph-wrap">
      <canvas id="hp-graph" width="920" height="560"></canvas>
      <div class="graph-legend" id="hp-legend"></div>
    </div>
    <div class="bar-note" id="hp-hint"></div>`;

  const clearHpLimits = () => {
    CMP.hp.limits = { xmin: '', xmax: '', ymin: '', ymax: '', zmin: '', zmax: '' };
  };
  $('#hp-algo').addEventListener('change', e => {
    CMP.hp.algo = e.target.value; CMP.hp.param = ''; CMP.hp.param2 = 'none';
    clearHpLimits(); renderHpPanel(rows, details);
  });
  $('#hp-param').addEventListener('change', e => {
    CMP.hp.param = e.target.value;
    if (CMP.hp.param2 === CMP.hp.param) CMP.hp.param2 = 'none';
    clearHpLimits(); renderHpPanel(rows, details);
  });
  $('#hp-param2').addEventListener('change', e => {
    CMP.hp.param2 = e.target.value;
    clearHpLimits(); renderHpPanel(rows, details);   // limit boxes change with the mode
  });
  $('#hp-y').addEventListener('change', e => { CMP.hp.y = e.target.value; drawHpGraph(rows); });
  $$('[data-hplim]', panel).forEach(inp => inp.addEventListener('change', () => {
    CMP.hp.limits[inp.dataset.hplim] = inp.value.trim();
    drawHpGraph(CMP.lastRows);
  }));
  $('#hp-zoom-in').addEventListener('click', () => hpZoom(0.8));
  $('#hp-zoom-out').addEventListener('click', () => hpZoom(1.25));
  $('#hp-reset').addEventListener('click', () => {
    clearHpLimits();
    CMP.hp.view = { yaw: 0.7, pitch: 0.45, zoom: 1.0 };
    syncHpLimitInputs();
    drawHpGraph(CMP.lastRows);
  });
  $('#hp-png').addEventListener('click', () => {
    // Export exactly the current view (limits, rotation, zoom) at 2× resolution.
    const off = document.createElement('canvas');
    off.width = 1840; off.height = 1120;
    drawHpGraph(CMP.lastRows, off);
    const a = document.createElement('a');
    a.href = off.toDataURL('image/png');
    a.download = 'hyperparameter_sweep.png';
    a.click();
  });

  drawHpGraph(rows);
}

function syncHpLimitInputs() {
  $$('[data-hplim]').forEach(inp => { inp.value = CMP.hp.limits[inp.dataset.hplim]; });
}

function hpZoom(factor) {
  if (hpIs3d()) {
    CMP.hp.view.zoom = Math.max(0.4, Math.min(3, CMP.hp.view.zoom / factor));
    drawHpGraph(CMP.lastRows);
    return;
  }
  const eff = CMP.hpEffLimits;
  if (!eff) return;
  for (const ax of ['x', 'y']) {
    const [mn, mx] = eff[ax];
    const c = (mn + mx) / 2, half = (mx - mn) / 2 * factor;
    CMP.hp.limits[ax + 'min'] = String(+(c - half).toPrecision(6));
    CMP.hp.limits[ax + 'max'] = String(+(c + half).toPrecision(6));
  }
  syncHpLimitInputs();
  drawHpGraph(CMP.lastRows);
}

function drawHpGraph(rows, targetCv) {
  const cv = targetCv || $('#hp-graph');
  if (!cv) return;
  const offscreen = !!targetCv;
  const p1 = CMP.hp.param, p2 = CMP.hp.param2, yKey = CMP.hp.y;

  if (!hpIs3d()) {
    // 2D: one series per configuration of the *other* params (matched across
    // runs); X = swept parameter, same-X values averaged.
    const series = new Map();
    for (const r of rows) {
      if (r.algoId !== CMP.hp.algo) continue;
      const x = Number((r.params || {})[p1]);
      const y = r.vals[yKey];
      if (!Number.isFinite(x) || y == null) continue;
      const rest = { ...r.params };
      delete rest[p1];
      const sig = JSON.stringify(rest);
      if (!series.has(sig)) {
        series.set(sig, { label: paramsSummary(rest) || 'defaults', algoId: r.algoId, byX: new Map() });
      }
      const s = series.get(sig);
      if (!s.byX.has(x)) s.byX.set(x, []);
      s.byX.get(x).push(y);
    }
    const lines = [...series.values()].map((s, i) => ({
      ...s, idx: i,
      pts: [...s.byX.entries()]
        .map(([x, ys]) => ({ x, y: ys.reduce((a, b) => a + b, 0) / ys.length }))
        .sort((a, b) => a.x - b.x),
    }));

    drawSeriesChart(cv, lines, hpParamLabel(p1), axisLabel(yKey), {
      legendEl: offscreen ? null : $('#hp-legend'),
      hintEl: offscreen ? null : $('#hp-hint'),
      hintText: 'One series per configuration of the remaining hyperparameters, across the selected runs. ' +
                'Scroll or ＋/－ to zoom; limit boxes set exact ranges.',
      limits: CMP.hp.limits,
      onEff: eff => { CMP.hpEffLimits = eff; },
      logicalH: 560,
    });
    if (!offscreen) bindHpInteractions(cv);
    return;
  }

  // 3D: X = param 1, Y = param 2, Z (vertical) = metric. One color per
  // configuration of the remaining params; identical (config, x, y) across
  // runs are averaged.
  const groups = new Map();
  for (const r of rows) {
    if (r.algoId !== CMP.hp.algo) continue;
    const x = Number((r.params || {})[p1]);
    const y = Number((r.params || {})[p2]);
    const z = r.vals[yKey];
    if (!Number.isFinite(x) || !Number.isFinite(y) || z == null) continue;
    const rest = { ...r.params };
    delete rest[p1];
    delete rest[p2];
    const sig = JSON.stringify(rest);
    if (!groups.has(sig)) {
      groups.set(sig, { label: paramsSummary(rest) || 'defaults', byXY: new Map() });
    }
    const g = groups.get(sig);
    const k = `${x}|${y}`;
    if (!g.byXY.has(k)) g.byXY.set(k, { x, y, zs: [] });
    g.byXY.get(k).zs.push(z);
  }
  const seriesList = [...groups.values()].map((g, i) => ({ label: g.label, idx: i }));
  const pts = [];
  [...groups.values()].forEach((g, i) => {
    for (const cell of g.byXY.values()) {
      pts.push({ x: cell.x, y: cell.y, z: cell.zs.reduce((a, b) => a + b, 0) / cell.zs.length,
                 color: slotHexByIndex(i) });
    }
  });

  if (!offscreen) {
    $('#hp-legend').innerHTML = seriesList.map(s =>
      `<span class="algo-cell"><span class="chip" style="${slotStyleByIndex(s.idx)}"></span>${esc(s.label)}</span>`).join('');
    $('#hp-hint').textContent =
      'X and Y are the two swept hyperparameters; the vertical axis is the metric. ' +
      'One color per configuration of the remaining params. Drag to rotate, scroll or ＋/－ to zoom.';
  }

  render3DScatter(cv, pts, {
    x: { label: hpParamLabel(p1) },
    y: { label: hpParamLabel(p2) },
    z: { label: axisLabel(yKey) },
  }, CMP.hp.view, {
    limits: CMP.hp.limits,
    onEff: eff => { CMP.hpEffLimits = eff; },
  });
  if (!offscreen) bindHpInteractions(cv);
}

/* Shared rotatable 3D scatter (normalized cube). pts: [{x,y,z,color,label?}]. */
function render3DScatter(cv, pts, axesMeta, view, opts = {}) {
  const LW = 920, LH = 560;
  const scaleT = cv.width / LW;
  const ctx = cv.getContext('2d');
  ctx.setTransform(scaleT, 0, 0, scaleT, 0, 0);

  const surface = cssVar('--surface'), ink = cssVar('--ink'), ink2 = cssVar('--ink-2'),
        muted = cssVar('--muted'), grid = cssVar('--grid'), baseline = cssVar('--baseline');

  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, LW, LH);
  ctx.font = '12px system-ui, sans-serif';

  if (!pts.length) {
    ctx.fillStyle = muted;
    ctx.fillText('No data points for these settings.', 30, 40);
    return;
  }

  const lim = opts.limits || {};
  const [xmin, xmax] = axisRangeFor(pts.map(p => p.x), lim.xmin, lim.xmax);
  const [ymin, ymax] = axisRangeFor(pts.map(p => p.y), lim.ymin, lim.ymax);
  const [zmin, zmax] = axisRangeFor(pts.map(p => p.z), lim.zmin, lim.zmax);
  if (opts.onEff) opts.onEff({ x: [xmin, xmax], y: [ymin, ymax], z: [zmin, zmax] });
  const norm = (v, mn, mx) => (v - mn) / (mx - mn) - 0.5;

  const scale = view.zoom * Math.min(LW, LH) * 0.52;
  const project = (nx, ny, nz) => {
    const x1 = nx * Math.cos(view.yaw) - ny * Math.sin(view.yaw);
    const y1 = nx * Math.sin(view.yaw) + ny * Math.cos(view.yaw);
    const y2 = y1 * Math.cos(view.pitch) - nz * Math.sin(view.pitch);
    const z2 = y1 * Math.sin(view.pitch) + nz * Math.cos(view.pitch);
    return [x1 * scale + LW / 2, -z2 * scale + LH / 2, y2];
  };

  const c = [-0.5, 0.5];
  const corners = [];
  for (const a of c) for (const b of c) for (const d of c) corners.push([a, b, d]);
  const pc = corners.map(k => project(k[0], k[1], k[2]));
  const edges = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    if (corners[i].filter((v, k) => v !== corners[j][k]).length === 1) edges.push([i, j]);
  }
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [a, b] of edges) { ctx.moveTo(pc[a][0], pc[a][1]); ctx.lineTo(pc[b][0], pc[b][1]); }
  ctx.stroke();

  const axes = [
    { label: axesMeta.x.label, from: [-0.5, -0.5, -0.5], to: [0.5, -0.5, -0.5], mn: xmin, mx: xmax },
    { label: axesMeta.y.label, from: [-0.5, -0.5, -0.5], to: [-0.5, 0.5, -0.5], mn: ymin, mx: ymax },
    { label: axesMeta.z.label, from: [-0.5, -0.5, -0.5], to: [-0.5, -0.5, 0.5], mn: zmin, mx: zmax },
  ];
  ctx.textAlign = 'center';
  for (const ax of axes) {
    const a = project(...ax.from), b = project(...ax.to);
    ctx.strokeStyle = baseline;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.fillStyle = ink2;
    ctx.fillText(ax.label, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 8);
    ctx.fillStyle = muted;
    ctx.fillText(tickFmt(ax.mn, ax.mx - ax.mn), a[0], a[1] + 14);
    ctx.fillText(tickFmt(ax.mx, ax.mx - ax.mn), b[0], b[1] + 14);
  }

  const inR = (v, mn, mx) => v >= mn && v <= mx;
  const proj = pts
    .filter(p => inR(p.x, xmin, xmax) && inR(p.y, ymin, ymax) && inR(p.z, zmin, zmax))
    .map(p => ({
      ...p, s: project(norm(p.x, xmin, xmax), norm(p.y, ymin, ymax), norm(p.z, zmin, zmax)),
    })).sort((a, b) => b.s[2] - a.s[2]);
  for (const p of proj) {
    ctx.fillStyle = p.color || cssVar('--cat-1');
    ctx.beginPath();
    ctx.arc(p.s[0], p.s[1], 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = surface;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (p.label) {
      ctx.fillStyle = ink;
      ctx.textAlign = 'left';
      ctx.fillText(p.label, p.s[0] + 10, p.s[1] + 4);
      ctx.textAlign = 'center';
    }
  }
}

/* Drag-rotate + wheel zoom for the hyperparameter sweep canvas. */
function bindHpInteractions(cv) {
  if (cv.dataset.bound === '1') return;
  cv.dataset.bound = '1';
  let dragging = false, lx = 0, ly = 0;
  cv.addEventListener('pointerdown', e => { dragging = true; lx = e.clientX; ly = e.clientY; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!dragging || !hpIs3d()) return;
    CMP.hp.view.yaw += (e.clientX - lx) * 0.01;
    CMP.hp.view.pitch = Math.max(-1.45, Math.min(1.45, CMP.hp.view.pitch + (e.clientY - ly) * 0.01));
    lx = e.clientX; ly = e.clientY;
    drawHpGraph(CMP.lastRows);
  });
  cv.addEventListener('pointerup', () => { dragging = false; });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    if (hpIs3d()) {
      CMP.hp.view.zoom = Math.max(0.4, Math.min(3, CMP.hp.view.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
      drawHpGraph(CMP.lastRows);
    } else {
      hpZoom(e.deltaY < 0 ? 0.85 : 1.18);
    }
  }, { passive: false });
}

/* ---- metric graph: 2D scatter or rotatable 3D scatter, canvas-rendered ---- */

const GVIEW = { yaw: 0.7, pitch: 0.45, zoom: 1.0, bound: false };

function axisLabel(key) {
  const o = axisOptions().find(o => o.key === key);
  return o ? o.label : key;
}

function slotStyleByIndex(i) {
  return `background: var(--cat-${i % 8 + 1})`;
}
function slotHexByIndex(i) {
  return cssVar(`--cat-${i % 8 + 1}`) || '#2a78d6';
}

/* Tick label precision adapted to the axis span (avoids "0.00 0.00 0.01 …"). */
function tickFmt(v, span) {
  if (!Number.isFinite(v)) return '';
  const d = span >= 100 ? 0 : span >= 10 ? 1 : span >= 1 ? 2 : span >= 0.1 ? 3 : span >= 0.01 ? 4 : 5;
  return Number(v.toFixed(d)).toString();
}

function niceTicks(min, max, n = 5) {
  if (!(max > min)) { max = min + 1; }
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = span / n / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = mult * step;
  const ticks = [];
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-9; v += s) ticks.push(v);
  return ticks;
}

/* Resolve axis range: manual limit strings override the autofit of `vals`. */
function axisRangeFor(vals, mnStr, mxStr) {
  let mn = Math.min(...vals), mx = Math.max(...vals);
  if (mn === mx) { mn -= 1; mx += 1; }
  const padv = (mx - mn) * 0.08;
  let lo = mn - padv, hi = mx + padv;
  const pmn = parseFloat(mnStr), pmx = parseFloat(mxStr);
  if (Number.isFinite(pmn)) lo = pmn;
  if (Number.isFinite(pmx)) hi = pmx;
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
}

/* All charts render against a logical size and scale to the target canvas, so
   the PNG export can redraw the exact current view at higher resolution. */
function drawCompareGraph(rows, targetCv) {
  const cv = targetCv || $('#cmp-graph');
  if (!cv) return;
  const offscreen = !!targetCv;
  if (CMP.graph.mode === 'line') { drawLineGraph(rows, cv, offscreen); return; }

  const LW = 920, LH = 560;
  const scaleT = cv.width / LW;
  const ctx = cv.getContext('2d');
  ctx.setTransform(scaleT, 0, 0, scaleT, 0, 0);

  const xs = CMP.graph.x, ys = CMP.graph.y, zs = CMP.graph.z;
  const is3d = zs !== 'none';

  const pts = rows
    .map(r => ({
      x: r.vals[xs], y: r.vals[ys], z: is3d ? r.vals[zs] : 0,
      label: `${r.runName} · ${r.algoLabel}`,
      algoId: r.algoId,
    }))
    .filter(p => p.x != null && p.y != null && (!is3d || p.z != null));

  if (!offscreen) {
    const legend = $('#graph-legend');
    const algosPresent = [...new Set(pts.map(p => p.algoId))];
    legend.innerHTML = algosPresent.map(id => {
      const spec = S.registry.find(r => r.id === id);
      return `<span class="algo-cell"><span class="chip" style="${chipStyle(id)}"></span>${esc(spec ? spec.label : id)}</span>`;
    }).join('');
    $('#graph-hint').textContent = is3d
      ? 'Each point is one run × algorithm instance. Drag to rotate, scroll or ＋/－ to zoom. Axes are normalized; labels show real values.'
      : 'Each point is one run × algorithm instance. Scroll or ＋/－ to zoom; edit the limit boxes for exact ranges.';
  }

  const surface = cssVar('--surface'), ink = cssVar('--ink'), ink2 = cssVar('--ink-2'),
        muted = cssVar('--muted'), grid = cssVar('--grid'), baseline = cssVar('--baseline');

  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, LW, LH);
  ctx.font = '12px system-ui, sans-serif';

  if (!pts.length) {
    ctx.fillStyle = muted;
    ctx.fillText('No data points for the chosen axes.', 30, 40);
    return;
  }

  const L = CMP.limits;

  if (!is3d) {
    const M = { l: 70, r: 24, t: 20, b: 52 };
    const W = LW - M.l - M.r, H = LH - M.t - M.b;
    const [xmin, xmax] = axisRangeFor(pts.map(p => p.x), L.xmin, L.xmax);
    const [ymin, ymax] = axisRangeFor(pts.map(p => p.y), L.ymin, L.ymax);
    CMP.lastEffLimits = { x: [xmin, xmax], y: [ymin, ymax] };
    const px = v => M.l + (v - xmin) / (xmax - xmin) * W;
    const py = v => M.t + H - (v - ymin) / (ymax - ymin) * H;

    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = muted;
    for (const t of niceTicks(xmin, xmax)) {
      if (t < xmin || t > xmax) continue;
      ctx.beginPath(); ctx.moveTo(px(t), M.t); ctx.lineTo(px(t), M.t + H); ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(tickFmt(t, xmax - xmin), px(t), M.t + H + 18);
    }
    for (const t of niceTicks(ymin, ymax)) {
      if (t < ymin || t > ymax) continue;
      ctx.beginPath(); ctx.moveTo(M.l, py(t)); ctx.lineTo(M.l + W, py(t)); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(tickFmt(t, ymax - ymin), M.l - 8, py(t) + 4);
    }
    ctx.strokeStyle = baseline;
    ctx.strokeRect(M.l, M.t, W, H);

    ctx.fillStyle = ink2;
    ctx.textAlign = 'center';
    ctx.fillText(axisLabel(xs), M.l + W / 2, LH - 12);
    ctx.save();
    ctx.translate(16, M.t + H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(axisLabel(ys), 0, 0);
    ctx.restore();

    // Clip marks to the plot area so zoomed-out points don't spill over axes.
    ctx.save();
    ctx.beginPath();
    ctx.rect(M.l, M.t, W, H);
    ctx.clip();
    for (const p of pts) {
      ctx.fillStyle = chipHex(p.algoId);
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = surface;    // 2px surface ring separates overlapping marks
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = ink;
      ctx.textAlign = 'left';
      ctx.fillText(p.label, px(p.x) + 10, py(p.y) + 4);
    }
    ctx.restore();
    if (!offscreen) bindGraphInteractions(cv);
    return;
  }

  // ---- 3D scatter: normalize each axis to a unit cube, rotate/zoom ----
  const [xmin, xmax] = axisRangeFor(pts.map(p => p.x), L.xmin, L.xmax);
  const [ymin, ymax] = axisRangeFor(pts.map(p => p.y), L.ymin, L.ymax);
  const [zmin, zmax] = axisRangeFor(pts.map(p => p.z), L.zmin, L.zmax);
  CMP.lastEffLimits = { x: [xmin, xmax], y: [ymin, ymax], z: [zmin, zmax] };
  const norm = (v, mn, mx) => (v - mn) / (mx - mn) - 0.5;

  const scale = GVIEW.zoom * Math.min(LW, LH) * 0.52;
  const project = (nx, ny, nz) => {
    const x1 = nx * Math.cos(GVIEW.yaw) - ny * Math.sin(GVIEW.yaw);
    const y1 = nx * Math.sin(GVIEW.yaw) + ny * Math.cos(GVIEW.yaw);
    const y2 = y1 * Math.cos(GVIEW.pitch) - nz * Math.sin(GVIEW.pitch);
    const z2 = y1 * Math.sin(GVIEW.pitch) + nz * Math.cos(GVIEW.pitch);
    return [x1 * scale + LW / 2, -z2 * scale + LH / 2, y2];
  };

  // cube edges
  const c = [-0.5, 0.5];
  const corners = [];
  for (const a of c) for (const b of c) for (const d of c) corners.push([a, b, d]);
  const pc = corners.map(k => project(k[0], k[1], k[2]));
  const edges = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    const diff = corners[i].filter((v, k) => v !== corners[j][k]).length;
    if (diff === 1) edges.push([i, j]);
  }
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const [a, b] of edges) { ctx.moveTo(pc[a][0], pc[a][1]); ctx.lineTo(pc[b][0], pc[b][1]); }
  ctx.stroke();

  // axis labels + min/max value labels at ends
  const axes = [
    { label: axisLabel(xs), from: [-0.5, -0.5, -0.5], to: [0.5, -0.5, -0.5], mn: xmin, mx: xmax },
    { label: axisLabel(ys), from: [-0.5, -0.5, -0.5], to: [-0.5, 0.5, -0.5], mn: ymin, mx: ymax },
    { label: axisLabel(zs), from: [-0.5, -0.5, -0.5], to: [-0.5, -0.5, 0.5], mn: zmin, mx: zmax },
  ];
  ctx.fillStyle = ink2;
  ctx.textAlign = 'center';
  for (const ax of axes) {
    const a = project(...ax.from), b = project(...ax.to);
    ctx.strokeStyle = baseline;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.fillStyle = ink2;
    ctx.fillText(ax.label, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 8);
    ctx.fillStyle = muted;
    ctx.fillText(tickFmt(ax.mn, ax.mx - ax.mn), a[0], a[1] + 14);
    ctx.fillText(tickFmt(ax.mx, ax.mx - ax.mn), b[0], b[1] + 14);
  }

  // points, far → near; points outside manual limits are dropped
  const inR = (v, mn, mx) => v >= mn && v <= mx;
  const proj = pts
    .filter(p => inR(p.x, xmin, xmax) && inR(p.y, ymin, ymax) && inR(p.z, zmin, zmax))
    .map(p => ({
      ...p, s: project(norm(p.x, xmin, xmax), norm(p.y, ymin, ymax), norm(p.z, zmin, zmax)),
    })).sort((a, b) => b.s[2] - a.s[2]);
  for (const p of proj) {
    ctx.fillStyle = chipHex(p.algoId);
    ctx.beginPath();
    ctx.arc(p.s[0], p.s[1], 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = surface;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = ink;
    ctx.textAlign = 'left';
    ctx.fillText(p.label, p.s[0] + 10, p.s[1] + 4);
  }

  if (!offscreen) bindGraphInteractions(cv);
}

/* Drag-rotate (3D) + wheel zoom (2D limits / 3D projection) — bound once. */
function bindGraphInteractions(cv) {
  if (cv.dataset.bound === '1') return;
  cv.dataset.bound = '1';
  let dragging = false, lx = 0, ly = 0;
  cv.addEventListener('pointerdown', e => { dragging = true; lx = e.clientX; ly = e.clientY; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!dragging || !is3dGraph()) return;
    GVIEW.yaw += (e.clientX - lx) * 0.01;
    GVIEW.pitch = Math.max(-1.45, Math.min(1.45, GVIEW.pitch + (e.clientY - ly) * 0.01));
    lx = e.clientX; ly = e.clientY;
    drawCompareGraph(CMP.lastRows);
  });
  cv.addEventListener('pointerup', () => { dragging = false; });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    if (is3dGraph()) {
      GVIEW.zoom = Math.max(0.4, Math.min(3, GVIEW.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
      drawCompareGraph(CMP.lastRows);
    } else {
      graphZoom(e.deltaY < 0 ? 0.85 : 1.18);
    }
  }, { passive: false });
}

/* Line graph: Y metric vs a run-level map property; one line per algorithm
   configuration (same algorithm id + identical params grouped across runs).
   Multiple runs at the same X are averaged. */
function drawLineGraph(rows, cv, offscreen) {
  const xs = CMP.graph.x, ys = CMP.graph.y;

  const series = new Map();
  for (const r of rows) {
    const x = r.vals[xs], y = r.vals[ys];
    if (x == null || y == null) continue;
    const sig = r.algoId + '|' + JSON.stringify(r.params || {});
    if (!series.has(sig)) series.set(sig, { label: r.algoLabel, algoId: r.algoId, byX: new Map() });
    const s = series.get(sig);
    if (!s.byX.has(x)) s.byX.set(x, []);
    s.byX.get(x).push(y);
  }
  const lines = [...series.values()].map((s, i) => ({
    ...s, idx: i,
    pts: [...s.byX.entries()]
      .map(([x, ys2]) => ({ x, y: ys2.reduce((a, b) => a + b, 0) / ys2.length, n: ys2.length }))
      .sort((a, b) => a.x - b.x),
  }));

  drawSeriesChart(cv, lines, axisLabel(xs), axisLabel(ys), {
    legendEl: offscreen ? null : $('#graph-legend'),
    hintEl: offscreen ? null : $('#graph-hint'),
    hintText: 'One line per algorithm configuration, across the selected runs. ' +
              'Points at the same X are averaged. Scroll or ＋/－ to zoom; limit boxes set exact ranges.',
    limits: CMP.limits,
    storeEff: true,
    logicalH: 560,
  });
  if (!offscreen) bindGraphInteractions(cv);
}

/* Shared multi-series line chart (used by the metric line graph and the
   hyperparameter sweep). Renders at a logical size scaled to the canvas so
   PNG exports can redraw the identical view at higher resolution. */
function drawSeriesChart(cv, lines, xLabelText, yLabelText, opts = {}) {
  const LW = 920, LH = opts.logicalH || 560;
  const scaleT = cv.width / LW;
  const ctx = cv.getContext('2d');
  ctx.setTransform(scaleT, 0, 0, scaleT, 0, 0);

  const surface = cssVar('--surface'), ink = cssVar('--ink'), ink2 = cssVar('--ink-2'),
        muted = cssVar('--muted'), grid = cssVar('--grid'), baseline = cssVar('--baseline');

  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, LW, LH);
  ctx.font = '12px system-ui, sans-serif';

  if (opts.legendEl) {
    opts.legendEl.innerHTML = lines.map(l =>
      `<span class="algo-cell"><span class="chip" style="${slotStyleByIndex(l.idx)}"></span>${esc(l.label)}</span>`).join('');
  }
  if (opts.hintEl) opts.hintEl.textContent = opts.hintText || '';

  if (!lines.length) {
    ctx.fillStyle = muted;
    ctx.fillText('No data points — nothing matches these settings.', 30, 40);
    return;
  }

  const allX = lines.flatMap(l => l.pts.map(p => p.x));
  const allY = lines.flatMap(l => l.pts.map(p => p.y));
  const lim = opts.limits || {};
  const [xmin, xmax] = axisRangeFor(allX, lim.xmin, lim.xmax);
  const [ymin, ymax] = axisRangeFor(allY, lim.ymin, lim.ymax);
  if (opts.storeEff) CMP.lastEffLimits = { x: [xmin, xmax], y: [ymin, ymax] };
  if (opts.onEff) opts.onEff({ x: [xmin, xmax], y: [ymin, ymax] });

  const M = { l: 70, r: 190, t: 20, b: 52 };   // wide right margin for line-end labels
  const W = LW - M.l - M.r, H = LH - M.t - M.b;
  const px = v => M.l + (v - xmin) / (xmax - xmin) * W;
  const py = v => M.t + H - (v - ymin) / (ymax - ymin) * H;

  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = muted;
  for (const t of niceTicks(xmin, xmax)) {
    if (t < xmin || t > xmax) continue;
    ctx.beginPath(); ctx.moveTo(px(t), M.t); ctx.lineTo(px(t), M.t + H); ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(fmt(t, 2), px(t), M.t + H + 18);
  }
  for (const t of niceTicks(ymin, ymax)) {
    if (t < ymin || t > ymax) continue;
    ctx.beginPath(); ctx.moveTo(M.l, py(t)); ctx.lineTo(M.l + W, py(t)); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(fmt(t, 2), M.l - 8, py(t) + 4);
  }
  ctx.strokeStyle = baseline;
  ctx.strokeRect(M.l, M.t, W, H);

  ctx.fillStyle = ink2;
  ctx.textAlign = 'center';
  ctx.fillText(xLabelText, M.l + W / 2, LH - 12);
  ctx.save();
  ctx.translate(16, M.t + H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabelText, 0, 0);
  ctx.restore();

  // Lines clipped to the plot area; labels drawn outside the clip.
  ctx.save();
  ctx.beginPath();
  ctx.rect(M.l, M.t, W, H);
  ctx.clip();
  for (const l of lines) {
    const color = slotHexByIndex(l.idx);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    l.pts.forEach((p, i) => {
      if (i === 0) ctx.moveTo(px(p.x), py(p.y)); else ctx.lineTo(px(p.x), py(p.y));
    });
    ctx.stroke();
    for (const p of l.pts) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = surface;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  ctx.restore();

  const usedLabelYs = [];
  for (const l of lines) {
    const last = l.pts[l.pts.length - 1];
    let ly = Math.max(M.t + 8, Math.min(M.t + H - 4, py(last.y) + 4));
    while (usedLabelYs.some(u => Math.abs(u - ly) < 14)) ly += 14;   // avoid label collisions
    usedLabelYs.push(ly);
    ctx.fillStyle = ink;
    ctx.textAlign = 'left';
    ctx.fillText(l.label.slice(0, 34), Math.min(px(last.x), M.l + W) + 10, ly);
  }
}

/* ===================== VIEWERS (3D slice + rotatable, ND traces + projection) ===================== */

async function loadGrid(msId, mapName) {
  const key = `${msId}/${mapName}`;
  if (!GRIDS.has(key)) {
    const g = await api(`/api/map_sets/${msId}/grid/${mapName}`);
    const bin = atob(g.data_b64);
    const data = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
    GRIDS.set(key, { shape: g.shape, data });   // shape is grid-order (reversed)
  }
  return GRIDS.get(key);
}

async function openRunViewer(d, stem) {
  const recs = (d.results || []).filter(r => r.map_stem === stem && r.solved);
  const ms = S.mapSets.find(m => m.id === d.map_set_id);
  if (!ms) { alert('Map set no longer exists.'); return; }
  const entry = (ms.maps || []).find(m => m.stem === stem);
  if (!entry) { alert('Map not found in map set.'); return; }

  const algos = d.algorithms || [];
  const pathSpecs = [];
  for (const r of recs) {
    try {
      const path = await api(`/api/runs/${d.id}/path/${recKey(r)}/${stem}`);
      const e2 = algos.find(a => algoKey(a) === recKey(r)) || { id: r.algorithm_id, label: r.algorithm_label, params: r.params };
      pathSpecs.push({ algoId: r.algorithm_id, key: recKey(r), label: instLabel(algos, e2), path });
    } catch (e) { /* no stored path */ }
  }
  openViewerModal(ms, entry, pathSpecs);
}

function openViewerModal(ms, entry, pathSpecs) {
  pathSpecs = pathSpecs || [];
  entry = (entry && entry.stem) ? entry : (ms.maps || [])[0];
  if (!entry) { alert('Map set has no maps.'); return; }
  showModal({
    title: `${ms.name} · ${entry.stem}`,
    hideConfirm: true,
    cancelLabel: 'Close',
    wide: true,
    bodyHTML: `
      ${(ms.maps || []).length > 1 && !pathSpecs.length ? `
        <div class="field" style="max-width:240px;margin-bottom:10px">
          <label>Map</label>
          <select id="viewer-map-sel">
            ${ms.maps.map(m => `<option value="${esc(m.stem)}" ${m.stem === entry.stem ? 'selected' : ''}>${esc(m.stem)}</option>`).join('')}
          </select>
        </div>` : ''}
      <div id="viewer-mount"><p class="empty">Loading voxels…</p></div>`,
    onOpen: () => {
      mountViewer($('#viewer-mount'), ms, entry, pathSpecs);
      const sel = $('#viewer-map-sel');
      if (sel) sel.addEventListener('change', () => {
        const e2 = ms.maps.find(m => m.stem === sel.value);
        $('#modal-title').textContent = `${ms.name} · ${e2.stem}`;
        mountViewer($('#viewer-mount'), ms, e2, []);
      });
    },
  });
}

async function mountViewer(mount, ms, entry, pathSpecs) {
  mount.innerHTML = '<p class="empty">Loading voxels…</p>';
  let grid;
  try { grid = await loadGrid(ms.id, entry.name); }
  catch (e) { mount.innerHTML = `<p class="empty">Failed to load grid: ${esc(e.message)}</p>`; return; }

  if (ms.dims === 3) mountViewer3D(mount, ms, entry, pathSpecs, grid);
  else mountViewerND(mount, ms, entry, pathSpecs, grid);
}

/* ---- shared: toggles row ---- */
function togglesHTML(pathSpecs) {
  return pathSpecs.length ? `<div class="viewer-toggles">
    ${pathSpecs.map(s => `<label class="check">
      <input type="checkbox" data-vt="${esc(s.key || s.algoId)}" checked>
      <span class="chip" style="${chipStyle(s.algoId)}"></span>${esc(s.label)}
    </label>`).join('')}
  </div>` : '';
}

/* ---- 3D viewer: slice pane + rotatable voxel pane ---- */
function mountViewer3D(mount, ms, entry, pathSpecs, grid) {
  const [D, H, W] = grid.shape;
  const start = entry.start, end = entry.end;
  const enabled = new Map(pathSpecs.map(s => [s.key || s.algoId, true]));

  mount.innerHTML = `
    ${togglesHTML(pathSpecs)}
    <div class="viewer-grid">
      <div class="viewer-pane">
        <h4>Slice view (z-layer)</h4>
        <canvas id="v-slice"></canvas>
        <div class="v-controls">
          <input type="range" id="v-z" min="0" max="${D - 1}" value="${Math.min(Math.round(start[2] ?? 0), D - 1)}">
          <span id="v-zlabel" style="min-width:70px"></span>
        </div>
        <span class="viewer-hint">Solid = path within ±0.5 layer · faded = within ±1.5 · green start · red end</span>
      </div>
      <div class="viewer-pane">
        <h4>3D view</h4>
        <canvas id="v-3d" width="460" height="460"></canvas>
        <span class="viewer-hint">Drag to rotate · scroll to zoom · gray = obstacles</span>
      </div>
    </div>`;

  const sliceCv = $('#v-slice', mount);
  const cs = Math.max(2, Math.floor(Math.min(380 / W, 380 / H)));
  sliceCv.width = W * cs;
  sliceCv.height = H * cs;
  const zInput = $('#v-z', mount);

  function drawSlice() {
    const z = Number(zInput.value);
    $('#v-zlabel', mount).textContent = `z = ${z} / ${D - 1}`;
    const ctx = sliceCv.getContext('2d');
    ctx.fillStyle = cssVar('--surface');
    ctx.fillRect(0, 0, sliceCv.width, sliceCv.height);
    ctx.fillStyle = cssVar('--ink');
    ctx.globalAlpha = 0.85;
    const base = z * H * W;
    for (let y = 0; y < H; y++) {
      const rowBase = base + y * W;
      for (let x = 0; x < W; x++) {
        if (grid.data[rowBase + x]) ctx.fillRect(x * cs, y * cs, cs, cs);
      }
    }
    ctx.globalAlpha = 1;
    for (const s of pathSpecs) {
      if (!enabled.get(s.key || s.algoId)) continue;
      const color = chipHex(s.algoId);
      for (const p of s.path) {
        const dz = Math.abs((p[2] ?? 0) - z);
        if (dz > 1.5) continue;
        ctx.globalAlpha = dz <= 0.5 ? 1 : 0.3;
        ctx.fillStyle = color;
        ctx.fillRect(Math.floor(p[0]) * cs, Math.floor(p[1]) * cs, cs, cs);
      }
    }
    ctx.globalAlpha = 1;
    if (Math.round(start[2] ?? 0) === z) { ctx.fillStyle = '#0ca30c'; ctx.fillRect(start[0] * cs, start[1] * cs, cs, cs); }
    if (Math.round(end[2] ?? 0) === z) { ctx.fillStyle = '#d03b3b'; ctx.fillRect(end[0] * cs, end[1] * cs, cs, cs); }
  }
  zInput.addEventListener('input', drawSlice);

  const cv3 = $('#v-3d', mount);
  const view = { yaw: 0.7, pitch: 0.45, zoom: 1.0 };
  const draw3D = make3DVoxelRenderer(cv3, view, grid, [W, H, D], pathSpecs, enabled, start, end);

  bindOrbit(cv3, view, draw3D);
  $$('[data-vt]', mount).forEach(cb => cb.addEventListener('change', () => {
    enabled.set(cb.dataset.vt, cb.checked);
    drawSlice();
    draw3D();
  }));

  drawSlice();
  draw3D();
}

/* Reusable rotatable voxel+path renderer (3D data or 3D projection of ND). */
function make3DVoxelRenderer(cv, view, grid, sizes, pathSpecs, enabled, start, end, dimsPick) {
  const ctx = cv.getContext('2d');
  const [SX, SY, SZ] = sizes;               // sizes along the three displayed axes
  const cx = SX / 2, cy = SY / 2, cz = SZ / 2;
  const maxDim = Math.max(SX, SY, SZ);
  const pick = dimsPick || [0, 1, 2];       // which user dims feed the 3 axes

  // Voxel projection: for true 3D grids collect (x,y,z); for ND grids collect
  // deduped projections onto the picked dims.
  let voxels = [];
  if (grid) {
    const gshape = grid.shape;              // grid order = reversed user order
    const n = gshape.length;
    const userSizes = gshape.slice().reverse();
    const strides = new Array(n);
    strides[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) strides[i] = strides[i + 1] * gshape[i + 1];
    const seen = n > 3 ? new Set() : null;
    const coord = new Array(n).fill(0);
    for (let flat = 0; flat < grid.data.length; flat++) {
      if (grid.data[flat]) {
        let rem = flat;
        for (let i = 0; i < n; i++) { coord[i] = Math.floor(rem / strides[i]); rem %= strides[i]; }
        // coord is grid-order; user dim u = coord[n-1-u]
        const p = [coord[n - 1 - pick[0]], coord[n - 1 - pick[1]], coord[n - 1 - pick[2]]];
        if (seen) {
          const k = p.join(',');
          if (seen.has(k)) continue;
          seen.add(k);
        }
        voxels.push(p);
      }
    }
    const MAX_VOX = 50000;
    if (voxels.length > MAX_VOX) {
      const stride = Math.ceil(voxels.length / MAX_VOX);
      voxels = voxels.filter((_, i) => i % stride === 0);
    }
  }

  function project(x, y, z) {
    const s = view.zoom * (380 / maxDim);
    const x1 = (x - cx) * Math.cos(view.yaw) - (y - cy) * Math.sin(view.yaw);
    const y1 = (x - cx) * Math.sin(view.yaw) + (y - cy) * Math.cos(view.yaw);
    const z1 = (z - cz);
    const y2 = y1 * Math.cos(view.pitch) - z1 * Math.sin(view.pitch);
    const z2 = y1 * Math.sin(view.pitch) + z1 * Math.cos(view.pitch);
    return [x1 * s + cv.width / 2, -z2 * s + cv.height / 2, y2];
  }

  const pickPt = pt => [pt[pick[0]] ?? 0, pt[pick[1]] ?? 0, pt[pick[2]] ?? 0];

  return function draw() {
    ctx.fillStyle = cssVar('--page');
    ctx.fillRect(0, 0, cv.width, cv.height);

    const corners = [[0, 0, 0], [SX, 0, 0], [SX, SY, 0], [0, SY, 0], [0, 0, SZ], [SX, 0, SZ], [SX, SY, SZ], [0, SY, SZ]];
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const pc = corners.map(k => project(k[0], k[1], k[2]));
    ctx.strokeStyle = cssVar('--baseline');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [a, b] of edges) { ctx.moveTo(pc[a][0], pc[a][1]); ctx.lineTo(pc[b][0], pc[b][1]); }
    ctx.stroke();

    if (voxels.length) {
      const projected = voxels.map(v => project(v[0] + 0.5, v[1] + 0.5, v[2] + 0.5));
      projected.sort((a, b) => b[2] - a[2]);
      const depths = projected.map(p => p[2]);
      const dMin = Math.min(...depths, 0), dMax = Math.max(...depths, 1);
      ctx.fillStyle = cssVar('--muted');
      for (const p of projected) {
        const t = (dMax - p[2]) / (dMax - dMin + 1e-9);
        ctx.globalAlpha = 0.10 + 0.22 * t;
        ctx.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3);
      }
      ctx.globalAlpha = 1;
    }

    for (const s of pathSpecs) {
      if (enabled && !enabled.get(s.key || s.algoId)) continue;
      ctx.strokeStyle = chipHex(s.algoId);
      ctx.lineWidth = 2;
      ctx.beginPath();
      s.path.forEach((pt, i) => {
        const [px, py, pz] = pickPt(pt);
        const q = project(px + 0.5, py + 0.5, pz + 0.5);
        if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
      });
      ctx.stroke();
    }

    if (start) {
      const [px, py, pz] = pickPt(start);
      const ps = project(px + 0.5, py + 0.5, pz + 0.5);
      ctx.fillStyle = '#0ca30c'; ctx.fillRect(ps[0] - 4, ps[1] - 4, 8, 8);
    }
    if (end) {
      const [px, py, pz] = pickPt(end);
      const pe = project(px + 0.5, py + 0.5, pz + 0.5);
      ctx.fillStyle = '#d03b3b'; ctx.fillRect(pe[0] - 4, pe[1] - 4, 8, 8);
    }
  };
}

function bindOrbit(cv, view, draw) {
  let dragging = false, lx = 0, ly = 0;
  cv.addEventListener('pointerdown', e => { dragging = true; lx = e.clientX; ly = e.clientY; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!dragging) return;
    view.yaw += (e.clientX - lx) * 0.01;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (e.clientY - ly) * 0.01));
    lx = e.clientX; ly = e.clientY;
    draw();
  });
  cv.addEventListener('pointerup', () => { dragging = false; });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    view.zoom = Math.max(0.3, Math.min(4, view.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
    draw();
  }, { passive: false });
}

/* ---- ND viewer: coordinate traces + 3-of-N projection ---- */
function mountViewerND(mount, ms, entry, pathSpecs, grid) {
  const dims = ms.dims;
  const sizes = ms.shape || grid.shape.slice().reverse();   // user-order sizes
  const enabled = new Map(pathSpecs.map(s => [s.key || s.algoId, true]));
  const pick = [0, 1, 2];

  const dimSel = (i) => `<select data-dimpick="${i}">
    ${sizes.map((_, d) => `<option value="${d}" ${pick[i] === d ? 'selected' : ''}>dim ${d} (x${d})</option>`).join('')}
  </select>`;

  mount.innerHTML = `
    ${togglesHTML(pathSpecs)}
    <div class="viewer-grid">
      <div class="viewer-pane">
        <h4>Coordinate traces — each axis vs. path step</h4>
        <canvas id="v-traces" width="460" height="420"></canvas>
        <span class="viewer-hint">One line per dimension per algorithm; smooth traces = smooth path. Dim labels at line ends.</span>
      </div>
      <div class="viewer-pane">
        <h4>3D projection of ${dims}D space</h4>
        <div class="v-controls" style="flex-wrap:wrap">
          <span>X:</span>${dimSel(0)} <span>Y:</span>${dimSel(1)} <span>Z:</span>${dimSel(2)}
        </div>
        <canvas id="v-nd3d" width="460" height="420"></canvas>
        <span class="viewer-hint">Drag to rotate · scroll to zoom · gray = obstacle projection onto chosen dims</span>
      </div>
    </div>`;

  // ---- traces ----
  const tcv = $('#v-traces', mount);
  function drawTraces() {
    const ctx = tcv.getContext('2d');
    ctx.fillStyle = cssVar('--surface');
    ctx.fillRect(0, 0, tcv.width, tcv.height);
    const specs = pathSpecs.filter(s => enabled.get(s.key || s.algoId));
    if (!specs.length) {
      ctx.fillStyle = cssVar('--muted');
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('No paths to display.', 20, 30);
      return;
    }
    const M = { l: 40, r: 46, t: 12, b: 26 };
    const W = tcv.width - M.l - M.r, H = tcv.height - M.t - M.b;
    const maxSteps = Math.max(...specs.map(s => s.path.length));
    const maxVal = Math.max(...sizes);
    ctx.strokeStyle = cssVar('--grid');
    ctx.lineWidth = 1;
    for (const t of niceTicks(0, maxVal, 4)) {
      const y = M.t + H - t / maxVal * H;
      ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + W, y); ctx.stroke();
      ctx.fillStyle = cssVar('--muted');
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(fmt(t, 0), M.l - 5, y + 4);
    }
    ctx.strokeStyle = cssVar('--baseline');
    ctx.strokeRect(M.l, M.t, W, H);
    ctx.fillStyle = cssVar('--ink-2');
    ctx.textAlign = 'center';
    ctx.fillText('path step →', M.l + W / 2, tcv.height - 8);

    // Line style: color = algorithm; per-dim identity via end-of-line label.
    for (const s of specs) {
      const color = chipHex(s.algoId);
      for (let d = 0; d < dims; d++) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.45 + 0.55 * (d % 2);   // alternate emphasis to separate dense lines
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        s.path.forEach((pt, i) => {
          const x = M.l + (i / Math.max(1, maxSteps - 1)) * W;
          const y = M.t + H - (pt[d] / maxVal) * H;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
        const lastPt = s.path[s.path.length - 1];
        ctx.fillStyle = cssVar('--ink');
        ctx.textAlign = 'left';
        ctx.font = '11px system-ui, sans-serif';
        const lx = M.l + W + 4;
        const ly = M.t + H - (lastPt[d] / maxVal) * H + 4;
        ctx.fillText(`x${d}`, lx, ly);
      }
    }
  }

  // ---- 3D projection ----
  const cv3 = $('#v-nd3d', mount);
  const view = { yaw: 0.7, pitch: 0.45, zoom: 1.0 };
  let draw3D;
  function rebuild3D() {
    const sz = [sizes[pick[0]], sizes[pick[1]], sizes[pick[2]]];
    draw3D = make3DVoxelRenderer(cv3, view, grid, sz, pathSpecs, enabled,
                                 entry.start, entry.end, pick.slice());
    draw3D();
  }
  bindOrbit(cv3, view, () => draw3D && draw3D());

  $$('[data-dimpick]', mount).forEach(sel => sel.addEventListener('change', () => {
    pick[Number(sel.dataset.dimpick)] = Number(sel.value);
    rebuild3D();
  }));
  $$('[data-vt]', mount).forEach(cb => cb.addEventListener('change', () => {
    enabled.set(cb.dataset.vt, cb.checked);
    drawTraces();
    draw3D && draw3D();
  }));

  drawTraces();
  rebuild3D();
}

/* ===================== init ===================== */

(async function init() {
  try {
    [S.registry, S.system] = await Promise.all([api('/api/registry'), api('/api/system')]);
    await refreshLists();
    // Sensible default: one instance of each algorithm that supports 2D.
    if (!F.algoList.length) {
      F.algoList = S.registry
        .filter(spec => specSupports(spec, 2))
        .map(spec => ({ id: spec.id, params: defaultParams(spec) }));
    }
    renderJobsIndicator();
    renderRunTab();
    if (anyActivity()) ensurePolling();
  } catch (e) {
    document.body.innerHTML = `<div style="padding:40px;font-family:system-ui">
      <h2>Failed to load</h2><p>${esc(e.message)}</p></div>`;
  }
})();
