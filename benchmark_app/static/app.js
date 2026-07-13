'use strict';

/* ===================== state ===================== */

const S = {
  registry: [],
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
  gen: { name: '', dims: 2, width: 100, height: 100, depth: 30,
         num_maps: 5, obstacles_min: 5, obstacles_max: 30, seed: '' },
  mapSetId: null,
  mapSel: null,                     // null = all maps, else Set of map names
  algos: {},                        // id -> {on: bool, params: {}}
  runName: '',
  timeout: 120,
};

const METRIC_OPTIONS = [
  { key: 'steering_penalty', label: 'Steering penalty @30° (recommended)', src: 'metrics' },
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

/* Metrics-lab state: preview holds an unsaved recompute for the open run. */
const LAB = { theta: 30, sweep: 5, preview: null, runId: null };

/* Cross-run compare state. */
const CMP = { sel: new Set(), metric: 'steering_penalty', cache: new Map() };

/* Decoded grid cache for 3D viewers: key "msId/mapName" -> {shape, data}. */
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

function statusChip(st) {
  return `<span class="status st-${esc(st)}">${esc(st)}</span>`;
}

function msLabel(ms) {
  const size = ms.dims === 3 ? `${ms.width}×${ms.height}×${ms.depth}` : `${ms.width}×${ms.height}`;
  return `${ms.name} — ${ms.dims}D · ${ms.num_maps} maps · ${size}`;
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
    try { await modalConfirmCb(); } catch (e) { alert(e.message); }
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

function currentDims() {
  if (F.source === 'new') return Number(F.gen.dims);
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
      <div id="algo-list"></div>
    </div>

    <div class="card">
      <h2>3 · Execute</h2>
      <div class="row">
        <div class="field" style="min-width:220px">
          <label>Run name (optional)</label>
          <input type="text" id="run-name" value="${esc(F.runName)}" placeholder="e.g. fvb vs astar, 100x100">
        </div>
        <div class="field">
          <label>Per-solve timeout (s)</label>
          <input type="number" id="run-timeout" value="${F.timeout}" min="5" step="5">
          <span class="hint">Hard kill per (map, algorithm)</span>
        </div>
        <div class="field">
          <label>Steering θ (°)</label>
          <input type="number" id="run-theta" value="${F.theta ?? 30}" min="0" max="180" step="5">
          <span class="hint">Penalty threshold</span>
        </div>
        <div class="field">
          <label>Steering sweep</label>
          <input type="number" id="run-sweep" value="${F.sweep ?? 5}" min="1" step="1">
          <span class="hint">Neighbor offsets</span>
        </div>
        <button class="btn btn-primary" id="start-run">Start run</button>
      </div>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:8px 0 0">
        Metric parameters only affect how smoothness is computed — you can recompute
        metrics on stored paths later without re-running the planners.
      </p>
      <div id="run-msg" style="margin-top:10px"></div>
    </div>`;

  $$('input[name="src"]', root).forEach(r => r.addEventListener('change', () => {
    F.source = r.value; renderRunTab();
  }));
  $('#run-name').addEventListener('input', e => { F.runName = e.target.value; });
  $('#run-timeout').addEventListener('input', e => { F.timeout = Number(e.target.value); });
  $('#run-theta').addEventListener('input', e => { F.theta = Number(e.target.value); });
  $('#run-sweep').addEventListener('input', e => { F.sweep = Number(e.target.value); });
  $('#start-run').addEventListener('click', startRun);

  renderMapsConfig();
  renderAlgoBlocks();
}

function renderMapsConfig() {
  const box = $('#maps-config');
  if (F.source === 'new') {
    const g = F.gen;
    const is3d = Number(g.dims) === 3;
    box.innerHTML = `
      <div class="row">
        <div class="field" style="min-width:200px">
          <label>Set name (optional)</label>
          <input type="text" data-g="name" value="${esc(g.name)}" placeholder="e.g. dense 100x100">
        </div>
        <div class="field"><label>Dimensions</label>
          <select data-g="dims">
            <option value="2" ${!is3d ? 'selected' : ''}>2D</option>
            <option value="3" ${is3d ? 'selected' : ''}>3D</option>
          </select>
        </div>
        <div class="field"><label>Width</label><input type="number" data-g="width" value="${g.width}" min="4"></div>
        <div class="field"><label>Height</label><input type="number" data-g="height" value="${g.height}" min="4"></div>
        ${is3d ? `<div class="field"><label>Depth</label><input type="number" data-g="depth" value="${g.depth}" min="4"></div>` : ''}
        <div class="field"><label># Maps</label><input type="number" data-g="num_maps" value="${g.num_maps}" min="1"></div>
        <div class="field"><label>Obstacles min</label><input type="number" data-g="obstacles_min" value="${g.obstacles_min}" min="0"></div>
        <div class="field"><label>Obstacles max</label><input type="number" data-g="obstacles_max" value="${g.obstacles_max}" min="0"></div>
        <div class="field"><label>Seed (optional)</label><input type="number" data-g="seed" value="${esc(g.seed)}" placeholder="random"></div>
      </div>
      <p class="hint" style="color:var(--muted);font-size:12px;margin:8px 0 0">
        Maps are verified solvable with A* before being accepted. Start/end points are placed away from the border.
      </p>`;
    $$('[data-g]', box).forEach(inp => inp.addEventListener('change', () => {
      F.gen[inp.dataset.g] = inp.type === 'number' && inp.value !== '' ? Number(inp.value) : inp.value;
      if (inp.dataset.g === 'dims') { renderRunTab(); }
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
      : `<div style="height:40px;display:flex;align-items:center;justify-content:center;color:var(--muted)">3D</div>`;
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

function renderAlgoBlocks() {
  const dims = currentDims();
  const box = $('#algo-list');
  box.innerHTML = S.registry.map(spec => {
    if (!F.algos[spec.id]) {
      F.algos[spec.id] = { on: false, params: Object.fromEntries(spec.params.map(p => [p.name, p.default])) };
    }
    const st = F.algos[spec.id];
    const unsupported = dims && !spec.dims.includes(dims);
    const paramsHTML = spec.params.map(p => {
      const val = st.params[p.name];
      if (p.type === 'bool') {
        return `<label class="check" style="margin-top:18px"><input type="checkbox" data-algo="${esc(spec.id)}" data-param="${esc(p.name)}" ${val ? 'checked' : ''}> ${esc(p.label)}</label>`;
      }
      return `<div class="field">
        <label title="${esc(p.help || '')}">${esc(p.label)}</label>
        <input type="number" data-algo="${esc(spec.id)}" data-param="${esc(p.name)}"
               value="${esc(val)}" ${p.min != null ? `min="${p.min}"` : ''} ${p.step != null ? `step="${p.step}"` : (p.type === 'float' ? 'step="any"' : '')}>
      </div>`;
    }).join('');
    return `<div class="algo-block ${unsupported ? 'disabled' : ''}">
      <div class="algo-head">
        <label class="check">
          <input type="checkbox" data-algo-toggle="${esc(spec.id)}" ${st.on && !unsupported ? 'checked' : ''} ${unsupported ? 'disabled' : ''}>
          <span class="chip" style="${chipStyle(spec.id)}"></span>
          <b>${esc(spec.label)}</b>
        </label>
        <span class="note">${unsupported ? `${spec.dims.join('/')}D only` : `supports ${spec.dims.join('/')}D`}</span>
      </div>
      ${st.on && !unsupported ? `<div class="algo-params">${paramsHTML}</div>` : ''}
    </div>`;
  }).join('');

  $$('[data-algo-toggle]', box).forEach(inp => inp.addEventListener('change', () => {
    F.algos[inp.dataset.algoToggle].on = inp.checked;
    renderAlgoBlocks();
  }));
  $$('[data-algo][data-param]', box).forEach(inp => inp.addEventListener('change', () => {
    const st = F.algos[inp.dataset.algo];
    st.params[inp.dataset.param] = inp.type === 'checkbox' ? inp.checked : Number(inp.value);
  }));
}

async function startRun() {
  const msg = $('#run-msg');
  msg.innerHTML = '';
  try {
    const dims = currentDims();
    const algos = S.registry
      .filter(spec => F.algos[spec.id] && F.algos[spec.id].on && (!dims || spec.dims.includes(dims)))
      .map(spec => ({ id: spec.id, params: F.algos[spec.id].params }));
    if (!algos.length) throw new Error('Select at least one algorithm.');

    const body = {
      name: F.runName || null,
      solve_timeout_s: F.timeout,
      algorithms: algos,
      metric_params: {
        steering_theta_degrees: F.theta ?? 30,
        steering_sweep_range: F.sweep ?? 5,
      },
    };
    if (F.source === 'new') {
      body.new_map_set = { ...F.gen };
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
      : `<span class="more">3D set — no thumbnails</span>`;
    return `<div class="card ms-card">
      <div style="display:flex;align-items:center;gap:8px">
        <h2 style="margin:0">${esc(ms.name)}</h2>
        ${statusChip(ms.status)}
      </div>
      <div class="ms-meta">
        ${ms.dims}D · ${(ms.maps || []).length}/${ms.num_maps} maps · ${ms.width}×${ms.height}${ms.dims === 3 ? '×' + ms.depth : ''}
        · obstacles ${ms.gen_params.obstacles_min}–${ms.gen_params.obstacles_max}
        ${ms.gen_params.seed != null ? `· seed ${ms.gen_params.seed}` : ''}<br>
        <span style="color:var(--muted)">${esc(ms.id)} · ${esc(ms.created)}</span>
      </div>
      <div class="ms-thumbs">${thumbs}</div>
      <div class="ms-actions">
        <button class="btn btn-sm" data-runon="${esc(ms.id)}" ${ms.status !== 'ready' ? 'disabled' : ''}>Run on this set</button>
        ${ms.dims === 3 ? `<button class="btn btn-sm" data-view3d="${esc(ms.id)}" ${ms.status !== 'ready' ? 'disabled' : ''}>View maps</button>` : ''}
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
        <code>start_end_points.csv</code> (the format the old CLI scripts produced,
        e.g. <code>Test Suite/output/2d_maps</code>). Files are copied — the source
        folder is not modified.
      </p>
      <div class="field" style="margin-top:10px">
        <label>Folder path</label>
        <input type="text" id="imp-path" placeholder="Test Suite/output/2d_maps" style="width:100%">
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
    <table class="data">
      <thead><tr>
        <th>Name</th><th>Status</th><th>Created</th><th>Map set</th><th>Algorithms</th>
        <th class="num">Solved</th><th></th>
      </tr></thead>
      <tbody>
        ${S.runs.map(r => {
          const algos = (r.algorithms || []).map(a =>
            `<span class="algo-cell"><span class="chip" style="${chipStyle(a.id)}"></span>${esc(a.label)}</span>`).join('<br>');
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
    </table>
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

function renderRunDetail() {
  const d = S.runDetail;
  const root = $('#tab-results');
  const previewOn = LAB.preview && LAB.runId === d.id;
  const results = previewOn ? LAB.preview.results : (d.results || []);
  const mp = previewOn ? LAB.preview.metric_params
                       : (d.metric_params || { steering_theta_degrees: 30, steering_sweep_range: 5 });
  const active = ['pending', 'running', 'cancelling'].includes(d.status);

  // progress from matching background job
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

  // ---- summary tiles ----
  const tiles = (d.algorithms || []).map(a => {
    const recs = results.filter(r => r.algorithm_id === a.id);
    const solved = recs.filter(r => r.solved);
    const mTime = meanBy(solved, r => r.time_s);
    const mPen = meanBy(solved, r => r.metrics && r.metrics.steering_penalty);
    const mLen = meanBy(solved, r => r.path_length_euclidean);
    return `<div class="tile">
      <div class="t-head"><span class="chip" style="${chipStyle(a.id)}"></span>${esc(a.label)}</div>
      <div class="t-stats">
        <span>solved <b>${solved.length}/${recs.length || '—'}</b></span>
        <span>mean time <b>${fmt(mTime, 3)}${mTime != null ? ' s' : ''}</b></span>
        <span>mean steering penalty <b>${fmt(mPen, 2)}</b></span>
        <span>mean path length <b>${fmt(mLen, 1)}</b></span>
      </div>
    </div>`;
  }).join('');

  // ---- comparison bars (single hue; direct value labels; lower = better) ----
  const mo = METRIC_OPTIONS.find(m => m.key === compareMetric) || METRIC_OPTIONS[0];
  const barData = (d.algorithms || []).map(a => {
    const solved = results.filter(r => r.algorithm_id === a.id && r.solved);
    const val = meanBy(solved, r => mo.src === 'metrics' ? (r.metrics && r.metrics[mo.key]) : r[mo.key]);
    return { id: a.id, label: a.label, val };
  }).filter(b => b.val != null);
  const maxVal = Math.max(...barData.map(b => b.val), 0) || 1;
  const barsHTML = barData
    .sort((x, y) => x.val - y.val)
    .map(b => `<div class="bar-row">
      <div class="b-label"><span class="chip" style="${chipStyle(b.id)}"></span>${esc(b.label)}</div>
      <div class="b-track"><div class="b-fill" style="width:${Math.max(2, 100 * b.val / maxVal)}%"></div></div>
      <div class="b-val">${fmt(b.val, 3)}</div>
    </div>`).join('');

  // ---- per-map table ----
  const rows = results.map((r, i) => {
    const pen = r.metrics ? r.metrics.steering_penalty : null;
    const metricsHTML = r.metrics
      ? Object.entries(r.metrics).map(([k, v]) =>
          `<div><span class="k">${esc(k)}</span><span class="v">${fmt(v, 5)}</span></div>`).join('')
      : '';
    return `<tr>
      <td>${esc(r.map_stem || r.map_name)}</td>
      <td><span class="algo-cell"><span class="chip" style="${chipStyle(r.algorithm_id)}"></span>${esc(r.algorithm_label)}</span></td>
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
            ${recs.map(r => `<figure>
              <img loading="lazy" src="/api/runs/${esc(d.id)}/image/${esc(r.algorithm_id)}/${esc(stem)}_path.png" alt="">
              <figcaption><span class="chip" style="${chipStyle(r.algorithm_id)}"></span>${esc(r.algorithm_label)}</figcaption>
            </figure>`).join('')}
          </div>
        </div>`).join('')
    : `<p class="empty">${d.dims === 3 ? 'No images for 3D runs.' : 'No path images (no solved maps yet).'}</p>`;

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

    ${d.dims === 3 && results.some(r => r.solved) ? `
    <div class="card">
      <h2>3D path viewer</h2>
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
          Slice-by-slice view plus a rotatable 3D view (drag to rotate, scroll to zoom) with each
          algorithm's path overlaid.
        </p>`
        : `<p class="empty">The map set was deleted — 3D view needs the map voxels and is unavailable.</p>`}
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
      </div>
      <div class="bars">${barsHTML || '<p class="empty">No solved results to compare.</p>'}</div>
      <div class="bar-note">Lower is better for all listed metrics.</div>
    </div>` : ''}

    <div class="card">
      <h2>Results by map</h2>
      ${results.length ? `<div style="overflow-x:auto"><table class="data">
        <thead><tr>
          <th>Map</th><th>Algorithm</th><th>Outcome</th>
          <th class="num">Time (s)</th><th class="num">Steps</th>
          <th class="num">Length</th><th class="num">Steering pen.</th><th></th>
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

  // Metrics lab
  const labTheta = $('#lab-theta'), labSweep = $('#lab-sweep');
  if (labTheta) {
    labTheta.addEventListener('input', e => { LAB.theta = Number(e.target.value); });
    labSweep.addEventListener('input', e => { LAB.sweep = Number(e.target.value); });
    $('#lab-preview').addEventListener('click', () => labRecompute(d.id, false));
    $('#lab-save').addEventListener('click', () => labRecompute(d.id, true));
  }
  const labReset = $('#lab-reset');
  if (labReset) labReset.addEventListener('click', () => { LAB.preview = null; renderRunDetail(); });

  // 3D viewer
  const v3dOpen = $('#v3d-open');
  if (v3dOpen) v3dOpen.addEventListener('click', () => {
    const stem = $('#v3d-map').value;
    openRunViewer(d, stem);
  });
}

async function labRecompute(runId, save) {
  try {
    const out = await api(`/api/runs/${runId}/metrics`, {
      method: 'POST',
      body: JSON.stringify({ theta_degrees: LAB.theta, sweep_range: LAB.sweep, save }),
    });
    CMP.cache.delete(runId);   // compare tab must refetch updated metrics
    if (save) {
      LAB.preview = null;
      await openRun(runId, true);   // reload saved state
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
  S.registry.forEach(spec => {
    const cfg = (d.algorithms || []).find(a => a.id === spec.id);
    F.algos[spec.id] = cfg
      ? { on: true, params: { ...cfg.params } }
      : { on: false, params: Object.fromEntries(spec.params.map(p => [p.name, p.default])) };
  });
  F.runName = d.name + ' (re-run)';
  F.timeout = d.solve_timeout_s || 120;
  F.theta = (d.metric_params || {}).steering_theta_degrees ?? 30;
  F.sweep = (d.metric_params || {}).steering_sweep_range ?? 5;
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

async function renderCompareTab() {
  const root = $('#tab-compare');
  const eligible = S.runs.filter(r =>
    ['done', 'cancelled', 'interrupted'].includes(r.status) && (r.summary || []).length);

  if (!eligible.length) {
    root.innerHTML = `<div class="card"><p class="empty">No completed runs to compare yet.</p></div>`;
    return;
  }
  // Drop selections for runs that no longer exist.
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
              · ${(r.algorithms || []).map(a => a.label).join(', ')}
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

async function renderCompareBody() {
  const body = $('#cmp-body');
  if (!body) return;
  if (CMP.sel.size < 1) {
    body.innerHTML = `<div class="card"><p class="empty">Select at least one run above.</p></div>`;
    return;
  }
  body.innerHTML = `<div class="card"><p class="empty">Loading…</p></div>`;

  const details = [];
  for (const id of CMP.sel) {
    if (!CMP.cache.has(id)) {
      try { CMP.cache.set(id, await api(`/api/runs/${id}`)); }
      catch (e) { CMP.sel.delete(id); continue; }
    }
    details.push(CMP.cache.get(id));
  }

  const mo = METRIC_OPTIONS.find(m => m.key === CMP.metric) || METRIC_OPTIONS[0];
  const rows = [];
  for (const d of details) {
    for (const a of (d.algorithms || [])) {
      const recs = (d.results || []).filter(r => r.algorithm_id === a.id);
      const solved = recs.filter(r => r.solved);
      rows.push({
        runId: d.id, runName: d.name, algoId: a.id, algoLabel: a.label,
        params: a.params, mapSet: d.map_set_name || d.map_set_id, dims: d.dims,
        theta: (d.metric_params || {}).steering_theta_degrees ?? 30,
        total: recs.length, solved: solved.length,
        meanTime: meanBy(solved, r => r.time_s),
        val: meanBy(solved, r => mo.src === 'metrics' ? (r.metrics && r.metrics[mo.key]) : r[mo.key]),
      });
    }
  }

  const mapSets = [...new Set(details.map(d => d.map_set_id))];
  const thetas = [...new Set(rows.map(r => r.theta))];
  const warns = [];
  if (mapSets.length > 1) warns.push('These runs used different map sets — metric differences may come from the maps, not the algorithms.');
  if (mo.src === 'metrics' && thetas.length > 1) warns.push(`Runs were scored with different steering θ (${thetas.join('°, ')}°) — use each run's Metrics lab to align them before comparing steering penalty.`);

  const withVal = rows.filter(r => r.val != null);
  const maxVal = Math.max(...withVal.map(r => r.val), 0) || 1;
  const barsHTML = withVal
    .sort((x, y) => x.val - y.val)
    .map(r => `<div class="bar-row">
      <div class="b-label" title="${esc(JSON.stringify(r.params))}">
        <span class="chip" style="${chipStyle(r.algoId)}"></span>
        <span>${esc(r.runName)} · ${esc(r.algoLabel)}</span>
      </div>
      <div class="b-track"><div class="b-fill" style="width:${Math.max(2, 100 * r.val / maxVal)}%"></div></div>
      <div class="b-val">${fmt(r.val, 3)}</div>
    </div>`).join('');

  body.innerHTML = `
    <div class="card">
      <h2>Comparison</h2>
      ${warns.map(w => `<div class="cmp-note">${esc(w)}</div>`).join('')}
      <div class="row">
        <div class="field" style="min-width:280px">
          <label>Metric (mean over solved maps)</label>
          <select id="cmp-tab-metric">
            ${METRIC_OPTIONS.map(m => `<option value="${m.key}" ${m.key === CMP.metric ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}
          </select>
        </div>
      </div>
      <div class="bars">${barsHTML || '<p class="empty">No solved results among selected runs.</p>'}</div>
      <div class="bar-note">Lower is better for all listed metrics.</div>
      <h3>Details</h3>
      <div style="overflow-x:auto"><table class="data">
        <thead><tr>
          <th>Run</th><th>Algorithm</th><th>Params</th><th>Map set</th>
          <th class="num">Solved</th><th class="num">Mean time (s)</th><th class="num">${esc(mo.label)}</th>
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr>
            <td><a class="plain" href="#" data-goto="${esc(r.runId)}">${esc(r.runName)}</a></td>
            <td><span class="algo-cell"><span class="chip" style="${chipStyle(r.algoId)}"></span>${esc(r.algoLabel)}</span></td>
            <td style="font-size:11.5px;color:var(--ink-2)">${esc(Object.entries(r.params || {}).map(([k, v]) => `${k}=${v}`).join(', '))}</td>
            <td>${esc(r.mapSet)} (${r.dims}D)</td>
            <td class="num">${r.solved}/${r.total}</td>
            <td class="num">${fmt(r.meanTime, 3)}</td>
            <td class="num">${fmt(r.val, 3)}</td>
          </tr>`).join('')}
        </tbody>
      </table></div>
    </div>`;

  $('#cmp-tab-metric').addEventListener('change', e => { CMP.metric = e.target.value; renderCompareBody(); });
  $$('[data-goto]', body).forEach(a => a.addEventListener('click', e => {
    e.preventDefault(); switchTab('results'); openRun(a.dataset.goto);
  }));
}

/* ===================== 3D VIEWER ===================== */

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

async function loadGrid(msId, mapName) {
  const key = `${msId}/${mapName}`;
  if (!GRIDS.has(key)) {
    const g = await api(`/api/map_sets/${msId}/grid/${mapName}`);
    const bin = atob(g.data_b64);
    const data = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
    GRIDS.set(key, { shape: g.shape, data });
  }
  return GRIDS.get(key);
}

/* Open the viewer for a run's map: overlays every solved algorithm path. */
async function openRunViewer(d, stem) {
  const recs = (d.results || []).filter(r => r.map_stem === stem && r.solved);
  const ms = S.mapSets.find(m => m.id === d.map_set_id);
  if (!ms) { alert('Map set no longer exists.'); return; }
  const entry = (ms.maps || []).find(m => m.stem === stem);
  if (!entry) { alert('Map not found in map set.'); return; }

  const pathSpecs = [];
  for (const r of recs) {
    try {
      const path = await api(`/api/runs/${d.id}/path/${r.algorithm_id}/${stem}`);
      pathSpecs.push({ algoId: r.algorithm_id, label: r.algorithm_label, path });
    } catch (e) { /* no stored path */ }
  }
  openViewerModal(ms, entry, pathSpecs);
}

/* Modal wrapper: map selector (within the set) + slice pane + rotatable 3D pane. */
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

  const [D, H, W] = grid.shape;
  const start = entry.start, end = entry.end;
  const enabled = new Map(pathSpecs.map(s => [s.algoId, true]));

  mount.innerHTML = `
    ${pathSpecs.length ? `<div class="viewer-toggles">
      ${pathSpecs.map(s => `<label class="check">
        <input type="checkbox" data-vt="${esc(s.algoId)}" checked>
        <span class="chip" style="${chipStyle(s.algoId)}"></span>${esc(s.label)}
      </label>`).join('')}
    </div>` : ''}
    <div class="viewer-grid">
      <div class="viewer-pane">
        <h4>Slice view (z-layer)</h4>
        <canvas id="v-slice"></canvas>
        <div class="v-controls">
          <input type="range" id="v-z" min="0" max="${D - 1}" value="${Math.min(start[2] ?? 0, D - 1)}">
          <span id="v-zlabel" style="min-width:70px"></span>
        </div>
        <span class="viewer-hint">Solid = path on this layer · faded = path within ±1 layer · green start · red end</span>
      </div>
      <div class="viewer-pane">
        <h4>3D view</h4>
        <canvas id="v-3d" width="460" height="460"></canvas>
        <span class="viewer-hint">Drag to rotate · scroll to zoom · gray = obstacles</span>
      </div>
    </div>`;

  /* ---- slice pane ---- */
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
      if (!enabled.get(s.algoId)) continue;
      const color = chipHex(s.algoId);
      for (const p of s.path) {
        const dz = Math.abs((p[2] ?? 0) - z);
        if (dz > 1) continue;
        ctx.globalAlpha = dz === 0 ? 1 : 0.3;
        ctx.fillStyle = color;
        ctx.fillRect(p[0] * cs, p[1] * cs, cs, cs);
      }
    }
    ctx.globalAlpha = 1;
    if ((start[2] ?? 0) === z) { ctx.fillStyle = '#0ca30c'; ctx.fillRect(start[0] * cs, start[1] * cs, cs, cs); }
    if ((end[2] ?? 0) === z) { ctx.fillStyle = '#d03b3b'; ctx.fillRect(end[0] * cs, end[1] * cs, cs, cs); }
  }
  zInput.addEventListener('input', drawSlice);

  /* ---- 3D pane ---- */
  const cv3 = $('#v-3d', mount);
  const ctx3 = cv3.getContext('2d');
  const view = { yaw: 0.7, pitch: 0.45, zoom: 1.0 };
  const cx = W / 2, cy = H / 2, cz = D / 2;
  const maxDim = Math.max(W, H, D);

  // Collect obstacle voxels once (subsample very dense grids for canvas perf).
  let voxels = [];
  for (let z = 0; z < D; z++) {
    for (let y = 0; y < H; y++) {
      const rowBase = z * H * W + y * W;
      for (let x = 0; x < W; x++) if (grid.data[rowBase + x]) voxels.push([x, y, z]);
    }
  }
  const MAX_VOX = 50000;
  if (voxels.length > MAX_VOX) {
    const stride = Math.ceil(voxels.length / MAX_VOX);
    voxels = voxels.filter((_, i) => i % stride === 0);
  }

  function project(x, y, z) {
    const s = view.zoom * (380 / maxDim);
    const x1 = (x - cx) * Math.cos(view.yaw) - (y - cy) * Math.sin(view.yaw);
    const y1 = (x - cx) * Math.sin(view.yaw) + (y - cy) * Math.cos(view.yaw);
    const z1 = (z - cz);
    const y2 = y1 * Math.cos(view.pitch) - z1 * Math.sin(view.pitch);
    const z2 = y1 * Math.sin(view.pitch) + z1 * Math.cos(view.pitch);
    return [x1 * s + cv3.width / 2, -z2 * s + cv3.height / 2, y2];
  }

  function draw3D() {
    ctx3.fillStyle = cssVar('--page');
    ctx3.fillRect(0, 0, cv3.width, cv3.height);

    // bounding box wireframe
    const corners = [[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0], [0, 0, D], [W, 0, D], [W, H, D], [0, H, D]];
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const pc = corners.map(c => project(c[0], c[1], c[2]));
    ctx3.strokeStyle = cssVar('--baseline');
    ctx3.lineWidth = 1;
    ctx3.beginPath();
    for (const [a, b] of edges) { ctx3.moveTo(pc[a][0], pc[a][1]); ctx3.lineTo(pc[b][0], pc[b][1]); }
    ctx3.stroke();

    // obstacles, far → near
    const projected = voxels.map(v => project(v[0] + 0.5, v[1] + 0.5, v[2] + 0.5));
    projected.sort((a, b) => b[2] - a[2]);
    const depths = projected.map(p => p[2]);
    const dMin = Math.min(...depths, 0), dMax = Math.max(...depths, 1);
    ctx3.fillStyle = cssVar('--muted');
    for (const p of projected) {
      const t = (dMax - p[2]) / (dMax - dMin + 1e-9);
      ctx3.globalAlpha = 0.10 + 0.22 * t;
      ctx3.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3);
    }
    ctx3.globalAlpha = 1;

    // paths on top
    for (const s of pathSpecs) {
      if (!enabled.get(s.algoId)) continue;
      ctx3.strokeStyle = chipHex(s.algoId);
      ctx3.lineWidth = 2;
      ctx3.beginPath();
      s.path.forEach((p, i) => {
        const q = project(p[0] + 0.5, p[1] + 0.5, (p[2] ?? 0) + 0.5);
        if (i === 0) ctx3.moveTo(q[0], q[1]); else ctx3.lineTo(q[0], q[1]);
      });
      ctx3.stroke();
    }

    const ps = project(start[0] + 0.5, start[1] + 0.5, (start[2] ?? 0) + 0.5);
    const pe = project(end[0] + 0.5, end[1] + 0.5, (end[2] ?? 0) + 0.5);
    ctx3.fillStyle = '#0ca30c'; ctx3.fillRect(ps[0] - 4, ps[1] - 4, 8, 8);
    ctx3.fillStyle = '#d03b3b'; ctx3.fillRect(pe[0] - 4, pe[1] - 4, 8, 8);
  }

  let dragging = false, lastX = 0, lastY = 0;
  cv3.addEventListener('pointerdown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; cv3.setPointerCapture(e.pointerId); });
  cv3.addEventListener('pointermove', e => {
    if (!dragging) return;
    view.yaw += (e.clientX - lastX) * 0.01;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (e.clientY - lastY) * 0.01));
    lastX = e.clientX; lastY = e.clientY;
    draw3D();
  });
  cv3.addEventListener('pointerup', () => { dragging = false; });
  cv3.addEventListener('wheel', e => {
    e.preventDefault();
    view.zoom = Math.max(0.3, Math.min(4, view.zoom * (e.deltaY < 0 ? 1.1 : 0.9)));
    draw3D();
  }, { passive: false });

  $$('[data-vt]', mount).forEach(cb => cb.addEventListener('change', () => {
    enabled.set(cb.dataset.vt, cb.checked);
    drawSlice();
    draw3D();
  }));

  drawSlice();
  draw3D();
}

function chipHex(algoId) {
  const idx = S.registry.findIndex(r => r.id === algoId);
  const slot = (idx >= 0 ? idx : 0) % 8 + 1;
  return cssVar(`--cat-${slot}`) || '#2a78d6';
}

/* ===================== init ===================== */

(async function init() {
  try {
    S.registry = await api('/api/registry');
    await refreshLists();
    S.registry.forEach(spec => {
      F.algos[spec.id] = { on: true,
                           params: Object.fromEntries(spec.params.map(p => [p.name, p.default])) };
    });
    renderJobsIndicator();
    renderRunTab();
    if (anyActivity()) ensurePolling();
  } catch (e) {
    document.body.innerHTML = `<div style="padding:40px;font-family:system-ui">
      <h2>Failed to load</h2><p>${esc(e.message)}</p></div>`;
  }
})();
