/* qualk web app: run, steer and monitor from the browser. Talks to qualk/server.py (REST + /ws). */
(() => {
  'use strict';
  const G = window.QualkGraph;
  const $ = id => document.getElementById(id);
  const esc = G.esc;
  const ACTIVE = ['starting', 'seeding', 'running', 'paused', 'stopping'];

  const S = {
    simulate: false, providers: {}, limits: {max_rounds: 200}, authRequired: false,
    status: {},            // run id -> latest status event
    runs: [],              // /api/runs listing
    current: null, models: {}, tab: 'graph', round: 0, follow: true, selected: null, threadSel: null,
    log: [], connected: false, timer: null,
  };
  let gv = null, gvA = null, gvB = null, viewFor = null, drawQueued = false, runsQueued = 0;

  /* ---- model of one run (replay data + live records) -------------------------------------- */
  class Model {
    constructor(data) {
      this.id = data.id; this.nodes = data.nodes || []; this.edges = data.edges || [];
      this.rounds = data.rounds || []; this.steer = data.steer || []; this.config = data.config || {};
      this.nodeIndex = new Map(this.nodes.map(n => [n.id, n]));
      this.edgeIds = new Set(this.edges.map(e => e.id));
      this.byRound = {}; this.seeds = []; this.maxRound = 0; this.override = null;
      this.rounds.forEach(r => this._note(r));
    }
    _note(r) {
      if (r.kind === 'seed') this.seeds.push(r); else this.byRound[r.round] = r;
      this.maxRound = Math.max(this.maxRound, r.round);
    }
    apply(rec) {
      if (rec.kind !== 'seed' && this.byRound[rec.round]) return;
      this.rounds.push(rec); this._note(rec);
      const d = rec.delta || {nodes: [], assertions: []};
      d.nodes.forEach(n => { if (!this.nodeIndex.has(n.id)) { const m = {...n, first: rec.round}; this.nodes.push(m); this.nodeIndex.set(n.id, m); } });
      d.assertions.forEach(a => {
        if (this.edgeIds.has(a.id) || (a.frame && a.frame !== 'world')) return;
        this.edgeIds.add(a.id);
        this.edges.push({id: a.id, s: a.subject, p: a.pred, o: a.object, conf: a.conf, val: a.valence, aff: a.affinity,
                         since: rec.round, until: null, role: a.role});
      });
    }
    threads(round) { return G.threadInfo(this.rounds, round, this.override); }
    get names() { return Object.fromEntries(this.nodes.map(n => [n.id, n.name])); }
  }

  /* ---- http ---------------------------------------------------------------------------------- */
  async function api(path, method = 'GET', body) {
    const res = await fetch(path, {method, headers: body ? {'Content-Type': 'application/json'} : {}, body: body ? JSON.stringify(body) : undefined});
    if (res.status === 401 && path !== '/api/login') { showLogin(); throw new Error('authentication required'); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    return data;
  }
  function toast(text, bad) { addLog({kind: bad ? 'error' : 'log', text}); if (bad) $('dock-note').textContent = text; }

  /* ---- live connection ------------------------------------------------------------------------ */
  let ws = null, retry = 1500;
  function connect() {
    ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    ws.onopen = () => { retry = 1500; setConn(true); };
    ws.onclose = () => { setConn(false); setTimeout(connect, retry); retry = Math.min(retry * 1.7, 10000); };
    ws.onmessage = ev => onEvent(JSON.parse(ev.data));
  }
  function setConn(on) {
    S.connected = on; $('conn').classList.toggle('on', on); $('conn').lastElementChild.textContent = on ? 'live' : 'reconnecting';
  }
  function onEvent(m) {
    if (m.type === 'hello') {
      S.simulate = m.simulate; S.providers = m.providers; S.limits = m.limits || S.limits; S.authRequired = m.auth_required;
      S.status = Object.fromEntries((m.status || []).map(s => [s.run, s]));
      (m.recent || []).forEach(addLog);
      refreshRuns().then(() => { if (S.current) reloadCurrent(); }); draw();
    } else if (m.type === 'status') {
      const before = S.status[m.run] && S.status[m.run].state;
      S.status[m.run] = m;
      if (!S.runs.some(r => r.id === m.run) || (before !== m.state && !ACTIVE.includes(m.state))) refreshRuns();
      if (before && before !== m.state && m.state === 'done' && m.run === S.current) reloadCurrent();
      draw();
    } else if (m.type === 'round') {
      const model = S.models[m.run];
      if (model) {
        model.apply(m.record);
        if (m.run === S.current && S.follow) S.round = model.maxRound;
        draw();
      }
    } else if (m.type === 'deleted') {
      forgetRuns(m.runs || []);
    } else if (m.type === 'threads') {
      const model = S.models[m.run];
      if (model) { model.override = {round: m.round, table: m.table, focus: m.focus}; draw(); }
    } else if (m.type === 'log' || m.type === 'error' || m.type === 'steer') { addLog(m); if (m.type === 'steer') draw(); }
  }
  function addLog(m) {
    const t = m.t ? new Date(m.t * 1000) : new Date();
    const text = m.type === 'steer' || m.kind === 'steer' ? `steer ${m.kind === 'steer' ? '' : m.kind} ${JSON.stringify(m.changes || {node: m.node, source: m.source})}` : m.text;
    S.log.push({t: t.toLocaleTimeString(), run: m.run || '', cls: m.type === 'error' || m.kind === 'error' ? 'err' : (m.type === 'steer' ? 'steer' : ''), text});
    if (S.log.length > 300) S.log.shift();
    if (S.tab === 'log') drawLog();
  }

  /* ---- runs -------------------------------------------------------------------------------------- */
  async function refreshRuns() {
    const now = Date.now(); if (now - runsQueued < 400) return; runsQueued = now;
    try { const d = await api('/api/runs'); S.runs = d.runs; draw(); } catch (e) { /* login or offline */ }
  }
  async function loadModel(id) {
    const data = await api('/api/runs/' + encodeURIComponent(id));
    S.models[id] = new Model(data);
    return S.models[id];
  }
  function runInfo(id) { return S.runs.find(r => r.id === id) || {}; }
  function partnerId(id) {
    const st = S.status[id]; const info = runInfo(id);
    return ((st && st.paired_with) || info.paired_with || [])[0] || null;
  }
  async function selectRun(id) {
    S.current = id; S.selected = null; S.follow = true; viewFor = null;
    try {
      const m = await loadModel(id);
      const p = partnerId(id); if (p) await loadModel(p).catch(() => {});
      S.round = m.maxRound;
    } catch (e) { toast(e.message, true); }
    if (!partnerId(id) && S.tab === 'compare') S.tab = 'graph';
    draw();
  }
  async function reloadCurrent() { if (S.current) { try { await loadModel(S.current); const p = partnerId(S.current); if (p) await loadModel(p); } catch (e) {} draw(); } }
  const cur = () => S.models[S.current];

  function liveState(id) { const st = S.status[id]; return st ? st.state : (runInfo(id).state || 'idle'); }

  /* ---- drawing ------------------------------------------------------------------------------------ */
  function draw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(() => { drawQueued = false; render(); }); } }

  function render() {
    $('sim-banner').hidden = !S.simulate;
    $('btn-logout').hidden = !S.authRequired;
    $('pills').innerHTML = ['search', 'llm', 'atlas'].map(k => {
      const on = S.simulate || S.providers[k];
      return `<span class="pill ${on ? 'ok' : 'off'}" title="${S.simulate ? 'simulated' : (on ? 'configured' : 'not configured')}">${k}</span>`;
    }).join('');
    renderRuns(); renderDock(); renderTabs(); renderTimeline(); renderInspector(); renderQPanel(); renderSteer();
    if (S.tab === 'graph') renderGraph(); else if (S.tab === 'compare') renderCompare(); else if (S.tab === 'threads') renderThreads(); else drawLog();
  }

  function renderRuns() {
    const ids = new Set(S.runs.map(r => r.id));
    const extra = Object.keys(S.status).filter(id => !ids.has(id)).map(id => ({id, cfg: S.status[id].cfg, round: S.status[id].round, nodes: 0, paired_with: S.status[id].paired_with, role: S.status[id].role}));
    $('run-list').innerHTML = [...extra, ...S.runs].map(r => {
      const st = liveState(r.id), live = S.status[r.id];
      const label = ACTIVE.includes(st) || st === 'done' || st === 'stopped' || st === 'error' ? st : 'idle';
      const prog = live ? `${live.done}/${live.total}` : `r${r.round}`;
      return `<div class="run ${r.id === S.current ? 'sel' : ''}" data-id="${esc(r.id)}"><button class="rdel" data-del="${esc(r.id)}" title="${ACTIVE.includes(st) ? 'Stop the run to delete it' : 'Delete this run'}" ${ACTIVE.includes(st) ? 'disabled' : ''}>&times;</button><b title="${esc(r.id)}">${(r.paired_with || []).length ? '&#8644; ' : ''}${esc(r.id)}</b>` +
        `<div class="meta"><span class="chip ${label}">${label}</span><span>${prog} &middot; ${r.nodes || '-'} nodes</span></div></div>`;
    }).join('') || '<div class="muted small">No runs yet.</div>';
  }

  function renderDock() {
    const id = S.current, st = id ? liveState(id) : '-', live = id && S.status[id];
    $('dock-name').textContent = id || 'No run selected';
    const chip = $('dock-state'); chip.textContent = st; chip.className = 'chip ' + st;
    const active = ACTIVE.includes(st);
    $('c-resume').disabled = st !== 'paused'; $('c-pause').disabled = st !== 'running';
    $('c-step').disabled = st !== 'paused'; $('c-stop').disabled = !active || st === 'stopping';
    $('c-continue-box').hidden = !id || active;
    $('c-delete').hidden = !id || active;
    $('progress-bar').style.width = live && live.total ? Math.min(100, 100 * live.done / live.total) + '%' : (id && !active ? '0%' : '0%');
    let note = live && live.error ? live.error : '';
    if (live && live.pinned) note += ` pinned next seed: ${(cur() && cur().names[live.pinned]) || live.pinned}.`;
    if (live && live.pending) note += ` ${live.pending} steering action(s) waiting for the next round.`;
    if (st === 'paused') note += ' Paused: Step runs one round; steering is applied when a round starts.';
    if (id && active && partnerId(id)) note += ' Paired run: controls apply to both runs.';
    $('dock-note').textContent = note.trim();
  }

  function renderTabs() {
    $('tab-compare').hidden = !(S.current && partnerId(S.current));
    const qi = cur() && cur().threads(S.round);
    $('tab-threads').textContent = qi ? `Questions (${qi.totals.open} open)` : 'Questions';
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.tab === S.tab));
    ['graph', 'compare', 'threads', 'log'].forEach(v => $('view-' + v).hidden = v !== S.tab);
  }

  function renderTimeline() {
    const m = cur(), max = m ? m.maxRound : 0;
    $('slider').max = max; $('slider').value = Math.min(S.round, max);
    $('rlabel').textContent = `round ${Math.min(S.round, max)} / ${max}`;
    $('follow').checked = S.follow;
    $('play').textContent = S.timer ? 'Pause' : 'Play';
  }

  function recAt(m, r) { return r === 0 ? m.seeds[0] : m.byRound[r]; }

  function freshView(svgId, current) {
    d3.select('#' + svgId).selectAll('*').remove();
    if (svgId !== 'graph') return new G.GraphView($(svgId));
    return new G.GraphView($(svgId), {onNodeClick: n => selectNode(S.selected === n.id ? null : n.id),
                                      onBackgroundClick: () => { S.threadSel = null; selectNode(null); }});
  }

  function renderGraph() {
    const m = cur();
    $('graph-empty').hidden = !!m && (m.nodes.length > 0);
    if (!m) return;
    if (viewFor !== S.current + '|graph') { gv = freshView('graph'); viewFor = S.current + '|graph'; }
    const info = m.threads(S.round), rec = recAt(m, S.round);
    gv.render(m.nodes, m.edges, S.round, rec, S.selected,
              info ? {threads: info.table, threadSel: S.threadSel, focus: info.focus, aimed: rec && rec.probe && rec.probe.thread} : {});
  }

  function selectNode(id) {
    S.selected = id;
    if (id) S.threadSel = null;
    const n = id && cur() && cur().nodeIndex.get(id);
    if (n) $('pin-input').value = n.name;
    draw();
  }

  function renderInspector() {
    const m = cur();
    if (!m) { $('side').innerHTML = '<div class="muted">Select a run to inspect its rounds.</div>'; return; }
    const rec = S.round === 0 ? null : m.byRound[S.round];
    const node = S.selected && m.nodeIndex.get(S.selected);
    const active = ACTIVE.includes(liveState(m.id));
    const card = node ? G.nodeCardHTML(node, m.edges, m.names, S.round, {pin: active}) : '';
    $('side').innerHTML = card + G.roundPanelHTML(rec, m.names, m.seeds, {qasmHref: r => `/api/runs/${encodeURIComponent(m.id)}/qasm/${r}`}) +
      (S.round === 0 ? '<div class="muted small">Each round the walk picks a concept, the web is searched for it, and what is read is added to the graph.</div>' : '');
  }

  /* ---- steering panel ------------------------------------------------------------------------------ */
  function renderQPanel() {
    const m = cur(), info = m && m.threads(S.round), box = $('qpanel');
    box.hidden = !info;
    if (info) box.innerHTML = G.openQuestionsHTML(info);
  }

  function renderSteer() {
    const id = S.current, st = id ? liveState(id) : '-', live = id && S.status[id];
    const show = !!id && ACTIVE.includes(st) && !!live;
    $('steer').hidden = !show;
    if (!show) return;
    const form = $('params-form');
    if (!form.contains(document.activeElement)) {
      Object.entries(live.cfg).forEach(([k, v]) => { const f = form.elements[k]; if (f) f.value = v; });
    }
    form.elements.walk.disabled = live.role === 'ctl' || (partnerId(id) != null);
    [...form.elements.backend.options].forEach(o => o.disabled = o.value === 'atlas' && !S.providers.atlas && !S.simulate);
    const m = cur();
    if (m) {
      const dl = $('pin-list'); const names = m.nodes.filter(n => n.kind !== 'thread').map(n => n.name);
      if (dl.dataset.n !== String(names.length)) { dl.dataset.n = names.length; dl.innerHTML = names.map(n => `<option value="${esc(n)}">`).join(''); }
    }
    $('pin-note').textContent = live.pinned ? `next seed pinned: ${(m && m.names[live.pinned]) || live.pinned}` : '';
  }

  async function steer(action, body) {
    try { const r = await api(`/api/runs/${encodeURIComponent(S.current)}/${action}`, 'POST', body || {}); return r; }
    catch (e) { toast(`${action}: ${e.message}`, true); }
  }

  /* ---- compare ---------------------------------------------------------------------------------------- */
  function renderCompare() {
    const a = cur(), pid = a && partnerId(a.id), b = pid && S.models[pid];
    if (!a || !b) return;
    const main = a.id.endsWith('-ctl') ? b : a, ctl = a.id.endsWith('-ctl') ? a : b;
    $('cmp-title-a').textContent = `${main.id}: quantum walk`; $('cmp-title-b').textContent = `${ctl.id}: diffusion control`;
    if (viewFor !== S.current + '|cmp') { gvA = freshView('graph-a'); gvB = freshView('graph-b'); viewFor = S.current + '|cmp'; }
    gvA.render(main.nodes, main.edges, S.round, recAt(main, S.round));
    gvB.render(ctl.nodes, ctl.edges, S.round, recAt(ctl, S.round));
    const lastPerRound = (m, keep) => [...new Map(m.rounds.filter(keep).map(r => [r.round, r])).values()];
    const series = (m, key) => lastPerRound(m, r => r.graph).map(r => ({x: r.round, y: r.graph[key]}));
    const nov = m => m.rounds.filter(r => r.novelty != null).map(r => ({x: r.round, y: r.novelty}));
    const two = (fn, fmt) => [{name: 'quantum', color: '--q', points: fn(main)}, {name: 'diffusion', color: '--c', points: fn(ctl)}];
    const upto = Math.max(main.maxRound, ctl.maxRound);
    const names = (m, r) => new Set(m.nodes.filter(n => n.first <= r).map(n => n.name.toLowerCase()));
    const overlap = [];
    for (let r = 0; r <= upto; r++) {
      const A = names(main, r), B = names(ctl, r), inter = [...A].filter(x => B.has(x)).length, union = new Set([...A, ...B]).size;
      overlap.push({x: r, y: union ? inter / union : 1});
    }
    const mark = S.round;
    G.lineChart($('ch-nodes'), two(m => series(m, 'nodes')), {title: 'concepts in the graph', mark});
    G.lineChart($('ch-rel'), two(m => series(m, 'relations')), {title: 'relation edges (walkable)', mark});
    G.lineChart($('ch-nov'), two(nov), {title: 'observation novelty per round', yMax: 1, mark});
    G.lineChart($('ch-overlap'), [{name: 'overlap', color: '--hit', points: overlap}], {title: 'concept overlap between the two graphs (Jaccard)', yMax: 1, mark});
    const ti = m => G.threadInfo(m.rounds, upto);
    const im = ti(main), ic = ti(ctl);
    G.lineChart($('ch-answered'), [{name: 'quantum', color: '--q', points: im ? im.series.answered : []},
                                  {name: 'diffusion', color: '--c', points: ic ? ic.series.answered : []}], {title: 'questions answered (cumulative)', mark});
    G.lineChart($('ch-open'), [{name: 'quantum', color: '--q', points: im ? im.series.open : []},
                              {name: 'diffusion', color: '--c', points: ic ? ic.series.open : []}], {title: 'questions open', mark});
    const mean = xs => xs.length ? xs.reduce((s, v) => s + v.y, 0) / xs.length : null;
    const tvds = main.rounds.filter(r => r.probe && r.probe.walk && r.probe.walk.total_variation_distance != null).map(r => ({y: r.probe.walk.total_variation_distance}));
    const f = v => v == null ? '-' : v.toFixed(3);
    $('cmp-stats').innerHTML = `<span>mean novelty quantum <b>${f(mean(nov(main)))}</b></span><span>diffusion <b>${f(mean(nov(ctl)))}</b></span>` +
      `<span>mean walk TVD (quantum vs diffusion on the same window) <b>${f(mean(tvds))}</b></span>` +
      `<span>answered <b>${im ? im.totals.answered : '-'}</b> vs <b>${ic ? ic.totals.answered : '-'}</b></span>` +
      `<span>concepts <b>${main.nodes.length}</b> vs <b>${ctl.nodes.length}</b></span><span>overlap now <b>${f(overlap[Math.min(S.round, upto)]?.y)}</b></span>`;
  }

  function renderThreads() {
    const m = cur(), box = $('th-list');
    const info = m && m.threads(S.round);
    if (!info) { $('th-kpis').innerHTML = ''; $('ch-threads').innerHTML = ''; $('th-activity').innerHTML = ''; box.innerHTML = '<div class="muted small">This run does not track questions (it was started with "Track open questions" off, or before questions existed).</div>'; return; }
    const t = info.totals, live = S.status[m.id], active = ACTIVE.includes(liveState(m.id));
    $('th-kpis').innerHTML =
      `<div class="kpi big"><b>${t.answered}</b><span>answered of ${t.opened} raised</span></div>` +
      `<div class="kpi"><b>${t.open}</b><span>open now</span></div><div class="kpi"><b>${t.advanced}</b><span>moved forward at least once</span></div>` +
      `<div class="kpi"><b>${info.aimed}</b><span>of ${info.probes} probes aimed at a question</span></div>` +
      `<div class="kpi"><b>${t.dormant + t.closed}</b><span>let go (dormant / closed)</span></div>` +
      `<div class="kpi" title="mean cosine between open questions: rising means they are collapsing onto one theme"><b>${info.spread == null ? '-' : info.spread.toFixed(2)}</b><span>question overlap</span></div>`;
    G.lineChart($('ch-threads'), [
      {name: 'raised', color: '--muted', points: info.series.opened},
      {name: 'open', color: '--q', points: info.series.open},
      {name: 'answered', color: '--ok', points: info.series.answered}], {title: 'questions: raised (grey), open (violet), answered (green)', mark: S.round});
    $('th-activity').innerHTML = G.threadActivityHTML(info);
    box.innerHTML = G.threadCardsHTML(info, {actions: active, selected: S.threadSel, focus: info.focus, names: m.names});
  }

  function drawLog() {
    const el = $('log');
    el.innerHTML = S.log.slice(-300).map(l => `<div class="${l.cls}">${esc(l.t)} ${l.run ? '[' + esc(l.run) + '] ' : ''}${esc(l.text)}</div>`).join('') || '<div class="muted">No events yet.</div>';
    el.scrollTop = el.scrollHeight;
  }

  /* ---- dialogs -------------------------------------------------------------------------------------------- */
  function openNew() {
    const f = $('new-form');
    [...f.elements.backend.options].forEach(o => o.disabled = o.value === 'atlas' && !S.providers.atlas && !S.simulate);
    f.elements.rounds.max = S.limits.max_rounds; $('new-error').hidden = true; $('dlg-new').showModal();
  }
  async function submitNew(ev) {
    ev.preventDefault();
    const f = $('new-form').elements, num = k => Number(f[k].value);
    const body = {topic: f.topic.value.trim(), seed: f.seed.value, rounds: num('rounds'), walk: f.walk.value, paired: f.paired.checked,
                  start_paused: f.start_paused.checked, threads: f.threads.checked, thread_aim: num('thread_aim'), dedupe: num('dedupe'), nodes: num('nodes'), steps: num('steps'), time: num('time'), shots: num('shots'),
                  backend: f.backend.value, explore: num('explore'), rng: num('rng'), name: f.name.value.trim()};
    if (!body.topic && !body.seed.trim()) return showNewError('Give a topic or a seed text.');
    if (body.paired && body.walk !== 'quantum') return showNewError('A paired run needs the quantum walk.');
    $('new-go').disabled = true;
    try {
      const r = await api('/api/runs', 'POST', body);
      $('dlg-new').close(); await refreshRuns();
      S.tab = 'graph'; await selectRun(r.ids[0]);
    } catch (e) { showNewError(e.message); } finally { $('new-go').disabled = false; }
  }
  function showNewError(t) { const e = $('new-error'); e.textContent = t; e.hidden = false; }

  async function openSettings() {
    const d = await api('/api/settings');
    const box = $('settings-fields'); box.innerHTML = '';
    Object.entries(d.settings).forEach(([key, s]) => {
      const row = document.createElement('div'); row.className = 'field';
      row.innerHTML = `<label for="set-${key}">${esc(s.label)}</label>` +
        `<input id="set-${key}" data-key="${key}" data-secret="${s.secret}" type="${s.secret ? 'password' : 'text'}" autocomplete="off" ` +
        `placeholder="${s.secret ? (s.set ? 'unchanged' : 'not set') : esc(s.placeholder)}" value="${s.secret ? '' : esc(s.value || '')}" ${d.locked ? 'disabled' : ''}>` +
        (s.secret ? `<span class="badge ${s.set ? '' : 'no'}">${s.set ? 'set' : 'not set'}</span>` : '<span></span>');
      box.appendChild(row);
    });
    $('settings-save').disabled = d.locked;
    $('settings-msg').textContent = d.locked ? 'A run is active: stop it to change settings.' : '';
    $('dlg-settings').showModal();
  }
  async function saveSettings(ev) {
    ev.preventDefault();
    const set = {};
    document.querySelectorAll('#settings-fields input').forEach(i => { if (i.value.trim()) set[i.dataset.key] = i.value.trim(); });
    try {
      const r = await api('/api/settings', 'PUT', {set});
      S.providers = r.providers;
      $('settings-msg').textContent = r.changed.length ? `Saved: ${r.changed.join(', ')}` : 'Nothing changed.';
      await openSettingsFieldsOnly(r.settings); draw();
    } catch (e) { $('settings-msg').textContent = e.message; }
  }
  async function openSettingsFieldsOnly(settings) {
    document.querySelectorAll('#settings-fields input').forEach(i => {
      const s = settings[i.dataset.key]; if (i.dataset.secret === 'true') { i.value = ''; i.placeholder = s.set ? 'unchanged' : 'not set'; i.nextElementSibling.textContent = s.set ? 'set' : 'not set'; i.nextElementSibling.className = 'badge ' + (s.set ? '' : 'no'); }
    });
  }
  async function runTest(what) {
    const msg = $('settings-msg'); msg.className = 'muted small'; msg.textContent = `testing ${what}...`;
    try { const r = await api('/api/settings/test/' + what, 'POST'); msg.className = 'small test-res ' + (r.ok ? 'ok' : 'bad'); msg.textContent = `${what}: ${r.ok ? 'OK' : 'FAILED'} - ${r.detail}`; }
    catch (e) { msg.className = 'small test-res bad'; msg.textContent = `${what}: ${e.message}`; }
  }

  /* ---- deleting runs ---------------------------------------------------------------------------------------- */
  let pendingDelete = null;
  function askDelete(id) {
    const partner = partnerId(id);
    pendingDelete = id;
    $('delete-text').innerHTML = `Delete <b>${esc(id)}</b>` + (partner ? ` <b>and its paired run ${esc(partner)}</b>` : '') + '?';
    $('delete-error').hidden = true; $('dlg-delete').showModal();
  }
  async function confirmDelete(ev) {
    ev.preventDefault();
    $('delete-go').disabled = true;
    try {
      const r = await api('/api/runs/' + encodeURIComponent(pendingDelete), 'DELETE');
      $('dlg-delete').close();
      const left = Object.entries(r.leftover || {});
      if (left.length) toast('Deleted what the app wrote; kept files it does not recognise: ' + left.map(([k, v]) => `${k}: ${v.join(', ')}`).join('; '), true);
      forgetRuns(r.deleted.concat(left.map(([k]) => k)));
    } catch (e) { const el = $('delete-error'); el.textContent = e.message; el.hidden = false; }
    finally { $('delete-go').disabled = false; }
  }
  function forgetRuns(ids) {
    ids.forEach(id => { delete S.models[id]; delete S.status[id]; });
    if (ids.includes(S.current)) { S.current = null; S.selected = null; S.threadSel = null; viewFor = null; S.round = 0; }
    runsQueued = 0; refreshRuns();
  }

  /* ---- login ------------------------------------------------------------------------------------------------ */
  function showLogin() { $('login').hidden = false; }
  async function submitLogin(ev) {
    ev.preventDefault();
    try { await api('/api/login', 'POST', {token: $('login-token').value.trim()}); location.reload(); }
    catch (e) { const el = $('login-error'); el.textContent = e.message; el.hidden = false; }
  }

  /* ---- wiring ----------------------------------------------------------------------------------------------------- */
  function bind() {
    $('side').addEventListener('click', e => {
      const link = e.target.closest('a[data-node]');
      if (link) { e.preventDefault(); return selectNode(link.dataset.node); }
      const act = e.target.closest('button[data-act]');
      if (!act) return;
      if (act.dataset.act === 'clear') selectNode(null);
      else if (act.dataset.act === 'pin' && S.selected) steer('pin', {node_id: S.selected});
    });
    addEventListener('keydown', e => { if (e.key === 'Escape' && S.selected && !document.querySelector('dialog[open]')) selectNode(null); });
    $('qpanel').addEventListener('click', e => {
      if (e.target.closest('button[data-act=questions]')) { S.tab = 'threads'; viewFor = null; return draw(); }
      const q = e.target.closest('.oq');
      if (q) { S.threadSel = S.threadSel === q.dataset.thread ? null : q.dataset.thread; S.selected = null; S.tab = 'graph'; viewFor = null; draw(); }
    });
    $('run-list').addEventListener('click', e => {
      const del = e.target.closest('button[data-del]');
      if (del) { e.stopPropagation(); return askDelete(del.dataset.del); }
      const el = e.target.closest('.run'); if (el) selectRun(el.dataset.id);
    });
    $('c-delete').onclick = () => S.current && askDelete(S.current);
    $('delete-cancel').onclick = () => $('dlg-delete').close();
    $('delete-form').onsubmit = confirmDelete;
    $('btn-new').onclick = openNew; $('new-cancel').onclick = () => $('dlg-new').close(); $('new-form').onsubmit = submitNew;
    $('btn-settings').onclick = () => openSettings().catch(e => toast(e.message, true));
    $('settings-close').onclick = () => $('dlg-settings').close(); $('settings-form').onsubmit = saveSettings;
    document.querySelectorAll('#settings-tests button').forEach(b => b.onclick = () => runTest(b.dataset.test));
    $('btn-logout').onclick = async () => { await api('/api/logout', 'POST'); location.reload(); };
    $('login-form').onsubmit = submitLogin;
    document.querySelectorAll('.tab').forEach(t => t.onclick = () => { S.tab = t.dataset.tab; viewFor = null; draw(); });
    ['resume', 'pause', 'step', 'stop'].forEach(a => $('c-' + a).onclick = () => steer(a));
    $('c-continue').onclick = async () => {
      try { await api(`/api/runs/${encodeURIComponent(S.current)}/continue`, 'POST', {rounds: Number($('c-continue-n').value) || 10}); S.follow = true; }
      catch (e) { toast(e.message, true); }
    };
    $('params-form').onsubmit = async ev => {
      ev.preventDefault();
      const live = S.status[S.current], f = $('params-form').elements, changes = {};
      ['walk', 'backend', 'nodes', 'steps', 'time', 'shots', 'explore', 'rounds', 'thread_aim', 'dedupe'].forEach(k => {
        if (f[k].disabled) return;
        const v = ['walk', 'backend'].includes(k) ? f[k].value : Number(f[k].value);
        if (v !== live.cfg[k]) changes[k] = v;
      });
      if (!Object.keys(changes).length) return toast('No parameter changed.');
      await steer('params', changes);
    };
    $('ask-form').onsubmit = async ev => {
      ev.preventDefault();
      const question = $('ask-q').value.trim(), involves = $('ask-involves').value.split(',').map(x => x.trim()).filter(Boolean);
      if (question.length < 4) return toast('Write the question first.', true);
      const r = await steer('ask', {question, involves, focus: $('ask-focus').checked});
      if (r) { $('ask-q').value = ''; $('ask-involves').value = ''; S.tab = 'threads'; draw(); }
    };
    $('th-list').addEventListener('click', async e => {
      const chip = e.target.closest('a[data-node]');
      if (chip) { e.preventDefault(); S.tab = 'graph'; viewFor = null; return selectNode(chip.dataset.node); }
      const btn = e.target.closest('button[data-thread-act]'), card = e.target.closest('[data-thread]');
      if (!btn || !card) return;
      const id = card.dataset.thread, act = btn.dataset.threadAct, info = cur().threads(S.round);
      if (act === 'show') { S.threadSel = S.threadSel === id ? null : id; S.selected = null; if (S.threadSel) { S.tab = 'graph'; viewFor = null; } draw(); }
      else if (act === 'focus') await steer('focus', {thread_id: info && info.focus === id ? null : id});
      else if (act === 'close') await steer('close_thread', {thread_id: id});
    });
    $('pin-go').onclick = async () => {
      const name = $('pin-input').value.trim().toLowerCase(), m = cur();
      const node = m && (m.nodeIndex.get(S.selected) && m.nodeIndex.get(S.selected).name.toLowerCase() === name ? m.nodeIndex.get(S.selected)
        : m.nodes.find(n => n.name.toLowerCase() === name));
      if (!node) return toast('Pick a concept from the list or click a node.', true);
      await steer('pin', {node_id: node.id});
    };
    $('inject-go').onclick = async () => {
      const url = $('inject-url').value.trim(), text = $('inject-text').value.trim();
      if (!!url === !!text) return toast('Give either a URL or some text.', true);
      const r = await steer('inject', url ? {url} : {text});
      if (r) { $('inject-url').value = ''; $('inject-text').value = ''; }
    };
    $('slider').oninput = () => { S.follow = false; S.round = Number($('slider').value); stopPlay(); draw(); };
    $('follow').onchange = () => { S.follow = $('follow').checked; if (S.follow && cur()) S.round = cur().maxRound; draw(); };
    $('play').onclick = () => {
      if (S.timer) return stopPlay();
      const m = cur(); if (!m) return;
      S.follow = false; if (S.round >= m.maxRound) S.round = 0;
      S.timer = setInterval(() => { if (S.round >= cur().maxRound) return stopPlay(); S.round += 1; draw(); }, 1200); draw();
    };
    addEventListener('resize', () => { viewFor = null; draw(); });
  }
  function stopPlay() { if (S.timer) { clearInterval(S.timer); S.timer = null; } draw(); }

  async function boot() {
    bind();
    const params = new URLSearchParams(location.search);
    if (params.get('token')) {
      try { await api('/api/login', 'POST', {token: params.get('token')}); } catch (e) { /* fall through to the login form */ }
      history.replaceState(null, '', location.pathname);
    }
    const a = await fetch('/api/auth').then(r => r.json());
    S.authRequired = a.required;
    if (a.required && !a.ok) return showLogin();
    try { const st = await api('/api/state'); S.simulate = st.simulate; S.providers = st.providers; S.limits = st.limits; S.status = Object.fromEntries((st.status || []).map(s => [s.run, s])); } catch (e) { return; }
    await refreshRuns();
    const active = Object.values(S.status).find(s => ACTIVE.includes(s.state));
    const first = active ? active.run : (S.runs[0] && S.runs[0].id);
    if (first) await selectRun(first);
    connect(); draw();
  }
  boot();
})();
