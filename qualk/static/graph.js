/* Shared drawing code: the world-graph view, the round inspector and the small charts.
 * Used by the web app (static/app.js) and inlined into the static export (viewer.py).
 * Plain script: defines window.QualkGraph. Requires d3 v7. */
(function () {
  const KIND_HUES = {agent: 265, site: 200, object: 30, event: 350, motif: 150, thread: 50};
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const pct = x => (x == null ? '-' : (100 * x).toFixed(1) + '%');
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const isDark = () => {
    const t = document.documentElement.dataset.theme;
    return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  };
  const kindColor = k => `hsl(${KIND_HUES[k] ?? 210}, ${isDark() ? 55 : 50}%, ${isDark() ? 66 : 48}%)`;
  const pairKey = (a, b) => [a, b].sort().join('|');

  /* ---- threads: shared helpers ------------------------------------------------------------ */
  const threadColor = i => `hsl(${(i * 67 + 20) % 360}, ${isDark() ? 60 : 55}%, ${isDark() ? 62 : 45}%)`;
  const srcLabel = src => {
    if (!src) return '';
    if (src.startsWith('web:http')) { try { return new URL(src.slice(4)).hostname.replace(/^www\./, ''); } catch (e) { return 'web'; } }
    if (src.startsWith('inject:')) return 'injected';
    return src.split(':')[0];
  };
  const CLOSURES = ['resolved', 'dormant', 'capped', 'closed'];

  /* Everything the UI needs to know about threads up to round `upto`, from the round records.
   * `override` ({round, table}) is a newer table from a steering action that has not produced a
   * round record yet. Returns null when the run has no thread data at all. */
  function threadInfo(rounds, upto, override) {
    const recs = rounds.filter(r => r.threads && r.round <= upto);
    if (!recs.length && !(override && override.table)) return null;
    let table = recs.length ? recs[recs.length - 1].threads.table : [];
    if (override && override.table && upto >= override.round) table = override.table;
    const events = [];
    recs.forEach(r => (r.threads.events || []).forEach(e => events.push({...e, round: r.round})));
    const per = {};
    const slot = id => (per[id] = per[id] || {events: [], aimed: []});
    events.forEach(e => slot(e.id).events.push(e));
    let probes = 0, aimed = 0;
    rounds.filter(r => r.kind === 'probe' && r.probe && r.round <= upto).forEach(r => {
      probes++;
      if (r.probe.thread) { aimed++; slot(r.probe.thread).aimed.push(r.round); }
    });
    const lastPerRound = new Map();
    recs.forEach(r => lastPerRound.set(r.round, r.threads));
    const cum = {opened: 0, answered: 0, other: 0};
    const byRoundEvents = {};
    events.forEach(e => (byRoundEvents[e.round] = byRoundEvents[e.round] || []).push(e));
    const open = [], answered = [], opened = [];
    [...lastPerRound.keys()].sort((a, b) => a - b).forEach(k => {
      (byRoundEvents[k] || []).forEach(e => {
        if (e.event === 'opened') cum.opened++; else if (e.event === 'resolved') cum.answered++; else if (CLOSURES.includes(e.event)) cum.other++;
      });
      open.push({x: k, y: lastPerRound.get(k).open}); answered.push({x: k, y: cum.answered}); opened.push({x: k, y: cum.opened});
    });
    const has = k => events.filter(e => e.event === k).length;
    return {table, events, per, probes, aimed, series: {open, answered, opened},
            totals: {opened: has('opened'), advanced: new Set(events.filter(e => e.event === 'advanced').map(e => e.id)).size,
                     answered: has('resolved'), dormant: has('dormant') + has('capped'), closed: has('closed'),
                     open: table.filter(t => t.state !== 'resolved').length},
            spread: recs.length ? recs[recs.length - 1].threads.spread : null,
            focus: override && override.table && upto >= override.round ? override.focus : (recs.length ? recs[recs.length - 1].threads.focus : null)};
  }

  function threadStatus(row, info) {
    const ev = (info.per[row.id] || {events: []}).events;
    if (row.state === 'resolved') {
      const closing = [...ev].reverse().find(e => CLOSURES.includes(e.event));
      return closing ? closing.event : 'resolved';
    }
    return ev.some(e => e.event === 'advanced') || row.state === 'complicated' ? 'advancing' : 'open';
  }
  const STATUS_LABEL = {open: 'open', advancing: 'advancing', resolved: 'answered', dormant: 'dormant', capped: 'dropped', closed: 'closed'};
  const EVENT_LABEL = {opened: 'raised', advanced: 'advanced', resolved: 'answered', dormant: 'went dormant', capped: 'dropped (pool full)', closed: 'closed by you'};

  /* The questions as cards: the question, what is currently understood, the concepts it rests on,
   * and the trail of pages that moved it. opts: {actions, selected, focus, names} */
  function threadCardsHTML(info, opts = {}) {
    if (!info || !info.table.length) return '<div class="muted small">No questions yet: they are raised when a page leaves something unanswered.</div>';
    const names = opts.names || {};
    const rank = r => ({advancing: 0, open: 0, resolved: 1, closed: 2, dormant: 2, capped: 2}[threadStatus(r, info)] ?? 2);
    const rows = [...info.table].sort((a, b) => rank(a) - rank(b) || (rank(a) === 0 ? b.pressure_eff - a.pressure_eff : b.last - a.last));
    return rows.map(t => {
      const st = threadStatus(t, info), per = info.per[t.id] || {events: [], aimed: []};
      const idx = info.table.findIndex(x => x.id === t.id);
      const trail = per.events.map(e => `<div class="ev"><span class="rn">r${e.round}</span> <b>${EVENT_LABEL[e.event] || e.event}</b>` +
        `${e.event === 'advanced' || e.event === 'resolved' ? ': ' + esc((e.gist || '').slice(0, 140)) : ''}` +
        `${e.source ? ' <span class="muted">via ' + esc(srcLabel(e.source)) + '</span>' : ''}</div>`).join('');
      const chips = t.involves.map(id => `<a href="#" class="chipc" data-node="${esc(id)}">${esc(names[id] || id.replace(/^[a-z]+:/, ''))}</a>`).join('');
      const active = st === 'open' || st === 'advancing';
      return `<div class="thread ${st} ${opts.selected === t.id ? 'sel' : ''}" data-thread="${esc(t.id)}" style="--tc:${threadColor(idx)}">` +
        `<div class="th-head"><span class="tchip ${st}">${STATUS_LABEL[st]}</span><b>${esc(t.name)}</b></div>` +
        (t.gist && t.gist !== t.name ? `<div class="q">${esc(t.gist)}</div>` : '') +
        `<div class="th-meta"><span>raised r${t.first}</span>` +
        (active ? `<span class="pbar" title="pressure ${t.pressure_eff} (fades when nothing touches it)"><i style="width:${Math.round(100 * Math.min(1, t.pressure_eff))}%"></i></span>` : '') +
        `<span>${per.aimed.length} probe${per.aimed.length === 1 ? '' : 's'} aimed at it</span>` +
        (opts.focus === t.id ? '<span class="focusing">focused</span>' : '') + `</div>` +
        `<div class="chips">${chips}</div>` +
        (trail ? `<div class="trail">${trail}</div>` : '') +
        (opts.show !== false ? `<div class="th-btns"><button data-thread-act="show">${opts.selected === t.id ? 'Hide' : 'Show on graph'}</button>` +
          (opts.actions && active ? `<button data-thread-act="focus">${opts.focus === t.id ? 'Stop focusing' : 'Focus probes here'}</button><button data-thread-act="close">Close</button>` : '') + '</div>' : '') +
        `</div>`;
    }).join('');
  }

  /* What happened to the questions in one round (for the round inspector). */
  function threadEventsHTML(rec) {
    const ev = rec && rec.threads && rec.threads.events;
    if (!ev || !ev.length) return '';
    return `<div class="tevents"><h2>Questions this round</h2>` + ev.map(e =>
      `<div class="tev ${e.event}"><b>${EVENT_LABEL[e.event] || e.event}</b> ${esc(e.name)}` +
      `${e.event === 'advanced' || e.event === 'resolved' ? '<div class="muted small">' + esc((e.gist || '').slice(0, 160)) + '</div>' : ''}</div>`).join('') + `</div>`;
  }

  /* ---- the graph ------------------------------------------------------------------------ */
  class GraphView {
    /* opts: onNodeClick(node), onBackgroundClick() */
    constructor(svgEl, opts = {}) {
      this.svgEl = svgEl;
      this.svg = d3.select(svgEl);
      this.opts = opts;
      this.root = this.svg.append('g');
      this.gHulls = this.root.append('g');
      this.gEdges = this.root.append('g');
      this.gEdgeLabels = this.root.append('g');
      this.gNodes = this.root.append('g');
      this.gLabels = this.root.append('g');
      this.pos = {};            // id -> {x, y}: the layout grows, it is not reshuffled every round
      this.sim = null;
      this.sig = '';            // which nodes/edges the running layout was built for
      this.lastFocus = null;
      this.autoFit = true;      // keep the whole graph in frame until the user pans or zooms
      this.ticks = 0;
      this.zoom = d3.zoom().scaleExtent([0.15, 5]).on('zoom', ev => {
        this.root.attr('transform', ev.transform);
        if (ev.sourceEvent) this.autoFit = false;          // a person moved the camera
      });
      this.svg.call(this.zoom).on('dblclick.zoom', null);
      if (opts.fitButton !== false && svgEl.parentNode) {
        const host = svgEl.parentNode;
        host.querySelectorAll(':scope > .fitbtn').forEach(b => b.remove());
        const btn = document.createElement('button');
        btn.className = 'fitbtn'; btn.textContent = 'Fit'; btn.title = 'Fit the whole graph in view';
        btn.onclick = () => { this.autoFit = true; this._fit(this.lastNodes || []); };
        host.appendChild(btn);
      }
      this.svg.on('click.background', ev => {
        if (ev.target === svgEl && this.opts.onBackgroundClick) this.opts.onBackgroundClick();
      });
    }

    /* Ease the view so that point `p` sits in the middle (or back to the whole graph when null). */
    _centerOn(p, W, H, k = 1.7) {
      const t = p ? d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-p.x, -p.y) : d3.zoomIdentity;
      this.svg.transition().duration(650).call(this.zoom.transform, t);
    }

    /* Frame every node with a margin. */
    _fit(nodes) {
      const W = this.svgEl.clientWidth || 600, H = this.svgEl.clientHeight || 400;
      if (!nodes.length) return;
      const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y), pad = 56;
      const x0 = d3.min(xs), x1 = d3.max(xs), y0 = d3.min(ys), y1 = d3.max(ys);
      const k = Math.max(0.15, Math.min(1.4, (W - 2 * pad) / Math.max(1, x1 - x0), (H - 2 * pad) / Math.max(1, y1 - y0)));
      const t = d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
      this.svg.transition().duration(500).call(this.zoom.transform, t);
    }

    /* nodes: [{id, kind, name, gist, first}], edges: [{id, s, p, o, conf, val, aff, since, until, role}]
     * extra: {threads: table rows (draws halos, hides thread nodes), threadSel: id to highlight,
     *         focus: focused thread id, aimed: thread id the round's probe was aimed at} */
    render(nodes, edges, round, rec, selected, extra = {}) {
      const el = this.svgEl;
      const W = el.clientWidth || 600, H = el.clientHeight || 400;
      this.svg.attr('viewBox', `0 0 ${W} ${H}`);
      const hideThreads = Array.isArray(extra.threads);
      const shown = nodes.filter(n => n.first <= round && !(hideThreads && n.kind === 'thread'));
      const ids = new Set(shown.map(n => n.id));
      const live = edges.filter(e => e.since <= round && (e.until == null || e.until > round) && ids.has(e.s) && ids.has(e.o));
      const nameOf = Object.fromEntries(nodes.map(n => [n.id, n.name]));
      const walk = (rec && rec.probe && rec.probe.walk) || {};
      const win = new Set(walk.nodes || []), seed = walk.seed_node, target = walk.target;
      const winEdge = new Set((walk.edges || []).map(e => pairKey(e.a, e.b)));
      const fresh = new Set(((rec && rec.delta) ? rec.delta.nodes : []).map(n => n.id));

      const simNodes = shown.map(n => {
        const p = this.pos[n.id] || (this.pos[n.id] = {x: W / 2 + (Math.random() - .5) * 80, y: H / 2 + (Math.random() - .5) * 80});
        return Object.assign(p, n);
      });
      const index = Object.fromEntries(simNodes.map(n => [n.id, n]));
      const links = live.map(e => ({source: index[e.s], target: index[e.o], e}));

      // Selection: the node, its neighbours and the links between them stay lit; the rest recedes.
      const sel = selected && index[selected] ? selected : null;
      const incident = sel ? links.filter(l => l.e.s === sel || l.e.o === sel) : [];
      const nbr = new Set(incident.map(l => l.e.s === sel ? l.e.o : l.e.s));
      const isIncident = new Set(incident.map(l => l.e.id));

      // Threads: a soft halo around the concepts each open question rests on; a chosen thread lights its concepts.
      const table = extra.threads || [];
      const hulls = table.filter(t => t.state !== 'resolved' || t.id === extra.threadSel)
        .map(t => ({t, color: threadColor(table.indexOf(t)), ids: t.involves.filter(id => index[id]),
                    strong: t.id === extra.threadSel || t.id === extra.focus || t.id === extra.aimed}))
        .filter(h => h.ids.length);
      const threadSel = extra.threadSel ? hulls.find(h => h.t.id === extra.threadSel) : null;
      const lit = !sel && threadSel ? new Set(threadSel.ids) : null;

      // Only a change in the graph itself restarts the layout; selecting never does.
      const sig = simNodes.map(n => n.id).join(',') + '|' + live.map(e => e.id).join(',');
      const changed = sig !== this.sig;
      if (changed) {
        const first = this.sig === '';
        this.sig = sig;
        if (this.sim) this.sim.stop();
        this.sim = d3.forceSimulation(simNodes)
          .force('link', d3.forceLink(links).distance(l => l.e.role === 'relation' ? 85 : 55).strength(.6))
          .force('charge', d3.forceManyBody().strength(-220).distanceMax(320))
          .force('x', d3.forceX(W / 2).strength(0.07))          // gravity keeps components from drifting apart
          .force('y', d3.forceY(H / 2).strength(0.07))
          .force('collide', d3.forceCollide(16)).alpha(first ? .8 : .35).alphaDecay(.04);
        this.ticks = 0;
      }

      const hull = this.gHulls.selectAll('path').data(hulls, h => h.t.id);
      hull.exit().remove();
      const hullAll = hull.enter().append('path').attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').merge(hull)
        .attr('fill', h => h.color).attr('stroke', h => h.color).attr('stroke-width', 30)
        .attr('fill-opacity', h => h.strong ? .2 : .08).attr('stroke-opacity', h => h.strong ? .2 : .08);
      hullAll.select('title').remove();
      hullAll.append('title').text(h => h.t.name);
      const hl = this.gHulls.selectAll('text').data(hulls.filter(h => h.strong), h => h.t.id);      // only the emphasised halo is named
      hl.exit().remove();
      const hullLabels = hl.enter().append('text').attr('class', 'thl').attr('text-anchor', 'middle').merge(hl)
        .style('fill', h => h.color).text(h => (h.t.name.length > 46 ? h.t.name.slice(0, 45) + '...' : h.t.name));

      const inWin = l => winEdge.has(pairKey(l.e.s, l.e.o));
      const link = this.gEdges.selectAll('line').data(links, l => l.e.id);
      link.exit().remove();
      const linkAll = link.enter().append('line').merge(link)
        .attr('stroke', l => l.e.val <= -0.3 ? css('--hostile') : (isIncident.has(l.e.id) || l.e.role === 'relation' ? css('--node') : css('--scaffold')))
        .attr('stroke-dasharray', l => l.e.role === 'relation' ? null : '3 3')
        .attr('stroke-width', l => isIncident.has(l.e.id) ? 3 : (inWin(l) ? 3.2 : (l.e.role === 'relation' ? 1.4 : 1)))
        .attr('stroke-opacity', l => sel ? (isIncident.has(l.e.id) ? 1 : .07)
          : lit ? (lit.has(l.e.s) && lit.has(l.e.o) ? 1 : .07) : (win.size && !inWin(l) ? .35 : .9));
      linkAll.select('title').remove();
      linkAll.append('title').text(l => `${nameOf[l.e.s]} -${l.e.p}-> ${nameOf[l.e.o]}  conf ${l.e.conf}  affinity ${l.e.aff ?? '?'}`);

      // Labels on the selected node's links: the predicate, with the direction it points.
      const rank = {};
      const labelData = incident.map(l => {
        const k = pairKey(l.e.s, l.e.o), i = rank[k] = (rank[k] ?? -1) + 1;
        return {l, i, out: l.e.s === sel};
      });
      const elab = this.gEdgeLabels.selectAll('text').data(labelData, d => d.l.e.id);
      elab.exit().remove();
      const elabAll = elab.enter().append('text').attr('class', 'el').attr('text-anchor', 'middle').merge(elab)
        .style('fill', d => d.l.e.val <= -0.3 ? css('--hostile') : css('--ink'))
        .text(d => (d.out ? `${d.l.e.p.replace(/_/g, ' ')} →` : `← ${d.l.e.p.replace(/_/g, ' ')}`));

      const onClick = this.opts.onNodeClick;
      const node = this.gNodes.selectAll('circle').data(simNodes, n => n.id);
      node.exit().remove();
      const nodeAll = node.enter().append('circle').attr('r', 0).style('cursor', onClick ? 'pointer' : 'default')
        .merge(node)
        .attr('fill', n => kindColor(n.kind))
        .attr('stroke', n => n.id === sel ? css('--ink') : n.id === target ? css('--hit') : n.id === seed ? css('--q') : fresh.has(n.id) ? css('--new') : css('--panel'))
        .attr('stroke-width', n => (n.id === sel || n.id === target || n.id === seed || fresh.has(n.id)) ? 3 : 1.2)
        .attr('opacity', n => sel ? (n.id === sel || nbr.has(n.id) ? 1 : .15)
          : lit ? (lit.has(n.id) ? 1 : .2) : (win.size && !win.has(n.id) ? .45 : 1))
        .on('click', onClick ? (ev, n) => { ev.stopPropagation(); onClick(n); } : null);
      nodeAll.transition().duration(300).attr('r', n => n.id === sel ? 10 : nbr.has(n.id) ? 7.5 : 6);
      nodeAll.select('title').remove();
      nodeAll.append('title').text(n => `${n.name} [${n.kind}]\n${n.gist || ''}`);

      const labeled = sel ? simNodes.filter(n => n.id === sel || nbr.has(n.id))
        : lit ? simNodes.filter(n => lit.has(n.id))
        : simNodes.filter(n => n.id === seed || n.id === target || win.has(n.id) || fresh.has(n.id) || simNodes.length < 25);
      const lab = this.gLabels.selectAll('text').data(labeled, n => n.id);
      lab.exit().remove();
      const labAll = lab.enter().append('text').attr('class', 'nl').attr('dx', 11).attr('dy', 3).merge(lab)
        .attr('font-weight', n => n.id === sel ? 700 : 400)
        .text(n => n.name.length > 30 ? n.name.slice(0, 29) + '...' : n.name);

      const hullPath = h => {
        const pts = h.ids.map(id => [index[id].x, index[id].y]);
        if (pts.length >= 3) return 'M' + d3.polygonHull(pts).join('L') + 'Z';
        if (pts.length === 2) return `M${pts[0]}L${pts[1]}`;
        return `M${pts[0][0]},${pts[0][1]}l0.01,0`;
      };
      const centroid = h => {
        const pts = h.ids.map(id => index[id]);
        return {x: d3.mean(pts, p => p.x), y: d3.min(pts, p => p.y) - 24};
      };
      const place = () => {
        hullAll.attr('d', hullPath);
        hullLabels.attr('x', h => centroid(h).x).attr('y', h => centroid(h).y);
        linkAll.attr('x1', l => l.source.x).attr('y1', l => l.source.y).attr('x2', l => l.target.x).attr('y2', l => l.target.y);
        elabAll.attr('x', d => (d.l.source.x + d.l.target.x) / 2).attr('y', d => (d.l.source.y + d.l.target.y) / 2 - 4 + d.i * 12);
        nodeAll.attr('cx', n => n.x).attr('cy', n => n.y);
        labAll.attr('x', n => n.x).attr('y', n => n.y);
      };
      // where the camera should rest: a selected node, or the middle of a highlighted thread
      const aim = () => sel ? {p: index[sel], k: 1.7}
        : threadSel ? {p: {x: d3.mean(threadSel.ids, id => index[id].x), y: d3.mean(threadSel.ids, id => index[id].y)}, k: 1.4} : null;
      this.lastNodes = simNodes;
      this.sim.on('tick', () => {
        place();
        if (this.autoFit && !aim() && ++this.ticks % 45 === 0) this._fit(simNodes);
      });
      this.sim.on('end', () => {
        const a = aim();
        if (a) this._centerOn(a.p, W, H, a.k); else if (this.autoFit) this._fit(simNodes);
      });
      place();                                    // paint at once, whether or not the layout is moving

      const focusKey = sel ? 'n:' + sel : threadSel ? 't:' + extra.threadSel : null;
      if (focusKey !== this.lastFocus) {          // a new selection (or none): glide to it
        this.lastFocus = focusKey;
        const a = aim();
        if (a) this._centerOn(a.p, W, H, a.k);
        else { this.autoFit = true; this._fit(simNodes); }
      }
    }
  }

  /* ---- the selected node's card ------------------------------------------------------------ */
  /* node: {id, kind, name, gist, first}; edges: the run's edges; names: id -> name. Links that are
   * open at `round`, relations first. Neighbour names carry data-node so the page can navigate. */
  function nodeCardHTML(node, edges, names, round, opts = {}) {
    const links = edges.filter(e => e.since <= round && (e.until == null || e.until > round) && (e.s === node.id || e.o === node.id))
      .sort((a, b) => (a.role === 'relation' ? 0 : 1) - (b.role === 'relation' ? 0 : 1) || b.conf - a.conf);
    const row = e => {
      const out = e.s === node.id, other = out ? e.o : e.s;
      const hostile = e.val <= -0.3;
      return `<div class="link ${e.role}"><span class="pred ${hostile ? 'hostile' : ''}">${out ? '' : '&larr; '}${esc(e.p.replace(/_/g, ' '))}${out ? ' &rarr;' : ''}</span> ` +
        `<a href="#" data-node="${esc(other)}">${esc(names[other] || other)}</a>` +
        `<span class="muted"> conf ${e.conf}${e.aff != null ? ' &middot; affinity ' + e.aff : ''}${hostile ? ' &middot; hostile' : ''}${e.role === 'scaffold' ? ' &middot; structure' : ''}</span></div>`;
    };
    const rel = links.filter(e => e.role === 'relation').length;
    return `<div class="nodecard"><h2>${esc(node.kind)} &middot; first seen round ${node.first}</h2>` +
      `<div class="nc-name">${esc(node.name)}</div>` +
      (node.gist ? `<div class="q">${esc(node.gist)}</div>` : '') +
      `<div class="kv"><span>links <b>${links.length}</b> (${rel} relations)</span></div>` +
      (links.length ? `<div class="links">${links.map(row).join('')}</div>` : '<div class="muted small">No links yet.</div>') +
      `<div class="nc-btns">${opts.pin ? '<button data-act="pin">Pin as next seed</button>' : ''}<button data-act="clear">Deselect</button></div></div>`;
  }

  /* ---- the round inspector --------------------------------------------------------------- */
  /* Every probe round reads the same way: 1 Choose -> 2 Search -> 3 Read -> 4 What it changed. */
  const ORIGIN = {
    spawn: ['Fresh probe', 'a rarely visited concept was drawn as the seed'],
    mutate: ['Variation', 'a well-scoring earlier probe with one of its concepts swapped'],
    cross: ['Recombination', 'two earlier probes mixed, one concept from each'],
    draw: ['Aimed at an open question', 'the probe starts from concepts that question rests on'],
  };
  const FALLBACK = {
    ineligible_seed: 'the seed is not part of any relation cluster',
    isolated_seed: 'the seed has no relations yet',
    no_eligible_target: 'every other concept in the window was used too recently',
    no_eligible_measurements: 'the walk measured no eligible concept',
    no_seedable_cluster: 'the graph has no cluster of 3 or more related concepts yet',
    one_nonthread_node: 'the graph has only one concept',
    no_nonthread_nodes: 'the graph has no concepts yet',
  };
  const WALK_LABEL = m => /diffusion/.test(m || '') ? 'the diffusion control walk'
    : /atlas/.test(m || '') ? 'the quantum walk (run on Moth Atlas)'
    : /numpy/.test(m || '') ? 'the quantum walk (exact simulation, window too wide for a circuit)' : 'the quantum walk (Qiskit circuit)';
  const tipItem = (label, value, help) => `<span title="${esc(help)}">${label} <b>${value}</b></span>`;
  const step = (n, title, body) => `<section class="step"><h3><i>${n}</i>${title}</h3>${body}</section>`;
  const safeLink = src => {
    const m = /^web:(https?:\/\/.+)$/.exec(src || '');
    if (!m) return esc(src || 'no page');
    let host = m[1]; try { host = new URL(m[1]).hostname.replace(/^www\./, ''); } catch (e) {}
    return `<a href="${esc(m[1])}" target="_blank" rel="noopener noreferrer">${esc(host)}</a>`;
  };

  function roundPanelHTML(rec, nodeName, seeds, opts = {}) {
    if (!rec) {
      const list = (seeds || []).map(s => `<div class="q">${esc(s.source)} - +${s.delta.nodes.length} concepts, +${s.delta.assertions.length} claims</div>`).join('');
      return `<div><h2>Seed</h2>${list || '<div class="muted">no seed record yet</div>'}</div>`;
    }
    const name = id => nodeName[id] || id;
    if (rec.kind === 'idle') return '<div class="muted">nothing to probe yet</div>';
    const d = rec.delta || {nodes: [], assertions: []};
    const nq = ((rec.threads && rec.threads.events) || []).length;
    const summary = `+${d.nodes.length} concept${d.nodes.length === 1 ? '' : 's'}, +${d.assertions.length} claim${d.assertions.length === 1 ? '' : 's'}` +
      (nq ? `, ${nq} question event${nq === 1 ? '' : 's'}` : '');
    const changed = step(4, 'What it changed',
      (d.nodes.length ? `<div class="added">${d.nodes.map(n => `<span>+ ${esc(n.name)}</span>`).join(', ')}</div>` : '<div class="muted small">No new concepts.</div>') +
      `<div class="muted small">${d.assertions.length} new claim${d.assertions.length === 1 ? '' : 's'} between concepts.</div>` + threadEventsHTML(rec));
    if (rec.kind === 'inject' || rec.kind === 'seed') {
      const title = rec.kind === 'seed' ? 'Seed passage' : 'Injected observation';
      return `<h1 class="rp-title">Round ${rec.round} <span>${title}</span></h1>` +
        `<div class="q">${safeLink(rec.source)}</div><div class="muted small">${esc(rec.excerpt || '')}</div>` + changed;
    }

    const w = (rec.probe && rec.probe.walk) || {}, p = rec.probe;
    const comps = (p.components || []).map(c => c.name);
    const [olabel, odesc] = ORIGIN[p.origin] || [p.origin, ''];
    const question = p.thread ? name(p.thread) : null;
    const route = w.target ? `<span class="seed">${esc(name(w.seed_node))}</span> &rarr; <span class="target">${esc(name(w.target))}</span>` : esc(comps.join('  /  '));
    let how;
    if (w.target) how = `The partner concept was chosen by <b>${WALK_LABEL(w.method)}</b>${w.forced ? ' (only one concept was eligible)' : ''}.`;
    else if (w.skipped === 'crossover') how = 'No walk: a recombined probe keeps the concepts of its two parents.';
    else if (w.fallback) how = `No walk this round: ${esc(FALLBACK[w.fallback] || w.fallback)}. A concept was drawn at random instead.`;
    else how = 'No graph walk in this run: concepts are drawn by visit count.';
    let html = `<h1 class="rp-title">Round ${rec.round} <span>${esc(olabel)}</span></h1><div class="rp-sum">${esc(summary)}</div>`;
    html += step(1, 'Choose where to look',
      `<div class="route">${route}</div><div class="why"><b>${esc(olabel)}</b>: ${esc(odesc)}${question ? ` &mdash; <i>${esc(question)}</i>` : ''}.</div><div class="how">${how}</div>`);

    if (w.nodes && w.quantum_probabilities) {
      const q = w.quantum_probabilities, c = w.classical_probabilities || [];
      const top = Math.max(...q, ...c, 1e-9);
      const fromDiffusion = /diffusion/.test(w.method || '');
      const rows = w.nodes.map((id, i) => ({id, i, q: q[i], c: c[i] ?? 0})).sort((a, b) => Math.max(b.q, b.c) - Math.max(a.q, a.c)).slice(0, 12);
      const bars = rows.map(x =>
        `<div class="lbl ${x.id === w.target ? 't' : ''}" title="${esc(name(x.id))}">${x.i === 0 ? '&#9679; ' : ''}${esc(name(x.id))}</div>` +
        `<div class="pair"><div class="bar q" style="width:${100 * x.q / top}%"></div><div class="bar c" style="width:${100 * x.c / top}%"></div></div>` +
        `<div class="nums"><span class="nq">${(100 * x.q).toFixed(0)}%</span><span class="nc">${(100 * x.c).toFixed(0)}%</span></div>`).join('');
      html += `<details class="walkbox" open><summary>The walk: where the walker lands</summary>` +
        `<p class="cap">Each row is a concept the walk considered (one qubit each; ${w.nodes.length} in the window). &#9679; marks the seed, where the walker starts. ` +
        `A bar is the chance of finding the walker on that concept after time t=${w.time}, scaled to the longest bar this round; the numbers are the exact percentages. ` +
        `The partner was drawn from the <b>${fromDiffusion ? 'orange (diffusion)' : 'purple (quantum' + (w.shots ? `, measured over ${w.shots} shots` : '') + ')'}</b> distribution: the green concept.</p>` +
        `<div class="legend"><span><i style="background:var(--q)"></i>quantum walk</span><span><i style="background:var(--c)"></i>diffusion: a classical random walker on the same graph</span></div>` +
        `<div class="bars" style="margin-top:6px">${bars}</div>` +
        `<div class="kv">` +
        tipItem('qubits', w.nodes.length, 'Concepts in the walk window: one qubit each.') +
        tipItem('links', (w.edges || []).length, 'Relations among those concepts: the couplings of the walk.') +
        tipItem('difference', (w.total_variation_distance ?? 0).toFixed(3), 'Total variation distance between the purple and orange distributions: 0 = identical, 1 = completely different. It is what the wave-like interference adds over ordinary hopping.') +
        (w.trotter_error != null ? tipItem('circuit error', w.trotter_error.toFixed(3), 'How far the finite-step circuit is from the ideal continuous quantum walk (Trotter error).') : '') +
        (w.sampling_error != null ? tipItem('sampling error', w.sampling_error.toFixed(3), 'Measured shots versus the ideal circuit output.') : '') +
        (w.two_qubit_gate_count != null ? tipItem('2-qubit gates', w.two_qubit_gate_count, 'Size of the circuit: four gates per link per step.') : '') +
        tipItem('backend', esc(w.method), 'Where the circuit ran.') + (w.atlas_job_id ? tipItem('Atlas job', esc(w.atlas_job_id), 'Job id on Moth Atlas.') : '') +
        (w.atlas_error ? `<span>atlas fallback: ${esc(w.atlas_error)}</span>` : '') +
        (w.qasm_path ? `<span>circuit ${opts.qasmHref ? `<a href="${esc(opts.qasmHref(rec.round))}" download>${esc(w.qasm_path)}</a>` : `<b>${esc(w.qasm_path)}</b>`}</span>` : '') +
        `</div></details>`;
    }

    const status = (rec.harvest || {}).status || '';
    const STATUS = {ok: 'found a new page', 'all-seen': 'every result had already been read', 'snippet-only': 'no page could be fetched; used the search snippets',
                    'fetch-failed': 'results found but none could be fetched', 'all-duplicate': 'every result was a near-duplicate of a page already read', 'no-results': 'the search returned nothing', 'search-error': 'the search failed'};
    const skipped = (rec.harvest || {}).skipped || [];
    html += step(2, 'Search', `<div class="q">${esc(rec.query)}</div><div class="muted small">${esc(STATUS[status] || status)}</div>` +
      (skipped.length ? `<div class="skipped" title="A page is skipped, without spending an LLM call, when its distance (1 - cosine similarity) to a page already read is below the run's threshold.">` +
        `Skipped ${skipped.length} near-duplicate${skipped.length === 1 ? '' : 's'}: ` +
        skipped.map(s => `${safeLink(s.source)} (distance ${s.distance}, like ${esc(String(s.of))})`).join('; ') + '</div>' : ''));
    if (rec.source) {
      const nov = rec.novelty;
      html += step(3, 'Read',
        `<div class="q">${safeLink(rec.source)}</div><div class="muted small">${esc(rec.excerpt || '').slice(0, 220)}</div>` +
        (nov != null ? `<div class="nov" title="distance to the closest of the last 32 pages read: ${rec.distance ?? 'none yet'}"><span class="nbar"><i style="width:${Math.round(100 * nov)}%"></i></span> novelty <b>${nov}</b></div>` +
          `<p class="cap">Novelty is how different this page is from the last 32 pages read: 1 minus its similarity to the closest one (0 = a near-duplicate; the first page read counts 0.5). ` +
          `Novelty times &ldquo;did it change the graph&rdquo; is the probe's score, and high-scoring probes are more likely to be reused as parents.</p>` : ''));
    } else {
      html += step(3, 'Read', '<div class="muted small">No page was read, so this probe scored 0 for novelty.</div>');
    }
    return html + changed;
  }

  /* ---- the questions overview: where every question is, and what has happened to it -------------- */
  function openQuestionsHTML(info, opts = {}) {
    if (!info) return '';
    const active = info.table.filter(t => t.state !== 'resolved').sort((a, b) => b.pressure_eff - a.pressure_eff);
    const t = info.totals;
    const rows = active.slice(0, opts.max || 7).map(q => {
      const st = threadStatus(q, info);
      return `<div class="oq" data-thread="${esc(q.id)}" style="--tc:${threadColor(info.table.indexOf(q))}" title="Show on graph"><span class="tchip ${st}">${STATUS_LABEL[st]}</span> ${esc(q.name)}</div>`;
    }).join('');
    return `<h2>Questions</h2><div class="kv"><span><b>${t.answered}</b> answered</span><span><b>${t.open}</b> open</span><span><b>${t.opened}</b> raised</span></div>` +
      (rows || '<div class="muted small">No open questions right now.</div>') +
      (active.length > (opts.max || 7) ? `<div class="muted small">and ${active.length - (opts.max || 7)} more</div>` : '') +
      `<button class="linkbtn" data-act="questions">All questions and their history &rarr;</button>`;
  }

  /* Every question event so far, newest first: the story of how the run processed its questions. */
  function threadActivityHTML(info, limit = 60) {
    if (!info || !info.events.length) return '<div class="muted small">Nothing has happened to any question yet.</div>';
    return [...info.events].reverse().slice(0, limit).map(e =>
      `<div class="act ${e.event}"><span class="rn">round ${e.round}</span> <b>${EVENT_LABEL[e.event] || e.event}</b> ${esc(e.name)}` +
      `${e.source ? ' <span class="muted">via ' + esc(srcLabel(e.source)) + '</span>' : ''}` +
      `${(e.event === 'advanced' || e.event === 'resolved') && e.gist ? '<div class="muted small">' + esc(e.gist.slice(0, 180)) + '</div>' : ''}</div>`).join('');
  }

  /* ---- a small line chart ---------------------------------------------------------------- */
  /* series: [{name, color (css var), points: [{x, y}]}]  */
  function lineChart(el, series, {title = '', yMax = null, format = v => v, mark = null} = {}) {
    const W = el.clientWidth || 300, H = el.clientHeight || 140, m = {l: 34, r: 10, t: 22, b: 20};
    const svg = d3.select(el).html('').append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').attr('height', '100%');
    const all = series.flatMap(s => s.points);
    svg.append('text').attr('x', m.l).attr('y', 13).attr('class', 'ct').text(title);
    if (!all.length) return;
    const x = d3.scaleLinear().domain(d3.extent(all, p => p.x)).range([m.l, W - m.r]);
    if (x.domain()[0] === x.domain()[1]) x.domain([x.domain()[0] - 1, x.domain()[1] + 1]);
    const y = d3.scaleLinear().domain([0, yMax ?? (d3.max(all, p => p.y) || 1)]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(Math.min(6, all.length)).tickFormat(d3.format('d')).tickSize(3)).attr('class', 'ax');
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickFormat(format).tickSize(-(W - m.l - m.r))).attr('class', 'ax grid');
    series.forEach(s => {
      if (!s.points.length) return;
      svg.append('path').datum(s.points).attr('fill', 'none').attr('stroke', css(s.color)).attr('stroke-width', 2)
        .attr('d', d3.line().x(p => x(p.x)).y(p => y(p.y)));
      svg.selectAll(null).data(s.points).enter().append('circle').attr('r', 2.5).attr('fill', css(s.color))
        .attr('cx', p => x(p.x)).attr('cy', p => y(p.y));
    });
    if (mark != null) svg.append('line').attr('x1', x(mark)).attr('x2', x(mark)).attr('y1', m.t).attr('y2', H - m.b)
      .attr('stroke', css('--muted')).attr('stroke-dasharray', '3 3');
  }

  window.QualkGraph = {GraphView, nodeCardHTML, roundPanelHTML, lineChart, threadInfo, threadStatus, threadCardsHTML,
                       threadEventsHTML, openQuestionsHTML, threadActivityHTML, threadColor, esc, pct, css};
})();
