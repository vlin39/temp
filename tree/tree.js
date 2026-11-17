/*
 * Skill Tree Planner
 *
 * Turns a JSON file into a tree of cards and arrows, lets you edit it, and
 * exports it as JSON again. Plain JavaScript: no build step, no libraries.
 *
 * The file has two parts:
 *   1. Tree logic - plain functions that never touch the page.
 *                   tree.test.js loads these directly.
 *   2. Page       - drawing, the edit panel, loading and exporting.
 */
(function () {
  'use strict';

  /* ======================================================================
   * 1. Tree logic
   * ====================================================================== */

  const SIZES = ['S', 'M', 'L', 'XL'];

  // The drawing grid, in pixels.
  const GRID = {
    cardWidth: 176,
    cardHeight: 62,
    columnWidth: 216,
    rowHeight: 76,
    bandLabel: 28,
    bandGap: 12,
    margin: 14,
  };

  function isText(value) {
    return typeof value === 'string' && value.trim() !== '';
  }

  function indexById(nodes) {
    return new Map(nodes.map((node) => [node.id, node]));
  }

  function firstDuplicate(values) {
    const seen = new Set();
    for (const value of values) {
      if (seen.has(value)) return value;
      seen.add(value);
    }
    return null;
  }

  /** Reads JSON text and returns a checked tree. Throws an Error with a readable reason. */
  function parseTree(text) {
    let data;
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new Error('This is not valid JSON.');
    }
    return checkTree(data);
  }

  /** Checks a plain object and returns a clean tree with every field filled in. */
  function checkTree(data) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      throw new Error('The file must hold one tree object.');
    }
    if (!Array.isArray(data.groups) || !Array.isArray(data.nodes)) {
      throw new Error('The tree needs a "groups" list and a "nodes" list.');
    }

    const groups = data.groups.map((group, index) => {
      if (!group || !isText(group.id)) throw new Error(`Group ${index + 1} has no id.`);
      return { id: group.id, name: isText(group.name) ? group.name : group.id };
    });
    const repeatedGroup = firstDuplicate(groups.map((group) => group.id));
    if (repeatedGroup) throw new Error(`The group id "${repeatedGroup}" is used twice.`);
    const groupIds = new Set(groups.map((group) => group.id));

    const nodes = data.nodes.map((node, index) => {
      if (!node || !isText(node.id)) throw new Error(`Node ${index + 1} has no id.`);
      if (!isText(node.title)) throw new Error(`Node "${node.id}" has no title.`);
      if (!groupIds.has(node.group)) {
        throw new Error(`Node "${node.id}" is in a group that does not exist: "${node.group}".`);
      }
      if (node.needs !== undefined && !Array.isArray(node.needs)) {
        throw new Error(`Node "${node.id}" has a "needs" that is not a list.`);
      }
      return {
        id: node.id,
        title: node.title,
        description: typeof node.description === 'string' ? node.description : '',
        group: node.group,
        size: SIZES.includes(node.size) ? node.size : 'M',
        done: node.done === true,
        needs: [...new Set(node.needs || [])],
      };
    });
    const repeatedNode = firstDuplicate(nodes.map((node) => node.id));
    if (repeatedNode) throw new Error(`The node id "${repeatedNode}" is used twice.`);

    const nodeIds = new Set(nodes.map((node) => node.id));
    for (const node of nodes) {
      for (const needed of node.needs) {
        if (needed === node.id) throw new Error(`Node "${node.id}" needs itself.`);
        if (!nodeIds.has(needed)) {
          throw new Error(`Node "${node.id}" needs "${needed}", which does not exist.`);
        }
      }
    }

    const loop = findLoop(nodes);
    if (loop) throw new Error(`Loop: ${loop.join(' needs ')}.`);

    return {
      title: isText(data.title) ? data.title : 'Untitled tree',
      version: 1,
      groups,
      nodes,
    };
  }

  /**
   * Looks for a chain of prerequisites that leads back to its own start.
   * Returns the chain, for example ['A', 'B', 'A'] when A needs B and B needs A,
   * or null when there is none. (Depth-first search.)
   */
  function findLoop(nodes) {
    const byId = indexById(nodes);
    const progress = new Map(); // id -> 'in progress' | 'finished'

    function visit(id, path) {
      if (progress.get(id) === 'finished') return null;
      if (progress.get(id) === 'in progress') return path.slice(path.indexOf(id));
      progress.set(id, 'in progress');
      const node = byId.get(id);
      for (const needed of node ? node.needs : []) {
        const loop = visit(needed, path.concat(needed));
        if (loop) return loop;
      }
      progress.set(id, 'finished');
      return null;
    }

    for (const node of nodes) {
      const loop = visit(node.id, [node.id]);
      if (loop) return loop;
    }
    return null;
  }

  /** 'done', 'available' (every prerequisite is done) or 'locked'. */
  function statusOf(node, byId) {
    if (node.done) return 'done';
    return node.needs.every((needed) => byId.get(needed).done) ? 'available' : 'locked';
  }

  function countStatuses(tree) {
    const byId = indexById(tree.nodes);
    const counts = { done: 0, available: 0, locked: 0 };
    for (const node of tree.nodes) counts[statusOf(node, byId)] += 1;
    return counts;
  }

  /** Collects every id reachable from `startId` by repeatedly following `next(id)`. */
  function reach(startId, next) {
    const found = new Set();
    const queue = [startId];
    while (queue.length > 0) {
      for (const id of next(queue.shift())) {
        if (!found.has(id)) {
          found.add(id);
          queue.push(id);
        }
      }
    }
    found.delete(startId);
    return found;
  }

  /** Every prerequisite of a node, and their prerequisites, and so on. */
  function allNeeds(id, byId) {
    return reach(id, (current) => byId.get(current).needs);
  }

  /** Every node that cannot be finished until this one is. */
  function allUnlocks(id, nodes) {
    const dependents = new Map(nodes.map((node) => [node.id, []]));
    for (const node of nodes) {
      for (const needed of node.needs) dependents.get(needed).push(node.id);
    }
    return reach(id, (current) => dependents.get(current));
  }

  /**
   * Decides where every card goes.
   * Column = how many prerequisites deep the node is. Each group gets its own
   * horizontal band, and nodes that share a band and a column stack downward.
   */
  function layoutTree(tree) {
    const byId = indexById(tree.nodes);
    const column = new Map();

    function columnOf(id) {
      if (!column.has(id)) {
        const needs = byId.get(id).needs;
        column.set(id, needs.length === 0 ? 0 : 1 + Math.max(...needs.map(columnOf)));
      }
      return column.get(id);
    }

    const positions = new Map();
    const bands = [];
    let y = GRID.margin;
    let lastColumn = 0;

    for (const group of tree.groups) {
      const rowsUsed = new Map(); // column -> cards already stacked there
      for (const node of tree.nodes.filter((member) => member.group === group.id)) {
        const col = columnOf(node.id);
        const row = rowsUsed.get(col) || 0;
        rowsUsed.set(col, row + 1);
        lastColumn = Math.max(lastColumn, col);
        positions.set(node.id, {
          x: GRID.margin + col * GRID.columnWidth,
          y: y + GRID.bandLabel + row * GRID.rowHeight,
        });
      }
      const height = GRID.bandLabel + Math.max(1, ...rowsUsed.values()) * GRID.rowHeight;
      bands.push({ id: group.id, name: group.name, y, height });
      y += height + GRID.bandGap;
    }

    return {
      positions,
      bands,
      column,
      width: GRID.margin * 2 + lastColumn * GRID.columnWidth + GRID.cardWidth,
      height: y - GRID.bandGap + GRID.margin,
    };
  }

  /* ---- Edits. Each returns a new tree, or throws an Error with the reason. ---- */

  function setTitle(tree, title) {
    if (!isText(title)) throw new Error('The tree needs a title.');
    return { ...tree, title: title.trim() };
  }

  function freshNodeId(tree, groupId) {
    const used = new Set(tree.nodes.map((node) => node.id));
    let number = 1;
    while (used.has(groupId + number)) number += 1;
    return groupId + number;
  }

  /** Adds a node at the end of the list. Its id is the group id plus the first free number. */
  function addNode(tree, groupId) {
    if (!tree.groups.some((group) => group.id === groupId)) {
      throw new Error('Add a branch before adding a node.');
    }
    const node = {
      id: freshNodeId(tree, groupId),
      title: 'New node',
      description: '',
      group: groupId,
      size: 'M',
      done: false,
      needs: [],
    };
    return { ...tree, nodes: tree.nodes.concat(node) };
  }

  /** Changes a node's id, title, description, group, size or done state. */
  function updateNode(tree, id, changes) {
    const newId = changes.id === undefined ? id : String(changes.id).trim();
    if (newId === '') throw new Error('A node needs an id.');
    if (newId !== id && tree.nodes.some((node) => node.id === newId)) {
      throw new Error(`The id "${newId}" is already used.`);
    }
    if (changes.title !== undefined && !isText(changes.title)) {
      throw new Error('A node needs a title.');
    }
    if (changes.group !== undefined && !tree.groups.some((group) => group.id === changes.group)) {
      throw new Error('That branch does not exist.');
    }
    if (changes.size !== undefined && !SIZES.includes(changes.size)) {
      throw new Error('Size must be S, M, L or XL.');
    }

    const nodes = tree.nodes.map((node) => {
      if (node.id === id) {
        return {
          ...node,
          id: newId,
          title: changes.title === undefined ? node.title : changes.title.trim(),
          description: changes.description === undefined ? node.description : changes.description,
          group: changes.group === undefined ? node.group : changes.group,
          size: changes.size === undefined ? node.size : changes.size,
          done: changes.done === undefined ? node.done : changes.done === true,
        };
      }
      if (newId !== id && node.needs.includes(id)) {
        return { ...node, needs: node.needs.map((needed) => (needed === id ? newId : needed)) };
      }
      return node;
    });
    return { ...tree, nodes };
  }

  /** Replaces a node's prerequisites. Refuses anything that would create a loop. */
  function setPrerequisites(tree, id, needs) {
    const ids = new Set(tree.nodes.map((node) => node.id));
    const clean = [...new Set(needs)];
    for (const needed of clean) {
      if (needed === id) throw new Error('A node cannot need itself.');
      if (!ids.has(needed)) throw new Error(`"${needed}" does not exist.`);
    }
    const nodes = tree.nodes.map((node) => (node.id === id ? { ...node, needs: clean } : node));
    const loop = findLoop(nodes);
    if (loop) throw new Error(`That would make a loop: ${loop.join(' needs ')}.`);
    return { ...tree, nodes };
  }

  /** Deletes a node and removes it from every other node's prerequisites. */
  function removeNode(tree, id) {
    const nodes = tree.nodes
      .filter((node) => node.id !== id)
      .map((node) =>
        node.needs.includes(id)
          ? { ...node, needs: node.needs.filter((needed) => needed !== id) }
          : node
      );
    return { ...tree, nodes };
  }

  function addGroup(tree, id, name) {
    const cleanId = String(id || '').trim();
    if (cleanId === '') throw new Error('A branch needs an id.');
    if (tree.groups.some((group) => group.id === cleanId)) {
      throw new Error(`The branch id "${cleanId}" is already used.`);
    }
    const group = { id: cleanId, name: isText(name) ? name.trim() : cleanId };
    return { ...tree, groups: tree.groups.concat(group) };
  }

  function renameGroup(tree, id, name) {
    if (!isText(name)) throw new Error('A branch needs a name.');
    const groups = tree.groups.map((group) =>
      group.id === id ? { ...group, name: name.trim() } : group
    );
    return { ...tree, groups };
  }

  function removeGroup(tree, id) {
    if (tree.nodes.some((node) => node.group === id)) {
      throw new Error('Move or delete the nodes in this branch first.');
    }
    return { ...tree, groups: tree.groups.filter((group) => group.id !== id) };
  }

  /** The tree as JSON text, with fields always written in the same order. */
  function serializeTree(tree) {
    const plain = {
      title: tree.title,
      version: 1,
      groups: tree.groups.map((group) => ({ id: group.id, name: group.name })),
      nodes: tree.nodes.map((node) => ({
        id: node.id,
        title: node.title,
        description: node.description,
        group: node.group,
        size: node.size,
        done: node.done,
        needs: node.needs.slice(),
      })),
    };
    return JSON.stringify(plain, null, 2) + '\n';
  }

  const logic = {
    SIZES,
    GRID,
    parseTree,
    checkTree,
    findLoop,
    statusOf,
    countStatuses,
    allNeeds,
    allUnlocks,
    layoutTree,
    setTitle,
    addNode,
    updateNode,
    setPrerequisites,
    removeNode,
    addGroup,
    renameGroup,
    removeGroup,
    serializeTree,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = logic;
  if (typeof document === 'undefined') return; // loaded by a test, not by a page

  /* ======================================================================
   * 2. Page
   * ====================================================================== */

  const STORAGE_KEY = 'skilltree.tree';
  const STARTER_FILE = 'roomorg.json';
  const STATUS_WORD = { done: 'done', available: 'ready', locked: 'locked' };
  const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

  const EXAMPLE_TREE = {
    title: 'Example tree',
    version: 1,
    groups: [{ id: 'A', name: 'First steps' }],
    nodes: [
      { id: 'A1', title: 'Load a JSON file', description: 'Use "Load JSON" or drop a file on the page.', group: 'A', size: 'S', done: true, needs: [] },
      { id: 'A2', title: 'Edit the tree', description: 'Select a card, then change it in the panel.', group: 'A', size: 'S', done: false, needs: ['A1'] },
      { id: 'A3', title: 'Export it again', description: 'Use "Export JSON" to save your changes.', group: 'A', size: 'S', done: false, needs: ['A2'] },
    ],
  };

  // Everything the page shows is drawn from these three values.
  let tree = checkTree(EXAMPLE_TREE);
  let selectedId = null;
  let starterTree = null; // offered under "Reset to starter" when one is available

  let zoom = 1; // how large the tree is drawn; 1 = full size

  let panelShowsId = null; // which node the panel last showed, to keep its scroll position

  const byElementId = (id) => document.getElementById(id);

  /** Builds an HTML element. Text always goes in as text, never as markup. */
  function el(tag, props, children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'value') node.value = value;
      else if (key === 'checked') node.checked = true;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of [].concat(children || [])) node.append(child);
    return node;
  }

  function svgEl(tag, attributes) {
    const node = document.createElementNS(SVG_NAMESPACE, tag);
    for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, value);
    return node;
  }

  function say(text, kind) {
    const message = byElementId('message');
    message.textContent = text;
    message.className = 'message' + (kind ? ' is-' + kind : '');
  }

  /**
   * Runs one edit. On failure nothing changes and the reason is shown.
   * `selectAfter` (optional) is the node id to select once the edit succeeds,
   * or null to clear the selection.
   */
  function edit(action, okMessage, selectAfter) {
    try {
      tree = action(tree);
    } catch (error) {
      say(error.message, 'error');
      render(); // puts the form back to the stored values
      return false;
    }
    if (selectAfter !== undefined) selectedId = selectAfter;
    remember();
    say(okMessage || '', 'ok');
    render();
    return true;
  }

  /** Makes a button ask for a second click before it does something destructive. */
  function twoStep(button, confirmLabel, action) {
    const label = button.textContent;
    let timer = null;
    const disarm = () => {
      timer = null;
      button.textContent = label;
      button.classList.remove('is-armed');
    };
    button.addEventListener('click', () => {
      if (timer) {
        clearTimeout(timer);
        disarm();
        action();
        return;
      }
      button.textContent = confirmLabel;
      button.classList.add('is-armed');
      timer = setTimeout(disarm, 4000);
    });
    return button;
  }

  /* ---- Drawing the tree ---- */

  function render() {
    if (selectedId && !tree.nodes.some((node) => node.id === selectedId)) selectedId = null;
    renderSummary();
    renderCanvas();
    renderPanel();
    byElementId('json-text').value = serializeTree(tree);
  }

  function renderSummary() {
    const counts = countStatuses(tree);
    byElementId('summary').textContent =
      `${tree.title}: ${tree.nodes.length} nodes, ${counts.done} done, ` +
      `${counts.available} ready, ${counts.locked} locked`;
  }

  function renderCanvas() {
    const canvas = byElementId('canvas');
    const layout = layoutTree(tree);
    const byId = indexById(tree.nodes);
    const needs = selectedId ? allNeeds(selectedId, byId) : new Set();
    const unlocks = selectedId ? allUnlocks(selectedId, tree.nodes) : new Set();
    const inChain = (id) => id === selectedId || needs.has(id) || unlocks.has(id);

    canvas.replaceChildren();
    canvas.style.width = layout.width + 'px';
    canvas.style.height = layout.height + 'px';
    canvas.style.zoom = zoom;

    layout.bands.forEach((band, index) => {
      const strip = el('div', { class: 'band' + (index % 2 ? ' band-alt' : '') }, [
        el('span', { class: 'band-name', text: band.name }),
      ]);
      strip.style.top = band.y + 'px';
      strip.style.height = band.height + 'px';
      canvas.append(strip);
    });

    const arrows = svgEl('svg', {
      class: 'arrows',
      width: layout.width,
      height: layout.height,
      'aria-hidden': 'true',
    });
    arrows.append(arrowHeads());
    for (const node of tree.nodes) {
      for (const needed of node.needs) {
        let kind = 'normal';
        if (selectedId) kind = inChain(needed) && inChain(node.id) ? 'strong' : 'dim';
        arrows.append(arrow(layout.positions.get(needed), layout.positions.get(node.id), kind));
      }
    }
    canvas.append(arrows);

    for (const node of tree.nodes) {
      let role = '';
      if (node.id === selectedId) role = 'is-selected';
      else if (needs.has(node.id)) role = 'is-need';
      else if (unlocks.has(node.id)) role = 'is-unlock';
      else if (selectedId) role = 'is-dim';
      canvas.append(card(node, layout.positions.get(node.id), statusOf(node, byId), role));
    }
  }

  /** One arrowhead shape per arrow style, so each can take its own colour. */
  function arrowHeads() {
    const defs = svgEl('defs');
    for (const kind of ['normal', 'strong', 'dim']) {
      const marker = svgEl('marker', {
        id: 'head-' + kind,
        viewBox: '0 0 10 10',
        refX: '9',
        refY: '5',
        markerUnits: 'userSpaceOnUse', // same size whatever the line thickness
        markerWidth: '9',
        markerHeight: '9',
        orient: 'auto-start-reverse',
      });
      marker.append(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', class: 'arrow-head ' + kind }));
      defs.append(marker);
    }
    return defs;
  }

  /** A curve from the right edge of one card to the left edge of another. */
  function arrow(from, to, kind) {
    const x1 = from.x + GRID.cardWidth;
    const y1 = from.y + GRID.cardHeight / 2;
    const x2 = to.x - 2;
    const y2 = to.y + GRID.cardHeight / 2;
    const bend = Math.max(24, (x2 - x1) / 2);
    return svgEl('path', {
      d: `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`,
      class: 'arrow ' + kind,
      'marker-end': `url(#head-${kind})`,
    });
  }

  function card(node, position, status, role) {
    const button = el(
      'button',
      {
        type: 'button',
        class: `card is-${status} ${role}`,
        title: node.description || node.title,
        'aria-pressed': String(node.id === selectedId),
        'data-id': node.id,
        onclick: (event) => {
          event.stopPropagation();
          select(node.id);
        },
      },
      [
        el('span', { class: 'card-top' }, [
          el('span', { class: 'card-id', text: node.id }),
          el('span', { class: 'card-size', text: node.size }),
          el('span', { class: 'card-status', text: STATUS_WORD[status] }),
        ]),
        el('span', { class: 'card-title', text: node.title }),
      ]
    );
    button.style.left = position.x + 'px';
    button.style.top = position.y + 'px';
    return button;
  }

  /** Changes the zoom. "Fit" makes the whole width of the tree visible at once. */
  function setZoom(value) {
    zoom = Math.min(1.5, Math.max(0.3, value));
    renderCanvas();
  }

  function zoomToFit() {
    const available = byElementId('canvas-wrap').clientWidth - 2;
    setZoom(Math.min(1, available / layoutTree(tree).width));
  }

  function findCard(id) {
    return [...byElementId('canvas').querySelectorAll('.card')].find(
      (button) => button.dataset.id === id
    );
  }

  function select(id) {
    selectedId = id;
    render();
    const chosen = id ? findCard(id) : null;
    if (chosen) {
      chosen.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      chosen.focus({ preventScroll: true });
    }
  }

  /* ---- The edit panel ---- */

  function renderPanel() {
    const panel = byElementId('panel');
    const node = selectedId ? indexById(tree.nodes).get(selectedId) : null;
    const oldList = panel.querySelector('.needs-list');
    const keepScroll = oldList && panelShowsId === selectedId ? oldList.scrollTop : 0;

    panel.replaceChildren(...(node ? nodeForm(node) : treeForm()));
    panelShowsId = selectedId;

    const newList = panel.querySelector('.needs-list');
    if (newList) newList.scrollTop = keepScroll;
  }

  function field(labelText, control) {
    return el('label', { class: 'field' }, [el('span', { class: 'field-label', text: labelText }), control]);
  }

  /** The panel when nothing is selected: the tree's title and its branches. */
  function treeForm() {
    const newId = el('input', { id: 'new-branch-id', type: 'text', placeholder: 'Id', 'aria-label': 'New branch id' });
    const newName = el('input', { id: 'new-branch-name', type: 'text', placeholder: 'Name', 'aria-label': 'New branch name' });

    const branchRows = tree.groups.map((group) => {
      const used = tree.nodes.some((node) => node.group === group.id);
      return el('div', { class: 'branch-row' }, [
        el('code', { class: 'branch-id', text: group.id }),
        el('input', {
          id: 'branch-name-' + group.id,
          type: 'text',
          value: group.name,
          'aria-label': `Name of branch ${group.id}`,
          onchange: (event) => edit((current) => renameGroup(current, group.id, event.target.value)),
        }),
        el('button', {
          type: 'button',
          class: 'button button-small',
          text: 'Remove',
          disabled: used,
          title: used ? 'Move or delete its nodes first' : 'Remove this empty branch',
          onclick: () => edit((current) => removeGroup(current, group.id), `Removed branch ${group.id}.`),
        }),
      ]);
    });

    return [
      el('h2', { text: 'Tree' }),
      field(
        'Title',
        el('input', {
          id: 'tree-title',
          type: 'text',
          value: tree.title,
          onchange: (event) => edit((current) => setTitle(current, event.target.value)),
        })
      ),
      el('h3', { text: 'Branches' }),
      el('div', { class: 'branch-list' }, branchRows),
      el('div', { class: 'branch-row' }, [
        newId,
        newName,
        el('button', {
          type: 'button',
          class: 'button button-small',
          text: 'Add',
          onclick: () => edit((current) => addGroup(current, newId.value, newName.value), 'Added a branch.'),
        }),
      ]),
      el('h3', { text: 'Reading the tree' }),
      el('ul', { class: 'legend' }, [
        el('li', {}, [el('span', { class: 'swatch is-done' }), 'done']),
        el('li', {}, [el('span', { class: 'swatch is-available' }), 'ready: everything it needs is done']),
        el('li', {}, [el('span', { class: 'swatch is-locked' }), 'locked: something it needs is not done']),
      ]),
      el('p', {
        class: 'hint',
        text: 'Arrows run from what a node needs to the node. Select a card to edit it and to trace what it needs and unlocks.',
      }),
    ];
  }

  /** The panel for one node. */
  function nodeForm(node) {
    const byId = indexById(tree.nodes);
    const change = (changes, okMessage) => edit((current) => updateNode(current, node.id, changes), okMessage);

    const idInput = el('input', {
      id: 'node-id',
      type: 'text',
      value: node.id,
      onchange: (event) => {
        const newId = event.target.value.trim();
        edit((current) => updateNode(current, node.id, { id: newId }), '', newId);
      },
    });

    const groupSelect = el(
      'select',
      { id: 'node-group', onchange: (event) => change({ group: event.target.value }) },
      tree.groups.map((group) =>
        el('option', { value: group.id, text: `${group.id}: ${group.name}`, selected: group.id === node.group })
      )
    );

    const sizeSelect = el(
      'select',
      { id: 'node-size', onchange: (event) => change({ size: event.target.value }) },
      SIZES.map((size) => el('option', { value: size, text: size, selected: size === node.size }))
    );

    const deleteButton = twoStep(
      el('button', { type: 'button', class: 'button button-danger', text: 'Delete node' }),
      'Click again to delete',
      () => edit((current) => removeNode(current, node.id), `Deleted ${node.id}.`, null)
    );

    return [
      el('div', { class: 'panel-head' }, [
        el('h2', { text: node.id }),
        el('button', { type: 'button', class: 'button button-small', text: 'Close', onclick: () => select(null) }),
      ]),
      field('Id', idInput),
      field(
        'Title',
        el('input', {
          id: 'node-title',
          type: 'text',
          value: node.title,
          onchange: (event) => change({ title: event.target.value }),
        })
      ),
      field(
        'Description',
        el('textarea', {
          id: 'node-description',
          rows: '4',
          value: node.description,
          onchange: (event) => change({ description: event.target.value }),
        })
      ),
      el('div', { class: 'field-pair' }, [field('Branch', groupSelect), field('Size', sizeSelect)]),
      el('label', { class: 'check' }, [
        el('input', {
          id: 'node-done',
          type: 'checkbox',
          checked: node.done,
          onchange: (event) => change({ done: event.target.checked }),
        }),
        'Done',
      ]),
      el('h3', { text: 'Needs' }),
      needsChecklist(node),
      el('h3', { text: 'Needs, in total' }),
      chipList([...allNeeds(node.id, byId)], 'Nothing. This node can be started any time.'),
      el('h3', { text: 'Unlocks' }),
      chipList([...allUnlocks(node.id, tree.nodes)], 'Nothing depends on this node yet.'),
      deleteButton,
    ];
  }

  /** A tick-list of every other node, grouped by branch. */
  function needsChecklist(node) {
    const list = el('div', { class: 'needs-list' });
    for (const group of tree.groups) {
      const others = tree.nodes.filter((other) => other.group === group.id && other.id !== node.id);
      if (others.length === 0) continue;
      list.append(el('div', { class: 'needs-group', text: group.name }));
      for (const other of others) {
        list.append(
          el('label', { class: 'check' }, [
            el('input', {
              id: 'need-' + other.id,
              type: 'checkbox',
              checked: node.needs.includes(other.id),
              onchange: () => {
                const ticked = [...list.querySelectorAll('input:checked')].map((box) => box.dataset.need);
                edit((current) => setPrerequisites(current, node.id, ticked));
              },
              'data-need': other.id,
            }),
            el('code', { text: other.id }),
            ' ' + other.title,
          ])
        );
      }
    }
    return list;
  }

  /** A row of small buttons, one per node id, in the tree's own order. */
  function chipList(ids, emptyText) {
    if (ids.length === 0) return el('p', { class: 'hint', text: emptyText });
    const wanted = new Set(ids);
    const chips = tree.nodes
      .filter((node) => wanted.has(node.id))
      .map((node) =>
        el('button', {
          type: 'button',
          class: 'chip',
          text: node.id,
          title: node.title,
          onclick: () => select(node.id),
        })
      );
    return el('div', { class: 'chips' }, chips);
  }

  /* ---- Loading, exporting, remembering ---- */

  function loadText(text, sourceName) {
    let loaded;
    try {
      loaded = parseTree(text);
    } catch (error) {
      say(`Could not load ${sourceName}. ${error.message} The current tree is unchanged.`, 'error');
      return;
    }
    replaceTree(loaded, `Loaded "${loaded.title}" (${loaded.nodes.length} nodes).`);
  }

  function replaceTree(newTree, okMessage) {
    tree = newTree;
    selectedId = null;
    remember();
    say(okMessage, 'ok');
    render();
  }

  async function loadFile(file) {
    if (!file) return;
    loadText(await file.text(), file.name);
  }

  function exportFileName() {
    const base = tree.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return (base || 'tree') + '.json';
  }

  /**
   * Hands a text file to the person. In an ordinary browser tab this is a
   * normal download. Some hosts run the page in a sandbox where pages cannot
   * download by themselves and offer a save function instead; use it if present.
   */
  async function saveFile(filename, text) {
    const host = window.claude && typeof window.claude.use === 'function'
      ? await window.claude.use('downloads')
      : null;
    if (host) {
      await host.save({ filename, data: text });
      return;
    }
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const link = el('a', { href: url, download: filename });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function exportTree() {
    const filename = exportFileName();
    try {
      await saveFile(filename, serializeTree(tree));
      say(`Exported ${filename}.`, 'ok');
    } catch (error) {
      const cancelled = error && error.code === 'declined';
      say(
        cancelled
          ? 'Export cancelled.'
          : 'The file could not be saved here. Copy the text under "JSON text" instead.',
        cancelled ? '' : 'error'
      );
    }
  }

  async function copyJson() {
    const box = byElementId('json-text');
    try {
      await navigator.clipboard.writeText(box.value);
      say('Copied the JSON text.', 'ok');
    } catch (error) {
      box.focus();
      box.select();
      say('Copying is blocked here. The text is selected: press Ctrl+C or Cmd+C.', 'error');
    }
  }

  /** Keeps the tree in this browser so a reload does not lose your work. */
  function remember() {
    try {
      localStorage.setItem(STORAGE_KEY, serializeTree(tree));
    } catch (error) {
      // Storage can be unavailable (private windows, blocked site data). Export still works.
    }
  }

  function recall() {
    try {
      const text = localStorage.getItem(STORAGE_KEY);
      return text ? parseTree(text) : null;
    } catch (error) {
      return null;
    }
  }

  /** The tree to start from, if the page was given one or can fetch the one beside it. */
  async function findStarterTree() {
    try {
      if (window.DEFAULT_TREE) return checkTree(window.DEFAULT_TREE);
      if (!location.protocol.startsWith('http')) return null; // a double-clicked file cannot fetch
      const response = await fetch(STARTER_FILE);
      return response.ok ? parseTree(await response.text()) : null;
    } catch (error) {
      return null;
    }
  }

  function wireControls() {
    byElementId('load-input').addEventListener('change', (event) => {
      loadFile(event.target.files[0]);
      event.target.value = ''; // so choosing the same file again still loads it
    });
    byElementId('export-button').addEventListener('click', exportTree);
    byElementId('copy-button').addEventListener('click', copyJson);
    byElementId('apply-text-button').addEventListener('click', () => {
      loadText(byElementId('json-text').value, 'the text');
    });
    byElementId('add-node-button').addEventListener('click', () => {
      const selected = selectedId ? indexById(tree.nodes).get(selectedId) : null;
      const groupId = selected ? selected.group : (tree.groups[0] || {}).id;
      if (edit((current) => addNode(current, groupId), 'Added a node.')) {
        select(tree.nodes[tree.nodes.length - 1].id); // the new node is always last
      }
    });
    twoStep(byElementId('new-tree-button'), 'Click again to clear', () => {
      replaceTree(
        checkTree({ title: 'New tree', groups: [{ id: 'A', name: 'Main' }], nodes: [] }),
        'Started an empty tree.'
      );
    });
    twoStep(byElementId('starter-button'), 'Click again to reset', () => {
      replaceTree(starterTree, `Reset to "${starterTree.title}".`);
    });

    byElementId('zoom-out-button').addEventListener('click', () => setZoom(zoom / 1.25));
    byElementId('zoom-in-button').addEventListener('click', () => setZoom(zoom * 1.25));
    byElementId('zoom-fit-button').addEventListener('click', zoomToFit);

    byElementId('canvas-wrap').addEventListener('click', () => {
      if (selectedId) select(null);
    });
    document.addEventListener('keydown', (event) => {
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if (event.key === 'Escape' && selectedId && !typing) select(null);
    });

    // Dropping a JSON file anywhere on the page loads it.
    window.addEventListener('dragover', (event) => event.preventDefault());
    window.addEventListener('drop', (event) => {
      event.preventDefault();
      loadFile(event.dataTransfer.files[0]);
    });
  }

  async function start() {
    wireControls();
    const remembered = recall();
    if (remembered) tree = remembered;
    render(); // show something straight away

    starterTree = await findStarterTree();
    byElementId('starter-button').hidden = !starterTree;
    if (starterTree && !remembered) {
      tree = starterTree;
      render();
    }
  }

  start();
})();
