// Tests for the tree logic in tree.js. Run with:  node --test tree/tree.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const T = require('./tree.js');

/** A small tree: A1 <- A2 <- A3, and B1 needs A1. */
function sample() {
  return T.checkTree({
    title: 'Sample',
    groups: [
      { id: 'A', name: 'First' },
      { id: 'B', name: 'Second' },
    ],
    nodes: [
      { id: 'A1', title: 'One', group: 'A', done: true },
      { id: 'A2', title: 'Two', group: 'A', needs: ['A1'] },
      { id: 'A3', title: 'Three', group: 'A', needs: ['A2'] },
      { id: 'B1', title: 'Other', group: 'B', needs: ['A1'] },
    ],
  });
}

test('checkTree fills in missing fields', () => {
  const node = sample().nodes[1];
  assert.deepEqual(node, {
    id: 'A2', title: 'Two', description: '', group: 'A', size: 'M', done: false, needs: ['A1'],
  });
});

test('parseTree refuses broken files with a reason', () => {
  const group = { id: 'A', name: 'First' };
  const cases = [
    ['{ not json', /not valid JSON/],
    [JSON.stringify([]), /one tree object/],
    [JSON.stringify({ nodes: [] }), /"groups" list/],
    [JSON.stringify({ groups: [group], nodes: [{ id: 'A1', group: 'A' }] }), /no title/],
    [JSON.stringify({ groups: [group], nodes: [{ id: 'A1', title: 'x', group: 'Z' }] }), /does not exist/],
    [JSON.stringify({ groups: [group], nodes: [{ id: 'A1', title: 'x', group: 'A' }, { id: 'A1', title: 'y', group: 'A' }] }), /used twice/],
    [JSON.stringify({ groups: [group], nodes: [{ id: 'A1', title: 'x', group: 'A', needs: ['A9'] }] }), /needs "A9", which does not exist/],
    [JSON.stringify({ groups: [group], nodes: [{ id: 'A1', title: 'x', group: 'A', needs: ['A1'] }] }), /needs itself/],
  ];
  for (const [text, reason] of cases) assert.throws(() => T.parseTree(text), reason);
});

test('findLoop reports the chain, or null when there is none', () => {
  assert.equal(T.findLoop(sample().nodes), null);
  const looped = [
    { id: 'A', needs: ['B'] },
    { id: 'B', needs: ['C'] },
    { id: 'C', needs: ['A'] },
  ];
  assert.deepEqual(T.findLoop(looped), ['A', 'B', 'C', 'A']);
});

test('a file with a loop is refused', () => {
  const text = JSON.stringify({
    groups: [{ id: 'A' }],
    nodes: [
      { id: 'A1', title: 'x', group: 'A', needs: ['A2'] },
      { id: 'A2', title: 'y', group: 'A', needs: ['A1'] },
    ],
  });
  assert.throws(() => T.parseTree(text), /Loop: A1 needs A2 needs A1/);
});

test('statusOf: done, available when every prerequisite is done, else locked', () => {
  const tree = sample();
  const byId = new Map(tree.nodes.map((node) => [node.id, node]));
  assert.equal(T.statusOf(byId.get('A1'), byId), 'done');
  assert.equal(T.statusOf(byId.get('A2'), byId), 'available');
  assert.equal(T.statusOf(byId.get('A3'), byId), 'locked');
  assert.deepEqual(T.countStatuses(tree), { done: 1, available: 2, locked: 1 });
});

test('allNeeds and allUnlocks follow the whole chain', () => {
  const tree = sample();
  const byId = new Map(tree.nodes.map((node) => [node.id, node]));
  assert.deepEqual([...T.allNeeds('A3', byId)].sort(), ['A1', 'A2']);
  assert.deepEqual([...T.allUnlocks('A1', tree.nodes)].sort(), ['A2', 'A3', 'B1']);
  assert.deepEqual([...T.allUnlocks('A3', tree.nodes)], []);
});

test('layoutTree puts nodes in columns by depth and bands by group', () => {
  const layout = T.layoutTree(sample());
  assert.deepEqual(
    ['A1', 'A2', 'A3', 'B1'].map((id) => layout.column.get(id)),
    [0, 1, 2, 1]
  );
  assert.equal(layout.bands.length, 2);
  // Further right for deeper nodes; the second band sits below the first.
  assert.ok(layout.positions.get('A2').x > layout.positions.get('A1').x);
  assert.ok(layout.positions.get('B1').y > layout.positions.get('A1').y);
  assert.ok(layout.bands[1].y >= layout.bands[0].y + layout.bands[0].height);
});

test('nodes sharing a band and a column stack without overlapping', () => {
  let tree = sample();
  tree = T.addNode(tree, 'A'); // A4, no prerequisites -> column 0, same band as A1
  const layout = T.layoutTree(tree);
  const first = layout.positions.get('A1');
  const second = layout.positions.get('A4');
  assert.equal(first.x, second.x);
  assert.ok(second.y - first.y >= T.GRID.cardHeight);
});

test('addNode picks the first free id in the group', () => {
  const tree = T.addNode(sample(), 'A');
  assert.equal(tree.nodes[tree.nodes.length - 1].id, 'A4');
  assert.throws(() => T.addNode(sample(), 'Z'), /Add a branch/);
});

test('updateNode renames an id everywhere it is needed', () => {
  const tree = T.updateNode(sample(), 'A1', { id: 'ROOT', title: 'Root' });
  assert.equal(tree.nodes[0].id, 'ROOT');
  assert.deepEqual(tree.nodes[1].needs, ['ROOT']);
  assert.deepEqual(tree.nodes[3].needs, ['ROOT']);
  assert.throws(() => T.updateNode(sample(), 'A1', { id: 'A2' }), /already used/);
  assert.throws(() => T.updateNode(sample(), 'A1', { title: '  ' }), /needs a title/);
});

test('setPrerequisites refuses a loop and leaves the tree alone', () => {
  const tree = sample();
  assert.throws(
    () => T.setPrerequisites(tree, 'A1', ['A3']),
    /That would make a loop: A1 needs A3 needs A2 needs A1/
  );
  assert.deepEqual(tree.nodes[0].needs, []);
  assert.deepEqual(T.setPrerequisites(tree, 'B1', ['A1', 'A3']).nodes[3].needs, ['A1', 'A3']);
});

test('removeNode also removes it from other nodes\' needs', () => {
  const tree = T.removeNode(sample(), 'A1');
  assert.deepEqual(tree.nodes.map((node) => node.id), ['A2', 'A3', 'B1']);
  assert.deepEqual(tree.nodes[0].needs, []);
  assert.deepEqual(tree.nodes[2].needs, []);
});

test('branches: add, rename, and remove only when empty', () => {
  let tree = T.addGroup(sample(), 'C', 'Third');
  assert.equal(tree.groups[2].name, 'Third');
  tree = T.renameGroup(tree, 'C', 'Renamed');
  assert.equal(tree.groups[2].name, 'Renamed');
  assert.equal(T.removeGroup(tree, 'C').groups.length, 2);
  assert.throws(() => T.removeGroup(tree, 'A'), /Move or delete/);
  assert.throws(() => T.addGroup(tree, 'A', 'Again'), /already used/);
});

test('export then load gives the same tree', () => {
  const tree = sample();
  assert.deepEqual(T.parseTree(T.serializeTree(tree)), tree);
});

test('roomorg.json loads, has no loops, and exports byte for byte', () => {
  const text = fs.readFileSync(path.join(__dirname, 'roomorg.json'), 'utf8');
  const tree = T.parseTree(text);
  assert.equal(T.serializeTree(tree), text);
  assert.ok(tree.nodes.length > 60);
  // Every branch node grows from the trunk's milestone.
  const byId = new Map(tree.nodes.map((node) => [node.id, node]));
  for (const node of tree.nodes.filter((member) => member.group !== 'T')) {
    assert.ok(T.allNeeds(node.id, byId).has('START'), `${node.id} should need START`);
  }
});
