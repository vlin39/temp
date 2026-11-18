import { describe, expect, test } from 'vitest';
import { parseHome } from '../src/model/parse';
import { createStarterHome } from '../src/model/roomData';
import { move, toggleOpen } from '../src/model/tree';
import type { Home } from '../src/model/types';
import { addItemIn, addOrganizerIn } from './helpers';

/** A home with something in every kind of place. */
function busyHome(): Home {
  let { home, id } = addItemIn(createStarterHome(), 'closet-b1-c1-4-box1', 'tshirt');
  ({ home } = addItemIn(home, 'closet-b1-c1-1', 'coat'));
  ({ home, id } = addItemIn(home, null, 'socks'));
  home = move(home, id, 'closet-b1-c1-3');
  ({ home, id } = addOrganizerIn(home, 'closet-b1-c1-4-box1', 'pouch'));
  ({ home } = addItemIn(home, id, 'scarf'));
  ({ home } = addOrganizerIn(home, null, 'backpack'));
  return toggleOpen(home, 'sink-unit-door1');
}

const problemsOf = (change: (home: any) => void): string[] => {
  const home = JSON.parse(JSON.stringify(busyHome()));
  change(home);
  const result = parseHome(JSON.stringify(home));
  return result.ok ? [] : result.problems;
};

describe('saved homes', () => {
  test('saving and reading back gives an identical home', () => {
    const home = busyHome();
    const result = parseHome(JSON.stringify(home));
    expect(result).toEqual({ ok: true, home });
  });

  test('the starter home passes its own checks', () => {
    expect(parseHome(JSON.stringify(createStarterHome())).ok).toBe(true);
  });
});

describe('broken files are refused with a reason', () => {
  test('not JSON, not a home, wrong version', () => {
    expect(parseHome('{ nope')).toEqual({ ok: false, problems: ['The file is not valid JSON.'] });
    expect(parseHome('[1, 2]').ok).toBe(false);
    expect(problemsOf((home) => { home.version = 3; })[0]).toMatch(/version/);
    expect(problemsOf((home) => { delete home.items; })[0]).toMatch(/missing its list of items/);
  });

  test('a field of the wrong kind', () => {
    expect(problemsOf((home) => { home.items[0].colour = 'red'; })).toEqual(['Item 1 has a missing or wrong "colour".']);
    expect(problemsOf((home) => { home.items[0].fold = 'crumpled'; })[0]).toMatch(/"fold"/);
    expect(problemsOf((home) => { home.items[0].parts = []; })[0]).toMatch(/"parts"/);
    expect(problemsOf((home) => { home.organizers[0].outer.w = -1; })[0]).toMatch(/Box or bag 1.*"outer"/);
    expect(problemsOf((home) => { home.furniture[0].facing = 'up'; })[0]).toMatch(/"facing"/);
    expect(problemsOf((home) => { home.furniture[0].columns[0].cells[0].height = 'tall'; })[0]).toMatch(/compartment.*"height"/);
    expect(problemsOf((home) => { home.organizers[0].kind = 'crate'; })[0]).toMatch(/"kind"/);
    expect(problemsOf((home) => { home.furniture[0].doors = [{ id: 'd', name: 'Door' }]; })[0]).toMatch(/door.*"cellIds"/);
    expect(problemsOf((home) => { home.open = 'yes'; })[0]).toMatch(/open doors/);
  });

  test('records that do not fit together', () => {
    expect(problemsOf((home) => { home.items[0].parentId = 'nowhere'; })[0]).toMatch(/place that does not exist/);
    expect(problemsOf((home) => { home.organizers[0].parentId = 'nowhere'; })[0]).toMatch(/place that does not exist/);
    expect(problemsOf((home) => { home.organizers[0].parentId = home.organizers[1].id; })[0]).toMatch(/box cannot go inside another box/);
    expect(problemsOf((home) => { home.items[0].parentId = 'sink-unit-c2-2'; })[0]).toMatch(/Nothing can be stored in Sink/);
    expect(problemsOf((home) => { home.furniture[0].doors[0].cellIds = ['closet-b2-c1-1']; })[0]).toMatch(/door in front of a compartment it does not have/);
    expect(problemsOf((home) => { home.furniture[0].roomId = 'attic'; })[0]).toMatch(/room that does not exist/);
    expect(problemsOf((home) => { home.items[1].id = home.items[0].id; })[0]).toMatch(/same id/);
    expect(problemsOf((home) => { home.furniture[0].columns[0].cells[0].height += 5; })[0]).toMatch(/do not add up/);
  });
});

describe('a home saved by the first version', () => {
  /** The starter home as version 1 saved it, with the old sink unit and a few things added. */
  function version1(): any {
    const home = JSON.parse(JSON.stringify(createStarterHome()));
    home.version = 1;
    delete home.open;
    for (const organizer of home.organizers) delete organizer.kind;
    for (const piece of home.furniture) delete piece.doors;

    const cell = (id: string, name: string, kind: string, height: number) => ({ id, name, kind, height, railDrop: 0, drawerWall: kind === 'drawer' ? 1.2 : 0 });
    const side = (n: number) => ({
      id: `sink-unit-c${n}`,
      width: 35,
      cells: [cell(`sink-unit-c${n}-1`, 'Lower', 'cabinet', 35.8), cell(`sink-unit-c${n}-2`, 'Upper', 'cabinet', 35.8)],
    });
    const sink = home.furniture.find((piece: any) => piece.id === 'sink-unit');
    sink.columns = [
      side(1),
      { id: 'sink-unit-c2', width: 72.8, cells: [cell('sink-unit-c2-1', 'Under the sink', 'cabinet', 59.6), cell('sink-unit-c2-2', 'Drawer', 'drawer', 12)] },
      side(3),
    ];
    const wall = home.furniture.find((piece: any) => piece.id === 'wall-cabinet');
    for (const c of wall.columns[0].cells) c.kind = 'cabinet';

    const item = (id: string, parentId: string) => ({
      id, kind: 'clothing', name: id, type: 'socks', colour: '#336699', parentId, order: 0,
      parts: [{ role: 'body', w: 10, l: 24, t: 1.6, count: 1 }], fold: 'flat', previousFold: null,
    });
    home.items = [item('in-old-drawer', 'sink-unit-c2-2'), item('in-wall-cabinet', 'wall-cabinet-c1-2'), item('in-box', 'closet-b1-c1-4-box1')];
    return home;
  }

  test('is brought up to date: boxes, doors, the corrected sink unit', () => {
    const result = parseHome(JSON.stringify(version1()));
    if (!result.ok) throw new Error(result.problems.join(' '));
    const { home } = result;
    const starter = createStarterHome();

    expect(home.version).toBe(2);
    expect(home.organizers.every((o) => o.kind === 'box')).toBe(true);
    expect(home.furniture.find((f) => f.id === 'sink-unit')).toEqual(starter.furniture.find((f) => f.id === 'sink-unit'));
    expect(home.furniture.map((f) => f.doors)).toEqual(starter.furniture.map((f) => f.doors));
    expect(home.furniture.find((f) => f.id === 'wall-cabinet')!.columns[0].cells.map((c) => c.kind)).toEqual(['shelf', 'shelf', 'shelf']);
    expect(home.open).toEqual(starter.open);
  });

  test('what was in the old drawer, now the sink, goes to the floor; the rest stays', () => {
    const result = parseHome(JSON.stringify(version1()));
    if (!result.ok) throw new Error(result.problems.join(' '));
    const where = (id: string) => result.home.items.find((item) => item.id === id)!.parentId;
    expect(where('in-old-drawer')).toBeNull();
    expect(where('in-wall-cabinet')).toBe('wall-cabinet-c1-2');
    expect(where('in-box')).toBe('closet-b1-c1-4-box1');
  });

  test('a damaged version 1 file is still refused with reasons', () => {
    const broken = version1();
    broken.items[0].colour = 'red';
    broken.furniture[0] = 7;
    const result = parseHome(JSON.stringify(broken));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.problems.join(' ')).toMatch(/Furniture 1 is not a record.*Item 1.*"colour"/);
  });
});
