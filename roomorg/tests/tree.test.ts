import { describe, expect, test } from 'vitest';
import { ORGANIZER_PRESETS, UNFINISHED_COLOUR } from '../src/model/presets';
import { Refusal } from '../src/model/refusal';
import { createStarterHome } from '../src/model/roomData';
import {
  addItem,
  addOrganizer,
  canHold,
  childId,
  childrenOf,
  destinations,
  indexHome,
  move,
  pathTo,
  remove,
  updateItem,
  updateOrganizer,
} from '../src/model/tree';
import type { Home } from '../src/model/types';

const RAIL = 'closet-b1-c1-1';
const SHELF = 'closet-b1-c1-3';
const BOX = 'closet-b1-c1-4-box1';

const itemIn = (home: Home, id: string) => home.items.find((item) => item.id === id)!;
const idsIn = (home: Home, parentId: string | null) => childrenOf(home, parentId).map(childId);

describe('adding', () => {
  test('a new item is laid flat, in the unfinished colour, last in its parent', () => {
    const first = addItem(createStarterHome(), SHELF, 'tshirt');
    const second = addItem(first.home, SHELF, 'jeans');
    const item = itemIn(second.home, second.id);
    expect(item).toMatchObject({ fold: 'flat', colour: UNFINISHED_COLOUR, parentId: SHELF, order: 1, type: 'jeans' });
    expect(item.parts.map((part) => part.role)).toEqual(['body', 'leg']);
    expect(idsIn(second.home, SHELF)).toEqual([first.id, second.id]);
  });

  test('an item added straight onto a rail is hung', () => {
    const { home, id } = addItem(createStarterHome(), RAIL, 'coat');
    expect(itemIn(home, id)).toMatchObject({ fold: 'hung', previousFold: 'flat' });
  });

  test('a box goes in a compartment or the tray, never in another box', () => {
    const { home, id } = addOrganizer(createStarterHome(), null, ORGANIZER_PRESETS[0]);
    expect(home.organizers.find((o) => o.id === id)!.parentId).toBeNull();
    expect(() => addOrganizer(home, BOX, ORGANIZER_PRESETS[0])).toThrow(Refusal);
  });
});

describe('moving', () => {
  test('canHold: compartments and the tray hold both, a box holds items only', () => {
    const index = indexHome(createStarterHome());
    expect(canHold(index, SHELF, 'organizer')).toBe(true);
    expect(canHold(index, null, 'organizer')).toBe(true);
    expect(canHold(index, BOX, 'item')).toBe(true);
    expect(canHold(index, BOX, 'organizer')).toBe(false);
    expect(canHold(index, 'no-such-place', 'item')).toBe(false);
  });

  test('onto a rail sets "hung"; off it restores the previous fold', () => {
    const added = addItem(createStarterHome(), SHELF, 'shirt');
    const folded = updateItem(added.home, added.id, { fold: 'thirds' });

    const hung = move(folded, added.id, RAIL);
    expect(itemIn(hung, added.id)).toMatchObject({ parentId: RAIL, fold: 'hung', previousFold: 'thirds' });

    const boxed = move(hung, added.id, BOX);
    expect(itemIn(boxed, added.id)).toMatchObject({ parentId: BOX, fold: 'thirds', previousFold: null });
  });

  test('a fold chosen while hanging is kept for when the item comes off', () => {
    const added = addItem(createStarterHome(), RAIL, 'shirt');
    const chosen = updateItem(added.home, added.id, { fold: 'half' });
    expect(itemIn(chosen, added.id)).toMatchObject({ fold: 'hung', previousFold: 'half' });
    expect(itemIn(move(chosen, added.id, null), added.id).fold).toBe('half');
  });

  test('position puts the thing among its new siblings, and reorders within a parent', () => {
    let home = createStarterHome();
    const [a, b, c] = ['tshirt', 'jeans', 'socks'].map((type) => {
      const added = addItem(home, SHELF, type);
      home = added.home;
      return added.id;
    });
    expect(idsIn(home, SHELF)).toEqual([a, b, c]);
    expect(idsIn(move(home, c, SHELF, 0), SHELF)).toEqual([c, a, b]);
    expect(idsIn(move(home, a, SHELF, 1), SHELF)).toEqual([b, a, c]);

    const tights = 'closet-b1-c1-2-box1';
    const moved = move(home, tights, SHELF, 1);
    expect(idsIn(moved, SHELF)).toEqual([a, tights, b, c]);
    expect(idsIn(moved, 'closet-b1-c1-2')).not.toContain(tights);
  });

  test('a box cannot be moved into a box, and unknown ids are refused', () => {
    const home = createStarterHome();
    expect(() => move(home, 'closet-b1-c1-2-box1', BOX)).toThrow(/cannot go inside another box/);
    expect(() => move(home, 'nothing', SHELF)).toThrow(Refusal);
  });
});

describe('removing', () => {
  test('deleting a box moves its items to the tray', () => {
    const first = addItem(createStarterHome(), BOX, 'tshirt');
    const second = addItem(first.home, BOX, 'jeans');
    const after = remove(second.home, BOX);
    expect(after.organizers.some((o) => o.id === BOX)).toBe(false);
    expect(idsIn(after, null)).toEqual([first.id, second.id]);
    expect(after.items).toHaveLength(2);
  });

  test('deleting an item removes only that item', () => {
    const { home, id } = addItem(createStarterHome(), BOX, 'tshirt');
    expect(remove(home, id).items).toEqual([]);
    expect(() => remove(home, 'nothing')).toThrow(Refusal);
  });
});

describe('editing', () => {
  test('changing the type resets the parts to that type', () => {
    const { home, id } = addItem(createStarterHome(), SHELF, 'tshirt');
    const retyped = updateItem(home, id, { type: 'trousers' });
    expect(itemIn(retyped, id).parts.map((part) => part.role)).toEqual(['body', 'leg']);
  });

  test('bad values are refused', () => {
    const { home, id } = addItem(createStarterHome(), SHELF, 'tshirt');
    expect(() => updateItem(home, id, { name: '  ' })).toThrow(Refusal);
    expect(() => updateItem(home, id, { colour: 'red' })).toThrow(Refusal);
    expect(() => updateItem(home, id, { parts: [{ role: 'body', w: 0, l: 10, t: 1, count: 1 }] })).toThrow(Refusal);
    expect(() => updateOrganizer(home, BOX, { wall: 30 })).toThrow(/no room inside/);
    expect(() => updateOrganizer(home, BOX, { outer: { w: 0, h: 20, d: 50 } })).toThrow(Refusal);
    expect(updateOrganizer(home, BOX, { name: 'Bags' }).organizers.find((o) => o.id === BOX)!.name).toBe('Bags');
  });
});

describe('finding', () => {
  test('pathTo names the room, furniture, compartment, box and item', () => {
    const { home, id } = addItem(createStarterHome(), BOX, 'tshirt');
    expect(pathTo(home, id)).toEqual(['Bedroom', 'Closet B1', 'Hand bags', 'Hand bags', 'New t-shirt']);
    const loose = addItem(home, null, 'socks');
    expect(pathTo(loose.home, loose.id)).toEqual(['Unsorted', 'New socks (pair)']);
  });

  test('destinations lists the tray, every compartment and every box', () => {
    const home = createStarterHome();
    const list = destinations(home);
    const cells = home.furniture.flatMap((f) => f.columns.flatMap((column) => column.cells));
    expect(list[0]).toEqual({ id: null, label: 'Unsorted tray', kind: 'tray' });
    expect(list.filter((d) => d.kind === 'cell')).toHaveLength(cells.length);
    expect(list.filter((d) => d.kind === 'organizer')).toHaveLength(home.organizers.length);
    expect(list.find((d) => d.id === BOX)!.label).toBe('Closet B1 › Hand bags › Hand bags');
  });
});
