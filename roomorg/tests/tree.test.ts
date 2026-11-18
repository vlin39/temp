import { describe, expect, test } from 'vitest';
import { isReachable, openingsOf } from '../src/model/furniture';
import { nearestHolder } from '../src/model/place';
import { UNFINISHED_COLOUR, organizerPreset } from '../src/model/presets';
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
  reveal,
  setAllOpen,
  toggleOpen,
  updateItem,
  updateOrganizer,
} from '../src/model/tree';
import type { Home } from '../src/model/types';
import { addItemIn, addOrganizerIn } from './helpers';

const RAIL = 'closet-b1-c1-1';
const SHELF = 'closet-b1-c1-3';
const BOX = 'closet-b1-c1-4-box1';

const itemIn = (home: Home, id: string) => home.items.find((item) => item.id === id)!;
const idsIn = (home: Home, parentId: string | null) => childrenOf(home, parentId).map(childId);

describe('adding', () => {
  test('a new item is laid flat, in the unfinished colour, last in its parent', () => {
    const first = addItemIn(createStarterHome(), SHELF, 'tshirt');
    const second = addItemIn(first.home, SHELF, 'jeans');
    const item = itemIn(second.home, second.id);
    expect(item).toMatchObject({ fold: 'flat', colour: UNFINISHED_COLOUR, parentId: SHELF, order: 1, type: 'jeans' });
    expect(item.parts.map((part) => part.role)).toEqual(['body', 'leg']);
    expect(idsIn(second.home, SHELF)).toEqual([first.id, second.id]);
  });

  test('an item added straight onto a rail is hung', () => {
    const { home, id } = addItemIn(createStarterHome(), RAIL, 'coat');
    expect(itemIn(home, id)).toMatchObject({ fold: 'hung', previousFold: 'flat' });
  });

  test('new things start on the floor', () => {
    const box = addOrganizer(createStarterHome(), organizerPreset('small'));
    expect(box.home.organizers.find((o) => o.id === box.id)).toMatchObject({ kind: 'box', parentId: null, order: 0 });
    const bag = addOrganizer(box.home, organizerPreset('tote'));
    expect(bag.home.organizers.find((o) => o.id === bag.id)).toMatchObject({ kind: 'bag', name: 'Tote bag', order: 1 });
    const item = addItem(bag.home, 'socks');
    expect(itemIn(item.home, item.id)).toMatchObject({ parentId: null, order: 2 });
  });
});

describe('moving', () => {
  test('canHold: what may go where', () => {
    let { home, id: bag } = addOrganizerIn(createStarterHome(), null, 'pouch');
    const second = addOrganizerIn(home, null, 'tote');
    const added = addItemIn(second.home, null, 'socks');
    const index = indexHome(added.home);
    const otherBox = 'closet-b1-c1-2-box1';

    // Compartments and the floor hold anything.
    for (const thing of [otherBox, bag, added.id]) {
      expect(canHold(index, SHELF, thing)).toBe(true);
      expect(canHold(index, null, thing)).toBe(true);
    }
    // A box holds bags and clothing, not another box.
    expect(canHold(index, BOX, bag)).toBe(true);
    expect(canHold(index, BOX, added.id)).toBe(true);
    expect(canHold(index, BOX, otherBox)).toBe(false);
    // A bag holds clothing only.
    expect(canHold(index, bag, added.id)).toBe(true);
    expect(canHold(index, bag, second.id)).toBe(false);
    expect(canHold(index, bag, otherBox)).toBe(false);
    // Nothing goes in the sink, inside itself, or somewhere that does not exist.
    expect(canHold(index, 'sink-unit-c2-2', added.id)).toBe(false);
    expect(canHold(index, bag, bag)).toBe(false);
    expect(canHold(index, 'no-such-place', added.id)).toBe(false);
  });

  test('clothing goes in a bag, and the bag goes in a box', () => {
    const bag = addOrganizerIn(createStarterHome(), null, 'pouch');
    const socks = addItemIn(bag.home, bag.id, 'socks');
    const home = move(socks.home, bag.id, BOX);
    expect(idsIn(home, BOX)).toEqual([bag.id]);
    expect(idsIn(home, bag.id)).toEqual([socks.id]);
    expect(pathTo(home, socks.id)).toEqual(['Bedroom', 'Closet B1', 'Hand bags', 'Hand bags', 'Pouch', 'New socks (pair)']);
    expect(() => move(home, BOX, bag.id)).toThrow(/only hold clothing/);
  });

  test('onto a rail sets "hung"; off it restores the previous fold', () => {
    const added = addItemIn(createStarterHome(), SHELF, 'shirt');
    const folded = updateItem(added.home, added.id, { fold: 'thirds' });

    const hung = move(folded, added.id, RAIL);
    expect(itemIn(hung, added.id)).toMatchObject({ parentId: RAIL, fold: 'hung', previousFold: 'thirds' });

    const boxed = move(hung, added.id, BOX);
    expect(itemIn(boxed, added.id)).toMatchObject({ parentId: BOX, fold: 'thirds', previousFold: null });
  });

  test('a fold chosen while hanging is kept for when the item comes off', () => {
    const added = addItemIn(createStarterHome(), RAIL, 'shirt');
    const chosen = updateItem(added.home, added.id, { fold: 'half' });
    expect(itemIn(chosen, added.id)).toMatchObject({ fold: 'hung', previousFold: 'half' });
    expect(itemIn(move(chosen, added.id, null), added.id).fold).toBe('half');
  });

  test('position puts the thing among its new siblings, and reorders within a parent', () => {
    let home = createStarterHome();
    const [a, b, c] = ['tshirt', 'jeans', 'socks'].map((type) => {
      const added = addItemIn(home, SHELF, type);
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

describe('where a new thing goes', () => {
  test('into what is selected if that may hold it, otherwise beside it', () => {
    const bag = addOrganizerIn(createStarterHome(), BOX, 'pouch');
    const sock = addItemIn(bag.home, null, 'socks');
    const tote = addOrganizerIn(sock.home, null, 'tote');
    const box = addOrganizerIn(tote.home, null, 'small');
    const home = box.home;

    // Clothing goes into a bag, a box or a compartment.
    expect(nearestHolder(home, bag.id, sock.id)).toBe(bag.id);
    expect(nearestHolder(home, BOX, sock.id)).toBe(BOX);
    // A bag aimed at a bag goes beside it, in the box; aimed at a box, into it.
    expect(nearestHolder(home, bag.id, tote.id)).toBe(BOX);
    expect(nearestHolder(home, BOX, tote.id)).toBe(BOX);
    // A box aimed at a bag in a box goes on that box's shelf.
    expect(nearestHolder(home, bag.id, box.id)).toBe('closet-b1-c1-4');
    // The sink holds nothing, so it is the floor; so is aiming at nothing.
    expect(nearestHolder(home, 'sink-unit-c2-2', sock.id)).toBeNull();
    expect(nearestHolder(home, null, sock.id)).toBeNull();
  });
});

describe('removing', () => {
  test('deleting a box moves its contents to the floor', () => {
    const first = addItemIn(createStarterHome(), BOX, 'tshirt');
    const second = addItemIn(first.home, BOX, 'jeans');
    const after = remove(second.home, BOX);
    expect(after.organizers.some((o) => o.id === BOX)).toBe(false);
    expect(idsIn(after, null)).toEqual([first.id, second.id]);
    expect(after.items).toHaveLength(2);
  });

  test('deleting an item removes only that item', () => {
    const { home, id } = addItemIn(createStarterHome(), BOX, 'tshirt');
    expect(remove(home, id).items).toEqual([]);
    expect(() => remove(home, 'nothing')).toThrow(Refusal);
  });
});

describe('editing', () => {
  test('changing the type resets the parts to that type', () => {
    const { home, id } = addItemIn(createStarterHome(), SHELF, 'tshirt');
    const retyped = updateItem(home, id, { type: 'trousers' });
    expect(itemIn(retyped, id).parts.map((part) => part.role)).toEqual(['body', 'leg']);
  });

  test('bad values are refused', () => {
    const { home, id } = addItemIn(createStarterHome(), SHELF, 'tshirt');
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
    const { home, id } = addItemIn(createStarterHome(), BOX, 'tshirt');
    expect(pathTo(home, id)).toEqual(['Bedroom', 'Closet B1', 'Hand bags', 'Hand bags', 'New t-shirt']);
    const loose = addItemIn(home, null, 'socks');
    expect(pathTo(loose.home, loose.id)).toEqual(['Floor', 'New socks (pair)']);
  });

  test('destinations lists the floor, every compartment that stores things, and every box and bag', () => {
    const bag = addOrganizerIn(createStarterHome(), null, 'pouch');
    const home = move(bag.home, bag.id, BOX);
    const loose = addOrganizerIn(home, null, 'tote');
    const list = destinations(loose.home);
    const cells = home.furniture.flatMap((f) => f.columns.flatMap((column) => column.cells));
    const storing = cells.filter((cell) => cell.kind !== 'fixture');

    expect(list[0]).toEqual({ id: null, label: 'Floor' });
    expect(list).toHaveLength(1 + storing.length + loose.home.organizers.length);
    expect(new Set(list.map((d) => d.id)).size).toBe(list.length);
    expect(list.some((d) => d.id === 'sink-unit-c2-2')).toBe(false);
    expect(list.find((d) => d.id === BOX)!.label).toBe('Closet B1 › Hand bags › Hand bags');
    expect(list.find((d) => d.id === bag.id)!.label).toBe('Closet B1 › Hand bags › Hand bags › Pouch');
    expect(list.find((d) => d.id === loose.id)!.label).toBe('Floor › Tote bag');
  });
});

describe('doors and drawers', () => {
  const piece = (home: Home, id: string) => home.furniture.find((f) => f.id === id)!;
  const cell = (home: Home, id: string) => indexHome(home).cells.get(id)!.cell;

  test('the closets start open; the sink unit and wall cabinet start closed', () => {
    const home = createStarterHome();
    expect(isReachable(piece(home, 'closet-b1'), cell(home, SHELF), home.open)).toBe(true);
    expect(isReachable(piece(home, 'sink-unit'), cell(home, 'sink-unit-c1-1'), home.open)).toBe(false);
    expect(isReachable(piece(home, 'wall-cabinet'), cell(home, 'wall-cabinet-c1-2'), home.open)).toBe(false);
    // Shelving with no doors is always reachable.
    expect(isReachable(piece(home, 'shelves-c1'), cell(home, 'shelves-c1-c1-1'), home.open)).toBe(true);
  });

  test('a door opens and closes, and one door covers both shelves of a side cabinet', () => {
    const sink = (home: Home) => piece(home, 'sink-unit');
    const opened = toggleOpen(createStarterHome(), 'sink-unit-door1');
    expect(isReachable(sink(opened), cell(opened, 'sink-unit-c1-1'), opened.open)).toBe(true);
    expect(isReachable(sink(opened), cell(opened, 'sink-unit-c1-2'), opened.open)).toBe(true);
    expect(isReachable(sink(opened), cell(opened, 'sink-unit-c3-1'), opened.open)).toBe(false);
    const closed = toggleOpen(opened, 'sink-unit-door1');
    expect(isReachable(sink(closed), cell(closed, 'sink-unit-c1-1'), closed.open)).toBe(false);
  });

  test('a drawer is reachable only while it is open, and the sink never is', () => {
    const home = createStarterHome();
    const drawer = 'sink-unit-c1-3';
    expect(isReachable(piece(home, 'sink-unit'), cell(home, drawer), home.open)).toBe(false);
    const opened = toggleOpen(home, drawer);
    expect(isReachable(piece(opened, 'sink-unit'), cell(opened, drawer), opened.open)).toBe(true);
    const all = setAllOpen(home, piece(home, 'sink-unit'), true);
    expect(isReachable(piece(all, 'sink-unit'), cell(all, 'sink-unit-c2-2'), all.open)).toBe(false);
  });

  test('open all and close all touch only that piece', () => {
    const home = createStarterHome();
    const all = setAllOpen(home, piece(home, 'sink-unit'), true);
    expect(all.open).toEqual(expect.arrayContaining(['sink-unit-door1', 'sink-unit-door2', 'sink-unit-door3', 'sink-unit-c1-3', 'sink-unit-c3-3']));
    expect(all.open).toEqual(expect.arrayContaining(home.open));
    const none = setAllOpen(all, piece(all, 'sink-unit'), false);
    expect([...none.open].sort()).toEqual([...home.open].sort());
  });

  test('reveal opens whatever is in front of a thing, and nothing else', () => {
    const inDrawer = addItemIn(createStarterHome(), 'sink-unit-c1-3', 'socks');
    expect(reveal(inDrawer.home, inDrawer.id).open).toEqual([...inDrawer.home.open, 'sink-unit-c1-3']);

    // Something in a bag in a box behind a door: the door opens.
    const box = addOrganizerIn(createStarterHome(), 'sink-unit-c2-1', 'small');
    const bag = addOrganizerIn(box.home, box.id, 'pouch');
    const deep = addItemIn(bag.home, bag.id, 'socks');
    expect(reveal(deep.home, deep.id).open).toEqual([...deep.home.open, 'sink-unit-door2']);

    // Already open, on open shelving, or on the floor: nothing changes.
    const shelved = addItemIn(createStarterHome(), SHELF, 'socks');
    expect(reveal(shelved.home, shelved.id)).toBe(shelved.home);
    const loose = addItemIn(createStarterHome(), null, 'socks');
    expect(reveal(loose.home, loose.id)).toBe(loose.home);
  });

  test('openingsOf lists doors and drawers', () => {
    const home = createStarterHome();
    expect(openingsOf(piece(home, 'sink-unit'))).toEqual(['sink-unit-door1', 'sink-unit-door2', 'sink-unit-door3', 'sink-unit-c1-3', 'sink-unit-c3-3']);
    expect(openingsOf(piece(home, 'shelves-c1'))).toEqual([]);
  });

  test('the sink unit: a drawer over two shelves on each side, the sink over a cabinet in the middle', () => {
    const sink = piece(createStarterHome(), 'sink-unit');
    const kinds = sink.columns.map((column) => [...column.cells].reverse().map((c) => c.kind));
    expect(kinds).toEqual([
      ['drawer', 'shelf', 'shelf'],
      ['fixture', 'shelf'],
      ['drawer', 'shelf', 'shelf'],
    ]);
  });
});
