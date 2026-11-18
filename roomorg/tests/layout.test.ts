import { describe, expect, test } from 'vitest';
import { arrange, arrangeFloor, arrangeRail, arrangeShelf, boxesThatDoNotFit, insideOf } from '../src/model/layout';
import { place, placeOrLeave } from '../src/model/place';
import { HANGER_GAP, PACKED_GAP, THING_GAP, clothingType, organizerPreset } from '../src/model/presets';
import { createStarterHome } from '../src/model/roomData';
import { updateItem, type Child } from '../src/model/tree';
import type { Cell, Fold, Item, Organizer } from '../src/model/types';
import { addItemIn, addOrganizerIn } from './helpers';

const item = (id: string, typeId: string, fold: Fold): Child => ({
  kind: 'item',
  item: {
    id,
    kind: 'clothing',
    name: id,
    type: typeId,
    colour: '#336699',
    parentId: null,
    order: 0,
    parts: clothingType(typeId).parts,
    fold,
    previousFold: null,
  } satisfies Item,
});

const box = (id: string, presetId: string): Child => {
  const preset = organizerPreset(presetId);
  const organizer: Organizer = { id, kind: preset.kind, name: id, parentId: null, order: 0, outer: preset.outer, wall: preset.wall };
  return { kind: 'organizer', organizer };
};

const shelf = { w: 96, h: 23, d: 57 };

describe('on a shelf', () => {
  test('boxes run left to right with a gap between them, flush with the front', () => {
    const placed = arrangeShelf(shelf, [box('a', 'large'), box('b', 'medium'), box('c', 'small')]);
    expect(placed.map((p) => p.x)).toEqual([0, 46 + THING_GAP, 46 + 28 + 2 * THING_GAP]);
    expect(placed.every((p) => p.y === 0 && p.z === 0 && p.fits)).toBe(true);
  });

  test('things are packed closer inside a box', () => {
    const placed = arrangeShelf(shelf, [box('a', 'small'), box('b', 'small')], PACKED_GAP);
    expect(placed[1].x).toBe(18 + PACKED_GAP);
  });

  test('something deeper than the shelf sticks out in front of it', () => {
    const [placed] = arrangeShelf({ w: 96, h: 23, d: 39 }, [box('a', 'large')]);
    expect(placed.z).toBe(39 - 50);
  });

  test('one large, one medium and one small fit; a second large does not', () => {
    const placed = arrangeShelf(shelf, [box('a', 'large'), box('b', 'medium'), box('c', 'small'), box('d', 'large')]);
    expect(placed.map((p) => p.fits)).toEqual([true, true, true, false]);
  });

  test('loose clothing stacks into a pile, then starts a new one', () => {
    const tees = ['a', 'b', 'c'].map((id) => item(id, 'tshirt', 'half'));
    const placed = arrangeShelf(shelf, tees);
    expect(placed.map((p) => p.x)).toEqual([0, 0, 0]);
    expect(placed[1].y).toBeCloseTo(placed[0].h);
    expect(placed[2].y).toBeCloseTo(placed[0].h + placed[1].h);

    // A wider item cannot sit on a narrower pile, so it starts the next pile.
    const mixed = arrangeShelf(shelf, [item('narrow', 'tshirt', 'quarters'), item('wide', 'tshirt', 'half')]);
    expect(mixed[1].x).toBeCloseTo(mixed[0].w + THING_GAP);
    expect(mixed[1].y).toBe(0);
  });

  test('a pile stops at the shelf above', () => {
    const sweaters = Array.from({ length: 6 }, (_, i) => item(`s${i}`, 'sweater', 'half'));
    const placed = arrangeShelf(shelf, sweaters);
    for (const p of placed) expect(p.y + p.h).toBeLessThanOrEqual(shelf.h);
    expect(new Set(placed.map((p) => p.x)).size).toBeGreaterThan(1);
  });

  test('a box does not join a pile, and clothing does not sit on a box', () => {
    const placed = arrangeShelf(shelf, [item('a', 'tshirt', 'half'), box('b', 'small'), item('c', 'tshirt', 'half')]);
    expect(placed.map((p) => p.y)).toEqual([0, 0, 0]);
    expect(placed[2].x).toBeCloseTo(placed[0].w + placed[1].w + 2 * THING_GAP);
  });

  test('anything too wide, too tall or too deep is marked as not fitting', () => {
    expect(arrangeShelf({ w: 40, h: 23, d: 57 }, [box('a', 'large')])[0].fits).toBe(false);
    expect(arrangeShelf({ w: 96, h: 15, d: 57 }, [box('a', 'large')])[0].fits).toBe(false);
    expect(arrangeShelf({ w: 96, h: 23, d: 39 }, [box('a', 'large')])[0].fits).toBe(false);
    // A T-shirt laid flat is 68 cm long: too deep for a 39 cm shelf, fine once folded in half.
    expect(arrangeShelf({ w: 98, h: 30, d: 39 }, [item('a', 'tshirt', 'flat')])[0].fits).toBe(false);
    expect(arrangeShelf({ w: 98, h: 30, d: 39 }, [item('a', 'tshirt', 'half')])[0].fits).toBe(true);
  });
});

describe('on a rail', () => {
  const cell: Cell = { id: 'c', name: 'Hanging', kind: 'hanging', height: 110, railDrop: 6, drawerWall: 0 };
  const space = { w: 96, h: 110, d: 57 };

  test('clothing hangs side by side, just below the rail', () => {
    const placed = arrangeRail(space, cell, [item('a', 'shirt', 'half'), item('b', 'jacket', 'flat')]);
    expect(placed[0].x).toBe(0);
    expect(placed[1].x).toBeCloseTo(placed[0].w);
    for (const p of placed) {
      expect(p.y + p.h).toBeCloseTo(space.h - cell.railDrop - HANGER_GAP);
      expect(p.z + p.d / 2).toBeCloseTo(space.d / 2); // centred under the rail
    }
    expect(placed.every((p) => p.fits)).toBe(true);
  });

  test('a full rail and an item too long to hang are marked', () => {
    const many = Array.from({ length: 45 }, (_, i) => item(`i${i}`, 'shirt', 'flat'));
    const placed = arrangeRail(space, cell, many);
    expect(placed[0].fits).toBe(true);
    expect(placed[44].fits).toBe(false);

    const short = { ...cell, height: 80 };
    expect(arrangeRail({ ...space, h: 80 }, short, [item('coat', 'coat', 'flat')])[0].fits).toBe(false);
  });

  test('boxes sit on the floor under the clothing', () => {
    const placed = arrangeRail(space, cell, [item('a', 'shirt', 'flat'), box('b', 'large')]);
    expect(placed[1]).toMatchObject({ kind: 'organizer', x: 0, y: 0, fits: true });
  });
});

describe('arrange', () => {
  test('looks up the space of a compartment or the inside of a box', () => {
    let home = createStarterHome();
    const added = addItemIn(home, 'closet-b1-c1-4-box1', 'tshirt');

    // Folded in half a T-shirt is still 50 cm wide: too wide for a large box. In quarters it fits.
    home = updateItem(added.home, added.id, { fold: 'half' });
    expect(arrange(home, 'closet-b1-c1-4-box1').map((p) => p.fits)).toEqual([false]);
    home = updateItem(home, added.id, { fold: 'quarters' });
    expect(arrange(home, 'closet-b1-c1-4-box1').map((p) => p.fits)).toEqual([true]);

    const onShelf = arrange(home, 'closet-b1-c1-2');
    expect(onShelf.map((p) => p.kind)).toEqual(['organizer', 'organizer', 'organizer']);
    expect(onShelf.every((p) => p.fits)).toBe(true);
    expect(arrange(home, 'no-such-place')).toEqual([]);
  });

  test('the inside of a box is its outside minus walls and floor', () => {
    const { home, id } = addOrganizerIn(createStarterHome(), null, 'large');
    const organizer = home.organizers.find((o) => o.id === id)!;
    expect(insideOf(organizer)).toEqual({ w: 46 - 0.6, h: 20 - 0.3, d: 50 - 0.6 });
  });

  test('every starter box fits where the table put it', () => {
    const home = createStarterHome();
    const shelves = new Set(home.organizers.map((o) => o.parentId!));
    for (const shelfId of shelves) expect(arrange(home, shelfId).every((p) => p.fits), shelfId).toBe(true);
  });
});

describe('on the floor', () => {
  test('things sit in front of the furniture, in rows that wrap', () => {
    const things = [box('a', 'large'), box('b', 'large'), box('c', 'large')];
    const { placed, depth } = arrangeFloor(things, 100);
    // Two large boxes fit across 100 cm; the third starts a second row, further out.
    expect(placed.map((p) => p.x)).toEqual([0, 46 + THING_GAP, 0]);
    expect(placed[0].z).toBe(-(THING_GAP + 50));
    expect(placed[2].z).toBe(-(2 * THING_GAP + 100));
    expect(depth).toBe(100 + 3 * THING_GAP);
    expect(placed.every((p) => p.fits)).toBe(true);
  });

  test('a pile of clothing stays together when rows wrap', () => {
    const tees = ['a', 'b', 'c'].map((id) => item(id, 'tshirt', 'half'));
    const { placed } = arrangeFloor([box('x', 'large'), ...tees], 60);
    expect(placed.slice(1).map((p) => p.x)).toEqual([0, 0, 0]);
    expect(new Set(placed.slice(1).map((p) => p.z)).size).toBe(1);
    expect(placed[1].z).toBeLessThan(placed[0].z);
  });

  test('something wider than a row still gets a row of its own', () => {
    const { placed } = arrangeFloor([box('a', 'large'), box('b', 'large')], 30);
    expect(placed.map((p) => p.x)).toEqual([0, 0]);
  });

  test('an empty floor', () => {
    expect(arrangeFloor([], 100)).toEqual({ placed: [], depth: THING_GAP });
  });
});

describe('boxes need room', () => {
  const FULL = 'closet-b2-c1-4'; // large + small + medium
  const ROOMY = 'closet-b1-c1-4'; // one large

  test('a box is refused where it does not fit, and goes where it does', () => {
    const home = createStarterHome();
    const jeans = 'closet-b2-c1-3-box1'; // a large box
    expect(() => place(home, jeans, FULL)).toThrow(/not enough room/);
    const moved = place(home, jeans, ROOMY);
    expect(moved.organizers.find((o) => o.id === jeans)!.parentId).toBe(ROOMY);
    expect(boxesThatDoNotFit(moved, ROOMY)).toBe(0);
  });

  test('a box can be reordered on its own shelf, and set on the floor', () => {
    const home = createStarterHome();
    expect(() => place(home, `${FULL}-box3`, FULL, 0)).not.toThrow();
    expect(place(home, `${FULL}-box1`, null).organizers.find((o) => o.id === `${FULL}-box1`)!.parentId).toBeNull();
  });

  test('a box too tall for a compartment is refused even when it is empty', () => {
    const { home, id } = addOrganizerIn(createStarterHome(), null, 'large');
    expect(() => place(home, id, 'sink-unit-c1-3')).toThrow(/not enough room/);
  });

  test('bags and clothing are soft: they are never refused for room, only marked', () => {
    let { home, id } = addOrganizerIn(createStarterHome(), null, 'backpack');
    home = place(home, id, FULL);
    expect(arrange(home, FULL).find((p) => p.id === id)!.fits).toBe(false);
    expect(boxesThatDoNotFit(home, FULL)).toBe(0);

    const coat = addItemIn(home, null, 'coat');
    expect(() => place(coat.home, coat.id, FULL)).not.toThrow();
  });

  test('a shelf that already has a box sticking out does not block other moves there', () => {
    // Shrink nothing; overfill a shelf directly, then reorder within it.
    const overfull = addOrganizerIn(createStarterHome(), FULL, 'large');
    expect(boxesThatDoNotFit(overfull.home, FULL)).toBe(1);
    expect(() => place(overfull.home, `${FULL}-box1`, FULL, 3)).not.toThrow();
  });

  test('placeOrLeave leaves a new thing on the floor and says why', () => {
    const { home, id } = addOrganizerIn(createStarterHome(), null, 'large');
    const left = placeOrLeave(home, id, FULL);
    expect(left.refused).toMatch(/not enough room/);
    expect(left.home.organizers.find((o) => o.id === id)!.parentId).toBeNull();
    const put = placeOrLeave(home, id, ROOMY);
    expect(put.refused).toBeNull();
    expect(put.home.organizers.find((o) => o.id === id)!.parentId).toBe(ROOMY);
  });
});
