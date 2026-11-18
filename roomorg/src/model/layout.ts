/*
 * Automatic arrangement.
 *
 * You choose where something goes and in what order; this decides exactly
 * where it sits. Anything that does not fit is still placed, and marked so
 * the screen can warn about it. (Whether a move is allowed at all is decided
 * in place.ts.)
 *
 * Positions are measured inside the space a thing sits in:
 *   x runs right from the left edge,
 *   y runs up from the bottom,
 *   z runs back from the front opening. A negative z is in front of it.
 */

import { hangLength, usableSpace, clearSpaces } from './furniture';
import { storedSize } from './folding';
import { FLOOR_PILE_HEIGHT, HANGER_GAP, PACKED_GAP, THING_GAP } from './presets';
import { childrenOf, childId, indexHome, type Child, type HomeIndex } from './tree';
import type { Cell, Home, Id, Organizer, Size3 } from './types';

/** Where one thing sits, and how big it is there. */
export interface Placed {
  id: Id;
  kind: 'organizer' | 'item';
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  d: number;
  fits: boolean;
}

/** The room inside an organizer: its outside minus its walls and floor. */
export function insideOf(organizer: Organizer): Size3 {
  return {
    w: organizer.outer.w - 2 * organizer.wall,
    h: organizer.outer.h - organizer.wall,
    d: organizer.outer.d - 2 * organizer.wall,
  };
}

function sizeOf(child: Child): Size3 {
  return child.kind === 'organizer' ? child.organizer.outer : storedSize(child.item);
}

/**
 * On a shelf, or inside an organizer or drawer: things line up left to right
 * with a gap between them, pushed to the front. Loose clothing stacks into a
 * pile while the next item is no wider than the pile and the pile still fits
 * under the shelf above.
 */
export function arrangeShelf(space: Size3, children: Child[], gap = THING_GAP): Placed[] {
  const placed: Placed[] = [];
  let x = 0;
  let pile: { x: number; width: number; height: number } | null = null;

  for (const child of children) {
    const size = sizeOf(child);
    const stacks =
      child.kind === 'item' && pile !== null && size.w <= pile.width && pile.height + size.h <= space.h;

    let at: { x: number; y: number };
    if (stacks && pile) {
      at = { x: pile.x, y: pile.height };
      pile.height += size.h;
    } else {
      at = { x, y: 0 };
      pile = child.kind === 'item' ? { x, width: size.w, height: size.h } : null;
      x += size.w + gap;
    }

    placed.push({
      id: childId(child),
      kind: child.kind,
      ...at,
      // Flush with the front. Something deeper than the space sticks out in front of it.
      z: Math.min(0, space.d - size.d),
      ...size,
      fits: at.x + size.w <= space.w && at.y + size.h <= space.h && size.d <= space.d,
    });
  }
  return placed;
}

/**
 * In a hanging compartment: clothing hangs side by side from the rail, and
 * organizers sit on the floor underneath.
 */
export function arrangeRail(space: Size3, cell: Cell, children: Child[]): Placed[] {
  const placed: Placed[] = [];
  const longest = hangLength(cell);
  let railX = 0;

  for (const child of children) {
    if (child.kind !== 'item') continue;
    const size = storedSize({ ...child.item, fold: 'hung' });
    const x = railX;
    railX += size.w;
    placed.push({
      id: child.item.id,
      kind: 'item',
      x,
      y: space.h - cell.railDrop - HANGER_GAP - size.h,
      z: (space.d - size.d) / 2, // centred under the rail
      ...size,
      fits: railX <= space.w && size.h <= longest && size.d <= space.d,
    });
  }

  const onTheFloor = children.filter((child) => child.kind === 'organizer');
  return [...placed, ...arrangeShelf(space, onTheFloor)];
}

/**
 * On the floor in front of a piece of furniture: rows of things, each row
 * `rowWidth` wide at most, the first row nearest the furniture. Returns the
 * placements and how far out from the furniture the rows reach.
 */
export function arrangeFloor(children: Child[], rowWidth: number): { placed: Placed[]; depth: number } {
  // Start from one long row, then cut it into rows. A pile stays together.
  const line = arrangeShelf({ w: Infinity, h: FLOOR_PILE_HEIGHT, d: Infinity }, children);

  const rows: Placed[][] = [];
  let rowStart = 0;
  for (const thing of line) {
    const startsColumn = thing.y === 0;
    const overflows = thing.x + thing.w - rowStart > rowWidth;
    if (rows.length === 0 || (startsColumn && overflows && thing.x > rowStart)) {
      rows.push([]);
      rowStart = thing.x;
    }
    rows[rows.length - 1].push({ ...thing, x: thing.x - rowStart });
  }

  const placed: Placed[] = [];
  let back = THING_GAP; // how far in front of the furniture this row's back edge is
  for (const row of rows) {
    for (const thing of row) placed.push({ ...thing, z: -(back + thing.d), fits: true });
    back += Math.max(...row.map((thing) => thing.d)) + THING_GAP;
  }
  return { placed, depth: back };
}

/** Where everything inside a compartment or an organizer sits. */
export function arrange(home: Home, parentId: Id, index: HomeIndex = indexHome(home)): Placed[] {
  const children = childrenOf(home, parentId);

  const organizer = index.organizers.get(parentId);
  if (organizer) return arrangeShelf(insideOf(organizer), children, PACKED_GAP);

  const cellRef = index.cells.get(parentId);
  if (!cellRef) return [];
  const space = usableSpace(cellRef.cell, clearSpaces(cellRef.furniture).get(parentId)!);
  return cellRef.cell.kind === 'hanging'
    ? arrangeRail(space, cellRef.cell, children)
    : arrangeShelf(space, children);
}

/** How many boxes in a compartment do not fit there. Bags and clothing are soft and are not counted. */
export function boxesThatDoNotFit(home: Home, parentId: Id): number {
  const index = indexHome(home);
  return arrange(home, parentId, index).filter((p) => !p.fits && index.organizers.get(p.id)?.kind === 'box').length;
}
