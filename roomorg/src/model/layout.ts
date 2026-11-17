/*
 * Automatic arrangement.
 *
 * You choose where something goes and in what order; this decides exactly
 * where it is drawn. Nothing is ever blocked: anything that does not fit is
 * still placed, and marked so the screen can warn about it.
 */

import { hangLength, usableSpace, clearSpaces } from './furniture';
import { storedSize } from './folding';
import { HANGER_GAP } from './presets';
import { childrenOf, childId, indexHome, type Child, type HomeIndex } from './tree';
import type { Cell, Home, Id, Organizer, Size3 } from './types';

/** A rectangle in the front view, measured from the bottom-left of the space it sits in. */
export interface Placed {
  id: Id;
  kind: 'organizer' | 'item';
  x: number;
  y: number;
  w: number;
  h: number;
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
 * On a shelf, or inside an organizer or drawer: things line up left to right.
 * Loose clothing stacks into a pile while the next item is no wider than the
 * pile and the pile still fits under the shelf above.
 */
export function arrangeShelf(space: Size3, children: Child[]): Placed[] {
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
      x += size.w;
    }

    placed.push({
      id: childId(child),
      kind: child.kind,
      ...at,
      w: size.w,
      h: size.h,
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
  let floorX = 0;

  for (const child of children) {
    if (child.kind === 'item') {
      const size = storedSize({ ...child.item, fold: 'hung' });
      const x = railX;
      railX += size.w;
      placed.push({
        id: child.item.id,
        kind: 'item',
        x,
        y: space.h - cell.railDrop - HANGER_GAP - size.h,
        w: size.w,
        h: size.h,
        fits: railX <= space.w && size.h <= longest && size.d <= space.d,
      });
    } else {
      const size = child.organizer.outer;
      const x = floorX;
      floorX += size.w;
      placed.push({
        id: child.organizer.id,
        kind: 'organizer',
        x,
        y: 0,
        w: size.w,
        h: size.h,
        fits: floorX <= space.w && size.h <= space.h && size.d <= space.d,
      });
    }
  }
  return placed;
}

/** Where everything inside a compartment or an organizer is drawn. */
export function arrange(home: Home, parentId: Id, index: HomeIndex = indexHome(home)): Placed[] {
  const children = childrenOf(home, parentId);

  const organizer = index.organizers.get(parentId);
  if (organizer) return arrangeShelf(insideOf(organizer), children);

  const cellRef = index.cells.get(parentId);
  if (!cellRef) return [];
  const space = usableSpace(cellRef.cell, clearSpaces(cellRef.furniture).get(parentId)!);
  return cellRef.cell.kind === 'hanging'
    ? arrangeRail(space, cellRef.cell, children)
    : arrangeShelf(space, children);
}

/** How far a set of placed things reaches to the right and upward. */
export function reach(placed: Placed[]): { w: number; h: number } {
  return {
    w: Math.max(0, ...placed.map((p) => p.x + p.w)),
    h: Math.max(0, ...placed.map((p) => p.y + p.h)),
  };
}
