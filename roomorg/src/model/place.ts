/*
 * Putting things down.
 *
 * `place` is what dragging, the "Where" list and the Add buttons use. It is
 * `move` from tree.ts plus one rule: a box is rigid, so it needs room. Bags
 * and clothing are soft, so they are never refused for lack of room, only
 * marked when they do not fit. The floor always has room.
 */

import { boxesThatDoNotFit } from './layout';
import { Refusal } from './refusal';
import { canHold, indexHome, move } from './tree';
import type { Home, Id } from './types';

/** Puts a thing into a parent at a position, or throws a Refusal saying why not. */
export function place(home: Home, id: Id, parentId: Id | null, position?: number): Home {
  const moved = move(home, id, parentId, position);
  const box = moved.organizers.find((organizer) => organizer.id === id && organizer.kind === 'box');
  if (box && parentId !== null && boxesThatDoNotFit(moved, parentId) > boxesThatDoNotFit(home, parentId)) {
    throw new Refusal(`There is not enough room for "${box.name}" there.`);
  }
  return moved;
}

/**
 * Puts a new thing where it was asked to go, or leaves it on the floor if it
 * cannot go there. `refused` is the reason when it stayed on the floor.
 */
export function placeOrLeave(home: Home, id: Id, parentId: Id | null): { home: Home; refused: string | null } {
  if (parentId === null) return { home, refused: null };
  try {
    return { home: place(home, id, parentId), refused: null };
  } catch (error) {
    if (error instanceof Refusal) return { home, refused: error.message };
    throw error;
  }
}

/**
 * The nearest place that may hold a thing, starting from `wish` and working
 * outward: a new box aimed at a box goes beside it. Null is the floor.
 */
export function nearestHolder(home: Home, wish: Id | null, id: Id): Id | null {
  const index = indexHome(home);
  let parentId = wish;
  while (parentId !== null && !canHold(index, parentId, id)) {
    parentId = index.organizers.get(parentId)?.parentId ?? null;
  }
  return parentId;
}
