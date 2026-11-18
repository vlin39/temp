/*
 * Where things are, and moving them.
 *
 * Organizers and items point at their parent by id:
 *   a compartment holds boxes, bags and clothing (unless it is a fixture),
 *   a box holds bags and clothing,
 *   a bag holds clothing,
 *   the floor (parent null) holds anything.
 */

import { doorOf, openingsOf } from './furniture';
import { newId } from './ids';
import { UNFINISHED_COLOUR, clothingType, type OrganizerPreset } from './presets';
import { Refusal } from './refusal';
import type { Cell, Column, Fold, Furniture, Home, Id, Item, Organizer } from './types';

export interface CellRef {
  furniture: Furniture;
  column: Column;
  cell: Cell;
}

/** Quick lookups by id, built once per change. */
export interface HomeIndex {
  cells: Map<Id, CellRef>;
  organizers: Map<Id, Organizer>;
  items: Map<Id, Item>;
}

/** Something that sits in a compartment, in an organizer or on the floor. */
export type Child = { kind: 'organizer'; organizer: Organizer } | { kind: 'item'; item: Item };

export type ParentKind = 'cell' | 'organizer' | 'floor';

/** A place something can be moved to, for the "Where" list. */
export interface Destination {
  id: Id | null;
  label: string;
}

export function indexHome(home: Home): HomeIndex {
  const cells = new Map<Id, CellRef>();
  for (const furniture of home.furniture) {
    for (const column of furniture.columns) {
      for (const cell of column.cells) cells.set(cell.id, { furniture, column, cell });
    }
  }
  return {
    cells,
    organizers: new Map(home.organizers.map((organizer) => [organizer.id, organizer])),
    items: new Map(home.items.map((item) => [item.id, item])),
  };
}

export const childId = (child: Child): Id => (child.kind === 'organizer' ? child.organizer.id : child.item.id);
const childOrder = (child: Child): number => (child.kind === 'organizer' ? child.organizer.order : child.item.order);

/** The organizers and items directly inside a parent, in order. */
export function childrenOf(home: Home, parentId: Id | null): Child[] {
  const children: Child[] = [
    ...home.organizers
      .filter((organizer) => organizer.parentId === parentId)
      .map((organizer): Child => ({ kind: 'organizer', organizer })),
    ...home.items.filter((item) => item.parentId === parentId).map((item): Child => ({ kind: 'item', item })),
  ];
  return children.sort((a, b) => childOrder(a) - childOrder(b));
}

export function parentKindOf(index: HomeIndex, parentId: Id | null): ParentKind | null {
  if (parentId === null) return 'floor';
  if (index.cells.has(parentId)) return 'cell';
  if (index.organizers.has(parentId)) return 'organizer';
  return null;
}

/**
 * Why this parent cannot hold this thing, or null if it can.
 * The sentence is written for the person using the app.
 */
export function whyNotHold(index: HomeIndex, parentId: Id | null, id: Id): string | null {
  const organizer = index.organizers.get(id);
  if (!organizer && !index.items.has(id)) return 'That no longer exists.';
  if (parentId === null) return null;
  if (parentId === id) return 'A thing cannot go inside itself.';

  const cellRef = index.cells.get(parentId);
  if (cellRef) return cellRef.cell.kind === 'fixture' ? `Nothing can be stored in ${cellRef.cell.name}.` : null;

  const holder = index.organizers.get(parentId);
  if (!holder) return 'That place no longer exists.';
  if (!organizer) return null; // clothing goes in any box or bag
  if (holder.kind === 'bag') return 'A bag can only hold clothing.';
  return organizer.kind === 'box' ? 'A box cannot go inside another box.' : null;
}

export function canHold(index: HomeIndex, parentId: Id | null, id: Id): boolean {
  return whyNotHold(index, parentId, id) === null;
}

export function isRail(index: HomeIndex, parentId: Id | null): boolean {
  return parentId !== null && index.cells.get(parentId)?.cell.kind === 'hanging';
}

/** The compartment something is in, however deeply it is nested. Undefined if it is on the floor. */
export function cellHolding(index: HomeIndex, id: Id): CellRef | undefined {
  let parentId = (index.items.get(id) ?? index.organizers.get(id))?.parentId ?? null;
  while (parentId !== null) {
    const cellRef = index.cells.get(parentId);
    if (cellRef) return cellRef;
    parentId = index.organizers.get(parentId)?.parentId ?? null;
  }
  return undefined;
}

/** An item going onto a rail becomes hung; coming off, it goes back to its earlier fold. */
function refold(item: Item, ontoRail: boolean): Item {
  if (ontoRail && item.fold !== 'hung') return { ...item, fold: 'hung', previousFold: item.fold };
  if (!ontoRail && item.fold === 'hung') return { ...item, fold: item.previousFold ?? 'flat', previousFold: null };
  return item;
}

/**
 * Puts an organizer or item into a parent, at `position` among what is already there
 * (omit it to put the thing last). This only checks what may hold what; place.ts adds
 * the rule that a box needs room.
 */
export function move(home: Home, id: Id, newParentId: Id | null, position = Infinity): Home {
  const index = indexHome(home);
  const reason = whyNotHold(index, newParentId, id);
  if (reason !== null) throw new Refusal(reason);

  const siblings = childrenOf(home, newParentId).map(childId).filter((sibling) => sibling !== id);
  const at = Math.max(0, Math.min(position, siblings.length));
  const ordered = [...siblings.slice(0, at), id, ...siblings.slice(at)];
  const rank = new Map(ordered.map((member, place) => [member, place]));
  const ontoRail = isRail(index, newParentId);

  return {
    ...home,
    organizers: home.organizers.map((organizer) =>
      rank.has(organizer.id) ? { ...organizer, parentId: newParentId, order: rank.get(organizer.id)! } : organizer,
    ),
    items: home.items.map((item) => {
      if (!rank.has(item.id)) return item;
      const moved = item.id === id ? refold(item, ontoRail) : item;
      return { ...moved, parentId: newParentId, order: rank.get(item.id)! };
    }),
  };
}

/** Deletes an item, or an organizer whose contents then go to the floor. */
export function remove(home: Home, id: Id): Home {
  if (home.items.some((item) => item.id === id)) {
    return { ...home, items: home.items.filter((item) => item.id !== id) };
  }
  if (!home.organizers.some((organizer) => organizer.id === id)) throw new Refusal('That no longer exists.');

  let nextPlace = childrenOf(home, null).length;
  const toFloor = <T extends Organizer | Item>(thing: T): T =>
    thing.parentId === id ? { ...thing, parentId: null, order: nextPlace++ } : thing;
  return {
    ...home,
    organizers: home.organizers.filter((organizer) => organizer.id !== id).map(toFloor),
    items: home.items.map(toFloor),
  };
}

/** Adds a box or a bag of a preset size to the floor. */
export function addOrganizer(home: Home, preset: OrganizerPreset): { home: Home; id: Id } {
  const organizer: Organizer = {
    id: newId(),
    kind: preset.kind,
    name: preset.name,
    parentId: null,
    order: childrenOf(home, null).length,
    outer: { ...preset.outer },
    wall: preset.wall,
  };
  return { home: { ...home, organizers: [...home.organizers, organizer] }, id: organizer.id };
}

/** Adds a clothing item, laid flat, to the floor. */
export function addItem(home: Home, typeId: string): { home: Home; id: Id } {
  const type = clothingType(typeId);
  const item: Item = {
    id: newId(),
    kind: 'clothing',
    name: `New ${type.name.toLowerCase()}`,
    type: type.id,
    colour: UNFINISHED_COLOUR,
    parentId: null,
    order: childrenOf(home, null).length,
    parts: type.parts.map((part) => ({ ...part })),
    fold: 'flat',
    previousFold: null,
  };
  return { home: { ...home, items: [...home.items, item] }, id: item.id };
}

const positive = (n: number) => Number.isFinite(n) && n > 0;

/** Changes an organizer's name, size or wall thickness. */
export function updateOrganizer(home: Home, id: Id, changes: Partial<Pick<Organizer, 'name' | 'outer' | 'wall'>>): Home {
  const current = home.organizers.find((organizer) => organizer.id === id);
  if (!current) throw new Refusal('That no longer exists.');
  const next = { ...current, ...changes };
  const what = current.kind === 'box' ? 'box' : 'bag';
  if (next.name.trim() === '') throw new Refusal(`A ${what} needs a name.`);
  if (![next.outer.w, next.outer.h, next.outer.d].every(positive)) throw new Refusal('A size must be more than zero.');
  if (!(next.wall >= 0)) throw new Refusal('A thickness cannot be less than zero.');
  if (2 * next.wall >= Math.min(next.outer.w, next.outer.d) || next.wall >= next.outer.h) {
    throw new Refusal(`The walls would leave no room inside the ${what}.`);
  }
  return { ...home, organizers: home.organizers.map((organizer) => (organizer.id === id ? next : organizer)) };
}

/**
 * Changes an item's name, colour, type, parts or fold.
 * Changing the type resets the parts to that type's starter shape.
 * While an item hangs on a rail, a new fold is kept for when it comes off.
 */
export function updateItem(
  home: Home,
  id: Id,
  changes: Partial<Pick<Item, 'name' | 'colour' | 'type' | 'parts'>> & { fold?: Fold },
): Home {
  const current = home.items.find((item) => item.id === id);
  if (!current) throw new Refusal('That item no longer exists.');

  const { fold, ...rest } = changes;
  let next: Item = { ...current, ...rest };
  if (changes.type !== undefined && changes.type !== current.type) {
    next.parts = clothingType(changes.type).parts.map((part) => ({ ...part }));
  }
  if (fold !== undefined && fold !== 'hung') {
    next = current.fold === 'hung' ? { ...next, previousFold: fold } : { ...next, fold };
  }

  if (next.name.trim() === '') throw new Refusal('An item needs a name.');
  if (!/^#[0-9a-f]{6}$/i.test(next.colour)) throw new Refusal('That is not a colour.');
  if (!next.parts.every((part) => [part.w, part.l, part.t].every(positive))) {
    throw new Refusal('A size must be more than zero.');
  }
  return { ...home, items: home.items.map((item) => (item.id === id ? next : item)) };
}

/** Opens a closed door or drawer, or closes an open one. */
export function toggleOpen(home: Home, id: Id): Home {
  return { ...home, open: home.open.includes(id) ? home.open.filter((other) => other !== id) : [...home.open, id] };
}

/** Opens every door and drawer of a piece, or closes them all. */
export function setAllOpen(home: Home, furniture: Furniture, open: boolean): Home {
  const ids = openingsOf(furniture);
  const others = home.open.filter((id) => !ids.includes(id));
  return { ...home, open: open ? [...others, ...ids] : others };
}

/** Opens whatever is closed in front of a thing: the door of its compartment, and the compartment itself if it is a drawer. */
export function reveal(home: Home, id: Id): Home {
  const cellRef = cellHolding(indexHome(home), id);
  if (!cellRef) return home;
  const inFront = [doorOf(cellRef.furniture, cellRef.cell.id)?.id, cellRef.cell.kind === 'drawer' ? cellRef.cell.id : undefined];
  const closed = inFront.filter((opening): opening is Id => opening !== undefined && !home.open.includes(opening));
  return closed.length === 0 ? home : { ...home, open: [...home.open, ...closed] };
}

/** The names leading to something: room, furniture, compartment, any box and bag, then itself. */
export function pathTo(home: Home, id: Id): string[] {
  const index = indexHome(home);
  const own = index.organizers.get(id) ?? index.items.get(id);
  if (!own) return [];

  const path = [own.name];
  let parentId = own.parentId;
  while (parentId !== null) {
    const cellRef = index.cells.get(parentId);
    if (cellRef) {
      const room = home.rooms.find((candidate) => candidate.id === cellRef.furniture.roomId);
      return [...(room ? [room.name] : []), cellRef.furniture.name, cellRef.cell.name, ...path];
    }
    const holder = index.organizers.get(parentId);
    if (!holder) break;
    path.unshift(holder.name);
    parentId = holder.parentId;
  }
  return ['Floor', ...path];
}

/** Every place something can be moved to: the floor, each compartment, and each box and bag. */
export function destinations(home: Home): Destination[] {
  const list: Destination[] = [{ id: null, label: 'Floor' }];
  const organizersIn = (parentId: Id | null, prefix: string) => {
    for (const child of childrenOf(home, parentId)) {
      if (child.kind !== 'organizer') continue;
      const label = `${prefix} › ${child.organizer.name}`;
      list.push({ id: child.organizer.id, label });
      organizersIn(child.organizer.id, label);
    }
  };

  organizersIn(null, 'Floor');
  for (const furniture of home.furniture) {
    for (const column of furniture.columns) {
      // Listed top to bottom, the way the piece looks from the front.
      for (const cell of [...column.cells].reverse()) {
        if (cell.kind === 'fixture') continue;
        const label = `${furniture.name} › ${cell.name}`;
        list.push({ id: cell.id, label });
        organizersIn(cell.id, label);
      }
    }
  }
  return list;
}
