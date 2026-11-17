/*
 * Where things are, and moving them.
 *
 * Organizers and items point at their parent by id:
 *   a compartment holds organizers and items,
 *   an organizer holds items,
 *   the unsorted tray (parent null) holds both.
 */

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

/** Something that sits in a compartment, an organizer or the tray. */
export type Child = { kind: 'organizer'; organizer: Organizer } | { kind: 'item'; item: Item };

export type ParentKind = 'cell' | 'organizer' | 'tray';

/** A place something can be moved to, for the "Move to" list. */
export interface Destination {
  id: Id | null;
  label: string;
  kind: ParentKind;
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
  if (parentId === null) return 'tray';
  if (index.cells.has(parentId)) return 'cell';
  if (index.organizers.has(parentId)) return 'organizer';
  return null;
}

/** May this parent hold this kind of thing? An organizer cannot hold another organizer. */
export function canHold(index: HomeIndex, parentId: Id | null, kind: Child['kind']): boolean {
  const parentKind = parentKindOf(index, parentId);
  if (parentKind === null) return false;
  return parentKind !== 'organizer' || kind === 'item';
}

export function isRail(index: HomeIndex, parentId: Id | null): boolean {
  return parentId !== null && index.cells.get(parentId)?.cell.kind === 'hanging';
}

/** The compartment a box or an item is in, directly or inside a box. Undefined if it is in the tray. */
export function cellHolding(index: HomeIndex, id: Id): CellRef | undefined {
  const own = index.items.get(id) ?? index.organizers.get(id);
  const holder = own?.parentId ? index.organizers.get(own.parentId) : undefined;
  const cellId = holder ? holder.parentId : own?.parentId;
  return cellId ? index.cells.get(cellId) : undefined;
}

/** An item going onto a rail becomes hung; coming off, it goes back to its earlier fold. */
function refold(item: Item, ontoRail: boolean): Item {
  if (ontoRail && item.fold !== 'hung') return { ...item, fold: 'hung', previousFold: item.fold };
  if (!ontoRail && item.fold === 'hung') return { ...item, fold: item.previousFold ?? 'flat', previousFold: null };
  return item;
}

/**
 * Puts an organizer or item into a parent, at `position` among what is already there
 * (omit it to put the thing last).
 */
export function move(home: Home, id: Id, newParentId: Id | null, position = Infinity): Home {
  const index = indexHome(home);
  const kind = index.organizers.has(id) ? 'organizer' : index.items.has(id) ? 'item' : null;
  if (kind === null) throw new Refusal('That no longer exists.');
  if (!canHold(index, newParentId, kind)) {
    throw new Refusal(kind === 'organizer' ? 'A box cannot go inside another box.' : 'That cannot go there.');
  }

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

/** Deletes an item, or an organizer whose contents then go to the unsorted tray. */
export function remove(home: Home, id: Id): Home {
  if (home.items.some((item) => item.id === id)) {
    return { ...home, items: home.items.filter((item) => item.id !== id) };
  }
  if (!home.organizers.some((organizer) => organizer.id === id)) throw new Refusal('That no longer exists.');

  let nextTrayPlace = childrenOf(home, null).length;
  return {
    ...home,
    organizers: home.organizers.filter((organizer) => organizer.id !== id),
    items: home.items.map((item) => (item.parentId === id ? { ...item, parentId: null, order: nextTrayPlace++ } : item)),
  };
}

/** Adds an organizer of a preset size to a compartment or the tray. */
export function addOrganizer(home: Home, parentId: Id | null, preset: OrganizerPreset): { home: Home; id: Id } {
  const organizer: Organizer = {
    id: newId(),
    name: `${preset.name} box`,
    parentId: null,
    order: 0,
    outer: { ...preset.outer },
    wall: preset.wall,
  };
  const added = { ...home, organizers: [...home.organizers, organizer] };
  return { home: move(added, organizer.id, parentId), id: organizer.id };
}

/** Adds a clothing item, laid flat, to a compartment, an organizer or the tray. */
export function addItem(home: Home, parentId: Id | null, typeId: string): { home: Home; id: Id } {
  const type = clothingType(typeId);
  const item: Item = {
    id: newId(),
    kind: 'clothing',
    name: `New ${type.name.toLowerCase()}`,
    type: type.id,
    colour: UNFINISHED_COLOUR,
    parentId: null,
    order: 0,
    parts: type.parts.map((part) => ({ ...part })),
    fold: 'flat',
    previousFold: null,
  };
  const added = { ...home, items: [...home.items, item] };
  return { home: move(added, item.id, parentId), id: item.id };
}

const positive = (n: number) => Number.isFinite(n) && n > 0;

/** Changes an organizer's name, size or wall thickness. */
export function updateOrganizer(home: Home, id: Id, changes: Partial<Pick<Organizer, 'name' | 'outer' | 'wall'>>): Home {
  const current = home.organizers.find((organizer) => organizer.id === id);
  if (!current) throw new Refusal('That box no longer exists.');
  const next = { ...current, ...changes };
  if (next.name.trim() === '') throw new Refusal('A box needs a name.');
  if (![next.outer.w, next.outer.h, next.outer.d].every(positive)) throw new Refusal('A size must be more than zero.');
  if (!(next.wall >= 0)) throw new Refusal('A thickness cannot be less than zero.');
  if (2 * next.wall >= Math.min(next.outer.w, next.outer.d) || next.wall >= next.outer.h) {
    throw new Refusal('The walls would leave no room inside the box.');
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

/** The names leading to something: room, furniture, compartment, organizer, then itself. */
export function pathTo(home: Home, id: Id): string[] {
  const index = indexHome(home);
  const organizer = index.organizers.get(id);
  const item = index.items.get(id);
  const own = organizer ?? item;
  if (!own) return [];

  const path = [own.name];
  let parentId = own.parentId;
  const holder = parentId === null ? undefined : index.organizers.get(parentId);
  if (holder) {
    path.unshift(holder.name);
    parentId = holder.parentId;
  }
  const cellRef = parentId === null ? undefined : index.cells.get(parentId);
  if (!cellRef) return ['Unsorted', ...path];

  const room = home.rooms.find((candidate) => candidate.id === cellRef.furniture.roomId);
  return [...(room ? [room.name] : []), cellRef.furniture.name, cellRef.cell.name, ...path];
}

/** Every place something can be moved to: the tray, each compartment, and each organizer. */
export function destinations(home: Home): Destination[] {
  const list: Destination[] = [{ id: null, label: 'Unsorted tray', kind: 'tray' }];
  const organizersIn = (parentId: Id | null, prefix: string) => {
    for (const child of childrenOf(home, parentId)) {
      if (child.kind === 'organizer') {
        list.push({ id: child.organizer.id, label: `${prefix} › ${child.organizer.name}`, kind: 'organizer' });
      }
    }
  };

  organizersIn(null, 'Unsorted tray');
  for (const furniture of home.furniture) {
    for (const column of furniture.columns) {
      // Listed top to bottom, the way the piece looks from the front.
      for (const cell of [...column.cells].reverse()) {
        const label = `${furniture.name} › ${cell.name}`;
        list.push({ id: cell.id, label, kind: 'cell' });
        organizersIn(cell.id, label);
      }
    }
  }
  return list;
}
