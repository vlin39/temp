/*
 * The saved data.
 *
 * Every length is in centimetres. Width runs left to right, height runs up,
 * depth runs front to back.
 *
 * Rooms, furniture, organizers and items are flat lists that point at their
 * parent by id, like rows in a database. Moving something changes one id.
 */

export type Id = string;

export interface Size3 {
  w: number;
  h: number;
  d: number;
}

/** A box-shaped space, placed relative to some origin. */
export interface Box extends Size3 {
  x: number;
  y: number;
}

export interface Room {
  id: Id;
  name: string;
  /** Where the room sits on the plan of the whole suite. */
  x: number;
  y: number;
  /** Floor size: w runs across the plan, d runs down it. */
  w: number;
  d: number;
}

/** Which way the front of a piece of furniture points on the plan. */
export type Facing = 'north' | 'east' | 'south' | 'west';

/**
 * - 'shelf': an open space things sit in.
 * - 'hanging': a space with a rail.
 * - 'drawer': slides out; things can only go in while it is open.
 * - 'fixture': takes up room but stores nothing (a sink).
 */
export type CellKind = 'shelf' | 'hanging' | 'drawer' | 'fixture';

/** One compartment of a piece of furniture. */
export interface Cell {
  id: Id;
  name: string;
  kind: CellKind;
  /** Clear inside height. */
  height: number;
  /** Hanging compartments: from the top of the compartment down to the rail. Otherwise 0. */
  railDrop: number;
  /** Drawers: thickness of the drawer's own walls. Otherwise 0. */
  drawerWall: number;
}

/** A vertical stack of compartments, listed bottom to top. */
export interface Column {
  id: Id;
  /** Clear inside width. */
  width: number;
  cells: Cell[];
}

/** A door on the front of a piece. It covers the rectangle around the compartments it names. */
export interface Door {
  id: Id;
  name: string;
  cellIds: Id[];
}

export interface PanelThickness {
  side: number;
  top: number;
  bottom: number;
  back: number;
}

export interface Furniture {
  id: Id;
  name: string;
  roomId: Id;
  /** The corner of its floor footprint nearest the room's top-left corner. Fixed for now. */
  x: number;
  y: number;
  /** Fixed for now. */
  facing: Facing;
  /** Outside size. */
  outer: Size3;
  panel: PanelThickness;
  shelfThickness: number;
  dividerThickness: number;
  /** Legs or a plinth under the piece. */
  baseHeight: number;
  /** Side by side, left to right. Empty for furniture with no storage (a bed, a desk). */
  columns: Column[];
  doors: Door[];
  /** True until one of its measurements is edited. */
  estimated: boolean;
}

/**
 * Something that holds other things.
 * A box is rigid: it holds bags and clothing, and needs room wherever it goes.
 * A bag is soft: it holds clothing, and can go inside a box.
 */
export interface Organizer {
  id: Id;
  kind: 'box' | 'bag';
  name: string;
  /** A compartment's id, a box's id (for a bag), or null when it is on the floor. */
  parentId: Id | null;
  /** Position among the things that share its parent. */
  order: number;
  outer: Size3;
  wall: number;
}

export type Fold = 'flat' | 'half' | 'thirds' | 'quarters' | 'rolled' | 'hung';

export type PartRole = 'body' | 'sleeve' | 'leg';

/** One rectangle of a garment laid flat. */
export interface Part {
  role: PartRole;
  /** Across. */
  w: number;
  /** Down. */
  l: number;
  /** Thickness lying flat (both layers of fabric). */
  t: number;
  /** 1, or 2 for a pair of sleeves or legs. */
  count: number;
}

export interface Item {
  id: Id;
  /** Only clothing for now; other kinds can be added later. */
  kind: 'clothing';
  name: string;
  /** A clothing type id from presets.ts. */
  type: string;
  /** '#rrggbb'. */
  colour: string;
  /** A compartment's id, an organizer's id, or null when it is on the floor. */
  parentId: Id | null;
  order: number;
  /** The laid-flat shape. The first 'body' part is the main one. */
  parts: Part[];
  fold: Fold;
  /** While the item hangs on a rail: the fold to go back to when it comes off. */
  previousFold: Fold | null;
}

export interface Home {
  version: 2;
  rooms: Room[];
  furniture: Furniture[];
  organizers: Organizer[];
  items: Item[];
  /** The ids of the doors and drawers that are open. */
  open: Id[];
  /** When a backup file was last downloaded (ISO date), or null. */
  lastBackupAt: string | null;
}
