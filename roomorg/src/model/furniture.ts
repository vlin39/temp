/*
 * Furniture measurements.
 *
 * A piece has an outside size and panel thicknesses. The clear space inside
 * each compartment is worked out from those, so nothing is stored twice.
 *
 * One rule must always hold (see checkPiece):
 *   in each column:  cell heights + the shelves between them = the inside height
 *   across columns:  column widths + the dividers between them = the inside width
 * Every edit below ends by restoring that rule, or refuses.
 */

import { HANGER_GAP, MIN_CLEAR } from './presets';
import { Refusal } from './refusal';
import type { Box, Cell, Column, Door, Furniture, Id } from './types';
import { toDisplay } from './units';

/**
 * What gives when a measurement changes:
 * - 'keepOutside': the outside size stays, and a neighbouring compartment absorbs the change.
 * - 'keepCompartments': the other compartments stay, and the piece grows or shrinks.
 */
export type ResizeMode = 'keepOutside' | 'keepCompartments';

export type Dimension = 'w' | 'h' | 'd';

export type ThicknessName = 'side' | 'top' | 'bottom' | 'back' | 'shelf' | 'divider' | 'base';

/** A panel to draw in the front view. x runs right from the left edge, y runs up from the floor. */
export interface PanelRect {
  kind: 'base' | 'side' | 'top' | 'bottom' | 'shelf' | 'divider';
  x: number;
  y: number;
  w: number;
  h: number;
}

const TOLERANCE = 0.01;

const sum = (numbers: number[]) => numbers.reduce((total, n) => total + n, 0);

export const innerWidth = (f: Furniture) => f.outer.w - 2 * f.panel.side;
export const innerHeight = (f: Furniture) => f.outer.h - f.baseHeight - f.panel.top - f.panel.bottom;
export const innerDepth = (f: Furniture) => f.outer.d - f.panel.back;

/** The height a column's compartments and shelves take up. */
function heightNeeded(f: Furniture, column: Column): number {
  return sum(column.cells.map((cell) => cell.height)) + (column.cells.length - 1) * f.shelfThickness;
}

/** The width the columns and dividers take up. */
function widthNeeded(f: Furniture): number {
  return sum(f.columns.map((column) => column.width)) + (f.columns.length - 1) * f.dividerThickness;
}

/** Where each compartment's empty space is, measured from the piece's front-bottom-left corner. */
export function clearSpaces(f: Furniture): Map<Id, Box> {
  const spaces = new Map<Id, Box>();
  let x = f.panel.side;
  for (const column of f.columns) {
    let y = f.baseHeight + f.panel.bottom;
    for (const cell of column.cells) {
      spaces.set(cell.id, { x, y, w: column.width, h: cell.height, d: innerDepth(f) });
      y += cell.height + f.shelfThickness;
    }
    x += column.width + f.dividerThickness;
  }
  return spaces;
}

/** The part of a compartment things can actually sit in: a drawer loses its own walls. */
export function usableSpace(cell: Cell, space: Box): Box {
  if (cell.kind !== 'drawer') return space;
  const wall = cell.drawerWall;
  return {
    x: space.x + wall,
    y: space.y + wall,
    w: space.w - 2 * wall,
    h: space.h - wall,
    d: space.d - 2 * wall,
  };
}

/**
 * The rectangle a door covers on the front of its piece: the area around its
 * compartments, widened to the middle of each neighbouring panel so that
 * doors side by side meet.
 */
export function doorRect(f: Furniture, door: Door): { x: number; y: number; w: number; h: number } {
  const spaces = clearSpaces(f);
  const covered = door.cellIds.flatMap((id) => (spaces.has(id) ? [spaces.get(id)!] : []));
  const left = Math.min(...covered.map((space) => space.x));
  const right = Math.max(...covered.map((space) => space.x + space.w));
  const bottom = Math.min(...covered.map((space) => space.y));
  const top = Math.max(...covered.map((space) => space.y + space.h));

  const floor = f.baseHeight + f.panel.bottom;
  const atLeftEdge = left <= f.panel.side + TOLERANCE;
  const atRightEdge = right >= f.outer.w - f.panel.side - TOLERANCE;
  const atBottom = bottom <= floor + TOLERANCE;
  const atTop = top >= f.outer.h - f.panel.top - TOLERANCE;

  const x = left - (atLeftEdge ? f.panel.side : f.dividerThickness / 2);
  const y = bottom - (atBottom ? f.panel.bottom : f.shelfThickness / 2);
  return {
    x,
    y,
    w: right + (atRightEdge ? f.panel.side : f.dividerThickness / 2) - x,
    h: top + (atTop ? f.panel.top : f.shelfThickness / 2) - y,
  };
}

/** The door in front of a compartment, if it has one. */
export function doorOf(f: Furniture, cellId: Id): Door | undefined {
  return f.doors.find((door) => door.cellIds.includes(cellId));
}

/** The ids of everything on a piece that opens: its doors and its drawers. */
export function openingsOf(f: Furniture): Id[] {
  const drawers = f.columns.flatMap((column) => column.cells.filter((cell) => cell.kind === 'drawer'));
  return [...f.doors.map((door) => door.id), ...drawers.map((drawer) => drawer.id)];
}

/**
 * Can you reach into this compartment right now? Not if it is behind a closed
 * door, is a closed drawer, or stores nothing. `open` is the home's list of
 * open doors and drawers.
 */
export function isReachable(f: Furniture, cell: Cell, open: Id[]): boolean {
  if (cell.kind === 'fixture') return false;
  if (cell.kind === 'drawer' && !open.includes(cell.id)) return false;
  const door = doorOf(f, cell.id);
  return !door || open.includes(door.id);
}

/** How long a garment can be and still hang clear of the compartment's floor. */
export function hangLength(cell: Cell): number {
  return cell.height - cell.railDrop - HANGER_GAP;
}

/** The panels to draw: base, sides, top, bottom, shelves and dividers. */
export function panels(f: Furniture): PanelRect[] {
  if (f.columns.length === 0) return [];
  const rects: PanelRect[] = [];
  const floor = f.baseHeight + f.panel.bottom;
  const add = (kind: PanelRect['kind'], x: number, y: number, w: number, h: number) => {
    if (w > 0 && h > 0) rects.push({ kind, x, y, w, h });
  };

  add('base', 0, 0, f.outer.w, f.baseHeight);
  add('bottom', 0, f.baseHeight, f.outer.w, f.panel.bottom);
  add('top', 0, f.outer.h - f.panel.top, f.outer.w, f.panel.top);
  add('side', 0, floor, f.panel.side, innerHeight(f));
  add('side', f.outer.w - f.panel.side, floor, f.panel.side, innerHeight(f));

  let x = f.panel.side;
  f.columns.forEach((column, index) => {
    let y = floor;
    column.cells.slice(0, -1).forEach((cell) => {
      y += cell.height;
      add('shelf', x, y, column.width, f.shelfThickness);
      y += f.shelfThickness;
    });
    x += column.width;
    if (index < f.columns.length - 1) add('divider', x, floor, f.dividerThickness, innerHeight(f));
    x += f.dividerThickness;
  });
  return rects;
}

/** The ways a piece breaks the rule at the top of this file. Empty means it is fine. */
export function checkPiece(f: Furniture): string[] {
  if (f.columns.length === 0) return [];
  const problems: string[] = [];
  if (Math.abs(widthNeeded(f) - innerWidth(f)) > TOLERANCE) {
    problems.push(`${f.name}: the columns and dividers do not add up to the inside width.`);
  }
  for (const column of f.columns) {
    if (column.cells.length === 0) problems.push(`${f.name}: a column has no compartments.`);
    if (Math.abs(heightNeeded(f, column) - innerHeight(f)) > TOLERANCE) {
      problems.push(`${f.name}: the compartments and shelves do not add up to the inside height.`);
    }
  }
  if (innerDepth(f) <= 0) problems.push(`${f.name}: the back panel is thicker than the piece is deep.`);
  return problems;
}

export function findCell(f: Furniture, cellId: Id): { column: Column; index: number } | null {
  for (const column of f.columns) {
    const index = column.cells.findIndex((cell) => cell.id === cellId);
    if (index >= 0) return { column, index };
  }
  return null;
}

/* ---- Restoring the rule ---- */

/*
 * Heights down a column and widths across a piece behave the same way, so the
 * two helpers below work on a plain row of sizes. `nameOf` names an entry for
 * the message when a change is refused.
 */

function tooSmall(name: string): Refusal {
  return new Refusal(`There is not enough room: ${name} would be left with less than ${toDisplay(MIN_CLEAR)}.`);
}

/** Adds `extra` to the largest size in the row (or takes it away, if negative). */
function giveToLargest(sizes: number[], extra: number, nameOf: (index: number) => string): number[] {
  if (Math.abs(extra) < 1e-9) return sizes;
  const largest = sizes.indexOf(Math.max(...sizes));
  if (sizes[largest] + extra < MIN_CLEAR) throw tooSmall(nameOf(largest));
  return sizes.map((size, index) => (index === largest ? size + extra : size));
}

/** Sets one size in the row; the next entry (or the previous one, at the end) gives or takes the difference. */
function tradeWithNeighbour(sizes: number[], index: number, value: number, nameOf: (index: number) => string): number[] {
  const neighbour = index + 1 < sizes.length ? index + 1 : index - 1;
  if (neighbour < 0) {
    throw new Refusal(
      `${nameOf(index)} has nothing beside it to give or take the difference. ` +
        'Change the size of the whole piece, or switch to "keep the other compartments".',
    );
  }
  const left = sizes[neighbour] - (value - sizes[index]);
  if (left < MIN_CLEAR) throw tooSmall(nameOf(neighbour));
  return sizes.map((size, i) => (i === index ? value : i === neighbour ? left : size));
}

const cellName = (column: Column) => (index: number) => `"${column.cells[index].name}"`;
const columnName = (index: number) => `column ${index + 1}`;

const withHeights = (column: Column, heights: number[]): Column => ({
  ...column,
  cells: column.cells.map((cell, index) => ({ ...cell, height: heights[index] })),
});

const withWidths = (f: Furniture, widths: number[]): Furniture => ({
  ...f,
  columns: f.columns.map((column, index) => ({ ...column, width: widths[index] })),
});

const heightsOf = (column: Column) => column.cells.map((cell) => cell.height);
const widthsOf = (f: Furniture) => f.columns.map((column) => column.width);

function fitHeights(f: Furniture, mode: ResizeMode): Furniture {
  let piece = f;
  if (mode === 'keepCompartments') {
    const tallest = Math.max(...f.columns.map((column) => heightNeeded(f, column)));
    piece = { ...f, outer: { ...f.outer, h: tallest + f.baseHeight + f.panel.top + f.panel.bottom } };
  }
  const available = innerHeight(piece);
  return {
    ...piece,
    columns: piece.columns.map((column) =>
      withHeights(column, giveToLargest(heightsOf(column), available - heightNeeded(piece, column), cellName(column))),
    ),
  };
}

function fitWidths(f: Furniture, mode: ResizeMode): Furniture {
  if (mode === 'keepCompartments') {
    return { ...f, outer: { ...f.outer, w: widthNeeded(f) + 2 * f.panel.side } };
  }
  return withWidths(f, giveToLargest(widthsOf(f), innerWidth(f) - widthNeeded(f), columnName));
}

/** Makes the piece add up again after a change, or refuses. */
function rebalance(f: Furniture, mode: ResizeMode): Furniture {
  if (f.columns.length === 0) return f;
  const piece = fitWidths(fitHeights(f, mode), mode);
  if (innerDepth(piece) < MIN_CLEAR) {
    throw new Refusal(`There is not enough depth for that. The inside would be less than ${toDisplay(MIN_CLEAR)} deep.`);
  }
  return piece;
}

/* ---- Edits. Each returns a new piece, or throws a Refusal. ---- */

/** The panels that sit between an inside measurement and the outside one. */
function panelsAlong(f: Furniture, dimension: Dimension): number {
  if (dimension === 'w') return 2 * f.panel.side;
  if (dimension === 'h') return f.baseHeight + f.panel.top + f.panel.bottom;
  return f.panel.back;
}

/** Sets the overall width, height or depth, typed as either an outside or an inside measurement. */
export function resizePiece(
  f: Furniture,
  dimension: Dimension,
  value: number,
  measuredFrom: 'outside' | 'inside',
): Furniture {
  const outerValue = measuredFrom === 'inside' ? value + panelsAlong(f, dimension) : value;
  if (!(outerValue > 0)) throw new Refusal('A size must be more than zero.');
  const resized = { ...f, outer: { ...f.outer, [dimension]: outerValue }, estimated: false };
  return rebalance(resized, 'keepOutside');
}

/** Sets one compartment's clear height. */
export function resizeCompartment(f: Furniture, cellId: Id, newHeight: number, mode: ResizeMode): Furniture {
  const found = findCell(f, cellId);
  if (!found) throw new Refusal('That compartment no longer exists.');
  const { column, index } = found;
  if (newHeight < MIN_CLEAR) throw tooSmall(cellName(column)(index));

  // In 'keepOutside' mode the shelf between this compartment and its neighbour moves.
  const heights =
    mode === 'keepOutside'
      ? tradeWithNeighbour(heightsOf(column), index, newHeight, cellName(column))
      : heightsOf(column).map((height, i) => (i === index ? newHeight : height));
  const columns = f.columns.map((c) => (c === column ? withHeights(c, heights) : c));
  return rebalance({ ...f, columns, estimated: false }, mode);
}

/** Sets one column's clear width. Only pieces with several columns have this. */
export function resizeColumn(f: Furniture, columnId: Id, newWidth: number, mode: ResizeMode): Furniture {
  const index = f.columns.findIndex((column) => column.id === columnId);
  if (index < 0) throw new Refusal('That column no longer exists.');
  if (newWidth < MIN_CLEAR) throw tooSmall(columnName(index));

  const widths =
    mode === 'keepOutside'
      ? tradeWithNeighbour(widthsOf(f), index, newWidth, columnName)
      : widthsOf(f).map((width, i) => (i === index ? newWidth : width));
  return rebalance({ ...withWidths(f, widths), estimated: false }, mode);
}

/** Sets a panel, shelf or divider thickness, or the height of the base. */
export function setThickness(f: Furniture, which: ThicknessName, value: number, mode: ResizeMode): Furniture {
  if (!(value >= 0)) throw new Refusal('A thickness cannot be less than zero.');
  let piece: Furniture;
  if (which === 'shelf') piece = { ...f, shelfThickness: value };
  else if (which === 'divider') piece = { ...f, dividerThickness: value };
  else if (which === 'base') piece = { ...f, baseHeight: value };
  else piece = { ...f, panel: { ...f.panel, [which]: value } };
  return rebalance({ ...piece, estimated: false }, mode);
}

/** Changes a compartment's name, rail position or drawer wall. */
export function updateCell(
  f: Furniture,
  cellId: Id,
  changes: Partial<Pick<Cell, 'name' | 'railDrop' | 'drawerWall'>>,
): Furniture {
  const found = findCell(f, cellId);
  if (!found) throw new Refusal('That compartment no longer exists.');
  const cell = { ...found.column.cells[found.index], ...changes };
  if (cell.name.trim() === '') throw new Refusal('A compartment needs a name.');
  if (!(cell.railDrop >= 0) || !(cell.drawerWall >= 0)) throw new Refusal('That cannot be less than zero.');
  if (cell.kind === 'hanging' && hangLength(cell) < MIN_CLEAR) {
    throw new Refusal('The rail would be too low for anything to hang from it.');
  }
  if (cell.kind === 'drawer' && found.column.width - 2 * cell.drawerWall < MIN_CLEAR) {
    throw new Refusal('The drawer walls would leave no room inside.');
  }
  const columns = f.columns.map((column) =>
    column === found.column
      ? { ...column, cells: column.cells.map((c, i) => (i === found.index ? cell : c)) }
      : column,
  );
  return { ...f, columns };
}

export function renamePiece(f: Furniture, name: string): Furniture {
  if (name.trim() === '') throw new Refusal('A piece of furniture needs a name.');
  return { ...f, name: name.trim() };
}
