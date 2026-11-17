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
import type { Box, Cell, Column, Furniture, Id } from './types';
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

function tooSmall(name: string): Refusal {
  return new Refusal(`There is not enough room: "${name}" would be left with less than ${toDisplay(MIN_CLEAR)}.`);
}

/** Gives a column's tallest compartment `extra` more height (or less, if negative). */
function growTallestCell(column: Column, extra: number): Column {
  if (Math.abs(extra) < 1e-9) return column;
  const tallest = column.cells.reduce((a, b) => (b.height > a.height ? b : a));
  const height = tallest.height + extra;
  if (height < MIN_CLEAR) throw tooSmall(tallest.name);
  return { ...column, cells: column.cells.map((cell) => (cell === tallest ? { ...cell, height } : cell)) };
}

function fitHeights(f: Furniture, mode: ResizeMode): Furniture {
  let piece = f;
  if (mode === 'keepCompartments') {
    const tallest = Math.max(...f.columns.map((column) => heightNeeded(f, column)));
    piece = { ...f, outer: { ...f.outer, h: tallest + f.baseHeight + f.panel.top + f.panel.bottom } };
  }
  const available = innerHeight(piece);
  return {
    ...piece,
    columns: piece.columns.map((column) => growTallestCell(column, available - heightNeeded(piece, column))),
  };
}

function fitWidths(f: Furniture, mode: ResizeMode): Furniture {
  if (mode === 'keepCompartments') {
    return { ...f, outer: { ...f.outer, w: widthNeeded(f) + 2 * f.panel.side } };
  }
  const extra = innerWidth(f) - widthNeeded(f);
  if (Math.abs(extra) < 1e-9) return f;
  const widest = f.columns.reduce((a, b) => (b.width > a.width ? b : a));
  const width = widest.width + extra;
  if (width < MIN_CLEAR) throw new Refusal(`There is not enough width for that. A column would be left with less than ${toDisplay(MIN_CLEAR)}.`);
  return { ...f, columns: f.columns.map((column) => (column === widest ? { ...column, width } : column)) };
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
  const cell = column.cells[index];
  if (newHeight < MIN_CLEAR) throw tooSmall(cell.name);

  let cells = column.cells.map((c) => (c === cell ? { ...c, height: newHeight } : c));
  if (mode === 'keepOutside') {
    // The shelf between this compartment and its neighbour moves.
    const neighbour = column.cells[index + 1] ?? column.cells[index - 1];
    if (!neighbour) {
      throw new Refusal(
        'This is the only compartment in its column, so its height follows the piece. ' +
          'Change the piece\'s height, or switch to "keep the other compartments".',
      );
    }
    const neighbourHeight = neighbour.height - (newHeight - cell.height);
    if (neighbourHeight < MIN_CLEAR) throw tooSmall(neighbour.name);
    cells = cells.map((c) => (c === neighbour ? { ...c, height: neighbourHeight } : c));
  }

  const columns = f.columns.map((c) => (c === column ? { ...c, cells } : c));
  return rebalance({ ...f, columns, estimated: false }, mode);
}

/** Sets one column's clear width. Only pieces with several columns have this. */
export function resizeColumn(f: Furniture, columnId: Id, newWidth: number, mode: ResizeMode): Furniture {
  const index = f.columns.findIndex((column) => column.id === columnId);
  if (index < 0) throw new Refusal('That column no longer exists.');
  if (newWidth < MIN_CLEAR) throw new Refusal(`A column needs at least ${toDisplay(MIN_CLEAR)}.`);

  const column = f.columns[index];
  let columns = f.columns.map((c) => (c === column ? { ...c, width: newWidth } : c));
  if (mode === 'keepOutside') {
    const neighbour = f.columns[index + 1] ?? f.columns[index - 1];
    if (!neighbour) {
      throw new Refusal(
        'This piece has one column, so its width follows the piece. ' +
          'Change the piece\'s width, or switch to "keep the other compartments".',
      );
    }
    const neighbourWidth = neighbour.width - (newWidth - column.width);
    if (neighbourWidth < MIN_CLEAR) {
      throw new Refusal(`There is not enough width: the next column would be left with less than ${toDisplay(MIN_CLEAR)}.`);
    }
    columns = columns.map((c) => (c === neighbour ? { ...c, width: neighbourWidth } : c));
  }
  return rebalance({ ...f, columns, estimated: false }, mode);
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
