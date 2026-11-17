/*
 * The starter home: the suite from the sketch, its furniture and compartments,
 * and the boxes named in the organization table.
 *
 * Which pieces exist, where they stand and which way they face are fixed here
 * for now. Their measurements can be edited in the app.
 *
 * Only the shelving units (C) use real measurements. Every other size is a
 * placeholder, marked `estimated`, until it is measured.
 */

import { ORGANIZER_PRESETS } from './presets';
import type { Cell, CellKind, Column, Facing, Furniture, Home, Organizer, Room, Size3 } from './types';
import { CM_PER_INCH } from './units';

const inches = (n: number) => n * CM_PER_INCH;

/** Board thickness assumed for panels and shelves until measured. */
const BOARD = 1.8;

interface CellSpec {
  name: string;
  kind?: CellKind;
  /** Clear height. Compartments without one share what is left equally. */
  height?: number;
}

interface PieceSpec {
  id: string;
  name: string;
  roomId: string;
  x: number;
  y: number;
  facing: Facing;
  outer: Size3;
  back?: number;
  base?: number;
  estimated?: boolean;
  /** Each column lists its compartments bottom to top. Omit for furniture with no storage. */
  columns?: { width?: number; cells: CellSpec[] }[];
}

/** Builds a piece whose compartments add up exactly to its inside size. */
function makePiece(spec: PieceSpec): Furniture {
  const specColumns = spec.columns ?? [];
  const base = spec.base ?? 0;
  const innerW = spec.outer.w - 2 * BOARD;
  const innerH = spec.outer.h - base - 2 * BOARD;

  const fixedWidth = specColumns.reduce((total, column) => total + (column.width ?? 0), 0);
  const freeColumns = specColumns.filter((column) => column.width === undefined).length;
  const spareWidth = innerW - fixedWidth - (specColumns.length - 1) * BOARD;

  const columns: Column[] = specColumns.map((column, columnIndex) => {
    const fixedHeight = column.cells.reduce((total, cell) => total + (cell.height ?? 0), 0);
    const freeCells = column.cells.filter((cell) => cell.height === undefined).length;
    const spareHeight = innerH - fixedHeight - (column.cells.length - 1) * BOARD;

    const cells: Cell[] = column.cells.map((cell, cellIndex) => {
      const kind = cell.kind ?? 'shelf';
      return {
        id: `${spec.id}-c${columnIndex + 1}-${cellIndex + 1}`,
        name: cell.name,
        kind,
        height: cell.height ?? spareHeight / freeCells,
        railDrop: kind === 'hanging' ? 6 : 0,
        drawerWall: kind === 'drawer' ? 1.2 : 0,
      };
    });
    return { id: `${spec.id}-c${columnIndex + 1}`, width: column.width ?? spareWidth / freeColumns, cells };
  });

  return {
    id: spec.id,
    name: spec.name,
    roomId: spec.roomId,
    x: spec.x,
    y: spec.y,
    facing: spec.facing,
    outer: spec.outer,
    panel: { side: BOARD, top: BOARD, bottom: BOARD, back: spec.back ?? 0.5 },
    shelfThickness: BOARD,
    dividerThickness: BOARD,
    baseHeight: base,
    columns,
    estimated: spec.estimated ?? true,
  };
}

/** A closet of style B: a hanging space with four shelves above it. Names run bottom to top. */
function closetB(id: string, name: string, x: number, y: number, facing: Facing, names: string[]): Furniture {
  const [hanging, ...shelves] = names;
  return makePiece({
    id,
    name,
    roomId: 'bedroom',
    x,
    y,
    facing,
    outer: { w: 100, h: 220, d: 58 },
    base: 6,
    columns: [{ cells: [{ name: hanging, kind: 'hanging', height: 110 }, ...shelves.map((shelf) => ({ name: shelf }))] }],
  });
}

/** The shelving unit you measured: inside 38.5 in long, 15.5 in deep, two levels of 11.9 in. */
function shelvesC(id: string, name: string, x: number, y: number, facing: Facing): Furniture {
  const level = inches(11.9);
  const back = 0.3;
  return makePiece({
    id,
    name,
    roomId: 'bedroom',
    x,
    y,
    facing,
    outer: { w: inches(38.5) + 2 * BOARD, h: 2 * level + 3 * BOARD, d: inches(15.5) + back },
    back,
    estimated: false,
    columns: [{ cells: [{ name: 'Lower shelf', height: level }, { name: 'Upper shelf', height: level }] }],
  });
}

function noStorage(id: string, name: string, roomId: string, x: number, y: number, facing: Facing, outer: Size3): Furniture {
  return makePiece({ id, name, roomId, x, y, facing, outer });
}

/** Boxes on one shelf, left to right, as [name, preset id] pairs. */
function boxes(cellId: string, entries: [string, string][]): Organizer[] {
  return entries.map(([name, presetId], order) => {
    const preset = ORGANIZER_PRESETS.find((candidate) => candidate.id === presetId)!;
    return {
      id: `${cellId}-box${order + 1}`,
      name,
      parentId: cellId,
      order,
      outer: { ...preset.outer },
      wall: preset.wall,
    };
  });
}

export function createStarterHome(): Home {
  const rooms: Room[] = [
    { id: 'bedroom', name: 'Bedroom', x: 0, y: 0, w: 420, d: 480 },
    { id: 'sink-area', name: 'Sink area', x: 420, y: 0, w: 200, d: 260 },
    { id: 'bathroom', name: 'Bathroom', x: 420, y: 260, w: 200, d: 220 },
  ];

  const furniture: Furniture[] = [
    // Bedroom. Closets B1 and B2 are the pair in the middle; B3 stands beside Closet A.
    closetB('closet-b1', 'Closet B1', 362, 240, 'west', [
      'Large jackets, formal wear, costumes',
      'Tights, corsets, scarves',
      'To do',
      'Hand bags',
      'Hats',
    ]),
    closetB('closet-b2', 'Closet B2', 362, 340, 'west', [
      'Light dresses, jumpsuits, jackets, coats',
      'Socks, underwear, bras',
      'Bottoms',
      'Tops',
      'Swim and mountain inner gear',
    ]),
    closetB('closet-b3', 'Closet B3', 140, 422, 'north', [
      'Shirts, blouses, pants, short jackets',
      'Pajamas, belts, tank tops',
      'To sort (lower)',
      'To sort (upper)',
      'Pillows and duvet',
    ]),
    makePiece({
      id: 'closet-a',
      name: 'Closet A',
      roomId: 'bedroom',
      x: 60,
      y: 422,
      facing: 'north',
      outer: { w: 80, h: 220, d: 58 },
      base: 6,
      columns: [
        {
          cells: [
            { name: 'Shoes (lower)', height: 25 },
            { name: 'Shoes (upper)', height: 25 },
            { name: 'Sheets' },
            { name: 'Towels' },
            { name: 'To do' },
            { name: 'Mountain outer gear' },
          ],
        },
      ],
    }),
    shelvesC('shelves-c1', 'Shelves C1', 110, 0, 'south'),
    shelvesC('shelves-c2', 'Shelves C2', 0, 50, 'east'),
    noStorage('desk', 'Desk', 'bedroom', 50, 60, 'south', { w: 110, h: 75, d: 55 }),
    noStorage('bed', 'Bed', 'bedroom', 0, 165, 'east', { w: 200, h: 50, d: 100 }),
    noStorage('tiny-table', 'Tiny table', 'bedroom', 125, 150, 'south', { w: 50, h: 45, d: 50 }),
    noStorage('tiny-chair', 'Tiny chair', 'bedroom', 135, 215, 'south', { w: 30, h: 50, d: 30 }),

    // Sink area.
    makePiece({
      id: 'shelves-d',
      name: 'Shelves D',
      roomId: 'sink-area',
      x: 120,
      y: 0,
      facing: 'south',
      outer: { w: 70, h: 100, d: 35 },
      columns: [{ cells: [{ name: 'Bottom shelf' }, { name: 'Middle shelf' }, { name: 'Top shelf' }] }],
    }),
    makePiece({
      id: 'sink-unit',
      name: 'Sink unit',
      roomId: 'sink-area',
      x: 145,
      y: 80,
      facing: 'west',
      outer: { w: 150, h: 85, d: 55 },
      base: 8,
      columns: [
        { width: 35, cells: [{ name: 'Left cabinet, lower', kind: 'cabinet' }, { name: 'Left cabinet, upper', kind: 'cabinet' }] },
        { cells: [{ name: 'Under the sink', kind: 'cabinet' }, { name: 'Drawer', kind: 'drawer', height: 12 }] },
        { width: 35, cells: [{ name: 'Right cabinet, lower', kind: 'cabinet' }, { name: 'Right cabinet, upper', kind: 'cabinet' }] },
      ],
    }),
    makePiece({
      id: 'wall-cabinet',
      name: 'Wall cabinet',
      roomId: 'sink-area',
      x: 180,
      y: 45,
      facing: 'west',
      outer: { w: 30, h: 75, d: 20 },
      columns: [
        { cells: [{ name: 'Bottom level', kind: 'cabinet' }, { name: 'Middle level', kind: 'cabinet' }, { name: 'Top level', kind: 'cabinet' }] },
      ],
    }),

    // Bathroom fixtures, shown on the floor plan only.
    noStorage('toilet', 'Toilet', 'bathroom', 130, 20, 'west', { w: 45, h: 40, d: 65 }),
    noStorage('bath', 'Bath and shower', 'bathroom', 20, 140, 'north', { w: 160, h: 55, d: 70 }),
  ];

  // The boxes named in the table, on the shelves it lists. (S), (M) and (L) are the box sizes.
  const organizers: Organizer[] = [
    ...boxes('closet-b1-c1-4', [['Hand bags', 'large']]),
    ...boxes('closet-b1-c1-2', [['Tights', 'small'], ['Corsets', 'medium'], ['Scarves', 'large']]),
    ...boxes('closet-b2-c1-5', [['Snorkeling stuff', 'medium'], ['Swimwear', 'small'], ['Coverups', 'small'], ['Mountain inner gear', 'small']]),
    ...boxes('closet-b2-c1-4', [['Wool sweaters', 'large'], ['Turtlenecks', 'small'], ['Thick cotton sweat-sets', 'medium']]),
    ...boxes('closet-b2-c1-3', [['Jeans', 'large'], ['Linen pants', 'medium'], ['Leggings', 'small']]),
    ...boxes('closet-b2-c1-2', [['Socks', 'large'], ['Underwear', 'small'], ['Bras', 'medium']]),
    ...boxes('closet-b3-c1-2', [['Pajamas', 'large'], ['Belts', 'small'], ['Tank tops', 'small']]),
  ];

  return { version: 1, rooms, furniture, organizers, items: [], lastBackupAt: null };
}
