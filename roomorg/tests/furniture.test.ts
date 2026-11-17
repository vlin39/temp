import { describe, expect, test } from 'vitest';
import {
  checkPiece,
  clearSpaces,
  innerDepth,
  innerHeight,
  innerWidth,
  panels,
  resizeColumn,
  resizeCompartment,
  resizePiece,
  setThickness,
  updateCell,
  usableSpace,
} from '../src/model/furniture';
import { MIN_CLEAR } from '../src/model/presets';
import { Refusal } from '../src/model/refusal';
import { createStarterHome } from '../src/model/roomData';
import type { Furniture } from '../src/model/types';
import { CM_PER_INCH } from '../src/model/units';

const home = createStarterHome();
const piece = (id: string): Furniture => home.furniture.find((f) => f.id === id)!;
const inches = (cm: number) => cm / CM_PER_INCH;

describe('starter furniture', () => {
  test('every piece adds up', () => {
    for (const f of home.furniture) expect(checkPiece(f), f.name).toEqual([]);
  });

  test('Shelves C gives two clear spaces of 38.5 x 15.5 x 11.9 in', () => {
    const spaces = [...clearSpaces(piece('shelves-c1')).values()];
    expect(spaces).toHaveLength(2);
    for (const space of spaces) {
      expect(inches(space.w)).toBeCloseTo(38.5);
      expect(inches(space.d)).toBeCloseTo(15.5);
      expect(inches(space.h)).toBeCloseTo(11.9);
    }
  });

  test('compartments stack upward with a shelf thickness between them', () => {
    const f = piece('shelves-c1');
    const [lower, upper] = [...clearSpaces(f).values()];
    expect(lower.y).toBeCloseTo(f.panel.bottom);
    expect(upper.y).toBeCloseTo(lower.y + lower.h + f.shelfThickness);
    expect(upper.y + upper.h + f.panel.top).toBeCloseTo(f.outer.h);
  });

  test('columns sit side by side with a divider between them', () => {
    const f = piece('sink-unit');
    const spaces = clearSpaces(f);
    const left = spaces.get('sink-unit-c1-1')!;
    const middle = spaces.get('sink-unit-c2-1')!;
    expect(left.x).toBeCloseTo(f.panel.side);
    expect(middle.x).toBeCloseTo(left.x + left.w + f.dividerThickness);
  });

  test('panels cover the outline: two sides, top, bottom, base, one shelf per gap', () => {
    const f = piece('closet-b1');
    const kinds = panels(f).map((p) => p.kind);
    expect(kinds.filter((k) => k === 'side')).toHaveLength(2);
    expect(kinds.filter((k) => k === 'shelf')).toHaveLength(4);
    expect(kinds).toContain('base');
    expect(panels(piece('bed'))).toEqual([]);
  });

  test('a drawer loses its own walls', () => {
    const f = piece('sink-unit');
    const drawer = f.columns[1].cells[1];
    const space = clearSpaces(f).get(drawer.id)!;
    const usable = usableSpace(drawer, space);
    expect(usable.w).toBeCloseTo(space.w - 2 * drawer.drawerWall);
    expect(usable.h).toBeCloseTo(space.h - drawer.drawerWall);
  });
});

describe('resizing a compartment', () => {
  const f = piece('closet-b1');
  const [hanging, shelf] = f.columns[0].cells;

  test('"keep outside" moves the shelf: the neighbour absorbs the change', () => {
    const next = resizeCompartment(f, shelf.id, shelf.height + 4, 'keepOutside');
    const cells = next.columns[0].cells;
    expect(next.outer).toEqual(f.outer);
    expect(cells[1].height).toBeCloseTo(shelf.height + 4);
    expect(cells[2].height).toBeCloseTo(f.columns[0].cells[2].height - 4);
    expect(checkPiece(next)).toEqual([]);
    expect(next.estimated).toBe(false);
  });

  test('"keep compartments" changes the outside by the difference', () => {
    const next = resizeCompartment(f, shelf.id, shelf.height + 4, 'keepCompartments');
    expect(next.outer.h).toBeCloseTo(f.outer.h + 4);
    expect(next.columns[0].cells[2].height).toBeCloseTo(f.columns[0].cells[2].height);
    expect(checkPiece(next)).toEqual([]);
  });

  test('typing measured heights one after another does not disturb the others', () => {
    let next = f;
    for (const cell of f.columns[0].cells.slice(1)) next = resizeCompartment(next, cell.id, 25, 'keepCompartments');
    expect(next.columns[0].cells.slice(1).map((cell) => cell.height)).toEqual([25, 25, 25, 25]);
    expect(next.columns[0].cells[0].height).toBeCloseTo(hanging.height);
  });

  test('squeezing a neighbour below the minimum is refused', () => {
    const tooTall = shelf.height + f.columns[0].cells[2].height - MIN_CLEAR + 1;
    expect(() => resizeCompartment(f, shelf.id, tooTall, 'keepOutside')).toThrow(Refusal);
    expect(() => resizeCompartment(f, shelf.id, MIN_CLEAR - 1, 'keepOutside')).toThrow(Refusal);
  });
});

describe('resizing a piece', () => {
  const f = piece('closet-b1');

  test('an outside height change is taken up by the tallest compartment', () => {
    const next = resizePiece(f, 'h', f.outer.h + 10, 'outside');
    expect(next.columns[0].cells[0].height).toBeCloseTo(f.columns[0].cells[0].height + 10);
    expect(checkPiece(next)).toEqual([]);
  });

  test('an inside measurement sets the outside through the panels', () => {
    const wide = resizePiece(f, 'w', 90, 'inside');
    expect(innerWidth(wide)).toBeCloseTo(90);
    expect(wide.outer.w).toBeCloseTo(90 + 2 * f.panel.side);
    const deep = resizePiece(f, 'd', 40, 'inside');
    expect(innerDepth(deep)).toBeCloseTo(40);
    expect(checkPiece(wide)).toEqual([]);
  });

  test('a piece too small for its compartments is refused', () => {
    expect(() => resizePiece(f, 'h', 60, 'outside')).toThrow(Refusal);
    expect(() => resizePiece(f, 'd', 2, 'outside')).toThrow(Refusal);
  });

  test('furniture with no storage can still be resized', () => {
    expect(resizePiece(piece('bed'), 'w', 190, 'outside').outer.w).toBe(190);
  });
});

describe('thicknesses and columns', () => {
  test('a thicker shelf takes room from the tallest compartment, or grows the piece', () => {
    const f = piece('closet-b1');
    const kept = setThickness(f, 'shelf', f.shelfThickness + 1, 'keepOutside');
    expect(kept.outer.h).toBeCloseTo(f.outer.h);
    expect(kept.columns[0].cells[0].height).toBeCloseTo(f.columns[0].cells[0].height - 4);

    const grown = setThickness(f, 'shelf', f.shelfThickness + 1, 'keepCompartments');
    expect(grown.outer.h).toBeCloseTo(f.outer.h + 4);
    expect(innerHeight(grown)).toBeCloseTo(innerHeight(f) + 4);
    expect(checkPiece(kept)).toEqual([]);
    expect(checkPiece(grown)).toEqual([]);
  });

  test('a wider column takes width from the column beside it', () => {
    const f = piece('sink-unit');
    const next = resizeColumn(f, f.columns[0].id, f.columns[0].width + 5, 'keepOutside');
    expect(next.columns[1].width).toBeCloseTo(f.columns[1].width - 5);
    expect(next.outer.w).toBeCloseTo(f.outer.w);
    expect(checkPiece(next)).toEqual([]);
  });

  test('a rail cannot be lowered until nothing can hang from it', () => {
    const f = piece('closet-b1');
    const hanging = f.columns[0].cells[0];
    expect(updateCell(f, hanging.id, { railDrop: 10 }).columns[0].cells[0].railDrop).toBe(10);
    expect(() => updateCell(f, hanging.id, { railDrop: hanging.height })).toThrow(Refusal);
    expect(() => updateCell(f, hanging.id, { name: ' ' })).toThrow(Refusal);
  });
});
