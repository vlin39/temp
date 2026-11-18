import { describe, expect, test } from 'vitest';
import { fabricVolume, flatOutline, storedSize, type Shape } from '../src/model/folding';
import { MIN_RAIL_SPACE, clothingType } from '../src/model/presets';
import type { Fold } from '../src/model/types';

const shape = (typeId: string, fold: Fold): Shape => ({ parts: clothingType(typeId).parts, fold });
const volumeOfBox = (size: { w: number; h: number; d: number }) => size.w * size.h * size.d;

describe('laid flat', () => {
  test('a T-shirt is a body with a sleeve down each side', () => {
    const outline = flatOutline(shape('tshirt', 'flat'));
    expect(outline.rects.map((r) => r.role)).toEqual(['body', 'sleeve', 'sleeve']);
    expect(outline.w).toBe(50 + 2 * 20);
    expect(outline.l).toBe(68);
  });

  test('trousers are hips with a leg under each half', () => {
    const outline = flatOutline(shape('trousers', 'flat'));
    const [hips, left, right] = outline.rects;
    expect(outline.rects.map((r) => r.role)).toEqual(['body', 'leg', 'leg']);
    expect(left.y).toBe(hips.h);
    expect(right.x + right.w).toBe(hips.x + hips.w);
    expect(outline.l).toBe(28 + 76);
  });

  test('the flat box is the outline, as thick as the thickest part', () => {
    expect(storedSize(shape('tshirt', 'flat'))).toEqual({ w: 90, d: 68, h: 0.6 });
  });
});

describe('folded', () => {
  test('the amount of fabric is the same for every fold', () => {
    const fabric = fabricVolume(shape('sweater', 'flat'));
    for (const fold of ['half', 'thirds', 'quarters', 'rolled'] as Fold[]) {
      const size = storedSize(shape('sweater', fold));
      const boxVolume = fold === 'rolled' ? (Math.PI / 4) * size.h * size.d * size.w : volumeOfBox(size);
      expect(boxVolume, fold).toBeCloseTo(fabric);
    }
  });

  test('"half" halves the depth and doubles the height of a sleeveless item', () => {
    const flat = storedSize(shape('skirt', 'flat'));
    const half = storedSize(shape('skirt', 'half'));
    expect(half.w).toBeCloseTo(flat.w);
    expect(half.d).toBeCloseTo(flat.d / 2);
    expect(half.h).toBeCloseTo(flat.h * 2);
  });

  test('sleeves make a folded item taller, not wider', () => {
    const withSleeves = storedSize(shape('longsleeve', 'half'));
    const body = clothingType('longsleeve').parts.filter((p) => p.role === 'body');
    const without = storedSize({ parts: body, fold: 'half' });
    expect(withSleeves.w).toBe(without.w);
    expect(withSleeves.d).toBe(without.d);
    expect(withSleeves.h).toBeGreaterThan(without.h);
  });

  test('quarters has half the footprint width of half', () => {
    expect(storedSize(shape('tshirt', 'quarters')).w).toBeCloseTo(storedSize(shape('tshirt', 'half')).w / 2);
  });
});

describe('hung', () => {
  test('a thin item still takes the minimum rail space', () => {
    const size = storedSize(shape('tank', 'hung'));
    expect(size.w).toBe(MIN_RAIL_SPACE);
    expect(size.h).toBe(62);
    expect(size.d).toBe(42);
  });

  test('a coat takes more rail than a shirt, and trousers hang their full length', () => {
    expect(storedSize(shape('coat', 'hung')).w).toBeGreaterThan(storedSize(shape('shirt', 'hung')).w);
    expect(storedSize(shape('trousers', 'hung')).h).toBe(28 + 76);
  });
});
