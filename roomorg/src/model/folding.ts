/*
 * Shape and fold.
 *
 * A garment laid flat is one to three connected rectangles: a body, and
 * perhaps a pair of sleeves or a pair of legs. In storage it is a single box.
 * Folding divides its footprint and raises its height so that the amount of
 * fabric stays the same. Sleeves and legs add fabric, so they make a folded
 * item thicker without changing its footprint.
 */

import { FOLD_DIVISORS, MIN_RAIL_SPACE } from './presets';
import type { Item, Part, PartRole, Size3 } from './types';

/** Just enough of an item to work out its shape. */
export type Shape = Pick<Item, 'parts' | 'fold'>;

/** One rectangle of the laid-flat drawing. x runs right, y runs down from the top. */
export interface OutlineRect {
  role: PartRole;
  x: number;
  y: number;
  w: number;
  h: number;
}

const EMPTY_PART: Part = { role: 'body', w: 0, l: 0, t: 0, count: 1 };

function part(shape: Shape, role: PartRole): Part | undefined {
  return shape.parts.find((candidate) => candidate.role === role);
}

/** The amount of fabric: every part's width x length x thickness. */
export function fabricVolume(shape: Shape): number {
  return shape.parts.reduce((total, p) => total + p.w * p.l * p.t * p.count, 0);
}

/**
 * The laid-flat drawing: the body, a sleeve down each side from the shoulder,
 * and a leg under each half of the hips.
 */
export function flatOutline(shape: Shape): { rects: OutlineRect[]; w: number; l: number } {
  const body = part(shape, 'body') ?? EMPTY_PART;
  const sleeve = part(shape, 'sleeve');
  const leg = part(shape, 'leg');
  const bodyX = sleeve ? sleeve.w : 0;

  const rects: OutlineRect[] = [{ role: 'body', x: bodyX, y: 0, w: body.w, h: body.l }];
  if (sleeve) {
    rects.push({ role: 'sleeve', x: 0, y: 0, w: sleeve.w, h: sleeve.l });
    if (sleeve.count > 1) rects.push({ role: 'sleeve', x: bodyX + body.w, y: 0, w: sleeve.w, h: sleeve.l });
  }
  if (leg) {
    rects.push({ role: 'leg', x: bodyX, y: body.l, w: leg.w, h: leg.l });
    if (leg.count > 1) rects.push({ role: 'leg', x: bodyX + body.w - leg.w, y: body.l, w: leg.w, h: leg.l });
  }

  return {
    rects,
    w: Math.max(...rects.map((r) => r.x + r.w)),
    l: Math.max(...rects.map((r) => r.y + r.h)),
  };
}

/** The single box the item takes up in storage, for its current fold. */
export function storedSize(shape: Shape): Size3 {
  const body = part(shape, 'body') ?? EMPTY_PART;
  const leg = part(shape, 'leg');
  const coreW = body.w; // sleeves fold in over the body
  const coreL = body.l + (leg ? leg.l : 0); // legs lie together and continue downward
  const volume = fabricVolume(shape);
  if (!(coreW > 0) || !(coreL > 0)) return { w: 0, h: 0, d: 0 };

  if (shape.fold === 'flat') {
    const outline = flatOutline(shape);
    return { w: outline.w, d: outline.l, h: Math.max(...shape.parts.map((p) => p.t)) };
  }

  if (shape.fold === 'rolled') {
    const diameter = Math.sqrt((4 * volume) / (Math.PI * coreW));
    return { w: coreW, d: diameter, h: diameter };
  }

  if (shape.fold === 'hung') {
    return {
      w: Math.max(MIN_RAIL_SPACE, volume / (coreW * coreL)), // space along the rail
      d: coreW, // shoulder width, front to back
      h: coreL, // how far it hangs down
    };
  }

  const { across, down } = FOLD_DIVISORS[shape.fold] ?? { across: 1, down: 1 };
  const w = coreW / across;
  const d = coreL / down;
  return { w, d, h: volume / (w * d) }; // same fabric on a smaller footprint, so taller
}
