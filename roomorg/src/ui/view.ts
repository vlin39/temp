/*
 * Drawing in depth.
 *
 * Furniture is drawn at an angle with an oblique projection: a front face
 * keeps its true shape, and anything further back is shifted right and up.
 * It looks like standing a little to the right of the piece and above it, so
 * you see the top and right side of everything.
 *
 * Because every front face stays a plain rectangle, sizes read true and the
 * shapes stay easy to click and drag.
 *
 * Positions are in centimetres: x runs right, y runs up, and z runs back from
 * the front. A negative z is in front (the floor, an open drawer or door).
 */

export interface View {
  /** Pixels per centimetre. */
  scale: number;
  /** Where the point (0, 0, 0) is, in pixels. */
  originX: number;
  originY: number;
  /** How far one centimetre of depth moves a point right and up, in centimetres. Both 0 for a flat front view. */
  kx: number;
  ky: number;
}

/** The angle the furniture is seen from. */
export const ANGLED = { kx: 0.32, ky: 0.42 };

/** Straight from the front, with no depth. */
export const FLAT = { kx: 0, ky: 0 };

export type Corner = [x: number, y: number, z: number];

export interface Cuboid {
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
  d: number;
}

/** Where a point in space lands on the screen. */
export function toPixels(view: View, x: number, y: number, z = 0): { x: number; y: number } {
  return {
    x: view.originX + (x + z * view.kx) * view.scale,
    y: view.originY - (y + z * view.ky) * view.scale,
  };
}

/** A flat shape in space, as the `points` of an SVG polygon. */
export function polygon(view: View, corners: Corner[]): string {
  return corners
    .map(([x, y, z]) => {
      const point = toPixels(view, x, y, z);
      return `${point.x},${point.y}`;
    })
    .join(' ');
}

/** The same view, measuring from a different point: the corner of a compartment, say. */
export function movedTo(view: View, x: number, y: number, z = 0): View {
  const origin = toPixels(view, x, y, z);
  return { ...view, originX: origin.x, originY: origin.y };
}

/** The front face of a cuboid, as an SVG rectangle. */
export function frontRect(view: View, box: Omit<Cuboid, 'd'>): { x: number; y: number; width: number; height: number } {
  const corner = toPixels(view, box.x, box.y + box.h, box.z);
  return { x: corner.x, y: corner.y, width: box.w * view.scale, height: box.h * view.scale };
}

/** The faces of a cuboid, each as a list of corners. `left` is seen from inside, as the wall of a compartment. */
export function faces(box: Cuboid): Record<'top' | 'bottom' | 'left' | 'right' | 'back', Corner[]> {
  const { x, y, z, w, h, d } = box;
  return {
    top: [[x, y + h, z], [x + w, y + h, z], [x + w, y + h, z + d], [x, y + h, z + d]],
    bottom: [[x, y, z], [x + w, y, z], [x + w, y, z + d], [x, y, z + d]],
    left: [[x, y, z], [x, y, z + d], [x, y + h, z + d], [x, y + h, z]],
    right: [[x + w, y, z], [x + w, y, z + d], [x + w, y + h, z + d], [x + w, y + h, z]],
    back: [[x, y, z + d], [x + w, y, z + d], [x + w, y + h, z + d], [x, y + h, z + d]],
  };
}

/** The smallest and largest screen positions of some corners, in centimetres, before scaling. */
export function extent(corners: Corner[], depth: { kx: number; ky: number }) {
  const across = corners.map(([x, , z]) => x + z * depth.kx);
  const up = corners.map(([, y, z]) => y + z * depth.ky);
  return { left: Math.min(...across), right: Math.max(...across), bottom: Math.min(...up), top: Math.max(...up) };
}
