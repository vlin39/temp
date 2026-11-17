/*
 * Lengths are stored in centimetres and shown in inches.
 * (A switch between inches and centimetres is node D1 in the skill tree.)
 */

export const CM_PER_INCH = 2.54;

function trimmed(value: number, decimals: number): string {
  return String(Number(value.toFixed(decimals)));
}

/** The number to put in an input box, in inches: '15.5'. */
export function formatLength(cm: number): string {
  return trimmed(cm / CM_PER_INCH, 2);
}

/** A length for reading: '15.5 in'. */
export function toDisplay(cm: number): string {
  return `${trimmed(cm / CM_PER_INCH, 1)} in`;
}

/** A size for reading: '38.5 x 11.9 in'. */
export function toDisplaySize(...lengths: number[]): string {
  return `${lengths.map((cm) => trimmed(cm / CM_PER_INCH, 1)).join(' × ')} in`;
}

/**
 * Reads what someone typed and returns centimetres, or null if it is not a
 * length above zero. Accepts '15.5', '15.5 in', '15.5"', '15 1/2', '1/2' and '40 cm'.
 * A bare number is inches.
 */
export function parseLength(text: string): number | null {
  const match = text
    .trim()
    .toLowerCase()
    .match(/^(\d+(?:\.\d+)?|\.\d+)?(?:\s*(\d+)\s*\/\s*(\d+))?\s*(in|inch|inches|"|cm)?$/);
  if (!match) return null;
  const [, whole, top, bottom, unit] = match;
  if (whole === undefined && top === undefined) return null;
  if (bottom !== undefined && Number(bottom) === 0) return null;

  const value = Number(whole ?? 0) + (top !== undefined ? Number(top) / Number(bottom) : 0);
  if (!(value > 0)) return null;
  return unit === 'cm' ? value : value * CM_PER_INCH;
}
