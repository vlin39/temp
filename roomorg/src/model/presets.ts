/*
 * Starter sizes and limits. Everything here is a default that can be tuned;
 * nothing else in the app hard-codes a measurement.
 */

import type { Fold, Part, Size3 } from './types';

/** No compartment or column may be squeezed below this. */
export const MIN_CLEAR = 5;

/** The least rail length one hanger takes, however thin the garment. */
export const MIN_RAIL_SPACE = 2.5;

/** From the rail down to the shoulders of a hung garment: the hook and the hanger. */
export const HANGER_GAP = 10;

/** The space left between neighbours on a shelf or on the floor. */
export const THING_GAP = 2;

/** The space left between neighbours inside a box or a bag, where things are packed closer. */
export const PACKED_GAP = 0.5;

/** How high a pile of clothing may grow on the floor before a new pile starts. */
export const FLOOR_PILE_HEIGHT = 45;

/** New items get this loud colour until you choose one, so unfinished ones stand out. */
export const UNFINISHED_COLOUR = '#ff4fa3';

export interface OrganizerPreset {
  id: string;
  name: string;
  kind: 'box' | 'bag';
  outer: Size3;
  wall: number;
}

/**
 * The boxes are sized so a closet shelf about 96 cm wide holds two large, or
 * one large + one medium + one small, with gaps between them.
 * The bags are one kind of thing in three sizes for now.
 */
export const ORGANIZER_PRESETS: OrganizerPreset[] = [
  { id: 'small', name: 'Small box', kind: 'box', outer: { w: 18, h: 20, d: 50 }, wall: 0.3 },
  { id: 'medium', name: 'Medium box', kind: 'box', outer: { w: 28, h: 20, d: 50 }, wall: 0.3 },
  { id: 'large', name: 'Large box', kind: 'box', outer: { w: 46, h: 20, d: 50 }, wall: 0.3 },
  { id: 'pouch', name: 'Pouch', kind: 'bag', outer: { w: 22, h: 14, d: 8 }, wall: 0.2 },
  { id: 'tote', name: 'Tote bag', kind: 'bag', outer: { w: 36, h: 34, d: 12 }, wall: 0.2 },
  { id: 'backpack', name: 'Backpack', kind: 'bag', outer: { w: 30, h: 45, d: 18 }, wall: 0.3 },
];

export function organizerPreset(id: string): OrganizerPreset {
  return ORGANIZER_PRESETS.find((preset) => preset.id === id) ?? ORGANIZER_PRESETS[1];
}

export interface ClothingType {
  id: string;
  name: string;
  /** The laid-flat shape a new item of this type starts with. */
  parts: Part[];
}

const body = (w: number, l: number, t: number): Part => ({ role: 'body', w, l, t, count: 1 });
const sleeves = (w: number, l: number, t: number): Part => ({ role: 'sleeve', w, l, t, count: 2 });
const legs = (w: number, l: number, t: number): Part => ({ role: 'leg', w, l, t, count: 2 });

export const CLOTHING_TYPES: ClothingType[] = [
  { id: 'tshirt', name: 'T-shirt', parts: [body(50, 68, 0.6), sleeves(20, 20, 0.5)] },
  { id: 'tank', name: 'Tank top', parts: [body(42, 62, 0.4)] },
  { id: 'longsleeve', name: 'Long-sleeve top', parts: [body(50, 68, 0.6), sleeves(16, 60, 0.5)] },
  { id: 'shirt', name: 'Shirt or blouse', parts: [body(52, 72, 0.5), sleeves(16, 62, 0.4)] },
  { id: 'sweater', name: 'Sweater', parts: [body(54, 66, 2.4), sleeves(18, 62, 2)] },
  { id: 'jacket', name: 'Jacket', parts: [body(56, 70, 3), sleeves(20, 64, 2.5)] },
  { id: 'coat', name: 'Coat', parts: [body(58, 100, 3.5), sleeves(20, 66, 3)] },
  { id: 'dress', name: 'Dress', parts: [body(48, 100, 0.8)] },
  { id: 'skirt', name: 'Skirt', parts: [body(44, 60, 0.8)] },
  { id: 'trousers', name: 'Trousers', parts: [body(44, 28, 1.2), legs(22, 76, 1)] },
  { id: 'jeans', name: 'Jeans', parts: [body(44, 28, 2), legs(22, 78, 1.6)] },
  { id: 'shorts', name: 'Shorts', parts: [body(44, 26, 1), legs(24, 14, 0.8)] },
  { id: 'leggings', name: 'Leggings or tights', parts: [body(34, 24, 0.6), legs(14, 70, 0.5)] },
  { id: 'underwear', name: 'Underwear', parts: [body(30, 22, 0.5)] },
  { id: 'bra', name: 'Bra', parts: [body(30, 14, 2.5)] },
  { id: 'socks', name: 'Socks (pair)', parts: [body(10, 24, 1.6)] },
  { id: 'scarf', name: 'Scarf', parts: [body(30, 180, 0.3)] },
  { id: 'other', name: 'Other', parts: [body(30, 30, 5)] },
];

export function clothingType(id: string): ClothingType {
  return CLOTHING_TYPES.find((type) => type.id === id) ?? CLOTHING_TYPES[CLOTHING_TYPES.length - 1];
}

export interface FoldOption {
  id: Fold;
  name: string;
}

/** The folds you can pick. 'hung' is not here: an item is hung by putting it on a rail. */
export const FOLD_OPTIONS: FoldOption[] = [
  { id: 'flat', name: 'Laid flat' },
  { id: 'half', name: 'Folded in half' },
  { id: 'thirds', name: 'Folded in thirds' },
  { id: 'quarters', name: 'Folded in quarters' },
  { id: 'rolled', name: 'Rolled' },
];

/** How many layers each fold makes across the width and down the length. */
export const FOLD_DIVISORS: Partial<Record<Fold, { across: number; down: number }>> = {
  half: { across: 1, down: 2 },
  thirds: { across: 1, down: 3 },
  quarters: { across: 2, down: 2 },
};
