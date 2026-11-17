/*
 * Reading a saved home (the browser's autosave, or a backup file) and checking
 * every part of it before it is used. A file that fails is refused whole, with
 * the reasons, so a damaged backup can never half-load.
 */

import { checkPiece } from './furniture';
import type { Home } from './types';

export type ParseResult = { ok: true; home: Home } | { ok: false; problems: string[] };

type Raw = Record<string, unknown>;

const FACINGS = ['north', 'east', 'south', 'west'];
const CELL_KINDS = ['shelf', 'hanging', 'drawer', 'cabinet'];
const FOLDS = ['flat', 'half', 'thirds', 'quarters', 'rolled', 'hung'];
const PART_ROLES = ['body', 'sleeve', 'leg'];

const isObject = (value: unknown): value is Raw => typeof value === 'object' && value !== null && !Array.isArray(value);
const isText = (value: unknown): value is string => typeof value === 'string';
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const isPositive = (value: unknown) => isNumber(value) && value > 0;
const isNotNegative = (value: unknown) => isNumber(value) && value >= 0;
const isOneOf = (options: string[]) => (value: unknown) => isText(value) && options.includes(value);
const isSize = (value: unknown) => isObject(value) && isPositive(value.w) && isPositive(value.h) && isPositive(value.d);

/** Collects what is wrong with one record. `checks` maps a field name to a test of its value. */
function checkFields(label: string, record: unknown, checks: Record<string, (value: unknown) => boolean>): string[] {
  if (!isObject(record)) return [`${label} is not a record.`];
  return Object.entries(checks)
    .filter(([field, passes]) => !passes(record[field]))
    .map(([field]) => `${label} has a missing or wrong "${field}".`);
}

function checkFurniture(piece: unknown, label: string): string[] {
  const problems = checkFields(label, piece, {
    id: isText,
    name: isText,
    roomId: isText,
    x: isNumber,
    y: isNumber,
    facing: isOneOf(FACINGS),
    outer: isSize,
    panel: (v) => isObject(v) && [v.side, v.top, v.bottom, v.back].every(isNotNegative),
    shelfThickness: isNotNegative,
    dividerThickness: isNotNegative,
    baseHeight: isNotNegative,
    columns: Array.isArray,
    estimated: (v) => typeof v === 'boolean',
  });
  if (problems.length > 0) return problems;

  for (const column of (piece as Raw).columns as unknown[]) {
    problems.push(...checkFields(`${label}, a column`, column, { id: isText, width: isPositive, cells: Array.isArray }));
    if (!isObject(column) || !Array.isArray(column.cells)) continue;
    for (const cell of column.cells) {
      problems.push(
        ...checkFields(`${label}, a compartment`, cell, {
          id: isText,
          name: isText,
          kind: isOneOf(CELL_KINDS),
          height: isPositive,
          railDrop: isNotNegative,
          drawerWall: isNotNegative,
        }),
      );
    }
  }
  return problems;
}

const isParts = (value: unknown) =>
  Array.isArray(value) &&
  value.length > 0 &&
  value.every(
    (part) => isObject(part) && isOneOf(PART_ROLES)(part.role) && [part.w, part.l, part.t, part.count].every(isPositive),
  );

/** Checks a parsed object. Returns it as a Home, or the list of what is wrong. */
export function checkHome(data: unknown): ParseResult {
  if (!isObject(data)) return { ok: false, problems: ['The file does not hold a saved home.'] };
  if (data.version !== 1) {
    return { ok: false, problems: ['The file is from a version of the app this one cannot read.'] };
  }
  const lists = ['rooms', 'furniture', 'organizers', 'items'];
  const missing = lists.filter((list) => !Array.isArray(data[list]));
  if (missing.length > 0) return { ok: false, problems: [`The file is missing its list of ${missing.join(', ')}.`] };

  const rooms = data.rooms as unknown[];
  const furniture = data.furniture as unknown[];
  const organizers = data.organizers as unknown[];
  const items = data.items as unknown[];
  const problems: string[] = [];

  rooms.forEach((room, i) =>
    problems.push(
      ...checkFields(`Room ${i + 1}`, room, { id: isText, name: isText, x: isNumber, y: isNumber, w: isPositive, d: isPositive }),
    ),
  );
  furniture.forEach((piece, i) => problems.push(...checkFurniture(piece, `Furniture ${i + 1}`)));
  organizers.forEach((organizer, i) =>
    problems.push(
      ...checkFields(`Box ${i + 1}`, organizer, {
        id: isText,
        name: isText,
        parentId: (v) => v === null || isText(v),
        order: isNumber,
        outer: isSize,
        wall: isNotNegative,
      }),
    ),
  );
  items.forEach((item, i) =>
    problems.push(
      ...checkFields(`Item ${i + 1}`, item, {
        id: isText,
        kind: (v) => v === 'clothing',
        name: isText,
        type: isText,
        colour: (v) => isText(v) && /^#[0-9a-f]{6}$/i.test(v),
        parentId: (v) => v === null || isText(v),
        order: isNumber,
        parts: isParts,
        fold: isOneOf(FOLDS),
        previousFold: (v) => v === null || isOneOf(FOLDS)(v),
      }),
    ),
  );
  if (!(data.lastBackupAt === null || isText(data.lastBackupAt))) problems.push('The backup date is wrong.');
  if (problems.length > 0) return { ok: false, problems };

  // Every record has the right shape. Now check that they fit together.
  const home = data as unknown as Home;
  const roomIds = new Set(home.rooms.map((room) => room.id));
  const cellIds = new Set(home.furniture.flatMap((piece) => piece.columns.flatMap((column) => column.cells.map((cell) => cell.id))));
  const organizerIds = new Set(home.organizers.map((organizer) => organizer.id));

  const allIds = [...roomIds, ...home.furniture.map((piece) => piece.id), ...cellIds, ...organizerIds, ...home.items.map((item) => item.id)];
  if (new Set(allIds).size !== allIds.length) problems.push('Two things in the file share the same id.');

  for (const piece of home.furniture) {
    if (!roomIds.has(piece.roomId)) problems.push(`${piece.name} is in a room that does not exist.`);
    problems.push(...checkPiece(piece));
  }
  for (const organizer of home.organizers) {
    if (organizer.parentId !== null && !cellIds.has(organizer.parentId)) {
      problems.push(`The box "${organizer.name}" is in a compartment that does not exist.`);
    }
  }
  for (const item of home.items) {
    if (item.parentId !== null && !cellIds.has(item.parentId) && !organizerIds.has(item.parentId)) {
      problems.push(`"${item.name}" is in a place that does not exist.`);
    }
  }
  return problems.length > 0 ? { ok: false, problems } : { ok: true, home };
}

/** Reads the text of a saved home. */
export function parseHome(text: string): ParseResult {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return { ok: false, problems: ['The file is not valid JSON.'] };
  }
  return checkHome(data);
}
