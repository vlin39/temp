/*
 * Reading a saved home (the browser's autosave, or a backup file) and checking
 * every part of it before it is used. A file that fails is refused whole, with
 * the reasons, so a damaged backup can never half-load.
 *
 * A file saved by an older version of the app is brought up to date first.
 */

import { checkPiece } from './furniture';
import { createStarterHome } from './roomData';
import { indexHome, whyNotHold } from './tree';
import type { Home } from './types';

export type ParseResult = { ok: true; home: Home } | { ok: false; problems: string[] };

type Raw = Record<string, unknown>;

const FACINGS = ['north', 'east', 'south', 'west'];
const CELL_KINDS = ['shelf', 'hanging', 'drawer', 'fixture'];
const ORGANIZER_KINDS = ['box', 'bag'];
const FOLDS = ['flat', 'half', 'thirds', 'quarters', 'rolled', 'hung'];
const PART_ROLES = ['body', 'sleeve', 'leg'];

const isObject = (value: unknown): value is Raw => typeof value === 'object' && value !== null && !Array.isArray(value);
const isText = (value: unknown): value is string => typeof value === 'string';
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const isPositive = (value: unknown) => isNumber(value) && value > 0;
const isNotNegative = (value: unknown) => isNumber(value) && value >= 0;
const isOneOf = (options: string[]) => (value: unknown) => isText(value) && options.includes(value);
const isSize = (value: unknown) => isObject(value) && isPositive(value.w) && isPositive(value.h) && isPositive(value.d);
const isTextList = (value: unknown) => Array.isArray(value) && value.every(isText);

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
    doors: Array.isArray,
    estimated: (v) => typeof v === 'boolean',
  });
  if (problems.length > 0) return problems;

  for (const door of (piece as Raw).doors as unknown[]) {
    problems.push(...checkFields(`${label}, a door`, door, { id: isText, name: isText, cellIds: isTextList }));
  }

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

/**
 * Brings a home saved by version 1 up to the shape of version 2, which added
 * doors, bags and the corrected sink unit:
 * - every organizer was a box;
 * - 'cabinet' compartments become shelves behind a door;
 * - each piece gets the doors the starter home gives it, where its compartments still match;
 * - the sink unit is replaced by the corrected one, since its compartments changed.
 * Anything not shaped as expected is passed through for checkHome to report.
 */
function upgradeFromVersion1(data: Raw): Raw {
  const starter = createStarterHome();
  const each = (list: unknown, change: (record: Raw) => unknown): unknown =>
    Array.isArray(list) ? list.map((entry) => (isObject(entry) ? change(entry) : entry)) : list;

  const furniture = each(data.furniture, (piece) => {
    const model = starter.furniture.find((candidate) => candidate.id === piece.id);
    if (model && piece.id === 'sink-unit') return model;

    const cellIds = new Set<unknown>();
    const columns = each(piece.columns, (column) => ({
      ...column,
      cells: each(column.cells, (cell) => {
        cellIds.add(cell.id);
        return cell.kind === 'cabinet' ? { ...cell, kind: 'shelf' } : cell;
      }),
    }));
    const doors = (model?.doors ?? []).filter((door) => door.cellIds.every((id) => cellIds.has(id)));
    return { ...piece, columns, doors };
  });

  return {
    ...data,
    version: 2,
    furniture,
    organizers: each(data.organizers, (organizer) => ({ kind: 'box', ...organizer })),
    open: starter.open,
  };
}

/**
 * After an upgrade: anything whose place is gone, or can no longer hold it
 * (the sink), goes to the floor, and open doors that do not exist are dropped.
 */
function settleAfterUpgrade(home: Home): Home {
  const index = indexHome(home);
  let nextPlace = home.organizers.length + home.items.length;
  const rehome = <T extends Home['organizers'][number] | Home['items'][number]>(thing: T): T =>
    whyNotHold(index, thing.parentId, thing.id) === null ? thing : { ...thing, parentId: null, order: nextPlace++ };
  const doorIds = new Set(home.furniture.flatMap((piece) => piece.doors.map((door) => door.id)));
  return {
    ...home,
    organizers: home.organizers.map(rehome),
    items: home.items.map(rehome),
    open: home.open.filter((id) => doorIds.has(id)),
  };
}

/** Checks a parsed object. Returns it as a Home, or the list of what is wrong. */
export function checkHome(saved: unknown): ParseResult {
  if (!isObject(saved)) return { ok: false, problems: ['The file does not hold a saved home.'] };
  const lists = ['rooms', 'furniture', 'organizers', 'items'];
  const missing = lists.filter((list) => !Array.isArray(saved[list]));
  if (missing.length > 0) return { ok: false, problems: [`The file is missing its list of ${missing.join(', ')}.`] };

  const data = saved.version === 1 ? upgradeFromVersion1(saved) : saved;
  if (data.version !== 2) {
    return { ok: false, problems: ['The file is from a version of the app this one cannot read.'] };
  }

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
      ...checkFields(`Box or bag ${i + 1}`, organizer, {
        id: isText,
        kind: isOneOf(ORGANIZER_KINDS),
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
  if (!isTextList(data.open)) problems.push('The list of open doors and drawers is wrong.');
  if (!(data.lastBackupAt === null || isText(data.lastBackupAt))) problems.push('The backup date is wrong.');
  if (problems.length > 0) return { ok: false, problems };

  // Every record has the right shape. Now check that they fit together.
  const home = saved.version === 1 ? settleAfterUpgrade(data as unknown as Home) : (data as unknown as Home);
  const roomIds = new Set(home.rooms.map((room) => room.id));
  const cellIds = new Set(home.furniture.flatMap((piece) => piece.columns.flatMap((column) => column.cells.map((cell) => cell.id))));
  const organizerIds = new Set(home.organizers.map((organizer) => organizer.id));
  const doorIds = home.furniture.flatMap((piece) => piece.doors.map((door) => door.id));

  const allIds = [
    ...roomIds,
    ...home.furniture.map((piece) => piece.id),
    ...cellIds,
    ...doorIds,
    ...organizerIds,
    ...home.items.map((item) => item.id),
  ];
  if (new Set(allIds).size !== allIds.length) problems.push('Two things in the file share the same id.');

  for (const piece of home.furniture) {
    if (!roomIds.has(piece.roomId)) problems.push(`${piece.name} is in a room that does not exist.`);
    const ownCells = new Set(piece.columns.flatMap((column) => column.cells.map((cell) => cell.id)));
    if (piece.doors.some((door) => door.cellIds.length === 0 || door.cellIds.some((id) => !ownCells.has(id)))) {
      problems.push(`${piece.name} has a door in front of a compartment it does not have.`);
    }
    problems.push(...checkPiece(piece));
  }

  const index = indexHome(home);
  for (const thing of [...home.organizers, ...home.items]) {
    if (thing.parentId !== null && !cellIds.has(thing.parentId) && !organizerIds.has(thing.parentId)) {
      problems.push(`"${thing.name}" is in a place that does not exist.`);
      continue;
    }
    const reason = whyNotHold(index, thing.parentId, thing.id);
    if (reason !== null) problems.push(`"${thing.name}" is somewhere it cannot be. ${reason}`);
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
