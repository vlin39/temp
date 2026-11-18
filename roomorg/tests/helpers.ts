import { organizerPreset } from '../src/model/presets';
import { addItem, addOrganizer, move } from '../src/model/tree';
import type { Home, Id } from '../src/model/types';

/** Adds a clothing item and puts it somewhere. */
export function addItemIn(home: Home, parentId: Id | null, typeId: string): { home: Home; id: Id } {
  const added = addItem(home, typeId);
  return { home: move(added.home, added.id, parentId), id: added.id };
}

/** Adds a box or bag of a preset size and puts it somewhere, without checking for room. */
export function addOrganizerIn(home: Home, parentId: Id | null, presetId: string): { home: Home; id: Id } {
  const added = addOrganizer(home, organizerPreset(presetId));
  return { home: move(added.home, added.id, parentId), id: added.id };
}
