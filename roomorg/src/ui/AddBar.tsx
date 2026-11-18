/*
 * Adding clothing, boxes and bags. A new thing goes into whatever is selected
 * (a compartment, a box or a bag), or beside it if it cannot go inside. With
 * nothing selected, or no room there, it goes on the floor.
 */

import { useState } from 'react';
import { nearestHolder, placeOrLeave } from '../model/place';
import { CLOTHING_TYPES, ORGANIZER_PRESETS, organizerPreset } from '../model/presets';
import { addItem, addOrganizer, reveal } from '../model/tree';
import type { Home, Id } from '../model/types';
import { useStore } from '../state/store';

export function AddBar() {
  const { state, index, dispatch, edit } = useStore();
  const [typeId, setTypeId] = useState(CLOTHING_TYPES[0].id);
  const [presetId, setPresetId] = useState('medium');
  const preset = organizerPreset(presetId);

  /** Where new things are aimed: the selected compartment, box or bag, or wherever the selected item is. */
  const selected = state.selected;
  const wish: Id | null = !selected ? null : selected.kind === 'item' ? (index.items.get(selected.id)?.parentId ?? null) : selected.id;

  /** Where new clothing will go, in words. */
  const target = (): string => {
    const cellRef = wish === null ? undefined : index.cells.get(wish);
    if (cellRef && cellRef.cell.kind !== 'fixture') return `into ${cellRef.furniture.name} › ${cellRef.cell.name}`;
    const organizer = wish === null ? undefined : index.organizers.get(wish);
    return organizer ? `into ${organizer.name}` : 'on the floor';
  };

  const add = (kind: 'item' | 'organizer', make: (home: Home) => { home: Home; id: Id }) => {
    const outcome = { refused: null as string | null };
    edit((home) => {
      const added = make(home);
      const put = placeOrLeave(added.home, added.id, nearestHolder(added.home, wish, added.id));
      outcome.refused = put.refused;
      return { home: reveal(put.home, added.id), select: { kind, id: added.id } };
    });
    if (outcome.refused) {
      dispatch({ type: 'notify', notice: { text: `${outcome.refused} It is on the floor instead.`, tone: 'error' } });
    }
  };

  return (
    <div className="add-bar">
      <div className="add-group">
        <select id="add-type" aria-label="Type of clothing to add" value={typeId} onChange={(event) => setTypeId(event.target.value)}>
          {CLOTHING_TYPES.map((type) => (
            <option key={type.id} value={type.id}>
              {type.name}
            </option>
          ))}
        </select>
        <button id="add-clothing" type="button" className="button button-primary" onClick={() => add('item', (home) => addItem(home, typeId))}>
          Add clothing
        </button>
      </div>
      <div className="add-group">
        <select id="add-size" aria-label="Kind of box or bag to add" value={presetId} onChange={(event) => setPresetId(event.target.value)}>
          {(['box', 'bag'] as const).map((kind) => (
            <optgroup key={kind} label={kind === 'box' ? 'Boxes' : 'Bags'}>
              {ORGANIZER_PRESETS.filter((candidate) => candidate.kind === kind).map((candidate) => (
                <option key={candidate.id} value={candidate.id}>
                  {candidate.name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <button id="add-container" type="button" className="button" onClick={() => add('organizer', (home) => addOrganizer(home, preset))}>
          Add {preset.kind}
        </button>
      </div>
      <p className="hint add-target">New clothing goes {target()}.</p>
    </div>
  );
}
