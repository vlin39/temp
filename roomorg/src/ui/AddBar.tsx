/*
 * Adding clothing and boxes. New things go into whatever is selected (a
 * compartment or a box), or into the unsorted tray when nothing is.
 */

import { useState } from 'react';
import { CLOTHING_TYPES, ORGANIZER_PRESETS } from '../model/presets';
import { addItem, addOrganizer } from '../model/tree';
import type { Id } from '../model/types';
import { useStore } from '../state/store';

export function AddBar() {
  const { state, index, edit } = useStore();
  const [typeId, setTypeId] = useState(CLOTHING_TYPES[0].id);
  const [presetId, setPresetId] = useState(ORGANIZER_PRESETS[1].id);

  /** The place a new thing of this kind goes, given what is selected. */
  const target = (kind: 'item' | 'organizer'): Id | null => {
    const selected = state.selected;
    if (!selected) return null;
    if (selected.kind === 'cell') return selected.id;

    // A box is selected: clothing goes into it. An item is selected: new things go where it is.
    const place = selected.kind === 'organizer' ? selected.id : (index.items.get(selected.id)?.parentId ?? null);
    const box = place === null ? undefined : index.organizers.get(place);
    // A box cannot go inside a box, so a new box goes beside it.
    return kind === 'organizer' && box ? box.parentId : place;
  };

  const nameOf = (parentId: Id | null): string => {
    if (parentId === null) return 'the unsorted tray';
    const cellRef = index.cells.get(parentId);
    return cellRef ? `${cellRef.furniture.name} › ${cellRef.cell.name}` : (index.organizers.get(parentId)?.name ?? 'the unsorted tray');
  };

  const addClothing = () =>
    edit((home) => {
      const added = addItem(home, target('item'), typeId);
      return { home: added.home, select: { kind: 'item', id: added.id } };
    });

  const addBox = () =>
    edit((home) => {
      const preset = ORGANIZER_PRESETS.find((candidate) => candidate.id === presetId)!;
      const added = addOrganizer(home, target('organizer'), preset);
      return { home: added.home, select: { kind: 'organizer', id: added.id } };
    });

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
        <button id="add-clothing" type="button" className="button button-primary" onClick={addClothing}>
          Add clothing
        </button>
      </div>
      <div className="add-group">
        <select id="add-size" aria-label="Size of box to add" value={presetId} onChange={(event) => setPresetId(event.target.value)}>
          {ORGANIZER_PRESETS.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.name}
            </option>
          ))}
        </select>
        <button id="add-box" type="button" className="button" onClick={addBox}>
          Add box
        </button>
      </div>
      <p className="hint add-target">New clothing goes into {nameOf(target('item'))}.</p>
    </div>
  );
}
