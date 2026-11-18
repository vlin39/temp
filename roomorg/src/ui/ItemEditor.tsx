/*
 * The side panel for the selected thing: a clothing item, a box or a bag.
 */

import { useState } from 'react';
import { flatOutline, storedSize } from '../model/folding';
import { arrange, insideOf } from '../model/layout';
import { place } from '../model/place';
import { CLOTHING_TYPES, FOLD_OPTIONS, ORGANIZER_PRESETS, clothingType } from '../model/presets';
import { canHold, childId, childrenOf, destinations, isRail, pathTo, remove, reveal, updateItem, updateOrganizer } from '../model/tree';
import type { Fold, Id, Item, Organizer, Part, Size3 } from '../model/types';
import { toDisplay, toDisplaySize } from '../model/units';
import { useStore } from '../state/store';
import { Contents } from './Contents';
import { overClass, useDrag } from './drag';
import { LengthField, TextField, TwoStepButton, cls } from './parts';
import { ThingChip } from './ThingChip';
import { FLAT, frontRect, type View } from './view';

const PART_NAMES: Record<Part['role'], string> = { body: 'Body', sleeve: 'Each sleeve', leg: 'Each leg' };

export function ItemEditor() {
  const { state, index } = useStore();
  const item = state.selected?.kind === 'item' ? index.items.get(state.selected.id) : undefined;
  const organizer = state.selected?.kind === 'organizer' ? index.organizers.get(state.selected.id) : undefined;
  if (item) return <ClothingEditor key={item.id} item={item} />;
  if (organizer) return <OrganizerEditor key={organizer.id} organizer={organizer} />;
  return null;
}

function PanelHead({ title }: { title: string }) {
  const { dispatch } = useStore();
  return (
    <div className="panel-head">
      <h2>{title}</h2>
      <button type="button" className="button button-small" onClick={() => dispatch({ type: 'select', selected: null })}>
        Close
      </button>
    </div>
  );
}

/**
 * Where a thing is, and a list to move it somewhere else without dragging.
 * The list only offers places that may hold it; a box is still refused where
 * there is no room. A door or drawer in the way is opened.
 */
function Whereabouts({ id }: { id: Id }) {
  const { state, index, edit } = useStore();
  const parentId = (index.items.get(id) ?? index.organizers.get(id))?.parentId ?? null;
  const places = destinations(state.home).filter((candidate) => canHold(index, candidate.id, id));

  return (
    <label className="field">
      <span className="field-label">Where: {pathTo(state.home, id).slice(0, -1).join(' › ')}</span>
      <select
        id="move-to"
        value={parentId ?? ''}
        onChange={(event) => edit((home) => reveal(place(home, id, event.target.value === '' ? null : event.target.value), id))}
      >
        {places.map((candidate) => (
          <option key={candidate.id ?? ''} value={candidate.id ?? ''}>
            {candidate.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ClothingEditor({ item }: { item: Item }) {
  const { index, tryEdit, edit } = useStore();
  const [pendingType, setPendingType] = useState<string | null>(null);

  const change = (changes: Parameters<typeof updateItem>[2]) => tryEdit((home) => updateItem(home, item.id, changes));
  const setPart = (part: Part, field: 'w' | 'l' | 't') => (cm: number) =>
    change({ parts: item.parts.map((p) => (p === part ? { ...p, [field]: cm } : p)) });

  const onRail = isRail(index, item.parentId);
  const fold = onRail ? (item.previousFold ?? 'flat') : item.fold;
  const stored = storedSize(item);
  const startingParts = JSON.stringify(clothingType(item.type).parts);
  const partsEdited = JSON.stringify(item.parts) !== startingParts;

  const chooseType = (typeId: string) => {
    // Changing the type replaces the part sizes, so ask first if they were edited.
    if (partsEdited) setPendingType(typeId);
    else change({ type: typeId });
  };

  return (
    <div className="panel-body">
      <PanelHead title="Clothing" />
      <TextField id="item-name" label="Name" value={item.name} onCommit={(name) => change({ name })} />

      <div className="field-row">
        <label className="field">
          <span className="field-label">Type</span>
          <select id="item-type" value={item.type} onChange={(event) => chooseType(event.target.value)}>
            {CLOTHING_TYPES.map((type) => (
              <option key={type.id} value={type.id}>
                {type.name}
              </option>
            ))}
          </select>
        </label>
        <label className="field field-narrow">
          <span className="field-label">Colour</span>
          <input id="item-colour" type="color" value={item.colour} onChange={(event) => change({ colour: event.target.value })} />
        </label>
      </div>
      {pendingType && (
        <div className="confirm">
          <p>Changing the type replaces the sizes you typed with that type's starting sizes.</p>
          <button
            type="button"
            className="button button-small"
            onClick={() => {
              change({ type: pendingType });
              setPendingType(null);
            }}
          >
            Change to {clothingType(pendingType).name}
          </button>
          <button type="button" className="button button-small" onClick={() => setPendingType(null)}>
            Keep {clothingType(item.type).name}
          </button>
        </div>
      )}

      <h3>Laid flat</h3>
      <FlatPreview item={item} />
      {item.parts.map((part, i) => (
        <fieldset key={part.role} className="part">
          <legend>{PART_NAMES[part.role]}</legend>
          <div className="field-row">
            <LengthField id={`part-${i}-w`} label="Width" value={part.w} onCommit={setPart(part, 'w')} />
            <LengthField id={`part-${i}-l`} label="Length" value={part.l} onCommit={setPart(part, 'l')} />
            <LengthField id={`part-${i}-t`} label="Thickness" value={part.t} onCommit={setPart(part, 't')} />
          </div>
        </fieldset>
      ))}

      <h3>Stored</h3>
      <label className="field">
        <span className="field-label">{onRail ? 'Fold when it comes off the rail' : 'Fold'}</span>
        <select id="item-fold" value={fold} onChange={(event) => change({ fold: event.target.value as Fold })}>
          {FOLD_OPTIONS.map((option) => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
      </label>
      <p className="stored-size">
        {onRail
          ? `Hung: takes ${toDisplay(stored.w)} of rail and hangs ${toDisplay(stored.h)} down.`
          : `Takes ${toDisplaySize(stored.w, stored.d, stored.h)} (wide × deep × tall).`}
      </p>

      <Whereabouts id={item.id} />
      <TwoStepButton className="button-danger" confirmLabel="Click again to delete" onConfirm={() => edit((home) => remove(home, item.id))}>
        Delete item
      </TwoStepButton>
    </div>
  );
}

/** The item's laid-flat shape, drawn to scale in its own colour. */
function FlatPreview({ item }: { item: Item }) {
  const outline = flatOutline(item);
  const scale = Math.min(200 / outline.w, 130 / outline.l);
  return (
    <svg
      className="flat-preview"
      width={outline.w * scale + 2}
      height={outline.l * scale + 2}
      role="img"
      aria-label={`Laid flat: ${toDisplaySize(outline.w, outline.l)}`}
    >
      {outline.rects.map((r, i) => (
        <rect key={i} className="item-body" x={1 + r.x * scale} y={1 + r.y * scale} width={r.w * scale} height={r.h * scale} fill={item.colour} />
      ))}
    </svg>
  );
}

function OrganizerEditor({ organizer }: { organizer: Organizer }) {
  const { state, tryEdit, edit } = useStore();
  const change = (changes: Parameters<typeof updateOrganizer>[2]) => tryEdit((home) => updateOrganizer(home, organizer.id, changes));
  const setOuter = (dimension: keyof Size3) => (cm: number) => change({ outer: { ...organizer.outer, [dimension]: cm } });
  const inside = insideOf(organizer);
  const children = childrenOf(state.home, organizer.id);
  const isBag = organizer.kind === 'bag';

  return (
    <div className="panel-body">
      <PanelHead title={isBag ? 'Bag' : 'Box'} />
      <TextField id="organizer-name" label="Name" value={organizer.name} onCommit={(name) => change({ name })} />

      <h3>Inside ({children.length})</h3>
      <InsideView organizer={organizer} />
      {children.length === 0 && (
        <p className="hint">
          Empty. Drag {isBag ? 'clothing' : 'clothing or a bag'} onto it, or add some while it is selected.
        </p>
      )}
      <div className="chip-row">
        {children.map((child) => (
          <ThingChip key={childId(child)} child={child} zone={organizer.id} counted={false} />
        ))}
      </div>

      <h3>Size</h3>
      <div className="preset-row">
        {ORGANIZER_PRESETS.filter((preset) => preset.kind === organizer.kind).map((preset) => (
          <button
            key={preset.id}
            type="button"
            className="button button-small"
            onClick={() => change({ outer: { ...preset.outer }, wall: preset.wall })}
          >
            {preset.name}
          </button>
        ))}
      </div>
      <div className="field-row">
        <LengthField id="organizer-w" label="Width" value={organizer.outer.w} onCommit={setOuter('w')} />
        <LengthField id="organizer-h" label="Height" value={organizer.outer.h} onCommit={setOuter('h')} />
        <LengthField id="organizer-d" label="Depth" value={organizer.outer.d} onCommit={setOuter('d')} />
        <LengthField id="organizer-wall" label="Wall" value={organizer.wall} onCommit={(cm) => change({ wall: cm })} allowZero />
      </div>
      <p className="stored-size">Inside: {toDisplaySize(inside.w, inside.d, inside.h)} (wide × deep × tall).</p>

      <Whereabouts id={organizer.id} />
      <TwoStepButton
        className="button-danger"
        confirmLabel="Click again: contents go to the floor"
        onConfirm={() => edit((home) => remove(home, organizer.id))}
      >
        Delete {organizer.kind}
      </TwoStepButton>
    </div>
  );
}

/** The inside of a box or bag seen straight from the front, larger than it appears on the shelf. */
function InsideView({ organizer }: { organizer: Organizer }) {
  const { state, index } = useStore();
  const { drag } = useDrag();
  const inside = insideOf(organizer);
  const placed = arrange(state.home, organizer.id, index);

  // Leave room to show things that are too big for it, up to three times its size.
  const drawnW = Math.min(Math.max(inside.w, ...placed.map((p) => p.x + p.w)), inside.w * 3);
  const drawnH = Math.min(Math.max(inside.h, ...placed.map((p) => p.y + p.h)), inside.h * 3);
  const margin = 12;
  const scale = Math.min(250 / drawnW, 150 / drawnH);
  const view: View = { ...FLAT, scale, originX: margin, originY: margin + drawnH * scale };

  return (
    <svg
      className="inside-view"
      width={drawnW * scale + 2 * margin}
      height={drawnH * scale + 2 * margin}
      role="group"
      aria-label={`Inside ${organizer.name}`}
    >
      <rect
        className={cls(organizer.kind === 'bag' ? 'bag-body' : 'organizer-body', overClass(drag, organizer.id))}
        data-drop={organizer.id}
        {...frontRect(view, { x: 0, y: 0, z: 0, w: inside.w, h: inside.h })}
      />
      <Contents placed={placed} view={view} maxRight={drawnW} />
    </svg>
  );
}
