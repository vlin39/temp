/*
 * The measurements of the open piece of furniture: its outside and inside
 * size, its panel thicknesses, and each compartment.
 */

import {
  innerDepth,
  innerWidth,
  renamePiece,
  resizeColumn,
  resizeCompartment,
  resizePiece,
  setThickness,
  updateCell,
  type Dimension,
  type ResizeMode,
  type ThicknessName,
} from '../model/furniture';
import type { Cell, Furniture, Home } from '../model/types';
import { useStore } from '../state/store';
import { LengthField, TextField, cls } from './parts';

const THICKNESSES: { name: ThicknessName; label: string }[] = [
  { name: 'side', label: 'Sides' },
  { name: 'top', label: 'Top' },
  { name: 'bottom', label: 'Bottom' },
  { name: 'back', label: 'Back' },
  { name: 'shelf', label: 'Shelves' },
  { name: 'divider', label: 'Dividers' },
  { name: 'base', label: 'Base height' },
];

function thicknessOf(piece: Furniture, name: ThicknessName): number {
  if (name === 'shelf') return piece.shelfThickness;
  if (name === 'divider') return piece.dividerThickness;
  if (name === 'base') return piece.baseHeight;
  return piece.panel[name];
}

export function MeasurePanel() {
  const { state, dispatch, tryEdit } = useStore();
  const piece = state.home.furniture.find((candidate) => candidate.id === state.openFurnitureId);
  if (!piece) return <p className="hint">Pick a piece of furniture to see its measurements.</p>;

  /** Applies a change to this piece. Returns the reason if it is refused. */
  const change = (update: (piece: Furniture) => Furniture): string | null =>
    tryEdit((home: Home) => ({
      ...home,
      furniture: home.furniture.map((candidate) => (candidate.id === piece.id ? update(candidate) : candidate)),
    }));

  const mode = state.resizeMode;
  const resize = (dimension: Dimension, from: 'outside' | 'inside') => (cm: number) =>
    change((current) => resizePiece(current, dimension, cm, from));
  const hasStorage = piece.columns.length > 0;
  const selectedCellId = state.selected?.kind === 'cell' ? state.selected.id : null;

  return (
    <div className="panel-body">
      <h2>Measurements</h2>
      <TextField id="piece-name" label="Name" value={piece.name} onCommit={(name) => change((current) => renamePiece(current, name))} />
      {piece.estimated && <p className="badge">Estimated sizes. Measure and type over them.</p>}

      <h3>Outside</h3>
      <div className="field-row">
        <LengthField id="outer-w" label="Width" value={piece.outer.w} onCommit={resize('w', 'outside')} />
        <LengthField id="outer-h" label="Height" value={piece.outer.h} onCommit={resize('h', 'outside')} />
        <LengthField id="outer-d" label="Depth" value={piece.outer.d} onCommit={resize('d', 'outside')} />
      </div>

      {hasStorage && (
        <>
          <h3>Inside</h3>
          <div className="field-row">
            <LengthField id="inner-w" label="Width" value={innerWidth(piece)} onCommit={resize('w', 'inside')} />
            <LengthField id="inner-d" label="Depth" value={innerDepth(piece)} onCommit={resize('d', 'inside')} />
          </div>

          <h3>Thickness</h3>
          <div className="field-row">
            {THICKNESSES.filter((thickness) => thickness.name !== 'divider' || piece.columns.length > 1).map(({ name, label }) => (
              <LengthField
                key={name}
                id={`thickness-${name}`}
                label={label}
                value={thicknessOf(piece, name)}
                onCommit={(cm) => change((current) => setThickness(current, name, cm, mode))}
                allowZero
              />
            ))}
          </div>

          <fieldset className="mode">
            <legend>When a compartment or thickness changes</legend>
            <ModeChoice value="keepOutside" current={mode} label="Keep the outside size (the neighbour gives or takes)" />
            <ModeChoice value="keepCompartments" current={mode} label="Keep the other compartments (the piece grows or shrinks)" />
          </fieldset>

          <h3>Compartments</h3>
          {piece.columns.map((column, columnIndex) => (
            <div key={column.id} className="column-block">
              {piece.columns.length > 1 && (
                <LengthField
                  id={`column-${column.id}`}
                  label={`Column ${columnIndex + 1} width`}
                  value={column.width}
                  onCommit={(cm) => change((current) => resizeColumn(current, column.id, cm, mode))}
                />
              )}
              {/* Listed top to bottom, the way the piece looks from the front. */}
              {[...column.cells].reverse().map((cell) => (
                <CellRow
                  key={cell.id}
                  cell={cell}
                  selected={cell.id === selectedCellId}
                  onSelect={() => dispatch({ type: 'select', selected: { kind: 'cell', id: cell.id } })}
                  onRename={(name) => change((current) => updateCell(current, cell.id, { name }))}
                  onHeight={(cm) => change((current) => resizeCompartment(current, cell.id, cm, mode))}
                  onRailDrop={(cm) => change((current) => updateCell(current, cell.id, { railDrop: cm }))}
                  onDrawerWall={(cm) => change((current) => updateCell(current, cell.id, { drawerWall: cm }))}
                />
              ))}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function ModeChoice({ value, current, label }: { value: ResizeMode; current: ResizeMode; label: string }) {
  const { dispatch } = useStore();
  return (
    <label className="check">
      <input
        id={`mode-${value}`}
        type="radio"
        name="resize-mode"
        checked={current === value}
        onChange={() => dispatch({ type: 'setResizeMode', mode: value })}
      />
      {label}
    </label>
  );
}

interface CellRowProps {
  cell: Cell;
  selected: boolean;
  onSelect: () => void;
  onRename: (name: string) => string | null;
  onHeight: (cm: number) => string | null;
  onRailDrop: (cm: number) => string | null;
  onDrawerWall: (cm: number) => string | null;
}

const KIND_NAMES: Record<Cell['kind'], string> = {
  shelf: 'Shelf',
  hanging: 'Hanging space',
  drawer: 'Drawer',
  fixture: 'Stores nothing',
};

function CellRow({ cell, selected, onSelect, onRename, onHeight, onRailDrop, onDrawerWall }: CellRowProps) {
  return (
    <div className={cls('cell-row', selected && 'is-selected')} onFocus={onSelect}>
      <TextField id={`cell-name-${cell.id}`} label={KIND_NAMES[cell.kind]} value={cell.name} onCommit={onRename} />
      <div className="field-row">
        <LengthField id={`cell-height-${cell.id}`} label="Clear height" value={cell.height} onCommit={onHeight} />
        {cell.kind === 'hanging' && (
          <LengthField id={`cell-rail-${cell.id}`} label="Rail below top" value={cell.railDrop} onCommit={onRailDrop} allowZero />
        )}
        {cell.kind === 'drawer' && (
          <LengthField id={`cell-wall-${cell.id}`} label="Drawer wall" value={cell.drawerWall} onCommit={onDrawerWall} allowZero />
        )}
      </div>
    </div>
  );
}
