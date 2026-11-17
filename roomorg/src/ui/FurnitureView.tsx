/*
 * The front view of the open piece of furniture, drawn to scale: its panels,
 * its compartments, and everything in them.
 */

import { clearSpaces, panels, usableSpace } from '../model/furniture';
import { arrange, reach } from '../model/layout';
import { cellHolding, childId, childrenOf } from '../model/tree';
import type { Box, Cell, Furniture } from '../model/types';
import { toDisplaySize } from '../model/units';
import { useStore } from '../state/store';
import { Contents } from './Contents';
import { useDrag } from './drag';
import { cls, fitText, useElementSize } from './parts';
import { ThingChip } from './ThingChip';

/** Room around the drawing so outlines and warning marks are not clipped. */
const MARGIN = 14;

/** How far past the piece's side overflowing things may be drawn, as a share of its width. */
const MAX_OVERHANG = 0.6;

export function FurnitureView() {
  const { state } = useStore();
  const [ref, size] = useElementSize<HTMLDivElement>();
  const piece = state.home.furniture.find((candidate) => candidate.id === state.openFurnitureId);

  return (
    <div className="furniture-view" ref={ref}>
      {!piece && <p className="hint">Pick a piece of furniture from the list or the floor plan.</p>}
      {piece && piece.columns.length === 0 && <p className="hint">{piece.name} has no storage to show.</p>}
      {piece && piece.columns.length > 0 && size.width > 0 && <Drawing piece={piece} availableWidth={size.width} />}
      {piece && piece.columns.length > 0 && <CompartmentList />}
    </div>
  );
}

/**
 * Names everything in the compartment being looked at: the selected one, or
 * the one holding the selected box or item. The chips can be dragged, which is
 * the easy way to pick up something too thin to grab in the drawing.
 */
function CompartmentList() {
  const { state, index } = useStore();
  const selected = state.selected;

  const cellRef = selected?.kind === 'cell' ? index.cells.get(selected.id) : selected ? cellHolding(index, selected.id) : undefined;
  if (!cellRef || cellRef.furniture.id !== state.openFurnitureId) {
    return <p className="hint compartment-list">Click a compartment to list what is in it.</p>;
  }

  const children = childrenOf(state.home, cellRef.cell.id);
  return (
    <div className="compartment-list">
      <span className="compartment-list-name">In {cellRef.cell.name}:</span>
      {children.length === 0 && <span className="hint">nothing yet</span>}
      {children.map((child) => (
        <ThingChip key={childId(child)} child={child} zone={cellRef.cell.id} counted={false} />
      ))}
    </div>
  );
}

function Drawing({ piece, availableWidth }: { piece: Furniture; availableWidth: number }) {
  const { state, index } = useStore();
  const spaces = clearSpaces(piece);
  const cells = piece.columns.flatMap((column) =>
    column.cells.map((cell) => {
      const space = spaces.get(cell.id)!;
      return { cell, space, usable: usableSpace(cell, space), placed: arrange(state.home, cell.id, index) };
    }),
  );

  // Things that do not fit hang out past the piece's side. Make room to show them, up to a limit.
  const rightmost = Math.max(piece.outer.w, ...cells.map(({ usable, placed }) => usable.x + reach(placed).w));
  const drawnWidth = Math.min(rightmost, piece.outer.w * (1 + MAX_OVERHANG));

  // A piece with one column has room beside it for compartment names; a wider piece labels inside.
  const gutter = piece.columns.length === 1 ? Math.min(170, Math.max(96, availableWidth * 0.28)) : 0;
  const maxHeight = Math.max(280, window.innerHeight * 0.62);
  const scale = Math.min((availableWidth - gutter - 2 * MARGIN) / drawnWidth, (maxHeight - 2 * MARGIN) / piece.outer.h);

  const height = piece.outer.h * scale;
  const left = gutter + MARGIN;
  const floor = MARGIN + height; // the pixel row of the furniture's floor line

  /** Turns a box measured from the piece's bottom-left corner into pixels. */
  const toPixels = (box: { x: number; y: number; w: number; h: number }) => ({
    x: left + box.x * scale,
    y: floor - (box.y + box.h) * scale,
    width: box.w * scale,
    height: box.h * scale,
  });

  return (
    <svg
      className="furniture-drawing"
      width={gutter + drawnWidth * scale + 2 * MARGIN}
      height={height + 2 * MARGIN}
      role="group"
      aria-label={`Front view of ${piece.name}`}
    >
      <rect className="interior" {...toPixels({ x: 0, y: 0, w: piece.outer.w, h: piece.outer.h })} />
      {cells.map(({ cell, space }) => (
        <CellShape key={cell.id} cell={cell} space={space} toPixels={toPixels} scale={scale} gutter={gutter} />
      ))}
      {panels(piece).map((panel, i) => (
        <rect key={i} className={`panel panel-${panel.kind}`} {...toPixels(panel)} />
      ))}
      {cells.map(({ cell, usable, placed }) => {
        const corner = toPixels({ x: usable.x, y: usable.y, w: 0, h: 0 });
        return (
          <Contents
            key={cell.id}
            placed={placed}
            originX={corner.x}
            originY={corner.y}
            scale={scale}
            maxRight={left + drawnWidth * scale}
          />
        );
      })}
    </svg>
  );
}

interface CellShapeProps {
  cell: Cell;
  space: Box;
  toPixels: (box: { x: number; y: number; w: number; h: number }) => { x: number; y: number; width: number; height: number };
  scale: number;
  /** Width of the label area to the left of the piece, or 0 to label inside the compartment. */
  gutter: number;
}

/** One compartment: the area you can drop things on, its rail if it has one, and its label. */
function CellShape({ cell, space, toPixels, scale, gutter }: CellShapeProps) {
  const { state, dispatch } = useStore();
  const { drag } = useDrag();
  const area = toPixels(space);
  const selected = state.selected?.kind === 'cell' && state.selected.id === cell.id;
  const size = toDisplaySize(space.w, space.h);
  const railY = area.y + cell.railDrop * scale;

  return (
    <g className={cls('cell', selected && 'is-selected', drag?.overZone === cell.id && 'is-over')}>
      <rect
        className="cell-zone"
        data-drop={cell.id}
        {...area}
        onClick={() => dispatch({ type: 'select', selected: { kind: 'cell', id: cell.id } })}
      >
        <title>{`${cell.name}: ${size} clear`}</title>
      </rect>
      {cell.kind === 'hanging' && <line className="rail" x1={area.x + 2} x2={area.x + area.width - 2} y1={railY} y2={railY} />}
      {gutter > 0 ? (
        <>
          <text className="cell-name" x={gutter + MARGIN - 8} y={area.y + area.height / 2 - 1} textAnchor="end">
            {fitText(cell.name, gutter - 4)}
          </text>
          <text className="cell-size" x={gutter + MARGIN - 8} y={area.y + area.height / 2 + 12} textAnchor="end">
            {size}
          </text>
        </>
      ) : (
        <text className="cell-name" x={area.x + 4} y={area.y + 12}>
          {fitText(cell.name, area.width - 8)}
        </text>
      )}
    </g>
  );
}
