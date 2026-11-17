/*
 * Choosing which piece of furniture to look at: a dropdown, and a floor plan
 * of the suite where each piece can be clicked.
 */

import type { Furniture, Home, Room } from '../model/types';
import { useStore } from '../state/store';
import { cls } from './parts';

/** Largest text size on the plan, in plan units (centimetres). About 8 pixels at the usual scale. */
const LABEL_SIZE = 20;

/** Below this, a label is too small to read and is left out. The hover text still names the piece. */
const SMALLEST_LABEL = 13;

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** A spot for a room's name that no furniture covers, trying the middle of the room first. */
function roomNameSpot(room: Room, taken: Rect[]): { x: number; y: number } {
  const width = room.name.length * LABEL_SIZE * 0.68;
  for (const down of [0.5, 0.3, 0.7, 0.14, 0.88]) {
    for (const across of [0.5, 0.36, 0.64, 0.25, 0.75]) {
      const centre = { x: room.x + across * room.w, y: room.y + down * room.d };
      const label = { x: centre.x - width / 2, y: centre.y - LABEL_SIZE / 2, w: width, h: LABEL_SIZE };
      const insideRoom = label.x >= room.x && label.x + label.w <= room.x + room.w;
      if (insideRoom && !taken.some((box) => overlaps(label, box))) return centre;
    }
  }
  return { x: room.x + room.w / 2, y: room.y + room.d / 2 };
}

/** The rectangle a piece covers on the floor. Facing east or west turns it a quarter turn. */
function footprint(piece: Furniture, room: Room): Rect {
  const sideways = piece.facing === 'east' || piece.facing === 'west';
  return {
    x: room.x + piece.x,
    y: room.y + piece.y,
    w: sideways ? piece.outer.d : piece.outer.w,
    h: sideways ? piece.outer.w : piece.outer.d,
  };
}

/** A thick line along the side of the footprint the piece opens toward. */
function frontEdge(piece: Furniture, box: Rect) {
  const right = box.x + box.w;
  const bottom = box.y + box.h;
  if (piece.facing === 'north') return { x1: box.x, y1: box.y, x2: right, y2: box.y };
  if (piece.facing === 'south') return { x1: box.x, y1: bottom, x2: right, y2: bottom };
  if (piece.facing === 'west') return { x1: box.x, y1: box.y, x2: box.x, y2: bottom };
  return { x1: right, y1: box.y, x2: right, y2: bottom };
}

export function FurniturePicker() {
  const { state, dispatch } = useStore();
  const { home, openFurnitureId } = state;
  const open = (furnitureId: string) => dispatch({ type: 'open', furnitureId });

  return (
    <div className="picker">
      <label className="field">
        <span className="field-label">Furniture</span>
        <select id="furniture-select" value={openFurnitureId ?? ''} onChange={(event) => open(event.target.value)}>
          {home.rooms.map((room) => (
            <optgroup key={room.id} label={room.name}>
              {home.furniture
                .filter((piece) => piece.roomId === room.id && piece.columns.length > 0)
                .map((piece) => (
                  <option key={piece.id} value={piece.id}>
                    {piece.name}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
      </label>
      <FloorPlan home={home} openId={openFurnitureId} onOpen={open} />
      <p className="hint">Click a piece on the plan to open it. The thick edge is its front.</p>
    </div>
  );
}

function FloorPlan({ home, openId, onOpen }: { home: Home; openId: string | null; onOpen: (id: string) => void }) {
  const width = Math.max(...home.rooms.map((room) => room.x + room.w));
  const depth = Math.max(...home.rooms.map((room) => room.y + room.d));
  const pad = 6;
  const roomOf = (piece: Furniture) => home.rooms.find((room) => room.id === piece.roomId);
  const footprints = home.furniture.flatMap((piece) => {
    const room = roomOf(piece);
    return room ? [footprint(piece, room)] : [];
  });

  return (
    <svg
      className="floor-plan"
      viewBox={`${-pad} ${-pad} ${width + 2 * pad} ${depth + 2 * pad}`}
      role="group"
      aria-label="Floor plan of the suite"
    >
      {home.rooms.map((room) => (
        <g key={room.id}>
          <rect className="plan-room" x={room.x} y={room.y} width={room.w} height={room.d} />
          <text
            className="plan-room-name"
            {...roomNameSpot(room, footprints)}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={LABEL_SIZE}
          >
            {room.name}
          </text>
        </g>
      ))}
      {home.furniture.map((piece) => {
        const room = roomOf(piece);
        if (!room) return null;
        return <PlanPiece key={piece.id} piece={piece} room={room} isOpen={piece.id === openId} onOpen={onOpen} />;
      })}
    </svg>
  );
}

function PlanPiece({ piece, room, isOpen, onOpen }: { piece: Furniture; room: Room; isOpen: boolean; onOpen: (id: string) => void }) {
  const box = footprint(piece, room);
  const hasStorage = piece.columns.length > 0;
  const tall = box.h > box.w;
  const centre = { x: box.x + box.w / 2, y: box.y + box.h / 2 };
  // Shrink the name to fit along the longer side; leave it out if that makes it unreadable.
  const labelSize = Math.min(LABEL_SIZE, Math.max(box.w, box.h) / (piece.name.length * 0.52));
  const fits = labelSize >= SMALLEST_LABEL && Math.min(box.w, box.h) >= labelSize;

  const interactive = hasStorage
    ? {
        role: 'button',
        tabIndex: 0,
        'aria-pressed': isOpen,
        onClick: () => onOpen(piece.id),
        onKeyDown: (event: React.KeyboardEvent) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onOpen(piece.id);
          }
        },
      }
    : {};

  return (
    <g className={cls('plan-piece', hasStorage && 'has-storage', isOpen && 'is-open')} {...interactive}>
      <title>{hasStorage ? `Open ${piece.name}` : piece.name}</title>
      <rect x={box.x} y={box.y} width={box.w} height={box.h} />
      {hasStorage && <line className="plan-front" {...frontEdge(piece, box)} />}
      {fits && (
        <text
          x={centre.x}
          y={centre.y}
          fontSize={labelSize}
          textAnchor="middle"
          dominantBaseline="central"
          transform={tall ? `rotate(-90 ${centre.x} ${centre.y})` : undefined}
        >
          {piece.name}
        </text>
      )}
    </g>
  );
}
