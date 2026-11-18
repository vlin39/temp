/*
 * The parts of a piece of furniture that open: doors and drawers. Clicking
 * one opens or closes it. Also the front of a fixture (a sink), which never
 * opens.
 *
 * What is behind a closed door or in a closed drawer is not drawn and cannot
 * be dropped onto: FurnitureView only draws compartments that are reachable.
 */

import type { KeyboardEvent, ReactNode } from 'react';
import { arrange } from '../model/layout';
import { doorRect, openingsOf, usableSpace } from '../model/furniture';
import { setAllOpen, toggleOpen } from '../model/tree';
import type { Box, Cell, Door, Furniture, Id } from '../model/types';
import { useStore } from '../state/store';
import { Contents } from './Contents';
import { overClass, useDrag } from './drag';
import { cls } from './parts';
import { faces, frontRect, movedTo, polygon, type Corner, type View } from './view';

/** A door wider than this is a pair of doors that meet in the middle. */
const DOUBLE_DOOR_FROM = 50;

/**
 * Where the free edge of an open door ends up, as shares of the door's width:
 * how far across from its hinge, and how far out toward you. A door hinged on
 * the left stands almost straight out; one hinged on the right swings wider,
 * so that from this angle neither hides its own compartment.
 */
const SWING = {
  left: { across: -0.17, out: 0.98 },
  right: { across: 0.71, out: 0.71 },
};

/** One door panel. `x`, `y`, `w` and `h` are its closed position on the front of the piece. */
export interface Leaf {
  x: number;
  y: number;
  w: number;
  h: number;
  hinge: 'left' | 'right';
}

/** The panels of a door: one, hinged on the side nearer the edge of the piece, or a pair. */
export function leavesOf(piece: Furniture, door: Door): Leaf[] {
  const rect = doorRect(piece, door);
  if (rect.w > DOUBLE_DOOR_FROM) {
    const half = rect.w / 2;
    return [
      { ...rect, w: half, hinge: 'left' },
      { ...rect, x: rect.x + half, w: half, hinge: 'right' },
    ];
  }
  return [{ ...rect, hinge: rect.x + rect.w / 2 <= piece.outer.w / 2 ? 'left' : 'right' }];
}

/** The corners of a door panel standing open. */
export function openLeaf(leaf: Leaf): Corner[] {
  const hingeX = leaf.hinge === 'left' ? leaf.x : leaf.x + leaf.w;
  const swing = SWING[leaf.hinge];
  const tipX = hingeX + swing.across * leaf.w;
  const tipZ = -swing.out * leaf.w;
  return [
    [hingeX, leaf.y, 0],
    [hingeX, leaf.y + leaf.h, 0],
    [tipX, leaf.y + leaf.h, tipZ],
    [tipX, leaf.y, tipZ],
  ];
}

/** An SVG group that works as a button: click, Enter or Space. */
function Pressable({ label, className, onPress, children }: { label: string; className: string; onPress: () => void; children: ReactNode }) {
  return (
    <g
      className={className}
      role="button"
      tabIndex={0}
      aria-label={label}
      onClick={onPress}
      onKeyDown={(event: KeyboardEvent) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onPress();
        }
      }}
    >
      <title>{label}</title>
      {children}
    </g>
  );
}

/** What the drawings of a compartment share: its class (selected, lit up, refusing) and selecting it on a click. */
export function useCell(cell: Cell) {
  const { state, dispatch } = useStore();
  const { drag } = useDrag();
  const selected = state.selected?.kind === 'cell' && state.selected.id === cell.id;
  return {
    className: cls('cell', selected && 'is-selected', overClass(drag, cell.id)),
    select: () => dispatch({ type: 'select', selected: { kind: 'cell', id: cell.id } }),
  };
}

function useToggle(id: Id) {
  const { state, edit } = useStore();
  return { open: state.home.open.includes(id), toggle: () => edit((home) => toggleOpen(home, id)) };
}

/** A door, closed across its compartments or standing open. */
export function DoorShape({ piece, door, view }: { piece: Furniture; door: Door; view: View }) {
  const { open, toggle } = useToggle(door.id);
  const leaves = leavesOf(piece, door);

  return (
    <Pressable className={cls('door', open && 'is-open')} label={`${open ? 'Close' : 'Open'}: ${door.name}`} onPress={toggle}>
      {leaves.map((leaf) => {
        if (open) return <polygon key={leaf.hinge} className="door-leaf" points={polygon(view, openLeaf(leaf))} />;
        const rect = frontRect(view, { ...leaf, z: 0 });
        const inset = Math.min(6, rect.width / 6);
        const knobX = leaf.hinge === 'left' ? rect.x + rect.width - 7 : rect.x + 7;
        return (
          <g key={leaf.hinge}>
            <rect className="door-leaf" {...rect} />
            <rect className="door-trim" x={rect.x + inset} y={rect.y + inset} width={rect.width - 2 * inset} height={rect.height - 2 * inset} />
            <circle className="knob" cx={knobX} cy={rect.y + rect.height / 2} r={2.5} />
          </g>
        );
      })}
    </Pressable>
  );
}

interface DrawerProps {
  cell: Cell;
  /** The compartment's clear space, measured from the piece's front-bottom-left corner. */
  space: Box;
  view: View;
}

/**
 * A drawer. Closed, it is a front panel. Open, it is pulled all the way out
 * and seen from above: its contents can be dragged in and out, and its near
 * walls are see-through so nothing is hidden behind them. Press the handle
 * to close it.
 */
export function DrawerShape({ cell, space, view }: DrawerProps) {
  const { state, index } = useStore();
  const { className, select } = useCell(cell);
  const { open, toggle } = useToggle(cell.id);
  const front = (z: number) => frontRect(view, { ...space, z });
  const handle = (rect: ReturnType<typeof front>) => ({
    x: rect.x + rect.width * 0.3,
    y: rect.y + rect.height / 2 - 3,
    width: rect.width * 0.4,
    height: 6,
    rx: 3,
  });

  if (!open) {
    const rect = front(0);
    return (
      <Pressable className="drawer" label={`Open: ${cell.name}`} onPress={toggle}>
        <rect className="door-leaf" {...rect} />
        <rect className="handle" {...handle(rect)} />
      </Pressable>
    );
  }

  const box = { ...space, z: -space.d };
  const sides = faces(box);
  const usable = usableSpace(cell, space);
  const rect = front(box.z);

  return (
    <g className="drawer is-open">
      <g className={className} data-drop={cell.id} onClick={select}>
        <title>{cell.name}</title>
        <polygon className="drawer-inside inside-back" points={polygon(view, sides.back)} />
        <polygon className="drawer-inside inside-floor" points={polygon(view, sides.bottom)} />
        <polygon className="drawer-inside inside-wall" points={polygon(view, sides.left)} />
      </g>
      <Contents
        placed={arrange(state.home, cell.id, index)}
        view={movedTo(view, usable.x, usable.y, box.z + cell.drawerWall)}
        maxRight={space.w * 1.6}
      />
      <polygon className="drawer-glass" points={polygon(view, sides.right)} />
      <rect className="drawer-glass" {...rect} />
      <Pressable className="drawer-handle" label={`Close: ${cell.name}`} onPress={toggle}>
        <rect className="handle" {...handle(rect)} />
        {/* A larger invisible area around the handle, so it is easy to press. */}
        <rect className="handle-reach" x={rect.x + rect.width * 0.2} y={rect.y + rect.height / 2 - 9} width={rect.width * 0.6} height={18} />
      </Pressable>
    </g>
  );
}

/** The fixed front of a compartment that stores nothing. Clicking it selects the compartment, to edit its size. */
export function FixtureFront({ cell, space, view }: DrawerProps) {
  const { className, select } = useCell(cell);
  return (
    <g className={cls('fixture', className)} onClick={select}>
      <title>{`${cell.name}: stores nothing`}</title>
      <rect className="fixture-front" {...frontRect(view, { ...space, z: 0 })} />
    </g>
  );
}

/** Opens every door and drawer of a piece, or closes them all. Nothing is shown for a piece that has none. */
export function OpenAllButton({ piece }: { piece: Furniture }) {
  const { state, edit } = useStore();
  const ids = openingsOf(piece);
  if (ids.length === 0) return null;
  const allOpen = ids.every((id) => state.home.open.includes(id));
  return (
    <button id="open-all" type="button" className="button button-small" onClick={() => edit((home) => setAllOpen(home, piece, !allOpen))}>
      {allOpen ? 'Close all' : 'Open all'}
    </button>
  );
}
