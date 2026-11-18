/*
 * The open piece of furniture, drawn to scale and at an angle: its panels and
 * compartments, its doors and drawers, everything in them, and the floor in
 * front of it with whatever has been set down there.
 *
 * It is painted from the back forward: the floor, the outside of the piece,
 * the inside of each compartment, the front edges of the panels, the things
 * on the shelves, then the doors and drawers, then the things on the floor.
 */

import { clearSpaces, doorRect, isReachable, panels, usableSpace } from '../model/furniture';
import { arrange, arrangeFloor } from '../model/layout';
import { cellHolding, childId, childrenOf } from '../model/tree';
import type { Box, Cell, Furniture } from '../model/types';
import { toDisplaySize } from '../model/units';
import { useStore } from '../state/store';
import { Contents } from './Contents';
import { FLOOR_ZONE, overClass, useDrag } from './drag';
import { DoorShape, DrawerShape, FixtureFront, leavesOf, openLeaf, useCell } from './Openings';
import { cls, fitText, useElementSize } from './parts';
import { ThingChip } from './ThingChip';
import { ANGLED, extent, faces, frontRect, movedTo, polygon, toPixels, type Corner, type View } from './view';

/** Room around the drawing so outlines and warning marks are not cut off. */
const MARGIN = 14;

/** How far past the piece's side overflowing things may be drawn, as a share of its width. */
const MAX_OVERHANG = 0.6;

/** The least floor shown in front of a piece, so there is always somewhere to set things down, and the most. */
const MIN_FLOOR_WIDTH = 90;
const MIN_FLOOR_DEPTH = 45;
const MAX_FLOOR_WIDTH = 240;

/**
 * The patch of floor things are set down on is far enough in front of the
 * piece that anything up to this tall does not hide the piece behind it.
 */
const FLOOR_CLEAR_HEIGHT = 34;

/** The back-left corner of that patch. It is also moved right, so on screen it sits straight below the piece. */
const FLOOR_CORNER = { x: (FLOOR_CLEAR_HEIGHT / ANGLED.ky) * ANGLED.kx, z: -FLOOR_CLEAR_HEIGHT / ANGLED.ky };

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
    return <p className="hint compartment-list">Click a compartment to list what is in it. Click a door or a drawer to open it.</p>;
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
  const { drag } = useDrag();
  const home = state.home;
  const { w: width, h: height, d: depth } = piece.outer;

  const spaces = clearSpaces(piece);
  const cells = piece.columns.flatMap((column) =>
    column.cells.map((cell, level) => {
      const space = spaces.get(cell.id)!;
      const reachable = isReachable(piece, cell, home.open);
      // Shelves and hanging spaces are drawn here. Drawers draw their own contents when open.
      const placed = reachable && cell.kind !== 'drawer' ? arrange(home, cell.id, index) : [];
      return { cell, space, usable: usableSpace(cell, space), reachable, placed, isTop: level === column.cells.length - 1 };
    }),
  );

  // Things that do not fit hang out past the piece's side. Make room to show them, up to a limit.
  const rightmost = Math.max(width, ...cells.flatMap(({ usable, placed }) => placed.map((p) => usable.x + p.x + p.w)));
  const drawnRight = Math.min(rightmost, width * (1 + MAX_OVERHANG));

  // A piece with one column has room beside it for compartment names; a wider piece labels inside.
  const gutter = piece.columns.length === 1 ? Math.min(170, Math.max(96, availableWidth * 0.26)) : 0;
  const room = { w: availableWidth - gutter - 2 * MARGIN, h: Math.max(320, window.innerHeight * 0.7) - 2 * MARGIN };

  /** Lays out the floor at a width, and finds the scale at which the whole scene fits the room there is. */
  const fitScene = (floorWidth: number) => {
    const floor = arrangeFloor(childrenOf(home, null), floorWidth);
    const floorDepth = Math.max(floor.depth, MIN_FLOOR_DEPTH);
    // Everything that can appear, with every door and drawer open, so the picture
    // keeps one size whichever of them are open.
    const bounds = extent(
      [
        [0, 0, 0],
        [drawnRight, height, 0],
        [0, height, depth],
        [width, height, depth],
        [FLOOR_CORNER.x, 0, FLOOR_CORNER.z - floorDepth],
        [FLOOR_CORNER.x + floorWidth, 0, FLOOR_CORNER.z],
        ...floor.placed.map((p): Corner => [FLOOR_CORNER.x + p.x + p.w, p.y + p.h, FLOOR_CORNER.z + p.z + p.d]),
        ...piece.doors.flatMap((door) => leavesOf(piece, door).flatMap(openLeaf)),
        ...cells.filter(({ cell }) => cell.kind === 'drawer').map(({ space }): Corner => [space.x, space.y, -space.d]),
      ],
      ANGLED,
    );
    const scale = Math.min(room.w / (bounds.right - bounds.left), room.h / (bounds.top - bounds.bottom));
    return { floor, floorWidth, floorDepth, bounds, scale };
  };

  // Start with a floor as wide as the piece. If that leaves room beside the piece (a tall, narrow closet),
  // widen the floor into it, so fewer things end up in front of one another.
  let scene = fitScene(Math.max(width, MIN_FLOOR_WIDTH));
  const widest = scene.bounds.left + room.w / scene.scale - FLOOR_CORNER.x;
  if (widest > scene.floorWidth) scene = fitScene(Math.min(widest, MAX_FLOOR_WIDTH));
  const { floor, floorWidth, floorDepth, bounds, scale } = scene;

  const view: View = {
    ...ANGLED,
    scale,
    originX: gutter + MARGIN - bounds.left * scale,
    originY: MARGIN + bounds.top * scale,
  };

  const outside = faces({ x: 0, y: 0, z: 0, w: width, h: height, d: depth });
  const floorView = movedTo(view, FLOOR_CORNER.x, 0, FLOOR_CORNER.z);
  const floorMiddle = toPixels(floorView, floorWidth / 2, 0, -floorDepth / 2);

  // Doors and drawers are painted from the bottom of the piece up, because
  // you look down on it: an open drawer hides part of the door below it.
  const openings = [
    ...piece.doors.map((door) => ({ bottom: doorRect(piece, door).y, shape: <DoorShape key={door.id} piece={piece} door={door} view={view} /> })),
    ...cells.flatMap(({ cell, space }) => {
      if (cell.kind === 'drawer') return [{ bottom: space.y, shape: <DrawerShape key={cell.id} cell={cell} space={space} view={view} /> }];
      if (cell.kind === 'fixture') return [{ bottom: space.y, shape: <FixtureFront key={cell.id} cell={cell} space={space} view={view} /> }];
      return [];
    }),
  ].sort((a, b) => a.bottom - b.bottom);

  return (
    <svg
      className="furniture-drawing"
      width={gutter + (bounds.right - bounds.left) * scale + 2 * MARGIN}
      height={(bounds.top - bounds.bottom) * scale + 2 * MARGIN}
      role="group"
      aria-label={`${piece.name}, seen from the front at an angle`}
    >
      <polygon
        className={cls('floor-zone', overClass(drag, FLOOR_ZONE))}
        data-drop={FLOOR_ZONE}
        data-append=""
        points={polygon(floorView, [[0, 0, 0], [floorWidth, 0, 0], [floorWidth, 0, -floorDepth], [0, 0, -floorDepth]])}
      />
      {floor.placed.length === 0 && (
        <text className="floor-hint" x={floorMiddle.x} y={floorMiddle.y + 4} textAnchor="middle">
          Floor: drag things here to set them down
        </text>
      )}

      <polygon className="piece-face piece-top" points={polygon(view, outside.top)} />
      <polygon className="piece-face piece-side" points={polygon(view, outside.right)} />
      <rect className="interior" {...frontRect(view, { x: 0, y: 0, z: 0, w: width, h: height })} />
      {cells.map(({ cell, space, isTop }) => cell.kind === 'fixture' && isTop && <Basin key={cell.id} space={space} top={height} depth={depth} view={view} />)}

      {cells.map(({ cell, space, reachable }) => (
        <g key={cell.id}>
          <clipPath id={`clip-${cell.id}`}>
            <rect {...frontRect(view, { ...space, z: 0 })} />
          </clipPath>
          {reachable && <Compartment cell={cell} space={space} view={view} labelInside={gutter === 0} />}
          {gutter > 0 && <GutterLabel cell={cell} space={space} view={view} right={gutter + MARGIN - 8} width={gutter - 4} />}
        </g>
      ))}

      {panels(piece).map((panel, i) => (
        <rect key={i} className={`panel panel-${panel.kind}`} {...frontRect(view, { ...panel, z: 0 })} />
      ))}

      {cells.map(({ cell, usable, placed }) => (
        <Contents
          key={cell.id}
          placed={placed}
          view={movedTo(view, usable.x, usable.y)}
          maxRight={drawnRight - usable.x}
          clipId={`clip-${cell.id}`}
        />
      ))}

      {openings.map((opening) => opening.shape)}

      <Contents placed={floor.placed} view={floorView} ordered={false} />
    </svg>
  );
}

interface CompartmentProps {
  cell: Cell;
  space: Box;
  view: View;
  /** True to write the compartment's name inside it. */
  labelInside: boolean;
}

/**
 * The inside of one open compartment: its back, floor and left wall, its rail
 * if it has one, and the area you can drop things on. A drawer's inside is
 * only the hole it slides into; the drawer itself is drawn by DrawerShape.
 */
function Compartment({ cell, space, view, labelInside }: CompartmentProps) {
  const { className, select } = useCell(cell);
  const opening = frontRect(view, { ...space, z: 0 });
  const walls = faces({ ...space, z: 0 });
  const isDrawer = cell.kind === 'drawer';
  const size = toDisplaySize(space.w, space.h);

  const railY = space.y + space.h - cell.railDrop;
  const railFrom = toPixels(view, space.x, railY, space.d / 2);
  const railTo = toPixels(view, space.x + space.w, railY, space.d / 2);

  return (
    <g className={isDrawer ? 'cell' : className}>
      <g clipPath={`url(#clip-${cell.id})`}>
        <polygon className="inside-back" points={polygon(view, walls.back)} />
        <polygon className="inside-floor" points={polygon(view, walls.bottom)} />
        <polygon className="inside-wall" points={polygon(view, walls.left)} />
        {cell.kind === 'hanging' && <line className="rail" x1={railFrom.x} y1={railFrom.y} x2={railTo.x} y2={railTo.y} />}
        {labelInside && !isDrawer && (
          <text className="cell-name" x={opening.x + 5} y={opening.y + 12}>
            {fitText(cell.name, opening.width - 10)}
          </text>
        )}
      </g>
      {!isDrawer && (
        <rect
          className="cell-zone"
          data-drop={cell.id}
          {...opening}
          onClick={select}
        >
          <title>{`${cell.name}: ${size} clear`}</title>
        </rect>
      )}
    </g>
  );
}

/** A compartment's name and clear size, written beside it. */
function GutterLabel({ cell, space, view, right, width }: { cell: Cell; space: Box; view: View; right: number; width: number }) {
  const middle = toPixels(view, 0, space.y + space.h / 2).y;
  return (
    <>
      <text className="cell-name" x={right} y={middle - 1} textAnchor="end">
        {fitText(cell.name, width)}
      </text>
      <text className="cell-size" x={right} y={middle + 12} textAnchor="end">
        {toDisplaySize(space.w, space.h)}
      </text>
    </>
  );
}

/** A basin and tap on top of a piece, above a compartment that is a sink. */
function Basin({ space, top, depth, view }: { space: Box; top: number; depth: number; view: View }) {
  const inset = Math.min(8, space.w / 6);
  const bowl: Corner[] = [
    [space.x + inset, top, depth * 0.18],
    [space.x + space.w - inset, top, depth * 0.18],
    [space.x + space.w - inset, top, depth * 0.72],
    [space.x + inset, top, depth * 0.72],
  ];
  const middle = space.x + space.w / 2;
  const tap: Corner[] = [
    [middle, top, depth * 0.86],
    [middle, top + 14, depth * 0.86],
    [middle, top + 14, depth * 0.6],
  ];
  return (
    <g className="basin">
      <polygon className="basin-bowl" points={polygon(view, bowl)} />
      <polyline className="basin-tap" points={polygon(view, tap)} />
    </g>
  );
}
