/*
 * Draws boxes, bags and clothing as SVG, at the positions worked out by
 * src/model/layout.ts. Used for the things in a compartment, in an open
 * drawer, on the floor, and inside a box or bag.
 *
 * Each thing is a block: a front face, plus a top and a right side when the
 * view has depth (see view.ts). `placed` must be in painting order, furthest
 * first, which is the order layout.ts returns.
 */

import type { KeyboardEvent, ReactNode } from 'react';
import { flatOutline } from '../model/folding';
import type { Placed } from '../model/layout';
import { childrenOf } from '../model/tree';
import type { Item, Organizer } from '../model/types';
import { useStore } from '../state/store';
import { FLOOR_ZONE, overClass, useDrag, type Draggable } from './drag';
import { cls, fitText, readableOn } from './parts';
import { toPixels, type View } from './view';

interface ContentsProps {
  /** Where each thing goes, in painting order. */
  placed: Placed[];
  /** A view that measures from the front-bottom-left corner of the space the things are in. */
  view: View;
  /** Things that overflow further right than this (in cm) are pulled back, so none is lost off-screen. */
  maxRight?: number;
  /** The id of an SVG clip path. Things that fit are behind the opening, so they are cut off at its edges. */
  clipId?: string;
  /** False where a drop cannot pick a position among the things (the floor), so it goes last. */
  ordered?: boolean;
}

/** One thing's place on the screen: the top-left corner of its front face, its size, and how far its back is shifted. */
interface Block {
  x: number;
  y: number;
  w: number;
  h: number;
  dx: number;
  dy: number;
}

/** The colour that stands in for a box or a bag where its real look is not drawn. */
export const ORGANIZER_COLOURS = { box: '#b7a58c', bag: '#7f95ad' };

export function Contents({ placed, view, maxRight = Infinity, clipId, ordered = true }: ContentsProps) {
  const { index } = useStore();

  const blockOf = (p: Placed): Block => {
    const x = Math.max(0, Math.min(p.x, maxRight - p.w));
    const corner = toPixels(view, x, p.y + p.h, p.z);
    return {
      ...corner,
      w: p.w * view.scale,
      h: p.h * view.scale,
      dx: p.d * view.kx * view.scale,
      dy: p.d * view.ky * view.scale,
    };
  };

  const draw = (p: Placed) => {
    const organizer = index.organizers.get(p.id);
    const item = index.items.get(p.id);
    if (organizer) return <OrganizerShape key={p.id} organizer={organizer} block={blockOf(p)} fits={p.fits} ordered={ordered} />;
    if (item) return <ItemShape key={p.id} item={item} block={blockOf(p)} fits={p.fits} ordered={ordered} />;
    return null;
  };

  if (!clipId) return <g>{placed.map(draw)}</g>;
  return (
    <>
      <g clipPath={`url(#${clipId})`}>{placed.filter((p) => p.fits).map(draw)}</g>
      <g>{placed.filter((p) => !p.fits).map(draw)}</g>
    </>
  );
}

/**
 * What every drawn thing shares: selecting, dragging, keyboard use and the hover text.
 * `zone` is the drop zone it sits in: its parent's id, or FLOOR_ZONE. Pass
 * `ordered: false` where a drop should not count this copy of the thing when
 * working out the order: on the floor, or where the thing is also drawn elsewhere.
 */
export function useThing(thing: Draggable, zone: string, { fits = true, ordered = true } = {}) {
  const { state, dispatch } = useStore();
  const { drag, startDrag } = useDrag();
  const selected = state.selected?.id === thing.id;
  return {
    className: cls('thing', selected && 'is-selected', !fits && 'is-misfit', drag?.id === thing.id && 'is-dragging'),
    props: {
      'data-thing': thing.id,
      'data-child-of': ordered ? zone : undefined,
      role: 'button',
      tabIndex: 0,
      'aria-pressed': selected,
      onPointerDown: (event: React.PointerEvent) => startDrag(event, thing),
      onKeyDown: (event: KeyboardEvent) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          dispatch({ type: 'select', selected: { kind: thing.kind, id: thing.id } });
        }
      },
    },
    hoverText: fits ? thing.label : `${thing.label} (does not fit here)`,
  };
}

/** A point on a block's top face: `across` and `back` each run from 0 to 1. */
const onTop = (block: Block, across: number, back: number) =>
  `${across * block.w + back * block.dx},${-back * block.dy}`;

/** The top and right side of a block. Nothing is drawn in a flat view. */
function DepthFaces({ block, className, fill }: { block: Block; className: string; fill?: string }) {
  if (block.dx === 0 && block.dy === 0) return null;
  const top = [onTop(block, 0, 0), onTop(block, 1, 0), onTop(block, 1, 1), onTop(block, 0, 1)].join(' ');
  const side = `${block.w},0 ${block.w + block.dx},${-block.dy} ${block.w + block.dx},${block.h - block.dy} ${block.w},${block.h}`;
  return (
    <>
      <polygon className={className} points={top} fill={fill} />
      <polygon className="tint tint-light" points={top} />
      <polygon className={className} points={side} fill={fill} />
      <polygon className="tint tint-dark" points={side} />
    </>
  );
}

interface ShellProps {
  block: Block;
  fits: boolean;
  className: string;
  label: string;
  hoverText: string;
  /** Extra attributes for the outer group: the dragging and selecting props, and a drop zone for organizers. */
  groupProps: Record<string, unknown>;
  children: ReactNode;
}

/**
 * The outer group of a drawn thing. It is moved into place with a CSS
 * transform, so a thing glides when its neighbours change, and its contents
 * are drawn from (0, 0) at the top-left of its front face.
 */
function Shell({ block, fits, className, label, hoverText, groupProps, children }: ShellProps) {
  return (
    <g className={className} aria-label={label} style={{ transform: `translate(${block.x}px, ${block.y}px)` }} {...groupProps}>
      <title>{hoverText}</title>
      <g className="thing-body">
        {children}
        {!fits && (
          <text className="misfit-mark" x={block.w / 2} y={-block.dy - 3} textAnchor="middle">
            !
          </text>
        )}
      </g>
    </g>
  );
}

interface ShapeProps {
  block: Block;
  fits: boolean;
  ordered: boolean;
}

function OrganizerShape({ organizer, block, fits, ordered }: ShapeProps & { organizer: Organizer }) {
  const { state } = useStore();
  const { drag } = useDrag();
  const isBag = organizer.kind === 'bag';
  const thing = useThing(
    { id: organizer.id, kind: 'organizer', label: organizer.name, colour: ORGANIZER_COLOURS[organizer.kind] },
    organizer.parentId ?? FLOOR_ZONE,
    { fits, ordered },
  );
  const bodyClass = isBag ? 'bag-body' : 'organizer-body';

  // One stripe per thing inside, so you can see what it holds without opening it.
  const colours = childrenOf(state.home, organizer.id).map((child) =>
    child.kind === 'item' ? child.item.colour : ORGANIZER_COLOURS[child.organizer.kind],
  );
  const labelHeight = block.h >= 26 && block.w >= 30 ? 14 : 0;
  const stripes = { x: 3, y: labelHeight + 2, w: block.w - 6, h: block.h - labelHeight - 5 };
  const stripeWidth = colours.length > 0 ? stripes.w / colours.length : 0;

  // A box is open at the top: its contents show there too.
  const rim = { across: Math.min(0.2, 3 / block.w), back: 0.12 };
  const opening = (from: number, to: number) =>
    [onTop(block, from, rim.back), onTop(block, to, rim.back), onTop(block, to, 1 - rim.back), onTop(block, from, 1 - rim.back)].join(' ');
  const openWidth = 1 - 2 * rim.across;
  const hasDepth = block.dx !== 0 || block.dy !== 0;

  return (
    <Shell
      block={block}
      fits={fits}
      className={cls(thing.className, 'organizer', isBag && 'bag', overClass(drag, organizer.id))}
      label={`${isBag ? 'Bag' : 'Box'}: ${organizer.name}, ${colours.length} inside`}
      hoverText={`${thing.hoverText}: ${colours.length} inside`}
      groupProps={{ ...thing.props, 'data-drop': organizer.id }}
    >
      <DepthFaces block={block} className={bodyClass} />
      {hasDepth && !isBag && (
        <>
          <polygon className="box-opening" points={opening(rim.across, 1 - rim.across)} />
          {colours.map((colour, i) => (
            <polygon
              key={i}
              className="box-filling"
              points={opening(rim.across + (i / colours.length) * openWidth, rim.across + ((i + 1) / colours.length) * openWidth)}
              fill={colour}
            />
          ))}
        </>
      )}
      <rect className={bodyClass} width={block.w} height={block.h} rx={isBag ? Math.min(6, block.h / 4) : 2} />
      {stripes.h > 2 &&
        colours.map((colour, i) => (
          <rect key={i} className="stripe" x={stripes.x + i * stripeWidth} y={stripes.y} width={stripeWidth} height={stripes.h} fill={colour} />
        ))}
      {isBag && (
        // A carrying strap, so a bag does not read as a box.
        <path
          className="bag-strap"
          d={`M${block.w * 0.28},1 C${block.w * 0.28},${-Math.min(block.h * 0.45, 14)} ${block.w * 0.72},${-Math.min(block.h * 0.45, 14)} ${block.w * 0.72},1`}
        />
      )}
      {labelHeight > 0 && (
        <text className="thing-label" x={block.w / 2} y={11} textAnchor="middle">
          {fitText(organizer.name, block.w - 6)}
        </text>
      )}
    </Shell>
  );
}

function ItemShape({ item, block, fits, ordered }: ShapeProps & { item: Item }) {
  const thing = useThing(
    { id: item.id, kind: 'item', label: item.name, colour: item.colour },
    item.parentId ?? FLOOR_ZONE,
    { fits, ordered },
  );
  const ink = readableOn(item.colour);
  const showOutline = block.w >= 16 && block.h >= 16;
  const showName = block.w >= 44 && block.h >= 14;

  return (
    <Shell
      block={block}
      fits={fits}
      className={cls(thing.className, 'item')}
      label={`Clothing: ${item.name}`}
      hoverText={thing.hoverText}
      groupProps={thing.props}
    >
      <DepthFaces block={block} className="item-body" fill={item.colour} />
      <rect className="item-body" width={block.w} height={block.h} fill={item.colour} />
      {showOutline && <ShapeOutline item={item} width={block.w} height={block.h} stroke={ink} />}
      {showName && (
        <text x={block.w / 2} y={block.h / 2 + 4} textAnchor="middle" fill={ink} className="thing-label">
          {fitText(item.name, block.w - 6)}
        </text>
      )}
    </Shell>
  );
}

/** The item's laid-flat shape, shrunk to sit on the front of its stored block. */
function ShapeOutline({ item, width, height, stroke }: { item: Pick<Item, 'parts' | 'fold'>; width: number; height: number; stroke: string }) {
  const outline = flatOutline(item);
  if (!(outline.w > 0) || !(outline.l > 0)) return null;
  const scale = Math.min((width - 6) / outline.w, (height - 6) / outline.l);
  if (scale <= 0) return null;
  const left = (width - outline.w * scale) / 2;
  const top = (height - outline.l * scale) / 2;
  return (
    <g className="shape-outline" stroke={stroke}>
      {outline.rects.map((r, i) => (
        <rect key={i} x={left + r.x * scale} y={top + r.y * scale} width={r.w * scale} height={r.h * scale} />
      ))}
    </g>
  );
}
