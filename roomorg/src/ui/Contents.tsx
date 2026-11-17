/*
 * Draws the organizers and clothing inside one compartment or one organizer,
 * as SVG, at the positions worked out by arrange() in src/model/layout.ts.
 */

import type { KeyboardEvent } from 'react';
import { flatOutline } from '../model/folding';
import type { Placed } from '../model/layout';
import { childrenOf } from '../model/tree';
import type { Item, Organizer } from '../model/types';
import { useStore } from '../state/store';
import { TRAY_ZONE, useDrag, type Draggable } from './drag';
import { cls, fitText, readableOn } from './parts';

interface ContentsProps {
  /** Where each thing goes, from arrange(). */
  placed: Placed[];
  /** Where the bottom-left corner of their space is in the drawing, in pixels. */
  originX: number;
  originY: number;
  /** Pixels per centimetre. */
  scale: number;
  /** The right edge of the drawing. Things that overflow further are pulled back to it, so none is lost off-screen. */
  maxRight: number;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** The colour that stands in for an organizer in the drag ghost and the tray. */
export const ORGANIZER_COLOUR = '#b7a58c';

export function Contents({ placed, originX, originY, scale, maxRight }: ContentsProps) {
  const { index } = useStore();

  const toRect = (p: Placed): Rect => {
    const w = p.w * scale;
    return {
      x: Math.max(originX, Math.min(originX + p.x * scale, maxRight - w)),
      y: originY - (p.y + p.h) * scale,
      w,
      h: p.h * scale,
    };
  };

  return (
    <g>
      {placed.map((p) => {
        const organizer = index.organizers.get(p.id);
        const item = index.items.get(p.id);
        if (organizer) return <OrganizerShape key={p.id} organizer={organizer} rect={toRect(p)} fits={p.fits} />;
        if (item) return <ItemShape key={p.id} item={item} rect={toRect(p)} fits={p.fits} />;
        return null;
      })}
    </g>
  );
}

/**
 * What every drawn thing shares: selecting, dragging, keyboard use and the hover text.
 * `zone` is the drop zone it sits in: its parent's id, or TRAY_ZONE.
 */
export function useThing(thing: Draggable, zone: string, fits = true) {
  const { state, dispatch } = useStore();
  const { drag, startDrag } = useDrag();
  const selected = state.selected?.id === thing.id;
  return {
    className: cls('thing', selected && 'is-selected', !fits && 'is-misfit', drag?.id === thing.id && 'is-dragging'),
    props: {
      'data-thing': thing.id,
      'data-child-of': zone,
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

function MisfitMark({ rect }: { rect: Rect }) {
  return (
    <text className="misfit-mark" x={rect.x + rect.w / 2} y={rect.y - 3} textAnchor="middle">
      !
    </text>
  );
}

function OrganizerShape({ organizer, rect, fits }: { organizer: Organizer; rect: Rect; fits: boolean }) {
  const { state } = useStore();
  const { drag } = useDrag();
  const thing = useThing(
    { id: organizer.id, kind: 'organizer', label: organizer.name, colour: ORGANIZER_COLOUR },
    organizer.parentId ?? TRAY_ZONE,
    fits,
  );

  // One stripe per item inside, so you can see what a closed box holds.
  const items = childrenOf(state.home, organizer.id).flatMap((child) => (child.kind === 'item' ? [child.item] : []));
  const labelHeight = rect.h >= 26 && rect.w >= 30 ? 14 : 0;
  const stripes = { x: rect.x + 3, y: rect.y + labelHeight + 2, w: rect.w - 6, h: rect.h - labelHeight - 5 };
  const stripeWidth = items.length > 0 ? stripes.w / items.length : 0;

  return (
    <g
      className={cls(thing.className, 'organizer', drag?.overZone === organizer.id && 'is-over')}
      data-drop={organizer.id}
      aria-label={`Box: ${organizer.name}, ${items.length} inside`}
      {...thing.props}
    >
      <title>{`${thing.hoverText}: ${items.length} inside`}</title>
      <rect className="organizer-body" x={rect.x} y={rect.y} width={rect.w} height={rect.h} rx={2} />
      {stripes.h > 2 &&
        items.map((item, i) => (
          <rect key={item.id} x={stripes.x + i * stripeWidth} y={stripes.y} width={stripeWidth} height={stripes.h} fill={item.colour} />
        ))}
      {labelHeight > 0 && (
        <text className="thing-label" x={rect.x + rect.w / 2} y={rect.y + 11} textAnchor="middle">
          {fitText(organizer.name, rect.w - 6)}
        </text>
      )}
      {!fits && <MisfitMark rect={rect} />}
    </g>
  );
}

function ItemShape({ item, rect, fits }: { item: Item; rect: Rect; fits: boolean }) {
  const thing = useThing(
    { id: item.id, kind: 'item', label: item.name, colour: item.colour },
    item.parentId ?? TRAY_ZONE,
    fits,
  );
  const ink = readableOn(item.colour);
  const showOutline = rect.w >= 16 && rect.h >= 16;
  const showName = rect.w >= 44 && rect.h >= 14;

  return (
    <g className={cls(thing.className, 'item')} aria-label={`Clothing: ${item.name}`} {...thing.props}>
      <title>{thing.hoverText}</title>
      <rect className="item-body" x={rect.x} y={rect.y} width={rect.w} height={rect.h} fill={item.colour} />
      {showOutline && <ShapeOutline item={item} rect={rect} stroke={ink} />}
      {showName && (
        <text x={rect.x + rect.w / 2} y={rect.y + rect.h / 2 + 4} textAnchor="middle" fill={ink} className="thing-label">
          {fitText(item.name, rect.w - 6)}
        </text>
      )}
      {!fits && <MisfitMark rect={rect} />}
    </g>
  );
}

/** The item's laid-flat shape, shrunk to sit on the face of its stored box. */
export function ShapeOutline({ item, rect, stroke }: { item: Pick<Item, 'parts' | 'fold'>; rect: Rect; stroke: string }) {
  const outline = flatOutline(item);
  if (!(outline.w > 0) || !(outline.l > 0)) return null;
  const scale = Math.min((rect.w - 6) / outline.w, (rect.h - 6) / outline.l);
  if (scale <= 0) return null;
  const left = rect.x + (rect.w - outline.w * scale) / 2;
  const top = rect.y + (rect.h - outline.l * scale) / 2;
  return (
    <g className="shape-outline" stroke={stroke}>
      {outline.rects.map((r, i) => (
        <rect key={i} x={left + r.x * scale} y={top + r.y * scale} width={r.w * scale} height={r.h * scale} />
      ))}
    </g>
  );
}
