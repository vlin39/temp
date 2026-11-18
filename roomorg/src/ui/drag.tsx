/*
 * Picking things up and putting them down, with mouse or touch.
 *
 * Built on the browser's pointer events rather than a library, because the
 * things being dragged are SVG shapes as well as HTML.
 *
 * - Anything draggable calls `startDrag` from its onPointerDown.
 * - Anywhere something can be dropped carries a `data-drop` attribute holding
 *   the parent id (or FLOOR_ZONE for the floor). Add `data-append` where a
 *   drop cannot pick a position, so the thing goes last.
 * - Each drawn thing carries `data-thing` (its id) and `data-child-of` (the
 *   zone it sits in), which is how a drop works out its place in the order.
 *
 * While a thing is carried, a copy of it follows the pointer at its real size.
 * The place under it lights up, or shows that it is refused (a box where
 * there is no room). A press that does not move is a click, and selects.
 */

import { createContext, useContext, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { place } from '../model/place';
import { Refusal } from '../model/refusal';
import { canHold } from '../model/tree';
import type { Id } from '../model/types';
import { useStore } from '../state/store';

/** The zone name for the floor, whose parent id is null. */
export const FLOOR_ZONE = 'floor';

/** How far the pointer must travel before a press becomes a drag. */
const DRAG_THRESHOLD = 5;

/** Room around the carried copy for its outline. */
const GHOST_PADDING = 4;

export interface Draggable {
  id: Id;
  kind: 'organizer' | 'item';
  label: string;
  colour: string;
}

export interface DragState extends Draggable {
  /** The zone under the pointer that could hold the thing, if any. */
  overZone: string | null;
  /** True when that zone refuses the thing: a box where there is no room. */
  refused: boolean;
}

interface DragApi {
  drag: DragState | null;
  startDrag: (event: ReactPointerEvent, thing: Draggable) => void;
}

interface Target {
  zone: string;
  /** True where a drop goes last rather than to a position. */
  append: boolean;
}

const DragContext = createContext<DragApi | null>(null);

const parentOfZone = (zone: string): Id | null => (zone === FLOOR_ZONE ? null : zone);

/** The topmost zone under a point that may hold the thing. A thing is never its own zone. */
function zoneAt(x: number, y: number, thingId: Id, mayHold: (zone: string) => boolean): Target | null {
  for (const element of document.elementsFromPoint(x, y)) {
    const holder = element.closest<HTMLElement | SVGElement>('[data-drop]');
    const zone = holder?.dataset.drop;
    if (holder && zone !== undefined && zone !== thingId && mayHold(zone)) {
      return { zone, append: holder.dataset.append !== undefined };
    }
  }
  return null;
}

/** Where in a zone's left-to-right order a drop at `clientX` lands. */
function dropPosition(zone: string, clientX: number, thingId: Id): number {
  const siblings = [...document.querySelectorAll<HTMLElement | SVGElement>('[data-child-of]')].filter(
    (element) => element.dataset.childOf === zone && element.dataset.thing !== thingId,
  );
  return siblings.filter((element) => {
    const box = element.getBoundingClientRect();
    return box.left + box.width / 2 < clientX;
  }).length;
}

/**
 * A copy of the thing being carried, to follow the pointer. It is plain DOM
 * rather than React, so moving it does not redraw the page.
 */
function makeGhost(source: Element): { node: HTMLElement; left: number; top: number } {
  const copy = source.cloneNode(true) as Element;
  for (const element of [copy, ...copy.querySelectorAll('*')]) {
    for (const name of ['data-drop', 'data-thing', 'data-child-of', 'tabindex', 'style']) element.removeAttribute(name);
  }
  copy.classList.remove('is-selected');

  const node = document.createElement('div');
  node.className = 'drag-ghost';
  const rect = source.getBoundingClientRect();
  let padding = 0;

  if (source instanceof SVGGraphicsElement) {
    // An SVG shape only draws inside an <svg>, so wrap the copy in one that is just big enough.
    padding = GHOST_PADDING;
    const box = source.getBBox();
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', String(box.width + 2 * padding));
    svg.setAttribute('height', String(box.height + 2 * padding));
    svg.setAttribute('viewBox', `${box.x - padding} ${box.y - padding} ${box.width + 2 * padding} ${box.height + 2 * padding}`);
    svg.append(copy);
    node.append(svg);
  } else {
    node.append(copy);
  }
  document.body.append(node);
  return { node, left: rect.left - padding, top: rect.top - padding };
}

export function DragProvider({ children }: { children: ReactNode }) {
  const store = useStore();
  const [drag, setDrag] = useState<DragState | null>(null);

  // The window listeners below outlive the render that created them, so they
  // read the store through this ref to see the current home.
  const latest = useRef(store);
  latest.current = store;

  const startDrag = (event: ReactPointerEvent, thing: Draggable) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    const source = event.currentTarget;
    const start = { x: event.clientX, y: event.clientY };
    let ghost: ReturnType<typeof makeGhost> | null = null;
    let shown = { zone: null as string | null, refused: false };

    const mayHold = (zone: string) => canHold(latest.current.index, parentOfZone(zone), thing.id);

    /** Why the thing cannot be put in a zone, or null if it can. */
    const whyNot = (zone: string): string | null => {
      try {
        place(latest.current.state.home, thing.id, parentOfZone(zone));
        return null;
      } catch (error) {
        if (error instanceof Refusal) return error.message;
        throw error;
      }
    };

    const onMove = (e: PointerEvent) => {
      if (!ghost && Math.hypot(e.clientX - start.x, e.clientY - start.y) < DRAG_THRESHOLD) return;
      ghost ??= makeGhost(source);
      ghost.node.style.transform = `translate(${ghost.left + e.clientX - start.x}px, ${ghost.top + e.clientY - start.y}px)`;

      const zone = zoneAt(e.clientX, e.clientY, thing.id, mayHold)?.zone ?? null;
      const changed = zone !== shown.zone;
      if (changed) shown = { zone, refused: zone !== null && whyNot(zone) !== null };
      ghost.node.classList.toggle('is-refused', shown.refused);
      // The page only needs redrawing when the carried thing is first picked up or moves over a different place.
      if (changed || !ghost.node.dataset.shown) {
        ghost.node.dataset.shown = 'yes';
        setDrag({ ...thing, overZone: shown.zone, refused: shown.refused });
      }
    };

    const stop = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', stop);
      ghost?.node.remove();
      setDrag(null);
    };

    const onUp = (e: PointerEvent) => {
      const carried = ghost !== null;
      stop();
      const { dispatch, edit } = latest.current;
      if (!carried) {
        dispatch({ type: 'select', selected: { kind: thing.kind, id: thing.id } });
        return;
      }
      const target = zoneAt(e.clientX, e.clientY, thing.id, mayHold);
      if (!target) return; // dropped nowhere: it stays where it was
      const position = target.append ? undefined : dropPosition(target.zone, e.clientX, thing.id);
      edit((home) => place(home, thing.id, parentOfZone(target.zone), position));
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', stop);
  };

  return <DragContext value={{ drag, startDrag }}>{children}</DragContext>;
}

/** The class for a place while something is carried over it: lit up, or refusing. False when nothing is over it. */
export function overClass(drag: DragState | null, zone: string): 'is-over' | 'is-refused' | false {
  if (drag?.overZone !== zone) return false;
  return drag.refused ? 'is-refused' : 'is-over';
}

export function useDrag(): DragApi {
  const api = useContext(DragContext);
  if (!api) throw new Error('useDrag must be used inside DragProvider');
  return api;
}
