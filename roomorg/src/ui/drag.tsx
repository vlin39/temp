/*
 * Dragging organizers and clothing, with mouse or touch.
 *
 * Built on the browser's pointer events rather than a library, because the
 * things being dragged are SVG shapes as well as HTML.
 *
 * - Anything draggable calls `startDrag` from its onPointerDown.
 * - Anywhere something can be dropped carries a `data-drop` attribute holding
 *   the parent id (or TRAY_ZONE for the unsorted tray).
 * - Each drawn thing carries `data-thing` (its id) and `data-child-of` (the
 *   zone it sits in), which is how a drop works out its place in the order.
 *
 * A press that does not move is a click, and selects the thing.
 */

import { createContext, useContext, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react';
import { canHold, move } from '../model/tree';
import type { Id } from '../model/types';
import { useStore } from '../state/store';

/** The zone name for the unsorted tray, whose parent id is null. */
export const TRAY_ZONE = 'tray';

/** How far the pointer must travel before a press becomes a drag. */
const DRAG_THRESHOLD = 5;

export interface Draggable {
  id: Id;
  kind: 'organizer' | 'item';
  label: string;
  colour: string;
}

interface DragState extends Draggable {
  x: number;
  y: number;
  /** The zone under the pointer that would accept the thing, if any. */
  overZone: string | null;
}

interface DragApi {
  drag: DragState | null;
  startDrag: (event: ReactPointerEvent, thing: Draggable) => void;
}

const DragContext = createContext<DragApi | null>(null);

const parentOfZone = (zone: string): Id | null => (zone === TRAY_ZONE ? null : zone);

/** The topmost zone under a point that passes `accepts`. A thing is never its own zone. */
function zoneAt(x: number, y: number, thingId: Id, accepts: (zone: string) => boolean): string | null {
  for (const element of document.elementsFromPoint(x, y)) {
    const zone = element.closest<HTMLElement | SVGElement>('[data-drop]')?.dataset.drop;
    if (zone !== undefined && zone !== thingId && accepts(zone)) return zone;
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
    const start = { x: event.clientX, y: event.clientY };
    let moved = false;

    const accepts = (zone: string) => canHold(latest.current.index, parentOfZone(zone), thing.kind);

    const onMove = (e: PointerEvent) => {
      if (!moved && Math.hypot(e.clientX - start.x, e.clientY - start.y) < DRAG_THRESHOLD) return;
      moved = true;
      setDrag({ ...thing, x: e.clientX, y: e.clientY, overZone: zoneAt(e.clientX, e.clientY, thing.id, accepts) });
    };

    const stop = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', stop);
      setDrag(null);
    };

    const onUp = (e: PointerEvent) => {
      stop();
      if (!moved) {
        latest.current.dispatch({ type: 'select', selected: { kind: thing.kind, id: thing.id } });
        return;
      }
      const zone = zoneAt(e.clientX, e.clientY, thing.id, accepts);
      if (zone === null) return;
      const position = dropPosition(zone, e.clientX, thing.id);
      latest.current.edit((home) => move(home, thing.id, parentOfZone(zone), position));
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', stop);
  };

  return (
    <DragContext value={{ drag, startDrag }}>
      {children}
      {drag && (
        <div className="drag-ghost" style={{ left: drag.x, top: drag.y }}>
          <span className="swatch" style={{ background: drag.colour }} />
          {drag.label}
        </div>
      )}
    </DragContext>
  );
}

export function useDrag(): DragApi {
  const api = useContext(DragContext);
  if (!api) throw new Error('useDrag must be used inside DragProvider');
  return api;
}
