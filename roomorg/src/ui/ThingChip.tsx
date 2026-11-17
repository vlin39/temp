/*
 * A named chip for an organizer or a clothing item. Chips appear wherever
 * things are listed: the unsorted tray, the selected compartment and an open
 * box. They can be clicked to select and dragged like the drawn shapes, which
 * matters for items too thin to grab in the drawing.
 */

import type { Child } from '../model/tree';
import { ORGANIZER_COLOUR, useThing } from './Contents';
import { cls } from './parts';

interface ThingChipProps {
  child: Child;
  /** The drop zone the thing sits in: its parent's id, or TRAY_ZONE. */
  zone: string;
  /**
   * True when this chip is the thing's only appearance on screen, so a drop
   * should count it when working out the order. False when the thing is also
   * drawn in a picture.
   */
  counted: boolean;
}

export function ThingChip({ child, zone, counted }: ThingChipProps) {
  const draggable =
    child.kind === 'item'
      ? { id: child.item.id, kind: 'item' as const, label: child.item.name, colour: child.item.colour }
      : { id: child.organizer.id, kind: 'organizer' as const, label: child.organizer.name, colour: ORGANIZER_COLOUR };
  const thing = useThing(draggable, zone);
  const props = counted ? thing.props : { ...thing.props, 'data-child-of': undefined };

  return (
    <div
      className={cls(thing.className, 'chip', child.kind === 'organizer' && 'chip-box')}
      data-drop={child.kind === 'organizer' ? child.organizer.id : undefined}
      {...props}
    >
      <span className="swatch" style={{ background: draggable.colour }} />
      {draggable.label}
      {child.kind === 'organizer' && <span className="chip-kind">box</span>}
    </div>
  );
}
