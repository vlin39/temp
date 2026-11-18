/*
 * A named chip for a box, a bag or a clothing item. Chips appear wherever
 * things are listed: the floor strip, the selected compartment and an open
 * box or bag. They can be clicked to select and dragged like the drawn shapes, which
 * matters for items too thin to grab in the drawing.
 */

import type { Child } from '../model/tree';
import { ORGANIZER_COLOURS, useThing } from './Contents';
import { cls } from './parts';

interface ThingChipProps {
  child: Child;
  /** The drop zone the thing sits in: its parent's id, or FLOOR_ZONE. */
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
      : { id: child.organizer.id, kind: 'organizer' as const, label: child.organizer.name, colour: ORGANIZER_COLOURS[child.organizer.kind] };
  const thing = useThing(draggable, zone, { ordered: counted });

  return (
    <div
      className={cls(thing.className, 'chip', child.kind === 'organizer' && 'chip-box')}
      data-drop={child.kind === 'organizer' ? child.organizer.id : undefined}
      {...thing.props}
    >
      <span className="swatch" style={{ background: draggable.colour }} />
      {draggable.label}
      {child.kind === 'organizer' && <span className="chip-kind">{child.organizer.kind}</span>}
    </div>
  );
}
