/*
 * The floor, as a strip along the bottom of the screen. Things with no place
 * yet are on the floor. The strip stays visible whichever piece of furniture
 * is open, so it is also how you carry something from one piece to another.
 * The same things are drawn on the floor in front of the open piece.
 */

import { childId, childrenOf } from '../model/tree';
import { useStore } from '../state/store';
import { FLOOR_ZONE, overClass, useDrag } from './drag';
import { cls } from './parts';
import { ThingChip } from './ThingChip';

export function FloorStrip() {
  const { state } = useStore();
  const { drag } = useDrag();
  const children = childrenOf(state.home, null);

  return (
    <section className={cls('floor-strip', overClass(drag, FLOOR_ZONE))} data-drop={FLOOR_ZONE} aria-label="On the floor">
      <h2>
        On the floor <span className="count">{children.length}</span>
      </h2>
      <div className="floor-things">
        {children.length === 0 && (
          <p className="hint">Nothing is on the floor. Set things down here to carry them to another piece of furniture.</p>
        )}
        {children.map((child) => (
          <ThingChip key={childId(child)} child={child} zone={FLOOR_ZONE} counted />
        ))}
      </div>
    </section>
  );
}
