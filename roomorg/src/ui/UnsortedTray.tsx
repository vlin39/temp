/*
 * The unsorted tray: things that have no place yet. It stays visible at the
 * bottom of the screen, so it also works as a way to carry something from one
 * piece of furniture to another.
 */

import { childId, childrenOf } from '../model/tree';
import { useStore } from '../state/store';
import { TRAY_ZONE, useDrag } from './drag';
import { cls } from './parts';
import { ThingChip } from './ThingChip';

export function UnsortedTray() {
  const { state } = useStore();
  const { drag } = useDrag();
  const children = childrenOf(state.home, null);

  return (
    <section className={cls('tray', drag?.overZone === TRAY_ZONE && 'is-over')} data-drop={TRAY_ZONE} aria-label="Unsorted tray">
      <h2>
        Unsorted <span className="count">{children.length}</span>
      </h2>
      <div className="tray-things">
        {children.length === 0 && (
          <p className="hint">Things with no place yet wait here. Drag anything here to carry it to another piece of furniture.</p>
        )}
        {children.map((child) => (
          <ThingChip key={childId(child)} child={child} zone={TRAY_ZONE} counted />
        ))}
      </div>
    </section>
  );
}
