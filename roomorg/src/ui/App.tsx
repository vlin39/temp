/*
 * The page: a top bar, then the furniture picker, the front view of the open
 * piece, and a side panel for whatever is selected, with the unsorted tray
 * along the bottom.
 */

import { StoreProvider, useStore } from '../state/store';
import { AddBar } from './AddBar';
import { DragProvider } from './drag';
import { FurniturePicker } from './FurniturePicker';
import { FurnitureView } from './FurnitureView';
import { Header } from './Header';
import { ItemEditor } from './ItemEditor';
import { MeasurePanel } from './MeasurePanel';
import { UnsortedTray } from './UnsortedTray';
import { toDisplaySize } from '../model/units';

export function App() {
  return (
    <StoreProvider>
      <DragProvider>
        <Page />
      </DragProvider>
    </StoreProvider>
  );
}

function Page() {
  const { state } = useStore();
  const piece = state.home.furniture.find((candidate) => candidate.id === state.openFurnitureId);
  const editingThing = state.selected?.kind === 'item' || state.selected?.kind === 'organizer';

  return (
    <div className="app">
      <Header />
      <main className="layout">
        <aside className="card sidebar" aria-label="Furniture">
          <FurniturePicker />
        </aside>
        <section className="card stage" aria-label="Front view">
          <div className="stage-head">
            <h2>{piece?.name ?? 'No furniture open'}</h2>
            {piece && (
              <span className="stage-size">
                {toDisplaySize(piece.outer.w, piece.outer.h, piece.outer.d)} outside
                {piece.estimated && <span className="badge-inline">estimated</span>}
              </span>
            )}
          </div>
          <AddBar />
          <FurnitureView />
        </section>
        <aside className="card side-panel" aria-label="Details">
          {editingThing ? <ItemEditor /> : <MeasurePanel />}
        </aside>
      </main>
      <UnsortedTray />
    </div>
  );
}
