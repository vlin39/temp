/*
 * The app's state: the home (everything that is saved) plus what is open and
 * selected on screen. Every screen reads it from here and changes it through
 * `edit` or `tryEdit`, which run a function from src/model and either save the
 * new home or report why the change was refused.
 */

import { createContext, useContext, useEffect, useMemo, useReducer, type Dispatch, type ReactNode } from 'react';
import type { ResizeMode } from '../model/furniture';
import { Refusal } from '../model/refusal';
import { indexHome, type HomeIndex } from '../model/tree';
import type { Home, Id } from '../model/types';
import { loadHome, saveHome } from './persistence';

export type Selection = { kind: 'cell' | 'organizer' | 'item'; id: Id };

export interface Notice {
  text: string;
  tone: 'ok' | 'error';
}

export interface AppState {
  home: Home;
  openFurnitureId: Id | null;
  selected: Selection | null;
  resizeMode: ResizeMode;
  notice: Notice | null;
}

type Action =
  | { type: 'setHome'; home: Home; select?: Selection | null }
  | { type: 'open'; furnitureId: Id }
  | { type: 'select'; selected: Selection | null }
  | { type: 'setResizeMode'; mode: ResizeMode }
  | { type: 'notify'; notice: Notice | null };

/** What an edit hands back: the new home, optionally with what to select afterwards. */
export type EditResult = Home | { home: Home; select: Selection | null };

export interface Store {
  state: AppState;
  /** Lookups by id for the current home. */
  index: HomeIndex;
  dispatch: Dispatch<Action>;
  /** Runs an edit. Returns the reason if it was refused, or null if it was saved. */
  tryEdit: (change: (home: Home) => EditResult) => string | null;
  /** Runs an edit and shows the reason in the notice line if it was refused. */
  edit: (change: (home: Home) => EditResult) => boolean;
}

function exists(home: Home, selection: Selection | null): boolean {
  if (selection === null) return true;
  const index = indexHome(home);
  if (selection.kind === 'cell') return index.cells.has(selection.id);
  return selection.kind === 'organizer' ? index.organizers.has(selection.id) : index.items.has(selection.id);
}

/** The piece to show when nothing else is open. */
function firstWithStorage(home: Home): Id | null {
  return home.furniture.find((piece) => piece.columns.length > 0)?.id ?? null;
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'setHome': {
      const selected = action.select === undefined ? state.selected : action.select;
      const stillOpen = action.home.furniture.some((piece) => piece.id === state.openFurnitureId);
      return {
        ...state,
        home: action.home,
        openFurnitureId: stillOpen ? state.openFurnitureId : firstWithStorage(action.home),
        selected: exists(action.home, selected) ? selected : null,
        notice: null,
      };
    }
    case 'open':
      return { ...state, openFurnitureId: action.furnitureId, selected: null };
    case 'select':
      return { ...state, selected: action.selected };
    case 'setResizeMode':
      return { ...state, resizeMode: action.mode };
    case 'notify':
      return { ...state, notice: action.notice };
  }
}

function initialState(): AppState {
  const { home, warning } = loadHome();
  return {
    home,
    openFurnitureId: firstWithStorage(home),
    selected: null,
    resizeMode: 'keepOutside',
    notice: warning ? { text: warning, tone: 'error' } : null,
  };
}

const StoreContext = createContext<Store | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const index = useMemo(() => indexHome(state.home), [state.home]);

  useEffect(() => {
    if (!saveHome(state.home)) {
      dispatch({ type: 'notify', notice: { text: 'This browser could not save your changes. Download a backup.', tone: 'error' } });
    }
  }, [state.home]);

  const tryEdit = (change: (home: Home) => EditResult): string | null => {
    try {
      const result = change(state.home);
      dispatch('home' in result ? { type: 'setHome', home: result.home, select: result.select } : { type: 'setHome', home: result });
      return null;
    } catch (error) {
      if (error instanceof Refusal) return error.message;
      throw error;
    }
  };

  const edit = (change: (home: Home) => EditResult): boolean => {
    const reason = tryEdit(change);
    if (reason !== null) dispatch({ type: 'notify', notice: { text: reason, tone: 'error' } });
    return reason === null;
  };

  return <StoreContext value={{ state, index, dispatch, tryEdit, edit }}>{children}</StoreContext>;
}

export function useStore(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error('useStore must be used inside StoreProvider');
  return store;
}
