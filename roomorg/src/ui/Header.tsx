/*
 * The top bar: the app's name, backups, and the line that reports what just
 * happened or why something was refused.
 */

import type { ChangeEvent } from 'react';
import { createStarterHome } from '../model/roomData';
import { backupFileName, backupText, readBackup, saveFile } from '../state/persistence';
import { useStore } from '../state/store';
import { TwoStepButton, cls } from './parts';

export function Header() {
  const { state, dispatch } = useStore();
  const say = (text: string, tone: 'ok' | 'error') => dispatch({ type: 'notify', notice: { text, tone } });

  const download = async () => {
    const now = new Date();
    const stamped = { ...state.home, lastBackupAt: now.toISOString() };
    try {
      await saveFile(backupFileName(now), backupText(stamped));
    } catch (error) {
      const declined = (error as { code?: string }).code === 'declined';
      say(declined ? 'Backup cancelled.' : 'The backup could not be saved here.', declined ? 'ok' : 'error');
      return;
    }
    dispatch({ type: 'setHome', home: stamped });
    say('Backup downloaded.', 'ok');
  };

  const restore = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = ''; // so choosing the same file again still works
    if (!file) return;
    const result = await readBackup(file);
    if (!result.ok) {
      const more = result.problems.length > 1 ? ` (and ${result.problems.length - 1} more)` : '';
      say(`${file.name} could not be restored: ${result.problems[0]}${more} Nothing was changed.`, 'error');
      return;
    }
    dispatch({ type: 'setHome', home: result.home, select: null });
    say(`Restored ${file.name}.`, 'ok');
  };

  const startOver = () => {
    dispatch({ type: 'setHome', home: createStarterHome(), select: null });
    say('Back to the starter room.', 'ok');
  };

  const lastBackup = state.home.lastBackupAt
    ? new Date(state.home.lastBackupAt).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
    : 'never';

  return (
    <header className="header">
      <h1>Closet Map</h1>
      <div className="header-actions">
        <button id="download-backup" type="button" className="button" onClick={download}>
          Download backup
        </button>
        <input id="restore-input" className="visually-hidden" type="file" accept=".json,application/json" onChange={restore} />
        <label className="button" htmlFor="restore-input">
          Restore backup
        </label>
        <TwoStepButton confirmLabel="Click again to erase everything" onConfirm={startOver}>
          Start over
        </TwoStepButton>
        <span className="last-backup">Last backup: {lastBackup}</span>
      </div>
      <p id="notice" className={cls('notice', state.notice && `is-${state.notice.tone}`)} role="status" aria-live="polite">
        {state.notice?.text}
      </p>
    </header>
  );
}
