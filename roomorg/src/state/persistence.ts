/*
 * Saving and loading.
 *
 * The home autosaves in this browser on every change. Browser storage is a
 * convenience, not a safe: it can be cleared, and it belongs to one address
 * on one device. The backup file is the real copy.
 */

import { parseHome, type ParseResult } from '../model/parse';
import { createStarterHome } from '../model/roomData';
import type { Home } from '../model/types';

const SAVED_KEY = 'closetmap.home.v1';
const UNREADABLE_KEY = 'closetmap.home.unreadable';

/** The autosaved home, or the starter home. `warning` is set if a saved copy could not be read. */
export function loadHome(): { home: Home; warning: string | null } {
  let text: string | null = null;
  try {
    text = localStorage.getItem(SAVED_KEY);
  } catch {
    // Storage is unavailable (a private window, blocked site data). Start fresh.
  }
  if (text === null) return { home: createStarterHome(), warning: null };

  const result = parseHome(text);
  if (result.ok) return { home: result.home, warning: null };
  try {
    localStorage.setItem(UNREADABLE_KEY, text); // keep it, in case it can be rescued
  } catch {
    // Nothing more can be done here.
  }
  return {
    home: createStarterHome(),
    warning: 'Your saved room could not be read, so the starter room is shown. Restore a backup to get it back.',
  };
}

/** Autosaves. Returns false if the browser refused, so the screen can say so. */
export function saveHome(home: Home): boolean {
  try {
    localStorage.setItem(SAVED_KEY, JSON.stringify(home));
    return true;
  } catch {
    return false;
  }
}

export function backupText(home: Home): string {
  return JSON.stringify(home, null, 2) + '\n';
}

export function backupFileName(date: Date): string {
  return `closet-map-backup-${date.toISOString().slice(0, 10)}.json`;
}

/** Reads a backup file someone chose. */
export async function readBackup(file: File): Promise<ParseResult> {
  return parseHome(await file.text());
}

interface HostDownloads {
  save(request: { filename: string; data: string }): Promise<unknown>;
}

/**
 * Hands a text file to the person. In an ordinary browser tab this is a normal
 * download. Some hosts run the page in a sandbox where pages cannot download by
 * themselves and offer a save function instead; that is used when present.
 */
export async function saveFile(filename: string, text: string): Promise<void> {
  const host = (window as { claude?: { use?: (name: string) => Promise<HostDownloads | null> } }).claude;
  const downloads = typeof host?.use === 'function' ? await host.use('downloads') : null;
  if (downloads) {
    await downloads.save({ filename, data: text });
    return;
  }
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
