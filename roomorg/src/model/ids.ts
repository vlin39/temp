import type { Id } from './types';

/** A new unique id for something the person adds. */
export function newId(): Id {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  // Older browsers and pages opened without https.
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
