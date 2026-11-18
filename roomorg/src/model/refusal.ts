/**
 * Thrown when an edit is not allowed, for example a shelf that would end up
 * with no room. The message is written for the person using the app.
 * Any other kind of error is a bug.
 */
export class Refusal extends Error {}
