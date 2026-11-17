/*
 * Small pieces shared by the screens: a length input, a two-step button, and
 * a few helpers for drawing.
 */

import { useEffect, useRef, useState, type ChangeEvent, type KeyboardEvent, type ReactNode } from 'react';
import { formatLength, parseLength } from '../model/units';

/** Joins class names, skipping anything false. */
export function cls(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(' ');
}

/** Black or white, whichever reads better on the given '#rrggbb' colour. */
export function readableOn(colour: string): string {
  const [r, g, b] = [1, 3, 5].map((start) => parseInt(colour.slice(start, start + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#000000' : '#ffffff';
}

/** Shortens text to fit a width in pixels, for SVG text, which cannot wrap or clip itself. */
export function fitText(text: string, width: number, pixelsPerCharacter = 6): string {
  const room = Math.floor(width / pixelsPerCharacter);
  if (text.length <= room) return text;
  return room < 2 ? '' : `${text.slice(0, room - 1)}…`;
}

/** The size of an element, kept up to date as the layout changes. */
export function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, size] as const;
}

/**
 * What the two fields below share: the text being typed is a draft that is
 * committed on Enter or on leaving the field. If `commit` returns a reason,
 * the stored value comes back and the reason is shown under the field.
 */
function useDraft(stored: string, commit: (text: string) => string | null) {
  const [text, setText] = useState(stored);
  const [problem, setProblem] = useState<string | null>(null);

  // When the stored value changes (this edit or another one), show it.
  useEffect(() => setText(stored), [stored]);

  const finish = () => {
    const reason = text.trim() === stored ? null : commit(text);
    setProblem(reason);
    if (reason !== null) setText(stored);
  };

  return {
    problem,
    inputProps: {
      type: 'text',
      value: text,
      'aria-invalid': problem !== null,
      onChange: (event: ChangeEvent<HTMLInputElement>) => setText(event.target.value),
      onBlur: finish,
      onKeyDown: (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') event.currentTarget.blur();
      },
    },
  };
}

interface LengthFieldProps {
  id: string;
  label: string;
  /** The current length, in centimetres. */
  value: number;
  /** Called with the new length in centimetres. Return the reason if it is refused, or null. */
  onCommit: (cm: number) => string | null;
  /** Thicknesses may be zero; sizes may not. */
  allowZero?: boolean;
}

/** An input for a length. It shows inches and accepts things like '15 1/2' or '40 cm'. */
export function LengthField({ id, label, value, onCommit, allowZero = false }: LengthFieldProps) {
  const draft = useDraft(formatLength(value), (text) => {
    const cm = parseLength(text, allowZero);
    return cm === null ? 'Type a length, such as 15.5 or 40 cm.' : onCommit(cm);
  });

  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <span className="length-input">
        <input id={id} inputMode="decimal" {...draft.inputProps} />
        <span className="unit">in</span>
      </span>
      {draft.problem && <span className="field-problem">{draft.problem}</span>}
    </label>
  );
}

interface TextFieldProps {
  id: string;
  label: string;
  value: string;
  /** Called with the new text. Return the reason if it is refused, or null. */
  onCommit: (text: string) => string | null;
}

/** An input for a name. */
export function TextField({ id, label, value, onCommit }: TextFieldProps) {
  const draft = useDraft(value, onCommit);
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input id={id} {...draft.inputProps} />
      {draft.problem && <span className="field-problem">{draft.problem}</span>}
    </label>
  );
}

interface TwoStepButtonProps {
  className?: string;
  /** What the button says once it has been pressed and is waiting for the second press. */
  confirmLabel: string;
  onConfirm: () => void;
  children: ReactNode;
}

/** A button for something destructive: the first press arms it, the second does it. */
export function TwoStepButton({ className, confirmLabel, onConfirm, children }: TwoStepButtonProps) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const timer = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(timer);
  }, [armed]);

  return (
    <button
      type="button"
      className={cls('button', className, armed && 'is-armed')}
      onClick={() => {
        if (armed) onConfirm();
        setArmed(!armed);
      }}
    >
      {armed ? confirmLabel : children}
    </button>
  );
}
