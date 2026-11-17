import { expect, test } from 'vitest';
import { formatLength, parseLength, toDisplay, toDisplaySize } from '../src/model/units';

test('lengths are shown in inches', () => {
  expect(toDisplay(39.37)).toBe('15.5 in');
  expect(toDisplay(2.54)).toBe('1 in');
  expect(toDisplaySize(97.79, 30.226)).toBe('38.5 × 11.9 in');
  expect(formatLength(30.226)).toBe('11.9');
});

test('typed lengths are read as inches unless they say cm', () => {
  expect(parseLength('15.5')).toBeCloseTo(39.37);
  expect(parseLength(' 15.5 in ')).toBeCloseTo(39.37);
  expect(parseLength('15.5"')).toBeCloseTo(39.37);
  expect(parseLength('15 1/2')).toBeCloseTo(39.37);
  expect(parseLength('1/2')).toBeCloseTo(1.27);
  expect(parseLength('.5')).toBeCloseTo(1.27);
  expect(parseLength('40 cm')).toBe(40);
  expect(parseLength('40cm')).toBe(40);
});

test('anything that is not a length above zero is refused', () => {
  for (const text of ['', 'abc', '0', '-3', '1/0', '12 ft', '1.2.3']) expect(parseLength(text), text).toBeNull();
});

test('what is shown can be typed back in', () => {
  for (const cm of [30.226, 97.79, 57.5, 1.8]) expect(parseLength(formatLength(cm))!).toBeCloseTo(cm, 1);
});
