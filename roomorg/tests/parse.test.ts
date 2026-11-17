import { describe, expect, test } from 'vitest';
import { parseHome } from '../src/model/parse';
import { createStarterHome } from '../src/model/roomData';
import { addItem, move } from '../src/model/tree';
import type { Home } from '../src/model/types';

/** A home with something in every kind of place. */
function busyHome(): Home {
  let { home, id } = addItem(createStarterHome(), 'closet-b1-c1-4-box1', 'tshirt');
  ({ home } = addItem(home, 'closet-b1-c1-1', 'coat'));
  ({ home, id } = addItem(home, null, 'socks'));
  return move(home, id, 'closet-b1-c1-3');
}

const problemsOf = (change: (home: any) => void): string[] => {
  const home = JSON.parse(JSON.stringify(busyHome()));
  change(home);
  const result = parseHome(JSON.stringify(home));
  return result.ok ? [] : result.problems;
};

describe('saved homes', () => {
  test('saving and reading back gives an identical home', () => {
    const home = busyHome();
    const result = parseHome(JSON.stringify(home));
    expect(result).toEqual({ ok: true, home });
  });

  test('the starter home passes its own checks', () => {
    expect(parseHome(JSON.stringify(createStarterHome())).ok).toBe(true);
  });
});

describe('broken files are refused with a reason', () => {
  test('not JSON, not a home, wrong version', () => {
    expect(parseHome('{ nope')).toEqual({ ok: false, problems: ['The file is not valid JSON.'] });
    expect(parseHome('[1, 2]').ok).toBe(false);
    expect(problemsOf((home) => { home.version = 2; })[0]).toMatch(/version/);
    expect(problemsOf((home) => { delete home.items; })[0]).toMatch(/missing its list of items/);
  });

  test('a field of the wrong kind', () => {
    expect(problemsOf((home) => { home.items[0].colour = 'red'; })).toEqual(['Item 1 has a missing or wrong "colour".']);
    expect(problemsOf((home) => { home.items[0].fold = 'crumpled'; })[0]).toMatch(/"fold"/);
    expect(problemsOf((home) => { home.items[0].parts = []; })[0]).toMatch(/"parts"/);
    expect(problemsOf((home) => { home.organizers[0].outer.w = -1; })[0]).toMatch(/Box 1.*"outer"/);
    expect(problemsOf((home) => { home.furniture[0].facing = 'up'; })[0]).toMatch(/"facing"/);
    expect(problemsOf((home) => { home.furniture[0].columns[0].cells[0].height = 'tall'; })[0]).toMatch(/compartment.*"height"/);
  });

  test('records that do not fit together', () => {
    expect(problemsOf((home) => { home.items[0].parentId = 'nowhere'; })[0]).toMatch(/place that does not exist/);
    expect(problemsOf((home) => { home.organizers[0].parentId = 'nowhere'; })[0]).toMatch(/compartment that does not exist/);
    expect(problemsOf((home) => { home.furniture[0].roomId = 'attic'; })[0]).toMatch(/room that does not exist/);
    expect(problemsOf((home) => { home.items[1].id = home.items[0].id; })[0]).toMatch(/same id/);
    expect(problemsOf((home) => { home.furniture[0].columns[0].cells[0].height += 5; })[0]).toMatch(/do not add up/);
  });
});
