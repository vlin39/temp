# Skill Tree Planner

A basic page that turns a JSON file into a skill tree and back. It shows which pieces of a project depend on which, lets you edit the tree, and exports the edited JSON.

`roomorg.json` in this folder is the skill tree for the organizer in `../roomorg/`.

## Open it

Double-click `index.html`. Nothing needs installing.

- Opened that way, the page starts with a three-node example. Use **Load JSON** (or drop a file on the page) to open `roomorg.json`.
- Served over http (for example `python3 -m http.server` in the repo root, then `/tree/index.html`), it loads `roomorg.json` by itself.
- Your edits are remembered in the browser, so a reload keeps them. The exported file is the real copy.

## Use it

| To do this | Do this |
| --- | --- |
| See what a node needs and unlocks | Select its card. What it needs gets a dashed outline, what it unlocks a solid one, and everything else fades. |
| See the whole tree | **Fit**. The **−** and **+** buttons zoom. |
| Edit a node | Select it, then change its id, title, description, branch, size, done state or needs in the panel. |
| Add or delete a node | **Add node** adds one to the selected node's branch. **Delete node** asks for a second click. |
| Edit branches | With nothing selected, the panel lists the branches: rename, add, or remove an empty one. |
| Save | **Export JSON** downloads the file. **JSON text** at the bottom shows the same text to copy, and can load pasted text. |

Card colours show state: done, ready (everything it needs is done), or locked.

A prerequisite that would create a loop is refused, with the loop spelled out. A file that is broken is refused with the reason, and the tree on screen stays as it was.

## File format

```json
{
  "title": "Roomorg skill tree",
  "version": 1,
  "groups": [
    { "id": "T", "name": "Trunk" }
  ],
  "nodes": [
    {
      "id": "T2",
      "title": "Furniture picker",
      "description": "A dropdown plus a simple clickable floor plan.",
      "group": "T",
      "size": "S",
      "done": false,
      "needs": ["T1"]
    }
  ]
}
```

- `groups` are the branches, in top-to-bottom display order.
- `needs` lists the ids a node depends on.
- `size` is `S`, `M`, `L` or `XL`.
- Only `id`, `title` and `group` are required on a node; the rest are filled in when the file loads.
- Exporting and loading the result always gives the same tree.

In `roomorg.json`, the `START` node is a milestone that needs the trunk (T1 to T8). Every branch node needs `START`, so "needs the first iteration" is one arrow.

## How the code is laid out

| File | What it holds |
| --- | --- |
| `index.html` | The page shell: toolbar, tree area, edit panel, JSON text box. |
| `tree.css` | Styling, with light and dark colours. |
| `tree.js` | Part 1 is the tree logic: plain functions that never touch the page. Part 2 is the page: drawing, the edit panel, loading and exporting. |
| `tree.test.js` | Tests for part 1. |
| `roomorg.json` | This project's tree. |

The main functions in `tree.js`:

- `parseTree(text)` reads and checks a file: unique ids, known groups, known prerequisites, no loops.
- `findLoop(nodes)` is a depth-first search that returns the chain of a loop, or nothing.
- `layoutTree(tree)` places the cards. A node's column is how many prerequisites deep it is; each branch is a horizontal band; nodes sharing a band and column stack downward.
- `allNeeds(id)` and `allUnlocks(id)` follow the arrows backward and forward, for tracing.
- `addNode`, `updateNode`, `setPrerequisites`, `removeNode` and the branch edits each return a new tree or throw an error with the reason. The page shows the reason and leaves the tree alone.
- `serializeTree(tree)` writes the JSON with fields always in the same order.
- `render()` redraws everything from three values: the tree, the selected id and the zoom.

Known limits: the layout is simple, so arrows can cross or run behind cards on a busy tree, and there is no undo. Export before large edits.

## Run the tests

```bash
node --test tree/tree.test.js
```
