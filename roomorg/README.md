# Closet Map

A visual organizer for a real room, played like a tidying game. Each piece of furniture is drawn to scale from its measurements and seen at an angle. You open its doors and drawers, and move boxes, bags and clothing between shelves, rails, drawers and the floor. "Closet Map" is a working title; `roomorg` is only the folder name.

Two iterations are built, both from the skill tree in [`../tree/roomorg.json`](../tree/roomorg.json): the trunk (T1 to T8), then the tidy-up game (G1 to G6). Every later feature is a node in that tree.

![A closet seen at an angle with its doors open, boxes on its shelves, clothing on its rail, and a box and two bags on the floor in front](docs/screenshot.png)

## Run it

**In GitHub Codespaces (nothing to install).** On the repository page choose Code → Codespaces → Create codespace, wait for the editor to open, then in its terminal:

```bash
cd roomorg
```

```bash
npm install
```

```bash
npm run dev
```

Codespaces offers to open the page when the server starts. If the page does not load, stop the server with Ctrl-C and use the built version instead:

```bash
npm run build
```

```bash
npm run preview
```

**On your own computer.** Install Node.js (`brew install node` on a Mac), then run the same commands.

| Command | What it does |
| --- | --- |
| `npm run dev` | Starts the app with live reload |
| `npm test` | Runs the unit tests |
| `npm run typecheck` | Checks the types |
| `npm run build` | Type-checks and builds the static files into `dist/` |
| `npm run preview` | Serves the built files |

## Use it

| To do this | Do this |
| --- | --- |
| Open a piece of furniture | Pick it in the dropdown, or click it on the floor plan. The thick edge on the plan is its front. |
| Open or close a door | Click it. |
| Open or close a drawer | Click it to pull it out. Press its handle to push it back. |
| Open or close everything on a piece | **Open all** or **Close all**, above the drawing. |
| Add clothing, a box or a bag | Click a compartment, a box or a bag, choose a type or size, then **Add clothing**, **Add box** or **Add bag**. With nothing selected it goes on the floor. |
| Move something | Drag it onto a shelf, a rail, an open drawer, a box, a bag or the floor. Or select it and use the **Where** list. |
| Take something out | Drag it to the floor in front of the piece, or to the floor strip along the bottom. |
| Reorder a shelf | Drag a thing left or right along it. |
| Carry something to another piece | Set it on the floor, open the other piece, pick it up again. The floor comes with you. |
| Pick up something thin | Click its compartment and drag its chip from the row under the drawing. |
| Fold something | Select it and choose a fold. Putting it on a rail hangs it; taking it off restores its fold. |
| See what a box or bag holds | The bands of colour on it are its contents. Click it to open it. |
| Change a measurement | Type in the Measurements panel and press Enter. Lengths are in inches; `15 1/2` and `40 cm` also work. |

### What goes where

| This | Holds |
| --- | --- |
| A shelf, a hanging space, an open drawer | Boxes, bags and clothing |
| A box | Bags and clothing |
| A bag | Clothing |
| The floor | Anything |

- **A box needs room.** It is rigid, so it only goes where it fits. While you carry it over a place with no room, that place shows a red dashed outline, and letting go there leaves the box where it was.
- **Clothing and bags are soft.** They are never refused. If one does not fit where it is (too wide, too tall, too deep, or too long to hang) it gets a dashed red outline and a `!`.
- **Closed means out of reach.** You cannot drop anything behind a closed door or into a closed drawer. Moving something there with the **Where** list opens it for you.
- **Neighbours keep a gap.** Things on a shelf or the floor sit 2 cm apart; inside a box or bag they pack closer.

### Measurements

- Every piece has an outside size, panel thicknesses (sides, top, bottom, back), a shelf thickness, a divider thickness and a base height. The clear space in each compartment is worked out from those.
- You can type the inside width or depth instead, as a tape measure gives it, and the outside follows.
- A switch decides what gives when a compartment or thickness changes: the neighbouring compartment (the outside stays), or the outside (the other compartments stay). Use the second when typing in measured heights one after another.
- Pieces marked "estimated" have placeholder sizes. Only the two shelving units use real measurements so far.
- Which pieces have doors, and which compartments they cover, is set in `src/model/roomData.ts`. The closets' double doors are a guess from the sketch.

### Your data

- Every change is saved in the browser straight away.
- Browser storage belongs to one address on one device and can be cleared, so **Download backup** regularly. The backup file is the real copy. **Restore backup** loads one; a damaged file is refused with the reason and changes nothing.
- A room or backup saved by the first iteration still opens. It gains doors and the corrected sink unit; anything that was in the sink unit's old drawer is put on the floor.
- Nothing leaves your browser: there is no server, no account and no tracking.
- **Start over** returns to the starter room. It asks for a second click.

## How the code is laid out

Dependencies point one way: `ui` uses `state`, and both use `model`. `model` knows nothing about the browser or React, so it is easy to test and can be reused when the Three.js room view arrives.

### `src/model/`: the rules, in plain TypeScript

| File | What it holds |
| --- | --- |
| `types.ts` | The saved data: `Home`, `Room`, `Furniture`, `Column`, `Cell`, `Door`, `Organizer` (a box or a bag), `Item`. Flat lists that point at their parent by id. All lengths in cm. |
| `roomData.ts` | `createStarterHome()`: the suite, its furniture, compartments and doors, and the boxes from the organization table. |
| `furniture.ts` | `clearSpaces()` works out each compartment's space. `resizePiece`, `resizeCompartment`, `resizeColumn` and `setThickness` change measurements and keep the piece adding up. `doorRect`, `isReachable` and `openingsOf` cover doors and drawers. |
| `folding.ts` | `flatOutline()` is the laid-flat shape. `storedSize()` is the single block an item takes up for its fold. |
| `layout.ts` | `arrange()` decides where everything in a compartment, box or bag sits, in three dimensions, and what does not fit. `arrangeFloor()` does the same for the floor. |
| `tree.ts` | Where things are and what may hold what: `childrenOf`, `canHold`, `move`, `remove`, `addItem`, `addOrganizer`, `updateItem`, `updateOrganizer`, `toggleOpen`, `setAllOpen`, `reveal`, `pathTo`, `destinations`. |
| `place.ts` | `place()` is `move` plus the rule that a box needs room. It is what dragging, the Where list and the Add buttons use. |
| `parse.ts` | `parseHome()` checks a saved home or backup file before it is used, and brings an older file up to date. |
| `presets.ts` | Box and bag sizes, clothing types with starting sizes, fold options, gaps and limits. |
| `units.ts` | Inches for display, cm for storage; reading typed lengths. |
| `refusal.ts`, `ids.ts` | The error type for a refused edit; new ids. |

### `src/state/`: what is on screen, and saving

| File | What it holds |
| --- | --- |
| `store.tsx` | One store: the home, the open piece, the selection, the resize switch and the notice line. Screens change the home through `edit` or `tryEdit`. |
| `persistence.ts` | Autosave, loading, and backup files. |

### `src/ui/`: the screens, in React

| File | What it holds |
| --- | --- |
| `App.tsx` | The page layout. |
| `Header.tsx` | Backups and the notice line. |
| `FurniturePicker.tsx` | The dropdown and the floor plan. |
| `view.ts` | The angled view: turning a point in space into a point on the screen. |
| `FurnitureView.tsx` | The open piece, to scale and at an angle, with the floor in front, and the row of chips for the selected compartment. |
| `Openings.tsx` | Doors, drawers, the fixed front of the sink, and the Open all button. |
| `Contents.tsx` | Draws boxes, bags and clothing as blocks, wherever they are. |
| `MeasurePanel.tsx` | The measurements of the open piece. |
| `ItemEditor.tsx` | The side panel for a selected clothing item, box or bag. |
| `AddBar.tsx` | Adding clothing, boxes and bags. |
| `FloorStrip.tsx` | The floor, as a strip along the bottom. |
| `ThingChip.tsx` | A named, draggable chip for a box, a bag or an item. |
| `drag.tsx` | Picking things up and putting them down, with mouse or touch, built on pointer events. |
| `parts.tsx` | Shared pieces: the length and name fields, the two-step button, small helpers. |
| `ui.css` | Styling, with light and dark colours. |

## How it works

**Furniture adds up.** In each column, the compartment heights plus the shelves between them must equal the inside height; across a piece, the column widths plus the dividers must equal the inside width. Every measurement edit ends by restoring that rule with one of two small helpers (`giveToLargest`, `tradeWithNeighbour`), or it is refused.

**Folding keeps the fabric.** A garment laid flat is a body with optional sleeves or legs. Its fabric volume is the sum of each part's width × length × thickness. A fold divides the footprint, so the height rises to keep the volume the same. Sleeves and legs add volume, which makes a folded item thicker without making it wider.

**Arrangement is automatic.** You choose the place and the order; `arrange()` chooses the position. On a shelf, things run left to right with a gap between them, flush with the front, and loose clothing stacks into piles. On a rail, clothing hangs side by side and boxes sit on the floor underneath. On the floor, things sit in rows in front of the piece.

**The angle is an oblique projection.** A front face keeps its true shape, and anything further back is shifted right and up (`view.ts`). Every front face stays a plain rectangle, so sizes read true and shapes stay easy to click. The drawing is still SVG, painted from the back forward.

**Boxes need room; soft things do not.** `place()` moves a thing, then refuses the move if it left one more box that does not fit than before. Counting that way means a shelf that is already overfull does not block you from tidying it.

**Doors and drawers are a list.** `Home.open` holds the ids of whatever is open. A compartment is reachable when it is not a fixture, not a closed drawer and not behind a closed door; only reachable compartments are drawn with their contents and accept drops.

**Edits are refused, not broken.** Model functions return a new home or throw a `Refusal` with a sentence written for the person using the app. `tryEdit` in the store catches it, leaves the home untouched, and hands the sentence back to the field that asked.

**Saved files are checked whole.** `parseHome()` checks every record's shape, then that the records fit together (parents exist and may hold their contents, ids are unique, furniture adds up). A file that fails is refused with the reasons. A file from the first iteration is upgraded before it is checked.

## Tests

```bash
npm test
```

86 unit tests cover clear spaces and resizing, folding, arrangement with gaps and depth, the floor, what may hold what, boxes needing room, doors and drawers, moving and removing, saved-file checking and upgrading, and reading typed lengths.

The screens were checked with a scripted browser run of 74 steps. It opens every piece; makes each kind of measurement edit; drags between a box, a bag, a shelf, a rail, a drawer and the floor; checks that a full shelf refuses a box; opens and closes doors and drawers by click and by keyboard; folds; reloads; backs up, starts over and restores; and opens a room saved by the first iteration.

`npm audit --omit=dev` is clean: the app ships only React. `npm audit` reports one advisory, a slow-pattern bug in `picomatch`, which Vite and Vitest use to match file names while building and testing. No fixed release existed when this was written. It is not in the built app.

## Known limits

- The angle is fixed, and it is one piece at a time. The whole room from each corner, in Three.js, is node R1.
- You choose the place and the order, not the exact spot: things line up by themselves.
- A box or bag on a shelf shows its contents as bands of colour. To see or move the things inside, click it and use the panel.
- An open door or drawer can be in front of the shelf beside or below it. Close it to reach behind.
- Bags are one kind in three sizes. Backpacks, storage cubes and pouches as distinct kinds is node G7.
- Hung clothing and things laid flat are thin in the drawing; use the chips under the drawing to grab them.
- Furniture positions, facing, compartment layout and doors are fixed in `roomData.ts`. Editing them in the app is nodes R7 and R9.
- Pressing on a box or item with a finger starts a drag instead of scrolling the page. Scroll from the empty areas.
- Data lives in one browser. Syncing between devices is node D5.

## More

- [`docs/DESIGN.md`](docs/DESIGN.md): decisions, your feedback word for word, the pseudocode, and why each library was or was not used.
- [`../tree/`](../tree/): the skill-tree page and this project's tree.
