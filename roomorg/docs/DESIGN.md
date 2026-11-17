# Closet Map: design notes

"Closet Map" is a working title. `roomorg` is only the folder name.

This file records what was decided, what you asked for in your own words, the pseudocode the code follows, and why each library was or wasn't used. It is updated with each iteration. The README covers how to run the app and how the code is laid out.

## Decisions

| Decision | Choice | Who decided |
| --- | --- | --- |
| Room drawing (arrives with node R1) | Three.js, fixed camera at each of the 4 corners | You |
| Language and tooling | TypeScript + Vite | You |
| Menus and forms | React | You |
| First iteration | The trunk, T1 to T8: furniture front views, measurements, organizers, clothing, folding, saving | You, after two rounds of feedback |
| Plan format | A skill tree, kept in `../tree/roomorg.json` | You |

## Feedback log

Your feedback, word for word, so it does not get lost. New feedback is added at the end.

**Round 1**

> For the preliminary version, the furniture and room shouldn't be adjustable---though I want it to be easy to add that functionality in the future. What I want to move currently are mainly the organizers (the boxes on the shelves) and the clothing (which should be able to go in a box, on a shelf, or be hung up).
>
> Would it be easier to first focus on one primary view of each piece of furniture (from one angle) that appears when you click on that item in the room (or select it from a dropdown menu, or something else)?
>
> The folding should be a setting that can be adjusted per clothing---so there should be a default shape ( default box should be the clothing laid out flat.
>
> The temp repository is, as suggested, a temporary repository for now. I would not be linking this in a portfolio---the current goal is to get as quickly as I can to a working product (though yes, in steps/iteration versions instead of all at once) so I can use said working product to organize my room.
>
> I would like more detailed descriptions of the various iterations/future developments---and preferably individually (or in individual lists) instead of grouped as an overall iteration step. So...actually more like a skill tree in gaem! (Actually, would that be a good coding project? To make a way to visualize larger projects as skill trees?)
>
> I like the idea of using GitHub codespaces for now if that actually works.

**Round 2**

> To clarify, the furniture location and orientation should be able to be edited in the app at a later iteration. The current changes that should be allowed are adjusting the dimensions of, say, shelves in a cabinet or in a shelving unit. Or the overall cabinet itself. (Which reminds me, we should make sure the model accounts for various measurements in the furniture, including minor things like the fact a shelf has a thickness and so do the walls of things like cabinets.)
>
> For each item starting as a box---would depicting pant legs as connected but separate boxes (and the same with sleeves) be too complicated for a first step? Or perhaps would it be easier to give a visual impression of overlaying the shape on a rectangular prism/cuboid?
>
> For the planning skill tree, ideally add a separate folder in the repo named "tree". This should be a pretty basic tree visualizer that can go from a json to a tree and then back (it opens a webpage where you can load a json, edit the tree/graph, and then export the edited json). I mainly think it might be useful to show what later developments rely on a previous development (and if so, what---like prerequisites).

**Round 3**

> If plain HTML + JavaScript is functional enough, that would be great.
>
> I also want to double-check what the T stands for in things like T1, T2, etc.
>
> Please provide the answers (if plain HTML + Javascript is enough and what the T stands for) in-chat.

Answered: plain HTML + JavaScript is enough for the tree page, and T stands for "trunk", the first iteration that every branch grows from.

**Round 4**

> to clarify, if i accept right now, what am i accepting?

Answered: accepting covered building the tree page and the trunk only, on the branch, with nothing merged. Every later node waits for you to pick it.

**Round 5**

> without deleting anything currently in the plan, provide the pseudocode for both tree and roomorg

Answered: the pseudocode was added to the plan, which you then approved. The organizer's half is below; the tree page's half is summarised in `../tree/README.md`.

## The first iteration (trunk, T1 to T8)

These eight pieces are the trunk of the skill tree (T1–T8): the first iteration that every branch grows from. They ship together because none is useful alone. There is no Three.js yet; the corner room view is the recommended next unlock (R1).

| Node | What it adds |
| --- | --- |
| T1 Room data | Your suite, its furniture and their compartments in one data file, taken from your sketch and table. Which pieces exist, where they stand and which way they face are fixed for now. |
| T2 Furniture picker | A dropdown plus a simple clickable floor plan of the suite. Picking a piece opens its front view. |
| T3 Front view | A to-scale drawing of one piece from the front: panels, shelves, hanging rail, drawers, cabinets. |
| T4 Measurements | Edit the size of a whole piece and of its shelves and compartments. Panel, shelf and divider thickness are part of the model (details below). |
| T5 Organizers | Add a box (small, medium, large, or a typed size), name it, drag it to another shelf, delete it. A box shows coloured stripes for what is inside; clicking it opens it. |
| T6 Clothing | Add a clothing item with a name, a type and a colour. Put it in a box, on a shelf, or on a rail, and drag it between them. An always-visible "unsorted" tray holds things not yet placed and carries them between pieces of furniture. |
| T7 Shape and fold | Each item starts as its laid-flat shape. A per-item fold setting reshapes it: flat, halves, thirds, quarters, rolled, hung (details below). |
| T8 Saving | Autosaves in the browser, plus "download backup" and "restore backup" for one file. |

**Measurements (T4).**
- Every piece has an outside size, a panel thickness (sides, top, bottom, back), a shelf thickness, a divider thickness, and a base height for legs or a plinth.
- Each compartment's clear inside size is calculated from those, so nothing is typed twice.
- You can type an inside measurement, as a tape measure gives you inside a shelf, and the outside adjusts; or type the outside.
- Changing one compartment's height moves the shelf above it, and the neighbouring compartment absorbs the difference.
- A rail sits a set distance below the top of its compartment, and hung items start below it with a hanger allowance.
- Drawers and organizers follow the same rule: outside size and wall thickness give the inside size.
- Starter values use your shelving measurements; closet sizes are estimates until you measure.

**Shape and fold (T7).**
- The laid-flat shape is one to three connected boxes set by the item's type: a body with two sleeves, or hips with two legs. Each part's size is editable.
- Once folded, in a box or hung, the item is a single box. Folding divides its footprint and raises its height, so the amount of fabric stays the same; sleeves and legs just make the folded box thicker.
- A simple outline of the type (shirt, trousers, dress and so on) is drawn on the stored box so items are recognisable at closet scale.

**How things arrange themselves.** Placement is automatic; you choose where something goes and in what order, not exact positions.
- On a shelf, boxes and loose clothing line up left to right, and loose clothing stacks into piles.
- On a rail, items hang side by side.
- Dropping an item on a rail switches it to "hung"; taking it off restores its previous fold.
- Anything that doesn't fit, including after you shrink a compartment, is drawn with a warning outline and is not blocked.

**Easy to extend later.**
- Rooms and furniture are ordinary data rows with sizes, positions and facing, so moving and rotating them (R7) adds controls without changing saved data.
- The saved file carries a version number, so later nodes can add fields without breaking your data.
- Items have a `kind`, so non-clothing items can be added (C14).

### Files planned for the trunk

This is the table from the plan. The README lists the files as built. Later nodes add `src/render/` (Three.js, at R1), more files in `ui/`, and `model/search.ts` and `model/fields.ts`.

| File | Language | Main parts |
| --- | --- | --- |
| `src/model/types.ts` | TypeScript | `Home`, `Room`, `Furniture`, `Compartment`, `Organizer`, `Item`: flat tables with parent links, like database rows |
| `src/model/roomData.ts` | TypeScript | Your suite: rooms, furniture, compartments |
| `src/model/furniture.ts` | TypeScript | `clearSpaces(furniture)`: inside sizes from outside size and thicknesses; `resizePiece`, `resizeCompartment` |
| `src/model/presets.ts` | TypeScript | Organizer sizes, clothing types with default part sizes and outlines, fold options |
| `src/model/folding.ts` | TypeScript | `storedSize(item)`: flat parts + fold → one box |
| `src/model/layout.ts` | TypeScript | `arrange(parent)`: where each child is drawn, and what doesn't fit |
| `src/model/tree.ts` | TypeScript | `childrenOf`, `pathTo`, `move`, `remove` |
| `src/model/units.ts` | TypeScript | Inches ↔ cm (sizes are stored in cm) |
| `src/state/store.tsx` | TSX | One store for the home document and selection; actions `add`, `update`, `move`, `remove`, `resize` |
| `src/state/persistence.ts` | TypeScript | Autosave, load, backup download, backup restore with validation |
| `src/ui/App.tsx` | TSX | Page layout |
| `src/ui/FurniturePicker.tsx` | TSX | Dropdown and floor plan (SVG) |
| `src/ui/FurnitureView.tsx` | TSX | Front view (SVG): panels, shelves, compartments |
| `src/ui/ContentsLayer.tsx` | TSX | Draws organizers and clothing from `arrange()`; drag and drop |
| `src/ui/MeasurePanel.tsx` | TSX | Size and thickness inputs for the open piece and its compartments |
| `src/ui/UnsortedTray.tsx` | TSX | Holding area |
| `src/ui/ItemEditor.tsx` | TSX | Side panel: name, colour, type, part sizes, fold, flat preview; organizer name and size |
| `src/ui/ui.css` | CSS | Styling |
| `tests/*.test.ts` | TypeScript | Clear spaces, folding, layout, moving, backup round trip |
| `index.html`, `package.json`, `tsconfig.json`, `vite.config.ts` | HTML, JSON, TypeScript | Page shell and project config |
| `README.md`, `docs/DESIGN.md` | Markdown | How to run; design, pseudocode, feedback log |

Libraries for Start: React, TypeScript, Vite and its test runner. Whether drag and drop uses a library (needed for touch screens) is decided in the import check. Three.js arrives with R1.

## Pseudocode

Five choices in the pseudocode go slightly beyond the trunk description above:
- The tree file gets one milestone node, `START`, that needs the trunk nodes. Branch nodes need `START` instead of eight separate arrows.
- Resizing has a two-way switch: keep the outside size (the neighbour absorbs the change) or keep the other compartments (the piece grows or shrinks). The second is for typing in tape-measure numbers one after another.
- Deleting an organizer sends its contents to the unsorted tray. It never deletes them.
- The starter data includes the boxes named in your table (for example "Hand bags", large), placed on the shelves you listed.
- Each hanger takes at least 2.5 cm of rail by default. That is a preset to tune against your "30–36 items" figure once the closet is measured.

All lengths are stored in cm. Width runs left to right, height up, depth front to back.

**Data shapes (`model/types.ts`)**

```
Size3: { w, h, d }

Home:
    version:       1
    rooms:         list of Room
    furniture:     list of Furniture
    organizers:    list of Organizer
    items:         list of Item
    lastBackupAt:  date, or none

Room:       id, name, x, y (place on the suite plan), w, d (floor size)

Furniture:
    id, name, roomId
    x, y:                 back-left corner within the room            # fixed in the trunk
    facing:               "north" | "east" | "south" | "west"         # fixed in the trunk
    outer:                Size3
    panel:                { side, top, bottom, back }                 # thicknesses
    shelfThickness, dividerThickness
    baseHeight:           legs or plinth
    columns:              list of Column, left to right               # empty = no storage (bed, desk)
    estimated:            true until you edit a measurement

Column:     id, width (clear inside), cells: list of Cell, bottom to top

Cell:                     # one compartment
    id, name              # "Hats"
    kind:                 "shelf" | "hanging" | "drawer" | "cabinet"
    height:               clear inside height
    railDrop:             top of the cell down to the rail             # hanging only
    drawerWall:           wall thickness                               # drawer only

Organizer:                # a box, basket or bin
    id, name
    parentId:             a Cell id, or none (in the tray)
    order:                position among its siblings
    outer:                Size3
    wall:                 thickness

Item:
    id, kind: "clothing", name
    type:                 a clothing type id                           # "tshirt"
    colour:               "#rrggbb"
    parentId:             a Cell id, an Organizer id, or none (in the tray)
    order:                position among its siblings
    parts:                list of Part                                 # the laid-flat shape
    fold:                 "flat" | "half" | "thirds" | "quarters" | "rolled" | "hung"
    previousFold:         the fold to restore when it comes off a rail, or none

Part:
    role:                 "body" | "sleeve" | "leg"
    w, l, t:              across, down, thickness lying flat
    count:                1, or 2 for sleeves and legs
```

Rule that must always hold for every piece of furniture:

```
in each column:   sum of cell heights + (cells - 1) * shelfThickness
                  = outer.h - baseHeight - panel.top - panel.bottom
across columns:   sum of column widths + (columns - 1) * dividerThickness
                  = outer.w - 2 * panel.side
every cell's depth = outer.d - panel.back
```

**Furniture measurements (`model/furniture.ts`)**

```
function clearSpaces(f):                          # where each compartment's empty space is
    x = f.panel.side
    for each column:
        y = f.baseHeight + f.panel.bottom
        for each cell, bottom to top:
            space[cell.id] = { x, y, w: column.width, h: cell.height, d: f.outer.d - f.panel.back }
            y = y + cell.height + f.shelfThickness
        x = x + column.width + f.dividerThickness
    return space

function usableSpace(cell, space):
    drawer:   shrink by cell.drawerWall on the sides, front, back and bottom
    hanging:  same box; hang length available = space.h - cell.railDrop - HANGER_GAP
    else:     space unchanged

function panels(f):         the rectangles to draw: base, sides, top, bottom, each shelf, each divider

function checkPiece(f):     list of ways f breaks the rule above (empty = fine)

function rebalance(f, mode):                      # restore the rule after any change
    if mode is "keep outside":
        in each column, the tallest cell takes up the height difference
        the widest column takes up the width difference
    if mode is "keep compartments":
        outer.h and outer.w change to fit the tallest and the full set of columns
    refuse (change nothing, return the reason) if any cell or column would go below MIN_CLEAR

function resizePiece(f, dimension, value, measuredFrom):
    if measuredFrom is "inside": value = value + the panels on that dimension
    f.outer[dimension] = value
    rebalance(f, "keep outside")                  # the compartments adjust to the new outside
    f.estimated = false

function resizeCompartment(f, cell, newHeight, mode):
    delta = newHeight - cell.height
    cell.height = newHeight
    if mode is "keep outside":
        neighbour = the cell above, or the cell below if there is none above
        neighbour.height = neighbour.height - delta       # the shelf between them moves
        refuse if neighbour.height < MIN_CLEAR
    else:
        rebalance(f, "keep compartments")                 # the piece grows or shrinks
    f.estimated = false

function setThickness(f, which, value, mode):
    set it; rebalance(f, mode)

column widths resize the same way as cell heights, with the column to the right as neighbour
```

**Presets (`model/presets.ts`)**

```
ORGANIZER_SIZES:   small, medium, large -> outer Size3 and wall
                   sized so a closet shelf holds two large, or one large + one medium + one small
CLOTHING_TYPES:    list of { id, name, parts with default sizes, outline drawing, default fold }
                   t-shirt:   body + 2 short sleeves        trousers: hips + 2 legs
                   sweater:   body + 2 long thick sleeves   dress, skirt, shorts, jacket,
                   underwear, socks, scarf, other: body only
FOLD_DIVISORS:     half -> (across 1, down 2)   thirds -> (1, 3)   quarters -> (2, 2)
MIN_RAIL_SPACE = 2.5      HANGER_GAP = 10      MIN_CLEAR = 5      (cm)
```

**Shape and fold (`model/folding.ts`)**

```
function fabricVolume(item):
    sum over parts of  w * l * t * count

function flatOutline(item):                       # for drawing the laid-flat shape
    body rectangle; a sleeve rectangle on each side at the top; a leg rectangle under each half of the hips

function storedSize(item) -> Size3:               # the single box it takes up in storage
    body  = the body part
    coreW = body.w                                # sleeves fold in over the body
    coreL = body.l + leg length if it has legs    # legs lie together and continue downward
    V     = fabricVolume(item)

    if fold is "flat":
        w = body.w + 2 * sleeve.w if it has sleeves
        d = coreL
        h = thickest part's t
    if fold is "half", "thirds" or "quarters":
        (a, b) = FOLD_DIVISORS[fold]
        w = coreW / a
        d = coreL / b
        h = V / (w * d)                           # same fabric on a smaller footprint, so taller
    if fold is "rolled":
        w = coreW
        d = h = square root of (4 * V / (pi * coreW))
    if fold is "hung":
        w = larger of MIN_RAIL_SPACE and V / (coreW * coreL)     # space along the rail
        d = coreW                                 # shoulder width, front to back
        h = coreL                                 # how far it hangs down
    return { w, h, d }
```

Sleeves and legs add to `V`, so they make a folded item thicker without changing its footprint.

**Automatic arrangement (`model/layout.ts`)**

```
Placed: { id, kind, x, y, w, h, fits }            # a rectangle in the front view, from the space's bottom-left

function arrange(parent, home):
    space    = usableSpace of a cell, or an organizer's inside (outer minus walls and floor)
    children = organizers and items whose parentId is parent.id, sorted by order
    if parent is a hanging cell: return arrangeRail(space, children)
    return arrangeShelf(space, children)

function arrangeShelf(space, children):
    x = 0; pile = none
    for each child:
        size = child.outer if it is an organizer, else storedSize(child)
        if child is an item and pile exists
           and size.w <= pile.width and pile.height + size.h <= space.h:
            place it on the pile at (pile.x, pile.height)          # loose clothing stacks
            pile.height = pile.height + size.h
        else:
            place it at (x, 0)
            pile = { x, width: size.w, height: size.h } if child is an item, else none
            x = x + size.w
        fits = right edge <= space.w and top edge <= space.h and size.d <= space.d
    return the placed list

function arrangeRail(space, children):
    x = 0
    for each item:                                # hung side by side
        size = storedSize(item)                   # its fold is "hung" here
        place it at (x, space.h - railDrop - HANGER_GAP - size.h)
        x = x + size.w
        fits = x <= space.w and size.h <= hang length available and size.d <= space.d
    organizers sit on the floor of the cell, left to right; fits = within width and depth
    return the placed list
```

**Moving things (`model/tree.ts`)**

```
function childrenOf(home, parentId):   organizers and items with that parent, sorted by order
function pathTo(home, id):             room > furniture > compartment > (organizer) > item

function canHold(parent, child):
    a cell holds organizers and items
    an organizer holds items only
    the tray holds both

function move(home, childId, newParentId, index):
    require canHold(new parent, child)
    if child is an item:
        if the new parent is a hanging cell and fold is not "hung":
            previousFold = fold; fold = "hung"
        if the new parent is not a hanging cell and fold is "hung":
            fold = previousFold, or the type's default fold
    child.parentId = newParentId
    renumber order among the new siblings with child at index
    renumber order among the old siblings

function remove(home, id):
    an organizer: its items move to the tray, then the organizer is deleted
    an item:      deleted
```

**Units (`model/units.ts`)**

```
function toDisplay(cm):        inches, rounded to 0.1   -> "15.5 in"
function parseLength(text):    accepts "15.5", "15.5 in", "15 1/2", "40 cm"; returns cm, or none if not a positive number
```

**Starter data (`model/roomData.ts`)**

```
function createStarterHome():
    rooms:      bedroom, sink area, bathroom, placed as in your floor plan
    furniture:
        Closet A:        1 column; shelves named Shoes, Shoes, Sheets, Towels, TODO?, Mountain outer gear
        Closet B x 3:    1 column; a hanging cell at the bottom, 4 shelves above, named from your table
        Shelves C x 2:   1 column; 2 shelves; inside 38.5 x 15.5 in, each level 11.9 in    (your measurements)
        Shelves D:       1 column; 3 shelves
        Sink unit:       3 columns; side cabinets with 2 levels each; middle = drawer over the under-sink cabinet
        Wall cabinet:    1 column; 3 levels
        Bed, desk, tiny table, tiny chair:  no columns (shown on the floor plan only)
        every size not from your measurements has estimated = true
    organizers: one per boxed entry in your table, with its S/M/L size, on its shelf
    items:      none
```

**State (`state/store.tsx`)**

```
state:
    home
    openFurnitureId:  or none
    selected:         { kind, id }, or none
    resizeMode:       "keep outside" | "keep compartments"

function reducer(state, action) -> a new state; the old one is never edited
    openFurniture(id) / select(kind, id) / clearSelection / setResizeMode(mode)
    addOrganizer(parentId, preset)          addItem(parentId, typeId, name)
    updateOrganizer(id, changes)            updateItem(id, changes)       # name, colour, parts, fold, size, wall
    move(childId, newParentId, index)       -> tree.move
    remove(id)                              -> tree.remove
    resizePiece(...) / resizeCompartment(...) / setThickness(...)         -> furniture.*
    replaceHome(home)                       # restore a backup, or reset to the starter data
    a refused change leaves the state as it was and sets a message for the field

after any action that changed home: persistence.saveSoon(home)
a Provider component hands state and dispatch to every UI component
```

**Saving (`state/persistence.ts`)**

```
function loadHome():
    text = browser storage under KEY, inside try/catch
    if there is none: return createStarterHome()
    result = parseHome(text)
    if it failed: keep the bad text under a second key, show a warning, return createStarterHome()
    return result

function saveSoon(home):          300 ms after the last change, write home as JSON, inside try/catch
                                  on failure show "Couldn't save. Download a backup."

function downloadBackup(home):    download "roomorg-backup-<date>.json"; set lastBackupAt = now

function restoreBackup(file):     result = parseHome(text of file)
                                  if it failed: show the reason, change nothing
                                  else: dispatch replaceHome(result)

function parseHome(text):
    read as JSON; check version; upgrade older versions one step at a time
    check every list and field type
    check every roomId and parentId points at something that exists
    check every piece of furniture with checkPiece
    return home, or the list of problems
```

**Screens (`ui/`)**

```
App:
    top bar:   FurniturePicker, "Download backup", "Restore backup", "Last backup: ..."
    middle:    FurnitureView of the open piece
    side:      ItemEditor if an organizer or item is selected, else MeasurePanel
    bottom:    UnsortedTray

FurniturePicker:
    dropdown of every piece that has storage, grouped by room -> openFurniture(id)
    floor plan (SVG): each room as a rectangle; each piece as its footprint with a label and its front edge marked
                      click a piece -> openFurniture(id); the open one is highlighted

FurnitureView(piece):
    scale = fit outer.w x outer.h into the space available, same scale both ways
    draw panels(piece) as filled shapes, so thickness is visible
    for each cell:
        draw its clear space as a drop zone, with its name and clear size
        hanging cells draw the rail
        buttons: "+ box", "+ clothing"
        ContentsLayer(cell)

ContentsLayer(parent):
    for each Placed in arrange(parent, home):
        organizer: outline, name, and one coloured stripe per item inside
        item:      filled with its colour, the type's outline, and its name if it is wide enough
        fits is false -> dashed warning outline and a "!"
        click -> select it
    dragging:
        pointer down on a thing -> a ghost follows the pointer
        zones under the pointer that pass canHold light up (cells, organizers, the tray)
        on drop -> index from the pointer's position among the siblings -> dispatch move(...)

MeasurePanel(piece):
    outside W, H, D; inside W, D (linked to the outside through the thicknesses)
    thicknesses: sides, top, bottom, back, shelf, divider; base height
    per column: width (only if there is more than one)
    per cell: name, clear height, rail drop (hanging), wall (drawer)
    switch: "Changing a compartment keeps the outside size / keeps the other compartments"
    each field: parseLength on Enter or leaving the field -> dispatch
                invalid or refused -> reason shown under the field, old value kept
    an "estimated" badge until the piece is edited

UnsortedTray:
    a strip of the organizers and items with no parent; a drop zone; "+ box", "+ clothing"

ItemEditor(selection):
    item:      name, type, colour, part sizes (body; sleeves or legs), fold
               previews: the laid-flat shape from flatOutline, and the stored size from storedSize
               where it is, from pathTo; "Move to tray"; "Delete"
               changing the type asks before resetting the part sizes
    organizer: name, size preset or W, H, D and wall
               a larger ContentsLayer of its inside; a list of its items
               "Delete" (contents go to the tray)
```

**Tests (`tests/`)**

```
furniture:    Shelves C gives two clear spaces of 38.5 x 15.5 x 11.9 in
              heights + shelves + panels add up to the outside height
              "keep outside" leaves the outside unchanged; "keep compartments" changes it by the difference
              a change that would squeeze a neighbour below MIN_CLEAR is refused
folding:      fabric volume is the same for half, thirds and quarters
              "half" halves the depth and doubles the height
              adding sleeves makes the folded box taller, not wider
              a hung item never takes less than MIN_RAIL_SPACE
layout:       boxes run left to right; loose clothing stacks into piles
              anything past the width, height or depth is marked as not fitting
              a rail marks overflow and items too long to hang
tree:         moving onto a rail sets "hung"; moving off restores the previous fold
              deleting an organizer moves its items to the tray
              canHold refuses an organizer inside an organizer
persistence:  download -> parse gives an identical home
              a broken file is refused with a reason and changes nothing
```

## Changes made while building

Departures from the pseudocode, each made to keep the code simpler or the app easier to use:

- **Where a piece stands.** `x, y` is the corner of its floor footprint nearest the room's top-left corner, with `facing` saying which way the front points. The pseudocode said "back-left corner", which changes meaning with every rotation.
- **Edits go through one helper.** The pseudocode listed one reducer action per edit. In the code, screens call the model functions through `tryEdit` or `edit` in the store, which either save the new home or return the reason it was refused. The reducer keeps five actions.
- **Saving is immediate.** The pseudocode waited 300 ms after the last change. The home is small, so it is saved on every change, and a reload straight after an edit loses nothing.
- **File checking has its own file.** `parseHome` lives in `model/parse.ts`, not in `persistence.ts`, so it can be tested without a browser.
- **No default fold per clothing type.** New items start flat, as you asked. An item taken off a rail goes back to its previous fold, or flat if it had none.
- **The outline on a stored box** is drawn from the item's own parts, so it changes when you edit a sleeve or leg. There is no separate drawing per type.
- **Adding things.** Two buttons add a box or a clothing item to the selected compartment or box, or to the unsorted tray when nothing is selected. The pseudocode put buttons on every compartment, which cluttered the drawing.
- **"Where" list.** The editor has a dropdown of every compartment and box, for moves between pieces of furniture and for keyboard use.
- **Chips for thin things.** An item laid flat is a pixel or two tall in the drawing and hard to grab. Clicking a compartment lists its contents as named chips under the drawing, and an open box lists its contents the same way. Chips drag like the drawn shapes.
- **Nothing is drawn off-screen.** Things that do not fit are drawn past the side of the piece, up to a limit, and pulled back to the edge beyond that, so an overfull shelf never hides anything.
- **Names are editable.** The Measurements panel can rename a piece and its compartments.
- **"Start over"** returns to the starter room, behind a second click.
- **Two shared helpers for resizing.** Heights down a column and widths across a piece follow the same two rules, so `furniture.ts` has `giveToLargest` and `tradeWithNeighbour` in place of four separate routines.
- **A few more small files** than the plan's table: `refusal.ts`, `ids.ts`, `drag.tsx`, `parts.tsx`, `ThingChip.tsx`, `AddBar.tsx` and `Header.tsx`. `ContentsLayer.tsx` became `Contents.tsx`. The README lists the files as built.

## Import check

For each need, the order was: something built into the browser or React first, then one well-known library, then writing it ourselves.

| Need | Choice | Why | Considered instead |
| --- | --- | --- | --- |
| Screens | React 19 (Meta) | Your decision | Svelte, no framework |
| Language and build | TypeScript 7 (Microsoft), Vite 8 (VoidZero and an open-source team) | Your decision | Plain JavaScript |
| Tests | Vitest 5 (the Vite team) | Shares Vite's config, and the model tests need no browser | Jest (separate TypeScript setup); Node's built-in runner (no TypeScript without extra setup) |
| Shared state | React's built-in `useReducer` and context | One small app-wide state | Zustand (Poimandres collective); worth adding when the Three.js room (R1) needs to read state outside React |
| Drag and drop | Browser pointer events plus `document.elementsFromPoint`, in one small hook | Works with mouse and touch, and on SVG drawings, with nothing to install | dnd-kit (individual-led, widely used; built for HTML lists, awkward on SVG); the browser's drag-and-drop API (no touch, unreliable on SVG) |
| Checking saved files | A hand-written `parseHome` | Five small record types; explicit checks give readable reasons | Zod (individual-led, widely used); worth adding when tags and custom categories (C6, C7) grow the format |
| Storage | `localStorage`, behind `loadHome` and `saveSoon` | The whole home is one small JSON text | IndexedDB through idb-keyval; needed once photos arrive (C13) |
| Ids | `crypto.randomUUID()` | Built in | nanoid, uuid |
| Colour picker | `<input type="color">` | Built in | react-colorful; needed when opacity arrives (C4) |
| Drawing | SVG, written as React elements | Everything in the trunk is rectangles and text, to scale | Canvas; Three.js, which arrives with R1 |
| Saving a file | A Blob and a link, or the host page's own save function when there is one | Built in | FileSaver.js |

Installed: two packages the app runs on (`react`, `react-dom`) and six used only to build and test. With their own dependencies that is 44 packages, all from the npm registry, with exact versions locked in `package-lock.json`. `npm audit` reported no known vulnerabilities when they were installed.

How this was checked: type checking, the test runner and the build were run end to end on the installed versions before any app code was written, and React's installed type definitions were read for the functions used. The documentation lookup tool had run out of its monthly quota, so its results are not part of this check.

## Appendix: answers to the questions in your first message

These are the answers from the first plan, kept for reference. Where the plan changed afterwards, the answer says so.

**When to research imports.** Your instinct was right, with one exception. The few load-bearing choices (how the room is drawn, the screen library, storage) shape the file layout, so they were settled in the outline. Detailed documentation review belongs after the pseudocode, when it is clear which functions are needed. JavaScript has even more overlapping packages than Python.

**Three.js vs Godot.**
- Three.js is a JavaScript library that draws 3D inside a normal web page. It is open source, the web standard, and maintained by a large community. There is no editor; you write code, and HTML forms and search sit beside it naturally.
- Godot is a full game engine with a visual editor and its own Python-like language. It is strong for a walking character, pathfinding, quests and app-store builds. Its web export is a large download and awkward for forms.
- Three.js fits the room view (R1) and Godot fits the walking-character stage (R11). The shared saved file keeps both open.

**Do you need to model anything, or use Blender?** No. Everything is drawn from dimensions. If you later want shaped furniture:
- Blockbench (free, open source) is the easiest fit for a blocky pixel look.
- MagicaVoxel (free, by an individual developer, widely used) is good for voxel models.
- Blender (Blender Foundation) is the most capable and the steepest to learn.
- For pixel sprites and textures: Aseprite (paid), or the free LibreSprite or Piskel.

**Plugins.** Enable none for now.

| Plugin in your catalog | Made by | Verdict |
| --- | --- | --- |
| frontend-design, Design | Anthropic | Optional; useful for a design critique or accessibility review later |
| Figma | Figma (the company) | Only if you start using Figma |
| playwright | Microsoft | Not needed; the build workspace already has it |
| parallax-threejs | An individual community author | Skip. It adds hooks and several browser-control servers, the broadest access of the list |
| Claude2Figma, figma-suite, Figma to Code | Individual community authors | Skip |

A plugin can add instructions, tools that reach outside services, and hooks that run on every action. Risk grows with what it can reach and how little is known about its author. The code libraries and who maintains them are in the import check above; the first plan also listed three, zustand, zod and idb-keyval, none of which the trunk ended up needing.

**Vulnerabilities.** Worth checking, and cheap. The app has no server and no accounts, and data stays in your browser. The real risks are the dependencies (few, locked, audited), imported backup files (checked before use), and text you type being treated as code (React escapes it by default). Later AI features must go through a small server so that no secret key sits in page code, and photos should have location data stripped.

**Tests.** Yes. Unit tests cover the pure logic: clear spaces, folding, layout, moving, and backup files. A browser run checks the screens. A GitHub check on every push would live outside `roomorg/`, so it is not added unless you ask.

**Evaluators and analyzers.** Strict TypeScript only for now, since speed matters more than polish in a temporary repo. If this moves to a portfolio repo, the useful additions are ESLint with complexity limits, test coverage, and Lighthouse (built into Chrome) for speed and accessibility. Online autograders are for coursework.

**Infinite loops and memory leaks.** Low risk, and confined to one browser tab that you can close. The dev server is an ordinary process; Ctrl-C stops it. The places that will need care arrive with the Three.js room (R1): its drawing loop and its graphics memory.

**Environment and installs.**
- To run the code yourself you need Node.js (`brew install node`) and, optionally, VS Code. Homebrew is fine.
- The per-project `node_modules` folder plus the lockfile is the equivalent of a Python virtual environment, so nothing else is installed globally.
- A version manager (nvm or fnm) only matters once two projects need different Node versions.
- Jupyter does not apply.
- To install nothing, use GitHub Codespaces or the preview page. The README has the steps.

**Docker.** Not now. It is a multi-GB install wrapped around a one-command dev server. It becomes useful when a Python service for photos and AI arrives.

**Figma.** Optional and not needed yet; your sketches are enough. It becomes useful for laying out the item pop-up and mannequin screens. It is the wrong tool for pixel art.

**Monetization and privacy.** Compatible with this design. Data is local-first with no accounts or tracking, and the web app can later be wrapped for app stores (X1, X2).

**A skill-tree visualizer as a project.** A good small one, and `../tree/` is now a first version of it. The hard part is graph layout, so a larger version should use a layout library, and it should not replace the work it tracks.
