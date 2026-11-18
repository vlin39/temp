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
| Second iteration | The tidy-up game, G1 to G6: an angled view, doors and drawers, the floor, gaps, boxes that need room, bags | You, in feedback round 6 |
| How the angled view is drawn | An oblique projection in SVG, inside the existing drawing code. Three.js still arrives with the whole-room view (R1). | Claude; say if you would rather have true 3D here |

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

**Round 6** (after trying the first iteration)

> Is this a working interactive webapp? I would like the furniture to be viewed from an angle and the storage organizers/boxes to have a gap between them. Furthermore, the bathroom sink is incorrect---on the left hand side, for example, there's a drawer then the top shelf of a cabinet and the lower shelf of a cabinet. In the middle is the sink, with a cabinet underneath. And on the right is a drawer, above the top shelf of a cabinet, and then there's the lower shelf. I'd also like to make sure the doors of each cabinet can be clicked to open (to reveal the shelves), and each box/storage organizer should be able to be "removed" and...for example, set on the floor. Other boxes should be able to do the same. Clothes should also be able to be set on the floor and shuffled between boxes, and boxes should be able to be shuffled between locations where there is enough room for them. (There should also be "bags"---later on, maybe distinct backpacks vs storage cubes vs pouches---that can hold clothes and can go into the storage boxes.) This should be similar to an organizing game, maybe a bit like A Little To the Left.

Answered: yes, it is a working interactive app. This became the second iteration, described under "The second iteration" below. How each sentence was read:

| You said | What was built |
| --- | --- |
| "viewed from an angle" | Each piece is drawn at an angle, so its top, its side and the depth of every shelf show (G1). |
| "a gap between them" | Neighbours on a shelf or the floor sit 2 cm apart (G4). |
| "the bathroom sink is incorrect" | The sink unit now has, on each side, a drawer above a cabinet with an upper and a lower shelf, and in the middle the sink above a cabinet (G2). |
| "doors of each cabinet can be clicked to open" | Doors click open and closed; drawers pull out. What is behind a closed door is hidden and out of reach (G2). |
| "removed and ... set on the floor" | Anything can be dragged to the floor in front of the piece. The floor replaces the unsorted tray (G3). |
| "shuffled between locations where there is enough room" | A box is refused where it does not fit, and the place shows that while you carry the box over it (G4). |
| "bags ... that can hold clothes and can go into the storage boxes" | Pouch, tote bag and backpack. A bag holds clothing and goes in a box. Distinct kinds of bag is a later node, G7. |
| "similar to an organizing game" | Things are picked up and put down: a copy follows the pointer at real size, places light up, and things settle into place (G6). The look is its own; nothing is taken from another game. |

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

## The second iteration (tidy-up game, G1 to G6)

Built from feedback round 6. These six nodes are a new branch, "Tidy-up game", in `../tree/roomorg.json`.

| Node | What it adds |
| --- | --- |
| G1 Angled view | Each piece is drawn at an angle. Boxes, bags and clothing are blocks with a top and a side. |
| G2 Doors and drawers | Doors click open and closed, drawers pull out, and the sink unit is laid out as it really is. |
| G3 The floor | Anything can be set down on the floor in front of the piece. The floor follows you from piece to piece. |
| G4 Gaps, and boxes need room | Neighbours keep a gap. A box only goes where it fits. |
| G5 Bags | Pouches, tote bags and backpacks hold clothing and go inside boxes. |
| G6 Pick up and put down | A carried thing follows the pointer at its real size; the place under it lights up or refuses; things settle. |

### Choices made here that you did not spell out

- **The angle is drawn in SVG, not Three.js.** An oblique projection keeps every front face a true rectangle, so the existing dragging, keyboard use and to-scale sizes carried over unchanged. True 3D arrives with the whole-room view (R1).
- **The closets have double doors, starting open.** Your sketch draws them, but you only mentioned cabinet doors. The sink unit and the wall cabinet start closed. Doors are listed per piece in `roomData.ts`.
- **Only boxes need room.** Clothing and bags are soft, so they are flagged when they do not fit, as before, and never refused.
- **The floor is one place.** The same things are on the floor whichever piece is open, which is how you carry something between pieces. It is drawn in front of the open piece and listed in the strip along the bottom.
- **The floor is drawn a little way in front of the piece**, so that a box or bag set down there does not hide the bottom shelves.
- **An open drawer has see-through near walls**, so its contents are not hidden behind its own front.
- **A box or bag on a shelf is not drawn open.** Its contents show as bands of colour, on its front and in its open top. Clicking it shows the inside in the side panel.
- **Bags cannot hold bags, and boxes cannot hold boxes.** A bag goes in a box, on a shelf, in a drawer or on the floor.
- **Arrangement stays automatic.** You choose the place and the order. Placing things at exact spots by hand would be a later node.
- **Older saves are upgraded.** A room saved by the first iteration gains doors and the corrected sink unit. Anything that was in the sink unit's old middle drawer goes to the floor, because that compartment is now the sink.

### Pseudocode

**Data shapes (`model/types.ts`), what changed**

```
Home:
    version:   2                              # was 1
    open:      list of ids                    # the doors and drawers that are open

Furniture:
    doors:     list of Door

Door:          id, name, cellIds              # covers the rectangle around those compartments

Cell.kind:     "shelf" | "hanging" | "drawer" | "fixture"
               # "cabinet" is gone: a cabinet is shelves behind a Door
               # "fixture" takes up room and stores nothing (the sink)

Organizer:
    kind:      "box" | "bag"
    parentId:  a Cell id, a box's id (for a bag), or none (on the floor)
```

**Doors and drawers (`model/furniture.ts`, `model/tree.ts`)**

```
function doorRect(f, door):
    the rectangle around the clear spaces of door.cellIds,
    widened to the middle of each neighbouring panel, or to the piece's edge

function isReachable(f, cell, open):
    false if cell is a fixture
    false if cell is a drawer and its id is not in open
    false if a door covers cell and that door's id is not in open
    otherwise true

function toggleOpen(home, id):        add id to home.open, or take it out
function setAllOpen(home, f, open):   the same for every door and drawer of one piece
function reveal(home, thingId):       open the door of the compartment holding the thing,
                                      and the compartment itself if it is a drawer
```

**What may hold what (`model/tree.ts`)**

```
function whyNotHold(parent, thing):   # a sentence, or none if it may
    the floor holds anything
    a thing cannot go inside itself
    a compartment holds anything, unless it is a fixture
    a box holds bags and clothing, not another box
    a bag holds clothing only

function canHold(parent, thing):      whyNotHold(parent, thing) is none
function move(home, id, parent, position):
    refuse with whyNotHold's sentence; otherwise as in the first iteration
    new boxes, bags and clothing always start on the floor
```

**Boxes need room (`model/place.ts`)**

```
function place(home, id, parent, position):       # used by dragging, the Where list and the Add buttons
    moved = move(home, id, parent, position)
    if the thing is a box and parent is not the floor
       and boxesThatDoNotFit(moved, parent) > boxesThatDoNotFit(home, parent):
        refuse: 'There is not enough room for "<name>" there.'
    return moved

function placeOrLeave(home, id, parent):          # for a newly added thing
    try place; if refused, leave it on the floor and return the reason

function nearestHolder(home, wish, id):           # where a new thing goes
    start at wish (what is selected); while it may not hold the thing, step out to its parent
    none means the floor
```

Comparing the number of misfit boxes before and after, and not just asking "does everything fit", means a shelf that is already overfull does not block you from reordering it or taking things off it.

**Arrangement in depth, with gaps (`model/layout.ts`)**

```
Placed: { id, kind, x, y, z, w, h, d, fits }      # z runs back from the front opening; negative is in front

function arrangeShelf(space, children, gap = THING_GAP):
    as before, plus:
        x advances by the thing's width + gap
        z = 0, flush with the front; a thing deeper than the space gets z = space.d - size.d, so it sticks out in front
    inside a box or bag the gap is PACKED_GAP

function arrangeRail(space, cell, children):
    as before, plus: hung clothing is centred under the rail, z = (space.d - size.d) / 2

function arrangeFloor(children, rowWidth):
    line = arrangeShelf(a space with no limits except pile height, children)
    cut the line into rows no wider than rowWidth; a pile stays in one row
    back = THING_GAP
    for each row, nearest the furniture first:
        each thing gets z = -(back + its depth)      # in front of the furniture, back edges lined up
        back = back + the row's deepest thing + THING_GAP
    return the placements, and back as the depth of floor used

function boxesThatDoNotFit(home, parent):
    how many placements in arrange(home, parent) are boxes with fits = false
```

The lists these return are already in painting order, furthest first, so the drawing code does not sort.

**The angled view (`ui/view.ts`)**

```
ANGLED = { kx: 0.32, ky: 0.42 }       # one cm of depth moves a point this far right and up
FLAT   = { kx: 0,    ky: 0 }          # used for the inside of a box in the side panel

function toPixels(view, x, y, z):
    screen x = view.originX + (x + z * view.kx) * view.scale
    screen y = view.originY - (y + z * view.ky) * view.scale

a block at (x, y, z) of size (w, h, d) is drawn as
    its front face: a rectangle
    its top face:   the front's top edge, pushed back by d
    its side face:  the front's right edge, pushed back by d
```

**The drawing (`ui/FurnitureView.tsx`, `ui/Openings.tsx`, `ui/Contents.tsx`)**

```
scale = fit everything that could appear (the piece, the floor, every door and drawer open) into the space available,
        so the picture does not change size when a door opens

paint, from the back forward:
    the floor patch                               # a drop zone for "the floor"
    the outside of the piece: its top and right side; a basin above a fixture at the top of a column
    for each reachable compartment:
        its back, floor and left wall, cut off at the edges of its opening
        its rail, if it has one
        its opening, as the drop zone
    the front edges of the panels
    for each reachable shelf or hanging space: its contents, cut off at the opening unless they do not fit
    doors, drawers and fixtures, from the bottom of the piece up:
        a closed door or drawer: a panel with a knob or handle; click to open
        an open door: its panels swung out toward you; click to close
        an open drawer: pulled all the way out, its contents inside, its near walls see-through; press the handle to close
        a fixture: a fixed panel
    the things on the floor

each thing: a group moved into place with a CSS transform, so it glides when its neighbours change,
            and drops into place when it is first drawn
    a box:      front with its name and a band of colour per thing inside; side; top with a dark opening showing the same bands
    a bag:      the same in another colour, with a strap and no opening
    clothing:   a block in its colour with its laid-flat outline on the front
```

**Picking up and putting down (`ui/drag.tsx`)**

```
on press:  remember the thing and where the pointer is
on move, once past 5 px:
    make a copy of the thing's own drawing and move it with the pointer
    zone = the topmost drop zone under the pointer that may hold the thing (canHold)
    refused = zone exists and place(home, thing, zone) would be refused
    the zone lights up, or shows a red dashed outline if refused
on release:
    not moved: select the thing
    no zone:   nothing changes
    otherwise: edit(place(home, thing, zone, position among the things already there))
               a refusal is shown in the notice line
```

**Reading an older file (`model/parse.ts`)**

```
function checkHome(saved):
    if saved.version is 1: data = upgradeFromVersion1(saved)
    check every record's shape (as before, plus doors, organizer kind and the open list)
    if it was version 1: anything whose parent is gone or may no longer hold it goes to the floor
    check the records fit together: every parent exists and whyNotHold says it may hold its contents

function upgradeFromVersion1(data):
    every organizer gets kind "box"
    every "cabinet" compartment becomes "shelf"
    each piece gets the doors the starter home gives it, where its compartments still match
    the sink unit is replaced by the corrected one
    open = the starter home's open doors
```

**Tests added**

```
layout:   gaps on a shelf and inside a box; flush with the front; sticking out when too deep; centred under a rail
          the floor: rows wrap, a pile stays together, something wider than a row still gets a row
place:    a box is refused where it does not fit and accepted where it does
          a box can be reordered on its own shelf and set on the floor
          bags and clothing are never refused for room, only marked
          an overfull shelf does not block other moves on it
          a new thing is left on the floor with the reason when its target has no room
tree:     what may hold what, including bags in boxes and the sink
          doors and drawers: starting state, toggling, open all, reveal
          the sink unit's layout
parse:    organizer kind, doors and the open list are checked
          a version 1 file is upgraded; what was in the old drawer goes to the floor
          a damaged version 1 file is still refused with reasons
```

### Import check for this iteration

Nothing was added. The angled view is arithmetic on points (`view.ts`) drawn with the same SVG elements, and the carried copy is a clone of the thing's own drawing.

| Need | Choice | Considered instead |
| --- | --- | --- |
| Depth | An oblique projection, under 100 lines in `view.ts` | Three.js: true 3D, but the dragging, keyboard use and labels would all be rebuilt on a canvas. It still arrives with R1. CSS 3D transforms: do not apply to SVG shapes. |
| Things gliding and settling | CSS transitions and one keyframe animation | A motion library such as Motion (formerly Framer Motion); not needed for two effects |
| The carried copy | `cloneNode` on the thing's own drawing | dnd-kit's drag overlay; it does not handle SVG shapes |

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
