# Glyph reference library

A shared, machine-readable catalogue of music-notation glyphs and **how each maps
to our compact note format**, so humans and AI agents (Claude, the browser AI)
refer to symbols the same way — without having to re-derive them each time.

## Compact note format (recap)
A bar holds up to four voice strings: `melody`, `melody2`, `bass`, `bass2`.
Each is space-separated tokens `Pitch+Octave(Duration)` or `R(Duration)`:

- Pitch: `A`–`G`, optional `#`/`b`, then octave, e.g. `C5`, `D#4`, `Bb3`. Rest = `R`.
- Duration tag: `w h q 8 16 32 64` (whole … 64th), optional `.` = dotted.
  `(4)` is **invalid** — a quarter is `(q)`.
- A 3/8 bar holds 6 sixteenths (1.5 quarter-beats); every voice should fill the bar.

## API (AI-friendly)
- `GET /glyphs` → `{glyphs:[{id,name,category,smufl,compact,meaning,status}], categories, count, unidentified}`
- `GET /glyphs/{id}` → one glyph
- `POST /glyphs` → catalogue a new/unidentified glyph
  `{name?, category, meaning?, compact?, smufl?, source?, status, image_b64?}`
- `PATCH /glyphs/{id}` → identify/annotate a previously-collected glyph
- `GET /glyphs/{id}/image` → the clipped PNG (for collected glyphs)

The seed catalogue lives in `app/core/glyphs.json` (version-controlled, read-only).
Collected/unidentified glyphs persist in `app/data/glyphs/` (gitignored).

## How to refer to a glyph
Use its **`id`** (stable) or **`smufl`** (the Standard Music Font Layout name) when
discussing or annotating; use **`compact`** when writing it into a transcription.
Examples: sixteenth rest = id `rest-16th`, smufl `rest16th`, compact `R(16)`.
Eighth note = a filled notehead with one beam/flag → compact `(8)`.

## Collecting unidentified glyphs
When you meet a symbol you can't map, **catalogue it** instead of guessing:
clip the region, `POST /glyphs` with `status:"unidentified"`, the image, and a
`source` (e.g. `fur_elise_p1 bar 12`). It appears in the dashboard **Glyphs** tab
(flagged "unidentified") for a human/AI to identify later via `PATCH`. In the
dashboard you can also paste (Ctrl+V) a clipped image straight into the
"Catalogue a glyph" form.
