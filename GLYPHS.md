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
- `GET /glyphs` → `{glyphs:[{id,name,category,smufl,compact,meaning,status,samples,image_url,image_missing}], categories, count, samples_total, with_samples, missing_samples, pieces, unidentified}`
- `GET /glyphs/by-piece/{piece}` → reverse lookup: which glyphs actually occur in a piece, with the samples (and note tokens) drawn from it
- `GET /glyphs/{id}` → one glyph (with its `samples`)
- `POST /glyphs` → catalogue a new/unidentified glyph
  `{name?, category, meaning?, compact?, smufl?, source?, status, image_b64?}`
- `PATCH /glyphs/{id}` → identify/annotate a previously-collected glyph
- `GET /glyphs/{id}/image` → the first sample PNG (back-compat)
- `GET /glyphs/{id}/image/{file}` → a specific named sample PNG

The seed catalogue lives in `app/core/glyphs.json` (version-controlled, read-only).
Collected/unidentified glyphs persist in `app/data/glyphs/` (gitignored).

## Glyph images are real score segments — never synthetic
A glyph's sample images are **real PNG crops from source scores** (not font
characters or drawn shapes), stored at `app/core/glyph_images/{id}/{file}.png`.
A glyph can hold **multiple samples from different pieces** — e.g. the treble
clef carries both `fur_elise.png` and `k545_mvt1.png` so the same symbol can be
compared across editions. `GET /glyphs` reports a `samples` list per glyph, sets
`image_url` to the first sample, and `image_missing: true` when none exist — the
dashboard then shows an explicit "NO SAMPLE" / "sample missing" placeholder
rather than fabricating a glyph. To add a sample, drop a PNG in
`app/core/glyph_images/{id}/` (any name; the file stem is used as the source
label) — optionally add a provenance entry (below) for the richer link.

## Samples ↔ music relationship (provenance)
`app/core/glyph_samples.json` links each sample to a **real occurrence in the
music**: `{file, piece, page, system, bar, region, note?, maps_to}`. `note` is the
actual compact token the symbol produced (e.g. the Für Elise sharp → `D#5`),
`maps_to` is its semantic role (voice / meter / velocity). This is the bridge
between the reference library and transcribed pieces — given a glyph you can find
where it really appears (`GET /glyphs/{id}`), and given a piece you can list its
glyphs and tokens (`GET /glyphs/by-piece/{piece}`). It futureproofs round-tripping
between OMR symbols and our note format.

## Time signatures are catalogued explicitly per combo
Each meter is its own glyph as we encounter it — `time-sig-common` (C = 4/4),
`time-sig-2-4`, `time-sig-3-4`, `time-sig-3-8` — each with its bar-fill size
(4/4 = 16 sixteenths, 3/4 = 12, 2/4 = 8, 3/8 = 6) and a real crop from the score
it came from (K.545 mvts I/III/II, Für Elise). Add new combos as new pieces
introduce them.

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
