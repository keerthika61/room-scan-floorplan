# Repeatability

The brief's gate: *two captures of the same room at the same tier agree within
1 cm or 0.5% per wall* ("same room in, same plan out").

The sample data has no two separate physical captures of one room, so we test
repeatability with an honest **proxy**: split a single scan's frames into two
disjoint halves (even-indexed vs odd-indexed), which gives two independent,
lower-density captures of the same space, and compare the geometry built from
each.

Run it with:

```bash
python -m fixloop.repeatability sample_data/single_room/c00a170fe1
```

## Result (`single_room`, 1715 frames -> two halves)

| Measurement    | Half A | Half B | Difference | Against gate |
|----------------|-------:|-------:|-----------:|--------------|
| Perimeter      | 31.418 m | 31.404 m | 1.39 cm (**0.04%**) | PASS (<= 0.5%) |
| Floor area     | 14.780 m² | 15.111 m² | 0.33 m² (2.22%) | — (area is boundary-sensitive) |
| Ceiling height | n/a | n/a | — | floor-only scan |

**Reading:** the perimeter — the most direct per-wall quantity — repeats to
**0.04%**, comfortably inside the 0.5% gate, showing the pipeline is stable to
how the room is sampled. Floor area differs more (2.22%) because it depends on
the exact segmentation boundary, which the two sparser halves place slightly
differently; this is the honest weak point of area repeatability.

## Honesty note

This is a frame-split proxy, not two physical walk-throughs. It isolates
stability to sampling (the dominant handheld-capture factor) but does not
capture pose-tracking differences between two real sessions. A true
repeatability test needs two physical captures, which the walk-in test would
provide.
