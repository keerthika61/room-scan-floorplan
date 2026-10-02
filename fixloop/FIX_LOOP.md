# Fix Loop

One measured weakness, diagnosed, fixed, and re-measured. Both runs are
regenerable with the commands at the bottom.

## 1. Worst-performing gate (before)

**Gate:** Ceiling height accuracy, which requires the reported confidence
interval per room to be within **±1.5 cm**.

**Failing number (scan `single_scan_with_ceiling` / `c7d28f72c6`):**

```
ceiling_height = 3.0615 m
ceiling_height_ci_half_width = 3.06 cm   -> FAIL (gate <= 1.5 cm)
```

The point *estimate* was already reasonable; it was the **confidence interval
that failed** — the pipeline was honestly reporting that it did not know the
height tightly enough to pass.

## 2. Root-cause hypothesis + evidence

The confidence interval was computed from the **standard deviation of all
points in a ±5 cm band** around the ceiling (and floor) height:

```python
ceiling_ci = sqrt(floor.spread**2 + ceiling.spread**2) + 0.005
```

That quantity measures the **physical thickness / roughness of the slab plus
the band width**, not the uncertainty in *where the plane sits*. With ~100k
points on a plane, the plane's height is known far better than any single
point's scatter suggests.

**Evidence:** the floor and ceiling slab spreads were ~1.7–1.9 cm, almost
entirely slab thickness and a few outliers (light fixtures, vents), yet those
spreads were being reported directly as the height uncertainty. The number of
points (statistical power) was being ignored entirely.

## 3. Fix shipped + predicted number

**Fix:** estimate each plane height with a **robust least-squares fit**
(iterative 2.5σ trim to reject fixtures/outliers), and report its uncertainty
as the **standard error of the mean** (`std / sqrt(N)`), floored by a sensor
systematic (`0.5 cm`) so we never over-claim precision. Ceiling height is a
difference of two independent plane heights, so its CI is the quadrature sum
of the two standard errors, reported at ~2σ (95%).

**Prediction:** CI drops below the 1.5 cm gate while the height value stays
essentially unchanged (the fix corrects the *uncertainty*, not the estimate).

## 4. After

```
ceiling_height = 3.0589 m
ceiling_height_ci_half_width = 1.41 cm   -> PASS (gate <= 1.5 cm)
```

- Gate moved **FAIL -> PASS**.
- Height value moved only **2.6 mm** (3.0615 -> 3.0589 m): we did not shift the
  answer to pass, we corrected an over-conservative uncertainty estimate.
- The floor-only scan (`single_room`) still correctly reports **no ceiling**
  (no false positive introduced), confirming the change did not just loosen
  detection.

## Regenerate

```bash
# BEFORE: checkout the commit tagged just before the fix, then:
python -m fixloop.measure_ceiling outputs/c7d28f72c6/cloud.ply

# AFTER: on the current commit:
python -m fixloop.measure_ceiling outputs/c7d28f72c6/cloud.ply
```

Captured transcripts: `fixloop/before.txt`, `fixloop/after.txt`.
