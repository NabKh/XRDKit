# Observed − simulated difference (Rietveld-style residual)

Worked examples of xrdkit's **observed / calculated / difference** figure — the
plot crystallographers use to judge how well a reference pattern accounts for a
measured one. Each figure has three stacked, x-shared panels:

1. **top** — measured pattern (open circles) with the scaled calculation (line);
2. **middle** — Bragg tick rows, one colour-matched row per phase;
3. **bottom** — the residual, *observed − calculated*, about zero.

## How the calculation is matched to the measurement

The pipeline (`xrdkit.plot.compute_difference`) follows standard practice:

1. interpolate the measurement onto a common 2θ grid;
2. estimate and subtract a **SNIP background** (window wider than the peaks, so
   it hugs the baseline without biting into Bragg intensity);
3. **zero-shift align** the calculation by cross-correlation (specimen
   displacement / zero-offset);
4. scale the calculated phase profiles to the measured counts by
   **non-negative least squares** — *not* max-normalisation. With several phases
   this returns one non-negative scale per phase (a phase-abundance proxy);
5. residual = measured − Σ(scaleₖ · phaseₖ).

The two reported numbers are honest *pattern-similarity* scores, **not** a
Rietveld goodness-of-fit (xrdkit does kinematic reference overlay with fixed
lattice/HKL intensities, not a structural refinement):

- **overlap** — cosine similarity ⟨obs, calc⟩ / (‖obs‖‖calc‖) ∈ [0, 1];
- **R_p** — profile R-factor, 100·Σ|obs − calc| / Σ|obs|.

## Run them

From the repository root:

```bash
python examples/observed_simulated_difference/run_examples.py
```

Vector PDFs (journal-ready, 600 dpi, editable fonts) and PNG previews are written
to `figures/`.

## The three examples

| # | file | system | D (nm) | overlap | R_p | zero-shift |
|---|------|--------|:------:|:-------:|:---:|:----------:|
| 1 | `ex1_platinum`     | fcc **Pt** metal        | 8 | 0.90 | 60 % | +0.32° |
| 2 | `ex2_rutile_tio2`  | **rutile TiO₂** oxide   | 9 | 0.86 | 70 % | +0.39° |
| 3 | `ex3_iro2_oer`     | **rutile IrO₂** (OER)   | 3 | 0.84 | 51 % | −0.67° |

**1 — Platinum.** A clean single-phase fcc metal: the calculation tracks all
reflections and the residual is small and flat apart from narrow derivative-shaped
wiggles where the peak position/width differ by a fraction of a degree. This is
what a good match looks like.

**2 — Rutile TiO₂.** The (211) reflection near 54.5° sits well below the value
on the ICDD card. The residual shows this as a single deep negative excursion —
the classic signature of **preferred orientation** (or a real intensity
departure), surfaced honestly rather than normalised away.

**3 — Rutile IrO₂.** The OER/PEMWE anode workhorse, here **nanocrystalline
(~3 nm)** so every Bragg peak is strongly size-broadened — the calculation only
matches once the Scherrer broadening is set accordingly. A residual structure
remains because a single rigid zero-shift cannot absorb an angle-dependent
lattice-parameter difference; that leftover is exactly the cue an experimentalist
reads as "refine the cell."

> Crystallite size **D** is a genuine fit parameter here: it sets the peak width
> via the Scherrer relation. The values above were chosen as the best match for
> each measurement (e.g. IrO₂ is best fit by ~3 nm, consistent with a
> nanostructured catalyst), not tuned to flatter a result.

## Data

`data/` holds the three two-column ASCII (`.xy`) patterns used above. They are
copies of the measured traces shipped at the repository root, kept here so the
examples run self-contained.
