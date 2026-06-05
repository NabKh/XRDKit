# xrdkit — scientific roadmap

Features that diffraction experimentalists routinely reach for and that fit
xrdkit's scope (lightweight reference-overlay + measured-pattern analysis, *not*
a full structural-refinement engine). Ordered within each group by value-to-effort.
"Quick" = a focused addition reusing machinery already in `core.py`/`plot.py`;
"Medium"/"Larger" need a new fitting routine or UI surface.

Nothing here is implemented yet — it is a menu to choose from, not a promise.

---

## A. Extending the difference / subtraction tool (closest to what was just added)

- **Two-experiment difference (operando before↔after).** *Quick.* Subtract one
  *measured* pattern from another instead of measured−calculated. Reuses the
  background/align/scale machinery; swap the calculated basis for a second
  `Measured`. This is the single most-wanted feature for electrocatalysis: track
  what changes after potential cycling, dissolution, or restructuring. Directly
  serves PEMWE/OER degradation studies.
- **Stacked operando / time-series view.** *Medium.* Load N patterns (a time or
  potential series) and render a waterfall or 2D heat-map with the same Nature
  styling, optionally with a difference-to-first-frame strip.
- **Residual diagnostics readout.** *Quick.* Flag whether the residual is
  derivative-shaped (position/zero-shift error), single-sided at one peak
  (preferred orientation), or broadly oscillating (lattice-parameter mismatch) —
  turning the picture into an actionable hint.

## B. Quantification (numbers reviewers ask for)

- **Semi-quantitative phase fractions (RIR / I-over-Icor).** *Medium.* The
  non-negative least-squares phase scales already computed for the difference
  plot become weight fractions once divided by each phase's Reference Intensity
  Ratio. Honest caveats (no microabsorption, kinematic) stated up front. High
  value for mixed catalyst oxides (IrO₂/RuO₂/TiO₂).
- **Lattice-parameter refinement from peak positions.** *Medium.* Least-squares
  refine the unit cell from indexed d-spacings, report a, c (+ e.s.d.) and the
  delta versus the reference card — the quantitative version of "the residual
  says refine the cell."
- **Crystallinity index.** *Quick.* Crystalline peak area / (crystalline +
  amorphous-halo area) from the SNIP split already available.

## C. Microstructure (size & strain)

- **Per-peak profile fitting.** *Medium.* Fit individual pseudo-Voigts to
  measured peaks → precise position, FWHM, and integrated area with uncertainties.
  This is the foundation the next two items build on.
- **Williamson–Hall plot.** *Medium.* Separate size broadening from microstrain
  by plotting βcosθ vs sinθ; report crystallite size and strain. A staple of
  nanomaterial/catalyst papers and a natural companion to the Scherrer code
  already in `core.py`.
- **Instrumental-broadening correction.** *Quick.* Subtract a standard's FWHM
  (e.g. LaB₆/Si) in quadrature before size/strain analysis, so reported sizes are
  not instrument-limited.

## D. Data handling experimentalists expect

- **Peak search + auto-table.** *Quick.* Detect peaks in a measured pattern and
  emit a 2θ / d / relative-intensity table (mirrors the existing reference
  `peaks_table`).
- **More instrument file formats.** *Quick–Medium.* Bruker `.raw`/`.brml`,
  Rigaku `.ras`, plain `.xrdml` (already partly supported). Lowering the
  "will it read my file?" barrier is what gets a tool actually used.
- **Q-space / d-space x-axis toggle.** *Quick.* Offer Q = 4π·sinθ/λ or d
  alongside 2θ — expected by synchrotron and total-scattering users and for
  comparing data taken at different wavelengths on one axis.
- **Wavelength conversion of a measured pattern.** *Quick.* Re-map a pattern from
  one λ to another so Cu-, Co-, and Mo-source data overlay correctly.

## E. Polish for the hosted app

- **Lazy/optional pymatgen + mp-api** so the public app boots light and the
  CIF/MP tabs load on demand (see `DEPLOY.md`). *Quick.*
- **Example-data picker** in the UI ("try it with Pt / TiO₂ / IrO₂") so a
  first-time experimentalist sees a result in one click. *Quick.*
- **Session export/import** (save the whole composed figure + settings as a
  small JSON) for reproducibility and sharing. *Medium.*

---

### Suggested first three to build next
1. **Two-experiment difference** (B-tier value, A-tier ease) — operando before/after.
2. **Example-data picker in the app** — turns the hosted link into an instant demo.
3. **Semi-quantitative phase fractions (RIR)** — reuses the NNLS scales already
   computed, gives reviewers a number.
