# xrdkit — methodology note

Scope: xrdkit produces **synthetic kinematic powder XRD reference patterns** for crystalline materials, intended for phase identification, peak-position diagnostics, and figure composition. It is not a Rietveld refinement tool.

## 1. Peak positions — Bragg's law

For each phase a unit cell (lattice parameters `a`, `c`) and a list of Miller indices `(hkl)` are stored or derived from a structure file. The interplanar spacing `d_hkl` is computed analytically per crystal system (`xrdkit/core.py:d_spacing`):

- cubic         `1/d² = (h²+k²+l²)/a²`
- hexagonal     `1/d² = (4/3)(h²+hk+k²)/a² + l²/c²`
- tetragonal    `1/d² = (h²+k²)/a² + l²/c²`

The 2θ position then follows from Bragg's law

```
λ = 2 d_hkl sinθ      →     2θ_hkl = 2 · arcsin( λ / (2·d_hkl) )
```

Changing the wavelength dropdown shifts every peak to its correct new position. Exact textbook crystallography, no approximation. For arbitrary structure files loaded through pymatgen the same calculation is done with full space-group symmetry handling.

## 2. Peak intensities — kinematic random-powder

For the eight built-in phases the relative intensities are the values from the ICDD / JCPDS powder-diffraction reference cards cited in `xrdkit/core.py`:

| Phase | Card |
|---|---|
| Pt | PDF 04-0802 |
| Ir | PDF 06-0598 |
| α-Ti | PDF 44-1294 |
| TiO₂-rutile | PDF 21-1276 |
| TiO₂-anatase | PDF 21-1272 |
| IrO₂-rutile | PDF 15-870 |
| RuO₂-rutile | PDF 40-1290 |
| TiO₂-brookite (proxy) | PDF 29-1360 |

Each card value corresponds to the kinematic intensity of a randomly-oriented polycrystalline sample at Cu Kα:

```
I_hkl ∝ |F_hkl|² · m_hkl · LP(θ_hkl) · exp(−2B sin²θ/λ²)
```

- `F_hkl` — structure factor, `Σ_j f_j(sinθ/λ) · exp[2πi(hx_j+ky_j+lz_j)]`
- `m_hkl` — multiplicity of (hkl)
- `LP(θ)` — Lorentz-polarisation factor `(1+cos²2θ)/(sin²θ cosθ)` for Bragg-Brentano geometry
- `exp(−2B sin²θ/λ²)` — Debye-Waller thermal factor

Card values are already integrated over all of that. Consequence: the intensities are rigorously correct at Cu Kα and approximately correct at other wavelengths (LP factor and `f_j(sinθ/λ)` weighting drift with λ).

For structures loaded from CIF / POSCAR / COD / Materials Project the full structure-factor calculation is performed by `pymatgen.analysis.diffraction.xrd.XRDCalculator` at the requested wavelength.

## 3. Peak shape — pseudo-Voigt with Scherrer broadening

Each Dirac stick `(2θ_hkl, I_hkl)` is convolved with a pseudo-Voigt profile (`xrdkit/core.py:pseudo_voigt`)

```
PV(x) = η · Lorentzian(x; γ) + (1−η) · Gaussian(x; σ)         η ∈ [0,1], default 0.5
```

with FWHM given by the Scherrer equation (`xrdkit/core.py:scherrer_fwhm_deg`)

```
β(2θ) = K · λ / (D · cosθ)         K = 0.9 (sphere shape factor), D in nm
```

The total pattern is `I(2θ) = Σ_hkl I_hkl · PV(2θ − 2θ_hkl, β_hkl)`.

## 4. Optional refinements

- **Kα1 / Kα2 doublet** — when enabled, the pattern is rendered as the sum of contributions at λ_Kα1 and λ_Kα2 = λ_Kα1 + Δλ with 2:1 intensity ratio (Cu Δλ = 0.00383 Å). Peak positions for the Kα2 component are derived by Bragg's law from the original `d_hkl`, not by a global 2θ-shift, which is correct at all angles.
- **Linear background slope** — a baseline of `slope · (2θ − 2θ_min)/(2θ_max − 2θ_min)` is added before normalisation, mimicking the broadband rise often seen with amorphous content or air scatter.
- **(hkl) labels** — Miller indices are printed above peaks with relative intensity above a user-chosen threshold. Helpful for paper figures.

## 4b. Observed − simulated difference

The difference tool (`xrdkit/plot.py:compute_difference`) compares a measured
pattern against the selected reference phases in the order used by Rietveld
practice:

1. interpolate the measurement onto the common 2θ grid;
2. estimate and subtract a background by **SNIP** (log-log-sqrt iterative
   peak-clipping; Ryan et al., *Nucl. Instrum. Methods B* **34**, 396 (1988)),
   with the clipping window set wider than the Bragg peaks so it follows the
   baseline rather than biting into intensity;
3. **zero-shift align** the calculation by maximising cosine overlap over a small
   2θ offset (specimen displacement / zero-offset);
4. scale the calculated phase profiles to the measured counts by **non-negative
   least squares** — one scale per phase, returned as a relative-abundance proxy.
   Unit weights are used: the profiles are background-free, so Σ p² is dominated
   by the peaks; 1/y_obs Poisson weights would hand authority to the zero baseline
   and collapse the scale.

The residual is `y_obs − Σ_k s_k · p_k`. Two figures of merit are reported, both
*pattern-similarity* measures and **not** a Rietveld goodness-of-fit:

```
overlap = ⟨y_obs, y_calc⟩ / (‖y_obs‖ ‖y_calc‖)         (cosine, 0–1)
R_p     = 100 · Σ|y_obs − y_calc| / Σ|y_obs|            (profile R-factor, %)
```

Worked examples are in `examples/observed_simulated_difference/`.

## 5. What "Normalised intensity (a.u.)" means

Inside `_prepare` each phase's computed pattern is divided by its own maximum so the strongest peak of each phase reaches 1.0. In the stacked layout an offset of 1.15 is added per phase to separate them visually. **Heights are not comparable between phases** — Pt scatters far more strongly than TiO₂ per unit volume. The label `Normalised intensity (a.u.)` makes the arbitrary-units convention explicit.

For mixed-phase quantification use Rietveld refinement (GSAS-II / FullProf / TOPAS) or Reference-Intensity-Ratio (`I/I_c`) analysis — xrdkit deliberately does not do this.

## 6. What xrdkit does not model

1. **Phase fractions** — heights are per-phase normalised, not weighted by abundance.
2. **Preferred orientation** — ICDD intensities assume random orientation; real sprayed / coated catalyst layers are textured.
3. **Instrument broadening** — divergence / Soller slits / detector response add Gaussian + Lorentzian contributions that are not modelled. Real FWHMs at large `D` are limited by the instrument, not by Scherrer.
4. **Strain / microstrain** — Williamson-Hall analysis is out of scope.
5. **Amorphous background, fluorescence** — only a linear baseline is offered.
6. **Structural refinement** — the observed−simulated difference tool (§4b) does a *linear* scale fit of fixed reference profiles to a measurement; it does not refine atomic positions, occupancies, or the unit cell, and reports similarity scores, not a Rietveld χ².

## 7. For an experimentalist who wants to compare their data

To overlay a measured diffractogram in the **Upload measured / computed XRD** tab:

**A. Get the pattern in 2-column ASCII** (`2θ_deg  intensity`). Most diffractometer GUIs export to `.xy`, `.csv`, or `.txt`. `.xrdml` is parsed natively. For Bruker `.raw` or Rigaku `.ras`, export via the vendor software.

**B. Match the wavelength**. If your diffractometer used Cu Kα, leave the wavelength dropdown at Cu Kα; if Co Kα, switch.

**C. Apply any per-file corrections**:
- **2θ offset** — instrument zero-shift (calibrate against Si NIST 640 / LaB₆).
- **intensity scale** — set negative to flip a computed pattern below the baseline for difference plots.
- **header rows to skip** — for files with comment lines.

**D. Decide preprocessing before loading**:
- Background subtraction (none / polynomial / spline) — xrdkit does not subtract for you.
- Kα₂ stripping (Rachinger) — if your data are Kα₂-stripped, render the references with Kα₂ off, and vice versa.

A typical workflow: tick the references you expect, set wavelength and 2θ range to match the scan, drag in your `.xy`, drag the `D (nm)` slider until reference peak widths roughly match the measured widths, look for unassigned peaks.

## 8. Computational / DFT users

The **Upload structure** tab accepts CIF, POSCAR / CONTCAR, `.vasp`, `.json`, `.xyz`, `.xsf` — anything pymatgen reads. Loaded structures are converted to a primitive cell, the space group is detected, and the kinematic powder pattern is computed.

A common workflow: relax your structure in VASP, drop the CONTCAR in, compare against a measured pattern uploaded in the next tab. DFT-relaxed lattices typically over-estimate `a` by 1–2 % (PBE/GGA), shifting peaks 0.1–0.3°; account for this when interpreting offsets.

## References

- Cullity, B. D. & Stock, S. R. *Elements of X-Ray Diffraction*, 3rd ed., Prentice Hall, 2001.
- Birkholz, M. *Thin Film Analysis by X-Ray Scattering*, Wiley-VCH, 2006.
- Pecharsky, V. K. & Zavalij, P. Y. *Fundamentals of Powder Diffraction and Structural Characterization*, 2nd ed., Springer, 2009.
- Scherrer, P. *Nachr. Ges. Wiss. Göttingen*, **1918**, 98–100.
- ICDD / JCPDS card numbers as cited per phase in §2.
- Materials Project: Jain et al., *APL Materials* **1**, 011002 (2013).
- Crystallography Open Database: Gražulis et al., *J. Appl. Cryst.* **42**, 726 (2009).
- pymatgen: Ong et al., *Comput. Mater. Sci.* **68**, 314 (2013).
