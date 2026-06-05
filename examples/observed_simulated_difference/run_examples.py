"""Worked examples: observed - simulated difference (Rietveld-style residual).

Each example loads a *measured* pattern, builds a *calculated* reference pattern
from one or more xrdkit phases, then background-subtracts, zero-shift aligns, and
scales the calculation to the measurement by non-negative least squares before
drawing the residual. The reported overlap (cosine) and R_p are pattern-similarity
scores for kinematic reference overlay -- not a Rietveld goodness-of-fit.

Run from the repository root (or this folder):

    python examples/observed_simulated_difference/run_examples.py

Outputs land in ./figures/ as both PDF (vector, journal-ready) and PNG (preview).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]          # examples/observed_simulated_difference -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # import *this* repo's xrdkit, not a sibling install

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xrdkit.db import builtin_phases
from xrdkit.plot import load_measured, compute_difference, make_difference_plot

DATA = HERE / "data"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
LIB = builtin_phases()

CU_KA = 1.54184  # Cu Kalpha (weighted average), Angstrom

# (data file, phase names, crystallite size nm, eta, 2theta range, label)
EXAMPLES = [
    dict(xy="accurate_platinum_experimental.xy", phases=["Pt"],
         D=8.0, eta=0.4, tt=(15.0, 90.0),
         out="ex1_platinum",
         note="fcc Pt -- clean single-phase metal; near-textbook match"),
    dict(xy="accurate_rutilo_tio2_experimental.xy", phases=["TiO₂ rutile"],
         D=9.0, eta=0.4, tt=(15.0, 80.0),
         out="ex2_rutile_tio2",
         note="rutile TiO2 -- oxide where (211) intensity departs from the ICDD "
              "card (preferred orientation), shown honestly in the residual"),
    dict(xy="realistic_iro2_experiment.xy", phases=["IrO₂ rutile"],
         D=3.0, eta=0.5, tt=(20.0, 90.0),
         out="ex3_iro2_oer",
         note="rutile IrO2 -- the OER/PEMWE workhorse anode catalyst; nanocrystalline "
              "(~3 nm) so the Bragg peaks are strongly size-broadened"),
]


def main():
    print(f"available phases: {list(LIB)}\n")
    summary = []
    for ex in EXAMPLES:
        phases = [LIB[n] for n in ex["phases"] if n in LIB]
        if not phases:
            print(f"!! skipping {ex['out']}: phases {ex['phases']} not found")
            continue
        meas = load_measured(DATA / ex["xy"], name=ex["xy"].split(".")[0])
        data = compute_difference(
            meas, phases, wavelength_A=CU_KA,
            crystallite_nm=ex["D"], eta=ex["eta"], tt_range=ex["tt"],
        )
        fig, _ = make_difference_plot(data, tt_range=ex["tt"])
        fig.savefig(FIGS / f"{ex['out']}.pdf")
        fig.savefig(FIGS / f"{ex['out']}.png", dpi=200)
        plt.close(fig)

        scales = ", ".join(f"{n}={float(s):.1f}"
                           for n, s in zip(data["names"], data["scales"]))
        print(f"== {ex['out']} ==")
        print(f"   {ex['note']}")
        print(f"   phase scale(s): {scales}")
        print(f"   zero-shift = {data['shift']:+.2f} deg   "
              f"overlap = {data['overlap']:.3f}   R_p = {data['Rp']:.1f} %")
        print(f"   wrote figures/{ex['out']}.pdf / .png\n")
        summary.append((ex["out"], data["overlap"], data["Rp"], data["shift"]))

    print("summary (overlap, R_p, zero-shift):")
    for name, ov, rp, sh in summary:
        print(f"  {name:<18} overlap={ov:.3f}  R_p={rp:5.1f}%  shift={sh:+.2f} deg")


if __name__ == "__main__":
    main()
