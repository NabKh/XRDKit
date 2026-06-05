"""Command-line entry point: render PDF / GIF without opening the GUI."""
from __future__ import annotations

import argparse
from pathlib import Path

from xrdkit.core import WAVELENGTHS
from xrdkit.db import (builtin_phases, from_structure_file, from_mp_id,
                        from_cod_id, assign_color)
from xrdkit.plot import save_pdf, save_gif, load_measured


def parse_args():
    p = argparse.ArgumentParser(
        prog="xrdkit",
        description="Render XRD reference figures from ICDD cards, structure "
                    "files, COD / Materials Project, and measured patterns.",
    )
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--phase", action="append", default=[],
                   help="built-in ICDD-card phase (repeatable). "
                        "Default: all built-in phases.")
    p.add_argument("--cif", action="append", default=[],
                   help="structure file: CIF / POSCAR / .vasp / .json (repeatable)")
    p.add_argument("--mp", action="append", default=[],
                   help="Materials Project ID, e.g. mp-126 (repeatable)")
    p.add_argument("--cod", action="append", default=[],
                   help="COD entry ID (repeatable)")
    p.add_argument("--measured", action="append", default=[],
                   help="measured or computed XRD file (.xy / .csv / .xrdml)")
    p.add_argument("--wavelength", default="Cu Kα",
                   choices=list(WAVELENGTHS.keys()))
    p.add_argument("--crystallite", type=float, default=30.0,
                   help="crystallite size (nm) for Scherrer broadening")
    p.add_argument("--tt-min", type=float, default=20.0)
    p.add_argument("--tt-max", type=float, default=90.0)
    p.add_argument("--show-hkl", action="store_true",
                   help="annotate peaks with their (h k l) Miller indices")
    p.add_argument("--kalpha12", action="store_true",
                   help="add the Kα2 satellite (intensity 0.5×Kα1)")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-pdf", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    wl = WAVELENGTHS[args.wavelength]
    tt = (args.tt_min, args.tt_max)

    library = builtin_phases()
    selected = []
    if args.phase:
        for name in args.phase:
            if name not in library:
                raise SystemExit(
                    f"unknown built-in phase {name!r}. options: {list(library)}")
            selected.append(library[name])
    elif not (args.cif or args.mp or args.cod):
        selected = list(library.values())

    used = [ph.color for ph in selected]
    for path in args.cif:
        c = assign_color(used); used.append(c)
        selected.append(from_structure_file(path, color=c))
    for mpid in args.mp:
        c = assign_color(used); used.append(c)
        selected.append(from_mp_id(mpid, color=c))
    for cid in args.cod:
        c = assign_color(used); used.append(c)
        selected.append(from_cod_id(cid, color=c))

    measured = []
    for mpath in args.measured:
        c = assign_color(used); used.append(c)
        measured.append(load_measured(mpath, color=c))

    if not selected and not measured:
        raise SystemExit("no phases or measured patterns selected")

    common = dict(wavelength_A=wl, crystallite_nm=args.crystallite,
                  tt_range=tt, show_hkl_labels=args.show_hkl,
                  kalpha12=args.kalpha12)
    if not args.no_pdf:
        save_pdf(out / "xrdkit_stacked.pdf", phases=selected,
                 measured=measured, stack=True, **common)
        save_pdf(out / "xrdkit_overlay.pdf", phases=selected,
                 measured=measured, stack=False, show_sticks=False, **common)
    if not args.no_gif:
        save_gif(out / "xrdkit.gif", phases=selected, measured=measured,
                 wavelength_A=wl, crystallite_nm=args.crystallite,
                 tt_range=tt, show_hkl_labels=args.show_hkl,
                 kalpha12=args.kalpha12)

    for ph in selected:
        print(f"  {ph.name:32s} [{ph.source}]")
    for m in measured:
        print(f"  {m.name:32s} [{m.source}]")


if __name__ == "__main__":
    main()
