"""Command-line entry point: render PDF / GIF without opening the GUI.

``xrdkit analyze FILE …`` runs the Pawley model comparison instead and writes
JSON / CSV results (see ``xrdkit analyze -h``)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from xrdkit.core import WAVELENGTHS, WAVELENGTHS_KA12
from xrdkit.db import (builtin_phases, from_structure_file, from_mp_id,
                        from_cod_id, assign_color)
from xrdkit.plot import save_pdf, save_gif, load_measured


def parse_args():
    p = argparse.ArgumentParser(
        prog="xrdkit",
        description="Render XRD reference figures from built-in phases, structure "
                    "files, COD / Materials Project, and measured patterns.",
    )
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--phase", action="append", default=[],
                   help="built-in reference phase (repeatable). "
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


def analyze(argv):
    from xrdkit import analysis as xa
    from xrdkit.plot import load_measured_excel

    wls = {**WAVELENGTHS_KA12, **WAVELENGTHS}
    p = argparse.ArgumentParser(
        prog="xrdkit analyze",
        description="Pawley-refine candidate cells against measured patterns and "
                    "rank them by BIC_eff. Writes <out>/<pattern>.json (every "
                    "model, with status and provenance) and, for the best model, "
                    "<pattern>_profile.csv and <pattern>_reflections.csv.")
    p.add_argument("files", nargs="+", help="pattern files (.xy/.xrdml/.brml/.xlsx)")
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--models", nargs="+", default=["Pm-3m", "R-3c", "Pnma"],
                   help=f"perovskite tilt models ({', '.join(xa.PEROVSKITE_MODELS)}) "
                        "and/or structure files (CIF, POSCAR)")
    p.add_argument("--wavelength", default="Cu Kα1/Kα2", choices=list(wls))
    p.add_argument("--tt-min", type=float, default=20.0)
    p.add_argument("--tt-max", type=float, default=90.0)
    p.add_argument("--offset", choices=["displacement", "zero"], default="displacement")
    p.add_argument("--zero-fixed", type=float, default=0.0,
                   help="instrument zero (°) held fixed when refining displacement")
    p.add_argument("--a-pc", type=float, default=None,
                   help="pseudocubic a seed (Å); default: estimated from the data")
    p.add_argument("--strip-ka2", action="store_true",
                   help="Rachinger-strip Kα2 first (default: fit the doublet)")
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wl = wls[a.wavelength]
    wl = wl[:2] if isinstance(wl, tuple) else wl
    models = []
    for md in a.models:
        if md in xa.PEROVSKITE_MODELS:
            models.append(md)
        else:
            from pymatgen.core import Structure
            sp = xa.phase_spec(Structure.from_file(md))
            sp["name"] = Path(md).stem
            models.append(sp)
    pats = []
    for f in a.files:
        pats += (load_measured_excel(f) if Path(f).suffix.lower() in (".xlsx", ".xlsm")
                 else [load_measured(f)])
    for m in pats:
        if a.strip_ka2:
            full = wls[a.wavelength]
            if not isinstance(full, tuple):
                raise SystemExit("--strip-ka2 needs a Kα1/Kα2 wavelength")
            m = xa.strip_kalpha2(m, wl1=full[0], wl2=full[1], ratio=full[2])
        res, df = xa.compare_models(
            m, models=models, wavelength=wl, tt_range=(a.tt_min, a.tt_max),
            a_pc=a.a_pc, refine_disp=a.offset == "displacement",
            refine_zero=a.offset == "zero", zero_fixed=a.zero_fixed)
        prov = {"input": m.source, "input_sha256": m.info.get("sha256"),
                "sheet": m.info.get("sheet"), "wavelength": wl,
                "tt_range": [a.tt_min, a.tt_max], "offset": a.offset,
                "zero_fixed": a.zero_fixed, "kalpha2_stripped": a.strip_ka2}
        ok = [r for r in res if not isinstance(r, Exception)]
        doc = {**prov, "warning": df.attrs.get("warning"),
               "models": [r.to_dict() for r in ok],
               "failed": [str(r) for r in res if isinstance(r, Exception)],
               "ranking": df["model"].tolist() if "model" in df else []}
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in m.name)
        (out / f"{stem}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        best = next((r for r in ok if len(df) and r.model == df["model"].iloc[0]), None)
        if best is not None:
            best.profile_frame().to_csv(out / f"{stem}_profile.csv", index=False)
            best.reflections_frame().to_csv(out / f"{stem}_reflections.csv", index=False)
            print(f"{m.name}: best {best.model} [{best.status}] "
                  f"a,b,c = {', '.join(f'{v:.5f}' for v in best.cell[:3])} Å "
                  f"Rwp {100 * best.Rwp:.2f} %")
        if df.attrs.get("warning"):
            print(f"  warning: {df.attrs['warning']}")


def main():
    if sys.argv[1:2] == ["analyze"]:
        return analyze(sys.argv[2:])
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
