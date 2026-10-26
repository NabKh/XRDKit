"""Phase abstraction and loaders: built-in references, structure files, COD, MP."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from xrdkit.core import MATERIALS as _PARENT_MATERIALS
from xrdkit.core import PALETTE as _PARENT_PALETTE
from xrdkit.core import two_theta_peaks as _card_two_theta_peaks


PeakList = List[Tuple[float, float, Tuple[int, int, int]]]


@dataclass
class Phase:
    name: str
    color: str
    source: str
    peaks_fn: Callable[[float], PeakList]
    info: Dict = field(default_factory=dict)

    def peaks(self, wavelength_A: float, tt_range=(5.0, 130.0)) -> PeakList:
        out = []
        for tt, I, hkl in self.peaks_fn(wavelength_A):
            if tt_range[0] <= tt <= tt_range[1]:
                out.append((tt, I, hkl))
        return out


def _card_phase(name: str, mat: dict, color: str, source: str) -> Phase:
    def fn(wl: float) -> PeakList:
        return _card_two_theta_peaks(mat, wl)
    info = {
        "system": mat["system"], "a": mat["a"], "c": mat.get("c"),
        "ref": mat.get("ref", ""), "spacegroup": mat.get("spacegroup", ""),
    }
    return Phase(name=name, color=color, source=source, peaks_fn=fn, info=info)


# Built-in phases computed from refined crystal structures (COD, public
# domain): space group, cell, Wyckoff positions and U_eq = (U11+U22+U33)/3
# copied from the cited CIF; B = 8π²·U_eq is averaged per element.
_STRUCTURE_CARDS = {
    "IrO₂ rutile": {
        "ref": "COD 1548817", "cite": "Zhao & Kulik, JCTC 14, 670 (2018)",
        "spacegroup": "P4_2/mnm", "cell": (4.5051, 4.5051, 3.1586, 90, 90, 90),
        "sites": [("Ir", (0.0, 0.0, 0.0), 0.009467),
                  ("O", (0.3077, 0.3077, 0.0), 0.010400)],
        "color": "#117A65",
    },
    "RuO₂ rutile": {
        "ref": "COD 1000058", "cite": "Bolzan et al., Acta Cryst. B 53, 373 (1997)",
        "spacegroup": "P4_2/mnm", "cell": (4.4968, 4.4968, 3.1049, 90, 90, 90),
        "sites": [("Ru", (0.0, 0.0, 0.0), 0.004067),
                  ("O", (0.3053, 0.3053, 0.0), 0.006533)],
        "color": "#7F4FAA",
    },
    "TiO₂ brookite": {
        "ref": "COD 9004137", "cite": "Meagher & Lager, Can. Mineral. 17, 77 (1979)",
        "spacegroup": "Pbca", "cell": (9.174, 5.449, 5.138, 90, 90, 90),
        "sites": [("Ti", (0.1289, 0.0972, 0.8628), 0.004733),
                  ("O", (0.0095, 0.1491, 0.1835), 0.005833),
                  ("O", (0.2314, 0.1110, 0.5366), 0.006763)],
        "color": "#B7950B",
    },
}


def _builtin_structure(card: dict):
    from pymatgen.core import Lattice, Structure
    lat = Lattice.from_parameters(*card["cell"])
    st = Structure.from_spacegroup(card["spacegroup"], lat,
                                   [e for e, _, _ in card["sites"]],
                                   [x for _, x, _ in card["sites"]])
    B: Dict[str, List[float]] = {}
    for e, _, u in card["sites"]:
        B.setdefault(e, []).append(8.0 * np.pi ** 2 * u)
    return st, {e: float(np.mean(v)) for e, v in B.items()}


def builtin_phases() -> Dict[str, Phase]:
    """Built-in reference phases keyed by display name.

    Pt, Ir, α-Ti, rutile and anatase use tabulated card-style line lists
    (PDF numbers as recorded in :data:`xrdkit.core.MATERIALS`; the values were
    not re-checked against ICDD). IrO₂, RuO₂ and brookite are kinematic
    patterns of the cited COD structures, with isotropic B factors."""
    out: Dict[str, Phase] = {}
    for name, mat in _PARENT_MATERIALS.items():
        out[name] = _card_phase(name, mat, _PARENT_PALETTE[name],
                                f"reference line list ({mat.get('ref', '')})")
    for name, card in _STRUCTURE_CARDS.items():
        st, B = _builtin_structure(card)
        a, b_, c = st.lattice.abc
        out[name] = _structure_phase(
            name, st, card["color"], f"{card['ref']} structure ({card['cite']})",
            {"formula": st.composition.reduced_formula, "ref": card["ref"],
             "spacegroup": card["spacegroup"], "a": a, "c": c,
             "lattice": dict(zip(("a", "b", "c", "α", "β", "γ"),
                                 list(st.lattice.abc) + list(st.lattice.angles))),
             "n_sites": len(st), "B_iso": B},
            debye_waller=B)
    return out


def _structure_phase(name: str, structure, color: str, source: str, info: dict,
                     debye_waller: Optional[Dict[str, float]] = None) -> Phase:
    """Phase whose lines are pymatgen kinematic intensities of ``structure``.

    ``debye_waller`` maps element → isotropic B (Å²); without it B = 0, which
    overestimates high-angle intensities (≈ 25 % at 2θ = 90° for B = 0.5 Å²,
    Cu Kα). Anomalous dispersion f′, f″ is not included."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    dw = dict(debye_waller) if debye_waller else None

    def fn(wl: float) -> PeakList:
        calc = XRDCalculator(wavelength=float(wl), debye_waller_factors=dw)
        pat = calc.get_pattern(structure, two_theta_range=(2.0, 160.0))
        peaks: PeakList = []
        for tt, I, hkls in zip(pat.x, pat.y, pat.hkls):
            hkl = tuple(int(v) for v in hkls[0]["hkl"]) if hkls else (0, 0, 0)
            if len(hkl) == 4:  # Miller–Bravais (h k i l) → (h k l)
                hkl = (hkl[0], hkl[1], hkl[3])
            peaks.append((float(tt), float(I), hkl))
        return peaks
    info = dict(info)
    info.setdefault("structure", structure)
    info.setdefault("B_iso", dw or {})
    return Phase(name=name, color=color, source=source, peaks_fn=fn, info=info)


def from_cif(path, name=None, color="#444444", source_label=None) -> Phase:
    """Backwards-compatible alias for :func:`from_structure_file`."""
    return from_structure_file(path, name=name, color=color, source_label=source_label)


def from_structure_file(path, name=None, color="#444444",
                         source_label=None, debye_waller=None) -> Phase:
    """Load any pymatgen-readable structure file (CIF, POSCAR, .vasp, .json, …).

    The cell is used as given (no primitive/conventional conversion).
    ``debye_waller``: element → B (Å²) for the intensities (default B = 0)."""
    from pymatgen.core import Structure
    p = Path(path)
    s = Structure.from_file(str(p))
    sg_sym, sg_num = s.get_space_group_info()
    formula = s.composition.reduced_formula
    if name is None:
        name = formula
    info = {
        "path": str(p), "formula": formula, "spacegroup": sg_sym,
        "spacegroup_number": sg_num,
        "lattice": dict(zip(("a", "b", "c", "α", "β", "γ"),
                            list(s.lattice.abc) + list(s.lattice.angles))),
        "n_sites": int(len(s)),
    }
    return _structure_phase(name, s, color,
                            source_label or f"file: {p.name}", info,
                            debye_waller=debye_waller)


def from_mp_id(mp_id, api_key=None, color="#444444", name=None) -> Phase:
    """Fetch a Materials Project structure by mp-id and wrap as a Phase."""
    from mp_api.client import MPRester
    key = api_key or os.environ.get("MP_API_KEY")
    with MPRester(key) as mpr:
        s = mpr.get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    sg_sym, sg_num = s.get_space_group_info()
    formula = s.composition.reduced_formula
    if name is None:
        name = f"{formula} ({mp_id})"
    info = {
        "mp_id": mp_id, "formula": formula, "spacegroup": sg_sym,
        "spacegroup_number": sg_num,
        "lattice": dict(zip(("a", "b", "c", "α", "β", "γ"),
                            list(s.lattice.abc) + list(s.lattice.angles))),
        "n_sites": int(len(s)),
    }
    return _structure_phase(name, s, color, f"Materials Project {mp_id}", info)


def search_mp(formula, api_key=None, max_results=12):
    """Query Materials Project for structures matching a formula."""
    from mp_api.client import MPRester
    key = api_key or os.environ.get("MP_API_KEY")
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "symmetry", "energy_above_hull"],
        )
    rows = []
    for d in docs:
        rows.append({
            "mp_id": str(d.material_id),
            "formula": d.formula_pretty,
            "spacegroup": getattr(d.symmetry, "symbol", "?"),
            "e_above_hull": d.energy_above_hull,
        })
    rows.sort(key=lambda x: (x["e_above_hull"] is None, x["e_above_hull"] or 0))
    return rows[:max_results]


def _cod_format_formula(f: str) -> str:
    """Format a chemical formula for the COD REST API.

    COD expects elements space-separated with the unit-count suffix dropped,
    e.g. 'Ir O2', 'Co3 O4', 'Mo S2'. Pymatgen's Composition handles the
    grouping; we strip explicit '1' suffixes.
    """
    from pymatgen.core import Composition
    try:
        s = Composition(f).formula
    except Exception:
        s = f
    return re.sub(r"([A-Z][a-z]?)1(?![0-9])", r"\1", s)


def search_cod(formula, max_results=25):
    """Query the Crystallography Open Database for structures matching a formula.

    Uses COD's OPTIMADE endpoint (the legacy ``/cod/result`` JSON search is no
    longer served). OPTIMADE cannot filter on the formula itself, so entries
    with exactly the formula's element set are fetched and those whose reduced
    formula matches are listed first, followed by non-stoichiometric and
    other compositions of the same elements.
    """
    from pymatgen.core import Composition
    from xrdkit.analysis import cod_optimade_query
    comp = Composition(_cod_format_formula(formula).replace(" ", ""))
    target = comp.reduced_formula
    els = sorted(el.symbol for el in comp.elements)
    rows = cod_optimade_query(els, nelements=len(els))
    rows = [r for r in rows if r["has_coordinates"]]
    rows.sort(key=lambda r: (r["formula"] != target,
                             r["R"] if r["R"] is not None else 1.0))
    return [{"cod_id": r["cod_id"], "formula": r["formula"],
             "spacegroup": r["spacegroup"], "title": r["title"]}
            for r in rows[:max_results]]


def from_cod_id(cod_id, color="#444444", name=None, cache_dir=None) -> Phase:
    """Fetch a COD CIF by ID (cached locally) and wrap as a Phase."""
    from xrdkit.core import http_get
    cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "xrdkit" / "cod")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{cod_id}.cif"
    if not target.exists():
        url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        text = http_get(url, timeout=30).text
        if "_cell_length_a" not in text:   # an error page, not a CIF: do not cache
            raise ValueError(f"COD {cod_id}: the server did not return a CIF")
        tmp = target.with_suffix(".part")
        tmp.write_text(text)
        tmp.replace(target)
    return from_structure_file(
        target, name=name or f"COD {cod_id}", color=color,
        source_label=f"COD {cod_id}",
    )


# Colour-cycle for user-added phases (Wong colour-blind-safe palette extended).
PALETTE_EXTRA = [
    "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF", "#393B79", "#637939", "#8C6D31",
    "#843C39", "#7B4173",
]


def assign_color(used_colors: List[str]) -> str:
    """Pick the next colour not already used."""
    for c in PALETTE_EXTRA:
        if c not in used_colors:
            return c
    return PALETTE_EXTRA[len(used_colors) % len(PALETTE_EXTRA)]
