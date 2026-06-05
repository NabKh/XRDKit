"""Phase abstraction and loaders: ICDD cards, local structure files, COD, MP."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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


# Additional ICDD reference cards beyond the original five PEMWE phases.
_EXTRA_CARDS = {
    "IrO₂ rutile": {
        "system": "tetragonal", "a": 4.4983, "c": 3.1544,
        "ref": "PDF 15-870", "spacegroup": "P4_2/mnm", "color": "#117A65",
        "hkl_I": [
            ((1,1,0), 100.0), ((1,0,1), 80.0), ((2,0,0), 25.0),
            ((1,1,1),  20.0), ((2,1,0), 10.0), ((2,1,1), 65.0),
            ((2,2,0),  30.0), ((0,0,2), 18.0), ((3,1,0), 20.0),
            ((3,0,1),  35.0), ((1,1,2), 18.0), ((3,1,1), 10.0),
        ],
    },
    "RuO₂ rutile": {
        "system": "tetragonal", "a": 4.4994, "c": 3.1071,
        "ref": "PDF 40-1290", "spacegroup": "P4_2/mnm", "color": "#7F4FAA",
        "hkl_I": [
            ((1,1,0), 100.0), ((1,0,1), 80.0), ((2,0,0), 25.0),
            ((1,1,1),  20.0), ((2,1,0), 10.0), ((2,1,1), 60.0),
            ((2,2,0),  28.0), ((0,0,2), 16.0), ((3,1,0), 20.0),
            ((3,0,1),  32.0), ((1,1,2), 18.0),
        ],
    },
    "TiO₂ brookite (proxy)": {
        # The true Pbca brookite cell is orthorhombic; this tetragonal proxy
        # gives the four strongest peaks roughly in the right places. Replace
        # with a proper brookite CIF via the "Upload structure" tab for any
        # serious brookite work.
        "system": "tetragonal", "a": 5.4558, "c": 5.1429,
        "ref": "PDF 29-1360 (tetragonal proxy)", "spacegroup": "Pbca (proxy)",
        "color": "#B7950B",
        "hkl_I": [
            ((1,2,1),  90.0), ((1,1,1), 100.0), ((1,2,0), 20.0),
            ((2,0,0),  20.0), ((0,1,2),  20.0),
        ],
    },
}


def builtin_phases() -> Dict[str, Phase]:
    """Return all built-in ICDD-card phases keyed by display name."""
    out: Dict[str, Phase] = {}
    for name, mat in _PARENT_MATERIALS.items():
        out[name] = _card_phase(name, mat, _PARENT_PALETTE[name],
                                "ICDD card (built-in)")
    for name, mat in _EXTRA_CARDS.items():
        out[name] = _card_phase(
            name, {k: v for k, v in mat.items() if k != "color"},
            mat["color"], f"ICDD card ({mat['ref']})",
        )
    return out


def _structure_phase(name: str, structure, color: str, source: str, info: dict) -> Phase:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    def fn(wl: float) -> PeakList:
        calc = XRDCalculator(wavelength=float(wl))
        pat = calc.get_pattern(structure, two_theta_range=(2.0, 160.0))
        peaks: PeakList = []
        for tt, I, hkls in zip(pat.x, pat.y, pat.hkls):
            hkl = tuple(int(v) for v in hkls[0]["hkl"]) if hkls else (0, 0, 0)
            peaks.append((float(tt), float(I), hkl))
        return peaks
    return Phase(name=name, color=color, source=source, peaks_fn=fn, info=info)


def from_cif(path, name=None, color="#444444", source_label=None) -> Phase:
    """Backwards-compatible alias for :func:`from_structure_file`."""
    return from_structure_file(path, name=name, color=color, source_label=source_label)


def from_structure_file(path, name=None, color="#444444",
                         source_label=None) -> Phase:
    """Load any pymatgen-readable structure file (CIF, POSCAR, .vasp, .json, …)."""
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
                            source_label or f"file: {p.name}", info)


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
    """Query the Crystallography Open Database for structures matching a formula."""
    import requests
    r = requests.get(
        "https://www.crystallography.net/cod/result",
        params={"formula": _cod_format_formula(formula), "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for entry in data[:max_results]:
        out.append({
            "cod_id": str(entry.get("file")),
            "formula": entry.get("formula"),
            "spacegroup": entry.get("sg"),
            "title": (entry.get("commonname") or entry.get("chemname")
                      or entry.get("mineral") or "")[:80],
        })
    return out


def from_cod_id(cod_id, color="#444444", name=None, cache_dir=None) -> Phase:
    """Fetch a COD CIF by ID (cached locally) and wrap as a Phase."""
    import requests
    cache = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "xrdkit" / "cod")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{cod_id}.cif"
    if not target.exists():
        url = f"https://www.crystallography.net/cod/{cod_id}.cif"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        target.write_text(r.text)
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
