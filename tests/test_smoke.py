"""Smoke tests — pure-Python, no network, no GUI."""
from __future__ import annotations

import numpy as np
import pytest

from xrdkit.core import (
    MATERIALS, WAVELENGTHS, WAVELENGTHS_KA12,
    d_spacing, two_theta_peaks, pseudo_voigt, scherrer_fwhm_deg,
)
from xrdkit.db import (
    builtin_phases, _cod_format_formula, assign_color, Phase,
)
from xrdkit.plot import make_figure, peaks_table


def test_builtin_phases_load():
    lib = builtin_phases()
    assert len(lib) == 8
    for ph in lib.values():
        assert isinstance(ph, Phase)
        assert ph.color.startswith("#")
        assert "ICDD card" in ph.source


def test_bragg_pt_111_cu_kalpha1():
    """Pt(111) at Cu Kα1 should be 39.76° (ICDD card value)."""
    peaks = two_theta_peaks(MATERIALS["Pt"], WAVELENGTHS["Cu Kα1"])
    p111 = next(p for p in peaks if p[2] == (1, 1, 1))
    assert abs(p111[0] - 39.764) < 0.03
    assert p111[1] == 100.0


def test_bragg_tio2_anatase_101_cu_kalpha1():
    """Anatase (101) at Cu Kα1 should be 25.28° (ICDD card value)."""
    peaks = two_theta_peaks(MATERIALS["TiO₂ anatase"], WAVELENGTHS["Cu Kα1"])
    p101 = next(p for p in peaks if p[2] == (1, 0, 1))
    assert abs(p101[0] - 25.28) < 0.03


def test_d_spacing_cubic_formula():
    """For cubic Pt(200), d = a/2 exactly."""
    mat = MATERIALS["Pt"]
    assert abs(d_spacing(mat, (2, 0, 0)) - mat["a"] / 2.0) < 1e-12


def test_d_spacing_hcp_formula():
    """For hcp α-Ti(002), d = c/2 exactly."""
    mat = MATERIALS["Ti (α-hcp)"]
    assert abs(d_spacing(mat, (0, 0, 2)) - mat["c"] / 2.0) < 1e-12


def test_wavelength_switch_shifts_peaks_correctly():
    """Switching Cu Kα → Co Kα shifts Pt(111) to higher 2θ by ~6°."""
    pks_cu = two_theta_peaks(MATERIALS["Pt"], WAVELENGTHS["Cu Kα"])
    pks_co = two_theta_peaks(MATERIALS["Pt"], WAVELENGTHS["Co Kα"])
    p111_cu = next(p[0] for p in pks_cu if p[2] == (1, 1, 1))
    p111_co = next(p[0] for p in pks_co if p[2] == (1, 1, 1))
    assert p111_co > p111_cu
    assert 5.0 < (p111_co - p111_cu) < 7.5


def test_scherrer_smaller_D_broader():
    """Smaller crystallites broaden peaks (Scherrer)."""
    b_small = scherrer_fwhm_deg(40.0, WAVELENGTHS["Cu Kα"], crystallite_nm=5)
    b_big   = scherrer_fwhm_deg(40.0, WAVELENGTHS["Cu Kα"], crystallite_nm=50)
    assert b_small > 5 * b_big


def test_pseudo_voigt_is_normalised_at_peak():
    """At x = x0 the pseudo-Voigt equals 1 regardless of η."""
    for eta in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert abs(pseudo_voigt(np.array([0.5]), 0.5, 0.2, eta=eta)[0] - 1.0) < 1e-9


def test_cod_formula_normalisation():
    """COD requires elements space-separated with '1' suffix dropped."""
    assert _cod_format_formula("IrO2") == "Ir O2"
    assert _cod_format_formula("Co3O4") == "Co3 O4"
    assert _cod_format_formula("Pt") == "Pt"
    assert _cod_format_formula("MoS2") == "Mo S2"


def test_assign_color_unique_until_palette_exhausted():
    used = []
    c1 = assign_color(used); used.append(c1)
    c2 = assign_color(used); used.append(c2)
    assert c1 != c2


def test_make_figure_with_builtins():
    """A figure with three built-in phases renders without error."""
    lib = builtin_phases()
    phases = [lib["Pt"], lib["Ir"], lib["IrO₂ rutile"]]
    fig, ax = make_figure(phases, wavelength_A=WAVELENGTHS["Cu Kα"])
    assert len(fig.axes) >= 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_make_figure_with_hkl_labels():
    """The (hkl)-labelling path runs without exception."""
    lib = builtin_phases()
    fig, ax = make_figure([lib["Pt"]], wavelength_A=WAVELENGTHS["Cu Kα"],
                           show_hkl_labels=True, hkl_threshold=5.0)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_make_figure_with_kalpha12():
    """Kα1/Kα2 doublet rendering executes without error."""
    lib = builtin_phases()
    fig, ax = make_figure([lib["Pt"]],
                           wavelength_A=WAVELENGTHS_KA12["Cu Kα1/Kα2"][:2],
                           kalpha12=True)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_peaks_table_returns_expected_columns():
    lib = builtin_phases()
    rows = peaks_table([lib["Pt"]], WAVELENGTHS["Cu Kα"], tt_range=(20, 90))
    assert rows
    cols = set(rows[0])
    for k in ("Phase", "2θ (°)", "d (Å)", "I (%)", "hkl", "Source"):
        assert k in cols
    # the strongest reflection of Pt should be (111) near 39.76°
    strongest = max(rows, key=lambda r: r["I (%)"])
    assert strongest["hkl"] == "1 1 1"
    assert abs(strongest["2θ (°)"] - 39.76) < 0.1
