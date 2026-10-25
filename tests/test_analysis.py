"""Known-answer tests on synthetic patterns (Kα1 + Kα2, Poisson counts).

Patterns are generated with pymatgen kinematic intensities and unit-area
pseudo-Voigt lines, so the refined cells must reproduce the input cells."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

from xrdkit import analysis as xa
from xrdkit.core import ka12_lines, pseudo_voigt_area
from xrdkit.plot import Measured, _clean_xy

L1, L2 = 1.54056, 1.54439
X = np.arange(20.0, 90.0, 0.02)


def _pattern(st, fwhm=0.12, eta=0.5, disp=0.0, seed=0, doublet=True):
    rng = np.random.default_rng(seed)
    y = 200.0 + 30.0 * np.exp(-(X - 20.0) / 30.0)
    for wl, w in ((L1, 1.0), (L2, 0.5)) if doublet else ((L1, 1.0),):
        pat = XRDCalculator(wavelength=wl).get_pattern(st, two_theta_range=(15, 95))
        for tt, inten in zip(pat.x, pat.y):
            th = np.radians(tt / 2)
            y = y + w * 2e4 * inten / 100 * pseudo_voigt_area(
                X, tt + disp * np.cos(th), fwhm, eta)
    return Measured("synthetic", X, rng.poisson(y).astype(float))


PT = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.9242), ["Pt"], [[0, 0, 0]])
STO = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
                                [[0, 0, 0], [.5, .5, .5], [.5, .5, 0]])
RUTILE = Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.5937, 2.9587),
                                   ["Ti", "O"], [[0, 0, 0], [.3048, .3048, 0]])


def _spec(sg, cell):
    return {"name": sg, "spacegroup": sg, "cell": cell, "Z": None}


@pytest.mark.parametrize("st, sg, cell0, abc", [
    (PT, "Fm-3m", (3.93, 3.93, 3.93, 90, 90, 90), (3.9242, 3.9242, 3.9242)),
    (RUTILE, "P4_2/mnm", (4.60, 4.60, 2.96, 90, 90, 90), (4.5937, 4.5937, 2.9587)),
])
def test_pawley_recovers_cell(st, sg, cell0, abc):
    r = xa.pawley_refine(_pattern(st), model=sg, cell0=cell0, check_systematics=False)
    assert r.status == xa.STATUS_OK
    assert np.allclose(r.cell[:3], abc, atol=5e-4)
    assert 0.8 < r.gof < 1.3
    assert 1.5 < r.durbin_watson < 2.5      # white-noise residuals


def test_pawley_perovskite_model_and_systematics():
    r = xa.pawley_refine(_pattern(STO, disp=0.05), model="Pm-3m", a_pc=3.90)
    assert abs(r.cell[0] - 3.905) < 2e-3
    assert r.cell_sys and np.isfinite(r.cell_sys[0])


def test_noise_is_unexplained():
    x = X
    m = Measured("noise", x, np.random.default_rng(1).poisson(
        np.full_like(x, 300.0)).astype(float))
    r = xa.pawley_refine(m, model="Pm-3m", a_pc=3.9, check_systematics=False)
    assert r.status == xa.STATUS_UNEXPLAINED


def test_fcc_is_flagged_as_not_perovskite():
    m = _pattern(PT)
    frac, mixed = xa._pc_index(m, 3.9242, L1, (20.0, 90.0), 0.015)
    assert frac > 0.9 and mixed < 0.05
    _, mixed_sto = xa._pc_index(_pattern(STO), 3.905, L1, (20.0, 90.0), 0.015)
    assert mixed_sto > 0.3


def _wh_peaks(D_nm=20.0, eps=0.002, scatter=0.0, seed=0):
    tt = np.array([30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    th = np.radians(tt / 2)
    bcos = 0.9 * L1 / (D_nm * 10) + 4 * eps * np.sin(th)
    bcos = bcos * (1 + scatter * np.random.default_rng(seed).standard_normal(tt.size))
    f = np.degrees(bcos / np.cos(th))
    df = pd.DataFrame({"two_theta": tt, "fwhm": f, "fwhm_esd": 0.002 * f,
                       "eta": np.ones_like(tt)})
    df.attrs["wavelength"] = L1
    return df


def test_williamson_hall_known_answer():
    wh = xa.williamson_hall(_wh_peaks(), correction="lorentzian")
    assert abs(wh["D_nm"] - 20.0) < 0.1
    assert abs(wh["strain"] - 0.002) < 1e-5
    assert wh["consistent"]


def test_williamson_hall_pure_size_is_valid_with_r2_near_zero():
    wh = xa.williamson_hall(_wh_peaks(eps=0.0, scatter=0.002, seed=3),
                            correction="lorentzian")
    assert wh["consistent"] and abs(wh["D_nm"] - 20.0) < 0.5


def test_williamson_hall_flags_anisotropic_scatter():
    wh = xa.williamson_hall(_wh_peaks(scatter=0.2), correction="lorentzian")
    assert not wh["consistent"]


def test_kalpha2_strip_removes_satellite():
    m = _pattern(PT)
    s = xa.strip_kalpha2(m, wl1=L1, wl2=L2, ratio=0.5)
    tt1 = 2 * np.degrees(np.arcsin(L1 / (2 * 3.9242 / np.sqrt(8))))   # Pt 220
    tt2 = 2 * np.degrees(np.arcsin(L2 / (2 * 3.9242 / np.sqrt(8))))
    i1 = np.argmin(abs(X - tt1))
    i2 = np.argmin(abs(X - tt2))
    base = 230.0
    assert (m.intensity[i2] - base) > 0.3 * (m.intensity[i1] - base)
    assert (s.intensity[i2] - base) < 0.1 * (s.intensity[i1] - base)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_pseudo_voigt_area_is_unit(eta):
    x = np.linspace(-200, 200, 400001)
    area = np.trapezoid(pseudo_voigt_area(x, 0.0, 0.2, eta), x)
    assert abs(area - 1.0) < 2e-3


def test_clean_xy_sorts_and_drops():
    x, _, w = _clean_xy([3.0, 1.0, np.nan, 2.0, 2.0], [1, 2, 3, 4, 5], "t")
    assert np.all(np.diff(x) > 0) and len(x) == 3 and w


def test_ka12_lines_from_mean_or_ka1():
    assert ka12_lines(1.54056)[:2] == (1.54056, 1.54439)
    assert ka12_lines(1.54184) is not None
    assert ka12_lines(1.0) is None


def test_plausible_formula():
    assert not xa._plausible_formula("O2")
    assert xa._plausible_formula("La0.6Sr0.4MnO3")


def test_find_peaks_table_on_pt():
    pk = xa.find_peaks_table(_pattern(PT, doublet=False))
    tt111 = 2 * np.degrees(np.arcsin(L1 / (2 * 3.9242 / np.sqrt(3))))
    assert np.min(abs(pk["two_theta"].to_numpy() - tt111)) < 0.01
    assert np.all(np.isfinite(pk["fwhm"]))


ANATASE = Structure.from_spacegroup("I4_1/amd", Lattice.tetragonal(3.785, 9.514),
                                    ["Ti", "O"], [[0, 0, 0], [0, 0, .2081]])
TI = Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(2.9508, 4.6855),
                               ["Ti"], [[1 / 3, 2 / 3, .25]])


def _mixture(parts, seed=0):
    rng = np.random.default_rng(seed)
    y = 200.0 + 30.0 * np.exp(-(X - 20.0) / 30.0)
    for st, amp in parts:
        for wl, w in ((L1, 1.0), (L2, 0.5)):
            pat = XRDCalculator(wavelength=wl).get_pattern(st, two_theta_range=(15, 95))
            for tt, inten in zip(pat.x, pat.y):
                y = y + w * amp * inten / 100 * pseudo_voigt_area(X, tt, 0.12, 0.5)
    return Measured("mixture", X, rng.poisson(y).astype(float))


def test_identify_phases_finds_both_phases_and_rejects_decoys():
    m = _mixture([(PT, 2e4), (RUTILE, 6e3)])
    cands = [{"id": n, "structure": s} for n, s in
             [("Pt", PT), ("rutile", RUTILE), ("anatase", ANATASE), ("STO", STO),
              ("Ti", TI)]]
    res = xa.identify_phases(m, candidates=cands, wavelength=(L1, L2, 0.5))
    assert sorted(res.phases["database id"]) == ["Pt", "rutile"]
    pt = res.phases.set_index("database id").loc["Pt"]
    ru = res.phases.set_index("database id").loc["rutile"]
    assert abs(pt["a (Å)"] - 3.9242) < 1e-3
    assert abs(ru["a (Å)"] - 4.5937) < 1e-3 and abs(ru["c (Å)"] - 2.9587) < 1e-3
    assert res.fit.gof < 1.3
    assert len(res.unexplained) <= 1   # at most one sub-1 % kinematic line


def test_identify_phases_on_noise_finds_nothing():
    m = Measured("noise", X, np.random.default_rng(2).poisson(
        np.full_like(X, 300.0)).astype(float))
    res = xa.identify_phases(m, candidates=[{"id": "Pt", "structure": PT}])
    assert res.phases.empty and res.fit is None
