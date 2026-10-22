"""Quantitative analysis of measured powder patterns.

* :func:`find_peaks_table` — background (SNIP), peak search and pseudo-Voigt
  fits with uncertainties (2θ, d, FWHM, height, area).
* :func:`strip_kalpha2` — Rachinger Kα2 stripping.
* :func:`williamson_hall` — size/strain from integral-breadth-free FWHM data
  with instrumental correction.
* :func:`pawley_refine` / :func:`compare_models` — whole-pattern Pawley fit of
  the unit cell for candidate space groups (cell, zero, sample displacement,
  Caglioti U V W, η, Chebyshev background; free reflection intensities).
* :func:`unassigned_peaks` / :func:`match_cod` — rank COD entries against the
  peaks a model does not explain (minor / secondary phases).

Conventions: angles in degrees 2θ, wavelengths in Å, d in Å. Uncertainties are
1σ from the least-squares covariance scaled by the reduced χ². Point-to-point
correlated residuals make these estimates optimistic (Bérar & Lelann, J. Appl.
Cryst. 24, 1, 1991); treat them as lower bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from xrdkit.core import pseudo_voigt
from xrdkit.plot import Measured, _snip_background

_PV_AREA_G = np.sqrt(np.pi / (4.0 * np.log(2.0)))


def _pv_area_factor(fwhm, eta):
    """Area of a unit-height pseudo-Voigt: FWHM·(η·π/2 + (1−η)·√(π/4ln2))."""
    return fwhm * (eta * 0.5 * np.pi + (1.0 - eta) * _PV_AREA_G)


def _wl_tuple(wavelength) -> Tuple[float, Optional[float], float]:
    """Normalise a wavelength spec to (λ1, λ2 or None, Kα2/Kα1 ratio)."""
    if isinstance(wavelength, (tuple, list)):
        wl1 = float(wavelength[0])
        wl2 = float(wavelength[1]) if len(wavelength) > 1 and wavelength[1] else None
        ratio = float(wavelength[2]) if len(wavelength) > 2 else 0.5
        return wl1, wl2, ratio
    return float(wavelength), None, 0.0


def _ka2_position(tt1, wl1, wl2):
    """2θ of the Kα2 line for a reflection whose Kα1 line sits at ``tt1``."""
    s = np.sin(np.radians(0.5 * np.asarray(tt1, float))) * (wl2 / wl1)
    return 2.0 * np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def _crop(m: Measured, tt_range=None):
    x = np.asarray(m.two_theta, float)
    y = np.asarray(m.intensity, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if tt_range is not None:
        k = (x >= tt_range[0]) & (x <= tt_range[1])
        x, y = x[k], y[k]
    return x, y


def estimate_background(x, y, window_deg: float = 2.5):
    """SNIP background with a clipping window of ``window_deg`` degrees."""
    step = float(np.median(np.diff(x))) if x.size > 1 else 0.02
    n_iter = int(max(4, min(400, round(window_deg / max(step, 1e-6)))))
    return _snip_background(y, n_iter=n_iter)


def counting_weights(x, y, mode: str = "auto"):
    """Least-squares weights 1/σ² and the weighting actually used.

    "poisson": σ² = y (raw detector counts). "uniform": σ² = σ_n², the
    point-to-point noise estimated from the data. "auto" picks Poisson only if
    the observed noise is compatible with √(background), i.e. the file holds
    raw counts; exported intensities that were scaled, averaged, smoothed or
    offset for stacking fail that test and get uniform weights.
    """
    y = np.asarray(y, float)
    bkg = estimate_background(x, y)
    sn = _robust_noise(y - bkg)
    if mode == "auto":
        ratio = sn / np.sqrt(max(float(np.median(bkg)), 1e-12))
        mode = "poisson" if 0.5 <= ratio <= 2.0 else "uniform"
    if mode == "poisson":
        return 1.0 / np.clip(y, 1.0, None), "poisson"
    return np.full_like(y, 1.0 / sn ** 2), "uniform"


def _robust_noise(y_net):
    """σ of the point-to-point noise from the MAD of first differences."""
    d = np.diff(y_net)
    if d.size == 0:
        return 1.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)) or 1.0


# ===========================================================================
#  1. Peak search and profile fitting
# ===========================================================================

def find_peaks_table(m: Measured, wavelength=1.54056, tt_range=None,
                     min_snr: float = 5.0, bkg_window_deg: float = 2.5,
                     kalpha2: bool = True, min_sep_deg: float = 0.06):
    """Locate and fit the Bragg peaks of a measured pattern.

    Each cluster of overlapping peaks is fitted together with pseudo-Voigt
    profiles (own 2θ, FWHM, height, η per peak) on top of the SNIP background
    plus a local linear correction. When ``kalpha2`` is true and ``wavelength``
    is a (λ1, λ2, ratio) tuple, every peak carries a Kα2 companion at fixed
    position and ratio, so the reported FWHM and area belong to Kα1 alone.

    Returns a pandas DataFrame, one row per peak, with 1σ columns ``*_esd``.
    """
    import pandas as pd
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    wl1, wl2, ratio = _wl_tuple(wavelength)
    use_ka2 = bool(kalpha2 and wl2)
    x, y = _crop(m, tt_range)
    if x.size < 10:
        raise ValueError("pattern has fewer than 10 points in the chosen range")
    step = float(np.median(np.diff(x)))
    bkg = estimate_background(x, y, bkg_window_deg)
    net = y - bkg
    sigma_n = _robust_noise(net)
    idx, props = find_peaks(net, prominence=min_snr * sigma_n,
                            distance=max(1, int(round(min_sep_deg / step))),
                            width=1.5)
    if idx.size == 0:
        return pd.DataFrame(columns=["two_theta", "two_theta_esd", "d", "d_esd",
                                     "fwhm", "fwhm_esd", "height", "height_esd",
                                     "area", "area_esd", "eta", "snr", "cluster"])
    widths = np.clip(props["widths"] * step, 2 * step, 3.0)
    if use_ka2:
        # find_peaks sees unresolved Kα1+Kα2 as one wider peak
        widths = np.clip(widths * 0.8, 2 * step, 3.0)
    centres = x[idx]

    # group peaks whose profiles overlap
    clusters, cur = [], [0]
    for i in range(1, idx.size):
        gap = centres[i] - centres[cur[-1]]
        reach = 2.5 * max(widths[i], widths[cur[-1]])
        if use_ka2:
            reach += _ka2_position(centres[cur[-1]], wl1, wl2) - centres[cur[-1]]
        if gap < reach:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)

    rows = []
    for cid, members in enumerate(clusters):
        t0 = centres[members]
        f0 = widths[members]
        h0 = np.clip(net[idx[members]], sigma_n, None)
        lo = t0.min() - 4.0 * f0.max()
        hi = t0.max() + 4.0 * f0.max() + (
            (_ka2_position(t0.max(), wl1, wl2) - t0.max()) if use_ka2 else 0.0)
        k = (x >= lo) & (x <= hi)
        xs, ys = x[k], net[k]
        sig = np.sqrt(np.clip(y[k], 1.0, None))
        xm = 0.5 * (lo + hi)
        n = len(members)

        def model(xx, *p):
            out = p[-2] + p[-1] * (xx - xm)
            for j in range(n):
                tt, f, h, e = p[4 * j:4 * j + 4]
                out = out + h * pseudo_voigt(xx, tt, f, e)
                if use_ka2:
                    out = out + ratio * h * pseudo_voigt(
                        xx, _ka2_position(tt, wl1, wl2), f, e)
            return out

        p0, lb, ub = [], [], []
        for j in range(n):
            p0 += [t0[j], f0[j], h0[j], 0.5]
            lb += [t0[j] - max(0.5 * f0[j], 2 * step), step, 0.0, 0.0]
            ub += [t0[j] + max(0.5 * f0[j], 2 * step), 4.0, np.inf, 1.0]
        p0 += [0.0, 0.0]
        lb += [-np.inf, -np.inf]
        ub += [np.inf, np.inf]
        try:
            if xs.size <= len(p0):
                raise RuntimeError("too few points")
            popt, pcov = curve_fit(model, xs, ys, p0=p0, sigma=sig,
                                   absolute_sigma=False, bounds=(lb, ub),
                                   maxfev=20000)
            perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
        except Exception:
            popt = np.asarray(p0, float)
            pcov = np.full((len(p0), len(p0)), np.nan)
            perr = np.full(len(p0), np.nan)

        for j in range(n):
            tt, f, h, e = popt[4 * j:4 * j + 4]
            s_tt, s_f, s_h, _ = perr[4 * j:4 * j + 4]
            area = h * _pv_area_factor(f, e)
            g = np.array([0.0, area / f if f else 0.0, area / h if h else 0.0,
                          h * f * (0.5 * np.pi - _PV_AREA_G)])
            cov = pcov[4 * j:4 * j + 4, 4 * j:4 * j + 4]
            s_area = float(np.sqrt(max(g @ cov @ g, 0.0))) if np.all(np.isfinite(cov)) else np.nan
            th = np.radians(0.5 * tt)
            d = wl1 / (2.0 * np.sin(th))
            s_d = d / np.tan(th) * np.radians(0.5 * s_tt)
            rows.append({
                "two_theta": tt, "two_theta_esd": s_tt, "d": d, "d_esd": s_d,
                "fwhm": f, "fwhm_esd": s_f, "height": h, "height_esd": s_h,
                "area": area, "area_esd": s_area, "eta": e,
                "snr": h / sigma_n, "cluster": cid,
            })
    df = pd.DataFrame(rows).sort_values("two_theta").reset_index(drop=True)
    # drop components the fit shrank below threshold or to a one-step spike
    df = df[(df["snr"] >= min_snr) & (df["fwhm"] > 1.05 * step)].reset_index(drop=True)
    df.attrs.update({"noise_sigma": sigma_n, "kalpha2_fitted": use_ka2,
                     "wavelength": wl1})
    return df


# ===========================================================================
#  2. Kα2 stripping (Rachinger, J. Sci. Instrum. 25, 254, 1948)
# ===========================================================================

def strip_kalpha2(m: Measured, wl1: float = 1.54056, wl2: float = 1.54439,
                  ratio: float = 0.5, bkg_window_deg: float = 2.5) -> Measured:
    """Return a Kα1-only copy of ``m`` by Rachinger's recursion.

    I₁(2θ) = I(2θ) − R·I₁(2θ′), where 2θ′ is the Kα1 position whose Kα2 line
    falls at 2θ. The recursion runs on the background-subtracted signal, and
    the background is added back unchanged. It assumes an exact
    Kα2/Kα1 ratio and identical line shapes; at low angle, where the doublet
    separation is below the step size, the correction is unreliable.
    """
    x, y = _crop(m)
    step = float(np.median(np.diff(x)))
    grid = np.arange(x[0], x[-1] + 0.5 * step, step)
    yg = np.interp(grid, x, y)
    bkg = estimate_background(grid, yg, bkg_window_deg)
    net = yg - bkg
    s = np.sin(np.radians(0.5 * grid)) * (wl1 / wl2)
    src = 2.0 * np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))  # 2θ′ < 2θ
    pos = (src - grid[0]) / step
    out = net.copy()
    for i in range(grid.size):
        p = pos[i]
        if p < 0:
            continue
        j = int(np.floor(p))
        if j >= i:  # separation below one step: nothing earlier to subtract
            continue
        fr = p - j
        nxt = out[j + 1] if j + 1 < i else out[j]
        out[i] = net[i] - ratio * ((1.0 - fr) * out[j] + fr * nxt)
    info = dict(m.info or {})
    info.update({"kalpha2_stripped": True, "wl1": wl1, "wl2": wl2, "ratio": ratio})
    return Measured(name=f"{m.name} (Kα2 stripped)", two_theta=grid,
                    intensity=out + bkg, color=m.color,
                    source=f"{m.source} · Rachinger Kα2 strip", info=info)


# ===========================================================================
#  3. Williamson–Hall
# ===========================================================================

def caglioti_fwhm(tt, U: float, V: float, W: float):
    """Instrumental FWHM (°) from Caglioti: √(U tan²θ + V tanθ + W)."""
    t = np.tan(np.radians(0.5 * np.asarray(tt, float)))
    return np.sqrt(np.clip(U * t * t + V * t + W, 1e-12, None))


def williamson_hall(peaks, wavelength: float = 1.54056, K: float = 0.9,
                    instrument=None, correction: str = "auto",
                    select: Optional[Sequence[bool]] = None) -> Dict:
    """Williamson–Hall analysis β·cosθ = Kλ/D + 4ε·sinθ.

    Parameters
    ----------
    peaks : DataFrame from :func:`find_peaks_table` (Kα1 FWHM in degrees).
    instrument : None, float (constant FWHM in °), or (U, V, W) Caglioti tuple
        from a line-profile standard (LaB6 SRM 660, Si SRM 640) measured on the
        same instrument and slits.
    correction : "lorentzian" (β = β_obs − β_inst), "gaussian"
        (β² = β_obs² − β_inst²) or "auto" (Lorentzian if mean η > 0.5).
    select : boolean mask of the peaks to use. Use only single, resolved
        reflections: unresolved splitting from a lower-symmetry cell broadens
        the apparent FWHM and is then misread as strain.
    """
    df = peaks if select is None else peaks[np.asarray(select, bool)]
    tt = df["two_theta"].to_numpy(float)
    f = df["fwhm"].to_numpy(float)
    f_esd = df["fwhm_esd"].to_numpy(float) if "fwhm_esd" in df else np.full_like(f, np.nan)
    if instrument is None:
        fi = np.zeros_like(f)
    elif np.isscalar(instrument):
        fi = np.full_like(f, float(instrument))
    else:
        fi = caglioti_fwhm(tt, *instrument)
    mode = correction
    if mode == "auto":
        mode = "lorentzian" if float(np.nanmean(df["eta"])) > 0.5 else "gaussian"
    if mode == "lorentzian":
        b = f - fi
        b_esd = f_esd
    else:
        b = np.sqrt(np.clip(f * f - fi * fi, 0.0, None))
        b_esd = np.where(b > 0, f * f_esd / np.where(b > 0, b, 1.0), np.nan)
    ok = b > 0
    th = np.radians(0.5 * tt)
    xw = 4.0 * np.sin(th)
    yw = np.radians(b) * np.cos(th)
    yw_esd = np.radians(b_esd) * np.cos(th)
    res = {"x": xw, "y": yw, "y_esd": yw_esd, "two_theta": tt, "used": ok,
           "correction": mode, "K": K, "wavelength": wavelength,
           "D_nm": np.nan, "D_nm_esd": np.nan, "strain": np.nan,
           "strain_esd": np.nan, "r2": np.nan, "note": ""}
    scherrer = np.where(ok, K * wavelength / np.where(yw > 0, yw, np.nan) / 10.0, np.nan)
    res["scherrer_nm"] = scherrer
    if ok.sum() < 3:
        res["note"] = "fewer than 3 usable peaks; W–H fit not attempted"
        return res
    X, Y = xw[ok], yw[ok]
    sw = yw_esd[ok]
    w = 1.0 / sw ** 2 if np.all(np.isfinite(sw)) and np.all(sw > 0) else np.ones_like(Y)
    A = np.vstack([np.ones_like(X), X]).T
    Aw = A * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(Aw, Y * np.sqrt(w), rcond=None)
    r = Y - A @ coef
    dof = max(len(Y) - 2, 1)
    s2 = float(np.sum(w * r * r) / dof)
    cov = np.linalg.pinv(Aw.T @ Aw) * s2
    c0, c1 = coef
    sc0, sc1 = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2)) or 1.0
    res.update({"intercept": c0, "intercept_esd": sc0, "slope": c1,
                "slope_esd": sc1, "strain": c1, "strain_esd": sc1,
                "r2": 1.0 - float(np.sum(r * r)) / ss_tot})
    if c0 > 0:
        D = K * wavelength / c0 / 10.0
        res["D_nm"] = D
        res["D_nm_esd"] = D * sc0 / c0
        if sc0 > c0:
            res["note"] = "intercept not significant: size broadening unresolved"
    else:
        res["note"] = ("intercept ≤ 0: size broadening below resolution or the "
                       "instrumental correction is too large")
    return res


# ===========================================================================
#  4. Pawley whole-pattern cell refinement
# ===========================================================================

_SQ2, _SQ3 = np.sqrt(2.0), np.sqrt(3.0)

# Starting cells (in units of the pseudocubic a_p) and formula units Z for
# ABO3 perovskites. Axis choices follow the standard ITA settings used by
# pymatgen; orthorhombic models are additionally started with the long axis
# permuted, so an unconventional setting cannot trap the fit.
PEROVSKITE_MODELS = {
    "Pm-3m": {"sg": "Pm-3m", "cell": (1, 1, 1, 90, 90, 90), "Z": 1,
              "label": "cubic a⁰a⁰a⁰"},
    "R-3c": {"sg": "R-3c", "cell": (_SQ2, _SQ2, 2 * _SQ3, 90, 90, 120), "Z": 6,
             "label": "rhombohedral a⁻a⁻a⁻ (hexagonal axes)"},
    "Pnma": {"sg": "Pnma", "cell": (_SQ2, 2, _SQ2, 90, 90, 90), "Z": 4,
             "label": "orthorhombic a⁻b⁺a⁻"},
    "Imma": {"sg": "Imma", "cell": (_SQ2, 2, _SQ2, 90, 90, 90), "Z": 4,
             "label": "orthorhombic a⁰b⁻b⁻"},
    "I4/mcm": {"sg": "I4/mcm", "cell": (_SQ2, _SQ2, 2, 90, 90, 90), "Z": 4,
               "label": "tetragonal a⁰a⁰c⁻"},
}


def _lattice_free(system: str, rhombo: bool):
    """Names of the free cell parameters for a crystal system."""
    if system == "cubic":
        return ["a"]
    if system in ("tetragonal", "hexagonal"):
        return ["a", "c"]
    if system == "trigonal":
        return ["a", "alpha"] if rhombo else ["a", "c"]
    if system == "orthorhombic":
        return ["a", "b", "c"]
    if system == "monoclinic":
        return ["a", "b", "c", "beta"]
    return ["a", "b", "c", "alpha", "beta", "gamma"]


def _cell_from_free(system, rhombo, free, base):
    a, b, c, al, be, ga = base
    p = dict(zip(_lattice_free(system, rhombo), free))
    if system == "cubic":
        return (p["a"],) * 3 + (90.0, 90.0, 90.0)
    if system == "tetragonal":
        return (p["a"], p["a"], p["c"], 90.0, 90.0, 90.0)
    if system in ("hexagonal", "trigonal") and not rhombo:
        return (p["a"], p["a"], p["c"], 90.0, 90.0, 120.0)
    if system == "trigonal":
        return (p["a"],) * 3 + (p["alpha"],) * 3
    if system == "orthorhombic":
        return (p["a"], p["b"], p["c"], 90.0, 90.0, 90.0)
    if system == "monoclinic":
        return (p["a"], p["b"], p["c"], 90.0, p["beta"], 90.0)
    return (p["a"], p["b"], p["c"], p["alpha"], p["beta"], p["gamma"])


def _recip_metric(cell):
    a, b, c, al, be, ga = cell
    al, be, ga = np.radians([al, be, ga])
    G = np.array([
        [a * a, a * b * np.cos(ga), a * c * np.cos(be)],
        [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
        [a * c * np.cos(be), b * c * np.cos(al), c * c],
    ])
    return np.linalg.inv(G)


def cell_volume(cell) -> float:
    a, b, c, al, be, ga = cell
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    return float(a * b * c * np.sqrt(max(1 - ca * ca - cb * cb - cg * cg
                                         + 2 * ca * cb * cg, 0.0)))


def _space_group(symbol: str):
    from pymatgen.symmetry.groups import SpaceGroup
    s = symbol.replace(" ", "")
    return SpaceGroup(s)


def allowed_reflections(sg_symbol: str, cell, d_min: float):
    """Symmetry-distinct reflections allowed by the space group down to d_min.

    Systematic absences come from the structure factor of one general-position
    orbit (two random seeds): |F| ≈ 0 ⇔ absent for every structure in the
    group. Reflections are merged when they coincide for *every* cell of the
    crystal system (Laue-equivalent or metrically forced, e.g. cubic 300/221);
    this test is done on a deliberately generic cell, so accidental overlaps
    of a pseudocubic starting cell are kept separate.

    Returns (hkl representatives [n×3 int], multiplicity [n]).
    """
    sg = _space_group(sg_symbol)
    system = sg.crystal_system
    rhombo = sg_symbol.replace(" ", "").endswith(":R")
    names = _lattice_free(system, rhombo)
    free0 = [dict(zip(["a", "b", "c", "alpha", "beta", "gamma"], cell))[k]
             for k in names]
    len_f = iter((1.0, 1.0137, 0.9871))
    ang_d = iter((1.37, 2.11, 0.73))
    generic = [v * next(len_f) if k in ("a", "b", "c") else v + next(ang_d)
               for v, k in zip(free0, names)]
    cell_g = _cell_from_free(system, rhombo, generic, cell)
    Gs = _recip_metric(cell)
    a_s = np.sqrt(np.diag(Gs))
    # one box for all axes, sized by the longest one, so the list stays valid
    # when a refinement permutes or strongly distorts the axes
    hmax = int(np.floor(1.0 / (d_min * a_s.min()) + 1e-9)) + 1
    rng = [np.arange(-hmax, hmax + 1)] * 3
    H = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, axis=1)]
    inv_d2 = np.einsum("ij,jk,ik->i", H, Gs, H)
    H = H[inv_d2 <= 1.0 / d_min ** 2 * 1.0001]

    ops = list(sg.symmetry_ops)
    R = np.array([op.rotation_matrix for op in ops])
    T = np.array([op.translation_vector for op in ops])
    keep = np.zeros(len(H), bool)
    for seed in (np.array([0.1234, 0.3571, 0.6789]), np.array([0.4142, 0.0731, 0.2236])):
        X = np.einsum("nij,j->ni", R, seed) + T
        F = np.exp(2j * np.pi * (H @ X.T)).sum(axis=1)
        keep |= np.abs(F) > 1e-3 * len(ops)
    H = H[keep]
    Gg = _recip_metric(cell_g)
    key = np.round(np.einsum("ij,jk,ik->i", H, Gg, H), 9)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    reps = np.zeros((len(uniq), 3), int)
    for gi in range(len(uniq)):
        members = H[inv == gi]
        # prefer the all-non-negative, lexicographically largest label
        score = (members >= 0).sum(axis=1) * 1000 + members @ np.array([100, 10, 1])
        reps[gi] = members[np.argmax(score)]
    return reps, counts


@dataclass
class PawleyResult:
    model: str
    spacegroup: str
    crystal_system: str
    cell: Tuple[float, ...]
    cell_esd: Tuple[float, ...]
    volume: float
    volume_esd: float
    Z: int
    zero: float
    zero_esd: float
    displacement: float
    displacement_esd: float
    U: float
    V: float
    W: float
    eta: float
    Rwp: float
    Rp: float
    Rexp: float
    gof: float
    chi2: float
    n_obs: int
    n_par: int
    bic: float
    x: np.ndarray = field(repr=False)
    y_obs: np.ndarray = field(repr=False)
    y_calc: np.ndarray = field(repr=False)
    y_bkg: np.ndarray = field(repr=False)
    ticks: np.ndarray = field(repr=False)
    hkl: np.ndarray = field(repr=False)
    intensities: np.ndarray = field(repr=False)
    correlations: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    extra: List[Dict] = field(default_factory=list)

    @property
    def volume_per_fu(self) -> float:
        return self.volume / self.Z

    @property
    def volume_per_fu_esd(self) -> float:
        return self.volume_esd / self.Z

    @property
    def a_pc(self) -> float:
        """Pseudocubic lattice parameter (V/Z)^(1/3) for ABO3 models."""
        return self.volume_per_fu ** (1.0 / 3.0)

    def as_row(self) -> Dict:
        a, b, c, al, be, ga = self.cell
        sa, sb, sc, sal, sbe, sga = self.cell_esd
        return {
            "model": self.model, "space group": self.spacegroup,
            "a (Å)": a, "σa": sa, "b (Å)": b, "σb": sb, "c (Å)": c, "σc": sc,
            "α (°)": al, "β (°)": be, "γ (°)": ga,
            "V (Å³)": self.volume, "σV": self.volume_esd, "Z": self.Z,
            "V/Z (Å³)": self.volume_per_fu, "σ(V/Z)": self.volume_per_fu_esd,
            "a_pc (Å)": self.a_pc,
            "zero (°)": self.zero, "σzero": self.zero_esd,
            "displ. (°)": self.displacement, "σdispl.": self.displacement_esd,
            "Rwp (%)": 100 * self.Rwp, "Rp (%)": 100 * self.Rp,
            "Rexp (%)": 100 * self.Rexp, "GoF": self.gof,
            "N_par": self.n_par, "BIC": self.bic,
        }


class _PawleyModel:
    """Variable-projection Pawley model.

    Nonlinear parameters (cell, zero, displacement, U, V, W, η) are refined by
    ``scipy.optimize.least_squares``; at every evaluation the reflection
    intensities (≥ 0) and Chebyshev background coefficients are the exact
    weighted linear least-squares solution (NNLS with ± background columns).
    Peak position: 2θ_obs = 2θ_Bragg + zero + D·cosθ, where D is the
    sample-displacement shift (D = −(2s/R)·180/π for displacement s and
    goniometer radius R).
    """

    def __init__(self, x, y, sg_symbol, cell0, wl1, wl2, ratio, n_bkg,
                 refine_zero, refine_disp, weighting="auto", extra=()):
        self.x, self.y = x, y
        self.w, self.weighting = counting_weights(x, y, weighting)
        self.sw = np.sqrt(self.w)
        self.sg_symbol = sg_symbol
        self.sg = _space_group(sg_symbol)
        self.system = self.sg.crystal_system
        self.rhombo = sg_symbol.replace(" ", "").endswith(":R")
        self.base = tuple(float(v) for v in cell0)
        self.free_names = _lattice_free(self.system, self.rhombo)
        self.wl1, self.wl2, self.ratio = wl1, wl2, ratio
        self.refine_zero, self.refine_disp = refine_zero, refine_disp
        lam_max = wl2 if wl2 else wl1
        tt_max = min(x.max() + 2.0, 179.0)
        self.d_min = lam_max / (2.0 * np.sin(np.radians(0.5 * tt_max)))
        self.hkl, self.mult = allowed_reflections(sg_symbol, self.base, self.d_min * 0.97)
        u = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-9) - 1.0
        T = np.polynomial.chebyshev.chebvander(u, max(int(n_bkg) - 1, 0))
        self.B = T
        self.Qb, self.Rb = np.linalg.qr(T * self.sw[:, None])
        self.names = list(self.free_names)
        if refine_zero:
            self.names.append("zero")
        if refine_disp:
            self.names.append("disp")
        self.names += ["U", "V", "W", "eta"]
        # additional phases refined jointly: own cell + FWHM scale, shared profile
        self.extra = []
        for k, ph in enumerate(extra):
            sg = _space_group(ph["spacegroup"])
            rh = ph["spacegroup"].replace(" ", "").endswith(":R")
            base = tuple(float(v) for v in ph["cell"])
            fn = _lattice_free(sg.crystal_system, rh)
            hkl, _ = allowed_reflections(ph["spacegroup"], base, self.d_min * 0.97)
            tag = ph.get("name") or f"phase{k + 2}"
            self.extra.append({"name": tag, "system": sg.crystal_system, "rhombo": rh,
                               "base": base, "free": fn, "hkl": hkl,
                               "refine_cell": ph.get("refine_cell", True),
                               "keys": [f"{tag}:{f}" for f in fn] + [f"{tag}:wscale"]})
            self.names += self.extra[-1]["keys"]

    def unpack(self, p):
        d = dict(zip(self.names, p))
        free = [d[k] for k in self.free_names]
        cell = _cell_from_free(self.system, self.rhombo, free, self.base)
        return (cell, d.get("zero", 0.0), d.get("disp", 0.0),
                d["U"], d["V"], d["W"], d["eta"])

    def extra_cells(self, p):
        d = dict(zip(self.names, p))
        out = []
        for ph in self.extra:
            free = [d[k] for k in ph["keys"][:-1]]
            out.append((_cell_from_free(ph["system"], ph["rhombo"], free, ph["base"]),
                        d[ph["keys"][-1]]))
        return out

    def positions(self, cell, zero, disp, wl, hkl=None):
        hkl = self.hkl if hkl is None else hkl
        Gs = _recip_metric(cell)
        inv_d2 = np.einsum("ij,jk,ik->i", hkl, Gs, hkl)
        s = 0.5 * wl * np.sqrt(np.clip(inv_d2, 0.0, None))
        ok = s < 1.0
        th = np.arcsin(np.clip(s, 0.0, 0.999999))
        tt = 2.0 * np.degrees(th) + zero + disp * np.cos(th)
        return tt, ok

    def design(self, p):
        cell, zero, disp, U, V, W, eta = self.unpack(p)
        eta = float(np.clip(eta, 0.0, 1.0))
        tt1, ok = self.positions(cell, zero, disp, self.wl1)
        f1 = caglioti_fwhm(tt1, U, V, W)
        sel = ok & (tt1 > self.x.min() - 3 * f1) & (tt1 < self.x.max() + 3 * f1)
        tt1, f1 = tt1[sel], f1[sel]
        dx = self.x[:, None] - tt1[None, :]
        A = pseudo_voigt(dx, 0.0, f1[None, :], eta) / _pv_area_factor(f1, eta)[None, :]
        if self.wl2:
            tt2, _ = self.positions(cell, zero, disp, self.wl2)
            tt2 = tt2[sel]
            f2 = caglioti_fwhm(tt2, U, V, W)
            A = A + self.ratio * (pseudo_voigt(self.x[:, None] - tt2[None, :], 0.0,
                                               f2[None, :], eta)
                                  / _pv_area_factor(f2, eta)[None, :])
        self.extra_ticks = []
        if self.extra:
            blocks = [A]
            for ph, (c, ws) in zip(self.extra, self.extra_cells(p)):
                cols, ticks = [], []
                for wl, fac in ((self.wl1, 1.0), (self.wl2, self.ratio)):
                    if not wl:
                        continue
                    t, okx = self.positions(c, zero, disp, wl, ph["hkl"])
                    if wl == self.wl1:
                        f = ws * caglioti_fwhm(t, U, V, W)
                        sx = okx & (t > self.x.min() - 3 * f) & (t < self.x.max() + 3 * f)
                        ticks = t[sx]
                    f = ws * caglioti_fwhm(t[sx], U, V, W)
                    cols.append(fac * pseudo_voigt(self.x[:, None] - t[sx][None, :], 0.0,
                                                   f[None, :], eta)
                                / _pv_area_factor(f, eta)[None, :])
                blocks.append(sum(cols))
                self.extra_ticks.append(ticks)
            A = np.hstack(blocks)
        return A, sel, tt1

    def solve_linear(self, p):
        """Exact weighted LSQ for intensities (≥ 0) and background (free).

        The background is projected out (QR of the weighted Chebyshev basis),
        and the remaining non-negative problem is reduced to its small
        triangular form before NNLS, so the cost is two thin QRs per call.
        """
        from scipy.optimize import nnls
        A, sel, tt1 = self.design(p)
        b = self.y * self.sw
        M = A * self.sw[:, None]
        Mp = M - self.Qb @ (self.Qb.T @ M)
        bp = b - self.Qb @ (self.Qb.T @ b)
        if M.shape[1]:
            Q, R = np.linalg.qr(Mp)
            I, _ = nnls(R, Q.T @ bp, maxiter=50 * max(M.shape[1], 1))
        else:
            I = np.zeros(0)
        cb = np.linalg.solve(self.Rb, self.Qb.T @ (b - M @ I))
        bkg = self.B @ cb
        return A @ I + bkg, bkg, I, sel, tt1

    def residual(self, p):
        yc, *_ = self.solve_linear(p)
        return self.sw * (self.y - yc)


def phase_spec(ph) -> Dict:
    """Secondary-phase spec for :func:`pawley_refine` from a Structure or dict.

    A pymatgen Structure is reduced to its conventional standard cell and
    international space-group symbol (symprec 0.01 Å).
    """
    if isinstance(ph, dict):
        out = dict(ph)
        out["cell"] = tuple(float(v) for v in out["cell"])
        return out
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(ph, symprec=0.01)
    conv = sga.get_conventional_standard_structure()
    sym = sga.get_space_group_symbol()
    if sga.get_crystal_system() == "trigonal" and sym.startswith("R"):
        sym += ":H"
    lat = conv.lattice
    return {"name": f"{ph.composition.reduced_formula} {sym}", "spacegroup": sym,
            "cell": tuple(lat.abc) + tuple(lat.angles), "refine_cell": True}


def pawley_refine(m: Measured, model: str = "R-3c", a_pc: Optional[float] = None,
                  cell0: Optional[Sequence[float]] = None, Z: Optional[int] = None,
                  wavelength=(1.54056, 1.54439, 0.5), tt_range=None,
                  n_bkg: int = 8, refine_zero: bool = False,
                  refine_disp: bool = True, fwhm0: float = 0.12,
                  multistart: bool = True, weighting: str = "auto",
                  extra_phases: Sequence = ()) -> PawleyResult:
    """Pawley fit of one measured pattern with one space-group model.

    ``model`` is a key of :data:`PEROVSKITE_MODELS` (then ``a_pc`` sets the
    starting cell; if omitted it is estimated from the strongest peak taken as
    the pseudocubic 110) or any space-group symbol together with ``cell0`` =
    (a, b, c, α, β, γ) and ``Z``.

    Zero offset and sample displacement are both nearly constant over a
    20–80° scan (cosθ changes by ~20 %), so refining both at once is
    ill-conditioned; the default refines displacement only. Lower symmetry
    always fits at least as well; compare models with :func:`compare_models`
    (BIC) and by which pseudocubic reflections are split.

    ``weighting`` is passed to :func:`counting_weights`. BIC is computed in
    its unknown-variance form N·ln(χ²/N) + P·ln N, which does not depend on
    the absolute intensity scale or on a wrong noise model.

    ``extra_phases`` are secondary phases refined jointly (Pawley intensities,
    own lattice constants, own FWHM scale on the shared Caglioti profile):
    pymatgen Structures or dicts {"name", "spacegroup", "cell", "refine_cell"}.
    Without them, lines of an impurity are absorbed as spurious intensity of
    the main phase and bias its cell.
    """
    from scipy.optimize import least_squares

    wl1, wl2, ratio = _wl_tuple(wavelength)
    x, y = _crop(m, tt_range)
    if model in PEROVSKITE_MODELS:
        spec = PEROVSKITE_MODELS[model]
        sg_symbol = spec["sg"]
        if a_pc is None:
            a_pc = estimate_a_pc(m, wl1, tt_range)
        a, b, c, al, be, ga = spec["cell"]
        cell0 = (a * a_pc, b * a_pc, c * a_pc, al, be, ga)
        Z = spec["Z"] if Z is None else Z
    else:
        sg_symbol = model
        if cell0 is None:
            raise ValueError("cell0 is required for a custom space group")
        Z = Z or 1
    cell0 = tuple(float(v) for v in cell0)
    Z = int(Z)

    extra = [phase_spec(e) for e in extra_phases]
    pm = _PawleyModel(x, y, sg_symbol, cell0, wl1, wl2, ratio, n_bkg,
                      refine_zero, refine_disp, weighting, extra)
    base_free = [dict(zip(["a", "b", "c", "alpha", "beta", "gamma"], cell0))[k]
                 for k in pm.free_names]

    def full_p(free):
        p = list(free)
        if refine_zero:
            p.append(0.0)
        if refine_disp:
            p.append(0.0)
        p += [0.0, 0.0, fwhm0 ** 2, 0.5]
        for ph in pm.extra:
            p += [dict(zip(["a", "b", "c", "alpha", "beta", "gamma"], ph["base"]))[k]
                  for k in ph["free"]] + [1.0]
        return np.array(p, float)

    lo, hi = [], []
    lens = [v for k, v in zip(pm.free_names, base_free) if k in ("a", "b", "c")]
    for k, v in zip(pm.free_names, base_free):
        if k in ("a", "b", "c") and pm.system == "orthorhombic":
            lo.append(0.9 * min(lens))
            hi.append(1.1 * max(lens))
        elif k in ("a", "b", "c"):
            lo.append(0.9 * v)
            hi.append(1.1 * v)
        else:
            lo.append(v - 5.0)
            hi.append(v + 5.0)
    if refine_zero:
        lo.append(-1.0)
        hi.append(1.0)
    if refine_disp:
        lo.append(-1.0)
        hi.append(1.0)
    lo += [0.0, -0.5, 1e-6, 0.0]
    hi += [1.0, 0.5, 1.0, 1.0]
    for ph in pm.extra:
        cd = dict(zip(["a", "b", "c", "alpha", "beta", "gamma"], ph["base"]))
        for k in ph["free"]:
            v = cd[k]
            dv = (0.03 * v if k in ("a", "b", "c") else 2.0) if ph["refine_cell"] else 1e-9 * max(v, 1)
            lo.append(v - dv)
            hi.append(v + dv)
        lo.append(0.2)
        hi.append(8.0)
    lo, hi = np.array(lo), np.array(hi)

    # --- starting points: splitting directions of the pseudocubic cell ------
    starts = [list(base_free)]
    if multistart:
        lengths = [i for i, k in enumerate(pm.free_names) if k in ("a", "b", "c")]
        if len(lengths) == 2:
            for f in (0.99, 0.995, 1.005, 1.01):
                st = list(base_free)
                st[lengths[1]] *= f
                starts.append(st)
        elif len(lengths) == 3:
            for f1 in (0.995, 1.005):
                for f2 in (0.995, 1.005):
                    st = list(base_free)
                    st[lengths[1]] *= f1
                    st[lengths[2]] *= f2
                    starts.append(st)
            # long axis on a / c instead of b (setting insurance)
            a0, b0, c0 = (base_free[i] for i in lengths)
            for perm in ((b0, a0, c0), (a0, c0, b0)):
                if not np.allclose(perm, (a0, b0, c0)):
                    st = list(base_free)
                    for i, v in zip(lengths, perm):
                        st[i] = v
                    starts.append(st)

    n_free = len(pm.free_names) + int(refine_zero) + int(refine_disp)

    def coarse(p_full):
        mask = np.zeros(len(p_full), bool)
        mask[:n_free] = True

        def r(sub):
            q = p_full.copy()
            q[mask] = sub
            return pm.residual(q)
        sol = least_squares(r, p_full[mask], bounds=(lo[mask], hi[mask]),
                            x_scale="jac", max_nfev=25, diff_step=1e-6)
        q = p_full.copy()
        q[mask] = sol.x
        return q, float(np.sum(sol.fun ** 2))

    best = None
    for s in starts:
        q, chi = coarse(np.clip(full_p(s), lo, hi))
        if best is None or chi < best[1]:
            best = (q, chi)
    p_start = best[0]

    sol = least_squares(pm.residual, p_start, bounds=(lo, hi), x_scale="jac",
                        max_nfev=400, diff_step=1e-6)
    p = sol.x
    yc, bkg, I, sel, tt1 = pm.solve_linear(p)
    r = y - yc
    w = pm.w
    chi2 = float(np.sum(w * r * r))
    n_obs = int(y.size)
    n_lin_active = int(np.sum(I > 0)) + pm.B.shape[1]
    n_par = len(p) + n_lin_active
    dof = max(n_obs - n_par, 1)
    Rwp = float(np.sqrt(chi2 / np.sum(w * y * y)))
    Rp = float(np.sum(np.abs(r)) / np.sum(np.abs(y)))
    Rexp = float(np.sqrt(dof / np.sum(w * y * y)))
    J = sol.jac
    try:
        cov = np.linalg.pinv(J.T @ J) * (chi2 / dof)
    except np.linalg.LinAlgError:
        cov = np.full((len(p), len(p)), np.nan)
    esd = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    cell, zero, disp, U, V, W, eta = pm.unpack(p)

    def cell_of(pp):
        return pm.unpack(pp)[0]

    # propagate covariance to the six cell constants and V
    nfree = len(pm.free_names)
    Jc = np.zeros((6, nfree))
    Jv = np.zeros(nfree)
    for i in range(nfree):
        h = 1e-6 * max(abs(p[i]), 1.0)
        pp, pn = p.copy(), p.copy()
        pp[i] += h
        pn[i] -= h
        cp, cn = np.array(cell_of(pp)), np.array(cell_of(pn))
        Jc[:, i] = (cp - cn) / (2 * h)
        Jv[i] = (cell_volume(cp) - cell_volume(cn)) / (2 * h)
    C = cov[:nfree, :nfree]
    cell_esd = tuple(float(v) for v in np.sqrt(np.clip(np.diag(Jc @ C @ Jc.T), 0, None)))
    V_esd = float(np.sqrt(max(Jv @ C @ Jv, 0.0)))

    def _esd(name):
        return float(esd[pm.names.index(name)]) if name in pm.names else 0.0

    extra_out = []
    n_main = int(np.sum(sel))
    off = n_main
    for ph, (c, ws), tk in zip(pm.extra, pm.extra_cells(p), pm.extra_ticks):
        nref = len(tk)
        Iph = I[off:off + nref]
        off += nref
        extra_out.append({
            "name": ph["name"], "cell": tuple(float(v) for v in c),
            "esd": {k.split(":")[1]: _esd(k) for k in ph["keys"][:-1]},
            "volume": cell_volume(c), "fwhm_scale": float(ws), "ticks": tk,
            "intensity_share": float(Iph.sum() / max(I.sum(), 1e-12)),
        })

    corr = {}
    sd = np.where(esd > 0, esd, np.nan)
    Cn = cov / np.outer(sd, sd)
    for i, j in combinations(range(len(p)), 2):
        if np.isfinite(Cn[i, j]) and abs(Cn[i, j]) > 0.8:
            corr[f"{pm.names[i]}–{pm.names[j]}"] = float(Cn[i, j])

    return PawleyResult(
        model=model, spacegroup=pm.sg.symbol, crystal_system=pm.system,
        cell=tuple(float(v) for v in cell), cell_esd=cell_esd,
        volume=cell_volume(cell), volume_esd=V_esd, Z=Z,
        zero=float(zero), zero_esd=_esd("zero"),
        displacement=float(disp), displacement_esd=_esd("disp"),
        U=float(U), V=float(V), W=float(W), eta=float(eta),
        Rwp=Rwp, Rp=Rp, Rexp=Rexp, gof=Rwp / Rexp if Rexp else np.nan,
        chi2=chi2, n_obs=n_obs, n_par=n_par,
        bic=n_obs * np.log(chi2 / n_obs) + n_par * np.log(n_obs),
        x=x, y_obs=y, y_calc=yc, y_bkg=bkg, ticks=tt1,
        hkl=pm.hkl[sel], intensities=I[:int(np.sum(sel))], correlations=corr,
        extra=extra_out,
        message=f"{sol.message} [{pm.weighting} weights]",
    )


def estimate_a_pc(m: Measured, wl1: float = 1.54056, tt_range=None) -> float:
    """Pseudocubic a of an ABO3 perovskite from its 200 reflection.

    The strongest peak with d between 1.85 and 2.06 Å (a_pc 3.70–4.12 Å) is
    taken as the pseudocubic 200, a = 2d. The 200 is preferred over the
    usually stronger 110 because common secondary phases (Ruddlesden–Popper
    A2BO4 103, rock-salt 111) fall near the 110 but not near the 200. Falls
    back to the strongest peak at d 2.6–2.95 Å as the 110. Check the result:
    pass ``a_pc`` explicitly when the perovskite is a minority phase.
    """
    x, y = _crop(m, tt_range)
    net = y - estimate_background(x, y)
    d = wl1 / (2.0 * np.sin(np.radians(0.5 * x)))
    for (dlo, dhi), fac in (((1.85, 2.06), 2.0), ((2.60, 2.95), _SQ2)):
        k = (d > dlo) & (d < dhi)
        if np.count_nonzero(k) < 5:
            continue
        xs, ys = x[k], net[k]
        i = int(np.argmax(ys))
        # centroid of the top of the strongest peak (robust to its own splitting)
        top = ys >= 0.5 * ys[i]
        j0, j1 = i, i
        while j0 > 0 and top[j0 - 1]:
            j0 -= 1
        while j1 < len(ys) - 1 and top[j1 + 1]:
            j1 += 1
        tt = float(np.sum(xs[j0:j1 + 1] * ys[j0:j1 + 1]) / np.sum(ys[j0:j1 + 1]))
        return float(fac * wl1 / (2.0 * np.sin(np.radians(0.5 * tt))))
    raise ValueError("no perovskite 200 or 110 in range; pass a_pc explicitly")


def compare_models(m: Measured, models: Sequence[str] = ("Pm-3m", "R-3c", "Pnma"),
                   **kwargs):
    """Refine every model and return (results list, DataFrame sorted by BIC).

    ΔBIC > 10 relative to the best model is conventionally strong evidence
    against it (Kass & Raftery, JASA 90, 773, 1995). Pawley fits cannot see
    weak superstructure (octahedral-tilt) reflections at lab-XRD
    signal-to-noise, so space groups that share the same metric distortion —
    e.g. R-3c vs R-3, Pnma vs Pbnm settings — are not distinguishable here.
    """
    import pandas as pd
    a_pc = kwargs.pop("a_pc", None)
    if a_pc is None:
        wl1 = _wl_tuple(kwargs.get("wavelength", 1.54056))[0]
        a_pc = estimate_a_pc(m, wl1, kwargs.get("tt_range"))
    results = []
    for md in models:
        try:
            results.append(pawley_refine(m, model=md, a_pc=a_pc, **kwargs))
        except Exception as e:  # keep the other models
            results.append(e)
    rows = []
    for md, r in zip(models, results):
        if isinstance(r, Exception):
            rows.append({"model": md, "error": f"{type(r).__name__}: {r}"})
        else:
            rows.append(r.as_row())
    df = pd.DataFrame(rows)
    if "BIC" in df:
        df["ΔBIC"] = df["BIC"] - df["BIC"].min()
        df = df.sort_values("BIC").reset_index(drop=True)
    return results, df


def make_pawley_plot(res: PawleyResult, figsize=(6.0, 4.0), zoom=None):
    """Observed / calculated / difference plot with reflection ticks."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    fig, (ax, axd) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1.2], "hspace": 0.05})
    x = res.x
    if zoom:
        k = (x >= zoom[0]) & (x <= zoom[1])
    else:
        k = slice(None)
    ax.plot(x[k], res.y_obs[k], "o", ms=1.6, mfc="none", mec="#000000", mew=0.5,
            label="observed")
    ax.plot(x[k], res.y_calc[k], color="#D55E00", lw=0.9, label="Pawley fit")
    ax.plot(x[k], res.y_bkg[k], color="#7F7F7F", lw=0.6, ls="--", label="background")
    ymin = float(np.min(res.y_obs[k]))
    ymax = float(np.max(res.y_obs[k]))
    tick_y = ymin - 0.06 * (ymax - ymin)
    tt = res.ticks
    if zoom:
        tt = tt[(tt >= zoom[0]) & (tt <= zoom[1])]
    ax.plot(tt, np.full_like(tt, tick_y), "|", color="#0072B2", ms=6, mew=0.7,
            label=f"{res.spacegroup} reflections")
    for i, (e, col) in enumerate(zip(res.extra, ["#009E73", "#CC79A7", "#E69F00", "#56B4E9"])):
        tk = np.asarray(e["ticks"], float)
        if zoom:
            tk = tk[(tk >= zoom[0]) & (tk <= zoom[1])]
        tick_y2 = tick_y - (i + 1) * 0.05 * (ymax - ymin)
        ax.plot(tk, np.full_like(tk, tick_y2), "|", color=col, ms=6, mew=0.7,
                label=e["name"])
    ylo = tick_y - (len(res.extra) + 1) * 0.05 * (ymax - ymin)
    ax.set_ylim(ylo, ymax * 1.05)
    ax.set_ylabel("Intensity (a.u.)")
    a, b, c, al, be, ga = res.cell
    ax.set_title(f"{res.model}: a={a:.4f}, b={b:.4f}, c={c:.4f} Å, "
                 f"V={res.volume:.2f} Å³ — Rwp {100 * res.Rwp:.2f} %", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    d = res.y_obs[k] - res.y_calc[k]
    axd.plot(x[k], d, color="#0072B2", lw=0.6)
    axd.axhline(0, color="#7F7F7F", lw=0.5)
    axd.set_ylabel("Obs − Calc")
    axd.set_xlabel(r"2$\theta$ (degrees)")
    if not zoom:
        axd.xaxis.set_major_locator(MultipleLocator(10))
        axd.xaxis.set_minor_locator(MultipleLocator(2))
    fig.align_ylabels([ax, axd])
    return fig


def make_wh_plot(wh: Dict, figsize=(3.6, 2.8)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    u = wh["used"]
    ax.errorbar(wh["x"][u], wh["y"][u] * 1e3, yerr=np.nan_to_num(wh["y_esd"][u]) * 1e3,
                fmt="o", ms=3, color="#0072B2", capsize=2, lw=0.8)
    if np.isfinite(wh.get("strain", np.nan)) and "intercept" in wh:
        xx = np.linspace(0, max(wh["x"].max(), 1e-3) * 1.05, 50)
        ax.plot(xx, (wh["intercept"] + wh["slope"] * xx) * 1e3, color="#D55E00", lw=1)
    ax.set_xlabel(r"4 sin$\theta$")
    ax.set_ylabel(r"$\beta$ cos$\theta$ (10$^{-3}$ rad)")
    D, e = wh.get("D_nm", np.nan), wh.get("strain", np.nan)
    ax.set_title(f"D = {D:.1f} nm, ε = {e * 100:.3f} %" if np.isfinite(D)
                 else f"ε = {e * 100:.3f} % (D undetermined)", fontsize=8)
    fig.tight_layout()
    return fig


# ===========================================================================
#  5. Unassigned peaks → COD candidates
# ===========================================================================

def unassigned_peaks(peaks, reflection_sets: Sequence[np.ndarray], tol: float = 0.15):
    """Peaks with no reflection of any given set within ``tol`` degrees."""
    tt = peaks["two_theta"].to_numpy(float)
    hit = np.zeros(tt.size, bool)
    for ref in reflection_sets:
        ref = np.asarray(ref, float)
        if ref.size:
            hit |= np.min(np.abs(tt[:, None] - ref[None, :]), axis=1) <= tol
    return peaks[~hit].reset_index(drop=True)


_OPTIMADE_COD = "https://www.crystallography.net/cod/optimade/v1/structures"


def cod_optimade_query(elements: Sequence[str], nelements: Optional[int] = None,
                       max_pages: int = 5, timeout: float = 60.0) -> List[Dict]:
    """Query COD over OPTIMADE for entries containing all ``elements``.

    Returns dicts with cod_id, formula (reduced), raw formula, spacegroup,
    title, R and the cell (a, b, c, α, β, γ) when COD lists it."""
    import requests
    from pymatgen.core import Composition
    els = ",".join(f'"{e}"' for e in elements)
    flt = f"elements HAS ALL {els}"
    if nelements:
        flt += f" AND nelements={int(nelements)}"
    params = {"filter": flt, "page_limit": 100,
              "response_fields": ("_cod_formula,_cod_sg,_cod_chemname,_cod_mineral,"
                                  "_cod_commonname,_cod_flags,_cod_Rall,_cod_a,_cod_b,"
                                  "_cod_c,_cod_alpha,_cod_beta,_cod_gamma")}
    url, out = _OPTIMADE_COD, []
    for _ in range(max_pages):
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        for d in j.get("data", []):
            a = d.get("attributes", {})
            raw = (a.get("_cod_formula") or "").strip("- ").strip()
            try:
                red = Composition(raw.replace(" ", "")).reduced_formula
            except Exception:
                red = raw
            cell = tuple(a.get(k) for k in ("_cod_a", "_cod_b", "_cod_c",
                                            "_cod_alpha", "_cod_beta", "_cod_gamma"))
            out.append({
                "cod_id": str(d.get("id")), "formula": red, "formula_raw": raw,
                "spacegroup": (a.get("_cod_sg") or "").strip(),
                "title": (a.get("_cod_commonname") or a.get("_cod_mineral")
                          or a.get("_cod_chemname") or "")[:80],
                "has_coordinates": "has coordinates" in (a.get("_cod_flags") or ""),
                "R": a.get("_cod_Rall"),
                "cell": cell if all(v is not None for v in cell) else None,
            })
        nxt = (j.get("links") or {}).get("next")
        if not nxt:
            break
        url = nxt["href"] if isinstance(nxt, dict) else nxt
        params = None
    return out


def _candidate_positions(entry, wl1, tt_range):
    """Allowed 2θ (Kα1) of a COD entry from its listed cell and space group."""
    if not entry.get("cell"):
        return None
    try:
        cell = tuple(float(v) for v in entry["cell"])
        sym = entry["spacegroup"].replace(" ", "")
        d_min = wl1 / (2.0 * np.sin(np.radians(0.5 * min(tt_range[1] + 1, 179))))
        hkl, _ = allowed_reflections(sym, cell, d_min)
        Gs = _recip_metric(cell)
        s = 0.5 * wl1 * np.sqrt(np.einsum("ij,jk,ik->i", hkl, Gs, hkl))
        tt = 2.0 * np.degrees(np.arcsin(np.clip(s[s < 1], 0, 1)))
        return tt[(tt >= tt_range[0]) & (tt <= tt_range[1])]
    except Exception:
        return None


_OPTIMADE_MP = "https://optimade.materialsproject.org/v1/structures"


def mp_optimade_query(elements: Sequence[str], nelements: Optional[int] = None,
                      max_ehull: float = 0.1, max_pages: int = 5,
                      timeout: float = 60.0) -> List[Dict]:
    """Materials Project structures over OPTIMADE (no API key needed).

    Returns dicts with mp_id, formula, spacegroup, e_above_hull (eV/atom) and
    a pymatgen ``structure``. These are DFT (GGA/GGA+U) relaxed cells, which
    are typically 1–2 % larger than experiment, so match with a lattice-scale
    scan (``max_strain`` in :func:`match_cod`)."""
    import requests
    from pymatgen.core import Lattice, Structure
    els = ",".join(f'"{e}"' for e in elements)
    flt = f"elements HAS ALL {els}"
    if nelements:
        flt += f" AND nelements={int(nelements)}"
    params = {"filter": flt, "page_limit": 100,
              "response_fields": ("chemical_formula_reduced,lattice_vectors,"
                                  "cartesian_site_positions,species_at_sites,"
                                  "_mp_stability")}
    url, out = _OPTIMADE_MP, []
    for _ in range(max_pages):
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        for d in j.get("data", []):
            a = d.get("attributes", {})
            stab = a.get("_mp_stability") or {}
            eh = None
            for k in ("gga_gga+u_r2scan", "gga_gga+u", "r2scan"):
                if k in stab and stab[k].get("energy_above_hull") is not None:
                    eh = float(stab[k]["energy_above_hull"])
                    break
            if eh is not None and eh > max_ehull:
                continue
            try:
                st = Structure(Lattice(a["lattice_vectors"]), a["species_at_sites"],
                               a["cartesian_site_positions"], coords_are_cartesian=True)
                sg = st.get_space_group_info(symprec=0.01)[0]
            except Exception:
                continue
            out.append({"mp_id": str(d.get("id")),
                        "formula": a.get("chemical_formula_reduced", ""),
                        "spacegroup": sg, "e_above_hull": eh, "structure": st})
        nxt = (j.get("links") or {}).get("next")
        if not nxt:
            break
        url = nxt["href"] if isinstance(nxt, dict) else nxt
        params = None
    return out


def _score(cand_tt, cand_I, target_tt, target_h, observed_tt, tol,
           strong: float = 20.0, weak: float = 5.0):
    """explained fraction (target-height weighted) × consistency of strong lines."""
    if cand_tt.size == 0:
        return 0.0, 0.0, 0.0, []
    near = np.abs(target_tt[:, None] - cand_tt[None, :]) <= tol
    if cand_I is not None:
        near &= (cand_I[None, :] >= weak)
    matched = near.any(axis=1)
    explained = float(target_h[matched].sum() / max(target_h.sum(), 1e-12))
    if cand_I is None:
        consistency = 1.0
    else:
        s = cand_I >= strong
        if not s.any():
            consistency = 0.0
        elif observed_tt.size == 0:
            consistency = 0.0
        else:
            seen = np.min(np.abs(cand_tt[s][:, None] - observed_tt[None, :]),
                          axis=1) <= tol
            consistency = float(cand_I[s][seen].sum() / cand_I[s].sum())
    return explained * consistency, explained, consistency, list(target_tt[matched])


def _rescale_tt(tt, wl, scale):
    """2θ after an isotropic lattice scaling d → scale·d."""
    s = np.sin(np.radians(0.5 * tt)) / scale
    ok = s < 1.0
    return 2.0 * np.degrees(np.arcsin(np.clip(s, 0, 1))), ok


def _best_scale_score(tt, I, wl, target_tt, target_h, observed_tt, tol, max_strain):
    """Score maximised over an isotropic lattice scale within ±max_strain."""
    scales = [1.0] if max_strain <= 0 else np.linspace(1 - max_strain, 1 + max_strain,
                                                      int(2 * max_strain / 0.001) + 1)
    best = (0.0, 0.0, 0.0, [], 1.0)
    for sc in scales:
        t2, ok = _rescale_tt(tt, wl, sc)
        res = _score(t2[ok], None if I is None else I[ok], target_tt, target_h,
                     observed_tt, tol)
        # prefer the smallest |strain| among equal scores
        if res[0] > best[0] + 1e-9 or (abs(res[0] - best[0]) <= 1e-9
                                       and abs(sc - 1) < abs(best[4] - 1)):
            best = (*res, float(sc))
    return best


def match_cod(target_peaks, all_peaks, elements: Sequence[str],
              wavelength: float = 1.54056, tt_range=(20.0, 80.0),
              tol: float = 0.15, max_subset: int = 3, prefilter_keep: int = 30,
              max_strain: float = 0.01, source: str = "auto", progress=None):
    """Rank database phases built from subsets of ``elements`` against peaks.

    1. OPTIMADE query for every element subset of size ≤ ``max_subset`` with
       exactly that many elements. ``source`` = "cod", "mp", or "auto" (COD,
       falling back to the Materials Project if COD cannot be reached).
       COD: entries with coordinates only, one per formula + space group
       (lowest R kept). MP: entries within 0.1 eV/atom of the hull.
    2. COD prefilter: allowed reflection positions from the listed cell and
       space group must explain the target (unassigned) peaks.
    3. Kinematic intensities (pymatgen; COD CIFs are downloaded for the
       ``prefilter_keep`` best) and score = explained × consistency, where
       consistency is the intensity-weighted fraction of the candidate's
       strong (≥ 20 %) lines present in ``all_peaks``. The score is maximised
       over an isotropic lattice scale within ±``max_strain`` (solid
       solutions, DFT over-binding); the scale is reported.

    A high score is a hypothesis to check by overlay, not an identification:
    one or two unassigned peaks are explained by many phases.
    Returns a DataFrame sorted by score; ``df.attrs["errors"]`` lists failed
    queries, ``df.attrs["source"]`` the database actually used and
    ``df.attrs["structures"]`` the MP structures by id (no API key needed).
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    wl1 = _wl_tuple(wavelength)[0]
    target_tt = target_peaks["two_theta"].to_numpy(float)
    target_h = target_peaks["height"].to_numpy(float)
    observed_tt = all_peaks["two_theta"].to_numpy(float)
    elements = [e.strip() for e in elements if e.strip()]
    subsets = [s for k in range(1, min(max_subset, len(elements)) + 1)
               for s in combinations(elements, k)]
    errors: List[str] = []

    def run(fn):
        def q(sub):
            try:
                return fn(sub, nelements=len(sub)), None
            except Exception as e:
                return [], f"{'-'.join(sub)}: {type(e).__name__}: {e}"[:200]
        with ThreadPoolExecutor(max_workers=6) as ex:
            out = list(ex.map(q, subsets))
        return [e for r, _ in out for e in r], [err for _, err in out if err]

    used = source
    rows: List[Dict] = []
    structures: Dict[str, object] = {}
    if source in ("cod", "auto"):
        hits, errs = run(cod_optimade_query)
        errors += [f"COD {e}" for e in errs]
        if source == "auto" and errs and len(errs) == len(subsets):
            used = "mp"
        else:
            used = "cod"
            best: Dict[Tuple[str, str], Dict] = {}
            for e in hits:
                if not e["has_coordinates"]:
                    continue
                key = (e["formula"], e["spacegroup"])
                r = e["R"] if e["R"] is not None else 1.0
                rb = best[key]["R"] if key in best and best[key]["R"] is not None else 1.0
                if key not in best or r < rb:
                    best[key] = e
            cands = list(best.values())
            if progress:
                progress(f"COD: {len(hits)} entries, {len(cands)} distinct "
                         "formula/space group")
            pre = []
            for e in cands:
                tt = _candidate_positions(e, wl1, tt_range)
                if tt is None:
                    pre.append((0.5, e))
                    continue
                sc = _best_scale_score(tt, None, wl1, target_tt, target_h,
                                       observed_tt, tol, max_strain)
                pre.append((sc[0], e))
            pre.sort(key=lambda t: -t[0])
            short = [e for s_, e in pre[:prefilter_keep] if s_ > 0]

            def full(e):
                from xrdkit.db import from_cod_id
                try:
                    pk = from_cod_id(e["cod_id"]).peaks(wl1, (tt_range[0] - 3,
                                                              tt_range[1] + 3))
                    return {"id": f"COD {e['cod_id']}", "formula": e["formula"],
                            "spacegroup": e["spacegroup"], "title": e["title"],
                            "peaks": pk}
                except Exception as err:
                    errors.append(f"COD {e['cod_id']} CIF: {type(err).__name__}: {err}"[:200])
                    return None
            with ThreadPoolExecutor(max_workers=6) as ex:
                rows = [r for r in ex.map(full, short) if r]
    if used == "mp":
        hits, errs = run(mp_optimade_query)
        errors += [f"MP {e}" for e in errs]
        if progress:
            progress(f"Materials Project: {len(hits)} structures within 0.1 eV/atom of hull")
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        calc = XRDCalculator(wavelength=wl1)
        for e in hits:
            try:
                pat = calc.get_pattern(e["structure"], two_theta_range=(
                    max(tt_range[0] - 3, 1.0), min(tt_range[1] + 3, 179.0)))
                pk = [(float(t), float(i), None) for t, i in zip(pat.x, pat.y)]
            except Exception:
                continue
            rows.append({"id": e["mp_id"], "formula": e["formula"],
                         "spacegroup": e["spacegroup"],
                         "title": f"E_hull {e['e_above_hull']:.3f} eV/atom"
                         if e["e_above_hull"] is not None else "",
                         "peaks": pk})
            structures[e["mp_id"]] = e["structure"]

    out = []
    for r in rows:
        tt = np.array([p[0] for p in r["peaks"]], float)
        I = np.array([p[1] for p in r["peaks"]], float)
        if I.size:
            I = 100.0 * I / I.max()
        sc, ex_, co, matched, scale = _best_scale_score(
            tt, I, wl1, target_tt, target_h, observed_tt, tol, max_strain)
        out.append({"id": r["id"], "formula": r["formula"],
                    "spacegroup": r["spacegroup"], "score": sc,
                    "explained": ex_, "consistency": co,
                    "lattice scale": scale,
                    "matched 2θ": ", ".join(f"{v:.2f}" for v in matched),
                    "note": r["title"]})
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["score", "explained"], ascending=False).reset_index(drop=True)
    df.attrs["errors"] = errors
    df.attrs["source"] = used
    df.attrs["structures"] = structures
    return df
