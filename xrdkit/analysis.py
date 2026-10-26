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

Conventions: angles in degrees 2θ, wavelengths in Å, d in Å. Pawley
uncertainties are 1σ from the least-squares covariance scaled by the reduced
χ² and then inflated for serially correlated residuals (Bérar & Lelann, J.
Appl. Cryst. 24, 1, 1991; see :func:`_serial_correlation`). They are still
statistical only: the zero/displacement choice adds a systematic error that is
reported separately (``cell_sys``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from xrdkit.core import pseudo_voigt
from xrdkit.core import _PV_AREA_G
from xrdkit.core import pv_area_factor as _pv_area_factor
from xrdkit.plot import Measured, _snip_background

# Status codes of a Pawley fit (PawleyResult.status)
# "unexplained": the model's reflections do not beat a background-only fit,
# because there is no signal or because this cell does not index it.
STATUS_OK, STATUS_POOR, STATUS_UNEXPLAINED = "ok", "poor_fit", "unexplained"
STATUS_NO_SIGNAL = STATUS_UNEXPLAINED   # backwards-compatible alias


def _blas_quiet():
    """numpy 2.x on macOS Accelerate emits spurious 'divide by zero / overflow
    encountered in matmul' warnings on finite inputs; callers check results."""
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


def _serial_correlation(r, max_lag: int):
    """Durbin–Watson d and esd inflation of correlated weighted residuals r.

    The inflation is √τ with τ = 1 + 2 Σ_k ρ_k the integrated autocorrelation
    of r, summed over lags k = 1 … ``max_lag`` (≈ one FWHM in points) and
    truncated at the first ρ_k ≤ 0. This is the idea of Bérar & Lelann (J.
    Appl. Cryst. 24, 1, 1991): residuals correlated over a peak width carry
    fewer independent observations, N_eff = N/τ. τ ≥ 1 by construction.
    """
    r = np.asarray(r, float) - np.mean(r)
    ss = float(np.sum(r * r))
    if ss <= 0 or r.size < 4:
        return 2.0, 1.0
    dw = float(np.sum(np.diff(r) ** 2) / ss)
    tau = 1.0
    for k in range(1, max(int(max_lag), 1) + 1):
        rho = float(np.sum(r[:-k] * r[k:])) / ss
        if rho <= 0:
            break
        tau += 2.0 * rho
    return dw, float(np.sqrt(tau))


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
    the data behave as raw counts: for Poisson counts var(y[i+1] − y[i]) =
    y[i+1] + y[i], so the normalised first differences have σ = 1 at any
    count level (robust MAD estimate; peak slopes are a minority of points).
    Exported intensities that were scaled, averaged, smoothed or offset for
    stacking give σ far from 1 and get uniform weights.
    """
    y = np.asarray(y, float)
    bkg = estimate_background(x, y)
    sn = _robust_noise(y - bkg)
    if mode == "auto":
        d = np.diff(y) / np.sqrt(np.clip(y[1:] + y[:-1], 1.0, None))
        z = 1.4826 * float(np.median(np.abs(d - np.median(d))))
        mode = "poisson" if 0.75 <= z <= 1.5 and np.min(y) >= 0 else "uniform"
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
    ok = ((df["snr"] >= min_snr) & (df["fwhm"] >= 3 * step)
          & np.isfinite(df["two_theta_esd"]) & np.isfinite(df["fwhm_esd"])
          & (df["height"] >= 3 * df["height_esd"]))
    df = df[ok].reset_index(drop=True)
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


def williamson_hall(peaks, wavelength=None, K: float = 0.9,
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
    if wavelength is None:
        wavelength = peaks.attrs.get("wavelength", 1.54056)
    wavelength = _wl_tuple(wavelength)[0]
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
    weighted = bool(np.all(np.isfinite(sw)) and np.all(sw > 0))
    w = 1.0 / sw ** 2 if weighted else np.ones_like(Y)
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
                "r2": 1.0 - float(np.sum(r * r)) / ss_tot,
                "chi2_red": s2 if weighted else np.nan, "consistent": True})
    # R² is not a validity test: a pure-size sample has slope ≈ 0 and R² ≈ 0.
    # The test is whether the points scatter about the line by more than their
    # FWHM esds, i.e. the broadening is not one isotropic size + strain.
    if weighted and s2 > 10.0:
        res["consistent"] = False
    if c0 > 0:
        D = K * wavelength / c0 / 10.0
        res["D_nm"] = D
        res["D_nm_esd"] = D * sc0 / c0
        if sc0 > c0:
            res["note"] = "intercept not significant: size broadening unresolved"
    else:
        res["note"] = ("intercept ≤ 0: size broadening below resolution or the "
                       "instrumental correction is too large")
    if not res["consistent"]:
        res["note"] = (res["note"] + "; " if res["note"] else "") + (
            f"points scatter {np.sqrt(s2):.1f}× their esds about the line: the "
            "broadening is not a single isotropic size + strain (anisotropic, "
            "overlapping or split lines); D and ε are not meaningful")
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
        with _blas_quiet():
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
    Z: Optional[int]
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
    Rwp_net: float = float("nan")      # Rwp of the background-subtracted pattern
    durbin_watson: float = float("nan")
    esd_scale: float = 1.0             # correlated-residual inflation applied to esds
    n_eff: float = float("nan")        # N/τ
    bic_eff: float = float("nan")      # BIC with N_eff
    bic_null_eff: float = float("nan")  # background-only model, same N_eff
    cell_sys: Tuple[float, ...] = ()   # |Δcell| zero-only vs displacement-only
    volume_sys: float = float("nan")
    status: str = STATUS_OK
    warnings: List[str] = field(default_factory=list)

    @property
    def volume_per_fu(self) -> float:
        return self.volume / self.Z if self.Z else float("nan")

    @property
    def volume_per_fu_esd(self) -> float:
        return self.volume_esd / self.Z if self.Z else float("nan")

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
            "a_pc (Å)": self.a_pc if self.model in PEROVSKITE_MODELS else float("nan"),
            "zero (°)": self.zero, "σzero": self.zero_esd,
            "displ. (°)": self.displacement, "σdispl.": self.displacement_esd,
            "Rwp (%)": 100 * self.Rwp, "Rwp net (%)": 100 * self.Rwp_net,
            "Rp (%)": 100 * self.Rp, "Rexp (%)": 100 * self.Rexp, "GoF": self.gof,
            "DW": self.durbin_watson, "esd ×": self.esd_scale,
            "N_par": self.n_par, "BIC": self.bic, "BIC_eff": self.bic_eff,
            "status": self.status,
        }

    def profile_frame(self):
        """Observed, calculated, background and difference per point."""
        import pandas as pd
        return pd.DataFrame({"two_theta": self.x, "y_obs": self.y_obs,
                             "y_calc": self.y_calc, "y_bkg": self.y_bkg,
                             "diff": self.y_obs - self.y_calc})

    def reflections_frame(self):
        """Kα1 reflection positions, hkl and Pawley (integrated) intensities."""
        import pandas as pd
        h = np.asarray(self.hkl).reshape(-1, 3) if len(self.hkl) else np.zeros((0, 3))
        n = min(len(self.ticks), len(h))
        return pd.DataFrame({"two_theta": np.asarray(self.ticks)[:n],
                             "h": h[:n, 0], "k": h[:n, 1], "l": h[:n, 2],
                             "intensity": np.asarray(self.intensities)[:n]})

    def to_dict(self, provenance: Optional[Dict] = None) -> Dict:
        """JSON-safe summary: the table row, cell systematics, notes and the
        provenance (input file hash, wavelength, range, version) given."""
        def clean(v):
            if isinstance(v, (np.floating, float)):
                return float(v) if np.isfinite(v) else None
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [clean(u) for u in v]
            if isinstance(v, dict):
                return {str(k): clean(u) for k, u in v.items()}
            return v
        from xrdkit import __version__
        return clean({"schema": "xrdkit.pawley/1", "xrdkit_version": __version__,
                      **(provenance or {}), **self.as_row(),
                      "cell": self.cell, "cell_esd": self.cell_esd,
                      "cell_sys": self.cell_sys, "volume_sys": self.volume_sys,
                      "n_obs": self.n_obs, "n_eff": self.n_eff,
                      "bic_null_eff": self.bic_null_eff, "U": self.U, "V": self.V,
                      "W": self.W, "eta": self.eta, "message": self.message,
                      "warnings": list(self.warnings),
                      "correlations": self.correlations,
                      "extra_phases": [{k: e[k] for k in e if k not in ("ticks", "hkl")}
                                       for e in self.extra]})


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
        self.zero_fixed = 0.0
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

    def thinned(self, k):
        """Copy on every k-th data point (same reflections and parameters)."""
        import copy
        if k <= 1:
            return self
        t = copy.copy(self)
        t.x, t.y, t.w, t.sw, t.B = (v[::k] for v in (self.x, self.y, self.w, self.sw, self.B))
        t.Qb, t.Rb = np.linalg.qr(t.B * t.sw[:, None])
        return t

    def unpack(self, p):
        d = dict(zip(self.names, p))
        free = [d[k] for k in self.free_names]
        cell = _cell_from_free(self.system, self.rhombo, free, self.base)
        return (cell, d.get("zero", self.zero_fixed), d.get("disp", 0.0),
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
        with _blas_quiet():
            Mp = M - self.Qb @ (self.Qb.T @ M)
            bp = b - self.Qb @ (self.Qb.T @ b)
            if M.shape[1]:
                Q, R = np.linalg.qr(Mp)
                I, _ = nnls(R, Q.T @ bp, maxiter=50 * max(M.shape[1], 1))
            else:
                I = np.zeros(0)
            cb = np.linalg.solve(self.Rb, self.Qb.T @ (b - M @ I))
            bkg = self.B @ cb
            yc = A @ I + bkg
        if not np.all(np.isfinite(yc)):
            raise FloatingPointError("non-finite calculated pattern")
        return yc, bkg, I, sel, tt1

    def chi2_null(self):
        """Weighted χ² of the background-only model (no reflections)."""
        b = self.y * self.sw
        with _blas_quiet():
            r = b - self.Qb @ (self.Qb.T @ b)
        return float(np.sum(r * r))

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
        if "cell" not in out:
            out = _spec_from_lattice_info(out)
        out["cell"] = tuple(float(v) for v in out["cell"])
        return out
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(ph, symprec=0.01)
    conv = sga.get_conventional_standard_structure()
    sym = sga.get_space_group_symbol()
    if sga.get_crystal_system() == "trigonal" and sym.startswith("R"):
        sym += ":H"
    lat = conv.lattice
    Z = int(conv.composition.get_reduced_composition_and_factor()[1])
    return {"name": f"{ph.composition.reduced_formula} {sym}", "spacegroup": sym,
            "cell": tuple(lat.abc) + tuple(lat.angles), "Z": Z, "refine_cell": True}


def _spec_from_lattice_info(info: Dict) -> Dict:
    """Spec from a reference card: {"spacegroup", "system", "a", "b"?, "c"?, angles?}."""
    sg = str(info.get("spacegroup") or "").strip()
    if not sg or "proxy" in sg:
        raise ValueError("reference card has no usable space group")
    _space_group(sg)  # validates the symbol
    sysname = str(info.get("system", "")).lower()
    a = float(info["a"])
    b = float(info.get("b") or a)
    c = float(info.get("c") or a)
    al, be, ga = (float(info.get(k) or 90.0) for k in ("alpha", "beta", "gamma"))
    if sysname in ("hexagonal", "hcp", "trigonal") and not info.get("gamma"):
        ga = 120.0
    if sg.startswith("R") and ga == 120.0 and not sg.endswith(":H"):
        sg += ":H"
    return {"name": info.get("name") or sg, "spacegroup": sg, "cell": (a, b, c, al, be, ga),
            "Z": info.get("Z"), "refine_cell": True}


def pawley_refine(m: Measured, model: str = "R-3c", a_pc: Optional[float] = None,
                  cell0: Optional[Sequence[float]] = None, Z: Optional[int] = None,
                  wavelength=(1.54056, 1.54439, 0.5), tt_range=None,
                  n_bkg: int = 8, refine_zero: bool = False,
                  refine_disp: bool = True, fwhm0: Optional[float] = None,
                  multistart: bool = True, weighting: str = "auto",
                  extra_phases: Sequence = (), zero_fixed: float = 0.0,
                  check_systematics: bool = True) -> PawleyResult:
    """Pawley fit of one measured pattern with one space-group model.

    ``model`` is a key of :data:`PEROVSKITE_MODELS` (then ``a_pc`` sets the
    starting cell; if omitted it is estimated from the strongest peak taken as
    the pseudocubic 110) or any space-group symbol together with ``cell0`` =
    (a, b, c, α, β, γ) and ``Z``.

    Zero offset and sample displacement are both nearly constant over a
    20–80° scan (cosθ changes by ~20 %), so refining both at once is
    ill-conditioned; the default refines displacement only. Which of the two
    is refined shifts the cell by more than its statistical esd, so with
    ``check_systematics`` the fit is repeated with the other one and the
    difference is reported as ``cell_sys`` / ``volume_sys``. If the zero is
    known from a standard (Si, LaB6) pass it as ``zero_fixed`` (°); the check
    is then skipped. A Measured with ``info["kalpha2_stripped"]`` is fitted
    with Kα1 only. Lower symmetry
    always fits at least as well; compare models with :func:`compare_models`
    (BIC) and by which pseudocubic reflections are split.

    ``weighting`` is passed to :func:`counting_weights`. BIC is computed in
    its unknown-variance form N·ln(χ²/N) + P·ln N, which does not depend on
    the absolute intensity scale or on a wrong noise model; ``bic_eff`` uses
    N_eff = N/τ for serially correlated residuals and is the one to compare.
    ``bic_null_eff`` is the background-only model on the same data: a fit
    that does not beat it by ΔBIC > 10 has status ``"unexplained"`` (no
    signal, or a cell that does not index the peaks).

    ``extra_phases`` are secondary phases refined jointly (Pawley intensities,
    own lattice constants, own FWHM scale on the shared Caglioti profile):
    pymatgen Structures or dicts {"name", "spacegroup", "cell", "refine_cell"}.
    Without them, lines of an impurity are absorbed as spurious intensity of
    the main phase and bias its cell.
    """
    from scipy.optimize import least_squares

    wl1, wl2, ratio = _wl_tuple(wavelength)
    notes: List[str] = []
    if (m.info or {}).get("kalpha2_stripped") and wl2:
        wl2, ratio = None, 0.0
        notes.append("input is Kα2-stripped: fitted with Kα1 only")
    x, y = _crop(m, tt_range)
    if fwhm0 is None:
        fwhm0 = _median_fwhm(x, y)
    if model in PEROVSKITE_MODELS and cell0 is None:
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
    cell0 = tuple(float(v) for v in cell0)
    Z = int(Z) if Z else None  # unknown Z → V/Z is not reported

    extra = [phase_spec(e) for e in extra_phases]
    pm = _PawleyModel(x, y, sg_symbol, cell0, wl1, wl2, ratio, n_bkg,
                      refine_zero, refine_disp, weighting, extra)
    pm.zero_fixed = float(zero_fixed)
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
    wmax = max(1.0, 4.0 * fwhm0 ** 2)  # allow nanocrystalline (FWHM of degrees)
    lo += [0.0, -0.5 * wmax, 1e-6, 0.0]
    hi += [wmax, 0.5 * wmax, wmax, 1.0]
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

    # basin search (scale scan, multistart) on thinned data,
    # >= 4 points per median FWHM; the final refinement and esds use every point
    step = float(np.median(np.diff(x))) if x.size > 1 else fwhm0
    pc = pm.thinned(int(np.clip(fwhm0 / (4.0 * max(step, 1e-9)), 1, 10)))

    # --- coarse isotropic scale scan (±3 %, no displacement): a start a
    # few 0.1 % off puts every peak beside its reflection and the local
    # optimiser then lets the displacement absorb the misfit
    len_idx = [i for i, k in enumerate(pm.free_names) if k in ("a", "b", "c")]
    if len_idx:
        best_s, best_c = 1.0, np.inf
        for s_ in np.linspace(0.97, 1.03, 121):
            st = list(base_free)
            for i in len_idx:
                st[i] *= s_
            c_ = float(np.sum(pc.residual(np.clip(full_p(st), lo, hi)) ** 2))
            if c_ < best_c:
                best_s, best_c = s_, c_
        for i in len_idx:
            base_free[i] *= best_s

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
            return pc.residual(q)
        sol = least_squares(r, p_full[mask], bounds=(lo[mask], hi[mask]),
                            x_scale="jac", max_nfev=25, diff_step=1e-6)
        q = p_full.copy()
        q[mask] = sol.x
        # rank starts on the full data: thinning can reorder near-degenerate basins
        return q, float(np.sum(pm.residual(q) ** 2)) if pc is not pm else float(np.sum(sol.fun ** 2))

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
    Rwp_net = float(np.sqrt(chi2 / max(np.sum(w * (y - bkg) ** 2), 1e-300)))
    Rp = float(np.sum(np.abs(r)) / np.sum(np.abs(y)))
    Rexp = float(np.sqrt(dof / np.sum(w * y * y)))

    # serial correlation over ~one FWHM → esd inflation and N_eff
    step = float(np.median(np.diff(x))) if x.size > 1 else fwhm0
    fw_mean = float(np.mean(caglioti_fwhm(tt1, *pm.unpack(p)[3:6]))) if tt1.size else fwhm0
    dw, esd_scale = _serial_correlation(pm.sw * r, int(np.ceil(fw_mean / max(step, 1e-9))))
    n_eff = n_obs / esd_scale ** 2

    J = sol.jac
    try:
        with _blas_quiet():
            cov = np.linalg.pinv(J.T @ J) * (chi2 / dof) * esd_scale ** 2
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
    with _blas_quiet():
        Cn = cov / np.outer(sd, sd)
    for i, j in combinations(range(len(p)), 2):
        if np.isfinite(Cn[i, j]) and abs(Cn[i, j]) > 0.8:
            corr[f"{pm.names[i]}–{pm.names[j]}"] = float(Cn[i, j])

    # zero-only vs displacement-only: systematic part of the cell error
    cell_sys, V_sys = (), float("nan")
    if check_systematics and refine_zero != refine_disp and zero_fixed == 0.0:
        import copy
        alt = copy.copy(pm)
        alt.names = ["zero" if k == "disp" else "disp" if k == "zero" else k
                     for k in pm.names]
        i0 = pm.names.index("disp" if refine_disp else "zero")
        cth = float(np.mean(np.cos(np.radians(0.5 * x))))
        p2 = p.copy()
        p2[i0] = p[i0] * cth if refine_disp else p[i0] / cth
        p2 = np.clip(p2, lo, hi)
        try:
            s2 = least_squares(alt.residual, p2, bounds=(lo, hi), x_scale="jac",
                               max_nfev=100, diff_step=1e-6)
            c2 = alt.unpack(s2.x)[0]
            cell_sys = tuple(float(abs(u - v)) for u, v in zip(c2, cell))
            V_sys = float(abs(cell_volume(c2) - cell_volume(cell)))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            pass

    # parameters stuck at a bound (η ∈ {0, 1} and U = 0 are legitimate limits)
    for k, v, a_, b_ in zip(pm.names, p, lo, hi):
        if k in ("eta", "U"):
            continue
        if min(v - a_, b_ - v) < 1e-3 * (b_ - a_):
            notes.append(f"{k} = {v:.5g} is at its bound [{a_:.5g}, {b_:.5g}]: "
                         "the fit is constrained there, its esd is meaningless")
    _prof = ("U", "V", "W", "eta")
    for (i, j) in combinations(range(len(p)), 2):
        k = f"{pm.names[i]}–{pm.names[j]}"
        c = corr.get(k, 0.0)
        if abs(c) > 0.95 and not (pm.names[i] in _prof and pm.names[j] in _prof):
            notes.append(f"{k} correlation {c:+.2f}: these are not separately determined")

    bic = n_obs * np.log(chi2 / n_obs) + n_par * np.log(n_obs)
    bic_eff = n_eff * np.log(chi2 / n_obs) + n_par * np.log(n_eff)
    chi2_0 = pm.chi2_null()
    bic_null_eff = n_eff * np.log(chi2_0 / n_obs) + pm.B.shape[1] * np.log(n_eff)
    gof = Rwp / Rexp if Rexp else np.nan
    status = STATUS_OK
    if not bic_eff < bic_null_eff - 10.0:
        status = STATUS_UNEXPLAINED
        notes.append(f"this model's reflections do not explain the pattern: "
                     f"BIC_eff − BIC_null = {bic_eff - bic_null_eff:+.1f} (needs "
                     "< −10 against a background-only fit). Either there is no "
                     "diffraction signal or this cell does not index the peaks")
    elif gof > 3:
        status = STATUS_POOR
        notes.append(f"GoF = {gof:.1f} > 3 — misfit (wrong cell, missing phase or "
                     "false minimum); check the difference curve")
    if esd_scale > 1.5:
        notes.append(f"residuals serially correlated (Durbin–Watson {dw:.2f}): "
                     f"esds inflated ×{esd_scale:.1f}")

    return PawleyResult(
        model=model, spacegroup=pm.sg.symbol, crystal_system=pm.system,
        cell=tuple(float(v) for v in cell), cell_esd=cell_esd,
        volume=cell_volume(cell), volume_esd=V_esd, Z=Z,
        zero=float(zero), zero_esd=_esd("zero"),
        displacement=float(disp), displacement_esd=_esd("disp"),
        U=float(U), V=float(V), W=float(W), eta=float(eta),
        Rwp=Rwp, Rp=Rp, Rexp=Rexp, gof=gof,
        chi2=chi2, n_obs=n_obs, n_par=n_par, bic=float(bic),
        x=x, y_obs=y, y_calc=yc, y_bkg=bkg, ticks=tt1,
        hkl=pm.hkl[sel], intensities=I[:int(np.sum(sel))], correlations=corr,
        extra=extra_out,
        message=f"{sol.message} [{pm.weighting} weights]",
        Rwp_net=Rwp_net, durbin_watson=dw, esd_scale=esd_scale, n_eff=n_eff,
        bic_eff=float(bic_eff), bic_null_eff=float(bic_null_eff),
        cell_sys=cell_sys, volume_sys=V_sys, status=status, warnings=notes,
    )


def _median_fwhm(x, y) -> float:
    """Median FWHM (°) of the clearly resolved peaks; seeds the profile."""
    from scipy.signal import find_peaks
    step = float(np.median(np.diff(x)))
    net = y - estimate_background(x, y)
    idx, pr = find_peaks(net, prominence=8 * _robust_noise(net), width=1.5)
    if idx.size == 0:
        return 0.12
    return float(np.clip(np.median(pr["widths"]) * step, 2 * step, 3.0))


def _pc_index(m: Measured, a_pc: float, wl1: float, tt_range, tol: float):
    """(indexed, mixed) fractions of the net intensity of the 8 strongest
    peaks: indexed by a primitive cubic cell ``a_pc``, and indexed only by
    mixed-parity pseudocubic hkl (100, 110, 210, …)."""
    from scipy.signal import find_peaks
    x, y = _crop(m, tt_range)
    net = y - estimate_background(x, y)
    idx, pr = find_peaks(net, prominence=5 * _robust_noise(net))
    if idx.size == 0:
        return 0.0, 0.0
    top = idx[np.argsort(pr["prominences"])[::-1][:8]]
    d = wl1 / (2 * np.sin(np.radians(x[top] / 2)))
    hkl = np.array([(h, k, l) for h in range(5) for k in range(h + 1)
                    for l in range(k + 1) if h + k + l > 0])
    d_pc = a_pc / np.sqrt((hkl ** 2).sum(1))
    mixed_hkl = (hkl % 2).sum(1) % 3 != 0  # not all-even / all-odd
    ok = np.zeros(len(d), bool)
    mixed = np.zeros(len(d), bool)
    for i, di in enumerate(d):
        hit = np.abs(d_pc / di - 1) <= tol
        ok[i] = hit.any()
        mixed[i] = hit.any() and mixed_hkl[hit].all()
    w = net[top]
    tot = max(w.sum(), 1e-12)
    return float(w[ok].sum() / tot), float(w[mixed].sum() / tot)


def perovskite_check(m: Measured, a_pc: float, wl1: float = 1.54056, tt_range=None,
                     tol: float = 0.015) -> float:
    """Fraction of the net intensity of the 8 strongest peaks that a primitive
    cubic cell with parameter ``a_pc`` indexes (relative d tolerance ``tol``,
    wide enough for split pseudocubic lines). A single-phase perovskite gives
    ≈ 1; values well below ~0.7 mean the pattern is not (only) a perovskite
    and the tilt-model comparison is meaningless.

    A face-centred pattern (fcc metal, rock salt, fluorite) also scores 1,
    because its lines are a subset of primitive cubic; see
    :func:`perovskite_mixed_fraction`.
    """
    return _pc_index(m, a_pc, wl1, tt_range, tol)[0]


def perovskite_mixed_fraction(m: Measured, a_pc: float, wl1: float = 1.54056,
                              tt_range=None, tol: float = 0.015) -> float:
    """Share of the strongest-peak intensity on mixed-parity pseudocubic hkl.

    In an ABO3 perovskite the pseudocubic 110 (F ≈ f_A + f_B − f_O) is the
    strongest line, so this share is large; a face-centred lattice has no
    mixed-parity reflections and gives ≈ 0.
    """
    return _pc_index(m, a_pc, wl1, tt_range, tol)[1]


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


def compare_models(m: Measured, models: Sequence = ("Pm-3m", "R-3c", "Pnma"),
                   **kwargs):
    """Refine every model and return (results list, DataFrame sorted by BIC).

    Each model is a key of :data:`PEROVSKITE_MODELS`, a pymatgen Structure, or
    a spec dict {"name", "spacegroup", "cell", "Z"} (see :func:`phase_spec`),
    so candidate structures of any material can be compared.

    Models are ranked by BIC with a common effective number of observations
    N_eff = N/τ, τ taken from the best (lowest χ²) fit, because serially
    correlated residuals make the nominal N overstate the evidence. ΔBIC > 10
    is conventionally strong evidence against a model (Kass & Raftery, JASA
    90, 773, 1995). Pnma and Pbnm are one space group (No. 62) in two axis
    settings, not two candidates. Groups that differ only by weak
    superstructure (octahedral-tilt or cation-order) reflections — e.g. R-3
    vs R-3c — are usually not distinguishable from lab-XRD intensities even
    though their reflection lists differ.
    """
    import pandas as pd
    a_pc = kwargs.pop("a_pc", None)
    kwargs.pop("cell0", None)
    kwargs.pop("Z", None)
    if a_pc is None and any(isinstance(md, str) and md in PEROVSKITE_MODELS
                            for md in models):
        wl1 = _wl_tuple(kwargs.get("wavelength", 1.54056))[0]
        a_pc = estimate_a_pc(m, wl1, kwargs.get("tt_range"))
    pc_frac = mixed_frac = None
    if a_pc is not None:
        wl1 = _wl_tuple(kwargs.get("wavelength", 1.54056))[0]
        pc_frac, mixed_frac = _pc_index(m, a_pc, wl1, kwargs.get("tt_range"), 0.015)
    def one(md):
        name = md if isinstance(md, str) else None
        try:
            if isinstance(md, str) and md in PEROVSKITE_MODELS:
                r = pawley_refine(m, model=md, a_pc=a_pc, **kwargs)
            elif isinstance(md, str):
                raise ValueError(f"'{md}' is not a perovskite tilt model; pass a "
                                 "Structure or a spec dict with its cell")
            else:
                spec = phase_spec(md)
                name = spec["name"]
                r = pawley_refine(m, model=spec["spacegroup"], cell0=spec["cell"],
                                  Z=spec.get("Z"), **kwargs)
                r.model = name
        except Exception as e:  # keep the other models
            r = e
        return name or str(getattr(md, "composition", md)), r

    out = [one(md) for md in models]
    names = [o[0] for o in out]
    results = [o[1] for o in out]
    rows = []
    for md, r in zip(names, results):
        if isinstance(r, Exception):
            rows.append({"model": md, "error": f"{type(r).__name__}: {r}"})
        else:
            rows.append(r.as_row())
    df = pd.DataFrame(rows)
    ok = [r for r in results if not isinstance(r, Exception)]
    if ok:
        ref = min(ok, key=lambda r: r.chi2)
        n_eff = ref.n_eff
        bmap = {}
        for md, r in zip(names, results):
            if not isinstance(r, Exception):
                r.bic_eff = float(n_eff * np.log(r.chi2 / r.n_obs) + r.n_par * np.log(n_eff))
                bmap[r.model if r.model else md] = r.bic_eff
        df["BIC_eff"] = [bmap.get(md, np.nan) for md in df["model"]]
        df["ΔBIC"] = df["BIC"] - df["BIC"].min()
        df["ΔBIC_eff"] = df["BIC_eff"] - df["BIC_eff"].min()
        df = df.sort_values("BIC_eff").reset_index(drop=True)
        df.attrs["n_eff"] = n_eff
    if "status" in df and (df["status"] == STATUS_UNEXPLAINED).all():
        df.attrs["warning"] = ("no model beats a background-only fit (ΔBIC_eff > −10): "
                               "none of these cells indexes the observed peaks (or "
                               "there is no signal); the ranking is not meaningful")
    if "Rwp (%)" in df and np.isfinite(df["Rwp (%)"].iloc[0]) and df["Rwp (%)"].iloc[0] > 20:
        df.attrs["warning"] = (df.attrs.get("warning", "") and df.attrs["warning"] + ". ") + (
            f"best candidate fits with Rwp = {df['Rwp (%)'].iloc[0]:.0f} %: none of the "
            "candidates describes this pattern; add the right phase(s)")
    if pc_frac is not None:
        df.attrs["perovskite_fraction"] = pc_frac
        df.attrs["perovskite_mixed_fraction"] = mixed_frac
        if pc_frac >= 0.7 and mixed_frac < 0.1:
            df.attrs["warning"] = (df.attrs.get("warning", "") and df.attrs["warning"] + ". ") + (
                f"the strong lines index as a face-centred cubic cell (a = {a_pc:.3f} Å): "
                f"only {100 * mixed_frac:.0f} % of their intensity is on mixed-parity "
                "pseudocubic hkl, but the 110 must be the strongest perovskite line. "
                "This looks like an fcc metal / rock-salt / fluorite phase, not a "
                "perovskite; the tilt-model ranking is not meaningful.")
        if pc_frac < 0.7:
            df.attrs["warning"] = (df.attrs.get("warning", "") and df.attrs["warning"] + ". ") + (
                f"only {100 * pc_frac:.0f} % of the strongest-peak intensity is indexed "
                f"by a pseudocubic cell a = {a_pc:.3f} Å: this pattern is not a "
                "single-phase perovskite, so the tilt-model ranking is not meaningful. "
                "Refine reference-phase candidates or add the other phases.")
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
    if not wh.get("consistent", True):
        title = "not a single size + strain (χ²_red > 10)"
    elif np.isfinite(D):
        title = f"D = {D:.1f} nm, ε = {e * 100:.3f} %"
    else:
        title = f"ε = {e * 100:.3f} % (D undetermined)"
    ax.set_title(title, fontsize=8)
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
                       max_pages: int = 50, timeout: float = 60.0) -> List[Dict]:
    """Query COD over OPTIMADE for entries containing all ``elements``.

    Returns dicts with cod_id, formula (reduced), raw formula, spacegroup,
    title, R and the cell (a, b, c, α, β, γ) when COD lists it."""
    from xrdkit.core import http_get
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
        r = http_get(url, params=params, timeout=timeout)
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
    else:
        # more entries than max_pages × 100: a silently truncated list would
        # drop candidates at random, so fail instead
        raise RuntimeError(f"more than {max_pages * 100} entries; list truncated")
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
                      max_ehull: float = 0.1, max_pages: int = 50,
                      timeout: float = 60.0) -> List[Dict]:
    """Materials Project structures over OPTIMADE (no API key needed).

    Returns dicts with mp_id, formula, spacegroup, e_above_hull (eV/atom) and
    a pymatgen ``structure``. These are DFT (GGA/GGA+U) relaxed cells, which
    are typically 1–2 % larger than experiment, so match with a lattice-scale
    scan (``max_strain`` in :func:`match_cod`)."""
    from xrdkit.core import http_get
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
        r = http_get(url, params=params, timeout=timeout)
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
    else:
        # more entries than max_pages × 100: a silently truncated list would
        # drop candidates at random, so fail instead
        raise RuntimeError(f"more than {max_pages * 100} entries; list truncated")
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


_GAS_ELEMENTS = {"H", "N", "O", "F", "Cl", "He", "Ne", "Ar", "Kr", "Xe"}


def _plausible_formula(formula: str, max_atoms: int = 100) -> bool:
    """False for entries that cannot be a phase in an oxide/alloy sample:
    solid molecular gases (O₂, N₂, …) and reduced formulas with more than
    ``max_atoms`` atoms (molecular/organic crystals). Partial-occupancy
    solid solutions (La₀.₆Sr₀.₄MnO₃) are kept."""
    from pymatgen.core import Composition
    try:
        c = Composition(formula).reduced_composition
    except Exception:
        return False
    els = {str(e) for e in c.elements}
    if len(els) == 1 and els <= _GAS_ELEMENTS:
        return False
    return float(sum(c.values())) <= max_atoms


# a cell with more distinct reflections than this in the measured range has
# lines within a few FWHM of any observed peak: its "match" is coincidence
MAX_REFLECTIONS = 1000


def _database_candidates(target_tt, target_h, observed_tt, elements, wl1, tt_range,
                         tol, max_subset, prefilter_keep, max_strain, source,
                         progress=None):
    """Database phases built from subsets of ``elements`` with kinematic lines.

    Returns (rows, errors, source used, failed); each row has id, formula,
    spacegroup, title, peaks [(2θ, I, hkl)] and the pymatgen ``structure``;
    ``failed`` lists the queries and downloads that failed after retries, i.e.
    the candidate list is incomplete when it is not empty. COD entries
    are prefiltered on their allowed reflection positions (cell + space group)
    before their CIFs are downloaded: the ``prefilter_keep`` best by explained
    target intensity, plus the ``prefilter_keep`` best by the fraction of
    their own lines that coincide with an observed peak (so that a minor
    phase with few lines is not crowded out by large cells that explain
    everything with dense, mostly absent reflections)."""
    from concurrent.futures import ThreadPoolExecutor

    subsets = [s for k in range(1, min(max_subset, len(elements)) + 1)
               for s in combinations(elements, k)]
    errors: List[str] = []
    failed: List[str] = []

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
    if source in ("cod", "auto"):
        hits, errs = run(cod_optimade_query)
        errors += [f"COD {e}" for e in errs]
        if source == "auto" and errs and len(errs) == len(subsets):
            used = "mp"
        else:
            used = "cod"
            failed += [f"COD query {e}" for e in errs]
            # one entry (lowest R) per formula, space group and cell: entries
            # of one compound measured at high pressure / temperature or of a
            # different composition within the formula's rounding have other
            # cells and must not hide the ambient one (0.5 % length bins)
            best: Dict[Tuple, Dict] = {}
            for e in hits:
                if not e["has_coordinates"]:
                    continue
                cbin = tuple(int(np.round(np.log(float(v)) / 0.005))
                             for v in e["cell"][:3]) if e.get("cell") else ()
                key = (e["formula"], e["spacegroup"], cbin)
                r = e["R"] if e["R"] is not None else 1.0
                rb = best[key]["R"] if key in best and best[key]["R"] is not None else 1.0
                if key not in best or r < rb:
                    best[key] = e
            cands = [e for e in best.values() if _plausible_formula(e["formula"])]
            if progress:
                progress(f"COD: {len(hits)} entries, {len(cands)} distinct "
                         "formula/space group/cell")
            # fraction of the 2θ range within ±tol of an observed peak: the
            # hit rate of lines placed at random (chance coincidence)
            lo_, hi_ = tt_range
            grid = np.linspace(lo_, hi_, 20001)
            chance = float(np.mean(np.min(np.abs(grid[:, None] - observed_tt[None, :]),
                                          axis=1) <= tol)) if observed_tt.size else 0.0
            by_expl, by_prec = [], []
            scales = np.linspace(1 - max_strain, 1 + max_strain,
                                 int(2 * max_strain / 0.001) + 1) if max_strain > 0 else [1.0]
            for i, e in enumerate(cands):
                tt = _candidate_positions(e, wl1, tt_range)
                if tt is None:
                    by_expl.append((0.5, i))
                    continue
                n_pos = np.unique(np.round(tt, 3)).size
                if n_pos > MAX_REFLECTIONS:
                    errors.append(f"COD {e['cod_id']} {e['formula']} {e['spacegroup']}: "
                                  f"skipped, {n_pos} distinct reflections in range "
                                  "(too dense to identify from a powder pattern)")
                    continue
                sc = _best_scale_score(tt, None, wl1, target_tt, target_h,
                                       observed_tt, tol, max_strain)
                by_expl.append((sc[0], i))
                if tt.size and observed_tt.size:
                    # precision at the scale that puts most of the candidate's
                    # own lines on observed peaks: the scale that explains
                    # most height can move a minor phase's lines onto the
                    # main phase's peaks
                    prec = -1.0
                    for s_ in scales:
                        t2, ok = _rescale_tt(tt, wl1, s_)
                        if not ok.any():
                            continue
                        hit = np.min(np.abs(t2[ok][:, None] - observed_tt[None, :]),
                                     axis=1) <= tol
                        if hit.sum() >= 2:
                            prec = max(prec, float(hit.mean()))
                    if prec >= 0:
                        # precision in excess of chance coincidence
                        by_prec.append(((prec - chance) / max(1.0 - chance, 1e-6), i))
            # rank compounds, not cells: per formula + space group only the
            # cell that best matches the observed peaks competes, so that
            # many temperature / pressure entries of one compound do not
            # crowd other compounds out of the shortlist
            def per_compound(scored):
                top: Dict[Tuple, Tuple[float, int]] = {}
                for s_, i in scored:
                    k = (cands[i]["formula"], cands[i]["spacegroup"])
                    if k not in top or s_ > top[k][0]:
                        top[k] = (s_, i)
                return sorted(top.values(), key=lambda t: -t[0])[:prefilter_keep]
            keep = {i for s_, i in per_compound(by_expl) if s_ > 0}
            keep |= {i for s_, i in per_compound(by_prec) if s_ > 0}
            short = [cands[i] for i in sorted(keep)]

            def full(e):
                from xrdkit.db import from_cod_id
                try:
                    ph = from_cod_id(e["cod_id"])
                    pk = ph.peaks(wl1, (tt_range[0] - 3, tt_range[1] + 3))
                    return {"id": f"COD {e['cod_id']}", "formula": e["formula"],
                            "spacegroup": e["spacegroup"], "title": e["title"],
                            "peaks": pk, "structure": ph.info.get("structure")}
                except Exception as err:
                    import requests
                    msg = f"COD {e['cod_id']} CIF: {type(err).__name__}: {err}"[:200]
                    errors.append(msg)
                    # a download that failed may succeed next time; a CIF that
                    # cannot be parsed fails the same way on every run
                    if isinstance(err, requests.RequestException) or \
                            "did not return a CIF" in str(err):
                        failed.append(msg)
                    return None
            with ThreadPoolExecutor(max_workers=6) as ex:
                rows = [r for r in ex.map(full, short) if r]
    if used == "mp":
        hits, errs = run(mp_optimade_query)
        errors += [f"MP {e}" for e in errs]
        failed += [f"MP query {e}" for e in errs]
        if progress:
            progress(f"Materials Project: {len(hits)} structures within 0.1 eV/atom of hull")
        for e in hits:
            if not _plausible_formula(e["formula"]):
                continue
            pk = _kinematic_lines(e["structure"], wl1, tt_range)
            if pk is None:
                continue
            rows.append({"id": e["mp_id"], "formula": e["formula"],
                         "spacegroup": e["spacegroup"],
                         "title": f"E_hull {e['e_above_hull']:.3f} eV/atom"
                         if e["e_above_hull"] is not None else "",
                         "peaks": pk, "structure": e["structure"]})
    return rows, errors, used, failed


def _kinematic_lines(structure, wl1, tt_range):
    """[(2θ, I, None)] of a Structure (pymatgen kinematic, B = 0), or None."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    try:
        pat = XRDCalculator(wavelength=wl1).get_pattern(structure, two_theta_range=(
            max(tt_range[0] - 3, 1.0), min(tt_range[1] + 3, 179.0)))
    except Exception:
        return None
    return [(float(t), float(i), None) for t, i in zip(pat.x, pat.y)]


def _lines_array(peaks):
    tt = np.array([p[0] for p in peaks], float)
    I = np.array([p[1] for p in peaks], float)
    if I.size and I.max() > 0:
        I = 100.0 * I / I.max()
    return tt, I


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
       prefiltered entries) and score = explained × consistency, where
       consistency is the intensity-weighted fraction of the candidate's
       strong (≥ 20 %) lines present in ``all_peaks``. The score is maximised
       over an isotropic lattice scale within ±``max_strain`` (solid
       solutions, DFT over-binding); the scale is reported.

    Molecular-gas solids and very large reduced formulas are dropped
    (:func:`_plausible_formula`).
    A high score is a hypothesis to check by overlay, not an identification:
    one or two unassigned peaks are explained by many phases. For a complete
    identification with whole-pattern tests use :func:`identify_phases`.
    Returns a DataFrame sorted by score; ``df.attrs["errors"]`` lists failed
    queries, ``df.attrs["source"]`` the database actually used and
    ``df.attrs["structures"]`` the structures by id.
    """
    import pandas as pd

    wl1 = _wl_tuple(wavelength)[0]
    target_tt = target_peaks["two_theta"].to_numpy(float)
    target_h = target_peaks["height"].to_numpy(float)
    observed_tt = all_peaks["two_theta"].to_numpy(float)
    elements = [e.strip() for e in elements if e.strip()]
    rows, errors, used, failed = _database_candidates(
        target_tt, target_h, observed_tt, elements, wl1, tt_range, tol,
        max_subset, prefilter_keep, max_strain, source, progress)

    out = []
    for r in rows:
        tt, I = _lines_array(r["peaks"])
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
    df.attrs["structures"] = {r["id"]: r["structure"] for r in rows
                              if r.get("structure") is not None}
    return df


@dataclass
class PhaseIdentification:
    """Result of :func:`identify_phases`."""
    phases: "object"            # DataFrame, one row per accepted phase
    fit: Optional[PawleyResult]  # final joint Pawley fit (None if nothing found)
    peaks: "object"             # all observed peaks
    unexplained: "object"       # observed peaks no accepted phase explains
    steps: List[Dict]           # per step: candidates tried, ΔBIC, decision
    summary: str                # plain-language summary
    caveats: List[str]
    errors: List[str]
    source: str
    structures: Dict[str, object] = field(default_factory=dict)
    fixed_fit: Optional[Dict] = None   # fixed-structure fit (profiles, scales, cells)
    complete: bool = True       # False: database search incomplete, nothing reported


def _bic_at(r: PawleyResult, n_eff: float) -> float:
    return float(n_eff * np.log(r.chi2 / r.n_obs) + r.n_par * np.log(n_eff))


def _spec_scaled(structure, scale: float, name: str) -> Dict:
    sp = phase_spec(structure)
    a, b, c, al, be, ga = sp["cell"]
    sp["cell"] = (a * scale, b * scale, c * scale, al, be, ga)
    sp["name"] = name
    return sp


# ---------------------------------------------------------------------------
#  Fixed-structure (Rietveld-type) whole-pattern fit used to screen phases
# ---------------------------------------------------------------------------

def _cell_ties(cell, tol: float = 1e-4):
    """Free cell parameters implied by the metric of ``cell`` itself.

    Equal lengths share one parameter, angles at 90°/120° stay fixed, equal
    free angles share one parameter. This keeps the metric symmetry of any
    setting without needing the crystal-system convention of the setting.
    Returns (base values of the free parameters, index of the free parameter
    for each of a, b, c, α, β, γ or None)."""
    free, idx = [], [None] * 6
    for i in range(3):
        j = next((j for j in range(i) if abs(cell[i] - cell[j]) < tol * cell[j]), None)
        if j is None:
            idx[i] = len(free)
            free.append(float(cell[i]))
        else:
            idx[i] = idx[j]
    for i in range(3, 6):
        if min(abs(cell[i] - 90.0), abs(cell[i] - 120.0)) < 1e-3:
            continue
        j = next((j for j in range(3, i) if idx[j] is not None
                  and abs(cell[i] - cell[j]) < 1e-3), None)
        if j is None:
            idx[i] = len(free)
            free.append(float(cell[i]))
        else:
            idx[i] = idx[j]
    return free, idx


def _cell_from_ties(free, base, idx):
    return tuple(float(free[idx[i]]) if idx[i] is not None else float(base[i])
                 for i in range(6))


def _structure_lines(structure, wl: float, tt_range, name: str = ""):
    """Reflections of a crystal structure for :class:`_FixedModel`.

    The structure is put in its spglib standard conventional cell. Intensities
    are pymatgen kinematic |F|²·Lp per cell, summed over all symmetry
    equivalents (so multiplicity is included), without temperature factor
    and without normalisation, so that scale factors of different phases are
    comparable (Hill & Howard, J. Appl. Cryst. 20, 467, 1987). Returns a dict
    with the cell, its ties, hkl (n×3), I, the cell mass (g/mol) — or None.
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        st = SpacegroupAnalyzer(structure, symprec=0.01).get_refined_structure()
    except Exception:
        st = structure
    try:
        pat = XRDCalculator(wavelength=wl).get_pattern(
            st, scaled=False, two_theta_range=(max(tt_range[0] - 3, 1.0),
                                               min(tt_range[1] + 3, 179.0)))
    except Exception:
        return None
    lat = st.lattice
    cell = tuple(lat.abc) + tuple(lat.angles)
    hkl = np.array([(f[0]["hkl"][0], f[0]["hkl"][1], f[0]["hkl"][-1]) for f in pat.hkls],
                   float).reshape(-1, 3)
    I = np.asarray(pat.y, float)
    if not I.size:
        return None
    # hkl must reproduce pymatgen's d (Miller–Bravais → Miller, setting check)
    d = 1.0 / np.sqrt(np.einsum("ij,jk,ik->i", hkl, _recip_metric(cell), hkl))
    ok = np.abs(d - np.asarray(pat.d_hkls)) < 1e-4 * d
    if ok.mean() < 0.99:
        return None
    free, idx = _cell_ties(cell)
    return {"name": name, "cell": cell, "free0": free, "ties": idx, "hkl": hkl[ok],
            "I": I[ok], "mass": float(st.composition.weight), "structure": st}


class _FixedModel:
    """Whole-pattern model with fixed kinematic intensities per phase.

    Nonlinear: sample displacement D (2θ shift D·cosθ), Caglioti U V W, η
    (shared) and per phase its free cell parameters, an overall isotropic
    temperature factor B (Å², intensities × exp(−2B sin²θ/λ²)) and a FWHM
    scale. Linear (exact, variable projection): one scale factor ≥ 0 per
    phase and a Chebyshev background. Atomic positions and occupancies are
    those of the database entry; preferred orientation, microabsorption and
    anisotropic broadening are not modelled.
    """

    def __init__(self, x, y, phases, wl1, wl2, ratio, n_bkg=8, weighting="auto"):
        self.x, self.y = x, y
        self.w, self.weighting = counting_weights(x, y, weighting)
        self.sw = np.sqrt(self.w)
        self.ph = list(phases)
        self.wl1, self.wl2, self.ratio = wl1, wl2, ratio
        u = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-9) - 1.0
        self.B = np.polynomial.chebyshev.chebvander(u, max(int(n_bkg) - 1, 0))
        self.Qb, self.Rb = np.linalg.qr(self.B * self.sw[:, None])
        self.off, n = [], 5
        for ph in self.ph:
            self.off.append(n)
            n += len(ph["free0"]) + 2
        self.n_nl = n

    def thinned(self, k):
        import copy
        if k <= 1:
            return self
        t = copy.copy(self)
        t.x, t.y, t.w, t.sw, t.B = (v[::k] for v in (self.x, self.y, self.w, self.sw, self.B))
        t.Qb, t.Rb = np.linalg.qr(t.B * t.sw[:, None])
        return t

    def phase_params(self, p, k):
        ph, o = self.ph[k], self.off[k]
        nf = len(ph["free0"])
        return _cell_from_ties(p[o:o + nf], ph["cell"], ph["ties"]), p[o + nf], p[o + nf + 1]

    def positions(self, cell, hkl, disp, wl):
        inv_d2 = np.einsum("ij,jk,ik->i", hkl, _recip_metric(cell), hkl)
        s = 0.5 * wl * np.sqrt(np.clip(inv_d2, 0.0, None))
        th = np.arcsin(np.clip(s, 0.0, 0.999999))
        return 2.0 * np.degrees(th) + disp * np.cos(th), s < 1.0, inv_d2

    def columns(self, p):
        disp, U, V, W, eta = p[:5]
        eta = float(np.clip(eta, 0.0, 1.0))
        cols = []
        for k, ph in enumerate(self.ph):
            cell, Bi, ws = self.phase_params(p, k)
            col = np.zeros_like(self.x)
            for wl, fac in ((self.wl1, 1.0), (self.wl2, self.ratio)):
                if not wl:
                    continue
                tt, ok, inv_d2 = self.positions(cell, ph["hkl"], disp, wl)
                f = ws * caglioti_fwhm(tt, U, V, W)
                sel = ok & (tt > self.x.min() - 6 * f) & (tt < self.x.max() + 6 * f)
                if not sel.any():
                    continue
                amp = ph["I"][sel] * np.exp(-0.5 * Bi * inv_d2[sel])
                prof = pseudo_voigt(self.x[:, None] - tt[sel][None, :], 0.0,
                                    f[sel][None, :], eta) / _pv_area_factor(f[sel], eta)[None, :]
                with _blas_quiet():
                    col += fac * (prof @ amp)
            cols.append(col)
        return np.column_stack(cols) if cols else np.zeros((self.x.size, 0))

    def solve(self, p):
        from scipy.optimize import nnls
        A = self.columns(p)
        b = self.y * self.sw
        M = A * self.sw[:, None]
        with _blas_quiet():
            Mp = M - self.Qb @ (self.Qb.T @ M)
            bp = b - self.Qb @ (self.Qb.T @ b)
            if M.shape[1]:
                cn = np.linalg.norm(Mp, axis=0)
                cn = np.where(cn > 0, cn, 1.0)
                Q, R = np.linalg.qr(Mp / cn)
                S, _ = nnls(R, Q.T @ bp)
                S = S / cn
            else:
                S = np.zeros(0)
            cb = np.linalg.solve(self.Rb, self.Qb.T @ (b - M @ S))
            bkg = self.B @ cb
            yc = A @ S + bkg
        if not np.all(np.isfinite(yc)):
            raise FloatingPointError("non-finite calculated pattern")
        return yc, bkg, S, A

    def residual(self, p):
        return self.sw * (self.y - self.solve(p)[0])

    def chi2_null(self):
        b = self.y * self.sw
        with _blas_quiet():
            r = b - self.Qb @ (self.Qb.T @ b)
        return float(np.sum(r * r))


def _is_length(ph, j):
    return any(ph["ties"][i] == j for i in range(3))


def _lines_tt(L, free, wl):
    """Kα1 2θ of the lines of ``L`` at free cell parameters ``free``."""
    cell = _cell_from_ties(free, L["cell"], L["ties"])
    s = 0.5 * wl * np.sqrt(np.einsum("ij,jk,ik->i", L["hkl"], _recip_metric(cell),
                                     L["hkl"]))
    return 2.0 * np.degrees(np.arcsin(np.clip(s, 0, 1))), s < 1


def _cell_gate(L, wl, target_tt, target_h, observed_tt, tol, strain):
    """Candidate score after adapting its cell to the observed peaks.

    Solid solutions and non-stoichiometric phases differ from the nearest
    database entry by an anisotropic strain of up to a few %, which a single
    isotropic scale cannot follow. 1. Isotropic scale scan (0.1 % grid,
    ±``strain``). 2. Lines ≥ 5 % are assigned to the nearest observed peak
    within a window (4, 2, 1, 1 × ``tol``) and the symmetry-free cell
    parameters refined by bounded least squares on 2θ (√I weights; lengths
    ±``strain``, free angles ±1°). The cell with the best :func:`_score` is
    kept, preferring the smaller change. Returns (score, explained,
    consistency, matched target 2θ, free cell parameters)."""
    from scipy.optimize import least_squares
    free0 = np.asarray(L["free0"], float)
    is_len = np.array([_is_length(L, j) for j in range(free0.size)])
    lo = np.where(is_len, free0 * (1 - strain), free0 - 1.0)
    hi = np.where(is_len, free0 * (1 + strain), free0 + 1.0)
    Irel = 100.0 * L["I"] / L["I"].max()

    def score(free):
        tt, ok = _lines_tt(L, free, wl)
        return _score(tt[ok], Irel[ok], target_tt, target_h, observed_tt, tol)

    best, best_free = None, free0
    for sc in np.linspace(1 - strain, 1 + strain, int(round(2 * strain / 0.001)) + 1):
        f = np.where(is_len, free0 * sc, free0)
        res = score(f)
        if best is None or res[0] > best[0] + 1e-9 or (
                abs(res[0] - best[0]) <= 1e-9 and abs(sc - 1) < abs(best_free[0] / free0[0] - 1)):
            best, best_free = res, f
    free = best_free.copy()
    strong = Irel >= 5.0
    for win in (4 * tol, 2 * tol, tol, tol):
        if not observed_tt.size:
            break
        tt, ok = _lines_tt(L, free, wl)
        sel = strong & ok
        if not sel.any():
            break
        dist = np.abs(tt[sel][:, None] - observed_tt[None, :])
        j = np.argmin(dist, axis=1)
        near = dist[np.arange(j.size), j] <= win
        if near.sum() < free.size + 1:
            break
        rows_ = np.flatnonzero(sel)[near]
        obs = observed_tt[j[near]]
        w = np.sqrt(Irel[rows_])

        def resid(f):
            return (_lines_tt(L, f, wl)[0][rows_] - obs) * w
        try:
            free = least_squares(resid, np.clip(free, lo, hi), bounds=(lo, hi),
                                 x_scale="jac").x
        except Exception:
            break
        res = score(free)
        if res[0] > best[0] + 1e-9:
            best, best_free = res, free.copy()
    return (*best[:4], best_free)


def _fixed_bounds(phases, fwhm0, strain):
    wmax = max(1.0, 4.0 * fwhm0 ** 2)
    lo = [-0.5, 0.0, -0.5 * wmax, 1e-6, 0.0]
    hi = [0.5, wmax, 0.5 * wmax, wmax, 1.0]
    for k, ph in enumerate(phases):
        for j, v in enumerate(ph["free0"]):
            dv = strain * v if _is_length(ph, j) else 1.0
            lo.append(v - dv)
            hi.append(v + dv)
        lo += [-0.5, 0.999999 if k == 0 else 0.25]
        hi += [5.0, 1.000001 if k == 0 else 6.0]
    return np.array(lo), np.array(hi)


def _fit_fixed(m_x, m_y, phases, wl, p0, strain, fwhm0, scan_last=True, n_bkg=8):
    """Fit :class:`_FixedModel`; the last phase's starting cell (from
    :func:`_cell_gate`) is first scanned over an isotropic scale of ±0.5 %
    with everything else fixed (a cell a few tenths of a percent off puts
    every line beside its peak, where a local optimiser cannot find it).
    Returns a result dict."""
    from scipy.optimize import least_squares
    wl1, wl2, ratio = wl
    fm = _FixedModel(m_x, m_y, phases, wl1, wl2, ratio, n_bkg)
    step = float(np.median(np.diff(m_x))) if m_x.size > 1 else fwhm0
    fc = fm.thinned(int(np.clip(fwhm0 / (4.0 * max(step, 1e-9)), 1, 10)))
    lo, hi = _fixed_bounds(phases, fwhm0, strain)
    p = np.clip(np.asarray(p0, float), lo, hi)
    if scan_last and phases:
        o = fm.off[-1]
        f0 = p[o:o + len(phases[-1]["free0"])].copy()
        lens = np.array([_is_length(phases[-1], j) for j in range(len(f0))])
        best = (np.inf, p)
        for s_ in np.arange(0.995, 1.005 + 1e-9, 0.0005):
            q = p.copy()
            q[o:o + len(f0)] = np.where(lens, f0 * s_, f0)
            c_ = float(np.sum(fc.residual(np.clip(q, lo, hi)) ** 2))
            if c_ < best[0]:
                best = (c_, q)
        p = np.clip(best[1], lo, hi)
    sol = least_squares(fc.residual, p, bounds=(lo, hi), x_scale="jac", max_nfev=150,
                        diff_step=1e-6)
    sol = least_squares(fm.residual, sol.x, bounds=(lo, hi), x_scale="jac", max_nfev=60,
                        diff_step=1e-6)
    p = sol.x
    yc, bkg, S, A = fm.solve(p)
    r = m_y - yc
    chi2 = float(np.sum(fm.w * r * r))
    n = int(m_y.size)
    fw = float(np.median(caglioti_fwhm(m_x, *p[1:4])))
    dw, es = _serial_correlation(fm.sw * r, int(np.ceil(fw / max(step, 1e-9))))
    n_eff = n / es ** 2
    # the first phase's FWHM scale is pinned to 1 by its bounds: not a parameter
    n_par = len(p) - 1 + int(np.sum(S > 0)) + fm.B.shape[1]
    cells, Bs, wss, areas, wt = [], [], [], [], []
    for k, ph in enumerate(phases):
        c, Bi, ws = fm.phase_params(p, k)
        cells.append(c)
        Bs.append(float(Bi))
        wss.append(float(ws))
        areas.append(float(S[k] * np.trapezoid(A[:, k], m_x)))
        # Hill–Howard: W_k ∝ S_k·M_k·V_k with S per cell of mass M_k, volume V_k
        wt.append(float(S[k] * ph["mass"] * cell_volume(c)))
    at, wtt = sum(areas) or 1.0, sum(wt) or 1.0
    chi2_0 = fm.chi2_null()
    return {"p": p, "chi2": chi2, "n_obs": n, "n_eff": n_eff, "n_par": n_par,
            "Rwp": float(np.sqrt(chi2 / np.sum(fm.w * m_y * m_y))),
            "Rwp_net": float(np.sqrt(chi2 / max(np.sum(fm.w * (m_y - bkg) ** 2), 1e-300))),
            "gof": float(np.sqrt(chi2 / max(n - n_par, 1))), "dw": dw,
            "cells": cells, "B": Bs, "wscale": wss, "scale": S,
            "share": [a / at for a in areas], "wt": [w / wtt for w in wt],
            "bic_null": float(n_eff * np.log(chi2_0 / n) + fm.B.shape[1] * np.log(n_eff)),
            "y_calc": yc, "y_bkg": bkg, "x": m_x, "y_obs": m_y}


def _bic_fixed(f, n_eff):
    return float(n_eff * np.log(f["chi2"] / f["n_obs"]) + f["n_par"] * np.log(n_eff))


def identify_phases(m: Measured, elements: Sequence[str] = (),
                    wavelength=(1.54056, 1.54439, 0.5), tt_range=None,
                    candidates: Optional[Sequence[Dict]] = None,
                    source: str = "auto", max_phases: int = 4,
                    tol: float = 0.15, max_strain: Optional[float] = None,
                    min_snr: float = 5.0, top_k: int = 4, min_new_peaks: int = 2,
                    min_consistency: float = 0.5, delta_bic: float = 10.0,
                    min_share: float = 0.01, prefilter_keep: int = 40,
                    progress=None) -> PhaseIdentification:
    """Identify the crystalline phases in a pattern from its elements alone.

    Sequential search–match with a whole-pattern test at every step:

    1. Peaks: :func:`find_peaks_table` (Kα1 positions, SNR ≥ ``min_snr``).
    2. Candidates: every COD / Materials Project structure made of any subset
       of ``elements`` (:func:`_database_candidates`), plus/or the given
       ``candidates`` (dicts with "id", "structure" and optionally "formula",
       "spacegroup"), with pymatgen kinematic line intensities.
    3. Each step scores every candidate against the peaks not yet explained:
       score = (fraction of unexplained peak height it explains) ×
       (intensity-weighted fraction of its own strong lines that are
       observed), at the cell found by :func:`_cell_gate`: an isotropic
       scale scan, then a bounded least-squares fit of the symmetry-free
       cell parameters to the 2θ of the assigned strong lines, lengths within
       ±``max_strain`` (default 3 %: solid solutions such as substituted
       Ruddlesden–Popper oxides differ anisotropically by 2–3 % from the
       nearest database end member; Dara, Chem. Mater. 2026,
       doi:10.1021/acs.chemmater.5c02820, uses an isotropic 1 %) and angles
       within ±1°. A candidate must explain ≥ ``min_new_peaks``
       unexplained peaks with lines ≥ 5 % and have consistency ≥
       ``min_consistency`` (tolerates moderate preferred orientation).
    4. The ``top_k`` survivors are each refined jointly with the phases
       already accepted in a fixed-structure (Rietveld-type) fit: kinematic
       intensities |F|²·Lp from the database structure, refined cell
       (symmetry-tied, ±(``max_strain`` + 0.5 %)), overall B, per-phase width
       scale, shared displacement and Caglioti profile, scales ≥ 0 and a
       Chebyshev background solved linearly. The one with the lowest BIC is
       accepted only if it lowers BIC (common N_eff = N/τ, τ from the better
       fitting model) by more than ``delta_bic`` (10; Kass & Raftery 1995 —
       with serially correlated, imperfect-model residuals this is a
       conservative heuristic, not a calibrated probability) relative to the fit
       without it — for the first phase, relative to a background-only fit —
       and takes ≥ ``min_share`` of the fitted intensity. Candidates within
       ``delta_bic`` of the accepted one are reported as alternatives: they
       are not distinguished by these data (polymorphs, isostructural
       solid-solution members, …).
    5. Stop when no candidate passes, nothing is left unexplained, or
       ``max_phases`` is reached.
    6. Backward check (a greedy pick cannot otherwise be undone): each
       accepted phase is removed in turn and the rest refitted; a phase whose
       removal does not raise BIC by more than ``delta_bic`` is dropped.

    7. A final joint Pawley fit (free intensities) of the accepted phases
       gives the reported cells and their esds.

    Weight fractions are Hill–Howard estimates W ∝ S·M_cell·V_cell from the
    fixed-structure scales (no preferred orientation, microabsorption or
    amorphous content). Limits stated in ``caveats``: a structure absent from
    the candidate list can be replaced by one with nearby lines (alternatives
    are listed, removal is checked but substitution is not); phases below
    ~1–3 wt %, amorphous material and elements not listed are invisible to
    this search.
    """
    import pandas as pd

    wl1, wl2, ratio = _wl_tuple(wavelength)
    x, _y = _crop(m, tt_range)
    tr = (float(x.min()), float(x.max())) if tt_range is None else tuple(tt_range)
    say = progress or (lambda s: None)
    # Kα1 positions: with a doublet every peak is fitted with its Kα2 companion,
    # otherwise resolved Kα2 lines would count as extra, unexplained peaks
    peaks = find_peaks_table(m, wavelength=(wl1, wl2, ratio) if wl2 else wl1,
                             tt_range=tr, min_snr=min_snr, kalpha2=wl2 is not None)
    obs_tt = peaks["two_theta"].to_numpy(float)
    obs_h = peaks["height"].to_numpy(float)
    say(f"{len(peaks)} peaks found (SNR ≥ {min_snr:g})")

    elements = [e.strip() for e in elements if str(e).strip()]
    rows: List[Dict] = []
    errors: List[str] = []
    failed: List[str] = []
    used = "given"
    if elements:
        say("searching the database …")
        rows, errors, used, failed = _database_candidates(
            obs_tt, obs_h, obs_tt, elements, wl1, tr, tol, len(elements),
            prefilter_keep, 0.03 if max_strain is None else max_strain, source, say)
    if failed:
        # a candidate that failed to download may be the true phase: never
        # report an identification made from an incomplete candidate list
        return PhaseIdentification(
            phases=pd.DataFrame(), fit=None, peaks=peaks, unexplained=peaks, steps=[],
            summary=(f"NO RESULT: the database search was incomplete ({len(failed)} "
                     f"{'query or download' if len(failed) == 1 else 'queries or downloads'}"
                     f" failed after retries, e.g. {failed[0]}). An identification "
                     "from an incomplete candidate list can be wrong, so none is "
                     "reported. Run it again; structures already downloaded are "
                     "cached, so a rerun continues where this one stopped."),
            caveats=[], errors=errors, source=used, complete=False)
    if max_strain is None:
        max_strain = 0.03
    for c in candidates or ():
        st = c["structure"]
        pk = _kinematic_lines(st, wl1, tr)
        if pk is None:
            continue
        try:
            sg = st.get_space_group_info(symprec=0.01)[0]
        except Exception:
            sg = ""
        rows.append({"id": c.get("id") or st.composition.reduced_formula,
                     "formula": c.get("formula") or st.composition.reduced_formula,
                     "spacegroup": c.get("spacegroup") or sg, "title": c.get("title", ""),
                     "peaks": pk, "structure": st})
    rows = [r for r in rows if r.get("structure") is not None and r["peaks"]]
    say(f"{len(rows)} candidate structures")

    accepted: List[Dict] = []   # {"row", "lines", "dbic", "alternatives"}
    fx: Optional[Dict] = None   # current fixed-structure fit
    caveat_extra: List[str] = []
    steps: List[Dict] = []
    remaining = peaks
    xs, ys = _crop(m, tr)
    fwhm0 = _median_fwhm(xs, ys)
    wlt = (wl1, wl2, ratio)
    strain = max_strain + 0.005   # margin beyond the gate cell for the fit
    slines: Dict[str, Optional[Dict]] = {}

    def struct_lines(r):
        if r["id"] not in slines:
            slines[r["id"]] = _structure_lines(r["structure"], wl1, tr,
                                               f"{r['formula']} {r['spacegroup']} ({r['id']})")
        return slines[r["id"]]

    def p_start(prev, free):
        base = list(prev["p"]) if prev else [0.0, 0.0, 0.0, fwhm0 ** 2, 0.5]
        return base + list(free) + [0.5, 1.0]

    def assigned_lines(acc, f):
        """Calculated lines ≥ 0.3 % of each accepted phase at its fitted cell."""
        out = []
        for k, a in enumerate(acc):
            L = a["lines"]
            cell = f["cells"][k] if f else L["cell"]
            inv_d2 = np.einsum("ij,jk,ik->i", L["hkl"], _recip_metric(cell), L["hkl"])
            s = 0.5 * wl1 * np.sqrt(inv_d2)
            keep = (s < 1) & (L["I"] >= 3e-3 * L["I"].max())
            out.append(2 * np.degrees(np.arcsin(np.clip(s[keep], 0, 1))))
        return out

    for step in range(max_phases):
        if remaining.empty:
            break
        t_tt = remaining["two_theta"].to_numpy(float)
        t_h = remaining["height"].to_numpy(float)
        taken = {(a["row"]["formula"], a["row"]["spacegroup"]) for a in accepted}
        scored = []
        for r in rows:
            if (r["formula"], r["spacegroup"]) in taken:
                continue
            L = struct_lines(r)
            if L is None:
                continue
            sc, ex_, co, matched, free = _cell_gate(
                L, wl1, t_tt, t_h, obs_tt, tol, max_strain)
            if len(matched) >= min_new_peaks and co >= min_consistency:
                scored.append((sc, ex_, co, tuple(matched), free, r))
        scored.sort(key=lambda t: -t[0])
        # one entry per compound: same elements, space group and matched
        # peaks (COD lists many entries and non-stoichiometric variants,
        # e.g. MnO, Mn0.94O, Mn0.98O, which would fill all top_k slots)
        seen, short = set(), []
        for t in scored:
            els_ = frozenset(str(e) for e in t[5]["structure"].composition.elements) \
                if t[5].get("structure") is not None else t[5]["formula"]
            key = (els_, t[5]["spacegroup"], tuple(np.round(sorted(t[3]), 2)))
            if key not in seen:
                seen.add(key)
                short.append(t)
            if len(short) == top_k:
                break
        rec = {"step": step + 1, "n_unexplained_before": len(remaining), "tried": []}
        if not short:
            rec["decision"] = ("stop: no candidate explains ≥ "
                               f"{min_new_peaks} of the remaining peaks with its "
                               "strong lines observed")
            steps.append(rec)
            break
        trials = []
        for sc, ex_, co, mt, free, r in short:
            nm = len(mt)
            name = f"{r['formula']} {r['spacegroup']} ({r['id']})"
            say(f"step {step + 1}: testing {name}")
            L = struct_lines(r)
            if L is None:
                rec["tried"].append({"phase": name, "error": "no usable reflections"})
                continue
            acc = accepted + [{"row": r, "lines": L}]
            try:
                f = _fit_fixed(xs, ys, [a["lines"] for a in acc], wlt,
                               p_start(fx, free), strain, fwhm0)
            except Exception as e:
                rec["tried"].append({"phase": name, "error": f"{type(e).__name__}: {e}"[:160]})
                continue
            if fx is None:
                f["dbic"] = _bic_fixed(f, f["n_eff"]) - f["bic_null"]
            else:
                # serial correlation from the better-fitting (larger) model:
                # the misfit of a model still missing a phase inflates τ
                n_eff = max(f["n_eff"], fx["n_eff"])
                f["dbic"] = _bic_fixed(f, n_eff) - _bic_fixed(fx, n_eff)
            trials.append((f, acc, name, sc, ex_, co, nm))
        if not trials:
            rec["decision"] = "stop: every candidate refinement failed"
            steps.append(rec)
            break
        trials.sort(key=lambda t: t[0]["dbic"])
        for f, acc, name, sc, ex_, co, nm in trials:
            rec["tried"].append({"phase": name, "score": round(sc, 3),
                                 "explains": f"{nm} peaks, {100 * ex_:.0f} % of "
                                             "unexplained height",
                                 "strong lines seen": f"{100 * co:.0f} %",
                                 "ΔBIC": round(f["dbic"], 1),
                                 "intensity share": round(f["share"][-1], 3),
                                 "Rwp (%)": round(100 * f["Rwp"], 2)})
        best = trials[0]
        if not (best[0]["dbic"] < -delta_bic and best[0]["share"][-1] >= min_share):
            rec["decision"] = (f"stop: best candidate {best[2]} does not improve the "
                               f"whole-pattern fit enough (ΔBIC {best[0]['dbic']:+.1f}, "
                               f"needs < −{delta_bic:g}; share "
                               f"{100 * best[0]['share'][-1]:.1f} %)")
            steps.append(rec)
            break
        alts = [t[2] for t in trials[1:] if t[0]["dbic"] - best[0]["dbic"] < delta_bic
                and t[0]["dbic"] < -delta_bic]
        fx, accepted = best[0], best[1]
        accepted[-1]["alternatives"] = alts
        accepted[-1]["dbic"] = best[0]["dbic"]
        rec["decision"] = f"accepted {best[2]}" + (
            f"; not distinguishable from: {', '.join(alts)}" if alts else "")
        steps.append(rec)
        remaining = unassigned_peaks(peaks, assigned_lines(accepted, fx), tol)

    # backward check: drop accepted phases the final fit does not need
    while fx is not None and len(accepted) > 1:
        worst = None
        for k in range(len(accepted)):
            sub = accepted[:k] + accepted[k + 1:]
            fm_off = 5 + sum(len(a["lines"]["free0"]) + 2 for a in accepted[:k])
            nk = len(accepted[k]["lines"]["free0"]) + 2
            p_sub = np.concatenate([fx["p"][:fm_off], fx["p"][fm_off + nk:]])
            try:
                f_sub = _fit_fixed(xs, ys, [a["lines"] for a in sub], wlt, p_sub,
                                   strain, fwhm0, scan_last=False)
            except Exception:
                continue
            n_eff = min(f_sub["n_eff"], fx["n_eff"])
            d = _bic_fixed(f_sub, n_eff) - _bic_fixed(fx, n_eff)  # > 0: phase k is needed
            if worst is None or d < worst[0]:
                worst = (d, k, f_sub, sub)
        if worst is None or worst[0] > delta_bic:
            break
        name = accepted[worst[1]]["lines"]["name"]
        steps.append({"step": "backward check", "tried": [],
                      "decision": f"dropped {name}: removing it changes BIC by "
                                  f"{worst[0]:+.1f} (needs > +{delta_bic:g} to keep)"})
        fx, accepted = worst[2], worst[3]

    # final Pawley fit (free intensities) at the screened cells: lattice
    # constants that do not depend on the fixed atomic model's intensity misfit
    fit: Optional[PawleyResult] = None
    if fx is not None:
        specs = []
        for k, a in enumerate(accepted):
            sp = phase_spec(a["lines"]["structure"])
            v0 = cell_volume(a["lines"]["cell"])
            sc_ = (cell_volume(fx["cells"][k]) / v0) ** (1.0 / 3.0)
            a_, b_, c_, al, be, ga = sp["cell"]
            sp["cell"] = (a_ * sc_, b_ * sc_, c_ * sc_, al, be, ga)
            sp["name"] = a["lines"]["name"]
            specs.append(sp)
        try:
            say("final Pawley fit of all accepted phases …")
            main = specs[0]
            fit = pawley_refine(m, model=main["spacegroup"], cell0=main["cell"],
                                Z=main.get("Z"), wavelength=wlt, tt_range=tr,
                                extra_phases=specs[1:], check_systematics=False)
            fit.model = main["name"]
        except Exception as e:
            errors.append(f"final Pawley fit failed: {type(e).__name__}: {e}"[:200])

    # report
    prow = []
    if fx is not None:
        p_cells, p_esd = [None] * len(accepted), [None] * len(accepted)
        if fit is not None:
            p_cells = [fit.cell] + [e["cell"] for e in fit.extra]
            p_esd = [fit.cell_esd] + [tuple(e["esd"].get(n, 0.0) for n in
                                            ("a", "b", "c", "alpha", "beta", "gamma"))
                                      for e in fit.extra]
        for k, a in enumerate(accepted):
            r = a["row"]
            c, esd = p_cells[k], p_esd[k]
            # the free-intensity fit must stay at the screened cell; if it
            # wandered (> 0.5 % in volume) its cell is not used
            if c is None or abs(cell_volume(c) / cell_volume(fx["cells"][k]) - 1) > 5e-3:
                if c is not None:
                    caveat_extra.append(
                        f"{r['formula']}: the free-intensity (Pawley) cell moved "
                        f"{100 * (cell_volume(c) / cell_volume(fx['cells'][k]) - 1):+.1f} % "
                        "in volume away from the fixed-structure fit; the fixed-structure "
                        "cell is reported, without esd.")
                c, esd = fx["cells"][k], (np.nan,) * 6
            prow.append({"phase": r["formula"], "space group": r["spacegroup"],
                         "database id": r["id"],
                         "a (Å)": c[0], "b (Å)": c[1], "c (Å)": c[2],
                         "α": c[3], "β": c[4], "γ": c[5],
                         "σa": esd[0], "σb": esd[1], "σc": esd[2],
                         "V (Å³)": cell_volume(c),
                         "weight fraction (%)": 100 * fx["wt"][k],
                         "intensity share (%)": 100 * fx["share"][k],
                         "B (Å²)": fx["B"][k],
                         "ΔBIC when added": a.get("dbic", np.nan),
                         "not distinguishable from": "; ".join(a.get("alternatives", []))})
    phases = pd.DataFrame(prow)
    unexplained = unassigned_peaks(peaks, assigned_lines(accepted, fx), tol) \
        if accepted else peaks
    tot = float(obs_h.sum()) or 1.0
    expl = 1.0 - float(unexplained["height"].sum()) / tot if len(peaks) else 0.0

    if not len(peaks):
        summary = "No diffraction peaks above the noise were found."
    elif not accepted:
        summary = ("No database phase made of the given elements explains the peaks "
                   "(see the steps). Check the element list, the wavelength and the "
                   "2θ range, or add a structure (CIF) by hand.")
    else:
        names = ", ".join(f"{p['phase']} ({p['space group']}, "
                          f"≈{p['weight fraction (%)']:.0f} wt %)" for p in prow)
        summary = (f"{len(accepted)} phase(s) found: {names}. Together they explain "
                   f"{100 * expl:.0f} % of the peak height "
                   f"({len(peaks) - len(unexplained)} of {len(peaks)} peaks).")
        if len(unexplained):
            summary += (" Unexplained peaks at 2θ = " + ", ".join(
                f"{v:.2f}°" for v in unexplained["two_theta"]) +
                ": another phase not in the database, an element not listed, "
                "or a superstructure.")
        if any(a.get("alternatives") for a in accepted):
            summary += (" Some phases cannot be told apart from close alternatives "
                        "with these data (listed in the table).")
        summary += (f" Fixed-structure whole-pattern fit: Rwp {100 * fx['Rwp']:.1f} %, "
                    f"GoF {fx['gof']:.2f}")
        if fit is not None:
            summary += (f"; free-intensity (Pawley) fit: Rwp {100 * fit.Rwp:.1f} %, "
                        f"GoF {fit.gof:.2f}")
        summary += " (GoF 1 = only counting noise left)."
    caveats = [
        "Weight fractions come from the fixed-structure fit (Hill & Howard 1987: "
        "W ∝ scale × cell mass × cell volume) with the database atomic positions, "
        "one overall B per phase, and no preferred-orientation or microabsorption "
        "correction: treat them as estimates (typically a few wt % off), not a "
        "refined Rietveld analysis.",
        "Phases below about 1–3 wt %, amorphous material and phases containing "
        "elements you did not list cannot be found.",
        "A phase is accepted only if its calculated pattern (positions and "
        "intensities from the database structure) lowers the BIC of the whole-pattern "
        "fit by more than 10; lattice constants and their esds are from the final "
        "free-intensity (Pawley) fit.",
        "Phases with nearly identical patterns (polymorphs, members of a solid "
        "solution) are reported as alternatives, not resolved.",
    ] + caveat_extra
    at_lo = [a["row"]["formula"] for k, a in enumerate(accepted) if fx["B"][k] <= -0.5 + 1e-3]
    if at_lo:
        caveats.append(
            f"{', '.join(at_lo)}: the overall B refined to its lower bound (−0.5 Å²), "
            "i.e. the calculated intensities fall off with angle faster than the "
            "data. Usual causes: the sample is a solid solution or non-stoichiometric "
            "member of the database structure (different atoms and occupancies), "
            "microabsorption or surface roughness, or processed (normalised / "
            "background-subtracted) data. Weight fractions of these phases are "
            "less reliable.")
    return PhaseIdentification(phases=phases, fit=fit, peaks=peaks,
                               unexplained=unexplained, steps=steps, summary=summary,
                               caveats=caveats, errors=errors, source=used,
                               structures={r["id"]: r["structure"] for r in rows},
                               fixed_fit=fx)
