"""Plotting and measured-data loading."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MultipleLocator

from xrdkit.core import NATURE_RC, pseudo_voigt, scherrer_fwhm_deg
from xrdkit.db import Phase


@dataclass
class Measured:
    name: str
    two_theta: np.ndarray
    intensity: np.ndarray
    color: str = "#000000"
    source: str = "measured"
    info: dict = field(default_factory=dict)


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _read_xrdml(p):
    """PANalytical .xrdml → (two_theta, intensity)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(p).getroot()
    start = end = step = None
    intensities = None
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "startPosition" and start is None:
            start = float(el.text)
        elif tag == "endPosition" and end is None:
            end = float(el.text)
        elif tag == "stepSize" and step is None:
            step = float(el.text)
        elif tag in ("intensities", "counts") and intensities is None:
            intensities = np.fromstring(el.text, sep=" ")
    if intensities is None or start is None or end is None:
        raise ValueError(f"could not parse .xrdml: {p}")
    if step is None:
        step = (end - start) / max(len(intensities) - 1, 1)
    return start + step * np.arange(len(intensities)), intensities


def _read_brml(p):
    """Bruker .brml (a ZIP of XML) → (two_theta, intensity).

    Best-effort: pulls intensities from <Datum> rows and reconstructs 2θ from
    the TwoTheta scan-axis Start/Increment when present, else from a 2θ column.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        cand = [n for n in names
                if n.split("/")[-1].lower().startswith("rawdata")
                and n.lower().endswith(".xml")]
        if not cand:
            cand = [n for n in names if n.lower().endswith(".xml")]
        if not cand:
            raise ValueError(f"no XML data found inside .brml: {p}")
        root = ET.fromstring(z.read(sorted(cand)[0]))

    def _local(el):
        return el.tag.split("}")[-1]

    def _val(el):
        txt = (el.text or "").strip() or el.get("Value") or ""
        try:
            return float(txt)
        except (TypeError, ValueError):
            return None

    start = incr = None
    for el in root.iter():
        if _local(el) == "ScanAxisInfo":
            axis = (el.get("AxisName") or "").lower()
            if "theta" not in axis:
                continue
            for ch in el:
                ln = _local(ch)
                if ln == "Start":
                    start = _val(ch)
                elif ln == "Increment":
                    incr = _val(ch)

    rows = []
    for el in root.iter():
        if _local(el) == "Datum" and el.text:
            nums = [float(x) for x in el.text.split(",")
                    if _NUM_RE.fullmatch(x.strip() or "x")]
            if nums:
                rows.append(nums)
    if not rows:
        raise ValueError(f"no <Datum> intensities found in .brml: {p}")

    intensity = np.array([r[-1] for r in rows], dtype=float)
    if start is not None and incr is not None:
        two_theta = start + incr * np.arange(len(intensity))
    elif len(rows[0]) >= 2:
        # fall back to the widest-spanning column as 2θ
        cols = np.array([r for r in rows if len(r) == len(rows[0])], dtype=float)
        spans = cols.max(axis=0) - cols.min(axis=0)
        two_theta = cols[:, int(np.argmax(spans[:-1]))]
        intensity = cols[:, -1]
    else:
        raise ValueError(f"could not reconstruct 2θ axis from .brml: {p}")
    return two_theta, intensity


def _read_ascii(p, delimiter=None, skiprows: int = 0):
    """Robust 2-column ASCII reader.

    Tolerates arbitrary metadata header lines (e.g. Bruker `'Id: ...` banners),
    blank lines, comment markers, and any of space/tab/comma/semicolon
    delimiters — every line that does not start with two parseable numbers is
    skipped automatically, so `skiprows` is rarely needed.
    """
    xs, ys = [], []
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for line in lines[skiprows:]:
        s = line.strip()
        if not s or s[0] in "#!%;'\"*/":
            continue
        nums = _NUM_RE.findall(s.replace(",", " "))
        if len(nums) < 2:
            continue
        try:
            x, y = float(nums[0]), float(nums[1])
        except ValueError:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        raise ValueError(f"{p}: found no 2-column numeric data (2θ, intensity)")
    return np.array(xs), np.array(ys)


def load_measured(path, name=None, color="#000000", delimiter=None,
                  skiprows: int = 0, scale: float = 1.0,
                  two_theta_offset: float = 0.0) -> Measured:
    """Read a measured XRD pattern from (almost) any lab's export.

    The format is detected from the file's *content*, not its extension, so a
    mislabelled or unfamiliar file still loads:

    * ZIP container (Bruker ``.brml``)        → :func:`_read_brml`
    * XML (PANalytical ``.xrdml``, others)    → :func:`_read_xrdml`
    * anything else                           → universal 2-column ASCII reader
      that extracts the first two numbers of every line and ignores arbitrary
      instrument headers, comments, and delimiters.
    """
    p = Path(path)
    with open(p, "rb") as fh:
        head = fh.read(8)

    if head[:4] == b"PK\x03\x04":                       # ZIP → Bruker .brml
        two_theta, y = _read_brml(p)
    elif head.lstrip()[:1] in (b"<",):                   # XML (xrdml etc.)
        try:
            two_theta, y = _read_xrdml(p)
        except Exception:
            two_theta, y = _read_brml(p)                 # some XML brml variants
    else:                                                # any ASCII table
        try:
            two_theta, y = _read_ascii(p, delimiter=delimiter, skiprows=skiprows)
        except ValueError:
            if p.suffix.lower() == ".xrdml":
                two_theta, y = _read_xrdml(p)
            else:
                raise
    two_theta = two_theta.astype(float) + two_theta_offset
    y = y.astype(float) * scale
    return Measured(name=name or p.stem, two_theta=two_theta, intensity=y,
                    color=color, source=f"file: {p.name}",
                    info={"path": str(p), "n_points": len(two_theta),
                          "tt_range": (float(two_theta.min()),
                                        float(two_theta.max()))})


def _compute_pattern_single(grid, peaks, wavelength_A, crystallite_nm, eta):
    y = np.zeros_like(grid)
    for tt, I, _ in peaks:
        fwhm = scherrer_fwhm_deg(tt, wavelength_A, crystallite_nm)
        y += I * pseudo_voigt(grid, tt, fwhm, eta)
    return y


def _compute_pattern(grid, peaks, wavelengths, crystallite_nm, eta):
    """Sum the pattern over Kα1 (+ optional Kα2) for the given peak list.

    `wavelengths` is a list of (λ_Å, weight) tuples. Peak positions in `peaks`
    were computed at the first (Kα1) wavelength; for the doublet we shift
    those by Δ2θ = -2·(d_λ/λ_1)·tan(θ) for each secondary λ.
    """
    if len(wavelengths) == 1:
        wl, w = wavelengths[0]
        return w * _compute_pattern_single(grid, peaks, wl, crystallite_nm, eta)
    wl1 = wavelengths[0][0]
    y = np.zeros_like(grid)
    for wl, weight in wavelengths:
        scale = wl / wl1
        shifted = []
        for tt, I, hkl in peaks:
            sin_t = np.sin(np.radians(0.5 * tt)) * scale
            if abs(sin_t) >= 1.0:
                continue
            shifted.append((2.0 * np.degrees(np.arcsin(sin_t)), I, hkl))
        y += weight * _compute_pattern_single(grid, shifted, wl, crystallite_nm, eta)
    return y


def _prepare(phases, wavelengths, crystallite_nm, tt_range, eta):
    grid = np.linspace(tt_range[0], tt_range[1], 6000)
    wl_primary = wavelengths[0][0]
    traces = []
    for ph in phases:
        peaks = ph.peaks(wl_primary, tt_range)
        if not peaks:
            traces.append({"phase": ph, "peaks": [], "y": np.zeros_like(grid),
                           "max_I": 1.0})
            continue
        y = _compute_pattern(grid, peaks, wavelengths, crystallite_nm, eta)
        ymax = y.max() if y.max() > 0 else 1.0
        traces.append({
            "phase": ph, "peaks": peaks, "y": y / ymax,
            "max_I": max(I for _, I, _ in peaks),
        })
    return grid, traces


def _style_axes(ax, tt_range, ymax):
    ax.set_xlim(*tt_range)
    ax.set_ylim(-0.05, ymax)
    ax.set_xlabel(r"2$\theta$ (degrees)")
    ax.set_ylabel("Normalised intensity (a.u.)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())


def _hkl_str(hkl) -> str:
    h, k, l = hkl
    def _f(x):
        return f"\\overline{{{abs(x)}}}" if x < 0 else f"{x}"
    return f"({_f(h)}{_f(k)}{_f(l)})"


def make_figure(
    phases: List[Phase],
    wavelength_A,
    measured: Optional[List[Measured]] = None,
    crystallite_nm: float = 30.0,
    eta: float = 0.5,
    tt_range: Tuple[float, float] = (20.0, 90.0),
    show_sticks: bool = True,
    show_profile: bool = True,
    show_hkl_labels: bool = False,
    hkl_threshold: float = 10.0,
    kalpha12: bool = False,
    kalpha2_ratio: float = 0.5,
    bkg_slope: float = 0.0,
    stack: bool = True,
    offset_step: float = 1.15,
    figsize=None,
):
    """Render the reference + measured figure.

    Parameters
    ----------
    phases : list of Phase
        Selected reference phases.
    wavelength_A : float or tuple
        Either a single Å value (Kα average) or, for Kα1/Kα2 mode, a
        ``(λ_Kα1, λ_Kα2)`` tuple in Å. When ``kalpha12=True`` and a single λ
        is passed, λ is used as Kα1 and λ_Kα2 = λ + 0.004 Å as a default.
    measured : list of Measured, optional
        Overlay traces (already-2θ-vs-I curves).
    show_hkl_labels : bool
        Print (h k l) just above each peak with I ≥ ``hkl_threshold`` %.
    kalpha12 : bool
        Add the Kα2 satellite at ``kalpha2_ratio`` × Kα1 intensity.
    bkg_slope : float
        Add a linear baseline of slope * (2θ − tt_min)/(tt_max − tt_min).
    """
    plt.rcParams.update(NATURE_RC)
    measured = list(measured or [])

    if isinstance(wavelength_A, (tuple, list)):
        wl1, wl2 = float(wavelength_A[0]), float(wavelength_A[1])
        wavelengths = [(wl1, 1.0), (wl2, kalpha2_ratio)] if kalpha12 else [(wl1, 1.0)]
    else:
        wl1 = float(wavelength_A)
        wavelengths = [(wl1, 1.0)]
        if kalpha12:
            wavelengths.append((wl1 + 0.004, kalpha2_ratio))

    n_p = len(phases)
    n_m = len(measured)
    n_total = n_p + n_m
    if figsize is None:
        figsize = (3.8, max(3.0, 0.55 * max(n_total, 1) + 1.6))
    fig, ax = plt.subplots(figsize=figsize)
    grid, traces = _prepare(phases, wavelengths, crystallite_nm, tt_range, eta)

    bkg = np.zeros_like(grid)
    if bkg_slope > 0:
        bkg = bkg_slope * (grid - tt_range[0]) / max(tt_range[1] - tt_range[0], 1e-9)

    for i, t in enumerate(traces):
        ph = t["phase"]
        offset = i * offset_step if stack else 0.0
        y_show = t["y"] + bkg
        y_show = y_show / max(y_show.max(), 1e-9) if y_show.max() > 0 else y_show

        if show_profile:
            ax.fill_between(grid, offset, y_show + offset,
                            color=ph.color, alpha=0.10, linewidth=0)
            ax.plot(grid, y_show + offset, color=ph.color, linewidth=1.2)
        if show_sticks and t["peaks"]:
            for tt, I, _ in t["peaks"]:
                h = I / t["max_I"]
                ax.plot([tt, tt], [offset, offset + 0.95 * h],
                        color=ph.color, linewidth=0.6, alpha=0.55,
                        solid_capstyle="butt")
        if show_hkl_labels and t["peaks"]:
            for tt, I, hkl in t["peaks"]:
                rel = I / t["max_I"] * 100.0
                if rel < hkl_threshold:
                    continue
                ax.text(tt, offset + 0.95 * (I / t["max_I"]) + 0.06,
                        f"${_hkl_str(hkl)}$",
                        color=ph.color, ha="center", va="bottom",
                        fontsize=6, rotation=90)
        if stack:
            ax.text(tt_range[1] - 1.0, offset + 0.92, ph.name,
                    color=ph.color, ha="right", va="center", fontsize=8)

    for j, m in enumerate(measured):
        offset = (n_p + j) * offset_step if stack else 0.0
        mask = (m.two_theta >= tt_range[0]) & (m.two_theta <= tt_range[1])
        x = m.two_theta[mask]
        y = m.intensity[mask].astype(float)
        if y.size and y.max() > 0:
            y = y / y.max()
        ax.plot(x, y + offset, color=m.color, linewidth=0.9, label=m.name)
        if stack:
            ax.text(tt_range[1] - 1.0, offset + 0.92, m.name, color=m.color,
                    ha="right", va="center", fontsize=8, fontstyle="italic")

    ymax = (n_total - 1) * offset_step + 1.30 if stack and n_total else 1.20
    _style_axes(ax, tt_range, ymax)
    if not stack:
        for t in traces:
            ax.plot([], [], color=t["phase"].color, linewidth=1.2,
                    label=t["phase"].name)
        for m in measured:
            ax.plot([], [], color=m.color, linewidth=0.9, label=m.name)
        ax.legend(loc="upper right", frameon=False, handlelength=1.4,
                  labelspacing=0.4)
    fig.tight_layout()
    return fig, ax


def save_pdf(path, **kwargs):
    fig, _ = make_figure(**kwargs)
    fig.savefig(path)
    plt.close(fig)


def save_gif(
    path,
    phases: List[Phase],
    wavelength_A,
    measured: Optional[List[Measured]] = None,
    crystallite_nm: float = 30.0,
    eta: float = 0.5,
    tt_range=(20.0, 90.0),
    figsize=None,
    fps: int = 14,
    fade_frames: int = 18,
    hold_frames: int = 14,
    tail_frames: int = 28,
    offset_step: float = 1.15,
    show_sticks: bool = True,
    show_hkl_labels: bool = False,
    hkl_threshold: float = 10.0,
    kalpha12: bool = False,
    kalpha2_ratio: float = 0.5,
):
    plt.rcParams.update(NATURE_RC)
    measured = list(measured or [])

    if isinstance(wavelength_A, (tuple, list)):
        wl1, wl2 = float(wavelength_A[0]), float(wavelength_A[1])
        wavelengths = [(wl1, 1.0), (wl2, kalpha2_ratio)] if kalpha12 else [(wl1, 1.0)]
    else:
        wl1 = float(wavelength_A)
        wavelengths = [(wl1, 1.0)]
        if kalpha12:
            wavelengths.append((wl1 + 0.004, kalpha2_ratio))

    n_p = len(phases)
    n_m = len(measured)
    n_total = n_p + n_m
    if figsize is None:
        figsize = (4.6, max(3.5, 0.6 * max(n_total, 1) + 2.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    grid, traces = _prepare(phases, wavelengths, crystallite_nm, tt_range, eta)
    ymax = (n_total - 1) * offset_step + 1.30 if n_total else 1.20
    _style_axes(ax, tt_range, ymax)

    profile_lines, stick_groups, labels, fills, hkl_texts = [], [], [], [], []
    for i, t in enumerate(traces):
        offset = i * offset_step
        ph = t["phase"]
        base = np.full_like(grid, offset)
        line, = ax.plot(grid, base, color=ph.color, linewidth=1.2, alpha=0.0)
        fill = ax.fill_between(grid, offset, offset, color=ph.color, alpha=0.0, linewidth=0)
        sl = []
        if show_sticks:
            for tt, I, _ in t["peaks"]:
                h = I / t["max_I"]
                ln, = ax.plot([tt, tt], [offset, offset], color=ph.color,
                              linewidth=0.6, alpha=0.0, solid_capstyle="butt")
                sl.append((ln, h))
        labels_for_phase = []
        if show_hkl_labels:
            for tt, I, hkl in t["peaks"]:
                rel = I / t["max_I"] * 100.0
                if rel < hkl_threshold:
                    continue
                txt = ax.text(tt, offset + 0.95 * (I / t["max_I"]) + 0.06,
                               f"${_hkl_str(hkl)}$",
                               color=ph.color, ha="center", va="bottom",
                               fontsize=6, rotation=90, alpha=0.0)
                labels_for_phase.append(txt)
        lbl = ax.text(tt_range[1] - 1.0, offset + 0.92, ph.name,
                      color=ph.color, ha="right", va="center",
                      fontsize=8, alpha=0.0)
        profile_lines.append(line)
        fills.append(fill)
        stick_groups.append(sl)
        labels.append(lbl)
        hkl_texts.append(labels_for_phase)

    meas_lines, meas_labels, meas_data = [], [], []
    for j, m in enumerate(measured):
        offset = (n_p + j) * offset_step
        mask = (m.two_theta >= tt_range[0]) & (m.two_theta <= tt_range[1])
        x = m.two_theta[mask]
        y = m.intensity[mask].astype(float)
        if y.size and y.max() > 0:
            y = y / y.max()
        line, = ax.plot(x, np.full_like(x, offset), color=m.color,
                        linewidth=0.9, alpha=0.0)
        lbl = ax.text(tt_range[1] - 1.0, offset + 0.92, m.name, color=m.color,
                      ha="right", va="center", fontsize=8,
                      fontstyle="italic", alpha=0.0)
        meas_lines.append(line)
        meas_labels.append(lbl)
        meas_data.append((x, y, offset))

    fig.tight_layout()

    total = n_total * (fade_frames + hold_frames) + tail_frames

    def update(frame):
        for i, t in enumerate(traces):
            ph = t["phase"]
            offset = i * offset_step
            start = i * (fade_frames + hold_frames)
            local = frame - start
            if local < 0:
                a = 0.0
            elif local < fade_frames:
                a = 0.5 - 0.5 * np.cos(np.pi * local / fade_frames)
            else:
                a = 1.0
            y_anim = offset + a * t["y"]
            profile_lines[i].set_ydata(y_anim)
            profile_lines[i].set_alpha(a)
            fills[i].remove()
            fills[i] = ax.fill_between(grid, offset, y_anim, color=ph.color,
                                       alpha=0.10 * a, linewidth=0)
            for ln, h in stick_groups[i]:
                ln.set_ydata([offset, offset + a * 0.95 * h])
                ln.set_alpha(0.55 * a)
            for txt in hkl_texts[i]:
                txt.set_alpha(a)
            labels[i].set_alpha(a)
        for j, (x, y, offset) in enumerate(meas_data):
            start = (n_p + j) * (fade_frames + hold_frames)
            local = frame - start
            if local < 0:
                a = 0.0
            elif local < fade_frames:
                a = 0.5 - 0.5 * np.cos(np.pi * local / fade_frames)
            else:
                a = 1.0
            meas_lines[j].set_ydata(offset + a * y)
            meas_lines[j].set_alpha(a)
            meas_labels[j].set_alpha(a)
        return []

    anim = animation.FuncAnimation(fig, update, frames=total,
                                   interval=1000 / fps, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=160)
    plt.close(fig)


def peaks_table(phases, wavelength_A, tt_range=(5.0, 130.0)):
    """Return a list of dicts: phase, 2θ, d, I (rel), hkl, source. Useful for
    rendering a peak-list table beneath the figure."""
    wl = float(wavelength_A[0]) if isinstance(wavelength_A, (tuple, list)) else float(wavelength_A)
    rows = []
    for ph in phases:
        peaks = ph.peaks(wl, tt_range)
        if not peaks:
            continue
        max_I = max(I for _, I, _ in peaks)
        for tt, I, hkl in peaks:
            theta = np.radians(tt / 2.0)
            d = wl / (2.0 * np.sin(theta)) if np.sin(theta) > 0 else float("nan")
            rows.append({
                "Phase":  ph.name,
                "2θ (°)": round(tt, 3),
                "d (Å)":  round(d, 4),
                "I (%)":  round(100.0 * I / max_I, 1),
                "hkl":    " ".join(map(str, hkl)),
                "Source": ph.source,
            })
    rows.sort(key=lambda r: (r["Phase"], r["2θ (°)"]))
    return rows


# ===========================================================================
#  Observed − simulated difference  (Rietveld-style obs/calc/diff figure)
#
#  Pipeline (the defensible order, per Toby 2006 / IUCr practice):
#    1. interpolate observed onto a common 2θ grid
#    2. estimate + subtract background (SNIP, Ryan et al. 1988)
#    3. zero-shift alignment of calculated vs observed (cross-correlation)
#    4. weighted non-negative least-squares scaling of the *calculated* phase
#       profiles to the *observed* counts — NOT max-normalisation
#    5. difference = y_obs − Σ_k s_k · p_k
#
#  Because xrdkit performs kinematic *reference overlay* (fixed lattice/HKL
#  intensities), not a full structural refinement, the reported residuals are
#  pattern-similarity figures of merit, NOT Rietveld goodness-of-fit. There is
#  deliberately no χ²/R_exp — those require counting statistics and a refined
#  parameter count that do not exist here.
# ===========================================================================


def _snip_background(y, n_iter: int = 24):
    """SNIP lower-envelope background (Ryan et al., NIMB 1988).

    Works in the log-log-sqrt (LLS) domain so the estimate hugs the baseline
    without biting into Bragg peaks. ``n_iter`` sets the maximum clipping
    window (larger → smoother, flatter background)."""
    yy = np.clip(np.asarray(y, float), 0.0, None)
    v = np.log(np.log(np.sqrt(yy + 1.0) + 1.0) + 1.0)
    for p in range(1, int(n_iter) + 1):
        lo = np.r_[np.full(p, v[0]), v[:-p]]
        hi = np.r_[v[p:], np.full(p, v[-1])]
        v = np.minimum(v, 0.5 * (lo + hi))
    b = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0
    return np.clip(b, 0.0, None)


def _best_zero_shift(grid, obs, calc, max_shift: float = 0.75, n: int = 151):
    """Return the 2θ zero-shift (°) that maximises cosine overlap of calc→obs."""
    if obs.size == 0 or calc.size == 0 or calc.max() <= 0:
        return 0.0
    on = obs / (np.linalg.norm(obs) + 1e-12)
    best_score, best_s = -np.inf, 0.0
    for s in np.linspace(-max_shift, max_shift, n):
        c = np.interp(grid, grid + s, calc, left=0.0, right=0.0)
        nc = np.linalg.norm(c)
        if nc <= 0:
            continue
        score = float(on @ (c / nc))
        if score > best_score:
            best_score, best_s = score, s
    return float(best_s)


def _fit_scales(obs, basis):
    """Non-negative least-squares phase scales: min ‖obs − Σ s_k p_k‖², s_k ≥ 0.

    Unit weights are used deliberately. The phase profiles are background-free,
    so the denominator Σ p_k² is dominated by the Bragg peaks (p² is tiny in the
    tails); 1/y_obs Poisson weights would instead hand authority to the baseline
    and collapse the scale to zero. Falls back to a closed-form single scale if
    SciPy is unavailable."""
    A = np.vstack(basis).T
    try:
        from scipy.optimize import nnls
        scales, _ = nnls(A, obs)
        return np.asarray(scales, float)
    except Exception:
        s_sum = basis[0].copy()
        for b in basis[1:]:
            s_sum = s_sum + b
        num = float(np.sum(obs * s_sum))
        den = float(np.sum(s_sum * s_sum)) + 1e-12
        s = max(num / den, 0.0)
        return np.full(len(basis), s / max(len(basis), 1))


def compute_difference(measured, phases, wavelength_A,
                       crystallite_nm: float = 30.0, eta: float = 0.5,
                       tt_range: Tuple[float, float] = (20.0, 90.0),
                       kalpha12: bool = False, kalpha2_ratio: float = 0.5,
                       subtract_background: bool = True, bkg_window_deg: float = 3.0,
                       align: bool = True, n_grid: int = 6000):
    """Background-subtract, align, and least-squares-scale a simulated multi-phase
    pattern to a measured one, returning everything needed to draw the residual.

    Parameters
    ----------
    measured : Measured
        The observed diffractogram.
    phases : list of Phase
        Reference phases forming the calculated pattern (fit as a basis set).
    subtract_background : bool
        Estimate and remove a SNIP background from the observed counts.
    align : bool
        Apply a small zero-shift (±0.4°) to best register calc onto observed.

    Returns
    -------
    dict with: grid, valid (in-range mask), obs (bkg-subtracted), obs_raw, bkg,
    calc, diff, scales, names, colors, peaks_per_phase, exp_name, shift,
    plus the residual figures of merit (Rp, Rwp, overlap).
    """
    if isinstance(wavelength_A, (tuple, list)):
        wl1, wl2 = float(wavelength_A[0]), float(wavelength_A[1])
        wavelengths = [(wl1, 1.0), (wl2, kalpha2_ratio)] if kalpha12 else [(wl1, 1.0)]
    else:
        wl1 = float(wavelength_A)
        wavelengths = [(wl1, 1.0)]
        if kalpha12:
            wavelengths.append((wl1 + 0.004, kalpha2_ratio))

    grid = np.linspace(tt_range[0], tt_range[1], int(n_grid))

    # --- calculated phase basis (each profile normalised to its own max) ------
    _, traces = _prepare(phases, wavelengths, crystallite_nm, tt_range, eta)
    keep = [t for t in traces if t["y"].max() > 0]
    basis = [t["y"].copy() for t in keep]
    names = [t["phase"].name for t in keep]
    colors = [t["phase"].color for t in keep]
    peaks_per_phase = [t["peaks"] for t in keep]

    # --- observed onto the common grid ---------------------------------------
    mask = (measured.two_theta >= tt_range[0]) & (measured.two_theta <= tt_range[1])
    xm = measured.two_theta[mask].astype(float)
    ym = measured.intensity[mask].astype(float)
    if xm.size >= 2:
        order = np.argsort(xm)
        xm, ym = xm[order], ym[order]
        obs_raw = np.interp(grid, xm, ym)
        valid = (grid >= xm.min()) & (grid <= xm.max())
    else:
        obs_raw = np.zeros_like(grid)
        valid = np.zeros_like(grid, dtype=bool)

    # SNIP clipping window must be wider than the Bragg peaks, else it bites
    # into them; size it from a 2θ window rather than a raw iteration count.
    dx = (tt_range[1] - tt_range[0]) / max(int(n_grid) - 1, 1)
    snip_iter = max(int(round(bkg_window_deg / max(dx, 1e-9))), 8)
    bkg = _snip_background(obs_raw, snip_iter) if subtract_background else np.zeros_like(obs_raw)
    obs = np.clip(obs_raw - bkg, 0.0, None)

    # --- alignment + non-negative least-squares scaling ----------------------
    shift = 0.0
    scales = np.zeros(len(basis))
    if basis:
        calc_sum0 = np.sum(basis, axis=0)
        if align:
            shift = _best_zero_shift(grid[valid], obs[valid], calc_sum0[valid]) if valid.any() \
                else _best_zero_shift(grid, obs, calc_sum0)
            basis = [np.interp(grid, grid + shift, b, left=0.0, right=0.0) for b in basis]
        idx = valid if valid.any() else np.ones_like(grid, dtype=bool)
        scales = _fit_scales(obs[idx], [b[idx] for b in basis])
        calc = np.zeros_like(grid)
        for s, b in zip(scales, basis):
            calc = calc + s * b
    else:
        calc = np.zeros_like(grid)

    diff = obs - calc
    out = {
        "grid": grid, "valid": valid,
        "obs": obs, "obs_raw": obs_raw, "bkg": bkg,
        "calc": calc, "diff": diff,
        "scales": scales, "names": names, "colors": colors,
        "peaks_per_phase": peaks_per_phase,
        "exp_name": measured.name, "phases": names, "shift": shift,
    }
    out.update(calculate_r_factors(out))
    return out


def calculate_r_factors(difference_data):
    """Pattern-similarity figures of merit on the background-subtracted, scaled
    profiles. These are profile residuals — *not* a Rietveld χ² goodness-of-fit.

    Rp   — profile R-factor,        100·Σ|y_o−y_c| / Σ|y_o|
    Rwp  — weighted profile R-factor, 100·√(Σw(y_o−y_c)² / Σw·y_o²),  w = 1/max(y_o,1)
    overlap — cosine similarity ⟨y_o,y_c⟩ / (‖y_o‖‖y_c‖), in [0, 1].
    """
    valid = difference_data.get("valid")
    obs = difference_data["obs"]
    calc = difference_data["calc"]
    if valid is not None and np.asarray(valid).any():
        obs = obs[valid]
        calc = calc[valid]
    diff = obs - calc

    sum_obs = float(np.sum(np.abs(obs)))
    Rp = 100.0 * float(np.sum(np.abs(diff))) / sum_obs if sum_obs > 0 else float("inf")

    w = 1.0 / np.maximum(obs, 1.0)
    den = float(np.sum(w * obs * obs))
    Rwp = 100.0 * np.sqrt(float(np.sum(w * diff * diff)) / den) if den > 0 else float("inf")

    no, nc = float(np.linalg.norm(obs)), float(np.linalg.norm(calc))
    overlap = float(obs @ calc) / (no * nc) if no > 0 and nc > 0 else 0.0

    def _r(x):
        return round(x, 2) if np.isfinite(x) else None
    return {"Rp": _r(Rp), "Rwp": _r(Rwp), "overlap": round(overlap, 3)}


def make_difference_plot(difference_data, tt_range=None, figsize=None,
                         obs_color="#111111", calc_color="#C8102E",
                         diff_color="#2E7D5B", marker_every: int = 14,
                         show_rfactors: bool = True):
    """Render the observed/calculated/difference figure in the xrdkit house style.

    Three stacked, x-shared panels: (top) observed open circles + calculated
    line; (middle) colour-matched Bragg tick rows, one per phase; (bottom) the
    flat residual about zero. Returns ``(fig, (ax_top, ax_ticks, ax_diff))``.
    """
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update(NATURE_RC)
    grid = difference_data["grid"]
    valid = difference_data.get("valid")
    if valid is None:
        valid = np.ones_like(grid, dtype=bool)
    obs = difference_data["obs"]
    calc = difference_data["calc"]
    diff = difference_data["diff"]
    names = difference_data["names"]
    colors = difference_data["colors"]
    peaks_per_phase = difference_data["peaks_per_phase"]
    if tt_range is None:
        tt_range = (float(grid[0]), float(grid[-1]))
    n_ph = max(len(names), 1)

    # display normalisation: scale the whole figure by the observed maximum
    norm = float(np.nanmax(np.where(valid, obs, np.nan))) if valid.any() else float(obs.max())
    norm = norm if norm and norm > 0 else 1.0
    o = np.where(valid, obs / norm, np.nan)
    c = np.where(valid, calc / norm, np.nan)
    d = np.where(valid, diff / norm, np.nan)

    if figsize is None:
        figsize = (3.8, 3.7)
    tick_h = 0.14 * n_ph + 0.10
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[3.4, tick_h, 1.05], hspace=0.07)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1], sharex=ax)
    axd = fig.add_subplot(gs[2], sharex=ax)

    # ---- top: observed (markers) + calculated (line) ------------------------
    ax.plot(grid[::marker_every], o[::marker_every], linestyle="none",
            marker="o", mfc="none", mec=obs_color, ms=2.6, mew=0.5,
            label="Observed")
    ax.plot(grid, c, color=calc_color, linewidth=1.1, label="Calculated")
    ax.set_ylim(-0.06, 1.18)
    ax.set_ylabel("Normalised intensity (a.u.)")
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.legend(loc="upper right", frameon=False, handlelength=1.3,
              labelspacing=0.3, borderaxespad=0.3)
    if show_rfactors:
        rp = difference_data.get("Rp")
        ov = difference_data.get("overlap")
        bits = []
        if ov is not None:  bits.append(rf"overlap$={ov:.3f}$")
        if rp is not None:  bits.append(rf"$R_p={rp:.1f}\%$")
        if bits:
            ax.text(0.02, 0.96, "\n".join(bits), transform=ax.transAxes,
                    ha="left", va="top", fontsize=6.5, color="#333333")
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---- middle: Bragg tick rows, one per phase -----------------------------
    for r, (pk, col, nm) in enumerate(zip(peaks_per_phase, colors, names)):
        yc = (n_ph - 1 - r)
        for tt, _I, _hkl in pk:
            if tt_range[0] <= tt <= tt_range[1]:
                axt.vlines(tt, yc + 0.18, yc + 0.82, color=col, linewidth=0.6)
        axt.text(tt_range[1] - 0.6, yc + 0.5, nm, color=col, ha="right",
                 va="center", fontsize=6.0)
    axt.set_ylim(-0.15, n_ph)
    axt.set_yticks([])
    axt.tick_params(axis="x", which="both", length=0)
    for sp in axt.spines.values():
        sp.set_visible(False)
    plt.setp(axt.get_xticklabels(), visible=False)

    # ---- bottom: residual ----------------------------------------------------
    axd.axhline(0.0, color="#9A9A9A", linewidth=0.6, zorder=0)
    axd.plot(grid, d, color=diff_color, linewidth=0.8)
    finite = d[np.isfinite(d)]
    amp = float(np.percentile(np.abs(finite), 99)) if finite.size else 0.1
    amp = max(amp, 0.04) * 1.5
    axd.set_ylim(-amp, amp)
    axd.set_ylabel("Obs − Calc")
    axd.yaxis.set_major_locator(plt.NullLocator())
    axd.set_xlabel(r"2$\theta$ (degrees)")
    axd.xaxis.set_major_locator(MultipleLocator(10))
    axd.xaxis.set_minor_locator(MultipleLocator(2))

    for a in (ax, axt, axd):
        a.set_xlim(*tt_range)
    fig.align_ylabels([ax, axd])
    return fig, (ax, axt, axd)
