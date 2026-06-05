"""Core constants, lattice geometry, and peak-shape utilities."""
from __future__ import annotations

import numpy as np


# Single-line (Kα-averaged or pure Kα1) wavelengths in Å.
WAVELENGTHS = {
    "Cu Kα": 1.54184,
    "Cu Kα1": 1.54056,
    "Co Kα": 1.78897,
    "Mo Kα": 0.70930,
    "Cr Kα": 2.28970,
    "Synchrotron 1.0000 Å": 1.00000,
}


# Kα1 / Kα2 doublets — (λ_Kα1, λ_Kα2, I_Kα2/I_Kα1).
# Standard 2:1 intensity ratio for Cu / Co / Mo / Cr characteristic X-rays.
WAVELENGTHS_KA12 = {
    "Cu Kα1/Kα2": (1.54056, 1.54439, 0.5),
    "Co Kα1/Kα2": (1.78897, 1.79285, 0.5),
    "Mo Kα1/Kα2": (0.70930, 0.71359, 0.5),
    "Cr Kα1/Kα2": (2.28970, 2.29361, 0.5),
}


PALETTE = {
    "Pt":           "#0072B2",
    "Ir":           "#D55E00",
    "Ti (α-hcp)":   "#009E73",
    "TiO₂ rutile":  "#CC79A7",
    "TiO₂ anatase": "#E69F00",
}


MATERIALS = {
    "Pt": {
        "system": "cubic", "a": 3.9242, "ref": "PDF 04-0802",
        "spacegroup": "Fm-3m",
        "hkl_I": [((1,1,1),100.0), ((2,0,0),53.0), ((2,2,0),31.0),
                  ((3,1,1),33.0), ((2,2,2),12.0), ((4,0,0),6.0), ((3,3,1),11.0)],
    },
    "Ir": {
        "system": "cubic", "a": 3.8394, "ref": "PDF 06-0598",
        "spacegroup": "Fm-3m",
        "hkl_I": [((1,1,1),100.0), ((2,0,0),47.0), ((2,2,0),27.0),
                  ((3,1,1),29.0), ((2,2,2),9.0), ((4,0,0),4.0), ((3,3,1),8.0)],
    },
    "Ti (α-hcp)": {
        "system": "hcp", "a": 2.9508, "c": 4.6855, "ref": "PDF 44-1294",
        "spacegroup": "P6_3/mmc",
        "hkl_I": [((1,0,0),30.0), ((0,0,2),26.0), ((1,0,1),100.0),
                  ((1,0,2),19.0), ((1,1,0),17.0), ((1,0,3),17.0),
                  ((1,1,2),18.0), ((2,0,1),15.0), ((2,0,2),4.0), ((1,0,4),10.0)],
    },
    "TiO₂ rutile": {
        "system": "tetragonal", "a": 4.5933, "c": 2.9592, "ref": "PDF 21-1276",
        "spacegroup": "P4_2/mnm",
        "hkl_I": [((1,1,0),100.0), ((1,0,1),50.0), ((2,0,0),8.0),
                  ((1,1,1),25.0), ((2,1,0),10.0), ((2,1,1),60.0),
                  ((2,2,0),20.0), ((0,0,2),10.0), ((3,1,0),10.0),
                  ((3,0,1),20.0), ((1,1,2),12.0)],
    },
    "TiO₂ anatase": {
        "system": "tetragonal", "a": 3.7842, "c": 9.5146, "ref": "PDF 21-1272",
        "spacegroup": "I4_1/amd",
        "hkl_I": [((1,0,1),100.0), ((1,0,3),10.0), ((0,0,4),20.0),
                  ((1,1,2),10.0), ((2,0,0),35.0), ((1,0,5),20.0),
                  ((2,1,1),20.0), ((2,0,4),14.0), ((1,1,6),6.0),
                  ((2,2,0),6.0), ((2,1,5),10.0)],
    },
}


def d_spacing(material: dict, hkl) -> float:
    """Interplanar spacing d_hkl in Å for a (h,k,l) reflection."""
    h, k, l = hkl
    a = material["a"]
    sys = material["system"]
    if sys == "cubic":
        return a / np.sqrt(h*h + k*k + l*l)
    if sys == "hcp":
        c = material["c"]
        return 1.0 / np.sqrt((4.0/3.0)*(h*h + h*k + k*k)/(a*a) + (l*l)/(c*c))
    if sys == "tetragonal":
        c = material["c"]
        return 1.0 / np.sqrt((h*h + k*k)/(a*a) + (l*l)/(c*c))
    raise ValueError(f"unsupported crystal system: {sys}")


def two_theta_peaks(material: dict, wavelength_A: float):
    """Return [(2θ_deg, I_rel, hkl), …] for a card-style material at a given λ."""
    out = []
    for hkl, I in material["hkl_I"]:
        d = d_spacing(material, hkl)
        ratio = wavelength_A / (2.0 * d)
        if abs(ratio) >= 1.0:
            continue
        out.append((2.0 * np.degrees(np.arcsin(ratio)), I, hkl))
    return out


def pseudo_voigt(x, x0, fwhm, eta: float = 0.5):
    """Pseudo-Voigt peak: η·Lorentzian + (1−η)·Gaussian, same FWHM."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = 0.5 * fwhm
    g = np.exp(-((x - x0) ** 2) / (2.0 * sigma * sigma))
    l = 1.0 / (1.0 + ((x - x0) / gamma) ** 2)
    return eta * l + (1.0 - eta) * g


def scherrer_fwhm_deg(two_theta_deg: float, wavelength_A: float,
                       crystallite_nm: float, K: float = 0.9) -> float:
    """Scherrer FWHM (degrees) for crystallite size D (nm) at a given 2θ."""
    theta = np.radians(0.5 * two_theta_deg)
    D_A = crystallite_nm * 10.0
    return np.degrees(K * wavelength_A / (D_A * np.cos(theta)))


# Vector-PDF rcParams: closed box, inward ticks, Helvetica/Arial fallback,
# 600 dpi, editable text (fonttype 42). Suitable for journal figures.
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "savefig.bbox": "tight",
}
