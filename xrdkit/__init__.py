"""xrdkit — interactive XRD reference patterns from ICDD cards, CIF files,
the Crystallography Open Database, the Materials Project, plus your own
measured or computed diffractograms.
"""
from xrdkit.core import (
    WAVELENGTHS, WAVELENGTHS_KA12, MATERIALS, NATURE_RC,
    d_spacing, two_theta_peaks, pseudo_voigt, scherrer_fwhm_deg,
)
from xrdkit.db import (
    Phase, builtin_phases, from_cif, from_structure_file,
    from_mp_id, from_cod_id, search_mp, search_cod, assign_color,
)
from xrdkit.plot import (
    Measured, load_measured, make_figure, save_pdf, save_gif, peaks_table,
    compute_difference, make_difference_plot, calculate_r_factors,
)

__version__ = "0.1.0"
__all__ = [
    "WAVELENGTHS", "WAVELENGTHS_KA12", "MATERIALS", "NATURE_RC",
    "d_spacing", "two_theta_peaks", "pseudo_voigt", "scherrer_fwhm_deg",
    "Phase", "builtin_phases",
    "from_cif", "from_structure_file", "from_mp_id", "from_cod_id",
    "search_mp", "search_cod", "assign_color",
    "Measured", "load_measured",
    "make_figure", "save_pdf", "save_gif", "peaks_table",
    "compute_difference", "make_difference_plot", "calculate_r_factors",
]
