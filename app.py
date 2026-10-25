"""Streamlit GUI for xrdkit."""
from __future__ import annotations

import io
import os
import sys

# one BLAS thread: hosted containers get 1-2 CPUs but OpenBLAS sizes its pool to
# the host's cores, and the oversubscribed pool slows every QR/matmul many-fold
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
try:
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)
except ImportError:
    pass
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# A redeploy reruns this script in a live interpreter that still holds the
# previously imported xrdkit modules; reload them when their source changed.
import importlib
import xrdkit.core
import xrdkit.plot
import xrdkit.db
import xrdkit.analysis

_XRDKIT_MODS = (xrdkit.core, xrdkit.plot, xrdkit.db, xrdkit.analysis)  # dependency order
_mtimes = tuple(os.path.getmtime(m_.__file__) for m_ in _XRDKIT_MODS)
if getattr(xrdkit, "_loaded_mtimes", _mtimes) != _mtimes:
    for m_ in _XRDKIT_MODS:
        importlib.reload(m_)
xrdkit._loaded_mtimes = _mtimes

from xrdkit.core import WAVELENGTHS, WAVELENGTHS_KA12, NATURE_RC
from xrdkit.db import (Phase, builtin_phases, from_structure_file, from_mp_id,
                        from_cod_id, search_mp, search_cod, assign_color)
from xrdkit.plot import (Measured, load_measured, load_measured_excel,
                          make_figure, save_gif,
                          peaks_table, compute_difference, make_difference_plot,
                          make_zoom_figure, auto_zoom_windows)
from xrdkit import analysis as xa


st.set_page_config(page_title="xrdkit",
                   page_icon="◬",
                   layout="wide", initial_sidebar_state="expanded")
plt.rcParams.update(NATURE_RC)


# ----- session state -----
if "library" not in st.session_state:
    st.session_state.library = builtin_phases()
    st.session_state.custom = {}
    st.session_state.measured = {}
    st.session_state.used_colors = [ph.color for ph in st.session_state.library.values()]
    st.session_state.cod_rows = []
    st.session_state.mp_rows = []
    st.session_state.processed_upload_ids = set()

LIB = st.session_state.library
CUSTOM = st.session_state.custom
MEASURED = st.session_state.measured
USED = st.session_state.used_colors


def _unique(name, taken):
    if name not in taken:
        return name
    k = 2
    while f"{name} #{k}" in taken:
        k += 1
    return f"{name} #{k}"


# ============ SIDEBAR ============
with st.sidebar:
    st.header("Reference cards")
    st.caption("Built-in reference phases (line lists and COD structures). Tick to add.")
    cols = st.columns(2)
    if cols[0].button("Show all", use_container_width=True):
        for n in LIB: st.session_state[f"bi_{n}"] = True
        st.rerun()
    if cols[1].button("Clear all", use_container_width=True):
        for n in LIB: st.session_state[f"bi_{n}"] = False
        st.rerun()
    builtin_on = {}
    for name, ph in LIB.items():
        ref = ph.info.get("ref", "")
        sg  = ph.info.get("spacegroup", "")
        label = f"{name}  ·  {ref}"
        if sg:
            label += f"  ·  {sg}"
        builtin_on[name] = st.checkbox(label, value=False, key=f"bi_{name}")

    if CUSTOM:
        st.markdown("**Loaded structures**")
        custom_on = {}
        for name in list(CUSTOM):
            cols = st.columns([5, 1])
            custom_on[name] = cols[0].checkbox(
                f"{name}  ·  {CUSTOM[name].source}", value=True, key=f"cu_{name}")
            if cols[1].button("✕", key=f"rm_cu_{name}"):
                del CUSTOM[name]; st.rerun()
    else:
        custom_on = {}

    if MEASURED:
        st.markdown("**Loaded measured / computed patterns**")
        meas_on = {}
        for name in list(MEASURED):
            cols = st.columns([5, 1])
            meas_on[name] = cols[0].checkbox(
                f"{name}  ·  {MEASURED[name].source}", value=True, key=f"me_{name}")
            if cols[1].button("✕", key=f"rm_me_{name}"):
                del MEASURED[name]; st.rerun()
    else:
        meas_on = {}

    st.divider()
    st.header("Plot settings")

    use_doublet = st.checkbox("Kα1 / Kα2 doublet", value=True,
                              help="render the Kα2 satellite at 0.5×Kα1 intensity")
    if use_doublet:
        wl_name = st.selectbox("Wavelength (Kα1 / Kα2)",
                                list(WAVELENGTHS_KA12.keys()), index=0)
        wl_value = WAVELENGTHS_KA12[wl_name][:2]
        ka2_ratio = WAVELENGTHS_KA12[wl_name][2]
    else:
        wl_name = st.selectbox("Wavelength", list(WAVELENGTHS.keys()), index=0)
        wl_value = WAVELENGTHS[wl_name]
        ka2_ratio = 0.5

    D_nm = st.slider("Crystallite size D (nm)", 2.0, 200.0, 30.0, step=1.0,
                     help="controls Scherrer FWHM: smaller D → broader peaks")
    tt_lo, tt_hi = st.slider("2θ range (°)", 5.0, 140.0, (20.0, 90.0), step=1.0)
    eta_val = st.slider("η  (pseudo-Voigt mixing)", 0.0, 1.0, 0.5, step=0.05,
                        help="0 = pure Gaussian, 1 = pure Lorentzian")
    bkg_slope = st.slider("Background slope", 0.0, 1.0, 0.0, step=0.05,
                          help="adds a linear baseline rising from 2θ_min to 2θ_max")
    layout_mode = st.radio("Layout", ["stacked", "overlay"], horizontal=True)
    cols = st.columns(2)
    y_scale = cols[0].selectbox("Intensity axis", ["linear", "sqrt", "log"],
                                help="√I and log reveal weak and impurity peaks")
    normalise = cols[1].selectbox(
        "Normalise", ["max", "background"],
        help="'background' subtracts a SNIP background before scaling to 1")
    cols = st.columns(2)
    show_profile = cols[0].checkbox("smooth profile", value=True)
    show_sticks = cols[1].checkbox("sticks (hkl)", value=True)
    cols = st.columns(2)
    show_hkl_labels = cols[0].checkbox("(hkl) labels on peaks", value=False)
    hkl_thr = cols[1].slider("label threshold (% I)", 1.0, 50.0, 10.0, step=1.0,
                              disabled=not show_hkl_labels)


# ============ MAIN AREA ============
_logo = HERE / "logo" / "logo_xrdkit_white.jpeg"
if _logo.exists():
    st.image(str(_logo), width=320)
st.title("XRDKit - powder-XRD pattern composer")
st.caption("Build XRD reference patterns for any crystalline material from built-in "
           "reference phases, CIF/POSCAR files, the COD, or the Materials Project. Overlay "
           "measured data and compute the observed-simulated difference.")

tab_struct, tab_cod, tab_mp, tab_meas = st.tabs([
    "Upload structure (CIF / POSCAR)",
    "Search COD",
    "Search Materials Project",
    "Upload measured / computed XRD",
])


# --- Upload structure ---
with tab_struct:
    st.write("Drop one or more **CIF**, **POSCAR / CONTCAR (.vasp)**, **.json**, "
             "**.xyz**, or **.xsf** files. The kinematic powder pattern is "
             "computed by pymatgen.")
    up_struct = st.file_uploader(
        "structure file(s)", type=None, accept_multiple_files=True,
        key="upload_struct",
    )
    if up_struct:
        for f in up_struct:
            fid = ("struct", f.name, f.size)
            if fid in st.session_state.processed_upload_ids:
                continue
            try:
                suffix = Path(f.name).suffix or ".cif"
                if suffix.lower() in (".xy", ".xye", ".brml", ".xrdml",
                                      ".dat", ".tsv", ".csv", ".txt", ".ras"):
                    raise ValueError(
                        f"`{suffix}` looks like a measured/computed diffractogram, "
                        "not a crystal structure. Upload it in the "
                        "**Upload measurement** tab instead.")
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(f.getvalue()); tmp_path = tmp.name
                color = assign_color(USED); USED.append(color)
                ph = from_structure_file(tmp_path, name=Path(f.name).stem, color=color)
                ph_name = _unique(ph.name, set(CUSTOM) | set(LIB))
                if ph_name != ph.name:
                    ph = Phase(name=ph_name, color=ph.color, source=ph.source,
                               peaks_fn=ph.peaks_fn, info=ph.info)
                CUSTOM[ph_name] = ph
                st.session_state.processed_upload_ids.add(fid)
                st.success(f"loaded **{ph_name}**  ·  spacegroup "
                           f"{ph.info.get('spacegroup','?')}  ·  "
                           f"{ph.info.get('n_sites','?')} sites")
            except Exception as e:
                st.error(f"`{f.name}`: {type(e).__name__}: {e}")


# --- COD search ---
with tab_cod:
    st.write("Crystallography Open Database — free, no API key. "
             "Type a formula (e.g. `IrO2`, `Co3O4`, `MoS2`), search, pick, add.")
    cols = st.columns([3, 1])
    cod_formula = cols[0].text_input("Formula", placeholder="IrO2",
                                     key="cod_formula")
    if cols[1].button("Search COD", use_container_width=True):
        if cod_formula.strip():
            try:
                with st.spinner("querying COD…"):
                    st.session_state.cod_rows = search_cod(cod_formula.strip(),
                                                            max_results=25)
            except Exception as e:
                st.error(f"COD search failed: {type(e).__name__}: {e}")
                st.session_state.cod_rows = []
    rows = st.session_state.cod_rows
    if rows:
        labels = [f"{r['cod_id']:>10} │ {r['formula']:<14} │ "
                  f"{(r['spacegroup'] or '?'):<14} │ {(r['title'] or '')[:50]}"
                  for r in rows]
        pick = st.selectbox(f"Results ({len(rows)})",
                            options=list(range(len(rows))),
                            format_func=lambda i: labels[i], key="cod_pick")
        if st.button("Add selected to plot", key="cod_add"):
            try:
                with st.spinner("fetching CIF…"):
                    color = assign_color(USED); USED.append(color)
                    ph = from_cod_id(rows[pick]["cod_id"], color=color)
                    ph_name = _unique(ph.name, set(CUSTOM) | set(LIB))
                    if ph_name != ph.name:
                        ph = Phase(name=ph_name, color=ph.color, source=ph.source,
                                   peaks_fn=ph.peaks_fn, info=ph.info)
                    CUSTOM[ph_name] = ph
                st.success(f"added **{ph_name}**")
                st.rerun()
            except Exception as e:
                st.error(f"COD fetch failed: {type(e).__name__}: {e}")


# --- MP search ---
with tab_mp:
    st.write("Materials Project — DFT-relaxed structures. Free key from "
             "[materialsproject.org/api](https://materialsproject.org/api). "
             "Note: DFT lattice parameters can differ from experimental by "
             "1–2 %, shifting peaks by 0.1–0.3°.")
    mp_key_val = st.text_input("MP API key", type="password",
                                value=os.environ.get("MP_API_KEY", ""),
                                key="mp_key_in")
    cols = st.columns([3, 1])
    mp_formula = cols[0].text_input("Formula", placeholder="Pt",
                                    key="mp_formula")
    if cols[1].button("Search MP", use_container_width=True):
        if not mp_key_val:
            st.error("paste your Materials Project API key first")
        elif mp_formula.strip():
            try:
                with st.spinner("querying Materials Project…"):
                    st.session_state.mp_rows = search_mp(
                        mp_formula.strip(), api_key=mp_key_val, max_results=15)
            except Exception as e:
                st.error(f"MP search failed: {type(e).__name__}: {e}")
                st.session_state.mp_rows = []
    rows = st.session_state.mp_rows
    if rows:
        labels = []
        for r in rows:
            eh = r["e_above_hull"]
            tail = f"  Eₕ={eh:.3f} eV/atom" if eh is not None else ""
            labels.append(f"{r['mp_id']:>10} │ {r['formula']:<10} │ "
                          f"{r['spacegroup']:<14}{tail}")
        pick = st.selectbox(f"Results ({len(rows)})",
                            options=list(range(len(rows))),
                            format_func=lambda i: labels[i], key="mp_pick")
        if st.button("Add selected to plot", key="mp_add"):
            try:
                with st.spinner("fetching structure…"):
                    color = assign_color(USED); USED.append(color)
                    ph = from_mp_id(rows[pick]["mp_id"], api_key=mp_key_val,
                                    color=color)
                    ph_name = _unique(ph.name, set(CUSTOM) | set(LIB))
                    if ph_name != ph.name:
                        ph = Phase(name=ph_name, color=ph.color, source=ph.source,
                                   peaks_fn=ph.peaks_fn, info=ph.info)
                    CUSTOM[ph_name] = ph
                st.success(f"added **{ph_name}**")
                st.rerun()
            except Exception as e:
                st.error(f"MP fetch failed: {type(e).__name__}: {e}")


# --- Measured upload ---
with tab_meas:
    st.write("Drop your measured or computed XRD pattern. "
             "Two-column ASCII (`.xy`, `.csv`, `.txt`, `.dat`, `.tsv` — "
             "instrument header lines are skipped automatically), PANalytical "
             "`.xrdml`, Bruker `.brml`, or Excel `.xlsx` (one pattern per sheet: "
             "2θ in column A, intensity in column B). Use offset / scale if needed.")
    cols = st.columns(4)
    meas_offset = cols[0].number_input("2θ offset (°)", value=0.0, step=0.01,
                                        format="%.3f", key="meas_offset")
    meas_scale = cols[1].number_input("intensity scale", value=1.0, step=0.1,
                                       format="%.3f", key="meas_scale")
    meas_skip = cols[2].number_input("header rows to skip", value=0, step=1,
                                      min_value=0, key="meas_skip")
    meas_color = cols[3].color_picker("colour", value="#000000", key="meas_color")
    up_meas = st.file_uploader(
        "measured / computed XRD file(s)", type=None, accept_multiple_files=True,
        key="upload_meas",
    )
    if up_meas:
        for f in up_meas:
            fid = ("meas", f.name, f.size)
            if fid in st.session_state.processed_upload_ids:
                continue
            try:
                suffix = Path(f.name).suffix or ".xy"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(f.getvalue()); tmp_path = tmp.name
                if suffix.lower() in (".xlsx", ".xlsm"):
                    loaded = load_measured_excel(
                        tmp_path, color=meas_color, skiprows=int(meas_skip),
                        scale=float(meas_scale),
                        two_theta_offset=float(meas_offset))
                else:
                    loaded = [load_measured(tmp_path, name=Path(f.name).stem,
                                            color=meas_color,
                                            skiprows=int(meas_skip),
                                            scale=float(meas_scale),
                                            two_theta_offset=float(meas_offset))]
                for m in loaded:
                    taken = list(USED) + [mm.color for mm in MEASURED.values()]
                    if m.color in taken:
                        m.color = assign_color(taken)
                    m_name = _unique(m.name, set(MEASURED))
                    if m_name != m.name:
                        m = Measured(name=m_name, two_theta=m.two_theta,
                                     intensity=m.intensity, color=m.color,
                                     source=m.source, info=m.info)
                    MEASURED[m_name] = m
                    st.success(f"loaded **{m_name}**  ·  {m.info['n_points']} pts  ·  "
                               f"2θ {m.info['tt_range'][0]:.1f}–{m.info['tt_range'][1]:.1f}°")
                    for w in m.info.get("warnings", []):
                        st.warning(f"{m_name}: {w}")
                    fwl = m.info.get("wavelength")
                    if fwl is not None:
                        side = float(np.atleast_1d(wl_value)[0])
                        msg = (f"{m_name}: file records Kα1 = {fwl[0]:.5f} Å"
                               + (f", Kα2 = {fwl[1]:.5f} Å" if fwl[1] else "")
                               + (f", Kα2/Kα1 = {fwl[2]:.3f}" if fwl[2] else ""))
                        if abs(fwl[0] - side) > 1e-4:
                            st.warning(msg + f"; the sidebar wavelength is "
                                       f"{side:.5f} Å — select the matching tube")
                        else:
                            st.caption(msg)
                st.session_state.processed_upload_ids.add(fid)
            except Exception as e:
                st.error(f"`{f.name}`: {type(e).__name__}: {e}")


# ============ LIVE PLOT ============
st.divider()

phases = [LIB[n] for n in LIB if builtin_on.get(n, False)]
phases += [CUSTOM[n] for n in CUSTOM if custom_on.get(n, True)]
measured = [MEASURED[n] for n in MEASURED if meas_on.get(n, True)]

if not phases and not measured:
    st.info("Pick a built-in phase on the left, upload a structure or measurement, "
            "or search the COD / Materials Project to begin.")
else:
    fig, _ = make_figure(
        phases=phases, measured=measured,
        wavelength_A=wl_value,
        crystallite_nm=D_nm, eta=eta_val,
        tt_range=(tt_lo, tt_hi),
        show_profile=show_profile, show_sticks=show_sticks,
        show_hkl_labels=show_hkl_labels, hkl_threshold=hkl_thr,
        kalpha12=use_doublet, kalpha2_ratio=ka2_ratio,
        bkg_slope=bkg_slope,
        stack=(layout_mode == "stacked"),
        y_scale=y_scale, normalise=normalise,
    )

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(fig, use_container_width=True)

    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf")
    plt.close(fig)
    pdf_buf.seek(0)

    with right:
        st.subheader("Export")
        st.download_button(
            "Download PDF (vector)",
            pdf_buf.getvalue(),
            file_name="xrdkit_pattern.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        gen_gif = st.button("Render animated GIF", use_container_width=True)
        if gen_gif:
            try:
                with st.spinner("rendering GIF…"):
                    tmp_gif = Path(tempfile.gettempdir()) / "xrdkit_pattern.gif"
                    save_gif(tmp_gif, phases=phases, measured=measured,
                             wavelength_A=wl_value,
                             crystallite_nm=D_nm, eta=eta_val,
                             tt_range=(tt_lo, tt_hi),
                             show_sticks=show_sticks,
                             show_hkl_labels=show_hkl_labels,
                             hkl_threshold=hkl_thr,
                             kalpha12=use_doublet, kalpha2_ratio=ka2_ratio)
                    st.session_state.last_gif = tmp_gif.read_bytes()
            except Exception as e:
                st.error(f"GIF render failed: {type(e).__name__}: {e}")
        if "last_gif" in st.session_state:
            st.download_button(
                "Download GIF",
                st.session_state.last_gif,
                file_name="xrdkit_pattern.gif",
                mime="image/gif",
                use_container_width=True,
            )

        st.subheader("Active in plot")
        for ph in phases:
            extra = []
            if ph.info.get("spacegroup"): extra.append(ph.info["spacegroup"])
            if ph.info.get("a") is not None:
                a = ph.info["a"]; c = ph.info.get("c")
                extra.append(f"a={a:.3f}Å" + (f", c={c:.3f}Å" if c else ""))
            tail = "  ·  ".join(extra)
            st.markdown(
                f"<span style='color:{ph.color}'>●</span> "
                f"**{ph.name}** <span style='color:#888;font-size:11px'>"
                f"({ph.source}{'  ·  ' + tail if tail else ''})</span>",
                unsafe_allow_html=True,
            )
        for m in measured:
            st.markdown(
                f"<span style='color:{m.color}'>●</span> "
                f"*{m.name}* <span style='color:#888;font-size:11px'>"
                f"({m.source})</span>",
                unsafe_allow_html=True,
            )

    # ----- peak list table -----
    if phases:
        st.subheader("Peak list")
        rows = peaks_table(phases, wavelength_A=wl_value,
                            tt_range=(tt_lo, tt_hi))
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download peak list (CSV)", csv_bytes,
                                file_name="xrdkit_peaks.csv", mime="text/csv")


# ============ OBSERVED − SIMULATED DIFFERENCE ============
st.divider()
st.header("Observed − simulated difference")
st.caption("The Rietveld-style observed / calculated / difference figure. The "
           "selected reference phases are background-subtracted (SNIP), "
           "zero-shift aligned, and scaled to the measured pattern by "
           "non-negative least squares — then the residual is drawn below. "
           "Reported R_p and overlap are pattern-similarity scores for kinematic "
           "reference overlay, not a Rietveld goodness-of-fit.")

if not measured:
    st.info("Upload a measured / computed pattern (tab above) and tick at least "
            "one reference phase to enable the difference figure.")
elif not phases:
    st.info("Tick at least one reference phase (left) to calculate against the "
            "measured pattern.")
else:
    dcols = st.columns([2, 1, 1])
    obs_name = dcols[0].selectbox(
        "Observed pattern", [m.name for m in measured], key="diff_obs")
    do_bkg = dcols[1].checkbox("subtract background", value=True, key="diff_bkg")
    do_align = dcols[2].checkbox("zero-shift align", value=True, key="diff_align")
    obs_meas = next(m for m in measured if m.name == obs_name)

    diff_data = compute_difference(
        obs_meas, phases, wavelength_A=wl_value,
        crystallite_nm=D_nm, eta=eta_val, tt_range=(tt_lo, tt_hi),
        kalpha12=use_doublet, kalpha2_ratio=ka2_ratio,
        subtract_background=do_bkg, align=do_align,
    )

    if not diff_data["names"]:
        st.warning("None of the selected phases produce peaks in this 2θ range.")
    else:
        dfig, _ = make_difference_plot(diff_data, tt_range=(tt_lo, tt_hi))
        dleft, dright = st.columns([3, 2])
        with dleft:
            st.pyplot(dfig, use_container_width=True)

        dpdf = io.BytesIO()
        dfig.savefig(dpdf, format="pdf")
        plt.close(dfig)
        dpdf.seek(0)

        with dright:
            st.subheader("Fit summary")
            ov = diff_data.get("overlap")
            rp = diff_data.get("Rp")
            rwp = diff_data.get("Rwp")
            mc = st.columns(2)
            mc[0].metric("overlap (cosine)", f"{ov:.3f}" if ov is not None else "—")
            mc[1].metric("R_p", f"{rp:.1f} %" if rp is not None else "—")
            st.caption(f"weighted R_wp′ = {rwp:.1f} %  ·  zero-shift = "
                       f"{diff_data['shift']:+.2f}°" if rwp is not None
                       else f"zero-shift = {diff_data['shift']:+.2f}°")
            st.markdown("**Phase scale factors**")
            for nm, col, s in zip(diff_data["names"], diff_data["colors"],
                                  diff_data["scales"]):
                st.markdown(
                    f"<span style='color:{col}'>●</span> {nm} — "
                    f"<code>{float(s):.1f}</code>", unsafe_allow_html=True)
            st.download_button(
                "Download difference figure (PDF)", dpdf.getvalue(),
                file_name="xrdkit_difference.pdf", mime="application/pdf",
                use_container_width=True)


# ============ ANALYSIS OF MEASURED PATTERNS ============
st.divider()
st.header("Analysis of measured patterns")
st.caption("Peak fitting, Kα2 stripping, Pawley cell refinement with space-group "
           "model comparison, Williamson–Hall, zoom panels and phase search for "
           "unassigned peaks. The wavelength is the one chosen in the sidebar "
           "(Kα1/Kα2 when the doublet box is ticked).")

an_by = {m.name: m for m in measured
         if np.sum((m.two_theta >= tt_lo) & (m.two_theta <= tt_hi)) >= 10}
an_names = list(an_by)
_short = [m.name for m in measured if m.name not in an_by]
if _short:
    st.warning("Fewer than 10 points in the 2θ range, not analysed: " + ", ".join(_short))
if not measured:
    st.info("Upload a measured pattern (tab above) to enable the analysis tools.")
elif an_by:
    if use_doublet:
        an_wl = (float(wl_value[0]), float(wl_value[1]), float(ka2_ratio))
    else:
        an_wl = (float(wl_value), None, 0.5)
    an_wl1 = an_wl[0]
    st.session_state.setdefault("peak_tables", {})
    st.session_state.setdefault("pawley", {})

    def _sig(name):  # data + settings a cached result depends on
        m_ = an_by[name]
        return (name, hash(m_.two_theta.tobytes()), hash(m_.intensity.tobytes()),
                an_wl, tt_lo, tt_hi)

    (t_ident, t_peaks, t_ka2, t_pawley, t_wh, t_zoom, t_match) = st.tabs([
        "Identify phases", "Peak table", "Kα2 stripping", "Cell refinement (Pawley)",
        "Williamson–Hall", "Zoom panels", "Unassigned peaks → database"])

    def _peaks(name, snr):
        key = (_sig(name), snr)
        if key not in st.session_state.peak_tables:
            st.session_state.peak_tables[key] = xa.find_peaks_table(
                an_by[name], wavelength=an_wl, tt_range=(tt_lo, tt_hi),
                min_snr=snr, kalpha2=an_wl[1] is not None)
        return st.session_state.peak_tables[key]

    with t_ident:
        st.markdown(
            "**Start here.** Enter the elements in your sample. The tool finds the "
            "peaks, takes every structure made of those elements from the "
            "Crystallography Open Database (or Materials Project if COD is "
            "unreachable), and adds phases one at a time: at each step the "
            "candidates whose calculated lines best match the still-unexplained "
            "peaks are refined together with the phases already found, and one is "
            "kept only if the whole-pattern fit becomes clearly better "
            "(ΔBIC < −10). Phases the final fit does not need are removed again. "
            "Works for any material; nothing here assumes a perovskite.")
        c = st.columns([2, 2, 1])
        id_name = c[0].selectbox("Pattern", an_names, key="id_name")
        id_els = c[1].text_input("Elements in the sample", key="id_els",
                                 placeholder="e.g. La, Sr, Mn, Ni, O",
                                 help="every element that can be in a crystalline "
                                      "phase; a phase with an element not listed "
                                      "cannot be found")
        id_src = c[2].selectbox("Database", ["auto", "cod", "mp"], key="id_src",
                                help="auto: COD, falling back to Materials Project")
        with st.expander("Options"):
            c = st.columns(4)
            id_max = c[0].number_input("max phases", 1, 8, 4, key="id_max")
            id_tol = c[1].number_input("peak match tolerance (°)", 0.05, 0.5, 0.15,
                                       step=0.01, key="id_tol")
            id_snr = c[2].number_input("min peak S/N", 2.0, 50.0, 5.0, step=1.0,
                                       key="id_snr")
            id_lib = c[3].checkbox("also try my reference phases", True, key="id_lib",
                                   help="library / uploaded phases that have a "
                                        "structure are added to the candidates")
        id_list = [e for e in id_els.replace(" ", "").replace(";", ",").split(",") if e]
        st.session_state.setdefault("phase_id", {})
        id_key = (_sig(id_name), tuple(id_list), id_src, id_max, id_tol, id_snr, id_lib)
        if st.button("Identify phases", key="id_run", type="primary",
                     disabled=not id_list,
                     help=None if id_list else "enter the elements first"):
            extra_c = []
            if id_lib:
                for n_, p_ in {**LIB, **CUSTOM}.items():
                    stc = p_.info.get("structure")
                    if stc is not None and {str(e) for e in stc.composition.elements} <= set(id_list):
                        extra_c.append({"id": n_, "structure": stc})
            msg = st.empty()
            with st.spinner("searching and refining (one to a few minutes)…"):
                try:
                    st.session_state.phase_id[id_key] = xa.identify_phases(
                        an_by[id_name], id_list, wavelength=an_wl,
                        tt_range=(tt_lo, tt_hi), candidates=extra_c, source=id_src,
                        max_phases=int(id_max), tol=id_tol, min_snr=id_snr,
                        progress=lambda t: msg.caption(t))
                except Exception as e:
                    st.error(f"Identification failed: {type(e).__name__}: {e}")
            msg.empty()
        pid = st.session_state.phase_id.get(id_key)
        if pid is not None:
            (st.success if len(pid.phases) and not len(pid.unexplained)
             else st.info if len(pid.phases) else st.warning)(pid.summary)
            if len(pid.phases):
                show = pid.phases.copy()
                for col in ("a (Å)", "b (Å)", "c (Å)", "V (Å³)"):
                    show[col] = show[col].round(4)
                for col in ("σa", "σb", "σc"):
                    show[col] = show[col].map(lambda v: f"{v:.1e}")
                for col in ("weight fraction (%)", "intensity share (%)"):
                    show[col] = show[col].round(1)
                show["B (Å²)"] = show["B (Å²)"].round(2)
                show["ΔBIC when added"] = show["ΔBIC when added"].round(0)
                st.dataframe(show, hide_index=True, use_container_width=True)
                f = pid.fit
                if f is None:
                    st.warning("The free-intensity (Pawley) fit failed; cells, weight "
                               "fractions and B above are from the fixed-structure fit "
                               "only, without esds.")
            if len(pid.phases) and pid.fit is not None:
                ifig = xa.make_pawley_plot(f, figsize=(7.0, 4.2))
                st.pyplot(ifig, use_container_width=True)
                plt.close(ifig)
                st.markdown(
                    "**How to read the fit numbers.** "
                    f"*Rwp* = {100 * f.Rwp:.1f} %: misfit relative to the total "
                    "intensity (it looks small when the background is high). "
                    f"*Rwp (net)* = {100 * f.Rwp_net:.1f} %: the same after removing "
                    "the background — the honest measure of how well the peaks are "
                    f"described. *GoF* = {f.gof:.2f}: 1 means the misfit is pure "
                    "counting noise; 1–2 is typical of good lab data; above 3 "
                    "something is missing (a phase, peak asymmetry, a wrong cell). "
                    f"*Durbin–Watson* = {f.durbin_watson:.2f}: 2 means the residual "
                    "is random noise; values near 0 mean neighbouring points are "
                    "wrong together, i.e. a systematic misfit — the cell esds are "
                    f"then multiplied by {f.esd_scale:.1f}. σ = 1σ statistical "
                    "uncertainty only (no zero-shift or instrument error).")
            if len(pid.phases):
                if len(pid.unexplained):
                    st.markdown("**Peaks no phase explains**")
                    st.dataframe(pid.unexplained[["two_theta", "d", "height", "snr"]],
                                 hide_index=True, use_container_width=True)
                pick = st.multiselect("Add found phases as reference phases (for the "
                                      "plot and the other tabs)",
                                      pid.phases["database id"].tolist(), key="id_pick")
                if st.button("Add", key="id_add", disabled=not pick):
                    from xrdkit.db import _structure_phase
                    for pid_ in pick:
                        try:
                            if pid_.startswith("COD "):
                                ph = from_cod_id(pid_.split()[1])
                            else:
                                stc = pid.structures[pid_]
                                sg = stc.get_space_group_info(symprec=0.01)
                                ph = _structure_phase(
                                    f"{stc.composition.reduced_formula} ({pid_})", stc,
                                    "#444444", f"{pid_}",
                                    {"formula": stc.composition.reduced_formula,
                                     "spacegroup": sg[0], "spacegroup_number": sg[1]})
                            ph.name = _unique(ph.name, {**LIB, **CUSTOM})
                            ph.color = assign_color(USED)
                            USED.append(ph.color)
                            CUSTOM[ph.name] = ph
                        except Exception as e:
                            st.error(f"Could not load {pid_}: {type(e).__name__}: {e}")
                    st.rerun()
            with st.expander("What this result can and cannot tell you"):
                for cv in pid.caveats:
                    st.markdown(f"- {cv}")
            with st.expander("Step-by-step decisions"):
                for s_ in pid.steps:
                    st.markdown(f"**Step {s_['step']}** — {s_['decision']}")
                    if s_["tried"]:
                        st.dataframe(pd.DataFrame(s_["tried"]), hide_index=True,
                                     use_container_width=True)
                st.caption(f"database used: {pid.source} · {len(pid.peaks)} peaks")
                for e in pid.errors[:5]:
                    st.caption(f"⚠ {e}")
                if len(pid.errors) > 5:
                    st.caption(f"… and {len(pid.errors) - 5} more")

    with t_peaks:
        c = st.columns([2, 1])
        pk_name = c[0].selectbox("Pattern", an_names, key="pk_name")
        pk_snr = c[1].number_input("min S/N", 2.0, 50.0, 5.0, step=1.0, key="pk_snr")
        with st.spinner("fitting peaks…"):
            pk = _peaks(pk_name, pk_snr)
        st.caption(f"{len(pk)} peaks · noise σ = {pk.attrs.get('noise_sigma', float('nan')):.3g}"
                   " · esd from the fit covariance (statistical only)."
                   + (" Kα2 companions fitted at fixed ratio." if an_wl[1] else
                      " Single wavelength: tick the Kα1/Kα2 box for laboratory tube data."))
        st.dataframe(pk, hide_index=True, use_container_width=True)
        st.download_button("Download peak table (CSV)", pk.to_csv(index=False).encode(),
                           file_name=f"{pk_name}_peaks.csv", mime="text/csv")

    with t_ka2:
        if an_wl[1] is None:
            st.info("Tick “Kα1 / Kα2 doublet” in the sidebar to select the Kα2 line.")
        else:
            ka_name = st.selectbox("Pattern", an_names, key="ka_name")
            stripped = xa.strip_kalpha2(an_by[ka_name], an_wl[0], an_wl[1], an_wl[2])
            kfig, kax = plt.subplots(figsize=(6.0, 2.6))
            kax.plot(an_by[ka_name].two_theta, an_by[ka_name].intensity, lw=0.6,
                     color="#999999", label="measured")
            kax.plot(stripped.two_theta, stripped.intensity, lw=0.7,
                     color="#0072B2", label="Kα1 only (Rachinger)")
            kax.set_xlim(tt_lo, tt_hi); kax.set_xlabel("2θ (°)"); kax.set_ylabel("I (counts)")
            kax.legend(frameon=False)
            st.pyplot(kfig, use_container_width=True); plt.close(kfig)
            st.caption("Rachinger assumes an exact Kα2/Kα1 ratio and identical line "
                       "shapes; small negative ripples next to sharp peaks are "
                       "the usual artefact. Fit the doublet (peak table / Pawley) "
                       "for quantitative work.")
            if st.button("Add stripped pattern to the plot", key="ka_add"):
                nm = _unique(stripped.name, MEASURED)
                stripped.name = nm
                MEASURED[nm] = stripped
                st.rerun()

    with t_pawley:
        st.markdown(
            "Whole-pattern Pawley fit (free intensities, Caglioti widths, pseudo-Voigt, "
            "Chebyshev background, sample displacement or zero shift) of each candidate "
            "structure. Candidates are ranked by BIC_eff, the BIC with the number of "
            "points divided by the serial correlation of the residuals (neighbouring "
            "points of a profile misfit are not independent); ΔBIC_eff > 10 is strong "
            "evidence against a candidate. A fit is `unexplained` unless it beats a "
            "background-only model by 10, and `poor_fit` if GoF > 3. A Pawley fit sees "
            "the cell metric only: space groups with the same cell shape whose "
            "distinguishing lines are too weak to observe are not separated, and a "
            "lower-symmetry cell always fits at least as well. esds are inflated for "
            "residual correlation; `cell sys` is the change when the 2θ offset is "
            "refined as zero shift instead of displacement (or vice versa) and is "
            "usually the larger error.")

        def _spec_of(ph):
            if ph.info.get("structure") is not None:
                sp = xa.phase_spec(ph.info["structure"])
            else:
                sp = xa.phase_spec({**ph.info, "name": ph.name})
            sp["name"] = ph.name
            return sp

        usable = {}
        for n_, p_ in {**LIB, **CUSTOM}.items():
            try:
                usable[n_] = _spec_of(p_)
            except Exception:
                pass
        st.session_state.setdefault("pw_manual", {})
        usable.update(st.session_state.pw_manual)

        TILT = {f"{k} (ABO₃ perovskite tilt model)": k for k in xa.PEROVSKITE_MODELS}
        c = st.columns([3, 1])
        pw_names = c[0].multiselect("Patterns", an_names, default=an_names, key="pw_names")
        pw_zero = c[1].radio("Offset", ["displacement", "zero"], key="pw_zero",
                             help="sample displacement (∝cosθ, Bragg–Brentano) or "
                                  "constant zero shift; refining both is ill-conditioned")
        pw_models = st.multiselect(
            "Candidate structures (main phase)", list(usable) + list(TILT),
            default=([ph.name for ph in phases if ph.name in usable]
                     or [n_ for n_ in TILT if TILT[n_] in ("Pm-3m", "R-3c", "Pnma")]),
            key="pw_cands",
            help="Reference phases (library, uploaded CIF, database hits, cells typed "
                 "below) start from their own cell. ABO₃ tilt models (Pm-3m, R-3c, "
                 "Pnma, I4/mcm, Imma, …) are built from one pseudocubic parameter "
                 "estimated from the data. Every candidate is refined and ranked by BIC.")
        bc = st.columns(4)
        pw_zfix = 0.0
        if pw_zero == "displacement":
            pw_zfix = bc[3].number_input(
                "fixed zero (°)", -1.0, 1.0, 0.0, 0.001, format="%.4f", key="pw_zfix",
                help="instrument zero from a standard (Si, LaB6) on the same set-up; "
                     "held fixed while displacement is refined")

        def _all_tilt():
            st.session_state.pw_cands = list(dict.fromkeys(
                [n_ for n_ in st.session_state.get("pw_cands", []) if n_ not in TILT]
                + list(TILT)))

        def _clear_cands():
            st.session_state.pw_cands = []

        bc[0].button("Select all ABO₃ tilt models", key="pw_all_tilt", on_click=_all_tilt)
        bc[1].button("Clear candidates", key="pw_clear", on_click=_clear_cands)
        pw_apc = 0.0
        if any(n_ in TILT for n_ in pw_models):
            pw_apc = bc[2].number_input("Pseudocubic a seed (Å, 0 = auto)", 0.0, 5.0, 0.0,
                                        0.001, format="%.3f", key="pw_apc",
                                        help="auto: from the pseudocubic 200 line. Set it "
                                             "when the perovskite is a minority phase.")
        with st.expander("Add a candidate by space group and cell"):
            mc = st.columns(4)
            m_sg = mc[0].text_input("Space group (H-M)", "", key="pw_m_sg",
                                    placeholder="e.g. Fd-3m, P6_3mc, R-3c:H")
            m_z = mc[1].number_input("Z (formula units)", 0, 192, 0, key="pw_m_z",
                                     help="0 = unknown; V/Z is then not reported")
            m_nm = mc[2].text_input("Name", "", key="pw_m_nm")
            mc2 = st.columns(6)
            m_cell = [mc2[i].number_input(lab, 0.0, 200.0 if i < 3 else 180.0,
                                          90.0 if i >= 3 else 0.0, format="%.4f",
                                          key=f"pw_m_{lab}")
                      for i, lab in enumerate(["a (Å)", "b (Å)", "c (Å)", "α", "β", "γ"])]
            if mc[3].button("Add candidate", key="pw_m_add"):
                try:
                    a_, b_, c_ = m_cell[:3]
                    b_ = b_ or a_
                    c_ = c_ or a_
                    sp = xa.phase_spec({"name": m_nm or m_sg, "spacegroup": m_sg.strip(),
                                        "cell": (a_, b_, c_, *m_cell[3:]),
                                        "Z": m_z or None, "refine_cell": True})
                    xa.allowed_reflections(sp["spacegroup"], sp["cell"], 1.0)
                    st.session_state.pw_manual[sp["name"]] = sp
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid candidate: {type(e).__name__}: {e}")
        cand_specs = [TILT[n_] if n_ in TILT else usable[n_] for n_ in pw_models]
        pw_extra = st.multiselect(
            "Additional phases (refined jointly: own cell ±3 %, own peak width)",
            [n_ for n_ in usable if n_ not in pw_models], key="pw_extra",
            help="For multiphase samples: impurity or secondary phases from the "
                 "library, uploaded CIFs, the database tab or typed cells. Strongly "
                 "overlapping lines make cells and intensity shares less certain "
                 "than the esds suggest.")
        if st.button("Run refinement", key="pw_run", type="primary") and pw_names and cand_specs:
            prog = st.progress(0.0)
            extras = [usable[n_] for n_ in pw_extra]
            for i, nm in enumerate(pw_names):
                with st.spinner(f"{nm}: {len(cand_specs)} candidates…"):
                    try:
                        res, df = xa.compare_models(
                            an_by[nm], models=cand_specs, wavelength=an_wl,
                            tt_range=(tt_lo, tt_hi), a_pc=(pw_apc or None),
                            extra_phases=extras,
                            refine_disp=(pw_zero == "displacement"),
                            refine_zero=(pw_zero == "zero"),
                            zero_fixed=float(pw_zfix))
                        st.session_state.pawley[_sig(nm)] = (res, df)
                        if df.attrs.get("warning"):
                            st.warning(f"{nm}: {df.attrs['warning']}")
                    except Exception as e:
                        st.error(f"{nm}: {type(e).__name__}: {e}")
                prog.progress((i + 1) / len(pw_names))
        done = [n for n in pw_names if _sig(n) in st.session_state.pawley]
        if done:
            summ = []
            for nm in done:
                res, df = st.session_state.pawley[_sig(nm)]
                if "BIC" not in df:
                    continue
                r0 = df.iloc[0].to_dict()
                dcol = "ΔBIC_eff" if "ΔBIC_eff" in df else "ΔBIC"
                r0["ΔBIC_eff to 2nd"] = float(df[dcol].iloc[1]) if len(df) > 1 else float("nan")
                summ.append({"sample": nm, **r0})
            if summ:
                st.subheader("Best model per sample")
                sdf = pd.DataFrame(summ)
                keep = ["sample", "model", "a (Å)", "σa", "b (Å)", "σb", "c (Å)", "σc",
                        "α (°)", "β (°)", "γ (°)", "V (Å³)", "σV", "Z", "V/Z (Å³)",
                        "a_pc (Å)", "Rwp (%)", "Rwp net (%)", "GoF", "DW", "esd ×",
                        "ΔBIC_eff to 2nd", "status"]
                keep = [k for k in keep if k in sdf and not sdf[k].isna().all()]
                st.dataframe(sdf[keep], hide_index=True,
                             use_container_width=True)
                st.download_button("Download summary (CSV)", sdf.to_csv(index=False).encode(),
                                   file_name="pawley_summary.csv", mime="text/csv")
            sel = st.selectbox("Inspect sample", done, key="pw_inspect")
            res, df = st.session_state.pawley[_sig(sel)]
            if df.attrs.get("warning"):
                st.warning(df.attrs["warning"])
            st.dataframe(df, hide_index=True, use_container_width=True)
            ok = [r for r in res if not isinstance(r, Exception)]
            if ok:
                mdl = st.selectbox("Model", [r.model for r in ok], key="pw_model_plot",
                                   index=[r.model for r in ok].index(df["model"].iloc[0])
                                   if df["model"].iloc[0] in [r.model for r in ok] else 0)
                rr = next(r for r in ok if r.model == mdl)
                (st.success if rr.status == xa.STATUS_OK else st.error)(
                    f"status: **{rr.status}** · Rwp {100 * rr.Rwp:.2f} % (net {100 * rr.Rwp_net:.2f} %)"
                    f" · GoF {rr.gof:.2f} · Durbin–Watson {rr.durbin_watson:.2f}"
                    f" · esds ×{rr.esd_scale:.2f} · {rr.message}")
                if rr.cell_sys:
                    st.caption("cell sys (zero ↔ displacement): " + ", ".join(
                        f"{k} {v:.2g}" for k, v in zip(("a", "b", "c", "α", "β", "γ"),
                                                       rr.cell_sys) if np.isfinite(v))
                        + (f" · V {rr.volume_sys:.2g} Å³" if np.isfinite(rr.volume_sys)
                           else ""))
                for w in rr.warnings:
                    st.warning(w)
                pfig = xa.make_pawley_plot(rr)
                st.pyplot(pfig, use_container_width=True)
                if rr.extra:
                    st.markdown("**Additional phases**")
                    st.dataframe(pd.DataFrame([
                        {"phase": e["name"],
                         **{k: round(v, 5) for k, v in zip("abc", e["cell"][:3])},
                         "V (Å³)": e["volume"], "FWHM scale": e["fwhm_scale"],
                         "intensity share": e["intensity_share"]} for e in rr.extra]),
                        hide_index=True, use_container_width=True)
                pbuf = io.BytesIO(); pfig.savefig(pbuf, format="pdf"); plt.close(pfig)
                st.download_button("Download fit (PDF)", pbuf.getvalue(),
                                   file_name=f"{sel}_{mdl.replace('/', '')}_pawley.pdf",
                                   mime="application/pdf")
                if rr.correlations:
                    st.caption("Strong correlations (|r| > 0.8): " + ", ".join(
                        f"{k} {v:+.2f}" for k, v in rr.correlations.items()))

    with t_wh:
        c = st.columns([2, 1, 1])
        wh_name = c[0].selectbox("Pattern", an_names, key="wh_name")
        wh_K = c[1].number_input("Shape factor K", 0.5, 1.5, 0.9, step=0.01, key="wh_K")
        wh_corr = c[2].selectbox("Instrument correction",
                                 ["auto", "lorentzian", "gaussian"], key="wh_corr")
        inst_mode = st.radio("Instrumental broadening",
                             ["none", "constant FWHM", "Caglioti U, V, W (from LaB6/Si)"],
                             horizontal=True, key="wh_inst")
        inst = None
        if inst_mode == "constant FWHM":
            inst = st.number_input("FWHM_inst (°)", 0.0, 1.0, 0.05, step=0.005,
                                   format="%.3f", key="wh_c")
        elif inst_mode.startswith("Caglioti"):
            cc = st.columns(3)
            inst = (cc[0].number_input("U", value=0.004, format="%.5f", key="wh_U"),
                    cc[1].number_input("V", value=-0.002, format="%.5f", key="wh_V"),
                    cc[2].number_input("W", value=0.002, format="%.5f", key="wh_W"))
        pk = _peaks(wh_name, 5.0)
        pk_sel = st.multiselect(
            "Peaks to use (single, resolved reflections only)",
            [f"{t:.3f}" for t in pk["two_theta"]],
            default=[f"{t:.3f}" for t in pk["two_theta"]], key=f"wh_sel_{wh_name}")
        mask = np.isin([f"{t:.3f}" for t in pk["two_theta"]], pk_sel)
        if mask.sum() >= 3:
            wh = xa.williamson_hall(pk, wavelength=an_wl1, K=wh_K, instrument=inst,
                                    correction=wh_corr, select=mask)
            wl, wr = st.columns([3, 2])
            wfig = xa.make_wh_plot(wh)
            wl.pyplot(wfig, use_container_width=True); plt.close(wfig)
            wh_ok = wh.get("consistent", True)
            wr.metric("D (nm)", f"{wh['D_nm']:.1f} ± {wh['D_nm_esd']:.1f}"
                      if np.isfinite(wh["D_nm"]) and wh_ok else "—")
            wr.metric("ε (%)", f"{100 * wh['strain']:.3f} ± {100 * wh['strain_esd']:.3f}"
                      if np.isfinite(wh["strain"]) and wh_ok else "—")
            chi = wh.get("chi2_red", np.nan)
            wr.caption(f"R² = {wh['r2']:.3f}"
                       + (f" · χ²_red = {chi:.2f}" if np.isfinite(chi) else "")
                       + f" · {wh['correction']} correction. "
                       + (wh.get("note") or ""))
            if np.isfinite(wh["strain"]) and wh["strain"] < 0:
                wr.warning("Negative strain: widths fall with angle, which a size + "
                           "microstrain model cannot produce (over-corrected "
                           "instrument function, or overlapping low-angle lines).")
            if inst is None:
                wr.warning("No instrumental correction: the instrument width adds to "
                           "every peak, so D is underestimated (a lower bound). ε is "
                           "biased by however the instrument width changes with angle, "
                           "in either direction. Measure LaB₆ or Si on the same "
                           "instrument and enter its U, V, W.")
        else:
            st.info("Select at least three peaks.")

    with t_zoom:
        zs = st.multiselect("Patterns", an_names, default=an_names, key="zoom_names")
        auto_w = auto_zoom_windows([an_by[n] for n in zs]) if zs else []
        zw = st.text_input("Windows (2θ ranges, °; empty = five strongest peaks)", "",
                           key="zoom_win",
                           placeholder="; ".join(f"{a:g}-{b:g}" for a, b in auto_w))
        try:
            wins = [tuple(float(v) for v in w.split("-")) for w in zw.split(";") if w.strip()]
            x_lo = min(float(an_by[n].two_theta.min()) for n in zs) if zs else 0.0
            x_hi = max(float(an_by[n].two_theta.max()) for n in zs) if zs else 0.0
            bad = [w for w in wins if len(w) != 2 or w[1] <= w[0]
                   or w[1] < x_lo or w[0] > x_hi]
            if bad:
                raise ValueError(f"windows must be lo-hi with lo < hi inside "
                                 f"{x_lo:g}–{x_hi:g}°: {bad}")
            wins = wins or auto_w
            if not wins:
                raise ValueError("no peaks found for automatic windows; type them")
            zc = st.columns(2)
            z_scale = zc[0].selectbox("Intensity axis", ["linear", "sqrt", "log"], key="zoom_y")
            z_bkg = zc[1].checkbox("subtract background", True, key="zoom_bkg")
            zfig = make_zoom_figure([an_by[n] for n in zs], windows=wins,
                                       phases=phases or None, wavelength_A=an_wl1,
                                       subtract_background=z_bkg, y_scale=z_scale)
            st.pyplot(zfig, use_container_width=True)
            zbuf = io.BytesIO(); zfig.savefig(zbuf, format="pdf"); plt.close(zfig)
            st.download_button("Download zoom panels (PDF)", zbuf.getvalue(),
                               file_name="xrdkit_zoom.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Zoom figure failed: {type(e).__name__}: {e}")

    with t_match:
        st.markdown(
            "Peaks not explained by the refined main phase (or by the ticked "
            "reference phases) are matched against database structures made of "
            "subsets of the elements you give. Score = fraction of unassigned "
            "intensity explained × fraction of the candidate's strong lines present. "
            "A high score is a hypothesis to check by overlay, not an identification.")
        c = st.columns([2, 2, 1])
        um_name = c[0].selectbox("Pattern", an_names, key="um_name")
        _els = sorted({str(e) for ph in phases
                       for e in (ph.info.get("structure").composition.elements
                                 if ph.info.get("structure") is not None else [])})
        um_els = c[1].text_input("Elements in the sample", ", ".join(_els), key="um_els",
                                 placeholder="e.g. Fe, Co, O",
                                 help="pre-filled from ticked reference phases with a "
                                      "structure; edit to your sample's composition")
        um_tol = c[2].number_input("tol (°)", 0.05, 0.5, 0.15, step=0.01, key="um_tol")
        c = st.columns(3)
        um_src = c[0].selectbox("Database", ["auto", "cod", "mp"], key="um_src",
                                help="auto: COD, falling back to Materials Project")
        um_strain = c[1].number_input("lattice-scale scan ± (%)", 0.0, 5.0, 1.0,
                                      step=0.5, key="um_strain") / 100
        um_sub = c[2].number_input("max elements per phase", 1, 6, 3, key="um_sub")
        refsets = [np.array([p[0] for p in ph.peaks(an_wl1, (tt_lo, tt_hi))])
                   for ph in phases]
        src = "ticked reference phases"
        if _sig(um_name) in st.session_state.pawley:
            res, df = st.session_state.pawley[_sig(um_name)]
            best = next((r for r in res if not isinstance(r, Exception)
                         and r.model == df["model"].iloc[0]), None)
            if best is not None and best.status == xa.STATUS_OK and not df.attrs.get("warning"):
                refsets.append(np.asarray(best.ticks, float))
                src = f"Pawley {best.model} reflections" + (" + ticked phases" if phases else "")
            elif best is not None:
                st.caption(f"Pawley best model ({best.model}, status {best.status}) not "
                           "used for assignment: the fit was flagged.")
        pk = _peaks(um_name, 5.0)
        un = xa.unassigned_peaks(pk, refsets, tol=um_tol) if refsets else pk
        st.caption(f"Assigned against: {src if refsets else 'nothing (run Pawley or tick a phase)'}"
                   f" · {len(un)} of {len(pk)} peaks unassigned")
        st.dataframe(un[["two_theta", "d", "height", "fwhm", "snr"]], hide_index=True,
                     use_container_width=True)
        um_list = [e for e in um_els.replace(" ", "").split(",") if e]
        if st.button("Search database", key="um_run", disabled=len(un) == 0 or not um_list,
                     help=None if um_list else "enter the elements in your sample first"):
            msg = st.empty()
            with st.spinner("querying OPTIMADE and scoring candidates…"):
                mdf = xa.match_cod(un, pk, um_list,
                                   wavelength=an_wl1, tt_range=(tt_lo, tt_hi),
                                   tol=um_tol, max_subset=int(um_sub),
                                   max_strain=um_strain, source=um_src,
                                   progress=lambda t: msg.caption(t))
            st.session_state.match_df = mdf
        if "match_df" in st.session_state:
            mdf = st.session_state.match_df
            st.caption(f"source: {mdf.attrs.get('source')}")
            for e in mdf.attrs.get("errors", [])[:5]:
                st.caption(f"⚠ {e}")
            st.dataframe(mdf.head(30), hide_index=True, use_container_width=True)
            if not mdf.empty:
                pick = st.selectbox("Add candidate as reference", mdf["id"].head(30).tolist(),
                                    key="um_pick")
                if st.button("Add as reference phase", key="um_add"):
                    try:
                        if pick.startswith("COD "):
                            ph = from_cod_id(pick.split()[1])
                        else:
                            from xrdkit.db import _structure_phase
                            stc = mdf.attrs["structures"][pick]
                            sg = stc.get_space_group_info(symprec=0.01)
                            ph = _structure_phase(
                                f"{stc.composition.reduced_formula} ({pick})", stc,
                                "#444444", f"Materials Project {pick} (DFT cell)",
                                {"formula": stc.composition.reduced_formula,
                                 "spacegroup": sg[0], "spacegroup_number": sg[1]})
                        ph.name = _unique(ph.name, {**LIB, **CUSTOM})
                        ph.color = assign_color(USED); USED.append(ph.color)
                        CUSTOM[ph.name] = ph
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load {pick}: {type(e).__name__}: {e}")


# ============ FOOTER ============
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px; line-height:1.6'>"
    "<b>Nabil Khossossi</b>, PhD<br>"
    "Researcher | AI-Driven Materials Discovery | Computational Chemistry"
    "</div>",
    unsafe_allow_html=True,
)
