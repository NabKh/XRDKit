"""Streamlit GUI for xrdkit."""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from xrdkit.core import WAVELENGTHS, WAVELENGTHS_KA12, NATURE_RC
from xrdkit.db import (Phase, builtin_phases, from_structure_file, from_mp_id,
                        from_cod_id, search_mp, search_cod, assign_color)
from xrdkit.plot import (Measured, load_measured, make_figure, save_gif,
                          peaks_table, compute_difference, make_difference_plot)


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
    st.header("Quick-pick reference cards")
    st.caption("Electrocatalysis presets, tick to add. Any crystalline material loads from the CIF / COD / Materials Project tabs. 
  Nothing shown by default")
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

    use_doublet = st.checkbox("Kα1 / Kα2 doublet", value=False,
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
st.title("XRD reference pattern composer")
st.caption("Compose figures from ICDD cards · CIF / POSCAR files · the "
           "Crystallography Open Database · the Materials Project, and overlay "
           "your own measured or computed diffractograms.")

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
             "Two-column ASCII (`.xy`, `.csv`, `.txt`, `.dat`, `.tsv`) or "
             "PANalytical `.xrdml`. Use offset / scale / header-skip if needed.")
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
                m = load_measured(tmp_path, name=Path(f.name).stem,
                                  color=meas_color, skiprows=int(meas_skip),
                                  scale=float(meas_scale),
                                  two_theta_offset=float(meas_offset))
                m_name = _unique(m.name, set(MEASURED))
                if m_name != m.name:
                    m = Measured(name=m_name, two_theta=m.two_theta,
                                 intensity=m.intensity, color=m.color,
                                 source=m.source, info=m.info)
                MEASURED[m_name] = m
                st.session_state.processed_upload_ids.add(fid)
                st.success(f"loaded **{m_name}**  ·  {m.info['n_points']} pts  ·  "
                           f"2θ {m.info['tt_range'][0]:.1f}–{m.info['tt_range'][1]:.1f}°")
            except Exception as e:
                st.error(f"`{f.name}`: {type(e).__name__}: {e}")


# ============ LIVE PLOT ============
st.divider()

phases = [LIB[n] for n in LIB if builtin_on.get(n, False)]
phases += [CUSTOM[n] for n in CUSTOM if custom_on.get(n, True)]
measured = [MEASURED[n] for n in MEASURED if meas_on.get(n, True)]

if not phases and not measured:
    st.info("Pick an ICDD card on the left, upload a structure or measurement, "
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


# ============ FOOTER ============
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px; line-height:1.6'>"
    "<b>Nabil Khossossi</b>, PhD<br>"
    "Researcher | AI-Driven Materials Discovery | Computational Chemistry"
    "</div>",
    unsafe_allow_html=True,
)
