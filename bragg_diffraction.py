import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bragg Diffraction", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    h1 { color: #e0e0e0; font-family: 'Courier New', monospace; letter-spacing: 2px; }
    h3 { color: #a0c4ff; font-family: 'Courier New', monospace; }
    .result-box {
        background: #1a1d2e;
        border: 1px solid #3a3f6e;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
    }
    .result-label { color: #7b8cde; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; }
    .result-value { color: #e8f4fd; font-size: 1.6em; font-weight: bold; margin-top: 4px; }
    .result-unit  { color: #7b8cde; font-size: 0.9em; }
    .warning-box  {
        background: #2a1a1a;
        border: 1px solid #8b3a3a;
        border-radius: 8px;
        padding: 14px 20px;
        color: #ff8080;
        font-family: 'Courier New', monospace;
    }
    .info-box {
        background: #1a2a1a;
        border: 1px solid #3a6e3a;
        border-radius: 8px;
        padding: 14px 20px;
        color: #80c080;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ── Physics functions ─────────────────────────────────────────────────────────

def d_spacing(crystal_type: str, a: float, h: int, k: int, l: int) -> float | None:
    """
    Return the d-spacing for the given crystal type and Miller indices.
    Returns None when the reflection is systematically absent.
    """
    if crystal_type == "Simple Cubic (SC)":
        return a / np.sqrt(h**2 + k**2 + l**2)

    elif crystal_type == "Body-Centered Cubic (BCC)":
        # Systematic absence: h+k+l must be even
        if (h + k + l) % 2 != 0:
            return None
        return a / np.sqrt(h**2 + k**2 + l**2)

    elif crystal_type == "Face-Centered Cubic (FCC)":
        # Systematic absence: h,k,l must be all odd or all even
        parities = {x % 2 for x in (h, k, l)}
        if len(parities) > 1:
            return None
        return a / np.sqrt(h**2 + k**2 + l**2)

    elif crystal_type == "Hexagonal (HCP)":
        # d_hkl for hexagonal: 1/d² = 4/3 * (h²+hk+k²)/a² + l²/c²
        # Assume c/a = 1.633 (ideal)
        c = 1.633 * a
        inv_d2 = (4 / 3) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2
        return 1 / np.sqrt(inv_d2)

    return None


def bragg_angle(wavelength_nm: float, d_nm: float) -> float | None:
    """Return Bragg angle θ in degrees, or None if no solution exists."""
    ratio = wavelength_nm / (2 * d_nm)
    if abs(ratio) > 1:
        return None
    return np.degrees(np.arcsin(ratio))


# ── Diagram ───────────────────────────────────────────────────────────────────

def draw_diagram(theta_deg: float, d_nm: float, wavelength_nm: float):
    theta = np.radians(theta_deg)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#13162a")

    # --- Draw atomic planes ---
    n_planes = 4
    y_start = 0.55
    dy = 0.18
    plane_color = "#3a6ea8"

    for i in range(n_planes):
        y = y_start - i * dy
        ax.axhline(y=y, xmin=0.05, xmax=0.95, color=plane_color,
                   linewidth=1.6, alpha=0.85, zorder=2)
        # Atoms on the plane
        for xp in np.linspace(0.10, 0.90, 9):
            ax.plot(xp, y, 'o', color="#a0c4ff", markersize=5,
                    alpha=0.9, zorder=3)

    # Plane labels
    ax.text(0.96, y_start,        "(hkl) planes", color="#6a9fd8",
            fontsize=8, va='center', fontfamily='monospace')

    # d-spacing arrow
    y_top = y_start
    y_bot = y_start - dy
    ax.annotate("", xy=(0.035, y_bot), xytext=(0.035, y_top),
                arrowprops=dict(arrowstyle="<->", color="#ffd166", lw=1.4))
    ax.text(0.005, (y_top + y_bot) / 2, f"d={d_nm:.3f} nm",
            color="#ffd166", fontsize=7.5, va='center',
            rotation=90, fontfamily='monospace')

    # --- Incident beam ---
    hit_x, hit_y = 0.50, y_start          # reflection point on top plane
    beam_len = 0.28

    # Direction vectors
    dx_in  =  np.cos(np.pi/2 - theta)     # beam goes right & down
    dy_in  = -np.sin(np.pi/2 - theta)
    dx_out =  np.cos(np.pi/2 - theta)
    dy_out =  np.sin(np.pi/2 - theta)

    # Incident ray
    ax.annotate("", xy=(hit_x, hit_y),
                xytext=(hit_x - beam_len*dx_in, hit_y - beam_len*dy_in),
                arrowprops=dict(arrowstyle="-|>", color="#ef476f",
                                lw=2, mutation_scale=14), zorder=5)

    # Reflected ray
    ax.annotate("", xy=(hit_x + beam_len*dx_out, hit_y + beam_len*dy_out),
                xytext=(hit_x, hit_y),
                arrowprops=dict(arrowstyle="-|>", color="#06d6a0",
                                lw=2, mutation_scale=14), zorder=5)

    # --- θ arc (angle to the PLANE, not normal) ---
    arc_r = 0.10
    # angle measured from the plane (horizontal), NOT from the normal
    angle_start = 180                      # pointing left along plane
    angle_end   = 180 + theta_deg          # rotated up by θ
    arc = patches.Arc((hit_x, hit_y), 2*arc_r, 2*arc_r,
                       angle=0, theta1=angle_start, theta2=angle_end,
                       color="#ffd166", lw=1.2, zorder=6)
    ax.add_patch(arc)
    mid_angle = np.radians((angle_start + angle_end) / 2)
    ax.text(hit_x + (arc_r + 0.03)*np.cos(mid_angle),
            hit_y + (arc_r + 0.03)*np.sin(mid_angle),
            f"θ={theta_deg:.1f}°", color="#ffd166",
            fontsize=9, ha='center', va='center', fontfamily='monospace')

    # --- Labels ---
    ax.text(hit_x - beam_len*dx_in*0.5 - 0.06,
            hit_y - beam_len*dy_in*0.5 + 0.02,
            "Incident\nbeam", color="#ef476f", fontsize=8,
            ha='center', fontfamily='monospace')
    ax.text(hit_x + beam_len*dx_out*0.6 + 0.02,
            hit_y + beam_len*dy_out*0.5 + 0.02,
            "Diffracted\nbeam", color="#06d6a0", fontsize=8,
            ha='center', fontfamily='monospace')

    # λ label on incident beam
    mid_ix = hit_x - beam_len*dx_in*0.5
    mid_iy = hit_y - beam_len*dy_in*0.5
    ax.text(mid_ix + 0.04, mid_iy - 0.04,
            f"λ={wavelength_nm:.3f} nm", color="#ffb3c6",
            fontsize=8, fontfamily='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 0.85)
    ax.axis("off")

    ax.text(0.5, 0.80, "Bragg Diffraction — 2D Diagram",
            color="#8899cc", fontsize=9, ha='center',
            fontfamily='monospace', transform=ax.transAxes)

    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("# ⟨ BRAGG DIFFRACTION SIMULATOR ⟩")
st.markdown("---")

col_in, col_out = st.columns([1, 1.6], gap="large")

with col_in:
    st.markdown("### Inputs")

    crystal = st.selectbox(
        "Crystal Structure",
        ["Simple Cubic (SC)",
         "Body-Centered Cubic (BCC)",
         "Face-Centered Cubic (FCC)",
         "Hexagonal (HCP)"],
    )

    lam = st.number_input(
        "Wavelength λ (nm)",
        min_value=0.001, max_value=10.0,
        value=0.154, step=0.001, format="%.3f",
        help="Cu Kα ≈ 0.154 nm"
    )

    a = st.number_input(
        "Lattice Constant a (nm)",
        min_value=0.01, max_value=10.0,
        value=0.405, step=0.001, format="%.3f",
        help="Aluminium ≈ 0.405 nm"
    )

    st.markdown("**Miller Indices (h, k, l)**")
    c1, c2, c3 = st.columns(3)
    with c1: h = st.number_input("h", value=1, step=1, min_value=-10, max_value=10)
    with c2: k = st.number_input("k", value=1, step=1, min_value=-10, max_value=10)
    with c3: l = st.number_input("l", value=1, step=1, min_value=-10, max_value=10)

    # Quick reference
    st.markdown("""
<div class='info-box'>
<b>Quick reference</b><br>
Cu Kα  : λ = 0.154 nm<br>
Al (FCC): a = 0.405 nm → (1,1,1) gives θ ≈ 19.3°<br>
Fe (BCC): a = 0.287 nm → (1,1,0) gives θ ≈ 32.5°<br>
NaCl (SC): a = 0.563 nm → (1,0,0) gives θ ≈ 7.9°
</div>
""", unsafe_allow_html=True)

# ── Compute ───────────────────────────────────────────────────────────────────
with col_out:
    st.markdown("### Results")

    if h == 0 and k == 0 and l == 0:
        st.markdown("<div class='warning-box'>⚠ (0,0,0) is not a valid Miller index.</div>",
                    unsafe_allow_html=True)
    else:
        d = d_spacing(crystal, a, h, k, l)

        if d is None:
            st.markdown(
                f"<div class='warning-box'>⚠ ({h},{k},{l}) is a <b>systematically absent</b> "
                f"reflection for {crystal}.<br>Structure factor F = 0 → no diffraction.</div>",
                unsafe_allow_html=True)
        else:
            theta = bragg_angle(lam, d)

            # d-spacing card
            st.markdown(f"""
<div class='result-box'>
  <div class='result-label'>d-spacing  d<sub>hkl</sub></div>
  <div class='result-value'>{d:.4f} <span class='result-unit'>nm</span></div>
</div>""", unsafe_allow_html=True)

            if theta is None:
                st.markdown(
                    "<div class='warning-box'>⚠ No solution: λ > 2d "
                    "(wavelength too large for this plane spacing).<br>"
                    "Bragg condition cannot be satisfied.</div>",
                    unsafe_allow_html=True)
            else:
                two_theta = 2 * theta

                st.markdown(f"""
<div class='result-box'>
  <div class='result-label'>Bragg Angle  θ</div>
  <div class='result-value'>{theta:.3f} <span class='result-unit'>degrees</span></div>
</div>
<div class='result-box'>
  <div class='result-label'>Diffractometer Angle  2θ</div>
  <div class='result-value'>{two_theta:.3f} <span class='result-unit'>degrees</span></div>
</div>""", unsafe_allow_html=True)

                # Bragg's law verification
                lam_check = 2 * d * np.sin(np.radians(theta))
                st.markdown(f"""
<div class='result-box'>
  <div class='result-label'>Bragg's Law Check  2d·sin(θ)</div>
  <div class='result-value'>{lam_check:.4f} <span class='result-unit'>nm</span>
  {"✓" if abs(lam_check - lam) < 1e-9 else "≈"} λ</div>
</div>""", unsafe_allow_html=True)

                st.markdown("### Diagram")
                fig = draw_diagram(theta, d, lam)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-family:monospace; color:#555; font-size:0.8em; text-align:center'>
Bragg's Law: <b>nλ = 2d·sin(θ)</b> &nbsp;|&nbsp; n=1 (first order) &nbsp;|&nbsp;
d-spacing from the plane-spacing formula for each Bravais lattice
</div>
""", unsafe_allow_html=True)
