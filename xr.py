import math

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import streamlit as st


def compute_d_spacing(crystal_type, a, c, h, k, l):
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices (0, 0, 0) are not valid.")

    if crystal_type in {"cubic", "bcc", "fcc"}:
        return a / math.sqrt(h**2 + k**2 + l**2)

    if crystal_type == "tetragonal":
        inv_d2 = (h**2 + k**2) / (a**2) + (l**2) / (c**2)
        return 1 / math.sqrt(inv_d2)

    if crystal_type == "hexagonal":
        inv_d2 = (4 / 3) * (h**2 + h * k + k**2) / (a**2) + (l**2) / (c**2)
        return 1 / math.sqrt(inv_d2)

    raise ValueError("Unsupported crystal type.")


def reflection_rule(crystal_type, h, k, l):
    h, k, l = abs(h), abs(k), abs(l)

    if crystal_type == "bcc":
        allowed = (h + k + l) % 2 == 0
        msg = "Allowed for BCC only when h + k + l is even."
        return allowed, msg

    if crystal_type == "fcc":
        allowed = (h % 2 == k % 2 == l % 2)
        msg = "Allowed for FCC only when h, k, l are all even or all odd."
        return allowed, msg

    return True, "Reflection allowed."


def compute_theta(wavelength, d, order=1):
    ratio = order * wavelength / (2 * d)
    if ratio > 1:
        return None
    return math.degrees(math.asin(ratio))


def plot_xrd_diagram(theta_deg, d_spacing):
    theta = math.radians(theta_deg)
    length = 3.0

    x_in = -length * math.cos(theta)
    y_in = length * math.sin(theta)
    x_out = length * math.cos(theta)
    y_out = length * math.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Crystal planes
    ax.plot([-3.5, 3.5], [0, 0], color="black", lw=2)
    ax.plot([-3.5, 3.5], [-0.9, -0.9], color="gray", lw=1.5)

    # Normal
    ax.plot([0, 0], [-1.4, 2.5], linestyle="--", color="gray", lw=1)

    # Incident and reflected rays
    ax.annotate(
        "",
        xy=(0, 0),
        xytext=(x_in, y_in),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#1f77b4"),
    )
    ax.annotate(
        "",
        xy=(x_out, y_out),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#d62728"),
    )

    # Angle arcs
    ax.add_patch(Arc((0, 0), 1.4, 1.4, angle=0, theta1=0, theta2=theta_deg, color="#d62728", lw=2))
    ax.add_patch(Arc((0, 0), 1.4, 1.4, angle=0, theta1=180 - theta_deg, theta2=180, color="#1f77b4", lw=2))

    ax.text(0.7, 0.18, r"$\theta$", fontsize=13)
    ax.text(-0.95, 0.18, r"$\theta$", fontsize=13)

    # d-spacing marker
    ax.annotate(
        "",
        xy=(2.7, 0),
        xytext=(2.7, -0.9),
        arrowprops=dict(arrowstyle="<->", lw=1.8, color="green"),
    )
    ax.text(2.85, -0.45, f"d = {d_spacing:.4f}", color="green", va="center", fontsize=11)

    # Labels
    ax.text(-2.8, 2.0, "Incident ray", color="#1f77b4", fontsize=11)
    ax.text(1.5, 2.0, "Reflected ray", color="#d62728", fontsize=11)
    ax.text(0.1, 2.15, "Normal", color="gray", fontsize=10)
    ax.text(-3.3, 0.08, "Crystal plane", fontsize=10)
    ax.text(-3.3, -0.82, "Crystal plane", fontsize=10, color="gray")

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.4, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def formula_text(crystal_type):
    if crystal_type in {"cubic", "bcc", "fcc"}:
        return r"\frac{1}{d^2}=\frac{h^2+k^2+l^2}{a^2}"
    if crystal_type == "tetragonal":
        return r"\frac{1}{d^2}=\frac{h^2+k^2}{a^2}+\frac{l^2}{c^2}"
    if crystal_type == "hexagonal":
        return r"\frac{1}{d^2}=\frac{4}{3}\frac{h^2+hk+k^2}{a^2}+\frac{l^2}{c^2}"
    return ""


def main():
    st.set_page_config(page_title="XRD Simulator", layout="centered")
    st.title("XRD Simulator Using Bragg's Law")

    st.markdown(r"Bragg's law:  $n\lambda = 2d\sin\theta$")
    st.caption("This app uses first-order diffraction: n = 1")

    with st.sidebar:
        st.header("Inputs")

        wavelength = st.number_input(
            "Wavelength, λ (Angstrom)",
            min_value=0.0001,
            value=1.5406,
            step=0.0001,
            format="%.4f",
        )

        crystal_type = st.selectbox(
            "Crystal Type",
            ["cubic", "bcc", "fcc", "tetragonal", "hexagonal"],
        )

        a = st.number_input(
            "Lattice Constant a (Angstrom)",
            min_value=0.0001,
            value=4.0000,
            step=0.0001,
            format="%.4f",
        )

        c = a
        if crystal_type in {"tetragonal", "hexagonal"}:
            c = st.number_input(
                "Lattice Constant c (Angstrom)",
                min_value=0.0001,
                value=5.0000,
                step=0.0001,
                format="%.4f",
            )

        st.subheader("Miller Indices")
        col1, col2, col3 = st.columns(3)
        with col1:
            h = int(st.number_input("h", value=1, step=1))
        with col2:
            k = int(st.number_input("k", value=1, step=1))
        with col3:
            l = int(st.number_input("l", value=1, step=1))

    try:
        d_spacing = compute_d_spacing(crystal_type, a, c, h, k, l)
        theta_deg = compute_theta(wavelength, d_spacing)
        allowed, rule_message = reflection_rule(crystal_type, h, k, l)

        st.subheader("Results")
        st.latex(formula_text(crystal_type))

        if crystal_type in {"bcc", "fcc"}:
            st.info("For BCC and FCC, the d-spacing formula is the same as cubic; selection rules determine whether the reflection is allowed.")

        col1, col2, col3 = st.columns(3)
        col1.metric("d-spacing (Angstrom)", f"{d_spacing:.4f}")

        if theta_deg is None:
            col2.metric("Theta, θ (deg)", "No solution")
            col3.metric("2θ (deg)", "No solution")
            st.error("No real Bragg angle exists because λ > 2d for the chosen values.")
        else:
            col2.metric("Theta, θ (deg)", f"{theta_deg:.3f}")
            col3.metric("2θ (deg)", f"{2 * theta_deg:.3f}")

            fig = plot_xrd_diagram(theta_deg, d_spacing)
            st.subheader("Diffraction Diagram")
            st.pyplot(fig)

        st.subheader("Details")
        st.write(f"Crystal type: **{crystal_type.upper()}**")
        st.write(f"Miller indices: **({h} {k} {l})**")
        st.write(f"Wavelength λ = **{wavelength:.4f} Angstrom**")
        st.write(f"Lattice constant a = **{a:.4f} Angstrom**")
        if crystal_type in {"tetragonal", "hexagonal"}:
            st.write(f"Lattice constant c = **{c:.4f} Angstrom**")

        if allowed:
            st.success(rule_message)
        else:
            st.warning(f"{rule_message} This peak is systematically absent for the selected structure.")

    except ValueError as e:
        st.error(str(e))


if __name__ == "__main__":
    main()