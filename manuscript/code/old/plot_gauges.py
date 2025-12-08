import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

plt.style.use("manuscript/code/gaugefixer.mplstyle")


if __name__ == '__main__':
    # --- Figure / axes ---
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection="3d")

    # Nice view angle (tweak if you want)
    elev, azim = 22, -60
    ax.view_init(elev=elev, azim=azim)
    ls = LightSource(azdeg=azim, altdeg=elev)

    # --- Plane: x + z = 0  ->  z = -x ---
    L = 3.0  # overall scene half-size
    theta1 = np.linspace(-L, L, 120)
    theta2 = np.linspace(-L, L, 120)
    X, Y = np.meshgrid(theta1, theta2)
    f1 = 0.5
    f2 = -1
    theta0 = (f1 + f2) / 2.0 + (X + Y) / 2.0

    # 2) shading via LightSource
    # estimate "height" for shading; normals from gradients
    dx, dy = np.gradient(theta0, theta1, theta2)
    # fake "elevation" field—works well enough for matte shading
    lightsrc = (315, 45)
    # Convert base color to RGB and shade
    face_color = "#4C78A8"
    shaded = ls.shade(theta0, cmap=cm.Greys, blend_mode="soft")
    shaded = np.clip(shaded, 0, 1)

    plane = ax.plot_surface(
        X,
        Y,
        theta0,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.35,
        facecolors=shaded,
        zsort="average",
        label="zero-sum gauge",
        rasterized=True,
    )

    # --- Line: intersection of x+y=2 and x+z=1 ---
    # Parametrize by x = t  => y = 2 - t, z = 1 - t
    theta0 = np.linspace(
        -L + 0.75, L - 0.75, 100
    )  # slightly extended range to show in view
    theta1 = f1 - theta0
    theta2 = f2 - theta0

    t0_fixed = (f1 + f2) / 2.0
    theta_fixed = [t0_fixed, f1 - t0_fixed, f2 - t0_fixed]
    above = theta0 > theta_fixed[0]

    vmin, vmax = theta0.min(), theta0.max()
    cmap = plt.get_cmap("Reds")
    for i in range(above.sum() - 1):
        ax.plot3D(
            theta1[above][i : i + 2],
            theta2[above][i : i + 2],
            theta0[above][i : i + 2],
            linewidth=0.75,
            label=r"$f=X\theta$" if i == 0 else None,
            color=cmap((theta0[above][i] - vmin) / (vmax - vmin)),
            rasterized=True,
        )
    ax.plot3D(
        theta1[~above],
        theta2[~above],
        theta0[~above],
        color=cmap((theta0[above][i] - vmin) / (vmax - vmin)),
        linewidth=0.75,
        linestyle="--",
        alpha=0.5,
        rasterized=True,
    )


    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0, 0, 0, 0))  #

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("none")
        pane.set_linewidth(0.5)
        pane.set_alpha(0.5)
        pane.set_facecolor((1, 1, 1, 0))
    ax.scatter(
        theta_fixed[1],
        theta_fixed[2],
        theta_fixed[0],
        c="darkred",
        s=2,
        label=r"Gauge-fixed $\theta$",
    )

    # Draw custom "spines" (axes) that cross at the origin (0,0,0)
    ax.plot(
        [-L, L], [0, 0], [0, 0], color="k", linewidth=0.75
    )  # x spine through origin
    ax.plot(
        [-L - 1, L + 1],
        [0, 0],
        [0, 0],
        color="k",
        linewidth=0.75,
        linestyle="--",
    )  # x spine through origin
    ax.plot(
        [0, 0], [-L, L], [0, 0], color="k", linewidth=0.75
    )  # y spine through origin
    ax.plot(
        [0, 0],
        [-L - 2, L + 2],
        [0, 0],
        color="k",
        linewidth=0.75,
        linestyle="--",
    )  # x spine through origin
    ax.plot(
        [0, 0], [0, 0], [-L, L], color="k", linewidth=0.75
    )  # z spine through origin
    ax.plot(
        [0, 0],
        [0, 0],
        [-L - 0.75, L + 0.75],
        color="k",
        linewidth=0.75,
        linestyle="--",
    )  # z spine through origin


    # Labels
    ticks = [-2, -1, 1, 2]
    for t in ticks:
        ax.plot3D([t, t], [0, 0], [-0.1, 0.1], color="k", lw=0.5)
        ax.text(t + 0.125, -0.3, -0.05, f"{t}", ha="center", va="top", fontsize=5)

        ax.plot3D([0, 0], [t, t], [-0.1, 0.1], color="k", lw=0.5)
        ax.text(0.0, t, -0.25, f"{t}", ha="center", va="top", fontsize=5)

        ax.plot3D([0, 0], [-0.1, 0.1], [t, t], color="k", lw=0.5)
        ax.text(0.1, 0.2, t, f"{t}", ha="left", va="center", fontsize=5)

    # Axis labels
    ax.text(L + 0.3, 0, 0.1, r"$\theta_1$", fontsize=6, weight="bold")
    ax.text(-0.1, L + 0.2, 0.3, r"$\theta_2$", fontsize=6, weight="bold")
    ax.text(0.15, 0, L + 0.2, r"$\theta_0$", fontsize=6, weight="bold")
    ax.set(xticks=[], yticks=[], zticks=[])

    # Ticks subtle
    ax.tick_params(axis="both", which="major", length=4, width=0.8, labelsize=9)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)
    ax.legend(loc=(0.1, 0.8))

    # Limits
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1))  # equal aspect

    plt.tight_layout()
    plt.savefig("manuscript/figures/gauge_fixing.png", dpi=300)
    plt.savefig("manuscript/figures/gauge_fixing.svg", dpi=300)
