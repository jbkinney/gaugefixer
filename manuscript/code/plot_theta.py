from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr

plt.style.use("manuscript/code/gaugefixer.mplstyle")

plot_theta_0 = True
plot_theta_lc_lclc = True
plot_theta_scatter = True



def _get_45deg_mesh(mat):
    """Create X and Y grids rotated -45 degreees."""
    # Define rotation matrix
    theta = -np.pi / 4
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Define unrotated coordinates on
    K = len(mat) + 1
    grid1d = np.arange(0, K) - 0.5
    X = np.tile(np.reshape(grid1d, [K, 1]), [1, K])
    Y = np.tile(np.reshape(grid1d, [1, K]), [K, 1])
    xy = np.array([X.ravel(), Y.ravel()])

    # Rotate coordinates
    xy_rot = R @ xy
    X_rot = xy_rot[0, :].reshape(K, K)
    Y_rot = xy_rot[1, :].reshape(K, K).T

    return X_rot, Y_rot


def heatmap_pairwise(
    values,
    alphabet,
    gpmap_type='pairwise',
    seq=None,
    seq_kwargs=None,
    ax=None,
    show_position=False,
    position_size=None,
    position_pad=1,
    show_alphabet=True,
    alphabet_size=None,
    alphabet_pad=1,
    show_seplines=True,
    sepline_kwargs=None,
    xlim_pad=0.1,
    ylim_pad=0.1,
    cbar=True,
    cbar_kwargs={},
    cax=None,
    clim=None,
    clim_quantile=1,
    ccenter=0,
    cmap="coolwarm",
    cmap_size="5%",
    cmap_pad=0.1,
):
    """
    Draw a heatmap illustrating pairwise or neighbor values, e.g. representing
    model parameters, mutational effects, etc.

    Note: The resulting plot has aspect ratio of 1 and is scaled so that pixels
    have half-diagonal lengths given by ``half_pixel_diag = 1/(C*2)``, and
    blocks of characters have half-diagonal lengths given by
    ``half_block_diag = 1/2``. This is done so that the horizontal distance
    between positions (as indicated by x-ticks) is 1.
    
    Adapted from MAVE-NN:
    https://github.com/jbkinney/mavenn/blob/master/mavenn/src/visualization.py

    Parameters
    ----------
    values: (np.array)
        An array, shape ``(L,C,L,C)``, containing pairwise or neighbor values.
        Note that only values at coordinates ``[l1, c1, l2, c2]`` with
        ``l2`` > ``l1`` will be plotted. NaN values will not be plotted.

    alphabet: 
        Alphabet name ``'dna'``, ``'rna'``, or ``'protein'``, or 1D array
        containing characters in the alphabet.

    seq: (str, None)
        The sequence to show, if any, using dots plotted on top of the heatmap.
        Must have length ``L`` and be comprised of characters in ``alphabet``.

    seq_kwargs: (dict)
        Arguments to pass to ``Axes.scatter()`` when drawing dots to illustrate
        the characters in ``seq``.

    ax: (matplotlib.axes.Axes)
        The ``Axes`` object on which the heatmap will be drawn.
        If ``None``, one will be created. If specified, ``cbar=True``,
        and ``cax=None``, ``ax`` will be split in two to make room for a
        colorbar.

    gpmap_type: (str)
        Determines how many pairwise parameters are plotted.
        Must be ``'pairwise'`` or ``'neighbor'``. If ``'pairwise'``, a
        triangular heatmap will be plotted. If ``'neighbor'``, a heatmap
        resembling a string of diamonds will be plotted.

    show_position: (bool)
        Whether to annotate the heatmap with position labels.

    position_size: (float)
        Font size to use for position labels. Must be >= 0.

    position_pad: (float)
        Additional padding, in units of ``half_pixel_diag``, used to space
        the position labels further from the heatmap.

    show_alphabet: (bool)
        Whether to annotate the heatmap with character labels.

    alphabet_size: (float)
        Font size to use for alphabet. Must be >= 0.

    alphabet_pad: (float)
        Additional padding, in units of ``half_pixel_diag``, used to space
        the alphabet labels from the heatmap.

    show_seplines: (bool)
        Whether to draw lines separating character blocks for different
        position pairs.

    sepline_kwargs: (dict)
        Keywords to pass to ``Axes.plot()`` when drawing seplines.

    xlim_pad: (float)
        Additional padding to add (in absolute units) both left and right of
        the heatmap.

    ylim_pad: (float)
        Additional padding to add (in absolute units) both above and below the
        heatmap.

    cbar: (bool)
        Whether to draw a colorbar next to the heatmap.
    
    cbar_kwargs: dict
        Keywords to pass to ``plt.colorbar()`` when drawing colorbar.

    cax: (matplotlib.axes.Axes, None)
        The ``Axes`` object on which the colorbar will be drawn, if requested.
        If ``None``, one will be created by splitting ``ax`` in two according
        to ``cmap_size`` and ``cmap_pad``.

    clim: (list, None)
        List of the form ``[cmin, cmax]``, specifying the maximum ``cmax``
        and minimum ``cmin`` values spanned by the colormap. Overrides
        ``clim_quantile``.

    clim_quantile: (float)
        Must be a float in the range [0,1]. ``clim`` will be automatically
        chosen to include this central quantile of values.

    ccenter: (float)
        Value at which to position the center of a diverging
        colormap. Setting ``ccenter=0`` often makes sense.

    cmap: (str, matplotlib.colors.Colormap)
        Colormap to use.

    cmap_size: (str)
        Fraction of ``ax`` width to be used for the colorbar. For formatting
        requirements, see the documentation for
        ``mpl_toolkits.axes_grid1.make_axes_locatable()``.

    cmap_pad: (float)
        Space between colorbar and the shrunken heatmap ``Axes``. For formatting
        requirements, see the documentation for
        ``mpl_toolkits.axes_grid1.make_axes_locatable()``.

    Returns
    -------
    ax: (matplotlib.axes.Axes)
        ``Axes`` object containing the heatmap.

    cb: (matplotlib.colorbar.Colorbar, None)
        Colorbar object linked to ``ax``, or ``None`` if no colorbar was drawn.
    """
    # Validate values
    L, C, L2, C2 = values.shape
    values = values.copy()
    ls = np.arange(L).astype(int)
    l1_grid = np.tile(np.reshape(ls, (L, 1, 1, 1)), (1, C, L, C))
    l2_grid = np.tile(np.reshape(ls, (1, 1, L, 1)), (L, C, 1, C))

    nan_ix = ~(l2_grid - l1_grid >= 1)


    # Set values at invalid positions to nan
    values[nan_ix] = np.nan

    # Reshape values into a matrix
    mat = values.reshape((L * C, L * C))
    mat = mat[:-C, :]
    mat = mat[:, C:]
    K = (L - 1) * C

    # Verify that mat is the right size
    assert mat.shape == (K, K), (
        f"mat.shape={mat.shape}; expected{(K, K)}. Should never happen."
    )

    # Get indices of finite elements of mat
    ix = np.isfinite(mat)

    # Set color lims to central 95% quantile
    if clim is None:
        clim = np.quantile(
            mat[ix], q=[(1 - clim_quantile) / 2, 1 - (1 - clim_quantile) / 2]
        )

    # Create axis if none already exists
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Needed to center colormap at zero
    if ccenter is not None:
        # Reset ccenter if is not compatible with clim
        if (clim[0] > ccenter) or (clim[1] < ccenter):
            ccenter = 0.5 * (clim[0] + clim[1])

        norm = TwoSlopeNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])

    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Get rotated mesh
    X_rot, Y_rot = _get_45deg_mesh(mat)

    # Normalize
    half_pixel_diag = 1 / (2 * C)
    pixel_side = 1 / (C * np.sqrt(2))
    X_rot = X_rot * pixel_side + half_pixel_diag
    Y_rot = Y_rot * pixel_side

    # Set parameters that depend on gpmap_type
    ysep_min = -0.5 - 0.001 * half_pixel_diag
    xlim = (-xlim_pad, L - 1 + xlim_pad)
    ysep_max = L / 2 + 0.001 * half_pixel_diag
    ylim = (-0.5 - ylim_pad, (L - 1) / 2 + ylim_pad)

    # Not sure why I have to do this
    Y_rot = -Y_rot

    # Draw rotated heatmap
    im = ax.pcolormesh(X_rot, Y_rot, mat, cmap=cmap, norm=norm)

    # Remove spines
    for loc, spine in ax.spines.items():
        spine.set_visible(False)

    # Set sepline kwargs
    if show_seplines:
        if sepline_kwargs is None:
            sepline_kwargs = {
                "color": "gray",
                "linestyle": "-",
                "linewidth": 0.5,
            }

        # Draw white lines to separate position pairs
        for n in range(0, K + 1, C):
            # TODO: Change extent so these are the right length
            x = X_rot[n, :]
            y = Y_rot[n, :]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

            x = X_rot[:, n]
            y = Y_rot[:, n]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

    # Set lims
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set aspect
    ax.set_aspect("equal")

    # Remove yticks
    ax.set_yticks([])

    # Set xticks
    xticks = np.arange(L).astype(int)
    ax.set_xticks(xticks)

    # If drawing characters
    if show_alphabet:
        # Draw c1 alphabet
        for i, c in enumerate(alphabet):
            x1 = (
                0.5 * half_pixel_diag
                + i * half_pixel_diag
                - alphabet_pad * half_pixel_diag
            )
            y1 = (
                -0.5 * half_pixel_diag
                - i * half_pixel_diag
                - alphabet_pad * half_pixel_diag
            )
            ax.text(
                x1,
                y1,
                c,
                va="center",
                ha="center",
                rotation=-45,
                fontsize=alphabet_size,
            )

        # Draw c2 alphabet
        for i, c in enumerate(alphabet):
            x2 = (
                0.5
                + 0.5 * half_pixel_diag
                + i * half_pixel_diag
                + alphabet_pad * half_pixel_diag
            )
            y2 = (
                -0.5
                + 0.5 * half_pixel_diag
                + i * half_pixel_diag
                - alphabet_pad * half_pixel_diag
            )
            ax.text(
                x2,
                y2,
                c,
                va="center",
                ha="center",
                rotation=45,
                fontsize=alphabet_size,
            )

    # Display positions if requested (only if model is pairwise)
    l1_positions = np.arange(0, L - 1)
    l2_positions = np.arange(1, L)
    half_block_diag = C * half_pixel_diag
    if show_position and gpmap_type == "pairwise":
        # Draw l2 positions
        for i, l2 in enumerate(l2_positions):
            x2 = (
                0.5 * half_block_diag
                + i * half_block_diag
                - position_pad * half_pixel_diag
            )
            y2 = (
                0.5 * half_block_diag
                + i * half_block_diag
                + position_pad * half_pixel_diag
            )
            ax.text(
                x2,
                y2,
                f"{l2:d}",
                va="center",
                ha="center",
                rotation=45,
                fontsize=position_size,
            )

        # Draw l1 positions
        for i, l1 in enumerate(l1_positions):
            x1 = (
                (L - 0.5) * half_block_diag
                + i * half_block_diag
                + position_pad * half_pixel_diag
            )
            y1 = (
                (L - 1.5) * half_block_diag
                - i * half_block_diag
                + position_pad * half_pixel_diag
            )
            ax.text(
                x1,
                y1,
                f"{l1:d}",
                va="center",
                ha="center",
                rotation=-45,
                fontsize=position_size,
            )

    elif show_position and gpmap_type == "neighbor":
        # Draw l2 positions
        for i, l2 in enumerate(l2_positions):
            x2 = (
                0.5 * half_block_diag
                + 2 * i * half_block_diag
                - position_pad * half_pixel_diag
            )
            y2 = 0.5 * half_block_diag + position_pad * half_pixel_diag
            ax.text(
                x2,
                y2,
                f"{l2:d}",
                va="center",
                ha="center",
                rotation=45,
                fontsize=position_size,
            )

        # Draw l1 positions
        for i, l1 in enumerate(l1_positions):
            x1 = (
                1.5 * half_block_diag
                + 2 * i * half_block_diag
                + position_pad * half_pixel_diag
            )
            y1 = +0.5 * half_block_diag + position_pad * half_pixel_diag
            ax.text(
                x1,
                y1,
                f"{l1:d}",
                va="center",
                ha="center",
                rotation=-45,
                fontsize=position_size,
            )

    # Mark wt sequence
    if seq:
        # Set seq_kwargs if not set in constructor
        if seq_kwargs is None:
            seq_kwargs = {"marker": ".", "color": "k", "s": 2}

        # Iterate over pairs of positions
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                # Break out of loop if gmap_type is "neighbor" and l2 > l1+1
                if (l2 - l1 > 1) and gpmap_type == "neighbor":
                    continue

                # Iterate over pairs of characters
                for i1, c1 in enumerate(alphabet):
                    for i2, c2 in enumerate(alphabet):
                        # If there is a match to the wt sequence,
                        if seq[l1] == c1 and seq[l2] == c2:
                            # Compute coordinates of point
                            x = (
                                half_pixel_diag
                                + (i1 + i2) * half_pixel_diag
                                + (l1 + l2 - 1) * half_block_diag
                            )
                            y = (i2 - i1) * half_pixel_diag + (
                                l2 - l1 - 1
                            ) * half_block_diag

                            # Plot point
                            ax.scatter(x, y, **seq_kwargs)

    # Create colorbar if requested, make one
    if cbar:
        if cax is None:
            cax = make_axes_locatable(ax).new_horizontal(
                size=cmap_size, pad=cmap_pad
            )
            fig.add_axes(cax)
        cb = plt.colorbar(im, cax=cax, **cbar_kwargs)

        # Otherwise, return None for cb
    else:
        cb = None

    return ax, cb


if __name__ == "__main__":
    positions = [-13, -12, -11, -10, -9]
    alphabet = list("ACGU")
    pos_alleles = [f"{p}{a}" for p, a in product(range(5), alphabet)]

    print("Load gauge-fixed parameters")
    fpath = "manuscript/results/theta_fixed.aligned.csv"
    theta = pd.read_csv(fpath, index_col=0)
    theta0 = theta.loc[theta["k"] == 0, :].copy()

    theta_add = theta.loc[theta["k"] == 1, :].copy()
    theta_add["position"] = [positions[int(x[1])] for x in theta_add["orbit"]]
    theta_lc_positions = {
        position: pd.pivot(
            theta_add, index="subseq", columns="position", values=str(position)
        )
        for position in positions
    }

    theta_pw = theta.loc[theta["k"] == 2, :].copy()
    theta_pw["pos_allele_1"] = [
        orbit[1] + subseq[0]
        for orbit, subseq in theta_pw[["orbit", "subseq"]].values
    ]
    theta_pw["pos_allele_2"] = [
        orbit[4] + subseq[1]
        for orbit, subseq in theta_pw[["orbit", "subseq"]].values
    ]

    theta_lclc_positions = {
        position: pd.pivot(
            theta_pw,
            index="pos_allele_1",
            columns="pos_allele_2",
            values=str(position),
        )
        .reindex(pos_alleles)
        .T.reindex(pos_alleles)
        .T
        for position in positions
    }
    function = pd.read_csv("manuscript/data/shine_dalgarno.csv", index_col=0)[
        "f"
    ]

    # Plot theta_0
    if plot_theta_0:
        registers = [-13, -12, -11, -10, -9]
        mean_y = function.mean()
        cons_y = function.loc["AAGGAGGUG"]
        mean_color = "C3"
        cons_color = "C2"
        print("Plot constant parameters plot")
        fig, ax = plt.subplots(1, 1, figsize=(1.5, 0.9))
        means = theta0[[str(x) for x in positions]].values.flatten()
        ax.scatter(registers, means, c="black", s=7)
        ax.plot(registers, means, c="black", lw=0.9)
        ax.axhline(mean_y, c=mean_color, linestyle="--", lw=0.5)
        ax.axhline(cons_y, c=cons_color, linestyle="--", lw=0.5)
        ax.set(
            xticks=registers,  # Set ticks BEFORE ticklabels
            xticklabels=registers,
            ylim=(0.4, 2.75),
        )
        ax.text(x=-10, y=cons_y, s="wild type", fontsize=5, ha="left", va="bottom", color=cons_color)
        ax.text(x=-13, y=mean_y, s="average", fontsize=5, ha="left", va="bottom", color=mean_color)
        ax.tick_params(axis='both', which='major', length=2)
        ax.set_xlabel("register", labelpad=1)
        ax.set_ylabel("parameter", labelpad=1)
        fig.tight_layout(pad=0.2)
        fig.savefig("manuscript/figures/Figure1E.png", dpi=300, transparent=True)
        fig.savefig("manuscript/figures/Figure1E.pdf", transparent=True)


    # Plot theta_lc and theta_lclc
    if plot_theta_lc_lclc:
        fig, axes = plt.subplots(
            5,
            2,
            figsize=(2.75, 4.5),
            width_ratios=[1, 2.75],
        )

        print("Plot additive parameters heatmaps")
        for ax, (position, theta_lc) in zip(axes[:,0], theta_lc_positions.items()):
            # Invert the y-axis order so lower y values are at the top
            sns.heatmap(
                theta_lc.iloc[::-1, :],  # reverse rows for y-axis inversion
                ax=ax,
                cmap="coolwarm",
                center=0,
                cbar=False,
                vmin=-1.5,
                vmax=1.5,
            )
            xs = np.arange(5)
            ys = np.array([0, 2, 2, 0, 2])
            # Adjust scatter y positions to match inverted y-axis
            ax.scatter(xs + 0.5, (3 - ys) + 0.5, c="black", s=2)
            ax.set(
                ylabel=f"register {position}",
                xlabel="",
                aspect=.8,
                ylim=(0, 4),   # 4 rows (ACGU): still needed for axis scaling, y inverted by data order
                xlim=(0, 5),   # 5 columns (positions 1-5)
            )
            # Keep yticklabels in same logical order (e.g., ACGU mapped to 0,1,2,3 originally)
            # But now since the DataFrame is inverted, flip the order of ticklabels
            ax.set_yticklabels(list('ACGU')[::-1], rotation=0, fontsize=6)
            ax.set_xticklabels(np.arange(1, 6), rotation=90, fontsize=6)
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)
            sns.despine(ax=ax, top=False, right=False)
        axes[-1, 0].set(xlabel="position in core")

        print("Plot pairwise parameters heatmap")
        for ax, (position, theta_lclc) in zip(
            axes[:, 1], theta_lclc_positions.items()
        ):
            theta_lclc = theta_lclc.to_numpy().reshape(5, 4, 5, 4)
            ax, cbar = heatmap_pairwise(
                theta_lclc,
                alphabet=alphabet,
                seq="AGGAG",
                ax=ax,
                clim=(-1.5, 1.5),
                ccenter=0,
                sepline_kwargs={"lw": 0.5, "c": "black"},
                show_alphabet=True,
                show_position=False,
                alphabet_size=4,
                cmap_size="4%",
            )
            cbar.set_ticks([-1.0, 0.0, 1.0], 
                           labels=["-1", "0", "1"], 
                           size=0,  # Turn off major tick length
                           minor=False, 
                           fontsize=6)
            # Set cbar tick_params to half the default length (default is 4.0, so use 2.0)
            cbar.ax.tick_params(length=2)  # halve the tick length
            cbar.set_label(r"parameter")
            # Set tick parameters to draw ticks inside and in white
            #cbar.ax.tick_params(direction='in', colors='white')
            #cbar.ax.tick_labels([f"{x:.1f}" for x in [-1.0, 0.0, 1.0]],)
            #ax.set(xticks=[])
            ax.set_xticklabels(np.arange(1, 6))
            ax.tick_params(axis='x', length=0, rotation=90, pad=0.5)
            
        axes[-1, 1].set(xlabel="position in core")
        # axes[-1, 1].set(
        #     xlabel="core position",
        #     xticks=np.arange(5),
        #     xticklabels=1 + np.arange(5),
        # )
        # axes[-1, 1].tick_params(axis='x', length=0, rotation=90)

        fig.tight_layout(h_pad=0.5, w_pad=1.0, pad=0.5)
        #plt.show()
        fig.savefig("manuscript/figures/Figure1FG.png", dpi=300, transparent=True)
        fig.savefig("manuscript/figures/Figure1FG.pdf", transparent=True)
        
        
    if plot_theta_scatter:
        fig, axes = plt.subplots(
            5,
            2,
            figsize=(1.9, 4.5),
            width_ratios=[1, 1],
            sharey=True,
            sharex=True
        )

        print("Plotting comparison of coefficients")
        positions_str = [str(x) for x in positions]
        pos_pairs = list(combinations(positions_str, 2))
        # pos_pairs = sorted(
        #     combinations(positions_str, 2),
        #     key=lambda x: (np.abs(int(x[0]) - int(x[1])), int(x[0])),
        # )
        theta_up_to_pw = theta.loc[theta["k"] <= 2, :]
        idx = ~np.all(theta_up_to_pw[positions_str].values == 0.0, axis=1)
        theta_up_to_pw = theta_up_to_pw.loc[idx, :]

        lims = (-2.5, 2.5)
        sizes = [6, 4, 2]
        lables = ["constant", "additive", "pairwise"]
        palette = {0: "C6", 1: "C8", 2: "C9"}
        
        
        ax_coords = [(i, j) for i in range(axes.shape[0]) for j in range(axes.shape[1])]
        
        for (p, q), (i, j) in zip(pos_pairs, ax_coords):           
            ax = axes[i, j]
            for k, theta_k in theta_up_to_pw.groupby("k"):
                ax.scatter(
                    theta_k[p],
                    theta_k[q],
                    c=palette[k],
                    s=sizes[k],
                    lw=0,
                    label=lables[k],
                )
            ax.axline((0, 0), slope=1, lw=0.5, linestyle="--", c="grey", zorder=-100)
            ax.set(aspect="equal", xlim=lims, ylim=lims)
            ax.set_xticks([-2,0,2])
            ax.set_yticks([-2,0,2])
            ax.tick_params(axis='both', which='major', length=2)
            ax.tick_params(axis='both', which='minor', length=2)

            if i == 4:
                ax.set_xlabel(f"1st register", labelpad=1)
                ax.set_xticklabels(["-2","0","2"])
            if j == 0:
                ax.set_ylabel(f"2nd register", labelpad=0.5)
                ax.set_yticklabels(["-2","0","2"])
            ax.text(
                0.05,
                0.95,
                f"{p} vs {q}",
                fontsize=6,
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
            r = pearsonr(theta_up_to_pw[p], theta_up_to_pw[q])[0]
            ax.text(
                0.95,
                0.05,
                "r={:.2f}".format(r),
                fontsize=6,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )
            
            # # If p and q differ by 1, make bounding box of axes have linewidth 2
            # if abs(int(p) - int(q)) == 1:
            #     for spine in ax.spines.values():
            #         spine.set_linewidth(2)
            
        handles, labels = ax.get_legend_handles_labels()
        
        # for ax in axes[:, -1]:
        #     ax.set(yticklabels=[])

        fig.tight_layout(h_pad=0.5, w_pad=0.5, pad=0.5)
        fig.subplots_adjust(top=0.95)
        
        fig.legend(
            handles,
            labels,
            loc="upper center",
            #bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            fontsize=5,
            frameon=True,
            handletextpad=0.2,           # reduce space between symbol and text
        )
        
        #plt.show()
        fig.savefig("manuscript/figures/Figure1H.png", dpi=300, transparent=True)
        fig.savefig("manuscript/figures/Figure1H.pdf", transparent=True)
