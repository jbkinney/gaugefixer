import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_model(
    model,
    jitter_amp=0.3,
    dy=0.5,
    linewidth=0.5,
    linealpha=0.5,
    ax=None,
    figsize=(6, 3),
    fontsize=8,
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    show_legend=True,
    legend_loc="upper right",
    legend_fontsize=None,
    dotsize=20,
):
    """
    Visualize model orbits showing their order and contributing positions.

    Parameters
    ----------
    model : HierarchicalModel
        The model instance to visualize.
    jitter_amp : float, default=0.3
        Amplitude of vertical jitter for overlapping orbit dots
    dy : float, default=0.5
        Spacing parameter for vertical jitter
    linewidth : float, default=0.5
        Width of the connecting lines
    linealpha : float, default=0.5
        Alpha/transparency of the connecting lines
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    figsize : tuple, default=(8,4)
        Figure size (width, height) in inches if creating new figure
    fontsize : int, default=8
        Default font size for all text elements
    axis_label_fontsize : int, optional
        Font size for axis labels. If None, uses fontsize.
    tick_label_fontsize : int, optional
        Font size for tick labels. If None, uses fontsize.
    show_legend : bool, default=True
        Whether to show the legend
    legend_loc : str, default='upper right'
        Location of the legend
    legend_fontsize : int, optional
        Font size for legend text. If None, uses fontsize.
    dotsize : int, default=50
        Size of the dots in the plot
    """

    # Set font sizes with fallbacks
    if axis_label_fontsize is None:
        axis_label_fontsize = fontsize
    if tick_label_fontsize is None:
        tick_label_fontsize = fontsize
    if legend_fontsize is None:
        legend_fontsize = fontsize

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot position dots on x-axis
    ax.scatter(
        model.positions,
        [0] * len(model.positions),
        facecolor="white",
        edgecolor="black",
        s=dotsize,
        zorder=10,
    )

    # Get unique orbit orders and create color map
    unique_orders = sorted(
        set(len(orbit) for orbit in model.orbits if len(orbit) > 0)
    )
    order_to_color = dict(
        zip(
            unique_orders,
            [f"C{i % 10}" for i in range(len(unique_orders))],
        )
    )

    # Group orbits by order and center position
    orbit_groups = {}
    for orbit in model.orbits:
        if len(orbit) > 0:
            center_pos = sum(orbit) / len(orbit)
            order = len(orbit)
            key = (order, center_pos)
            if key not in orbit_groups:
                orbit_groups[key] = []
            orbit_groups[key].append(orbit)

    # Compute orbit counts by order
    n_orbits_by_order = {}
    for (order, center_pos), orbits in orbit_groups.items():
        n_orbits_by_order[order] = n_orbits_by_order.get(order, 0) + len(
            orbits
        )

    # Track which orders have been added to legend
    orders_in_legend = set()

    # Plot orbits with jitter
    for (order, center_pos), orbits in orbit_groups.items():
        color = order_to_color[order]
        K = len(orbits)

        for k, orbit in enumerate(orbits):
            # Calculate jittered y-position
            jittered_y = order + jitter_amp * np.tanh((k - (K - 1) / 2) * dy)

            # Plot orbit dot with label for legend (only for first occurrence of each order)
            if order not in orders_in_legend:
                label = f"order {order}: {n_orbits_by_order[order]} orbits"
                orders_in_legend.add(order)
            else:
                label = None

            ax.scatter(
                center_pos,
                jittered_y,
                color=color,
                s=dotsize,
                zorder=5,
                label=label,
            )

            # Draw lines to contributing positions
            for pos in orbit:
                ax.plot(
                    [center_pos, pos],
                    [jittered_y, 0],
                    color=color,
                    alpha=linealpha,
                    linewidth=linewidth,
                    zorder=1,
                )

    # Customize plot appearance
    ax.set_xlabel("position", fontsize=axis_label_fontsize)
    ax.set_ylabel("order", fontsize=axis_label_fontsize)

    # Set y-axis limits with some padding
    ax.set_ylim(-0.2, max(len(orbit) for orbit in model.orbits) + 0.5)
    ax.yaxis.set_ticks(range(1, max(len(orbit) for orbit in model.orbits) + 1))

    # Remove axes lines except y-axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Remove x-axis ticks but keep labels
    ax.tick_params(axis="x", length=0, labelsize=tick_label_fontsize)
    ax.set_xticks(model.positions)
    ax.set_xticklabels(model.positions)

    # Set tick label font size for y-axis
    ax.tick_params(axis="y", labelsize=tick_label_fontsize)

    # Add legend if requested
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

    plt.tight_layout()

    # Only show if we created a new figure
    if ax is None:
        plt.show()

