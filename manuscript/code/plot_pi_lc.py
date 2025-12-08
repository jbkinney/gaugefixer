import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("manuscript/code/gaugefixer.mplstyle")

plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['xtick.major.pad'] = 1.5

if __name__ == "__main__":
    positions = [-13, -12, -11, -10, -9]
    alphabet = list("ACGU")

    fig, subplots = plt.subplots(
        5,
        1,
        figsize=(1.5, 3.4),
    )

    print("Plot probability distributions")
    pi_uniform = [0.25 * np.ones(4)]
    pi_motif = [
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
    ]
    for axes, (p, position) in zip(subplots, enumerate(positions)):
        pi_lc = pi_uniform * p + pi_motif + pi_uniform * (4 - p)
        pi_lc = pd.DataFrame(pi_lc, columns=alphabet)
        logo = logomaker.Logo(pi_lc, ax=axes, show_spines=False, baseline_width=0)
        
        # Add bracket and "Core" label above the AGGAG motif
        bracket_y = 1.15
        text_y = 1.15
        x_left = p - 0.5
        x_right = p + 4 + 0.5

        # Draw bracket: horizontal line with small vertical ends
        axes.plot([x_left, x_left, x_right, x_right], 
                  [bracket_y - 0.08, bracket_y, bracket_y, bracket_y - 0.08], 
                  color='black', linewidth=0.5, clip_on=False)

        # Add "Core" label
        axes.text((x_left + x_right) / 2, text_y, 'core', 
                  ha='center', va='bottom', fontsize=7, clip_on=False)

        axes.set(
            ylim=(0, 1),
            yticks=[], #(0, 1),
            yticklabels=[], #["0", "1"],
            ylabel=f"register {position}",
            xticks=np.arange(9),
            xticklabels=np.arange(-13, -4),
        )
        axes.tick_params(axis='x', rotation=90, length=0, pad=1.5)  # xtick marks of length 0
        axes.tick_params(axis='y', length=2)
        if position == -9:  # Use 'position' instead of 'p'
            axes.set_xlabel("position relative to AUG", labelpad=1)
        # Hide the x-axis line but keep the xtick labels
        axes.spines['bottom'].set_visible(False)

        for p in list(range(p)) + list(range(p + 5, 9)):
            for c in alphabet:
                logo.style_single_glyph(p, c, alpha=0.3)

    fig.tight_layout(h_pad=1, w_pad=0)
    fig.savefig("manuscript/figures/Figure1D.pdf", transparent=True)
