import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use("manuscript/code/gaugefixer.mplstyle")


if __name__ == "__main__":
    print("Loading benchmarking results")
    times = pd.read_csv("manuscript/results/benchmarking.csv", index_col=0)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(4, 1.8),
        sharex=True,
        sharey=False,
    )
    models = ['AllOrderModel', 'PairwiseModel']
    colors = ["C0", "C1"]
    palette = dict(zip(models, colors))
    variables = {
        "time": "runtime (s)",
        "peak_memory": "memory (MB)",
    }
    
    print("Plotting gauge-fixing running times and peak memory usage")
    for i, (axes, (y, ylabel)) in enumerate(zip(subplots, variables.items())):
        data = times.loc[times['dense_matrix'], :].copy()
        labels = ['all-order (standard)', 'pairwise (standard)']
        labels_dict = dict(zip(models, labels))
        data['label'] = [labels_dict[x] for x in data['model']]
        palette = dict(zip(labels, colors))
        sns.lineplot(
            x="n_features",
            y=y,
            hue="label",
            data=data,
            ax=axes,
            palette=palette,
            errorbar="sd",
            err_style="bars",
            err_kws={"capsize": 0.75, "elinewidth": 0.75, "capthick": 0.75},
            linestyle='--',
            lw=0.75,
        )
        data = times.loc[~times['dense_matrix'], :].copy()
        labels = ['all-order (GaugeFixer)', 'pairwise (GaugeFixer)']
        labels_dict = dict(zip(models, labels))
        data['label'] = [labels_dict[x] for x in data['model']]
        palette = dict(zip(labels, colors))
        sns.lineplot(
            x="n_features",
            y=y,
            hue="label",
            data=data,
            ax=axes,
            palette=palette,
            errorbar="sd",
            err_style="bars",
            err_kws={"capsize": 0.75, "elinewidth": 0.75, "capthick": 0.75},
            lw=0.75,
            linestyle='solid',
        )
        
        if i == 0:
            yticks=(1E-5,1E-4,1E-3,1E-2,1E-1,1E0)
            ylim=(3E-6, 3E0)
        else:
            yticks=(1E-3,1E-2,1E-1,1E0,1E1,1E2,1E3)
            ylim=(3E-4, 3E3)
        
        axes.set(
            xlabel="num parameters",
            ylabel=ylabel,
            xscale="log",
            yscale="log",
            xlim=(90, 2e7),
            ylim=ylim,
            xticks=(1E2,1E3,1E4,1E5,1E6,1E7),
            yticks=yticks,
            ymargin=0.2,
        )
        from matplotlib.ticker import LogLocator, NullFormatter

        # Set log-scale minor tickmarks
        axes.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.legend_.set_visible(False)
        axes.legend(loc=4)

    fig.tight_layout(pad=0.2, w_pad=1)
    
    print("Saving plots")
    fig.savefig("manuscript/figures/Figure1BC.png", dpi=300, transparent=True)
    fig.savefig("manuscript/figures/Figure1BC.pdf", transparent=True)
    print("Done")