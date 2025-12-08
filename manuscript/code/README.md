# GaugeFixer: Overcoming Non-identifiability in Sequence-to-Function Models

In this folder, we provide code and data necessary to reproduce the analysis performed in the manuscript describing GaugeFixer.

### Requirements

The following Python libraries are required for reproducing the results:

- gaugefixer
- seaborn
- matplotlib
- logomaker

### Benchmarking analysis

First, we compute the memory and computational requirements and scaling with the number of parameters for the different models:

```bash
python manuscript/code/benchmark.py
```

This will create a file with the running times and memory requirements over 10 different iterations for each model configuration. 

To make the figures representing these computational requirements, run

```bash
python manuscript/code/plot_benchmark.py
```

### Shine-Dalgarno sequence analysis

Here we use `GaugeFixer` to analyze the fitness landscape of the Shine-Dalgarno sequence.

We first compute the gauge-fixed parameters in the different hierarhical gauges of interest by running

```bash
python manuscript/code/fix_gauge.py
```

Then, we make the plots corresponding to the sequence probability distributions that represent the different peaks in the landscape and plot the gauge-fixed coefficients in the hierarchical gauges defined by them:

```bash
python manuscript/code/plot_pi_lc.py
python manuscript/code/plot_theta.py
```

### Output

Expected output files:

Results tables

- `manuscript/results/benchmark.csv`: this file contains the results of the benchmarking analysis.
- `manuscript/results/theta_fixed.aligned.csv`: this file contains the values of the gauge-fixed parameters aligned by the core positions.

Figure panels

- `manuscript/figures/Figure1BC.svg`: benchmarking results. 
- `manuscript/figures/Figure1D.svg`: logo plots representing site-independent probability distributions.
- `manuscript/figures/Figure1E.svg`: panels representing the gauge-fixed constant parameters in each of the corresponding hierarchical gauges. 
- `manuscript/figures/Figure1FG.svg`: panels representing the gauge-fixed additive and pairwise parameters in each of the corresponding hierarchical gauges. 
- `manuscript/figures/Figure1H.svg`: panels comparing the parameters of local pairwise models in each of the corresponding hierarchical gauges. 

