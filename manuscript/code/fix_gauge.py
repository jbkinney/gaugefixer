import numpy as np
import pandas as pd

from gaugefixer import AllOrderModel
from gaugefixer.utils import get_subsets_of_set, get_orbits_features


if __name__ == "__main__":
    print("Loading Shine-Dalgarno landscape")
    function = pd.read_csv("manuscript/data/shine_dalgarno.csv", index_col=0)

    print("Initializing AllOrderModel")
    model = AllOrderModel(L=9, alphabet_name="rna")
    model.set_landscape(function["f"])

    pi_uniform = [0.25 * np.ones(4)]
    pi_motif = [
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 1, 0]),
    ]

    print("Fixing the gauge around AGGAG motif at each position")
    thetas = {}
    positions = [-13, -12, -11, -10, -9]
    for p, position in enumerate(positions):
        print(f"\tPosition {p}")
        pi_lc = pi_uniform * p + pi_motif + pi_uniform * (4 - p)
        theta_fixed = model.get_fixed_params(gauge="hierarchical", pi_lc=pi_lc)
        thetas[position] = theta_fixed
    thetas = pd.DataFrame(thetas)
    thetas["orbit"] = [x[0] for x in thetas.index]
    thetas["subseq"] = [x[1] for x in thetas.index]

    print("Aligning local models around each register")
    theta_ps = {}
    orbits = get_subsets_of_set((0, 1, 2, 3, 4))
    features = get_orbits_features(orbits, model.alphabet_list)
    for p, position in enumerate(positions):
        features_p = [
            (tuple(x + p for x in orbit), subseq) for orbit, subseq in features
        ]
        theta_p = thetas.loc[features_p, position]  # type: ignore
        theta_p.index = features
        theta_ps[position] = theta_p
    theta_ps = pd.DataFrame(theta_ps)
    theta_ps["orbit"] = [x[0] for x in theta_ps.index]
    theta_ps["subseq"] = [x[1] for x in theta_ps.index]
    theta_ps["k"] = [len(x) for x in theta_ps["subseq"]]
    theta_ps.to_csv("manuscript/results/theta_fixed.aligned.csv")
