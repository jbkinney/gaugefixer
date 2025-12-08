import tracemalloc
from time import time

import numpy as np
import pandas as pd

from gaugefixer import AllOrderModel, PairwiseModel

if __name__ == "__main__":
    results = []
    n_points = 12
    alphabet_name = "protein"
    models = [
        (AllOrderModel, range(2, 6)),
        (PairwiseModel, np.geomspace(3, 150, n_points).astype(int)),
    ]

    print(f"Computing matrix-vector products in {alphabet_name} space")
    for model, seq_lengths in models:
        print(model)
        for L in seq_lengths:
            print("\tSequence length = {}".format(L))
            m = model(alphabet_name=alphabet_name, L=L)
            m.set_random_params()
            
            for use_dense_matrix in [False, True]:
                if use_dense_matrix and m.n_features > 10000:
                    continue

                for i in range(11):
                    tracemalloc.start()
                    current1, peak1 = tracemalloc.get_traced_memory()
                    t0 = time()
                    theta_fixed = m.get_fixed_params(
                        gauge="zero-sum", use_dense_matrix=use_dense_matrix
                    )
                    t1 = time() - t0
                    current2, peak2 = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    # Skip the first run (warm-up)
                    if i == 0:
                        continue

                    results.append(
                        {
                            "model": model.__name__,
                            "n_features": m.n_features,
                            "alphabet_name": alphabet_name,
                            "time": t1,
                            "current_memory": (current2 - current1) / 1e6,
                            "peak_memory": (peak2 - peak1) / 1e6,
                            "dense_matrix": use_dense_matrix,
                        }
                    )

    results = pd.DataFrame(results)
    results.to_csv("manuscript/results/benchmarking.csv")
