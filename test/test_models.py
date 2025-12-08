#!/usr/bin/env python
import unittest
from itertools import combinations, product

import numpy as np
import pandas as pd

from gaugefixer.models import (
    AdditiveModel,
    AllOrderModel,
    KadjacentModel,
    KorderModel,
    HierarchicalModel,
    NeighborModel,
    PairwiseModel,
)


class TestModels(unittest.TestCase):
    def setUp(self):
        model = AllOrderModel(alphabet_name="dna", L=3)
        model.set_random_params()
        self.model = model
        self.seqs = ["AAA", "AAC", "AAG", "AAT", "CCA", "CCG", "TTT"]
        self.all_seqs = [''.join(x) for x in product(list('ACGT'), repeat=3)]

    def test_model(self):
        """Test model initialization."""
        model = AllOrderModel(alphabet_name="dna", L=3)

        # Check basic attributes
        self.assertEqual(model.L, 3)
        self.assertEqual(model.alphabet, ["A", "C", "G", "T"])
        self.assertEqual(model.alpha, 4)
        self.assertEqual(model.positions, [0, 1, 2])

        # Check orbits
        self.assertEqual(len(model.orbits), 2**model.L)

        # Check that features are created
        self.assertGreater(len(model.features), 0)
        self.assertGreater(model.n_features, 0)
        self.assertEqual(model.n_features, 5**model.L)

    def test_model_generatin_orbits(self):
        """Test model initialization with custom orbits"""
        orbits = [(0, 1)]
        model = HierarchicalModel(
            alphabet_name="dna", L=3, generating_orbits=orbits
        )

        # Check basic attributes
        self.assertEqual(model.L, 3)
        self.assertEqual(model.alphabet, ["A", "C", "G", "T"])
        self.assertEqual(model.alpha, 4)
        self.assertEqual(model.positions, [0, 1, 2])

        # Check orbits
        self.assertEqual(len(model.orbits), 4)

        # Check that features are created
        self.assertGreater(len(model.features), 0)
        self.assertGreater(model.n_features, 0)
        self.assertEqual(model.n_features, 5**2)

    def test_orbits_with_positions(self):
        """Test orbits with positions"""
        positions = [-2, -1, 0, 1, 2]
        orbits = [(-2, -1), (0,), (1, 2)]

        model = HierarchicalModel(
            alphabet_name="dna",
            L=len(positions),
            positions=positions,
            generating_orbits=orbits,
        )
        self.assertGreater(model.n_orbits, len(orbits))
        self.assertTrue(set(orbits).issubset(set(model.get_orbits())))

    def test_models_orbits(self):
        """Test get_orbits_for_allorder_model function."""
        L = 3
        models = [
            AllOrderModel(alphabet_name="dna", L=L),
            AdditiveModel(alphabet_name="dna", L=L),
            PairwiseModel(alphabet_name="dna", L=L),
            KorderModel(alphabet_name="dna", L=L, K=3),
            NeighborModel(alphabet_name="dna", L=L),
            KadjacentModel(alphabet_name="dna", L=L, K=3),
        ]
        expected = [
            [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)],
            [(), (0,), (1,), (2,)],
            [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)],
            [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)],
            [(), (0,), (1,), (2,), (0, 1), (1, 2)],
            [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)],
        ]

        for model, exp in zip(models, expected):
            self.assertEqual(set(model.orbits), set(exp))

        L = 4
        models = [
            AllOrderModel(alphabet_name="dna", L=L),
            AdditiveModel(alphabet_name="dna", L=L),
            PairwiseModel(alphabet_name="dna", L=L),
            KorderModel(alphabet_name="dna", L=L, K=3),
            NeighborModel(alphabet_name="dna", L=L),
            KadjacentModel(alphabet_name="dna", L=L, K=3),
            HierarchicalModel(
                alphabet_name="dna", L=L, generating_orbits=[(0, 1), (2, 3)]
            ),
        ]
        expected = [
            [
                (),
                (0,),
                (1,),
                (2,),
                (3,),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
                (0, 1, 2, 3),
            ],
            [(), (0,), (1,), (2,), (3,)],
            [
                (),
                (0,),
                (1,),
                (2,),
                (3,),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            ],
            [
                (),
                (0,),
                (1,),
                (2,),
                (3,),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
            ],
            [(), (0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3)],
            [
                (),
                (0,),
                (1,),
                (2,),
                (3,),
                (0, 1),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 3),
                (0, 1, 2),
                (1, 2, 3),
            ],
            [(), (0,), (1,), (2,), (3,), (0, 1), (2, 3)],
        ]
        for model, exp in zip(models, expected):
            self.assertGreater(model.n_orbits, 0)
            self.assertIsNotNone(model.orbits)
            self.assertEqual(set(model.orbits), set(exp))

    def test_orbits_wong2018_example(self):
        """Test orbits with the Wong2018 example from the notebook."""
        rna_alphabet = ["A", "C", "G", "U"]
        positions = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        generating_orbits = [(-4, -3, -2, -1), (0, 1, 2, 3, 4, 5), (-1, 0, 1)]
        alphabet_list = (
            [rna_alphabet] * 4 + [["G"]] + [["C", "U"]] + [rna_alphabet] * 4
        )

        model = HierarchicalModel(
            alphabet_list=alphabet_list,
            generating_orbits=generating_orbits,
            positions=positions,
        )

        # Check basic properties
        self.assertEqual(model.positions, positions)
        self.assertEqual(model.L, 10)  # Length of positions

        # Check that orbits are created
        self.assertIsNotNone(model.orbits)
        self.assertGreater(len(model.orbits), 0)
        self.assertGreater(model.n_orbits, len(generating_orbits))

        # Check that features are created
        self.assertGreater(model.n_features, 0)
        self.assertIsNotNone(model.features)

    def test_orbits_with_different_alphabets(self):
        """Test orbits with position-specific alphabets."""
        positions = [0, 1, 2]
        generating_orbits = [(0,), (1, 2)]
        alphabet_list = [["A", "C"], ["G", "T"], ["A", "G", "T"]]

        model = HierarchicalModel(
            alphabet_list=alphabet_list,
            positions=positions,
            generating_orbits=generating_orbits,
        )

        # Check basic properties
        self.assertEqual(model.L, 3)
        self.assertEqual(model.alphabet_list, alphabet_list)

        # Check that features are created correctly
        features = model.get_features()
        self.assertGreater(len(features), 0)

        # Check that subsequences use correct alphabets
        for orbit, subsequence in features:
            if len(orbit) > 0:
                for i, pos in enumerate(orbit):
                    pos_idx = positions.index(pos)
                    self.assertIn(subsequence[i], alphabet_list[pos_idx])

    def test_orbits_validation(self):
        """Test validation of generating_orbits parameter."""
        with self.assertRaises(KeyError):
            HierarchicalModel(
                alphabet_name="dna",
                L=4,
                positions=[0, 1, 2, 3],
                generating_orbits=[(0, 1, 5)],  # 5 is not in positions
            )

    def test_orbits_without_positions(self):
        """Test that orbits works with default positions when positions not provided."""
        model = HierarchicalModel(
            alphabet=["A", "C", "G", "T"],
            L=2,
            generating_orbits=[(0, 1)],
        )

        # Check that default positions are used
        self.assertEqual(model.positions, [0, 1])
        self.assertEqual(model.orbits, [(), (0,), (1,), (0, 1)])

        # Check that orbits are generated correctly
        orbits = model.get_orbits()
        self.assertGreater(len(orbits), 0)

    def test_orbits_consistency_with_features(self):
        """Test that orbits are consistent with generated features."""
        positions_list = [[0, 1, 2, 3], [0, 5, 6, 10]]
        orbits_list = [[(0, 1), (2, 3)], [(0, 5), (6, 10)]]

        for positions, generating_orbits in zip(positions_list, orbits_list):
            model = HierarchicalModel(
                alphabet=["A", "C", "G", "T"],
                L=len(positions),
                positions=positions,
                generating_orbits=generating_orbits,
            )

            features = model.get_features()
            orbits = model.get_orbits()

            # Check that all feature orbits are in the generated orbits
            feature_orbits = set(orbit for orbit, _ in features)
            orbit_set = set(orbits)

            # All feature orbits should be in the generated orbits
            self.assertTrue(feature_orbits.issubset(orbit_set))

            # Check that subsequences match orbit lengths
            for orbit, subsequence in features:
                self.assertEqual(len(subsequence), len(orbit))

    def test_valid_K_order_model(self):
        """Test that invalid K raises ValueError."""

        for kmodel in [KorderModel, KadjacentModel]:
            with self.assertRaises(ValueError):
                kmodel(alphabet_name="dna", L=3, K=0)

            with self.assertRaises(ValueError):
                kmodel(alphabet_name="dna", L=3, K=-1)

            with self.assertRaises(ValueError):
                kmodel(alphabet_name="dna", L=3, K=4)

    def test_set_params(self):
        """Test setting theta values"""
        model = AllOrderModel(alphabet_name="dna", L=3)
        theta = np.random.normal(size=model.n_features)
        theta = pd.Series(theta, index=model.features)
        model.set_params(theta)

        with self.assertRaises(ValueError):
            model.set_params(theta.iloc[1:])

        with self.assertRaises(ValueError):
            theta = np.random.normal(size=model.n_features + 1)
            theta = pd.Series(theta, index=model.features + [((-1,), "A")])
            model.set_params(theta)

        with self.assertRaises(KeyError):
            theta = np.random.normal(size=model.n_features)
            theta = pd.Series(
                theta, index=model.features[:-1] + [((-1,), "A")]
            )
            model.set_params(theta)

        with self.assertRaises(ValueError):
            theta = np.random.normal(size=model.n_features)
            theta[0] = np.nan
            theta = pd.Series(theta, index=model.features)
            model.set_params(theta)

        with self.assertRaises(ValueError):
            theta = np.random.normal(size=model.n_features)
            theta[0] = np.inf
            theta = pd.Series(theta, index=model.features)
            model.set_params(theta)

    def test_set_landscape(self):
        """Test setting function values"""
        model = self.model
        seqs = self.all_seqs
        f = np.random.normal(size=model.size)
        f = pd.Series(f, index=seqs)
        model.set_landscape(f)
        assert np.allclose(model(seqs), f)

        with self.assertRaises(ValueError):
            f = np.random.normal(size=model.size + 1)
            f = pd.Series(f, index=seqs + ["AAH"])
            model.set_landscape(f)

        with self.assertRaises(ValueError):
            f = np.random.normal(size=model.size)
            f[0] = np.nan
            f = pd.Series(f, index=seqs)
            model.set_landscape(f)

        with self.assertRaises(ValueError):
            seqs[0] = "AAH"
            f = np.random.normal(size=model.size)
            f = pd.Series(f, index=seqs)
            model.set_landscape(f)

    def test_model_values(self):
        """Test function calculation"""
        f = self.model(self.seqs)
        x = self.model._encode_seqs(self.seqs)
        self.assertEqual(f.shape[0], len(self.seqs))
        self.assertFalse(np.any(np.isnan(f)))
        self.assertFalse(np.any(np.isinf(f)))
        assert np.allclose(f, np.dot(x, self.model.get_params()))

    def test_model_values_additive(self):
        """Test function calculation in additive model"""
        model = AdditiveModel(alphabet=list("AB"), L=3)
        theta = np.array([0.0, 1, 0, 1, 0, 1, 0])
        theta = pd.Series(theta, index=model.features)
        model.set_params(theta)

        seqs = ["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"]
        expected = np.array([3, 2, 2, 1, 2, 1, 1, 0.0])
        values = model(seqs)
        assert np.allclose(values, expected)
        self.assertIsInstance(values, np.ndarray)

    def test_model_values_different_models(self):
        """Test score_seqs works with different model types."""
        L = 3
        models = [
            AllOrderModel(alphabet_name="dna", L=L),
            AdditiveModel(alphabet_name="dna", L=L),
            PairwiseModel(alphabet_name="dna", L=L),
            KorderModel(alphabet_name="dna", L=L, K=3),
            NeighborModel(alphabet_name="dna", L=L),
            KadjacentModel(alphabet_name="dna", L=L, K=3),
        ]
        for model in models:
            theta = np.random.normal(size=model.n_features)
            theta = pd.Series(theta, index=model.features)
            model.set_params(theta)

        for m1, m2 in combinations(models, 2):
            f1, f2 = m1(self.seqs), m2(self.seqs)
            self.assertEqual(f1.shape, f2.shape)
            self.assertEqual(f1.dtype, np.float64)
            self.assertEqual(f2.dtype, np.float64)
            self.assertFalse(np.allclose(f1, f2))

    def test_model_values_invalid_sequence(self):
        """Test that invalid sequence raises ValueError."""
        with self.assertRaises(ValueError):
            self.model(["AA"])

        with self.assertRaises(ValueError):
            self.model(["ACB"])

        with self.assertRaises(ValueError):
            self.model([])

        with self.assertRaises(ValueError):
            self.model(["AAAA"])


class TestGaugeFixing(unittest.TestCase):
    def test_get_fixed_params(self):
        """Test gauge fixing"""
        model = AllOrderModel(alphabet_name="dna", L=3)
        model.set_random_params()
        seqs = ["AAA", "AAC", "AAG", "AAT", "CCA", "CCG", "TTT"]
        f1 = model(seqs)

        pi_lc = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.4, 0.3, 0.2, 0.1]),
        ]
        gauges = [
            {"gauge": "zero-sum"},
            {"gauge": "wild-type", "wt_seq": "AAA"},
            {"gauge": "trivial"},
            {"gauge": "euclidean"},
            {"gauge": "equitable"},
            {"gauge": "hierarchical", "pi_lc": pi_lc},
            {"lda": 10.0, "pi_lc": pi_lc},
        ]
        for kwargs in gauges:
            theta_fixed = model.get_fixed_params(**kwargs)
            model.set_params(theta_fixed)
            f2 = model(seqs)
            assert np.allclose(f1, f2)

    def test_get_fixed_params_dense_matrix(self):
        """Test gauge fixing"""
        model = PairwiseModel(alphabet_name="dna", L=4)
        model.set_random_params()
        theta_original = model.theta.copy()
        theta_fixed1 = model.get_fixed_params(
            gauge="zero-sum", use_dense_matrix=True
        )
        model.set_params(theta_original)
        theta_fixed2 = model.get_fixed_params(
            gauge="zero-sum", use_dense_matrix=False
        )
        assert np.allclose(theta_fixed1, theta_fixed2)

    def test_get_fixed_params_hierarchical_models(self):
        """Test gauge fixing in hierarchical models"""
        L = 4
        models = [
            AdditiveModel(alphabet_name="dna", L=L),
            AllOrderModel(alphabet_name="dna", L=L),
            PairwiseModel(alphabet_name="dna", L=L),
            KorderModel(alphabet_name="dna", L=L, K=3),
            NeighborModel(alphabet_name="dna", L=L),
            KadjacentModel(alphabet_name="dna", L=L, K=3),
        ]
        pi_lc = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        gauges = [
            {"gauge": "zero-sum"},
            {"gauge": "wild-type", "wt_seq": "AGAA"},
            {"gauge": "hierarchical", "pi_lc": pi_lc},
        ]
        seqs = ["AAAC", "AACG", "GAAG", "TAAT", "TCCA", "ACCG", "TATT"]

        for model in models:
            model.set_random_params()
            theta = model.theta.copy()
            f1 = model(seqs)

            for kwargs in gauges:
                theta_fixed = model.get_fixed_params(**kwargs)
                model.set_params(theta_fixed)
                f2 = model(seqs)
                theta_fixed_fixed = model.get_fixed_params(**kwargs)
                assert np.allclose(theta_fixed, theta_fixed_fixed)
                assert np.allclose(f1, f2)
                assert np.logical_not(np.allclose(theta, theta_fixed))

    def test_alphabets_get_fixed_params(self):
        """Test gauge fixing with different alphabets"""
        self.alphabets = [
            {"alphabet": list("AB"), "L": 3},
            {"alphabet": list("ABC"), "L": 3},
            {"alphabet_list": [list("ABC"), list("ABCD"), list("AB")]},
            {"alphabet_name": "dna", "L": 3},
            {"alphabet_name": "DNA", "L": 3},
            {"alphabet_name": "protein", "L": 3},
            {"alphabet_name": "binary", "L": 3},
        ]
        for kwargs in self.alphabets:
            model = AllOrderModel(**kwargs)
            model.set_random_params()
            theta = model.theta.copy()
            seqs = ["".join(x) for x in product(*model.alphabet_list)]
            f1 = model(seqs)

            theta_fixed = model.get_fixed_params(gauge="zero-sum")
            model.set_params(theta_fixed)
            f2 = model(seqs)
            theta_fixed_fixed = model.get_fixed_params(gauge="zero-sum")
            assert np.allclose(theta_fixed, theta_fixed_fixed)
            assert np.allclose(f1, f2)
            assert np.logical_not(np.allclose(theta, theta_fixed))

    def test_fix_invalid_gauges(self):
        """Test failure with generally invalid gauges"""
        L = 4
        models = [
            AllOrderModel(alphabet_name="dna", L=L),
            AdditiveModel(alphabet_name="dna", L=L),
            PairwiseModel(alphabet_name="dna", L=L),
            KorderModel(alphabet_name="dna", L=L, K=3),
            NeighborModel(alphabet_name="dna", L=L),
            KadjacentModel(alphabet_name="dna", L=L, K=3),
        ]
        pi_lc = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        lda = 10.0
        gauges = [
            {"gauge": "wild-type", "pi_lc": pi_lc},
            {"gauge": "zero-sum", "lda": lda},
            {"gauge": "hierarchical"},
            {"gauge": "trivial", "wt_seq": "AAA"},
            {"gauge": "euclidean", "pi_lc": pi_lc},
            {"gauge": "equitable", "pi_lc": pi_lc},
            {"gauge": "hierarchical", "pi_lc": pi_lc, "lda": lda},
        ]
        for model in models:
            theta = np.random.normal(size=model.n_features)
            theta = pd.Series(theta, index=model.features)
            model.set_params(theta)
            for kwargs in gauges:
                with self.assertRaises((AssertionError, TypeError)):
                    model.get_fixed_params(**kwargs)

    def test_fix_invalid_gauges_hierarhical_models(self):
        """Test failure with invalid gauges in hierarchical models"""
        L = 4
        models = [
            AdditiveModel(alphabet_name="dna", L=L),
            PairwiseModel(alphabet_name="dna", L=L),
            KorderModel(alphabet_name="dna", L=L, K=3),
            NeighborModel(alphabet_name="dna", L=L),
            KadjacentModel(alphabet_name="dna", L=L, K=3),
        ]
        pi_lc = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.25, 0.25, 0.25, 0.25]),
        ]
        gauges = [
            {"gauge": "trivial"},
            {"gauge": "euclidean"},
            {"gauge": "equitable"},
            {"lda": 10.0, "pi_lc": pi_lc},
        ]
        for model in models:
            theta = np.random.normal(size=model.n_features)
            theta = pd.Series(theta, index=model.features)
            model.set_params(theta)
            for kwargs in gauges:
                with self.assertRaises((AssertionError, TypeError)):
                    model.get_fixed_params(**kwargs)


if __name__ == "__main__":
    unittest.main()
