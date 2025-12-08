#!/usr/bin/env python
import unittest
from itertools import combinations

import numpy as np

from gaugefixer.encoder import BinarySequenceEncoder
from gaugefixer.utils import random_seqs


class TestBinarySequenceEncoder(unittest.TestCase):
    def setUp(self):
        self.L = 3
        self.alphabet = list("ACGT")
        self.alphabet_list = [self.alphabet] * self.L
        self.orbits = [(), (0,), (1,), (2,)]
        self.params = [
            {"orbits": [(), (0,), (1,), (2,)], "n": 13},
            {"orbits": [(), (0,), (1,), (2,), (0, 1), (1, 2)], "n": 45},
            {"orbits": [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)], "n": 61},
            {
                "orbits": [
                    (),
                    (0,),
                    (1,),
                    (2,),
                    (0, 1),
                    (0, 2),
                    (1, 2),
                    (0, 1, 2),
                ],
                "n": 61 + 64,
            },
        ]
        self.encoder = BinarySequenceEncoder(self.alphabet_list, self.orbits)
        self.encoder_regex = BinarySequenceEncoder(
            self.alphabet_list, self.orbits, use_regex=True
        )
        self.seqs = ["ACG", "TGC", "AAA", "CCC"]
        self.x = np.array(
            [
                [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            ]
        )

    def test_initialization(self):
        """Test encoder initialization."""
        for params in self.params:
            encoder = BinarySequenceEncoder(
                self.alphabet_list, params["orbits"]
            )
            self.assertEqual(encoder.L, 3)
            self.assertEqual(encoder.max_alpha, 4)
            self.assertEqual(encoder.positions, [0, 1, 2])
            self.assertEqual(encoder.n_features, params["n"])

    def test_zero_length(self):
        """Test with minimal sequence length."""
        with self.assertRaises(ValueError):
            BinarySequenceEncoder([], [()])

    def test_single_character_validation(self):
        """Test that alphabet characters must be single characters."""
        with self.assertRaises(ValueError):
            BinarySequenceEncoder(alphabet_list=[["AA", "BB"]] * 3, orbits=[()])

    def test_empty_alphabet(self):
        """Test that empty alphabet raises error."""
        with self.assertRaises(ValueError):
            BinarySequenceEncoder(alphabet_list=[[]] * 3, orbits=[()])

    def test_orbit_to_slice_dict(self):
        """Test that orbit_to_slice_dict is created correctly."""
        self.assertIsInstance(self.encoder.orbit_to_slice_dict, dict)
        self.assertGreater(len(self.encoder.orbit_to_slice_dict), 0)

        # Check that slices are non-overlapping and cover all features
        total_features = sum(
            slice_obj.stop - slice_obj.start
            for slice_obj in self.encoder.orbit_to_slice_dict.values()
        )
        self.assertEqual(total_features, self.encoder.n_features)

    def test_regex_patterns_and_compiled(self):
        """Test that regex patterns are created."""
        self.assertEqual(
            len(self.encoder_regex.regex_patterns),
            self.encoder_regex.n_features,
        )
        self.assertEqual(
            len(self.encoder_regex.regex_compiled),
            self.encoder_regex.n_features,
        )

        # Check that patterns are strings
        for pattern in self.encoder_regex.regex_patterns:
            self.assertIsInstance(pattern, str)

    def test_features_structure(self):
        """Test that features have correct structure."""

        for positions, subsequence in self.encoder.features:
            self.assertIsInstance(positions, tuple)
            self.assertIsInstance(subsequence, str)

            # Positions should be within range [0, L)
            for pos in positions:
                self.assertGreaterEqual(pos, 0)
                self.assertLess(pos, self.encoder.L)

            # Subsequence length should match number of positions
            self.assertEqual(len(subsequence), len(positions))

            # All characters in subsequence should be in alphabet
            for pos, char in zip(positions, subsequence):
                self.assertIn(char, self.encoder.alphabet_list[pos])

    def test_encode_seqs(self):
        """Test encode_seqs with numpy algorithm."""
        x_fe = self.encoder(self.seqs)

        self.assertEqual(x_fe.shape, (4, self.encoder.n_features))
        self.assertEqual(x_fe.dtype, np.int8)
        self.assertTrue(np.all((x_fe == 0) | (x_fe == 1)))
        self.assertTrue(np.allclose(x_fe, self.x))

    def test_encode_seqs_regex_algorithm(self):
        """Test encode_seqs with regex algorithm."""
        x_fe = self.encoder_regex(self.seqs)

        self.assertEqual(x_fe.shape, (4, self.encoder_regex.n_features))
        self.assertEqual(x_fe.dtype, np.int8)
        self.assertTrue(np.all((x_fe == 0) | (x_fe == 1)))
        self.assertTrue(np.allclose(x_fe, self.x))

    def test_encode_seqs_algorithms_consistency(self):
        """Test that numpy and regex algorithms produce same results."""
        seqs = random_seqs(alphabet=self.alphabet, L=self.L, num_sequences=100)
        for params in self.params:
            encoder1 = BinarySequenceEncoder(
                self.alphabet_list, params["orbits"]
            )
            encoder2 = BinarySequenceEncoder(
                self.alphabet_list, params["orbits"], use_regex=True
            )
            x1 = encoder1(seqs)
            x2 = encoder2(seqs)
            assert np.allclose(x1, x2)

    def test_encode_seqs_empty_list(self):
        """Test encode_seqs with empty sequence list."""
        x_fe = self.encoder([])

        # Should return empty array with correct shape
        self.assertEqual(x_fe.shape, (0, self.encoder.n_features))
        # dtype might vary for empty arrays, so just check it's numeric
        self.assertTrue(np.issubdtype(x_fe.dtype, np.number))

    def test_encode_seqs_single_sequence(self):
        """Test encode_seqs with single sequence."""
        # Check output shape
        x_fe = self.encoder(["ACG"])
        self.assertEqual(x_fe.shape, (1, self.encoder.n_features))
        self.assertEqual(x_fe.dtype, np.int8)

    def test_encode_seqs_wrong_sequence_length(self):
        """Test encode_seqs with wrong sequence length."""
        self.assertRaises(ValueError, self.encoder, ["AA"])
        self.assertRaises(ValueError, self.encoder, ["AAAA"])

    def test_encode_seqs_invalid_sequence_type(self):
        """Test encode_seqs with invalid sequence type."""
        self.assertRaises(TypeError, self.encoder, [213])

    def test_encode_seqs_invalid_input_type(self):
        """Test encode_seqs with invalid input type."""
        self.assertRaises(TypeError, self.encoder, "AAA")

    def test_encode_seqs_different_models(self):
        """Test encode_seqs works with different model types."""
        encoders = [
            BinarySequenceEncoder(self.alphabet_list, params["orbits"])
            for params in self.params
        ]

        for encoder1, encoder2 in combinations(encoders, 2):
            x1 = encoder1(self.seqs)
            x2 = encoder2(self.seqs)
            self.assertNotEqual(x1.shape, x2.shape)


class TestTensorFunctions(unittest.TestCase):
    """Test tensor product and broadcasting functions."""

    def test_broadcast_moveaxis(self):
        """Test broadcast_moveaxis function."""
        encoder = BinarySequenceEncoder(
            alphabet_list=[list("AB")], orbits=[(), (0,)]
        )

        # Test basic case
        arr = np.array([[1, 2], [3, 4]])  # Shape (2, 2)
        result = encoder.broadcast_moveaxis(arr, 0, [2])
        expected_shape = (2, 2)  # (N, alpha_0) where alpha_0=2
        self.assertEqual(result.shape, expected_shape)

        # Test with different k
        result = encoder.broadcast_moveaxis(arr, 0, [2])
        expected_shape = (2, 2)  # (N, alpha_0) where alpha_0=2
        self.assertEqual(result.shape, expected_shape)

        # Test with single sample
        arr = np.array([[1, 2]])  # Shape (1, 2)
        result = encoder.broadcast_moveaxis(arr, 0, [2])
        expected_shape = (1, 2)  # (N, alpha_0) where alpha_0=2
        self.assertEqual(result.shape, expected_shape)

    def test_tensor_product_onehot(self):
        """Test tensor_product_onehot function."""
        # Test with 3D input
        encoder = BinarySequenceEncoder(
            alphabet_list=[list("AB")] * 2, orbits=[(), (0,), (1,), (0, 1)]
        )
        x = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])  # Shape (2, 2, 2)

        # Test with empty positions
        result = encoder.tensor_product_onehot(x, ())
        expected = np.ones((2, 1))
        np.testing.assert_array_equal(result, expected)

        # Test with single position
        result = encoder.tensor_product_onehot(x, (0,))
        expected_shape = (2, 2)
        self.assertEqual(result.shape, expected_shape)

        # Test with two positions
        result = encoder.tensor_product_onehot(x, (0, 1))
        expected_shape = (2, 2, 2)
        self.assertEqual(result.shape, expected_shape)

        # Test error cases
        with self.assertRaises(IndexError):
            encoder.tensor_product_onehot(x, (2,))  # Invalid position

        with self.assertRaises(ValueError):
            encoder.tensor_product_onehot(
                np.array([[1, 2]]), (0,)
            )  # Wrong dimensions

    def test_seqs_to_x_ohe(self):
        """Test seqs_to_x_ohe function."""
        seqs = ["AB", "BA"]
        encoder = BinarySequenceEncoder(
            alphabet_list=[list("AB")] * 2, orbits=[(), (0,), (1,), (0, 1)]
        )
        result = encoder.seqs_to_x_ohe(seqs)
        expected_shape = (2, 2, 2)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.dtype, np.int8)

        # Test with different alphabet
        encoder = BinarySequenceEncoder(
            alphabet_list=[list("ABC")] * 3, orbits=[(), (0,), (1,), (0, 1)]
        )
        seqs = ["ABC", "CBA"]
        result = encoder.seqs_to_x_ohe(seqs)
        expected_shape = (2, 3, 3)
        self.assertEqual(result.shape, expected_shape)

        # Test error cases
        with self.assertRaises(ValueError):
            encoder.seqs_to_x_ohe([])

        with self.assertRaises(ValueError):
            encoder.seqs_to_x_ohe(["AB", "ABC"])

        with self.assertRaises(ValueError):
            encoder.seqs_to_x_ohe(["AB", "AD"])


if __name__ == "__main__":
    unittest.main()
