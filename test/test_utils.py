#!/usr/bin/env python
import unittest

import numpy as np

from itertools import product
from typeguard import TypeCheckError
from gaugefixer.utils import (
    get_orbits_features,
    get_site_projection_matrix,
    get_subsets_of_multiple_sets,
    get_subsets_of_set,
    kron_matvec,
    named_alphabets_dict,
    random_seqs,
    sorted_tuples,
    validate_alphabet_params,
    get_all_seqs,
)


class UtilsTests(unittest.TestCase):
    def test_get_site_projection_matrix(self):
        pi_lc = np.array([0.4, 0.6])
        lda = np.inf
        expected_matrix = np.array(
            [[1, 0.4, 0.6], [0, 0.6, -0.6], [0, -0.4, 0.4]]
        )
        m = get_site_projection_matrix(pi_lc, lda)
        assert np.allclose(m, expected_matrix)

    def test_kron_matvec(self):
        matrices = [np.random.normal(size=(3, 2))] * 2
        m = np.kron(*matrices)
        v = np.random.normal(size=m.shape[1])
        u1 = m @ v
        u2 = kron_matvec(matrices, v)
        assert np.allclose(u1, u2)

    def test_get_all_seqs(self):
        alphabet_list = [["A", "B"], ["A", "B"]]
        seqs1 = get_all_seqs(alphabet_list)
        seqs2 = ["".join(s) for s in product(*alphabet_list)]
        assert seqs1 == seqs2

        alphabet_list = [["A", "C", "G", "T"], ["*"], ["*"]]
        seqs1 = get_all_seqs(alphabet_list)
        seqs2 = ["".join(s) for s in product(*alphabet_list)]
        assert seqs1 == seqs2


class TestRandomSeqs(unittest.TestCase):
    """Test random_seqs function with comprehensive test cases."""

    def test_alphabet_name_dna(self):
        """Test with DNA alphabet name."""
        seqs = random_seqs(alphabet_name="dna", L=5)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 5)
        valid_dna = set("ACGT")
        self.assertTrue(all(c in valid_dna for c in seqs[0]))

    def test_alphabet_name_dna_uppercase(self):
        """Test with DNA alphabet name (uppercase)."""
        seqs = random_seqs(alphabet_name="DNA", L=3)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 3)
        valid_dna = set("ACGT")
        self.assertTrue(all(c in valid_dna for c in seqs[0]))

    def test_alphabet_name_rna(self):
        """Test with RNA alphabet name."""
        seqs = random_seqs(alphabet_name="rna", L=4)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 4)
        valid_rna = set("ACGU")
        self.assertTrue(all(c in valid_rna for c in seqs[0]))

    def test_alphabet_name_rna_uppercase(self):
        """Test with RNA alphabet name (uppercase)."""
        seqs = random_seqs(alphabet_name="RNA", L=2)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 2)
        valid_rna = set("ACGU")
        self.assertTrue(all(c in valid_rna for c in seqs[0]))

    def test_alphabet_name_protein(self):
        """Test with protein alphabet name."""
        seqs = random_seqs(alphabet_name="protein", L=3)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 3)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        self.assertTrue(all(c in valid_aa for c in seqs[0]))

    def test_alphabet_name_amino_acid(self):
        """Test with amino_acid alphabet name."""
        seqs = random_seqs(alphabet_name="amino_acid", L=2)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 2)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        self.assertTrue(all(c in valid_aa for c in seqs[0]))

    def test_alphabet_name_binary(self):
        """Test with binary alphabet name."""
        seqs = random_seqs(alphabet_name="binary", L=6)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 6)
        valid_binary = set("01")
        self.assertTrue(all(c in valid_binary for c in seqs[0]))

    def test_alphabet_name_ternary(self):
        """Test with ternary alphabet name."""
        seqs = random_seqs(alphabet_name="ternary", L=4)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 4)
        valid_ternary = set("012")
        self.assertTrue(all(c in valid_ternary for c in seqs[0]))

    def test_alphabet_name_decimal(self):
        """Test with decimal alphabet name."""
        seqs = random_seqs(alphabet_name="decimal", L=3)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 3)
        valid_decimal = set("0123456789")
        self.assertTrue(all(c in valid_decimal for c in seqs[0]))

    def test_alphabet_name_invalid(self):
        """Test with invalid alphabet name."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_name="invalid", L=3)

    def test_alphabet_name_missing_L(self):
        """Test alphabet_name without L parameter."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_name="dna")

    def test_alphabet_name_zero_L(self):
        """Test alphabet_name with L=0."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_name="dna", L=0)

    def test_alphabet_name_negative_L(self):
        """Test alphabet_name with negative L."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_name="dna", L=-1)

    def test_custom_alphabet(self):
        """Test with custom alphabet."""
        custom_alphabet = ["A", "B", "C"]
        seqs = random_seqs(alphabet=custom_alphabet, L=4)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 4)
        valid_chars = set("ABC")
        self.assertTrue(all(c in valid_chars for c in seqs[0]))

    def test_custom_alphabet_single_char(self):
        """Test with single character alphabet."""
        custom_alphabet = ["X"]
        seqs = random_seqs(alphabet=custom_alphabet, L=3)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(seqs[0], "XXX")

    def test_custom_alphabet_numeric(self):
        """Test with numeric alphabet."""
        custom_alphabet = ["1", "2", "3", "4", "5"]
        seqs = random_seqs(alphabet=custom_alphabet, L=2)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 2)
        valid_nums = set("12345")
        self.assertTrue(all(c in valid_nums for c in seqs[0]))

    def test_custom_alphabet_special_chars(self):
        """Test with special characters."""
        custom_alphabet = ["@", "#", "$", "%"]
        with self.assertRaises(Warning):
            random_seqs(alphabet=custom_alphabet, L=3)

    def test_custom_alphabet_missing_L(self):
        """Test custom alphabet without L parameter."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=["A", "B", "C"])

    def test_custom_alphabet_zero_L(self):
        """Test custom alphabet with L=0."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=["A", "B", "C"], L=0)

    def test_custom_alphabet_negative_L(self):
        """Test custom alphabet with negative L."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=["A", "B", "C"], L=-1)

    def test_custom_alphabet_empty(self):
        """Test with empty alphabet."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=[], L=3)

    def test_custom_alphabet_multi_char(self):
        """Test with multi-character strings in alphabet."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=["AB", "CD"], L=3)

    def test_custom_alphabet_non_string(self):
        """Test with non-string elements in alphabet."""

        with self.assertRaises(TypeCheckError):
            random_seqs(alphabet=[1, 2, 3], L=3)

    def test_alphabet_list_uniform(self):
        """Test with uniform alphabet_list."""
        alphabet_list_input = [["A", "C", "G", "T"]] * 5
        seqs = random_seqs(alphabet_list=alphabet_list_input)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 5)
        valid_dna = set("ACGT")
        self.assertTrue(all(c in valid_dna for c in seqs[0]))

    def test_alphabet_list_varied(self):
        """Test with varied alphabet_list."""
        alphabet_list_input = [["A", "C"], ["G", "T"], ["A", "G"], ["C", "T"]]
        seqs = random_seqs(alphabet_list=alphabet_list_input)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 4)
        # Check that each position uses the correct alphabet
        self.assertIn(seqs[0][0], ["A", "C"])
        self.assertIn(seqs[0][1], ["G", "T"])
        self.assertIn(seqs[0][2], ["A", "G"])
        self.assertIn(seqs[0][3], ["C", "T"])

    def test_alphabet_list_single_position(self):
        """Test with single position alphabet_list."""
        alphabet_list_input = [["X", "Y", "Z"]]
        seqs = random_seqs(alphabet_list=alphabet_list_input)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 1)
        self.assertIn(seqs[0], ["X", "Y", "Z"])

    def test_alphabet_list_mixed_sizes(self):
        """Test with alphabet_list of mixed sizes."""
        alphabet_list_input = [["A"], ["B", "C"], ["D", "E", "F"]]
        seqs = random_seqs(alphabet_list=alphabet_list_input)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 3)
        self.assertEqual(seqs[0][0], "A")  # Only one choice
        self.assertIn(seqs[0][1], ["B", "C"])
        self.assertIn(seqs[0][2], ["D", "E", "F"])

    def test_alphabet_list_empty_alphabet(self):
        """Test with empty alphabet in alphabet_list."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_list=[[]])

    def test_alphabet_list_multi_char(self):
        """Test with multi-character strings in alphabet_list."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_list=[["AB", "CD"]])

    def test_alphabet_list_non_string(self):
        """Test with non-string elements in alphabet_list."""
        from typeguard import TypeCheckError

        with self.assertRaises(TypeCheckError):
            random_seqs(alphabet_list=[[1, 2, 3]])

    def test_alphabet_list_with_alphabet_name(self):
        """Test alphabet_list with alphabet_name (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_list=[["A", "B"]], alphabet_name="dna")

    def test_alphabet_list_with_alphabet(self):
        """Test alphabet_list with alphabet (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_list=[["A", "B"]], alphabet=["C", "D"])

    def test_alphabet_list_with_L(self):
        """Test alphabet_list with L (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_list=[["A", "B"]], L=3)

    def test_alphabet_name_with_alphabet(self):
        """Test alphabet_name with alphabet (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_name="dna", alphabet=["A", "B"], L=3)

    def test_no_parameters(self):
        """Test with no parameters (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs()

    def test_alphabet_name_with_alphabet_list(self):
        """Test alphabet_name with alphabet_list (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet_name="dna", alphabet_list=[["A", "B"]])

    def test_alphabet_with_alphabet_list(self):
        """Test alphabet with alphabet_list (should fail)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=["A", "B"], alphabet_list=[["C", "D"]])

    def test_num_sequences_default(self):
        """Test default num_sequences parameter."""
        seqs = random_seqs(alphabet=["A", "C", "G", "T"], L=5)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 5)

    def test_num_sequences_multiple(self):
        """Test with multiple sequences."""
        seqs = random_seqs(alphabet=["A", "C", "G", "T"], L=5, num_sequences=3)
        self.assertEqual(len(seqs), 3)
        for seq in seqs:
            self.assertEqual(len(seq), 5)
            valid_dna = set("ACGT")
            self.assertTrue(all(c in valid_dna for c in seq))

    def test_num_sequences_large(self):
        """Test with large number of sequences."""
        seqs = random_seqs(
            alphabet=["A", "C", "G", "T"], L=3, num_sequences=100
        )
        self.assertEqual(len(seqs), 100)
        for seq in seqs:
            self.assertEqual(len(seq), 3)
            valid_dna = set("ACGT")
            self.assertTrue(all(c in valid_dna for c in seq))

    def test_num_sequences_zero(self):
        """Test with num_sequences=0."""
        seqs = random_seqs(alphabet=["A", "C", "G", "T"], L=5, num_sequences=0)
        self.assertEqual(len(seqs), 0)

    def test_num_sequences_negative(self):
        """Test with negative num_sequences (should raise ValueError)."""
        with self.assertRaises(ValueError):
            random_seqs(alphabet=["A", "C", "G", "T"], L=5, num_sequences=-1)

    def test_random_seed_reproducibility(self):
        """Test reproducibility with random seed."""
        # Test that same seed produces same results
        seqs1 = random_seqs(
            alphabet=["A", "C", "G", "T"],
            L=10,
            num_sequences=2,
            random_seed=42,
        )
        seqs2 = random_seqs(
            alphabet=["A", "C", "G", "T"],
            L=10,
            num_sequences=2,
            random_seed=42,
        )
        self.assertEqual(seqs1, seqs2)

        # Test that different seeds produce different results
        seqs1 = random_seqs(
            alphabet=["A", "C", "G", "T"],
            L=10,
            num_sequences=1,
            random_seed=42,
        )
        seqs2 = random_seqs(
            alphabet=["A", "C", "G", "T"],
            L=10,
            num_sequences=1,
            random_seed=123,
        )
        # While it's possible they could be the same by chance, it's very unlikely
        # for length 10 sequences
        self.assertNotEqual(seqs1, seqs2)

    def test_random_seed_none(self):
        """Test that random_seed=None works (no seed set)."""
        seqs = random_seqs(alphabet=["A", "C", "G", "T"], L=5, random_seed=None)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 5)

    def test_random_seed_with_alphabet_name(self):
        """Test random seed with alphabet_name."""
        seqs1 = random_seqs(
            alphabet_name="dna", L=5, num_sequences=2, random_seed=42
        )
        seqs2 = random_seqs(
            alphabet_name="dna", L=5, num_sequences=2, random_seed=42
        )
        self.assertEqual(seqs1, seqs2)

    def test_random_seed_with_alphabet_list(self):
        """Test random seed with alphabet_list."""
        alphabet_list = [["A", "C"], ["G", "T"]]
        seqs1 = random_seqs(
            alphabet_list=alphabet_list, num_sequences=2, random_seed=42
        )
        seqs2 = random_seqs(
            alphabet_list=alphabet_list, num_sequences=2, random_seed=42
        )
        self.assertEqual(seqs1, seqs2)

    def test_edge_case_length_one(self):
        """Test with L=1 for various input types."""
        # Test alphabet_name with L=1
        seqs = random_seqs(alphabet_name="dna", L=1)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 1)
        self.assertIn(seqs[0], ["A", "C", "G", "T"])

        # Test alphabet with L=1
        seqs = random_seqs(alphabet=["X", "Y"], L=1)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 1)
        self.assertIn(seqs[0], ["X", "Y"])

        # Test alphabet_list with single position
        seqs = random_seqs(alphabet_list=[["P", "Q"]])
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 1)
        self.assertIn(seqs[0], ["P", "Q"])

    def test_large_L(self):
        """Test with large L values."""
        # Test with L=100
        seqs = random_seqs(alphabet_name="binary", L=100)
        self.assertEqual(len(seqs), 1)
        self.assertEqual(len(seqs[0]), 100)
        valid_binary = set("01")
        self.assertTrue(all(c in valid_binary for c in seqs[0]))

    def test_return_type_consistency(self):
        """Test that return types are consistent."""
        # Test alphabet_name
        seqs = random_seqs(alphabet_name="dna", L=3)
        self.assertIsInstance(seqs, list)
        self.assertTrue(all(isinstance(seq, str) for seq in seqs))

        # Test alphabet
        seqs = random_seqs(alphabet=["A", "B"], L=2)
        self.assertIsInstance(seqs, list)
        self.assertTrue(all(isinstance(seq, str) for seq in seqs))

        # Test alphabet_list
        seqs = random_seqs(alphabet_list=[["X"], ["Y"]])
        self.assertIsInstance(seqs, list)
        self.assertTrue(all(isinstance(seq, str) for seq in seqs))

    def test_sequence_uniqueness(self):
        """Test that sequences can be unique (not all identical)."""
        # Generate many sequences and check they're not all the same
        seqs = random_seqs(
            alphabet=["A", "C", "G", "T"], L=5, num_sequences=100
        )
        self.assertEqual(len(seqs), 100)
        # With 4^5 = 1024 possible sequences, 100 sequences should have some variety
        unique_seqs = set(seqs)
        self.assertGreater(len(unique_seqs), 1)  # Should have some variety

    def test_single_character_alphabet_multiple_sequences(self):
        """Test single character alphabet with multiple sequences."""
        seqs = random_seqs(alphabet=["A"], L=3, num_sequences=5)
        self.assertEqual(len(seqs), 5)
        for seq in seqs:
            self.assertEqual(seq, "AAA")

    def test_alphabet_list_with_multiple_sequences(self):
        """Test alphabet_list with multiple sequences."""
        alphabet_list = [["A", "B"], ["C", "D"]]
        seqs = random_seqs(alphabet_list=alphabet_list, num_sequences=3)
        self.assertEqual(len(seqs), 3)
        for seq in seqs:
            self.assertEqual(len(seq), 2)
            self.assertIn(seq[0], ["A", "B"])
            self.assertIn(seq[1], ["C", "D"])


class TestValidateAlphabetParams(unittest.TestCase):
    """Test validate_alphabet_params function with comprehensive test cases."""

    def test_alphabet_name_dna(self):
        """Test with DNA alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="dna", L=5
        )
        expected_alphabet = ["A", "C", "G", "T"]
        self.assertEqual(L, 5)
        self.assertEqual(len(alphabet_list), 5)
        for i in range(5):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_dna_uppercase(self):
        """Test with DNA alphabet name (uppercase)."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="DNA", L=3
        )
        expected_alphabet = ["A", "C", "G", "T"]
        self.assertEqual(L, 3)
        self.assertEqual(len(alphabet_list), 3)
        for i in range(3):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_rna(self):
        """Test with RNA alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="rna", L=4
        )
        expected_alphabet = ["A", "C", "G", "U"]
        self.assertEqual(L, 4)
        self.assertEqual(len(alphabet_list), 4)
        for i in range(4):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_rna_uppercase(self):
        """Test with RNA alphabet name (uppercase)."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="RNA", L=2
        )
        expected_alphabet = ["A", "C", "G", "U"]
        self.assertEqual(L, 2)
        self.assertEqual(len(alphabet_list), 2)
        for i in range(2):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_protein(self):
        """Test with protein alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="protein", L=3
        )
        expected_alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        self.assertEqual(L, 3)
        self.assertEqual(len(alphabet_list), 3)
        for i in range(3):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_amino_acid(self):
        """Test with amino_acid alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="amino_acid", L=2
        )
        expected_alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        self.assertEqual(L, 2)
        self.assertEqual(len(alphabet_list), 2)
        for i in range(2):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_binary(self):
        """Test with binary alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="binary", L=6
        )
        expected_alphabet = ["0", "1"]
        self.assertEqual(L, 6)
        self.assertEqual(len(alphabet_list), 6)
        for i in range(6):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_ternary(self):
        """Test with ternary alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="ternary", L=4
        )
        expected_alphabet = ["0", "1", "2"]
        self.assertEqual(L, 4)
        self.assertEqual(len(alphabet_list), 4)
        for i in range(4):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_decimal(self):
        """Test with decimal alphabet name."""
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="decimal", L=3
        )
        expected_alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.assertEqual(L, 3)
        self.assertEqual(len(alphabet_list), 3)
        for i in range(3):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_name_invalid(self):
        """Test with invalid alphabet name."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_name="invalid", L=3)

    def test_alphabet_name_missing_L(self):
        """Test alphabet_name without L parameter."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_name="dna")

    def test_alphabet_name_zero_L(self):
        """Test alphabet_name with L=0."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_name="dna", L=0)

    def test_alphabet_name_negative_L(self):
        """Test alphabet_name with negative L."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_name="dna", L=-1)

    def test_custom_alphabet(self):
        """Test with custom alphabet."""
        custom_alphabet = ["A", "B", "C"]
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet=custom_alphabet, L=4
        )
        self.assertEqual(L, 4)
        self.assertEqual(len(alphabet_list), 4)
        for i in range(4):
            self.assertEqual(alphabet_list[i], custom_alphabet)

    def test_custom_alphabet_single_char(self):
        """Test with single character alphabet."""
        custom_alphabet = ["X"]
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet=custom_alphabet, L=3
        )
        self.assertEqual(L, 3)
        self.assertEqual(len(alphabet_list), 3)
        for i in range(3):
            self.assertEqual(alphabet_list[i], custom_alphabet)

    def test_custom_alphabet_numeric(self):
        """Test with numeric alphabet."""
        custom_alphabet = ["1", "2", "3", "4", "5"]
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet=custom_alphabet, L=2
        )
        self.assertEqual(L, 2)
        self.assertEqual(len(alphabet_list), 2)
        for i in range(2):
            self.assertEqual(alphabet_list[i], custom_alphabet)

    def test_custom_alphabet_special_chars(self):
        """Test with special characters."""
        custom_alphabet = ["@", "#", "$", "%"]
        with self.assertRaises(Warning):
            _, _, alphabet_list, L = validate_alphabet_params(
                alphabet=custom_alphabet, L=3
            )

    def test_custom_alphabet_missing_L(self):
        """Test custom alphabet without L parameter."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet=["A", "B", "C"])

    def test_custom_alphabet_zero_L(self):
        """Test custom alphabet with L=0."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet=["A", "B", "C"], L=0)

    def test_custom_alphabet_negative_L(self):
        """Test custom alphabet with negative L."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet=["A", "B", "C"], L=-1
            )

    def test_custom_alphabet_empty(self):
        """Test with empty alphabet."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet=[], L=3)

    def test_custom_alphabet_multi_char(self):
        """Test with multi-character strings in alphabet."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet=["AB", "CD"], L=3)

    def test_custom_alphabet_non_string(self):
        """Test with non-string elements in alphabet."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet=[1, 2, 3], L=3)

    def test_alphabet_list_uniform(self):
        """Test with uniform alphabet_list."""
        alphabet_list_input = [["A", "C", "G", "T"]] * 5
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_list=alphabet_list_input
        )
        self.assertEqual(L, 5)
        self.assertEqual(len(alphabet_list), 5)
        expected_alphabet = ["A", "C", "G", "T"]
        for i in range(5):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_alphabet_list_varied(self):
        """Test with varied alphabet_list."""
        alphabet_list_input = [["A", "C"], ["G", "T"], ["A", "G"], ["C", "T"]]
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_list=alphabet_list_input
        )
        self.assertEqual(L, 4)
        self.assertEqual(len(alphabet_list), 4)
        self.assertEqual(alphabet_list[0], ["A", "C"])
        self.assertEqual(alphabet_list[1], ["G", "T"])
        self.assertEqual(alphabet_list[2], ["A", "G"])
        self.assertEqual(alphabet_list[3], ["C", "T"])

    def test_alphabet_list_single_position(self):
        """Test with single position alphabet_list."""
        alphabet_list_input = [["X", "Y", "Z"]]
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_list=alphabet_list_input
        )
        self.assertEqual(L, 1)
        self.assertEqual(len(alphabet_list), 1)
        self.assertEqual(alphabet_list[0], ["X", "Y", "Z"])

    def test_alphabet_list_mixed_sizes(self):
        """Test with alphabet_list of mixed sizes."""
        alphabet_list_input = [["A"], ["B", "C"], ["D", "E", "F"]]
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_list=alphabet_list_input
        )
        self.assertEqual(L, 3)
        self.assertEqual(len(alphabet_list), 3)
        self.assertEqual(alphabet_list[0], ["A"])
        self.assertEqual(alphabet_list[1], ["B", "C"])
        self.assertEqual(alphabet_list[2], ["D", "E", "F"])

    def test_alphabet_list_empty_alphabet(self):
        """Test with empty alphabet in alphabet_list."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_list=[[]])

    def test_alphabet_list_multi_char(self):
        """Test with multi-character strings in alphabet_list."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_list=[["AB", "CD"]])

    def test_alphabet_list_non_string(self):
        """Test with non-string elements in alphabet_list."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(alphabet_list=[[1, 2, 3]])

    def test_alphabet_list_with_alphabet_name(self):
        """Test alphabet_list with alphabet_name (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet_list=[["A", "B"]], alphabet_name="dna"
            )

    def test_alphabet_list_with_alphabet(self):
        """Test alphabet_list with alphabet (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet_list=[["A", "B"]], alphabet=["C", "D"]
            )

    def test_alphabet_list_with_L(self):
        """Test alphabet_list with L (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet_list=[["A", "B"]], L=3
            )

    def test_alphabet_name_with_alphabet(self):
        """Test alphabet_name with alphabet (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet_name="dna", alphabet=["A", "B"], L=3
            )

    def test_no_parameters(self):
        """Test with no parameters (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params()

    def test_alphabet_name_with_alphabet_list(self):
        """Test alphabet_name with alphabet_list (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet_name="dna", alphabet_list=[["A", "B"]]
            )

    def test_alphabet_with_alphabet_list(self):
        """Test alphabet with alphabet_list (should fail)."""
        with self.assertRaises(ValueError):
            _, _, _, _ = validate_alphabet_params(
                alphabet=["A", "B"], alphabet_list=[["C", "D"]]
            )

    def test_edge_case_length_one(self):
        """Test with L=1 for various input types."""
        # Test alphabet_name with L=1
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="dna", L=1
        )
        self.assertEqual(L, 1)
        self.assertEqual(len(alphabet_list), 1)
        self.assertEqual(alphabet_list[0], ["A", "C", "G", "T"])

        # Test alphabet with L=1
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet=["X", "Y"], L=1
        )
        self.assertEqual(L, 1)
        self.assertEqual(len(alphabet_list), 1)
        self.assertEqual(alphabet_list[0], ["X", "Y"])

        # Test alphabet_list with single position
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_list=[["P", "Q"]]
        )
        self.assertEqual(L, 1)
        self.assertEqual(len(alphabet_list), 1)
        self.assertEqual(alphabet_list[0], ["P", "Q"])

    def test_large_L(self):
        """Test with large L values."""
        # Test with L=100
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="binary", L=100
        )
        self.assertEqual(L, 100)
        self.assertEqual(len(alphabet_list), 100)
        expected_alphabet = ["0", "1"]
        for i in range(100):
            self.assertEqual(alphabet_list[i], expected_alphabet)

    def test_return_type_consistency(self):
        """Test that return types are consistent."""
        # Test alphabet_name
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_name="dna", L=3
        )
        self.assertIsInstance(alphabet_list, list)
        self.assertIsInstance(L, int)
        self.assertTrue(
            all(
                isinstance(pos_alphabet, list) for pos_alphabet in alphabet_list
            )
        )
        self.assertTrue(
            all(
                isinstance(char, str)
                for pos_alphabet in alphabet_list
                for char in pos_alphabet
            )
        )

        # Test alphabet
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet=["A", "B"], L=2
        )
        self.assertIsInstance(alphabet_list, list)
        self.assertIsInstance(L, int)
        self.assertTrue(
            all(
                isinstance(pos_alphabet, list) for pos_alphabet in alphabet_list
            )
        )
        self.assertTrue(
            all(
                isinstance(char, str)
                for pos_alphabet in alphabet_list
                for char in pos_alphabet
            )
        )

        # Test alphabet_list
        _, _, alphabet_list, L = validate_alphabet_params(
            alphabet_list=[["X"], ["Y"]]
        )
        self.assertIsInstance(alphabet_list, list)
        self.assertIsInstance(L, int)
        self.assertTrue(
            all(
                isinstance(pos_alphabet, list) for pos_alphabet in alphabet_list
            )
        )
        self.assertTrue(
            all(
                isinstance(char, str)
                for pos_alphabet in alphabet_list
                for char in pos_alphabet
            )
        )


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for tuple manipulation."""

    def test_sorted_tuples(self):
        """Test sorted_tuples function."""
        tuples = [(2, 1), (1,), (3, 2, 1), (1, 2), (2,)]
        expected = [(1,), (2,), (1, 2), (2, 1), (3, 2, 1)]
        result = sorted_tuples(tuples)
        self.assertEqual(result, expected)

        # Test with empty list
        self.assertEqual(sorted_tuples([]), [])

        # Test with single tuple
        self.assertEqual(sorted_tuples([(1, 2)]), [(1, 2)])

    def test_get_subsets_of_set(self):
        """Test get_subsets_of_set function."""
        s = (0, 1, 2)
        result = get_subsets_of_set(s)
        expected = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        self.assertEqual(set(result), set(expected))

        # Test with empty tuple
        self.assertEqual(get_subsets_of_set(()), [()])

        # Test with single element
        self.assertEqual(get_subsets_of_set((0,)), [(), (0,)])

    def test_get_subsets_of_multiple_sets(self):
        """Test get_subsets_of_multiple_sets function."""
        sets = [(0, 1), (1, 2)]
        result = get_subsets_of_multiple_sets(sets)
        expected = [(), (0,), (1,), (2,), (0, 1), (1, 2)]
        self.assertEqual(set(result), set(expected))

        # Test with empty list
        self.assertEqual(get_subsets_of_multiple_sets([]), [])

        # Test with overlapping sets
        sets = [(0, 1), (0, 1)]
        result = get_subsets_of_multiple_sets(sets)
        expected = [(), (0,), (1,), (0, 1)]
        self.assertEqual(set(result), set(expected))


class TestFeatureGeneration(unittest.TestCase):
    """Test feature generation functions."""

    def test_get_orbits_features(self):
        """Test get_features_from_orbits function."""
        orbits = [(), (0,), (1,), (0, 1)]
        alphabet = ["A", "B"]
        alphabet_list = [alphabet, alphabet]  # Convert to list of alphabets
        result = get_orbits_features(orbits, alphabet_list)
        expected = [
            ((), ""),
            ((0,), "A"),
            ((0,), "B"),
            ((1,), "A"),
            ((1,), "B"),
            ((0, 1), "AA"),
            ((0, 1), "AB"),
            ((0, 1), "BA"),
            ((0, 1), "BB"),
        ]
        self.assertEqual(set(result), set(expected))

        # Test with empty alphabet
        result = get_orbits_features(
            [(0,)], [[]]
        )  # Use orbit that references position 0
        self.assertEqual(result, [])  # Empty alphabet should return empty list

        # Test with multi-character alphabet (function accepts this)
        result = get_orbits_features([(0,), (1,)], [["AB"], ["C"]])
        expected = [((0,), "AB"), ((1,), "C")]
        self.assertEqual(result, expected)


class TestNamedAlphabetsDict(unittest.TestCase):
    """Test named_alphabets_dict."""

    def test_named_alphabets_dict_structure(self):
        """Test that named_alphabets_dict has the expected structure."""
        # Test that it's a dictionary
        self.assertIsInstance(named_alphabets_dict, dict)

        # Test that it contains expected keys
        expected_keys = {
            "dna",
            "DNA",
            "rna",
            "RNA",
            "protein",
            "protein*",
            "amino_acid",
            "binary",
            "ternary",
            "decimal",
        }
        self.assertEqual(set(named_alphabets_dict.keys()), expected_keys)

        # Test DNA alphabet
        self.assertEqual(named_alphabets_dict["dna"], list("ACGT"))

        # Test RNA alphabet
        self.assertEqual(named_alphabets_dict["rna"], list("ACGU"))

        # Test protein alphabet
        expected_protein = list("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(named_alphabets_dict["protein"], expected_protein)

    def test_named_alphabets_dict_immutability(self):
        """Test that named_alphabets_dict values are lists (not tuples)."""
        # Test that values are lists
        for key, value in named_alphabets_dict.items():
            self.assertIsInstance(value, list)
            self.assertNotIsInstance(value, tuple)

        # Test that we can access individual characters
        dna_chars = named_alphabets_dict["dna"]
        self.assertEqual(dna_chars[0], "A")
        self.assertEqual(dna_chars[1], "C")
        self.assertEqual(dna_chars[2], "G")
        self.assertEqual(dna_chars[3], "T")


if __name__ == "__main__":
    import sys

    sys.argv = ["", "UtilsTests"]
    unittest.main()
