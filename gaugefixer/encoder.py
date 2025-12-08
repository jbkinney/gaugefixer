from __future__ import annotations

import re

import numpy as np

from gaugefixer.utils import get_orbits_features, validate_alphabet_params


class BinarySequenceEncoder(object):
    """
    Sequence encoder using binary features.
    """

    def __init__(
        self,
        alphabet_list: list[list[str]],
        orbits: list[tuple],
        features: list[tuple] | None = None,
        use_regex=False,
    ):
        """
        Initialize the encoder with specified parameters.

        Parameters
        ----------
        alphabet_list : list[list[str]]
            List of alphabets, where each alphabet is a list of characters to sample
            from for that specific position. The length of alphabet_list determines
            the sequence length. Cannot be empty.
        orbits : list[tuple]
            List of tuples defining the interactions (orbits) to include in the feature set.
        features : list[tuple] or None, optional
            Predefined features to use. If None, features will be generated from orbits.
        use_regex : bool, default=False
            Whether to compile regex patterns for encoding. If False, only numpy-based
            encoding will be used. If True, both numpy and regex-based encoding will be available.
        """
        # Validate and record alphabet parameters
        self.alphabet_list, self.L = validate_alphabet_params(
            alphabet_list=alphabet_list
        )[2:]
        if self.L == 0:
            raise ValueError("Alphabet list cannot be empty")
        self.orbits = orbits
        self.alphas = [len(alphabet) for alphabet in self.alphabet_list]
        self.max_alpha = max(self.alphas) if self.alphas else 0
        self.positions = list(range(self.L))
        self.define_features(features)

        # Create alphabet_list as 2D array of int8s
        self.alphabet_list_arr = np.zeros(
            shape=[self.L, self.max_alpha], dtype=np.int8
        )
        for i, alphabet in enumerate(self.alphabet_list):
            alpha = len(alphabet)
            self.alphabet_list_arr[i, :alpha] = np.frombuffer(
                bytes("".join(alphabet), "utf-8"), np.int8, alpha
            )

        # Compute orbits_to_slice_dict
        start_idx = 0
        self.orbit_to_slice_dict = {}
        for orbit in self.orbits:
            num_features_in_orbit = int(
                np.prod([self.alphas[i] for i in orbit])
            )
            stop_idx = start_idx + num_features_in_orbit
            self.orbit_to_slice_dict[orbit] = slice(start_idx, stop_idx)
            start_idx = stop_idx

        self.use_regex = use_regex
        if self.use_regex:
            self.compile_regex()

    def define_features(self, features: list[tuple] | None) -> None:
        """
        Define the features for the encoder.

        Parameters
        ----------
        features : list[tuple] or None
            Predefined features to use. If None, features will be generated
            from the orbits and alphabet.

        Notes
        -----
        This method sets the `features` attribute to the provided list of
        features or generates them using the orbits and alphabet. It also
        calculates and sets the total number of features (`n_features`).
        """
        if features is not None:
            self.features = features
        else:
            self.features = get_orbits_features(self.orbits, self.alphabet_list)
        self.n_features = len(self.features)

    def compile_regex(self) -> None:
        """
        Compile the regular expressions to extract sequence features from sequences.
        """
        # Compile a regular expression for each sequence features
        # Create regex patterns for each feature
        self.regex_patterns = []
        self.regex_compiled = []
        for orbit, subsequence in self.features:
            pattern = "".join(
                [
                    subsequence[orbit.index(i)] if i in orbit else "."
                    for i in range(self.L)
                ]
            )
            self.regex_patterns.append(pattern)
            self.regex_compiled.append(re.compile(pattern))

    def broadcast_moveaxis(
        self, arr: np.ndarray, k: int, alphas: list[int]
    ) -> np.ndarray:
        """
        Broadcast array to target shape using moveaxis.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (N, C).
        k : int
            Position to place C dimension.
        alphas : list[int]
            Dims other than first dim in target space

        Returns
        -------
        np.ndarray
            Broadcasted array of shape (N, alpha_1, ..., alpha_K).
        """
        N, _ = arr.shape

        # Get number of characters for slice
        alpha_k = alphas[k]
        assert arr.shape[1] >= alpha_k, (
            f"arr.shape[1] ({arr.shape[1]}) < alpha ({alpha_k})"
        )

        # Add singleton dimensions
        shape = [N] + [1] * len(alphas)
        shape[k + 1] = alpha_k  # Put alpha_k at the k+1'th position

        # Reshape and broadcast
        reshaped = arr[:, :alpha_k].reshape(shape)
        target_shape = [N] + list(alphas)

        return np.broadcast_to(reshaped, target_shape)

    def tensor_product_onehot(
        self, x: np.ndarray, positions: tuple
    ) -> np.ndarray:
        """
        Create tensor product from one-hot encoded array at specified positions.

        Parameters
        ----------
        x : np.ndarray
            One-hot encoded array of shape (N, L, max_alpha).
        positions : tuple of int or empty tuple
            Positions to extract for tensor product.

        Returns
        -------
        np.ndarray
            Tensor product array.
        """
        if x.ndim != 3:
            raise ValueError("Input array must be 3-dimensional")

        N = x.shape[0]
        alphas = [self.alphas[pos] for pos in positions]

        # Compute element-wise product
        if len(positions) == 0:
            result = np.ones(shape=(N, 1))
        else:
            # Extract slices at specified positions
            x_slices = [x[:, pos, :] for pos in positions]

            # Reshape slices to match output shape
            x_slices_reshaped = [
                self.broadcast_moveaxis(x_slice_i, i, alphas)
                for i, x_slice_i in enumerate(x_slices)
            ]
            result = np.prod(x_slices_reshaped, axis=0)

            out_shape = tuple([N] + alphas)
            assert result.shape == out_shape, f"{result.shape=}, {out_shape=}"

        return result

    def x_ohe_to_x_fe(self, x_ohe: np.ndarray) -> np.ndarray:
        """
        Convert one-hot encoded sequences to feature matrix.

        Parameters
        ----------
        x_ohe : np.ndarray
            One-hot encoded sequences of shape (N, L, max_alpha).

        Returns
        -------
        np.ndarray
            Feature matrix of shape (N, n_features).
        """
        if x_ohe.ndim != 3:
            raise ValueError("Input array must be 3-dimensional")

        N = x_ohe.shape[0]
        x_fe = np.zeros(shape=(N, self.n_features))
        for orbit, x_slice in self.orbit_to_slice_dict.items():
            x = self.tensor_product_onehot(x_ohe, orbit).reshape(N, -1)
            x_fe[:, x_slice] = x
        return x_fe.astype(np.int8)

    def seqs_to_x_ohe(self, seqs: list[str]) -> np.ndarray:
        """
        Convert sequences to one-hot encoded matrix.

        Parameters
        ----------
        seqs : list of str
            List of sequences, each of length L.

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape (N, L, max_alpha) as np.int8.
        """
        if not seqs:
            raise ValueError("Sequence list cannot be empty")

        if not all(len(seq) == self.L for seq in seqs):
            raise ValueError("All sequences must have the same length")

        N = len(seqs)

        # Shape sequences as array of int8s
        x_arr = np.frombuffer(
            bytes("".join(seqs), "utf-8"), np.int8, N * self.L
        ).reshape([N, self.L])

        # Compute (N,L,C) grid of one-hot encoded values
        x_ohe = (
            x_arr[:, :, np.newaxis] == self.alphabet_list_arr[np.newaxis, :, :]
        ).astype(np.int8)

        # TODO: this looks more like a test than something we want to do all the time
        # Verify that sums across characters are all 1
        x_ohe_sums = np.sum(x_ohe, axis=2)
        valid_arr = x_ohe_sums == 1
        invalid_indices = np.argwhere(~valid_arr)
        if len(invalid_indices) > 0:
            n, p = invalid_indices[0]
            raise ValueError(
                f"Invalid character in seqs[{n=}] at position {p=}: f{seqs[n][p]=}; must be in alphabet '{self.alphabet_list[p]=}"
            )

        return x_ohe

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """
        Encode multiple sequences into binary feature vectors.

        Parameters
        ----------
        seqs : list of str
            List of input sequences to encode. All sequences must be of length L.

        Returns
        -------
        numpy.ndarray
            A 2D binary array (of type int8) of shape (n_sequences, n_features).
        """
        # Input validation
        if not isinstance(seqs, list):
            raise TypeError(f"seqs must be a list, not {type(seqs)}")

        if not seqs:
            return np.array([]).reshape(0, self.n_features)

        # Validate all sequences
        for i, seq in enumerate(seqs):
            if not isinstance(seq, str):
                raise TypeError(
                    f"Sequence {i} must be a string, not {type(seq)}"
                )

            if len(seq) != self.L:
                raise ValueError(
                    f"Sequence {i} length ({len(seq)}) must match the expected length ({self.L})"
                )

        # Convert sequences to one-hot encoding
        if self.use_regex:
            # Use regex-based encoding
            x_fe = np.array(
                [
                    [bool(regex.match(seq)) for regex in self.regex_compiled]
                    for seq in seqs
                ]
            ).astype(np.int8)
        else:
            # Use numpy-based encoding
            x_ohe = self.seqs_to_x_ohe(seqs)
            x_fe = self.x_ohe_to_x_fe(x_ohe)

        return x_fe
