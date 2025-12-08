from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import gaugefixer.docstrings as docs

from gaugefixer.encoder import BinarySequenceEncoder
from gaugefixer.fixer import GaugeFixer
from gaugefixer.utils import (
    get_orbits_features,
    get_subsets_of_multiple_sets,
    validate_alphabet_params,
    get_all_seqs,
)


class HierarchicalModel(object):
    __doc__ = f"""
    Linear hierarchical model for sequence-function relationships

    Parameters
    ----------
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.GENERATING_ORBITS}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        generating_orbits: list[tuple] | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        # Validate and record alphabet parameters
        params = validate_alphabet_params(
            alphabet_name, alphabet, alphabet_list, L
        )
        self.alphabet_name, self.alphabet, self.alphabet_list, self.L = params

        # Set alpha if alphabet is provided
        if self.alphabet is not None:
            self.alpha = len(self.alphabet)
        else:
            self.alpha = None
        self.alphas = [len(alphabet) for alphabet in self.alphabet_list]
        self._define_positions(positions)
        self._set_orbits(generating_orbits)
        self._define_features()
        self.size = np.prod(self.alphas)

        self.encoder = BinarySequenceEncoder(
            alphabet_list=self.alphabet_list,
            orbits=self.orbits,
            features=self.features,
        )
        self.fixer = GaugeFixer(
            alphabet_list=self.alphabet_list,
            generating_orbits=self.generating_orbits,
            features=self.features,
        )
        if theta is not None:
            self.set_params(theta)

    def _encode_seqs(self, seqs: list[str]) -> np.ndarray:
        """
        Encode multiple sequences into binary feature vectors.

        Parameters
        ----------
        {seqs}

        Returns
        -------
        {seqs_features}
        """
        return self.encoder(seqs)

    _encode_seqs.__doc__ = _encode_seqs.__doc__.format(  # type: ignore
        seqs=docs.SEQS, seqs_features=docs.BINARY_FEATURES
    )

    def get_params(self) -> pd.Series | pd.DataFrame:
        """
        Returns parameters for a linear.

        Returns
        -------
        {theta}
        """
        return self.theta

    get_params.__doc__ = get_params.__doc__.format(  # type: ignore
        theta=docs.THETA
    )

    def get_fixed_params(
        self,
        gauge: str | None = None,
        wt_seq: str | None = None,
        pi_lc: list[np.ndarray] | None = None,
        use_dense_matrix: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Returns gauge-fixed parameters for a linear model.

        Parameters
        ----------
        {gauge}
        {wt_seq}
        {pi_lc}
        {dense_matrix}

        Returns
        -------
        {theta}
        """
        return self.fixer(
            self.theta,
            gauge=gauge,
            wt_seq=wt_seq,
            pi_lc=pi_lc,
            lda=None,
            use_dense_matrix=use_dense_matrix,
        )

    get_fixed_params.__doc__ = get_fixed_params.__doc__.format(  # type: ignore
        gauge=docs.GAUGE_HIERARCHICAL,
        wt_seq=docs.WT_SEQ,
        pi_lc=docs.PI_LC,
        theta=docs.THETA_OUT,
        dense_matrix=docs.DENSE_MATRIX,
    )

    def _define_positions(self, positions: list[int] | None = None) -> None:
        if positions is not None:
            if len(positions) != self.L:
                raise ValueError(
                    f"positions, if specified, must be the same length as L; {len(positions)=}, {self.L=}"
                )
            if len(positions) != len(set(positions)):
                raise ValueError(
                    f"positions, if specified, must be a list of unique positions; {positions=}"
                )
            self.positions = positions
        else:
            self.positions = list(range(self.L))

        self.pos_to_idx = {pos: i for i, pos in enumerate(self.positions)}

    def _set_orbits(
        self, generating_orbits: list[tuple] | None = None
    ) -> None:
        """
        Set the orbits of the linear model and defines the associated features.

        Parameters
        ----------
        generating_orbits : list of tuple, optional
            Generating orbits to use. If positions is provided,
            the generating orbits are expected to be in terms of these positions.

        """
        if generating_orbits is not None:
            self.generating_orbits = generating_orbits.copy()
        else:
            self.generating_orbits = self._calc_generating_orbits()

        generating_orbits_idx = [
            tuple(self.pos_to_idx[pos] for pos in orbit)
            for orbit in self.generating_orbits
        ]

        self.orbits = get_subsets_of_multiple_sets(generating_orbits_idx)
        self.n_orbits = len(self.orbits)

        # Verify consistency with L
        assert all(
            [max(orbit) < self.L for orbit in self.orbits if len(orbit) > 0]
        )

    def _define_features(self) -> None:
        """
        Generate features from orbits and alphabet.
        """
        self.features = get_orbits_features(self.orbits, self.alphabet_list)
        self.n_features = len(self.features)

    def get_features(self) -> list[tuple]:
        """
        Get the features of the model, using user-specified positions if provided.

        Returns
        -------
        {features}
        """
        return [
            (tuple(self.positions[i] for i in orbit), subsequence)
            for orbit, subsequence in self.features
        ]

    get_features.__doc__ = get_features.__doc__.format(  # type: ignore
        features=docs.FEATURES_OUT
    )

    def get_generating_orbits(self) -> list[tuple]:
        """
        Get the generating orbits of the hierarchical model, using
        user-specified positions if provided.

        Returns
        -------
        {generating_orbits}
        """
        return [
            tuple(self.positions[i] for i in orbit)
            for orbit in self.generating_orbits
        ]

    get_generating_orbits.__doc__ = get_generating_orbits.__doc__.format(  # type: ignore
        generating_orbits=docs.GENERATING_ORBITS
    )

    def get_orbits(self) -> list[tuple]:
        """
        Get the orbits of the encoder, using user-specified positions if provided.

        Returns
        -------
        {orbits}
        """
        return [
            tuple(self.positions[i] for i in orbit) for orbit in self.orbits
        ]

    get_orbits.__doc__ = get_orbits.__doc__.format(  # type: ignore
        orbits=docs.ORBITS_OUT
    )

    def set_params(self, theta: pd.Series | pd.DataFrame) -> None:
        """
        Define the values of the model parameters theta.

        Parameters
        ----------
        {theta}

        """
        if theta.shape[0] != self.n_features:
            raise ValueError(
                f"theta must have length {self.n_features}; got {theta.shape[0]}"
            )

        self.theta = theta.loc[self.features]

        # Check for numerical issues in theta
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            raise ValueError("Theta contains NaN or infinite values")

    set_params.__doc__ = set_params.__doc__.format(theta=docs.THETA_OUT)  # type: ignore

    def set_random_params(self) -> None:
        """
        Initialize model with random values of the parameters.
        """
        theta_values = np.random.normal(size=self.n_features)
        theta = pd.Series(theta_values, index=self.features)
        self.theta = theta

    def _calc_generating_orbits(self) -> list[tuple]:
        """
        Get the generating orbits of the hierarchical model.
        """
        msg = """`HierarchicalModel` does not have a
                 `_calc_generating_orbits` method. Provide `generating_orbits` 
                 when defining the model instead"""
        raise NotImplementedError(msg)

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """
        Evaluate the model at specific input sequences.

        Parameters
        ----------
        {seqs}

        Returns
        -------
        {f}
        """
        # Input validation
        if not seqs:
            raise ValueError("seqs cannot be empty")

        if not hasattr(self, "theta"):
            raise ValueError("Model parameters theta have not been set.")

        # Embed all sequences and compute scores
        x_flat = self._encode_seqs(seqs)
        scores = x_flat @ self.theta.values

        # Check for numerical issues in result
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            raise ValueError("Computed scores contain NaN or infinite values")

        return scores

    __call__.__doc__ = __call__.__doc__.format(seqs=docs.SEQS, f=docs.F)  # type: ignore

    def _get_description(self) -> dict:
        description = {"L": self.L}

        if self.alphabet_name is not None:
            description["alphabet_name"] = self.alphabet_name  # type: ignore
        elif self.alphabet is not None:
            description["alphabet"] = self.alphabet  # type: ignore
        else:
            description["alphabet_list"] = self.alphabet_list  # type: ignore

        description["n_features"] = self.n_features
        description["n_orbits"] = self.n_orbits

        if hasattr(self, "K"):
            description["K"] = getattr(self, "K")
        return description

    def __repr__(self) -> str:
        items = [
            "{}={}".format(k, v) for k, v in self._get_description().items()
        ]
        return f"{self.__class__.__name__}({','.join(items)})"

    def __str__(self) -> str:
        items = [
            "\t{}={}".format(k, v) for k, v in self._get_description().items()
        ]
        return "\n".join([f"{self.__class__.__name__}:", *items])


class AllOrderModel(HierarchicalModel):
    __doc__ = f"""
    All-order model for sequence-function relationships

    Parameters
    ----------
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        super().__init__(
            alphabet=alphabet,
            alphabet_name=alphabet_name,
            alphabet_list=alphabet_list,
            L=L,
            positions=positions,
            theta=theta,
        )

    def get_fixed_params(
        self,
        gauge: str | None = None,
        wt_seq: str | None = None,
        pi_lc: list[np.ndarray] | None = None,
        lda: float | np.ndarray | None = None,
        use_dense_matrix: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Returns gauge-fixed parameters for the model.

        Parameters
        ----------
        {gauge}
        {wt_seq}
        {pi_lc}
        {lda}
        {dense_matrix}

        Returns
        -------
        {theta}
        """
        return self.fixer(
            self.theta,
            gauge=gauge,
            wt_seq=wt_seq,
            pi_lc=pi_lc,
            lda=lda,
            use_dense_matrix=use_dense_matrix,
        )

    get_fixed_params.__doc__ = get_fixed_params.__doc__.format(  # type: ignore
        gauge=docs.GAUGE_ALL_ORDERS,
        wt_seq=docs.WT_SEQ,
        pi_lc=docs.PI_LC,
        lda=docs.LDA,
        theta=docs.THETA_FIXED,
        dense_matrix=docs.DENSE_MATRIX,
    )

    def set_landscape(self, f: pd.Series) -> None:
        """
        Define the model parameters theta from function values.

        Parameters
        ----------
        {f}
        """
        if f.shape[0] != self.size:
            raise ValueError(
                f"f must have length {self.size}; got {f.shape[0]}"
            )
        try:
            f_seqs = f.loc[get_all_seqs(self.alphabet_list)]
        except KeyError:
            raise ValueError("f must be indexed by all possible sequences")
        if np.any(np.isnan(f_seqs)):
            raise ValueError("f contains NaN values")
        if np.any(np.isinf(f_seqs)):
            raise ValueError("f contains inf values")

        theta = pd.Series(np.zeros(self.n_features), index=self.features)
        positions = tuple(range(self.L))
        seq_features = [(positions, seq) for seq in f.index]
        theta.update(pd.Series(f.values, index=seq_features))
        self.set_params(theta)

    set_landscape.__doc__ = set_landscape.__doc__.format(f=docs.F)  # type: ignore

    def _calc_generating_orbits(self) -> list[tuple]:
        """
        Get the generating orbits of the all-orders model.
        """
        return [tuple(list(range(self.L)))]


class KorderModel(HierarchicalModel):
    __doc__ = f"""
    K-order model for sequence-function relationships

    Parameters
    ----------
    {docs.K}
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        K: int,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        if K < 1:
            raise ValueError(f"K must be at least 1; got {K}")
        self.K = K

        super().__init__(
            alphabet=alphabet,
            alphabet_name=alphabet_name,
            alphabet_list=alphabet_list,
            L=L,
            positions=positions,
            theta=theta,
        )

    def _calc_generating_orbits(self):
        return [tuple(orbit) for orbit in combinations(range(self.L), self.K)]


class AdditiveModel(KorderModel):
    __doc__ = f"""
    Additive model for sequence-function relationships

    Parameters
    ----------
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        super().__init__(
            alphabet=alphabet,
            alphabet_name=alphabet_name,
            alphabet_list=alphabet_list,
            L=L,
            K=1,
            positions=positions,
            theta=theta,
        )


class PairwiseModel(KorderModel):
    __doc__ = f"""
    Pairwise model for sequence-function relationships

    Parameters
    ----------
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        super().__init__(
            alphabet=alphabet,
            alphabet_name=alphabet_name,
            alphabet_list=alphabet_list,
            L=L,
            K=2,
            positions=positions,
            theta=theta,
        )


class KadjacentModel(HierarchicalModel):
    __doc__ = f"""
    K-adjacent model for sequence-function relationships

    Parameters
    ----------
    {docs.K}
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        K: int,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        if K < 1:
            raise ValueError(f"K must be at least 1; got {K}")
        self.K = K

        super().__init__(
            alphabet=alphabet,
            alphabet_name=alphabet_name,
            alphabet_list=alphabet_list,
            L=L,
            positions=positions,
            theta=theta,
        )

    def _calc_generating_orbits(self):
        return [
            tuple(range(i, i + self.K)) for i in range(self.L - self.K + 1)
        ]


class NeighborModel(KadjacentModel):
    __doc__ = f"""
    Neighbor model for sequence-function relationships

    Parameters
    ----------
    {docs.K}
    {docs.ALPHABET}
    {docs.ALPHABET_NAME}
    {docs.ALPHABET_LIST}
    {docs.L}
    {docs.POSITIONS}
    {docs.THETA}
    """

    def __init__(
        self,
        alphabet: list[str] | None = None,
        alphabet_name: str | None = None,
        alphabet_list: list[list[str]] | None = None,
        L: int | None = None,
        positions: list[int] | None = None,
        theta: pd.Series | pd.DataFrame | None = None,
    ):
        super().__init__(
            K=2,
            alphabet=alphabet,
            alphabet_name=alphabet_name,
            alphabet_list=alphabet_list,
            L=L,
            positions=positions,
            theta=theta,
        )
