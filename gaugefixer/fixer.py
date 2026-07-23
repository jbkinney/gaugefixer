from __future__ import annotations

import numpy as np
import pandas as pd

import gaugefixer.docstrings as docs
from gaugefixer.utils import (
    get_orbits_features,
    get_subsets_of_multiple_sets,
    get_site_projection_matrix,
    kron_matvec,
    get_generating_orbits_param_idx,
)


class ParameterProjector(object):
    """
    Projector object to transform parameters in hierarchical linear
    sequence-function models.

    Parameters
    ----------
    generating_orbits_param_idx: dict of tuple to np.ndarray
        Dictionary mapping each generating orbit to the indices of
        the input and output parameter vectors.
    Ps : list of np.ndarray
            List of site-specific projection matrices.
    use_dense_matrix: bool
        Fix the gauge building the explicit dense projection matrix.
        Implemented mainly for testing and benchmarking.

    """

    def __init__(
        self,
        generating_orbits_param_idx: dict[tuple, np.ndarray],
        Ps: list[np.ndarray],
        use_dense_matrix: bool = False,
    ):
        self.generating_orbits_param_idx = generating_orbits_param_idx
        self.Ps = Ps
        self.use_dense_matrix = use_dense_matrix

        self.n_features_input = (
            max(idx[0].max() for idx in generating_orbits_param_idx.values())
            + 1
        )
        self.n_features_output = (
            max(idx[1].max() for idx in generating_orbits_param_idx.values())
            + 1
        )
        if use_dense_matrix:
            self.P = self.get_dense_projection_matrix(Ps)

    def get_dense_projection_matrix(self, Ps: list[np.ndarray]) -> np.ndarray:
        P = np.zeros((self.n_features_output, self.n_features_input))

        for orbit, (
            idx_in,
            idx_out,
        ) in self.generating_orbits_param_idx.items():
            orbit_P = np.array([[1.0]])
            for p in orbit:
                orbit_P = np.kron(orbit_P, Ps[p])
            P[np.ix_(idx_in, idx_out)] = orbit_P

        return P

    def _project(self, theta: np.ndarray, Ps: list) -> pd.Series | pd.DataFrame:
        # Normalize input to 2D
        was_series = len(theta.shape) == 1
        if was_series:
            theta = theta[:, np.newaxis].copy()  # type: ignore

        # Initialize 2D new theta array
        theta_new = np.zeros((self.n_features_output, theta.shape[1]))

        for orbit, (
            idx_in,
            idx_out,
        ) in self.generating_orbits_param_idx.items():
            orbit_theta = theta[idx_in, :]

            if len(orbit) > 0:
                orbit_Ps = [Ps[i] for i in orbit]
                orbit_theta_fixed = kron_matvec(orbit_Ps, orbit_theta)  # type: ignore
                theta_new[idx_out, :] += orbit_theta_fixed
            else:
                theta_new[idx_out, :] = orbit_theta

            theta[idx_in, :] = 0.0  # type: ignore

        # Denormalize output to match input type
        if was_series:
            theta_new = theta_new[:, 0]
        return theta_new

    def __call__(
        self,
        theta: np.ndarray,
    ) -> pd.Series | pd.DataFrame:
        """
        Projects parameters into new reference using the site-specific
        projection matrices iteratively using generating orbits.

        Parameters
        ----------
        theta : np.ndarray
            Model parameters to be projected.

        Returns
        -------
        theta_new: np.ndarray
            Projected parameters in new reference.
        """
        if self.use_dense_matrix:
            theta_new = self.P @ theta
        else:
            theta_new = self._project(theta, self.Ps)

        return theta_new


class GaugeFixer(object):
    __doc__ = f"""
    GaugeFixer object to fix the gauge in linear sequence-function models

    Parameters
    ----------
    {docs.ALPHABET_LIST}
    {docs.GENERATING_ORBITS}
    {docs.FEATURES}
    """

    def __init__(
        self,
        alphabet_list: list[list[str]],
        generating_orbits: list[tuple],
        features: list[tuple] | None = None,
    ):
        self.alphabet_list = alphabet_list
        self.ext_alphabet_list = [
            ["*"] + alphabet for alphabet in alphabet_list
        ]
        self.alphas = [len(alphabet) for alphabet in self.alphabet_list]
        self.L = len(alphabet_list)
        self.generating_orbits = generating_orbits
        self.orbits = get_subsets_of_multiple_sets(generating_orbits)
        self.define_features(features)
        self.generating_orbits_param_idx = get_generating_orbits_param_idx(
            generating_orbits, alphabet_list
        )
        self.max_order = max(len(orbit) for orbit in self.orbits)

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

    def _get_pi_lc_wt(
        self,
        wt_seq: str | None = None,
    ) -> list[np.ndarray]:
        """
        Get position-specific allele frequencies corresponding
        to a wild-type sequence.

        Parameters
        ----------
        {wt_seq}

        Returns
        -------
        pi_lc : list of np.ndarray
            Position-specific allele frequencies.
        """
        pi_lc = [
            np.array([c == wt_c for c in self.alphabet_list[i]]).astype(float)
            for i, wt_c in enumerate(wt_seq)
        ]
        return pi_lc

    def _get_pi_lc_uniform(
        self,
    ) -> list[np.ndarray]:
        """
        Get position-specific allele frequencies corresponding
        to a uniform distribution.

        Returns
        -------
        pi_lc : list of np.ndarray
            Position-specific allele frequencies.
        """
        pi_lc = [np.full(a, 1.0 / a) for a in self.alphas]
        return pi_lc

    def _get_pi_lc_lda(
        self,
        gauge: str | None,
        pi_lc: list[np.ndarray] | None = None,
        wt_seq: str | None = None,
        lda: float | np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray] | None]:
        """
        Get position-specific background frequencies and regularization parameters.

        Parameters
        ----------
        {gauge}
        {pi_lc}
        {wt_seq}
        {lda}

        Returns
        -------
        pi_lc, lda : tuple of (list of np.ndarray, np.ndarray)
            Position-specific background frequencies and regularization parameters.
        """

        if (
            gauge == "wild-type"
            and lda is None
            and pi_lc is None
            and isinstance(wt_seq, str)
        ):
            assert len(wt_seq) == self.L
            for i, allele in enumerate(wt_seq):
                assert allele in self.alphabet_list[i]
            lda = np.full(self.L, np.inf)
            pi_lc = self._get_pi_lc_wt(wt_seq)

        elif (
            gauge == "zero-sum"
            and lda is None
            and pi_lc is None
            and wt_seq is None
        ):
            lda = np.full(self.L, np.inf)
            pi_lc = self._get_pi_lc_uniform()

        elif (
            gauge == "hierarchical"
            and lda is None
            and isinstance(pi_lc, list)
            and wt_seq is None
        ):
            assert len(pi_lc) == self.L
            assert all(len(p) == a for p, a in zip(pi_lc, self.alphas))
            assert all(np.allclose(pi.sum(), 1.0) for pi in pi_lc)
            lda = np.full(self.L, np.inf)

        elif (
            gauge == "trivial"
            and lda is None
            and pi_lc is None
            and wt_seq is None
            and self.max_order == self.L
        ):
            lda = np.zeros(self.L)
            pi_lc = self._get_pi_lc_uniform()

        elif (
            gauge == "euclidean"
            and lda is None
            and pi_lc is None
            and wt_seq is None
            and self.max_order == self.L
        ):
            lda = np.ones(self.L)
            pi_lc = self._get_pi_lc_uniform()

        elif (
            gauge == "equitable"
            and lda is None
            and pi_lc is None
            and wt_seq is None
            and self.max_order == self.L
        ):
            lda = np.array(self.alphas)
            pi_lc = self._get_pi_lc_uniform()

        elif (
            gauge is None
            and isinstance(lda, float)
            and isinstance(pi_lc, list)
            and wt_seq is None
            and self.max_order == self.L
        ):
            lda = np.full(self.L, lda)

        elif (
            gauge is None
            and isinstance(lda, np.ndarray)
            and isinstance(pi_lc, list)
            and wt_seq is None
            and self.max_order == self.L
        ):
            pass

        else:
            assert False, (
                f"Invalid combination of inputs {gauge=}, {lda=}, {pi_lc=}, {wt_seq=}, {self.max_order=}."
            )

        return pi_lc, lda

    _get_pi_lc_lda.__doc__ = _get_pi_lc_lda.__doc__.format(  # type: ignore
        gauge=docs.GAUGE_ALL_ORDERS,
        wt_seq=docs.WT_SEQ,
        pi_lc=docs.PI_LC,
        lda=docs.LDA,
    )

    def _get_site_P(
        self,
        pi_lc: list[np.ndarray],
        lda: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Compute site-specific projection matrices.

        Parameters
        ----------
        {pi_lc}
        {lda}

        Returns
        -------
        {Ps}
        """
        return [
            get_site_projection_matrix(pi, lda_i)
            for pi, lda_i in zip(pi_lc, lda)
        ]

    _get_site_P.__doc__ = _get_site_P.__doc__.format(  # type: ignore
        pi_lc=docs.PI_LC,
        lda=docs.LDA,
        Ps=docs.PS,
    )

    def __call__(
        self,
        theta: pd.Series | pd.DataFrame,
        gauge: str | None = None,
        wt_seq: str | None = None,
        pi_lc: list[np.ndarray] | None = None,
        lda: float | np.ndarray | None = None,
        use_dense_matrix: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Fixes the gauge of the model parameters.

        Parameters
        ----------
        {theta}
        {gauge}
        {wt_seq}
        {pi_lc}
        {lda}
        {dense_matrix}

        Returns
        -------
        {theta_fixed}
        """

        pi_lc, lda = self._get_pi_lc_lda(gauge, pi_lc, wt_seq, lda=lda)
        Ps = self._get_site_P(pi_lc, lda)
        idx = {k: (v, v) for k, v in self.generating_orbits_param_idx.items()}
        project = ParameterProjector(
            generating_orbits_param_idx=idx,
            Ps=Ps,
            use_dense_matrix=use_dense_matrix,
        )

        theta_fixed = project(theta.values)
        if isinstance(theta, pd.Series):
            theta_fixed = pd.Series(theta_fixed, index=self.features)
        else:
            theta_fixed = pd.DataFrame(
                theta_fixed, index=self.features, columns=theta.columns
            )
        return theta_fixed

    __call__.__doc__ = __call__.__doc__.format(  # type: ignore
        theta=docs.THETA_OUT,
        gauge=docs.GAUGE_ALL_ORDERS,
        wt_seq=docs.WT_SEQ,
        pi_lc=docs.PI_LC,
        lda=docs.LDA,
        theta_fixed=docs.THETA_FIXED,
        dense_matrix=docs.DENSE_MATRIX,
    )
