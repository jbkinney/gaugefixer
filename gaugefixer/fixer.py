from __future__ import annotations

import numpy as np
import pandas as pd

import gaugefixer.docstrings as docs
from gaugefixer.utils import (
    get_orbits_features,
    get_orbits_subsequences,
    get_subsets_of_multiple_sets,
    get_site_projection_matrix,
    kron_matvec,
    get_all_seqs,
)


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
        self.define_generating_orbits_idx()
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
            self.features = get_orbits_features(
                self.orbits, self.alphabet_list
            )
        self.n_features = len(self.features)

    def define_generating_orbits_idx(self) -> None:
        subseqs = get_orbits_subsequences(self.orbits, self.alphabet_list)
        self.subsequence_idx = dict(zip(subseqs, np.arange(self.n_features)))
        self.generating_orbits_idx = {}
        for orbit in self.generating_orbits:
            orbit_subseqs = self.get_suborbits_subsequences(orbit)
            idx = np.fromiter(
                (self.subsequence_idx[s] for s in orbit_subseqs),
                dtype=np.int64,
            )
            self.generating_orbits_idx[orbit] = idx

    def _get_pi_lc_lda(
        self,
        gauge: str | None,
        pi_lc: list[np.ndarray] | None = None,
        wt_seq: str | None = None,
        lda: float | np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
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
            pi_lc = [
                np.array([c == wt_c for c in self.alphabet_list[i]]).astype(
                    float
                )
                for i, wt_c in enumerate(wt_seq)
            ]

        elif (
            gauge == "zero-sum"
            and lda is None
            and pi_lc is None
            and wt_seq is None
        ):
            lda = np.full(self.L, np.inf)
            pi_lc = [np.full(a, 1.0 / a) for a in self.alphas]

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
            pi_lc = [np.full(a, 1.0 / a) for a in self.alphas]

        elif (
            gauge == "euclidean"
            and lda is None
            and pi_lc is None
            and wt_seq is None
            and self.max_order == self.L
        ):
            lda = np.ones(self.L)
            pi_lc = [np.full(a, 1.0 / a) for a in self.alphas]

        elif (
            gauge == "equitable"
            and lda is None
            and pi_lc is None
            and wt_seq is None
            and self.max_order == self.L
        ):
            lda = np.array(self.alphas)
            pi_lc = [np.full(a, 1.0 / a) for a in self.alphas]

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
        self, pi_lc: list[np.ndarray], lda: np.ndarray
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

    def get_suborbits_subsequences(
        self, orbit: tuple, wildcard: str = "*"
    ) -> list[str]:
        """
        Generate subsequences within a given orbit.

        Parameters
        ----------
        orbit : tuple
            Tuple of site indices defining the orbit (e.g., (i, j, ...)).
        wildcard : str, optional
            Character to use as a wildcard for suborbits. Default is '*'.

        Returns
        -------
        list[str]
            List of subsequences for the orbit. Each subsequence is a string
            representing a combination of alleles (or wildcards) at the specified sites.
        """
        alphabets = [
            [wildcard] + alphabet if p in orbit else [wildcard]
            for p, alphabet in enumerate(self.alphabet_list)
        ]
        subseqs = get_all_seqs(alphabets)
        return subseqs

    def get_dense_projection_matrix(self, Ps: list) -> np.ndarray:
        idxs = [
            {c: i for i, c in enumerate(alphabet)}
            for alphabet in self.ext_alphabet_list
        ]
        allele_idxs = np.array(
            [
                [idx[c] for c, idx in zip(seq, idxs)]
                for seq in self.subsequence_idx.keys()
            ]
        ).T
        P = np.ones((self.n_features, self.n_features))
        for p, P_p in zip(range(self.L), Ps):
            P *= P_p[allele_idxs[p], :][:, allele_idxs[p]]
        return P

    def _fix_using_generating_orbits(
        self, theta: pd.Series | pd.DataFrame, Ps: list
    ) -> pd.Series | pd.DataFrame:
        """
        Fixes parameters iteratively using generating orbits.

        Parameters
        ----------
        {theta}
        {Ps}

        Returns
        -------
        {theta_fixed}
        """
        # Normalize input to 2D
        was_series = isinstance(theta, pd.Series)
        if was_series:
            theta_array = theta.to_numpy()[:, np.newaxis].copy()  # type: ignore
        else:
            theta_array = theta.to_numpy().copy()  # type: ignore

        # Initialize 2D theta_fixed array
        theta_fixed = np.zeros((self.n_features, theta_array.shape[1]))

        for orbit, orbit_idx in self.generating_orbits_idx.items():
            orbit_theta = theta_array[orbit_idx, :]

            if len(orbit) > 0:
                orbit_Ps = [Ps[i] for i in orbit]
                orbit_theta_fixed = kron_matvec(orbit_Ps, orbit_theta)  # type: ignore
                theta_fixed[orbit_idx, :] += orbit_theta_fixed
            else:
                theta_fixed[orbit_idx, :] = orbit_theta

            theta_array[orbit_idx, :] = 0.0  # type: ignore

        # Denormalize output to match input type
        if was_series:
            return pd.Series(theta_fixed[:, 0], index=self.features)
        else:
            return pd.DataFrame(
                theta_fixed, index=self.features, columns=theta.columns
            )

    _fix_using_generating_orbits.__doc__ = (
        _fix_using_generating_orbits.__doc__.format(  # type: ignore
            theta=docs.THETA_OUT,
            Ps=docs.PS,
            theta_fixed=docs.THETA_FIXED,
        )
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
        if use_dense_matrix:
            P = self.get_dense_projection_matrix(Ps)
            theta_fixed = P @ theta.values
            if isinstance(theta, pd.Series):
                theta_fixed = pd.Series(theta_fixed, index=self.features)
            else:
                theta_fixed = pd.DataFrame(
                    theta_fixed, index=self.features, columns=theta.columns
                )
        else:
            theta_fixed = self._fix_using_generating_orbits(theta, Ps)
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
