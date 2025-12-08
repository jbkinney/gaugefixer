from __future__ import annotations

import numpy as np
import random
from typeguard import typechecked
from itertools import combinations, chain, product

# Define named alphabets
named_alphabets_dict = {
    "dna": list("ACGT"),
    "DNA": list("ACGT"),
    "rna": list("ACGU"),
    "RNA": list("ACGU"),
    "protein": list("ACDEFGHIKLMNPQRSTVWY"),
    "amino_acid": list("ACDEFGHIKLMNPQRSTVWY"),
    "protein*": list("*ACDEFGHIKLMNPQRSTVWY"),
    "binary": list("01"),
    "ternary": list("012"),
    "decimal": list("0123456789"),
}


# Visible ASCII characters are defined as characters with ASCII codes between 33 and 126.
# Includes all numbers, letters, punctuation, and symbols.
# Note: the space character (ASCII code 32) is not included.
_rna_alphabet = named_alphabets_dict["RNA"]

encoder_params_dict = {
    "Olson2014": {
        "L": 55,
        "model_type": "pairwise",
        "alphabet_name": "protein",
    },
    "Wu2016": {
        "L": 4,
        "model_type": "allorder",
        "alphabet_name": "protein",
        "user_positions": [39, 40, 41, 54],
    },
    "RNAP": {
        "L": 41,
        "model_type": "Kadjacent",
        "K": 3,
        "alphabet_name": "DNA",
    },
    "Wong2018": {
        "model_type": "custom",
        "user_positions": [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        "user_orbits": [
            (-4, -3, -2, -1, 0),
            (0, 1, 2, 3, 4, 5),
        ],  # All-order on each half of the 5'ss
        "alphabet_list": [_rna_alphabet] * 4
        + [["G"]]
        + [["C", "U"]]
        + [_rna_alphabet] * 4,
    },
}


def is_visible_ascii(char):
    """
    Visible ASCII characters are defined as characters with ASCII codes between 33 and 126.
    Includes all numbers, letters, punctuation, and symbols.
    Note: the space character (ASCII code 32) is not included.
    """
    return 33 <= ord(char) <= 126


def get_site_projection_matrix(
    pi_lc: np.ndarray, lda: float = np.inf
) -> np.ndarray:
    """
    Compute the site-specific projection matrix into the gauge space.

    Parameters
    ----------
    pi_lc : np.ndarray
        Custom position-specific background frequencies.
    lda : float, optional
        A parameter that controls the scaling of the projection. Defaults to
        infinity (`np.inf`), which results in a specific scaling behavior.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the site projection matrix. The matrix
        has dimensions `(alpha + 1, alpha + 1)`, where `alpha` is the length
        of `pi_lc`.
    """
    # Handle numerical stability for lda
    if np.isinf(lda):
        eta = 1.0
    else:
        # Prevent division by zero and overflow
        if abs(lda) < 1e-15:
            eta = 0.0
        elif abs(lda) > 1e15:
            eta = 1.0
        else:
            eta = lda / (1 + lda)

    alpha = pi_lc.shape[0]
    P0x = eta * pi_lc
    Px0 = np.full((alpha, 1), 1 - eta)
    Pxx = np.eye(alpha) - eta * np.expand_dims(pi_lc, axis=0)
    P = np.vstack([np.hstack([eta, P0x]), np.hstack([Px0, Pxx])])

    # Check for numerical issues
    if np.any(np.isnan(P)) or np.any(np.isinf(P)):
        raise ValueError("Projection matrix contains NaN or infinite values")

    return P


def tensordot(linop, v, axis):
    """
    Compute the tensor dot product along a specified axis.

    Parameters
    ----------
    linop : np.ndarray
        Linear operator (matrix) to apply.
    v : np.ndarray
        Tensor to contract with the linear operator.
    axis : int
        Axis along which the contraction is performed.

    Returns
    -------
    np.ndarray
        Resulting tensor after the contraction.
    """
    # Input validation
    if np.any(np.isnan(linop)) or np.any(np.isinf(linop)):
        raise ValueError("Linear operator contains NaN or infinite values")
    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
        raise ValueError("Input tensor contains NaN or infinite values")

    u = np.moveaxis(v, axis, 0)  # Shape becomes (contract_dim,...rest_of_dims)
    u_reshaped = u.reshape(
        u.shape[0], -1
    )  # Shape: (contract_dim, rest_product)

    # Use more numerically stable matrix multiplication
    try:
        # Suppress warnings for this operation and handle them explicitly
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            x = linop @ u_reshaped

        # Check for numerical issues in the result
        if np.any(np.isnan(x)):
            # Replace NaN values with zeros (common in gauge fixing)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        elif np.any(np.isinf(x)):
            # Handle overflow by clipping to reasonable range
            max_val = np.finfo(np.float64).max / 1e10  # Conservative limit
            x = np.clip(x, -max_val, max_val)

    except (OverflowError, FloatingPointError):
        # Fallback: use more conservative computation
        # Scale down inputs to prevent overflow
        scale_factor = 1e-10
        linop_scaled = linop * scale_factor
        u_reshaped_scaled = u_reshaped * scale_factor

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            x = linop_scaled @ u_reshaped_scaled
            x = x / (scale_factor * scale_factor)  # Scale back

        # Final safety check
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    final_shape = (linop.shape[0],) + u.shape[1:]  # (m, ...rest_of_dims)
    x = x.reshape(final_shape)
    return x


def kron_matvec(matrices: list[np.ndarray], vector: np.ndarray) -> np.ndarray:
    """
    Efficiently compute the matrix-vector product where the matrix is
    a Kronecker product of smaller matrices.

    Parameters
    ----------
    matrices : list of np.ndarray
        List of matrices whose Kronecker product forms the large matrix.
    vector : np.ndarray
        Vector(s) to multiply with the Kronecker product matrix.
        Can be 1D (shape (n,)) or 2D (shape (n, m)) where each column
        is a separate vector to transform.

    Returns
    -------
    np.ndarray
        Result of the matrix-vector multiplication.
        Returns 1D if input was 1D, 2D if input was 2D.
    """
    # Input validation
    if not matrices:
        raise ValueError("Matrices list cannot be empty")

    # Check for numerical issues in matrices
    for i, m in enumerate(matrices):
        if np.any(np.isnan(m)) or np.any(np.isinf(m)):
            raise ValueError(f"Matrix {i} contains NaN or infinite values")

    # Check for numerical issues in vector
    if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
        raise ValueError("Input vector contains NaN or infinite values")

    # Normalize input to 2D
    was_1d = vector.ndim == 1
    if was_1d:
        vector = vector[:, np.newaxis]

    matrices = matrices
    tensor_shape = [m_i.shape[1] for m_i in matrices]

    m = np.prod([m_i.shape[0] for m_i in matrices])
    n = np.prod(tensor_shape)

    if vector.shape[0] != n:
        msg = f"Vector length ({vector.shape[0]}) does not match the "
        msg += f"Kronecker product matrix size ({n})"
        raise ValueError(msg)

    # Reshape to include the number of vectors as the last dimension
    num_vectors = vector.shape[1]
    u_tensor = vector.reshape(tensor_shape + [num_vectors])
    for i, mat in enumerate(matrices):
        u_tensor = tensordot(mat, u_tensor, i)
    
    # Transpose all dimensions except the last one (num_vectors) to reverse their order
    # This is needed to match the Kronecker product structure
    # u_tensor has shape (m1, m2, ..., mk, num_vectors)
    # We want to reverse the order of m1, ..., mk while keeping num_vectors last
    k = len(matrices)
    axes_permutation = list(range(k-1, -1, -1)) + [k]
    u_tensor = u_tensor.transpose(axes_permutation)
    
    # Flatten all spatial dimensions, keeping num_vectors as the last dimension
    # Result shape: (m, num_vectors) where m = m1 * m2 * ... * mk
    m_int = int(m)
    u = u_tensor.reshape(m_int, num_vectors)

    # Final check for numerical issues in result
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

    # Denormalize output if input was 1D
    if was_1d:
        u = u.squeeze()

    return u


def validate_alphabet_params(
    alphabet_name: str | None = None,
    alphabet: list[str] | None = None,
    alphabet_list: list[list[str]] | None = None,
    L: int | None = None,
) -> tuple[str | None, list[str] | None, list[list[str]], int]:
    """
    Validate alphabet parameters

    Parameters
    ----------
    alphabet_name : str, optional
        Name of a predefined alphabet to use. Must be a key in named_alphabets_dict.
        Either alphabet or alphabet_name must be provided, but not both.
    alphabet : list of str, optional
        The set of possible characters that can appear in the sequences.
        Either alphabet or alphabet_name must be provided, but not both.
    alphabet_list : list of list of str, optional
        List of alphabets, where each alphabet is a list of characters to sample
        from for that specific position. The length of alphabet_list determines
        the sequence length. Cannot be used with alphabet, alphabet_name, or L parameters.
    L : int or None, optional
        The length of the sequences for which features are being generated. Must
        be provided if alphabet or alphabet_name is provided.

    Returns
    -------
    tuple[str | None, list[str] | None, list[list[str]] | None, int]
        The validated alphabet list and length.
    """

    # Validate L
    if L is not None and L <= 0:
        raise ValueError(
            f"{L=}; if not None, L must be a positive integer, got {L}"
        )

    # If alphabet_name is provided, validate it and set alphabet
    if alphabet_name is not None:
        if alphabet is not None:
            raise ValueError(
                "Cannot specify both 'alphabet' and 'alphabet_name'"
            )
        if alphabet_name not in named_alphabets_dict:
            raise ValueError(
                f"{alphabet_name=}; must be one of {list(named_alphabets_dict.keys())}; got {alphabet_name}"
            )
        alphabet = named_alphabets_dict[alphabet_name]
        if L is None:
            raise ValueError(
                "Cannot specify 'alphabet_name' or 'alphabet' without specifying'L'"
            )
        alphabet_list = [alphabet.copy() for _ in range(L)]

    # If alphabet or alphabet_name is provided, validate it and set alphabet_list
    if alphabet is not None:
        if L is None:
            raise ValueError(
                "Cannot specify 'alphabet_name' or 'alphabet' without specifying'L'"
            )
        if len(alphabet) == 0:
            raise ValueError("Alphabet must be non-empty")
        if len(set(alphabet)) != len(alphabet):
            raise ValueError(
                f"Named alphabet {alphabet_name} contains duplicate characters"
            )
        if not all(
            isinstance(char, str) and len(char) == 1 for char in alphabet
        ):
            raise ValueError("Alphabet must be lists of single characters")
        if not all(is_visible_ascii(char) for char in alphabet):
            raise ValueError(
                "Alphabet must contain only visible ASCII characters. Invalid character found."
            )
        if not alphabet == sorted(alphabet):
            raise Warning(
                f"Alphabet {alphabet} is not sorted; sorting alphabet."
            )
        alphabet = sorted(alphabet)
        alphabet_list = [alphabet] * L

    elif alphabet_list is not None:
        if L is None:
            L = len(alphabet_list)
        elif L != len(alphabet_list):
            raise ValueError(
                f"Values for L and alphabet_list are inconsistent. {L=}; {len(alphabet_list)=}"
            )
        for i, this_alphabet in enumerate(alphabet_list):
            if not all(
                isinstance(char, str) and len(char) == 1
                for char in this_alphabet
            ):
                raise ValueError(
                    "All alphabets must be lists of single characters"
                )
            if not all(is_visible_ascii(char) for char in this_alphabet):
                raise ValueError(
                    "All alphabets must contain only visible ASCII characters. Invalid character found."
                )
            if len(this_alphabet) == 0:
                raise ValueError("All alphabets must be non-empty")
            if this_alphabet != sorted(this_alphabet):
                raise Warning(
                    f"Alphabet for position {i} is not sorted; sorting alphabet."
                )

        alphabet_list = [
            sorted(this_alphabet) for this_alphabet in alphabet_list
        ]

    else:
        raise ValueError("Invalid combination of parameters")

    return (alphabet_name, alphabet, alphabet_list, L)


@typechecked
def random_seqs(
    alphabet_name: str | None = None,
    alphabet: list[str] | None = None,
    alphabet_list: list[list[str]] | None = None,
    L: int | None = None,
    num_sequences: int = 1,
    random_seed: int | None = None,
) -> list[str]:
    """
    Generate random sequences from a given alphabet or list of alphabets.

    Creates random sequences by sampling characters from either a single alphabet
    used for all positions, or from position-specific alphabets. Each sequence
    is generated independently.

    Parameters
    ----------
    alphabet_name : str, optional
        Name of a predefined alphabet to use. Must be a key in named_alphabets_dict.
        Either alphabet or alphabet_name must be provided, but not both.
    alphabet : list[str], optional
        List of characters to sample from when generating sequences.
        Each character should be a single string. Must be provided along with L.
    alphabet_list : list[list[str]], optional
        List of alphabets, where each alphabet is a list of characters to sample
        from for that specific position. The length of alphabet_list determines
        the sequence length. Cannot be used with alphabet and L parameters.
    L : int, optional
        Length of each generated sequence. Must be a positive integer.
        Required if using the alphabet parameter.
    num_sequences : int, default=1
        Number of sequences to generate. Must be a positive integer.
    random_seed : int, optional
        Random seed for reproducible results. If None, uses the current
        random state. Default is None.

    Returns
    -------
    list[str]
        List of randomly generated sequences. Each sequence is a string
        containing characters sampled from the alphabet(s).

    Notes
    -----
    Either 'alphabet' and 'L' must be provided together, or 'alphabet_list'
    must be provided. These two options are mutually exclusive.
    """

    # Validate inputs
    _, _, alphabet_list, L = validate_alphabet_params(
        alphabet_name, alphabet, alphabet_list, L
    )

    # Validate num_sequences
    if num_sequences < 0:
        raise ValueError(
            f"num_sequences must be non-negative, got {num_sequences}"
        )

    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Generate sequences
    if alphabet_list is not None:
        sequences = [
            "".join(random.choice(alphabet_list[i]) for i in range(L))
            for _ in range(num_sequences)
        ]
    else:
        raise ValueError("alphabet list must be provided or guessed")

    return sequences


@typechecked
def sorted_tuples(tuples: list[tuple]) -> list[tuple]:
    """
    Sort tuples by length first, then lexicographically.

    Parameters
    ----------
    tuples : list of tuple
        List of tuples to be sorted.

    Returns
    -------
    list of tuple
        Sorted tuples.
    """
    return sorted(tuples, key=lambda t: (len(t), t))


@typechecked
def get_subsets_of_set(s: tuple) -> list[tuple]:
    """
    Get all subsets of a set.

    Parameters
    ----------
    s : tuple
        Input set as a tuple.

    Returns
    -------
    list of tuple
        All subsets as tuples.
    """
    return [
        tuple(subset)
        for r in range(len(s) + 1)
        for subset in combinations(s, r)
    ]


@typechecked
def get_subsets_of_multiple_sets(sets: list[tuple]) -> list[tuple]:
    """
    Get sorted list of all subsets present in any of the sets from multiple sets.

    Parameters
    ----------
    sets : list of tuple
        List of sets as tuples.

    Returns
    -------
    list of tuple
        All subsets from all sets, sorted.
    """
    return sorted_tuples(
        list(set(chain.from_iterable([get_subsets_of_set(s) for s in sets])))
    )


@typechecked
def get_orbits_features(
    orbits: list[tuple], alphabet_list: list[list[str]]
) -> list[tuple]:
    """
    Build features for given orbits using the provided alphabets.

    Parameters
    ----------
    orbits : list[tuple]
        Each orbit is a tuple of integer positions. An empty tuple
        denotes the empty feature (no positions).
    alphabet_list : list[list[str]]
        List of per-position alphabets; alphabet_list[i] is the list of
        allowed characters at position i.

    Returns
    -------
    list[tuple]
        A list of features. Each feature is a tuple (orbit, seq) where
        seq is a string obtained by taking one character from each
        position in the orbit in order. For the empty orbit the feature
        ((), "") is returned.
    """
    features = []
    for orbit in orbits:
        if len(orbit) == 0:
            features.append(((), ""))
        else:
            alphabets = [alphabet_list[i] for i in orbit]
            features.extend([(orbit, s) for s in get_all_seqs(alphabets)])
    return features


@typechecked
def get_orbits_subsequences(
    orbits: list[tuple], alphabet_list: list[list[str]], wildcard: str = "*"
) -> list[str]:
    """
    Build features for given orbits using the provided alphabets.

    Parameters
    ----------
    orbits : list[tuple]
        Each orbit is a tuple of integer positions. An empty tuple
        denotes the empty feature (no positions).
    alphabet_list : list[list[str]]
        List of per-position alphabets; alphabet_list[i] is the list of
        allowed characters at position i.
    wildcard : str

    Returns
    -------
    list[str]
        A list of subsequences
    """
    subseqs = []
    L = len(alphabet_list)
    for orbit in orbits:
        if len(orbit) == 0:
            subseqs.append(wildcard * L)
        else:
            alphabets = [
                alphabet if p in orbit else ["*"]
                for p, alphabet in enumerate(alphabet_list)
            ]
            subseqs.extend(get_all_seqs(alphabets))
    return subseqs


def get_all_seqs(alphabet_list, seqs=[""]):
    if len(alphabet_list) == 0:
        return seqs
    elif len(alphabet_list) > 100:
        return ["".join(x) for x in product(*alphabet_list)]

    alphabet = alphabet_list[-1]

    new_seqs = []
    for c in alphabet:
        new_seqs.extend([c + s for s in seqs])

    return get_all_seqs(alphabet_list[:-1], seqs=new_seqs)
