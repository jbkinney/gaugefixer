ALPHABET = '''
alphabet : list of str, optional
    The set of possible characters that can appear in the sequences.
    Either ``alphabet`` or ``alphabet_name`` must be provided, but not both.
'''

ALPHABET_NAME = '''
alphabet_name : str, optional
    Name of a predefined alphabet to use, one of
    {"dna", "rna", "protein", "binary", "ternary", "decimal"}.
    Either ``alphabet`` or ``alphabet_name`` must be provided, but not both.
'''

ALPHABET_LIST = '''
alphabet_list : list of list of str, optional
    List of alphabets, where each alphabet is a list of characters to sample
    from for that specific position. The length of ``alphabet_list`` determines
    the sequence length. Cannot be used with ``alphabet``, ``alphabet_name``, or ``L`` parameters.
'''

L = '''
L : int or None, optional
        The length of the sequences for which features are being generated.
'''

K = '''
K : int
    The order of the model (size of interaction terms).
'''

ORBITS = '''
orbits : list of tuple, optional
    Covering orbits to use. If ``positions`` is provided,
    the covering orbits are expected to be in terms of these ``positions``.
'''

ORBITS_NO_POS = '''
orbits : list of tuple, optional
    Covering orbits to use.
'''

ORBITS_OUT = '''
orbits : list of tuple, optional
    Model's set of orbits.
'''

GENERATING_ORBITS = '''
orbits : list of tuple, optional
    Generating set of orbits, such that the model includes all features
    for all subsets of sites for each generating orbit.
'''

POSITIONS = '''
positions : list of int, optional
    Positions to use if different from ``range(self.L)``.
    If ``positions=None``, all positions are included.
'''

THETA = '''
theta : pd.Series of shape (n_features,), optional
    Model parameters indexed by features.
'''

THETA_OUT = '''
theta : pd.Series of shape (n_features,)  
    Model parameters indexed by features.
'''

THETA_FIXED = '''
theta_fixed : pd.Series of shape (n_features,)
    Gauge-fixed parameters indexed by features.
'''

SEQS = '''
seqs : list of str
    List of input sequences. All sequences must be of length ``L``.
'''

F = '''
f : pd.Series
    Landscape values indexed by sequences.
'''

BINARY_FEATURES = '''
x : numpy.ndarray
    A 2D binary array (of type int8) of shape (n_sequences, n_features).
'''

GAUGE_ALL_ORDERS = '''
gauge : str or None, optional

    Specifies the type of gauge fixing to apply:

    - ``'wild-type'``: Fix parameters relative to a wild-type sequence.

    - ``'zero-sum'``: Use uniform background frequencies (``pi_lc=1/n_alleles``).

    - ``'hierarchical'``: Hierarchical gauge fixing with provided ``pi_lc``.

    - ``'trivial'``: No gauge fixing (``lambda=0``).

    - ``'euclidean'``: Euclidean gauge fixing (``lambda=1``).

    - ``'equitable'``: Equitable gauge fixing (``lambda=n_alleles``).

    - ``None``: Custom gauge fixing with provided ``lambda`` and ``pi_lc``.
'''

GAUGE_HIERARCHICAL = '''
gauge : str or None, optional

    Specifies the type of gauge fixing to apply:

    - ``'wild-type'``: Fix parameters relative to a wild-type sequence.

    - ``'zero-sum'``: Use uniform background frequencies (``pi_lc=1/n_alleles``).

    - ``'hierarchical'``: Hierarchical gauge fixing with provided ``pi_lc``.
'''

WT_SEQ = '''
wt_seq : str or None, optional
    Wild-type sequence for ``gauge='wild-type'`` gauge fixing.
'''

PI_LC = '''
pi_lc : list of np.ndarray or None, optional
    Pi parameter, which specifies specifies the probability of each character at each position
    used when computing the variance explained by lower-order features.
'''

LDA = '''
lda : float, np.ndarray or None, optional
    Lambda parameter, which controls how much variance should be explained by the lower-order features.
'''

FEATURES = '''
features : list[tuple] or None, optional
    Predefined features to use. If ``None``, features will be generated from ``orbits``.
'''

FEATURES_OUT = '''
features : list[tuple]
    Model's list of binary features.
'''

PS = '''
Ps : list of np.ndarray
    List of site-specific projection matrices.
'''

DENSE_MATRIX='''
use_dense_matrix: bool
    Fix the gauge building the explicit dense projection matrix.
    Implemented mainly for testing and benchmarking.
'''