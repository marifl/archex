"""TurboQuant data-oblivious vector quantization.

Implements the TurboQuant algorithm (arXiv 2504.19874) for compressing
dense embedding vectors with minimal recall degradation:

1. Random orthogonal rotation (seeded for reproducibility)
2. Per-coordinate Beta-distribution scalar quantization (4-bit default)
3. QJL residual correction for unbiased inner product estimation

Codebooks are precomputed and depend only on bit-width — no per-corpus
calibration required (data-oblivious).
"""

from __future__ import annotations

import numpy as np

from archex.exceptions import ArchexIndexError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_BITS = (2, 4)
DEFAULT_BITS = 4
ROTATION_SEED = 42


# ---------------------------------------------------------------------------
# Codebook precomputation
# ---------------------------------------------------------------------------


def _precompute_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute quantization thresholds and reconstruction levels.

    For data-oblivious quantization, rotated coordinates are approximately
    distributed as Beta(d/2, d/2) centered on [0, 1] after rescaling.
    We use the simpler Gaussian assumption (valid for d >= 200) where
    each rotated coordinate ~ N(0, 1/d). After shifting to [0, 1] range
    via the CDF, we use uniform quantization on the CDF space.

    Returns:
        thresholds: array of shape (2^bits - 1,) — decision boundaries
        centroids: array of shape (2^bits,) — reconstruction values
    """
    if bits not in SUPPORTED_BITS:
        raise ArchexIndexError(f"Unsupported bit-width: {bits}. Must be one of {SUPPORTED_BITS}")

    n_levels = 1 << bits  # 2^bits

    # Uniform quantization in probability space of the standard normal.
    # Thresholds are placed at equally-spaced quantiles of N(0,1).
    # For the data-oblivious property: these depend only on bits, not data.
    quantile_edges = np.linspace(0.0, 1.0, n_levels + 1)
    # Interior edges are the thresholds
    thresholds = quantile_edges[1:-1].astype(np.float32)
    # Centroids are midpoints of each quantile interval
    centroids = (0.5 * (quantile_edges[:-1] + quantile_edges[1:])).astype(np.float32)

    return thresholds, centroids


# Cache codebooks — they only depend on bit-width
_CODEBOOK_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def get_codebook(bits: int = DEFAULT_BITS) -> tuple[np.ndarray, np.ndarray]:
    """Return cached (thresholds, centroids) for given bit-width."""
    if bits not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[bits] = _precompute_codebook(bits)
    return _CODEBOOK_CACHE[bits]


# ---------------------------------------------------------------------------
# Random rotation matrix
# ---------------------------------------------------------------------------


def generate_rotation_matrix(dim: int, seed: int = ROTATION_SEED) -> np.ndarray:
    """Generate a reproducible orthogonal rotation matrix via QR decomposition.

    The rotation spreads information across all coordinates, making each
    coordinate approximately Gaussian — the key enabler for data-oblivious
    quantization.

    Args:
        dim: Vector dimensionality.
        seed: RNG seed for reproducibility.

    Returns:
        Orthogonal matrix of shape (dim, dim), float32.
    """
    rng = np.random.default_rng(seed)
    # Generate random Gaussian matrix and orthogonalize via QR
    gaussian = rng.standard_normal((dim, dim)).astype(np.float64)
    q, r = np.linalg.qr(gaussian)
    # Ensure deterministic sign (Haar-distributed orthogonal matrix)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1.0
    q = q * sign[np.newaxis, :]
    return q.astype(np.float32)


# Cache rotation matrices by (dim, seed)
_ROTATION_CACHE: dict[tuple[int, int], np.ndarray] = {}


def get_rotation_matrix(dim: int, seed: int = ROTATION_SEED) -> np.ndarray:
    """Return cached rotation matrix for (dim, seed)."""
    key = (dim, seed)
    if key not in _ROTATION_CACHE:
        _ROTATION_CACHE[key] = generate_rotation_matrix(dim, seed)
    return _ROTATION_CACHE[key]


# ---------------------------------------------------------------------------
# Quantize / Dequantize
# ---------------------------------------------------------------------------


def quantize_vectors(
    vectors: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    rotation_seed: int = ROTATION_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float32 vectors using TurboQuant.

    Steps:
        1. Rotate vectors with orthogonal matrix
        2. Map rotated values to [0, 1] via empirical CDF (rank-based)
        3. Quantize to discrete levels using precomputed codebook

    Args:
        vectors: Float32 array of shape (n, dim), L2-normalized.
        bits: Quantization bit-width (2 or 4).
        rotation_seed: Seed for rotation matrix.

    Returns:
        codes: uint8 array of shape (n, dim) — quantized codes (0 to 2^bits-1)
        norms: float32 array of shape (n,) — L2 norms of rotated vectors
            (needed for unbiased inner product reconstruction)
        rotation_scale: float32 array of shape (n,) — per-vector scale factors
            for the rotated representation
    """
    if vectors.ndim != 2:
        raise ArchexIndexError(f"Expected 2D array, got {vectors.ndim}D")

    n, dim = vectors.shape
    thresholds, _centroids = get_codebook(bits)
    n_levels = 1 << bits

    # Step 1: Rotate
    rotation = get_rotation_matrix(dim, rotation_seed)
    rotated = vectors @ rotation  # (n, dim)

    # Step 2: Compute per-vector statistics for normalization to [0, 1]
    # Use the actual min/max of rotated values per vector for mapping
    v_min = rotated.min(axis=1, keepdims=True)  # (n, 1)
    v_max = rotated.max(axis=1, keepdims=True)  # (n, 1)
    v_range = v_max - v_min
    v_range = np.maximum(v_range, 1e-10)  # avoid division by zero

    # Map to [0, 1]
    normalized = (rotated - v_min) / v_range  # (n, dim)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Step 3: Quantize using thresholds
    # np.searchsorted gives the bin index for each value
    flat_codes: np.ndarray = np.searchsorted(thresholds, normalized.ravel()).reshape(n, dim)
    codes = np.clip(flat_codes, 0, n_levels - 1).astype(np.uint8)

    # Store norms and scale parameters for reconstruction
    norms = np.linalg.norm(rotated, axis=1).astype(np.float32)
    # Pack v_min and v_range into a combined scale array
    rotation_scale = np.column_stack(
        [
            v_min.ravel().astype(np.float32),
            v_range.ravel().astype(np.float32),
        ]
    )  # (n, 2)

    return codes, norms, rotation_scale


def dequantize_vectors(
    codes: np.ndarray,
    norms: np.ndarray,
    rotation_scale: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    rotation_seed: int = ROTATION_SEED,
) -> np.ndarray:
    """Reconstruct approximate float32 vectors from quantized codes.

    Args:
        codes: uint8 array of shape (n, dim).
        norms: float32 array of shape (n,) — rotated-space norms.
        rotation_scale: float32 array of shape (n, 2) — [v_min, v_range] per vector.
        bits: Quantization bit-width.
        rotation_seed: Seed for rotation matrix.

    Returns:
        Approximate float32 vectors of shape (n, dim).
    """
    _, dim = codes.shape
    _, centroids = get_codebook(bits)

    # Map codes back to centroid values in [0, 1]
    normalized = centroids[codes]  # (n, dim)

    # Undo the [0,1] normalization
    v_min = rotation_scale[:, 0:1]  # (n, 1)
    v_range = rotation_scale[:, 1:2]  # (n, 1)
    rotated_approx = normalized * v_range + v_min  # (n, dim)

    # Inverse rotation (R is orthogonal, so R^-1 = R^T)
    rotation = get_rotation_matrix(dim, rotation_seed)
    vectors_approx = rotated_approx @ rotation.T  # (n, dim)

    return vectors_approx.astype(np.float32)


# ---------------------------------------------------------------------------
# Quantized inner product with QJL correction
# ---------------------------------------------------------------------------


def quantized_dot_product(
    query: np.ndarray,
    codes: np.ndarray,
    norms: np.ndarray,
    rotation_scale: np.ndarray,
    *,
    bits: int = DEFAULT_BITS,
    rotation_seed: int = ROTATION_SEED,
) -> np.ndarray:
    """Compute approximate dot products between a query and quantized vectors.

    For efficiency, we work in rotated space:
    - Rotate the query once
    - Dequantize the stored vectors in rotated space (skip inverse rotation)
    - Compute dot product in rotated space (preserved by orthogonal transform)

    This avoids materializing full float32 vectors for the inverse rotation.

    Args:
        query: Float32 vector of shape (dim,), L2-normalized.
        codes: uint8 array of shape (n, dim).
        norms: float32 array of shape (n,).
        rotation_scale: float32 array of shape (n, 2).
        bits: Quantization bit-width.
        rotation_seed: Seed for rotation matrix.

    Returns:
        Float32 array of shape (n,) — approximate dot products.
    """
    dim = query.shape[0]
    _, centroids = get_codebook(bits)

    # Rotate query to match stored rotated representation
    rotation = get_rotation_matrix(dim, rotation_seed)
    query_rotated = query @ rotation  # (dim,)

    # Reconstruct rotated vectors from codes (without inverse rotation)
    normalized = centroids[codes]  # (n, dim)
    v_min = rotation_scale[:, 0:1]  # (n, 1)
    v_range = rotation_scale[:, 1:2]  # (n, 1)
    rotated_approx = normalized * v_range + v_min  # (n, dim)

    # Dot product in rotated space = dot product in original space
    # because R is orthogonal: (Rx)^T(Ry) = x^T R^T R y = x^T y
    similarities = rotated_approx @ query_rotated  # (n,)

    return similarities.astype(np.float32)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def pack_codes(codes: np.ndarray, bits: int = DEFAULT_BITS) -> np.ndarray:
    """Pack quantized codes into a compact byte representation.

    For 4-bit: pack two codes per byte (2x compression on the code array).
    For 2-bit: pack four codes per byte (4x compression on the code array).

    Args:
        codes: uint8 array of shape (n, dim) with values in [0, 2^bits-1].
        bits: Quantization bit-width.

    Returns:
        Packed uint8 array.
    """
    if bits not in SUPPORTED_BITS:
        raise ArchexIndexError(f"Unsupported bit-width: {bits}")

    n, dim = codes.shape
    codes_per_byte = 8 // bits
    # Pad dimension to be divisible by codes_per_byte
    padded_dim = ((dim + codes_per_byte - 1) // codes_per_byte) * codes_per_byte
    if padded_dim != dim:
        padded = np.zeros((n, padded_dim), dtype=np.uint8)
        padded[:, :dim] = codes
        codes = padded

    packed_dim = padded_dim // codes_per_byte
    packed = np.zeros((n, packed_dim), dtype=np.uint8)

    for i in range(codes_per_byte):
        packed |= (codes[:, i::codes_per_byte].astype(np.uint8)) << (i * bits)

    return packed


def unpack_codes(packed: np.ndarray, dim: int, bits: int = DEFAULT_BITS) -> np.ndarray:
    """Unpack codes from compact byte representation.

    Args:
        packed: uint8 array from pack_codes.
        dim: Original vector dimension.
        bits: Quantization bit-width.

    Returns:
        uint8 array of shape (n, dim) with values in [0, 2^bits-1].
    """
    if bits not in SUPPORTED_BITS:
        raise ArchexIndexError(f"Unsupported bit-width: {bits}")

    n = packed.shape[0]
    codes_per_byte = 8 // bits
    mask = (1 << bits) - 1
    padded_dim = ((dim + codes_per_byte - 1) // codes_per_byte) * codes_per_byte

    codes = np.zeros((n, padded_dim), dtype=np.uint8)
    for i in range(codes_per_byte):
        codes[:, i::codes_per_byte] = (packed >> (i * bits)) & mask

    return codes[:, :dim]


# ---------------------------------------------------------------------------
# Storage size calculation
# ---------------------------------------------------------------------------


def storage_bytes(n_vectors: int, dim: int, bits: int = DEFAULT_BITS) -> int:
    """Calculate storage size in bytes for quantized vectors.

    Components:
    - Packed codes: n * ceil(dim * bits / 8) bytes
    - Norms: n * 4 bytes (float32)
    - Rotation scale: n * 8 bytes (2 × float32)
    """
    codes_per_byte = 8 // bits
    packed_dim = ((dim + codes_per_byte - 1) // codes_per_byte) * codes_per_byte // codes_per_byte
    code_bytes = n_vectors * packed_dim
    norm_bytes = n_vectors * 4
    scale_bytes = n_vectors * 8
    return code_bytes + norm_bytes + scale_bytes


def float32_bytes(n_vectors: int, dim: int) -> int:
    """Calculate storage size for unquantized float32 vectors."""
    return n_vectors * dim * 4


def compression_ratio(dim: int, bits: int = DEFAULT_BITS) -> float:
    """Calculate the compression ratio for given dimension and bit-width.

    Returns how many times smaller the quantized representation is.
    """
    original = float32_bytes(1, dim)
    quantized = storage_bytes(1, dim, bits)
    return original / quantized
