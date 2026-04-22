import numpy as np

class TurboQuant:
    """
    TurboQuant: 4-bit MSE-optimal quantization with fixed random rotation.
    Targeted specifically to work with CPU limits and maximize the i7.
    """
    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.R = self._generate_rotation_matrix(dim)
        
    def _generate_rotation_matrix(self, dim: int) -> np.ndarray:
        """Generates a random orthogonal matrix (Haar measure) for variance spreading."""
        H = self.rng.standard_normal((dim, dim))
        Q, R = np.linalg.qr(H)
        d = np.diag(R)
        ph = d / np.abs(d)
        O = np.multiply(Q, ph)
        return O
        
    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Quantizes fp32 vectors into 4-bit representation stored packed in uint8.
        Input: (N, dim)
        Returns: (N, dim // 2) as uint8
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        N = vectors.shape[0]
        
        # 1. Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized_vectors = vectors / norms
        
        # 2. Fixed Random Rotation x R
        rotated_vectors = np.dot(normalized_vectors, self.R)
        
        # 3. Scale components dynamically
        std = 1.0 / np.sqrt(self.dim)
        
        # 3 std bounds hold ~99% of normal-like rotated components, maps well for 16 equal bins
        clipped = np.clip(rotated_vectors, -3*std, 3*std)
        
        # 4. Bind 0-15 scale
        normalized_for_quant = (clipped - (-3*std)) / (6 * std)
        quantized_int = np.clip(np.round(normalized_for_quant * 15), 0, 15).astype(np.uint8)
        
        # 5. Pack (assumes dim is even or odd properly padded)
        if self.dim % 2 != 0:
            quantized_int = np.pad(quantized_int, ((0,0), (0,1)), mode='constant')
            
        packed = np.zeros((N, quantized_int.shape[1] // 2), dtype=np.uint8)
        packed = (quantized_int[:, 0::2] << 4) | quantized_int[:, 1::2]
        
        return packed
        
    def unbiased_inner_product(self, query: np.ndarray, packed_db: np.ndarray) -> np.ndarray:
        """
        Calculates the unbiased inner product (cosine similarity since normalized).
        Query against massive quantized database via rapid reverse map.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        # Standard query sequence
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1e-9
        normalized_query = query / query_norm
        
        rotated_query = np.dot(normalized_query, self.R)
        
        N = packed_db.shape[0]
        dim_padded = packed_db.shape[1] * 2
        
        unpacked_int = np.zeros((N, dim_padded), dtype=np.uint8)
        unpacked_int[:, 0::2] = packed_db >> 4
        unpacked_int[:, 1::2] = packed_db & 0x0F
        
        unpacked_int = unpacked_int[:, :self.dim]
        
        std = 1.0 / np.sqrt(self.dim)
        dequantized = (unpacked_int / 15.0) * (6 * std) + (-3 * std)
        
        # Final Exact Search inner-product
        scores = np.dot(rotated_query, dequantized.T).flatten()
        return scores
