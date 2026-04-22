import os
import json
import numpy as np
from src.storage.turbo_quant import TurboQuant
from src.models import ClaimRecord, DenialAnalysis

class TurboStore:
    def __init__(self, store_dir: str = None, dim: int = 384):
        self.dim = dim
        self.quantizer = TurboQuant(dim=dim)
        
        if store_dir is None:
            store_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vector_store")
            
        self.store_dir = store_dir
        self.index_file = os.path.join(self.store_dir, "index.tq")
        self.meta_file = os.path.join(self.store_dir, "meta.json")
        
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir, exist_ok=True)
            
    def _build_meta_from_claim(self, claim: ClaimRecord, analysis: DenialAnalysis) -> dict:
        return {
            "claim_id": claim.claim_id,
            "carc_code": claim.carc_code,
            "insurance_type": claim.insurance_type,
            "procedure_code": claim.procedure_code,
            "payer_name": claim.payer_name,
            "recoverability": analysis.recoverability,
            "root_cause": analysis.root_cause
        }

    def upsert_claim(self, claim: ClaimRecord, analysis: DenialAnalysis, embedding: np.ndarray):
        """
        Adds a single claim's embedding and metadata to the quantizer storage.
        """
        existing_meta = []
        if os.path.exists(self.meta_file):
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
                
        # Handle duplicates/updates
        keep_indices = [i for i, m in enumerate(existing_meta) if m.get("claim_id") != claim.claim_id]
        
        if os.path.exists(self.index_file):
            existing_index_mmap = np.memmap(self.index_file, dtype=np.uint8, mode='r')
            existing_index = np.array(existing_index_mmap).reshape(-1, self.dim // 2)
            del existing_index_mmap
            existing_index = existing_index[keep_indices]
            existing_meta = [existing_meta[i] for i in keep_indices]
        else:
            existing_index = None

        new_packed = self.quantizer.quantize(embedding)
        if existing_index is not None and len(existing_index) > 0:
            combined_index = np.vstack([existing_index, new_packed])
        else:
            combined_index = new_packed
            
        combined_meta = existing_meta + [self._build_meta_from_claim(claim, analysis)]
        
        with open(self.index_file, 'wb') as f:
            f.write(combined_index.tobytes())
            
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(combined_meta, f, ensure_ascii=False, indent=2)

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Retrieves similar claims based on the query embedding.
        """
        if not os.path.exists(self.index_file) or not os.path.exists(self.meta_file):
            return []
            
        packed_index = np.memmap(self.index_file, dtype=np.uint8, mode='r').reshape(-1, self.dim // 2)
            
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        if len(meta) == 0:
            return []
            
        scores = self.quantizer.unbiased_inner_product(query_embedding, packed_index)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            res = meta[idx].copy()
            res["score"] = float(scores[idx])
            results.append(res)
            
        del packed_index
        return results
