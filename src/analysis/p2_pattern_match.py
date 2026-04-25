from src.models import ClaimRecord
from src.storage.claim_vector_store import TurboStore
from src.storage.embedder import embedder_client

class PatternMatcher:
    def __init__(self, store: TurboStore = None):
        self.store = store or TurboStore()
        
    def _build_embedding_text(self, claim: ClaimRecord) -> str:
        """
        Creates a rich semantic representation of the claim for embedding.
        Focuses on the combination of payer, procedure, diagnosis and reason.
        """
        return f"CARC:{claim.carc_code} payer:{claim.payer_name} proc:{claim.procedure_code} dx:{claim.principal_diagnosis} ins:{claim.insurance_type}"

    def embed_claim(self, claim: ClaimRecord):
        """
        Generates the 384-dimensional numpy embedding for the claim.
        """
        text = self._build_embedding_text(claim)
        return embedder_client.embed_text(text)

    def get_similar_context(self, claim: ClaimRecord, top_k: int = 5) -> str:
        """
        Retrieves similar historical claims and formats them into a context string 
        for injection into the LLM prompt.
        """
        embedding = self.embed_claim(claim)
        
        # Squeeze in case shape is (1, dim)
        if len(embedding.shape) > 1 and embedding.shape[0] == 1:
            embedding = embedding[0]
            
        similar_claims = self.store.search_similar(embedding, top_k=top_k)
        
        if not similar_claims:
            return "No similar historical claims found in the database."
            
        lines = []
        for c in similar_claims:
            score_pct = int(c.get('score', 0) * 100)
            lines.append(
                f"- Claim {c['claim_id']} ({score_pct}% match): "
                f"CARC {c['carc_code']}, Payer: {c['payer_name']}, "
                f"Outcome: {c['recoverability'].upper()}. "
                f"Root cause: {c['root_cause']}"
            )
            
        return "\n".join(lines)
