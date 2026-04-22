from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            # Force CPU usage specifically to save VRAM and use the powerful CPU
            cls._instance.model = SentenceTransformer(
                "Snowflake/snowflake-arctic-embed-xs",
                device="cpu",
                trust_remote_code=True
            )
            cls._instance.dim = 384  # snowflake-arctic-embed-xs output dimension
        return cls._instance
        
    def embed_text(self, text: str | list[str]):
        """Returns numpy array of embeddings"""
        if isinstance(text, str):
            text = [text]
        # Using prefix per snowflake arctic instructions for generic retrieval
        prefixed_texts = [f"Represent this sentence for searching relevant passages: {t}" for t in text]
        return self.model.encode(prefixed_texts, normalize_embeddings=True)

# Export a configured singleton
embedder_client = Embedder()
