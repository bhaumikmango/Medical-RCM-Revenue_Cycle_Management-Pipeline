import logging
from typing import List
import pandas as pd

from src.ingestion.loader import Loader
from src.ingestion.carc_lookup import CarcLookup
from src.analysis.p1_root_cause import RootCauseAnalyzer
from src.storage.claim_vector_store import TurboStore
from src.analysis.p2_pattern_match import PatternMatcher
from src.storage.claim_store import ClaimStore
from src.analysis.p3_clustering import DenialClusterer

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.loader = Loader()
        self.carc_lookup = CarcLookup("data/raw/carc_reference.csv")
        self.p1_analyzer = RootCauseAnalyzer(self.carc_lookup)
        
        self.vector_store = TurboStore()
        self.p2_matcher = PatternMatcher(store=self.vector_store)
        
        self.claim_store = ClaimStore()
        self.clusterer = DenialClusterer()

    def run_batch_analysis(self, claims_data: List[dict]):
        """
        Runs the end-to-end P1/P2 pipeline on a list of unjoined claims.
        claims_data should be a list of dicts: {"835": {...}, "837": {...}}
        """
        for data in claims_data:
            d835 = data.get("835", {})
            d837 = data.get("837", {})
            
            # Step 1: Ingest
            claim_record = self.loader._to_claim_record(d835, d837)
            
            # Step 2: Get Context
            similar_ctx = self.p2_matcher.get_similar_context(claim_record)
            
            # Step 3: P1 Analysis
            analysis = self.p1_analyzer.analyze_claim(claim_record, similar_ctx)
            
            # Step 4: Persist
            self.claim_store.save_claim(claim_record)
            self.claim_store.save_analysis(analysis)
            
            embedding = self.p2_matcher.embed_claim(claim_record)
            self.vector_store.upsert_claim(claim_record, analysis, embedding)
            
            logger.info(f"Processed {claim_record.claim_id}: {analysis.recoverability}")

    def run_clustering(self):
        """
        Runs the P3 clustering on all denied claims currently in the SQLite store.
        """
        combined_data = self.claim_store.get_all_denied_claims_with_analysis()
        clusters = self.clusterer.cluster_denials(combined_data)
        return clusters
