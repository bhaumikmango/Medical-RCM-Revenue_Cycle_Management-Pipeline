from fastapi import APIRouter, HTTPException
from src.pipeline import Pipeline
from app.core.config import settings

router = APIRouter(prefix="/api")
pipeline = Pipeline()

@router.get("/stats")
async def get_stats():
    """Returns high-level statistics for the dashboard."""
    results = pipeline.claim_store.get_all_denied_claims_with_analysis()
    if not results:
        return {
            "total_claims": 0,
            "total_value": 0,
            "recoverable_count": 0,
            "recoverable_value": 0,
            "recovery_rate": 0,
            "avg_confidence": 0
        }
    
    total_val = sum(c['claim']['claim_amount'] for c in results)
    recoverable = [c for c in results if c['analysis']['recoverability'] == 'recoverable']
    rec_val = sum(c['claim']['claim_amount'] for c in recoverable)
    
    return {
        "total_claims": len(results),
        "total_value": total_val,
        "recoverable_count": len(recoverable),
        "recoverable_value": rec_val,
        "recovery_rate": round(len(recoverable) / len(results) * 100, 1),
        "avg_confidence": round(sum(c['analysis']['confidence'] for c in results) / len(results), 2)
    }

@router.get("/claims")
async def get_claims():
    """Returns all claims with their analysis."""
    return pipeline.claim_store.get_all_denied_claims_with_analysis()

@router.get("/trends")
async def get_trends():
    """Returns systemic denial trends."""
    return pipeline.run_trend_analysis(min_claims=2)

@router.get("/clusters")
async def get_clusters():
    """Returns P3 clusters."""
    return pipeline.run_clustering()
