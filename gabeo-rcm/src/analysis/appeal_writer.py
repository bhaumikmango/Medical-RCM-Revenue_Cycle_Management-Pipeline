import os
import datetime
import logging
from src.models import ClaimRecord, DenialAnalysis
from src.llm.client import llm_client

logger = logging.getLogger(__name__)

async def generate_appeal_letter(analysis: DenialAnalysis, claim: ClaimRecord) -> str:
    """
    Generates a professional insurance appeal letter based on denial analysis and claim data.
    
    Args:
        analysis (DenialAnalysis): The P1 analysis output.
        claim (ClaimRecord): The original claim data.
        
    Returns:
        str: The full text of the appeal letter.
    """
    # Load prompt template
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts", "appeal_letter.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Appeal letter prompt template not found at {prompt_path}")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Pre-process evidence into numbered plain-English sentences
    evidence_list_formatted = "\n".join(
        f"{i+1}. {e}" for i, e in enumerate(analysis.evidence)
    )

    # Prepare data for template
    # Note: procedure_description is not in models.py, using a placeholder or code
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Map data to template placeholders
    try:
        prompt = prompt_template.format(
            claim_id=claim.claim_id,
            service_date=claim.service_date_from,
            received_date=claim.received_date,
            procedure_code=claim.procedure_code,
            procedure_description="Healthcare Service", # Default placeholder
            principal_diagnosis=claim.principal_diagnosis or "N/A",
            payer_name=claim.payer_name,
            insurance_type=claim.insurance_type,
            subscriber_id=claim.subscriber_id,
            denial_reason=claim.carc_description,
            carc_code=claim.carc_code,
            current_date=current_date,
            root_cause=analysis.root_cause,
            recoverability=analysis.recoverability,
            evidence_list_formatted=evidence_list_formatted
        )
    except KeyError as e:
        logger.error(f"Missing placeholder in appeal_letter.txt: {e}")
        raise ValueError(f"Prompt template requires missing key: {e}")

    # Generate letter using async client
    logger.info(f"Generating appeal letter for claim {claim.claim_id}...")
    letter_text = await llm_client.generate_async(prompt)
    
    if not letter_text:
        logger.error(f"Failed to generate appeal letter for {claim.claim_id}")
        return "Error: Could not generate appeal letter."

    return letter_text.strip()
