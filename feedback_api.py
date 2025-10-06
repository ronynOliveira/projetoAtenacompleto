from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime

router = APIRouter()

# Configuração de logging para feedback
feedback_logger = logging.getLogger("FeedbackLogger")
feedback_logger.setLevel(logging.INFO)
handler = logging.FileHandler("feedback.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
feedback_logger.addHandler(handler)

class FeedbackRequest(BaseModel):
    task_id: str
    feedback_type: str # "up" or "down"
    comment: str = None # Optional comment

@router.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest):
    try:
        timestamp = datetime.now().isoformat()
        log_message = f"Feedback recebido - Task ID: {feedback.task_id}, Tipo: {feedback.feedback_type}, Comentário: {feedback.comment if feedback.comment else 'N/A'}"
        feedback_logger.info(log_message)
        return {"status": "success", "message": "Feedback registrado com sucesso."}
    except Exception as e:
        feedback_logger.error(f"Erro ao registrar feedback: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao processar feedback.")
