from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import laya

router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    print("Loading Laya router onto GPU...")
    router = laya.Router(device="cuda", preload=True)
    yield
    router = None

app = FastAPI(title="Laya Decision API", lifespan=lifespan)

class DecisionRequest(BaseModel):
    state: Union[str, Dict[str, Any]]
    questions: Dict[str, Dict[str, Any]]

@app.post("/predict")
def predict_decision(payload: DecisionRequest):
    if router is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        formatted_questions = {}
        for q_id, q_data in payload.questions.items():
            instructions = q_data.get("instructions", f"Choose the best {q_id}")
            criteria = q_data.get("criteria", {})
            
            # If options passed as list, convert to dict
            if not criteria and "options" in q_data:
                criteria = {opt: "" for opt in q_data["options"]}

            formatted_questions[q_id] = {
                "instructions": instructions,
                "criteria": criteria,
                "type": q_data.get("type", "choice")
            }

        result = router.predict(state=payload.state, questions=formatted_questions)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))