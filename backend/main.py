from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import StartupIdea
from app.services.idea_extraction import ServiceError
from app.services.orchestration import ValidationReportService
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).with_name(".env"))
app = FastAPI(title="AI Startup Validator API", version="1.0.0")
origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"message": "Startup Validator backend is running"}

@app.post("/api/ideas")
def submit_idea(idea: StartupIdea):
    exclude = set() if "estimated_budget_currency" in idea.model_fields_set else {"estimated_budget_currency"}
    return {"message": "Startup idea received successfully", "idea": idea.model_dump(exclude=exclude)}

@app.post("/api/ideas/analyze")
async def analyze_idea(idea: StartupIdea):
    try:
        return await ValidationReportService().analyze(idea)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
