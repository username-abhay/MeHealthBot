# api/main.py
from fastapi import FastAPI
from .routes import router 

app = FastAPI(title="Health-Log Chatbot API", version="1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router, prefix="/api")

# Optional root endpoint
@app.get("/")
def root():
    return {"message": "Health-Log Chatbot API is running."}
