from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from storage import get_storage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/dashboard")
def dashboard_data():
    storage = get_storage()
    bubble = storage.get_latest_bubble_reading()

    if not bubble:
        return {"error": "no data"}

    return {
        "vix": bubble["vix"],
        "bubble_index": bubble["bubble_index"],
        "credit_spread": bubble["credit_spread_ig"], 
        "master_signal": bubble["regime"],
        "narrative": f"Bubble Index at {bubble['bubble_index']}, regime: {bubble['regime']}."
    }
