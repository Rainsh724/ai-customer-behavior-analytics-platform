from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.customers import router as customer_router

app = FastAPI(title="Rahin Customer Export API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(customer_router)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "rahin-customer-export-api"}
