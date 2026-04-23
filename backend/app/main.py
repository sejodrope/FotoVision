from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import init_db
from app.api.routes import diagnosis, history, models_info
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="FitoVision API",
    description="Detecção automática de anomalias fitossanitárias em hortaliças folhosas.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnosis.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(models_info.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "demo_mode": settings.demo_mode}
