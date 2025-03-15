from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.utils.logger import get_logger, setup_logging
from app.utils.config import make_cache_dir


load_dotenv()
setup_logging()
logger = get_logger()
make_cache_dir()

app = FastAPI(
    title="Enhanced Text Analysis API",
    description="An API for text analysis using ML and LLM models",
    version="1.0.0",
)
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Enhanced Text Analysis API!"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 