from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.routers import claims
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

# Mount static files
if os.path.exists(settings.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Include routers
app.include_router(claims.router)

@app.get("/")
async def read_index():
    index_path = os.path.join(settings.TEMPLATE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Index file not found in templates directory."}
