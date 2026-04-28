import os
import uvicorn

if __name__ == "__main__":
    reload = os.getenv("FITOVISION_RELOAD", "true").lower() == "true"
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=reload,
    )
