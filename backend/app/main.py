import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import get_settings
from .services.classifier import Classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("krishimithra")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at start-up, not per request. If that fails the API still
    # boots: /health reports "degraded" and /api/leaf/predict answers 503.
    try:
        app.state.classifier = Classifier.load(get_settings())
    except Exception:
        log.exception("Model failed to load; running in degraded mode")
        app.state.classifier = None
    yield
    app.state.classifier = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="KrishiMithra API",
        version="2.0.0",
        description="Disease/Pest CNN tool for the Crop Health Agent.",
        lifespan=lifespan,
    )
    wildcard = "*" in settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else list(settings.cors_origins),
        allow_credentials=not wildcard,  # browsers reject credentials combined with a "*" origin
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)

    # One deployable service: if the web app has been built (npm run build), serve it too.
    if settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        def root():
            return {"service": "KrishiMithra API", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
