import os

import certifi
from fastapi import FastAPI, Request, status
from fastapi.responses import ORJSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.routes import router as v1_router
from app.core.config import Settings
from app.core.errors import BadRequest, DependencyError
from app.core.lifespan import build_lifespan
from app.core.logging import configure_logging


# Optional cross-platform trust store injection.
# If truststore isn't installed yet, app still boots with certifi bundle.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def create_app() -> FastAPI:
    settings = Settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        default_response_class=ORJSONResponse,
        lifespan=build_lifespan(settings),
    )

    # Add CORS middleware to allow frontend requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default port
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(BadRequest)
    async def bad_request_handler(request: Request, exc: BadRequest):
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        # Convert ValueError to BadRequest format for consistency
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(DependencyError)
    async def dependency_error_handler(request: Request, exc: DependencyError):
        return ORJSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return ORJSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Format validation errors into a readable message
        errors = exc.errors()
        if errors:
            # Extract the first error message for clarity
            first_error = errors[0]
            field = ".".join(str(loc) for loc in first_error.get("loc", []))
            error_msg = first_error.get("msg", "Validation error")
            detail = f"Validation error for field '{field}': {error_msg}"
            if len(errors) > 1:
                detail += f" (and {len(errors) - 1} more error(s))"
        else:
            detail = "Validation error"
        
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": detail},
        )

    app.include_router(v1_router, prefix="/api/v1")
    return app


app = create_app()