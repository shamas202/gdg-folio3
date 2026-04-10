from fastapi import APIRouter

from app.api.v1.endpoints.catalog import router as catalog_router
from app.api.v1.endpoints.detection import router as detection_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.search import router as search_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(search_router, tags=["search"])
router.include_router(catalog_router, tags=["catalog"])
router.include_router(detection_router, tags=["detection"])
