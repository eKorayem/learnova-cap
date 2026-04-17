from fastapi import FastAPI, APIRouter, Depends
from dotenv import load_dotenv
from helpers.config import get_settings, Settings
import os

load_dotenv(".env")

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)


@base_router.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy"}


@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    return {
        "App Name": app_name,
        "App Version": app_version
    }

@base_router.get("/system-info")
async def system_info():
    import torch
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else None,
        "gpu_memory_free": f"{(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB" if torch.cuda.is_available() else None
    }