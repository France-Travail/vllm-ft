# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)


from vllm.entrypoints.openai import utils_ft
from vllm.version import ORIGINAL_VLLM_VERSION


@router.get("/v1/launch_arguments")
async def show_launch_arguments(raw_request: Request):
    if raw_request.app.state.arguments is None:
        return base(raw_request).create_error_response(
            message="Launch arguments is not enabled")
    else:
        return JSONResponse(content=raw_request.app.state.arguments)


@router.get("/v1/info")
async def get_info(raw_request: Request):
    model_name = None
    if raw_request.app.state.served_model_names:
        model_name = raw_request.app.state.served_model_names[0]
    content = {
        "application": "vllm_ft",
        "version": utils_ft.get_package_version().split("+")[0],
        "vllm_version": ORIGINAL_VLLM_VERSION,
        "model_name": model_name,
        "max_length": raw_request.app.state.model_config.max_model_len,
        "extra_information": raw_request.app.state.extra_information
    }
    return JSONResponse(content=content)
