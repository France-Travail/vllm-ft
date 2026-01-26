# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    try:
        await engine_client(raw_request).check_health()
        return Response(status_code=200)
    except EngineDeadError:
        return Response(status_code=503)


def attach_router(app):
    app.include_router(router)


@router.get("/liveness")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.get("/readiness")
async def get_readiness(raw_request: Request) -> Response:
    """Readiness probe for k8s"""
    try :
        model_executor = raw_request.app.state.openai_serving_chat.engine.engine.model_executor
        model_runner = model_executor.driver_worker.model_runner

        # check if model weight are loaded in gpu memory
        model_weights = model_runner.model_memory_usage

        # check if KV cache has been set up
        num_cpu_blocks = model_runner.num_cpu_blocks
        num_gpu_blocks = model_runner.num_gpu_blocks

        if model_weights > 0 and num_cpu_blocks > 0  and num_gpu_blocks > 0 :
            return Response(status_code=200)
    except: HTTPException(status_code=500, detail="Model not loaded yet or KV cache not setup yet")