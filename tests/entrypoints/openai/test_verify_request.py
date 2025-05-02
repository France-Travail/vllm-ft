import pytest
from fastapi import HTTPException

from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai import protocol as vllm_protocol


def test_verify_request():
    # Without HTTPException
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=1.0,
        echo=True,
        stream=None
    )
    assert api_server.verify_request(request) is None

    # With HTTPException echo and stream
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        echo=True,
        stream=True
    )
    with pytest.raises(HTTPException) as error:
        api_server.verify_request(request)
    assert error.value.detail == "Use both echo and stream breaks backend"

    # With HTTPException temperature and top_p equals to 0
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=0,
        top_p=0
    )
    with pytest.raises(HTTPException) as error:
        api_server.verify_request(request)
    assert error.value.detail == "Use temperature and top_p equal to 0 breaks the model"

    # With HTTPException high temperature and top_k equals to 1
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=2.4,
        top_k=1
    )
    with pytest.raises(HTTPException) as error:
        api_server.verify_request(request)
    assert error.value.detail == "Use temperature with high value: 2.4 and top_k equals to 1 : 1 breaks the model"

    # With HTTPException top_p and top_k equals to 1
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        top_p=1,
        top_k=1
    )
    with pytest.raises(HTTPException) as error:
        api_server.verify_request(request)
    assert error.value.detail == "Use top_p and top_k equal to 1 breaks the model"

    # With HTTPException max_tokens less than min_tokens
    request = vllm_protocol.ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        max_tokens=50,
        min_tokens=100
    )
    with pytest.raises(HTTPException) as error:
        api_server.verify_request(request)
    assert error.value.detail == "Use max_tokens: 50 less than min_tokens : 100 breaks the model"