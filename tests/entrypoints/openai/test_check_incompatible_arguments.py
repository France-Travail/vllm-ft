import pytest

from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest

def test_ChatCompletionRequest_check_incompatible_arguments():
    request = ChatCompletionRequest(
        messages=[{"role":"user", "content": "How are you ?"}],
        model="my_model",
        temperature=1.0,
        echo=True,
        stream=None
    ).model_dump()
    assert ChatCompletionRequest.check_incompatible_arguments(request) == request

    # With ValueError echo and stream
    with pytest.raises(ValueError, match="Using both echo and stream breaks backend") as error:
        request = ChatCompletionRequest(
            messages=[{"role":"user", "content": "How are you ?"}],
            model="my_model",
            echo=True,
            stream=True
        )

    # With ValueError temperature and top_p equals to 0
    with pytest.raises(ValueError, match="Using temperature and top_p equal to 0 breaks the model") as error:
        request = ChatCompletionRequest(
            messages=[{"role":"user", "content": "How are you ?"}],
            model="my_model",
            temperature=0,
            top_p=0
        )

    # With ValueError high temperature and top_k equals to 1
    with pytest.raises(ValueError, match="Using temperature with high value: 2.4 and top_k equals to 1 breaks the model") as error:
        request = ChatCompletionRequest(
            messages=[{"role":"user", "content": "How are you ?"}],
            model="my_model",
            temperature=2.4,
            top_k=1
        )

    # With ValueError top_p and top_k equals to 1
    with pytest.raises(ValueError, match="Using top_p and top_k equal to 1 breaks the model") as error:
        request = ChatCompletionRequest(
            messages=[{"role":"user", "content": "How are you ?"}],
            model="my_model",
            top_p=1,
            top_k=1
        )

    # With ValueError max_tokens less than min_tokens
    with pytest.raises(ValueError, match="Using max_tokens: 50 less than min_tokens : 100 breaks the model") as error:
        request = ChatCompletionRequest(
            messages=[{"role":"user", "content": "How are you ?"}],
            model="my_model",
            max_tokens=50,
            min_tokens=100
        )


def test_CompletionRequest_check_incompatible_arguments():
    request = CompletionRequest(
        prompt="How are you ?",
        model="my_model",
        temperature=1.0,
        echo=True,
        stream=None
    ).model_dump()
    assert CompletionRequest.check_incompatible_arguments(request) == request

    # With ValueError echo and stream
    with pytest.raises(ValueError, match="Using both echo and stream breaks backend") as error:
        request = CompletionRequest(
            prompt="How are you ?",
            model="my_model",
            echo=True,
            stream=True
        )

    # With ValueError temperature and top_p equals to 0
    with pytest.raises(ValueError, match="Using temperature and top_p equal to 0 breaks the model") as error:
        request = CompletionRequest(
            prompt="How are you ?",
            model="my_model",
            temperature=0,
            top_p=0
        )

    # With ValueError high temperature and top_k equals to 1
    with pytest.raises(ValueError, match="Using temperature with high value: 2.4 and top_k equals to 1 breaks the model") as error:
        request = CompletionRequest(
            prompt="How are you ?",
            model="my_model",
            temperature=2.4,
            top_k=1
        )

    # With ValueError top_p and top_k equals to 1
    with pytest.raises(ValueError, match="Using top_p and top_k equal to 1 breaks the model") as error:
        request = CompletionRequest(
            prompt="How are you ?",
            model="my_model",
            top_p=1,
            top_k=1
        )

    # With ValueError max_tokens less than min_tokens
    with pytest.raises(ValueError, match="Using max_tokens: 50 less than min_tokens : 100 breaks the model") as error:
        request = CompletionRequest(
            prompt="How are you ?",
            model="my_model",
            max_tokens=50,
            min_tokens=100
        )